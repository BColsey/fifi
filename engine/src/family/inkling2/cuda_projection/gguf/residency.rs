//! Adapt contiguous NVFP4 stacks in a GGUF mapping to the shared expert tier.

use std::{collections::BTreeMap, ops::Range};

use super::super::{PackedBuffers, bad};
use crate::{
    EngError,
    expert_tier::{
        ExpertId, ExpertPayload, ExpertPayloadSource, GpuExpertStore, GpuLayerPlan, Placement,
        Residency, place_gpu_layers,
    },
    family::mapped_file::MappedFile,
    kernels::cuda::{CudaContext, DeviceRange},
    weights::repack_nvfp4_into,
};

pub(in crate::family::inkling2) struct Nvfp4Stack<'a> {
    pub bytes: &'a [u8],
    pub file_range: Range<usize>,
    pub tensor_scales: Option<&'a [u8]>,
    pub shape: [u32; 2],
}

pub(in crate::family::inkling2) struct Nvfp4Layer<'a> {
    pub layer: u32,
    pub parts: [Nvfp4Stack<'a>; 2],
}

pub(super) struct ResidentExperts {
    placement: Residency,
    banks: BTreeMap<u32, [Option<PackedBuffers>; 2]>,
}

impl ResidentExperts {
    pub(super) fn load(
        ctx: &CudaContext,
        experts: u32,
        layers: Vec<Nvfp4Layer<'_>>,
        file: &MappedFile,
    ) -> Result<Self, EngError> {
        let mut banks = BTreeMap::new();
        let Some(first) = layers.first() else {
            return Ok(Self {
                placement: Residency::default(),
                banks,
            });
        };
        let bytes = layer_bytes(first);
        // A validated Inkling backbone gives every NVFP4 layer the same geometry.
        if layers.iter().any(|layer| layer_bytes(layer) != bytes) {
            return Err(bad("GGUF NVFP4 layers disagree on their packed sizes"));
        }
        // The shared placement walk consumes its order from the end.
        let ids: Vec<_> = layers.iter().rev().map(|layer| layer.layer).collect();
        let mut source = Source {
            layers,
            file,
            experts,
        };
        let mut store = Store {
            ctx,
            experts,
            banks: &mut banks,
        };
        let placement = place_gpu_layers(
            GpuLayerPlan::Auto,
            &ids,
            experts,
            bytes,
            &mut store,
            &mut source,
        )?;
        Ok(Self { placement, banks })
    }

    pub(super) fn ranges(
        &self,
        id: ExpertId,
        part: usize,
        lengths: [usize; 2],
    ) -> Result<Option<[DeviceRange<'_>; 2]>, EngError> {
        if self.placement.placement(id) != Placement::Gpu {
            return Ok(None);
        }
        let buffers = self
            .banks
            .get(&id.layer)
            .and_then(|bank| bank.get(part))
            .and_then(Option::as_ref)
            .ok_or_else(|| bad("GGUF tier assigned an expert without a device bank"))?;
        buffers.ranges_len(id.expert as usize, lengths).map(Some)
    }

    pub(super) fn placement(&self) -> &Residency {
        &self.placement
    }

    pub(super) fn bytes(&self) -> u64 {
        self.banks
            .values()
            .flatten()
            .flatten()
            .map(|p| (p.codes.bytes() + p.scales.bytes()) as u64)
            .sum()
    }
}

fn layer_bytes(layer: &Nvfp4Layer<'_>) -> u64 {
    layer.parts.iter().map(|part| part.bytes.len() as u64).sum()
}

struct Source<'a> {
    layers: Vec<Nvfp4Layer<'a>>,
    file: &'a MappedFile,
    experts: u32,
}

impl Source<'_> {
    fn layer(&self, layer: u32) -> Result<&Nvfp4Layer<'_>, EngError> {
        self.layers
            .iter()
            .find(|entry| entry.layer == layer)
            .ok_or_else(|| bad("GGUF tier requested an unknown layer"))
    }
}

impl ExpertPayloadSource for Source<'_> {
    fn parts(&self) -> usize {
        2
    }

    fn payload(&mut self, id: ExpertId, part: usize) -> Result<ExpertPayload, EngError> {
        let part = self
            .layer(id.layer)?
            .parts
            .get(part)
            .ok_or_else(|| bad("GGUF tier requested an unknown expert part"))?;
        if id.expert >= self.experts {
            return Err(bad("GGUF tier requested an out-of-range expert"));
        }
        let size = part.bytes.len() / self.experts as usize;
        let offset = id.expert as usize * size;
        let n = part.shape[0] as usize * part.shape[1] as usize;
        let mut data = vec![0; n / 2];
        let mut scales = vec![0; n / 16];
        repack_nvfp4_into(&part.bytes[offset..][..size], &mut data, &mut scales)?;
        let tensor_scale = part.tensor_scales.map_or(1.0, |raw| {
            f32::from_le_bytes(raw.as_chunks::<4>().0[id.expert as usize])
        });
        if !tensor_scale.is_finite() || tensor_scale < 0.0 {
            return Err(bad("GGUF tier found an invalid expert scale"));
        }
        Ok(ExpertPayload {
            data,
            scales,
            tensor_scale,
            shape: part.shape,
        })
    }

    fn release_layer(&mut self, layer: u32) -> Result<(), EngError> {
        for part in &self.layer(layer)?.parts {
            self.file.drop_cache(part.file_range.clone())?;
        }
        Ok(())
    }
}

struct Store<'a> {
    ctx: &'a CudaContext,
    experts: u32,
    banks: &'a mut BTreeMap<u32, [Option<PackedBuffers>; 2]>,
}

impl GpuExpertStore for Store<'_> {
    fn upload(
        &mut self,
        id: ExpertId,
        part: usize,
        payload: &ExpertPayload,
    ) -> Result<(), EngError> {
        let [data, scales] = [payload.data.as_slice(), payload.scales.as_slice()];
        let slot = self
            .banks
            .entry(id.layer)
            .or_default()
            .get_mut(part)
            .ok_or_else(|| bad("GGUF tier requested an unknown device bank"))?;
        if slot.is_none() {
            let count = self.experts as usize;
            *slot = Some(PackedBuffers {
                codes: self.ctx.allocate(data.len() * count)?,
                scales: self.ctx.allocate(scales.len() * count)?,
            });
        }
        let bank = slot.as_ref().expect("allocated bank");
        let [d, s] = bank.ranges_len(id.expert as usize, [data.len(), scales.len()])?;
        self.ctx.upload_at(d.buffer(), d.offset(), data)?;
        self.ctx.upload_at(s.buffer(), s.offset(), scales)
    }

    fn free_bytes(&self) -> Result<u64, EngError> {
        Ok(self.ctx.memory_info()?.free as u64)
    }

    fn reserved_bytes(&self) -> u64 {
        // Dense, streaming and activation buffers were admitted and allocated first.
        0
    }

    fn finish_placement(&mut self, _: u64) -> Result<(), EngError> {
        Ok(())
    }
}
