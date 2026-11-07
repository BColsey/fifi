//! The device half of the feeds: the PLE key and value projections on the card, and the sink that
//! writes a token start to the device.
//!
//! WHAT: [`DevicePle`] keeps the stored F16 `key_proj` (`width x ple_embed`) and `value_proj`
//! (`hidden x ple_embed`) on the device as they are in the pack (65 MiB, no F32 copy) and runs
//! them with `CudaContext::dense_gemv`. [`DevicePle::enqueue`] writes the embedding and queues the
//! two launches on the step stream; [`DevicePle::collect`] reads the keys and the value back
//! (44 KiB), behind the queued work. [`DeviceStart`] is the [`StartSink`] over the device words
//! block and two F32 buffers: it builds the `StepParams` block (token and position) and writes the
//! initial stream and the PLE embedding, all ordered on the step stream.
//!
//! WHY the projections run here: they depend on the token only, but the token is known when the
//! step starts, so the host would have the 12,800 x 2,560 product to finish while the device runs
//! layers 0 and 1 (about 1 ms). The host product took 2.6 ms (F16, 8 threads) to 4-5 ms (1-4
//! threads) at p50 and up to 14 ms at p99 on this machine, and it competes with the pinned expert
//! copies for memory bandwidth. The device reads the 65 MiB in well under 0.1 ms. The rest of the
//! PLE block (norms, gate, convolution ring, about 100 us) stays on the host in `ple::step`, which
//! takes the downloaded keys and value in place of its two `linear` calls.
//!
//! Nothing here synchronizes except `collect` (a read behind the queue) and the pageable writes of
//! `DeviceBuffer::write_at`, which return when the copy finished, as the Qwen3.6 start write does.

use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceStream,
        resident::{
            dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape},
            step_params::StepParams,
        },
    },
};

use super::StartSink;
use crate::family::flash_next::cuda::ple_feed::{PleKeyValue, PleProjection};
use std::rc::Rc;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// The PLE projections resident on the device.
pub(in crate::family) struct DevicePle {
    key_weight: DeviceBuffer,
    value_weight: DeviceBuffer,
    embedding: DeviceBuffer,
    keys: DeviceBuffer,
    value: DeviceBuffer,
    width: usize,
    hidden: usize,
    embed: usize,
}

impl DevicePle {
    /// Uploads the projections as stored: `key_f16` is `width x embed` binary16 values and
    /// `value_f16` is `hidden x embed`, little-endian row-major (the pack layout `[out, in]`).
    pub(in crate::family) fn upload(
        context: &CudaContext,
        key_f16: &[u8],
        value_f16: &[u8],
        width: usize,
        hidden: usize,
        embed: usize,
    ) -> Result<Self, EngError> {
        let overflow = || EngError::BadRequest("ple projection sizes overflow".to_owned());
        let key_bytes = width
            .checked_mul(embed)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(overflow)?;
        let value_bytes = hidden
            .checked_mul(embed)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(overflow)?;
        if key_f16.len() != key_bytes || value_f16.len() != value_bytes {
            return Err(EngError::Checkpoint(format!(
                "ple projections hold {} and {} bytes, expected {key_bytes} and {value_bytes}",
                key_f16.len(),
                value_f16.len()
            )));
        }
        let place = |bytes: &[u8]| -> Result<DeviceBuffer, EngError> {
            let buffer = context.allocate(bytes.len())?;
            context.upload_at(&buffer, 0, bytes)?;
            Ok(buffer)
        };
        Ok(Self {
            key_weight: place(key_f16)?,
            value_weight: place(value_f16)?,
            embedding: context.allocate(embed * 4)?,
            keys: context.allocate(width * 4)?,
            value: context.allocate(hidden * 4)?,
            width,
            hidden,
            embed,
        })
    }

    /// Device bytes held by this object.
    pub(in crate::family) fn device_bytes(&self) -> usize {
        (self.width + self.hidden) * self.embed * 2 + (self.embed + self.width + self.hidden) * 4
    }

    /// Writes `embedding` and queues both projections on `stream`. The results stay on the device
    /// until [`Self::collect`].
    pub(in crate::family) fn enqueue(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        embedding: &[f32],
    ) -> Result<(), EngError> {
        if embedding.len() != self.embed {
            return Err(EngError::BadRequest(format!(
                "the PLE embedding has {} values, expected {}",
                embedding.len(),
                self.embed
            )));
        }
        self.embedding.write_at(stream, 0, &f32_bytes(embedding))?;
        for (rows, weight, y, y_len) in [
            (self.width, &self.key_weight, &self.keys, self.width),
            (self.hidden, &self.value_weight, &self.value, self.hidden),
        ] {
            context.dense_gemv(
                stream,
                DenseGemvCall {
                    shape: DenseGemvShape::packed(1, rows, self.embed),
                    dtype: DenseDtype::F16,
                    w: weight.range(0, rows * self.embed * 2)?,
                    x: self.embedding.range(0, self.embed * 4)?,
                    y: y.range(0, y_len * 4)?,
                },
            )?;
        }
        Ok(())
    }

    /// Reads the keys and the value of the last [`Self::enqueue`], behind the queued work.
    pub(in crate::family) fn collect(
        &self,
        stream: &DeviceStream,
    ) -> Result<PleKeyValue, EngError> {
        let mut keys = vec![0u8; self.width * 4];
        let mut value = vec![0u8; self.hidden * 4];
        self.keys.read_at(stream, 0, &mut keys)?;
        self.value.read_at(stream, 0, &mut value)?;
        Ok(PleKeyValue {
            keys: f32_values(&keys),
            value: f32_values(&value),
        })
    }

    /// [`Self::enqueue`] and [`Self::collect`].
    pub(in crate::family) fn project(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        embedding: &[f32],
    ) -> Result<PleKeyValue, EngError> {
        self.enqueue(context, stream, embedding)?;
        self.collect(stream)
    }
}

/// The device route of the PLE step: [`PleProjection`] over [`DevicePle`], for `PleFeed`. It owns
/// shared handles to the step's context and stream, so the projection queues behind the layers
/// already enqueued on that stream and `project` returns when its results are read back.
pub(in crate::family) struct DevicePleRoute {
    ple: DevicePle,
    context: Rc<CudaContext>,
    stream: Rc<DeviceStream>,
}

impl DevicePleRoute {
    pub(in crate::family) fn new(
        ple: DevicePle,
        context: Rc<CudaContext>,
        stream: Rc<DeviceStream>,
    ) -> Self {
        Self {
            ple,
            context,
            stream,
        }
    }
}

impl PleProjection for DevicePleRoute {
    fn project(&mut self, embedding: &[f32]) -> Result<PleKeyValue, EngError> {
        self.ple.project(&self.context, &self.stream, embedding)
    }
}

/// The token start on the device: where the Flash state keeps the step parameters, the hyper
/// stream and the PLE embedding. The offsets come from the state layout.
pub(in crate::family) struct DeviceStart<'a> {
    pub(in crate::family) stream: &'a DeviceStream,
    pub(in crate::family) words: &'a DeviceBuffer,
    pub(in crate::family) params_offset: usize,
    pub(in crate::family) hyper: &'a DeviceBuffer,
    pub(in crate::family) hyper_offset: usize,
    pub(in crate::family) ple: &'a DeviceBuffer,
    pub(in crate::family) ple_offset: usize,
}

impl StartSink for DeviceStart<'_> {
    fn write_step_params(&mut self, token: i32, pos: i32) -> Result<(), EngError> {
        let params = StepParams::new(token, pos)?;
        self.words
            .write_at(self.stream, self.params_offset, &params.to_bytes())
    }

    fn write_stream(&mut self, stream: &[f32]) -> Result<(), EngError> {
        self.hyper
            .write_at(self.stream, self.hyper_offset, &f32_bytes(stream))
    }

    fn write_ple_embedding(&mut self, embedding: &[f32]) -> Result<(), EngError> {
        self.ple
            .write_at(self.stream, self.ple_offset, &f32_bytes(embedding))
    }
}

#[cfg(test)]
mod tests;
