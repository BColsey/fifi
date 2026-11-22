//! A resident snapshot as bytes, for the disk tier: one object per layer, in model order.
//!
//! A DeltaNet layer is its recurrent state then its convolution ring, both F32. An attention
//! layer is its cache tensors in the order `KvCache::tensors` lists them (keys then values; in the
//! 8-bit form codes, scales, biases of each side), each cut to the valid positions: for every key
//! value head the first `len` rows. The capacity slack is not written; a snapshot read back gets
//! a capacity of the next growth quantum and zeroed slack, exactly what a state that had grown
//! to `len` holds.

use super::super::{
    attn::{KvBuffers, KvCache, Q8Tensor},
    kv_plan::next_capacity,
    ops::{dim, short_import},
    prefix_pages::{Page, allowance_pages},
    weights::{RMixer, ResidentWeights},
};
use super::{
    BUFFER_GENERATION, LayerState, ResidentSnapshot, ResidentState, fixed_bytes, kv_bytes,
};
use crate::{
    EngError,
    family::qwen36::settings::{KvDtype, Q8_GROUP},
    kernels::MlxKernels,
    sched::SessionBudget,
    tensor::{Dtype, ExecutionTensor, Tensor},
};
use std::{cell::RefCell, rc::Rc, sync::atomic::Ordering};

/// The width and element type of each tensor of one attention layer's cache, in the order
/// `KvCache::tensors` lists them.
fn kv_parts(weights: &ResidentWeights) -> Result<Vec<(u32, Dtype)>, EngError> {
    let width = dim(weights.shape.attn.head_dim)?;
    let act = weights.consts.act;
    Ok(match weights.kv {
        KvDtype::F32 => vec![(width, Dtype::F32); 2],
        KvDtype::Bf16 => vec![(width, Dtype::Bf16); 2],
        KvDtype::Q8 => {
            let side = [
                (width / 4, Dtype::U32),
                (width / Q8_GROUP, act),
                (width / Q8_GROUP, act),
            ];
            side.into_iter().chain(side).collect()
        }
    })
}

/// The bytes of the DeltaNet state and ring of one layer, both F32.
fn delta_bytes(weights: &ResidentWeights) -> Result<usize, EngError> {
    let d = &weights.shape.delta;
    Ok(
        4 * (dim(d.value_heads)? as usize * dim(d.value_dim)? as usize * dim(d.key_dim)? as usize
            + dim(d.conv_dim())? as usize * (dim(d.conv_kernel - 1)? as usize)),
    )
}

/// The bytes of each layer's object for a snapshot of `positions`, in model order.
pub(in crate::family::qwen36::resident) fn object_bytes(
    weights: &ResidentWeights,
    positions: usize,
) -> Result<Vec<usize>, EngError> {
    let d = &weights.shape.delta;
    let delta = 4
        * (dim(d.value_heads)? as usize * dim(d.value_dim)? as usize * dim(d.key_dim)? as usize
            + dim(d.conv_dim())? as usize * (dim(d.conv_kernel - 1)? as usize));
    let heads = dim(weights.shape.attn.heads_kv)? as usize;
    let attn: usize = kv_parts(weights)?
        .iter()
        .map(|&(width, dtype)| heads * positions * width as usize * dtype.byte_width() as usize)
        .sum();
    Ok(weights
        .layers
        .iter()
        .map(|layer| match layer.mixer {
            RMixer::Delta(_) => delta,
            RMixer::Attn(_) => attn,
        })
        .collect())
}

/// The host copy of one snapshot: its objects back to back, read from the device when the disk
/// tier first needs them and kept until the snapshot goes.
#[derive(Default)]
pub(in crate::family::qwen36::resident) struct HostCopy(RefCell<Option<Rc<Vec<u8>>>>);

impl HostCopy {
    pub(in crate::family::qwen36::resident) fn get(&self) -> Option<Rc<Vec<u8>>> {
        self.0.borrow().clone()
    }

    pub(in crate::family::qwen36::resident) fn set(&self, bytes: Vec<u8>) {
        *self.0.borrow_mut() = Some(Rc::new(bytes));
    }
}

fn tensor_bytes(kernels: &mut MlxKernels, tensor: &ExecutionTensor) -> Result<Tensor, EngError> {
    kernels.readback(tensor)
}

fn host(tensor: &Tensor) -> Result<&[u8], EngError> {
    tensor.host_bytes().ok_or_else(|| EngError::Backend {
        operation: "snapshot read-back",
        message: "the read-back tensor holds no host bytes".into(),
    })
}

impl ResidentSnapshot {
    /// Reads the snapshot back into its host copy unless that exists, and checks it against the
    /// layout the weights give a snapshot of this length.
    pub(in crate::family::qwen36::resident) fn make_host_copy(
        &self,
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
    ) -> Result<(), EngError> {
        if self.host.get().is_some() {
            return Ok(());
        }
        let objects = self.to_objects(kernels)?;
        let want = object_bytes(weights, self.pos as usize)?;
        if objects.iter().map(Vec::len).collect::<Vec<_>>() != want {
            return Err(EngError::BadRequest(
                "a snapshot read back does not match the model layout".into(),
            ));
        }
        self.host.set(objects.concat());
        Ok(())
    }

    /// The host copy, if one was made.
    pub(in crate::family::qwen36::resident) fn host_copy(&self) -> &HostCopy {
        &self.host
    }

    /// The snapshot as one object per layer. Reads every array back, so it synchronizes.
    pub(in crate::family::qwen36::resident) fn to_objects(
        &self,
        kernels: &mut MlxKernels,
    ) -> Result<Vec<Vec<u8>>, EngError> {
        let mut objects = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let mut bytes = Vec::new();
            match layer {
                LayerState::Delta { rec, ring } => {
                    for tensor in [rec, ring] {
                        let read = tensor_bytes(kernels, tensor)?;
                        if read.dtype() != Dtype::F32 {
                            return Err(EngError::BadRequest(
                                "a DeltaNet state array is not F32".into(),
                            ));
                        }
                        bytes.extend_from_slice(host(&read)?);
                    }
                }
                LayerState::Attn(cache) => {
                    for tensor in cache.tensors() {
                        let read = tensor_bytes(kernels, tensor)?;
                        let &[_, heads, cap, width] = read.shape() else {
                            return Err(EngError::BadRequest(
                                "an attention cache tensor is not [1, heads, cap, width]".into(),
                            ));
                        };
                        if cache.len > cap as usize {
                            return Err(EngError::BadRequest(
                                "an attention cache tensor is shorter than its length".into(),
                            ));
                        }
                        let row = width as usize * read.dtype().byte_width() as usize;
                        let data = host(&read)?;
                        for head in 0..heads as usize {
                            let start = head * cap as usize * row;
                            bytes.extend_from_slice(&data[start..start + cache.len * row]);
                        }
                    }
                }
            }
            objects.push(bytes);
        }
        Ok(objects)
    }
}

impl ResidentState {
    /// What `snapshot_from_objects` charges the session budget for a snapshot of `positions`
    /// (the fixed DeltaNet bytes, the page-table allowance and the cache bytes), so a caller can
    /// check the room for it before it reads anything.
    pub(in crate::family::qwen36::resident) fn loaded_charge(
        &self,
        weights: &ResidentWeights,
        positions: usize,
    ) -> Result<u64, EngError> {
        let overflow = || EngError::MemoryPlan("snapshot bytes overflow u64".into());
        let delta_layers = weights
            .layers
            .iter()
            .filter(|layer| matches!(layer.mixer, RMixer::Delta(_)))
            .count();
        let attn_layers = weights.layers.len() - delta_layers;
        let fixed = (delta_layers as u64)
            .checked_mul(delta_bytes(weights)? as u64)
            .ok_or_else(overflow)?;
        let pos = u32::try_from(positions)
            .map_err(|_| EngError::BadRequest("a stored snapshot is too long".into()))?;
        let positions_charged = if self.kv_ceiling > 0 {
            next_capacity(0, positions, 0, self.kv_step, self.kv_ceiling)?
        } else {
            positions
        };
        let kv = (attn_layers as u64)
            .checked_mul(positions_charged as u64)
            .and_then(|n| n.checked_mul(self.kv_growth_per_layer))
            .ok_or_else(overflow)?;
        fixed
            .checked_add(allowance_pages(pos) * std::mem::size_of::<Page>() as u64)
            .and_then(|n| n.checked_add(kv))
            .ok_or_else(overflow)
    }

    /// A snapshot read from `objects` (the bytes `to_objects` wrote for `positions`), charged to
    /// `budget` like one taken from a live state. The caches get the capacity this state's growth
    /// rule gives `positions`; the slack is zero.
    pub(in crate::family::qwen36::resident) fn snapshot_from_objects(
        &self,
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
        budget: &mut SessionBudget,
        positions: usize,
        objects: &[&[u8]],
    ) -> Result<ResidentSnapshot, EngError> {
        let sizes = object_bytes(weights, positions)?;
        if objects.len() != sizes.len()
            || objects
                .iter()
                .zip(&sizes)
                .any(|(got, &want)| got.len() != want)
        {
            return Err(EngError::BadRequest(
                "a stored snapshot does not match the model layout".into(),
            ));
        }
        let d = &weights.shape.delta;
        let rec_shape = vec![dim(d.value_heads)?, dim(d.value_dim)?, dim(d.key_dim)?];
        let ring_shape = vec![dim(d.conv_dim())?, dim(d.conv_kernel - 1)?];
        let rec_bytes = 4 * rec_shape.iter().map(|&n| n as usize).product::<usize>();
        let heads = dim(weights.shape.attn.heads_kv)? as usize;
        let parts = kv_parts(weights)?;
        let cap = next_capacity(0, positions, 0, self.kv_step, self.kv_ceiling)?;
        let mut host_tensors: Vec<Tensor> = Vec::new();
        let mut plan: Vec<(usize, usize)> = Vec::new();
        for (object, layer) in objects.iter().zip(&weights.layers) {
            let first = host_tensors.len();
            match layer.mixer {
                RMixer::Delta(_) => {
                    host_tensors.push(Tensor::try_from_host(
                        Dtype::F32,
                        rec_shape.clone(),
                        object[..rec_bytes].to_vec(),
                    )?);
                    host_tensors.push(Tensor::try_from_host(
                        Dtype::F32,
                        ring_shape.clone(),
                        object[rec_bytes..].to_vec(),
                    )?);
                }
                RMixer::Attn(_) => {
                    let mut at = 0usize;
                    for &(width, dtype) in &parts {
                        let row = width as usize * dtype.byte_width() as usize;
                        let mut padded = vec![0u8; heads * cap * row];
                        for head in 0..heads {
                            let from = at + head * positions * row;
                            let to = head * cap * row;
                            padded[to..to + positions * row]
                                .copy_from_slice(&object[from..from + positions * row]);
                        }
                        at += heads * positions * row;
                        host_tensors.push(Tensor::try_from_host(
                            dtype,
                            vec![1, dim(heads)?, dim(cap)?, width],
                            padded,
                        )?);
                    }
                }
            }
            plan.push((first, host_tensors.len()));
        }
        let refs: Vec<&Tensor> = host_tensors.iter().collect();
        let mut imported = kernels.import_many(&refs)?.into_iter();
        let mut layers = Vec::with_capacity(plan.len());
        for (layer, &(first, last)) in weights.layers.iter().zip(&plan) {
            let mut take = |count: usize| -> Result<Vec<ExecutionTensor>, EngError> {
                (0..count)
                    .map(|_| imported.next().ok_or_else(short_import))
                    .collect()
            };
            layers.push(match layer.mixer {
                RMixer::Delta(_) => {
                    let mut two = take(last - first)?.into_iter();
                    LayerState::Delta {
                        rec: two.next().ok_or_else(short_import)?,
                        ring: two.next().ok_or_else(short_import)?,
                    }
                }
                RMixer::Attn(_) => {
                    let mut tensors = take(last - first)?.into_iter();
                    let mut next = || tensors.next().ok_or_else(short_import);
                    let buffers = match weights.kv {
                        KvDtype::F32 => KvBuffers::F32 {
                            k: next()?,
                            v: next()?,
                        },
                        KvDtype::Bf16 => KvBuffers::Bf16 {
                            k: next()?,
                            v: next()?,
                        },
                        KvDtype::Q8 => KvBuffers::Q8 {
                            k: Q8Tensor {
                                codes: next()?,
                                scales: next()?,
                                biases: next()?,
                            },
                            v: Q8Tensor {
                                codes: next()?,
                                scales: next()?,
                                biases: next()?,
                            },
                        },
                    };
                    let mut cache =
                        KvCache::with_form(weights.kv, self.kv_step).with_ceiling(self.kv_ceiling);
                    cache.buffers = Some(buffers);
                    cache.len = positions;
                    cache.cap = cap;
                    LayerState::Attn(cache)
                }
            });
        }
        let pos = u32::try_from(positions)
            .map_err(|_| EngError::BadRequest("a stored snapshot is too long".into()))?;
        for (layer, state) in weights.layers.iter().zip(&layers) {
            if let (RMixer::Attn(_), LayerState::Attn(cache)) = (&layer.mixer, state) {
                cache.check_buffers(
                    dim(weights.shape.attn.heads_kv)?,
                    dim(weights.shape.attn.head_dim)?,
                    weights.kv,
                )?;
            }
        }
        let charge = budget.acquire_growth(
            fixed_bytes(&layers)?
                .checked_add(allowance_pages(pos) * std::mem::size_of::<Page>() as u64)
                .ok_or_else(|| EngError::MemoryPlan("snapshot bytes overflow u64".into()))?,
        )?;
        let kv = budget.acquire_growth(kv_bytes(
            &layers,
            self.kv_growth_per_layer,
            self.kv_ceiling > 0,
        ))?;
        Ok(ResidentSnapshot {
            layers,
            pos,
            charge,
            kv_permits: vec![kv],
            generation: BUFFER_GENERATION.fetch_add(1, Ordering::Relaxed),
            host: HostCopy::default(),
        })
    }
}
