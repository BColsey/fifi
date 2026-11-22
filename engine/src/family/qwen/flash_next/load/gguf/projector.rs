//! Mapped GGUF host projection and value-head layout conversion.

use super::{ngram::GgufNgramRows, tensors::Mapped};
use crate::{
    EngError,
    family::{project::Projector, recipe::Global},
    weights::{GgufType, GgufWeights, dequantize, dequantize_into},
};

/// Chunk of a tensor decoded at a time, in stored bytes.
const CHUNK_BYTES: usize = 24 << 20;

pub(super) fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::Checkpoint(format!("Flash-Next GGUF host forward: {}", detail.as_ref()))
}

/// How a projection's rows or input are reordered from the file's value-head order to the
/// step's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Remap {
    None,
    /// The outputs from `skip` on are value heads of `hd` values each, tiled in the file.
    ValueOut {
        skip: usize,
        hd: usize,
        key_heads: usize,
        value_heads: usize,
    },
    /// The input is value heads of `hd` values each, grouped; the file wants them tiled.
    ValueIn {
        hd: usize,
        key_heads: usize,
        value_heads: usize,
    },
}

/// Position in the file's tiled order of the value head the step calls `h`.
fn tiled_position(h: usize, key_heads: usize, value_heads: usize) -> usize {
    let per_key = value_heads / key_heads;
    let (group, rep) = (h / per_key, h % per_key);
    rep * key_heads + group
}

/// Reorders `values[skip..]`, heads of `unit` values, from tiled to grouped order.
pub(super) fn untile(
    values: &[f32],
    skip: usize,
    unit: usize,
    key_heads: usize,
    value_heads: usize,
) -> Vec<f32> {
    let mut out = values.to_vec();
    for h in 0..value_heads {
        let from = skip + tiled_position(h, key_heads, value_heads) * unit;
        out[skip + h * unit..skip + (h + 1) * unit].copy_from_slice(&values[from..from + unit]);
    }
    out
}

/// Reorders `values`, heads of `unit` values, from grouped to tiled order.
pub(super) fn tile(values: &[f32], unit: usize, key_heads: usize, value_heads: usize) -> Vec<f32> {
    let mut out = values.to_vec();
    for h in 0..value_heads {
        let to = tiled_position(h, key_heads, value_heads) * unit;
        out[to..to + unit].copy_from_slice(&values[h * unit..(h + 1) * unit]);
    }
    out
}

#[derive(Debug, Clone)]
pub(super) struct Part {
    name: String,
    ty: GgufType,
    pub(super) input: usize,
    pub(super) out: usize,
}

/// A weight of the host step: tensors read from the file on use.
#[derive(Debug, Clone)]
pub(super) struct HostWeight {
    pub(super) parts: Vec<Part>,
    remap: Remap,
    /// Whether the tensor is a stack of experts (the last dimension indexes them).
    experts: usize,
    ngram: bool,
}

impl HostWeight {
    pub(super) fn new(entries: &[&Mapped], remap: Remap) -> Self {
        let parts = entries
            .iter()
            .map(|entry| Part {
                name: entry.name.clone(),
                ty: entry.ty,
                input: entry.dims[0] as usize,
                out: entry.dims.get(1).copied().unwrap_or(1) as usize,
            })
            .collect();
        let experts = entries[0].dims.get(2).copied().unwrap_or(0) as usize;
        Self {
            parts,
            remap,
            experts,
            ngram: entries[0].role == Global::NgramTable.into(),
        }
    }
}

/// The projector of the host step over a GGUF.
pub(super) struct HostProjector<'a> {
    weights: &'a GgufWeights,
    ngram: &'a GgufNgramRows,
    threads: usize,
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    cuda: Option<super::cuda_probe::Projection<'a>>,
}

impl<'a> HostProjector<'a> {
    pub(super) fn new(weights: &'a GgufWeights, ngram: &'a GgufNgramRows, threads: usize) -> Self {
        Self {
            weights,
            ngram,
            threads: threads.max(1),
            #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
            cuda: None,
        }
    }

    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    pub(super) fn with_cuda(mut self, cuda: super::cuda_probe::Projection<'a>) -> Self {
        self.cuda = Some(cuda);
        self
    }

    /// `rows` rows of `part` from row `first` of expert `expert`, dotted with `x`.
    fn matvec_range(
        weights: &GgufWeights,
        part: &Part,
        base: u64,
        first: usize,
        count: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let block = part.ty.block_elems() as usize;
        let row_bytes = part.input / block * part.ty.block_bytes() as usize;
        let per_chunk = (CHUNK_BYTES / row_bytes).max(1);
        let mut out = Vec::with_capacity(count);
        let mut bytes = Vec::new();
        let mut decoded = Vec::new();
        let mut done = 0;
        while done < count {
            let rows = per_chunk.min(count - done);
            bytes.resize(rows * row_bytes, 0);
            decoded.resize(rows * part.input, 0f32);
            weights.read_range(
                &part.name,
                base + ((first + done) * row_bytes) as u64,
                &mut bytes,
            )?;
            dequantize_into(part.ty, &bytes, &mut decoded)?;
            out.extend(
                decoded
                    .chunks_exact(part.input)
                    .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>()),
            );
            done += rows;
        }
        Ok(out)
    }

    fn matvec(
        &mut self,
        part: &Part,
        expert: Option<u32>,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        if x.len() != part.input {
            return Err(refuse(format!(
                "{} takes {} inputs, got {}",
                part.name,
                part.input,
                x.len()
            )));
        }
        let block = part.ty.block_elems() as usize;
        let row_bytes = (part.input / block * part.ty.block_bytes() as usize) as u64;
        let base = expert.map_or(0, |e| u64::from(e) * part.out as u64 * row_bytes);
        #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
        if let Some(cuda) = self.cuda.as_mut()
            && cuda.supports(part.ty)
        {
            let mut out = Vec::with_capacity(part.out);
            let mut bytes = Vec::new();
            let chunk_rows = (CHUNK_BYTES as u64 / row_bytes).max(1) as usize;
            for first in (0..part.out).step_by(chunk_rows) {
                let count = chunk_rows.min(part.out - first);
                bytes.resize(count * row_bytes as usize, 0);
                self.weights
                    .read_range(&part.name, base + first as u64 * row_bytes, &mut bytes)?;
                out.extend(cuda.run(part.ty, &bytes, x)?);
            }
            return Ok(out);
        }
        let threads = self.threads.min(part.out.div_ceil(64)).max(1);
        let share = part.out.div_ceil(threads);
        let weights = self.weights;
        let pieces: Vec<Result<Vec<f32>, EngError>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let first = (t * share).min(part.out);
                    let count = share.min(part.out - first);
                    scope.spawn(move || Self::matvec_range(weights, part, base, first, count, x))
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("matvec thread"))
                .collect()
        });
        let mut out = Vec::with_capacity(part.out);
        for piece in pieces {
            out.extend(piece?);
        }
        Ok(out)
    }
}

impl Projector for HostProjector<'_> {
    type Weight = HostWeight;

    fn linear(&mut self, weight: &HostWeight, x: &[f32]) -> Result<Vec<f32>, EngError> {
        let tiled_input;
        let x = match weight.remap {
            Remap::ValueIn {
                hd,
                key_heads,
                value_heads,
            } => {
                tiled_input = tile(x, hd, key_heads, value_heads);
                &tiled_input[..]
            }
            _ => x,
        };
        let mut out = Vec::new();
        for part in &weight.parts {
            out.extend(self.matvec(part, None, x)?);
        }
        if let Remap::ValueOut {
            skip,
            hd,
            key_heads,
            value_heads,
        } = weight.remap
        {
            out = untile(&out, skip, hd, key_heads, value_heads);
        }
        Ok(out)
    }

    fn expert_linear(
        &mut self,
        stack: &HostWeight,
        expert: u32,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        if expert as usize >= stack.experts {
            return Err(refuse(format!("expert {expert} of {}", stack.experts)));
        }
        self.matvec(&stack.parts[0], Some(expert), x)
    }

    fn rows(&mut self, table: &HostWeight, rows: &[u32]) -> Result<Vec<f32>, EngError> {
        let part = &table.parts[0];
        let mut out = Vec::with_capacity(rows.len() * part.input);
        for &row in rows {
            if table.ngram {
                out.extend(self.ngram.row(u64::from(row))?);
            } else {
                let block = part.ty.block_elems() as usize;
                let row_bytes = part.input / block * part.ty.block_bytes() as usize;
                let mut bytes = vec![0u8; row_bytes];
                self.weights.read_range(
                    &part.name,
                    u64::from(row) * row_bytes as u64,
                    &mut bytes,
                )?;
                out.extend(dequantize(part.ty, &bytes)?);
            }
        }
        Ok(out)
    }
}
