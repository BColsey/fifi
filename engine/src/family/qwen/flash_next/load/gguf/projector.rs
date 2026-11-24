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
            && !(part.ty == GgufType::Q80 && std::env::var_os("FIFI_FLASH_GGUF_HOST_Q80").is_some())
            && !(part.ty == GgufType::Q6K && std::env::var_os("FIFI_FLASH_GGUF_HOST_Q6K").is_some())
            && !(expert.is_some() && std::env::var_os("FIFI_FLASH_GGUF_HOST_EXPERTS").is_some())
        {
            let mut out = Vec::with_capacity(part.out);
            let mut bytes = Vec::new();
            let chunk_rows = (CHUNK_BYTES as u64 / row_bytes).max(1) as usize;
            for first in (0..part.out).step_by(chunk_rows) {
                let count = chunk_rows.min(part.out - first);
                bytes.resize(count * row_bytes as usize, 0);
                self.weights
                    .read_range(&part.name, base + first as u64 * row_bytes, &mut bytes)?;
                let expert_f16 = expert.is_some()
                    && matches!(part.ty, GgufType::Iq2S | GgufType::Iq4Nl)
                    && std::env::var_os("FIFI_FLASH_GGUF_EXPERT_F16").is_some();
                let trunk_f16 = expert.is_none()
                    && matches!(part.ty, GgufType::Q80 | GgufType::Q6K)
                    && std::env::var_os("FIFI_FLASH_GGUF_TRUNK_F16").is_some();
                let use_f16 = expert_f16 || trunk_f16;
                // The 8-bit-activation arithmetic only when a comparison asks for it by name.
                let q8 = std::env::var_os("FIFI_FLASH_GGUF_PROJECTION_Q8").is_some();
                out.extend(if use_f16 {
                    cuda.run_f16(part.ty, &bytes, x)?
                } else if q8 {
                    cuda.run_q8(part.ty, &bytes, x)?
                } else {
                    cuda.run(part.ty, &bytes, x)?
                });
            }
            if std::env::var_os("FIFI_FLASH_GGUF_TRACE_PROJECTIONS").is_some() {
                let reference = Self::matvec_range(self.weights, part, base, 0, part.out, x)?;
                let (error, norm) =
                    out.iter()
                        .zip(&reference)
                        .fold((0.0_f64, 0.0_f64), |(e, n), (&a, &b)| {
                            (
                                e + (f64::from(a) - f64::from(b)).powi(2),
                                n + f64::from(b).powi(2),
                            )
                        });
                let rel = if norm == 0.0 {
                    if error == 0.0 { 0.0 } else { f64::INFINITY }
                } else {
                    (error / norm).sqrt()
                };
                eprintln!(
                    "projection trace: name={} expert={expert:?} type={:?} rows={} relative_l2={rel}",
                    part.name, part.ty, part.out
                );
                if !rel.is_finite() || rel > 0.05 {
                    return Err(refuse(format!(
                        "projection trace exceeded 0.05 relative L2: {} expert={expert:?} type={:?} relative_l2={rel}",
                        part.name, part.ty
                    )));
                }
            }
            #[cfg(test)]
            capture_layer(part, expert, x, &out);
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
        #[cfg(test)]
        capture_layer(part, expert, x, &out);
        Ok(out)
    }
}

#[cfg(test)]
fn capture_layer(part: &Part, expert: Option<u32>, x: &[f32], out: &[f32]) {
    let Some(path) = std::env::var_os("FIFI_FLASH_GGUF_LAYER0_CAPTURE") else {
        return;
    };
    let wanted = if let Ok(value) = std::env::var("FIFI_FLASH_GGUF_CAPTURE_PREFIX") {
        let count: usize = value.parse().expect("capture prefix must be an integer");
        part.name
            .strip_prefix("blk.")
            .and_then(|name| name.split_once('.'))
            .and_then(|(layer, _)| layer.parse::<usize>().ok())
            .is_some_and(|layer| layer < count)
    } else {
        let layer = std::env::var("FIFI_FLASH_GGUF_CAPTURE_LAYER")
            .map(|value| {
                value
                    .parse::<usize>()
                    .expect("capture layer must be an integer")
            })
            .unwrap_or(0);
        part.name.starts_with(&format!("blk.{layer}."))
    };
    if !wanted {
        return;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open layer-zero capture");
    serde_json::to_writer(&mut file, &serde_json::json!({"name": part.name, "expert": expert, "type": format!("{:?}", part.ty), "input": x, "output": out})).expect("write layer-zero capture");
    std::io::Write::write_all(&mut file, b"\n").expect("finish layer-zero record");
}

/// Marks token boundaries for opt-in prefix or exact-layer captures.
/// The driver must create a new, empty file before the first token.
#[cfg(test)]
pub(super) fn capture_token_boundary(position: usize, token: u32) {
    if std::env::var("FIFI_FLASH_GGUF_CAPTURE_PREFIX").is_err()
        && std::env::var_os("FIFI_FLASH_GGUF_CAPTURE_MARK_TOKENS").is_none()
    {
        return;
    }
    let Some(path) = std::env::var_os("FIFI_FLASH_GGUF_LAYER0_CAPTURE") else {
        return;
    };
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .expect("open prefix capture");
    if position == 0 {
        assert_eq!(
            file.metadata().expect("capture metadata").len(),
            0,
            "prefix capture must start empty"
        );
    }
    serde_json::to_writer(
        &mut file,
        &serde_json::json!({"position": position, "token": token}),
    )
    .expect("write capture token boundary");
    std::io::Write::write_all(&mut file, b"\n").expect("finish capture token boundary");
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
        #[cfg(test)]
        if std::env::var_os("FIFI_FLASH_GGUF_TRACE_ROUTES").is_some()
            && stack.parts[0].name.contains("ffn_gate_exps")
        {
            eprintln!("expert route: name={} expert={expert}", stack.parts[0].name);
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
