//! Read-only weight access for the Inkling short-context CPU reference.

use std::{ops::Range, path::Path};

use super::{TextSpec, bad, checkpoint};
use crate::{
    EngError,
    family::mapped_file::MappedFile,
    kernels::{
        nvfp4::E4M3_VALUES,
        nvfp4_cpu::{Activation, Isa, Nvfp4Matrix},
    },
    weights::nemotron_shards::NemotronShardReader as ShardReader,
};

pub(super) struct Weights {
    metadata: ShardReader,
    mapping: MappedFile,
    isa: Isa,
}

#[derive(Clone, Copy)]
struct Floats<'a> {
    bytes: &'a [u8],
    width: usize,
}

impl Floats<'_> {
    fn len(self) -> usize {
        self.bytes.len() / self.width
    }

    fn values(self) -> impl ExactSizeIterator<Item = f32> {
        self.bytes.chunks_exact(self.width).map(move |bytes| {
            if self.width == 2 {
                f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)
            } else {
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            }
        })
    }

    fn slice(self, range: Range<usize>) -> Option<Self> {
        Some(Self {
            bytes: self
                .bytes
                .get(range.start.checked_mul(self.width)?..range.end.checked_mul(self.width)?)?,
            width: self.width,
        })
    }
}

impl Weights {
    pub(super) fn open(directory: &Path, spec: &TextSpec) -> Result<Self, EngError> {
        let (path, metadata) = checkpoint::open_checkpoint(directory, spec)?;
        let mapping = MappedFile::open(&path, metadata.file_len())
            .map_err(|error| bad(format!("map text weights: {error}")))?;
        Ok(Self {
            metadata,
            mapping,
            isa: Isa::detect(),
        })
    }

    fn bytes(&self, name: &str) -> Result<&[u8], EngError> {
        self.metadata
            .absolute_range(name)
            .and_then(|range| self.mapping.as_ref().get(range))
            .ok_or_else(|| bad(format!("missing or out-of-range tensor {name}")))
    }

    fn floats(&self, name: &str) -> Result<Floats<'_>, EngError> {
        let width = match self.metadata.tensor(name).map(|tensor| tensor.dtype()) {
            Some("BF16") => 2,
            Some("F32") => 4,
            _ => return Err(bad(format!("expected floating weights in {name}"))),
        };
        Ok(Floats {
            width,
            bytes: self.bytes(name)?,
        })
    }

    pub(super) fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>, EngError> {
        let data = self.floats(name)?;
        if data.len() != len {
            return Err(bad(format!(
                "{name} has {} elements, expected {len}",
                data.len()
            )));
        }
        Ok(data.values().collect())
    }

    pub(super) fn row(&self, name: &str, row: usize, cols: usize) -> Result<Vec<f32>, EngError> {
        let data = self.floats(name)?;
        let start = row
            .checked_mul(cols)
            .ok_or_else(|| bad("row offset overflows"))?;
        let end = start
            .checked_add(cols)
            .ok_or_else(|| bad("row span overflows"))?;
        data.slice(start..end)
            .map(|range| range.values().collect())
            .ok_or_else(|| bad(format!("{name} row {row} escapes the tensor")))
    }

    pub(super) fn project(&self, name: &str, x: &[f32], rows: usize) -> Result<Vec<f32>, EngError> {
        let data = self.floats(name)?;
        let [stored_rows, stored_cols] = self
            .metadata
            .tensor(name)
            .ok_or_else(|| bad(format!("missing {name}")))?
            .shape()
        else {
            return Err(bad(format!("{name} is not a matrix")));
        };
        if *stored_cols != x.len() as u64 || *stored_rows < rows as u64 || x.is_empty() {
            return Err(bad(format!(
                "{name} projection dimensions disagree with its input or output"
            )));
        }
        let bytes_per_row = data.width * x.len();
        Ok(data
            .bytes
            .chunks_exact(bytes_per_row)
            .take(rows)
            .map(|bytes| {
                Floats {
                    bytes,
                    width: data.width,
                }
                .values()
                .zip(x)
                .map(|(w, value)| w * value)
                .sum::<f32>()
            })
            .collect())
    }

    pub(super) fn activate(&self, x: &[f32]) -> Result<Activation, EngError> {
        Activation::new(x).map_err(|error| bad(format!("prep expert activation: {error}")))
    }

    pub(super) fn expert(
        &self,
        name: &str,
        expert: usize,
        input: &Activation,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let tensor = self
            .metadata
            .tensor(name)
            .ok_or_else(|| bad(format!("missing {name}")))?;
        let [experts, rows, cols] = tensor.shape() else {
            return Err(bad(format!("{name} must have expert, row and column axes")));
        };
        if expert >= *experts as usize {
            return Err(bad(format!("expert {expert} escapes {name}")));
        }
        let rows = *rows as usize;
        let cols = *cols as usize;
        match tensor.dtype() {
            "U8" => self.nvfp4_expert(name, expert, input, rows, cols),
            "F8_E4M3" => self.mxfp8_expert(name, expert, x, rows, cols),
            _ => Err(bad(format!("unsupported expert precision for {name}"))),
        }
    }

    fn expert_slice(&self, name: &str, expert: usize, len: usize) -> Result<&[u8], EngError> {
        let start = expert
            .checked_mul(len)
            .ok_or_else(|| bad("expert offset overflows"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| bad("expert span overflows"))?;
        self.bytes(name)?
            .get(start..end)
            .ok_or_else(|| bad(format!("expert {expert} escapes {name}")))
    }

    fn nvfp4_expert(
        &self,
        name: &str,
        expert: usize,
        input: &Activation,
        rows: usize,
        packed_cols: usize,
    ) -> Result<Vec<f32>, EngError> {
        let packed = self.expert_slice(name, expert, rows * packed_cols)?;
        let scales = self.expert_slice(&format!("{name}_scale"), expert, rows * packed_cols / 8)?;
        let amax = self.row(&format!("{name}_amax"), expert, 1)?[0];
        let divisor = self.vec(&format!("{name}_outer_scale_denominator"), 1)?[0];
        if !amax.is_finite() || amax < 0.0 || !divisor.is_finite() || divisor <= 0.0 {
            return Err(bad(format!("invalid NVFP4 outer scale for {name}")));
        }
        let mut y = vec![0.0; rows];
        if amax == 0.0 {
            return Ok(y);
        }
        Nvfp4Matrix::new(packed, scales, amax / divisor, rows, packed_cols * 2)
            .and_then(|matrix| matrix.matvec_activation(self.isa, input, &mut y))
            .map_err(|error| bad(format!("{name}: {error}")))?;
        Ok(y)
    }

    fn mxfp8_expert(
        &self,
        name: &str,
        expert: usize,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, EngError> {
        if cols != x.len() || !cols.is_multiple_of(32) {
            return Err(bad(format!("invalid MXFP8 input for {name}")));
        }
        let packed = self.expert_slice(name, expert, rows * cols)?;
        let scales = self.expert_slice(&format!("{name}_scale"), expert, rows * cols / 32)?;
        packed
            .chunks_exact(cols)
            .zip(scales.chunks_exact(cols / 32))
            .map(|(row, scales)| mxfp8_dot(row, scales, x))
            .collect()
    }
}

pub(super) fn mxfp8_dot(weights: &[u8], scales: &[u8], x: &[f32]) -> Result<f32, EngError> {
    if x.is_empty()
        || weights.len() != x.len()
        || !x.len().is_multiple_of(32)
        || scales.len() != x.len() / 32
    {
        return Err(bad("invalid MXFP8 row geometry"));
    }
    let mut sum = 0.0;
    let (weights, _) = weights.as_chunks::<32>();
    let (x, _) = x.as_chunks::<32>();
    for ((weights, x), &code) in weights.iter().zip(x).zip(scales) {
        if code == 255 {
            return Err(bad("MXFP8 E8M0 scale is NaN"));
        }
        // E8M0 byte zero encodes 2^-127, an F32 subnormal.
        let scale = f32::from_bits(if code == 0 {
            1 << 22
        } else {
            u32::from(code) << 23
        });
        let dot: f32 = weights
            .iter()
            .zip(x)
            .map(|(&weight, value)| E4M3_VALUES[weight as usize] * value)
            .sum();
        sum += scale * dot;
    }
    if !sum.is_finite() {
        return Err(bad("non-finite MXFP8 projection"));
    }
    Ok(sum)
}

#[test]
fn mxfp8_dot_applies_block_exponents_without_expanding_experts() {
    let mut weights = [0x38; 64]; // E4M3 1.0
    weights[32..].fill(0xb8); // E4M3 -1.0
    let x = [1.0; 64];
    assert_eq!(mxfp8_dot(&weights, &[127, 126], &x).unwrap(), 16.0);
    assert!(mxfp8_dot(&weights, &[127, 255], &x).is_err());
    assert!(mxfp8_dot(&weights, &[127], &x).is_err());
}
