//! The n-gram memory of a Flash-Next GGUF: `per_layer_token_embd.weight`, one tensor of
//! `rows x 160` values (320,001,536 rows, 28.8 GB in IQ4_NL) that the hashed n-grams of the
//! token history index. Rows are read from the shard file in place, one positioned read per
//! row, and decoded on the host with the same CPU dequantizer the rest of the GGUF code uses.
//!
//! The table never sits in memory: a row is `160 / 32` blocks of 18 bytes (90 bytes) in IQ4_NL,
//! and the page cache keeps whatever the access pattern makes hot. The hash constants that pick
//! the rows come from the metadata ([`NgramHash`](super::config::NgramHash)); the rows are
//! stored as plain values, with none of the per-head bias and per-row scale the EXL3 table
//! folds into its decode.

use std::{
    fs::File,
    os::unix::fs::{FileExt, MetadataExt},
};

use super::{config::NgramHash, tensors::Mapped};
use crate::{
    EngError,
    weights::{GgufType, GgufWeights, dequantize_into},
};

fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::Checkpoint(format!("Flash-Next GGUF n-gram table: {}", detail.as_ref()))
}

/// Rows of the n-gram table, readable from any thread.
#[derive(Debug)]
pub(in crate::family::qwen::flash_next) struct GgufNgramRows {
    file: File,
    /// Absolute file offset of row 0.
    base: u64,
    rows: u64,
    row_dim: usize,
    row_bytes: usize,
    ty: GgufType,
    hash: NgramHash,
}

impl GgufNgramRows {
    /// Opens the shard file that holds the table (through the local-storage guard) and checks
    /// the tensor's range against the file length. `table` is the mapped
    /// `per_layer_token_embd.weight`.
    pub(in crate::family::qwen::flash_next) fn open(
        weights: &GgufWeights,
        table: &Mapped,
        hash: NgramHash,
    ) -> Result<Self, EngError> {
        let tensor = weights
            .header()
            .tensor(&table.name)
            .ok_or_else(|| refuse(format!("{} is not in the file", table.name)))?;
        let [row_dim, rows] = tensor.dims() else {
            return Err(refuse(format!(
                "{} has {} dimensions, the table is two-dimensional",
                table.name,
                tensor.dims().len()
            )));
        };
        let ty = tensor.ggml_type();
        let row_dim = usize::try_from(*row_dim).map_err(|_| refuse("row width overflows"))?;
        let block = ty.block_elems();
        if !(row_dim as u64).is_multiple_of(block) {
            return Err(refuse(format!(
                "a row of {row_dim} values is not whole {} blocks",
                ty.name()
            )));
        }
        let row_bytes = usize::try_from(row_dim as u64 / block * ty.block_bytes())
            .map_err(|_| refuse("row size overflows"))?;
        let path = weights
            .paths()
            .get(tensor.shard())
            .ok_or_else(|| refuse("the table's shard is not open"))?;
        crate::storage_guard::require_local(path, "n-gram table")?;
        // Preserve the shard identity already opened by GgufWeights, even if its path changes.
        let file = weights
            .file_of(tensor)
            .try_clone()
            .map_err(|error| refuse(format!("cannot clone {}: {error}", path.display())))?;
        let len = file
            .metadata()
            .map_err(|error| refuse(format!("cannot stat {}: {error}", path.display())))?
            .len();
        if tensor.abs_range().end > len {
            return Err(refuse(format!(
                "{} ends at byte {}, past the {len} bytes of {}",
                table.name,
                tensor.abs_range().end,
                path.display()
            )));
        }
        Ok(Self {
            file,
            base: tensor.abs_offset(),
            rows: *rows,
            row_dim,
            row_bytes,
            ty,
            hash,
        })
    }

    /// Check that two readers address the same opened shard and table geometry.
    pub(super) fn same_source(&self, other: &Self) -> Result<bool, EngError> {
        let a = self
            .file
            .metadata()
            .map_err(|e| refuse(format!("cannot stat reader: {e}")))?;
        let b = other
            .file
            .metadata()
            .map_err(|e| refuse(format!("cannot stat reader: {e}")))?;
        Ok(a.dev() == b.dev()
            && a.ino() == b.ino()
            && self.base == other.base
            && self.rows == other.rows
            && self.row_dim == other.row_dim
            && self.row_bytes == other.row_bytes
            && self.ty == other.ty
            && self.hash == other.hash)
    }

    /// Rows in the table.
    pub(in crate::family::qwen::flash_next) fn rows(&self) -> u64 {
        self.rows
    }

    /// Values in one row.
    pub(in crate::family::qwen::flash_next) fn row_dim(&self) -> usize {
        self.row_dim
    }

    /// Stored bytes of one row.
    pub(in crate::family::qwen::flash_next) fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// The hash constants that map n-grams to rows.
    pub(in crate::family::qwen::flash_next) fn hash(&self) -> &NgramHash {
        &self.hash
    }

    /// Reads and decodes global row `row` into `out` (`row_dim` values).
    pub(in crate::family::qwen::flash_next) fn row_into(
        &self,
        row: u64,
        out: &mut [f32],
    ) -> Result<(), EngError> {
        if row >= self.rows {
            return Err(refuse(format!(
                "row {row} is outside the {} rows",
                self.rows
            )));
        }
        if out.len() != self.row_dim {
            return Err(refuse(format!(
                "an output of {} values for rows of {}",
                out.len(),
                self.row_dim
            )));
        }
        let mut packed = vec![0u8; self.row_bytes];
        self.file
            .read_exact_at(&mut packed, self.base + row * self.row_bytes as u64)
            .map_err(|error| refuse(format!("row {row}: {error}")))?;
        dequantize_into(self.ty, &packed, out)
    }

    /// Reads and decodes global row `row`.
    pub(in crate::family::qwen::flash_next) fn row(&self, row: u64) -> Result<Vec<f32>, EngError> {
        let mut out = vec![0f32; self.row_dim];
        self.row_into(row, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen::flash_next::load::gguf::testkit::tiny_table;

    /// One IQ4_NL block: scale 0.5, element `j` takes the low nibble of byte `j`, element
    /// `j + 16` the high nibble.
    fn block(nibbles: [u8; 32]) -> [u8; 18] {
        let mut out = [0u8; 18];
        out[..2].copy_from_slice(&0x3800u16.to_le_bytes());
        for j in 0..16 {
            out[2 + j] = nibbles[j] | (nibbles[j + 16] << 4);
        }
        out
    }

    /// The format's value table, written out from the specification.
    const KVALUES: [i32; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];

    #[test]
    fn rows_decode_from_the_file_by_position() {
        let rows = 7u64;
        // Row r, block b, element e holds the nibble (r + b + e) % 16.
        let mut payload = Vec::new();
        let mut want = Vec::new();
        for r in 0..rows as usize {
            for b in 0..5usize {
                let nibbles: [u8; 32] = std::array::from_fn(|e| ((r + b + e) % 16) as u8);
                payload.extend_from_slice(&block(nibbles));
                want.extend(nibbles.iter().map(|&n| 0.5 * KVALUES[n as usize] as f32));
            }
        }
        let fixture = tiny_table(GgufType::Iq4Nl, rows, payload);
        let table = GgufNgramRows::open(&fixture.weights, &fixture.table, fixture.hash.clone())
            .expect("open the table");
        assert_eq!(
            (table.rows(), table.row_dim(), table.row_bytes()),
            (rows, 160, 90)
        );
        for r in (0..rows as usize).rev() {
            let got = table.row(r as u64).expect("row");
            assert_eq!(got, want[r * 160..(r + 1) * 160], "row {r}");
        }
    }

    #[test]
    fn a_row_outside_the_table_and_a_wrong_output_width_are_refused() {
        let fixture = tiny_table(GgufType::Iq4Nl, 2, vec![0; 2 * 90]);
        let table = GgufNgramRows::open(&fixture.weights, &fixture.table, fixture.hash.clone())
            .expect("open");
        let error = table.row(2).expect_err("past the end").describe();
        assert!(error.contains("row 2 is outside the 2 rows"), "{error}");
        let error = table
            .row_into(0, &mut [0.0; 3])
            .expect_err("width")
            .describe();
        assert!(error.contains("3 values for rows of 160"), "{error}");
    }

    #[test]
    fn other_row_types_decode_too() {
        // F16 rows: 160 values of 2 bytes.
        let mut payload = Vec::new();
        for r in 0..3u16 {
            for e in 0..160u16 {
                payload.extend_from_slice(&half_bits(f32::from(r * 160 + e)).to_le_bytes());
            }
        }
        let fixture = tiny_table(GgufType::F16, 3, payload);
        let table = GgufNgramRows::open(&fixture.weights, &fixture.table, fixture.hash.clone())
            .expect("open");
        assert_eq!(table.row_bytes(), 320);
        let row = table.row(2).expect("row");
        assert_eq!(row[0], 320.0);
        assert_eq!(row[159], 479.0);
    }

    /// F16 bits of a small non-negative integer.
    fn half_bits(value: f32) -> u16 {
        if value == 0.0 {
            return 0;
        }
        let bits = value.to_bits();
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mantissa = (bits >> 13) & 0x3ff;
        ((exponent as u16) << 10) | mantissa as u16
    }
}
