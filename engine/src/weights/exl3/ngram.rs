//! Hashed n-gram embedding tables stored in the `exl3_ngram_trellis` format, version 1.
//!
//! The Qwen3.8 Flash-Next EXL3 packs keep the PLE n-gram table in a separate safetensors file
//! (`ngram_embedding.safetensors`). Each table row is one tail-biting trellis ring over the mul1
//! codebook, packed as `1 + ROW_DIM * K / 16` little-endian 16-bit words:
//!
//! - word 0 holds the row scale as binary16 bits;
//! - the other words hold a `ROW_DIM * K`-bit stream, least significant bit first. Bits
//!   `[i * K, (i + 1) * K)` are the symbol of position `i`.
//!
//! The 16-bit trellis state of position `i` stacks the symbols of positions `i, i - 1, i - 2, ...`
//! (modulo `ROW_DIM`) from bit 0 upwards, `K` bits each. A decoded value is
//! `mul1(state_i) * scale + head_bias[head][i]`, in `f32`, where `head` is the hash head whose
//! global row range holds the row.
//!
//! This module reads the header and decodes rows that the caller has read from the file. It does
//! not hash token histories; the family code computes the global row numbers.

// Staged: the Flash-Next EXL3 loader is the first production caller.
#![cfg_attr(not(test), allow(dead_code))]

use std::ops::Range;

use half::f16;
use serde_json::{Map, Value};

use super::Codebook;
use crate::{
    EngError,
    tensor::Dtype,
    weights::{HeaderEntry, ShardHeader, header::parse_header_json},
};

/// Values per table row. The format fixes it.
pub(crate) const ROW_DIM: usize = 160;
const FORMAT: &str = "exl3_ngram_trellis";
const VERSION: &str = "1";
const MAX_BITS: u32 = 8;

/// What the model configuration requires of the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NgramLayout {
    /// Hash heads: `(ngram_size - 1) * heads_per_ngram`.
    pub(crate) heads: usize,
    /// Values per head row: `ple_embed_dim / heads`.
    pub(crate) row_dim: usize,
    /// Tokens per n-gram, which is also the number of hash multipliers.
    pub(crate) ngram: usize,
}

/// The small tensors of the table file that are read once at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NgramAux {
    HeadBias,
    HeadOffsets,
    HeadVocabSizes,
    LayerMultipliers,
}

impl NgramAux {
    const ALL: [Self; 4] = [
        Self::HeadBias,
        Self::HeadOffsets,
        Self::HeadVocabSizes,
        Self::LayerMultipliers,
    ];

    fn suffix(self) -> &'static str {
        match self {
            Self::HeadBias => "head_bias",
            Self::HeadOffsets => "head_offsets",
            Self::HeadVocabSizes => "head_vocab_sizes",
            Self::LayerMultipliers => "layer_multipliers",
        }
    }
}

/// A validated `exl3_ngram_trellis` file: its geometry and the absolute byte ranges of its tensors.
#[derive(Debug, Clone)]
pub(crate) struct NgramTable {
    bits: u32,
    words: usize,
    rows: u64,
    shard_rows: u64,
    /// Absolute file byte range of each shard's packed rows, in global row order.
    shards: Vec<Range<u64>>,
    /// Absolute file byte ranges of the auxiliary tensors, in `NgramAux::ALL` order.
    aux: [Range<u64>; 4],
    layout: NgramLayout,
}

impl NgramTable {
    /// Validates the file header of the table under `key`.
    ///
    /// `prefix_and_header` holds the 8-byte safetensors prefix and the JSON header. `byte_len` is
    /// the length of the whole file, which must end exactly where the last tensor ends.
    pub(crate) fn from_header(
        prefix_and_header: &[u8],
        byte_len: u64,
        key: &str,
        layout: NgramLayout,
    ) -> Result<Self, EngError> {
        if layout.row_dim != ROW_DIM {
            return Err(invalid(&format!(
                "the model needs {}-value rows; the format stores {ROW_DIM}",
                layout.row_dim
            )));
        }
        let header = ShardHeader::read(
            &mut &prefix_and_header[..],
            byte_len,
            prefix_and_header.len() as u64,
        )?;
        // ShardHeader keeps the tensor entries only, so the metadata is read from the JSON again.
        let metadata = read_metadata(prefix_and_header)?;
        let bits = parse_metadata(&metadata)?;
        let words = 1 + ROW_DIM * bits as usize / 16;
        let rows = metadata_number(&metadata, "rows")?;
        let shard_rows = metadata_number(&metadata, "shard_rows")?;
        let found = header.iter().count();
        let shard_count = shard_count(rows, shard_rows, found)?;
        let shards = shard_ranges(&header, key, (shard_rows, words as u64), shard_count)?;
        let aux = aux_ranges(&header, key, layout)?;
        let expected_tensors = shard_count + NgramAux::ALL.len();
        if found != expected_tensors {
            return Err(invalid(&format!(
                "the file holds {found} tensors; the table needs exactly {expected_tensors}"
            )));
        }
        Ok(Self {
            bits,
            words,
            rows,
            shard_rows,
            shards,
            aux,
            layout,
        })
    }

    pub(crate) fn rows(&self) -> u64 {
        self.rows
    }

    /// Bits per stored value (`K`).
    pub(crate) fn bits(&self) -> u32 {
        self.bits
    }

    /// Bytes of one packed row.
    pub(crate) fn row_bytes(&self) -> usize {
        self.words * 2
    }

    /// Absolute file byte range of one global row.
    pub(crate) fn row_range(&self, row: u64) -> Result<Range<u64>, EngError> {
        if row >= self.rows {
            return Err(EngError::BadRequest(format!(
                "n-gram row {row} is outside the {} table rows",
                self.rows
            )));
        }
        let shard = &self.shards[(row / self.shard_rows) as usize];
        let start = shard.start + (row % self.shard_rows) * self.row_bytes() as u64;
        Ok(start..start + self.row_bytes() as u64)
    }

    /// Absolute file byte range of one auxiliary tensor.
    pub(crate) fn aux_range(&self, which: NgramAux) -> Range<u64> {
        // NgramAux::ALL lists the variants in declaration order, so the discriminant indexes it.
        self.aux[which as usize].clone()
    }
}

/// Decodes rows of one validated table.
#[derive(Debug, Clone)]
pub(crate) struct NgramDecoder {
    bits: u32,
    words: usize,
    rows: u64,
    /// First global row of each head.
    head_offsets: Vec<u64>,
    /// One past the last global row of each head; rows between heads belong to none.
    head_ends: Vec<u64>,
    /// `[heads, ROW_DIM]` row-major.
    head_bias: Vec<f32>,
}

impl NgramDecoder {
    /// `head_offsets` and `head_vocab_sizes` are the decoded I64 tensors and `head_bias` the
    /// decoded binary16 bits of the file's `head_bias` tensor.
    pub(crate) fn new(
        table: &NgramTable,
        head_offsets: &[i64],
        head_vocab_sizes: &[i64],
        head_bias: &[u16],
    ) -> Result<Self, EngError> {
        let heads = table.layout.heads;
        if head_offsets.len() != heads
            || head_vocab_sizes.len() != heads
            || head_bias.len() != heads * ROW_DIM
        {
            return Err(invalid("hash head tensors disagree with the head count"));
        }
        let mut offsets = Vec::with_capacity(heads);
        let mut ends = Vec::with_capacity(heads);
        let mut next = 0u64;
        let misplaced =
            || invalid("hash head row ranges must be nonempty, ordered, and inside the table");
        for (&offset, &size) in head_offsets.iter().zip(head_vocab_sizes) {
            let (Ok(offset), Ok(size)) = (u64::try_from(offset), u64::try_from(size)) else {
                return Err(invalid("a hash head offset or vocabulary size is negative"));
            };
            let Some(end) = offset.checked_add(size).filter(|&end| end <= table.rows) else {
                return Err(misplaced());
            };
            if size == 0 || offset < next {
                return Err(misplaced());
            }
            offsets.push(offset);
            ends.push(end);
            next = end;
        }
        let head_bias: Vec<f32> = head_bias
            .iter()
            .map(|&bits| f16::from_bits(bits).to_f32())
            .collect();
        if head_bias.iter().any(|value| !value.is_finite()) {
            return Err(invalid("head_bias holds a value that is not finite"));
        }
        Ok(Self {
            bits: table.bits,
            words: table.words,
            rows: table.rows,
            head_offsets: offsets,
            head_ends: ends,
            head_bias,
        })
    }

    /// The hash head whose row range holds `row`: the last head that starts at or before it, if
    /// the row is before that head's end. A row past every head (the table's padding rows) or in
    /// a gap between heads is refused.
    pub(crate) fn head_of(&self, row: u64) -> Result<usize, EngError> {
        let after = self.head_offsets.partition_point(|&offset| offset <= row);
        if row >= self.rows || after == 0 || row >= self.head_ends[after - 1] {
            return Err(EngError::BadRequest(format!(
                "n-gram row {row} belongs to no hash head"
            )));
        }
        Ok(after - 1)
    }

    /// Decodes the packed little-endian bytes of global row `row` into `out`.
    pub(crate) fn decode_row(
        &self,
        row: u64,
        packed: &[u8],
        out: &mut [f32; ROW_DIM],
    ) -> Result<(), EngError> {
        if packed.len() != self.words * 2 {
            return Err(EngError::BadRequest(format!(
                "an n-gram row is {} bytes, received {}",
                self.words * 2,
                packed.len()
            )));
        }
        let head = self.head_of(row)?;
        let word = |index: usize| u16::from_le_bytes([packed[2 * index], packed[2 * index + 1]]);
        let scale = f16::from_bits(word(0)).to_f32();
        if !scale.is_finite() {
            return Err(EngError::Checkpoint(format!(
                "n-gram row {row} has a scale that is not finite"
            )));
        }
        let symbols = self.symbols(|index| word(1 + index));
        let bias = &self.head_bias[head * ROW_DIM..(head + 1) * ROW_DIM];
        // Symbols that reach into the 16-bit state; the cast below keeps its low 16 bits.
        let per_state = 16usize.div_ceil(self.bits as usize);
        for (position, value) in out.iter_mut().enumerate() {
            let mut state = 0u32;
            for back in 0..per_state {
                let symbol = symbols[(position + ROW_DIM - back) % ROW_DIM];
                state |= u32::from(symbol) << (back as u32 * self.bits);
            }
            let code = f16::from_bits(Codebook::Mul1.decode_state(state as u16)).to_f32();
            *value = code * scale + bias[position];
        }
        Ok(())
    }

    /// The `K`-bit symbol of every ring position, read through a two-word window so a symbol may
    /// straddle a word boundary; the window wraps at the end of the ring.
    fn symbols(&self, stream_word: impl Fn(usize) -> u16) -> [u8; ROW_DIM] {
        let stream_words = self.words - 1;
        let mask = (1u32 << self.bits) - 1;
        let mut symbols = [0u8; ROW_DIM];
        for (position, symbol) in symbols.iter_mut().enumerate() {
            let bit = position * self.bits as usize;
            let low = u32::from(stream_word(bit / 16));
            let high = u32::from(stream_word((bit / 16 + 1) % stream_words));
            *symbol = (((low | (high << 16)) >> (bit % 16)) & mask) as u8;
        }
        symbols
    }
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("{FORMAT} table: {message}"))
}

fn read_metadata(prefix_and_header: &[u8]) -> Result<Map<String, Value>, EngError> {
    let json = prefix_and_header
        .get(8..)
        .ok_or_else(|| invalid("the header is shorter than its prefix"))?;
    let root = parse_header_json(json, "n-gram table header")?;
    root.get("__metadata__")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| invalid("the header has no metadata"))
}

fn metadata_text<'a>(metadata: &'a Map<String, Value>, name: &str) -> Result<&'a str, EngError> {
    metadata
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(&format!("metadata field {name} is missing")))
}

/// A canonical decimal: digits only, no leading zero but for "0" itself, within u64.
fn metadata_number(metadata: &Map<String, Value>, name: &str) -> Result<u64, EngError> {
    let text = metadata_text(metadata, name)?;
    let canonical =
        text.bytes().all(|byte| byte.is_ascii_digit()) && (text == "0" || !text.starts_with('0'));
    text.parse()
        .ok()
        .filter(|_| canonical)
        .ok_or_else(|| invalid(&format!("metadata field {name} is not a decimal integer")))
}

/// Checks the format identity and returns `K`.
fn parse_metadata(metadata: &Map<String, Value>) -> Result<u32, EngError> {
    let format = metadata_text(metadata, "format")?;
    if format != FORMAT {
        return Err(invalid(&format!("unsupported format {format:?}")));
    }
    let version = metadata_text(metadata, "version")?;
    if version != VERSION {
        return Err(invalid(&format!("unsupported version {version:?}")));
    }
    let codebook = metadata_text(metadata, "codebook")?;
    if codebook != "mul1" {
        return Err(invalid(&format!("unsupported codebook {codebook:?}")));
    }
    let row_dim = metadata_number(metadata, "row_dim")?;
    if row_dim != ROW_DIM as u64 {
        return Err(invalid(&format!("unsupported row_dim {row_dim}")));
    }
    let bits = metadata_number(metadata, "K")?;
    if !(1..=u64::from(MAX_BITS)).contains(&bits) {
        return Err(invalid(&format!("unsupported K {bits}")));
    }
    Ok(bits as u32)
}

/// The shard count of `rows` rows in shards of `shard_rows`, refused unless it divides exactly and
/// fits in the `tensors` the header lists: the header is untrusted, and the count sizes a vector.
fn shard_count(rows: u64, shard_rows: u64, tensors: usize) -> Result<usize, EngError> {
    if shard_rows == 0 || rows == 0 || !rows.is_multiple_of(shard_rows) {
        return Err(invalid(&format!(
            "{rows} rows do not divide into shards of {shard_rows}"
        )));
    }
    let count = rows / shard_rows;
    if count > tensors as u64 {
        return Err(invalid(&format!(
            "{rows} rows in shards of {shard_rows} need {count} shard tensors; the header lists \
             {tensors} tensors"
        )));
    }
    // At most `tensors`, so it fits usize.
    Ok(count as usize)
}

/// The byte ranges of the `count` shard tensors, `[shard_rows, words]` I16 each, in row order.
fn shard_ranges(
    header: &ShardHeader,
    key: &str,
    (shard_rows, words): (u64, u64),
    count: usize,
) -> Result<Vec<Range<u64>>, EngError> {
    (0..count)
        .map(|index| {
            let name = format!("{key}.shard_{index}.trellis");
            let entry = expect_entry(header, &name, Dtype::I16, &[shard_rows, words])?;
            Ok(absolute(header.data_start(), entry))
        })
        .collect()
}

/// The byte ranges of the auxiliary tensors, in `NgramAux::ALL` order.
fn aux_ranges(
    header: &ShardHeader,
    key: &str,
    layout: NgramLayout,
) -> Result<[Range<u64>; 4], EngError> {
    let heads = layout.heads as u64;
    let shapes: [&[u64]; 4] = [
        &[heads, ROW_DIM as u64],
        &[heads],
        &[heads],
        &[layout.ngram as u64],
    ];
    let dtypes = [Dtype::F16, Dtype::I64, Dtype::I64, Dtype::I64];
    let mut aux = [0..0, 0..0, 0..0, 0..0];
    for (slot, ((which, dtype), shape)) in aux
        .iter_mut()
        .zip(NgramAux::ALL.iter().zip(dtypes).zip(shapes))
    {
        let name = format!("{key}.{}", which.suffix());
        *slot = absolute(
            header.data_start(),
            expect_entry(header, &name, dtype, shape)?,
        );
    }
    Ok(aux)
}

fn expect_entry<'a>(
    header: &'a ShardHeader,
    name: &str,
    dtype: Dtype,
    shape: &[u64],
) -> Result<&'a HeaderEntry, EngError> {
    let entry = header
        .entry(name)
        .ok_or_else(|| invalid(&format!("tensor {name} is missing")))?;
    let found: Vec<u64> = entry.shape().iter().map(|&dim| u64::from(dim)).collect();
    if entry.dtype() != dtype || found != shape {
        return Err(invalid(&format!(
            "tensor {name} is {:?} {found:?}; the table needs {dtype:?} {shape:?}",
            entry.dtype()
        )));
    }
    Ok(entry)
}

fn absolute(data_start: u64, entry: &HeaderEntry) -> Range<u64> {
    let offsets = entry.offsets();
    data_start + offsets.start..data_start + offsets.end
}

#[cfg(test)]
#[path = "ngram_tests.rs"]
mod tests;
