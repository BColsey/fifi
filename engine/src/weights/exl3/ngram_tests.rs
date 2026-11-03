//! Expected values come from the committed real header and rows of the Qwen3.8 Flash-Next EXL3
//! pack (see `engine/tests/fixtures/README.md`), from the official BF16 source rows, or from hand
//! packing below; none comes from the decoder under test.

use std::ops::Range;

use half::f16;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::{NgramAux, NgramDecoder, NgramLayout, NgramTable, ROW_DIM};
use crate::{EngError, weights::exl3::Codebook};

const KEY: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
const LAYOUT: NgramLayout = NgramLayout {
    heads: 16,
    row_dim: 160,
    ngram: 3,
};

const HEADER: &[u8] =
    include_bytes!("../../../tests/fixtures/flash_next_exl3_ngram/ngram_embedding.header");
const MANIFEST: &[u8] =
    include_bytes!("../../../tests/fixtures/flash_next_exl3_ngram/manifest.json");
const PACKED: &[u8] =
    include_bytes!("../../../tests/fixtures/flash_next_exl3_ngram/packed_u16.bin");
const DECODED: &[u8] =
    include_bytes!("../../../tests/fixtures/flash_next_exl3_ngram/decoded_f32.bin");
const SOURCE: &[u8] =
    include_bytes!("../../../tests/fixtures/flash_next_exl3_ngram/source_bf16.bin");
const BIAS: &[u8] =
    include_bytes!("../../../tests/fixtures/flash_next_exl3_ngram/head_bias_f16.bin");

/// Length of the real `ngram_embedding.safetensors` (Hub tree listing of the pinned revision).
const REAL_FILE_BYTES: u64 = 32_640_183_408;

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

fn manifest() -> Value {
    serde_json::from_slice(MANIFEST).unwrap()
}

fn integers(value: &Value, name: &str) -> Vec<i64> {
    value[name]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}

fn real_table() -> NgramTable {
    NgramTable::from_header(HEADER, REAL_FILE_BYTES, KEY, LAYOUT).unwrap()
}

fn real_decoder(bias: &[u16]) -> NgramDecoder {
    let manifest = manifest();
    NgramDecoder::new(
        &real_table(),
        &integers(&manifest, "head_offsets"),
        &integers(&manifest, "head_vocab_sizes"),
        bias,
    )
    .unwrap()
}

#[test]
fn the_fixture_files_are_the_recorded_bytes() {
    for (bytes, digest) in [
        (
            HEADER,
            "a2c0cbede2bad9435fa2890bb8cf40a942d2df1517e1a84ce19da8429e6ae59b",
        ),
        (
            MANIFEST,
            "d7eb4132c3a5d998426f8af9f5cc2e30c69fc4e2e7e9c14774976df64edd5a33",
        ),
        (
            PACKED,
            "08ab8ead3de6f11e4fc8e7c4eabb9768f041ee8c26621b603933b106e64af301",
        ),
        (
            DECODED,
            "8de02d8763890f4225e1d039049cfcb1c6d82e23fa2c6ccd9098290b20f2283f",
        ),
        (
            SOURCE,
            "526c444aabc2c97311b640000c6333ea8700f2fe7d6c48185d9ed864ffe9672a",
        ),
        (
            BIAS,
            "c1e3097f5c1e7996e2198ddca3a7244a09868072c5538a6a34fade6b3c8b1899",
        ),
    ] {
        assert_eq!(sha256(bytes), digest);
    }
}

#[test]
fn the_real_header_validates_to_the_published_geometry() {
    let table = real_table();
    assert_eq!(table.rows(), 320_001_536);
    assert_eq!(table.bits(), 5);
    assert_eq!(table.row_bytes(), 102);
    // The data section starts after the 8-byte prefix and the 21,328-byte JSON header.
    assert_eq!(table.row_range(0).unwrap(), 21_336 + 5_400..21_336 + 5_502);
    assert_eq!(
        table.row_range(2_500_012).unwrap(),
        21_336 + 255_006_624..21_336 + 255_006_726
    );
    assert_eq!(
        table.row_range(320_001_535).unwrap(),
        REAL_FILE_BYTES - 102..REAL_FILE_BYTES
    );
    assert!(matches!(
        table.row_range(320_001_536),
        Err(EngError::BadRequest(_))
    ));
    let expected: [(NgramAux, Range<u64>); 4] = [
        (NgramAux::HeadBias, 21_336..26_456),
        (NgramAux::HeadOffsets, 26_456..26_584),
        (NgramAux::HeadVocabSizes, 26_584..26_712),
        (NgramAux::LayerMultipliers, 26_712..26_736),
    ];
    for (which, range) in expected {
        assert_eq!(table.aux_range(which), range, "{which:?}");
    }
}

#[test]
fn the_mul1_codebook_matches_the_independent_table() {
    let table: Vec<u8> = (0..=u16::MAX)
        .flat_map(|state| Codebook::Mul1.decode_state(state).to_le_bytes())
        .collect();
    assert_eq!(
        sha256(&table),
        manifest()["mul1_table_f16_sha256"].as_str().unwrap()
    );
}

fn relative_error_and_cosine(decoded: &[f32], source: &[f32]) -> (f64, f64) {
    let (mut err, mut norm, mut dot, mut dec) = (0f64, 0f64, 0f64, 0f64);
    for (&d, &s) in decoded.iter().zip(source) {
        let (d, s) = (f64::from(d), f64::from(s));
        err += (d - s) * (d - s);
        norm += s * s;
        dot += d * s;
        dec += d * d;
    }
    ((err / norm).sqrt(), dot / (dec * norm).sqrt())
}

/// The decoded rows of the prompt [760, 6511, 314, 9338, 369] (5 positions x 16 heads).
fn decode_fixture_rows(decoder: &NgramDecoder) -> Vec<[f32; ROW_DIM]> {
    let rows = manifest()["global_rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 80);
    rows.iter()
        .zip(PACKED.as_chunks::<102>().0)
        .map(|(&row, packed)| {
            let mut out = [0f32; ROW_DIM];
            decoder.decode_row(row, packed, &mut out).unwrap();
            out
        })
        .collect()
}

fn source_rows() -> Vec<f32> {
    u16s(SOURCE)
        .into_iter()
        .map(|bits| f32::from_bits(u32::from(bits) << 16))
        .collect()
}

#[test]
fn real_rows_decode_to_the_spec_values_and_track_the_bf16_source() {
    let decoder = real_decoder(&u16s(BIAS));
    let manifest = manifest();
    let heads: Vec<usize> = integers(&manifest, "heads")
        .into_iter()
        .map(|h| h as usize)
        .collect();
    let rows = integers(&manifest, "global_rows");
    for (&row, &head) in rows.iter().zip(&heads) {
        assert_eq!(decoder.head_of(row as u64).unwrap(), head, "row {row}");
    }
    let decoded = decode_fixture_rows(&decoder);
    let expected: Vec<f32> = DECODED
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    let source = source_rows();
    for (index, values) in decoded.iter().enumerate() {
        let span = index * ROW_DIM..(index + 1) * ROW_DIM;
        for (position, (&got, &want)) in values.iter().zip(&expected[span.clone()]).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "row {index} position {position}"
            );
        }
        // The pack's measured quantization error is at most 3.50% relative L2 with cosine at
        // least 0.99938 on these rows; a decoding fault lands far outside these bounds.
        let (relative, cosine) = relative_error_and_cosine(values, &source[span]);
        assert!(relative <= 0.04, "row {index}: relative L2 {relative}");
        assert!(cosine >= 0.999, "row {index}: cosine {cosine}");
    }
}

#[test]
fn the_bias_of_another_head_moves_the_rows_away_from_the_source() {
    let bias = u16s(BIAS);
    let shifted: Vec<u16> = bias[ROW_DIM..]
        .iter()
        .chain(&bias[..ROW_DIM])
        .copied()
        .collect();
    let decoded = decode_fixture_rows(&real_decoder(&shifted));
    let source = source_rows();
    let mean = decoded
        .iter()
        .enumerate()
        .map(|(index, values)| {
            relative_error_and_cosine(values, &source[index * ROW_DIM..(index + 1) * ROW_DIM]).0
        })
        .sum::<f64>()
        / decoded.len() as f64;
    assert!(mean > 0.1, "mean relative L2 {mean}");
}

// ---- synthetic tables -------------------------------------------------------------------------

struct Synthetic {
    metadata: Map<String, Value>,
    /// Name, dtype, shape, in file order.
    tensors: Vec<(String, &'static str, Vec<u64>)>,
}

fn element_bytes(dtype: &str) -> u64 {
    match dtype {
        "I16" | "F16" => 2,
        "I64" => 8,
        other => panic!("{other}"),
    }
}

impl Synthetic {
    fn new(bits: u64, shard_rows: u64, shards: u64) -> Self {
        let words = 1 + 10 * bits;
        let mut metadata = Map::new();
        for (name, value) in [
            ("format", "exl3_ngram_trellis".to_string()),
            ("version", "1".to_string()),
            ("K", bits.to_string()),
            ("codebook", "mul1".to_string()),
            ("row_dim", "160".to_string()),
            ("rows", (shard_rows * shards).to_string()),
            ("shard_rows", shard_rows.to_string()),
        ] {
            metadata.insert(name.into(), Value::String(value));
        }
        let mut tensors = vec![
            (format!("{KEY}.head_bias"), "F16", vec![2, 160]),
            (format!("{KEY}.head_offsets"), "I64", vec![2]),
            (format!("{KEY}.head_vocab_sizes"), "I64", vec![2]),
            (format!("{KEY}.layer_multipliers"), "I64", vec![3]),
        ];
        for shard in 0..shards {
            tensors.push((
                format!("{KEY}.shard_{shard}.trellis"),
                "I16",
                vec![shard_rows, words],
            ));
        }
        Self { metadata, tensors }
    }

    /// The prefix and header bytes and the length of the file they describe.
    fn header(&self) -> (Vec<u8>, u64) {
        let mut root = Map::new();
        root.insert("__metadata__".into(), Value::Object(self.metadata.clone()));
        let mut offset = 0u64;
        for (name, dtype, shape) in &self.tensors {
            let len = shape.iter().product::<u64>() * element_bytes(dtype);
            root.insert(
                name.clone(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len]}),
            );
            offset += len;
        }
        let text = serde_json::to_vec(&Value::Object(root)).unwrap();
        let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&text);
        let byte_len = bytes.len() as u64 + offset;
        (bytes, byte_len)
    }

    fn table(&self) -> Result<NgramTable, EngError> {
        let (bytes, byte_len) = self.header();
        NgramTable::from_header(&bytes, byte_len, KEY, small_layout())
    }
}

fn small_layout() -> NgramLayout {
    NgramLayout {
        heads: 2,
        row_dim: 160,
        ngram: 3,
    }
}

fn checkpoint_message(result: Result<NgramTable, EngError>) -> String {
    match result {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected a checkpoint error, got {other:?}"),
    }
}

/// Packs one ring with the symbol of position `i` at stream bits `[i * K, (i + 1) * K)`,
/// least significant bit first, behind the binary16 scale word.
fn pack_ring(symbols: &[u32], bits: u32, scale: f32) -> Vec<u8> {
    let mut words = vec![0u16; 1 + ROW_DIM * bits as usize / 16];
    words[0] = f16::from_f32(scale).to_bits();
    for (position, &symbol) in symbols.iter().enumerate() {
        for bit in 0..bits as usize {
            let stream_bit = position * bits as usize + bit;
            let set = u16::from((symbol >> bit) & 1 == 1);
            words[1 + stream_bit / 16] |= set << (stream_bit % 16);
        }
    }
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[test]
fn hand_packed_rings_decode_every_position_including_the_wrap() {
    for bits in [2u32, 3, 5, 8] {
        let table = Synthetic::new(u64::from(bits), 4, 2).table().unwrap();
        let mut bias = vec![0u16; 2 * ROW_DIM];
        for value in &mut bias[ROW_DIM..] {
            *value = f16::from_f32(0.25).to_bits();
        }
        let decoder = NgramDecoder::new(&table, &[0, 3], &[3, 5], &bias).unwrap();
        let mut seed = 0x2545_f491u32 ^ bits;
        let symbols: Vec<u32> = (0..ROW_DIM)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 16) & ((1 << bits) - 1)
            })
            .collect();
        let packed = pack_ring(&symbols, bits, 0.5);
        let mut out = [0f32; ROW_DIM];
        // Row 5 lies in head 1 (rows 3..8).
        decoder.decode_row(5, &packed, &mut out).unwrap();
        let history = 16usize.div_ceil(bits as usize);
        for (position, &value) in out.iter().enumerate() {
            let mut state = 0u32;
            for back in 0..history {
                state |= symbols[(position + ROW_DIM - back) % ROW_DIM] << (back as u32 * bits);
            }
            let code = f16::from_bits(Codebook::Mul1.decode_state((state & 0xffff) as u16));
            assert_eq!(
                value,
                code.to_f32() * 0.5 + 0.25,
                "K {bits} position {position}"
            );
        }
    }
}

#[test]
fn the_metadata_must_name_this_exact_format() {
    for (field, value, expected) in [
        ("format", "exl3_other", "unsupported format"),
        ("version", "2", "unsupported version"),
        ("codebook", "mcg", "unsupported codebook"),
        ("row_dim", "128", "unsupported row_dim"),
        ("K", "9", "unsupported K"),
        ("K", "0", "unsupported K"),
        ("K", "+5", "not a decimal integer"),
        ("rows", "9", "do not divide"),
    ] {
        let mut synthetic = Synthetic::new(5, 4, 2);
        synthetic
            .metadata
            .insert(field.into(), Value::String(value.into()));
        let message = checkpoint_message(synthetic.table());
        assert!(message.contains(expected), "{field}={value}: {message}");
    }
    let mut synthetic = Synthetic::new(5, 4, 2);
    synthetic.metadata.remove("shard_rows");
    assert!(checkpoint_message(synthetic.table()).contains("shard_rows is missing"));
}

#[test]
fn every_tensor_must_have_its_place_dtype_and_shape() {
    let mut wrong_words = Synthetic::new(5, 4, 2);
    wrong_words.tensors[4].2 = vec![4, 50];
    assert!(checkpoint_message(wrong_words.table()).contains("shard_0.trellis is I16 [4, 50]"));

    let mut missing = Synthetic::new(5, 4, 2);
    missing.tensors.pop();
    assert!(checkpoint_message(missing.table()).contains("shard_1.trellis is missing"));

    let mut extra = Synthetic::new(5, 4, 2);
    extra
        .tensors
        .push((format!("{KEY}.shard_2.trellis"), "I16", vec![4, 51]));
    assert!(checkpoint_message(extra.table()).contains("holds 7 tensors"));

    let mut bias = Synthetic::new(5, 4, 2);
    bias.tensors[0].2 = vec![2, 128];
    assert!(checkpoint_message(bias.table()).contains("head_bias is F16 [2, 128]"));

    let mut multipliers = Synthetic::new(5, 4, 2);
    multipliers.tensors[3].1 = "F16";
    assert!(checkpoint_message(multipliers.table()).contains("layer_multipliers is F16"));
}

#[test]
fn a_model_with_other_row_widths_or_a_wrong_file_length_is_refused() {
    let (bytes, byte_len) = Synthetic::new(5, 4, 2).header();
    let layout = NgramLayout {
        row_dim: 128,
        ..small_layout()
    };
    let message = checkpoint_message(NgramTable::from_header(&bytes, byte_len, KEY, layout));
    assert!(message.contains("128-value rows"), "{message}");
    let message = checkpoint_message(NgramTable::from_header(
        &bytes,
        byte_len + 1,
        KEY,
        small_layout(),
    ));
    assert!(
        message.contains("does not match the data section end"),
        "{message}"
    );
}

#[test]
fn hash_head_ranges_must_be_ordered_nonempty_and_inside_the_table() {
    let table = Synthetic::new(5, 4, 2).table().unwrap();
    let bias = vec![0u16; 2 * ROW_DIM];
    for (offsets, sizes) in [
        ([0i64, 2], [3i64, 5]),
        ([0, 3], [3, 6]),
        ([0, 3], [0, 5]),
        ([-1, 3], [3, 5]),
        ([3, 0], [5, 3]),
    ] {
        let error = NgramDecoder::new(&table, &offsets, &sizes, &bias).unwrap_err();
        assert!(
            matches!(error, EngError::Checkpoint(_)),
            "{offsets:?} {sizes:?}"
        );
    }
    let error = NgramDecoder::new(&table, &[0], &[3], &bias).unwrap_err();
    assert!(matches!(error, EngError::Checkpoint(m) if m.contains("head count")));
    let mut infinite = bias.clone();
    infinite[7] = f16::INFINITY.to_bits();
    let error = NgramDecoder::new(&table, &[0, 3], &[3, 5], &infinite).unwrap_err();
    assert!(matches!(error, EngError::Checkpoint(m) if m.contains("not finite")));
}

#[test]
fn bad_rows_are_refused_before_any_value_is_written() {
    let table = Synthetic::new(5, 4, 2).table().unwrap();
    let decoder = NgramDecoder::new(&table, &[1, 3], &[2, 5], &vec![0u16; 2 * ROW_DIM]).unwrap();
    let packed = pack_ring(&[0; ROW_DIM], 5, 1.0);
    let mut out = [7f32; ROW_DIM];
    for (row, bytes) in [(0u64, &packed[..]), (8, &packed[..]), (2, &packed[..100])] {
        let error = decoder.decode_row(row, bytes, &mut out).unwrap_err();
        assert!(
            matches!(error, EngError::BadRequest(_)),
            "row {row}: {error:?}"
        );
    }
    let mut not_finite = packed.clone();
    not_finite[..2].copy_from_slice(&f16::NAN.to_bits().to_le_bytes());
    let error = decoder.decode_row(2, &not_finite, &mut out).unwrap_err();
    assert!(matches!(error, EngError::Checkpoint(m) if m.contains("not finite")));
    assert!(out.iter().all(|&value| value == 7.0));
}
