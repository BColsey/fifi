//! The guarded opener on a sparse file that carries the real header of the Qwen3.8 Flash-Next
//! 3.05 bpw n-gram table (the committed fixture), its 80 recorded rows and its auxiliary tensors.
//! Everything else in the 32.6 GB file is a hole.

use std::{fs::File, io::Write, os::unix::fs::FileExt, path::Path};

use super::NgramFile;
use crate::{
    EngError,
    storage_guard::LocalGuard,
    weights::exl3::ngram::{NgramAux, NgramDecoder, NgramLayout, NgramTable, ROW_DIM},
};

const KEY: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
const LAYOUT: NgramLayout = NgramLayout {
    heads: 16,
    row_dim: 160,
    ngram: 3,
};
const REAL_FILE_BYTES: u64 = 32_640_183_408;
const HEADER: &[u8] =
    include_bytes!("../../../tests/fixtures/flash_next_exl3_ngram/ngram_embedding.header");
const MANIFEST: &[u8] =
    include_bytes!("../../../tests/fixtures/flash_next_exl3_ngram/manifest.json");
const PACKED: &[u8] =
    include_bytes!("../../../tests/fixtures/flash_next_exl3_ngram/packed_u16.bin");
const DECODED: &[u8] =
    include_bytes!("../../../tests/fixtures/flash_next_exl3_ngram/decoded_f32.bin");
const BIAS: &[u8] =
    include_bytes!("../../../tests/fixtures/flash_next_exl3_ngram/head_bias_f16.bin");

fn manifest_integers(name: &str) -> Vec<i64> {
    let manifest: serde_json::Value = serde_json::from_slice(MANIFEST).unwrap();
    manifest[name]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap())
        .collect()
}

fn le(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The sparse file: header, auxiliary tensors and the 80 recorded rows in place; holes elsewhere.
fn write_table(path: &Path) {
    let mut file = File::create(path).unwrap();
    file.write_all(HEADER).unwrap();
    file.set_len(REAL_FILE_BYTES).unwrap();
    let table = NgramTable::from_header(HEADER, REAL_FILE_BYTES, KEY, LAYOUT).unwrap();
    for (which, bytes) in [
        (NgramAux::HeadBias, BIAS.to_vec()),
        (
            NgramAux::HeadOffsets,
            le(&manifest_integers("head_offsets")),
        ),
        (
            NgramAux::HeadVocabSizes,
            le(&manifest_integers("head_vocab_sizes")),
        ),
        (
            NgramAux::LayerMultipliers,
            le(&manifest_integers("layer_multipliers")),
        ),
    ] {
        file.write_all_at(&bytes, table.aux_range(which).start)
            .unwrap();
    }
    for (index, row) in manifest_integers("global_rows").iter().enumerate() {
        let packed = &PACKED[index * table.row_bytes()..(index + 1) * table.row_bytes()];
        file.write_all_at(packed, table.row_range(*row as u64).unwrap().start)
            .unwrap();
    }
}

fn open(path: &Path) -> Result<NgramFile, EngError> {
    NgramFile::open(path, &LocalGuard::new().unwrap(), KEY, LAYOUT)
}

fn message(result: Result<NgramFile, EngError>) -> String {
    result.expect_err("refused").describe()
}

#[test]
fn rows_read_from_the_file_decode_to_the_recorded_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ngram_embedding.safetensors");
    write_table(&path);
    let file = open(&path).unwrap();
    assert_eq!(file.table().rows(), 320_001_536);
    assert_eq!(file.table().row_bytes(), 102);
    let decoder = NgramDecoder::new(
        file.table(),
        &manifest_integers("head_offsets"),
        &manifest_integers("head_vocab_sizes"),
        &file
            .read_aux(NgramAux::HeadBias)
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let rows = manifest_integers("global_rows");
    assert_eq!(rows.len(), 80);
    let mut packed = vec![0u8; file.table().row_bytes()];
    for (index, &row) in rows.iter().enumerate() {
        file.read_row(row as u64, &mut packed).unwrap();
        assert_eq!(
            packed,
            PACKED[index * 102..(index + 1) * 102],
            "row {row} bytes"
        );
        let mut values = [0f32; ROW_DIM];
        decoder
            .decode_row(row as u64, &packed, &mut values)
            .unwrap();
        for (position, value) in values.iter().enumerate() {
            let at = (index * ROW_DIM + position) * 4;
            let want = f32::from_le_bytes(DECODED[at..at + 4].try_into().unwrap());
            assert_eq!(
                value.to_bits(),
                want.to_bits(),
                "row {row} position {position}"
            );
        }
    }
    assert_eq!(
        file.read_aux(NgramAux::LayerMultipliers).unwrap(),
        le(&manifest_integers("layer_multipliers"))
    );
}

#[test]
fn a_row_read_refuses_a_row_outside_the_table_and_a_buffer_of_the_wrong_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ngram_embedding.safetensors");
    write_table(&path);
    let file = open(&path).unwrap();
    let mut packed = vec![0u8; 102];
    assert!(matches!(
        file.read_row(320_001_536, &mut packed),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        file.read_row(0, &mut packed[..101]),
        Err(EngError::BadRequest(_))
    ));
}

#[test]
fn a_file_of_the_wrong_length_or_a_lying_prefix_is_refused_before_any_row_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ngram_embedding.safetensors");
    write_table(&path);
    // One byte short: the last tensor no longer ends at the end of the file.
    File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(REAL_FILE_BYTES - 1)
        .unwrap();
    let text = message(open(&path));
    assert!(
        text.contains("does not match the data section end"),
        "{text}"
    );
    // A prefix that claims more header than the file holds (nothing is allocated for it).
    File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .write_all_at(&u64::MAX.to_le_bytes(), 0)
        .unwrap();
    let text = message(open(&path));
    assert!(text.contains("the header claims"), "{text}");
}

#[test]
fn a_header_for_another_layout_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ngram_embedding.safetensors");
    write_table(&path);
    let wrong = NgramLayout { heads: 8, ..LAYOUT };
    let result = NgramFile::open(&path, &LocalGuard::new().unwrap(), KEY, wrong);
    assert!(matches!(result, Err(EngError::Checkpoint(_))));
}

// Fails if `NgramFile::open` stops guarding the path: the file does not exist, so without the
// guard the error would be the missing file.
#[cfg(target_os = "linux")]
#[test]
fn a_table_on_a_network_mount_is_refused_before_it_is_opened() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ngram_embedding.safetensors");
    let text = crate::storage_guard::seam::with_nfs_at(dir.path(), || message(open(&path)));
    assert!(text.contains("must be on local storage"), "{text}");
    write_table(&path);
    assert!(open(&path).is_ok());
}

// Fails if the guard judges the link and not its target: the link is local, the file it names is
// on the network mount.
#[cfg(target_os = "linux")]
#[test]
fn a_local_link_to_a_table_on_a_network_mount_is_refused() {
    let share = tempfile::tempdir().unwrap();
    let local = tempfile::tempdir().unwrap();
    let target = share.path().join("ngram_embedding.safetensors");
    write_table(&target);
    let link = local.path().join("ngram_embedding.safetensors");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let text = crate::storage_guard::seam::with_nfs_at(share.path(), || message(open(&link)));
    assert!(text.contains("must be on local storage"), "{text}");
    assert!(open(&link).is_ok());
}
