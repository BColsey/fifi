use super::{embedding_row, stream_row};
use crate::{
    EngError,
    family::{exl3_pack::EMBED_NAME, flash_next::cuda::testkit::checkpoint},
};
use half::bf16;

fn table(vocab: usize, hidden: usize) -> (Vec<f32>, Vec<u8>) {
    let values: Vec<f32> = (0..vocab * hidden)
        .map(|i| bf16::from_f32((i as f32 - 20.0) * 0.37).to_f32())
        .collect();
    let bytes = values
        .iter()
        .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    (values, bytes)
}

#[test]
fn a_row_is_the_exact_widening_of_its_stored_bf16() {
    let (values, bytes) = table(7, 5);
    let source = checkpoint(&[(EMBED_NAME, "BF16", vec![7, 5], bytes)]);
    for token in [0u32, 3, 6] {
        let row = embedding_row(&source, token, 5).unwrap();
        assert_eq!(row, values[token as usize * 5..token as usize * 5 + 5]);
    }
}

#[test]
fn the_stream_repeats_the_row_once_per_branch() {
    let (values, bytes) = table(4, 3);
    let source = checkpoint(&[(EMBED_NAME, "BF16", vec![4, 3], bytes)]);
    let stream = stream_row(&source, 2, 3, 4).unwrap();
    assert_eq!(stream.len(), 12);
    for branch in 0..4 {
        assert_eq!(stream[branch * 3..branch * 3 + 3], values[6..9]);
    }
}

#[test]
fn a_token_outside_the_vocabulary_and_a_wrong_table_are_refused() {
    let (_, bytes) = table(4, 3);
    let source = checkpoint(&[(EMBED_NAME, "BF16", vec![4, 3], bytes.clone())]);
    assert!(matches!(
        embedding_row(&source, 4, 3),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        embedding_row(&source, 0, 4),
        Err(EngError::Checkpoint(_))
    ));
    let f16 = checkpoint(&[(EMBED_NAME, "F16", vec![4, 3], bytes)]);
    assert!(matches!(
        embedding_row(&f16, 0, 3),
        Err(EngError::Checkpoint(_))
    ));
    let empty = checkpoint(&[("other", "BF16", vec![1], vec![0, 0])]);
    assert!(embedding_row(&empty, 0, 3).is_err());
}

#[test]
fn a_row_that_widens_to_a_nonfinite_value_is_refused() {
    // 0x7f80 is BF16 infinity.
    let source = checkpoint(&[(EMBED_NAME, "BF16", vec![1, 2], vec![0x80, 0x3f, 0x80, 0x7f])]);
    assert!(matches!(
        embedding_row(&source, 0, 2),
        Err(EngError::Checkpoint(_))
    ));
}
