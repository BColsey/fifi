//! Host encoders for the blocks the device reads: a plain float matrix becomes Q8_0 blocks, so a
//! dense role that a file stores as F32, F16 or BF16 runs on the native kernels every other
//! dense role reads.

use crate::{EngError, weights::GgufType, weights::dequantize};

/// Values per Q8_0 block, and the bytes of one block (an f16 scale and 32 signed bytes).
const Q8_0_VALUES: usize = 32;
const Q8_0_BYTES: usize = 34;

/// The float types a dense role may carry in a file and the encoder reads.
pub fn is_plain_float(ty: GgufType) -> bool {
    matches!(ty, GgufType::F32 | GgufType::F16 | GgufType::Bf16)
}

/// Bytes of the Q8_0 form of `values` float values.
pub fn q8_0_len(values: usize) -> usize {
    values / Q8_0_VALUES * Q8_0_BYTES
}

/// Encodes `values` into `out` as Q8_0 blocks, one scale per 32 values (`amax / 127`, stored as
/// f16, the quotient rounded to the nearest signed byte). The length must be a multiple of 32
/// and `out` exactly [`q8_0_len`] of it.
pub fn encode_q8_0_into(values: &[f32], out: &mut [u8]) -> Result<(), EngError> {
    if !values.len().is_multiple_of(Q8_0_VALUES) || out.len() != q8_0_len(values.len()) {
        return Err(EngError::Checkpoint(format!(
            "q8_0 encode: {} values into {} bytes is not whole blocks of {Q8_0_VALUES}",
            values.len(),
            out.len()
        )));
    }
    for (block, bytes) in values
        .as_chunks::<Q8_0_VALUES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<Q8_0_BYTES>().0)
    {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        // The scale the device decodes is the stored f16, so quantize against that value.
        let scale = half::f16::from_f32(amax / 127.0);
        let inverse = if scale.to_f32() == 0.0 {
            0.0
        } else {
            scale.to_f32().recip()
        };
        bytes[..2].copy_from_slice(&scale.to_bits().to_le_bytes());
        for (byte, v) in bytes[2..].iter_mut().zip(block) {
            *byte = (v * inverse).round().clamp(-127.0, 127.0) as i8 as u8;
        }
    }
    Ok(())
}

/// Encodes `values` as Q8_0 blocks; see [`encode_q8_0_into`].
pub fn encode_q8_0(values: &[f32]) -> Result<Vec<u8>, EngError> {
    let mut out = vec![0u8; q8_0_len(values.len())];
    encode_q8_0_into(values, &mut out)?;
    Ok(out)
}

/// Encodes whole rows of a float tensor of type `ty` (`src`, in the file's bytes) into `out`,
/// the Q8_0 form of the same values. The caller hands one run of rows at a time, so a large
/// matrix never exists in its source, f32 and encoded forms together.
pub fn encode_plain_float_rows(ty: GgufType, src: &[u8], out: &mut [u8]) -> Result<(), EngError> {
    if !is_plain_float(ty) {
        return Err(EngError::Checkpoint(format!(
            "q8_0 encode: {} is not a plain float type",
            ty.name()
        )));
    }
    encode_q8_0_into(&dequantize(ty, src)?, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_round_trips_within_half_a_step() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) * 0.37).collect();
        let bytes = encode_q8_0(&values).expect("encode");
        assert_eq!(bytes.len(), 2 * Q8_0_BYTES);
        let back = dequantize(GgufType::Q80, &bytes).expect("decode");
        for (block, decoded) in values.chunks(32).zip(back.chunks(32)) {
            let step = block.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
            for (a, b) in block.iter().zip(decoded) {
                assert!((a - b).abs() <= step * 0.51 + 1e-3, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn a_zero_block_encodes_to_zero_bytes_and_a_ragged_length_is_refused() {
        assert_eq!(
            encode_q8_0(&[0.0; 32]).expect("zero"),
            vec![0u8; Q8_0_BYTES]
        );
        assert!(encode_q8_0(&[1.0; 31]).is_err());
    }

    #[test]
    fn f32_and_f16_sources_encode_to_the_same_blocks_when_the_values_are_exact_in_f16() {
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.25 - 4.0).collect();
        let f32_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        let (mut from_f32, mut from_f16) = (vec![0u8; Q8_0_BYTES], vec![0u8; Q8_0_BYTES]);
        encode_plain_float_rows(GgufType::F32, &f32_bytes, &mut from_f32).expect("f32");
        encode_plain_float_rows(GgufType::F16, &f16_bytes, &mut from_f16).expect("f16");
        assert_eq!(from_f32, from_f16);
        assert!(encode_plain_float_rows(GgufType::Q80, &[], &mut []).is_err());
    }
}
