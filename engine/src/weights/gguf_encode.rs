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

/// Encodes `values` as Q8_0 blocks, one scale per 32 values (`amax / 127`, stored as f16, the
/// quotient rounded to the nearest signed byte). The length must be a multiple of 32.
pub fn encode_q8_0(values: &[f32]) -> Result<Vec<u8>, EngError> {
    if !values.len().is_multiple_of(Q8_0_VALUES) {
        return Err(EngError::Checkpoint(format!(
            "q8_0 encode: {} values is not a multiple of {Q8_0_VALUES}",
            values.len()
        )));
    }
    let mut out = Vec::with_capacity(values.len() / Q8_0_VALUES * Q8_0_BYTES);
    for block in values.as_chunks::<Q8_0_VALUES>().0 {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        // The scale the device decodes is the stored f16, so quantize against that value.
        let scale = half::f16::from_f32(amax / 127.0);
        let d = scale.to_f32();
        let inverse = if d == 0.0 { 0.0 } else { d.recip() };
        out.extend_from_slice(&scale.to_bits().to_le_bytes());
        out.extend(
            block
                .iter()
                .map(|v| (v * inverse).round().clamp(-127.0, 127.0) as i8 as u8),
        );
    }
    Ok(out)
}

/// The Q8_0 bytes of a float tensor of type `ty` read from `source`.
pub fn plain_float_as_q8_0(ty: GgufType, source: &[u8]) -> Result<Vec<u8>, EngError> {
    if !is_plain_float(ty) {
        return Err(EngError::Checkpoint(format!(
            "q8_0 encode: {} is not a plain float type",
            ty.name()
        )));
    }
    encode_q8_0(&dequantize(ty, source)?)
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
        assert_eq!(
            plain_float_as_q8_0(GgufType::F32, &f32_bytes).expect("f32"),
            plain_float_as_q8_0(GgufType::F16, &f16_bytes).expect("f16")
        );
        assert!(plain_float_as_q8_0(GgufType::Q80, &[]).is_err());
    }
}
