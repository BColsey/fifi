//! The residual stream between the host feeds and the device.
//!
//! WHAT: [`read_stream`] copies the F32 stream (`branches * hidden`, 40 KiB) from a device buffer
//! to the host, which the PLE step needs as the query side of its gate; [`write_stream`] writes an
//! F32 vector of the same width into one, for the embedding row of a token (the initial stream) and
//! for the PLE update. Both are synchronous copies on the context's default path
//! (`CudaContext::download` and `upload_at`) and so run outside graph capture, at the host/device
//! hand-off points of the step.

use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceBuffer},
};

/// The first `values` F32 values of `buffer`.
pub(in crate::family) fn read_stream(
    context: &CudaContext,
    buffer: &DeviceBuffer,
    values: usize,
) -> Result<Vec<f32>, EngError> {
    let mut bytes = vec![0u8; values * size_of::<f32>()];
    context.download(buffer, &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect())
}

/// Writes `values` at the start of `buffer`.
pub(in crate::family) fn write_stream(
    context: &CudaContext,
    buffer: &DeviceBuffer,
    values: &[f32],
) -> Result<(), EngError> {
    let bytes = crate::family::qwen36::exl3_tensors::f32_le_bytes(values);
    context.upload_at(buffer, 0, &bytes)
}

#[cfg(test)]
mod tests {
    use super::{read_stream, write_stream};
    use crate::{family::project::cuda::CudaProjector, kernels::cuda::device_present};

    /// The one device claim of this file: a 40 KiB stream written and read back is bit-equal, and
    /// a second write replaces the first.
    #[test]
    #[ignore = "needs a GPU (under 1 MiB)"]
    fn a_stream_written_to_the_card_reads_back_bit_equal() {
        assert!(device_present("Flash-Next stream io"), "a GPU is required");
        let projector = CudaProjector::new().expect("projector");
        let context = projector.context();
        let width = 4 * 2560;
        let buffer = context.upload(&vec![0u8; width * 4]).expect("buffer");
        for seed in [1.0f32, -3.5] {
            let values: Vec<f32> = (0..width)
                .map(|i| seed * (i as f32 * 0.37 - 11.0))
                .collect();
            write_stream(context, &buffer, &values).expect("write");
            let back = read_stream(context, &buffer, width).expect("read");
            assert!(
                back.iter()
                    .zip(&values)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "seed {seed}"
            );
        }
    }
}
