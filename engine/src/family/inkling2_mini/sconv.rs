//! BF16 causal short convolution for the fixed Mini text decoder.

use half::bf16;

use super::{CONV_TAPS, HEAD_DIM, HIDDEN, KV_HEADS, VISITS, is_local, visits};
use crate::EngError;

const TAPS: usize = CONV_TAPS as usize;
const HISTORY: usize = TAPS - 1;
const KV_WIDTH: usize = (KV_HEADS * HEAD_DIM) as usize;

/// The four independent input histories of one decoder visit.
#[derive(Clone, Copy, Debug)]
pub enum InklingConvStream {
    Key,
    Value,
    AttentionOutput,
    MlpOutput,
}

impl InklingConvStream {
    fn width(self) -> usize {
        match self {
            Self::Key | Self::Value => KV_WIDTH,
            Self::AttentionOutput | Self::MlpOutput => HIDDEN,
        }
    }

    fn uses_dynamic_taps(self, layer: usize) -> bool {
        matches!(self, Self::Key | Self::Value) && !is_local(layer)
    }
}

const STREAMS: [InklingConvStream; 4] = [
    InklingConvStream::Key,
    InklingConvStream::Value,
    InklingConvStream::AttentionOutput,
    InklingConvStream::MlpOutput,
];

/// Checkpoint tap weights and, for full-attention K/V, the projected tap deltas for this token.
pub enum InklingConvTaps<'a> {
    /// Checkpoint `weight` in channel-major order, with oldest tap first.
    Static(&'a [bf16]),
    /// Checkpoint `static_w` in lag-major order, with newest tap first.
    Dynamic {
        weights: &'a [bf16],
        /// Projected BF16 deltas in head-major order, with newest tap first.
        deltas: &'a [bf16],
    },
}

/// One sequence's short-convolution history. Repeated physical layers have different visit slots.
pub struct InklingConvState {
    history: [[Vec<bf16>; STREAMS.len()]; VISITS],
}

impl Default for InklingConvState {
    fn default() -> Self {
        Self {
            history: std::array::from_fn(|_| {
                STREAMS.map(|stream| vec![bf16::ZERO; HISTORY * stream.width()])
            }),
        }
    }
}

impl InklingConvState {
    /// Process the next token of one stream at a schedule visit, starting with zero padding.
    /// Inputs and outputs are contiguous BF16 channel rows. The caller advances each
    /// stream in sequence order and owns any surrounding projection, clipping and normalization.
    pub fn step(
        &mut self,
        visit: usize,
        stream: InklingConvStream,
        input: &[bf16],
        taps: InklingConvTaps<'_>,
        output: &mut [bf16],
    ) -> Result<(), EngError> {
        let schedule = visits();
        let visit = schedule.get(visit).ok_or_else(|| {
            EngError::BadRequest(format!("Inkling convolution visit {visit} is out of range"))
        })?;
        let width = stream.width();
        if input.len() != width || output.len() != width {
            return Err(EngError::BadRequest(format!(
                "Inkling convolution expects input and output widths of {width}"
            )));
        }

        let history = &mut self.history[visit.state_slot][stream as usize];
        let [oldest, middle, previous] =
            std::array::from_fn(|row| &history[row * width..(row + 1) * width]);
        let rows = [oldest, middle, previous, input];
        // The reference accumulates static taps oldest-first and dynamic taps newest-first.
        // https://docs.vllm.ai/en/latest/api/vllm/models/inkling2/nvidia/ops/sconv/
        match taps {
            InklingConvTaps::Static(weights)
                if !stream.uses_dynamic_taps(visit.weight_layer)
                    && weights.len() == width * TAPS =>
            {
                let scale = ((TAPS + 1) as f32).sqrt().recip();
                for (channel, result) in output.iter_mut().enumerate() {
                    let weights = &weights[channel * TAPS..(channel + 1) * TAPS];
                    let mut sum = 0.0;
                    for (row, weight) in rows.iter().zip(weights) {
                        sum = row[channel].to_f32().mul_add(weight.to_f32(), sum);
                    }
                    *result = bf16::from_f32((sum + input[channel].to_f32()) * scale);
                }
            }
            InklingConvTaps::Dynamic { weights, deltas }
                if stream.uses_dynamic_taps(visit.weight_layer)
                    && weights.len() == width * TAPS
                    && deltas.len() == KV_HEADS as usize * TAPS =>
            {
                let scale = ((2 * TAPS + 1) as f32).sqrt().recip();
                for (channel, result) in output.iter_mut().enumerate() {
                    let head = channel / HEAD_DIM as usize;
                    let mut sum = 0.0;
                    for (lag, row) in rows.iter().rev().enumerate() {
                        let weight = weights[lag * width + channel].to_f32()
                            + deltas[head * TAPS + lag].to_f32();
                        sum = row[channel].to_f32().mul_add(weight, sum);
                    }
                    *result = bf16::from_f32((sum + input[channel].to_f32()) * scale);
                }
            }
            _ => {
                return Err(EngError::BadRequest(format!(
                    "Inkling convolution tap layout does not match {stream:?} at visit {}",
                    visit.state_slot
                )));
            }
        }
        history.copy_within(width.., 0);
        history[(HISTORY - 1) * width..].copy_from_slice(input);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_taps_follow_sequence_order_and_keep_visits_and_streams_independent() {
        let mut state = InklingConvState::default();
        let mut output = vec![bf16::ZERO; HIDDEN];
        let weights: Vec<_> = [1., 2., 4., 8.].map(bf16::from_f32).repeat(HIDDEN);
        for (visit, stream, sample, numerator) in [
            (6, InklingConvStream::AttentionOutput, 1., 9.),
            (6, InklingConvStream::AttentionOutput, 2., 22.),
            (18, InklingConvStream::AttentionOutput, 8., 72.),
            (6, InklingConvStream::MlpOutput, 4., 36.),
            (6, InklingConvStream::AttentionOutput, 3., 37.),
            (6, InklingConvStream::AttentionOutput, 4., 53.),
            (6, InklingConvStream::AttentionOutput, 5., 69.),
        ] {
            state
                .step(
                    visit,
                    stream,
                    &vec![bf16::from_f32(sample); HIDDEN],
                    InklingConvTaps::Static(&weights),
                    &mut output,
                )
                .unwrap();
            assert!(
                output
                    .iter()
                    .all(|&y| y == bf16::from_f32(numerator / 5_f32.sqrt()))
            );
        }
    }

    #[test]
    fn mlp_convolution_rounds_before_the_residual_update() {
        let mut state = InklingConvState::default();
        let input = vec![bf16::ONE; HIDDEN];
        let weights = vec![bf16::ZERO; HIDDEN * TAPS];
        let mut output = vec![bf16::ZERO; HIDDEN];
        state
            .step(
                0,
                InklingConvStream::MlpOutput,
                &input,
                InklingConvTaps::Static(&weights),
                &mut output,
            )
            .unwrap();
        let residual_with_addend = -0.4609375;
        let combined = bf16::from_f32(residual_with_addend + output[0].to_f32());
        assert_eq!(combined.to_f32(), -0.013671875);
        let rounded_once = bf16::from_f32(residual_with_addend + 5_f32.sqrt().recip());
        assert_ne!(combined, rounded_once);
    }

    #[test]
    fn dynamic_taps_use_head_deltas_and_reject_wrong_layout_without_advancing() {
        let mut state = InklingConvState::default();
        let width = KV_WIDTH;
        let mut output = vec![bf16::ZERO; width];
        let input = vec![bf16::ONE; width];
        let mut weights = vec![bf16::ZERO; width * TAPS];
        weights[width..2 * width].fill(bf16::from_f32(2.));
        let mut deltas = vec![bf16::ZERO; KV_HEADS as usize * TAPS];
        deltas[TAPS..2 * TAPS].copy_from_slice(&[
            bf16::from_f32(2.),
            bf16::ONE,
            bf16::ZERO,
            bf16::ZERO,
        ]);
        for (visit, taps) in [
            (5, InklingConvTaps::Static(&weights)),
            (
                0,
                InklingConvTaps::Dynamic {
                    weights: &weights,
                    deltas: &deltas,
                },
            ),
            (
                VISITS,
                InklingConvTaps::Dynamic {
                    weights: &weights,
                    deltas: &deltas,
                },
            ),
        ] {
            assert!(
                state
                    .step(visit, InklingConvStream::Key, &input, taps, &mut output)
                    .is_err()
            );
        }
        for (head0, head1) in [(1. / 3., 1.), (1., 2.)] {
            state
                .step(
                    5,
                    InklingConvStream::Key,
                    &input,
                    InklingConvTaps::Dynamic {
                        weights: &weights,
                        deltas: &deltas,
                    },
                    &mut output,
                )
                .unwrap();
            assert_eq!(output[0], bf16::from_f32(head0));
            assert_eq!(output[HEAD_DIM as usize], bf16::from_f32(head1));
            assert_eq!(output[2 * HEAD_DIM as usize], bf16::from_f32(head0));
        }
    }
}
