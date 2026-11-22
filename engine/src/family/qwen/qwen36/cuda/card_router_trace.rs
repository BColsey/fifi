//! Opt-in first-fault diagnostics for the pair's eager one-row router.
//!
//! `Q36_P2P_ROUTER_TRACE=1` reads the sticky step status after each card-0
//! router launch. This synchronizes the stream and can change timing or hide a
//! race; a stalled CUDA operation can also stall the diagnostic. It is
//! unsuitable for performance measurements. On the first router
//! fault of a token it logs only indices and finiteness counts, never tensor
//! values or token text. Device-side softmax sums are not captured, so finite
//! inputs do not identify the GPU predicate that set the fault bit.

use std::sync::OnceLock;

use super::super::pair_router_report::{enabled, summarize};
use super::moe::{MoeBuffers, MoeLayerWeights};
use crate::EngError;
use crate::kernels::cuda::resident::router::ROUTER_STATUS_BIT;
use crate::kernels::cuda::{DeviceRange, DeviceStream};

pub(super) struct RouterTrace {
    position: usize,
    finished: bool,
}

impl RouterTrace {
    pub(super) fn from_env(position: usize) -> Option<Self> {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        ENABLED
            .get_or_init(|| enabled(std::env::var_os("Q36_P2P_ROUTER_TRACE").as_deref()))
            .then_some(Self {
                position,
                finished: false,
            })
    }

    /// Best effort only: the diagnostic does not itself return an error or skip
    /// the normal status fold or peer scheduling. A failed read ends tracing for
    /// this token and is reported, not interpreted as a healthy router. The read
    /// may surface a CUDA error that causes a subsequent launch to fail.
    pub(super) fn observe(
        &mut self,
        layer: usize,
        stream: &DeviceStream,
        weights: &MoeLayerWeights<'_>,
        bufs: &MoeBuffers<'_>,
    ) {
        if self.finished {
            return;
        }
        match self.read_fault(stream, weights, bufs) {
            Ok(Some((status, summary))) => {
                self.finished = true;
                eprintln!(
                    "q36 pair router trace: position={} layer={} status={:#x} expert_nonfinite={} first_bad_expert={:?} gate_finite={} input_nonfinite={} first_bad_input={:?}",
                    self.position,
                    layer,
                    status,
                    summary.experts_nonfinite,
                    summary.first_bad_expert,
                    summary.gate_finite,
                    summary.input_nonfinite,
                    summary.first_bad_input,
                );
            }
            Ok(None) => {}
            Err(error) => {
                self.finished = true;
                eprintln!(
                    "q36 pair router trace: position={} layer={} read failed: {:?}",
                    self.position, layer, error
                );
            }
        }
    }

    fn read_fault(
        &self,
        stream: &DeviceStream,
        weights: &MoeLayerWeights<'_>,
        bufs: &MoeBuffers<'_>,
    ) -> Result<Option<(u32, super::super::pair_router_report::Summary)>, EngError> {
        let mut status = [0u8; 4];
        bufs.route
            .status
            .buffer()
            .read_at(stream, bufs.route.status.offset(), &mut status)?;
        let status = u32::from_le_bytes(status);
        if status & ROUTER_STATUS_BIT as u32 == 0 {
            return Ok(None);
        }
        let expert_count = weights.facts().gate.experts;
        let logits = read_f32(stream, bufs.route.logits, expert_count + 1)?;
        let input = read_f32(stream, bufs.route.norm_row, bufs.route.norm_row.bytes() / 4)?;
        Ok(Some((
            status,
            summarize(&logits[..expert_count], logits[expert_count], &input),
        )))
    }
}

fn read_f32(
    stream: &DeviceStream,
    range: DeviceRange<'_>,
    count: usize,
) -> Result<Vec<f32>, EngError> {
    let bytes_len = count
        .checked_mul(4)
        .ok_or_else(|| EngError::BadRequest("router trace byte count overflow".to_owned()))?;
    if bytes_len > range.bytes() {
        return Err(EngError::BadRequest(
            "router trace exceeds tensor range".to_owned(),
        ));
    }
    let mut bytes = vec![0u8; bytes_len];
    range.buffer().read_at(stream, range.offset(), &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}
