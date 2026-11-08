//! The card implementation of [`TokenDevice`]: the decode step (`step.rs`) with the split MoE block
//! (`moe_block.rs`) over the expert bank.
//!
//! WHAT: [`CardDevice`] owns a [`FlashStep`] and a [`SplitMoeAt`] and maps the driver's calls onto the
//! step's launches. The token start is written with synchronous copies on the step's stream (the
//! step parameters in the words block, the stream in the step arena), so they are ordered before the
//! first layer. The stream read-back at the PLE layer and the logits read-back wait for the queued
//! work.
//!
//! The split block is the step's MoE seam: `SplitMoeAt` (in `step.rs`) maps a model layer to the bank
//! row and delegates to the block's own `run`.

use std::ops::Range;

use super::{
    driver::TokenDevice,
    feeds::StartSink,
    step::{FlashStep, SplitMoeAt},
    step_plan::Slot,
};
use crate::{
    EngError, family::qwen36::exl3_tensors::f32_le_bytes,
    kernels::cuda::resident::step_params::StepParams,
};

pub(super) struct CardDevice {
    step: FlashStep<'static>,
    moe: SplitMoeAt<'static>,
    /// `branches * hidden`.
    stream_width: usize,
    /// The width of the head's output (the padded vocabulary).
    logits_width: usize,
}

impl CardDevice {
    pub(super) fn new(step: FlashStep<'static>, moe: SplitMoeAt<'static>) -> Self {
        let stream_width = step.geometry.branches * step.geometry.hidden;
        let logits_width = step.model.vocab;
        Self {
            step,
            moe,
            stream_width,
            logits_width,
        }
    }

    /// The kernel the single-row EXL3 projections of this context run on (`legacy`, `wide1`, ...),
    /// from `FIFI_EXL3_SINGLE_GATHER` or the shuffle kernel; the Flash-Next load line does not print it.
    #[cfg(test)]
    pub(super) fn single_row_kernel_name(&self) -> Result<&'static str, EngError> {
        self.step.ctx.single_row_kernel_name()
    }

    /// The bank's statistics line for the load log and the per-job report.
    pub(super) fn bank_misses(&self) -> u64 {
        self.moe.inner.bank().state().stats().misses
    }
}

impl StartSink for CardDevice {
    fn write_step_params(&mut self, token: i32, pos: i32) -> Result<(), EngError> {
        let params = StepParams::new(token, pos)?;
        let range = self.step.params()?;
        range
            .buffer()
            .write_at(self.step.stream, range.offset(), &params.to_bytes())
    }

    fn write_stream(&mut self, stream: &[f32]) -> Result<(), EngError> {
        let range = self.step.slot(Slot::Stream)?;
        range
            .buffer()
            .write_at(self.step.stream, range.offset(), &f32_le_bytes(stream))
    }

    /// The PLE projections stage their own embedding; the start needs no copy of it.
    fn write_ple_embedding(&mut self, _embedding: &[f32]) -> Result<(), EngError> {
        Ok(())
    }
}

impl TokenDevice for CardDevice {
    fn begin_token(&mut self) -> Result<(), EngError> {
        self.step.begin_token()
    }

    fn layers(&mut self, range: Range<usize>, pos: usize) -> Result<(), EngError> {
        if range.is_empty() {
            return Ok(());
        }
        self.step.run_layers(range, pos, &mut self.moe)
    }

    fn read_stream(&mut self) -> Result<Vec<f32>, EngError> {
        self.step
            .read_f32(self.step.slot(Slot::Stream)?, self.stream_width)
    }

    fn write_stream_back(&mut self, stream: &[f32]) -> Result<(), EngError> {
        StartSink::write_stream(self, stream)
    }

    fn head(&mut self) -> Result<(), EngError> {
        self.step.head()
    }

    fn read_logits(&mut self) -> Result<Vec<f32>, EngError> {
        self.step
            .read_f32(self.step.slot(Slot::Logits)?, self.logits_width)
    }

    /// Drains the token's work and refuses a nonzero status word or EXL3 flag (the step poisons
    /// itself, and the driver poisons with it).
    fn check_health(&mut self) -> Result<(), EngError> {
        self.step.finish_token()
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.step.reset()
    }

    fn bank_misses(&self) -> Option<u64> {
        Some(CardDevice::bank_misses(self))
    }
}
