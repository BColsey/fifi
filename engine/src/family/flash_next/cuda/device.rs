//! The card implementation of [`TokenDevice`]: the decode step (`step.rs`) with the split MoE block
//! (`moe_block.rs`) over the expert bank.
//!
//! WHAT: [`CardDevice`] owns a [`FlashStep`] and a [`SplitMoe`] and maps the driver's calls onto the
//! step's launches. The token start is written with synchronous copies on the step's stream (the
//! step parameters in the words block, the stream in the step arena), so they are ordered before the
//! first layer. The stream read-back at the PLE layer and the logits read-back wait for the queued
//! work.
//!
//! The split block is the step's MoE seam: `impl MoeBlock for SplitMoe` delegates to the block's own
//! `run`, which has the seam's signature.

use std::ops::Range;

use super::{
    driver::TokenDevice,
    feeds::StartSink,
    step::{FlashStep, MoeBlock},
    step_plan::Slot,
};
use crate::{
    EngError,
    family::{
        flash_next::cuda::moe_block::SplitMoe,
        qwen36::cuda::{
            exl3_moe::Exl3MoeBufs,
            moe::{Launch, MoeDims},
        },
    },
    kernels::cuda::resident::step_params::{StepParams, StepStatus},
};

impl MoeBlock for SplitMoe<'_> {
    fn run(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError> {
        SplitMoe::run(self, launch, layer, bufs, dims)
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub(super) struct CardDevice {
    step: FlashStep<'static>,
    moe: SplitMoe<'static>,
    /// `branches * hidden`.
    stream_width: usize,
    /// The width of the head's output (the padded vocabulary).
    logits_width: usize,
}

impl CardDevice {
    pub(super) fn new(step: FlashStep<'static>, moe: SplitMoe<'static>) -> Self {
        let stream_width = step.geometry.branches * step.geometry.hidden;
        let logits_width = step.model.vocab;
        Self {
            step,
            moe,
            stream_width,
            logits_width,
        }
    }

    /// The bank's statistics line for the load log and the per-job report.
    pub(super) fn bank_misses(&self) -> u64 {
        self.moe.bank().state().stats().misses
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
            .write_at(self.step.stream, range.offset(), &f32_bytes(stream))
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
        for layer in range {
            self.step.run_layer(layer, pos, &mut self.moe)?;
        }
        Ok(())
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

    fn check_health(&mut self) -> Result<(), EngError> {
        let (status, flag) = self.step.health()?;
        if let Some(error) = StepStatus(status).error() {
            return Err(error);
        }
        if flag != 0 {
            return Err(EngError::Backend {
                operation: "exl3_linear",
                message: "an EXL3 projection reported a non-finite value".to_string(),
            });
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.step.reset()
    }
}
