//! One-shot research entry; ordinary ExecutorOps::decode never arms it.
use super::QwenResidentStep;
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        qwen36::activation::{
            ActivationRecord, Capture, CaptureRequest, ExecutionSettings, LAYER, PromptBoundary,
            prefill_fingerprint, refusal,
        },
        session::DecodeSession,
    },
    tokenizer::DecodeBatch,
};

impl QwenResidentStep<'_> {
    pub(crate) fn activation_settings(
        &self,
        session: &DecodeSession,
    ) -> Result<ExecutionSettings, EngError> {
        for (name, active) in [
            (
                "graph capture/replay",
                self.capture_on || self.graph.is_some(),
            ),
            ("tiered experts", self.tiered_mode),
            ("paged KV", self.res.paged.is_some()),
            ("sparse attention", self.res.sparse.is_some()),
            ("speculative decode", self.speculator.is_some()),
            ("row window", self.window.is_some()),
            ("layer trace", crate::family::qwen36::trace::trace_enabled()),
        ] {
            if active {
                return Err(refusal(&format!(
                    "{name} is not supported by this extractor"
                )));
            }
        }
        if self.res.table.len() <= LAYER {
            return Err(refusal("model does not contain block 19"));
        }
        Ok(ExecutionSettings {
            weight_format: format!("{:?}", self.layout),
            kv_dtype: format!("{:?}", self.state.layout().kv.mode),
            context_tokens: self.context_limit(),
            prefill_chunk_tokens: self.m_sess_cap as usize,
            prefill_path: self
                .prefill_report
                .path_taken()
                .map(|path| path.label().to_owned())
                .unwrap_or_else(|| "unset".into()),
            prefill_pieces_sha256: prefill_fingerprint(
                self.prefill_report.start_rows(),
                self.prefill_report.feed_rows_calls(),
            ),
            sync_each: self.sync_each,
            section_marks: self.marks.is_some(),
            device_shortlist: self.shortlist_on,
            prefix_reused: session.restored_anchor().is_some(),
        })
    }

    /// Performs exactly the normal first decode forward, including its original sampling policy.
    /// The record is returned only if that whole decode succeeds; every outcome disarms capture.
    /// A decode that ends without a forward (cancellation, a token limit) is a refusal, not a
    /// record: capture is only meaningful on the forward that samples the first answer token.
    pub(crate) fn decode_with_activation(
        &mut self,
        session: &mut DecodeSession,
        generation: &mut GenerationState,
        request: CaptureRequest,
    ) -> Result<(DecodeBatch, ActivationRecord), EngError> {
        if self.activation.borrow().is_some() {
            return Err(refusal("another capture is already armed"));
        }
        request.check_pending(PromptBoundary {
            history: session.history(),
            prompt_len: session.prompt_len(),
            pending: session.pending(),
            fed: self.tokens(),
        })?;
        if request.hidden() != self.res.geometry.hidden {
            return Err(refusal(
                "request hidden width does not match the loaded model",
            ));
        }
        let capture = Capture::new(request, self.activation_settings(session)?)?;
        *self.activation.get_mut() = Some(capture);
        let result = <Self as ExecutorOps>::decode(self, session, generation);
        let capture = self
            .activation
            .get_mut()
            .take()
            .ok_or_else(|| refusal("capture was disarmed"))?;
        capture.finish(result)
    }

    pub(super) fn capture_activation(&self, position: usize) -> Result<(), EngError> {
        let mut active = self.activation.borrow_mut();
        let Some(capture) = active.as_mut() else {
            return Ok(());
        };
        capture.observe(LAYER, position, || {
            let range = self.view(super::resid_out(LAYER))?;
            if range.bytes() != self.res.geometry.hidden * size_of::<f32>() {
                return Err(refusal("post-block slot is not one raw F32 hidden row"));
            }
            // read_at is stream-ordered and blocking: the row is host-owned before any
            // later op of this token can read, add into or overwrite the slot.
            super::super::trace::f32_range(
                &super::Launch {
                    ctx: self.context,
                    stream: &self.stream,
                },
                &range,
            )
        })
    }
}
