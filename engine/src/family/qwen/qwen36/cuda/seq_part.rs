//! The part of the resident step that belongs to one sequence.
//!
//! One step serves several sessions by time slicing: the weights, the context, the stream, the
//! activation arena, the prefill arena and the prefix store are the step's own and shared. What a
//! sequence owns is its KV planes, its DeltaNet and convolution state and step words (all in
//! [`ResidentState`]), the graph captured over those buffers, and the per-job reports. A
//! [`SeqPart`] holds exactly that. `swap_seq` exchanges it with the part the step runs, a move of
//! a few handles with no device work, so a step that serves one session never swaps and runs the
//! code it always ran.
//!
//! A captured graph bakes in the addresses of the state it ran over, so the graph travels with the
//! state: the field order of [`SeqPart`] drops the graph first.

use super::{
    super::{
        dims::StepOptions,
        run_report::{PrefillReport, RunReport},
        state::ResidentState,
    },
    QwenResidentStep,
};
use crate::{EngError, family::decode_step::state_layout::StateLayout, kernels::cuda::GraphExec};

/// One sequence's state on the device and the bookkeeping that follows it.
pub(crate) struct SeqPart<'a> {
    // Declaration order is drop order: the graph that bakes in the state's addresses goes first.
    graph: Option<GraphExec<'a>>,
    state: ResidentState,
    warmed: bool,
    needs_reset: bool,
    report: RunReport,
    prefill_report: PrefillReport,
}

impl SeqPart<'_> {
    /// The KV cells the sequence's planes hold.
    pub(crate) fn cells(&self) -> usize {
        self.state.layout().kv.capacity
    }
}

/// The forms a step can run in, as the sessions executor asks about them.
#[derive(Clone, Copy, Default)]
struct SessionForms {
    paged: bool,
    sparse: bool,
    tiered: bool,
    banked: bool,
    krow: bool,
    rollback: bool,
    speculative: bool,
}

/// The form that keeps state outside the part a session owns, when the step runs one. The
/// tiered form is not one: its banks, host pool, staging and scratch belong to the step and are
/// used inside one call, so sessions share them as they share the weights.
fn session_refusal(forms: SessionForms) -> Option<&'static str> {
    let _ = forms.tiered;
    if forms.paged {
        Some("paged KV")
    } else if forms.sparse {
        Some("sparse decode")
    } else if forms.banked {
        Some("banked EXL3 experts")
    } else if forms.krow {
        Some("the K-row verify window")
    } else if forms.rollback {
        Some("device rollback")
    } else if forms.speculative {
        Some("speculative decoding")
    } else {
        None
    }
}

impl<'a> QwenResidentStep<'a> {
    /// Whether this step can serve more than one session: the plain and the tiered resident
    /// step. The tiered runtime (banks, host pool, staging, scratch) is the step's own and is used
    /// inside one call, so sessions share it as they share the weights. The paged, sparse,
    /// banked, K-row, rollback and speculative forms keep state outside the part a session owns.
    pub(crate) fn check_sessions(&self) -> Result<(), EngError> {
        let refused = session_refusal(SessionForms {
            paged: self.res.paged.is_some(),
            sparse: self.res.sparse.is_some(),
            tiered: self.tiered_mode,
            banked: self.exl3_bank.is_some(),
            krow: self.res.krow.is_some() || self.krow_wanted,
            rollback: self.rollback_wanted || self.rollback.is_some(),
            speculative: self.speculator.is_some(),
        });
        match refused {
            Some(what) => Err(EngError::BadRequest(format!(
                "engine config: max_seqs above 1 needs the plain resident step, not {what}"
            ))),
            None => Ok(()),
        }
    }

    /// A fresh sequence part whose KV planes hold `cells` positions, zeroed and ready to run. The
    /// stream is the step's own, so the part's first token orders after its own zero fill.
    pub(crate) fn new_seq(&mut self, cells: usize) -> Result<SeqPart<'a>, EngError> {
        self.check_sessions()?;
        let options = self.seq_options(cells);
        let state = ResidentState::allocate(self.context, &self.stream, &self.res.dims, &options)?;
        let mut part = SeqPart {
            graph: None,
            state,
            warmed: false,
            needs_reset: false,
            report: RunReport::default(),
            prefill_report: PrefillReport::default(),
        };
        // The report of a part carries the step's chunk facts, as the build's own does.
        self.swap_seq(&mut part);
        self.refill_prefill_report();
        self.swap_seq(&mut part);
        Ok(part)
    }

    /// Exchanges the part the step runs with `part`. Device work already queued for the part
    /// that leaves stays ordered on the one stream, ahead of anything the part that arrives runs.
    pub(crate) fn swap_seq(&mut self, part: &mut SeqPart<'a>) {
        std::mem::swap(&mut self.graph, &mut part.graph);
        std::mem::swap(&mut self.state, &mut part.state);
        std::mem::swap(&mut self.warmed, &mut part.warmed);
        std::mem::swap(&mut self.needs_reset, &mut part.needs_reset);
        std::mem::swap(&mut self.report, &mut part.report);
        std::mem::swap(&mut self.prefill_report, &mut part.prefill_report);
    }

    /// The options of a sequence part whose KV planes hold `cells` positions.
    pub(in crate::family::qwen::qwen36::cuda) fn seq_options(&self, cells: usize) -> StepOptions {
        let layout = self.state.layout();
        StepOptions {
            kv_mode: layout.kv.mode,
            context_tokens: cells,
            delta_slots: layout.delta.slots,
            forked_experts: self.forked_experts,
            tiered: None,
            layout: self.layout,
        }
    }

    /// Device bytes of a sequence part whose KV planes hold `cells` positions.
    pub(crate) fn seq_bytes_at(&self, cells: usize) -> u64 {
        StateLayout::new(&self.res.dims, &self.seq_options(cells))
            .and_then(|layout| layout.bytes())
            .map_or(u64::MAX, |bytes| bytes as u64)
    }

    /// Free device memory now, or `None` when the card cannot be read.
    pub(crate) fn free_device_bytes(&self) -> Option<u64> {
        self.context.memory_info().ok().map(|info| info.free as u64)
    }

    /// Re-lays the planes of `part` (which is out of the step) in `cells` cells with the sequence
    /// intact: the part's graph baked the old planes, so it goes first and the next token warms
    /// and captures again. On an error the part keeps its planes.
    pub(crate) fn grow_seq(
        &mut self,
        part: &mut SeqPart<'a>,
        cells: usize,
    ) -> Result<(), EngError> {
        self.check_sessions()?;
        let options = self.seq_options(cells);
        // Queued work may still read the old planes or run the old graph.
        self.stream.synchronize()?;
        let grown = part
            .state
            .grown(self.context, &self.stream, &self.res.dims, &options)?;
        part.graph = None;
        part.warmed = false;
        part.state = grown;
        Ok(())
    }

    /// The KV cells the running part holds.
    pub(crate) fn seq_cells(&self) -> usize {
        self.state.layout().kv.capacity
    }
}

impl crate::family::pooled_sessions::PartHost for QwenResidentStep<'static> {
    type Part = SeqPart<'static>;
    const LABEL: &'static str = "q36 cuda";

    fn check_sessions(&self) -> Result<(), EngError> {
        QwenResidentStep::check_sessions(self)
    }

    fn funded_sessions(&self) -> Option<usize> {
        self.sessions_funded
    }

    fn context_cells(&self) -> usize {
        self.seq_cells()
    }

    fn ceiling_cells(&self) -> usize {
        self.ceiling_cells.unwrap_or_else(|| self.seq_cells())
    }

    fn new_part(&mut self, cells: usize) -> Result<Self::Part, EngError> {
        self.new_seq(cells)
    }

    fn swap_part(&mut self, part: &mut Self::Part) {
        self.swap_seq(part);
    }

    fn grow_part(&mut self, part: &mut Self::Part, cells: usize) -> Result<(), EngError> {
        self.grow_seq(part, cells)
    }

    fn before_decode(&mut self) {
        self.log_prefill_job();
    }

    fn part_cells(part: &Self::Part) -> usize {
        part.cells()
    }

    fn part_fed(part: &Self::Part) -> usize {
        part.state.tokens()
    }

    fn part_bytes_at(&self, cells: usize) -> u64 {
        self.seq_bytes_at(cells)
    }

    fn free_device_bytes(&self) -> Option<u64> {
        QwenResidentStep::free_device_bytes(self)
    }

    fn finish_save(&mut self) -> Result<(), EngError> {
        self.complete_prefix_save()
    }

    fn eos_ids(&self) -> std::sync::Arc<[u32]> {
        std::sync::Arc::clone(&self.rules.eos)
    }

    fn sampler_vocab(&self) -> usize {
        self.rules.sampler_vocab
    }

    fn embedding_rows(&self) -> usize {
        self.res.dims.shape.vocab
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tiered_step_can_serve_sessions_and_every_form_with_outside_state_cannot() {
        assert_eq!(session_refusal(SessionForms::default()), None);
        assert_eq!(
            session_refusal(SessionForms {
                tiered: true,
                ..SessionForms::default()
            }),
            None,
            "the tiered runtime is the step's own"
        );
        let refusing = [
            (
                SessionForms {
                    paged: true,
                    ..SessionForms::default()
                },
                "paged KV",
            ),
            (
                SessionForms {
                    sparse: true,
                    ..SessionForms::default()
                },
                "sparse decode",
            ),
            (
                SessionForms {
                    banked: true,
                    ..SessionForms::default()
                },
                "banked EXL3 experts",
            ),
            (
                SessionForms {
                    krow: true,
                    ..SessionForms::default()
                },
                "the K-row verify window",
            ),
            (
                SessionForms {
                    rollback: true,
                    ..SessionForms::default()
                },
                "device rollback",
            ),
            (
                SessionForms {
                    speculative: true,
                    ..SessionForms::default()
                },
                "speculative decoding",
            ),
        ];
        for (forms, what) in refusing {
            assert_eq!(session_refusal(forms), Some(what));
            // A tiered step that is also one of them is still refused for it.
            assert_eq!(
                session_refusal(SessionForms {
                    tiered: true,
                    ..forms
                }),
                Some(what)
            );
        }
    }
}
