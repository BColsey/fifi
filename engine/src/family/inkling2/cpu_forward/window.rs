//! Transactional verification over the Inkling text target.

use std::sync::Arc;

#[cfg(test)]
use crate::family::session::TokenStepper;

use super::{CpuForward, ForwardRow, VisitState, bad};
use crate::{
    EngError,
    family::{
        inkling2::cpu_attention::VisitUndo,
        sampler::Sampler,
        speculate::{
            config::MAX_DRAFT_ROWS,
            driver::{DraftOutput, TargetConditioning, WindowModel},
            window::Commit,
        },
    },
};

pub(super) struct PendingWindow {
    state: Vec<VisitState>,
    rows: Vec<PendingRow>,
}

pub(super) struct PendingRow {
    undo: Vec<VisitUndo>,
    head_input: Vec<f32>,
    features: Option<Arc<[f32]>>,
}

impl PendingRow {
    pub(super) fn new(undo: Vec<VisitUndo>, row: ForwardRow) -> Result<Self, EngError> {
        Ok(Self {
            undo,
            head_input: row
                .head_input
                .ok_or_else(|| bad("verify row returned no head input"))?,
            features: row.features,
        })
    }
}

impl CpuForward {
    pub(in crate::family::inkling2) fn prepare_draft(
        &mut self,
        taps: Vec<usize>,
        max_drafts: usize,
    ) -> Result<(), EngError> {
        if self.position != 0 || self.failed {
            return Err(bad("speculation requires a reset Inkling target"));
        }
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let super::Weights::Gguf(weights) = &self.weights
            && weights.resident_attention().is_some()
        {
            return Err(bad("Inkling speculation requires host attention"));
        }
        if max_drafts == 0 || max_drafts >= MAX_DRAFT_ROWS as usize {
            return Err(bad("speculation row limit is out of range"));
        }
        if taps.last().is_some_and(|&tap| tap > self.visits.len())
            || taps.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(bad(
                "target taps must be distinct increasing execution visits",
            ));
        }
        self.taps = taps;
        self.window_limit = 1 + max_drafts;
        Ok(())
    }

    pub(in crate::family::inkling2) fn draft_window_enabled(&self) -> bool {
        self.window_limit > 0
    }

    pub(super) fn feature_bytes_per_row(&self) -> u64 {
        if self.taps.is_empty() {
            return 0;
        }
        self.taps.len() as u64 * self.spec.hidden_size * size_of::<f32>() as u64
            + (size_of::<Arc<[f32]>>() + 2 * size_of::<usize>()) as u64
    }

    /// Reserve a cloned cache's row handles plus the per-row undo and head input.
    pub(in crate::family::inkling2) fn window_workspace_bytes(&self) -> Result<u64, EngError> {
        if !self.draft_window_enabled() {
            return Ok(0);
        }
        let handle_bytes = self.spec.kv_plan(self.context as u32)?.windowed_bytes / 16;
        let one_row_kv = self.spec.kv_plan(1)?.windowed_bytes * 2;
        let undo = self.visits.len() as u64 * 256 * 1024;
        let head_input = self.spec.hidden_size * size_of::<f32>() as u64;
        let transient_logits = self.spec.unpadded_vocab_size * size_of::<f32>() as u64;
        let features = self.feature_bytes_per_row();
        Ok(handle_bytes
            + transient_logits
            + self.window_limit as u64 * (one_row_kv + undo + head_input + features))
    }

    fn verify_serial_rows(
        &self,
        state: &mut [VisitState],
        inputs: &[u32],
    ) -> Result<Vec<PendingRow>, EngError> {
        inputs
            .iter()
            .map(|&token| {
                let undo = state
                    .iter()
                    .zip(&self.visits)
                    .map(|(slot, visit)| slot.checkpoint(self.kv_capacity(visit.attention)))
                    .collect();
                PendingRow::new(undo, self.forward_row(state, token, true)?)
            })
            .collect()
    }

    fn verify_rows(
        &self,
        state: &mut [VisitState],
        inputs: &[u32],
    ) -> Result<Vec<PendingRow>, EngError> {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if self.grouped_verify && inputs.len() > 1 {
            return self.grouped_verify_rows(state, inputs);
        }
        self.verify_serial_rows(state, inputs)
    }
}

impl DraftOutput for CpuForward {
    fn embedding(&self, token: u32) -> Result<Vec<f32>, EngError> {
        if token >= self.vocab_size() {
            return Err(bad("draft anchor is outside the Inkling vocabulary"));
        }
        Ok(self.embed(token)?.current)
    }

    fn project(&self, hidden: &[f32]) -> Result<Vec<f32>, EngError> {
        if hidden.len() != self.hidden() {
            return Err(bad("draft hidden row does not match the Inkling head"));
        }
        let _span = self.phase_span(super::Work::DraftVocab);
        self.project_logits(hidden)
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    fn project_corrected_greedy(
        &self,
        rows: [&[f32]; 2],
        vocab: usize,
        cap: f32,
    ) -> Result<Option<u32>, EngError> {
        match &self.weights {
            super::Weights::Gguf(weights) => {
                let _span = self.phase_span(super::Work::DraftVocab);
                weights.draft_greedy(rows, vocab, cap)
            }
            super::Weights::Native(_) => Ok(None),
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    fn cuda_context(&self) -> Option<&crate::kernels::cuda::CudaContext> {
        self.draft_cuda_context()
    }
}

impl WindowModel for CpuForward {
    fn draft_conditioning(&self) -> Option<TargetConditioning<'_>> {
        (!self.taps.is_empty()).then_some(TargetConditioning {
            features: &self.features,
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            lineage: &self.draft_lineage,
            output: self,
        })
    }

    fn run_window(&mut self, inputs: &[u32]) -> Result<(), EngError> {
        self.window = None;
        if self.failed
            || inputs.is_empty()
            || inputs.len() > self.window_limit
            || inputs.len() > self.context - self.position
            || inputs.iter().any(|&id| id >= self.vocab_size())
        {
            return Err(bad("verify window is outside the Inkling text limits"));
        }
        let mut state = self.state.clone();
        let rows = self.verify_rows(&mut state, inputs)?;
        self.window = Some(PendingWindow { state, rows });
        Ok(())
    }

    fn draw(
        &mut self,
        row: usize,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<u32, EngError> {
        let row = self
            .window
            .as_ref()
            .and_then(|window| window.rows.get(row))
            .ok_or_else(|| bad("verify window does not contain the requested row"))?;
        // The acceptance loop draws only the prefix it reaches.
        let logits = self.logits_for_head(&row.head_input)?;
        self.sample_target(&logits, sampler, sampler_vocab)
    }

    fn commit_window(&mut self, commit: &Commit) -> Result<(), EngError> {
        let window = self
            .window
            .as_ref()
            .ok_or_else(|| bad("no Inkling verify window to commit"))?;
        if !(1..=window.rows.len()).contains(&commit.kept_rows)
            || commit.committed_rows != self.position + commit.kept_rows
            || commit.discarded_rows != window.rows.len() - commit.kept_rows
            || commit.state_slot != commit.kept_rows - 1
        {
            return Err(bad("verdict does not match the Inkling verify window"));
        }
        let PendingWindow {
            mut state,
            mut rows,
        } = self.window.take().unwrap();
        for rejected in rows.drain(commit.kept_rows..).rev() {
            for (visit, undo) in state.iter_mut().zip(rejected.undo) {
                visit.rewind(undo);
            }
        }
        self.features
            .extend(rows.into_iter().filter_map(|row| row.features));
        self.state = state;
        self.position = commit.committed_rows;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::speculate::{accept::accept_run, window::Window};

    const TAPS: [usize; 4] = [5, 23, 41, 48];

    #[test]
    fn taps_follow_execution_order_and_include_the_terminal_row() {
        let mut taps = super::super::TapRow::new(&[0, 3, 5], 2);
        for visit in 0..=5 {
            taps.capture(visit, &[visit as f32, -(visit as f32)]);
        }
        assert_eq!(
            taps.finish().unwrap().as_ref(),
            &[0.0, -0.0, 3.0, -3.0, 5.0, -5.0]
        );
    }

    #[test]
    #[ignore = "requires a local Inkling 2 Mini GGUF via FIFI_INKLING_MODEL_DIR"]
    fn real_window_keeps_exact_target_rows_and_per_visit_state() {
        check_target_window(|_| {});
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    #[ignore = "requires a leased CUDA device and a local Inkling 2 Mini GGUF"]
    fn cuda_window_keeps_exact_target_rows_and_per_visit_state() {
        check_target_window(|target| target.enable_cuda(0).unwrap());
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    #[ignore = "requires a leased CUDA device and a fully resident packed Inkling 2 Mini GGUF"]
    fn cuda_grouped_window_keeps_exact_target_rows_and_per_visit_state() {
        check_target_window(|target| {
            target.enable_cuda(0).unwrap();
            assert!(
                target.enable_grouped_verify(),
                "grouped CUDA verifier unavailable"
            );
        });
    }

    fn check_target_window(enable_backend: impl FnOnce(&mut CpuForward)) {
        use std::{env, path::Path};

        let path = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set FIFI_INKLING_MODEL_DIR");
        let mut target = CpuForward::open(Path::new(&path), 8).unwrap();
        // Cross the local attention boundary inside the window.
        target.spec.sliding_window_size = 2;
        for taps in [vec![5, 5], vec![23, 5], vec![49]] {
            assert!(target.prepare_draft(taps, 3).is_err());
        }
        target.prepare_draft(TAPS.to_vec(), 3).unwrap();
        enable_backend(&mut target);
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        {
            assert!(!target.supports_grouped_prefill());
            assert!(target.prefill_grouped(&[1, 2]).is_err());
        }
        target.feed_rows(&[1, 2]).unwrap();
        let base = target.snapshot();
        let mut sampler = Sampler::greedy();
        let vocab = target.vocab_size() as usize;

        let (mut input, mut solo_draws, mut solo_states) = (3, Vec::new(), Vec::new());
        for _ in 0..4 {
            let logits = TokenStepper::logits(&mut target, input).unwrap();
            let features = target.features.last().unwrap();
            assert_eq!(features.len(), TAPS.len() * target.hidden());
            assert_eq!(
                target
                    .logits_for_head(&features[3 * target.hidden()..])
                    .unwrap(),
                logits
            );
            input = sampler.sample(&logits).unwrap();
            solo_draws.push(input);
            solo_states.push(target.snapshot());
        }

        for (drafts, kept) in [
            (solo_draws[..3].to_vec(), 4),
            (
                vec![
                    solo_draws[0],
                    (solo_draws[1] + 1) % vocab as u32,
                    solo_draws[2],
                ],
                2,
            ),
            (
                vec![
                    (solo_draws[0] + 1) % vocab as u32,
                    solo_draws[1],
                    solo_draws[2],
                ],
                1,
            ),
        ] {
            target.restore(&base);
            let window = Window::new(base.position, drafts, vec![]);
            let start = std::time::Instant::now();
            target.run_window(&window.input(3)).unwrap();
            eprintln!(
                "verified {} rows in {:.3} ms before retaining {kept}",
                window.rows(),
                1_000.0 * start.elapsed().as_secs_f64()
            );
            assert_eq!(target.position(), base.position);
            assert_eq!(target.state, base.visits);
            assert_eq!(target.features, base.features);
            assert!(
                target
                    .commit_window(&Commit {
                        committed_rows: base.position,
                        kept_rows: 0,
                        discarded_rows: window.rows(),
                        state_slot: 0,
                    })
                    .is_err()
            );
            let mut drawn = Vec::new();
            let accepted = accept_run(
                window.drafts(),
                window.rows(),
                |row| {
                    drawn.push(row);
                    target.draw(row, &mut sampler, vocab)
                },
                |_| false,
            )
            .unwrap();
            assert_eq!(drawn, (0..kept).collect::<Vec<_>>());
            assert_eq!(accepted.tokens, solo_draws[..kept]);
            target
                .commit_window(&window.commit(&accepted).unwrap())
                .unwrap();
            let reference = &solo_states[kept - 1];
            assert_eq!(target.position(), reference.position);
            for (index, (actual, expected)) in
                target.state.iter().zip(&reference.visits).enumerate()
            {
                assert!(
                    actual == expected,
                    "state differed at visit {index}, kept {kept}"
                );
            }
            let features = target.draft_conditioning().unwrap().features;
            assert_eq!(features.len(), reference.features.len());
            for (index, (actual, expected)) in features.iter().zip(&reference.features).enumerate()
            {
                assert!(
                    actual
                        .iter()
                        .map(|x| x.to_bits())
                        .eq(expected.iter().map(|x| x.to_bits())),
                    "conditioning differed at row {index}, kept {kept}"
                );
            }
        }
        target.run_window(&[3, 4]).unwrap();
        target.restore(&base);
        assert!(target.draw(0, &mut sampler, vocab).is_err());
        assert_eq!(target.position(), base.position);
        assert_eq!(target.state, base.visits);
        assert_eq!(target.features, base.features);
    }
}
