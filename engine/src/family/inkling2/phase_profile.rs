//! Host wall time for Inkling GGUF work within the executor's prompt and decode calls.

use std::{cell::Cell, time::Instant};

use serde::Serialize;
use serde_json::{Value, json};

#[derive(Clone, Copy)]
pub(super) enum Work {
    CpuAttention,
    DeviceAttention,
    Convolution,
    Controller,
    Projection,
    Router,
    Experts,
    TargetVocab,
    DraftVocab,
    Sampling,
    AttentionControlPrepare,
    AttentionControlProject,
    AttentionControlApply,
    MlpControlPrepare,
    MlpControlProject,
    MlpControlApply,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    ResidentVisit,
}

impl Work {
    const ALL: &'static [(Self, &str)] = &[
        (Self::CpuAttention, "cpu_attention"),
        (Self::DeviceAttention, "device_attention"),
        (Self::Convolution, "convolution"),
        (Self::Controller, "controller"),
        (Self::Projection, "projection"),
        (Self::Router, "router"),
        (Self::Experts, "experts"),
        (Self::TargetVocab, "target_vocab"),
        (Self::DraftVocab, "draft_vocab"),
        (Self::Sampling, "sampling"),
        (Self::AttentionControlPrepare, "attn_control_prepare"),
        (Self::AttentionControlProject, "attn_control_project"),
        (Self::AttentionControlApply, "attn_control_apply"),
        (Self::MlpControlPrepare, "mlp_control_prepare"),
        (Self::MlpControlProject, "mlp_control_project"),
        (Self::MlpControlApply, "mlp_control_apply"),
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        (Self::ResidentVisit, "resident_visit"),
    ];
}

#[derive(Clone, Copy)]
enum Stage {
    Prefill,
    FirstDecode,
    Decode,
}

#[derive(Clone, Copy, Default, Serialize)]
struct Sample {
    ns: u64,
    calls: u64,
}

impl Sample {
    fn record(cell: &Cell<Self>, started: Instant) {
        let mut sample = cell.get();
        let ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        sample.ns = sample.ns.saturating_add(ns);
        sample.calls = sample.calls.saturating_add(1);
        cell.set(sample);
    }
}

#[derive(Default)]
struct Period {
    wall: Cell<Sample>,
    committed_rows: Cell<usize>,
    emitted_tokens: Cell<usize>,
    work: [Cell<Sample>; Work::ALL.len()],
}

impl Period {
    fn receipt(&self) -> Value {
        let work: serde_json::Map<_, _> = Work::ALL
            .iter()
            .map(|&(phase, name)| (name, self.work[phase as usize].get()))
            .filter(|(name, sample)| *name != "resident_visit" || sample.calls > 0)
            .map(|(name, sample)| (name.into(), json!(sample)))
            .collect();
        json!({
            "wall": self.wall.get(),
            "committed_rows": self.committed_rows.get(),
            "emitted_tokens": self.emitted_tokens.get(),
            "work": work,
        })
    }
}

pub(super) struct Step {
    started: Instant,
    stage: Stage,
    position: usize,
}

pub(super) struct Span<'a> {
    started: Instant,
    sample: &'a Cell<Sample>,
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        Sample::record(self.sample, self.started);
    }
}

#[derive(Default)]
pub(super) struct PhaseProfile {
    active: Cell<Option<Stage>>,
    periods: [Period; 3],
}

impl PhaseProfile {
    pub(super) fn begin_prefill(&self, position: usize) -> Step {
        self.begin(Stage::Prefill, position)
    }

    pub(super) fn begin_decode(&self, position: usize) -> Step {
        let stage = if self.periods[Stage::FirstDecode as usize].wall.get().calls == 0 {
            Stage::FirstDecode
        } else {
            Stage::Decode
        };
        self.begin(stage, position)
    }

    fn begin(&self, stage: Stage, position: usize) -> Step {
        debug_assert!(self.active.get().is_none());
        self.active.set(Some(stage));
        Step {
            started: Instant::now(),
            stage,
            position,
        }
    }

    pub(super) fn span(&self, work: Work) -> Option<Span<'_>> {
        let stage = self.active.get()?;
        Some(Span {
            started: Instant::now(),
            sample: &self.periods[stage as usize].work[work as usize],
        })
    }

    pub(super) fn finish(&self, step: Step, position: usize, tokens: usize) {
        self.active.set(None);
        let period = &self.periods[step.stage as usize];
        Sample::record(&period.wall, step.started);
        period.committed_rows.set(
            period
                .committed_rows
                .get()
                .saturating_add(position.saturating_sub(step.position)),
        );
        period
            .emitted_tokens
            .set(period.emitted_tokens.get().saturating_add(tokens));
    }

    pub(super) fn take_receipt(&mut self, completed: bool) -> Option<Value> {
        if self
            .periods
            .iter()
            .all(|period| period.wall.get().calls == 0)
        {
            return None;
        }
        let [prefill, first_decode, decode] = std::mem::take(&mut self.periods);
        Some(json!({
            "completed": completed,
            "prefill": prefill.receipt(),
            "first_decode": first_decode.receipt(),
            "decode": decode.receipt(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_tracks_prompt_and_generation_calls_separately() {
        let mut profile = PhaseProfile::default();
        assert!(profile.span(Work::CpuAttention).is_none());

        let step = profile.begin_prefill(7);
        drop(profile.span(Work::CpuAttention));
        drop(profile.span(Work::CpuAttention));
        profile.finish(step, 10, 0);
        let step = profile.begin_prefill(10);
        drop(profile.span(Work::Controller));
        for phase in [
            Work::AttentionControlPrepare,
            Work::AttentionControlProject,
            Work::AttentionControlApply,
            Work::MlpControlPrepare,
            Work::MlpControlProject,
            Work::MlpControlApply,
        ] {
            drop(profile.span(phase));
        }
        profile.finish(step, 11, 0);

        let step = profile.begin_decode(11);
        drop(profile.span(Work::TargetVocab));
        profile.finish(step, 15, 4);
        let step = profile.begin_decode(15);
        drop(profile.span(Work::Sampling));
        profile.finish(step, 16, 1);

        let report = profile.take_receipt(true).unwrap();
        assert_eq!(report["prefill"]["wall"]["calls"], 2);
        assert_eq!(report["prefill"]["committed_rows"], 4);
        assert_eq!(report["prefill"]["work"]["cpu_attention"]["calls"], 2);
        assert_eq!(report["prefill"]["work"]["controller"]["calls"], 1);
        for name in [
            "attn_control_prepare",
            "attn_control_project",
            "attn_control_apply",
            "mlp_control_prepare",
            "mlp_control_project",
            "mlp_control_apply",
        ] {
            assert_eq!(report["prefill"]["work"][name]["calls"], 1);
        }
        assert_eq!(report["first_decode"]["emitted_tokens"], 4);
        assert_eq!(report["first_decode"]["work"]["target_vocab"]["calls"], 1);
        assert_eq!(report["decode"]["emitted_tokens"], 1);
        assert_eq!(report["decode"]["work"]["sampling"]["calls"], 1);
        assert!(profile.take_receipt(false).is_none());
    }
}
