//! Staged GGUF token feed for the Flash-Next CUDA driver.

use std::collections::VecDeque;

use super::{
    GgufModel,
    ngram::GgufNgramRows,
    ple_weights::GgufPleWeights,
    projector::{HostProjector, HostWeight},
};
use crate::{
    EngError,
    family::{
        project::Projector,
        qwen::flash_next::{
            cuda::driver::{StartVectors, TokenFeeds},
            ple::{self, PleState},
            step::ModelShape,
        },
    },
};

struct Pending {
    token: u32,
    rows: Vec<(u32, Vec<f32>)>,
}

/// Rows gathered before device work; linear projections still use the mapped projector.
struct CachedProjector<'a, 'b> {
    inner: &'a mut HostProjector<'b>,
    table: &'a HostWeight,
    rows: &'a [(u32, Vec<f32>)],
    next: usize,
}

impl Projector for CachedProjector<'_, '_> {
    type Weight = HostWeight;
    fn linear(&mut self, w: &HostWeight, x: &[f32]) -> Result<Vec<f32>, EngError> {
        self.inner.linear(w, x)
    }
    fn expert_linear(
        &mut self,
        w: &HostWeight,
        expert: u32,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.inner.expert_linear(w, expert, x)
    }
    fn rows(&mut self, table: &HostWeight, rows: &[u32]) -> Result<Vec<f32>, EngError> {
        if !std::ptr::eq(table, self.table) || rows.len() != 1 {
            return Err(EngError::BadRequest(
                "GGUF PLE cached table mismatch".into(),
            ));
        }
        let (id, values) = self
            .rows
            .get(self.next)
            .ok_or_else(|| EngError::BadRequest("GGUF PLE cached row missing".into()))?;
        if *id != rows[0] {
            return Err(EngError::BadRequest("GGUF PLE cached row mismatch".into()));
        }
        self.next += 1;
        Ok(values.clone())
    }
}

pub(in super::super::super) struct GgufTokenFeeds<'a> {
    weights: GgufPleWeights,
    projector: HostProjector<'a>,
    shape: ModelShape,
    state: PleState,
    pending: VecDeque<Pending>,
}

impl<'a> GgufTokenFeeds<'a> {
    pub(in super::super::super) fn new(
        model: &'a GgufModel,
        ngram: &'a GgufNgramRows,
        threads: usize,
    ) -> Result<Self, EngError> {
        let shape = model.config().model_shape()?;
        Ok(Self {
            weights: GgufPleWeights::load(model)?,
            projector: HostProjector::new(&model.weights, ngram, threads),
            state: PleState::new(&shape.ple),
            shape,
            pending: VecDeque::new(),
        })
    }
}

impl TokenFeeds for GgufTokenFeeds<'_> {
    fn begin(&mut self, token: u32) -> Result<StartVectors, EngError> {
        self.begin_chunk(&[token], false)
            .map(|mut v| v.pop().expect("one token"))
    }
    fn begin_chunk(
        &mut self,
        tokens: &[u32],
        _profile: bool,
    ) -> Result<Vec<StartVectors>, EngError> {
        let w = self.weights.view();
        let mut history = self.state.history().to_vec();
        let mut pending = VecDeque::with_capacity(tokens.len());
        let mut starts = Vec::with_capacity(tokens.len());
        for &token in tokens {
            if token as usize >= self.shape.vocab {
                return Err(EngError::BadRequest("GGUF token outside vocabulary".into()));
            }
            let row = self.projector.rows(&self.weights.embed, &[token])?;
            if row.len() != self.shape.hyper.hidden() {
                return Err(EngError::BadRequest("GGUF embedding width mismatch".into()));
            }
            let stream = row.repeat(self.shape.hyper.branches());
            let plan = ple::plan_rows(&self.shape.ple, &w, &history, token)?;
            let mut rows = Vec::with_capacity(plan.rows.len());
            let mut embedding = Vec::new();
            for (table, local) in plan.rows {
                let values = self.projector.rows(&w.shards[table], &[local])?;
                embedding.extend(values.iter().map(|x| x * w.weight_scale));
                rows.push((local, values));
            }
            starts.push(StartVectors {
                stream,
                ple_embedding: embedding,
            });
            pending.push_back(Pending { token, rows });
            history = plan.history;
        }
        self.pending = pending;
        Ok(starts)
    }
    fn ple_update(&mut self, token: u32, stream: &[f32]) -> Result<Vec<f32>, EngError> {
        // Match PackFeeds: consume matching rows before the fallible update. The driver
        // poisons an attempted step on failure; retry requires reset or resume.
        let pending = self
            .pending
            .pop_front_if(|p| p.token == token)
            .ok_or_else(|| EngError::BadRequest(format!("no GGUF PLE rows for token {token}")))?;
        let w = self.weights.view();
        let mut projector = CachedProjector {
            inner: &mut self.projector,
            table: &w.shards[0],
            rows: &pending.rows,
            next: 0,
        };
        let mut next = self.state.clone();
        let update = ple::step(
            &mut projector,
            &self.shape.ple,
            &w,
            &mut next,
            token,
            stream,
        )?;
        if projector.next != pending.rows.len() {
            return Err(EngError::BadRequest("unused GGUF PLE cached rows".into()));
        }
        self.state = next;
        Ok(update)
    }
    fn reset(&mut self) {
        self.pending.clear();
        self.state = PleState::new(&self.shape.ple);
    }
    fn ple_state(&self) -> Option<(&[u32], &[f32])> {
        Some((self.state.history(), self.state.ring_values()))
    }
    fn ple_resume(&mut self, history: &[u32], ring: &[f32]) -> Result<(), EngError> {
        self.pending.clear();
        self.state.resume(history, ring)
    }
}
