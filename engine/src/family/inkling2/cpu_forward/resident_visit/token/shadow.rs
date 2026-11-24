//! Admission is read-only with respect to the sequence. After commitment there
//! is no fallback even for an enqueue error: a kernel may have appended K/V.
use super::*;

impl CpuForward {
    pub(in crate::family::inkling2::cpu_forward) fn admit_resident_token(
        &mut self,
    ) -> Result<bool, EngError> {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if !*ENABLED
            .get_or_init(|| std::env::var("FIFI_INKLING_CUDA_RESIDENT_TOKEN").as_deref() == Ok("1"))
            || self.draft_window_enabled()
            || !self.taps.is_empty()
        {
            return Ok(false);
        }
        let Weights::Gguf(w) = &self.weights else {
            return Ok(false);
        };
        let Some(cuda) = w.cuda() else {
            return Ok(false);
        };
        if let Some(ws) = &self.resident_token {
            return Ok(ws.ready(cuda, &self.state, self.position));
        }
        match TokenWorkspace::open(self, cuda, &self.state) {
            Ok(Some(ws)) => {
                self.resident_token = Some(ws);
                Ok(true)
            }
            Ok(None) | Err(EngError::MemoryPlan(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }
    pub(in crate::family::inkling2::cpu_forward) fn commit_resident_token(
        &mut self,
        id: u32,
    ) -> Result<Vec<f32>, EngError> {
        self.window = None;
        self.failed = true;
        let mut states = std::mem::take(&mut self.state);
        let mut ws = self.resident_token.take().expect("admitted resident token");
        let outcome: Result<Vec<f32>, EngError> = (|| {
            let _span = self.phase_span(Work::ResidentVisit);
            let Weights::Gguf(w) = &self.weights else {
                unreachable!("admitted GGUF token")
            };
            let cuda = w.cuda().expect("admitted CUDA token");
            let profile = self
                .weights
                .cuda_projection()
                .and_then(|p| p.begin_token().map(|start| (p, start)));
            let logits = ws.run(self, cuda, &states, id)?;
            ws.replay(self, cuda, &mut states)?;
            if let Some((p, start)) = profile {
                p.end_token(start, self.position, true);
            }
            Ok(logits)
        })();
        self.resident_token = Some(ws);
        let logits = outcome.inspect_err(|_| self.invalidate_device_attention())?;
        self.state = states;
        self.position += 1;
        self.failed = false;
        static USED: Once = Once::new();
        USED.call_once(|| {
            eprintln!(
                "Inkling CUDA used whole resident decode token: {} visits, one readback fence",
                self.visits.len()
            )
        });
        Ok(logits)
    }
}

impl TokenWorkspace {
    fn replay(
        &mut self,
        f: &CpuForward,
        cuda: &GgufCuda,
        states: &mut [VisitState],
    ) -> Result<(), EngError> {
        // All device status words and the logits have passed before a single
        // host cursor is advanced. Decode the journal only after that fence.
        let mut depth = Stream::new(self.initial.read(&self.pinned)?, f.cache_depth_mix);
        let mut prelude = None;
        for (index, s) in self.slots.iter().enumerate() {
            let w = &s.w;
            let visit = w.plan.visit;
            let l = &w.layout;
            if s.save {
                prelude = Some(depth.clone());
            }
            if s.restart {
                depth = prelude.as_ref().expect("saved prelude").clone();
            }
            let attention = s.attention.read(&self.pinned)?;
            if w.plan.shape.seal {
                depth.seal(attention);
            } else {
                depth.add_current(&attention);
            }
            depth.add_current(&s.residual.read(&self.pinned)?);
            let receipt_row = |start, bytes| {
                Region {
                    start: s.receipt.start + (start - l.attention_status.start),
                    bytes,
                }
                .read(&self.pinned)
            };
            let kv = w.plan.shape.widths.kv_channels * WORD;
            let [raw_k, raw_v, k, v] = [0, 1, 2, 3].map(|i| receipt_row(l.log.start + i * kv, kv));
            let state = &mut states[visit.state_slot];
            state.publish_resident_visit(
                [
                    raw_k?,
                    raw_v?,
                    k?,
                    v?,
                    s.attn_raw.read(&self.pinned)?,
                    receipt_row(l.mlp_raw.start, l.mlp_raw.bytes)?,
                ],
                (f.math.sconv_kernel_size - 1, f.kv_capacity(visit.attention)),
            );
            self.lineage[index] = state.resident_history_identity();
        }
        for s in &self.slots {
            cuda.publish_local_decode(s.w.plan.visit, f.position);
        }
        Ok(())
    }
}
