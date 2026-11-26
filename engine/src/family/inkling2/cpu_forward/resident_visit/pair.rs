//! Bounded cross-visit transaction: distinct per-slot logs, a device depth handoff,
//! one stream and one pinned owner for all transfers and the completion fence.
use super::*;

#[cfg(test)]
mod tests;

pub(in super::super) struct Pair {
    visits: [Workspace; 2],
    transfer: PinnedBuf,
    // Weak identity includes the request branch, unlike a bare position. A
    // restore or host-only visit cannot accidentally inherit stale histories.
    lineage: [Option<std::sync::Weak<Vec<f32>>>; 2],
}

impl CpuForward {
    pub(in super::super) fn try_resident_pair(
        &self,
        visits: [LayerVisit; 2],
        states: &mut [VisitState],
        depth: &mut Stream,
        bank: &MessageBank,
    ) -> Result<bool, EngError> {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if !*ENABLED
            .get_or_init(|| std::env::var("FIFI_INKLING_CUDA_RESIDENT_TOKEN").as_deref() == Ok("1"))
            || self.draft_window_enabled()
        {
            return Ok(false);
        }
        let Weights::Gguf(weights) = &self.weights else {
            return Ok(false);
        };
        let Some(cuda) = weights.cuda() else {
            return Ok(false);
        };
        if visits[0].attention != Attention::Local {
            return Ok(false);
        }
        let global = visits[1].attention == Attention::Global;
        let mut cached = self.resident_pair[usize::from(global)].borrow_mut();
        let pos = states[visits[0].state_slot].position();
        if cached
            .as_ref()
            .is_some_and(|p| p.visits[0].plan.visit.state_slot != visits[0].state_slot)
            || visits.iter().any(|v| {
                states[v.state_slot].position() != pos
                    || !match v.attention {
                        Attention::Local => cuda.resident_local_decode_supported(*v, pos),
                        Attention::Global => cuda.resident_global_supported(*v, pos),
                    }
            })
        {
            return Ok(false);
        }
        if cached.is_none() {
            match Pair::open(self, cuda, visits, depth, bank) {
                Ok(Some(pair)) => *cached = Some(pair),
                Ok(None) | Err(EngError::MemoryPlan(_)) => return Ok(false),
                Err(e) => return Err(e),
            }
        }
        let pair = cached.as_mut().expect("admitted pair");
        let _span = self.phase_span(Work::ResidentVisit);
        let outputs = pair
            .run(cuda, pos, states, depth, bank)
            .inspect_err(|_| self.invalidate_device_attention())?;
        for ((output, workspace), visit) in outputs.into_iter().zip(&pair.visits).zip(visits) {
            cuda.publish_local_decode(visit, pos);
            states[visit.state_slot].publish_resident_visit(
                output.history,
                (
                    self.math.sconv_kernel_size - 1,
                    self.kv_capacity(visit.attention),
                ),
            );
            depth.publish_resident_current(output.current, workspace.plan.shape.seal);
        }
        pair.lineage = visits.map(|v| states[v.state_slot].resident_history_identity());
        static USED: [Once; 2] = [Once::new(), Once::new()];
        USED[usize::from(global)].call_once(|| {
            eprintln!(
                "Inkling CUDA used resident decode pair: slots {} and {}, one completion fence",
                visits[0].state_slot, visits[1].state_slot
            )
        });
        Ok(true)
    }
}
impl Pair {
    fn open(
        forward: &CpuForward,
        cuda: &GgufCuda,
        visits: [LayerVisit; 2],
        depth: &Stream,
        bank: &MessageBank,
    ) -> Result<Option<Self>, EngError> {
        let blocks = depth.block_count();
        let Some(first) = Workspace::open(forward, cuda, visits[0], (blocks, bank))? else {
            return Ok(None);
        };
        let Some(second) = Workspace::open(
            forward,
            cuda,
            visits[1],
            (blocks + usize::from(first.plan.shape.seal), bank),
        )?
        else {
            return Ok(None);
        };
        let transfer = cuda
            .context()
            .pinned_buf(first.transfer.len() + second.transfer.len())?;
        Ok(Some(Self {
            visits: [first, second],
            transfer,
            lineage: [None, None],
        }))
    }
    fn run(
        &mut self,
        cuda: &GgufCuda,
        pos: usize,
        states: &[VisitState],
        depth: &Stream,
        bank: &MessageBank,
    ) -> Result<[Output; 2], EngError> {
        let ctx = cuda.context();
        let [a, b] = &mut self.visits;
        let warm = [&*a, &*b]
            .into_iter()
            .zip(std::mem::take(&mut self.lineage))
            .all(|(w, saved)| {
                saved
                    .zip(states[w.plan.visit.state_slot].resident_history_identity())
                    .is_some_and(|(left, right)| left.ptr_eq(&right))
            });
        a.encode(&states[a.plan.visit.state_slot], Some(depth), bank, !warm)?;
        b.encode(&states[b.plan.visit.state_slot], None, bank, !warm)?;
        let offset = a.transfer.len();
        self.transfer.write(0, &a.encoded)?;
        self.transfer.write(offset, &b.encoded)?;
        let stream = &a.stream;
        let queued = (|| {
            ctx.copy_from_pinned(
                stream,
                &mut self.transfer,
                0,
                a.arena.range(
                    0,
                    if warm {
                        a.layout.bank.end()
                    } else {
                        a.encoded.len()
                    },
                )?,
            )?;
            if warm {
                let s = a.layout.attention_status.start;
                ctx.copy_from_pinned(
                    stream,
                    &mut self.transfer,
                    s,
                    a.arena.range(s, a.encoded.len() - s)?,
                )?;
            }
            let l = &b.layout;
            ctx.copy_from_pinned(stream, &mut self.transfer, offset, l.pos.device(&b.arena)?)?;
            let start = if warm {
                l.attention_status.start
            } else {
                l.bank.end()
            };
            ctx.copy_from_pinned(
                stream,
                &mut self.transfer,
                offset + start,
                b.arena.range(start, b.encoded.len() - start)?,
            )?;
            ctx.copy_device(
                stream,
                a.layout.bank.device(&a.arena)?,
                l.bank.device(&b.arena)?,
            )?;
            for (i, w) in [&*a, &*b].into_iter().enumerate() {
                if i == 1 {
                    ctx.copy_device(
                        stream,
                        a.layout.depth.device(&a.arena)?,
                        b.arena.range(l.depth.start, a.layout.depth.bytes)?,
                    )?;
                }
                Execution {
                    plan: &w.plan,
                    layout: &w.layout,
                    weights: &w.weights,
                    arena: &w.arena,
                    shared: None,
                    stream,
                    context: ctx,
                }
                .enqueue(cuda, pos)?;
            }
            // Deliberately after BOTH visits. No receipt, cursor or row is host-visible at handoff.
            for (base, w) in [(0, &*a), (offset, &*b)] {
                let h = w.plan.shape.widths.hidden;
                for s in [
                    w.layout.receipts(),
                    w.layout.depth.row(h),
                    w.layout.histories[2].row(h),
                ] {
                    ctx.copy_to_pinned(
                        stream,
                        s.device(&w.arena)?,
                        &mut self.transfer,
                        base + s.start,
                    )?;
                }
            }
            Ok(())
        })();
        self.transfer.synchronize_after_enqueue(stream, queued)?;
        Ok([
            a.readback(&self.transfer, 0)?,
            b.readback(&self.transfer, offset)?,
        ])
    }
}
