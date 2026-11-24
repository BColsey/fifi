//! Whole-token transaction. Depth writes are in-place on alternating pass
//! planes. Sealed prelude, previous-pass bank, and each visit's convolution
//! state never cross PCIe during the chain. One coalesced receipt contains
//! logits, status words, and the raw rows needed to rebuild host state.
use super::*;
use crate::kernels::cuda::{NativeWeight, resident::embed::EmbedNativeCall};

struct Slot {
    w: Workspace,
    active: usize,
    save: bool,
    restart: bool,
    receipt: Region,
    attn_raw: Region,
    attention: Region,
    residual: Region,
}

pub(in super::super) struct TokenWorkspace {
    slots: Vec<Slot>,
    stream: DeviceStream,
    depth: [DeviceBuffer; 2],
    prelude: DeviceBuffer,
    prelude_bytes: usize,
    embedding: NativeWeight,
    constants: DeviceBuffer,
    params: DeviceBuffer,
    scratch: DeviceBuffer,
    journal: DeviceBuffer,
    pinned: PinnedBuf,
    logits: Region,
    health: Region,
    lineage: Vec<Option<std::sync::Weak<Vec<f32>>>>,
}

impl TokenWorkspace {
    fn layout(f: &CpuForward) -> Vec<(bool, bool, usize, Option<(usize, usize)>, usize)> {
        let c = &f.spec.core;
        let width = c.core_end - c.core_start;
        let suffix = c.core_start + width * c.num_passes;
        let (mut blocks, mut pre, mut bank, mut active) = (0, 0, None, 0);
        f.visits
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let save = i == c.core_start;
                if save {
                    pre = blocks;
                }
                let restart =
                    i > c.core_start && i < suffix && (i - c.core_start).is_multiple_of(width);
                if restart {
                    bank = Some((blocks, pre + 1));
                    blocks = pre;
                    active ^= 1;
                }
                if i == suffix {
                    bank = None;
                }
                let info = (save, restart, blocks, bank, active);
                blocks += usize::from(
                    v.weight_layer
                        .is_multiple_of(f.spec.n_layers_per_attn_res_block),
                );
                info
            })
            .collect()
    }

    fn eligible(cuda: &GgufCuda, v: LayerVisit, pos: usize, banked: bool) -> bool {
        if !banked {
            return cuda.resident_plain_supported(v, pos);
        }
        match v.attention {
            Attention::Local => cuda.resident_local_decode_supported(v, pos),
            Attention::Global => cuda.resident_global_supported(v, pos),
        }
    }

    pub(in super::super) fn open(
        f: &CpuForward,
        cuda: &GgufCuda,
        states: &[VisitState],
    ) -> Result<Option<Self>, EngError> {
        let h = f.hidden();
        let vocab = f.spec.unpadded_vocab_size as usize;
        if !cuda.resident_head_supported([vocab, h])
            || f.draft_window_enabled()
            || !f.taps.is_empty()
        {
            return Ok(None);
        }
        let geometry = Self::layout(f);
        if f.visits.iter().zip(&geometry).any(|(&v, g)| {
            states[v.state_slot].position() != f.position
                || !Self::eligible(cuda, v, f.position, g.3.is_some())
        }) {
            return Ok(None);
        }
        let Weights::Gguf(weights) = &f.weights else {
            return Ok(None);
        };
        let ctx = cuda.context();
        let mut end = 0;
        let logits = reserve(&mut end, vocab);
        let health = reserve(&mut end, 4);
        let (mut slots, mut depth_bytes, mut prelude_bytes, mut cold_bytes) = (Vec::new(), 0, 0, 8);
        for (&visit, (save, restart, blocks, bank, active)) in f.visits.iter().zip(geometry) {
            let Some(w) = Workspace::open_geometry(f, cuda, visit, blocks, bank)? else {
                return Ok(None);
            };
            depth_bytes = depth_bytes.max(w.layout.depth.bytes);
            if save {
                prelude_bytes = (blocks + 1) * h * WORD;
            }
            cold_bytes += w.layout.histories.iter().map(|r| r.bytes).sum::<usize>();
            let receipt = reserve(&mut end, w.layout.receipts().bytes / WORD);
            slots.push(Slot {
                w,
                active,
                save,
                restart,
                receipt,
                attn_raw: reserve(&mut end, h),
                attention: reserve(&mut end, h),
                residual: reserve(&mut end, h),
            });
        }
        let mut c = Vec::new();
        for name in [
            "model.llm.embed_norm.weight",
            "model.llm.attn_res.output_query",
            "model.llm.norm.weight",
        ] {
            append(&mut c, &f.weights.vec(name, h)?);
        }
        let constants = ctx.upload(&c)?;
        let zeros = slots
            .iter()
            .map(|s| s.w.layout.log.start - s.w.layout.attention_status.start)
            .max()
            .unwrap_or(0)
            .max(health.bytes);
        let params = ctx.upload(&vec![0; 8 + zeros])?;
        let depth = [ctx.allocate(depth_bytes)?, ctx.allocate(depth_bytes)?];
        let prelude = ctx.allocate(prelude_bytes)?;
        let journal = ctx.allocate(end)?;
        let pinned = ctx.pinned_buf(end.max(cold_bytes))?;
        let scratch = ctx.allocate(2 * h * WORD)?;
        let Some(embedding) = weights.resident_embedding(ctx)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            lineage: vec![None; slots.len()],
            slots,
            stream: ctx.stream()?,
            depth,
            prelude,
            prelude_bytes,
            embedding,
            constants,
            params,
            scratch,
            journal,
            pinned,
            logits,
            health,
        }))
    }

    pub(in super::super) fn ready(
        &self,
        cuda: &GgufCuda,
        states: &[VisitState],
        pos: usize,
    ) -> bool {
        self.slots.iter().all(|s| {
            let v = s.w.plan.visit;
            states[v.state_slot].position() == pos
                && Self::eligible(cuda, v, pos, s.w.plan.shape.banked)
        })
    }

    pub(in super::super) fn run(
        &mut self,
        f: &CpuForward,
        cuda: &GgufCuda,
        states: &[VisitState],
        id: u32,
    ) -> Result<Vec<f32>, EngError> {
        let ctx = cuda.context();
        let stream = &self.stream;
        let h = f.hidden();
        let row = h * WORD;
        let pos = i32::try_from(f.position).map_err(|_| bad("resident position overflows"))?;
        let mut boot = Vec::new();
        let mut input = Vec::new();
        input.extend(id.to_ne_bytes());
        input.extend(pos.to_ne_bytes());
        for (slot, identity) in self.slots.iter().zip(&mut self.lineage) {
            let w = &slot.w;
            let s = &states[w.plan.visit.state_slot];
            let warm = identity
                .take()
                .zip(s.resident_history_identity())
                .is_some_and(|(a, b)| a.ptr_eq(&b));
            if warm {
                continue;
            }
            let at = input.len();
            for (region, history) in w
                .layout
                .histories
                .into_iter()
                .zip(s.resident_conv_histories())
            {
                let start = input.len();
                let past = w.plan.shape.widths.taps - 1;
                if history.len() > past
                    || history
                        .iter()
                        .any(|v| v.len() * WORD * past != region.bytes)
                {
                    return Err(bad("resident token history geometry differs"));
                }
                for v in history {
                    append(&mut input, v);
                }
                input.resize(start + region.bytes, 0);
            }
            boot.push((w, at, input.len() - at));
        }
        self.pinned.write(0, &input)?;
        let params = &self.params;
        let head = |i| self.health.device(&self.journal)?.slice(i * WORD, WORD);
        let queued = (|| {
            ctx.copy_from_pinned(stream, &mut self.pinned, 0, params.range(0, 8)?)?;
            for (w, at, bytes) in boot {
                ctx.copy_from_pinned(
                    stream,
                    &mut self.pinned,
                    at,
                    w.arena.range(w.layout.histories[0].start, bytes)?,
                )?;
            }
            ctx.copy_device(
                stream,
                params.range(8, self.health.bytes)?,
                self.health.device(&self.journal)?,
            )?;
            let raw = self.scratch.range(0, row)?;
            ctx.embed_native(
                stream,
                EmbedNativeCall {
                    weight: &self.embedding,
                    ids: params.range(0, WORD)?,
                    x: raw,
                    status: head(0)?,
                    m: 1,
                },
            )?;
            ctx.enqueue_row_weighted_norm(
                stream,
                [
                    raw,
                    self.constants.range(0, row)?,
                    self.depth[0].range(0, row)?,
                    head(1)?,
                ],
                h,
                f.math.rms_norm_eps,
            )?;
            for s in &self.slots {
                let w = &s.w;
                let l = &w.layout;
                let d = &self.depth[s.active];
                if s.save {
                    ctx.copy_device(
                        stream,
                        d.range(0, self.prelude_bytes)?,
                        self.prelude.range(0, self.prelude_bytes)?,
                    )?;
                }
                if s.restart {
                    ctx.copy_device(
                        stream,
                        self.prelude.range(0, self.prelude_bytes)?,
                        d.range(0, self.prelude_bytes)?,
                    )?;
                }
                let bytes = l.log.start - l.attention_status.start;
                ctx.copy_device(
                    stream,
                    params.range(8, bytes)?,
                    w.arena.range(l.attention_status.start, bytes)?,
                )?;
                let b = &self.depth[s.active ^ 1];
                Execution {
                    plan: &w.plan,
                    layout: l,
                    weights: &w.weights,
                    arena: &w.arena,
                    stream,
                    context: ctx,
                    shared: Some(Shared {
                        depth: d.range(0, d.bytes())?,
                        bank: b.range(0, b.bytes())?,
                        pos: params.range(WORD, WORD)?,
                    }),
                }
                .enqueue(cuda, f.position)?;
                for (from, to) in [
                    (l.receipts(), s.receipt),
                    (l.histories[2].row(h), s.attn_raw),
                    (l.attention_out, s.attention),
                    (l.mlp_conv, s.residual),
                ] {
                    ctx.copy_device(stream, from.device(&w.arena)?, to.device(&self.journal)?)?;
                }
            }
            let last = self.slots.last().ok_or_else(|| bad("empty token chain"))?;
            let shape = last.w.plan.shape;
            let blocks = shape.blocks + usize::from(shape.seal);
            let normed = self.scratch.range(row, row)?;
            ctx.inkling_stream_mix(
                stream,
                MixCall {
                    state: self.depth[last.active].range(0, row * (blocks + 1))?,
                    query: self.constants.range(row, row)?,
                    out: raw,
                    health: head(2)?,
                },
                MixShape {
                    first_block: 0,
                    blocks,
                    width: h,
                    eps: MIX_EPS,
                },
            )?;
            ctx.enqueue_row_weighted_norm(
                stream,
                [raw, self.constants.range(row * 2, row)?, normed, head(3)?],
                h,
                f.math.rms_norm_eps,
            )?;
            cuda.enqueue_resident_head(
                stream,
                normed,
                self.logits.device(&self.journal)?,
                f.math.final_logit_softcapping,
            )?;
            ctx.copy_to_pinned(
                stream,
                self.journal.range(0, self.journal.bytes())?,
                &mut self.pinned,
                0,
            )
        })();
        self.pinned.synchronize_after_enqueue(stream, queued)?;
        let check = |span: Region| -> Result<(), EngError> {
            if span.read(&self.pinned)?.iter().any(|x| x.to_bits() != 0) {
                return Err(bad("resident token reported invalid device state"));
            }
            Ok(())
        };
        check(self.health)?;
        for s in &self.slots {
            check(Region {
                start: s.receipt.start,
                bytes: s.w.layout.log.start - s.w.layout.attention_status.start,
            })?;
        }
        let logits = self.logits.read(&self.pinned)?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(bad("resident head produced nonfinite logits"));
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests;
