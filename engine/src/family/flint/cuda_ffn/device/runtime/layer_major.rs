//! Layer-major prefill: every row of a piece finishes layer L before any row
//! starts layer L+1, so a layer's routed experts see the whole piece at once
//! instead of eight rows at a time.
//!
//! In the tile pass (`FIFI_EXL3_LAYER_MAJOR=1`) the attention half and the
//! shared-expert tail of a layer still run in the narrow blocks the KV ring was
//! sized for, through the same code as the token-major walk. Only the routed
//! experts are regrouped: one routing call, one gather, and rounds of experts
//! sized to the slot arena, each expert running its rows as a few tall tiles.
//!
//! In the jobs pass (`=jobs`) the piece is one block: the KV rings are planned
//! for a piece, the attention half, the dense feed-forward, the shared expert
//! and the residual are single tall steps, and a round runs its experts as
//! job-table launches. Resident experts are staged first, so only the last
//! round copies weights.
//!
//! The two [`LayerPass`] hooks are the seam a family-independent pass
//! scheduler would later drive.

use super::{
    AttentionPass, Decoder, Group, LAYERS, Result, State, VOCAB, WIDTH, bad, exl3, kv_guard_gate,
    model_forward_error, residual_range,
};
use crate::EngError;
use crate::family::flint::{
    cuda_ffn::plan::TOP_K, exl3_resident::FlintFfnView, settings::PREFILL_MAX_ROWS,
};
use crate::kernels::cuda::{ArenaTensor, CudaContext, DeviceArena, DeviceRange};
use std::{cell::Cell, ops::Range};

/// Rows one layer-major piece may hold: the router and the pair gathers are
/// admitted up to this height in a single call.
pub(crate) const LAYER_MAJOR_PIECE_ROWS: usize = exl3::LAYER_MAX_ROWS;

/// The two halves of a layer, each over a row range of the piece. A scheduler
/// that does not know Flint drives a model through exactly this pair.
pub(super) trait LayerPass {
    /// Runs attention for `rows` of the piece and stages each row's
    /// post-attention residual and FFN input.
    fn execute_layer_attention(&mut self, layer: usize, rows: Range<usize>) -> Result<()>;
    /// Runs the layer's feed-forward block for `rows`: the routed experts of
    /// a MoE layer, the dense FFN of the others, then the residual add, and
    /// leaves each row's layer output as the next layer's input.
    fn execute_layer_routed_ffn(&mut self, layer: usize, rows: Range<usize>) -> Result<()>;
}

pub(super) fn drive(pass: &mut impl LayerPass, layers: usize, rows: Range<usize>) -> Result<()> {
    for layer in 0..layers {
        pass.execute_layer_attention(layer, rows.clone())?;
        pass.execute_layer_routed_ffn(layer, rows.clone())?;
    }
    Ok(())
}

pub(super) fn blocks(rows: Range<usize>, height: usize) -> impl Iterator<Item = Range<usize>> {
    let end = rows.end;
    rows.step_by(height.max(1))
        .map(move |start| start..(start + height.max(1)).min(end))
}

#[derive(Clone, Copy)]
#[repr(usize)]
enum Plane {
    /// The layer's input rows; overwritten with its output rows.
    State,
    /// Published post-attention norm: the FFN input.
    Norm,
    /// Post-attention residual stream.
    Residual,
}

const PLANES: usize = Plane::Residual as usize + 1;

/// The widest matrix dimension of the model, for bounding the shared-expert scratch.
const WIDEST_MATRIX: usize = 4096;

/// Device staging of an armed session: three activation planes of one piece
/// and the expert-sorted pair faces.
pub(super) struct Staging {
    planes: DeviceArena,
    faces: exl3::PairFaces,
    rows: usize,
    /// Whether the pass runs a piece as one block of tall projections (the jobs pass) or in
    /// narrow blocks.
    jobs: bool,
}

impl Staging {
    pub(super) fn new(ctx: &CudaContext, rows: usize, jobs: bool) -> Result<Self> {
        Ok(Self {
            planes: ctx.arena(&plane_tensors(rows)?)?,
            faces: exl3::PairFaces::new(
                ctx,
                rows * TOP_K,
                WIDTH,
                super::ROUTED_INTERMEDIATE,
                jobs,
            )?,
            rows,
            jobs,
        })
    }

    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    /// Bytes this staging allocates beyond the two pair faces that
    /// `State::layer_major_staging_bytes` already charges: the three planes,
    /// the router-input copy a routed layer holds while it routes, and the
    /// per-tile scratch.
    pub(super) fn extra_bytes(rows: usize, jobs: bool) -> Result<u64> {
        let plane = u64::try_from(plane_bytes(rows)?)
            .map_err(|_| bad("layer-major plane charge overflows"))?;
        let faces = exl3::PairFaces::bytes(rows * TOP_K, WIDTH, super::ROUTED_INTERMEDIATE, jobs)?;
        let pairs = u64::try_from(rows * TOP_K * WIDTH * 4 * 2)
            .map_err(|_| bad("layer-major face charge overflows"))?;
        // Faces beyond the two pair planes are the tile scratch and alignment.
        (PLANES as u64 + 1)
            .checked_mul(plane)
            .and_then(|bytes| bytes.checked_add(faces.saturating_sub(pairs)))
            .ok_or_else(|| bad("layer-major extra charge overflows"))
    }

    fn plane(&self, plane: Plane, rows: Range<usize>) -> Result<DeviceRange<'_>> {
        if rows.is_empty() || rows.end > self.rows {
            return Err(bad("layer-major row range is outside the staged piece"));
        }
        let base = self.planes.tensor(plane as usize)?;
        let row_bytes = WIDTH * 4;
        base.buffer().range(
            base.offset() + rows.start * row_bytes,
            rows.len() * row_bytes,
        )
    }
}

fn plane_bytes(rows: usize) -> Result<usize> {
    rows.checked_mul(WIDTH * 4)
        .ok_or_else(|| bad("layer-major plane bytes overflow"))
}

fn plane_tensors(rows: usize) -> Result<[ArenaTensor; PLANES]> {
    let bytes = plane_bytes(rows)?;
    Ok([ArenaTensor {
        bytes,
        first: 0,
        last: 0,
    }; PLANES])
}

impl State {
    /// Allocates the layer-major staging for pieces of up to `rows` rows. The
    /// token-major walk never reads it; an unarmed session allocates nothing.
    pub(crate) fn arm_layer_major(
        &mut self,
        ctx: &CudaContext,
        rows: usize,
        jobs: bool,
    ) -> Result<()> {
        if !(1..=LAYER_MAJOR_PIECE_ROWS).contains(&rows) {
            return Err(bad("layer-major staging rows are outside 1..=4096"));
        }
        self.layer_major = Some(Staging::new(ctx, rows, jobs)?);
        Ok(())
    }

    /// Bytes the allocation cache may keep idle in a jobs pass of `rows`-row pieces, so the tall
    /// frames of one layer are reused by the next instead of being allocated and freed. It is the
    /// retention the work reserve already holds for one tall attention half, so it adds no charge.
    pub(crate) fn layer_jobs_idle_bytes(rows: usize) -> Result<u64> {
        super::Health::retained_bytes(rows)
    }

    /// What a jobs pass of `rows`-row pieces holds beyond the narrow-block staging: the
    /// shared-expert scratch at piece height. The late-health retention of one tall attention
    /// half is already in the work reserve, which sizes it from the KV plan's chunk rows.
    pub(crate) fn layer_jobs_extra_bytes(rows: usize) -> Result<u64> {
        rows.checked_mul(3 * WIDEST_MATRIX * std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| bad("layer-major jobs scratch charge overflows"))
    }

    /// Rows the armed staging holds; `None` on an unarmed session.
    #[cfg(test)]
    pub(crate) fn test_layer_major_rows(&self) -> Option<usize> {
        self.layer_major.as_ref().map(Staging::rows)
    }

    /// The extra staging charge of [`Self::arm_layer_major`] on top of
    /// `layer_major_staging_bytes(rows)`, for open-time admission.
    pub(crate) fn layer_major_extra_bytes(rows: usize, jobs: bool) -> Result<u64> {
        Staging::extra_bytes(rows, jobs)
    }
}

/// One piece in flight: borrows the session and walks its layers.
struct Pass<'a, 'm> {
    decoder: &'a Decoder<'m>,
    state: &'a mut State,
    /// Absolute position of the piece's first row.
    position: usize,
    /// Rows per narrow block, the height the KV ring was sized for.
    block: usize,
    /// Most rows one block of the pass may carry: eight in narrow blocks, the piece in a jobs pass.
    rows_cap: usize,
    progress: &'a Cell<usize>,
    /// Shared-expert scratch, built at the first MoE layer.
    shared: Option<exl3::Scratch>,
}

impl LayerPass for Pass<'_, '_> {
    fn execute_layer_attention(&mut self, layer: usize, rows: Range<usize>) -> Result<()> {
        // Ring ordinals strictly below 3*layer+1 executed before this layer's
        // router; an abort here may replace its refusal with an earlier ring
        // fault, never a later one.
        self.progress.set(3 * layer + 1);
        #[cfg(test)]
        super::super::castprobe::set_layer(layer);
        let d = self.decoder;
        let view = d.model.layer(layer)?;
        let State {
            kv,
            hot,
            health,
            kv_guard,
            layer_major,
            ..
        } = &mut *self.state;
        let staging = layer_major
            .as_ref()
            .ok_or_else(|| bad("layer-major staging is not armed"))?;
        let cache = &mut kv[layer];
        for block in blocks(rows, self.block) {
            let height = block.len();
            // The ring holds window + block rows, so each block appends and
            // commits before the next one starts: the live range of a block
            // never covers more than the capacity allows.
            let append = cache
                .cursor
                .begin_append(self.position + block.start, height)?;
            let (temps, residual, norm) = d.attention_half(
                &view,
                layer,
                staging.plane(Plane::State, block.clone())?,
                ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
                AttentionPass {
                    rows: height,
                    rows_cap: self.rows_cap,
                    health: health.as_ref(),
                    hot: &mut *hot,
                    kv_guard: kv_guard.as_ref(),
                    check_group: true,
                },
                None,
            )?;
            for (source, plane) in [(&residual, Plane::Residual), (&norm, Plane::Norm)] {
                d.ctx.copy_device(
                    d.stream,
                    source.range(0, source.bytes())?,
                    staging.plane(plane, block.clone())?,
                )?;
            }
            #[cfg(test)]
            super::super::stage_clock::mark(d.stream, "attn: plane copies");
            #[cfg(test)]
            super::TEST_COMMIT_ENDS.with(|ends| ends.borrow_mut().push(append.live_range().end));
            append.commit();
            #[cfg(test)]
            super::super::stage_clock::mark(d.stream, "attn: commit");
            drop(temps);
            #[cfg(test)]
            super::super::stage_clock::mark(d.stream, "attn: temporaries released");
            #[cfg(test)]
            super::test_count(3);
        }
        Ok(())
    }

    fn execute_layer_routed_ffn(&mut self, layer: usize, rows: Range<usize>) -> Result<()> {
        #[cfg(test)]
        super::super::castprobe::set_layer(layer);
        let d = self.decoder;
        let view = d.model.layer(layer)?;
        let moe = matches!(view.ffn, FlintFfnView::Moe { .. });
        if moe && self.shared.is_none() {
            self.shared = Some(exl3::Scratch::for_rows(d.ctx, &view.ffn, self.block)?);
        }
        let Self {
            state,
            block,
            rows_cap,
            progress,
            shared,
            ..
        } = self;
        let State {
            hot,
            grouped,
            layer_major,
            ..
        } = &mut **state;
        let staging = layer_major
            .as_ref()
            .ok_or_else(|| bad("layer-major staging is not armed"))?;
        let mut routed = if moe {
            let scratch = shared
                .as_mut()
                .ok_or_else(|| bad("layer-major shared scratch is missing"))?;
            Some(
                exl3::RoutedLayer::run(
                    d.ctx,
                    d.stream,
                    &view.ffn,
                    &staging.faces,
                    scratch,
                    staging.plane(Plane::Norm, rows.clone())?,
                    hot.router()?,
                )
                .map_err(model_forward_error)?,
            )
        } else {
            None
        };
        // The routed experts of this layer have completed: from here an
        // abort's provenance window includes the layer's parked FFN word.
        progress.set(3 * layer + 2);
        #[cfg(test)]
        if super::super::route_trace::inject::after_ffn_abort(layer) {
            return Err(EngError::Backend {
                operation: "flint_exl3_ffn",
                message: "test-injected post-FFN abort".into(),
            });
        }
        for part in blocks(rows, *block) {
            let norm = staging.plane(Plane::Norm, part.clone())?;
            #[cfg(test)]
            super::super::stage_clock::mark(d.stream, "moe: before the shared expert");
            let ffn = match routed.as_mut() {
                Some(routed) => routed.finish_block(part.start, norm),
                None => exl3::forward_reusing_to(
                    d.ctx,
                    d.stream,
                    &view.ffn,
                    norm,
                    exl3::ScratchLoans {
                        router: Some(hot.router()?),
                        grouped: Some(&*grouped),
                        ffn_flag: None,
                        device_picks: false,
                    },
                    *rows_cap,
                ),
            }
            .map_err(model_forward_error)?;
            #[cfg(test)]
            super::super::stage_clock::mark(d.stream, "ffn: shared expert, dense ffn, combine");
            let next = residual_range(
                &mut Group::new(d.ctx, d.stream, None, 0)?.tall(*rows_cap),
                staging.plane(Plane::Residual, part.clone())?,
                ffn.range(0, ffn.bytes())?,
                None,
            )?;
            d.ctx.copy_device(
                d.stream,
                next.range(0, next.bytes())?,
                staging.plane(Plane::State, part)?,
            )?;
            #[cfg(test)]
            super::super::stage_clock::mark(d.stream, "ffn: residual and state copy");
        }
        progress.set(3 * layer + 3);
        #[cfg(test)]
        if super::super::route_trace::inject::after_residual_abort(layer) {
            return Err(EngError::Backend {
                operation: "flint_exl3_residual",
                message: "test-injected post-residual abort".into(),
            });
        }
        Ok(())
    }
}

impl Decoder<'_> {
    /// Feeds `tokens` as one layer-major piece: all rows finish each layer
    /// before the next layer starts. `block` is the narrow block height the
    /// KV ring was planned for. The same quiescent-reset contract as the
    /// token-major walk: the session is poisoned for the whole piece and a
    /// failure leaves it so until reset.
    pub(crate) fn feed_rows_layer_major(
        &self,
        state: &mut State,
        tokens: &[u32],
        block: usize,
    ) -> Result<()> {
        let capacity = state.layer_major.as_ref().map_or(0, Staging::rows);
        if tokens.is_empty() || tokens.len() > capacity {
            return Err(bad(
                "EXL3 layer-major piece needs 1..=armed staging rows (armed sessions only)",
            ));
        }
        let jobs = state
            .layer_major
            .as_ref()
            .is_some_and(|staging| staging.jobs);
        let widest = if jobs { capacity } else { PREFILL_MAX_ROWS };
        if !(1..=widest).contains(&block) {
            return Err(bad(&format!(
                "EXL3 layer-major block is outside 1..={widest} rows"
            )));
        }
        if tokens.iter().any(|&token| token as usize >= VOCAB) {
            return Err(bad("EXL3 token is outside the vocabulary"));
        }
        if state.poisoned {
            return Err(bad("EXL3 decoder requires quiescent reset after failure"));
        }
        if state
            .fed()
            .checked_add(tokens.len())
            .is_none_or(|end| end > state.plan.context_rows())
        {
            return Err(EngError::ContextLimit("EXL3 context is full".into()));
        }
        state.poisoned = true;
        let progress = Cell::new(0usize);
        if let Err(original) = self.run_piece(state, tokens, block, &progress) {
            return Err(self.failure(state, tokens.len(), original, progress.get()));
        }
        state.poisoned = false;
        Ok(())
    }

    fn run_piece(
        &self,
        state: &mut State,
        tokens: &[u32],
        block: usize,
        progress: &Cell<usize>,
    ) -> Result<()> {
        let position = state.fed();
        if let Some(kv_guard) = &state.kv_guard {
            kv_guard.write_at(self.stream, 0, &[0u8; 4])?;
        }
        let staging = state
            .layer_major
            .as_ref()
            .ok_or_else(|| bad("layer-major staging is not armed"))?;
        for part in blocks(0..tokens.len(), block) {
            let embedded = self.embedding(&tokens[part.clone()])?;
            self.ctx.copy_device(
                self.stream,
                embedded.range(0, embedded.bytes())?,
                staging.plane(Plane::State, part)?,
            )?;
        }
        drive(
            &mut Pass {
                decoder: self,
                state,
                position,
                block,
                rows_cap: block.max(PREFILL_MAX_ROWS),
                progress,
                shared: None,
            },
            LAYERS,
            0..tokens.len(),
        )?;
        // One word serves every layer and block of the piece. The blocks have
        // committed by now, so a rejected piece leaves the session poisoned
        // until reset, as any failed piece does.
        if let Some(kv_guard) = &state.kv_guard {
            kv_guard_gate(self.stream, kv_guard)?;
        }
        self.stream.synchronize()
    }
}

#[cfg(test)]
#[path = "layer_major_tests.rs"]
mod tests;
