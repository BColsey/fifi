//! Check all slots and reserve headroom before creating a stream or staging
//! constants. The allocation calls remain fallible: a free-memory reading is
//! not an allocation guarantee. Both forms of decline precede K/V mutation.
use super::*;
impl TokenWorkspace {
    pub(super) fn valid_geometry(f: &CpuForward, states: &[VisitState]) -> bool {
        let c = &f.spec.core;
        f.math.validate(f.context, &f.spec).is_ok()
            && f.spec.n_layers_per_attn_res_block != 0
            && c.core_start < c.core_end
            && c.core_end <= f.spec.stored_layers()
            && c.num_passes > 0
            && c.core_end
                .checked_sub(c.core_start)
                .and_then(|n| n.checked_mul(c.num_passes))
                .and_then(|n| n.checked_add(c.core_start))
                .is_some_and(|end| end <= f.visits.len())
            && states.len() == f.visits.len()
            && !states.is_empty()
            && f.visits.iter().enumerate().all(|(i, v)| v.state_slot == i)
    }
    #[allow(clippy::type_complexity)]
    pub(super) fn fits(
        f: &CpuForward,
        cuda: &GgufCuda,
        geometry: &[(bool, bool, usize, Option<(usize, usize)>, usize)],
        free: usize,
    ) -> Result<bool, EngError> {
        let Weights::Gguf(w) = &f.weights else {
            return Ok(false);
        };
        let h = f.hidden();
        let vocab = f.spec.unpadded_vocab_size as usize;
        let Some(embed) = w.resident_embedding_bytes(h, f.spec.vocab_size as usize)? else {
            return Ok(false);
        };
        // Preserve the loader's expert-prefill growth floor plus a margin for
        // shared prep, driver objects and allocation rounding. Count per-slot
        // arenas, journal rows and weight constants before allocating any.
        let Some(floor) = cuda.token_growth_reserve(&f.spec, f.context) else {
            return Ok(false);
        };
        let mut demand = embed + floor + 2 * 1024 * 1024 + (vocab + 6 * h + 4) * WORD;
        let mut max_depth = 0;
        for (&v, g) in f.visits.iter().zip(geometry) {
            let banked = g.3.is_some();
            let global = v.attention == Attention::Global;
            let Some(widths) = cuda.resident_local_decode_widths(v) else {
                return Ok(false);
            };
            let Some(status) = cuda.resident_local_decode_status_words(v, banked) else {
                return Ok(false);
            };
            let (bank_blocks, bank_first) = g.3.unwrap_or_default();
            let shape = Shape {
                widths,
                global,
                banked,
                blocks: g.2,
                bank_blocks,
                bank_first,
                seal: v
                    .weight_layer
                    .is_multiple_of(f.spec.n_layers_per_attn_res_block),
            };
            if shape.blocks + usize::from(shape.seal) > MAX_BLOCKS
                || shape.bank_blocks > MAX_BLOCKS
                || shape.bank_first > shape.bank_blocks
            {
                return Ok(false);
            }
            let prefix = format!("model.llm.layers.{}", v.weight_layer);
            if w.resident_expert_dtypes(&format!("{prefix}.mlp"))?
                .is_none()
            {
                return Ok(false);
            }
            let mlp = if banked {
                let names = control_names(&prefix, Role::Mlp);
                cuda.resident_mlp_supported(
                    v.weight_layer as u32,
                    names.each_ref().map(String::as_str),
                    [f.spec.core.gate_rank as usize, h],
                )
            } else {
                cuda.resident_plain_mlp_supported(v.weight_layer as u32, h)
            };
            if !mlp {
                return Ok(false);
            }
            let layout = Layout::new(shape, status);
            max_depth = max_depth.max(layout.depth.bytes);
            let constants = (11
                + usize::from(v.weight_layer > 0)
                + 4 * usize::from(global)
                + 4 * usize::from(banked))
                * h
                + widths.query_channels
                + widths.relative_words
                + 2 * widths.kv_channels * widths.taps
                + f.spec.n_routed_experts as usize;
            demand = demand.saturating_add(
                layout.bytes + layout.receipts().bytes + (3 * h + constants) * WORD,
            );
            if demand > free {
                return Ok(false);
            }
        }
        Ok(demand.saturating_add(3 * max_depth) <= free)
    }
}
