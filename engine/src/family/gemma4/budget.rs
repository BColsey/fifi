//! KV-cache budget of the resident Bolt (Nemotron 3.5 Bolt) executor.
//!
//! The resident Bolt cache stores KV8 planes (one int8 code per value plus one
//! fp16 scale per 32-value block) on the physical layers only: the shared
//! trailing layers attend into the planes of the last physical layer of their
//! own kind, so they cost no cache bytes. The derivation reads the family
//! descriptor, so the geometry has one source of truth.

#![cfg_attr(
    all(not(test), not(feature = "native-cuda")),
    expect(
        dead_code,
        reason = "the host build has no caller; the native-cuda Bolt loader admits it"
    )
)]

use super::config::MaxSeqs;
use crate::family::device_budget::KvPlan;
use crate::family::seq_plan::SeqCost;
use crate::family::spec::AttentionKind;
use crate::family::spec::NEMOTRON_BOLT_E4B;

/// KV8 bytes one context position costs on one layer: one code byte per k and v
/// value plus one fp16 scale per 32-value block of each plane.
fn kv8_layer_bytes(heads: u32, head: u32) -> Option<u64> {
    let codes = (heads as u64).checked_mul(head as u64)? * 2;
    let scales = (heads as u64).checked_mul(head as u64 / 32)? * 2 * 2;
    codes.checked_add(scales)
}

/// The Bolt cache cost of one context position, summed over the twenty-four
/// physical layers (twenty sliding at head 256, four full at head 512).
/// `None` if the arithmetic overflows.
pub(crate) fn bolt_kv_bytes_per_position() -> Option<u64> {
    let spec = NEMOTRON_BOLT_E4B;
    let mut total = 0u64;
    for &kind in &spec.layers_attention[..spec.first_shared_layer() as usize] {
        let head = match kind {
            AttentionKind::Full => spec.full_head_dim(),
            _ => spec.head_dim,
        };
        total = total.checked_add(kv8_layer_bytes(spec.heads_kv, head)?)?;
    }
    Some(total)
}

/// Device bytes the load keeps free beside its weights, KV cache and scratch.
/// The resident admission folds it into the demand through the shared
/// `plan::admit`, the same headroom rule the other resident executors use, so
/// an exact fit of the allocations alone is refused.
pub(crate) const BOLT_HEADROOM_BYTES: u64 = 512 << 20;

/// Bytes one extra session holds beside its cache planes: the raw and the capped logits row its
/// slot adds to the window buffers, and its two position words.
#[cfg_attr(
    not(feature = "native-cuda"),
    allow(dead_code, reason = "only the CUDA starter plans sessions")
)]
pub(crate) const BOLT_SEQ_FIXED_BYTES: u64 = 2 * NEMOTRON_BOLT_E4B.vocab as u64 * 4 + 8;

/// The rows of one prefill piece of the Bolt step: the model takes this many prompt rows in one
/// launch of its projections.
pub(crate) const BOLT_PIECE_ROWS: usize = 8;

const _: () = assert!(BOLT_AUTO_PROMPT_ROWS >= BOLT_PIECE_ROWS);

/// The prompt rows per tick that the engine feeds while several sessions are live, when the entry
/// does not set `prompt_rows`.
#[cfg_attr(
    not(feature = "native-cuda"),
    allow(dead_code, reason = "only the CUDA starter reads the share")
)]
pub(crate) const BOLT_AUTO_PROMPT_ROWS: usize = 8;

/// Bolt's batching choices: idle prompts take whole pieces, and a pooled session that idles for the
/// default interval is released. Other executors keep the worker's defaults until their batching
/// is measured.
#[cfg_attr(
    not(feature = "native-cuda"),
    allow(dead_code, reason = "only the CUDA executor answers the policy")
)]
pub(crate) fn bolt_batch_policy() -> crate::family::executor::BatchPolicy {
    crate::family::executor::BatchPolicy {
        idle_pieces: crate::family::executor::IDLE_PIECES_PER_TICK,
        idle_trim: Some(crate::family::executor::DEFAULT_SESSION_IDLE_TRIM),
    }
}

/// What one more session costs the sequence planner.
#[cfg_attr(
    not(feature = "native-cuda"),
    allow(dead_code, reason = "only the CUDA starter plans sessions")
)]
pub(crate) fn bolt_seq_cost() -> Option<SeqCost> {
    Some(SeqCost {
        fixed: BOLT_SEQ_FIXED_BYTES,
        per_position: bolt_kv_bytes_per_position()?,
    })
}

/// The session request the start plans with, and what it says about the decode graph.
#[cfg_attr(
    not(feature = "native-cuda"),
    allow(dead_code, reason = "only the CUDA starter plans sessions")
)]
pub(crate) struct SessionAsk {
    /// The count to plan with.
    pub(crate) seqs: MaxSeqs,
    /// A line for the start output when the decode graph changed the request.
    pub(crate) note: Option<String>,
}

/// An explicit decode graph (`FIFI_BOLT_DECODE_GRAPH=1`) is a request the start honours. The graph
/// replays one session's one-token step, so with an automatic count the start plans one session
/// and says why; with an explicit count above one the windows win and the graph is off. Without the
/// graph the request is unchanged.
#[cfg_attr(
    not(feature = "native-cuda"),
    allow(dead_code, reason = "only the CUDA starter plans sessions")
)]
pub(crate) fn session_ask(seqs: MaxSeqs, graph_wanted: bool) -> SessionAsk {
    match (seqs, graph_wanted) {
        (MaxSeqs::Auto, true) => SessionAsk {
            seqs: MaxSeqs::Count(1),
            note: Some(
                "FIFI_BOLT_DECODE_GRAPH=1 asks for the decode graph, which serves one session: planning 1 session; set max_seqs to plan more, which turns the graph off"
                    .to_owned(),
            ),
        },
        (MaxSeqs::Count(count), true) if count > 1 => SessionAsk {
            seqs,
            note: Some(format!(
                "max_seqs {count} runs windows, which the decode graph does not capture: FIFI_BOLT_DECODE_GRAPH=1 is ignored"
            )),
        },
        _ => SessionAsk { seqs, note: None },
    }
}

/// The cache the resident executor holds: one plane set per configured position.
/// The starter admits this plan against free device memory before loading; the
/// positions come from the caller (the executor pins its cell count). `None`
/// if the per-position cost overflows.
pub(crate) fn bolt_kv_plan(positions: u64) -> Option<KvPlan> {
    Some(KvPlan {
        positions,
        bytes_per_position: bolt_kv_bytes_per_position()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bolt_costs_30464_bytes_per_position() {
        // Twenty sliding layers at 1088 bytes plus four full layers at 2176.
        assert_eq!(20 * 1088 + 4 * 2176, 30_464);
        assert_eq!(bolt_kv_bytes_per_position(), Some(30_464));
    }

    #[test]
    fn an_extra_session_costs_its_planes_and_its_window_rows() {
        let cost = bolt_seq_cost().expect("cost");
        assert_eq!(cost.per_position, 30_464);
        // Raw and capped logits rows of 262,144 floats, and two position words.
        assert_eq!(cost.fixed, 2 * 262_144 * 4 + 8);
    }

    #[test]
    fn an_explicit_decode_graph_with_an_automatic_count_plans_one_session() {
        let ask = session_ask(MaxSeqs::Auto, true);
        assert_eq!(ask.seqs, MaxSeqs::Count(1));
        assert!(
            ask.note
                .expect("a reason")
                .contains("FIFI_BOLT_DECODE_GRAPH=1")
        );
        // An explicit count above one wins over the graph, and says so.
        let ask = session_ask(MaxSeqs::Count(4), true);
        assert_eq!(ask.seqs, MaxSeqs::Count(4));
        assert!(ask.note.expect("a reason").contains("is ignored"));
        // One session keeps the graph silently; without the graph nothing changes.
        assert_eq!(session_ask(MaxSeqs::Count(1), true).note, None);
        let ask = session_ask(MaxSeqs::Auto, false);
        assert_eq!((ask.seqs, ask.note), (MaxSeqs::Auto, None));
    }

    /// What the automatic count plans on small cards, from the numbers of a start: the card, the
    /// file the memory plan picks (the 4,033.5 MiB Q4_K file when Q8_0 does not fit), the first
    /// session and the scratch, about 300 MiB the desktop holds, and the headroom.
    #[test]
    fn the_default_plans_one_session_on_a_6_gb_card_and_three_on_an_8_gb_card() {
        use crate::family::seq_plan::{SeqInputs, plan_seqs};
        let mib = |value: f64| (value * f64::from(1u32 << 20)) as u64;
        let cost = bolt_seq_cost().expect("cost");
        let first_session = 30_464 * 32_768;
        let plan = |card_mib: f64| {
            let free = mib(card_mib - 300.0 - 4_033.5 - 9.0) - first_session;
            plan_seqs(
                SeqInputs {
                    free,
                    headroom: BOLT_HEADROOM_BYTES,
                    cost,
                    context: 32_768,
                    row_cap: BOLT_PIECE_ROWS,
                    ceiling: BOLT_PIECE_ROWS,
                },
                None,
            )
            .expect("plan")
        };
        // 6,144 MiB: 849 MiB remain after the first session, 337 MiB above the headroom, which
        // is less than the 954 MiB of one more session.
        assert_eq!(plan(6_144.0).seqs, 1);
        // 8,192 MiB: 2,897 MiB remain, 2,385 above the headroom: two more sessions.
        assert_eq!(plan(8_192.0).seqs, 3);
    }

    #[test]
    fn bolt_opts_into_the_whole_piece_budget_and_the_thirty_second_trim() {
        let policy = bolt_batch_policy();
        assert_eq!(policy.idle_pieces, 8);
        assert_eq!(policy.idle_trim, Some(std::time::Duration::from_secs(30)));
        assert_ne!(policy, crate::family::executor::BatchPolicy::default());
    }

    #[test]
    fn the_headroom_keeps_half_a_gib_free() {
        // The resident admission folds this into the demand, so an exact fit of
        // the allocations alone is refused.
        assert_eq!(BOLT_HEADROOM_BYTES, 512 << 20);
    }

    #[test]
    fn one_layer_costs_codes_plus_scales() {
        // Sliding: 2 heads x 256 codes x 2 planes + 2 x 8 blocks x 2 B x 2 planes.
        assert_eq!(kv8_layer_bytes(2, 256), Some(1088));
        // Full: 2 heads x 512 codes x 2 planes + 2 x 16 blocks x 2 B x 2 planes.
        assert_eq!(kv8_layer_bytes(2, 512), Some(2176));
        // The humanizer local geometry stays in range of the arithmetic.
        assert_eq!(kv8_layer_bytes(8, 256), Some(4352));
        assert_eq!(kv8_layer_bytes(1, 512), Some(1088));
    }

    #[test]
    fn the_shared_layers_cost_nothing() {
        let spec = NEMOTRON_BOLT_E4B;
        assert_eq!(spec.first_shared_layer(), 24);
        assert_eq!(spec.layers, 42);
    }

    #[test]
    fn the_plan_reports_the_cell_count_charge() {
        // The executor's full cache: 32,768 cells at 30,464 bytes each is just
        // under one GiB, the number a serving preflight must admit.
        let plan = bolt_kv_plan(32_768).expect("bounded plan");
        assert_eq!(plan.positions, 32_768);
        assert_eq!(plan.bytes_per_position, 30_464);
        assert_eq!(plan.positions * plan.bytes_per_position, 998_244_352);
    }

    #[test]
    #[cfg(feature = "native-cuda")]
    fn the_allocation_geometry_matches_the_budget_derivation() {
        use crate::family::gemma4::resident::bolt::{
            GLOBAL_HEAD, KV_HEADS, LOCAL_HEAD, PHYSICAL_LAYERS,
        };
        let spec = NEMOTRON_BOLT_E4B;
        // The executor's hard-coded geometry is the descriptor's geometry.
        assert_eq!(KV_HEADS as u32, spec.heads_kv);
        assert_eq!(LOCAL_HEAD as u32, spec.head_dim);
        assert_eq!(GLOBAL_HEAD as u32, spec.full_head_dim());
        assert_eq!(PHYSICAL_LAYERS as u32, spec.first_shared_layer());
        // Sum the per-position allocation exactly as the executor lays out the
        // planes: k and v codes plus fp16 scales, full attention every sixth
        // layer. Sliding layers reserve the full cell count too — the window
        // bounds attention, not the allocation — so they cost full cells here.
        let mut allocated = 0u64;
        for layer in 0..PHYSICAL_LAYERS {
            let head = if layer % 6 == 5 {
                GLOBAL_HEAD
            } else {
                LOCAL_HEAD
            } as u64;
            let codes = 2 * KV_HEADS as u64 * head;
            let scales = 2 * KV_HEADS as u64 * (head / 32) * 2;
            allocated += codes + scales;
        }
        assert_eq!(Some(allocated), bolt_kv_bytes_per_position());
        assert_eq!(
            BOLT_PIECE_ROWS,
            crate::family::gemma4::resident::bolt::Bolt::PREFILL_BATCH
        );
    }
}
