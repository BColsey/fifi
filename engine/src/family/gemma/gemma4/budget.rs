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

/// The kinds of the physical layers, in layer order.
fn physical_kinds() -> &'static [AttentionKind] {
    let spec = NEMOTRON_BOLT_E4B;
    &spec.layers_attention[..spec.first_shared_layer() as usize]
}

fn layer_head(kind: AttentionKind) -> u32 {
    match kind {
        AttentionKind::Full => NEMOTRON_BOLT_E4B.full_head_dim(),
        _ => NEMOTRON_BOLT_E4B.head_dim,
    }
}

/// The Bolt cache cost of one context position: only the full-attention layers (four, at head
/// 512) keep every position. A sliding layer keeps a ring of fixed size, charged per session by
/// [`bolt_ring_bytes`]. `None` if the arithmetic overflows.
pub(crate) fn bolt_kv_bytes_per_position() -> Option<u64> {
    let spec = NEMOTRON_BOLT_E4B;
    let mut total = 0u64;
    for &kind in physical_kinds() {
        if kind == AttentionKind::Full {
            total = total.checked_add(kv8_layer_bytes(spec.heads_kv, layer_head(kind))?)?;
        }
    }
    Some(total)
}

/// The cells of one sliding layer's ring when a step appends at most `rows` tokens: the same
/// formula the kernels admit by.
pub(crate) fn bolt_ring_cells(rows: usize) -> Option<usize> {
    crate::family::kv_ring::ring_cells(NEMOTRON_BOLT_E4B.sliding_window as usize, rows)
}

/// The bytes of every sliding layer's ring, which each session holds from its first token
/// whatever its context: `None` if the arithmetic overflows.
pub(crate) fn bolt_ring_bytes(rows: usize) -> Option<u64> {
    let spec = NEMOTRON_BOLT_E4B;
    let cells = u64::try_from(bolt_ring_cells(rows)?).ok()?;
    let mut total = 0u64;
    for &kind in physical_kinds() {
        if kind != AttentionKind::Full {
            total = total.checked_add(
                cells.checked_mul(kv8_layer_bytes(spec.heads_kv, layer_head(kind))?)?,
            )?;
        }
    }
    Some(total)
}

/// The bytes of a prefix snapshot of `fed` positions: every position of the full-attention layers
/// and the last `sliding_window` positions of the sliding layers. The budget tests and the
/// snapshot device test read it; nothing sizes by it at start yet.
#[cfg(test)]
pub(crate) fn bolt_snapshot_bytes(fed: usize) -> Option<u64> {
    let spec = NEMOTRON_BOLT_E4B;
    let fed = u64::try_from(fed).ok()?;
    let kept = fed.min(u64::from(spec.sliding_window));
    let mut total = 0u64;
    for &kind in physical_kinds() {
        let positions = if kind == AttentionKind::Full {
            fed
        } else {
            kept
        };
        total = total.checked_add(
            positions.checked_mul(kv8_layer_bytes(spec.heads_kv, layer_head(kind))?)?,
        )?;
    }
    Some(total)
}

/// The bytes of the sliding layers' kept window in a prefix snapshot: the last `sliding_window`
/// positions of every sliding layer.
pub(crate) fn bolt_window_snapshot_bytes() -> Option<u64> {
    let spec = NEMOTRON_BOLT_E4B;
    let mut total = 0u64;
    for &kind in physical_kinds() {
        if kind != AttentionKind::Full {
            total = total.checked_add(
                u64::from(spec.sliding_window)
                    .checked_mul(kv8_layer_bytes(spec.heads_kv, layer_head(kind))?)?,
            )?;
        }
    }
    Some(total)
}

/// Device bytes the load keeps free beside its weights, KV cache and scratch.
/// The resident admission folds it into the demand through the shared
/// `plan::admit`, the same headroom rule the other resident executors use, so
/// an exact fit of the allocations alone is refused.
pub(crate) const BOLT_HEADROOM_BYTES: u64 = crate::family::mem::DEVICE.bolt_headroom;

/// Bytes one extra session holds beside its cache planes: the raw and the capped logits row its
/// slot adds to the window buffers, and its two position words.
#[cfg_attr(
    not(feature = "native-cuda"),
    allow(dead_code, reason = "only the CUDA starter plans sessions")
)]
pub(crate) const BOLT_SEQ_FIXED_BYTES: u64 = 2 * NEMOTRON_BOLT_E4B.vocab as u64 * 4 + 8;

/// The prefill chunk sizes a Bolt start may plan, largest first: the shared prefill admission
/// takes the largest the card leaves room for. The smallest is one decode window.
pub(crate) const BOLT_PREFILL_LADDER: &[usize] = &[64, 32, 16, 8];

/// The most sessions one decode window steps together (the decode attention and the head launch
/// are planned for this many rows); the session planner's row cap and ceiling.
pub(crate) const BOLT_WINDOW_SEQS: usize = 8;

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

/// What one more session costs the sequence planner: its logits rows and position words, its
/// sliding layers' rings for a prefill chunk of `rows`, and the cells of its full-attention layers.
#[cfg_attr(
    not(feature = "native-cuda"),
    allow(dead_code, reason = "only the CUDA starter plans sessions")
)]
pub(crate) fn bolt_seq_cost(rows: usize) -> Option<SeqCost> {
    Some(SeqCost {
        fixed: BOLT_SEQ_FIXED_BYTES.checked_add(bolt_ring_bytes(rows)?)?,
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
    fn a_context_position_costs_the_four_full_layers_only() {
        // Four full layers at 2176 bytes; the twenty sliding layers keep a ring instead.
        assert_eq!(bolt_kv_bytes_per_position(), Some(4 * 2176));
        assert_eq!(bolt_kv_bytes_per_position(), Some(8_704));
    }

    #[test]
    fn a_session_holds_the_rings_of_its_sliding_layers_from_its_first_token() {
        // Window 512 and a chunk of 64 rows: 575 cells at 1088 bytes in each of twenty layers.
        assert_eq!(bolt_ring_cells(64), Some(575));
        assert_eq!(bolt_ring_bytes(64), Some(20 * 575 * 1088));
        assert_eq!(bolt_ring_bytes(8), Some(20 * 519 * 1088));
        // The kept window of a snapshot is 512 positions of the same layers.
        assert_eq!(bolt_window_snapshot_bytes(), Some(20 * 512 * 1088));
        // The one formula: the ring a layer is built with is the one the budget charges.
        let ring = NEMOTRON_BOLT_E4B.kv_ring(AttentionKind::Sliding, 64);
        assert_eq!(ring.cells(usize::MAX), 575);
    }

    #[test]
    fn a_snapshot_costs_every_full_position_and_one_window_of_the_sliding_layers() {
        // Short of the window every layer keeps all its positions.
        assert_eq!(bolt_snapshot_bytes(100), Some(100 * 30_464));
        assert_eq!(bolt_snapshot_bytes(512), Some(512 * 30_464));
        // Past it the sliding layers stop growing: the window plus the full layers' positions.
        let window = bolt_window_snapshot_bytes().expect("window bytes");
        for fed in [513usize, 2_000, 32_768] {
            assert_eq!(
                bolt_snapshot_bytes(fed),
                Some(fed as u64 * bolt_kv_bytes_per_position().expect("per position") + window),
                "fed {fed}"
            );
        }
        assert_eq!(bolt_snapshot_bytes(0), Some(0));
    }

    #[test]
    fn an_extra_session_costs_its_planes_and_its_window_rows() {
        let cost = bolt_seq_cost(64).expect("cost");
        assert_eq!(cost.per_position, 8_704);
        // Raw and capped logits rows of 262,144 floats, two position words, and the rings.
        assert_eq!(cost.fixed, 2 * 262_144 * 4 + 8 + 20 * 575 * 1088);
    }

    #[test]
    fn the_ring_cuts_what_a_long_session_costs_by_the_sliding_layers_share() {
        // A session of 32,768 cells: it cost 30,464 bytes a cell before.
        let before = 30_464u64 * 32_768;
        let cost = bolt_seq_cost(64).expect("cost");
        let after = cost.per_position * 32_768 + (cost.fixed - BOLT_SEQ_FIXED_BYTES);
        assert!(after * 3 < before, "{after} against {before}");
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
    fn the_default_plans_four_sessions_on_a_6_gb_card_and_eight_on_an_8_gb_card() {
        use crate::family::seq_plan::{SeqInputs, plan_seqs};
        let mib = |value: f64| (value * f64::from(1u32 << 20)) as u64;
        let cost = bolt_seq_cost(8).expect("cost");
        let first_session = cost.per_position * 32_768 + (cost.fixed - BOLT_SEQ_FIXED_BYTES);
        let plan = |card_mib: f64| {
            let free = mib(card_mib - 300.0 - 4_033.5 - 9.0) - first_session;
            plan_seqs(
                SeqInputs {
                    free,
                    headroom: BOLT_HEADROOM_BYTES,
                    cost,
                    context: 32_768,
                    row_cap: BOLT_WINDOW_SEQS,
                    ceiling: BOLT_WINDOW_SEQS,
                },
                None,
            )
            .expect("plan")
        };
        // 6,144 MiB: 1,519 MiB remain after the first session, 1,007 above the headroom, which
        // funds three more sessions of 285 MiB (one used to fit: 954 MiB each).
        assert_eq!(plan(6_144.0).seqs, 4);
        // 8,192 MiB: 3,055 MiB above the headroom fund ten more; the window of eight rows caps it
        // (three before the rings).
        assert_eq!(plan(8_192.0).seqs, 8);
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
        // The full-attention layers' cache: 32,768 cells at 8,704 bytes each; the sliding layers'
        // rings are charged per session beside it.
        let plan = bolt_kv_plan(32_768).expect("bounded plan");
        assert_eq!(plan.positions, 32_768);
        assert_eq!(plan.bytes_per_position, 8_704);
        assert_eq!(plan.positions * plan.bytes_per_position, 285_212_672);
    }

    #[test]
    #[cfg(feature = "native-cuda")]
    fn the_allocation_geometry_matches_the_budget_derivation() {
        use crate::family::gemma::gemma4::resident::bolt::{
            GLOBAL_HEAD, KV_HEADS, LOCAL_HEAD, PHYSICAL_LAYERS,
        };
        let spec = NEMOTRON_BOLT_E4B;
        // The executor's hard-coded geometry is the descriptor's geometry.
        assert_eq!(KV_HEADS as u32, spec.heads_kv);
        assert_eq!(LOCAL_HEAD as u32, spec.head_dim);
        assert_eq!(GLOBAL_HEAD as u32, spec.full_head_dim());
        assert_eq!(PHYSICAL_LAYERS as u32, spec.first_shared_layer());
        // Sum the per-position allocation exactly as the executor lays out the planes: k and v
        // codes plus fp16 scales. Only the full-attention layers (every sixth) grow with the
        // context; each sliding layer holds `bolt_ring_cells` cells whatever the context.
        let (mut grows, mut rings) = (0u64, 0u64);
        for layer in 0..PHYSICAL_LAYERS {
            let global = layer % 6 == 5;
            let head = if global { GLOBAL_HEAD } else { LOCAL_HEAD } as u64;
            let codes = 2 * KV_HEADS as u64 * head;
            let scales = 2 * KV_HEADS as u64 * (head / 32) * 2;
            if global {
                grows += codes + scales;
            } else {
                rings += (codes + scales) * bolt_ring_cells(8).expect("cells") as u64;
            }
        }
        assert_eq!(Some(grows), bolt_kv_bytes_per_position());
        assert_eq!(Some(rings), bolt_ring_bytes(8));
        assert_eq!(
            BOLT_WINDOW_SEQS,
            crate::family::gemma::gemma4::resident::bolt::Bolt::WINDOW_ROWS
        );
        // The smallest rung is one decode window; the ladder runs largest first.
        assert_eq!(BOLT_PREFILL_LADDER.last(), Some(&BOLT_WINDOW_SEQS));
        assert!(BOLT_PREFILL_LADDER.windows(2).all(|pair| pair[0] > pair[1]));
    }
}
