//! The card attention core's driver pieces, stage 1: the head-alignment
//! rule a shard table's K/V deal must satisfy before a card prep exists,
//! the card prep's shape over the EXISTING rows kernel, and the
//! both-halves gate between the two K/V gathers and the cache append.
//!
//! The prep reuses `attn_prep_rows` unchanged: the fused projection row
//! layout derives from head counts, so the CARD's counts rebuild the
//! card's own offsets (per q head at column h*512, `[q 256 | gate 256]`,
//! then the k heads, then the v heads - see resident/attn_prep.h), the norm
//! weights are one shared `[head_dim]` row every head reads, and the rope
//! phase is the absolute token position the device word already carries.
//! Only the row slots are sliced: this card's K and V extents sit at their
//! GLOBAL rows of the full replicated row buffers, so the gathered row is
//! assembled in place and the append consumes a complete row or nothing.
use crate::EngError;
use crate::kernels::cuda::resident::attn_prep::{AttnPrepShape, SlotMode};

use super::mixer_geometry::MixerGeometry;

/// The card prep's K/V head count, with the rule that guards it: the shard
/// table deals K/V rows in 32-row blocks, but the rows kernel's per-head
/// norm and rope run over WHOLE heads - a range that splits a head has no
/// card prep. Two cards of the 35B shape (8 KV heads of 256 rows) split
/// evenly; a table whose deal does not is refused here, in planning, not
/// mis-prepped mid-token.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the multi-device step, its first production caller"
    )
)]
pub(super) fn card_kv_heads(k_range: (usize, usize), head_dim: usize) -> Result<usize, EngError> {
    let (start, end) = k_range;
    if start % head_dim != 0 || end % head_dim != 0 {
        return Err(EngError::BadRequest(format!(
            "card attn: K/V rows {start}..{end} split a {head_dim}-row head; the card prep \
             takes whole heads"
        )));
    }
    Ok((end - start) / head_dim)
}

/// The card prep's shape: the rows kernel over the card geometry's counts
/// and fused row, one decode row, over a cache of `capacity` cells. The
/// same builder states the one-context reference shape of the rig (the
/// full geometry with the full head counts).
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the multi-device step, its first production caller"
    )
)]
pub(super) fn card_prep_shape(
    g: &MixerGeometry,
    kv_heads: usize,
    capacity: usize,
) -> AttnPrepShape {
    let a = &g.attn;
    AttnPrepShape {
        m: 1,
        n_head: a.heads_q,
        n_kv_head: kv_heads,
        head_dim: a.head_dim,
        n_rot: a.n_rot,
        has_gate: true,
        slot: SlotMode::Position,
        rows: capacity,
        ld: a.ld,
    }
}

/// The byte offset of this card's extent inside a full replicated K/V row
/// buffer: the range's global first K row, as f32 bytes. Whole heads keep
/// the slice 16-byte aligned, which the rows entry's own fit checks then
/// re-prove on the range it is handed.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the multi-device step, its first production caller"
    )
)]
pub(super) fn card_slice_offset(k_range: (usize, usize)) -> usize {
    k_range.0 * 4
}

/// The both-halves gate: the cache append runs only when every gather
/// status word read back clean. A timeout on either half must not reach
/// the append - the token is poisoned, and v1 abandons the session: a
/// stale-V append (K gathered, V timed out) would write a half-valid row
/// into a cache a live session could keep serving from. The two halves
/// share no sequence word, which is exactly why this gate exists (see
/// `TokenOp::KvGatherV`).
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the multi-device step, its first production caller"
    )
)]
pub(super) fn gathers_clean(statuses: &[u32]) -> Result<(), EngError> {
    if statuses.iter().all(|word| *word == 0) {
        return Ok(());
    }
    Err(EngError::Checkpoint(
        "card attn: a K/V gather did not complete; the session is abandoned before the cache \
         append"
            .to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_card_prep_takes_whole_heads_and_refuses_a_split_one() {
        assert_eq!(card_kv_heads((0, 1024), 256).unwrap(), 4);
        assert_eq!(card_kv_heads((1024, 2048), 256).unwrap(), 4);
        // A 3-card deal of 2048 rows cuts at 704: mid-head, refused.
        assert!(card_kv_heads((0, 704), 256).is_err());
        assert!(card_kv_heads((704, 1376), 256).is_err());
    }

    #[test]
    fn the_gate_passes_only_when_every_half_is_clean() {
        gathers_clean(&[0, 0]).expect("both halves clean");
        // K gathered, V timed out: the exact half-failure the gate exists
        // for. Neither half may carry the other's completion.
        let error = gathers_clean(&[0, 2]).expect_err("a timed-out half");
        assert!(format!("{error:?}").contains("abandoned"), "{error:?}");
        assert!(gathers_clean(&[1, 0]).is_err());
        assert!(gathers_clean(&[0, 0, 8]).is_err());
    }
}
