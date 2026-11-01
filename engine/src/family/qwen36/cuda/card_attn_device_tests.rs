//! Two-logical-context device rig of the card attention core's first
//! stage, on the one physical device of this box: per-card prep through
//! the real rows kernel, both K/V gathers through the real peer ring, the
//! both-halves gate, and the Int8G32 append through the real kv8 kernel -
//! compared EXACTLY, cache planes included, against the one-context
//! full-shape reference on the same inputs.
//!
//! What the rig pins, per the slice contract:
//! - both KV-head groups: card 1's extent starts at global row 1024, a
//!   nonzero offset no local index reproduces;
//! - nonidentical cards: every fabricated column value depends on its
//!   GLOBAL column, so an accidental local/global mix-up cannot pass;
//! - a nonzero starting position (37) and successive tokens (the cursor
//!   advances 37, 38, ... while every fabricated value changes per step,
//!   so a stale row or a frozen cursor cannot pass);
//! - the cache contents after the last append (int8 codes AND fp16
//!   scale bits), not a downstream logit;
//! - the poison path: with the V gather dead, the both-halves gate
//!   refuses with the session-fatal error, the status words fold to
//!   exactly the ring-timeout step bit, and NO append runs (the planes
//!   stay zero) even though the K gather completed;
//! - the aliasing rule of the in-place gather: card c's part is a slice of
//!   its own full-row buffer, safe because the ring publishes every part
//!   (reading the slice) BEFORE any owner's wait_copy writes, and the
//!   publishes and copies of one card share that card's stream; no card
//!   ever writes another card's buffer.
//!
//! The Fp16 arm (the `kv_f16_append` kernel) is staged: it lands with its
//! own kernel slice, and until then the card path refuses to append fp16.
use super::card_attn::{card_kv_heads, card_prep_shape, card_slice_offset, gathers_clean};
use super::common_tests::dims;
use super::mixer_geometry::MixerGeometry;
use super::multi::{STEP_RING_TIMEOUT, fold_ring_status};
use super::native_device_rig::device_present;
use super::shard::ShardTable;
use crate::EngError;
use crate::family::project::cuda::served_admission;
use crate::kernels::cuda::peer_ring::{PeerRing, STATUS_TIMEOUT};
use crate::kernels::cuda::resident::attn_prep::{AttnPrepRowsCall, RopeParams};
use crate::kernels::cuda::resident::kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

/// The cache capacity of the rig: above the starting position, small
/// enough to keep the planes tiny.
const CELLS: usize = 64;
/// The nonzero starting position of the appended row: rope phases and
/// plane writes must both be absolute.
const POS: i32 = 37;
const F32: usize = size_of::<f32>();

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn download_f32(context: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>, EngError> {
    let mut bytes = vec![0u8; buffer.bytes()];
    context.download(buffer, &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn download_bytes(context: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<u8>, EngError> {
    let mut bytes = vec![0u8; buffer.bytes()];
    context.download(buffer, &mut bytes)?;
    Ok(bytes)
}

fn download_u32(context: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<u32>, EngError> {
    Ok(download_bytes(context, buffer)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

/// The fabricated value of one GLOBAL fused-row column at one token
/// step: distinct per column and per step, spread around a quarter, so a
/// stale row or a cached gather can never pass as a later token's.
fn column_value(global_col: usize, step: usize) -> f32 {
    0.25 + ((global_col * 7 + step * 613 + 3) % 997) as f32 / 64.0
}

/// The card's k/v slice of a full-row buffer: its own extent only, at
/// its absolute offset.
fn kv_slice(buffer: &DeviceBuffer, kv_run: (usize, usize)) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(card_slice_offset(kv_run), (kv_run.1 - kv_run.0) * F32)
}

/// Upload with the copy queued on `stream` (ordered behind its work, then
/// waited on). The audit in NOW.md showed `upload` would also be safe — it
/// ends in a device-wide synchronize — this just keeps every phase of the
/// rig on one stream by construction.
fn upload_on(
    context: &CudaContext,
    stream: &DeviceStream,
    bytes: &[u8],
) -> Result<DeviceBuffer, EngError> {
    let buffer = context.allocate(bytes.len())?;
    buffer.write_at(stream, 0, bytes)?;
    Ok(buffer)
}

/// The full fused projection row of one token step in the rows-kernel
/// layout (resident/attn_prep.h: per q head h at column h*512 sits
/// `[q 256 | gate 256]`, then the k heads, then the v heads), and the
/// per-card fused rows sliced out of it by the table.
fn fused_rows(
    full: &MixerGeometry,
    q_runs: &[std::ops::Range<usize>],
    kv_runs: &[std::ops::Range<usize>],
    step: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let a = &full.attn;
    let mut row = vec![0.0; a.ld];
    for head in 0..a.heads_q {
        let base = head * 2 * a.head_dim;
        for c in 0..a.head_dim {
            row[base + c] = column_value(head * a.head_dim + c, step);
            row[base + a.head_dim + c] = column_value(a.q_width + head * a.head_dim + c, step);
        }
    }
    for col in 0..a.kv_rows {
        row[a.k_off + col] = column_value(a.k_off + col, step);
        row[a.v_off + col] = column_value(a.v_off + col, step);
    }
    let cards = q_runs
        .iter()
        .zip(kv_runs)
        .map(|(q_run, kv_run)| {
            let q_width = q_run.len() * 2 * a.head_dim;
            // kv_run counts KV-row floats (one head is head_dim of them),
            // unlike q_run, which counts heads.
            let kv_rows = kv_run.len();
            let mut card = vec![0.0; q_width + 2 * kv_rows];
            for (j, head) in q_run.clone().enumerate() {
                let base = head * 2 * a.head_dim;
                card[j * 2 * a.head_dim..(j + 1) * 2 * a.head_dim]
                    .copy_from_slice(&row[base..base + 2 * a.head_dim]);
            }
            for (j, col) in kv_run.clone().enumerate() {
                card[q_width + j] = row[a.k_off + col];
                card[q_width + kv_rows + j] = row[a.v_off + col];
            }
            card
        })
        .collect();
    (row, cards)
}

/// The buffers one card owns in its own context.
struct CardSide {
    q: DeviceBuffer,
    k_rows: DeviceBuffer,
    v_rows: DeviceBuffer,
    norm: DeviceBuffer,
    pos: DeviceBuffer,
    status: DeviceBuffer,
    gather_status: DeviceBuffer,
    k_codes: DeviceBuffer,
    k_scales: DeviceBuffer,
    v_codes: DeviceBuffer,
    v_scales: DeviceBuffer,
}

/// The readable outputs of the whole device phase, host data only.
struct Pipeline {
    reference_k: Vec<f32>,
    reference_v: Vec<f32>,
    reference_q: Vec<f32>,
    reference_codes: (Vec<u8>, Vec<u8>),
    reference_scales: (Vec<u8>, Vec<u8>),
    cards: Vec<CardResult>,
    /// The both-halves gate over the FINAL step's downloaded status
    /// words, exactly as production reads it before any append: `Ok` on a
    /// clean run; the refusal message when a gather was poisoned.
    gate: Result<(), String>,
    /// `fold_ring_status` of the same words: the step-status bits a
    /// poisoned gather leaves for the session teardown.
    folded: u32,
}

struct CardResult {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    codes: (Vec<u8>, Vec<u8>),
    scales: (Vec<u8>, Vec<u8>),
}

/// The whole device phase over `steps` successive tokens: `?` only,
/// asserts live in the caller, every device owner drops when this
/// returns. With `poison_v`, the V gather of the FIRST step is replaced
/// by the status word a timed-out waiter would leave, so the gate path
/// runs with a genuinely dead half and no append ever launches.
fn run_two_context_pipeline(steps: usize, poison_v: bool) -> Result<Pipeline, EngError> {
    let admission = served_admission();
    let table = ShardTable::new(&dims().shape, 2, 256)?;
    let full = MixerGeometry::new(&dims())?;
    let geometry = |card| MixerGeometry::for_card(&dims(), &table, card);
    let card_geometry = [geometry(0)?, geometry(1)?];
    let kv_runs = [table.kv_rows(0)?, table.kv_rows(1)?];
    let q_runs = [table.query_heads(0)?, table.query_heads(1)?];
    let card_kv = [
        card_kv_heads(
            (kv_runs[0].start, kv_runs[0].end),
            card_geometry[0].attn.head_dim,
        )?,
        card_kv_heads(
            (kv_runs[1].start, kv_runs[1].end),
            card_geometry[1].attn.head_dim,
        )?,
    ];
    let norm: Vec<f32> = (0..full.attn.head_dim)
        .map(|i| 1.0 + (i % 64) as f32 / 128.0)
        .collect();
    let rope = RopeParams {
        eps: full.eps,
        rope_base: full.attn.rope_base,
    };
    let kv_total = full.attn.kv_rows;
    let norm_bytes = norm.len() * F32;

    // The two logical cards: one context each, on the one device (the
    // adapter admits two per process, so the reference below shares card
    // 0's context - sequential phases over separate buffers).
    let contexts = [
        CudaContext::with_admission_on(0, &admission)?,
        CudaContext::with_admission_on(0, &admission)?,
    ];
    let streams = [contexts[0].stream()?, contexts[1].stream()?];

    // The one-context reference: the full shape over the full row, and its
    // own kv8 append per token step into its own planes.
    let reference_context = &contexts[0];
    let reference_stream = &streams[0];
    let ref_q = upload_on(
        reference_context,
        reference_stream,
        &f32_bytes(&vec![0.0; full.attn.q_width]),
    )?;
    let ref_k = upload_on(
        reference_context,
        reference_stream,
        &f32_bytes(&vec![0.0; kv_total]),
    )?;
    let ref_v = upload_on(
        reference_context,
        reference_stream,
        &f32_bytes(&vec![0.0; kv_total]),
    )?;
    let ref_norm = upload_on(reference_context, reference_stream, &f32_bytes(&norm))?;
    let ref_pos = upload_on(reference_context, reference_stream, &u32_bytes(&[0]))?;
    let ref_status = upload_on(reference_context, reference_stream, &u32_bytes(&[0]))?;
    let kv8_shape = |heads_kv: usize| Kv8Shape {
        rows: 1,
        n_kv_head: heads_kv,
        head_dim: full.attn.head_dim,
        cells: CELLS,
        code_scale: Kv8CodeScale::StoredHalf,
    };
    let code_bytes = full.attn.heads_kv * CELLS * full.attn.head_dim;
    let scale_bytes = full.attn.heads_kv * CELLS * full.attn.head_dim / 32 * 2;
    let ref_k_codes = upload_on(reference_context, reference_stream, &vec![0u8; code_bytes])?;
    let ref_k_scales = upload_on(reference_context, reference_stream, &vec![0u8; scale_bytes])?;
    let ref_v_codes = upload_on(reference_context, reference_stream, &vec![0u8; code_bytes])?;
    let ref_v_scales = upload_on(reference_context, reference_stream, &vec![0u8; scale_bytes])?;
    for step in 0..steps {
        let (full_row, _) = fused_rows(&full, &q_runs, &kv_runs, step);
        let pos = POS + step as i32;
        let ref_proj = upload_on(reference_context, reference_stream, &f32_bytes(&full_row))?;
        ref_pos.write_at(reference_stream, 0, &u32_bytes(&[pos as u32]))?;
        reference_context.attn_prep_rows(
            reference_stream,
            AttnPrepRowsCall {
                shape: card_prep_shape(&full, full.attn.heads_kv, CELLS),
                rope,
                admit_pos: pos as usize,
                proj: ref_proj.range(0, full.attn.ld * F32)?,
                q_norm: ref_norm.range(0, norm_bytes)?,
                k_norm: ref_norm.range(0, norm_bytes)?,
                k_rows: ref_k.range(0, kv_total * F32)?,
                v_rows: ref_v.range(0, kv_total * F32)?,
                q: ref_q.range(0, full.attn.q_width * F32)?,
                pos: ref_pos.range(0, 4)?,
                status: ref_status.range(0, 4)?,
            },
        )?;
        reference_context.kv8_append(
            reference_stream,
            kv8_shape(full.attn.heads_kv),
            Kv8Data {
                k: ref_k.range(0, kv_total * F32)?,
                v: ref_v.range(0, kv_total * F32)?,
                k_codes: ref_k_codes.range(0, code_bytes)?,
                k_scales: ref_k_scales.range(0, scale_bytes)?,
                v_codes: ref_v_codes.range(0, code_bytes)?,
                v_scales: ref_v_scales.range(0, scale_bytes)?,
            },
            ref_pos.range(0, 4)?,
            ref_status.range(0, 4)?,
        )?;
    }
    reference_stream.synchronize()?;

    let mut sides = Vec::new();
    for card in 0..2 {
        let context = &contexts[card];
        let g = &card_geometry[card];
        // Buffers persist across the token steps; every step's prep writes
        // q/k/v through its own uploaded proj row before anything reads
        // them.
        let q = upload_on(
            context,
            &streams[card],
            &f32_bytes(&vec![0.0; g.attn.q_width]),
        )?;
        let k_rows = upload_on(context, &streams[card], &f32_bytes(&vec![0.0; kv_total]))?;
        let v_rows = upload_on(context, &streams[card], &f32_bytes(&vec![0.0; kv_total]))?;
        let norm = upload_on(context, &streams[card], &f32_bytes(&norm))?;
        sides.push(CardSide {
            q,
            k_rows,
            v_rows,
            norm,
            pos: upload_on(context, &streams[card], &u32_bytes(&[POS as u32]))?,
            status: upload_on(context, &streams[card], &u32_bytes(&[0]))?,
            gather_status: upload_on(context, &streams[card], &u32_bytes(&[0]))?,
            k_codes: upload_on(context, &streams[card], &vec![0u8; code_bytes])?,
            k_scales: upload_on(context, &streams[card], &vec![0u8; scale_bytes])?,
            v_codes: upload_on(context, &streams[card], &vec![0u8; code_bytes])?,
            v_scales: upload_on(context, &streams[card], &vec![0u8; scale_bytes])?,
        });
    }

    // The per-step loop: prep both cards over that step's rows, gather K
    // then V (each one ring sequence), gate, append. The ring's sequence
    // state advances across steps by design (slot parity alternates),
    // which is exactly what a decode loop exercises.
    let ring = PeerRing::new(&contexts, kv_total)?;
    let mut gate = Ok(());
    let mut folded = 0u32;
    for step in 0..steps {
        let (_, card_rows_now) = fused_rows(&full, &q_runs, &kv_runs, step);
        let pos = POS + step as i32;
        for card in 0..2 {
            let g = &card_geometry[card];
            let proj = upload_on(
                &contexts[card],
                &streams[card],
                &f32_bytes(&card_rows_now[card]),
            )?;
            sides[card]
                .pos
                .write_at(&streams[card], 0, &u32_bytes(&[pos as u32]))?;
            let kv_run = (kv_runs[card].start, kv_runs[card].end);
            contexts[card].attn_prep_rows(
                &streams[card],
                AttnPrepRowsCall {
                    shape: card_prep_shape(g, card_kv[card], CELLS),
                    rope,
                    admit_pos: pos as usize,
                    proj: proj.range(0, g.attn.ld * F32)?,
                    q_norm: sides[card].norm.range(0, norm_bytes)?,
                    k_norm: sides[card].norm.range(0, norm_bytes)?,
                    k_rows: kv_slice(&sides[card].k_rows, kv_run)?,
                    v_rows: kv_slice(&sides[card].v_rows, kv_run)?,
                    q: sides[card].q.range(0, g.attn.q_width * F32)?,
                    pos: sides[card].pos.range(0, 4)?,
                    status: sides[card].status.range(0, 4)?,
                },
            )?;
        }
        for half in ['k', 'v'] {
            if half == 'v' && poison_v && step == 0 {
                // GatherV timed out on card 1: its waiter leaves the
                // timeout bit, exactly what wait_copy writes. Card 0's
                // word stays clean, as a clean waiter's does.
                sides[1]
                    .gather_status
                    .write_at(&streams[1], 0, &u32_bytes(&[STATUS_TIMEOUT]))?;
                break;
            }
            let mut parts = Vec::new();
            let mut outs = Vec::new();
            let mut statuses = Vec::new();
            for card in 0..2 {
                let buffer = if half == 'k' {
                    &sides[card].k_rows
                } else {
                    &sides[card].v_rows
                };
                parts.push(buffer.range(
                    card_slice_offset((kv_runs[card].start, kv_runs[card].end)),
                    kv_runs[card].len() * F32,
                )?);
                outs.push(buffer.range(0, kv_total * F32)?);
                statuses.push(sides[card].gather_status.range(0, 4)?);
            }
            ring.all_gather_row(
                &contexts, &streams, &parts, &outs, &statuses, &kv_runs, kv_total,
            )?;
        }
        streams[0].synchronize()?;
        streams[1].synchronize()?;

        // The both-halves gate, BEFORE any append: every card's gather
        // status word must read back clean, or the device phase ends here
        // with the session-fatal refusal and no append launched.
        let word = |card: usize| -> Result<u32, EngError> {
            Ok(download_u32(&contexts[card], &sides[card].gather_status)?[0])
        };
        let words = [word(0)?, word(1)?];
        folded |= fold_ring_status(&words);
        gate = gathers_clean(&words).map_err(|err| match err {
            EngError::Checkpoint(message) => message,
            other => format!("{other:?}"),
        });
        if gate.is_err() {
            break;
        }

        // The Int8G32 append on every card, over the full gathered rows.
        for card in 0..2 {
            contexts[card].kv8_append(
                &streams[card],
                kv8_shape(full.attn.heads_kv),
                Kv8Data {
                    k: sides[card].k_rows.range(0, kv_total * F32)?,
                    v: sides[card].v_rows.range(0, kv_total * F32)?,
                    k_codes: sides[card].k_codes.range(0, code_bytes)?,
                    k_scales: sides[card].k_scales.range(0, scale_bytes)?,
                    v_codes: sides[card].v_codes.range(0, code_bytes)?,
                    v_scales: sides[card].v_scales.range(0, scale_bytes)?,
                },
                sides[card].pos.range(0, 4)?,
                sides[card].status.range(0, 4)?,
            )?;
        }
        streams[0].synchronize()?;
        streams[1].synchronize()?;
    }

    let mut cards = Vec::new();
    for card in 0..2 {
        let context = &contexts[card];
        cards.push(CardResult {
            q: download_f32(context, &sides[card].q)?,
            k: download_f32(context, &sides[card].k_rows)?,
            v: download_f32(context, &sides[card].v_rows)?,
            codes: (
                download_bytes(context, &sides[card].k_codes)?,
                download_bytes(context, &sides[card].v_codes)?,
            ),
            scales: (
                download_bytes(context, &sides[card].k_scales)?,
                download_bytes(context, &sides[card].v_scales)?,
            ),
        });
    }
    Ok(Pipeline {
        reference_k: download_f32(reference_context, &ref_k)?,
        reference_v: download_f32(reference_context, &ref_v)?,
        reference_q: download_f32(reference_context, &ref_q)?,
        reference_codes: (
            download_bytes(reference_context, &ref_k_codes)?,
            download_bytes(reference_context, &ref_v_codes)?,
        ),
        reference_scales: (
            download_bytes(reference_context, &ref_k_scales)?,
            download_bytes(reference_context, &ref_v_scales)?,
        ),
        cards,
        gate,
        folded,
    })
}

/// The equality contract every clean run pins: gathered rows, q rows,
/// and the cache planes, per card, against the one-context reference.
fn assert_cards_match_reference(pipeline: &Pipeline, table: &ShardTable) {
    let head_dim = 256;
    for (card, result) in pipeline.cards.iter().enumerate() {
        // The gathered rows are EXACTLY the one-context reference: every
        // value, on both halves, from both cards' buffers.
        assert_eq!(result.k, pipeline.reference_k, "card {card} K rows");
        assert_eq!(result.v, pipeline.reference_v, "card {card} V rows");
        // The card's query slot equals the reference's head run (the
        // global head offsets, not local indices, select it).
        let run = table.query_heads(card).expect("q run");
        let start = run.start * head_dim;
        assert_eq!(
            &result.q[..],
            &pipeline.reference_q[start..start + result.q.len()],
            "card {card} q rows"
        );
        // The cache planes after the last append: codes and scale bits,
        // identical across cards and to the reference.
        assert_eq!(
            result.codes.0, pipeline.reference_codes.0,
            "card {card} k codes"
        );
        assert_eq!(
            result.codes.1, pipeline.reference_codes.1,
            "card {card} v codes"
        );
        assert_eq!(
            result.scales.0, pipeline.reference_scales.0,
            "card {card} k scales"
        );
        assert_eq!(
            result.scales.1, pipeline.reference_scales.1,
            "card {card} v scales"
        );
    }
}

#[test]
fn two_contexts_prep_gather_and_append_the_full_kv_row() {
    if !device_present("two_contexts_prep_gather_and_append_the_full_kv_row") {
        return;
    }
    let pipeline = run_two_context_pipeline(1, false).expect("device phase");
    let table = ShardTable::new(&dims().shape, 2, 256).expect("table");
    assert!(pipeline.gate.is_ok(), "a clean run must gate clean");
    assert_eq!(pipeline.folded, 0, "clean status words fold to zero");
    assert_cards_match_reference(&pipeline, &table);
}

#[test]
fn two_contexts_append_successive_tokens() {
    if !device_present("two_contexts_append_successive_tokens") {
        return;
    }
    // Three steps: the append cursor advances 37, 38, 39 while every
    // fabricated value changes per step, so a stale row, a cached
    // gather, or a pos that failed to advance cannot pass.
    let pipeline = run_two_context_pipeline(3, false).expect("device phase");
    let table = ShardTable::new(&dims().shape, 2, 256).expect("table");
    assert!(pipeline.gate.is_ok());
    assert_eq!(pipeline.folded, 0);
    assert_cards_match_reference(&pipeline, &table);
}

#[test]
fn a_poisoned_v_gather_stops_the_session_before_the_append() {
    if !device_present("a_poisoned_v_gather_stops_the_session_before_the_append") {
        return;
    }
    let pipeline = run_two_context_pipeline(1, true).expect("device phase");
    // The gate refuses: the exact session-fatal refusal a dead V gather
    // must produce, before any append is launched.
    let refusal = pipeline.gate.expect_err("a poisoned V gather");
    assert!(
        refusal.contains("did not complete"),
        "the refusal names the gather: {refusal}"
    );
    // The same words fold to exactly the ring-timeout step bit: no other
    // bit fires and no other source is blamed.
    assert_eq!(pipeline.folded, STEP_RING_TIMEOUT);
    // GatherK completed before the poison: both cards hold the full
    // gathered K row of step 0.
    for (card, result) in pipeline.cards.iter().enumerate() {
        assert_eq!(result.k, pipeline.reference_k, "card {card} K rows");
        // No append ran: the cache planes stay zero on every card, K and
        // V, codes and scales alike.
        assert!(
            result.codes.0.iter().all(|&b| b == 0),
            "card {card} k codes must stay untouched"
        );
        assert!(
            result.codes.1.iter().all(|&b| b == 0),
            "card {card} v codes must stay untouched"
        );
        assert!(
            result.scales.0.iter().all(|&b| b == 0) && result.scales.1.iter().all(|&b| b == 0),
            "card {card} scales must stay untouched"
        );
    }
}
