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
//! - a nonzero starting position (37): the rope phase is absolute;
//! - the cache contents immediately after the append (int8 codes AND fp16
//!   scale bits), not a downstream logit;
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
use super::native_device_rig::device_present;
use super::shard::ShardTable;
use crate::EngError;
use crate::family::project::cuda::served_admission;
use crate::kernels::cuda::peer_ring::PeerRing;
use crate::kernels::cuda::resident::attn_prep::{AttnPrepRowsCall, RopeParams};
use crate::kernels::cuda::resident::kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

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

/// The fabricated value of one GLOBAL fused-row column: distinct per
/// column, spread around a quarter, never repeating per card.
fn column_value(global_col: usize) -> f32 {
    0.25 + ((global_col * 7 + 3) % 997) as f32 / 64.0
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

/// The full fused projection row in the rows-kernel layout
/// (resident/attn_prep.h: per q head h at column h*512 sits
/// `[q 256 | gate 256]`, then the k heads, then the v heads), and the
/// per-card fused rows sliced out of it by the table.
fn fused_rows(
    full: &MixerGeometry,
    q_runs: &[std::ops::Range<usize>],
    kv_runs: &[std::ops::Range<usize>],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let a = &full.attn;
    let mut row = vec![0.0; a.ld];
    for head in 0..a.heads_q {
        let base = head * 2 * a.head_dim;
        for c in 0..a.head_dim {
            row[base + c] = column_value(head * a.head_dim + c);
            row[base + a.head_dim + c] = column_value(a.q_width + head * a.head_dim + c);
        }
    }
    for col in 0..a.kv_rows {
        row[a.k_off + col] = column_value(a.k_off + col);
        row[a.v_off + col] = column_value(a.v_off + col);
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
}

struct CardResult {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    codes: (Vec<u8>, Vec<u8>),
    scales: (Vec<u8>, Vec<u8>),
}

/// The whole device phase: `?` only, asserts live in the caller, every
/// device owner drops when this returns.
fn run_two_context_pipeline() -> Result<Pipeline, EngError> {
    let admission = served_admission();
    let table = ShardTable::new(&dims().shape, 2, 256)?;
    let full = MixerGeometry::new(&dims())?;
    let geometry = |card| MixerGeometry::for_card(&dims(), &table, card);
    let card_geometry = [geometry(0)?, geometry(1)?];
    let kv_runs = [table.kv_rows(0)?, table.kv_rows(1)?];
    let q_runs = [table.query_heads(0)?, table.query_heads(1)?];
    let (full_row, card_rows) = fused_rows(&full, &q_runs, &kv_runs);
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
    // own kv8 append into its own planes.
    let reference_context = &contexts[0];
    let reference_stream = &streams[0];
    let ref_proj = upload_on(reference_context, reference_stream, &f32_bytes(&full_row))?;
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
    let ref_pos = upload_on(
        reference_context,
        reference_stream,
        &u32_bytes(&[POS as u32]),
    )?;
    let ref_status = upload_on(reference_context, reference_stream, &u32_bytes(&[0]))?;
    reference_context.attn_prep_rows(
        reference_stream,
        AttnPrepRowsCall {
            shape: card_prep_shape(&full, full.attn.heads_kv, CELLS),
            rope,
            admit_pos: POS as usize,
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
    reference_stream.synchronize()?;

    let mut sides = Vec::new();
    for card in 0..2 {
        let context = &contexts[card];
        let g = &card_geometry[card];
        let kv_heads = card_kv_heads((kv_runs[card].start, kv_runs[card].end), g.attn.head_dim)?;
        let proj = upload_on(context, &streams[card], &f32_bytes(&card_rows[card]))?;
        let q = upload_on(
            context,
            &streams[card],
            &f32_bytes(&vec![0.0; g.attn.q_width]),
        )?;
        let k_rows = upload_on(context, &streams[card], &f32_bytes(&vec![0.0; kv_total]))?;
        let v_rows = upload_on(context, &streams[card], &f32_bytes(&vec![0.0; kv_total]))?;
        let norm_buf = upload_on(context, &streams[card], &f32_bytes(&norm))?;
        let pos = upload_on(context, &streams[card], &u32_bytes(&[POS as u32]))?;
        let status = upload_on(context, &streams[card], &u32_bytes(&[0]))?;
        context.attn_prep_rows(
            &streams[card],
            AttnPrepRowsCall {
                shape: card_prep_shape(g, kv_heads, CELLS),
                rope,
                admit_pos: POS as usize,
                proj: proj.range(0, g.attn.ld * F32)?,
                q_norm: norm_buf.range(0, norm_bytes)?,
                k_norm: norm_buf.range(0, norm_bytes)?,
                k_rows: k_rows.range(
                    card_slice_offset((kv_runs[card].start, kv_runs[card].end)),
                    kv_runs[card].len() * F32,
                )?,
                v_rows: v_rows.range(
                    card_slice_offset((kv_runs[card].start, kv_runs[card].end)),
                    kv_runs[card].len() * F32,
                )?,
                q: q.range(0, g.attn.q_width * F32)?,
                pos: pos.range(0, 4)?,
                status: status.range(0, 4)?,
            },
        )?;
        sides.push(CardSide {
            q,
            k_rows,
            v_rows,
            pos,
            status,
            gather_status: upload_on(context, &streams[card], &u32_bytes(&[0]))?,
            k_codes: upload_on(context, &streams[card], &vec![0u8; code_bytes])?,
            k_scales: upload_on(context, &streams[card], &vec![0u8; scale_bytes])?,
            v_codes: upload_on(context, &streams[card], &vec![0u8; code_bytes])?,
            v_scales: upload_on(context, &streams[card], &vec![0u8; scale_bytes])?,
        });
    }

    // The two gathers, K then V, each one ring sequence. Parts are the
    // card's own slices, outs the full rows.
    let ring = PeerRing::new(&contexts, kv_total)?;
    for half in ['k', 'v'] {
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

    // The both-halves gate, BEFORE any append: every card's gather status
    // word must read back clean, or the device phase ends here with the
    // session-fatal refusal and no append launched.
    for card in 0..2 {
        gathers_clean(&download_u32(&contexts[card], &sides[card].gather_status)?)?;
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
    })
}

#[test]
fn two_contexts_prep_gather_and_append_the_full_kv_row() {
    if !device_present("two_contexts_prep_gather_and_append_the_full_kv_row") {
        return;
    }
    let pipeline = run_two_context_pipeline().expect("device phase");
    let table = ShardTable::new(&dims().shape, 2, 256).expect("table");
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
        // The cache planes immediately after the append: codes and scale
        // bits, identical across cards and to the reference.
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
