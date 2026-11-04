//! The Flash-Next QSA attention on CUDA (`CudaContext::qsa_attn`) against the host
//! `attend_selected` of this module, which the Mac MLX path is checked against, at the Flash-Next
//! attention geometry: 24 query heads over 2 kv heads of 256 (a group of 12), rope over the first
//! 64 values with base 1e7, a sigmoid output gate, blocks of 4 positions.
//!
//! A cache of 1100 positions holds random keys and values rounded to fp16 (the device cache
//! type); the host reads the same rounded values, so the comparison measures the attention and
//! not the cache rounding. For each case a random query and gate row, a random query norm and
//! random block scores give the host's selection (`select_positions`); the device gets the host's
//! normed and rotated q (the input `attn_prep` would produce) and the selection's blocks in
//! ascending order (the form `qsa_select` writes). Cases cover a position with no completed block,
//! the first completed block, a block boundary on each side, selections smaller than the
//! completed blocks, every block selected (budget 2048), a single-block budget, and 1, 7, 32 and
//! 128 splits. Each output row (one head) must lie within `2^-19 * max(max|h|, 1)` of the host.

use super::cuda_tests::{Rng, download, f32_bytes, uniform};
use super::{QsaDims, attend_selected, select_positions};
use crate::EngError;
use crate::family::flash_next::math::{rms0, rope_half};
use crate::kernels::cuda::resident::qsa_attn::{QsaAttnCall, QsaAttnShape};
use crate::kernels::cuda::resident::qsa_index::QsaPosition;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, device_present};

const N_HEAD: usize = 24;
const N_KV: usize = 2;
const HEAD: usize = 256;
const RATIO: usize = 4;
const ROWS: usize = 1100;
const LD: usize = N_HEAD * 2 * HEAD;
/// 2^-19: the first device run measured at most 2.0e-7 over every case.
const CLASS: f64 = 1.0 / 524_288.0;

fn dims(budget: usize) -> QsaDims {
    QsaDims {
        heads_q: N_HEAD as u32,
        heads_kv: N_KV as u32,
        head_dim: HEAD as u32,
        idx_heads: 4,
        idx_dim: 128,
        ratio: RATIO as u32,
        budget: budget as u32,
        rotary_dim: 64,
        theta: 1.0e7,
        hidden: 64,
        eps: 1e-6,
    }
}

/// Random values rounded to fp16, as fp16 bits and as the f32 the host reads.
fn half_values(rng: &mut Rng, count: usize) -> (Vec<u8>, Vec<f32>) {
    let values = uniform(rng, -1.0, 1.0, count);
    let halves: Vec<half::f16> = values.iter().map(|&v| half::f16::from_f32(v)).collect();
    (
        halves
            .iter()
            .flat_map(|h| h.to_bits().to_le_bytes())
            .collect(),
        halves.iter().map(|h| h.to_f32()).collect(),
    )
}

struct Case {
    pos: usize,
    budget: usize,
    splits: usize,
}

/// One case's host output and device output.
fn run_case(
    gpu: (&CudaContext, &DeviceStream),
    cache: (&DeviceBuffer, &DeviceBuffer),
    host_kv: (&[f32], &[f32]),
    case: &Case,
    rng: &mut Rng,
) -> Result<(Vec<f32>, Vec<f32>), EngError> {
    let (ctx, stream) = gpu;
    let dims = dims(case.budget);
    let pos = case.pos as u32;
    let qp = uniform(rng, -2.0, 2.0, LD);
    let q_norm = uniform(rng, -0.5, 0.5, HEAD);
    let blocks = (case.pos + 1) / RATIO;
    let score: Vec<f64> = (0..blocks).map(|_| rng.unit()).collect();
    let selected = select_positions(dims, &score, pos)?;
    let host = attend_selected(dims, &qp, &q_norm, host_kv, &selected, pos)?;
    let take = blocks.min(case.budget / RATIO);
    let mut chosen: Vec<i32> = selected[..take * RATIO]
        .iter()
        .step_by(RATIO)
        .map(|&p| (p / RATIO) as i32)
        .collect();
    chosen.sort_unstable();
    chosen.resize(case.budget / RATIO, -1);
    let mut q = Vec::with_capacity(N_HEAD * HEAD);
    for h in 0..N_HEAD {
        let mut qh = rms0(&qp[h * 2 * HEAD..h * 2 * HEAD + HEAD], &q_norm, dims.eps)?;
        rope_half(&mut qh, dims.rotary_dim as usize, pos, dims.theta)?;
        q.extend(qh);
    }
    let shape = QsaAttnShape {
        n_head: N_HEAD,
        n_kv_head: N_KV,
        rows: ROWS,
        ratio: RATIO,
        budget_blocks: case.budget / RATIO,
        max_splits: case.splits,
    };
    let q_buf = ctx.upload(&f32_bytes(&q))?;
    let proj = ctx.upload(&f32_bytes(&qp))?;
    let chosen_bytes: Vec<u8> = chosen.iter().flat_map(|b| b.to_le_bytes()).collect();
    let chosen_buf = ctx.upload(&chosen_bytes)?;
    let part = ctx.upload(&vec![0xFF; shape.part_bytes()?])?;
    let out = ctx.upload(&vec![0xFF; N_HEAD * HEAD * 4])?;
    ctx.qsa_attn(
        stream,
        QsaAttnCall {
            shape,
            ld: LD,
            scale: 1.0 / 16.0,
            position: QsaPosition::Host(case.pos),
            q: whole(&q_buf)?,
            proj: whole(&proj)?,
            k_cache: whole(cache.0)?,
            v_cache: whole(cache.1)?,
            chosen: whole(&chosen_buf)?,
            part: whole(&part)?,
            out: whole(&out)?,
            status: None,
        },
    )?;
    stream.synchronize()?;
    Ok((host, download(ctx, &out, N_HEAD * HEAD)))
}

fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

/// The largest `|d - h|` of a head row over `max(max|h|, 1)` of that row; NaN counts as
/// infinite.
fn worst_row(dev: &[f32], host: &[f32]) -> f64 {
    dev.chunks(HEAD)
        .zip(host.chunks(HEAD))
        .map(|(d, h)| {
            let scale = h.iter().fold(1.0f64, |m, v| m.max(f64::from(*v).abs()));
            let worst = d.iter().zip(h).fold(0.0f64, |m, (d, h)| {
                let e = (f64::from(*d) - f64::from(*h)).abs();
                if e.is_nan() { f64::INFINITY } else { m.max(e) }
            });
            worst / scale
        })
        .fold(0.0, f64::max)
}

#[test]
fn cuda_qsa_attention_matches_the_host_over_the_host_selection() -> Result<(), EngError> {
    if !device_present("cuda_qsa_attention_matches_the_host_over_the_host_selection") {
        return Ok(());
    }
    let mut rng = Rng(0x0a77_e5d0);
    let (k_bits, k_host) = half_values(&mut rng, ROWS * N_KV * HEAD);
    let (v_bits, v_host) = half_values(&mut rng, ROWS * N_KV * HEAD);
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let k = ctx.upload(&k_bits)?;
    let v = ctx.upload(&v_bits)?;
    let cases = [
        Case {
            pos: 0,
            budget: 64,
            splits: 32,
        },
        Case {
            pos: 2,
            budget: 64,
            splits: 32,
        },
        Case {
            pos: 3,
            budget: 64,
            splits: 32,
        },
        Case {
            pos: 66,
            budget: 64,
            splits: 32,
        },
        Case {
            pos: 67,
            budget: 64,
            splits: 32,
        },
        Case {
            pos: 600,
            budget: 64,
            splits: 32,
        },
        Case {
            pos: 1099,
            budget: 64,
            splits: 32,
        },
        Case {
            pos: 1099,
            budget: 2048,
            splits: 32,
        },
        Case {
            pos: 1098,
            budget: 4,
            splits: 32,
        },
        Case {
            pos: 1097,
            budget: 64,
            splits: 1,
        },
        Case {
            pos: 1097,
            budget: 64,
            splits: 7,
        },
        Case {
            pos: 1097,
            budget: 64,
            splits: 128,
        },
    ];
    for case in &cases {
        let (host, dev) = run_case(
            (&ctx, &stream),
            (&k, &v),
            (&k_host, &v_host),
            case,
            &mut rng,
        )?;
        let err = worst_row(&dev, &host);
        println!(
            "REPORTED: flash-next qsa attention position {} budget {} splits {}: {err:.3e} \
             (class {CLASS:.3e})",
            case.pos, case.budget, case.splits
        );
        assert!(
            err <= CLASS,
            "position {} budget {} splits {}: {err:.3e} above the class",
            case.pos,
            case.budget,
            case.splits
        );
    }
    Ok(())
}
