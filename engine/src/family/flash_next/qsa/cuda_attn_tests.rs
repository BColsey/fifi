//! The Flash-Next QSA attention on CUDA (`CudaContext::qsa_attn`) against the host
//! `attend_selected` of this module, which the Mac MLX path is checked against.
//!
//! The device gets what `attn_prep` and `qsa_select` would give it: the host's normed and rotated
//! q (rope over the first 64 values, base 1e7) and each token's selected blocks in ascending order.
//! Keys and values are random and rounded to fp16 (the device cache type); the host reads the same
//! rounded values, so the comparison measures the attention and not the cache rounding. Each
//! output row (one head of one token) must lie within `2^-19 * max(max|h|, 1)` of the host.
//!
//! Cases:
//! - the Flash-Next geometry (24 query heads over 2 kv heads of 256, a group of 12, blocks of 4,
//!   a cache of 1100 positions): a position with no completed block, the first completed block,
//!   a block boundary on each side, selections smaller than the completed blocks, every block
//!   selected (budget 512 blocks), a single-block budget, and 1, 7, 32 and 128 splits;
//! - prefill chunks in that geometry (37 tokens from position 600, 70 tokens from position 0):
//!   one call per chunk against the host per token, bit-identical to one call per token and to
//!   the same chunk at a device position word;
//! - the kernel limits: a group of 32 (1024-thread blocks) at ratio 3; 64 kv heads with a group
//!   of 1 at ratio 64; ratio 1 with a budget of 2^24 blocks; and a cache of 131200 positions of 64
//!   kv heads (4.3 GB, one buffer for keys and values), where the attended rows lie past 2^31
//!   fp16 values. That last case compares against the host on a compact cache of only the
//!   attended rows: the softmax does not depend on where a row sits, only on its values and its
//!   order, and the compact rows keep the device's order.

use super::cuda_tests::{Rng, download, f32_bytes, uniform};
use super::{QsaDims, attend_selected, select_positions};
use crate::EngError;
use crate::family::flash_next::math::{rms0, rope_half};
use crate::kernels::cuda::resident::qsa_attn::{QsaAttnCall, QsaAttnShape};
use crate::kernels::cuda::resident::qsa_index::QsaPosition;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, device_present};

const HEAD: usize = 256;
const ROTARY: u32 = 64;
const THETA: f32 = 1.0e7;
const EPS: f32 = 1e-6;
/// 2^-19: the first device run measured at most 2.0e-7 over every Flash-Next case.
const CLASS: f64 = 1.0 / 524_288.0;

#[derive(Clone, Copy, Debug)]
struct Geom {
    n_head: usize,
    n_kv: usize,
    ratio: usize,
    rows: usize,
    budget_blocks: usize,
    splits: usize,
}

const FLASH_NEXT: Geom = Geom {
    n_head: 24,
    n_kv: 2,
    ratio: 4,
    rows: 1100,
    budget_blocks: 16,
    splits: 32,
};

impl Geom {
    fn ld(self) -> usize {
        self.n_head * 2 * HEAD
    }

    fn host_dims(self) -> QsaDims {
        let as_u32 = |v: usize| u32::try_from(v).expect("dimension");
        QsaDims {
            heads_q: as_u32(self.n_head),
            heads_kv: as_u32(self.n_kv),
            head_dim: HEAD as u32,
            idx_heads: 4,
            idx_dim: 128,
            ratio: as_u32(self.ratio),
            budget: as_u32(self.budget_blocks * self.ratio),
            rotary_dim: ROTARY,
            theta: THETA,
            hidden: 64,
            eps: EPS,
        }
    }

    fn shape(self, m: usize) -> QsaAttnShape {
        QsaAttnShape {
            m,
            n_head: self.n_head,
            n_kv_head: self.n_kv,
            rows: self.rows,
            ratio: self.ratio,
            budget_blocks: self.budget_blocks,
            max_splits: self.splits,
        }
    }
}

/// One token's device inputs (q, its projection row, its chosen blocks) and host output.
struct Token {
    q: Vec<f32>,
    qp: Vec<f32>,
    chosen: Vec<i32>,
    host: Vec<f32>,
}

/// The host's normed and rotated q of every head of `qp` at `pos`.
fn rotated_q(geom: Geom, qp: &[f32], q_norm: &[f32], pos: u32) -> Result<Vec<f32>, EngError> {
    let mut q = Vec::with_capacity(geom.n_head * HEAD);
    for h in 0..geom.n_head {
        let mut qh = rms0(&qp[h * 2 * HEAD..h * 2 * HEAD + HEAD], q_norm, EPS)?;
        rope_half(&mut qh, ROTARY as usize, pos, THETA)?;
        q.extend(qh);
    }
    Ok(q)
}

/// A random token at `pos`: random block scores give the host's selection; the device gets its
/// blocks ascending, padded with -1 to the budget.
fn token(
    geom: Geom,
    host_kv: (&[f32], &[f32]),
    q_norm: &[f32],
    pos: usize,
    rng: &mut Rng,
) -> Result<Token, EngError> {
    let dims = geom.host_dims();
    let at = u32::try_from(pos).expect("position");
    let qp = uniform(rng, -2.0, 2.0, geom.ld());
    let blocks = (pos + 1) / geom.ratio;
    let score: Vec<f64> = (0..blocks).map(|_| rng.unit()).collect();
    let selected = select_positions(dims, &score, at)?;
    let host = attend_selected(dims, &qp, q_norm, host_kv, &selected, at)?;
    let take = blocks.min(geom.budget_blocks);
    let mut chosen: Vec<i32> = selected[..take * geom.ratio]
        .iter()
        .step_by(geom.ratio)
        .map(|&p| i32::try_from(p / geom.ratio).expect("block"))
        .collect();
    chosen.sort_unstable();
    chosen.resize(geom.budget_blocks, -1);
    Ok(Token {
        q: rotated_q(geom, &qp, q_norm, at)?,
        qp,
        chosen,
        host,
    })
}

/// A device context, a stream, the fp16 caches, and the host's copy of the caches.
struct Gpu<'a> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    k: &'a DeviceBuffer,
    v: &'a DeviceBuffer,
    host_kv: (&'a [f32], &'a [f32]),
}

/// The device output of one call of `tokens` at `position`, f32 `[m][n_head][256]`.
fn device_chunk(
    gpu: &Gpu<'_>,
    geom: Geom,
    tokens: &[&Token],
    position: QsaPosition<'_>,
) -> Result<Vec<f32>, EngError> {
    let ctx = gpu.ctx;
    let m = tokens.len();
    let shape = geom.shape(m);
    let gather = |f: fn(&Token) -> &[f32]| -> Vec<f32> {
        tokens.iter().flat_map(|t| f(t).iter().copied()).collect()
    };
    let q = ctx.upload(&f32_bytes(&gather(|t| t.q.as_slice())))?;
    let proj = ctx.upload(&f32_bytes(&gather(|t| t.qp.as_slice())))?;
    let chosen_bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|t| t.chosen.iter().flat_map(|b| b.to_le_bytes()))
        .collect();
    let chosen = ctx.upload(&chosen_bytes)?;
    let part = ctx.upload(&vec![0xFF; shape.part_bytes()?])?;
    let out = ctx.upload(&vec![0xFF; m * geom.n_head * HEAD * 4])?;
    ctx.qsa_attn(
        gpu.stream,
        QsaAttnCall {
            shape,
            ld: geom.ld(),
            scale: 1.0 / 16.0,
            position,
            q: whole(&q)?,
            proj: whole(&proj)?,
            k_cache: whole(gpu.k)?,
            v_cache: whole(gpu.v)?,
            chosen: whole(&chosen)?,
            part: whole(&part)?,
            out: whole(&out)?,
            status: None,
        },
    )?;
    gpu.stream.synchronize()?;
    Ok(download(ctx, &out, m * geom.n_head * HEAD))
}

fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
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

/// The largest `|d - h|` of a head row over `max(max|h|, 1)` of that row; NaN counts as
/// infinite.
fn worst_row(dev: &[f32], host: &[f32]) -> f64 {
    assert_eq!(dev.len(), host.len());
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

fn check(label: &str, dev: &[f32], host: &[f32]) {
    let err = worst_row(dev, host);
    println!("REPORTED: flash-next qsa attention {label}: {err:.3e} (class {CLASS:.3e})");
    assert!(err <= CLASS, "{label}: {err:.3e} above the class");
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Random fp16 caches of `geom` uploaded, and the host copies.
struct Cache {
    k: DeviceBuffer,
    v: DeviceBuffer,
    k_host: Vec<f32>,
    v_host: Vec<f32>,
}

impl Cache {
    fn gpu<'a>(&'a self, ctx: &'a CudaContext, stream: &'a DeviceStream) -> Gpu<'a> {
        Gpu {
            ctx,
            stream,
            k: &self.k,
            v: &self.v,
            host_kv: (&self.k_host, &self.v_host),
        }
    }
}

fn random_cache(ctx: &CudaContext, geom: Geom, rng: &mut Rng) -> Result<Cache, EngError> {
    let count = geom.rows * geom.n_kv * HEAD;
    let (k_bits, k_host) = half_values(rng, count);
    let (v_bits, v_host) = half_values(rng, count);
    Ok(Cache {
        k: ctx.upload(&k_bits)?,
        v: ctx.upload(&v_bits)?,
        k_host,
        v_host,
    })
}

#[test]
fn cuda_qsa_attention_matches_the_host_over_the_host_selection() -> Result<(), EngError> {
    if !device_present("cuda_qsa_attention_matches_the_host_over_the_host_selection") {
        return Ok(());
    }
    let mut rng = Rng(0x0a77_e5d0);
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let cache = random_cache(&ctx, FLASH_NEXT, &mut rng)?;
    let gpu = cache.gpu(&ctx, &stream);
    let q_norm = uniform(&mut rng, -0.5, 0.5, HEAD);
    // (position, budget blocks, splits)
    let cases = [
        (0, 16, 32),
        (2, 16, 32),
        (3, 16, 32),
        (66, 16, 32),
        (67, 16, 32),
        (600, 16, 32),
        (1099, 16, 32),
        (1099, 512, 32),
        (1098, 1, 32),
        (1097, 16, 1),
        (1097, 16, 7),
        (1097, 16, 128),
    ];
    for (pos, budget_blocks, splits) in cases {
        let geom = Geom {
            budget_blocks,
            splits,
            ..FLASH_NEXT
        };
        let t = token(geom, gpu.host_kv, &q_norm, pos, &mut rng)?;
        let dev = device_chunk(&gpu, geom, &[&t], QsaPosition::Host(pos))?;
        let label = format!("position {pos} budget {budget_blocks} splits {splits}");
        check(&label, &dev, &t.host);
    }
    Ok(())
}

/// One prefill chunk of `m` tokens from `p0`: against the host per token, bit-identical to one
/// call per token and to the chunk at a device position word.
fn check_chunk(
    gpu: &Gpu<'_>,
    geom: Geom,
    (p0, m): (usize, usize),
    rng: &mut Rng,
) -> Result<(), EngError> {
    let q_norm = uniform(rng, -0.5, 0.5, HEAD);
    let tokens: Vec<Token> = (p0..p0 + m)
        .map(|pos| token(geom, gpu.host_kv, &q_norm, pos, rng))
        .collect::<Result<_, _>>()?;
    let refs: Vec<&Token> = tokens.iter().collect();
    let chunk = device_chunk(gpu, geom, &refs, QsaPosition::Host(p0))?;
    let row = geom.n_head * HEAD;
    for (t, token) in tokens.iter().enumerate() {
        let dev = &chunk[t * row..(t + 1) * row];
        check(&format!("chunk {p0} + {m}, token {t}"), dev, &token.host);
        let alone = device_chunk(gpu, geom, &[token], QsaPosition::Host(p0 + t))?;
        assert_eq!(
            bits(dev),
            bits(&alone),
            "chunk {p0} + {m}, token {t}: differs from its own call"
        );
    }
    let word = gpu
        .ctx
        .upload(&i32::try_from(p0).expect("p0").to_le_bytes())?;
    let at_word = device_chunk(
        gpu,
        geom,
        &refs,
        QsaPosition::Device {
            word: whole(&word)?,
            admit: p0,
        },
    )?;
    assert_eq!(
        bits(&at_word),
        bits(&chunk),
        "chunk {p0} + {m}: the device word differs"
    );
    Ok(())
}

#[test]
fn cuda_qsa_attention_of_a_prefill_chunk_matches_the_host_per_token() -> Result<(), EngError> {
    if !device_present("cuda_qsa_attention_of_a_prefill_chunk_matches_the_host_per_token") {
        return Ok(());
    }
    let mut rng = Rng(0x0a77_c4a1);
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let cache = random_cache(&ctx, FLASH_NEXT, &mut rng)?;
    let gpu = cache.gpu(&ctx, &stream);
    check_chunk(&gpu, FLASH_NEXT, (600, 37), &mut rng)?;
    check_chunk(&gpu, FLASH_NEXT, (0, 70), &mut rng)
}

#[test]
fn cuda_qsa_attention_matches_the_host_at_the_kernel_limits() -> Result<(), EngError> {
    if !device_present("cuda_qsa_attention_matches_the_host_at_the_kernel_limits") {
        return Ok(());
    }
    let mut rng = Rng(0x0a77_11a1);
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    // (geometry, first position, tokens)
    let limits = [
        // A group of 32: 1024-thread split blocks.
        (
            Geom {
                n_head: 32,
                n_kv: 1,
                ratio: 3,
                rows: 700,
                budget_blocks: 40,
                splits: 16,
            },
            696,
            3,
        ),
        // 64 kv heads, a group of 1, blocks of 64.
        (
            Geom {
                n_head: 64,
                n_kv: 64,
                ratio: 64,
                rows: 773,
                budget_blocks: 4,
                splits: 5,
            },
            768,
            3,
        ),
        // Ratio 1 and a budget of 2^24 blocks: every one of the 300 blocks is chosen.
        (
            Geom {
                n_head: 4,
                n_kv: 2,
                ratio: 1,
                rows: 300,
                budget_blocks: 1 << 24,
                splits: 3,
            },
            298,
            2,
        ),
    ];
    for (geom, p0, m) in limits {
        let cache = random_cache(&ctx, geom, &mut rng)?;
        println!("REPORTED: flash-next qsa attention limit {geom:?}");
        check_chunk(&cache.gpu(&ctx, &stream), geom, (p0, m), &mut rng)?;
    }
    Ok(())
}

/// The cache rows a token at `pos` attends with `chosen` blocks, in the device's order.
fn attended_rows(geom: Geom, chosen: &[i32], pos: usize) -> Vec<usize> {
    let blocks = (pos + 1) / geom.ratio;
    let mut rows: Vec<usize> = chosen
        .iter()
        .flat_map(|&b| {
            let b = usize::try_from(b).expect("block");
            b * geom.ratio..(b + 1) * geom.ratio
        })
        .collect();
    rows.extend(blocks * geom.ratio..=pos);
    rows
}

#[test]
fn cuda_qsa_attention_reads_rows_past_2_pow_31_cache_values() -> Result<(), EngError> {
    if !device_present("cuda_qsa_attention_reads_rows_past_2_pow_31_cache_values") {
        return Ok(());
    }
    let geom = Geom {
        n_head: 64,
        n_kv: 64,
        ratio: 64,
        rows: 131_200,
        budget_blocks: 4,
        splits: 8,
    };
    // Position 131189: 2049 completed blocks, a tail of 54 rows; the last row starts at fp16
    // value 131189 * 64 * 256, past 2^31.
    let pos = 131_189;
    let chosen = [0, 1024, 2047, 2048];
    let rows = attended_rows(geom, &chosen, pos);
    assert!(rows.last().expect("rows") * geom.n_kv * HEAD > 1 << 31);
    let mut rng = Rng(0x0a77_b16a);
    let row_values = geom.n_kv * HEAD;
    let (bits_kv, host_kv) = half_values(&mut rng, rows.len() * row_values);
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    // One buffer serves as keys and values (both are only read), so the host's values equal its
    // keys too. Rows the token does not attend stay uninitialized.
    let cache = ctx.allocate(geom.rows * row_values * 2)?;
    for (i, &r) in rows.iter().enumerate() {
        let src = &bits_kv[i * row_values * 2..(i + 1) * row_values * 2];
        ctx.upload_at(&cache, r * row_values * 2, src)?;
    }
    // The host: a compact cache of the attended rows in the device's order, the token at the
    // last of them, every compact block selected; the rotation is at the compact position.
    let compact = Geom {
        rows: rows.len(),
        ..geom
    };
    let at = rows.len() - 1;
    let q_norm = uniform(&mut rng, -0.5, 0.5, HEAD);
    let qp = uniform(&mut rng, -2.0, 2.0, geom.ld());
    let at_u32 = u32::try_from(at).expect("position");
    let blocks = rows.len() / geom.ratio;
    let score: Vec<f64> = (0..blocks).map(|b| (blocks - b) as f64).collect();
    let selected = select_positions(compact.host_dims(), &score, at_u32)?;
    assert_eq!(selected, (0..rows.len()).collect::<Vec<_>>());
    let host = attend_selected(
        compact.host_dims(),
        &qp,
        &q_norm,
        (&host_kv, &host_kv),
        &selected,
        at_u32,
    )?;
    let t = Token {
        q: rotated_q(geom, &qp, &q_norm, at_u32)?,
        qp,
        chosen: chosen.to_vec(),
        host,
    };
    let gpu = Gpu {
        ctx: &ctx,
        stream: &stream,
        k: &cache,
        v: &cache,
        host_kv: (&host_kv, &host_kv),
    };
    let dev = device_chunk(&gpu, geom, &[&t], QsaPosition::Host(pos))?;
    check("rows past 2^31 cache values", &dev, &t.host);
    Ok(())
}
