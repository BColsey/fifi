//! Head dimension 512 on Q8 affine planes (Bolt's global layers: 8 query heads over 2 kv
//! heads, 32-value groups, signed codes, f16 scales, no bias, scale 1.0).
//!
//! Three references. The host reference reads the dequantized K and V in f64, rounds q to fp16
//! as the kernel does and computes the softmax in f64, on sampled rows. The decode reference
//! is `attn_decode_gemma` on the same device planes, every row, 16 tokens per launch with the
//! kv_len word of the launch. The refusal rows check that every rule the wide kernel adds is
//! a host `BadRequest` that changes no device bytes.
//!
//! The kernel rounds q, K, V and P to fp16 and accumulates in f32, so the bound is relative
//! to the largest value of V, not to f32 rounding: `TOLERANCE` below.

use super::super::{
    AttnKvLayout, AttnKvView, AttnScalarLayout, AttnStrides, cpf_attn_wide_split,
    cpf_attn_wide_window_split,
};
use super::*;
use crate::kernels::cuda::resident::attn_decode_gemma::AttnDecodeGemmaCall;
use crate::tensor::Dtype;

const WIDE: usize = 512;
const HEADS: usize = 8;
const KV_HEADS: usize = 2;
const GROUPS: usize = WIDE / 32;
/// Absolute bound of kernel against reference, in units of the largest |V| of the cache. It
/// was frozen from the first measurement of the grid below: the largest observed error is
/// 1.4e-3 against the f64 reference (sampled rows) and 3.7e-3 against decode (every row), both
/// at the peaked softmax; the flat softmax stays under 5e-4. The bound is twice the largest
/// observed value, rounded up.
const TOLERANCE: f64 = 8e-3;

struct Cache {
    cells: usize,
    /// Signed codes `[kv head][cell][512]`, as the planes `kv8_chunk` writes.
    codes: [Vec<u8>; 2],
    /// f16 scale bits `[kv head][cell][16]`.
    scales: [Vec<u16>; 2],
    /// The dequantized values, exact in f32 (an 11-bit scale times an 8-bit code).
    values: [Vec<f32>; 2],
    max_v: f64,
}

fn cache(cells: usize, seed: u64, k_sd: f64, v_sd: f64) -> Cache {
    let mut rng = Rng::new(seed);
    let cell_count = KV_HEADS * cells;
    let mut built = [(); 2].map(|()| {
        (
            vec![0u8; cell_count * WIDE],
            vec![0u16; cell_count * GROUPS],
            vec![0f32; cell_count * WIDE],
        )
    });
    for (plane, (codes, scales, values)) in built.iter_mut().enumerate() {
        let sd = if plane == 0 { k_sd } else { v_sd };
        for cell in 0..cell_count {
            for group in 0..GROUPS {
                let x: Vec<f32> = (0..32).map(|_| (rng.normal() * sd) as f32).collect();
                let amax = x.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
                let bits = f16_bits(amax / 127.0);
                let scale = f16_to_f32(bits);
                scales[cell * GROUPS + group] = bits;
                for (i, v) in x.iter().enumerate() {
                    let code = (v / scale).round().clamp(-127.0, 127.0) as i8;
                    let at = cell * WIDE + group * 32 + i;
                    codes[at] = code as u8;
                    values[at] = scale * f32::from(code);
                }
            }
        }
    }
    let [(kc, ks, kv), (vc, vs, vv)] = built;
    let max_v = vv.iter().fold(0f64, |a, v| a.max(f64::from(v.abs())));
    Cache {
        cells,
        codes: [kc, vc],
        scales: [ks, vs],
        values: [kv, vv],
        max_v,
    }
}

fn layout(cells: usize) -> AttnKvLayout {
    AttnKvLayout {
        storage: AttnKvStorage::Q8Affine {
            group_size: 32,
            signed: true,
        },
        payload_strides: AttnStrides {
            time_stride_bytes: WIDE,
            head_stride_bytes: cells * WIDE,
        },
        scales: Some(AttnScalarLayout {
            dtype: Dtype::F16,
            strides: AttnStrides {
                time_stride_bytes: GROUPS * 2,
                head_stride_bytes: cells * GROUPS * 2,
            },
            group_stride_bytes: 2,
        }),
        bias: None,
    }
}

fn le16s(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

/// The device side of a cache: code and scale buffers of both planes.
struct Planes {
    codes: [DeviceBuffer; 2],
    scales: [DeviceBuffer; 2],
    cells: usize,
}

fn upload(ctx: &CudaContext, cache: &Cache) -> Planes {
    Planes {
        codes: [0, 1].map(|p| testkit::upload_bytes(ctx, &cache.codes[p])),
        scales: [0, 1].map(|p| testkit::upload_bytes(ctx, &le16s(&cache.scales[p]))),
        cells: cache.cells,
    }
}

impl Planes {
    fn views(&self) -> Result<[AttnKvView<'_>; 2], EngError> {
        let layout = layout(self.cells);
        let extents = layout.extents(self.cells, KV_HEADS, WIDE)?;
        let view = |p: usize| -> Result<AttnKvView<'_>, EngError> {
            Ok(AttnKvView {
                layout,
                payload: self.codes[p].range(0, extents.payload_bytes)?,
                scales: Some(self.scales[p].range(0, extents.scale_bytes.unwrap_or(0))?),
                bias: None,
            })
        };
        Ok([view(0)?, view(1)?])
    }
}

fn q_values(m: usize, seed: u64, sd: f64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..m * HEADS * WIDE)
        .map(|_| (rng.normal() * sd) as f32)
        .collect()
}

fn shape(m: usize, cells: usize, p0: usize) -> AttnPrefillShape {
    AttnPrefillShape {
        m,
        n_head: HEADS,
        n_kv_head: KV_HEADS,
        head_dim: WIDE,
        cache_rows: cells,
        ld: 0,
        p0: p0 as i64,
    }
}

fn plan(shape: &AttnPrefillShape, window: usize) -> Result<CpfAttnSplit, EngError> {
    if window > 0 {
        cpf_attn_wide_window_split(window)
    } else {
        cpf_attn_wide_split(shape.cache_rows)
    }
}

/// Device buffers of one prefill call: q, part, out and the status word.
struct Io {
    q: DeviceBuffer,
    part: DeviceBuffer,
    out: DeviceBuffer,
    status: DeviceBuffer,
    part_bytes: usize,
    out_bytes: usize,
}

fn io(
    ctx: &CudaContext,
    shape: &AttnPrefillShape,
    split: CpfAttnSplit,
    q: &[f32],
) -> Result<Io, EngError> {
    let part_bytes = cpf_attn_part_bytes(
        CpfAttnGeometry {
            m: shape.m,
            n_head: shape.n_head,
            n_kv_head: shape.n_kv_head,
            head_dim: shape.head_dim,
        },
        CpfAttnPlan {
            q_tile: CPF_ATTN_Q_TILE,
            max_splits: split.max_splits,
            min_keys: split.min_keys,
        },
    )?;
    let out_bytes = shape.m * HEADS * WIDE * 4;
    Ok(Io {
        q: testkit::upload_bytes(ctx, &le(q)),
        part: testkit::upload_filled(ctx, PART_FILL, part_bytes),
        out: testkit::upload_filled(ctx, OUT_FILL, out_bytes),
        status: testkit::upload_bytes(ctx, &0i32.to_le_bytes()),
        part_bytes,
        out_bytes,
    })
}

fn call<'b>(
    shape: AttnPrefillShape,
    split: CpfAttnSplit,
    span: CpfAttnSpan,
    planes: &'b Planes,
    io: &'b Io,
) -> Result<AttnPrefillCall<'b>, EngError> {
    let views = planes.views()?;
    Ok(AttnPrefillCall {
        storage: views[0].layout.storage,
        shape,
        span,
        split,
        q_tile: CPF_ATTN_Q_TILE,
        scale: 1.0,
        q: io.q.range(0, io.out_bytes)?,
        proj: None,
        k_cache: views[0].payload,
        v_cache: views[1].payload,
        kv_views: Some(views),
        part: io.part.range(0, io.part_bytes)?,
        out: io.out.range(0, io.out_bytes)?,
        status: Some(io.status.range(0, 4)?),
        sinks: None,
    })
}

/// The f64 reference of one (token, head) row on the dequantized cache.
fn reference_row(
    q: &[f32],
    cache: &Cache,
    t: usize,
    h: usize,
    pos: usize,
    window: usize,
) -> Vec<f64> {
    let kv = h / (HEADS / KV_HEADS);
    let row = &q[(t * HEADS + h) * WIDE..][..WIDE];
    let rounded: Vec<f64> = row
        .iter()
        .map(|&x| f64::from(f16_to_f32(f16_bits(x))))
        .collect();
    let lo = if window > 0 {
        (pos + 1).saturating_sub(window)
    } else {
        0
    };
    let key =
        |plane: usize, j: usize| &cache.values[plane][(kv * cache.cells + j) * WIDE..][..WIDE];
    let scores: Vec<f64> = (lo..=pos)
        .map(|j| {
            key(0, j)
                .iter()
                .zip(&rounded)
                .map(|(&k, &q)| f64::from(k) * q)
                .sum()
        })
        .collect();
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut out = vec![0.0; WIDE];
    for (w, j) in weights.iter().zip(lo..=pos) {
        for (o, &v) in out.iter_mut().zip(key(1, j)) {
            *o += w * f64::from(v);
        }
    }
    out.iter().map(|o| o / total).collect()
}

struct Observation {
    prefill: Vec<f32>,
    decode: Vec<f32>,
    status: i32,
    guards: bool,
}

/// One call on the device, then the decode attention row by row on the same planes.
fn observe(
    cache: &Cache,
    q: &[f32],
    m: usize,
    p0: usize,
    window: usize,
    decode: bool,
) -> Result<Observation, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let planes = upload(&ctx, cache);
    let span = CpfAttnSpan {
        window: window as i64,
        ring: 0,
    };
    // The fixed-stride plan keeps one partial per split for every row of a tile, so a long
    // cache bounds the rows one call can take; a row does not depend on its chunk, so the
    // chunk runs as consecutive calls of the most rows the partial budget allows.
    let part_of = |rows: usize| -> Result<usize, EngError> {
        let probe = shape(rows, cache.cells, p0);
        let split = plan(&probe, window)?;
        cpf_attn_part_bytes(
            CpfAttnGeometry {
                m: rows,
                n_head: probe.n_head,
                n_kv_head: probe.n_kv_head,
                head_dim: probe.head_dim,
            },
            CpfAttnPlan {
                q_tile: CPF_ATTN_Q_TILE,
                max_splits: split.max_splits,
                min_keys: split.min_keys,
            },
        )
    };
    let per_tile = part_of(CPF_ATTN_Q_TILE)?;
    let step = (CPF_ATTN_PART_CAP / per_tile).max(1) * CPF_ATTN_Q_TILE;
    let row = HEADS * WIDE;
    let mut prefill = Vec::with_capacity(m * row);
    let mut status = 0;
    let mut guards = false;
    let mut start = 0;
    while start < m {
        let rows = step.min(m - start);
        let shape = shape(rows, cache.cells, p0 + start);
        let split = plan(&shape, window)?;
        let io = io(&ctx, &shape, split, &q[start * row..(start + rows) * row])?;
        ctx.attn_prefill(&stream, call(shape, split, span, &planes, &io)?)?;
        stream.synchronize()?;
        prefill.extend(testkit::floats(&testkit::read_bytes(
            &ctx,
            &io.out,
            io.out_bytes,
        )));
        let word = i32::from_le_bytes(
            testkit::read_bytes(&ctx, &io.status, 4)[..4]
                .try_into()
                .expect("the status word"),
        );
        status = if status == 0 { word } else { status };
        guards |= testkit::read_bytes(&ctx, &io.part, io.part_bytes)
            .iter()
            .any(|&b| b != PART_FILL);
        start += rows;
    }
    let mut rows = vec![0f32; if decode { m * HEADS * WIDE } else { 0 }];
    if decode {
        let launches = m.div_ceil(16);
        let words: Vec<u8> = (0..launches)
            .flat_map(|i| ((p0 + 16 * i + 1) as i32).to_le_bytes())
            .collect();
        let lens = testkit::upload_bytes(&ctx, &words);
        let dec_out = testkit::upload_filled(&ctx, OUT_FILL, m * row * 4);
        let q_all = testkit::upload_bytes(&ctx, &le(q));
        let codes_bytes = KV_HEADS * cache.cells * WIDE;
        let scale_bytes = KV_HEADS * cache.cells * GROUPS * 2;
        for i in 0..launches {
            let (t0, n) = (16 * i, 16.min(m - 16 * i));
            let row_bytes = HEADS * WIDE * 4;
            ctx.attn_decode_gemma(
                &stream,
                AttnDecodeGemmaCall {
                    out: dec_out.range(t0 * row_bytes, n * row_bytes)?,
                    status: None,
                    q: q_all.range(t0 * row_bytes, n * row_bytes)?,
                    k_codes: planes.codes[0].range(0, codes_bytes)?,
                    k_scales: planes.scales[0].range(0, scale_bytes)?,
                    v_codes: planes.codes[1].range(0, codes_bytes)?,
                    v_scales: planes.scales[1].range(0, scale_bytes)?,
                    kv_len: lens.range(4 * i, 4)?,
                    m: n,
                    n_q_head: HEADS,
                    n_kv_head: KV_HEADS,
                    head_dim: WIDE,
                    cells: cache.cells,
                    window,
                    scale: 1.0,
                    split: None,
                    ring: 0,
                },
            )?;
        }
        stream.synchronize()?;
        rows = testkit::floats(&testkit::read_bytes(&ctx, &dec_out, m * row * 4));
    }
    Ok(Observation {
        prefill,
        decode: rows,
        status,
        guards,
    })
}

fn worst(actual: &[f32], expected: impl Fn(usize) -> f64, count: usize) -> f64 {
    (0..count)
        .map(|i| (f64::from(actual[i]) - expected(i)).abs())
        .fold(0.0, f64::max)
}

/// Chunks of 1, 8, 64, 256 and 1,024 rows at p0 = 0, 511, 4,095 and 16,383, all against the
/// decode attention row by row, sampled rows against the f64 reference.
#[test]
fn d512_matches_the_f64_reference_and_the_decode_attention_row_by_row() {
    if !testkit::device_present("d512_matches_the_f64_reference_and_the_decode_attention") {
        return;
    }
    let cells = 16_383 + 1_024;
    let cache = cache(cells, 0xd512, 0.12, 1.0);
    let mut report = Vec::new();
    // The full grid at a flat softmax, then a peaked one (scores spread over about 11 nats).
    let mut grid = Vec::new();
    for p0 in [0usize, 511, 4_095, 16_383] {
        for m in [1usize, 8, 64, 256, 1_024] {
            grid.push((p0, m, 0.5));
        }
    }
    for p0 in [511usize, 4_095, 16_383] {
        for m in [8usize, 256] {
            grid.push((p0, m, 4.0));
        }
    }
    for (p0, m, q_sd) in grid {
        {
            let q = q_values(m, 0x51 + (p0 + m) as u64, q_sd);
            let seen = observe(&cache, &q, m, p0, 0, true).expect("the device phase");
            assert_eq!(seen.status, 0, "status at p0 {p0} m {m}");
            assert!(seen.guards, "the part buffer was written at p0 {p0} m {m}");
            assert!(
                seen.prefill.iter().all(|v| v.is_finite()),
                "non-finite output at p0 {p0} m {m}"
            );
            let versus_decode = worst(
                &seen.prefill,
                |i| f64::from(seen.decode[i]),
                seen.prefill.len(),
            );
            // Sampled rows: the first, a middle and the last token, a low and a high head.
            let mut versus_f64 = 0f64;
            for t in [0, m / 2, m - 1] {
                for h in [0, HEADS - 1] {
                    let want = reference_row(&q, &cache, t, h, p0 + t, 0);
                    let base = (t * HEADS + h) * WIDE;
                    versus_f64 =
                        versus_f64.max(worst(&seen.prefill[base..base + WIDE], |i| want[i], WIDE));
                }
            }
            println!(
                "D512 p0 {p0} m {m} q_sd {q_sd}: vs f64 {:.3e}, vs decode {:.3e} (units of max|V| {:.3})",
                versus_f64 / cache.max_v,
                versus_decode / cache.max_v,
                cache.max_v
            );
            report.push((p0, m, versus_f64 / cache.max_v, versus_decode / cache.max_v));
        }
    }
    for (p0, m, f64_error, decode_error) in report {
        assert!(
            f64_error <= TOLERANCE,
            "p0 {p0} m {m}: f64 error {f64_error:.3e}"
        );
        assert!(
            decode_error <= TOLERANCE,
            "p0 {p0} m {m}: decode error {decode_error:.3e}"
        );
    }
}

/// A window cuts each row to its own last keys; two runs of one call are byte identical.
#[test]
fn d512_window_rows_and_repeat_runs_are_exact() {
    if !testkit::device_present("d512_window_rows_and_repeat_runs_are_exact") {
        return;
    }
    let (m, p0, window) = (37usize, 700usize, 512usize);
    let cache = cache(p0 + m + 3, 0xd513, 0.12, 1.0);
    let q = q_values(m, 0x77, 0.5);
    let first = observe(&cache, &q, m, p0, window, true).expect("the device phase");
    let again = observe(&cache, &q, m, p0, window, false).expect("the device phase");
    assert_eq!(first.status, 0);
    assert_eq!(
        le(&first.prefill),
        le(&again.prefill),
        "the same call must give the same bytes"
    );
    let mut versus_f64 = 0f64;
    for t in [0, 7, 8, 20, m - 1] {
        for h in [1, 4, 6] {
            let want = reference_row(&q, &cache, t, h, p0 + t, window);
            let base = (t * HEADS + h) * WIDE;
            versus_f64 =
                versus_f64.max(worst(&first.prefill[base..base + WIDE], |i| want[i], WIDE));
        }
    }
    let versus_decode = worst(
        &first.prefill,
        |i| f64::from(first.decode[i]),
        first.prefill.len(),
    );
    println!(
        "D512 window {window}: vs f64 {:.3e}, vs decode {:.3e}",
        versus_f64 / cache.max_v,
        versus_decode / cache.max_v
    );
    assert!(versus_f64 / cache.max_v <= TOLERANCE);
    assert!(versus_decode / cache.max_v <= TOLERANCE);
}

/// Every rule the wide kernel adds is a host refusal that changes no device bytes.
#[test]
fn d512_refuses_what_the_wide_kernel_does_not_read() {
    if !testkit::device_present("d512_refuses_what_the_wide_kernel_does_not_read") {
        return;
    }
    let (m, p0) = (8usize, 64usize);
    let cache = cache(p0 + m, 0xd514, 0.12, 1.0);
    let q = q_values(m, 0x78, 0.5);
    let phase = || -> Result<Vec<String>, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let planes = upload(&ctx, &cache);
        let base = shape(m, cache.cells, p0);
        let split = plan(&base, 0)?;
        let io = io(&ctx, &base, split, &q)?;
        let before = [
            testkit::read_bytes(&ctx, &io.part, io.part_bytes),
            testkit::read_bytes(&ctx, &io.out, io.out_bytes),
        ];
        let mut failures = Vec::new();
        let mut expect_refusal = |label: &str, result: Result<(), EngError>| {
            if !matches!(result, Err(EngError::BadRequest(_))) {
                failures.push(format!("{label}: {result:?}"));
            }
        };
        let good = || call(base, split, CpfAttnSpan::FULL, &planes, &io);
        // Five query heads over two kv heads is no 4:1 group.
        let mut bad = good()?;
        bad.shape.n_head = 6;
        expect_refusal("group", ctx.attn_prefill(&stream, bad));
        // A bias plane.
        let mut bad = good()?;
        if let Some(views) = bad.kv_views.as_mut() {
            views[0].layout.bias = views[0].layout.scales;
        }
        expect_refusal(
            "bias layout without a range",
            ctx.attn_prefill(&stream, bad),
        );
        // 64-value groups.
        let mut bad = good()?;
        if let Some(views) = bad.kv_views.as_mut() {
            views[0].layout.storage = AttnKvStorage::Q8Affine {
                group_size: 64,
                signed: true,
            };
        }
        bad.storage = AttnKvStorage::Q8Affine {
            group_size: 64,
            signed: true,
        };
        expect_refusal("group size 64", ctx.attn_prefill(&stream, bad));
        // A payload row stride the 16-byte loads cannot address.
        let mut bad = good()?;
        if let Some(views) = bad.kv_views.as_mut() {
            views[1].layout.payload_strides.time_stride_bytes = WIDE + 8;
        }
        expect_refusal("unaligned row stride", ctx.attn_prefill(&stream, bad));
        // F16 storage has no wide kernel.
        let mut bad = good()?;
        bad.storage = AttnKvStorage::F16;
        bad.kv_views = None;
        expect_refusal("f16 storage", ctx.attn_prefill(&stream, bad));
        stream.synchronize()?;
        let after = [
            testkit::read_bytes(&ctx, &io.part, io.part_bytes),
            testkit::read_bytes(&ctx, &io.out, io.out_bytes),
        ];
        if before != after {
            failures.push("a refused call changed device bytes".to_owned());
        }
        Ok(failures)
    };
    let failures = phase().expect("the device phase");
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The timing report: per-call time of the wide kernel against the decode attention doing the
/// same rows (16 tokens per launch), with the bytes and flops each moves against the 3090's
/// peaks. Run by name: it holds the card for about a minute.
#[test]
#[ignore = "the D512 timing report: run by name in one GPU hold"]
fn d512_report_ms() {
    if !testkit::device_present("d512_report_ms") {
        return;
    }
    let cells = 32_768 + 1_024;
    let cache = cache(cells, 0xd515, 0.12, 1.0);
    let phase = || -> Result<Vec<String>, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let planes = upload(&ctx, &cache);
        let mut lines = Vec::new();
        for m in [256usize, 1_024] {
            for context in [4_096usize, 16_384, 32_768] {
                let p0 = context - m;
                let shape = shape(m, cells, p0);
                let split = plan(&shape, 0)?;
                let q = q_values(m, 0x99, 0.5);
                let io = io(&ctx, &shape, split, &q)?;
                let time = |launch: &dyn Fn() -> Result<(), EngError>,
                            reps: usize|
                 -> Result<f64, EngError> {
                    launch()?; // warm-up
                    stream.synchronize()?;
                    let (start, stop) = (ctx.event_timing()?, ctx.event_timing()?);
                    stream.record_timing(&start)?;
                    for _ in 0..reps {
                        launch()?;
                    }
                    stream.record_timing(&stop)?;
                    stream.synchronize()?;
                    Ok(f64::from(start.elapsed_ms_since(&stop)?) / reps as f64)
                };
                let wide = time(
                    &|| {
                        ctx.attn_prefill(
                            &stream,
                            call(shape, split, CpfAttnSpan::FULL, &planes, &io)?,
                        )
                    },
                    5,
                )?;
                let launches = m.div_ceil(16);
                let words: Vec<u8> = (0..launches)
                    .flat_map(|i| ((p0 + 16 * i + 1) as i32).to_le_bytes())
                    .collect();
                let lens = testkit::upload_bytes(&ctx, &words);
                let dec_out = testkit::upload_filled(&ctx, OUT_FILL, io.out_bytes);
                let codes_bytes = KV_HEADS * cells * WIDE;
                let scale_bytes = KV_HEADS * cells * GROUPS * 2;
                let row_bytes = HEADS * WIDE * 4;
                let decode = time(
                    &|| {
                        for i in 0..launches {
                            ctx.attn_decode_gemma(
                                &stream,
                                AttnDecodeGemmaCall {
                                    out: dec_out.range(16 * i * row_bytes, 16 * row_bytes)?,
                                    status: None,
                                    q: io.q.range(16 * i * row_bytes, 16 * row_bytes)?,
                                    k_codes: planes.codes[0].range(0, codes_bytes)?,
                                    k_scales: planes.scales[0].range(0, scale_bytes)?,
                                    v_codes: planes.codes[1].range(0, codes_bytes)?,
                                    v_scales: planes.scales[1].range(0, scale_bytes)?,
                                    kv_len: lens.range(4 * i, 4)?,
                                    m: 16,
                                    n_q_head: HEADS,
                                    n_kv_head: KV_HEADS,
                                    head_dim: WIDE,
                                    cells,
                                    window: 0,
                                    scale: 1.0,
                                    split: None,
                                    ring: 0,
                                },
                            )?;
                        }
                        Ok(())
                    },
                    1,
                )?;
                // 4 flops per (row, key, dim) for QK and PV over every query head.
                let keys = (p0 as f64) + (m as f64 + 1.0) / 2.0;
                let flops = 4.0 * WIDE as f64 * HEADS as f64 * keys * m as f64;
                // The planes are read once per kv head and tile if nothing is cached; report the
                // algorithmic minimum (each K and V byte once per chunk) as the traffic.
                let traffic =
                    (p0 + m) as f64 * KV_HEADS as f64 * 2.0 * (WIDE as f64 + GROUPS as f64 * 2.0);
                lines.push(format!(
                    "D512 m {m:>4} ctx {context:>5} splits {:>3}: wide {wide:8.3} ms ({:5.2} TFLOP/s = {:4.1}% of 71, {:6.1} GB/s min traffic = {:4.1}% of 936) | decode x{launches} {decode:9.3} ms | speedup {:6.1}x",
                    split.max_splits,
                    flops / (wide * 1e-3) / 1e12,
                    100.0 * flops / (wide * 1e-3) / 1e12 / 71.0,
                    traffic / (wide * 1e-3) / 1e9,
                    100.0 * traffic / (wide * 1e-3) / 1e9 / 936.0,
                    decode / wide,
                ));
            }
        }
        Ok(lines)
    };
    for line in phase().expect("the device phase") {
        println!("{line}");
    }
}

// ---- structured values: a lost or repeated key changes the output visibly ------------------

/// The scale of every structured plane: a power of two, exact in f16, bf16 and f32.
const STRUCT_SCALE: f32 = 0.0625;

/// The V code at (position, kv head, dim): a small integer that differs for every position,
/// head and dimension pattern, so any key a call drops, repeats or reads from the wrong row
/// moves the mean of its row.
fn v_code(pos: usize, head: usize, dim: usize, signed: bool) -> i32 {
    let c = ((pos * 7 + dim * 3 + head * 11) % 31) as i32;
    if signed { c - 15 } else { c }
}

fn v_value(pos: usize, head: usize, dim: usize, signed: bool) -> f64 {
    f64::from(STRUCT_SCALE) * f64::from(v_code(pos, head, dim, signed))
}

/// The device planes of a structured cache: K codes all 3, V codes by `v_code` of the position
/// each row holds (`None`: bytes no row may read), power-of-two scales in `dtype`.
struct StructPlanes {
    codes: [DeviceBuffer; 2],
    scales: [DeviceBuffer; 2],
    layout: AttnKvLayout,
    cells: usize,
    origin: usize,
}

fn scalar_bytes(dtype: Dtype) -> Vec<u8> {
    match dtype {
        Dtype::F16 => f16_bits(STRUCT_SCALE).to_le_bytes().to_vec(),
        Dtype::Bf16 => ((STRUCT_SCALE.to_bits() >> 16) as u16)
            .to_le_bytes()
            .to_vec(),
        _ => STRUCT_SCALE.to_le_bytes().to_vec(),
    }
}

fn struct_planes(
    ctx: &CudaContext,
    cells: usize,
    position_of_row: &dyn Fn(usize) -> Option<usize>,
    signed: bool,
    dtype: Dtype,
    origin: usize,
) -> StructPlanes {
    let width = scalar_bytes(dtype).len();
    let layout = AttnKvLayout {
        storage: AttnKvStorage::Q8Affine {
            group_size: 32,
            signed,
        },
        payload_strides: AttnStrides {
            time_stride_bytes: WIDE,
            head_stride_bytes: cells * WIDE,
        },
        scales: Some(AttnScalarLayout {
            dtype,
            strides: AttnStrides {
                time_stride_bytes: GROUPS * width,
                head_stride_bytes: cells * GROUPS * width,
            },
            group_stride_bytes: width,
        }),
        bias: None,
    };
    let mut k = vec![0u8; origin];
    let mut v = vec![0u8; origin];
    for head in 0..KV_HEADS {
        for row in 0..cells {
            for dim in 0..WIDE {
                k.push(3);
                v.push(match position_of_row(row) {
                    Some(pos) => v_code(pos, head, dim, signed) as i8 as u8,
                    None => 0x55,
                });
            }
        }
    }
    let scale = scalar_bytes(dtype);
    let scales: Vec<u8> = (0..KV_HEADS * cells * GROUPS)
        .flat_map(|_| scale.iter().copied())
        .collect();
    let pad = |mut bytes: Vec<u8>| {
        bytes.resize(bytes.len() + 64, 0);
        bytes
    };
    StructPlanes {
        codes: [
            testkit::upload_bytes(ctx, &pad(k)),
            testkit::upload_bytes(ctx, &pad(v)),
        ],
        scales: [
            testkit::upload_bytes(ctx, &scales),
            testkit::upload_bytes(ctx, &scales),
        ],
        layout,
        cells,
        origin,
    }
}

impl StructPlanes {
    fn views(&self) -> Result<[AttnKvView<'_>; 2], EngError> {
        let extents = self.layout.extents(self.cells, KV_HEADS, WIDE)?;
        let view = |p: usize| -> Result<AttnKvView<'_>, EngError> {
            Ok(AttnKvView {
                layout: self.layout,
                payload: self.codes[p].range(self.origin, extents.payload_bytes)?,
                scales: Some(self.scales[p].range(0, extents.scale_bytes.unwrap_or(0))?),
                bias: None,
            })
        };
        Ok([view(0)?, view(1)?])
    }

    fn call<'b>(
        &'b self,
        shape: AttnPrefillShape,
        split: CpfAttnSplit,
        span: CpfAttnSpan,
        io: &'b Io,
    ) -> Result<AttnPrefillCall<'b>, EngError> {
        let views = self.views()?;
        Ok(AttnPrefillCall {
            storage: self.layout.storage,
            shape,
            span,
            split,
            q_tile: CPF_ATTN_Q_TILE,
            scale: 1.0,
            q: io.q.range(0, io.out_bytes)?,
            proj: None,
            k_cache: views[0].payload,
            v_cache: views[1].payload,
            kv_views: Some(views),
            part: io.part.range(0, io.part_bytes)?,
            out: io.out.range(0, io.out_bytes)?,
            status: Some(io.status.range(0, 4)?),
            sinks: None,
        })
    }
}

#[derive(Clone, Copy)]
struct StructCase {
    p0: usize,
    m: usize,
    window: usize,
    ring: usize,
    signed: bool,
    dtype: Dtype,
    /// `Some(cut)`: the keys below `cut` and from `cut` on are staged as two segments.
    cut: Option<usize>,
}

impl StructCase {
    fn new(p0: usize, m: usize) -> Self {
        Self {
            p0,
            m,
            window: 0,
            ring: 0,
            signed: true,
            dtype: Dtype::F16,
            cut: None,
        }
    }
}

/// The rows a ring cache of `ring` rows holds after a chunk at `p0`: the newest position of
/// the union of the chunk's windows in each row.
fn ring_position(row: usize, ring: usize, p0: usize, m: usize, window: usize) -> Option<usize> {
    let last = p0 + m - 1;
    let low = (p0 + 1).saturating_sub(window);
    let pos = last.checked_sub((last + ring - row % ring) % ring)?;
    (pos >= low && pos % ring == row).then_some(pos)
}

/// Runs one structured case with q = 0, so every score is 0 and the output of a row is the exact
/// mean of the V rows it sees: returns the output, the status word and the split used.
fn run_structured(case: StructCase) -> Result<(Vec<f32>, i32, CpfAttnSplit), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let end = case.p0 + case.m;
    let span = CpfAttnSpan {
        window: case.window as i64,
        ring: case.ring as i64,
    };
    let q = vec![0f32; case.m * HEADS * WIDE];
    let (status, out, split) = if let Some(cut) = case.cut {
        let low = struct_planes(&ctx, cut, &|row| Some(row), case.signed, case.dtype, 0);
        let high = struct_planes(
            &ctx,
            end - cut,
            &|row| Some(cut + row),
            case.signed,
            case.dtype,
            0,
        );
        let split = CpfAttnSplit {
            max_splits: 2,
            min_keys: 256,
        };
        let shape_low = shape(case.m, cut, case.p0);
        let shape_high = shape(case.m, end - cut, case.p0);
        let io = io(&ctx, &shape_low, split, &q)?;
        for (planes, shape, origin, rows, index) in [
            (&low, shape_low, 0, cut, 0),
            (&high, shape_high, cut, end - cut, 1),
        ] {
            ctx.attn_prefill_part(
                &stream,
                planes.call(shape, split, span, &io)?,
                AttnPrefillSegment {
                    key_origin: origin,
                    key_rows: rows,
                    part_index: index,
                },
            )?;
        }
        ctx.attn_prefill_combine(&stream, low.call(shape_low, split, span, &io)?)?;
        stream.synchronize()?;
        read_struct(&ctx, &io, split)
    } else {
        let rows = if case.ring > 0 { case.ring } else { end };
        let planes = if case.ring > 0 {
            struct_planes(
                &ctx,
                rows,
                &|row| ring_position(row, case.ring, case.p0, case.m, case.window),
                case.signed,
                case.dtype,
                0,
            )
        } else {
            struct_planes(&ctx, rows, &|row| Some(row), case.signed, case.dtype, 0)
        };
        let shape = shape(case.m, rows, case.p0);
        let split = plan(&shape, case.window)?;
        let io = io(&ctx, &shape, split, &q)?;
        ctx.attn_prefill(&stream, planes.call(shape, split, span, &io)?)?;
        stream.synchronize()?;
        read_struct(&ctx, &io, split)
    };
    Ok((out, status, split))
}

fn read_struct(ctx: &CudaContext, io: &Io, split: CpfAttnSplit) -> (i32, Vec<f32>, CpfAttnSplit) {
    let status = i32::from_le_bytes(
        testkit::read_bytes(ctx, &io.status, 4)[..4]
            .try_into()
            .expect("the status word"),
    );
    (
        status,
        testkit::floats(&testkit::read_bytes(ctx, &io.out, io.out_bytes)),
        split,
    )
}

/// Worst per-row error of a structured run, as a fraction of the row's own output magnitude.
/// `floor` drops the keys below that position from the reference: 0 for the real one, above 0
/// to show that the bound notices a run that lost its first keys.
fn structured_error(case: StructCase, out: &[f32], floor: usize) -> f64 {
    let end = case.p0 + case.m;
    // Prefix sums over absolute positions, per kv head and dimension.
    let mut prefix = vec![vec![0f64; (end + 1) * WIDE]; KV_HEADS];
    for (head, sums) in prefix.iter_mut().enumerate() {
        for pos in 0..end {
            for dim in 0..WIDE {
                sums[(pos + 1) * WIDE + dim] =
                    sums[pos * WIDE + dim] + v_value(pos, head, dim, case.signed);
            }
        }
    }
    let mut worst = 0f64;
    for t in 0..case.m {
        let pos = case.p0 + t;
        let low = if case.window > 0 {
            (pos + 1).saturating_sub(case.window)
        } else {
            0
        }
        .max(floor)
        .min(pos);
        let count = (pos + 1 - low) as f64;
        for h in 0..HEADS {
            let head = h / (HEADS / KV_HEADS);
            let want: Vec<f64> = (0..WIDE)
                .map(|d| {
                    (prefix[head][(pos + 1) * WIDE + d] - prefix[head][low * WIDE + d]) / count
                })
                .collect();
            let magnitude = want.iter().fold(1e-3f64, |a, v| a.max(v.abs()));
            let base = (t * HEADS + h) * WIDE;
            let error = (0..WIDE)
                .map(|d| (f64::from(out[base + d]) - want[d]).abs())
                .fold(0.0, f64::max);
            worst = worst.max(error / magnitude);
        }
    }
    worst
}

/// With q = 0 the kernel's arithmetic is exact (equal weights, V exact in f16, sums exact in
/// f32), so the bound is about ten times the largest error of the first measurement (5.8e-8),
/// relative to the row's own magnitude: a key or a tile dropped, repeated or read from the wrong
/// row moves a row by far more.
const STRUCTURED_BOUND: f64 = 1e-6;

#[test]
fn d512_structured_values_expose_a_lost_repeated_or_misplaced_key() {
    if !testkit::device_present("d512_structured_values") {
        return;
    }
    let mut cases = vec![
        StructCase::new(0, 1),
        StructCase::new(0, 5),
        // kv_len 4,127 is not a multiple of 16 and 37 rows are not a multiple of 8.
        StructCase::new(4_090, 37),
        StructCase::new(1_000, 256),
        StructCase {
            window: 512,
            ..StructCase::new(2_000, 100)
        },
        // Unsigned codes and the other scale dtypes.
        StructCase {
            signed: false,
            dtype: Dtype::Bf16,
            ..StructCase::new(700, 33)
        },
        StructCase {
            signed: true,
            dtype: Dtype::F32,
            ..StructCase::new(700, 33)
        },
        StructCase {
            signed: false,
            dtype: Dtype::F32,
            ..StructCase::new(517, 9)
        },
        // A ring cache: 1,024 rows, window 512, the union of the chunk's windows staged.
        StructCase {
            window: 512,
            ring: 1_024,
            ..StructCase::new(4_090, 37)
        },
        // The part and combine entries over two segments.
        StructCase {
            cut: Some(2_000),
            ..StructCase::new(2_990, 37)
        },
        StructCase {
            cut: Some(33),
            signed: false,
            dtype: Dtype::Bf16,
            ..StructCase::new(90, 40)
        },
    ];
    cases.push(StructCase::new(300, 64));
    let (mut split_cases, mut tail_cases) = (0, 0);
    for case in cases {
        let (out, status, split) = run_structured(case).expect("the device phase");
        assert_eq!(status, 0, "status at p0 {} m {}", case.p0, case.m);
        assert!(out.iter().all(|v| v.is_finite()));
        let error = structured_error(case, &out, 0);
        println!(
            "D512 structured p0 {} m {} window {} ring {} signed {} {:?} cut {:?} splits {}: worst row error {error:.3e}",
            case.p0,
            case.m,
            case.window,
            case.ring,
            case.signed,
            case.dtype,
            case.cut,
            split.max_splits
        );
        assert!(
            error <= STRUCTURED_BOUND,
            "p0 {} m {} window {} ring {} cut {:?}: {error:e}",
            case.p0,
            case.m,
            case.window,
            case.ring,
            case.cut
        );
        if case.p0 == 4_090 && case.cut.is_none() && case.ring == 0 {
            // The bound has teeth: a reference that lacks the first 16 keys (one lost tile) is
            // far outside it.
            let lost = structured_error(case, &out, 16);
            assert!(
                lost > 100.0 * STRUCTURED_BOUND,
                "a lost tile moves a row by only {lost:e}"
            );
        }
        split_cases += usize::from(split.max_splits > 1 && case.cut.is_none());
        tail_cases += usize::from((case.p0 + case.m) % 16 != 0 && case.m % 8 != 0);
    }
    assert!(
        split_cases >= 3,
        "cases with more than one split: {split_cases}"
    );
    assert!(
        tail_cases >= 3,
        "cases with a partial last tile and chunk: {tail_cases}"
    );
}

/// Refusals the structured planes reach: a mixed F16 V plane, a V plane with 64-value groups, a
/// payload base off the 16-byte grid. Every refused call leaves the status word and the
/// buffers it names as they were.
#[test]
fn d512_refuses_mixed_wide_group_and_misaligned_planes_with_status_untouched() {
    if !testkit::device_present("d512_refuses_mixed_wide_group_and_misaligned_planes") {
        return;
    }
    let (m, p0) = (8usize, 56usize);
    let phase = || -> Result<Vec<String>, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let cells = p0 + m;
        let good = struct_planes(&ctx, cells, &|row| Some(row), true, Dtype::F16, 0);
        let shifted = struct_planes(&ctx, cells, &|row| Some(row), true, Dtype::F16, 8);
        let f16_rows = testkit::upload_filled(&ctx, 0, KV_HEADS * cells * WIDE * 2 + 64);
        let base = shape(m, cells, p0);
        let split = plan(&base, 0)?;
        let io = io(&ctx, &base, split, &vec![0f32; m * HEADS * WIDE])?;
        let snapshot = |ctx: &CudaContext| {
            [
                testkit::read_bytes(ctx, &io.part, io.part_bytes),
                testkit::read_bytes(ctx, &io.out, io.out_bytes),
                testkit::read_bytes(ctx, &io.status, 4),
            ]
        };
        let before = snapshot(&ctx);
        let mut failures = Vec::new();
        let mut refused = |label: &str, result: Result<(), EngError>| {
            if !matches!(result, Err(EngError::BadRequest(_))) {
                failures.push(format!("{label}: {result:?}"));
            }
        };
        // The mixed profile: K Q8, V F16.
        let mut bad = good.call(base, split, CpfAttnSpan::FULL, &io)?;
        if let Some(views) = bad.kv_views.as_mut() {
            views[1] = AttnKvView {
                layout: AttnKvLayout {
                    storage: AttnKvStorage::F16,
                    payload_strides: AttnStrides {
                        time_stride_bytes: WIDE * 2,
                        head_stride_bytes: cells * WIDE * 2,
                    },
                    scales: None,
                    bias: None,
                },
                payload: f16_rows.range(0, KV_HEADS * cells * WIDE * 2)?,
                scales: None,
                bias: None,
            };
            bad.v_cache = views[1].payload;
        }
        refused("mixed f16 V plane", ctx.attn_prefill(&stream, bad));
        // A V plane with 64-value groups beside a K plane with 32.
        let mut bad = good.call(base, split, CpfAttnSpan::FULL, &io)?;
        if let Some(views) = bad.kv_views.as_mut() {
            views[1].layout.storage = AttnKvStorage::Q8Affine {
                group_size: 64,
                signed: true,
            };
        }
        refused(
            "V plane with 64-value groups",
            ctx.attn_prefill(&stream, bad),
        );
        // A payload base at byte 8 of its buffer.
        refused(
            "payload base off the 16-byte grid",
            ctx.attn_prefill(&stream, shifted.call(base, split, CpfAttnSpan::FULL, &io)?),
        );
        stream.synchronize()?;
        if before != snapshot(&ctx) {
            failures.push("a refused call changed the part, out or status bytes".to_owned());
        }
        // The same planes still run, with a zero status word.
        ctx.attn_prefill(&stream, good.call(base, split, CpfAttnSpan::FULL, &io)?)?;
        stream.synchronize()?;
        if snapshot(&ctx)[2] != 0i32.to_le_bytes() {
            failures.push("a good call set the status word".to_owned());
        }
        Ok(failures)
    };
    let failures = phase().expect("the device phase");
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The device gate: the card the tests run on provides what the wide kernel needs.
#[test]
fn d512_device_gate_admits_the_test_card() {
    if !testkit::device_present("d512_device_gate_admits_the_test_card") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    ctx.check_wide_device()
        .expect("sm_80 PTX and an 80 KB opt-in on the test card");
}

/// Sixteen query heads over one kv head (the shape of a Gemma global layer) run as four virtual
/// kv heads of four query heads whose views have head stride 0, so every virtual head reads the
/// one real head. With q = 0 each row is the exact mean of that head's V rows.
#[test]
fn d512_serves_sixteen_heads_over_one_kv_head_through_a_zero_stride_view() {
    if !testkit::device_present("d512_serves_sixteen_heads_over_one_kv_head") {
        return;
    }
    let (m, p0, heads, virtual_kv) = (13usize, 300usize, 16usize, 4usize);
    let phase = || -> Result<Vec<f32>, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let cells = p0 + m;
        let planes = struct_planes(&ctx, cells, &|row| Some(row), true, Dtype::F16, 0);
        let mut layout = planes.layout;
        layout.payload_strides.head_stride_bytes = 0;
        if let Some(scales) = layout.scales.as_mut() {
            scales.strides.head_stride_bytes = 0;
        }
        let extents = layout.extents(cells, virtual_kv, WIDE)?;
        let view = |p: usize| -> Result<AttnKvView<'_>, EngError> {
            Ok(AttnKvView {
                layout,
                payload: planes.codes[p].range(0, extents.payload_bytes)?,
                scales: Some(planes.scales[p].range(0, extents.scale_bytes.unwrap_or(0))?),
                bias: None,
            })
        };
        let views = [view(0)?, view(1)?];
        let shape = AttnPrefillShape {
            m,
            n_head: heads,
            n_kv_head: virtual_kv,
            head_dim: WIDE,
            cache_rows: cells,
            ld: 0,
            p0: p0 as i64,
        };
        let geometry = CpfAttnGeometry {
            m,
            n_head: heads,
            n_kv_head: virtual_kv,
            head_dim: WIDE,
        };
        let plan = cpf_attn_wide_split(cells)?;
        let part_bytes = cpf_attn_part_bytes(
            geometry,
            CpfAttnPlan {
                q_tile: CPF_ATTN_Q_TILE,
                max_splits: plan.max_splits,
                min_keys: plan.min_keys,
            },
        )?;
        let out_bytes = m * heads * WIDE * 4;
        let q = testkit::upload_bytes(&ctx, &vec![0u8; out_bytes]);
        let part = testkit::upload_filled(&ctx, PART_FILL, part_bytes);
        let out = testkit::upload_filled(&ctx, OUT_FILL, out_bytes);
        ctx.attn_prefill(
            &stream,
            AttnPrefillCall {
                storage: layout.storage,
                shape,
                span: CpfAttnSpan::FULL,
                split: CpfAttnSplit {
                    max_splits: plan.max_splits,
                    min_keys: plan.min_keys,
                },
                q_tile: CPF_ATTN_Q_TILE,
                scale: 1.0,
                q: q.range(0, out_bytes)?,
                proj: None,
                k_cache: views[0].payload,
                v_cache: views[1].payload,
                kv_views: Some(views),
                part: part.range(0, part_bytes)?,
                out: out.range(0, out_bytes)?,
                status: None,
                sinks: None,
            },
        )?;
        stream.synchronize()?;
        Ok(testkit::floats(&testkit::read_bytes(&ctx, &out, out_bytes)))
    };
    let out = phase().expect("the device phase");
    let mut worst = 0f64;
    for t in 0..m {
        let pos = p0 + t;
        let want: Vec<f64> = (0..WIDE)
            .map(|d| (0..=pos).map(|j| v_value(j, 0, d, true)).sum::<f64>() / (pos + 1) as f64)
            .collect();
        let magnitude = want.iter().fold(1e-3f64, |a, v| a.max(v.abs()));
        for h in 0..heads {
            let base = (t * heads + h) * WIDE;
            let error = (0..WIDE)
                .map(|d| (f64::from(out[base + d]) - want[d]).abs())
                .fold(0.0, f64::max);
            worst = worst.max(error / magnitude);
        }
    }
    println!("D512 16 heads over 1 kv head: worst row error {worst:.3e}");
    assert!(worst <= STRUCTURED_BOUND, "{worst:e}");
}

/// Runs `parts` (start position, rows) of one prompt in order over one cache and returns the
/// output rows of the parts, concatenated. `q` holds the rows of positions `q_origin ..` of the
/// prompt.
fn run_parts(
    cache: &Cache,
    q: &[f32],
    q_origin: usize,
    parts: &[(usize, usize)],
    window: usize,
) -> Result<Vec<f32>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let planes = upload(&ctx, cache);
    let row = HEADS * WIDE;
    let span = CpfAttnSpan {
        window: window as i64,
        ring: 0,
    };
    let mut out = Vec::new();
    for &(p0, m) in parts {
        let shape = shape(m, cache.cells, p0);
        let split = plan(&shape, window)?;
        let at = (p0 - q_origin) * row;
        let io = io(&ctx, &shape, split, &q[at..at + m * row])?;
        ctx.attn_prefill(&stream, call(shape, split, span, &planes, &io)?)?;
        stream.synchronize()?;
        out.extend(testkit::floats(&testkit::read_bytes(
            &ctx,
            &io.out,
            io.out_bytes,
        )));
    }
    Ok(out)
}

/// The same rows cut five ways must give the same bytes as chunks of 8: `rows` rows from position
/// `base` over a cache of `cells`, as chunks of 5, of `big` (the most rows one call takes under
/// the 64 MB partial budget), of 1, and resumed at odd positions; `only` checks one of those four
/// layouts, so a slow tool can take them in separate runs.
fn assert_rows_independent_of_chunking(
    cells: usize,
    base: usize,
    rows: usize,
    window: usize,
    big: usize,
    only: Option<usize>,
) {
    let cache = cache(cells, 0xd516 + window as u64, 0.12, 1.0);
    let q = q_values(rows, 0x5eed, 1.5);
    let cut = |size: usize| -> Vec<(usize, usize)> {
        (0..rows)
            .step_by(size)
            .map(|at| (base + at, size.min(rows - at)))
            .collect()
    };
    let reference = run_parts(&cache, &q, base, &cut(8), window).expect("chunks of 8");
    let layouts = [
        ("chunks of 5", cut(5)),
        ("the largest chunk the part budget takes", cut(big)),
        ("chunks of 1", cut(1)),
        ("resumed at odd positions", {
            let mut parts = Vec::new();
            let mut at = 0;
            for size in [13, 11, 7, 3, 15].into_iter().cycle() {
                if at == rows {
                    break;
                }
                let size = size.min(big).min(rows - at);
                parts.push((base + at, size));
                at += size;
            }
            parts
        }),
    ];
    let mut failures = Vec::new();
    for (index, (label, parts)) in layouts.into_iter().enumerate() {
        if only.is_some_and(|only| only != index) {
            continue;
        }
        let got = run_parts(&cache, &q, base, &parts, window).expect(label);
        if le(&got) != le(&reference) {
            let first = got
                .iter()
                .zip(&reference)
                .position(|(a, b)| a.to_bits() != b.to_bits())
                .unwrap_or(0);
            failures.push(format!(
                "{label}: first differing value at row {} head {}",
                first / (HEADS * WIDE),
                first % (HEADS * WIDE) / WIDE
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "cells {cells} base {base} window {window}: {failures:#?}"
    );
}

/// A row's output is a function of its position and the cache alone: the same bytes whether the
/// prompt runs as chunks of 8, 5, 64 or 1 rows, or resumed at an odd position, so a prompt served
/// from a restored prefix, in pieces of any size, or alone gives the same cache and logits.
#[test]
fn d512_rows_do_not_depend_on_how_the_prompt_is_chunked() {
    if !testkit::device_present("d512_rows_do_not_depend_on_how_the_prompt_is_chunked") {
        return;
    }
    assert_rows_independent_of_chunking(700, 0, 643, 0, 64, None);
}

/// The same at the depth of the 32K cache, where every one of the 128 splits holds keys. One test
/// per layout, so a sanitizer can take them in separate runs.
fn deep_layout(only: usize, name: &str) {
    if !testkit::device_present(name) {
        return;
    }
    assert_rows_independent_of_chunking(32_768, 32_600, 128, 0, 16, Some(only));
}

#[test]
fn d512_rows_do_not_depend_on_chunking_at_32k_cells_chunks_of_5() {
    deep_layout(0, "d512_deep_chunks_of_5");
}

#[test]
fn d512_rows_do_not_depend_on_chunking_at_32k_cells_the_largest_chunk() {
    deep_layout(1, "d512_deep_largest_chunk");
}

#[test]
fn d512_rows_do_not_depend_on_chunking_at_32k_cells_chunks_of_1() {
    deep_layout(2, "d512_deep_chunks_of_1");
}

#[test]
fn d512_rows_do_not_depend_on_chunking_at_32k_cells_odd_resumes() {
    deep_layout(3, "d512_deep_odd_resumes");
}

/// A sliding window moves the first visible key with the tile; the splits stay on the absolute
/// stride, so the windowed rows do not depend on the chunking either.
#[test]
fn d512_windowed_rows_do_not_depend_on_how_the_prompt_is_chunked() {
    if !testkit::device_present("d512_windowed_rows_do_not_depend_on_how_the_prompt_is_chunked") {
        return;
    }
    assert_rows_independent_of_chunking(1_500, 0, 1_100, 512, 64, None);
    assert_rows_independent_of_chunking(4_096, 3_000, 700, 300, 64, None);
}

/// Runs one 8-row call at `p0` over a cache of `cells` with `split`, over a cache whose rows are
/// all valid, and returns the call's result with the output and status bytes as they ended.
fn call_with_plan(
    cells: usize,
    p0: usize,
    rows: usize,
    window: usize,
    split: CpfAttnSplit,
) -> (Result<(), EngError>, Vec<u8>, i32) {
    let cache = cache(cells, 0xd517, 0.12, 1.0);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let planes = upload(&ctx, &cache);
    let shape = shape(rows, cells, p0);
    let q = q_values(rows, 0x5eed, 1.5);
    let io = io(&ctx, &shape, split, &q).expect("buffers");
    let span = CpfAttnSpan {
        window: window as i64,
        ring: 0,
    };
    let result =
        call(shape, split, span, &planes, &io).and_then(|call| ctx.attn_prefill(&stream, call));
    if result.is_err() && super::super::wide_split_check_skipped() {
        // A device-side refusal latches the context: nothing can be read back, and nothing was
        // written because the launch never ran.
        return (result, Vec::new(), 0);
    }
    stream.synchronize().expect("sync");
    let out = testkit::read_bytes(&ctx, &io.out, io.out_bytes);
    let status = i32::from_le_bytes(
        testkit::read_bytes(&ctx, &io.status, 4)[..4]
            .try_into()
            .unwrap(),
    );
    (result, out, status)
}

/// One plan that cannot reach the last visible key: refused by the host check before any device
/// write, and, with the host check skipped, by the device-side check of the same rule. The device
/// refusal latches the backend of the thread, so each case runs in its own test.
fn assert_unreachable_plan_refused(cells: usize, p0: usize, window: usize, split: CpfAttnSplit) {
    let (result, out, status) = call_with_plan(cells, p0, 8, window, split);
    let error = result.expect_err("the host check refuses").describe();
    assert!(error.contains("wide split"), "{error}");
    assert!(out.iter().all(|&b| b == OUT_FILL), "out untouched");
    assert_eq!(status, 0, "status untouched");
    super::super::SKIP_WIDE_SPLIT_CHECK.with(|skip| skip.set(true));
    let (result, out, status) = call_with_plan(cells, p0, 8, window, split);
    super::super::SKIP_WIDE_SPLIT_CHECK.with(|skip| skip.set(false));
    let error = result.expect_err("the device check refuses").describe();
    assert!(
        !error.contains("wide split"),
        "the device check refuses, not the host check: {error}"
    );
    assert!(out.is_empty() && status == 0, "refused on the device");
}

/// A linear cache of 4,096 cells planned for 2,048.
#[test]
fn d512_refuses_a_linear_plan_that_cannot_reach_the_last_key() {
    if !testkit::device_present("d512_refuses_a_linear_plan_that_cannot_reach_the_last_key") {
        return;
    }
    let split = cpf_attn_wide_split(2_048).expect("plan");
    assert_unreachable_plan_refused(4_096, 3_000, 0, split);
}

/// A 512 window planned without the stride the first visible key is rounded down by.
#[test]
fn d512_refuses_a_window_plan_without_the_rounding_stride() {
    if !testkit::device_present("d512_refuses_a_window_plan_without_the_rounding_stride") {
        return;
    }
    let split = cpf_attn_wide_split(512 + CPF_ATTN_TILE_ROWS).expect("plan");
    assert_unreachable_plan_refused(2_048, 1_500, 512, split);
}

/// A chunk whose partials do not fit the 64 MB budget at the stride plan of a deep cache is a
/// memory-plan refusal that names the cap, and writes nothing.
#[test]
fn d512_refuses_a_chunk_whose_partials_exceed_the_part_cap() {
    if !testkit::device_present("d512_refuses_a_chunk_whose_partials_exceed_the_part_cap") {
        return;
    }
    let cells = 17_407;
    let split = cpf_attn_wide_split(cells).expect("plan");
    let (result, out, status) = call_with_plan(cells, 100, 64, 0, split);
    let error = result.expect_err("over the cap");
    assert!(
        matches!(&error, EngError::MemoryPlan(text) if text.contains("part cap")),
        "{error:?}"
    );
    assert!(out.iter().all(|&b| b == OUT_FILL));
    assert_eq!(status, 0);
}
