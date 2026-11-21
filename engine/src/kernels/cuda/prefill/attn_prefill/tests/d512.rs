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

use super::super::{AttnKvLayout, AttnKvView, AttnScalarLayout, AttnStrides};
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
    let plan = cpf_attn_launch_plan(
        CpfAttnGeometry {
            m: shape.m,
            n_head: shape.n_head,
            n_kv_head: shape.n_kv_head,
            head_dim: shape.head_dim,
        },
        shape.p0 as usize + shape.m,
        window,
        CPF_ATTN_SM_COUNT,
    )?;
    Ok(CpfAttnSplit {
        max_splits: plan.max_splits,
        min_keys: plan.min_keys,
    })
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
    let shape = shape(m, cache.cells, p0);
    let split = plan(&shape, window)?;
    let span = CpfAttnSpan {
        window: window as i64,
        ring: 0,
    };
    let io = io(&ctx, &shape, split, q)?;
    ctx.attn_prefill(&stream, call(shape, split, span, &planes, &io)?)?;
    stream.synchronize()?;
    let prefill = testkit::floats(&testkit::read_bytes(&ctx, &io.out, io.out_bytes));
    let status = i32::from_le_bytes(
        testkit::read_bytes(&ctx, &io.status, 4)[..4]
            .try_into()
            .expect("the status word"),
    );
    let guards = testkit::read_bytes(&ctx, &io.part, io.part_bytes)
        .iter()
        .any(|&b| b != PART_FILL);
    let mut rows = vec![0f32; if decode { m * HEADS * WIDE } else { 0 }];
    if decode {
        let launches = m.div_ceil(16);
        let words: Vec<u8> = (0..launches)
            .flat_map(|i| ((p0 + 16 * i + 1) as i32).to_le_bytes())
            .collect();
        let lens = testkit::upload_bytes(&ctx, &words);
        let dec_out = testkit::upload_filled(&ctx, OUT_FILL, io.out_bytes);
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
                    q: io.q.range(t0 * row_bytes, n * row_bytes)?,
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
                },
            )?;
        }
        stream.synchronize()?;
        rows = testkit::floats(&testkit::read_bytes(&ctx, &dec_out, io.out_bytes));
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
