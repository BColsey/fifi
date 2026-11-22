//! Property tests of the device sampler. No full row is read back to the host.
use super::*;
use crate::kernels::cuda::{CudaContext, testkit};

fn settings(
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    seed: u64,
    pos: u64,
) -> SampleParams {
    SampleParams {
        temperature,
        top_k,
        top_p,
        min_p,
        seed,
        position: pos,
    }
}

fn draw(ctx: &CudaContext, stream: &DeviceStream, logits: &[f32], params: SampleParams) -> u32 {
    let n = logits.len();
    let l = testkit::upload_f32(ctx, logits);
    let probs = testkit::upload_filled(ctx, 0, n * 4);
    let token = testkit::upload_filled(ctx, 0, 4);
    let logprob = testkit::upload_filled(ctx, 0, 4);
    let status = testkit::upload_filled(ctx, 0, 4);
    ctx.sample(
        stream,
        SampleCall {
            token: token.range(0, 4).expect("token"),
            logprob: logprob.range(0, 4).expect("logprob"),
            status: status.range(0, 4).expect("status"),
            probs: probs.range(0, n * 4).expect("probs"),
            logits: l.range(0, n * 4).expect("logits"),
            n,
            params,
        },
    )
    .expect("sample");
    stream.synchronize().expect("drain");
    let bytes = testkit::read_bytes(ctx, &token, 4);
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[test]
fn zero_temperature_takes_the_lowest_argmax() {
    if !testkit::device_present("sampler greedy") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut logits = vec![0.0f32; 64];
    logits[9] = 5.0;
    logits[41] = 5.0;
    assert_eq!(
        draw(&ctx, &stream, &logits, settings(0.0, 0, 1.0, 0.0, 7, 0)),
        9
    );
}

#[test]
fn top_k_one_takes_the_argmax_for_every_position() {
    if !testkit::device_present("sampler top_k one") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut logits = vec![0.0f32; 64];
    logits[17] = 3.0;
    for pos in 0..8u64 {
        assert_eq!(
            draw(&ctx, &stream, &logits, settings(1.0, 1, 1.0, 0.0, 11, pos)),
            17
        );
    }
}

#[test]
fn draws_repeat_by_position_and_vary_across_positions() {
    if !testkit::device_present("sampler determinism") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let logits = vec![0.0f32; 64];
    let first = draw(&ctx, &stream, &logits, settings(1.0, 0, 1.0, 0.0, 3, 5));
    let again = draw(&ctx, &stream, &logits, settings(1.0, 0, 1.0, 0.0, 3, 5));
    assert_eq!(first, again, "same seed and position");
    let mut distinct = std::collections::BTreeSet::new();
    for pos in 0..64u64 {
        distinct.insert(draw(
            &ctx,
            &stream,
            &logits,
            settings(1.0, 0, 1.0, 0.0, 3, pos),
        ));
    }
    assert!(
        distinct.len() > 8,
        "only {} distinct tokens",
        distinct.len()
    );
}

#[test]
fn a_peaked_distribution_selects_the_dominant_token() {
    if !testkit::device_present("sampler peaked") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut logits = vec![-40.0f32; 64];
    logits[33] = 40.0;
    for pos in 0..16u64 {
        assert_eq!(
            draw(&ctx, &stream, &logits, settings(1.0, 0, 0.95, 0.0, 5, pos)),
            33
        );
    }
}

#[test]
fn refuses_a_top_p_of_zero() {
    if !testkit::device_present("sampler refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let n = 64;
    let l = testkit::upload_f32(&ctx, &vec![0.0f32; n]);
    let probs = testkit::upload_filled(&ctx, 0, n * 4);
    let token = testkit::upload_filled(&ctx, 0, 4);
    let logprob = testkit::upload_filled(&ctx, 0, 4);
    let status = testkit::upload_filled(&ctx, 0, 4);
    let err = ctx
        .sample(
            &stream,
            SampleCall {
                token: token.range(0, 4).expect("token"),
                logprob: logprob.range(0, 4).expect("logprob"),
                status: status.range(0, 4).expect("status"),
                probs: probs.range(0, n * 4).expect("probs"),
                logits: l.range(0, n * 4).expect("logits"),
                n,
                params: settings(1.0, 0, 0.0, 0.0, 1, 0),
            },
        )
        .expect_err("top_p refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}

#[test]
fn a_zero_uniform_skips_excluded_chunks() {
    if !testkit::device_present("sampler zero uniform") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut logits = [-30.0; 32];
    logits[31] = 30.0;
    assert_eq!(
        draw(
            &ctx,
            &stream,
            &logits,
            settings(1.0, 1, 1.0, 0.0, 6_786_339, 1)
        ),
        31
    );
}

#[test]
fn top_k_one_breaks_equal_logits_by_lowest_index() {
    if !testkit::device_present("sampler tied top_k") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    assert_eq!(
        draw(&ctx, &stream, &[0.0; 32], settings(1.0, 1, 1.0, 0.0, 3, 5)),
        0
    );
}

#[test]
fn top_p_uses_the_mass_remaining_after_top_k() {
    if !testkit::device_present("sampler composed top_k top_p") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut logits = [-100.0; 32];
    for (index, weight) in [1.0_f32, 0.75, 0.5, 0.25].into_iter().enumerate() {
        logits[index] = weight.ln();
    }
    for position in 0..16 {
        assert_eq!(
            draw(
                &ctx,
                &stream,
                &logits,
                settings(1.0, 2, 0.5, 0.0, 3, position)
            ),
            0,
            "position {position}"
        );
    }
}

/// A row with a few strong tokens, a body and a flat tail, `n` wide and not a multiple of
/// anything: the shape of a model row at the sampler width.
fn model_like_row(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let noise = ((i * 7919 + 13) % 1009) as f32 / 1009.0 * 3.0 - 1.5;
            let head = if i % 97 == 0 { 4.0 } else { 0.0 };
            noise + head - (i as f32 / n as f32)
        })
        .collect()
}

/// Two-sample chi-square of two histograms over the same categories, merging the categories the
/// two together saw fewer than ten times. Returns the statistic and its degrees of freedom.
fn two_sample_chi_square(a: &[u32], b: &[u32]) -> (f64, usize) {
    let (mut merged_a, mut merged_b) = (0.0f64, 0.0f64);
    let (mut statistic, mut cells) = (0.0f64, 0usize);
    let (total_a, total_b) = (a.iter().sum::<u32>() as f64, b.iter().sum::<u32>() as f64);
    let (ka, kb) = ((total_b / total_a).sqrt(), (total_a / total_b).sqrt());
    for (&x, &y) in a.iter().zip(b) {
        if x + y >= 10 {
            let d = ka * f64::from(x) - kb * f64::from(y);
            statistic += d * d / f64::from(x + y);
            cells += 1;
        } else {
            merged_a += f64::from(x);
            merged_b += f64::from(y);
        }
    }
    if merged_a + merged_b >= 10.0 {
        let d = ka * merged_a - kb * merged_b;
        statistic += d * d / (merged_a + merged_b);
        cells += 1;
    }
    (statistic, cells.saturating_sub(1))
}

#[test]
fn the_device_draws_follow_the_host_sampler_distribution() {
    use crate::family::sampler::{Sampler, SamplingParams};
    if !testkit::device_present("sampler distribution") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let n = 1000;
    let row = model_like_row(n);
    let draws = 6000u64;
    for (temperature, top_p, top_k) in [(1.0f64, 0.95f64, 20u32), (0.7, 1.0, 0), (1.3, 0.9, 0)] {
        let mut device = vec![0u32; n];
        for position in 0..draws {
            let params = settings(
                temperature as f32,
                top_k as usize,
                top_p as f32,
                0.0,
                0x5eed,
                position,
            );
            device[draw(&ctx, &stream, &row, params) as usize] += 1;
        }
        let mut host = vec![0u32; n];
        let mut sampler = Sampler::new(
            SamplingParams::new(temperature, top_p, top_k).expect("params"),
            0xfeed,
        );
        for _ in 0..draws {
            host[sampler.sample(&row).expect("host draw") as usize] += 1;
        }
        let (statistic, df) = two_sample_chi_square(&device, &host);
        // The statistic is near `df` with a spread of sqrt(2 df): five spreads is far beyond
        // chance and well inside any real mismatch of the kept set or its weights.
        let bound = df as f64 + 5.0 * (2.0 * df as f64).sqrt();
        assert!(
            statistic < bound,
            "T {temperature} top_p {top_p} top_k {top_k}: chi-square {statistic:.1} over {df} cells, bound {bound:.1}"
        );
        // The same kept set: nothing the host never draws, and the reverse.
        let outside = device
            .iter()
            .zip(&host)
            .filter(|&(&d, &h)| (d > 0) != (h > 0) && d + h >= 20)
            .count();
        assert_eq!(outside, 0, "T {temperature} top_p {top_p} top_k {top_k}");
    }
}

#[test]
fn a_draw_at_the_sampler_width_of_a_model_takes_what_time() {
    if !testkit::device_present("sampler timing") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let n = 248_077;
    let row = model_like_row(n);
    let l = testkit::upload_f32(&ctx, &row);
    let probs = testkit::upload_filled(&ctx, 0, n * 4);
    let token = testkit::upload_filled(&ctx, 0, 4);
    let logprob = testkit::upload_filled(&ctx, 0, 4);
    let status = testkit::upload_filled(&ctx, 0, 4);
    let launch = |position: u64, params: (f32, usize, f32)| {
        ctx.sample(
            &stream,
            SampleCall {
                token: token.range(0, 4).expect("token"),
                logprob: logprob.range(0, 4).expect("logprob"),
                status: status.range(0, 4).expect("status"),
                probs: probs.range(0, n * 4).expect("probs"),
                logits: l.range(0, n * 4).expect("logits"),
                n,
                params: settings(params.0, params.1, params.2, 0.0, 9, position),
            },
        )
        .expect("sample");
    };
    for (name, params) in [
        (
            "temperature 1 top_k 20 top_p 0.95",
            (1.0f32, 20usize, 0.95f32),
        ),
        ("temperature 0.7 top_k 0 top_p 1", (0.7, 0, 1.0)),
        ("greedy", (0.0, 0, 1.0)),
    ] {
        launch(0, params);
        stream.synchronize().expect("warm");
        let runs = 200u64;
        let began = std::time::Instant::now();
        for position in 0..runs {
            launch(position, params);
        }
        stream.synchronize().expect("drain");
        eprintln!(
            "device sampler at n {n}, {name}: {:.1} us per draw",
            began.elapsed().as_secs_f64() * 1e6 / runs as f64
        );
    }
}

/// One draw from the ranked top `count` of `row`, through the shortlist entry point.
fn draw_from_shortlist(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ranked: &[(f32, u32)],
    params: ShortlistDraw,
) -> (u32, u32) {
    let count = ranked.len();
    let mut bytes = Vec::with_capacity(count * 8 + 4);
    for &(logit, id) in ranked {
        bytes.extend_from_slice(&logit.to_le_bytes());
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    bytes.extend_from_slice(&0u32.to_le_bytes());
    let record = ctx.upload(&bytes).expect("upload");
    let token = testkit::upload_filled(ctx, 0, 4);
    let logprob = testkit::upload_filled(ctx, 0, 4);
    let status = testkit::upload_filled(ctx, 0, 4);
    ctx.sample_shortlist(
        stream,
        SampleShortlistCall {
            token: token.range(0, 4).expect("token"),
            logprob: logprob.range(0, 4).expect("logprob"),
            status: status.range(0, 4).expect("status"),
            entries: record.range(0, count * 8).expect("entries"),
            flag: record.range(count * 8, 4).expect("flag"),
            count,
            draw: params,
        },
    )
    .expect("shortlist draw");
    stream.synchronize().expect("drain");
    let id = testkit::read_bytes(ctx, &token, 4);
    let state = testkit::read_bytes(ctx, &status, 4);
    (
        u32::from_le_bytes([id[0], id[1], id[2], id[3]]),
        u32::from_le_bytes([state[0], state[1], state[2], state[3]]),
    )
}

#[test]
fn the_shortlist_draws_follow_the_host_sampler_distribution() {
    use crate::family::sampler::{Sampler, SamplingParams};
    if !testkit::device_present("shortlist draw distribution") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let n = 1000;
    let row = model_like_row(n);
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        row[b as usize]
            .total_cmp(&row[a as usize])
            .then_with(|| a.cmp(&b))
    });
    let draws = 6000u64;
    for (temperature, top_p, top_k) in [(1.0f64, 0.95f64, 20u32), (0.7, 1.0, 20), (1.3, 0.9, 8)] {
        let ranked: Vec<(f32, u32)> = order[..20].iter().map(|&i| (row[i as usize], i)).collect();
        let mut device = vec![0u32; n];
        for position in 0..draws {
            let (id, status) = draw_from_shortlist(
                &ctx,
                &stream,
                &ranked,
                ShortlistDraw {
                    temperature: temperature as f32,
                    top_k: top_k as usize,
                    top_p: top_p as f32,
                    seed: 0x5eed,
                    position,
                },
            );
            assert_eq!(status, 0);
            device[id as usize] += 1;
        }
        let mut host = vec![0u32; n];
        let mut sampler = Sampler::new(
            SamplingParams::new(temperature, top_p, top_k).expect("params"),
            0xfeed,
        );
        for _ in 0..draws {
            host[sampler.sample(&row).expect("host draw") as usize] += 1;
        }
        let (statistic, df) = two_sample_chi_square(&device, &host);
        let bound = df as f64 + 5.0 * (2.0 * df as f64).sqrt();
        assert!(
            statistic < bound,
            "T {temperature} top_p {top_p} top_k {top_k}: chi-square {statistic:.1} over {df} cells, bound {bound:.1}"
        );
    }
}

#[test]
fn a_shortlist_draw_keeps_the_status_bits_it_does_not_own() {
    if !testkit::device_present("shortlist draw status") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let ranked = [(2.0f32, 7u32), (1.0, 3), (0.5, 9)];
    let mut bytes = Vec::new();
    for &(logit, id) in &ranked {
        bytes.extend_from_slice(&logit.to_le_bytes());
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    bytes.extend_from_slice(&0u32.to_le_bytes());
    let record = ctx.upload(&bytes).expect("upload");
    let token = testkit::upload_filled(&ctx, 0, 4);
    let logprob = testkit::upload_filled(&ctx, 0, 4);
    let status = ctx.upload(&8u32.to_le_bytes()).expect("upload");
    ctx.sample_shortlist(
        &stream,
        SampleShortlistCall {
            token: token.range(0, 4).expect("token"),
            logprob: logprob.range(0, 4).expect("logprob"),
            status: status.range(0, 4).expect("status"),
            entries: record.range(0, 24).expect("entries"),
            flag: record.range(24, 4).expect("flag"),
            count: 3,
            draw: ShortlistDraw {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                seed: 1,
                position: 0,
            },
        },
    )
    .expect("greedy shortlist draw");
    stream.synchronize().expect("drain");
    let picked = testkit::read_bytes(&ctx, &token, 4);
    let kept = testkit::read_bytes(&ctx, &status, 4);
    assert_eq!(
        u32::from_le_bytes([picked[0], picked[1], picked[2], picked[3]]),
        7
    );
    assert_eq!(u32::from_le_bytes([kept[0], kept[1], kept[2], kept[3]]), 8);
}

#[test]
fn the_wide_shortlist_ranks_like_the_host_and_its_draws_follow_the_host_sampler() {
    use crate::family::sampler::{Sampler, SamplingParams};
    use crate::kernels::cuda::resident::shortlist::{ShortlistRowsCall, WIDE_K, shape};
    if !testkit::device_present("wide shortlist draw") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let n = 1000;
    let row = model_like_row(n);
    let layout = shape(WIDE_K).expect("wide layout");
    let logits = testkit::upload_f32(&ctx, &row);
    let scratch = testkit::upload_filled(&ctx, 0, layout.result_offset);
    let records = testkit::upload_filled(&ctx, 0, layout.part_bytes);
    ctx.shortlist_rows(
        &stream,
        ShortlistRowsCall {
            k: WIDE_K,
            m: 1,
            n,
            ld: n,
            logits: logits.range(0, n * 4).expect("logits"),
            scratch: scratch.range(0, layout.result_offset).expect("scratch"),
            scratch_stride: layout.result_offset,
            out: records.range(0, layout.part_bytes).expect("records"),
        },
    )
    .expect("wide shortlist");
    stream.synchronize().expect("drain");
    let bytes = testkit::read_bytes(&ctx, &records, layout.part_bytes);
    // The record is the host's stable ranking: larger logit first, then lower id.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        row[b as usize]
            .total_cmp(&row[a as usize])
            .then_with(|| a.cmp(&b))
    });
    for (rank, &id) in order[..WIDE_K].iter().enumerate() {
        let at = rank * 8;
        let device_logit = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let device_id = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap());
        assert_eq!(
            (device_id, device_logit.to_bits()),
            (id, row[id as usize].to_bits()),
            "rank {rank}"
        );
    }
    assert_eq!(&bytes[WIDE_K * 8..WIDE_K * 8 + 4], &[0; 4], "finite row");
    let draws = 6000u64;
    for (temperature, top_p, top_k) in [(1.0f64, 0.95f64, 64u32), (0.8, 1.0, 40)] {
        let token = testkit::upload_filled(&ctx, 0, 4);
        let logprob = testkit::upload_filled(&ctx, 0, 4);
        let status = testkit::upload_filled(&ctx, 0, 4);
        let mut device = vec![0u32; n];
        for position in 0..draws {
            ctx.sample_shortlist(
                &stream,
                SampleShortlistCall {
                    token: token.range(0, 4).expect("token"),
                    logprob: logprob.range(0, 4).expect("logprob"),
                    status: status.range(0, 4).expect("status"),
                    entries: records.range(0, WIDE_K * 8).expect("entries"),
                    flag: records.range(WIDE_K * 8, 4).expect("flag"),
                    count: WIDE_K,
                    draw: ShortlistDraw {
                        temperature: temperature as f32,
                        top_k: top_k as usize,
                        top_p: top_p as f32,
                        seed: 0xb017,
                        position,
                    },
                },
            )
            .expect("wide draw");
            stream.synchronize().expect("drain");
            let id = testkit::read_bytes(&ctx, &token, 4);
            device[u32::from_le_bytes([id[0], id[1], id[2], id[3]]) as usize] += 1;
        }
        let mut host = vec![0u32; n];
        let mut sampler = Sampler::new(
            SamplingParams::new(temperature, top_p, top_k).expect("params"),
            0xfeed,
        );
        for _ in 0..draws {
            host[sampler.sample(&row).expect("host draw") as usize] += 1;
        }
        let (statistic, df) = two_sample_chi_square(&device, &host);
        let bound = df as f64 + 5.0 * (2.0 * df as f64).sqrt();
        assert!(
            statistic < bound,
            "T {temperature} top_p {top_p} top_k {top_k}: chi-square {statistic:.1} over {df} cells, bound {bound:.1}"
        );
    }
}
