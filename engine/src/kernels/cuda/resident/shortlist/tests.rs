use super::*;
use crate::family::sampler::{Sampler, SamplingParams};
use crate::kernels::cuda::testkit;

struct RowEvidence {
    row: Vec<f32>,
    candidates: Vec<(u32, f32)>,
    non_finite: u32,
}

fn device_rows() -> Result<Vec<RowEvidence>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut evidence = Vec::new();
    for n in [1, 3, 19, 20, 21, 257, 248_077] {
        let mut row: Vec<f32> = (0..n)
            .map(|i| ((i * 37 + 11) % 1021) as f32 / 32.0 - 15.9375)
            .collect();
        if n == 3 {
            row = vec![-0.0, 0.0, -0.0];
        }
        let raw: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
        let input = ctx.upload(&raw)?;
        let scratch = ctx.upload(&[0u8; 3072])?;
        ctx.shortlist(
            &stream,
            ShortlistCall {
                logits: input.range(0, n * 4)?,
                scratch: scratch.range(0, 3072)?,
                n,
            },
        )?;
        let mut bytes = [0u8; RESULT_BYTES];
        scratch.read_at(&stream, RESULT_OFFSET, &mut bytes)?;
        let candidates = (0..n.min(K))
            .map(|i| {
                let o = i * 8;
                (
                    u32::from_le_bytes([bytes[o + 4], bytes[o + 5], bytes[o + 6], bytes[o + 7]]),
                    f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]),
                )
            })
            .collect();
        evidence.push(RowEvidence {
            row,
            candidates,
            non_finite: u32::from_le_bytes([bytes[160], bytes[161], bytes[162], bytes[163]]),
        });
    }
    Ok(evidence)
}

#[test]
fn device_shortlist_preserves_stable_ids_bits_and_seeded_host_tokens() {
    if !testkit::device_present("device shortlist parity") {
        return;
    }
    // Device owners are gone before any assertion, so a failed oracle cannot latch CUDA.
    let evidence = device_rows().expect("device phase");
    for RowEvidence {
        row,
        candidates,
        non_finite,
    } in evidence
    {
        assert_eq!(non_finite, 0);
        let mut expected: Vec<(u32, f32)> = row
            .iter()
            .copied()
            .enumerate()
            .map(|(i, x)| (i as u32, x))
            .collect();
        expected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        expected.truncate(K);
        assert_eq!(
            candidates
                .iter()
                .map(|(id, x)| (*id, x.to_bits()))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|(id, x)| (*id, x.to_bits()))
                .collect::<Vec<_>>()
        );
        assert_eq!(candidates[0].0, Sampler::greedy().sample(&row).unwrap());
        for seed in [0, 7, u64::MAX] {
            let settings = SamplingParams::new(1.0, 0.95, 20).unwrap();
            let mut host = Sampler::new(settings, seed);
            let mut compact = Sampler::new(settings, seed);
            for _ in 0..128 {
                assert_eq!(
                    compact.sample_shortlist(&candidates, row.len()).unwrap(),
                    host.sample(&row).unwrap()
                );
            }
        }
    }
}

fn device_range_checks() -> Result<(u32, Vec<bool>), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut row = vec![1.0f32; 257];
    row[256] = f32::NEG_INFINITY;
    let raw: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
    let input = ctx.upload(&raw)?;
    let scratch = ctx.upload(&[0u8; 3072])?;
    let call = || -> Result<ShortlistCall<'_>, EngError> {
        Ok(ShortlistCall {
            logits: input.range(0, 257 * 4)?,
            scratch: scratch.range(0, 3072)?,
            n: 257,
        })
    };
    ctx.shortlist(&stream, call()?)?;
    let mut bytes = [0u8; RESULT_BYTES];
    scratch.read_at(&stream, RESULT_OFFSET, &mut bytes)?;
    let mut refusals = Vec::new();
    let mut bad = call()?;
    bad.n = 0;
    refusals.push(ctx.shortlist(&stream, bad).is_err());
    let mut bad = call()?;
    bad.n = 258;
    refusals.push(ctx.shortlist(&stream, bad).is_err());
    let mut bad = call()?;
    bad.scratch = scratch.range(0, SCRATCH_BYTES - 1)?;
    refusals.push(ctx.shortlist(&stream, bad).is_err());
    let mut bad = call()?;
    bad.scratch = scratch.range(1, 3071)?;
    refusals.push(ctx.shortlist(&stream, bad).is_err());
    let mut bad = call()?;
    bad.logits = scratch.range(0, 257 * 4)?;
    refusals.push(ctx.shortlist(&stream, bad).is_err());
    Ok((
        u32::from_le_bytes([bytes[160], bytes[161], bytes[162], bytes[163]]),
        refusals,
    ))
}

#[test]
fn device_shortlist_rejects_bad_ranges_and_certifies_the_entire_row() {
    if !testkit::device_present("device shortlist bounds") {
        return;
    }
    let (non_finite, refusals) = device_range_checks().expect("device phase");
    assert_eq!(non_finite, 1);
    assert!(refusals.into_iter().all(|refused| refused));
}

/// The device softmax mass and the host f64 total of the same row at the same temperature.
fn device_mass(row: &[f32], temperature: f64) -> Result<(f64, f64), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let n = row.len();
    let raw: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
    let input = ctx.upload(&raw)?;
    let scratch = ctx.upload(&[0u8; 3072])?;
    ctx.shortlist(
        &stream,
        ShortlistCall {
            logits: input.range(0, n * 4)?,
            scratch: scratch.range(0, 3072)?,
            n,
        },
    )?;
    ctx.shortlist_mass(
        &stream,
        ShortlistMassCall {
            logits: input.range(0, n * 4)?,
            scratch: scratch.range(0, 3072)?,
            n,
            inv_temperature: (1.0 / temperature) as f32,
        },
    )?;
    let mut parts = [0u8; MASS_PARTS * 8];
    scratch.read_at(&stream, MASS_OFFSET, &mut parts)?;
    let device: f64 = parts
        .as_chunks::<8>()
        .0
        .iter()
        .map(|part| f64::from_le_bytes(*part))
        .sum();
    let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f64> = row
        .iter()
        .map(|&x| ((f64::from(x) - f64::from(top)) / temperature).exp())
        .collect();
    weights.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let host = weights.iter().sum();
    Ok((device, host))
}

#[test]
fn the_device_mass_is_the_host_total_within_the_sampler_bound() {
    if !testkit::device_present("device shortlist mass") {
        return;
    }
    let bound = crate::family::sampler::DEVICE_MASS_RELATIVE_BOUND;
    let mut worst = 0.0_f64;
    for n in [1, 3, 20, 257, 4_096, 248_077, 248_320] {
        for temperature in [0.3, 0.7, 1.0, 1.5] {
            // A row shaped like a model row: strong head, wide spread, flat tail.
            let row: Vec<f32> = (0..n)
                .map(|i| {
                    let noise = ((i * 7919 + 13) % 1009) as f32 / 1009.0 * 4.0;
                    let head = if i < 12 { 12.0 - i as f32 } else { 0.0 };
                    noise + head
                })
                .collect();
            let (device, host) = device_mass(&row, temperature).expect("device phase");
            let relative = ((device - host) / host).abs();
            worst = worst.max(relative);
            assert!(
                relative < bound / 3.0,
                "n {n} T {temperature}: device {device} host {host} relative {relative}"
            );
        }
    }
    eprintln!("worst relative error of the device mass: {worst:e} (sampler bound {bound:e})");
}

#[test]
fn the_mass_launch_refuses_a_bad_temperature_or_a_short_scratch() {
    if !testkit::device_present("device shortlist mass bounds") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let row = vec![1.0f32; 257];
    let raw: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
    let input = ctx.upload(&raw).expect("upload");
    let scratch = ctx.upload(&[0u8; 3072]).expect("upload");
    let call = |inv_temperature: f32, scratch_bytes: usize, n: usize| ShortlistMassCall {
        logits: input.range(0, 257 * 4).expect("range"),
        scratch: scratch.range(0, scratch_bytes).expect("range"),
        n,
        inv_temperature,
    };
    assert!(ctx.shortlist_mass(&stream, call(0.0, 3072, 257)).is_err());
    assert!(
        ctx.shortlist_mass(&stream, call(f32::NAN, 3072, 257))
            .is_err()
    );
    assert!(
        ctx.shortlist_mass(&stream, call(1.0, MASS_SCRATCH_BYTES - 1, 257))
            .is_err()
    );
    assert!(ctx.shortlist_mass(&stream, call(1.0, 3072, 258)).is_err());
    assert!(ctx.shortlist_mass(&stream, call(1.0, 3072, 0)).is_err());
}
