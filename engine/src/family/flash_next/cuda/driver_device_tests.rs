//! The token driver on the real pack, on a card: the first Flash-Next token through the same
//! assembly the engine start uses (`start::build_driver`), against the exllamav3 oracle.
//!
//! Environment:
//! - `FIFI_FLASH_NEXT_EXL3_PACK`: the full local pack directory (required).
//! - `FIFI_FLASH_NEXT_DRIVER_ORACLE`: the oracle run JSON; `FIFI_FLASH_NEXT_DRIVER_PROMPT`
//!   names the prompt in it (default `chat-short`). The prompt ids are the oracle's.
//! - `FIFI_FLASH_NEXT_DRIVER_STEPS`: greedy tokens to draw (default 16).
//! - `FIFI_FLASH_NEXT_DRIVER_CONTEXT`: positions of the caches (default 512, a multiple of 4).
//! - `FIFI_FLASH_NEXT_DRIVER_SLOTS`: expert slots per layer (default: the load plan's).
//! - `FIFI_FLASH_NEXT_DRIVER_OUT`: a directory for the run record (`engine.json`, step 0 logits).
//!
//! The prompt is fed one token at a time (all but the last), then the driver draws `STEPS` greedy
//! tokens inside the tokenizer's vocabulary. Reported: the generated ids against the oracle's,
//! the top-1 agreement per step, the top-20 overlap of step 0, and the relative L2 and the largest
//! difference of the step-0 row against the oracle's full row. Nothing here asserts a speed.
//!
//! Needs a GPU (about 21 GiB), about 25 GiB of pinned host memory, `--test-threads=1`; ignored by
//! default.

use std::path::{Path, PathBuf};

use std::time::Duration;

use super::{
    driver::{TokenDriver, TokenTimes},
    start::build_driver,
};
use crate::{
    family::{
        flash_next::settings::{CudaSettings, KvFormatName},
        session::TokenStepper,
    },
    kernels::cuda::device_present,
};

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// The `[rows, width]` float32 rows of a `.npy` file (version 1 or 2, C order).
fn read_npy(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not an npy file");
    let (header_len, start) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        _ => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
    };
    bytes[start + header_len..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in row.iter().enumerate() {
        if *value > row[best] {
            best = index;
        }
    }
    best
}

fn top(row: &[f32], count: usize) -> Vec<(usize, f32)> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|a, b| row[*b].total_cmp(&row[*a]).then(a.cmp(b)));
    order.into_iter().take(count).map(|i| (i, row[i])).collect()
}

/// One token's measurements: the driver's phases, the host argmax and the bank misses it caused.
#[derive(Clone, Copy)]
struct Sample {
    times: TokenTimes,
    wall: Duration,
    argmax: Duration,
    misses: u64,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn mean_and_p50(values: &mut [f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    values.sort_by(f64::total_cmp);
    (
        values.iter().sum::<f64>() / values.len() as f64,
        values[values.len() / 2],
    )
}

/// Prints the stage table of `samples`: mean and p50 in ms per token.
fn table(title: &str, samples: &[Sample]) {
    type Field = fn(&Sample) -> f64;
    let rows: [(&str, Field); 11] = [
        ("embed + n-gram rows", |s| ms(s.times.rows)),
        ("start write", |s| ms(s.times.start)),
        ("layers before PLE (queued)", |s| ms(s.times.layers_before)),
        ("stream read-back (device wait)", |s| ms(s.times.ple_sync)),
        ("host PLE step", |s| ms(s.times.ple_host)),
        ("stream write-back", |s| ms(s.times.ple_write)),
        ("layers from PLE on (queued)", |s| ms(s.times.layers_after)),
        ("head + logits read-back", |s| ms(s.times.head)),
        ("health read", |s| ms(s.times.health)),
        ("host argmax", |s| ms(s.argmax)),
        ("TOTAL wall", |s| ms(s.wall)),
    ];
    println!(
        "stage table, {title}, {} tokens (ms per token: mean / p50)",
        samples.len()
    );
    for (name, field) in rows {
        let mut values: Vec<f64> = samples.iter().map(field).collect();
        let (mean, p50) = mean_and_p50(&mut values);
        println!("  {name:<32} {mean:>9.3} {p50:>9.3}");
    }
    let mut misses: Vec<f64> = samples.iter().map(|s| s.misses as f64).collect();
    let (mean, p50) = mean_and_p50(&mut misses);
    let total: u64 = samples.iter().map(|s| s.misses).sum();
    println!("  bank misses per token            {mean:>9.2} {p50:>9.1}  (total {total})");
}

fn sample(
    driver: &TokenDriver<super::device::CardDevice, super::pack_feeds::PackFeeds>,
    wall: Duration,
    argmax: Duration,
    before: u64,
) -> Sample {
    Sample {
        times: driver.last_times(),
        wall,
        argmax,
        misses: driver.bank_misses().unwrap_or(0).saturating_sub(before),
    }
}

#[test]
#[ignore = "needs a GPU (about 21 GiB), the full pack in FIFI_FLASH_NEXT_EXL3_PACK and the oracle run JSON"]
fn the_first_real_tokens_match_the_oracle() {
    assert!(
        device_present("Flash-Next token driver"),
        "a GPU is required"
    );
    let pack = PathBuf::from(var("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK"));
    let steps: usize =
        var("FIFI_FLASH_NEXT_DRIVER_STEPS").map_or(16, |v| v.parse().expect("steps"));
    let oracle_path =
        PathBuf::from(var("FIFI_FLASH_NEXT_DRIVER_ORACLE").expect("FIFI_FLASH_NEXT_DRIVER_ORACLE"));
    let wanted_id = var("FIFI_FLASH_NEXT_DRIVER_PROMPT").unwrap_or_else(|| "chat-short".into());
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&oracle_path).expect("oracle json")).expect("json");
    let actual_vocab = oracle["vocab"]["actual_vocab"]
        .as_u64()
        .expect("actual_vocab") as usize;
    let record = oracle["prompts"]
        .as_array()
        .expect("prompts")
        .iter()
        .find(|p| p["id"] == wanted_id.as_str())
        .unwrap_or_else(|| panic!("no prompt {wanted_id}"));
    let ids = |key: &str| -> Vec<u32> {
        record[key]
            .as_array()
            .unwrap_or_else(|| panic!("no {key}"))
            .iter()
            .map(|v| v.as_u64().expect("an id") as u32)
            .collect()
    };
    let prompt = ids("prompt_ids");
    let expected = ids("generated");
    let context: u32 =
        var("FIFI_FLASH_NEXT_DRIVER_CONTEXT").map_or(512, |v| v.parse().expect("context"));
    assert!(
        prompt.len() + steps < context as usize,
        "the context is too small"
    );
    let wanted = CudaSettings {
        context,
        kv_format: KvFormatName::F16,
        slots: var("FIFI_FLASH_NEXT_DRIVER_SLOTS").map(|v| v.parse().expect("slots")),
    };
    let began = std::time::Instant::now();
    let mut driver = build_driver(
        &|stage| println!("stage: {stage}"),
        &pack,
        &wanted,
        actual_vocab,
        None,
    )
    .expect("build the driver");
    println!("load: {:.1} s", began.elapsed().as_secs_f64());

    let (last, head) = prompt.split_last().expect("a prompt");
    let fed = std::time::Instant::now();
    let mut prompt_samples = Vec::new();
    for &token in head {
        let before = driver.bank_misses().unwrap_or(0);
        let opened = std::time::Instant::now();
        driver.feed(token).expect("feed a prompt token");
        prompt_samples.push(sample(&driver, opened.elapsed(), Duration::ZERO, before));
    }
    println!(
        "prompt: {} tokens fed in {:.2} s",
        head.len(),
        fed.elapsed().as_secs_f64()
    );
    let mut input = *last;
    let mut generated: Vec<u32> = Vec::new();
    let mut records = Vec::new();
    let mut first_row: Vec<f32> = Vec::new();
    let mut step_samples = Vec::new();
    for step in 0..steps {
        let before = driver.bank_misses().unwrap_or(0);
        let began = std::time::Instant::now();
        let row = driver
            .logits(input)
            .unwrap_or_else(|e| panic!("step {step}: {e:?}"));
        let wall = began.elapsed();
        let visible = &row[..actual_vocab];
        let picking = std::time::Instant::now();
        let pick = argmax(visible);
        step_samples.push(sample(&driver, wall, picking.elapsed(), before));
        let wall_ms = ms(wall);
        let finite = visible.iter().all(|v| v.is_finite());
        let want = expected.get(step).copied();
        println!(
            "step {step}: input {input} -> argmax {pick} (oracle {want:?}), top-1 logit {:.4}, finite {finite}, {wall_ms:.1} ms, {} bank misses",
            visible[pick],
            step_samples.last().map_or(0, |s| s.misses)
        );
        assert!(finite, "step {step}: non-finite logits");
        records.push(serde_json::json!({
            "i": step, "token": pick, "argmax": pick,
            "top": top(visible, 20).iter().map(|(i, v)| serde_json::json!([i, v])).collect::<Vec<_>>(),
        }));
        if step == 0 {
            first_row = row.clone();
        }
        generated.push(pick as u32);
        input = pick as u32;
    }
    table("prompt tokens (no head)", &prompt_samples);
    table("generated tokens", &step_samples);
    println!("generated {generated:?}");
    println!("oracle    {:?}", &expected[..steps.min(expected.len())]);
    let agree = generated
        .iter()
        .zip(&expected)
        .take_while(|(a, b)| a == b)
        .count();
    println!("greedy prefix agreement: {agree} of {steps}");
    if agree < steps.min(expected.len()) {
        let oracle_steps = record["steps"].as_array().expect("steps");
        let at = &oracle_steps[agree];
        let top = at["top"].as_array().expect("top");
        println!(
            "first divergence at step {agree}: ours {} oracle {}; oracle top-1 {} top-2 {} margin {}",
            generated[agree], expected[agree], top[0][1], top[1][1], at["margin"]
        );
    }

    // Step 0 against the oracle's full row.
    let oracle_rows = read_npy(
        &oracle_path.parent().expect("dir").join(
            record["full_logits"]["file"]
                .as_str()
                .expect("full_logits file"),
        ),
    );
    let width = record["full_logits"]["width"].as_u64().expect("width") as usize;
    let oracle0 = &oracle_rows[..width];
    let (got, want) = (&first_row[..actual_vocab], &oracle0[..actual_vocab]);
    let (mut diff, mut norm, mut max_abs) = (0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let d = f64::from(*g) - f64::from(*w);
        diff += d * d;
        norm += f64::from(*w).powi(2);
        max_abs = max_abs.max(d.abs());
    }
    let log_sum_exp = |row: &[f32]| -> f64 {
        let top = f64::from(row[argmax(row)]);
        top + row
            .iter()
            .map(|v| (f64::from(*v) - top).exp())
            .sum::<f64>()
            .ln()
    };
    let (lse_got, lse_want) = (log_sum_exp(got), log_sum_exp(want));
    let kl: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| {
            let log_want = f64::from(*w) - lse_want;
            log_want.exp() * (log_want - (f64::from(*g) - lse_got))
        })
        .sum();
    println!("step 0 KL(oracle || ours) {kl:.4e}");
    let got_top: Vec<usize> = top(got, 20).iter().map(|t| t.0).collect();
    let want_top: Vec<usize> = top(want, 20).iter().map(|t| t.0).collect();
    let overlap = got_top.iter().filter(|i| want_top.contains(i)).count();
    println!(
        "step 0 vs the oracle row: relative L2 {:.3e}, max abs {max_abs:.3e}, argmax {} vs {}, top-20 overlap {overlap}/20",
        (diff / norm).sqrt(),
        argmax(got),
        argmax(want)
    );
    if let Some(out) = var("FIFI_FLASH_NEXT_DRIVER_OUT") {
        let out = PathBuf::from(out);
        std::fs::create_dir_all(&out).expect("out dir");
        let run = serde_json::json!({
            "schema": "fifi-flash-next-oracle/1", "role": "engine", "prompt": wanted_id,
            "prompt_ids": prompt, "generated": generated, "steps": records,
            "agree_prefix": agree,
        });
        std::fs::write(
            out.join("engine.json"),
            serde_json::to_vec_pretty(&run).expect("json"),
        )
        .expect("write engine.json");
        let raw: Vec<u8> = first_row.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out.join("step0.logits.f32"), raw).expect("write logits");
    }
}
