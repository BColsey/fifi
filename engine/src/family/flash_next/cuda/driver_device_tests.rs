//! The token driver on the real pack, on a card: the first Flash-Next token through the same
//! assembly the engine start uses (`start::build_driver`), against the exllamav3 oracle.
//!
//! Environment:
//! - `FIFI_FLASH_NEXT_EXL3_PACK`: the full local pack directory (required).
//! - `FIFI_FLASH_NEXT_S7_ORACLE`: the oracle run JSON (`oracle-v1.json`); `FIFI_FLASH_NEXT_S7_PROMPT`
//!   names the prompt in it (default `chat-short`). The prompt ids are the oracle's.
//! - `FIFI_FLASH_NEXT_S7_STEPS`: greedy tokens to draw (default 16).
//! - `FIFI_FLASH_NEXT_S7_CONTEXT`: positions of the caches (default 512, a multiple of 4).
//! - `FIFI_FLASH_NEXT_S7_SLOTS`: expert slots per layer (default: the load plan's).
//! - `FIFI_FLASH_NEXT_S7_OUT`: a directory for the run record (`engine.json`, step 0 logits).
//!
//! The prompt is fed one token at a time (all but the last), then the driver draws `STEPS` greedy
//! tokens inside the tokenizer's vocabulary. Reported: the generated ids against the oracle's,
//! the top-1 agreement per step, the top-20 overlap of step 0, and the relative L2 and the largest
//! difference of the step-0 row against the oracle's full row. Nothing here asserts a speed.
//!
//! Needs a GPU (about 21 GiB), about 45 GiB of pinned host memory, `--test-threads=1`; ignored by
//! default.

use std::path::{Path, PathBuf};

use super::start::build_driver;
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

#[test]
#[ignore = "needs a GPU (about 21 GiB), the full pack in FIFI_FLASH_NEXT_EXL3_PACK and the oracle run JSON"]
fn the_first_real_tokens_match_the_oracle() {
    assert!(
        device_present("Flash-Next token driver"),
        "a GPU is required"
    );
    let pack = PathBuf::from(var("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK"));
    let steps: usize = var("FIFI_FLASH_NEXT_S7_STEPS").map_or(16, |v| v.parse().expect("steps"));
    let oracle_path =
        PathBuf::from(var("FIFI_FLASH_NEXT_S7_ORACLE").expect("FIFI_FLASH_NEXT_S7_ORACLE"));
    let wanted_id = var("FIFI_FLASH_NEXT_S7_PROMPT").unwrap_or_else(|| "chat-short".into());
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
        var("FIFI_FLASH_NEXT_S7_CONTEXT").map_or(512, |v| v.parse().expect("context"));
    assert!(
        prompt.len() + steps < context as usize,
        "the context is too small"
    );
    let wanted = CudaSettings {
        context,
        kv_format: KvFormatName::F16,
        slots: var("FIFI_FLASH_NEXT_S7_SLOTS").map(|v| v.parse().expect("slots")),
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
    for &token in head {
        driver.feed(token).expect("feed a prompt token");
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
    for step in 0..steps {
        let began = std::time::Instant::now();
        let row = driver
            .logits(input)
            .unwrap_or_else(|e| panic!("step {step}: {e:?}"));
        let ms = began.elapsed().as_secs_f64() * 1e3;
        let visible = &row[..actual_vocab];
        let pick = argmax(visible);
        let finite = visible.iter().all(|v| v.is_finite());
        let want = expected.get(step).copied();
        println!(
            "step {step}: input {input} -> argmax {pick} (oracle {want:?}), top-1 logit {:.4}, finite {finite}, {ms:.1} ms",
            visible[pick]
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
    println!("generated {generated:?}");
    println!("oracle    {:?}", &expected[..steps.min(expected.len())]);
    let agree = generated
        .iter()
        .zip(&expected)
        .take_while(|(a, b)| a == b)
        .count();
    println!("greedy prefix agreement: {agree} of {steps}");

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
    let got_top: Vec<usize> = top(got, 20).iter().map(|t| t.0).collect();
    let want_top: Vec<usize> = top(want, 20).iter().map(|t| t.0).collect();
    let overlap = got_top.iter().filter(|i| want_top.contains(i)).count();
    println!(
        "step 0 vs the oracle row: relative L2 {:.3e}, max abs {max_abs:.3e}, argmax {} vs {}, top-20 overlap {overlap}/20",
        (diff / norm).sqrt(),
        argmax(got),
        argmax(want)
    );
    if let Some(out) = var("FIFI_FLASH_NEXT_S7_OUT") {
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
