//! Parity of the device-resident pass against the flag-off path, in two processes: one writes
//! the reference (flag off), the other compares (flag on).
//!
//! ```text
//! FIFI_INKLING_PASS_PARITY_OUT=ref.json cargo test ... resident_pass_parity -- --ignored
//! FIFI_INKLING_CUDA_RESIDENT_PREFILL_PASS=1 FIFI_INKLING_PASS_PARITY_REF=ref.json cargo test ...
//! ```
use super::*;
use std::{env, fs, path::Path};

fn fingerprint(model: &mut CpuForward, tokens: &[u32], prefix_rows: usize) -> serde_json::Value {
    let start = std::time::Instant::now();
    model.prefill_grouped(&tokens[..prefix_rows]).unwrap();
    let prefill_us = start.elapsed().as_micros() as u64;
    let visits: Vec<String> = (0..model.state.len())
        .map(|slot| format!("{:016x}", model.state[slot].parity_fingerprint()))
        .collect();
    let logits: Vec<f32> = tokens[prefix_rows..]
        .iter()
        .flat_map(|&token| model.token(token, true).unwrap().unwrap())
        .collect();
    serde_json::json!({
        "rows": prefix_rows,
        "prefill_us": prefill_us,
        "visits": visits,
        "logits": logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
    })
}

#[test]
#[ignore = "requires a leased CUDA device, the Inkling GGUF and fixed prompt token IDs"]
fn resident_pass_parity() {
    let checkpoint = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set the Inkling GGUF path");
    let tokens: Vec<u32> = env::var("FIFI_INKLING_TOKEN_IDS")
        .expect("set comma-separated token IDs")
        .split(',')
        .map(|value| value.parse().expect("valid token ID"))
        .collect();
    let rows: usize = env::var("FIFI_INKLING_PASS_ROWS")
        .map(|value| value.parse().expect("rows"))
        .unwrap_or(64);
    assert!((2..=512).contains(&rows) && rows < tokens.len());
    let reference = env::var_os("FIFI_INKLING_PASS_PARITY_REF");
    let output = env::var_os("FIFI_INKLING_PASS_PARITY_OUT");
    assert_ne!(reference.is_none(), output.is_none(), "set exactly one of REF or OUT");
    let mut model = CpuForward::open(Path::new(&checkpoint), tokens.len()).unwrap();
    model.enable_cuda(0).unwrap();
    let value = fingerprint(&mut model, &tokens, rows);
    if let Some(out) = output {
        fs::write(out, serde_json::to_vec(&value).unwrap()).unwrap();
        eprintln!("Inkling resident pass reference written ({rows} rows)");
        return;
    }
    let expected: serde_json::Value =
        serde_json::from_slice(&fs::read(reference.unwrap()).unwrap()).unwrap();
    let visits = value["visits"].as_array().unwrap();
    let expected_visits = expected["visits"].as_array().unwrap();
    let unequal_visits = visits.iter().zip(expected_visits).filter(|(a, b)| a != b).count();
    let logits: Vec<u32> = serde_json::from_value(value["logits"].clone()).unwrap();
    let expected_logits: Vec<u32> = serde_json::from_value(expected["logits"].clone()).unwrap();
    let max_abs = logits
        .iter()
        .zip(&expected_logits)
        .map(|(a, b)| (f32::from_bits(*a) - f32::from_bits(*b)).abs())
        .fold(0.0, f32::max);
    eprintln!(
        "Inkling resident pass parity {}",
        serde_json::json!({"rows": rows, "unequal_visits": unequal_visits, "visits": visits.len(),
            "logits_max_abs": max_abs, "prefill_us_pass": value["prefill_us"], "prefill_us_host": expected["prefill_us"]})
    );
    assert_eq!(visits.len(), expected_visits.len());
    assert_eq!(unequal_visits, 0, "visit state differs from the flag-off path");
    assert!(max_abs < 1e-2, "logits differ from the flag-off path by {max_abs}");
}
