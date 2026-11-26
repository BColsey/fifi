//! Parity of the device-resident pass against the flag-off path, in two processes: one writes
//! the reference (flag off), the other compares (flag on).
//!
//! ```text
//! FIFI_INKLING_PASS_PARITY_OUT=ref.json cargo test ... resident_pass_parity -- --ignored
//! FIFI_INKLING_CUDA_RESIDENT_PREFILL_PASS=1 FIFI_INKLING_PASS_PARITY_REF=ref.json cargo test ...
//! ```
use super::*;
use std::{env, fs, path::Path};

/// Decode steps whose logits the reference keeps, besides the last step.
const LOGIT_STEPS: usize = 3;

fn fingerprint(model: &mut CpuForward, tokens: &[u32], prefix_rows: usize) -> serde_json::Value {
    let start = std::time::Instant::now();
    model.prefill_grouped(&tokens[..prefix_rows]).unwrap();
    let prefill_us = start.elapsed().as_micros() as u64;
    let visits: Vec<String> = (0..model.state.len())
        .map(|slot| format!("{:016x}", model.state[slot].parity_fingerprint()))
        .collect();
    // Per-part fingerprints of every visit, and the rows of the visits a dump names (default: the
    // first visit and the first banked one), so a differing run shows which part differs first.
    let (mut parts, mut detail) = (Vec::new(), serde_json::Map::new());
    for slot in 0..model.state.len() {
        let (slot_parts, slot_detail) = model.state[slot].parity_parts();
        parts.push(serde_json::json!(slot_parts));
        if [0, 18].contains(&slot) {
            detail.insert(slot.to_string(), slot_detail);
        }
    }
    let mut parts = serde_json::json!({"parts": parts, "detail": detail});
    if let Some(slots) = env::var("FIFI_INKLING_PASS_DUMP_SLOTS").ok() {
        // Every row of the named visits (a per-row comparison; large).
        let rows: Vec<serde_json::Value> = slots
            .split(',')
            .map(|slot| model.state[slot.trim().parse::<usize>().unwrap()].parity_rows())
            .collect();
        parts["rows"] = serde_json::json!(rows);
    }
    // Every decode step is teacher-forced; the logits of the first steps and of the last step are
    // kept (the full vocabulary), so the reference stays small for any block size.
    let steps = tokens.len() - prefix_rows;
    let mut logits = Vec::new();
    let mut argmax: Vec<u32> = Vec::with_capacity(steps);
    for &token in &tokens[prefix_rows..] {
        let full = model.token(token, true).unwrap().unwrap();
        argmax.push(
            full.iter()
                .enumerate()
                .max_by(|(a, x), (b, y)| x.total_cmp(y).then(b.cmp(a)))
                .map_or(0, |(a, _)| a as u32),
        );
        logits.extend(full);
    }
    let vocab = logits.len() / steps;
    let kept: Vec<u32> = (0..steps)
        .filter(|&step| step < LOGIT_STEPS || step + 1 == steps)
        .flat_map(|step| logits[step * vocab..(step + 1) * vocab].iter().map(|x| x.to_bits()))
        .collect();
    serde_json::json!({
        "rows": prefix_rows,
        "prefill_us": prefill_us,
        "visits": visits,
        "logits": kept,
        "argmax": argmax,
        "steps": steps,
        "parts": parts,
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
    let argmax: Vec<u32> = serde_json::from_value(value["argmax"].clone()).unwrap();
    let expected_argmax: Vec<u32> = serde_json::from_value(expected["argmax"].clone()).unwrap();
    let argmax_mismatches = argmax
        .iter()
        .zip(&expected_argmax)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "Inkling resident pass parity {}",
        serde_json::json!({"rows": rows, "unequal_visits": unequal_visits, "visits": visits.len(),
            "logits_max_abs": max_abs, "argmax_mismatches": argmax_mismatches, "steps": argmax.len(),
            "prefill_us_pass": value["prefill_us"], "prefill_us_host": expected["prefill_us"]})
    );
    assert_eq!(visits.len(), expected_visits.len());
    if unequal_visits != 0 {
        if let Some(out) = env::var_os("FIFI_INKLING_PASS_PARTS_OUT") {
            fs::write(out, serde_json::to_vec(&serde_json::json!({"pass": value["parts"], "host": expected["parts"]})).unwrap()).unwrap();
        }
    }
    assert_eq!(unequal_visits, 0, "visit state differs from the flag-off path");
    assert!(max_abs < 1e-2, "logits differ from the flag-off path by {max_abs}");
    assert_eq!(argmax, expected_argmax, "a greedy decode step picks another token");
}

/// Every constant the pass plans for every visit resolves in the checkpoint (host only: no CUDA).
#[test]
#[ignore = "requires the Inkling GGUF"]
fn pass_constant_names() {
    let checkpoint = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set the Inkling GGUF path");
    let model = CpuForward::open(Path::new(&checkpoint), 512 + 64).unwrap();
    let mut consts = Consts::default();
    let mut geometries: [Option<Geometry>; 2] = [None, None];
    for visit in &model.visits {
        let heads = model.pass_heads(visit.attention);
        let kind = usize::from(visit.attention == Attention::Global);
        let geometry = *geometries[kind].get_or_insert(Geometry {
            query: heads[0],
            kv: heads[1],
            channels: heads[0] * HEAD_DIM,
            kv_channels: heads[1] * HEAD_DIM,
            relative: heads[0] * model.math.d_rel,
        });
        CpuForward::visit_consts(&model, *visit, geometry, &mut consts).unwrap_or_else(|error| {
            panic!("visit {} (layer {}): {:?}", visit.state_slot, visit.weight_layer, error.describe())
        });
    }
    eprintln!("Inkling resident pass constants resolve for {} visits: {} words", model.visits.len(), consts.bytes.len());
}
