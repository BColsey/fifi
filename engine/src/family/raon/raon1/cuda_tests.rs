//! Device tests of the Raon-1 forward pass against the host reference. Both need the published
//! Q4_K_M file (`RAON_Q4_KM`) and a device; a host without either skips with a note.
//!
//! The host reference is slow (minutes for a short prompt), so its all-position logits can be
//! kept in the file named by `RAON_REF_LOGITS`: the first run writes it, later runs read it.

use std::path::PathBuf;

use super::cuda::{HeadRows, Raon1Device};
use super::reference::{attn_half, ffn_half, head, hyper};
use super::{Raon1Model, open_raon_1_mini_preview};
use crate::kernels::cuda::device_present;

/// The first twelve tokens of the frozen fixture prompt.
const PROMPT: [u32; 12] = [
    166_863, 54_935, 7_821, 44, 11_456, 189_096, 257_706, 142_907, 189_286, 9_305, 39_154, 166_294,
];

/// The largest gap between the host's top two logits at which a top-1 difference is accepted.
const NEAR_TIE: f32 = 0.1;

/// The largest difference over range accepted for a residual stage (`hc_attn_out`, `l_out`) of
/// any layer. Measured over 25 layers of a 4-token prompt: at most 1.1e-3 before the first
/// routing difference, 3.3e-2 after the routing differences of layers 5 to 20.
const STAGE_TOLERANCE: f32 = 4e-2;

/// The largest margin, between the host's sixth and seventh biased router scores, at which the
/// device may route a token to a different set of experts. Measured: at most 0.037, in every
/// case a near tie that activation quantization decides.
const MAX_FLIP_MARGIN: f32 = 0.05;

/// The largest absolute logit difference accepted over the logits of the sequence test.
/// Measured: 1.37 over 24 positions.
const MAX_LOGIT_DIFF: f32 = 2.0;

/// Greedy tokens decoded one at a time after the prompt.
const DECODED: usize = 12;

fn model_path() -> Option<PathBuf> {
    match std::env::var("RAON_Q4_KM") {
        Ok(path) => Some(PathBuf::from(path)),
        Err(_) => {
            eprintln!("skipped: RAON_Q4_KM is unset");
            None
        }
    }
}

/// The host reference's residual after the embedding: the token row in all four streams.
fn host_embedding(model: &Raon1Model, tokens: &[u32]) -> Vec<f32> {
    let spec = model.spec();
    let h = spec.hidden as usize;
    let hc = spec.streams as usize;
    let table = model.read_f32("token_embd.weight").expect("embedding");
    let mut inp = vec![0f32; h * hc * tokens.len()];
    for (t, &token) in tokens.iter().enumerate() {
        let row = &table[token as usize * h..(token as usize + 1) * h];
        for s in 0..hc {
            let at = (t * hc + s) * h;
            inp[at..at + h].copy_from_slice(row);
        }
    }
    inp
}

/// The host logits of every position of `tokens`.
fn host_logits(model: &Raon1Model, tokens: &[u32]) -> Vec<Vec<f32>> {
    let spec = model.spec();
    let hy = hyper(model, spec).expect("hyperparameters");
    let hc = spec.streams as usize;
    let h = spec.hidden as usize;
    let mut inp = host_embedding(model, tokens);
    let mut sink = |_: &str, _: &[f32]| {};
    for layer in 0..spec.layers {
        inp = attn_half(model, spec, &hy, &inp, layer, &mut sink).expect("attention half");
        inp = ffn_half(model, spec, &hy, &inp, layer, &mut sink, None).expect("ffn half");
    }
    (0..tokens.len())
        .map(|t| {
            let prefix = &inp[..(t + 1) * h * hc];
            head(model, spec, &hy, prefix, t + 1, &mut sink).expect("head")
        })
        .collect()
}

fn top_k(row: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    order.truncate(k);
    order
}

/// Prefill logits for every position of the prompt, then greedy single-token decode: the device
/// logits row of every position of the whole sequence.
fn device_sequence(device: &mut Raon1Device) -> (Vec<u32>, Vec<Vec<f32>>) {
    device.reset().expect("reset");
    let mut tokens = PROMPT.to_vec();
    let mut rows = device.feed(&PROMPT, 0, HeadRows::All).expect("prompt feed");
    for _ in 0..DECODED {
        let next = top_k(rows.last().expect("a logits row"), 1)[0] as u32;
        let mut out = device
            .feed(&[next], tokens.len(), HeadRows::Last)
            .expect("decode step");
        tokens.push(next);
        rows.append(&mut out);
    }
    (tokens, rows)
}

fn read_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn read_rows(bytes: &[u8], vocab: usize) -> Vec<Vec<f32>> {
    bytes
        .chunks_exact(vocab * 4)
        .map(|row| {
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        })
        .collect()
}

fn write_rows(path: &std::path::Path, rows: &[Vec<f32>]) {
    let bytes: Vec<u8> = rows
        .iter()
        .flat_map(|row| row.iter().flat_map(|v| v.to_le_bytes()))
        .collect();
    std::fs::write(path, bytes).expect("write logits");
}

/// Writes the greedy sequence of the device (prompt plus decoded tokens) to
/// `RAON_DEVICE_SEQ` and its logits rows, one per position, to `RAON_DEVICE_SEQ.logits`.
#[test]
fn the_device_writes_its_greedy_sequence_and_logits() {
    let Some(path) = model_path() else { return };
    let Ok(out) = std::env::var("RAON_DEVICE_SEQ").map(PathBuf::from) else {
        eprintln!("skipped: RAON_DEVICE_SEQ is unset");
        return;
    };
    if !device_present("the_device_writes_its_greedy_sequence_and_logits") {
        return;
    }
    let mut device = Raon1Device::load(&path, Some(4096), &mut |_| {}).expect("device load");
    let (tokens, rows) = device_sequence(&mut device);
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    std::fs::write(&out, bytes).expect("write the sequence");
    write_rows(&out.with_extension("logits"), &rows);
    eprintln!("device sequence {tokens:?}");
}

/// Scores the device sequence on the host reference (slow; the reference logits are kept in
/// `RAON_REF_LOGITS`) and compares every position: top-1, the top-20 set, the largest absolute
/// difference.
#[test]
fn the_host_reference_agrees_with_the_device_logits_at_every_position() {
    let Some(path) = model_path() else { return };
    let (Ok(seq), Ok(ref_path)) = (
        std::env::var("RAON_DEVICE_SEQ").map(PathBuf::from),
        std::env::var("RAON_REF_LOGITS").map(PathBuf::from),
    ) else {
        eprintln!("skipped: RAON_DEVICE_SEQ or RAON_REF_LOGITS is unset");
        return;
    };
    let model = open_raon_1_mini_preview(&path, None).expect("open");
    let vocab = model.spec().vocab as usize;
    let tokens = read_u32s(&std::fs::read(&seq).expect("the device sequence"));
    let device_rows = read_rows(
        &std::fs::read(seq.with_extension("logits")).expect("the device logits"),
        vocab,
    );
    assert_eq!(tokens.len(), device_rows.len());
    let host_rows = match std::fs::read(&ref_path) {
        Ok(bytes) if bytes.len() == tokens.len() * vocab * 4 => read_rows(&bytes, vocab),
        _ => {
            let rows = host_logits(&model, &tokens);
            write_rows(&ref_path, &rows);
            rows
        }
    };
    let mut worst = 0f32;
    let mut failures = Vec::new();
    for (position, (host, dev)) in host_rows.iter().zip(&device_rows).enumerate() {
        let diff = host
            .iter()
            .zip(dev)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        worst = worst.max(diff);
        let (host20, dev20) = (top_k(host, 20), top_k(dev, 20));
        let mut sorted_host = host20.clone();
        let mut sorted_dev = dev20.clone();
        sorted_host.sort_unstable();
        sorted_dev.sort_unstable();
        let top1 = host20[0] == dev20[0];
        let sets = sorted_host == sorted_dev;
        let overlap = sorted_host
            .iter()
            .filter(|id| sorted_dev.contains(id))
            .count();
        let margin = host[host20[0]] - host[host20[1]];
        eprintln!(
            "position {position:2}: top1 host {:6} device {:6} {} | top20 sets {} (overlap {overlap}) | max abs diff {diff:.4} | host margin {margin:.3}",
            host20[0],
            dev20[0],
            if top1 { "equal" } else { "DIFFER" },
            if sets { "equal" } else { "DIFFER" },
        );
        // A top-1 flip is accepted only where the host's own top two are within the
        // activation-quantization noise of each other; the top-20 sets may differ at the
        // boundary of the set.
        if (!top1 && margin > NEAR_TIE) || diff > MAX_LOGIT_DIFF {
            failures.push(position);
        }
    }
    eprintln!("max abs logit difference over all positions: {worst:.4}");
    assert!(
        failures.is_empty(),
        "top-1 differs beyond a near tie, or the logits differ by more than {MAX_LOGIT_DIFF}, at positions {failures:?}"
    );
}

/// A deterministic prompt of ids spread over the vocabulary (not text).
fn synthetic_tokens(count: usize) -> Vec<u32> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            1000 + ((state >> 33) % 200_000) as u32
        })
        .collect()
}

/// The per-layer stages of the first `layers` layers, device against host. The device feeds
/// `tokens[..split]` untraced, then traces the feed of the rest; the host runs the whole
/// sequence and its rows from `split` on are compared. Prints the difference over the range of
/// every stage, and for every MoE layer the routed experts of the device against the host's
/// top six by biased score with the host's margin to the seventh. Returns the stages whose
/// difference exceeds `tolerance` of their range and the routing disagreements.
fn compare_stages(
    path: &std::path::Path,
    tokens: &[u32],
    split: usize,
    layers: usize,
    tolerance: f32,
) -> (Vec<String>, Vec<(String, f32)>) {
    let model = open_raon_1_mini_preview(path, None).expect("open");
    let spec = model.spec();
    let hy = hyper(&model, spec).expect("hyperparameters");
    let (hc, h, ne, nu) = (
        spec.streams as usize,
        spec.hidden as usize,
        spec.experts as usize,
        hy.used_experts,
    );
    let nt = tokens.len();
    let mut host: Vec<super::cuda::Stage> = Vec::new();
    let mut inp = host_embedding(&model, tokens);
    for layer in 0..layers {
        let mut sink = |name: &str, data: &[f32]| {
            if [
                "attn_norm",
                "hc_attn_out",
                "ffn_norm",
                "l_out",
                "ffn_moe_probs_biased",
            ]
            .iter()
            .any(|stage| name == format!("{stage}-{layer}"))
            {
                host.push((name.to_owned(), data.to_vec()));
            }
        };
        inp = attn_half(&model, spec, &hy, &inp, layer, &mut sink).expect("attention half");
        inp = ffn_half(&model, spec, &hy, &inp, layer, &mut sink, None).expect("ffn half");
    }
    let mut device = Raon1Device::load(path, Some(nt.max(4096).next_multiple_of(256)), &mut |_| {})
        .expect("device load");
    device.reset().expect("reset");
    if split > 0 {
        device
            .feed(&tokens[..split], 0, HeadRows::None)
            .expect("prefix feed");
    }
    device.trace_next_feed();
    device
        .feed(&tokens[split..], split, HeadRows::None)
        .expect("traced feed");
    let traced = device.take_trace();
    // A chunked feed traces the same stage once per chunk: join them in order.
    let joined = |name: &str| -> Vec<f32> {
        traced
            .iter()
            .filter(|(n, _)| n == name)
            .flat_map(|(_, data)| data.iter().copied())
            .collect()
    };
    let (mut bad, mut routing) = (Vec::new(), Vec::new());
    for layer in 0..layers {
        for stage in ["attn_norm", "hc_attn_out", "ffn_norm", "l_out"] {
            // The norm outputs have a small range and are reported only; the residuals are gated.
            let gated = stage == "hc_attn_out" || stage == "l_out";
            let name = format!("{stage}-{layer}");
            let (_, all) = host.iter().find(|(n, _)| *n == name).expect("host stage");
            let per_token = all.len() / nt;
            let want = &all[split * per_token..];
            let got = joined(&name);
            assert_eq!(want.len(), got.len(), "{name}: length");
            let range = want.iter().fold(0f32, |acc, v| acc.max(v.abs())).max(1e-6);
            let diff = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "{name:16} diff {diff:.5} of range {range:.4} ({:.2e})",
                diff / range
            );
            if gated && (diff / range > tolerance || !diff.is_finite()) {
                bad.push(name);
            }
        }
        let Some((_, biased)) = host
            .iter()
            .find(|(n, _)| *n == format!("ffn_moe_probs_biased-{layer}"))
        else {
            continue;
        };
        let ids = joined(&format!("moe_ids-{layer}"));
        let mut flips = 0;
        for t in split..nt {
            let row = &biased[t * ne..(t + 1) * ne];
            let order = top_k(row, nu + 1);
            let margin = row[order[nu - 1]] - row[order[nu]];
            let mut host_set: Vec<usize> = order[..nu].to_vec();
            host_set.sort_unstable();
            let at = (t - split) * nu;
            let mut dev_set: Vec<usize> = ids[at..at + nu].iter().map(|&v| v as usize).collect();
            dev_set.sort_unstable();
            if host_set != dev_set {
                flips += 1;
                routing.push((format!("layer {layer} token {t}"), margin));
            }
        }
        eprintln!(
            "moe-{layer:<11} routed sets differ at {flips} of {} tokens",
            nt - split
        );
    }
    let _ = (hc, h);
    (bad, routing)
}

fn stage_layers(default: usize) -> usize {
    std::env::var("RAON_STAGE_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The per-layer stages of the first layers (all 25 with `RAON_STAGE_LAYERS=25`), for locating
/// a defect and for reading the noise floor.
#[test]
fn the_device_stages_match_the_host_reference_for_the_first_layers() {
    let Some(path) = model_path() else { return };
    if !device_present("the_device_stages_match_the_host_reference_for_the_first_layers") {
        return;
    }
    let (bad, routing) = compare_stages(&path, &PROMPT[..4], 0, stage_layers(4), STAGE_TOLERANCE);
    eprintln!("routing disagreements: {routing:?}");
    assert!(
        bad.is_empty(),
        "stages differ beyond {STAGE_TOLERANCE} of range: {bad:?}"
    );
}

/// A prompt past one chunk: the delta-rule state and the convolution history carry across the
/// chunk boundary at 256 tokens.
#[test]
fn the_device_stages_match_across_a_chunk_boundary() {
    let Some(path) = model_path() else { return };
    if !device_present("the_device_stages_match_across_a_chunk_boundary") {
        return;
    }
    let tokens = synthetic_tokens(300);
    let (bad, routing) = compare_stages(&path, &tokens, 0, stage_layers(4), STAGE_TOLERANCE);
    eprintln!("routing disagreements: {routing:?}");
    assert!(
        bad.is_empty(),
        "stages differ beyond {STAGE_TOLERANCE} of range: {bad:?}"
    );
}

/// A decode step over more than 1,024 keys: the latent attention splits the keys and reduces the
/// partial softmaxes.
#[test]
fn the_device_stages_match_with_split_key_attention() {
    let Some(path) = model_path() else { return };
    if !device_present("the_device_stages_match_with_split_key_attention") {
        return;
    }
    let tokens = synthetic_tokens(1101);
    let (bad, routing) = compare_stages(&path, &tokens, 1100, stage_layers(4), STAGE_TOLERANCE);
    eprintln!("routing disagreements: {routing:?}");
    assert!(
        bad.is_empty(),
        "stages differ beyond {STAGE_TOLERANCE} of range: {bad:?}"
    );
}

/// Final logits recorded by an independent implementation over the frozen 42-token fixture
/// prompt, separate from the host reference. The device's top three must be the same ids, within the quantization noise.
#[test]
fn the_device_top_logits_match_the_recorded_logits_on_the_fixture_prompt() {
    let Some(path) = model_path() else { return };
    if !device_present("the_device_top_logits_match_the_recorded_logits_on_the_fixture_prompt") {
        return;
    }
    const FIXTURE: [u32; 42] = [
        166863, 54935, 7821, 44, 11456, 189096, 257706, 142907, 189286, 9305, 39154, 166294,
        107408, 51684, 46, 224037, 18624, 166295, 11234, 166471, 167757, 142907, 22233, 5407, 5066,
        46, 254733, 27335, 44, 4565, 46, 124319, 156431, 44, 235664, 73629, 177858, 9678, 444,
        2452, 44, 3248,
    ];
    const RECORDED_TOP3: [(usize, f32); 3] = [(44, 21.71462), (58, 19.871094), (369, 19.577736)];
    let mut device = Raon1Device::load(&path, Some(4096), &mut |_| {}).expect("device load");
    device.reset().expect("reset");
    let rows = device
        .feed(&FIXTURE, 0, HeadRows::Last)
        .expect("fixture feed");
    let row = &rows[0];
    let top = top_k(row, 3);
    for (rank, &(id, want)) in RECORDED_TOP3.iter().enumerate() {
        eprintln!(
            "rank {rank}: recorded {id} {want:.4} | device {} {:.4}",
            top[rank], row[top[rank]]
        );
        assert_eq!(top[rank], id, "rank {rank} id");
        assert!(
            (row[id] - want).abs() < 0.5,
            "logit of {id}: {} vs {want}",
            row[id]
        );
    }
}

/// Prefill and decode rates at the prompt lengths of `RAON_BENCH` (comma separated tokens),
/// on the context the free device memory admits. Prints one line per length.
#[test]
fn the_device_reports_prefill_and_decode_rates() {
    let Some(path) = model_path() else { return };
    let Ok(lengths) = std::env::var("RAON_BENCH") else {
        eprintln!("skipped: RAON_BENCH is unset");
        return;
    };
    if !device_present("the_device_reports_prefill_and_decode_rates") {
        return;
    }
    let mut device = Raon1Device::load(&path, None, &mut |cells| {
        eprintln!("planned context {cells}");
    })
    .expect("device load");
    for length in lengths
        .split(',')
        .map(|v| v.trim().parse::<usize>().expect("length"))
    {
        let tokens: Vec<u32> = (0..length)
            .map(|i| 1000 + (i as u32 * 7919) % 200_000)
            .collect();
        device.reset().expect("reset");
        let start = std::time::Instant::now();
        device.feed(&tokens, 0, HeadRows::Last).expect("prefill");
        let prefill = start.elapsed().as_secs_f64();
        let steps = 32;
        let start = std::time::Instant::now();
        for step in 0..steps {
            device
                .feed(&[1234 + step as u32], length + step, HeadRows::Last)
                .expect("decode");
        }
        let decode = start.elapsed().as_secs_f64();
        eprintln!(
            "prompt {length}: prefill {:.1} tok/s ({prefill:.2} s) | decode {:.1} tok/s",
            length as f64 / prefill,
            steps as f64 / decode
        );
    }
}
