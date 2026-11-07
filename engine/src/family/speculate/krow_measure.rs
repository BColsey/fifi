//! The host arithmetic of the K-row measurement tests (`krow_alpha_device_tests`,
//! `krow_step_device_tests`): everything that needs no device and can be tested on the host.
//!
//! The measurements replace the estimated per-row costs of the K-row verifier design with numbers
//! from the real pack: the marginal cost of one more row through the rows kernel (alpha), the
//! phases of one captured step, the expert union of K consecutive tokens, and the routed-expert
//! chain of K rows. This file holds the definitions those numbers share, so a device run and a
//! host test read the same formulas:
//!
//! * alpha(R) = (t(R) - t(1)) / ((R - 1) * t(1)), where t(1) is the time of one single-row launch
//!   and t(R) the time of one launch at R rows. The time-weighted alpha of a set of roles weights
//!   each role by its own t(1) of one whole token (a sweep over every layer that has the role), so
//!   it is (sum of t(R) - t(1)) / ((R - 1) * sum of t(1)).
//! * the expert union of K tokens in one layer is the number of distinct ids among their K * top_k
//!   picks; windows slide one token at a time.

/// The row counts of the alpha measurement. 1 is the single-row (wide1) reference; 3 and 6 run the
/// rows kernel of the next power of two (4 and 8) with fewer rows.
pub(crate) const ROWS: [usize; 6] = [1, 2, 3, 4, 6, 8];
/// The window sizes of the expert union.
pub(crate) const UNION_WINDOWS: [usize; 7] = [2, 3, 4, 5, 6, 7, 8];

use serde_json::{Map, Value, json};

use super::window_cost::{median, summary};
use crate::family::AttentionKind;

/// Variables that may stay set in a measurement process: the pack, and the device requirement the
/// runner sets. Everything else under a prefix of [`REFUSED_PREFIXES`] changes which kernel or
/// which numerics the step runs, or what it reports, and makes the test refuse to start.
pub(crate) const ALLOWED_VARS: [&str; 2] =
    ["FIFI_EXL3_CHECKPOINT", "INFERENCE_CUDA_REQUIRE_DEVICE"];
/// Prefixes of the engine's kernel, numerics and diagnostic switches (read from the environment
/// reads of `qwen36/cuda`, `kernels/cuda` and `project/cuda`: GDN input-projection fusion, sparse
/// attention, the KV8 attention variants, the device shortlist, the Q36 ablations, the per-op
/// sync, the single-row, gather, dp4a, indexed-pair and shared-in-routed EXL3 switches, the graph
/// bench phases).
pub(crate) const REFUSED_PREFIXES: [&str; 6] = [
    "FIFI_EXL3_",
    "INFERENCE_CUDA_",
    "INFERENCE_QWEN36_",
    "INFERENCE_Q36_",
    "Q36_",
    "FIFI_GRAPH_BENCH_",
];
/// Single variables outside those prefixes that change a step.
pub(crate) const REFUSED_EXACT: [&str; 1] = ["FIFI_DELTA_TRACE"];

/// The names among `names` that a measurement refuses to run with.
pub(crate) fn refused_switches(names: impl IntoIterator<Item = String>) -> Vec<String> {
    names
        .into_iter()
        .filter(|name| {
            !ALLOWED_VARS.contains(&name.as_str())
                && (REFUSED_PREFIXES.iter().any(|p| name.starts_with(p))
                    || REFUSED_EXACT.contains(&name.as_str()))
        })
        .collect()
}

/// The whole startup check of a measurement process: no refused variable, and the shared expert
/// not inside the routed launches (`shared` is the process-wide reading of
/// `FIFI_EXL3_SHARED_IN_ROUTED`: `Ok(false)` is off, `Ok(true)` on, `Err` a bad value). An `Err`
/// names every reason.
pub(crate) fn check_default_path(
    names: impl IntoIterator<Item = String>,
    shared: Result<bool, String>,
) -> Result<(), String> {
    let mut reasons = Vec::new();
    let refused = refused_switches(names);
    if !refused.is_empty() {
        reasons.push(format!(
            "switches set that change what the step runs: {refused:?}"
        ));
    }
    match shared {
        Ok(false) => {}
        Ok(true) => reasons.push("FIFI_EXL3_SHARED_IN_ROUTED is on".to_owned()),
        Err(error) => reasons.push(format!("FIFI_EXL3_SHARED_IN_ROUTED: {error}")),
    }
    if reasons.is_empty() {
        Ok(())
    } else {
        Err(reasons.join("; "))
    }
}

/// The receipt code of a launch of `rows` rows run as separate single-row launches
/// (`SEP_CODE_OFFSET + rows`), against the plain `rows` of one batched launch.
pub(crate) const SEP_CODE_OFFSET: usize = 16;

/// What the union record does and does not cover (S6 of the review of the first window).
pub(crate) const UNION_CAVEAT: &str = "true-continuation windows only: the K rows are the next K tokens of the real text, i.e. every draft accepted; a window holding rejected drafts has other tokens in its later rows and a different union; depth 1K to 4K (a 1024-row prefill and 3000 decoded tokens), not 8K or 32K; picks are read on the EAGER body (capture off), assuming the captured step routes the same (D1)";

/// The size in bytes of the device-to-device marker copy that the receipt pass queues before the
/// launch of op `op` at `rows` rows, so that a profile can tell which kernels belong to which
/// (op, rows): 4096 * (op + 1) + 16 * rows.
pub(crate) fn receipt_marker_bytes(op: usize, rows: usize) -> usize {
    4096 * (op + 1) + 16 * rows
}

/// The `(op, rows)` a marker size names, `None` for any other size.
pub(crate) fn decode_receipt_marker(bytes: usize) -> Option<(usize, usize)> {
    let (op, rest) = (bytes / 4096, bytes % 4096);
    (op >= 1 && rest.is_multiple_of(16) && (1..=64).contains(&(rest / 16)))
        .then(|| (op - 1, rest / 16))
}

/// alpha(R) = (t(R) - t(1)) / ((R - 1) * t(1)); `None` for R = 1 or a non-positive t(1).
pub(crate) fn alpha(t1: f64, t_rows: f64, rows: usize) -> Option<f64> {
    if rows < 2 || !t1.is_finite() || t1 <= 0.0 || !t_rows.is_finite() {
        return None;
    }
    Some((t_rows - t1) / ((rows - 1) as f64 * t1))
}

/// The time-weighted alpha of `roles` (each `(t(1), t(R))` of a whole-token sweep): the sum of the
/// marginal times over (R - 1) times the sum of the solo times.
pub(crate) fn weighted_alpha(roles: &[(f64, f64)], rows: usize) -> Option<f64> {
    let solo: f64 = roles.iter().map(|r| r.0).sum();
    let marginal: f64 = roles.iter().map(|r| r.1 - r.0).sum();
    if rows < 2 || !solo.is_finite() || solo <= 0.0 || !marginal.is_finite() {
        return None;
    }
    Some(marginal / ((rows - 1) as f64 * solo))
}

/// A float output that differs between the rows kernel and the single-row launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Mismatch {
    pub(crate) row: usize,
    pub(crate) col: usize,
    /// The element index in the flat `[rows][row_values]` output.
    pub(crate) index: usize,
    pub(crate) left_bits: u32,
    pub(crate) right_bits: u32,
}

/// The first differing 32-bit word of two equally long little-endian f32 buffers laid out as
/// `[rows][row_values]`, compared by bits (a signed zero or a NaN payload counts). `None` when
/// the buffers are equal; a length that is not a whole number of words or differs between the two
/// is reported as a mismatch at index `usize::MAX`.
pub(crate) fn first_word_difference(
    left: &[u8],
    right: &[u8],
    row_values: usize,
) -> Option<Mismatch> {
    if left.len() != right.len() || !left.len().is_multiple_of(4) || row_values == 0 {
        return Some(Mismatch {
            row: usize::MAX,
            col: usize::MAX,
            index: usize::MAX,
            left_bits: 0,
            right_bits: 0,
        });
    }
    let word = |bytes: &[u8], at: usize| {
        u32::from_le_bytes([
            bytes[4 * at],
            bytes[4 * at + 1],
            bytes[4 * at + 2],
            bytes[4 * at + 3],
        ])
    };
    (0..left.len() / 4).find_map(|index| {
        let (a, b) = (word(left, index), word(right, index));
        (a != b).then_some(Mismatch {
            row: index / row_values,
            col: index % row_values,
            index,
            left_bits: a,
            right_bits: b,
        })
    })
}

/// The sizes of the distinct experts among `k` consecutive tokens of one layer, for every start.
/// `picks` holds the `top_k` ids of each token in order. A negative id (a faulty router row) is a
/// refusal: the union of a window with an invalid pick says nothing.
pub(crate) fn union_sizes(picks: &[Vec<i32>], k: usize) -> Result<Vec<usize>, String> {
    if k == 0 {
        return Err("a window of zero tokens".into());
    }
    if let Some(bad) = picks.iter().flatten().find(|id| **id < 0) {
        return Err(format!("the router picked the invalid id {bad}"));
    }
    Ok(picks
        .windows(k)
        .map(|window| {
            let mut ids: Vec<i32> = window.iter().flatten().copied().collect();
            ids.sort_unstable();
            ids.dedup();
            ids.len()
        })
        .collect())
}

/// The p-th percentile (0..=100) by the nearest-rank rule on a copy of `values`; `None` if empty.
pub(crate) fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=100.0).contains(&p) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    Some(sorted[rank.min(sorted.len()) - 1])
}

pub(crate) fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// `items` rotated left by `round` places: each round starts the sequence at a different item, so
/// drift of the clocks or the temperature does not always hit the same measurement first.
pub(crate) fn rotated<T: Clone>(items: &[T], round: usize) -> Vec<T> {
    if items.is_empty() {
        return Vec::new();
    }
    let shift = round % items.len();
    items[shift..]
        .iter()
        .chain(&items[..shift])
        .cloned()
        .collect()
}

/// A small deterministic generator of roughly normal values (the sum of four uniforms, centred and
/// scaled to unit variance), seeded per measurement so a rerun feeds the same inputs.
pub(crate) struct Inputs(u64);

impl Inputs {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u32(&mut self) -> u32 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }

    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / 4_294_967_296.0
    }

    pub(crate) fn normal(&mut self) -> f32 {
        // Four uniforms on [0, 1) have mean 2 and variance 1/3.
        let sum: f32 = (0..4).map(|_| self.unit()).sum();
        (sum - 2.0) * 3.0_f32.sqrt()
    }

    pub(crate) fn values(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.normal()).collect()
    }
}

pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ---- the records of the phase and union tests (host logic, host-tested) ----

pub(crate) struct Reading {
    pub(crate) wall_ms: f64,
    pub(crate) whole_ms: f64,
    pub(crate) embed_ms: f64,
    pub(crate) tail_ms: f64,
    pub(crate) groups: Vec<f64>,
}

fn medians(rows: &[Vec<f64>]) -> Vec<f64> {
    let width = rows.first().map_or(0, Vec::len);
    (0..width)
        .map(|at| median(&rows.iter().map(|row| row[at]).collect::<Vec<_>>()).unwrap_or(f64::NAN))
        .collect()
}

pub(crate) fn depth_json(
    depth: usize,
    readings: &[Reading],
    plan: &[(usize, usize, AttentionKind)],
) -> Value {
    let col = |f: &dyn Fn(&Reading) -> f64| -> Vec<f64> { readings.iter().map(f).collect() };
    let kind_total = |reading: &Reading, kind: AttentionKind| -> f64 {
        reading
            .groups
            .iter()
            .zip(plan)
            .filter(|(_, (_, _, k))| *k == kind)
            .map(|(ms, (_, size, _))| ms * *size as f64)
            .sum()
    };
    let gdn = col(&|r| kind_total(r, AttentionKind::DeltaNet));
    let full = col(&|r| kind_total(r, AttentionKind::Full));
    let embed = col(&|r| r.embed_ms);
    let tail = col(&|r| r.tail_ms);
    let whole = col(&|r| r.whole_ms);
    let sum_over_whole = col(&|r| {
        (r.embed_ms
            + r.tail_ms
            + kind_total(r, AttentionKind::DeltaNet)
            + kind_total(r, AttentionKind::Full))
            / r.whole_ms
    });
    let per_group = medians(
        &readings
            .iter()
            .map(|r| r.groups.clone())
            .collect::<Vec<_>>(),
    );
    let groups: Vec<Value> = plan
        .iter()
        .zip(&per_group)
        .map(|((layer, size, kind), ms)| {
            json!({"layer": layer, "run_size": size, "kind": format!("{kind:?}"), "median_ms": ms})
        })
        .collect();
    let m = |v: &[f64]| median(v).unwrap_or(f64::NAN);
    json!({
        "depth_start": depth,
        "tokens_timed": readings.len(),
        "step_wall_ms": summary(&col(&|r| r.wall_ms)),
        "step_gpu_event_ms": summary(&whole),
        "idle_gap_ms_median": m(&col(&|r| r.wall_ms - r.whole_ms)),
        "embed_ms": summary(&embed),
        "tail_ms (final norm, head, argmax, advance)": summary(&tail),
        "gdn_layers_ms_scaled_to_30 (mixer + moe)": summary(&gdn),
        "full_attention_layers_ms_scaled_to_10 (mixer + moe)": summary(&full),
        "marked_sum_over_whole_token": summary(&sum_over_whole),
        "median_shares_of_gpu_event": {
            "embed": m(&embed) / m(&whole), "tail": m(&tail) / m(&whole),
            "gdn_layers": m(&gdn) / m(&whole), "full_attention_layers": m(&full) / m(&whole),
        },
        "groups": groups,
        "raw_step_wall_ms": col(&|r| r.wall_ms),
        "raw_gpu_event_ms": whole,
    })
}

pub(crate) fn union_json(picks: &[Vec<Vec<i32>>]) -> Value {
    let mut per_k = Map::new();
    for k in std::iter::once(1).chain(UNION_WINDOWS) {
        let mut all: Vec<f64> = Vec::new();
        let mut layers = Vec::new();
        for (layer, tokens) in picks.iter().enumerate() {
            let sizes: Vec<f64> = union_sizes(tokens, k)
                .unwrap_or_else(|e| panic!("layer {layer} K {k}: {e}"))
                .into_iter()
                .map(|n| n as f64)
                .collect();
            layers.push(json!({
                "layer": layer,
                "windows": sizes.len(),
                "mean": mean(&sizes),
                "p90": percentile(&sizes, 90.0),
            }));
            all.extend(sizes);
        }
        let top_k = picks[0][0].len() as f64;
        per_k.insert(
            k.to_string(),
            json!({
                "windows_total": all.len(),
                "mean": mean(&all),
                "p90": percentile(&all, 90.0),
                "mean_per_layer_p90": mean(
                    &layers.iter().filter_map(|l| l["p90"].as_f64()).collect::<Vec<_>>()
                ),
                "mean_over_k_times_top_k": mean(&all).map(|m| m / (k as f64 * top_k)),
                "per_layer": layers,
            }),
        );
    }
    Value::Object(per_k)
}
