//! The KV-cache-form quality leg: teacher-forced scoring of one cache form (fp16 or int8) over
//! long windows of a fixed token stream, on the GGUF weights and the chunked prefill path the
//! default start serves.
//!
//! One form per process, driven by the environment (the served API cannot return per-position
//! log-probabilities). Every window prefills the tokens before its scored region through the
//! chunked path, then feeds each true token one at a time and reads the full logits row of the
//! next position. A row is narrowed to the sampler vocabulary and scored in f64: the true-token
//! log-probability, the greedy argmax and the top rows of the distribution go to a per-window
//! sidecar. When a reference run's sidecars are named, the leg also writes the divergence of the
//! reference distribution from its own over the reference's stored top rows (no bound on the
//! whole-vocabulary KL, the terms outside being signed) and the reference's tail mass beside it.
//!
//! The pairing, the bootstrap and the verdict run on the host from the sidecars of two runs.

use super::{
    common_tests::{dims, inventory},
    dims::{KvMode, StepOptions},
    m3_run::{CHUNK_ROWS, SAMPLER_VOCAB, err, rules, stamp},
    prefill_driver::{FIXED_ADMISSION_MARGIN, PrefillBudget},
    resources::StepResources,
    step::{ChunkVerdict, QwenResidentStep},
};
use crate::{
    EngError,
    family::{
        gguf_model::open_qwen3_6_35b_a3b,
        project::cuda::{CudaProjector, LoadOptions, Reserve, WeightLayout},
        qwen::qwen36::resident::m3_rows,
    },
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Rows kept per position in the sidecar.
const TOP_ROWS: usize = 256;
/// Bytes of one position record: true id, log-probability of the true id, argmax id, the top
/// ids and their log-probabilities, the divergence over the reference's top rows, and the
/// reference's tail mass.
const RECORD_BYTES: usize = 4 + 4 + 4 + TOP_ROWS * 8 + 4 + 4;
const MAGIC: &[u8; 4] = b"KVQ1";

fn env(name: &str) -> Result<String, EngError> {
    std::env::var(name).map_err(|_| err(&format!("{name} is not set")))
}

/// A window of the id stream: where its prefill starts, where scoring starts, and where it ends.
type Window = (usize, usize, usize);

fn layout_of(path: &Path) -> Result<(usize, Vec<Window>), EngError> {
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| err(&format!("layout: {e}")))?)
            .map_err(|e| err(&format!("layout: {e}")))?;
    let context = value["context_tokens"]
        .as_u64()
        .ok_or_else(|| err("layout: context_tokens"))? as usize;
    let windows = value["windows"]
        .as_array()
        .ok_or_else(|| err("layout: windows"))?
        .iter()
        .map(|w| {
            let at = |i: usize| w[i].as_u64().map(|v| v as usize);
            match (at(0), at(1), at(2)) {
                (Some(lo), Some(score_lo), Some(hi)) if lo < score_lo && score_lo < hi => {
                    Ok((lo, score_lo, hi))
                }
                _ => Err(err("layout: a window is not [lo, score_lo, hi]")),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((context, windows))
}

fn ids_of(path: &Path, sha: &str) -> Result<Vec<u32>, EngError> {
    let bytes = std::fs::read(path).map_err(|e| err(&format!("ids: {e}")))?;
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    if digest != sha {
        return Err(err(&format!("the id stream is {digest}, not {sha}")));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

/// The `TOP_ROWS` largest entries of `row`, as ids (ties to the lower id) and their
/// log-probabilities under `lse`.
fn top_rows(row: &[f32], lse: f64) -> (Vec<u32>, Vec<f32>) {
    let mut order: Vec<u32> = (0..row.len() as u32).collect();
    let by_value = |a: &u32, b: &u32| {
        row[*b as usize]
            .total_cmp(&row[*a as usize])
            .then_with(|| a.cmp(b))
    };
    order.select_nth_unstable_by(TOP_ROWS, by_value);
    order.truncate(TOP_ROWS);
    order.sort_by(by_value);
    let logprobs = order
        .iter()
        .map(|&i| (f64::from(row[i as usize]) - lse) as f32)
        .collect();
    (order, logprobs)
}

struct Reference {
    ids: Vec<u32>,
    logprobs: Vec<f32>,
}

/// The reference run's top rows at scored position `index` of window `w`.
fn reference_record(bytes: &[u8], index: usize) -> Reference {
    let at = 16 + index * RECORD_BYTES + 12;
    let ids = (0..TOP_ROWS)
        .map(|i| u32::from_le_bytes(bytes[at + 4 * i..at + 4 * i + 4].try_into().unwrap()))
        .collect();
    let base = at + 4 * TOP_ROWS;
    let logprobs = (0..TOP_ROWS)
        .map(|i| f32::from_le_bytes(bytes[base + 4 * i..base + 4 * i + 4].try_into().unwrap()))
        .collect();
    Reference { ids, logprobs }
}

/// One window on a fresh step: the prefill, then one forced step per scored position. Returns
/// the window's sidecar bytes.
fn run_window(
    step: &mut QwenResidentStep<'_>,
    kv: KvMode,
    free_after_load: usize,
    ids: &[u32],
    (lo, score_lo, hi): Window,
    reference: Option<&[u8]>,
    w: usize,
) -> Result<Vec<u8>, EngError> {
    let t0 = std::time::Instant::now();
    println!("kvq window {w} start: prefill {} ids", score_lo - lo);
    if kv == KvMode::Int8G32 {
        step.set_kv8_chunked_prefill(true)?;
    }
    step.set_m_sess_cap(CHUNK_ROWS);
    let verdict = step.admit_prefill(
        PrefillBudget {
            device_free: free_after_load.saturating_sub(FIXED_ADMISSION_MARGIN),
            extras: Vec::new(),
        },
        true,
        true,
        false,
    )?;
    if !matches!(verdict, ChunkVerdict::Ready { .. }) {
        return Err(err(&format!(
            "window {w}: the chunked arena verdict {verdict:?} is not Ready"
        )));
    }
    for piece in ids[lo..score_lo - 1].chunks(CHUNK_ROWS as usize) {
        step.feed_rows_chunked(piece, CHUNK_ROWS as usize)?;
    }
    println!("kvq window {w} prefill done [{}]", stamp(&t0));
    let mut out = Vec::with_capacity(16 + (hi - score_lo) * RECORD_BYTES);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(w as u32).to_le_bytes());
    out.extend_from_slice(&((hi - score_lo) as u32).to_le_bytes());
    out.extend_from_slice(&(TOP_ROWS as u32).to_le_bytes());
    for p in score_lo..hi {
        step.run_token(ids[p - 1], true)?;
        let full = step
            .row_bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .take(SAMPLER_VOCAB)
            .map(|c| f32::from_le_bytes(*c))
            .collect::<Vec<f32>>();
        if full.iter().any(|v| !v.is_finite()) {
            return Err(err(&format!("window {w} position {p}: a non-finite logit")));
        }
        let lse = m3_rows::log_sum_exp(&full);
        let true_logprob = m3_rows::logprob_true(&full, ids[p]).map_err(err)? as f32;
        let argmax = m3_rows::argmax_low_id(&full).map_err(err)?;
        let (top_ids, top_logprobs) = top_rows(&full, lse);
        let (divergence, tail) = match reference {
            Some(bytes) => {
                let r = reference_record(bytes, p - score_lo);
                let (mut kl, mut mass) = (0f64, 0f64);
                for (id, lp) in r.ids.iter().zip(&r.logprobs) {
                    let p_ref = f64::from(*lp).exp();
                    let q_lp = f64::from(full[*id as usize]) - lse;
                    kl += p_ref * (f64::from(*lp) - q_lp);
                    mass += p_ref;
                }
                (kl as f32, (1.0 - mass) as f32)
            }
            None => (0.0, 0.0),
        };
        out.extend_from_slice(&ids[p].to_le_bytes());
        out.extend_from_slice(&true_logprob.to_le_bytes());
        out.extend_from_slice(&argmax.to_le_bytes());
        for id in &top_ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
        for lp in &top_logprobs {
            out.extend_from_slice(&lp.to_le_bytes());
        }
        out.extend_from_slice(&divergence.to_le_bytes());
        out.extend_from_slice(&tail.to_le_bytes());
        if (p - score_lo) % 256 == 0 {
            println!(
                "kvq window {w} scored {}/{} [{}]",
                p - score_lo,
                hi - score_lo,
                stamp(&t0)
            );
        }
    }
    println!("kvq window {w} done [{}]", stamp(&t0));
    Ok(out)
}

/// Scores the windows `KVQ_WINDOWS` (`lo:hi`) with the cache form `KVQ_KV` (`f16` or `q8`).
#[test]
#[ignore = "the 3090 under fifi-gpu-run: KVQ_KV, KVQ_GGUF_DIR, KVQ_IDS, KVQ_IDS_SHA, KVQ_LAYOUT, KVQ_WINDOWS, KVQ_OUT, optional KVQ_REF"]
fn kv_quality_leg() {
    let kv = match env("KVQ_KV").expect("KVQ_KV").as_str() {
        "f16" => KvMode::Fp16,
        "q8" => KvMode::Int8G32,
        other => panic!("KVQ_KV is f16|q8, not {other}"),
    };
    let ids = ids_of(
        Path::new(&env("KVQ_IDS").expect("KVQ_IDS")),
        &env("KVQ_IDS_SHA").expect("KVQ_IDS_SHA"),
    )
    .expect("ids");
    let (context, windows) =
        layout_of(Path::new(&env("KVQ_LAYOUT").expect("KVQ_LAYOUT"))).expect("layout");
    let bounds: Vec<usize> = env("KVQ_WINDOWS")
        .expect("KVQ_WINDOWS")
        .split(':')
        .map(|x| x.parse().expect("window bound"))
        .collect();
    assert!(bounds.len() == 2 && bounds[0] <= bounds[1] && bounds[1] <= windows.len());
    let out_dir = PathBuf::from(env("KVQ_OUT").expect("KVQ_OUT"));
    std::fs::create_dir_all(&out_dir).expect("out dir");
    let reference_dir = std::env::var("KVQ_REF").ok().map(PathBuf::from);
    for &(_, _, hi) in &windows {
        assert!(hi <= ids.len(), "a window ends past the id stream");
    }

    let projector = CudaProjector::new().expect("projector");
    let model = open_qwen3_6_35b_a3b(
        &PathBuf::from(env("KVQ_GGUF_DIR").expect("KVQ_GGUF_DIR"))
            .join("Qwen3.6-35B-A3B-Q4_K_M.gguf"),
        None,
    )
    .expect("model");
    let weights = projector
        .load_model(
            &model,
            LoadOptions {
                reserve: Reserve {
                    kv_cache: 0,
                    work: 0,
                    headroom: 0,
                },
                drop_page_cache: None,
                layout: WeightLayout::Native,
            },
        )
        .expect("load")
        .device;
    let layout_kind = weights.layout().expect("layout kind");
    let free_after_load = projector.memory_info().expect("memory").free;
    println!("kvq kv={:?} free after load {free_after_load}", kv);
    let options = StepOptions {
        kv_mode: kv,
        context_tokens: context + CHUNK_ROWS as usize,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: layout_kind,
    };
    let res =
        StepResources::new(projector.context(), dims(), &inventory(), &options).expect("resources");
    for (w, window) in windows.iter().enumerate().take(bounds[1]).skip(bounds[0]) {
        let path = out_dir.join(format!("w{w:02}.kvq"));
        let expected = 16 + (window.2 - window.1) * RECORD_BYTES;
        if std::fs::metadata(&path).is_ok_and(|m| m.len() as usize == expected) {
            println!("kvq window {w}: sidecar present, skipping");
            continue;
        }
        let reference = reference_dir
            .as_ref()
            .map(|dir| std::fs::read(dir.join(format!("w{w:02}.kvq"))).expect("reference sidecar"));
        let mut step =
            QwenResidentStep::new(projector.context(), &res, &weights, &options, rules())
                .expect("step");
        let bytes = run_window(
            &mut step,
            kv,
            free_after_load,
            &ids,
            *window,
            reference.as_deref(),
            w,
        )
        .unwrap_or_else(|e| panic!("window {w}: {e:?}"));
        std::fs::write(&path, bytes).expect("sidecar write");
    }
}
