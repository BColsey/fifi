//! Measurements of the real captured step and of real routing for a K-row verify window:
//! MEASUREMENT tests that have run in exclusive GPU windows (the records are kept beside the run
//! plan). They gate nothing about speed.
//!
//! Two tests, one per process, each on the GPU held exclusively and each loading the whole
//! 4.00 bpw pack (about 17 GiB, so a lease of about 21 GiB with the KV cache and the arena):
//!
//! * `krow_phase_record`: the captured single-row step (wide1, fp16 KV, capture on) at depths 8,192
//!   and 32,768 (`FIFI_KROW_DEPTHS`). The step records the in-graph section marks that already
//!   exist (`section_timing`): the embedding, the tail (final norm, head, argmax, advance) and the
//!   mixer-plus-MoE pair of one layer of each run of identical layer kinds. Qwen3.6 alternates
//!   three GDN layers with one full-attention layer, so the marks give the GDN layer and every
//!   attention layer separately, scaled to the whole token. The finer split (routed experts, the
//!   shared expert, the dense projections, the norms, conv and delta) is the kernel-family
//!   breakdown of an `nsys` profile of this same test (`tools/reduce_families.py`, run by the
//!   runner script with `FIFI_KROW_DEPTHS=8192`); the marks stay in the graph for both, and the
//!   record states the observer effect (the marks add event nodes; the profiler adds more).
//! * `krow_union_record`: real routing. It prefills a prefix, then decodes real tokens of the text
//!   (`FIFI_SPEC_TEXT`, teacher forced) one at a time on the EAGER body with a test-only tap that
//!   reads the routed expert ids of every layer after its MoE block (the eager body is the one
//!   that can read; the router kernel and its inputs are the same as in the captured step, which
//!   is assumed to route the same; that is not measured here). From the ids it reports, per layer and over all layers, the mean and
//!   the 90th percentile of the number of distinct experts in windows of K = 2 to 8 consecutive
//!   tokens (sliding by one token). Then, on the same real ids, it times the routed experts of K
//!   rows run as K single-row id-indexed chains back to back in one stream (the per-row
//!   launch pattern), against one chain, for K = 1, 2, 3, 4, 8, and against a control whose rows pick disjoint
//!   experts (no weight overlap between the rows).
//!
//! The record carries the identity of the run (commit, binary, pack, KV mode, capture, the
//! effective single-row kernel and the shared-expert mode of the step); both tests refuse
//! to start if any variable under the refused prefixes of `krow_measure` (`FIFI_EXL3_`,
//! `INFERENCE_CUDA_`, `INFERENCE_QWEN36_`, `INFERENCE_Q36_`, `Q36_`, `FIFI_GRAPH_BENCH_`) or
//! `FIFI_DELTA_TRACE` is set, except `FIFI_EXL3_CHECKPOINT` and `INFERENCE_CUDA_REQUIRE_DEVICE`, or if
//! `FIFI_EXL3_SHARED_IN_ROUTED=1` is set (any set value, even `0`, is refused by the `FIFI_EXL3_` name rule). The engine now loads the shared expert inside the routed launches by
//! default; these tests measure the separate shared block that the K-row path runs, so each step is
//! forced to it with `set_shared_in_routed(false)` before its first token, and the record names the
//! step's reported mode.
//!
//! Environment: `FIFI_EXL3_CHECKPOINT`, `FIFI_SPEC_TEXT`, `FIFI_SPEC_TOKENIZER`, `FIFI_KROW_COMMIT`
//! (required), `FIFI_KROW_OUT` (the JSON path), `FIFI_KROW_PACK_MANIFEST_SHA256` and
//! `FIFI_KROW_IDS_SHA256` (pins), `FIFI_KROW_DEPTHS` (8192,32768), `FIFI_KROW_TOKENS` (60 timed
//! tokens per depth), `FIFI_KROW_WARM` (6), `FIFI_KROW_PREFIX` (1024 rows of the union prefill),
//! `FIFI_KROW_UNION_TOKENS` (3000 decoded tokens), `FIFI_KROW_CHAIN_LAYERS` (3,15,27,39),
//! `FIFI_KROW_CHAIN_WINDOWS` (24 windows per layer), `FIFI_KROW_PICKS_OUT` (a path: the ids of
//! every token and layer, for a rerun of the analysis on the host), `FIFI_KROW_DEADLINE_S` (780).

use std::{fs, time::Instant};

use serde_json::{Value, json};

use super::{
    common_tests::{dims, inventory},
    d1_window_device_tests::{load_pack, text_stream},
    dims::KvMode,
    exl3_moe::{Exl3MoeBufs, Exl3MoeWeights, Scratch, routed_on_device},
    krow_alpha_device_tests::{Flush, Stopwatch, capture_graph, emit, identity, refuse_switches},
    layer_table::LayerTable,
    moe::{Launch, MoeDims, RouteBufs},
    resources::STEP_ROWS,
    window_bench_device_tests::bench_step,
};
use crate::family::speculate::krow_measure::{
    Inputs, Reading, UNION_CAVEAT, depth_json, f32_bytes, mean, rotated, shared_in_routed_mode,
    union_json,
};
use crate::{
    family::{
        AttentionKind,
        session::TokenStepper,
        speculate::window_cost::{ids_sha256, median, sha256_file, summary},
    },
    kernels::{
        cuda::{DeviceBuffer, DeviceRange, device_present},
        single_row::require_kernel_ran,
    },
};

const F32: usize = 4;
/// Positions past the deepest depth that the cache holds.
const CONTEXT_SLACK: usize = 1_536;

fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().map_or(default, |text| {
        text.parse().unwrap_or_else(|_| panic!("{name}: {text:?}"))
    })
}

fn env_list(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .map(|part| {
            part.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name}: {part:?}"))
        })
        .collect()
}

/// The token-id part of the identity: the text, the tokenizer and the ids used.
fn token_identity(ids: &[u32]) -> Value {
    let text = std::env::var("FIFI_SPEC_TEXT").expect("FIFI_SPEC_TEXT");
    let tokenizer = std::env::var("FIFI_SPEC_TOKENIZER").expect("FIFI_SPEC_TOKENIZER");
    let sha = ids_sha256(ids);
    if let Ok(expected) = std::env::var("FIFI_KROW_IDS_SHA256") {
        assert_eq!(
            sha, expected,
            "the token ids are not the ones the run plan froze"
        );
    }
    json!({
        "text_path": text,
        "text_sha256": sha256_file(std::path::Path::new(&text)).expect("text sha256"),
        "tokenizer_path": tokenizer,
        "tokenizer_sha256": sha256_file(std::path::Path::new(&tokenizer)).expect("tokenizer sha256"),
        "ids_used": ids.len(),
        "ids_sha256": sha,
    })
}

/// The shared-expert mode the step itself reports in its load-line suffix: `Some(false)` for the
/// separate block, `Some(true)` for the shared expert inside the routed launches.
fn shared_in_routed_mode_of(step: &super::step::QwenResidentStep<'_>) -> Option<bool> {
    shared_in_routed_mode(&step.shared_in_routed_suffix())
}

/// The engine loads the shared expert inside the routed launches by default; the K-row path runs the
/// separate block, so the measurement forces it before the first token and checks the step's own
/// reported mode.
fn force_separate_shared_block(step: &mut super::step::QwenResidentStep<'_>) {
    step.set_shared_in_routed(false)
        .expect("force the separate shared block");
    assert_eq!(
        shared_in_routed_mode_of(step),
        Some(false),
        "the step must report the separate shared block: {}",
        step.shared_in_routed_suffix()
    );
}

/// Records the forced mode in the identity. `forced_off_by_hook` is true because
/// `force_separate_shared_block` ran and asserted the step's reported mode first.
fn record_shared_block(id: &mut Value, step: &super::step::QwenResidentStep<'_>) {
    id["shared_in_routed"]["forced_off_by_hook"] = json!(true);
    id["shared_in_routed"]["step_mode"] = json!(step.shared_in_routed_suffix().trim());
}

// ---------------------------------------------------------------------------------------------
// The phases of the captured step.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "the 3090, held exclusively (21 GiB lease): loads the real 4.00 bpw pack; a measurement, one future window"]
fn krow_phase_record() {
    refuse_switches();
    if !device_present("krow phase record") {
        return;
    }
    let started = Instant::now();
    let depths = env_list("FIFI_KROW_DEPTHS", "8192,32768");
    assert!(depths.windows(2).all(|w| w[0] < w[1]), "depths rise");
    let tokens: usize = env_num("FIFI_KROW_TOKENS", 60);
    let warm: usize = env_num("FIFI_KROW_WARM", 6);
    let deadline_s: f64 = env_num("FIFI_KROW_DEADLINE_S", 780.0);
    let deepest = *depths.last().expect("a depth");
    let need = deepest + warm + tokens + CONTEXT_SLACK;
    let (_tokenizer, stream_ids) = text_stream(need);
    let text = &stream_ids[..need];

    let loaded = load_pack();
    assert_eq!(loaded.kernel, "wide1", "the served single-row kernel");
    let mut step = bench_step(
        &loaded,
        KvMode::Fp16,
        deepest + CONTEXT_SLACK,
        true,
        true,
        None,
    )
    .expect("step");
    force_separate_shared_block(&mut step);
    let kinds = dims().shape.kinds.clone();
    let plan: Vec<(usize, usize, AttentionKind)> = step
        .marked_groups()
        .iter()
        .map(|g| (g.layer, g.size, kinds[g.layer]))
        .collect();
    println!(
        "KROW_CONFIG groups={} load_s={:.1}",
        plan.len(),
        started.elapsed().as_secs_f64()
    );

    let mut cells = Vec::new();
    let mut stopped_early = false;
    for &depth in &depths {
        if started.elapsed().as_secs_f64() > deadline_s {
            stopped_early = true;
            break;
        }
        let at = step.tokens();
        let began = Instant::now();
        step.feed_rows(&text[at..depth]).expect("prefill");
        println!(
            "KROW_PREFILL from={at} to={depth} wall_s={:.2}",
            began.elapsed().as_secs_f64()
        );
        // The first token of the process captures the graph; the warm tokens follow it.
        let mut readings = Vec::new();
        for index in 0..warm + tokens {
            let id = text[step.tokens()];
            let began = Instant::now();
            step.run_token(id, false).expect("token");
            let wall_ms = began.elapsed().as_secs_f64() * 1e3;
            let times = step
                .section_times()
                .expect("section times")
                .expect("marks are on");
            if index >= warm {
                readings.push(Reading {
                    wall_ms,
                    whole_ms: f64::from(times.whole_ms()),
                    embed_ms: f64::from(times.embed_ms()),
                    tail_ms: f64::from(times.tail_ms()),
                    groups: times
                        .group_marked_ms()
                        .iter()
                        .map(|v| f64::from(*v))
                        .collect(),
                });
            }
        }
        let cell = depth_json(depth, &readings, &plan);
        println!(
            "KROW_PHASE depth={depth} wall_ms={} gpu_ms={}",
            cell["step_wall_ms"]["median"], cell["step_gpu_event_ms"]["median"]
        );
        cells.push(cell);
    }
    let launches = step.kernel_launches();
    let kernel_check = require_kernel_ran(&launches, loaded.kernel);
    let mut id = identity("Fp16", true, loaded.kernel);
    record_shared_block(&mut id, &step);
    id["token_ids"] = token_identity(text);
    let record = json!({
        "schema": 1,
        "kind": "krow_phase",
        "identity": id,
        "method": {
            "timing": "in-graph event pairs of the existing section marks (marks on, capture on); the step wall is the host time of run_token without logits",
            "observer_effect": "the marks add a few event-record nodes to every replay; the nsys leg adds the profiler's own",
            "groups": "the second layer of each run of identical layer kinds (Qwen3.6: three GDN layers then one full-attention layer) times the run size",
            "warm_tokens": warm, "timed_tokens": tokens,
            "text": "teacher forced from the pinned text, so the routing is that of real text",
        },
        "single_row_launches": launches.describe(),
        "kernel_check": format!("{kernel_check:?}"),
        "cells": cells,
        "stopped_early": stopped_early,
        "wall_s": started.elapsed().as_secs_f64(),
    });
    emit(&record);
    assert!(
        kernel_check.is_ok(),
        "the intended kernel did not run alone: {kernel_check:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// Real routing: the expert union and the per-row routed chains.
// ---------------------------------------------------------------------------------------------

/// The K-row routed chain of one layer: per row its own normalized row, ids and expert buffers, the
/// scratch shared by all rows (the chain is serial on the stream).
struct ChainRig {
    norm: Vec<DeviceBuffer>,
    ids: Vec<DeviceBuffer>,
    gu: Vec<DeviceBuffer>,
    act: Vec<DeviceBuffer>,
    down: Vec<DeviceBuffer>,
    xh: DeviceBuffer,
    z: DeviceBuffer,
    flag: DeviceBuffer,
    dummy: DeviceBuffer,
}

impl ChainRig {
    fn new(ctx: &crate::kernels::cuda::CudaContext, moe: &MoeDims, rows: usize) -> Self {
        let alloc = |bytes: usize| ctx.allocate(bytes.max(16)).expect("allocate");
        let h = moe.hidden * F32;
        let mut inputs = Inputs::new(0x6b72_6f77);
        let norm = (0..rows)
            .map(|_| {
                let buffer = alloc(h);
                buffer
                    .write_at(
                        &ctx.stream().expect("stream"),
                        0,
                        &f32_bytes(&inputs.values(moe.hidden)),
                    )
                    .expect("write the row");
                buffer
            })
            .collect();
        Self {
            norm,
            ids: (0..rows).map(|_| alloc(moe.top_k * 4)).collect(),
            gu: (0..rows)
                .map(|_| alloc(moe.top_k * 2 * moe.inter * F32))
                .collect(),
            act: (0..rows)
                .map(|_| alloc(moe.top_k * moe.inter * F32))
                .collect(),
            down: (0..rows).map(|_| alloc(moe.top_k * h)).collect(),
            xh: alloc(2 * moe.top_k * moe.hidden * F32),
            z: alloc(2 * moe.top_k * moe.inter.max(moe.hidden) * F32),
            flag: alloc(4),
            dummy: alloc(h.max(2 * moe.shared_inter * F32)),
        }
    }

    fn full(buffer: &DeviceBuffer) -> DeviceRange<'_> {
        buffer.range(0, buffer.bytes()).expect("range")
    }

    fn bufs(&self, row: usize) -> Exl3MoeBufs<'_> {
        let d = Self::full(&self.dummy);
        Exl3MoeBufs {
            route: RouteBufs {
                resid_in: d,
                resid_out: d,
                norm_row: Self::full(&self.norm[row]),
                logits: d,
                ids: Self::full(&self.ids[row]),
                weights: d,
                shared_gate: d,
                status: d,
            },
            expert_gu: Self::full(&self.gu[row]),
            expert_act: Self::full(&self.act[row]),
            expert_down: Self::full(&self.down[row]),
            shared_gu: d,
            shared_act: d,
            shared_out: d,
            scratch: Scratch {
                xh: Self::full(&self.xh),
                z: Self::full(&self.z),
                flag: Self::full(&self.flag),
            },
        }
    }
}

fn ids_bytes(ids: &[i32]) -> Vec<u8> {
    ids.iter().flat_map(|id| id.to_le_bytes()).collect()
}

/// The per-layer lists of picks from the tap: `picks[layer][token]` is the `top_k` ids.
fn picks_by_layer(tap: Vec<(usize, Vec<i32>)>, layers: usize) -> Vec<Vec<Vec<i32>>> {
    let mut picks = vec![Vec::new(); layers];
    for (layer, ids) in tap {
        picks[layer].push(ids);
    }
    picks
}

#[test]
#[ignore = "the 3090, held exclusively (21 GiB lease): loads the real 4.00 bpw pack; a measurement, one future window"]
fn krow_union_record() {
    refuse_switches();
    if !device_present("krow union record") {
        return;
    }
    let started = Instant::now();
    let prefix: usize = env_num("FIFI_KROW_PREFIX", 1024);
    let decoded: usize = env_num("FIFI_KROW_UNION_TOKENS", 3000);
    let deadline_s: f64 = env_num("FIFI_KROW_DEADLINE_S", 780.0);
    let chain_layers = env_list("FIFI_KROW_CHAIN_LAYERS", "3,15,27,39");
    let chain_windows: usize = env_num("FIFI_KROW_CHAIN_WINDOWS", 24);
    let need = prefix + decoded + 16;
    let (_tokenizer, stream_ids) = text_stream(need);
    let text = &stream_ids[..need];

    let loaded = load_pack();
    assert_eq!(loaded.kernel, "wide1", "the served single-row kernel");
    // The eager body: the capture off, so the picks can be read after each block.
    let mut step = bench_step(
        &loaded,
        KvMode::Fp16,
        need + CONTEXT_SLACK,
        false,
        false,
        None,
    )
    .expect("step");
    force_separate_shared_block(&mut step);
    let layers = dims().shape.kinds.len();
    let began = Instant::now();
    step.feed_rows(&text[..prefix]).expect("prefill");
    println!(
        "KROW_PREFILL rows={prefix} wall_s={:.2} load_s={:.1}",
        began.elapsed().as_secs_f64(),
        (began - started).as_secs_f64()
    );
    *step.moe_pick_tap.borrow_mut() = Some(Vec::new());
    let began = Instant::now();
    let mut done = 0;
    for _ in 0..decoded {
        if started.elapsed().as_secs_f64() > deadline_s * 0.6 {
            break;
        }
        let id = text[step.tokens()];
        step.run_token(id, false).expect("token");
        done += 1;
    }
    let tap = step.moe_pick_tap.borrow_mut().take().expect("tap");
    println!(
        "KROW_DECODE tokens={done} wall_s={:.1} tapped={}",
        began.elapsed().as_secs_f64(),
        tap.len()
    );
    assert_eq!(tap.len(), done * layers, "one pick row per layer per token");
    let picks = picks_by_layer(tap, layers);
    assert!(
        picks.iter().all(|p| p.len() == done),
        "every layer saw every token"
    );
    if let Some(path) = std::env::var_os("FIFI_KROW_PICKS_OUT") {
        let table: Vec<Vec<&Vec<i32>>> = (0..done)
            .map(|t| picks.iter().map(|layer| &layer[t]).collect())
            .collect();
        fs::write(&path, serde_json::to_string(&table).expect("json")).expect("write the picks");
    }
    let union = union_json(&picks);
    let launches = step.kernel_launches();
    let kernel_check = require_kernel_ran(&launches, loaded.kernel);

    // The routed chains on the real ids.
    let ctx = loaded.projector.context();
    let stream = ctx.stream().expect("stream");
    let table = LayerTable::new(&dims(), &inventory()).expect("layer table");
    let moe = MoeDims::new(&dims(), STEP_ROWS);
    let flush = Flush::new(ctx).expect("flush");
    let watch = Stopwatch::new(ctx).expect("events");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let ks = [1usize, 2, 3, 4, 8];
    let mut chain_cells = Vec::new();
    for &layer in &chain_layers {
        let weights =
            Exl3MoeWeights::resolve(loaded.weights, &table, layer, &moe).expect("moe weights");
        let rig = ChainRig::new(ctx, &moe, 8);
        let mut graphs = Vec::new();
        for &k in &ks {
            let graph = capture_graph(ctx, &stream, || {
                for row in 0..k {
                    routed_on_device(&launch, &weights, &rig.bufs(row), &moe)?;
                }
                Ok(())
            })
            .unwrap_or_else(|e| panic!("capture the chain of layer {layer}, K {k}: {e:?}"));
            graphs.push(graph);
        }
        let last_start = done.saturating_sub(8);
        let mut real: Vec<Vec<(f64, usize)>> = vec![Vec::new(); ks.len()];
        let mut disjoint: Vec<Vec<f64>> = vec![Vec::new(); ks.len()];
        for window in 0..chain_windows {
            let start = if chain_windows > 1 {
                window * last_start / (chain_windows - 1)
            } else {
                0
            };
            for &index in &rotated(&(0..ks.len()).collect::<Vec<_>>(), window) {
                let k = ks[index];
                let mut union_ids: Vec<i32> = Vec::new();
                for row in 0..k {
                    let ids = &picks[layer][start + row];
                    union_ids.extend(ids);
                    rig.ids[row]
                        .write_at(&stream, 0, &ids_bytes(ids))
                        .expect("ids");
                }
                union_ids.sort_unstable();
                union_ids.dedup();
                let ms = watch
                    .replay(ctx, &stream, Some(&flush), &graphs[index])
                    .expect("replay");
                real[index].push((ms, union_ids.len()));
                // The control: row r picks experts 8 * r + (window mod 8) + 0..8, all distinct.
                let shift = (window % 8) as i32;
                for row in 0..k {
                    let ids: Vec<i32> = (0..moe.top_k as i32)
                        .map(|slot| (8 * row as i32 + slot + shift) % moe.n_expert as i32)
                        .collect();
                    rig.ids[row]
                        .write_at(&stream, 0, &ids_bytes(&ids))
                        .expect("ids");
                }
                disjoint[index].push(
                    watch
                        .replay(ctx, &stream, Some(&flush), &graphs[index])
                        .expect("replay"),
                );
            }
        }
        let t1 = median(&real[0].iter().map(|p| p.0).collect::<Vec<_>>()).unwrap_or(f64::NAN);
        let rows_json: Vec<Value> = ks
            .iter()
            .enumerate()
            .map(|(index, &k)| {
                let us: Vec<f64> = real[index].iter().map(|p| p.0 * 1e3).collect();
                let unions: Vec<f64> = real[index].iter().map(|p| p.1 as f64).collect();
                let control: Vec<f64> = disjoint[index].iter().map(|v| v * 1e3).collect();
                json!({
                    "k": k,
                    "real_chain_us": summary(&us),
                    "real_over_one_chain": median(&us).map(|m| m / (t1 * 1e3)),
                    "real_distinct_experts_mean": mean(&unions),
                    "disjoint_control_us": summary(&control),
                    "disjoint_over_one_chain": median(&control).map(|m| m / (t1 * 1e3)),
                })
            })
            .collect();
        println!("KROW_CHAIN layer={layer} one_chain_us={:.1}", t1 * 1e3);
        chain_cells.push(json!({"layer": layer, "windows": chain_windows, "rows": rows_json}));
        drop(graphs);
    }

    let mut id = identity("Fp16", false, loaded.kernel);
    record_shared_block(&mut id, &step);
    id["token_ids"] = token_identity(text);
    let record = json!({
        "schema": 1,
        "kind": "krow_union",
        "identity": id,
        "method": {
            "union": "distinct expert ids among the K * top_k picks of K consecutive decoded tokens, sliding by one token, per layer; the picks are read on the eager body after each MoE block",
            "routing_text": "teacher forced from the pinned text after a chunk-prefilled prefix; the routing is that of real text",
            "chain": "routed_on_device (gate/up indexed launch, swiglu, down indexed launch) once per row, K times back to back in one captured graph, ids written per window outside the event pair, L2 flushed before the start event; the control gives every row 8 experts no other row picks",
            "prefix_rows": prefix, "decoded_tokens": done,
            "caveat": UNION_CAVEAT,
        },
        "single_row_launches": launches.describe(),
        "kernel_check": format!("{kernel_check:?}"),
        "union": union,
        "caveat": UNION_CAVEAT,
        "chains": chain_cells,
        "wall_s": started.elapsed().as_secs_f64(),
    });
    emit(&record);
    assert!(
        kernel_check.is_ok(),
        "the intended kernel did not run alone: {kernel_check:?}"
    );
}
