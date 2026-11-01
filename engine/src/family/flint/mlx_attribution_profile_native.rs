//! Sixteen-step per-op-class host-wall attribution for one-row decode.
//!
//! Profile-build diagnostics only: a forced sync after each op class removes
//! pipelining, so class sums are attribution, not latency. The layer loop is a
//! test-only replica of `mlx_forward.rs` at this commit; it issues the same
//! kernels in the same order with one extra evaluate at each class boundary.
use super::{Model, State, clean_observation, git, metadata, qualification_paths};
use crate::family::flint::mlx_model::Ffn;
use crate::family::flint::mlx_route::{self, Route};
use crate::family::flint::mlx_state::{ChunkCachePlan, Plane, WINDOW};
use crate::family::flint::precision;
use crate::family::flint::precision_native;
use crate::family::sampler::Sampler;
use crate::kernels::mlx::{KvBlock, SharedAttention};
use crate::kernels::{BinaryOp, MlxKernels};
use crate::tensor::ExecutionTensor;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    time::Instant,
};

const CLASSES: &[&str] = &[
    "norms_residual",
    "attn_qkv_proj",
    "attn_core_kv_score",
    "attn_out_and_kv_append",
    "router",
    "routed_gate_up",
    "routed_down",
    "shared_expert",
    "combine",
    "dense_ffn",
    "head",
];

fn err(error: impl std::fmt::Debug) -> io::Error {
    io::Error::other(format!("{error:?}"))
}

/// Which tensors a class boundary syncs on: every output of the class, so
/// no projection work stays lazy into the next timed class.
trait SyncOn {
    fn sync_targets(&self) -> Vec<&ExecutionTensor>;
}

impl SyncOn for ExecutionTensor {
    fn sync_targets(&self) -> Vec<&ExecutionTensor> {
        vec![self]
    }
}

impl SyncOn for Route {
    fn sync_targets(&self) -> Vec<&ExecutionTensor> {
        vec![&self.health, self.ids.ids(), &self.weights]
    }
}

impl SyncOn for crate::kernels::mlx::BoundedIndices {
    fn sync_targets(&self) -> Vec<&ExecutionTensor> {
        vec![self.ids()]
    }
}

impl SyncOn for () {
    fn sync_targets(&self) -> Vec<&ExecutionTensor> {
        Vec::new()
    }
}

/// Carries a non-tensor passenger (a commit guard) through `hit` without
/// syncing anything for it; its owner syncs inside its own class.
struct Passenger<T>(T);

impl<T> SyncOn for Passenger<T> {
    fn sync_targets(&self) -> Vec<&ExecutionTensor> {
        Vec::new()
    }
}

impl<A: SyncOn, B: SyncOn> SyncOn for (A, B) {
    fn sync_targets(&self) -> Vec<&ExecutionTensor> {
        let mut targets = self.0.sync_targets();
        targets.extend(self.1.sync_targets());
        targets
    }
}

/// Per-class forced-sync host wall. `hit` times one op class including its
/// boundary `evaluate`, so the sum attributes the pipelined step time across
/// classes at the cost of the extra synchronizations.
#[derive(Default)]
struct Classes {
    nanos: BTreeMap<&'static str, u64>,
    syncs: BTreeMap<&'static str, u32>,
}

impl Classes {
    fn hit<T: SyncOn>(
        &mut self,
        kernels: &mut MlxKernels,
        name: &'static str,
        run: impl FnOnce(&mut MlxKernels) -> Result<T, crate::EngError>,
    ) -> io::Result<T> {
        let start = Instant::now();
        let value = run(kernels).map_err(err)?;
        let targets = value.sync_targets();
        // One evaluate call syncs every target at once: the overhead
        // correction subtracts one empty sync per hit, not one per tensor.
        kernels.evaluate(&targets).map_err(err)?;
        *self.nanos.entry(name).or_insert(0) += start.elapsed().as_nanos() as u64;
        *self.syncs.entry(name).or_insert(0) += 1;
        Ok(value)
    }
}

fn split_heads(
    k: &mut MlxKernels,
    x: &ExecutionTensor,
    heads: u32,
) -> Result<ExecutionTensor, crate::EngError> {
    let rows = x.shape().first().copied().unwrap_or(0);
    let x = k.reshape(x, &[rows, heads, 128])?;
    let x = k.transpose(&x, &[1, 0, 2])?;
    Ok(x)
}

fn merge_heads(
    k: &mut MlxKernels,
    x: &ExecutionTensor,
) -> Result<ExecutionTensor, crate::EngError> {
    let x = k.reshape(x, &[32, 1, 128])?;
    let x = k.transpose(&x, &[1, 0, 2])?;
    k.reshape(&x, &[1, 4096])
}

#[test]
#[ignore = "requires exclusive parent-owned Mac window and native-fault-qualification"]
fn achieved_read_bandwidth() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window"),
    );
    clean_observation()?;
    let (_root, _token_path, out) = qualification_paths()?;
    fs::create_dir(&out)?;
    let mut info = json!({"source_commit":git("HEAD")?,"source_tree":git("HEAD^{tree}")?,
        "method":"10 full-buffer reduce(Sum) passes over an 8 GiB F32 resident tensor, one evaluate per pass",
        "completed":false});
    let result = bandwidth(&out, &mut info);
    info["completed"] = result.is_ok().into();
    if let Err(error) = &result {
        info["error"] = json!(error.to_string());
    }
    metadata(&out, &info)?;
    result
}

/// Achieved device read bandwidth for the decode floor: one row scaled to
/// 8 GiB of F32 with `repeat`, then timed full reductions. Each pass reads
/// the whole buffer once; `bytes / time` is the achieved rate.
fn bandwidth(out: &Path, info: &mut Value) -> io::Result<()> {
    let _serial = crate::native_test_lock();
    let mut kernels = MlxKernels::try_new().map_err(err)?;
    let seed = crate::tensor::Tensor::try_from_host(
        crate::tensor::Dtype::F32,
        vec![1, 4096],
        vec![1.0_f32; 4096]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    )
    .map_err(err)?;
    let row = kernels.import_tensor(&seed).map_err(err)?;
    let rows = 1u32 << 19; // 2^19 x 4096 F32 elements = 8 GiB.
    let wide = kernels.repeat(&row, rows, 0).map_err(err)?;
    // Native ops are F32-only (check_operand), so the probe reads F32; at 8
    // GiB the achieved rate measures the memory subsystem, not a dtype path.
    let buffer = wide;
    kernels.evaluate(&[&buffer]).map_err(err)?;
    let bytes = u64::from(rows) * 4096 * 4;
    let passes = 10u32;
    let start = Instant::now();
    for _ in 0..passes {
        let sums = kernels
            .reduce(crate::kernels::ReduceOp::Sum, &buffer, 1, false)
            .map_err(err)?;
        kernels.evaluate(&[&sums]).map_err(err)?;
    }
    let nanos = start.elapsed().as_nanos() as u64;
    let gbps = bytes as f64 * f64::from(passes) / nanos as f64;
    info["bandwidth"] = json!({"bytes_per_pass": bytes, "passes": passes,
        "nanos_total": nanos, "achieved_read_gbps": gbps,
        "dtype": "f32", "shape": [rows, 4096]});
    fs::write(
        out.join("bandwidth.json"),
        serde_json::to_vec_pretty(&info["bandwidth"])?,
    )?;
    Ok(())
}

#[test]
#[ignore = "requires exclusive parent-owned Mac window and native-fault-qualification"]
fn sixteen_step_class_attribution() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window")
    );
    clean_observation()?;
    if std::env::var_os("FIFI_FLINT_MLX_BATCH_PREFILL").is_some()
        || std::env::var_os("FLINT_QUAL_STAGE_SYNC").is_some()
        || !crate::family::flint::mlx_forward::coalesce_residual_flagged()
        || !crate::family::flint::mlx_forward::async_chain_flagged()
    {
        return Err(io::Error::other(
            "attribution requires the serving piece-size and stage-sync envs unset, residual coalescing and the async chain ON (the evaluates_per_token row reports the chained cadence)",
        ));
    }
    let requested = match std::env::var("FLINT_ATTRIB_DEPTH") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => "4096".to_string(),
        Err(error) => return Err(io::Error::other(format!("FLINT_ATTRIB_DEPTH: {error}"))),
    };
    let depth: usize = match requested.as_str() {
        "4096" => 4096,
        "16384" => 16384,
        "30720" => 30720,
        _ => {
            return Err(io::Error::other(
                "FLINT_ATTRIB_DEPTH must be 4096, 16384, or 30720",
            ));
        }
    };
    // The 4K leg stays pinned to the frozen speed stimulus file; the deeper legs
    // rebuild the same arithmetic sequence every run, so each digest is a
    // constant and no stimulus file travels with the harness. The token env
    // is a 4K-leg requirement only.
    let (root, token_path, out) = if depth == 4096 {
        let (root, token, out) = qualification_paths()?;
        (root, Some(token), out)
    } else {
        let (root, out) = generated_leg_paths()?;
        (root, None, out)
    };
    let (ids, stimulus_source, digest) = if depth == 4096 {
        let token_path = token_path
            .as_ref()
            .ok_or_else(|| io::Error::other("4096 leg requires FLINT_QUAL_TOKENS"))?;
        if !fs::metadata(token_path)?.is_file() {
            return Err(io::Error::other("attribution input must be a regular file"));
        }
        let mut raw = Vec::new();
        fs::File::open(token_path)?
            .take(1024 * 1024 + 1)
            .read_to_end(&mut raw)?;
        let digest = format!("{:x}", Sha256::digest(&raw));
        // Cross-check the hardcoded digest against the speed manifest when
        // one is provided, so a transcription typo in the constant cannot pass
        // silently: the constant and the manifest must name the same stimulus.
        if let Some(manifest_path) = std::env::var_os("FLINT_QUAL_SPEED_MANIFEST") {
            let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)
                .map_err(|error| io::Error::other(format!("bad speed manifest: {error}")))?;
            let recorded = manifest["input"]["sha256"]
                .as_str()
                .ok_or_else(|| io::Error::other("speed manifest lacks input.sha256"))?;
            if recorded != "d72c7b49525c2c5fda9e60f6b6e5b1573a83c52199380524315d62a189185843" {
                return Err(io::Error::other(
                    "digest constant disagrees with the speed manifest",
                ));
            }
        }
        if digest != "d72c7b49525c2c5fda9e60f6b6e5b1573a83c52199380524315d62a189185843" {
            return Err(io::Error::other(
                "attribution requires the frozen 4097-ID speed stimulus",
            ));
        }
        let ids: Vec<u32> = serde_json::from_slice(&raw)?;
        if ids.len() != 4097 || ids.iter().any(|&id| id >= 150_144) {
            return Err(io::Error::other("invalid frozen speed IDs"));
        }
        (ids, "frozen-speed-file", digest)
    } else {
        let ids = deterministic_stimulus(depth);
        let raw = serde_json::to_vec(&ids)
            .map_err(|error| io::Error::other(format!("stimulus encode: {error}")))?;
        let digest = format!("{:x}", Sha256::digest(&raw));
        let pinned = match depth {
            16384 => "e9b9b3bdb5f9e986d90102ad46179188342bf712290b6073d8b028b7b002ef58",
            _ => "28729cec50cd5c029ae8ce25794973bbd1fc9f20cadf8fd44ab31354cc122e8b",
        };
        if digest != pinned {
            return Err(io::Error::other(
                "deterministic stimulus digest mismatch: the generator changed",
            ));
        }
        (ids, "deterministic-arithmetic", digest)
    };
    fs::create_dir(&out)?;
    let mut info = json!({"source_commit":git("HEAD")?,"source_tree":git("HEAD^{tree}")?,
        "model_revision":"3b93f9053b530488405ae9fca51d4900d6ce21f0","token_ids_sha256":digest,
        "requested_depth":depth,"stimulus_source":stimulus_source,
        "prefill_piece":512,"prefix_fed_untimed":depth-1,"requested_forwards":16,"start_position":depth-1,
        "first_feed_id":ids[depth-1],"vocab":150144,"ignores_eos":true,"d1_coalescing":true,
        "forced_sync_per_class":true,"classes_sum_is_attribution_not_latency":true,
        "host_wall_profile_build_not_production_throughput":true,
        "layer_loop":"test-only replica of mlx_forward.rs at source_commit",
        "empty_sync_calibration":"8 evaluates of a resident sink tensor per step","completed":false});
    let result = observe(&root, &out, &ids, &mut info, depth);
    info["completed"] = result.is_ok().into();
    if let Err(error) = &result {
        info["error"] = json!(error.to_string());
    }
    metadata(&out, &info)?;
    result
}

fn prefix(model: &mut Model, state: &mut State, ids: &[u32], depth: usize) -> io::Result<()> {
    for start in (0usize..depth).step_by(512) {
        let begin = start.saturating_sub(1);
        let end = (start + 512).min(depth) - 1;
        if model
            .prefill_rows(state, &ids[begin..end], false)
            .map_err(err)?
            .is_some()
        {
            return Err(io::Error::other("no-head prefix returned logits"));
        }
    }
    if state.pos as usize != depth - 1 {
        return Err(io::Error::other("prefix position mismatch"));
    }
    Ok(())
}

/// The deeper-leg stimulus: one arithmetic sequence per depth, rebuilt every
/// run, so the digest pinned in the test is the only state that travels.
fn deterministic_stimulus(depth: usize) -> Vec<u32> {
    (0..depth + 1)
        .map(|i| ((i as u64 * 2654435761 + depth as u64) % 150144) as u32)
        .collect()
}

/// Model and output paths for the generated-stimulus legs: the same checks
/// qualification_paths applies, minus the token file the deep legs never read.
fn generated_leg_paths() -> io::Result<(PathBuf, PathBuf)> {
    use super::{env_path, scratch};
    let root = env_path("FLINT_QUAL_MODEL")?.canonicalize()?;
    let requested = env_path("FLINT_QUAL_OUTPUT")?;
    if !requested.is_absolute() {
        return Err(io::Error::other(
            "qualification paths must use task scratch",
        ));
    }
    let parent = requested
        .parent()
        .ok_or_else(|| io::Error::other("output parent missing"))?
        .canonicalize()?;
    let name = requested
        .file_name()
        .ok_or_else(|| io::Error::other("output name missing"))?
        .to_owned();
    let out = parent.join(name);
    if !scratch(&parent) || out.starts_with(&root) {
        return Err(io::Error::other("output must be scratch outside weights"));
    }
    Ok((root, out))
}

fn observe(root: &Path, out: &Path, ids: &[u32], info: &mut Value, depth: usize) -> io::Result<()> {
    let _serial = crate::native_test_lock();
    let mut model = Model::open(root).map_err(err)?;
    // The evaluate cadence the served decode uses, measured from the layer
    // count and the chain group, not from a hand-copied number.
    let layers = model.layers.len();
    let group = crate::family::flint::mlx_forward::GROUP_LAYERS;
    let chain_evaluates = layers / group + usize::from(layers % group != 0);
    {
        // Discard one full warm prefix+decode pass, like the sync profile.
        let mut warm = State::new();
        prefix(&mut model, &mut warm, ids, depth)?;
        let mut sampler = Sampler::greedy();
        let mut pending = ids[depth - 1];
        let unsplit_started = Instant::now();
        for _ in 0..16 {
            let row = model
                .step(&mut warm, pending, true)
                .map_err(err)?
                .ok_or_else(|| io::Error::other("warm step returned no logits"))?;
            pending = sampler.sample(&row).map_err(err)?;
        }
        info["unsplit_warm_ms_per_step"] =
            json!(unsplit_started.elapsed().as_secs_f64() * 1000.0 / 16.0);
    }
    let mut state = State::new();
    prefix(&mut model, &mut state, ids, depth)?;
    let mut sampler = Sampler::greedy();
    let mut pending = ids[depth - 1];
    let mut classes = Classes::default();
    let mut empty_sync_nanos = 0u64;
    let mut empty_sync_count = 0u32;
    for _step in 0..16 {
        let mut health = Vec::new();
        // Keep the custom decode loop on the same grow-on-demand planes as
        // mlx_forward: grow to this row, then plan at the grown capacity
        // (the planes' real size), not the raw context limit.
        state
            .ensure_full_capacity(&mut model.kernels, state.pos + 1)
            .map_err(err)?;
        let packed = model.source.embedding_rows(&[pending]).map_err(err)?;
        let embedding = precision::embedding(&packed).map_err(err)?;
        let mut x = model.kernels.import_tensor(&embedding).map_err(err)?;
        for index in 0..model.layers.len() {
            let layer = &model.layers[index];
            let plan =
                ChunkCachePlan::new(index, state.pos, 1, state.full_capacity).map_err(err)?;
            let normalized = classes.hit(&mut model.kernels, "norms_residual", |k| {
                k.rms_norm(&x, &layer.input_norm, 1e-6)
            })?;
            let ((q, keys), values) = classes.hit(&mut model.kernels, "attn_qkv_proj", |k| {
                let q = k.qmm(&normalized, &layer.q)?;
                let keys = k.qmm(&normalized, &layer.k)?;
                let values = k.qmm(&normalized, &layer.v)?;
                let q = split_heads(k, &q, 32)?;
                let keys = split_heads(k, &keys, 4)?;
                let values = split_heads(k, &values, 4)?;
                let rope = plan.step.rope.map(|_| plan.p0);
                let (q, keys) = precision_native::normalize_qk(
                    k,
                    (&q, &keys),
                    (&layer.q_norm, &layer.k_norm),
                    rope,
                )?;
                let q = k.reshape(&q, &[1, 32, 1, 128])?;
                Ok(((q, keys), values))
            })?;
            let (core, Passenger(guard)) =
                classes.hit(&mut model.kernels, "attn_core_kv_score", |k| {
                    let (keys, values, guard) = Plane::prepare_chunk(
                        k,
                        &mut state.planes[index],
                        &plan,
                        &keys,
                        &values,
                        false,
                    )?;
                    let slots = keys.shape()[1];
                    let keys = k.reshape(&keys, &[1, 4, slots, 128])?;
                    let values = k.reshape(&values, &[1, 4, slots, 128])?;
                    let window = plan.step.rope.map(|_| WINDOW);
                    let key_origin = if window.is_some() {
                        plan.p0.saturating_sub(WINDOW - 1)
                    } else {
                        0
                    };
                    let output = k.shared_attention(SharedAttention {
                        q: &q,
                        kv: KvBlock::Float {
                            keys: &keys,
                            values: &values,
                        },
                        query_start: plan.p0,
                        key_origin,
                        window,
                        sinks: Some(&layer.sink),
                        scale: 1.0 / 128.0_f32.sqrt(),
                    })?;
                    let merged = merge_heads(k, &output)?;
                    Ok((merged, Passenger(guard)))
                })?;
            let projected = classes.hit(&mut model.kernels, "attn_out_and_kv_append", |k| {
                let projected = k.qmm(&core, &layer.o)?;
                Plane::commit_chunk(k, &mut state.planes[index], guard, &projected)?;
                Ok(projected)
            })?;
            let (normalized, added) = classes.hit(&mut model.kernels, "norms_residual", |k| {
                let added = k.binary(BinaryOp::Add, &x, &projected)?;
                let normalized = k.rms_norm(&added, &layer.post_norm, 1e-6)?;
                Ok((normalized, added))
            })?;
            let ffn = match &layer.ffn {
                Ffn::Dense(w) => {
                    let out = classes
                        .hit(&mut model.kernels, "dense_ffn", |k| w.apply(k, &normalized))?;
                    classes.hit(&mut model.kernels, "norms_residual", move |k| {
                        k.binary(BinaryOp::Add, &added, &out)
                    })?
                }
                Ffn::Moe {
                    router,
                    routed,
                    shared,
                } => {
                    let route = classes.hit(&mut model.kernels, "router", |k| {
                        let logits = k.mm(&normalized, router)?;
                        mlx_route::select(k, &logits)
                    })?;
                    health.push(route.health);
                    let ((product, sorted_ids), order) =
                        classes.hit(&mut model.kernels, "routed_gate_up", |k| {
                            let ids_tensor = route.ids.ids();
                            let keys = k.u32_to_f32(ids_tensor)?;
                            let order = k.argsort(&keys, 0)?;
                            let sorted_ids = k.take_along_axis(ids_tensor, &order, 0)?;
                            let sorted_ids = k.bound_indices(&sorted_ids, route.ids.bound())?;
                            let repeated = k.repeat(&normalized, 6, 0)?;
                            let sorted_x = k.take(&repeated, &order, 0)?;
                            let sorted_x = k.reshape(&sorted_x, &[6, 1, 2048])?;
                            let gate =
                                k.gather_qmm_sorted_bounded(&sorted_x, &routed.gate, &sorted_ids)?;
                            let up =
                                k.gather_qmm_sorted_bounded(&sorted_x, &routed.up, &sorted_ids)?;
                            let gate = k.silu(&gate)?;
                            let product = k.binary(BinaryOp::Mul, &gate, &up)?;
                            Ok(((product, sorted_ids), order))
                        })?;
                    let ordered = classes.hit(&mut model.kernels, "routed_down", |k| {
                        let output =
                            k.gather_qmm_sorted_bounded(&product, &routed.down, &sorted_ids)?;
                        let output = k.reshape(&output, &[6, 2048])?;
                        let order = k.u32_to_f32(&order)?;
                        let inverse = k.argsort(&order, 0)?;
                        let ordered = k.take(&output, &inverse, 0)?;
                        k.reshape(&ordered, &[1, 6, 2048])
                    })?;
                    let shared_out = classes.hit(&mut model.kernels, "shared_expert", |k| {
                        shared.apply(k, &normalized)
                    })?;
                    classes.hit(&mut model.kernels, "combine", |k| {
                        let rows =
                            precision_native::combine(k, &ordered, &route.weights, &shared_out)?;
                        k.binary(BinaryOp::Add, &added, &rows)
                    })?
                }
            };
            x = ffn;
            // Empty-sync calibration: sync a small resident tensor repeatedly.
            for _ in 0..8 {
                let start = Instant::now();
                model.kernels.evaluate(&[&layer.sink]).map_err(err)?;
                empty_sync_nanos += start.elapsed().as_nanos() as u64;
                empty_sync_count += 1;
            }
        }
        state.pos += 1;
        let logits = classes.hit(&mut model.kernels, "head", |k| {
            let last = k.slice(&x, &[0, 0], &[1, 2048])?;
            let normalized = k.rms_norm(&last, &model.norm, 1e-6)?;
            k.qmm(&normalized, &model.head)
        })?;
        let row = mlx_route::read_step(&mut model.kernels, Some(&logits), &health)
            .map_err(err)?
            .ok_or_else(|| io::Error::other("attribution head returned no logits"))?;
        pending = sampler.sample(&row).map_err(err)?;
    }
    let empty_mean_ns = if empty_sync_count == 0 {
        0
    } else {
        empty_sync_nanos / u64::from(empty_sync_count)
    };
    let mut report = serde_json::Map::new();
    let mut total_raw_ms = 0.0_f64;
    let mut total_net_ms = 0.0_f64;
    for name in CLASSES {
        let ns = classes.nanos.get(*name).copied().unwrap_or(0);
        let count = classes.syncs.get(*name).copied().unwrap_or(0);
        let raw_ms = ns as f64 / 1e6 / 16.0;
        let net_ms = (ns.saturating_sub(u64::from(count) * empty_mean_ns)) as f64 / 1e6 / 16.0;
        total_raw_ms += raw_ms;
        total_net_ms += net_ms;
        report.insert(
            (*name).into(),
            json!({
                "ms_per_step_raw": raw_ms,
                "ms_per_step_minus_empty_sync": net_ms,
                "forced_syncs_per_step": f64::from(count) / 16.0,
            }),
        );
    }
    info["attribution"] = json!({
        "classes": Value::Object(report),
        "sum_ms_per_step_raw": total_raw_ms,
        "sum_ms_per_step_minus_empty_sync": total_net_ms,
        "empty_sync_mean_ns": empty_mean_ns,
        "pipelined_reference_ms_per_step": 67.5,
        "unsplit_warm_ms_per_step": info["unsplit_warm_ms_per_step"].clone(),
        "evaluates_per_token": {
            "async_chain": crate::family::flint::mlx_forward::async_chain_flagged(),
            "layer_chain_evaluates": chain_evaluates,
            "readback": 1,
            "source": "mlx_forward GROUP_LAYERS chains plus the health readback",
        },
        "fence_vs_latency": {
            "class_sum_ms_per_step": total_raw_ms,
            "unsplit_warm_ms_per_step": info["unsplit_warm_ms_per_step"].clone(),
            "note": "the class sum syncs at every class boundary; the warm pass does not",
        },
        "note": "class sums are forced-sync attribution, not latency",
    });
    fs::write(
        out.join("attribution.json"),
        serde_json::to_vec_pretty(&info["attribution"])?,
    )?;
    Ok(())
}
