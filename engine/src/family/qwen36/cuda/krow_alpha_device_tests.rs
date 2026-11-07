//! The marginal cost of one more row through the EXL3 rows kernel, measured on the real 4.00 bpw
//! Qwen3.6-35B-A3B pack, against the single-row `wide1` launch: a MEASUREMENT test that has run in
//! two exclusive GPU windows (the records are kept beside the run plan). It gates nothing about speed.
//!
//! One test per process on the GPU held exclusively: `krow_alpha_record`. It loads the whole pack
//! (about 17 GiB, so a lease of 20 GiB), measures, and writes one JSON record (`FIFI_KROW_OUT`).
//!
//! **What is measured.** For each dense projection role of the step, with the real tensors of
//! every layer that has the role:
//!
//! * `rows`: one launch per layer at `R` rows (R = 1, 2, 3, 4, 6, 8) through `exl3_linear`, the call the
//!   step makes. Under the served single-row kernel (`wide1`) a call of R >= 2 rows is the rows
//!   kernel; R = 1 is the production one-tile body. A sweep over all layers of the role is the
//!   role's cost per token.
//! * `sep`: for each layer, R separate single-row (`wide1`) launches back to back on the stream,
//!   the layer-major order of a graph that runs each row as its own launch. The second and later launches
//!   find the layer's weights in the L2 cache when the matrix is small.
//! * `chunk4` (R = 8 only): two 4-row launches per layer, the alternative to one 8-row launch
//!   if the 8-row launch is worse than two 4-row ones.
//!
//! A sweep is captured once into a CUDA graph and replayed (a host launch loop would be bound by
//! the host, not the device, as the step is not: it replays one graph). The time of a replay is
//! the CUDA event pair around it, on the GPU timeline. The L2 cache is flushed before every timed
//! replay by a 128 MiB device copy that is outside the event pair, because the step reads each
//! weight once per token from DRAM, and a repeated sweep of a small role (the 10 attention k and
//! v matrices are 5 MB) would otherwise be served from the 6 MB L2. Rounds rotate the order of the
//! configurations so that clock or thermal drift does not always hit the same one first. The figure
//! of a configuration is the median of its warm rounds.
//!
//! alpha(R) = (t(R) - t(1)) / ((R - 1) * t(1)) per role, and time-weighted over a family: the sum
//! of (t(R) - t(1)) over the sum of (R - 1) * t(1), so a role weighs by its own per-token solo
//! time (`krow_measure`).
//!
//! **The bitwise check.** For every role, every layer, every R in {2, 3, 4, 6, 8}
//! and two input sets, the output of the R-row launch is compared bit for bit with the R
//! single-row launches (row j of one against the single-row output for input row j). Every
//! mismatch is listed with its role, layer, R, row, column, flat index and both bit patterns; none
//! is hidden. A non-zero arithmetic flag is a mismatch too. The router (logits, ids, weights,
//! shared gate) and the plain GEMVs of GDN beta and alpha are checked the same way. The test
//! writes the record first and fails afterwards, so a mismatch is in the file.
//!
//! **The shared expert.** Gate, up and down are measured as three separate matrices (each with
//! `exl3_linear`, rows = R). In addition `shared_gate_up_pair` times a pair layout that keeps
//! the one-row arithmetic: the reference block runs gate and up as ONE grouped launch of two
//! one-row calls; at R rows the pair becomes 2R one-row calls in one launch (R = 1 is the
//! reference), against R separate reference launches.
//!
//! Not measured here: the routed experts and the phases of the step (`krow_step_device_tests`).
//!
//! Environment: `FIFI_EXL3_CHECKPOINT` (the pack), `FIFI_KROW_COMMIT` (required), `FIFI_KROW_OUT`
//! (the JSON path, required), `FIFI_KROW_ROUNDS` (30), `FIFI_KROW_WARM_FROM` (3: rounds before it
//! are left out of the medians), `FIFI_KROW_DEADLINE_S` (600: after it no new round begins),
//! `FIFI_KROW_ONLY` (a comma list of role labels to measure, for a short run),
//! `FIFI_KROW_PACK_MANIFEST_SHA256` (pins the pack), `FIFI_KROW_RECEIPT` (`1`: skip the sweep and
//! only launch each op once per row count, batched and as separate single-row launches, behind
//! marker copies, for a profiler to read kernel names and launch counts; see `receipt_pass`).
//! The test refuses to start if any variable under the refused prefixes of `krow_measure`
//! (`FIFI_EXL3_`, `INFERENCE_CUDA_`, `INFERENCE_QWEN36_`, `INFERENCE_Q36_`, `Q36_`,
//! `FIFI_GRAPH_BENCH_`) or `FIFI_DELTA_TRACE` is set, except the two allowed ones
//! (`FIFI_EXL3_CHECKPOINT`, `INFERENCE_CUDA_REQUIRE_DEVICE`), or if `FIFI_EXL3_SHARED_IN_ROUTED`
//! is on (`refuse_switches`).

use std::{fs, path::PathBuf, time::Instant};

use serde_json::{Map, Value, json};

use super::{
    common_tests::{dims, inventory},
    d1_window_device_tests::{Loaded, load_pack},
    layer_table::LayerTable,
    moe::MoeDims,
    resources::GlobalNames,
};
use crate::family::speculate::krow_measure::{
    ALLOWED_VARS, Inputs, REFUSED_EXACT, REFUSED_PREFIXES, ROWS, SEP_CODE_OFFSET, alpha,
    check_default_path, f32_bytes, first_word_difference, receipt_marker_bytes, rotated,
    weighted_alpha,
};
use crate::{
    EngError,
    family::{
        gguf_model::Role,
        project::cuda::{CudaWeight, Exl3MatrixView, exl3::Exl3Io},
        qwen36::exl3_merge_plan::shared_in_routed_explicit,
        speculate::window_cost::{PackManifest, median, sha256_file, summary},
    },
    kernels::{
        cuda::{
            CapturePlan, CudaContext, DeviceBuffer, DeviceStream, GraphExec, TimingEvent,
            device_present,
            resident::router::{
                RouterCall, RouterLogitsCall, RouterOut, RouterShape, RouterWeights,
            },
        },
        single_row::require_kernel_ran,
    },
};

/// The family a role belongs to in the weighted figures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Family {
    /// The EXL3 dense projections whose rows the rows kernel can share: GDN and attention
    /// projections, the shared-expert matrices and the head.
    Dense,
    /// The shared expert's gate and up as one grouped launch of one-row calls (2R calls
    /// for R rows), measured instead of the two separate matrices.
    SharedPair,
    /// The plain F32 GEMVs of GDN beta and alpha.
    Plain,
    /// The MoE router (logits, selection, shared gate).
    Router,
}

/// One measured role: the label, the gguf role, the family and whether the matrix is EXL3.
#[derive(Clone, Copy, Debug)]
pub(super) struct RoleSpec {
    pub(super) label: &'static str,
    pub(super) role: Role,
    pub(super) family: Family,
}

/// The dense roles of the step. The shared-expert gate, up and down are separate matrices, the way
/// the reference MoE block runs them (`exl3_moe::shared`). The head is `Role::Output`, a single
/// matrix, handled as a global tensor by the device test.
pub(super) const DENSE_ROLES: [RoleSpec; 10] = [
    RoleSpec {
        label: "gdn_qkv",
        role: Role::AttnQkv,
        family: Family::Dense,
    },
    RoleSpec {
        label: "gdn_z",
        role: Role::AttnGate,
        family: Family::Dense,
    },
    RoleSpec {
        label: "gdn_out",
        role: Role::SsmOut,
        family: Family::Dense,
    },
    RoleSpec {
        label: "attn_q",
        role: Role::AttnQ,
        family: Family::Dense,
    },
    RoleSpec {
        label: "attn_k",
        role: Role::AttnK,
        family: Family::Dense,
    },
    RoleSpec {
        label: "attn_v",
        role: Role::AttnV,
        family: Family::Dense,
    },
    RoleSpec {
        label: "attn_o",
        role: Role::AttnOutput,
        family: Family::Dense,
    },
    RoleSpec {
        label: "shared_gate",
        role: Role::FfnGateShexp,
        family: Family::Dense,
    },
    RoleSpec {
        label: "shared_up",
        role: Role::FfnUpShexp,
        family: Family::Dense,
    },
    RoleSpec {
        label: "shared_down",
        role: Role::FfnDownShexp,
        family: Family::Dense,
    },
];

/// The plain (raw F32) GEMV roles of the step.
pub(super) const PLAIN_ROLES: [RoleSpec; 2] = [
    RoleSpec {
        label: "gdn_beta",
        role: Role::SsmBeta,
        family: Family::Plain,
    },
    RoleSpec {
        label: "gdn_alpha",
        role: Role::SsmAlpha,
        family: Family::Plain,
    },
];

const F32: usize = 4;
/// The largest row count of any measurement.
const MAX_ROWS: usize = 8;
/// The L2 flush copy: far larger than the 6 MB L2 of the 3090.
const FLUSH_BYTES: usize = 128 << 20;

// ---------------------------------------------------------------------------------------------
// Guards, identity and the timing primitives shared with `krow_step_device_tests`.
// ---------------------------------------------------------------------------------------------

/// Refuses to run unless the process is on the default path: none of the kernel, numerics or
/// diagnostic switches of `krow_measure::REFUSED_PREFIXES` / `REFUSED_EXACT` is set (the single-row
/// kernel is then the registered `wide1`, the shared expert is not inside the routed launches, and
/// no gather, dp4a, indexed-pair, GDN input-projection fusion, sparse attention, KV8 variant,
/// device shortlist, Q36 ablation or per-op sync option is on). Only `krow_measure::ALLOWED_VARS`
/// may stay set.
pub(super) fn refuse_switches() {
    let names = std::env::vars_os().map(|(name, _)| name.to_string_lossy().into_owned());
    let shared = shared_in_routed_explicit().map_err(|e| format!("{e:?}"));
    if let Err(reasons) = check_default_path(names, shared) {
        panic!("the measurement runs the default path: {reasons}");
    }
}

fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().map_or(default, |text| {
        text.parse().unwrap_or_else(|_| panic!("{name}: {text:?}"))
    })
}

/// The identity block of a record: the commit the runner names, the SHA-256 of this binary, the
/// pack manifest, the KV mode and capture mode, the effective single-row kernel and the effective
/// value of `FIFI_EXL3_SHARED_IN_ROUTED` (nothing is stamped with a time: the runner uses `date -u`).
pub(super) fn identity(kv_mode: &str, capture: bool, kernel: &str) -> Value {
    let commit = std::env::var("FIFI_KROW_COMMIT").unwrap_or_default();
    assert!(
        !commit.is_empty(),
        "FIFI_KROW_COMMIT: the runner names the commit"
    );
    let pack_dir =
        PathBuf::from(std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT"));
    let pack = PackManifest::read(&pack_dir).expect("pack manifest");
    if let Ok(expected) = std::env::var("FIFI_KROW_PACK_MANIFEST_SHA256") {
        assert_eq!(
            pack.manifest_sha256, expected,
            "the pack is not the one the run plan froze"
        );
    }
    let exe = std::env::current_exe().expect("current_exe");
    let mut switches: Vec<(String, String)> = std::env::vars_os()
        .map(|(n, v)| {
            (
                n.to_string_lossy().into_owned(),
                v.to_string_lossy().into_owned(),
            )
        })
        .filter(|(name, _)| {
            ["FIFI_", "Q36_", "INFERENCE_CUDA_", "INFERENCE_QWEN36_"]
                .iter()
                .any(|prefix| name.starts_with(prefix))
        })
        .collect();
    switches.sort();
    json!({
        "commit": commit,
        "binary_path": exe.display().to_string(),
        "binary_sha256": sha256_file(&exe).expect("binary sha256"),
        "pack": pack.to_json(),
        "kv_mode": kv_mode,
        "capture": capture,
        "single_row_kernel": kernel,
        "shared_in_routed": {
            "env": std::env::var("FIFI_EXL3_SHARED_IN_ROUTED").ok(),
            "effective": shared_in_routed_explicit().unwrap_or(true),
        },
        "refused_prefixes": REFUSED_PREFIXES,
        "refused_exact": REFUSED_EXACT,
        "allowed_vars": ALLOWED_VARS,
        "switches": Value::Object(
            switches.into_iter().map(|(name, value)| (name, Value::String(value))).collect()
        ),
    })
}

/// Writes the record to `FIFI_KROW_OUT` and prints it on one line for the log.
pub(super) fn emit(record: &Value) {
    let text = serde_json::to_string_pretty(record).expect("json");
    if let Some(path) = std::env::var_os("FIFI_KROW_OUT") {
        fs::write(&path, &text).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
    }
    println!(
        "KROW_RECORD {}",
        serde_json::to_string(record).expect("json")
    );
}

/// A 128 MiB device-to-device copy that streams the L2 cache out.
pub(super) struct Flush {
    from: DeviceBuffer,
    to: DeviceBuffer,
}

impl Flush {
    pub(super) fn new(ctx: &CudaContext) -> Result<Self, EngError> {
        Ok(Self {
            from: ctx.allocate(FLUSH_BYTES)?,
            to: ctx.allocate(FLUSH_BYTES)?,
        })
    }

    pub(super) fn run(&self, ctx: &CudaContext, stream: &DeviceStream) -> Result<(), EngError> {
        ctx.copy_device(
            stream,
            self.from.range(0, FLUSH_BYTES)?,
            self.to.range(0, FLUSH_BYTES)?,
        )
    }
}

/// The two timing events of a measurement.
pub(super) struct Stopwatch {
    start: TimingEvent,
    end: TimingEvent,
}

impl Stopwatch {
    pub(super) fn new(ctx: &CudaContext) -> Result<Self, EngError> {
        Ok(Self {
            start: ctx.event_timing()?,
            end: ctx.event_timing()?,
        })
    }

    /// The GPU time in milliseconds of one replay of `graph`, behind an L2 flush that is outside
    /// the event pair. Waits for the stream.
    pub(super) fn replay(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        flush: Option<&Flush>,
        graph: &GraphExec<'_>,
    ) -> Result<f64, EngError> {
        if let Some(flush) = flush {
            flush.run(ctx, stream)?;
        }
        stream.record_timing(&self.start)?;
        graph.launch(stream)?;
        stream.record_timing(&self.end)?;
        stream.synchronize()?;
        Ok(f64::from(self.start.elapsed_ms_since(&self.end)?))
    }
}

/// Captures `body` into a graph after one eager run that loads every module it launches.
pub(super) fn capture_graph(
    ctx: &CudaContext,
    stream: &DeviceStream,
    mut body: impl FnMut() -> Result<(), EngError>,
) -> Result<GraphExec<'static>, EngError> {
    body()?;
    stream.synchronize()?;
    let plan = CapturePlan {
        pins: &[],
        holds: Vec::new(),
    };
    let (graph, ()) = ctx.capture(stream, plan, body)?;
    Ok(graph)
}

// ---------------------------------------------------------------------------------------------
// The measured operations.
// ---------------------------------------------------------------------------------------------

struct Env<'a> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Which {
    A,
    B,
}

impl Which {
    fn at(self) -> usize {
        usize::from(self == Self::B)
    }
}

enum Kind {
    /// One EXL3 matrix per layer, projected through `exl3_linear`.
    Dense(Vec<Exl3MatrixView<'static>>),
    /// The shared expert's gate and up of each layer as one grouped launch of one-row calls,
    /// `[gate_0, up_0, gate_1, up_1, ...]`.
    Pair(Vec<(Exl3MatrixView<'static>, Exl3MatrixView<'static>)>),
    /// A raw F32 `[n_expert][hidden]` GEMV per layer (`router_logits_rows`).
    Plain {
        w: Vec<&'static DeviceBuffer>,
        hidden: usize,
        n_expert: usize,
    },
    /// The MoE router per layer: logits, selection, shared gate.
    Router {
        w: Vec<(&'static DeviceBuffer, &'static DeviceBuffer)>,
        hidden: usize,
        n_expert: usize,
        top_k: usize,
    },
}

/// One compared output of a launch: the name, the bytes and the values per row.
type Blob = (&'static str, Vec<u8>, usize);

struct Op {
    label: String,
    family: Option<Family>,
    /// The layers that carry the role (empty for the head).
    layers: Vec<usize>,
    kind: Kind,
    x: DeviceBuffer,
    xh: Option<DeviceBuffer>,
    z: Option<DeviceBuffer>,
    flag: DeviceBuffer,
    out: [DeviceBuffer; 2],
    /// Router only: ids, pick weights and shared gate, for each of the two outputs.
    side: [Vec<DeviceBuffer>; 2],
}

fn whole(buffer: &DeviceBuffer) -> Result<crate::kernels::cuda::DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

impl Op {
    fn count(&self) -> usize {
        match &self.kind {
            Kind::Dense(v) => v.len(),
            Kind::Pair(v) => v.len(),
            Kind::Plain { w, .. } => w.len(),
            Kind::Router { w, .. } => w.len(),
        }
    }

    fn shapes(&self) -> Vec<[usize; 3]> {
        let mut shapes: Vec<[usize; 3]> = match &self.kind {
            Kind::Dense(v) => v
                .iter()
                .map(|m| [m.shape[0], m.shape[1], usize::from(m.bits_twice)])
                .collect(),
            Kind::Pair(v) => v
                .iter()
                .map(|(g, _)| [g.shape[0], g.shape[1], usize::from(g.bits_twice)])
                .collect(),
            Kind::Plain {
                hidden, n_expert, ..
            }
            | Kind::Router {
                hidden, n_expert, ..
            } => vec![[*hidden, *n_expert, 0]],
        };
        shapes.sort_unstable();
        shapes.dedup();
        shapes
    }

    fn new(
        ctx: &CudaContext,
        label: &str,
        family: Option<Family>,
        layers: Vec<usize>,
        kind: Kind,
    ) -> Result<Self, EngError> {
        let buffer = |bytes: usize| ctx.allocate(bytes.max(16));
        let (k_max, n_max, mult, side_top_k) = match &kind {
            Kind::Dense(v) => (
                v.iter().map(|m| m.shape[0]).max().unwrap_or(0),
                v.iter().map(|m| m.shape[1]).max().unwrap_or(0),
                1,
                0,
            ),
            Kind::Pair(v) => (
                v.iter().map(|(g, _)| g.shape[0]).max().unwrap_or(0),
                v.iter().map(|(g, _)| g.shape[1]).max().unwrap_or(0),
                2,
                0,
            ),
            Kind::Plain {
                hidden, n_expert, ..
            } => (*hidden, *n_expert, 1, 0),
            Kind::Router {
                hidden,
                n_expert,
                top_k,
                ..
            } => (*hidden, *n_expert + 1, 1, *top_k),
        };
        let exl3 = matches!(kind, Kind::Dense(_) | Kind::Pair(_));
        let side = |_: usize| -> Result<Vec<DeviceBuffer>, EngError> {
            if side_top_k == 0 {
                return Ok(Vec::new());
            }
            Ok(vec![
                buffer(MAX_ROWS * side_top_k * F32)?,
                buffer(MAX_ROWS * side_top_k * F32)?,
                buffer(MAX_ROWS * F32)?,
            ])
        };
        Ok(Self {
            label: label.to_owned(),
            family,
            layers,
            x: buffer(MAX_ROWS * k_max * F32)?,
            xh: exl3
                .then(|| buffer(MAX_ROWS * mult * k_max * F32))
                .transpose()?,
            z: exl3
                .then(|| buffer(MAX_ROWS * mult * n_max * F32))
                .transpose()?,
            flag: buffer(F32)?,
            out: [
                buffer(MAX_ROWS * mult * n_max * F32)?,
                buffer(MAX_ROWS * mult * n_max * F32)?,
            ],
            side: [side(0)?, side(1)?],
            kind,
        })
    }

    /// Fills the input rows with `seed`'s values and clears the arithmetic flag.
    fn fill(&self, env: &Env<'_>, seed: u64) -> Result<(), EngError> {
        let values = self.x.bytes() / F32;
        let bytes = f32_bytes(&Inputs::new(seed).values(values));
        self.x.write_at(env.stream, 0, &bytes)?;
        self.flag.write_at(env.stream, 0, &[0u8; F32])
    }

    /// Writes 0xff over the outputs a launch must overwrite.
    fn poison(&self, env: &Env<'_>, which: Which) -> Result<(), EngError> {
        let out = &self.out[which.at()];
        out.write_at(env.stream, 0, &vec![0xff; out.bytes()])?;
        for side in &self.side[which.at()] {
            side.write_at(env.stream, 0, &vec![0xff; side.bytes()])?;
        }
        Ok(())
    }

    /// Queues the matrix `mi` for input rows `row0..row0 + nrows`, writing output `which`.
    fn launch_matrix(
        &self,
        env: &Env<'_>,
        mi: usize,
        (row0, nrows): (usize, usize),
        which: Which,
    ) -> Result<(), EngError> {
        let out = &self.out[which.at()];
        match &self.kind {
            Kind::Dense(views) => {
                let view = &views[mi];
                let [k, n] = view.shape;
                let (xh, z) = (self.xh.as_ref().expect("xh"), self.z.as_ref().expect("z"));
                let io = Exl3Io {
                    x: self.x.range(row0 * k * F32, nrows * k * F32)?,
                    xh: xh.range(row0 * k * F32, nrows * k * F32)?,
                    z: z.range(row0 * n * F32, nrows * n * F32)?,
                    out: out.range(row0 * n * F32, nrows * n * F32)?,
                    flag: self.flag.range(0, F32)?,
                    rows: nrows,
                    fused: false,
                };
                env.ctx.exl3_linear(env.stream, view.linear_call(&io))
            }
            Kind::Pair(pairs) => {
                let (gate, up) = &pairs[mi];
                let [k, n] = gate.shape;
                let (xh, z) = (self.xh.as_ref().expect("xh"), self.z.as_ref().expect("z"));
                let mut calls = Vec::with_capacity(2 * nrows);
                for row in row0..row0 + nrows {
                    for (half, view) in [gate, up].into_iter().enumerate() {
                        let pair = 2 * row + half;
                        let io = Exl3Io {
                            x: self.x.range(row * k * F32, k * F32)?,
                            xh: xh.range(pair * k * F32, k * F32)?,
                            z: z.range(pair * n * F32, n * F32)?,
                            out: out.range(pair * n * F32, n * F32)?,
                            flag: self.flag.range(0, F32)?,
                            rows: 1,
                            fused: false,
                        };
                        calls.push(view.linear_call(&io));
                    }
                }
                env.ctx.exl3_grouped_step(env.stream, &calls)
            }
            Kind::Plain {
                w,
                hidden,
                n_expert,
            } => env.ctx.router_logits_rows(
                env.stream,
                RouterLogitsCall {
                    shape: RouterShape {
                        m: nrows,
                        hidden: *hidden,
                        n_expert: *n_expert,
                        top_k: 1,
                    },
                    x: self.x.range(row0 * hidden * F32, nrows * hidden * F32)?,
                    w: whole(w[mi])?,
                    logits: out.range(row0 * n_expert * F32, nrows * n_expert * F32)?,
                },
            ),
            Kind::Router {
                w,
                hidden,
                n_expert,
                top_k,
            } => {
                let cols = n_expert + 1;
                let side = &self.side[which.at()];
                env.ctx.router(
                    env.stream,
                    RouterCall {
                        shape: RouterShape {
                            m: nrows,
                            hidden: *hidden,
                            n_expert: *n_expert,
                            top_k: *top_k,
                        },
                        x: self.x.range(row0 * hidden * F32, nrows * hidden * F32)?,
                        weights: RouterWeights {
                            w: whole(w[mi].0)?,
                            w_shared: Some(whole(w[mi].1)?),
                        },
                        logits: out.range(row0 * cols * F32, nrows * cols * F32)?,
                        out: RouterOut {
                            ids: side[0].range(row0 * top_k * F32, nrows * top_k * F32)?,
                            weights: side[1].range(row0 * top_k * F32, nrows * top_k * F32)?,
                            shared_gate: Some(side[2].range(row0 * F32, nrows * F32)?),
                            status: self.flag.range(0, F32)?,
                        },
                    },
                )
            }
        }
    }

    /// One sweep: every layer's matrix, `rows` input rows in launches of `chunk` rows.
    fn sweep(
        &self,
        env: &Env<'_>,
        rows: usize,
        chunk: usize,
        which: Which,
    ) -> Result<(), EngError> {
        for mi in 0..self.count() {
            let mut row0 = 0;
            while row0 < rows {
                let nrows = chunk.min(rows - row0);
                self.launch_matrix(env, mi, (row0, nrows), which)?;
                row0 += nrows;
            }
        }
        Ok(())
    }

    fn read(
        &self,
        env: &Env<'_>,
        buffer: &DeviceBuffer,
        bytes: usize,
    ) -> Result<Vec<u8>, EngError> {
        let mut out = vec![0u8; bytes];
        buffer.read_at(env.stream, 0, &mut out)?;
        Ok(out)
    }

    /// The outputs of matrix `mi` for `rows` rows, from output `which`.
    fn outputs(
        &self,
        env: &Env<'_>,
        mi: usize,
        rows: usize,
        which: Which,
    ) -> Result<Vec<Blob>, EngError> {
        let out = &self.out[which.at()];
        Ok(match &self.kind {
            Kind::Dense(views) => {
                let n = views[mi].shape[1];
                vec![("out", self.read(env, out, rows * n * F32)?, n)]
            }
            Kind::Pair(pairs) => {
                let n = pairs[mi].0.shape[1];
                vec![("out", self.read(env, out, rows * 2 * n * F32)?, 2 * n)]
            }
            Kind::Plain { n_expert, .. } => {
                vec![(
                    "logits",
                    self.read(env, out, rows * n_expert * F32)?,
                    *n_expert,
                )]
            }
            Kind::Router {
                n_expert, top_k, ..
            } => {
                let side = &self.side[which.at()];
                vec![
                    (
                        "logits",
                        self.read(env, out, rows * (n_expert + 1) * F32)?,
                        n_expert + 1,
                    ),
                    ("ids", self.read(env, &side[0], rows * top_k * F32)?, *top_k),
                    (
                        "weights",
                        self.read(env, &side[1], rows * top_k * F32)?,
                        *top_k,
                    ),
                    ("shared_gate", self.read(env, &side[2], rows * F32)?, 1),
                ]
            }
        })
    }

    /// Writes 0xff over the rotated-input and partial scratch, so a byte a launch did not write is
    /// visible afterwards.
    fn poison_scratch(&self, env: &Env<'_>) -> Result<(), EngError> {
        for buffer in [self.xh.as_ref(), self.z.as_ref()].into_iter().flatten() {
            buffer.write_at(env.stream, 0, &vec![0xff; buffer.bytes()])?;
        }
        Ok(())
    }

    /// The buffers (name, first byte offset) in which a byte past the `rows` real rows of the last
    /// launch into output A was written: the main output, and for an EXL3 matrix the rotated input
    /// and the partials. Empty when the padded rows of the rows kernel left everything untouched.
    fn written_past_the_rows(
        &self,
        env: &Env<'_>,
        mi: usize,
        rows: usize,
    ) -> Result<Vec<(&'static str, usize)>, EngError> {
        let untouched = |bytes: &[u8], from: usize| {
            bytes[from..]
                .iter()
                .position(|b| *b != 0xff)
                .map(|at| from + at)
        };
        let mut found = Vec::new();
        let out = &self.out[Which::A.at()];
        let whole = self.read(env, out, out.bytes())?;
        let used = match &self.kind {
            Kind::Dense(v) => rows * v[mi].shape[1] * F32,
            Kind::Pair(v) => rows * 2 * v[mi].0.shape[1] * F32,
            Kind::Plain { n_expert, .. } => rows * n_expert * F32,
            Kind::Router { n_expert, .. } => rows * (n_expert + 1) * F32,
        };
        if let Some(at) = untouched(&whole, used) {
            found.push(("out", at));
        }
        if let (Kind::Dense(v), Some(xh), Some(z)) = (&self.kind, &self.xh, &self.z) {
            let [k, n] = v[mi].shape;
            let xh_bytes = self.read(env, xh, xh.bytes())?;
            if let Some(at) = untouched(&xh_bytes, rows * k * F32) {
                found.push(("xh", at));
            }
            let z_bytes = self.read(env, z, z.bytes())?;
            if let Some(at) = untouched(&z_bytes, rows * n * F32) {
                found.push(("z", at));
            }
        }
        Ok(found)
    }

    /// The bitwise check of this role at `rows` rows: the R-row launch against R single-row
    /// launches, for every layer, on the inputs of `seed`. Returns the number of comparisons and
    /// the mismatches (each a JSON object).
    fn check(
        &self,
        env: &Env<'_>,
        rows: usize,
        seed: u64,
    ) -> Result<(usize, Vec<Value>), EngError> {
        let mut comparisons = 0;
        let mut mismatches = Vec::new();
        self.fill(env, seed)?;
        for mi in 0..self.count() {
            let layer = self.layers.get(mi).copied();
            self.poison(env, Which::A)?;
            self.poison(env, Which::B)?;
            self.poison_scratch(env)?;
            self.flag.write_at(env.stream, 0, &[0u8; F32])?;
            self.launch_matrix(env, mi, (0, rows), Which::A)?;
            env.stream.synchronize()?;
            let flag_rows = self.read(env, &self.flag, F32)?;
            // The rows kernel of the next instantiation (R = 3 rides rows<4>, R = 5..7 rides rows<8>)
            // pads the missing rows: they read zeros (`x[row] = row < a.rows ? xh[..] : 0`) and the
            // output store is guarded by `row < a.rows`. Nothing may be written past the real rows.
            for (name, offset) in self.written_past_the_rows(env, mi, rows)? {
                mismatches.push(json!({
                    "role": self.label, "layer": layer, "rows": rows, "seed": seed,
                    "output": format!("{name}: bytes past the {rows} real rows were written"),
                    "byte_offset": offset,
                }));
            }
            comparisons += 1;
            self.flag.write_at(env.stream, 0, &[0u8; F32])?;
            for row in 0..rows {
                self.launch_matrix(env, mi, (row, 1), Which::B)?;
            }
            env.stream.synchronize()?;
            let flag_single = self.read(env, &self.flag, F32)?;
            let rows_out = self.outputs(env, mi, rows, Which::A)?;
            let single_out = self.outputs(env, mi, rows, Which::B)?;
            for ((name, left, per_row), (_, right, _)) in rows_out.iter().zip(&single_out) {
                comparisons += 1;
                if let Some(found) = first_word_difference(left, right, *per_row) {
                    mismatches.push(json!({
                        "role": self.label, "layer": layer, "rows": rows, "seed": seed,
                        "output": name, "row": found.row, "col": found.col, "index": found.index,
                        "rows_kernel_bits": format!("{:#010x}", found.left_bits),
                        "single_row_bits": format!("{:#010x}", found.right_bits),
                    }));
                }
            }
            for (name, flag) in [("rows", &flag_rows), ("single", &flag_single)] {
                if flag.iter().any(|b| *b != 0) {
                    mismatches.push(json!({
                        "role": self.label, "layer": layer, "rows": rows, "seed": seed,
                        "output": format!("{name} arithmetic flag"), "flag_bytes": flag,
                    }));
                }
            }
        }
        Ok((comparisons, mismatches))
    }
}

// ---------------------------------------------------------------------------------------------
// Building the operations from the loaded pack.
// ---------------------------------------------------------------------------------------------

fn exl3_view(loaded: &Loaded, name: &str) -> Result<Exl3MatrixView<'static>, EngError> {
    match loaded.weights.get(name) {
        Some(CudaWeight::Exl3 { weight, .. }) => weight.view(),
        _ => Err(EngError::Checkpoint(format!(
            "{name} is not an EXL3 matrix of the pack"
        ))),
    }
}

fn raw_buffer(loaded: &Loaded, name: &str) -> Result<&'static DeviceBuffer, EngError> {
    match loaded.weights.get(name) {
        Some(CudaWeight::Raw { buffer, .. }) => Ok(buffer),
        _ => Err(EngError::Checkpoint(format!(
            "{name} is not a raw tensor of the pack"
        ))),
    }
}

/// The layers of the model that carry `role`, with their tensor names.
fn layers_with(table: &LayerTable, role: Role) -> Vec<(usize, String)> {
    (0..table.len())
        .filter_map(|layer| Some((layer, table.name(layer, role).ok()?.to_owned())))
        .collect()
}

fn build_ops(loaded: &Loaded, ctx: &CudaContext, only: &[String]) -> Result<Vec<Op>, EngError> {
    let table = LayerTable::new(&dims(), &inventory())?;
    let moe = MoeDims::new(&dims(), 1);
    let names = GlobalNames::new(&inventory())?;
    let wanted = |label: &str| only.is_empty() || only.iter().any(|o| o == label);
    let mut ops = Vec::new();
    for spec in DENSE_ROLES {
        if !wanted(spec.label) {
            continue;
        }
        let found = layers_with(&table, spec.role);
        let mut layers = Vec::new();
        let mut views = Vec::new();
        for (layer, name) in found {
            views.push(exl3_view(loaded, &name)?);
            layers.push(layer);
        }
        assert!(!views.is_empty(), "no layer carries {}", spec.label);
        ops.push(Op::new(
            ctx,
            spec.label,
            Some(spec.family),
            layers,
            Kind::Dense(views),
        )?);
    }
    if wanted("shared_gate_up_pair") {
        let mut layers = Vec::new();
        let mut pairs = Vec::new();
        for (layer, gate) in layers_with(&table, Role::FfnGateShexp) {
            let up = table.name(layer, Role::FfnUpShexp)?;
            let (g, u) = (exl3_view(loaded, &gate)?, exl3_view(loaded, up)?);
            assert_eq!(g.shape, u.shape, "shared gate and up share a shape");
            pairs.push((g, u));
            layers.push(layer);
        }
        ops.push(Op::new(
            ctx,
            "shared_gate_up_pair",
            Some(Family::SharedPair),
            layers,
            Kind::Pair(pairs),
        )?);
    }
    if wanted("lm_head") {
        let view = exl3_view(loaded, &names.head)?;
        ops.push(Op::new(
            ctx,
            "lm_head",
            Some(Family::Dense),
            Vec::new(),
            Kind::Dense(vec![view]),
        )?);
    }
    for spec in PLAIN_ROLES {
        if !wanted(spec.label) {
            continue;
        }
        let mut layers = Vec::new();
        let mut w = Vec::new();
        for (layer, name) in layers_with(&table, spec.role) {
            w.push(raw_buffer(loaded, &name)?);
            layers.push(layer);
        }
        let n_expert = w[0].bytes() / (moe.hidden * F32);
        assert!(
            w.iter().all(|b| b.bytes() == n_expert * moe.hidden * F32),
            "{}: the layers differ in shape",
            spec.label
        );
        ops.push(Op::new(
            ctx,
            spec.label,
            Some(spec.family),
            layers,
            Kind::Plain {
                w,
                hidden: moe.hidden,
                n_expert,
            },
        )?);
    }
    if wanted("moe_router") {
        let mut layers = Vec::new();
        let mut w = Vec::new();
        for (layer, name) in layers_with(&table, Role::FfnGateInp) {
            let shared = table.name(layer, Role::FfnGateInpShexp)?;
            w.push((raw_buffer(loaded, &name)?, raw_buffer(loaded, shared)?));
            layers.push(layer);
        }
        ops.push(Op::new(
            ctx,
            "moe_router",
            Some(Family::Router),
            layers,
            Kind::Router {
                w,
                hidden: moe.hidden,
                n_expert: moe.n_expert,
                top_k: moe.top_k,
            },
        )?);
    }
    Ok(ops)
}

// ---------------------------------------------------------------------------------------------
// The measurement.
// ---------------------------------------------------------------------------------------------

/// How a configuration launches its rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum How {
    /// One launch of `rows` rows per layer (the rows kernel for rows >= 2).
    Rows,
    /// `rows` single-row launches per layer, back to back.
    Sep,
    /// Launches of four rows per layer (8 rows as 2 x 4).
    Chunk4,
}

impl How {
    fn name(self) -> &'static str {
        match self {
            Self::Rows => "rows",
            Self::Sep => "sep",
            Self::Chunk4 => "chunk4",
        }
    }

    fn chunk(self, rows: usize) -> usize {
        match self {
            Self::Rows => rows,
            Self::Sep => 1,
            Self::Chunk4 => 4,
        }
    }
}

struct Config {
    op: usize,
    how: How,
    rows: usize,
    graph: GraphExec<'static>,
    ms: Vec<f64>,
}

fn configs_of() -> Vec<(How, usize)> {
    let mut configs = vec![(How::Rows, 1)];
    for rows in &ROWS[1..] {
        configs.push((How::Rows, *rows));
        configs.push((How::Sep, *rows));
    }
    configs.push((How::Chunk4, 8));
    configs
}

fn us_summary(ms: &[f64]) -> Value {
    summary(&ms.iter().map(|v| v * 1e3).collect::<Vec<_>>())
}

fn median_ms(configs: &[Config], op: usize, how: How, rows: usize) -> Option<f64> {
    let config = configs
        .iter()
        .find(|c| c.op == op && c.how == how && c.rows == rows)?;
    median(&config.ms)
}

/// The median sweep time of `how` at `rows` for an op; R = 1 is the same launch for every `how`.
fn t_of(configs: &[Config], op: usize, how: How, rows: usize) -> Option<f64> {
    if rows == 1 {
        return median_ms(configs, op, How::Rows, 1);
    }
    median_ms(configs, op, how, rows)
}

fn family_json(ops: &[Op], configs: &[Config], members: &[usize], how: How) -> Value {
    let mut out = Map::new();
    out.insert(
        "members".into(),
        json!(
            members
                .iter()
                .map(|&i| ops[i].label.clone())
                .collect::<Vec<_>>()
        ),
    );
    let t1: f64 = members
        .iter()
        .filter_map(|&i| t_of(configs, i, how, 1))
        .sum();
    out.insert("t1_us".into(), json!(t1 * 1e3));
    let mut alphas = Map::new();
    let mut times = Map::new();
    for &rows in &ROWS[1..] {
        if how == How::Chunk4 && rows != 8 {
            continue;
        }
        let pairs: Vec<(f64, f64)> = members
            .iter()
            .filter_map(|&i| Some((t_of(configs, i, how, 1)?, t_of(configs, i, how, rows)?)))
            .collect();
        if pairs.len() != members.len() {
            continue;
        }
        times.insert(
            rows.to_string(),
            json!(pairs.iter().map(|p| p.1).sum::<f64>() * 1e3),
        );
        alphas.insert(rows.to_string(), json!(weighted_alpha(&pairs, rows)));
    }
    out.insert("t_us".into(), Value::Object(times));
    out.insert("alpha".into(), Value::Object(alphas));
    Value::Object(out)
}

/// The run-time receipt of which kernel each (op, rows) launches: for every op and every R, one
/// eager launch of layer 0's matrix, after a device-to-device marker copy whose size encodes the
/// (op, R) (`receipt_marker_bytes`). Run under `nsys profile` (tools/nsys_receipt_leg.sh), the
/// kernels between two markers are the kernels of that launch, and `tools/reduce_receipt.py` turns
/// them into names per op and R. A final marker closes the last launch. No timing is taken.
fn receipt_pass(env: &Env<'_>, ops: &[Op], kernel: &str) {
    let from = env.ctx.allocate(1 << 20).expect("marker source");
    let to = env.ctx.allocate(1 << 20).expect("marker destination");
    let marker = |bytes: usize| {
        env.ctx
            .copy_device(
                env.stream,
                from.range(0, bytes).expect("range"),
                to.range(0, bytes).expect("range"),
            )
            .expect("marker copy");
        env.stream.synchronize().expect("sync");
    };
    for (index, op) in ops.iter().enumerate() {
        op.fill(env, 7).expect("fill");
        for &rows in &ROWS {
            // One batched launch of R rows.
            marker(receipt_marker_bytes(index, rows));
            op.launch_matrix(env, 0, (0, rows), Which::A)
                .unwrap_or_else(|e| panic!("receipt launch {} R {rows}: {e:?}", op.label));
            env.stream.synchronize().expect("sync");
            // R separate single-row launches: the repeated execution the batched one replaces.
            if rows > 1 {
                marker(receipt_marker_bytes(index, SEP_CODE_OFFSET + rows));
                for row in 0..rows {
                    op.launch_matrix(env, 0, (row, 1), Which::A)
                        .unwrap_or_else(|e| panic!("receipt sep {} R {rows}: {e:?}", op.label));
                }
                env.stream.synchronize().expect("sync");
            }
        }
    }
    marker(receipt_marker_bytes(ops.len(), 1));
    let kinds: Vec<Value> = ops
        .iter()
        .map(|op| {
            let kind = match op.kind {
                Kind::Dense(_) => "dense",
                Kind::Pair(_) => "pair",
                Kind::Plain { .. } => "plain",
                Kind::Router { .. } => "router",
            };
            json!({"label": op.label, "kind": kind})
        })
        .collect();
    let record = json!({
        "schema": 1,
        "kind": "krow_receipt",
        "identity": identity("n/a (no KV in this test)", false, kernel),
        "ops": kinds,
        "rows": ROWS,
        "sep_code_offset": SEP_CODE_OFFSET,
        "marker": "device-to-device copy of 4096 * (op + 1) + 16 * code bytes before each launch group, code = rows for one batched launch of that many rows and 16 + rows for that many separate single-row launches; the last marker has op = ops.len(); layer 0 of the op",
    });
    emit(&record);
}

#[test]
#[ignore = "the 3090, held exclusively: loads the whole 4.00 bpw pack; run by name"]
fn krow_alpha_record() {
    refuse_switches();
    if !device_present("krow alpha record") {
        return;
    }
    let began = Instant::now();
    let rounds: usize = env_num("FIFI_KROW_ROUNDS", 30);
    let warm_from: usize = env_num("FIFI_KROW_WARM_FROM", 3);
    let deadline_s: f64 = env_num("FIFI_KROW_DEADLINE_S", 600.0);
    assert!(rounds > warm_from, "at least one warm round");
    let only: Vec<String> = std::env::var("FIFI_KROW_ONLY")
        .map(|text| text.split(',').map(|s| s.trim().to_owned()).collect())
        .unwrap_or_default();

    let loaded = load_pack();
    let ctx = loaded.projector.context();
    let stream = ctx.stream().expect("stream");
    let env = Env {
        ctx,
        stream: &stream,
    };
    assert_eq!(loaded.kernel, "wide1", "the served single-row kernel");
    let ops = build_ops(&loaded, ctx, &only).expect("ops");
    println!(
        "KROW_CONFIG ops={} load_s={:.1}",
        ops.len(),
        began.elapsed().as_secs_f64()
    );
    // The receipt mode: the kernel names and launch counts of each (op, R) under a profiler, no timing.
    if std::env::var("FIFI_KROW_RECEIPT").as_deref() == Ok("1") {
        receipt_pass(&env, &ops, loaded.kernel);
        return;
    }

    // The bitwise check, role by role. Every mismatch is kept.
    let mut checks = Vec::new();
    let mut all_mismatches: Vec<Value> = Vec::new();
    for op in &ops {
        let mut by_rows = Map::new();
        for &rows in &ROWS[1..] {
            let mut comparisons = 0;
            let mut mismatches = Vec::new();
            for seed in [1u64, 2] {
                let (n, found) = op.check(&env, rows, seed).expect("bitwise check");
                comparisons += n;
                mismatches.extend(found);
            }
            println!(
                "KROW_BITWISE role={} rows={rows} comparisons={comparisons} mismatches={}",
                op.label,
                mismatches.len()
            );
            by_rows.insert(
                rows.to_string(),
                json!({
                    "comparisons": comparisons,
                    "mismatch_count": mismatches.len(),
                    "mismatches": mismatches.iter().take(20).collect::<Vec<_>>(),
                }),
            );
            all_mismatches.extend(mismatches);
        }
        checks.push((op.label.clone(), Value::Object(by_rows)));
    }
    // The kernel that really ran: wide1 launched, no fallback.
    let counts = ctx.exl3_launch_counts();
    let kernel_check = require_kernel_ran(&counts, loaded.kernel);
    println!("KROW_LAUNCHES {} -> {kernel_check:?}", counts.describe());

    // The timing: one graph per configuration, rounds that rotate the order.
    let flush = Flush::new(ctx).expect("flush buffers");
    let watch = Stopwatch::new(ctx).expect("events");
    let mut configs = Vec::new();
    for (index, op) in ops.iter().enumerate() {
        op.fill(&env, 7).expect("fill");
        for (how, rows) in configs_of() {
            let graph = capture_graph(ctx, &stream, || {
                op.sweep(&env, rows, how.chunk(rows), Which::A)
            })
            .unwrap_or_else(|e| panic!("capture {} {} {rows}: {e:?}", op.label, how.name()));
            for _ in 0..3 {
                watch
                    .replay(ctx, &stream, Some(&flush), &graph)
                    .expect("warm replay");
            }
            configs.push(Config {
                op: index,
                how,
                rows,
                graph,
                ms: Vec::new(),
            });
        }
    }
    println!(
        "KROW_CONFIG graphs={} capture_done_s={:.1}",
        configs.len(),
        began.elapsed().as_secs_f64()
    );
    let order: Vec<usize> = (0..configs.len()).collect();
    let mut completed = 0;
    for round in 0..rounds {
        if began.elapsed().as_secs_f64() > deadline_s {
            break;
        }
        for &index in &rotated(&order, round * 7) {
            let ms = watch
                .replay(ctx, &stream, Some(&flush), &configs[index].graph)
                .expect("timed replay");
            if round >= warm_from {
                configs[index].ms.push(ms);
            }
        }
        completed += 1;
    }
    println!("KROW_CONFIG rounds_completed={completed} of {rounds}");

    // The record.
    let mut roles = Map::new();
    for (index, op) in ops.iter().enumerate() {
        let mut us = Map::new();
        let mut alphas = Map::new();
        for how in [How::Rows, How::Sep, How::Chunk4] {
            let mut u = Map::new();
            let mut a = Map::new();
            for config in configs.iter().filter(|c| c.op == index && c.how == how) {
                u.insert(config.rows.to_string(), us_summary(&config.ms));
                if config.rows > 1
                    && let (Some(t1), Some(tr)) =
                        (median_ms(&configs, index, How::Rows, 1), median(&config.ms))
                {
                    a.insert(config.rows.to_string(), json!(alpha(t1, tr, config.rows)));
                }
            }
            us.insert(how.name().into(), Value::Object(u));
            alphas.insert(how.name().into(), Value::Object(a));
        }
        let bitwise = checks
            .iter()
            .find(|(label, _)| *label == op.label)
            .map(|(_, v)| v.clone());
        roles.insert(
            op.label.clone(),
            json!({
                "layers": op.layers.len().max(1),
                "launches_per_token": op.count(),
                "shapes_k_n_bits_twice": op.shapes(),
                "us": us,
                "alpha": alphas,
                "bitwise": bitwise,
            }),
        );
    }
    let by_family = |wanted: &[Family], skip: &[&str]| -> Vec<usize> {
        ops.iter()
            .enumerate()
            .filter(|(_, op)| {
                op.family.is_some_and(|f| wanted.contains(&f)) && !skip.contains(&op.label.as_str())
            })
            .map(|(i, _)| i)
            .collect()
    };
    let dense = by_family(&[Family::Dense], &[]);
    let dense_no_head = by_family(&[Family::Dense], &["lm_head"]);
    let mut with_pair = by_family(&[Family::Dense], &["shared_gate", "shared_up"]);
    with_pair.extend(by_family(&[Family::SharedPair], &[]));
    let plain = by_family(&[Family::Plain], &[]);
    let router = by_family(&[Family::Router], &[]);
    let families = json!({
        "dense": family_json(&ops, &configs, &dense, How::Rows),
        "dense_sep": family_json(&ops, &configs, &dense, How::Sep),
        "dense_chunk4": family_json(&ops, &configs, &dense, How::Chunk4),
        "dense_no_head": family_json(&ops, &configs, &dense_no_head, How::Rows),
        "dense_with_shared_pair": family_json(&ops, &configs, &with_pair, How::Rows),
        "plain_gemv": family_json(&ops, &configs, &plain, How::Rows),
        "plain_gemv_chunk4": family_json(&ops, &configs, &plain, How::Chunk4),
        "moe_router": family_json(&ops, &configs, &router, How::Rows),
        "moe_router_chunk4": family_json(&ops, &configs, &router, How::Chunk4),
    });
    let kernel = loaded.kernel;
    let mut record = json!({
        "schema": 1,
        "kind": "krow_alpha",
        "identity": identity("n/a (no KV in this test)", false, kernel),
        "method": {
            "timing": "CUDA event pair around one replay of a graph of one whole-token sweep; L2 flushed by a 128 MiB device copy before the start event (the 0.35 ms flush hides the host cost of cudaGraphLaunch, which holds for graphs of 30 to about 1000 nodes); medians of the warm rounds",
            "headline_family": "dense_with_shared_pair is the step-faithful dense family (the production block launches the shared gate and up as one grouped pair); dense counts them as two separate launches",
            "rows": "R >= 2 rows through exl3_linear: for the EXL3 matrices grouped_project_rows<2|4|8> under wide1 (R = 3 rides <4>, R = 6 rides <8>; receipted by the nsys pass of FIFI_KROW_RECEIPT=1), R = 1 is the production wide1 body (grouped_project_small_wide<1>). The router and the plain GEMVs are NOT a rows kernel: m = 2..7 is the router decode kernel (m-grid), m = 8 the tiled kernel, which 4-row chunking avoids, so use their chunk4 figure at 8",
            "padded_rows": "native/resident/exl3.cu at the base: the dispatch on the largest group row count (1467-1477) runs grouped_project_small_wide<1> for 1 row, grouped_project_rows<2> for 2, <4> for 3 and 4, <8> for 5 to 8, so R = 3 launches the <4> instantiation and R = 6 the <8> one. The missing rows read x = 0 (line 1109, x[row] = row < a.rows ? xh[..] : 0) and are never stored (line 1170, if (row < R && row < a.rows)); every row has its own sums[row][col] and partial[w][row][col], so a padded row cannot change rows 0..R-1. Checked here per role, layer and seed: the bytes of out, xh and z past the real rows stay untouched (0xff) and rows 0..R-1 equal the single-row launches bit for bit. The instantiation and the launch count per R come from the nsys receipt (receipt_kernels), not from this reading of the code",
            "sep": "R single-row launches per layer back to back",
            "chunk4": "R = 8 as two 4-row launches per layer (the configuration that avoids the tiled kernel for the router and beta/alpha)",
            "rounds": rounds, "warm_from": warm_from, "rounds_completed": completed,
        },
        "single_row_launches": counts.describe(),
        "kernel_check": format!("{kernel_check:?}"),
        "roles": roles,
        "families": families,
        "bitwise_mismatch_total": all_mismatches.len(),
        "wall_s": began.elapsed().as_secs_f64(),
    });
    record["identity"]["shared_in_routed"]["step_effective"] = Value::Null;
    emit(&record);
    drop(configs);
    assert!(
        kernel_check.is_ok(),
        "the intended kernel did not run alone: {kernel_check:?}"
    );
    assert!(
        all_mismatches.is_empty(),
        "{} bitwise mismatches (listed in the record)",
        all_mismatches.len()
    );
}

#[test]
fn the_role_table_lists_the_shared_expert_matrices_separately() {
    let labels: Vec<&str> = DENSE_ROLES
        .iter()
        .chain(&PLAIN_ROLES)
        .map(|r| r.label)
        .collect();
    for want in [
        "shared_gate",
        "shared_up",
        "shared_down",
        "gdn_qkv",
        "gdn_z",
        "gdn_out",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_o",
        "gdn_beta",
        "gdn_alpha",
    ] {
        assert!(labels.contains(&want), "{want}");
    }
    let mut unique = labels.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), labels.len(), "labels are unique");
    assert_eq!(ROWS, [1, 2, 3, 4, 6, 8]);
}
