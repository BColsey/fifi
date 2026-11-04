//! CUDA Graph A/B microbenchmark of the Qwen3.8 Flash-Next MoE block on one GPU, on the real
//! weights of one layer. Measurement only: it adds no execution path and changes no executor or
//! default.
//!
//! Arm A queues the shared EXL3 MoE block (`run_exl3_moe` with device picks: router, 10 picks of
//! 512, shared expert, combine) eagerly. Arm B captures that same call once and replays the graph.
//! Both arms use the same weights, buffers, input rows and stream. The id-indexed launches read the
//! router's ids when the kernels run, so one graph serves any routing: the graph is not a
//! static-pick ceiling. Every expert of the layer is resident in device memory.
//!
//! The split block is the same work in three pieces, so that a miss can be placed between them:
//! the router; the resident picks (the first `10 - m` in pick order) with the shared expert; and,
//! after the compute stream waits for the copies of `m` experts, the missed picks (the last `m`)
//! with the combine. The missed picks still read the resident bank. The copies go to a staging
//! area that no kernel reads, so the copy-then-compute dependency and the stream traffic are real,
//! but the bytes the tail reads are not the copied ones. Which picks miss is fixed by position;
//! the expert ids are still chosen on the device.
//!
//! Copy scheduling, for a layer with misses:
//! - host: the host waits for the router to finish (an event synchronize, standing in for the
//!   mailbox poll), then queues the copies. One host thread does everything, so in arm A the
//!   copies are queued only after the resident launches.
//! - device: the transfer stream waits for the router on the device and the copies were queued
//!   before. This is the lower bound with no host round trip; a real host cannot know the misses
//!   before the router has run. The host queues those copies before the resident piece, so their
//!   queueing delays the resident launch in both arms.
//!
//! Phases (one `RESBENCH` JSON line per case; times in microseconds with mean, P50, P95, P99):
//! - graph_build: capture plus instantiate time, node count and device memory (the drop in free
//!   memory) of the block graph, of a 48-block token graph and of the split pieces; the first
//!   replay, and the memory of all graphs after it.
//! - graph_check: eager, graph, eager split and graph split outputs for 8 input rows, 200 graph
//!   replays of one row, and the token graph. Before every run the output row is cleared, the ids
//!   are set to -1 and every intermediate the block writes is set to NaN, so a skipped launch
//!   shows. Bitwise equality and relative L2 against arm A; the router status and the arithmetic
//!   flag stay 0. Any relative L2 above 1e-5, or rows that all pick the same experts, fail the run.
//!   The gate's copy and one layer's ten miss copies are read back and compared with the store.
//! - graph_block: one block, isolated (each block waits for the host to queue it) and gated (the
//!   compute stream waits behind a 24-expert copy while the host queues, so the device runs the
//!   queued kernels back to back: device work without host launch gaps; `late` counts the reps
//!   where the host was still queueing when the gate opened). Then 48 blocks back to back, the
//!   token graph, and 48 split layers without misses.
//! - graph_miss_layer: one split layer with m = 0..10 misses, per arm and copy scheduling: layer
//!   time, router, resident part, the stall waiting for copies, the lag from the router's end to
//!   the first copy, and the copy span.
//! - graph_trace: whole tokens of 48 split layers with the per-layer miss counts of the LRU replay
//!   (`FIFI_FLASH_NEXT_MISS_TRACES`, `name=path,...`; one line per token, one count per layer), and
//!   the same tokens with every expert resident. Per-token samples are printed for the report.
//!
//! - graph_variants (only when named in `FIFI_GRAPH_BENCH_PHASES`): the block under the production
//!   id-indexed launch and each bench-only prototype of it (`IndexedPrototype`: the DP4A decode,
//!   one launch of the 20 gate and up pairs, both), all on the served single-row kernel (`wide1`).
//!   Each variant's eager and graph outputs must equal the production eager outputs bit for bit
//!   on 4 rows; then graph isolated, graph gated, eager isolated and 48 graph launches back to
//!   back, the variants taking turns in every trial. `FIFI_EXL3_SINGLE_GATHER` and
//!   `FIFI_EXL3_INDEXED_PROTOTYPE` must be unset: the phase chooses the kernels itself.
//!
//! `FIFI_EXL3_INDEXED_PROTOTYPE` (`dp4a`, `pairs20`, `both`) runs every other phase on that
//! prototype, so its miss and trace costs can be set against a run without it.
//!
//! `FIFI_GRAPH_BENCH_PHASES` (comma list of check, block, variants, miss, trace; all but variants
//! by default) limits the phases; the build phase always runs. Run only inside an exclusive GPU window.
//!
//! What the numbers leave out: the classify kernel and the mailbox of a real miss path, the
//! resident bank's eviction and the slot table; the other layers' attention and recurrent work
//! (a token here is 48 MoE blocks of the same layer); and copies into the slots the kernels read.
//! The arms run in a fixed order (eager first in graph_block, graph first in graph_miss_layer and
//! graph_trace), so slow clock or thermal drift falls on one side; the two runs of a window show it.

use std::{ops::Range, time::Instant};

use super::{
    exl3_cuda_tests::{Router, input, upload},
    residency_bench::{
        Block, EXPERT_BYTES, Res, clock_mark, device_us, emit, fail, open_layer, read_store,
        resolve_block, us, whole,
    },
};
use crate::{
    family::{
        gguf_model::Role,
        project::cuda::{
            CudaProjector, CudaWeight, CudaWeights,
            exl3::{Exl3Bank, Exl3Io, Exl3Weight},
        },
        qwen36::cuda::{
            exl3_moe::Exl3MoeBufs,
            moe::{Launch, MoeDims},
        },
    },
    kernels::cuda::{
        CapturePlan, CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream, GraphExec,
        PinnedBuf, TimingEvent,
        prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape},
        resident::{
            exl3::{IndexedPrototype, SINGLE_GATHER_VAR, SingleRow},
            moe_combine::{CombineCall, CombineOut, CombineShape, SharedTerm},
            router::{RouterCall, RouterOut, RouterShape, RouterWeights},
        },
    },
};

const F32: usize = size_of::<f32>();
/// MoE layers of a Flash-Next token.
const LAYERS: usize = 48;
/// Staging slots of the miss copies; the gate copy fills all of them at once.
const COPY_SLOTS: usize = 16;
const GATE_EXPERTS: usize = 24;
const CAPTURES: usize = 20;
const CHECK_ROWS: usize = 8;
const REPEATS: usize = 200;
const WARM: usize = 30;
const BLOCK_REPS: usize = 200;
const TOKEN_WARM: usize = 10;
const TOKEN_REPS: usize = 60;
const MISS_WARM: usize = 20;
const MISS_REPS: usize = 150;
const TRACE_WARM: usize = 20;
/// The largest relative L2 distance from arm A that the check accepts.
const CHECK_LIMIT: f64 = 1e-5;

// Timing marks of one layer or token.
const START: usize = 0;
const ROUTED: usize = 1;
const RESIDENT: usize = 2;
const TAIL: usize = 3;
const COPY0: usize = 4;
const COPY1: usize = 5;
const END: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Exec {
    Eager,
    Graph,
}

impl Exec {
    fn tag(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Graph => "graph",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sched {
    Host,
    Device,
}

impl Sched {
    fn tag(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Device => "device",
        }
    }
}

/// One piece of the split block; the count is the number of missed picks.
#[derive(Clone, Copy)]
enum Piece {
    Router,
    Resident(usize),
    Tail(usize),
}

// ------------------------------------------------------------------ the split block

/// The weights of the block as the split pieces (and the projection benchmark) read them.
pub(super) struct Parts<'w> {
    router: &'w DeviceBuffer,
    router_shared: &'w DeviceBuffer,
    pub(super) gate: &'w Exl3Bank,
    pub(super) up: &'w Exl3Bank,
    pub(super) down: &'w Exl3Bank,
    shared_gate: &'w Exl3Weight,
    shared_up: &'w Exl3Weight,
    shared_down: &'w Exl3Weight,
}

impl<'w> Parts<'w> {
    /// The tensors of `layer`, with the shape checks of `Exl3MoeWeights::resolve_by`.
    pub(super) fn resolve(weights: &'w CudaWeights, layer: usize, moe: &MoeDims) -> Res<Self> {
        let keys = super::exl3_cuda_tests::keys(layer);
        let get = |role: Role| -> Res<&'w CudaWeight> {
            let key = keys
                .iter()
                .find(|(candidate, _)| *candidate == role)
                .map(|(_, key)| key.as_str())
                .ok_or_else(|| fail(format!("no key for {role:?}")))?;
            weights
                .get(key)
                .ok_or_else(|| fail(format!("no tensor {key}")))
        };
        let raw = |role: Role, values: usize| match get(role)? {
            CudaWeight::Raw { buffer, .. } if buffer.bytes() == values * F32 => Ok(buffer),
            _ => Err(fail(format!(
                "{role:?} is not a raw F32 tensor of {values}"
            ))),
        };
        let bank = |role: Role, shape: [usize; 2]| match get(role)? {
            CudaWeight::Exl3Bank { bank, .. }
                if bank.count() == moe.n_expert && bank.shape() == shape =>
            {
                Ok(bank)
            }
            _ => Err(fail(format!("{role:?} is not a bank of {shape:?}"))),
        };
        let single = |role: Role, shape: [usize; 2]| match get(role)? {
            CudaWeight::Exl3 { weight, .. } if weight.shape == shape => Ok(weight),
            _ => Err(fail(format!("{role:?} is not an EXL3 matrix of {shape:?}"))),
        };
        let (h, i, s) = (moe.hidden, moe.inter, moe.shared_inter);
        Ok(Self {
            router: raw(Role::FfnGateInp, moe.n_expert * h)?,
            router_shared: raw(Role::FfnGateInpShexp, h)?,
            gate: bank(Role::FfnGateExps, [h, i])?,
            up: bank(Role::FfnUpExps, [h, i])?,
            down: bank(Role::FfnDownExps, [i, h])?,
            shared_gate: single(Role::FfnGateShexp, [h, s])?,
            shared_up: single(Role::FfnUpShexp, [h, s])?,
            shared_down: single(Role::FfnDownShexp, [s, h])?,
        })
    }
}

/// The block's weights, buffers and dimensions, for the split pieces.
struct Composed<'a> {
    parts: Parts<'a>,
    bufs: Exl3MoeBufs<'a>,
    dims: MoeDims,
}

/// Row `index` of `width` bytes of `range`.
fn row<'b>(range: &DeviceRange<'b>, width: usize, index: usize) -> Res<DeviceRange<'b>> {
    range.buffer().range(range.offset() + index * width, width)
}

/// Rows `rows` of `width` bytes of `range`.
fn rows<'b>(range: &DeviceRange<'b>, width: usize, rows: &Range<usize>) -> Res<DeviceRange<'b>> {
    range
        .buffer()
        .range(range.offset() + rows.start * width, rows.len() * width)
}

fn route(l: &Launch<'_>, c: &Composed<'_>) -> Res<()> {
    let (b, d) = (&c.bufs.route, &c.dims);
    l.ctx.router(
        l.stream,
        RouterCall {
            shape: RouterShape {
                m: d.rows,
                hidden: d.hidden,
                n_expert: d.n_expert,
                top_k: d.top_k,
            },
            x: b.norm_row,
            weights: RouterWeights {
                w: whole(c.parts.router)?,
                w_shared: Some(whole(c.parts.router_shared)?),
            },
            logits: b.logits,
            out: RouterOut {
                ids: b.ids,
                weights: b.weights,
                shared_gate: Some(b.shared_gate),
                status: b.status,
            },
        },
    )
}

fn swiglu(
    l: &Launch<'_>,
    gu: DeviceRange<'_>,
    act: DeviceRange<'_>,
    (rows, width): (usize, usize),
) -> Res<()> {
    l.ctx.swiglu_rows(
        l.stream,
        SwigluRowsCall {
            shape: SwigluRowsShape {
                rows,
                n_ff: width,
                up_off: width,
                ld_in: 2 * width,
                ld_out: width,
            },
            gu,
            act,
        },
    )
}

/// One pair of an id-indexed launch: the member is `ids[slot]`, read on the device.
struct Indexed<'a> {
    bank: &'a Exl3Bank,
    slot: usize,
    x: DeviceRange<'a>,
    out: DeviceRange<'a>,
}

/// The id-indexed group `pairs` (one shape) in launches of at most `exl3_indexed_capacity`
/// pairs, each from scratch row 0, as in the shared block.
fn indexed(l: &Launch<'_>, c: &Composed<'_>, pairs: &[Indexed<'_>]) -> Res<()> {
    let s = &c.bufs.scratch;
    for chunk in pairs.chunks(l.ctx.exl3_indexed_capacity()) {
        let [k, n] = chunk[0].bank.shape();
        let mut calls = Vec::with_capacity(chunk.len());
        for (index, pair) in chunk.iter().enumerate() {
            let io = Exl3Io {
                x: pair.x,
                xh: row(&s.xh, k * F32, index)?,
                z: row(&s.z, n * F32, index)?,
                out: pair.out,
                flag: s.flag,
                rows: 1,
                fused: false,
            };
            let slot = u32::try_from(pair.slot).map_err(|_| fail("a pick slot past u32"))?;
            calls.push(pair.bank.span()?.indexed_call(&io, c.bufs.route.ids, slot));
        }
        l.ctx
            .exl3_grouped_indexed(l.stream, &calls, l.ctx.exl3_indexed_cols(n)?)?;
    }
    Ok(())
}

/// The routed experts of the pick slots `picks`: gate and up, the SwiGLU, down. The layout of the
/// rows is the shared block's, so the combine reads every pick where the whole block puts it.
fn experts(l: &Launch<'_>, c: &Composed<'_>, picks: Range<usize>) -> Res<()> {
    if picks.is_empty() {
        return Ok(());
    }
    let (h, i, b) = (c.dims.hidden, c.dims.inter, &c.bufs);
    let mut group = Vec::with_capacity(2 * picks.len());
    for slot in picks.clone() {
        for (half, bank) in [c.parts.gate, c.parts.up].into_iter().enumerate() {
            group.push(Indexed {
                bank,
                slot,
                x: b.route.norm_row,
                out: row(&b.expert_gu, i * F32, 2 * slot + half)?,
            });
        }
    }
    indexed(l, c, &group)?;
    let gu = rows(&b.expert_gu, 2 * i * F32, &picks)?;
    let act = rows(&b.expert_act, i * F32, &picks)?;
    swiglu(l, gu, act, (picks.len(), i))?;
    let mut group = Vec::with_capacity(picks.len());
    for slot in picks {
        group.push(Indexed {
            bank: c.parts.down,
            slot,
            x: row(&b.expert_act, i * F32, slot)?,
            out: row(&b.expert_down, h * F32, slot)?,
        });
    }
    indexed(l, c, &group)
}

/// The shared expert: gate and up in one grouped launch, the SwiGLU, down.
fn shared(l: &Launch<'_>, c: &Composed<'_>) -> Res<()> {
    let (h, s, b) = (c.dims.hidden, c.dims.shared_inter, &c.bufs);
    let views = [c.parts.shared_gate.view()?, c.parts.shared_up.view()?];
    let mut ios = Vec::with_capacity(2);
    for half in 0..2 {
        ios.push(Exl3Io {
            x: b.route.norm_row,
            xh: row(&b.scratch.xh, h * F32, half)?,
            z: row(&b.scratch.z, s * F32, half)?,
            out: row(&b.shared_gu, s * F32, half)?,
            flag: b.scratch.flag,
            rows: 1,
            fused: false,
        });
    }
    let calls: Vec<_> = views
        .iter()
        .zip(&ios)
        .map(|(view, io)| view.linear_call(io))
        .collect();
    l.ctx.exl3_grouped_step(l.stream, &calls)?;
    swiglu(l, b.shared_gu, b.shared_act, (1, s))?;
    let view = c.parts.shared_down.view()?;
    let io = Exl3Io {
        x: b.shared_act,
        xh: row(&b.scratch.xh, s * F32, 0)?,
        z: row(&b.scratch.z, h * F32, 0)?,
        out: b.shared_out,
        flag: b.scratch.flag,
        rows: 1,
        fused: false,
    };
    l.ctx.exl3_grouped_step(l.stream, &[view.linear_call(&io)])
}

fn combine(l: &Launch<'_>, c: &Composed<'_>) -> Res<()> {
    let (b, d) = (&c.bufs, &c.dims);
    l.ctx.moe_combine(
        l.stream,
        CombineCall {
            shape: CombineShape {
                m: d.rows,
                n: d.hidden,
                top_k: d.top_k,
                ld_down: d.hidden,
            },
            res: b.route.resid_in,
            down: b.expert_down,
            weights: b.route.weights,
            shared: Some(SharedTerm {
                gate: b.route.shared_gate,
                row: b.shared_out,
            }),
            out: CombineOut::To(b.route.resid_out),
        },
    )
}

fn queue_piece(l: &Launch<'_>, c: &Composed<'_>, piece: Piece) -> Res<()> {
    let k = c.dims.top_k;
    match piece {
        Piece::Router => route(l, c),
        Piece::Resident(m) => {
            experts(l, c, 0..k - m)?;
            shared(l, c)
        }
        Piece::Tail(m) => {
            experts(l, c, k - m..k)?;
            combine(l, c)
        }
    }
}

// ------------------------------------------------------------------ the rig

struct Graphs {
    full: GraphExec<'static>,
    token: GraphExec<'static>,
    router: GraphExec<'static>,
    resident: Vec<GraphExec<'static>>,
    tail: Vec<GraphExec<'static>>,
}

/// One run's output of the block.
struct Output {
    resid: Vec<f32>,
    ids: Vec<i32>,
    flag: u32,
    status: u32,
}

/// Device state of the benchmark. The graphs are dropped first, the streams last.
struct Rig<'a> {
    graphs: Option<Graphs>,
    ctx: &'a CudaContext,
    block: &'a Block<'a>,
    composed: Composed<'a>,
    /// Page-locked copy of the layer's experts: the source of every copy.
    pinned: PinnedBuf,
    /// The copies' destination, read by no kernel.
    staging: DeviceBuffer,
    routed: DeviceEvent,
    copied: DeviceEvent,
    opened: DeviceEvent,
    marks: Vec<TimingEvent>,
    next: usize,
    compute: DeviceStream,
    transfer: DeviceStream,
}

impl Rig<'_> {
    fn launch(&self) -> Launch<'_> {
        Launch {
            ctx: self.ctx,
            stream: &self.compute,
        }
    }

    fn graphs(&self) -> Res<&Graphs> {
        self.graphs
            .as_ref()
            .ok_or_else(|| fail("the graphs are not built"))
    }

    fn graph(&self, piece: Piece) -> Res<&GraphExec<'static>> {
        let g = self.graphs()?;
        let found = match piece {
            Piece::Router => Some(&g.router),
            Piece::Resident(m) => g.resident.get(m),
            Piece::Tail(m) => g.tail.get(m),
        };
        found.ok_or_else(|| fail("no graph for that piece"))
    }

    /// Queues the whole block.
    fn full(&self, exec: Exec) -> Res<()> {
        match exec {
            Exec::Eager => self.block.queue(self.ctx, &self.compute),
            Exec::Graph => self.graphs()?.full.launch(&self.compute),
        }
    }

    fn piece(&self, exec: Exec, piece: Piece) -> Res<()> {
        match exec {
            Exec::Eager => queue_piece(&self.launch(), &self.composed, piece),
            Exec::Graph => self.graph(piece)?.launch(&self.compute),
        }
    }

    /// Queues one copy per expert of `m` experts into the staging slots, between two marks.
    fn copies(&mut self, m: usize, timed: bool) -> Res<()> {
        if timed {
            self.transfer.record_timing(&self.marks[COPY0])?;
        }
        for j in 0..m {
            let id = self.next % self.composed.dims.n_expert;
            self.next += 1;
            let dst = self
                .staging
                .range((j % COPY_SLOTS) * EXPERT_BYTES, EXPERT_BYTES)?;
            self.ctx
                .copy_from_pinned(&self.transfer, &mut self.pinned, id * EXPERT_BYTES, dst)?;
        }
        if timed {
            self.transfer.record_timing(&self.marks[COPY1])?;
        }
        self.transfer.record(&self.copied)
    }

    /// Queues one split layer with `m` missed picks; `timed` records the stage marks.
    fn layer(&mut self, exec: Exec, sched: Sched, m: usize, timed: bool) -> Res<()> {
        self.piece(exec, Piece::Router)?;
        if timed {
            self.compute.record_timing(&self.marks[ROUTED])?;
        }
        if m > 0 {
            self.compute.record(&self.routed)?;
            if sched == Sched::Device {
                self.transfer.wait(&self.routed)?;
                self.copies(m, timed)?;
            }
        }
        self.piece(exec, Piece::Resident(m))?;
        if timed {
            self.compute.record_timing(&self.marks[RESIDENT])?;
        }
        if m > 0 {
            if sched == Sched::Host {
                self.routed.synchronize()?;
                self.copies(m, timed)?;
            }
            self.compute.wait(&self.copied)?;
        }
        if timed {
            self.compute.record_timing(&self.marks[TAIL])?;
        }
        self.piece(exec, Piece::Tail(m))
    }

    /// Holds the compute stream behind one copy of 24 experts (about 1.7 ms) so that the work the
    /// host queues next runs back to back once it opens; marks the copy's span. Returns the instant
    /// just before the copy was queued: the copy cannot start earlier.
    fn gate(&mut self) -> Res<Instant> {
        self.transfer.record_timing(&self.marks[COPY0])?;
        let dst = self.staging.range(0, GATE_EXPERTS * EXPERT_BYTES)?;
        let queued = Instant::now();
        self.ctx
            .copy_from_pinned(&self.transfer, &mut self.pinned, 0, dst)?;
        self.transfer.record_timing(&self.marks[COPY1])?;
        self.transfer.record(&self.opened)?;
        self.compute.wait(&self.opened)?;
        Ok(queued)
    }

    /// Waits for both streams.
    fn settle(&mut self) -> Res<()> {
        self.compute.synchronize()?;
        self.pinned.synchronize(&self.transfer)
    }

    fn mark_us(&self, from: usize, to: usize) -> Res<f64> {
        device_us(&self.marks[from], &self.marks[to])
    }

    fn set_row(&self, seed: usize) -> Res<()> {
        let bytes: Vec<u8> = input(self.composed.dims.hidden, seed)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let row = self.composed.bufs.route.norm_row;
        row.buffer().write_at(&self.compute, row.offset(), &bytes)
    }

    /// Clears the output row, sets the ids to -1 and every intermediate the block writes (pick
    /// weights, shared gate, expert and shared-expert rows) to NaN, runs `body`, waits and reads
    /// the outputs back. A skipped launch leaves NaN or -1 behind, which `sound` refuses.
    fn run_once(&mut self, body: impl FnOnce(&mut Self) -> Res<()>) -> Res<Output> {
        let d = self.composed.dims;
        let b = &self.composed.bufs;
        let out = b.route.resid_out;
        out.buffer()
            .write_at(&self.compute, out.offset(), &vec![0u8; d.hidden * F32])?;
        let ids = b.route.ids;
        ids.buffer()
            .write_at(&self.compute, ids.offset(), &vec![0xffu8; d.top_k * 4])?;
        let poisoned = [
            b.route.weights,
            b.route.shared_gate,
            b.expert_gu,
            b.expert_act,
            b.expert_down,
            b.shared_gu,
            b.shared_act,
            b.shared_out,
        ];
        for range in poisoned {
            let nan = f32::NAN.to_le_bytes().repeat(range.bytes() / F32);
            range
                .buffer()
                .write_at(&self.compute, range.offset(), &nan)?;
        }
        body(self)?;
        self.settle()?;
        let b = &self.composed.bufs;
        let read = |range: DeviceRange<'_>, bytes: usize| -> Res<Vec<u8>> {
            let mut raw = vec![0u8; bytes];
            range
                .buffer()
                .read_at(&self.compute, range.offset(), &mut raw)?;
            Ok(raw)
        };
        let words = |raw: Vec<u8>| -> Vec<[u8; 4]> { raw.as_chunks::<4>().0.to_vec() };
        let first = |raw: Vec<u8>| u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        Ok(Output {
            resid: words(read(b.route.resid_out, d.hidden * F32)?)
                .into_iter()
                .map(f32::from_le_bytes)
                .collect(),
            ids: words(read(b.route.ids, d.top_k * 4)?)
                .into_iter()
                .map(i32::from_le_bytes)
                .collect(),
            flag: first(read(b.scratch.flag, 4)?),
            status: first(read(b.route.status, 4)?),
        })
    }
}

fn capture(
    rig: &Rig<'_>,
    body: impl FnOnce(&Rig<'_>) -> Res<()>,
) -> Res<(GraphExec<'static>, f64)> {
    let start = Instant::now();
    let plan = CapturePlan {
        pins: &[],
        holds: Vec::new(),
    };
    let (graph, ()) = rig.ctx.capture(&rig.compute, plan, || body(rig))?;
    Ok((graph, us(start)))
}

fn free_bytes(rig: &Rig<'_>) -> Res<usize> {
    Ok(rig.ctx.memory_info()?.free)
}

// ------------------------------------------------------------------ phases

/// Captures `body` and returns the graph, the capture plus instantiate microseconds and the
/// device memory it took (the drop in free memory).
fn capture_measured(
    rig: &Rig<'_>,
    body: impl FnOnce(&Rig<'_>) -> Res<()>,
) -> Res<(GraphExec<'static>, f64, usize)> {
    let before = free_bytes(rig)?;
    let (graph, took) = capture(rig, body)?;
    Ok((graph, took, before.saturating_sub(free_bytes(rig)?)))
}

fn queue_block(r: &Rig<'_>) -> Res<()> {
    r.block.queue(r.ctx, &r.compute)
}

/// The block graph, captured `CAPTURES` times for the capture cost; the first is kept.
fn capture_block(rig: &Rig<'_>) -> Res<GraphExec<'static>> {
    let (full, first, vram) = capture_measured(rig, queue_block)?;
    let mut times = vec![first];
    for _ in 1..CAPTURES {
        times.push(capture(rig, queue_block)?.1);
    }
    let extra = format!(",\"nodes\":{},\"vram_bytes\":{vram}", full.nodes());
    emit(
        "graph_build",
        "block_capture_instantiate",
        "us",
        &times,
        &extra,
    );
    Ok(full)
}

/// One graph of a token's 48 blocks.
fn capture_token(rig: &Rig<'_>) -> Res<GraphExec<'static>> {
    let (token, took, vram) =
        capture_measured(rig, |r| (0..LAYERS).try_for_each(|_| queue_block(r)))?;
    let extra = format!(",\"nodes\":{},\"vram_bytes\":{vram}", token.nodes());
    emit(
        "graph_build",
        "token48_capture_instantiate",
        "us",
        &[took],
        &extra,
    );
    Ok(token)
}

/// The router graph and the resident and tail graphs for every miss count.
fn capture_split(
    rig: &Rig<'_>,
) -> Res<(
    GraphExec<'static>,
    Vec<GraphExec<'static>>,
    Vec<GraphExec<'static>>,
)> {
    let k = rig.composed.dims.top_k;
    let before = free_bytes(rig)?;
    let start = Instant::now();
    let piece = |piece: Piece| move |r: &Rig<'_>| queue_piece(&r.launch(), &r.composed, piece);
    let router = capture(rig, piece(Piece::Router))?.0;
    let mut resident = Vec::with_capacity(k + 1);
    let mut tail = Vec::with_capacity(k + 1);
    for m in 0..=k {
        resident.push(capture(rig, piece(Piece::Resident(m)))?.0);
        tail.push(capture(rig, piece(Piece::Tail(m)))?.0);
    }
    let took = us(start);
    let vram = before.saturating_sub(free_bytes(rig)?);
    let nodes = |graphs: &[GraphExec<'static>]| {
        let list: Vec<String> = graphs.iter().map(|g| g.nodes().to_string()).collect();
        list.join(",")
    };
    let extra = format!(
        ",\"router_nodes\":{},\"resident_nodes\":[{}],\"tail_nodes\":[{}],\"vram_bytes\":{vram}",
        router.nodes(),
        nodes(&resident),
        nodes(&tail)
    );
    emit(
        "graph_build",
        "split_capture_instantiate_all",
        "us",
        &[took],
        &extra,
    );
    Ok((router, resident, tail))
}

/// Warms every kernel, captures every graph, replays the block graph once (its first launch
/// uploads it) and keeps the graphs in the rig.
fn build_phase(rig: &mut Rig<'_>) -> Res<()> {
    let k = rig.composed.dims.top_k;
    // A first launch loads its kernel's module, which a capture refuses: warm every piece first.
    rig.full(Exec::Eager)?;
    for m in 0..=k {
        for piece in [Piece::Router, Piece::Resident(m), Piece::Tail(m)] {
            rig.piece(Exec::Eager, piece)?;
        }
    }
    rig.settle()?;
    let before = free_bytes(rig)?;
    let full = capture_block(rig)?;
    let token = capture_token(rig)?;
    let (router, resident, tail) = capture_split(rig)?;
    let start = Instant::now();
    rig.compute.record_timing(&rig.marks[START])?;
    full.launch(&rig.compute)?;
    rig.compute.record_timing(&rig.marks[END])?;
    rig.settle()?;
    let wall = us(start);
    let vram = before.saturating_sub(free_bytes(rig)?);
    let extra = format!(",\"wall_us\":{wall:.1},\"vram_bytes_all_graphs_after\":{vram}");
    emit(
        "graph_build",
        "block_first_replay",
        "us",
        &[rig.mark_us(START, END)?],
        &extra,
    );
    rig.graphs = Some(Graphs {
        full,
        token,
        router,
        resident,
        tail,
    });
    Ok(())
}

/// Bitwise and relative-L2 agreement of one arm with arm A.
#[derive(Default)]
struct Agreement {
    compared: usize,
    exact: usize,
    worst: f64,
}

impl Agreement {
    fn add(&mut self, want: &Output, got: &Output) {
        let same = want
            .resid
            .iter()
            .zip(&got.resid)
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && want.ids == got.ids;
        let (diff, norm) =
            want.resid
                .iter()
                .zip(&got.resid)
                .fold((0.0f64, 0.0f64), |(d, n), (&a, &b)| {
                    let (a, b) = (f64::from(a), f64::from(b));
                    (d + (a - b) * (a - b), n + a * a)
                });
        let rel = (diff / norm.max(f64::MIN_POSITIVE)).sqrt();
        let rel = if want.ids == got.ids && !rel.is_nan() {
            rel
        } else {
            f64::INFINITY
        };
        self.compared += 1;
        self.exact += usize::from(same);
        self.worst = self.worst.max(rel);
    }

    fn json(&self) -> String {
        format!(
            "{{\"n\":{},\"exact\":{},\"worst_rel_l2\":{:e}}}",
            self.compared, self.exact, self.worst
        )
    }
}

/// `Err` unless the run wrote a finite, non-zero row, valid ids, and left the flags clear.
fn sound(out: &Output, n_expert: usize) -> Res<()> {
    let ids_ok = out
        .ids
        .iter()
        .all(|&id| usize::try_from(id).is_ok_and(|id| id < n_expert));
    let row_ok = out.resid.iter().all(|v| v.is_finite()) && out.resid.iter().any(|&v| v != 0.0);
    if !ids_ok || !row_ok || out.flag != 0 || out.status != 0 {
        return Err(fail(format!(
            "graph check: ids {:?}, finite non-zero row {row_ok}, flag {}, status {}",
            out.ids, out.flag, out.status
        )));
    }
    Ok(())
}

/// Arm agreement with arm A over the check rows, the repeats and the token graph, and the
/// number of distinct pick sets the rows chose.
fn equivalence(rig: &mut Rig<'_>) -> Res<([Agreement; 5], usize)> {
    let (k, n_expert) = (rig.composed.dims.top_k, rig.composed.dims.n_expert);
    let mut agree: [Agreement; 5] = Default::default();
    let mut pick_sets: Vec<Vec<i32>> = Vec::new();
    let mut check = |slot: usize, want: &Output, got: Output| -> Res<()> {
        sound(&got, n_expert)?;
        agree[slot].add(want, &got);
        Ok(())
    };
    for seed in 0..CHECK_ROWS {
        rig.set_row(seed)?;
        let eager = rig.run_once(|r| r.full(Exec::Eager))?;
        sound(&eager, n_expert)?;
        check(0, &eager, rig.run_once(|r| r.full(Exec::Graph))?)?;
        for m in 0..=k {
            for (slot, exec) in [(1, Exec::Eager), (2, Exec::Graph)] {
                let split = rig.run_once(|r| {
                    for piece in [Piece::Router, Piece::Resident(m), Piece::Tail(m)] {
                        r.piece(exec, piece)?;
                    }
                    Ok(())
                })?;
                check(slot, &eager, split)?;
            }
        }
        let mut set = eager.ids.clone();
        set.sort_unstable();
        if !pick_sets.contains(&set) {
            pick_sets.push(set);
        }
    }
    rig.set_row(0)?;
    let first = rig.run_once(|r| r.full(Exec::Graph))?;
    for _ in 0..REPEATS {
        check(3, &first, rig.run_once(|r| r.full(Exec::Graph))?)?;
    }
    let token = rig.run_once(|r| r.graphs()?.token.launch(&r.compute))?;
    check(4, &first, token)?;
    Ok((agree, pick_sets.len()))
}

/// Reads `experts` staged experts back from slot `slot` on and compares each with the store's
/// expert `ids(j)`; `Err` on the first difference.
fn compare_staged(
    rig: &Rig<'_>,
    host: &[u8],
    experts: usize,
    id: impl Fn(usize) -> usize,
) -> Res<()> {
    let mut got = vec![0u8; EXPERT_BYTES];
    for j in 0..experts {
        rig.staging
            .read_at(&rig.compute, j * EXPERT_BYTES, &mut got)?;
        let want = &host[id(j) * EXPERT_BYTES..(id(j) + 1) * EXPERT_BYTES];
        if got != want {
            return Err(fail(format!(
                "copy check: staging slot {j} is not expert {}",
                id(j)
            )));
        }
    }
    Ok(())
}

/// The gate's copy and one layer's miss copies (host scheduling, every pick missed) land in the
/// staging area byte for byte; returns the number of experts compared.
fn copy_check(rig: &mut Rig<'_>, host: &[u8]) -> Res<usize> {
    let (k, n) = (rig.composed.dims.top_k, rig.composed.dims.n_expert);
    rig.gate()?;
    rig.settle()?;
    compare_staged(rig, host, GATE_EXPERTS, |j| j)?;
    let first = rig.next;
    rig.layer(Exec::Graph, Sched::Host, k, false)?;
    rig.settle()?;
    compare_staged(rig, host, k, |j| (first + j) % n)?;
    Ok(GATE_EXPERTS + k)
}

fn check_phase(rig: &mut Rig<'_>, host: &[u8]) -> Res<()> {
    let names = ["graph", "split_eager", "split_graph", "repeat", "token"];
    let (agree, pick_sets) = equivalence(rig)?;
    let copies = copy_check(rig, host)?;
    let fields: Vec<String> = names
        .iter()
        .zip(&agree)
        .map(|(name, a)| format!("\"{name}\":{}", a.json()))
        .collect();
    println!(
        "RESBENCH {{\"phase\":\"graph_check\",\"case\":\"equivalence\",\"rows\":{CHECK_ROWS},\
         \"distinct_pick_sets\":{pick_sets},\"copies_checked\":{copies},{}}}",
        fields.join(",")
    );
    if let Some((name, a)) = names
        .iter()
        .zip(&agree)
        .find(|(_, a)| a.worst > CHECK_LIMIT)
    {
        return Err(fail(format!(
            "graph check: {name} is {:e} from arm A (limit {CHECK_LIMIT:e})",
            a.worst
        )));
    }
    if pick_sets < 2 {
        return Err(fail("graph check: every row picked the same experts"));
    }
    Ok(())
}

/// One rep of one block: device interval, host queue time, wall time, and (gated) whether the
/// host was still queueing when the gate opened.
struct BlockRep {
    device: f64,
    queue: f64,
    wall: f64,
    late: bool,
}

fn block_rep(rig: &mut Rig<'_>, exec: Exec, gated: bool) -> Res<BlockRep> {
    rep_with(rig, gated, &|r| r.full(exec))
}

/// One rep of whatever `queue` queues on the compute stream, timed as `block_rep` times a block.
fn rep_with(rig: &mut Rig<'_>, gated: bool, queue: &dyn Fn(&Rig<'_>) -> Res<()>) -> Res<BlockRep> {
    let start = Instant::now();
    let opened_from = if gated { Some(rig.gate()?) } else { None };
    rig.compute.record_timing(&rig.marks[START])?;
    let queue_start = Instant::now();
    queue(rig)?;
    let queued = Instant::now();
    rig.compute.record_timing(&rig.marks[END])?;
    rig.settle()?;
    let wall = us(start);
    let late = match opened_from {
        Some(gate) => {
            let since_gate = queued.duration_since(gate).as_secs_f64() * 1e6;
            since_gate >= rig.mark_us(COPY0, COPY1)?
        }
        None => false,
    };
    Ok(BlockRep {
        device: rig.mark_us(START, END)?,
        queue: queued.duration_since(queue_start).as_secs_f64() * 1e6,
        wall,
        late,
    })
}

fn block_phase(rig: &mut Rig<'_>) -> Res<()> {
    for exec in [Exec::Eager, Exec::Graph] {
        for gated in [false, true] {
            let reps = (0..WARM + BLOCK_REPS)
                .map(|_| block_rep(rig, exec, gated))
                .collect::<Res<Vec<_>>>()?;
            let kept = &reps[WARM..];
            let late = kept.iter().filter(|r| r.late).count();
            let pick = |f: fn(&BlockRep) -> f64| kept.iter().map(f).collect::<Vec<_>>();
            let tag = format!(
                "{}_{}",
                exec.tag(),
                if gated { "gated" } else { "isolated" }
            );
            let extra = format!(",\"late\":{late}");
            emit(
                "graph_block",
                &format!("{tag}_device"),
                "us",
                &pick(|r| r.device),
                &extra,
            );
            emit(
                "graph_block",
                &format!("{tag}_host_queue"),
                "us",
                &pick(|r| r.queue),
                "",
            );
            if !gated {
                emit(
                    "graph_block",
                    &format!("{tag}_wall"),
                    "us",
                    &pick(|r| r.wall),
                    "",
                );
            }
        }
    }
    token_cases(rig)
}

/// A token's 48 blocks back to back: eager, 48 graph launches, one token graph, and 48 split
/// layers without misses in both arms.
fn token_cases(rig: &mut Rig<'_>) -> Res<()> {
    #[derive(Clone, Copy)]
    enum How {
        Blocks(Exec),
        TokenGraph,
        Split(Exec),
    }
    let cases = [
        ("eager_blocks", How::Blocks(Exec::Eager)),
        ("graph_blocks", How::Blocks(Exec::Graph)),
        ("graph_token", How::TokenGraph),
        ("eager_split_m0", How::Split(Exec::Eager)),
        ("graph_split_m0", How::Split(Exec::Graph)),
    ];
    for (name, how) in cases {
        let (mut device, mut issued, mut wall) = (Vec::new(), Vec::new(), Vec::new());
        for rep in 0..TOKEN_WARM + TOKEN_REPS {
            let start = Instant::now();
            rig.compute.record_timing(&rig.marks[START])?;
            match how {
                How::Blocks(exec) => (0..LAYERS).try_for_each(|_| rig.full(exec))?,
                How::TokenGraph => rig.graphs()?.token.launch(&rig.compute)?,
                How::Split(exec) => {
                    for _ in 0..LAYERS {
                        rig.layer(exec, Sched::Device, 0, false)?;
                    }
                }
            }
            let queued = us(start);
            rig.compute.record_timing(&rig.marks[END])?;
            rig.settle()?;
            if rep < TOKEN_WARM {
                continue;
            }
            device.push(rig.mark_us(START, END)?);
            issued.push(queued);
            wall.push(us(start));
        }
        emit(
            "graph_block",
            &format!("token_{name}_device"),
            "us",
            &device,
            "",
        );
        emit(
            "graph_block",
            &format!("token_{name}_host_queue"),
            "us",
            &issued,
            "",
        );
        emit(
            "graph_block",
            &format!("token_{name}_wall"),
            "us",
            &wall,
            "",
        );
    }
    Ok(())
}

/// The variants of the variants phase: the production id-indexed launch (`base`) and its
/// bench-only prototypes. The existing wide kernels were measured slower (2026-11-04) and are
/// left out.
const VARIANTS: [(&str, IndexedPrototype); 4] = [
    ("base", IndexedPrototype::Off),
    ("dp4a", IndexedPrototype::Dp4a),
    ("pairs20", IndexedPrototype::Pairs20),
    ("both", IndexedPrototype::Both),
];
/// The single-row kernel of every variant: a served EXL3 pack starts on it
/// (`qwen36::cuda::start`).
const SERVED: SingleRow = SingleRow::Wide(1);
const VARIANT_ROWS: usize = 4;
const VARIANT_TRIALS: usize = 5;
const VARIANT_REPS: usize = 40;
const VARIANT_TOKENS: usize = 6;

/// Captures the block under every variant and checks that the eager and graph outputs of each
/// equal the production eager outputs bit for bit on `VARIANT_ROWS` rows (NaN-poisoned runs).
fn variant_graphs(rig: &mut Rig<'_>) -> Res<Vec<GraphExec<'static>>> {
    let n_expert = rig.composed.dims.n_expert;
    rig.ctx.set_single_row_default(SERVED);
    rig.ctx.set_indexed_prototype(IndexedPrototype::Off);
    let mut want = Vec::with_capacity(VARIANT_ROWS);
    for seed in 0..VARIANT_ROWS {
        rig.set_row(seed)?;
        want.push(rig.run_once(|r| r.full(Exec::Eager))?);
    }
    let (mut graphs, mut agree) = (Vec::with_capacity(VARIANTS.len()), Agreement::default());
    for (name, prototype) in VARIANTS {
        rig.ctx.set_indexed_prototype(prototype);
        // The variant's kernels load on their first eager launch; a capture refuses a first launch.
        rig.full(Exec::Eager)?;
        rig.settle()?;
        let graph = capture(rig, queue_block)?.0;
        for (seed, want) in want.iter().enumerate() {
            rig.set_row(seed)?;
            let eager = rig.run_once(|r| r.full(Exec::Eager))?;
            let replay = rig.run_once(|r| graph.launch(&r.compute))?;
            for got in [eager, replay] {
                sound(&got, n_expert)?;
                agree.add(want, &got);
            }
            if agree.exact != agree.compared {
                return Err(fail(format!(
                    "variant {name}: row {seed} differs from production (worst relative L2 {:e})",
                    agree.worst
                )));
            }
        }
        graphs.push(graph);
    }
    rig.set_row(0)?;
    println!(
        "RESBENCH {{\"phase\":\"graph_variants\",\"case\":\"equivalence\",\"rows\":{VARIANT_ROWS},\
         \"variants\":{},\"against_base\":{}}}",
        VARIANTS.len(),
        agree.json()
    );
    Ok(graphs)
}

/// One token of whatever `queue` queues, device time from the first to the last mark.
fn token_with(rig: &mut Rig<'_>, queue: &dyn Fn(&Rig<'_>) -> Res<()>) -> Res<f64> {
    rig.compute.record_timing(&rig.marks[START])?;
    queue(rig)?;
    rig.compute.record_timing(&rig.marks[END])?;
    rig.settle()?;
    rig.mark_us(START, END)
}

/// Every variant's block: graph isolated and gated, eager isolated (under the variant's setting)
/// and 48 graph launches back to back. The variants take turns inside every trial, in an order
/// that rotates, and the first trial is a warm-up, so slow clock drift spreads over all of them.
fn variants_phase(rig: &mut Rig<'_>) -> Res<()> {
    // The variable is read once per process and wins over every default, so all the variants
    // would run one kernel under their own names.
    for var in [SINGLE_GATHER_VAR, IndexedPrototype::VAR] {
        if std::env::var_os(var).is_some() {
            return Err(fail(format!("the variants phase needs {var} unset")));
        }
    }
    let graphs = variant_graphs(rig)?;
    let names = [
        "graph_isolated",
        "graph_gated",
        "eager_isolated",
        "graph_token48",
    ];
    let mut samples: Vec<[Vec<f64>; 4]> = VARIANTS.iter().map(|_| Default::default()).collect();
    // Gated reps per variant where the host was still queueing when the gate opened.
    let mut late = [0usize; VARIANTS.len()];
    for trial in 0..=VARIANT_TRIALS {
        clock_mark(&format!("variants_trial{trial}"));
        // The order rotates from trial to trial, so no variant always follows the same one.
        for turn in 0..VARIANTS.len() {
            let v = (turn + trial) % VARIANTS.len();
            let ((_, prototype), graph) = (&VARIANTS[v], &graphs[v]);
            let replay = |r: &Rig<'_>| graph.launch(&r.compute);
            let mut got: [Vec<f64>; 4] = Default::default();
            for _ in 0..VARIANT_REPS {
                got[0].push(rep_with(rig, false, &replay)?.device);
                let gated = rep_with(rig, true, &replay)?;
                got[1].push(gated.device);
                if trial > 0 {
                    late[v] += usize::from(gated.late);
                }
            }
            rig.ctx.set_indexed_prototype(*prototype);
            for _ in 0..VARIANT_REPS {
                got[2].push(block_rep(rig, Exec::Eager, false)?.device);
            }
            for _ in 0..VARIANT_TOKENS {
                got[3].push(token_with(rig, &|r| {
                    (0..LAYERS).try_for_each(|_| replay(r))
                })?);
            }
            if trial > 0 {
                for (all, new) in samples[v].iter_mut().zip(got) {
                    all.extend(new);
                }
            }
        }
    }
    rig.ctx.set_single_row_default(SingleRow::Legacy);
    rig.ctx.set_indexed_prototype(IndexedPrototype::Off);
    for (((name, _), got), late) in VARIANTS.iter().zip(&samples).zip(late) {
        for (what, values) in names.iter().zip(got) {
            let extra = if *what == "graph_gated" {
                format!(",\"late\":{late}")
            } else {
                String::new()
            };
            emit(
                "graph_variants",
                &format!("{name}_{what}"),
                "us",
                values,
                &extra,
            );
        }
    }
    Ok(())
}

fn miss_phase(rig: &mut Rig<'_>) -> Res<()> {
    let k = rig.composed.dims.top_k;
    for exec in [Exec::Graph, Exec::Eager] {
        miss_case(rig, (exec, Sched::Device), 0, &format!("{}_m0", exec.tag()))?;
        for sched in [Sched::Device, Sched::Host] {
            for m in 1..=k {
                let tag = format!("{}_{}_m{m}", exec.tag(), sched.tag());
                miss_case(rig, (exec, sched), m, &tag)?;
            }
        }
    }
    Ok(())
}

/// One split layer with `m` misses, waited for after each rep.
fn miss_case(rig: &mut Rig<'_>, (exec, sched): (Exec, Sched), m: usize, tag: &str) -> Res<()> {
    let names = [
        "layer", "wall", "router", "resident", "stall", "lag", "copy",
    ];
    let mut samples: [Vec<f64>; 7] = Default::default();
    for rep in 0..MISS_WARM + MISS_REPS {
        let start = Instant::now();
        rig.compute.record_timing(&rig.marks[START])?;
        rig.layer(exec, sched, m, true)?;
        rig.compute.record_timing(&rig.marks[END])?;
        rig.settle()?;
        let wall = us(start);
        if rep < MISS_WARM {
            continue;
        }
        let mut got = vec![
            rig.mark_us(START, END)?,
            wall,
            rig.mark_us(START, ROUTED)?,
            rig.mark_us(ROUTED, RESIDENT)?,
            rig.mark_us(RESIDENT, TAIL)?,
        ];
        if m > 0 {
            got.push(rig.mark_us(ROUTED, COPY0)?);
            got.push(rig.mark_us(COPY0, COPY1)?);
        }
        for (into, value) in samples.iter_mut().zip(got) {
            into.push(value);
        }
    }
    for (name, values) in names.iter().zip(&samples) {
        if !values.is_empty() {
            emit(
                "graph_miss_layer",
                &format!("{tag}_{name}"),
                "us",
                values,
                "",
            );
        }
    }
    Ok(())
}

/// Per-layer miss counts per token, by trace name.
type Traces = Vec<(String, Vec<Vec<usize>>)>;

/// The traces named by `FIFI_FLASH_NEXT_MISS_TRACES` (`name=path,...`): every line one token of
/// 48 counts, each at most `top_k`.
fn read_traces(top_k: usize) -> Res<Traces> {
    let list = std::env::var("FIFI_FLASH_NEXT_MISS_TRACES")
        .map_err(|_| fail("FIFI_FLASH_NEXT_MISS_TRACES (name=path,...)"))?;
    let mut traces = Vec::new();
    for entry in list.split(',') {
        let (name, path) = entry
            .split_once('=')
            .ok_or_else(|| fail(format!("a trace entry without a name: {entry}")))?;
        let text = std::fs::read_to_string(path).map_err(|e| fail(format!("{path}: {e}")))?;
        let mut tokens = Vec::new();
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let counts = line
                .split_whitespace()
                .map(|v| v.parse::<usize>().map_err(|_| fail(format!("{path}: {v}"))))
                .collect::<Res<Vec<_>>>()?;
            if counts.len() != LAYERS || counts.iter().any(|&m| m > top_k) {
                return Err(fail(format!(
                    "{path}: a token is not {LAYERS} counts of 0..={top_k}"
                )));
            }
            tokens.push(counts);
        }
        if tokens.is_empty() {
            return Err(fail(format!("{path}: no tokens")));
        }
        traces.push((name.to_owned(), tokens));
    }
    Ok(traces)
}

fn trace_phase(rig: &mut Rig<'_>, traces: &Traces) -> Res<()> {
    let tokens = traces.first().map_or(0, |(_, t)| t.len());
    let resident = vec![vec![0usize; LAYERS]; tokens];
    let runs: [(Exec, &[Sched]); 2] = [
        (Exec::Graph, &[Sched::Host, Sched::Device]),
        (Exec::Eager, &[Sched::Host, Sched::Device]),
    ];
    for (exec, scheds) in runs {
        let tag = format!("{}_resident", exec.tag());
        token_run(rig, (exec, Sched::Device), &tag, &resident)?;
        for (name, tokens) in traces {
            for &sched in scheds {
                let tag = format!("{}_{}_{name}", exec.tag(), sched.tag());
                token_run(rig, (exec, sched), &tag, tokens)?;
            }
        }
    }
    Ok(())
}

/// Every token of `tokens` (48 split layers each), waited for after each token.
fn token_run(
    rig: &mut Rig<'_>,
    (exec, sched): (Exec, Sched),
    tag: &str,
    tokens: &[Vec<usize>],
) -> Res<()> {
    let warm = &tokens[..TRACE_WARM.min(tokens.len())];
    let (mut device, mut wall) = (
        Vec::with_capacity(tokens.len()),
        Vec::with_capacity(tokens.len()),
    );
    for (t, layers) in warm.iter().chain(tokens).enumerate() {
        let start = Instant::now();
        rig.compute.record_timing(&rig.marks[START])?;
        for &m in layers {
            rig.layer(exec, sched, m, false)?;
        }
        rig.compute.record_timing(&rig.marks[END])?;
        rig.settle()?;
        if t < warm.len() {
            continue;
        }
        device.push(rig.mark_us(START, END)?);
        wall.push(us(start));
    }
    for (what, values) in [("device", &device), ("wall", &wall)] {
        let list: Vec<String> = values.iter().map(|v| format!("{v:.1}")).collect();
        emit(
            "graph_trace",
            &format!("{tag}_{what}"),
            "us",
            values,
            &format!(",\"samples\":[{}]", list.join(",")),
        );
    }
    Ok(())
}

// ------------------------------------------------------------------ the test

#[test]
#[ignore = "timing: needs an exclusive GPU window (3 GiB) and FIFI_FLASH_NEXT_EXL3_PACK"]
fn flash_next_moe_graph_ab_microbench() {
    run().expect("the MoE graph A/B microbenchmark");
}

fn run() -> Res<()> {
    let phases = std::env::var("FIFI_GRAPH_BENCH_PHASES")
        .unwrap_or_else(|_| "check,block,miss,trace".to_owned());
    let want = |phase: &str| phases.split(',').any(|p| p == phase);
    let (checkpoint, layer, moe) = open_layer("Flash-Next MoE graph A/B microbenchmark")?;
    if moe.top_k > COPY_SLOTS || COPY_SLOTS > GATE_EXPERTS {
        return Err(fail(
            "the staging area must hold one layer's misses and the gate",
        ));
    }
    let traces = if want("trace") {
        read_traces(moe.top_k)?
    } else {
        Vec::new()
    };
    let (_, host) = read_store(&checkpoint, layer, &moe)?;
    let projector = CudaProjector::new()?;
    let ctx = projector.context();
    let router = Router::read(&checkpoint, layer, &moe)?;
    let weights = upload(&projector, &checkpoint, (layer, &router), &moe)?;
    let block = resolve_block(ctx, &weights, layer, moe)?;
    let mut pinned = ctx.pinned_buf(host.len())?;
    pinned.write(0, &host)?;
    let mut rig = Rig {
        graphs: None,
        ctx,
        block: &block,
        composed: Composed {
            parts: Parts::resolve(&weights, layer, &moe)?,
            bufs: block.bufs()?,
            dims: moe,
        },
        pinned,
        staging: ctx.upload(&vec![0u8; GATE_EXPERTS * EXPERT_BYTES])?,
        routed: ctx.event()?,
        copied: ctx.event()?,
        opened: ctx.event()?,
        marks: (0..=END).map(|_| ctx.event_timing()).collect::<Res<_>>()?,
        next: 0,
        compute: ctx.stream()?,
        transfer: ctx.stream()?,
    };
    let prototype = IndexedPrototype::from_env()?;
    ctx.set_indexed_prototype(prototype);
    println!(
        "RESBENCH {{\"phase\":\"graph_prototype\",\"case\":\"{}\"}}",
        prototype.name()
    );
    build_phase(&mut rig)?;
    if want("check") {
        check_phase(&mut rig, &host)?;
    }
    drop(host);
    if want("block") {
        block_phase(&mut rig)?;
    }
    if want("variants") {
        variants_phase(&mut rig)?;
    }
    if want("miss") {
        miss_phase(&mut rig)?;
    }
    if want("trace") {
        trace_phase(&mut rig, &traces)?;
    }
    Ok(())
}
