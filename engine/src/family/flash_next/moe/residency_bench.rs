//! Expert-residency microbenchmark for Qwen3.8 Flash-Next on one GPU: what serving a
//! non-resident routed expert costs on the critical path. Measurement only: it adds no execution
//! path, changes no executor and widens no limit.
//!
//! `FIFI_FLASH_NEXT_EXL3_PACK` names a local pack holding every MoE tensor of one layer
//! (`FIFI_FLASH_NEXT_EXL3_MOE_LAYER`, 3 by default). All 512 routed experts of that layer are read
//! into one device-ready host store (per expert: gate, up, down, each trellis then `suh` and `svh`
//! widened to f32; 1,881,600 bytes), once in page-locked memory for the copies and once in
//! ordinary memory for the CPU kernel. Every phase prints one `RESBENCH` JSON line per case with
//! the sample count, mean, P50, P95, P99 and maximum.
//!
//! Phases:
//! - pinned: page-locked allocation (with its zeroing) and the change in `MemAvailable`, and one
//!   expert copied from ordinary memory against one copied from page-locked memory;
//! - h2d_single: one expert per copy, device interval and host wall time, the first copy of each
//!   expert (cold) and a repeat (warm);
//! - h2d_batch: k experts copied back to back, k from 1 to 48, and runs of 64 adjacent experts as
//!   one copy against 64 copies;
//! - cpu: the AVX-512 EXL3 kernel on 1, 2, 4 and 10 real experts at 1, 6, 8 and 12 threads, cold
//!   (every call reads experts no recent call read; the layer's 963 MB rotates through the caches)
//!   and warm (the same experts every call);
//! - gpu_pass: the shared EXL3 MoE block (router, 10 device picks, shared expert, combine) alone;
//! - overlap: that block together with CPU expert work, with copies on another stream, and the CPU
//!   kernel while copies stream;
//! - handoff: the device-to-host-to-device round trip of one layer through the mapped mailbox
//!   (`tiered_classify`, then `tiered_wait_add`, a host server answering), with no miss, with misses
//!   answered at once, with misses computed by the CPU kernel, and with the GPU block between the
//!   two kernels; plus a synchronous id read-back, a slot-table row write and the host bookkeeping
//!   of a per-layer LRU;
//! - prefill_ring: 135, 300 and 328 experts streamed through a two-half ring of 64 slots, alone and
//!   behind GPU work, and a whole 328-expert set as 64-expert copies.
//!
//! The mailbox carries at most 8 misses (the tiered kernels' limit, kept); Flash-Next picks 10. Run
//! only inside an exclusive GPU window: the numbers are timings.

use std::{
    path::Path,
    sync::atomic::{AtomicBool, AtomicU8, Ordering},
    time::{Duration, Instant},
};

use half::f16;

use super::exl3_cuda_tests::{Router, input, keys, matrix, upload};
use crate::{
    EngError,
    family::{
        exl3_pack::{ExpertPart, Root, expert_name_in},
        flash_next::load::{exl3_pack::inspect_flash_pack, model_shape},
        project::cuda::CudaProjector,
        qwen36::cuda::{
            exl3_moe::{Exl3MoeBufs, Exl3MoeWeights, Scratch, run_exl3_moe},
            moe::{Launch, MoeDims, RouteBufs},
        },
    },
    kernels::{
        cuda::{
            CudaContext, DeviceBuffer, DeviceRange, DeviceStream, PinnedBuf, TimingEvent,
            device_present,
            resident::{
                router::{RouterShape, router_logits_bytes},
                tiered_classify::{CTRL_BYTES, ClassifyCall, ClassifyShape, REQUEST_BYTES},
                tiered_wait_add::WaitAddCall,
            },
        },
        exl3_cpu::{Exl3Isa, Exl3Projection, ExpertCpu, ExpertPick, ExpertView},
        nvfp4_cpu::WaitPolicy,
    },
    tensor::{Dtype, Tensor},
    weights::exl3::Checkpoint,
};

const F32: usize = 4;
const EXPERT_BYTES: usize = 1_881_600;
const HANDOFF_PICKS: usize = 8;

type Res<T> = Result<T, EngError>;

fn fail(what: impl Into<String>) -> EngError {
    EngError::BadRequest(what.into())
}

// ------------------------------------------------------------------ statistics and output

/// Nearest-rank percentiles of `samples`, printed as one JSON line.
fn emit(phase: &str, case: &str, unit: &str, samples: &[f64], extra: &str) {
    let mut v = samples.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        println!("RESBENCH {{\"phase\":\"{phase}\",\"case\":\"{case}\",\"n\":0{extra}}}");
        return;
    }
    let rank = |q: f64| v[((q * n as f64).ceil() as usize).clamp(1, n) - 1];
    let mean = v.iter().sum::<f64>() / n as f64;
    println!(
        "RESBENCH {{\"phase\":\"{phase}\",\"case\":\"{case}\",\"unit\":\"{unit}\",\"n\":{n},\
         \"mean\":{mean:.3},\"p50\":{:.3},\"p95\":{:.3},\"p99\":{:.3},\"max\":{:.3}{extra}}}",
        rank(0.50),
        rank(0.95),
        rank(0.99),
        v[n - 1]
    );
}

fn mem_available_gib() -> f64 {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    text.lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kib| kib.parse::<f64>().ok())
        .map_or(f64::NAN, |kib| kib / 1_048_576.0)
}

/// A deterministic xorshift for expert choices.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// `k` distinct values below `n`.
    fn distinct(&mut self, k: usize, n: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(k);
        while out.len() < k {
            let value = (self.next() % n as u64) as usize;
            if !out.contains(&value) {
                out.push(value);
            }
        }
        out
    }
}

/// The whole of `buffer` as one range.
fn whole(buffer: &DeviceBuffer) -> Res<DeviceRange<'_>> {
    buffer.range(0, buffer.bytes())
}

/// Device microseconds between two recorded timing events.
fn device_us(from: &TimingEvent, to: &TimingEvent) -> Res<f64> {
    Ok(f64::from(from.elapsed_ms_since(to)?) * 1e3)
}

fn us(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e6
}

// ------------------------------------------------------------------ the expert store

/// Byte layout of one device-ready expert: per part (gate, up, down) its trellis, then `suh` and
/// `svh` as f32.
#[derive(Clone, Copy)]
struct Layout {
    hidden: usize,
    inter: usize,
    words: usize,
}

impl Layout {
    fn shape(&self, part: usize) -> [usize; 2] {
        if part == 2 {
            [self.inter, self.hidden]
        } else {
            [self.hidden, self.inter]
        }
    }

    fn part_bytes(&self, part: usize) -> [usize; 3] {
        let [k, n] = self.shape(part);
        [(k / 16) * (n / 16) * self.words * 2, 4 * k, 4 * n]
    }

    fn expert_bytes(&self) -> usize {
        (0..3)
            .map(|p| self.part_bytes(p).iter().sum::<usize>())
            .sum()
    }

    /// The CPU kernel's view of the expert stored at the start of `bytes`.
    fn view<'a>(&self, bytes: &'a [u8]) -> Res<ExpertView<'a>> {
        let mut at = 0;
        let mut parts = Vec::with_capacity(3);
        for part in 0..3 {
            let [t, s, v] = self.part_bytes(part);
            let trellis = &bytes[at..at + t];
            let suh = &bytes[at + t..at + t + s];
            let svh = &bytes[at + t + s..at + t + s + v];
            parts.push(Exl3Projection::new(
                self.shape(part),
                self.words,
                trellis,
                suh,
                svh,
            )?);
            at += t + s + v;
        }
        Ok(ExpertView {
            gate: parts[0],
            up: parts[1],
            down: parts[2],
        })
    }
}

fn widened(tensor: &Tensor) -> Res<Vec<u8>> {
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| fail("an EXL3 scale without host storage"))?;
    match tensor.dtype() {
        Dtype::F32 => Ok(bytes.to_vec()),
        Dtype::F16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|b| f16::from_le_bytes(*b).to_f32().to_le_bytes())
            .collect()),
        other => Err(fail(format!("an EXL3 scale stored as {other:?}"))),
    }
}

/// Every routed expert of `layer`, device-ready, in id order.
fn read_store(checkpoint: &Checkpoint, layer: usize, moe: &MoeDims) -> Res<(Layout, Vec<u8>)> {
    let mut layout = None;
    let mut host = Vec::new();
    for expert in 0..moe.n_expert {
        for part in ExpertPart::ALL {
            let m = matrix(
                checkpoint,
                &expert_name_in(Root::Layers, layer, expert, part),
            )?;
            let words = usize::from(m.bitrate_twice()) * 8;
            let found = *layout.get_or_insert(Layout {
                hidden: moe.hidden,
                inter: moe.inter,
                words,
            });
            if found.words != words {
                return Err(fail("experts of one layer at different rates"));
            }
            let trellis = m
                .trellis()
                .host_bytes()
                .ok_or_else(|| fail("a trellis without host storage"))?;
            host.extend_from_slice(trellis);
            host.extend(widened(m.suh())?);
            host.extend(widened(m.svh())?);
        }
    }
    let layout = layout.ok_or_else(|| fail("no experts"))?;
    if layout.expert_bytes() != EXPERT_BYTES || host.len() != moe.n_expert * EXPERT_BYTES {
        return Err(fail(format!(
            "expert bytes {} (store {}), expected {EXPERT_BYTES}",
            layout.expert_bytes(),
            host.len()
        )));
    }
    Ok((layout, host))
}

// ------------------------------------------------------------------ the resident GPU block

/// The MoE block of one layer with persistent buffers, run with device picks.
struct Block<'w> {
    weights: Exl3MoeWeights<'w>,
    owned: Vec<DeviceBuffer>,
    moe: MoeDims,
}

impl Block<'_> {
    fn bufs(&self) -> Res<Exl3MoeBufs<'_>> {
        let r = |i: usize| self.owned[i].range(0, self.owned[i].bytes());
        Ok(Exl3MoeBufs {
            route: RouteBufs {
                resid_in: r(0)?,
                resid_out: r(1)?,
                norm_row: r(2)?,
                logits: r(3)?,
                ids: r(4)?,
                weights: r(5)?,
                shared_gate: r(6)?,
                status: r(7)?,
            },
            expert_gu: r(8)?,
            expert_act: r(9)?,
            expert_down: r(10)?,
            shared_gu: r(11)?,
            shared_act: r(12)?,
            shared_out: r(13)?,
            scratch: Scratch {
                xh: r(14)?,
                z: r(15)?,
                flag: r(16)?,
            },
        })
    }

    /// Queues the block on `stream`; nothing synchronizes.
    fn queue(&self, ctx: &CudaContext, stream: &DeviceStream) -> Res<()> {
        let launch = Launch { ctx, stream };
        run_exl3_moe(&launch, &self.weights, &self.bufs()?, &self.moe, true)
    }
}

fn block_buffers(ctx: &CudaContext, moe: &MoeDims) -> Res<Vec<DeviceBuffer>> {
    let zeros = |bytes: usize| ctx.upload(&vec![0u8; bytes]);
    let (h, k) = (moe.hidden * F32, moe.top_k);
    let logits = router_logits_bytes(RouterShape {
        m: 1,
        hidden: moe.hidden,
        n_expert: moe.n_expert,
        top_k: k,
    })?;
    let row: Vec<u8> = input(moe.hidden, 0)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    Ok(vec![
        zeros(h)?,
        zeros(h)?,
        ctx.upload(&row)?,
        zeros(logits)?,
        zeros(k * 4)?,
        zeros(k * F32)?,
        zeros(F32)?,
        zeros(4)?,
        zeros(k * 2 * moe.inter * F32)?,
        zeros(k * moe.inter * F32)?,
        zeros(k * h)?,
        zeros(2 * moe.shared_inter * F32)?,
        zeros(moe.shared_inter * F32)?,
        zeros(h)?,
        zeros(2 * k * moe.hidden * F32)?,
        zeros(2 * k * moe.inter.max(moe.hidden) * F32)?,
        zeros(4)?,
    ])
}

// ------------------------------------------------------------------ the benchmark state

struct Bench<'a> {
    ctx: &'a CudaContext,
    layout: Layout,
    /// Page-locked copy of the store: the source of every host-to-device copy.
    pinned: PinnedBuf,
    /// Ordinary-memory copy of the store: what the CPU kernel reads.
    host: Vec<u8>,
    /// 128 device slots of one expert each.
    slots: DeviceBuffer,
    compute: DeviceStream,
    transfer: DeviceStream,
    marks: [TimingEvent; 4],
    x: Vec<f32>,
}

impl Bench<'_> {
    fn expert(&self, id: usize) -> Res<ExpertView<'_>> {
        self.layout
            .view(&self.host[id * EXPERT_BYTES..(id + 1) * EXPERT_BYTES])
    }

    /// Queues one copy of `count` adjacent experts from `id` into slots from `slot`.
    fn copy(&mut self, id: usize, slot: usize, count: usize) -> Res<()> {
        let dst = self
            .slots
            .range(slot * EXPERT_BYTES, count * EXPERT_BYTES)?;
        self.ctx
            .copy_from_pinned(&self.transfer, &mut self.pinned, id * EXPERT_BYTES, dst)
    }

    fn settle(&mut self) -> Res<()> {
        self.transfer.synchronize()?;
        self.pinned.synchronize(&self.transfer)
    }

    /// One CPU forward over `ids` (unit weights); returns the wall time in microseconds.
    fn cpu_forward(&self, cpu: &ExpertCpu, ids: &[usize], out: &mut [f32]) -> Res<f64> {
        let picks = ids
            .iter()
            .map(|&id| {
                Ok(ExpertPick {
                    expert: self.expert(id)?,
                    weight: 1.0,
                })
            })
            .collect::<Res<Vec<_>>>()?;
        let start = Instant::now();
        cpu.forward(&self.x, &picks, out)?;
        Ok(us(start))
    }
}

fn new_cpu(threads: usize) -> Res<ExpertCpu> {
    let isa = Exl3Isa::detect();
    if !isa.is_supported() || isa != Exl3Isa::Avx512 {
        return Err(fail(
            "the AVX-512 EXL3 kernel is not available on this host",
        ));
    }
    ExpertCpu::new(
        threads,
        WaitPolicy::SpinThenPark(Duration::from_micros(500)),
        Some(isa),
    )
}

/// The expert ids of call `call` of a cold sequence: a permutation of the layer, so an expert
/// returns only after every other expert of the layer was read.
fn cold_ids(call: usize, count: usize, n: usize) -> Vec<usize> {
    (0..count).map(|j| ((call * count + j) * 37) % n).collect()
}

// ------------------------------------------------------------------ phases

fn pinned_phase(b: &mut Bench<'_>) -> Res<()> {
    for gib in [0.25f64, 1.0] {
        let bytes = (gib * (1u64 << 30) as f64) as usize;
        let before = mem_available_gib();
        let start = Instant::now();
        let buf = b.ctx.pinned_buf(bytes)?;
        let alloc = us(start);
        let after = mem_available_gib();
        let start = Instant::now();
        drop(buf);
        let free = us(start);
        emit(
            "pinned",
            &format!("alloc_zeroed_{gib}gib"),
            "us",
            &[alloc],
            &format!(
                ",\"free_us\":{free:.1},\"mem_available_before_gib\":{before:.2},\
                 \"mem_available_after_gib\":{after:.2}"
            ),
        );
    }
    let mut pageable = Vec::new();
    let mut pinned = Vec::new();
    let mut rng = Rng(7);
    for _ in 0..60 {
        let id = (rng.next() % 512) as usize;
        let start = Instant::now();
        b.slots.write_at(
            &b.compute,
            0,
            &b.host[id * EXPERT_BYTES..(id + 1) * EXPERT_BYTES],
        )?;
        b.compute.synchronize()?;
        pageable.push(us(start));
        let start = Instant::now();
        b.copy(id, 1, 1)?;
        b.settle()?;
        pinned.push(us(start));
    }
    emit(
        "pinned",
        "one_expert_pageable_write_at",
        "us",
        &pageable[10..],
        "",
    );
    emit("pinned", "one_expert_pinned_copy", "us", &pinned[10..], "");
    Ok(())
}

fn h2d_single(b: &mut Bench<'_>) -> Res<()> {
    let order: Vec<usize> = (0..256).map(|i| (i * 149 + 11) % 512).collect();
    for (case, rounds) in [("cold_first_copy", 1), ("warm_repeat", 2)] {
        let (mut device, mut wall) = (Vec::new(), Vec::new());
        for round in 0..rounds {
            for (i, &id) in order.iter().enumerate() {
                let start = Instant::now();
                b.transfer.record_timing(&b.marks[0])?;
                b.copy(id, i % 64, 1)?;
                b.transfer.record_timing(&b.marks[1])?;
                b.settle()?;
                if round + 1 == rounds {
                    wall.push(us(start));
                    device.push(device_us(&b.marks[0], &b.marks[1])?);
                }
            }
        }
        emit("h2d_single", case, "us", &device, ",\"clock\":\"device\"");
        emit("h2d_single", case, "us", &wall, ",\"clock\":\"host\"");
    }
    Ok(())
}

fn h2d_batch(b: &mut Bench<'_>) -> Res<()> {
    let mut rng = Rng(11);
    for k in [1usize, 2, 4, 8, 10, 16, 32, 48] {
        let (mut device, mut wall) = (Vec::new(), Vec::new());
        for rep in 0..60 {
            let ids = rng.distinct(k, 512);
            let start = Instant::now();
            b.transfer.record_timing(&b.marks[0])?;
            for (slot, &id) in ids.iter().enumerate() {
                b.copy(id, slot, 1)?;
            }
            b.transfer.record_timing(&b.marks[1])?;
            b.settle()?;
            if rep >= 10 {
                wall.push(us(start));
                device.push(device_us(&b.marks[0], &b.marks[1])?);
            }
        }
        let per: Vec<f64> = device.iter().map(|t| t / k as f64).collect();
        emit(
            "h2d_batch",
            &format!("k{k}"),
            "us",
            &device,
            ",\"clock\":\"device\"",
        );
        emit(
            "h2d_batch",
            &format!("k{k}"),
            "us",
            &wall,
            ",\"clock\":\"host\"",
        );
        emit(
            "h2d_batch",
            &format!("k{k}_per_expert"),
            "us",
            &per,
            ",\"clock\":\"device\"",
        );
    }
    for (case, per_copy) in [("run64_one_copy", 64usize), ("run64_64_copies", 1)] {
        let mut device = Vec::new();
        for rep in 0..20 {
            let first = (rep * 64) % 448;
            b.transfer.record_timing(&b.marks[0])?;
            for j in (0..64).step_by(per_copy) {
                b.copy(first + j, j, per_copy)?;
            }
            b.transfer.record_timing(&b.marks[1])?;
            b.settle()?;
            device.push(device_us(&b.marks[0], &b.marks[1])?);
        }
        emit(
            "h2d_batch",
            case,
            "us",
            &device[4..],
            ",\"clock\":\"device\"",
        );
    }
    Ok(())
}

fn cpu_phase(b: &Bench<'_>) -> Res<()> {
    let mut out = vec![0.0f32; b.layout.hidden];
    for threads in [1usize, 6, 8, 12] {
        let cpu = new_cpu(threads)?;
        for count in [1usize, 2, 4, 10] {
            let reps = if threads == 1 { 40 } else { 120 };
            for (case, cold) in [("cold", true), ("warm", false)] {
                let mut samples = Vec::new();
                for call in 0..reps + 10 {
                    let ids = cold_ids(if cold { call } else { 0 }, count, 512);
                    let t = b.cpu_forward(&cpu, &ids, &mut out)?;
                    if call >= 10 {
                        samples.push(t);
                    }
                }
                let per: Vec<f64> = samples.iter().map(|t| t / count as f64).collect();
                let gbps = count as f64 * EXPERT_BYTES as f64
                    / (samples.iter().sum::<f64>() / samples.len() as f64 * 1e3);
                emit(
                    "cpu",
                    &format!("{case}_t{threads}_m{count}"),
                    "us",
                    &samples,
                    &format!(
                        ",\"per_expert_mean_us\":{:.2},\"gb_per_s\":{gbps:.1}",
                        mean(&per)
                    ),
                );
            }
        }
    }
    Ok(())
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn gpu_pass(b: &mut Bench<'_>, block: &Block<'_>) -> Res<Vec<f64>> {
    let mut device = Vec::new();
    for rep in 0..230 {
        b.compute.record_timing(&b.marks[2])?;
        block.queue(b.ctx, &b.compute)?;
        b.compute.record_timing(&b.marks[3])?;
        b.compute.synchronize()?;
        if rep >= 30 {
            device.push(device_us(&b.marks[2], &b.marks[3])?);
        }
    }
    emit("gpu_pass", "moe_block_device_picks", "us", &device, "");
    Ok(device)
}

fn overlap_phase(b: &mut Bench<'_>, block: &Block<'_>) -> Res<()> {
    let cpu = new_cpu(6)?;
    let mut out = vec![0.0f32; b.layout.hidden];
    for count in [1usize, 2, 4] {
        let (mut wall, mut gpu, mut cpu_t) = (Vec::new(), Vec::new(), Vec::new());
        for call in 0..110 {
            let ids = cold_ids(call, count, 512);
            let start = Instant::now();
            b.compute.record_timing(&b.marks[2])?;
            block.queue(b.ctx, &b.compute)?;
            b.compute.record_timing(&b.marks[3])?;
            let t = b.cpu_forward(&cpu, &ids, &mut out)?;
            b.compute.synchronize()?;
            if call >= 10 {
                wall.push(us(start));
                gpu.push(device_us(&b.marks[2], &b.marks[3])?);
                cpu_t.push(t);
            }
        }
        let tag = format!("block_with_cpu_m{count}");
        emit("overlap", &format!("{tag}_wall"), "us", &wall, "");
        emit("overlap", &format!("{tag}_gpu"), "us", &gpu, "");
        emit("overlap", &format!("{tag}_cpu"), "us", &cpu_t, "");
    }
    // The block while ten experts copy on the transfer stream.
    let mut gpu = Vec::new();
    for rep in 0..60 {
        for j in 0..10 {
            b.copy((rep * 10 + j) % 512, j, 1)?;
        }
        b.compute.record_timing(&b.marks[2])?;
        block.queue(b.ctx, &b.compute)?;
        b.compute.record_timing(&b.marks[3])?;
        b.compute.synchronize()?;
        b.settle()?;
        if rep >= 10 {
            gpu.push(device_us(&b.marks[2], &b.marks[3])?);
        }
    }
    emit("overlap", "block_with_10_copies_gpu", "us", &gpu, "");
    // The CPU kernel while experts stream to the device (DRAM shared with the copy engine).
    for count in [1usize, 4] {
        let mut during = Vec::new();
        for burst in 0..8 {
            for j in 0..96 {
                b.copy((burst * 96 + j) % 512, j % 128, 1)?;
            }
            for call in 0..10 {
                during.push(b.cpu_forward(
                    &cpu,
                    &cold_ids(burst * 10 + call, count, 512),
                    &mut out,
                )?);
            }
            b.settle()?;
        }
        emit(
            "overlap",
            &format!("cpu_m{count}_while_copying"),
            "us",
            &during,
            "",
        );
    }
    Ok(())
}

/// The device buffers of the mailbox round trip of one layer; see the module doc.
struct Handoff {
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    slot_row: DeviceBuffer,
    norm_row: DeviceBuffer,
    /// slot_ids_out, weights_out, log_out, ctrl, word, resid, and two stand-ins that only pass
    /// the range checks of the request line and the row (the request goes to the mailbox).
    outs: [DeviceBuffer; 8],
}

impl Handoff {
    const IDS: [i32; HANDOFF_PICKS] = [11, 97, 203, 301, 405, 17, 255, 499];

    fn new(ctx: &CudaContext, hidden: usize) -> Res<Self> {
        let zeros = |bytes: usize| ctx.upload(&vec![0u8; bytes]);
        let ids: Vec<u8> = Self::IDS.iter().flat_map(|v| v.to_le_bytes()).collect();
        let weights: Vec<u8> = [0.125f32; HANDOFF_PICKS]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let row: Vec<u8> = input(hidden, 1)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Ok(Self {
            ids: ctx.upload(&ids)?,
            weights: ctx.upload(&weights)?,
            slot_row: zeros(512 * 4)?,
            norm_row: ctx.upload(&row)?,
            outs: [
                zeros(HANDOFF_PICKS * 4)?,
                zeros(HANDOFF_PICKS * 4)?,
                zeros(HANDOFF_PICKS * 8)?,
                zeros(CTRL_BYTES)?,
                zeros(4)?,
                zeros(hidden * F32)?,
                zeros(REQUEST_BYTES)?,
                zeros(hidden * F32)?,
            ],
        })
    }

    /// Marks the first `misses` picks non-resident and every other expert resident.
    fn set_misses(&self, stream: &DeviceStream, misses: usize) -> Res<()> {
        let mut row: Vec<i32> = (0..512).map(|e| e % 184).collect();
        for &id in &Self::IDS[..misses] {
            row[id as usize] = -1;
        }
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.slot_row.write_at(stream, 0, &bytes)?;
        stream.synchronize()
    }

    fn classify_call(&self, hidden: usize) -> Res<ClassifyCall<'_>> {
        let r = whole;
        let [slot_ids_out, weights_out, log_out, ctrl, word, _, req, row] = &self.outs;
        Ok(ClassifyCall {
            shape: ClassifyShape {
                n_expert: 512,
                slots: 184,
                top_k: HANDOFF_PICKS,
                hidden,
                layer: 1,
            },
            ids: r(&self.ids)?,
            weights: r(&self.weights)?,
            slot_row: r(&self.slot_row)?,
            norm_row: r(&self.norm_row)?,
            slot_ids_out: r(slot_ids_out)?,
            weights_out: r(weights_out)?,
            log_out: r(log_out)?,
            ctrl: r(ctrl)?,
            word: r(word)?,
            mbox_req: r(req)?,
            mbox_row: r(row)?,
        })
    }
}

/// What the mailbox server does with a request: answer a zero partial at once, or compute the
/// missed experts with the CPU kernel first.
const SERVE_IMMEDIATE: u8 = 0;
const SERVE_CPU: u8 = 1;

fn handoff_phase(b: &Bench<'_>, block: &Block<'_>) -> Res<()> {
    let hidden = b.layout.hidden;
    let rig = Handoff::new(b.ctx, hidden)?;
    let mailbox = b.ctx.mailbox(hidden)?;
    let port = mailbox.port()?;
    let stop = AtomicBool::new(false);
    let mode = AtomicU8::new(SERVE_IMMEDIATE);
    let (host, layout) = (&b.host, b.layout);
    let cases: [(&str, usize, u8, bool); 9] = [
        ("no_miss", 0, SERVE_IMMEDIATE, false),
        ("miss1_immediate", 1, SERVE_IMMEDIATE, false),
        ("miss8_immediate", 8, SERVE_IMMEDIATE, false),
        ("miss1_cpu", 1, SERVE_CPU, false),
        ("miss2_cpu", 2, SERVE_CPU, false),
        ("miss4_cpu", 4, SERVE_CPU, false),
        ("miss1_cpu_behind_block", 1, SERVE_CPU, true),
        ("miss2_cpu_behind_block", 2, SERVE_CPU, true),
        ("miss4_cpu_behind_block", 4, SERVE_CPU, true),
    ];
    let [_, _, _, ctrl, word, resid, _, _] = &rig.outs;
    let r = whole;
    std::thread::scope(|scope| -> Res<()> {
        let server = scope.spawn(|| -> Res<u64> {
            let cpu = new_cpu(6)?;
            let (mut row, mut partial) = (vec![0.0f32; hidden], vec![0.0f32; hidden]);
            let (mut after, mut served) = (0u64, 0u64);
            while !stop.load(Ordering::Acquire) {
                let Some(request) = port.poll(after) else {
                    std::hint::spin_loop();
                    continue;
                };
                if !port.read_row(&mut row) {
                    return Err(fail("the mailbox row could not be read"));
                }
                if mode.load(Ordering::Acquire) == SERVE_CPU {
                    let count = request.count as usize;
                    let picks = request.ids[..count]
                        .iter()
                        .map(|&id| {
                            let at = id as usize * EXPERT_BYTES;
                            Ok(ExpertPick {
                                expert: layout.view(&host[at..at + EXPERT_BYTES])?,
                                weight: 1.0,
                            })
                        })
                        .collect::<Res<Vec<_>>>()?;
                    cpu.forward(&row, &picks, &mut partial)?;
                } else {
                    partial.fill(0.0);
                }
                if !port.answer(request.seq, Ok(&partial)) {
                    return Err(fail("the mailbox answer was refused"));
                }
                after = request.seq;
                served += 1;
            }
            Ok(served)
        });
        let rounds = || -> Res<()> {
            for &(case, misses, serve, behind) in &cases {
                rig.set_misses(&b.compute, misses)?;
                mode.store(serve, Ordering::Release);
                let mut device = Vec::new();
                for rep in 0..230 {
                    b.compute.record_timing(&b.marks[2])?;
                    b.ctx.tiered_classify_to_mailbox(
                        &b.compute,
                        rig.classify_call(hidden)?,
                        &mailbox,
                    )?;
                    if behind {
                        block.queue(b.ctx, &b.compute)?;
                    }
                    b.ctx.tiered_wait_add(
                        &b.compute,
                        WaitAddCall {
                            hidden,
                            ctrl: r(ctrl)?,
                            word: r(word)?,
                            resid: r(resid)?,
                            mailbox: &mailbox,
                            timeout_ns: 2_000_000_000,
                        },
                    )?;
                    b.compute.record_timing(&b.marks[3])?;
                    b.compute.synchronize()?;
                    if rep >= 30 {
                        device.push(device_us(&b.marks[2], &b.marks[3])?);
                    }
                }
                let mut bits = [0u8; 4];
                word.read_at(&b.compute, 0, &mut bits)?;
                let bits = i32::from_le_bytes(bits);
                emit("handoff", case, "us", &device, &format!(",\"word\":{bits}"));
                if bits & 0b110 != 0 {
                    return Err(fail(format!("{case}: the tiered word reports {bits}")));
                }
            }
            Ok(())
        };
        let result = rounds();
        stop.store(true, Ordering::Release);
        let served = server
            .join()
            .map_err(|_| fail("the mailbox server panicked"))??;
        println!("RESBENCH {{\"phase\":\"handoff\",\"case\":\"served\",\"requests\":{served}}}");
        result
    })?;
    host_costs(b)
}

/// The host-side costs around a residency decision: a synchronous read-back of 10 ids, a
/// slot-table row write, and the bookkeeping of one layer's LRU (184 slots, 10 picks).
fn host_costs(b: &Bench<'_>) -> Res<()> {
    let ids = b.ctx.upload(&[0u8; 40])?;
    let table = b.ctx.upload(&[0u8; 2048])?;
    let (mut read, mut write) = (Vec::new(), Vec::new());
    let mut bytes = [0u8; 40];
    for rep in 0..530 {
        let start = Instant::now();
        ids.read_at(&b.compute, 0, &mut bytes)?;
        let t = us(start);
        let start = Instant::now();
        table.write_at(&b.compute, 0, &[rep as u8; 2048])?;
        b.compute.synchronize()?;
        if rep >= 30 {
            read.push(t);
            write.push(us(start));
        }
    }
    emit("handoff", "sync_readback_40b", "us", &read, "");
    emit("handoff", "slot_row_write_2kb", "us", &write, "");
    let mut last = vec![-1i64; 512];
    let mut resident = vec![false; 512];
    let (mut count, mut rng) = (0usize, Rng(3));
    let steps = 200_000usize;
    let start = Instant::now();
    let mut misses = 0usize;
    for step in 0..steps {
        let picks = rng.distinct(10, 512);
        for &e in &picks {
            if !resident[e] {
                misses += 1;
                if count == 184 {
                    let victim = (0..512)
                        .filter(|&v| resident[v] && !picks.contains(&v))
                        .min_by_key(|&v| last[v])
                        .ok_or_else(|| fail("no victim"))?;
                    resident[victim] = false;
                    count -= 1;
                }
                resident[e] = true;
                count += 1;
            }
            last[e] = step as i64;
        }
    }
    let per = start.elapsed().as_secs_f64() * 1e9 / steps as f64;
    println!(
        "RESBENCH {{\"phase\":\"handoff\",\"case\":\"lru_layer_step\",\"unit\":\"ns\",\
         \"mean\":{per:.1},\"misses_per_step\":{:.2},\"note\":\"uniform picks, linear victim scan\"}}",
        misses as f64 / steps as f64
    );
    Ok(())
}

fn prefill_ring(b: &mut Bench<'_>, block: &Block<'_>) -> Res<()> {
    let mut rng = Rng(19);
    for count in [135usize, 300, 328] {
        for behind in [false, true] {
            let mut device = Vec::new();
            for rep in 0..8 {
                let ids = rng.distinct(count, 512);
                if behind {
                    for _ in 0..40 {
                        block.queue(b.ctx, &b.compute)?;
                    }
                }
                b.transfer.record_timing(&b.marks[0])?;
                for (j, &id) in ids.iter().enumerate() {
                    b.copy(id, j % 128, 1)?;
                }
                b.transfer.record_timing(&b.marks[1])?;
                b.settle()?;
                b.compute.synchronize()?;
                if rep >= 2 {
                    device.push(device_us(&b.marks[0], &b.marks[1])?);
                }
            }
            let tag = if behind { "_behind_gpu" } else { "" };
            emit(
                "prefill_ring",
                &format!("stream{count}{tag}"),
                "us",
                &device,
                &format!(
                    ",\"gb_per_s\":{:.2}",
                    count as f64 * EXPERT_BYTES as f64 / (mean(&device) * 1e3)
                ),
            );
        }
    }
    let mut device = Vec::new();
    for rep in 0..8 {
        b.transfer.record_timing(&b.marks[0])?;
        let mut done = 0;
        while done < 328 {
            let run = (328 - done).min(64);
            b.copy((rep * 7 + done) % (512 - 64), (done / 64 % 2) * 64, run)?;
            done += run;
        }
        b.transfer.record_timing(&b.marks[1])?;
        b.settle()?;
        if rep >= 2 {
            device.push(device_us(&b.marks[0], &b.marks[1])?);
        }
    }
    emit(
        "prefill_ring",
        "stream328_as_64_expert_copies",
        "us",
        &device,
        "",
    );
    Ok(())
}

// ------------------------------------------------------------------ the test

#[test]
#[ignore = "timing: needs an exclusive GPU window (3 GiB) and FIFI_FLASH_NEXT_EXL3_PACK"]
fn flash_next_expert_residency_microbench() {
    run().expect("the residency microbenchmark");
}

fn run() -> Res<()> {
    let dir = std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK")
        .ok_or_else(|| fail("FIFI_FLASH_NEXT_EXL3_PACK"))?;
    let layer = std::env::var("FIFI_FLASH_NEXT_EXL3_MOE_LAYER")
        .map_or(Ok(3), |v| v.parse().map_err(|_| fail("a layer number")))?;
    if !device_present("Flash-Next residency microbenchmark") {
        return Err(fail("a GPU is required"));
    }
    let checkpoint = Checkpoint::open(Path::new(&dir))?;
    inspect_flash_pack(&checkpoint.source, &checkpoint.model_config)?;
    let (shape, _) = model_shape(&checkpoint.model_config)?;
    let shape = shape.moe;
    let moe = MoeDims {
        rows: 1,
        hidden: shape.hidden,
        n_expert: shape.experts,
        top_k: shape.top_k,
        inter: shape.inter,
        shared_inter: shape.shared_inter,
    };
    let start = Instant::now();
    let (layout, host) = read_store(&checkpoint, layer, &moe)?;
    println!(
        "RESBENCH {{\"phase\":\"setup\",\"case\":\"read_store\",\"seconds\":{:.1},\"bytes\":{},\
         \"words\":{},\"mem_available_gib\":{:.2}}}",
        start.elapsed().as_secs_f64(),
        host.len(),
        layout.words,
        mem_available_gib()
    );
    let projector = CudaProjector::new()?;
    let ctx = projector.context();
    let router = Router::read(&checkpoint, layer, &moe)?;
    let weights = upload(&projector, &checkpoint, (layer, &router), &moe)?;
    let keys = keys(layer);
    let block = Block {
        weights: Exl3MoeWeights::resolve_by(&weights, layer, &moe, |role| {
            keys.iter()
                .find(|(candidate, _)| *candidate == role)
                .map(|(_, key)| key.as_str())
                .ok_or_else(|| fail(format!("no key for {role:?}")))
        })?,
        owned: block_buffers(ctx, &moe)?,
        moe,
    };
    let start = Instant::now();
    let mut pinned = ctx.pinned_buf(host.len())?;
    let alloc = start.elapsed().as_secs_f64();
    let start = Instant::now();
    pinned.write(0, &host)?;
    println!(
        "RESBENCH {{\"phase\":\"setup\",\"case\":\"pinned_store\",\"bytes\":{},\"alloc_zeroed_s\":{alloc:.3},\
         \"fill_s\":{:.3},\"mem_available_gib\":{:.2}}}",
        host.len(),
        start.elapsed().as_secs_f64(),
        mem_available_gib()
    );
    let mut bench = Bench {
        ctx,
        layout,
        pinned,
        x: input(layout.hidden, 2),
        host,
        slots: ctx.upload(&vec![0u8; 128 * EXPERT_BYTES])?,
        compute: ctx.stream()?,
        transfer: ctx.stream()?,
        marks: [
            ctx.event_timing()?,
            ctx.event_timing()?,
            ctx.event_timing()?,
            ctx.event_timing()?,
        ],
    };
    h2d_single(&mut bench)?;
    h2d_batch(&mut bench)?;
    pinned_phase(&mut bench)?;
    cpu_phase(&bench)?;
    gpu_pass(&mut bench, &block)?;
    overlap_phase(&mut bench, &block)?;
    handoff_phase(&bench, &block)?;
    prefill_ring(&mut bench, &block)?;
    Ok(())
}
