//! Device test (3090) of one routed MoE stage through `routed_gemm`: the gate stack on the
//! dequantizing GEMM, the up stack on the vendored grouped MMQ (so one stage mixes the two
//! paths and the q8_1 buffer is built on demand for the stack that reads it), and a down stack on
//! the dequantizing GEMM reading slot rows. The outputs are compared with an f64 reference over
//! the oracle's dequantized weights. Run with `--ignored --test-threads=1`.

use super::prefill::{RoutedX, routed_gemm, routed_quantize};
use crate::family::exl3::launch::MoeDims;
use crate::kernels::cuda::prefill::mmq_mmid::{HostIds, MmidIds, MmidIdsOut, MmidShape};
use crate::kernels::cuda::prefill::mmq_quant::MmqPacking;
use crate::kernels::cuda::prefill::mmq_size::{fixup_floats_max, q8_1_bytes};
use crate::kernels::cuda::{
    CHUNK_LIMIT, CudaContext, DeviceBuffer, DeviceStream, NativeWeight, QuantTensor,
};
use crate::weights::{GgufType, dequantize};

/// A xorshift stream: the test needs repeatable bytes and normals, nothing more.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Sum of twelve uniforms minus six: close enough to a standard normal.
    fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.unit()).sum::<f64>() - 6.0
    }
}

/// Finite f16 scales between 2^-8 and 2^-5.
const SCALES: [u16; 4] = [0x1c00, 0x2000, 0x2400, 0x2800];

const E: usize = 8;
const TOP_K: usize = 2;
const M: usize = 24;
const HIDDEN: usize = 512;
const INTER: usize = 256;

/// `rows` rows of `k` values of `ty` as random blocks with finite scales.
fn blocks(ty: GgufType, rows: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let (bb, be) = (ty.block_bytes() as usize, ty.block_elems() as usize);
    let mut bytes = vec![0u8; rows * (k / be) * bb];
    for chunk in bytes.chunks_mut(8) {
        let word = rng.next_u64().to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
    for block in bytes.chunks_mut(bb) {
        let d = SCALES[(rng.next_u64() % 4) as usize].to_le_bytes();
        block[0..2].copy_from_slice(&d);
        if ty == GgufType::Q4K {
            block[2..4].copy_from_slice(&d);
        }
    }
    bytes
}

fn stack(ctx: &CudaContext, ty: GgufType, bytes: &[u8], rows: usize, k: usize) -> NativeWeight {
    ctx.upload_native(
        QuantTensor {
            name: "routed stack",
            ty,
            n_rows: E * rows,
            k,
            source_len: bytes.len(),
        },
        CHUNK_LIMIT,
        &mut |at, buf| {
            buf.copy_from_slice(&bytes[at..at + buf.len()]);
            Ok(())
        },
    )
    .expect("upload the stack")
    .with_experts(Some(E))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_f32(stream: &DeviceStream, buffer: &DeviceBuffer, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    buffer.read_at(stream, 0, &mut bytes).expect("read back");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn filled(ctx: &CudaContext, byte: u8, len: usize) -> DeviceBuffer {
    ctx.upload(&vec![byte; len]).expect("upload")
}

/// The reference of slot `s` (expert `e`, activation row `x`): `w[e][row] . x`, summed in f64.
fn reference(weights: &[f32], k: usize, rows: usize, e: usize, x: &[f32]) -> Vec<(f64, f64)> {
    (0..rows)
        .map(|r| {
            let w = &weights[(e * rows + r) * k..(e * rows + r + 1) * k];
            let (mut dot, mut abs) = (0.0f64, 0.0f64);
            for (a, b) in w.iter().zip(x) {
                let t = f64::from(*a) * f64::from(*b);
                dot += t;
                abs += t.abs();
            }
            (dot, abs)
        })
        .collect()
}

#[test]
#[ignore = "device: needs the card"]
fn a_routed_stage_mixes_the_dequantizing_gemm_and_the_vendored_mmq() {
    let Ok(ctx) = CudaContext::new() else {
        return;
    };
    let stream = ctx.stream().expect("stream");
    let mut rng = Rng(0x0051_07ed_2545_f491);
    let (gate_ty, up_ty, down_ty) = (GgufType::Iq3Xxs, GgufType::Q4K, GgufType::Iq4Xs);
    let gate_bytes = blocks(gate_ty, E * INTER, HIDDEN, &mut rng);
    let up_bytes = blocks(up_ty, E * INTER, HIDDEN, &mut rng);
    let down_bytes = blocks(down_ty, E * HIDDEN, INTER, &mut rng);
    let (gate, up, down) = (
        stack(&ctx, gate_ty, &gate_bytes, INTER, HIDDEN),
        stack(&ctx, up_ty, &up_bytes, INTER, HIDDEN),
        stack(&ctx, down_ty, &down_bytes, HIDDEN, INTER),
    );
    let (gate_w, up_w, down_w) = (
        dequantize(gate_ty, &gate_bytes).expect("gate"),
        dequantize(up_ty, &up_bytes).expect("up"),
        dequantize(down_ty, &down_bytes).expect("down"),
    );
    // Routing: token t picks experts (t % E, (t * 3 + 1) % E), distinct because E is even.
    let slots = M * TOP_K;
    let expert_of: Vec<usize> = (0..slots)
        .map(|s| {
            let (t, r) = (s / TOP_K, s % TOP_K);
            if r == 0 { t % E } else { (t * 3 + 1) % E }
        })
        .collect();
    assert!((0..M).all(|t| expert_of[t * TOP_K] != expert_of[t * TOP_K + 1]));
    let mut ids_dst = Vec::new();
    let mut bounds = vec![0i32];
    for e in 0..E {
        ids_dst.extend((0..slots).filter(|&s| expert_of[s] == e).map(|s| s as i32));
        bounds.push(ids_dst.len() as i32);
    }
    let shape = MmidShape {
        n_experts: E,
        m: M,
        top_k: TOP_K,
    };
    let src1 = filled(&ctx, 0, slots * 4);
    let dst = filled(&ctx, 0, (slots + M + 127) * 4);
    let bnd = filled(&ctx, 0, (E + 1) * 4);
    let ids = MmidIds::from_host(
        &stream,
        shape,
        HostIds {
            ids_dst: &ids_dst,
            expert_bounds: &bounds,
        },
        MmidIdsOut {
            ids_src1: src1.range(0, slots * 4).expect("src1"),
            ids_dst: dst.range(0, (slots + M + 127) * 4).expect("dst"),
            expert_bounds: bnd.range(0, (E + 1) * 4).expect("bounds"),
        },
    )
    .expect("ids");
    let dims = MoeDims {
        rows: M,
        hidden: HIDDEN,
        n_expert: E,
        top_k: TOP_K,
        inter: INTER,
        shared_inter: INTER,
    };
    let x2: Vec<f32> = (0..M * HIDDEN).map(|_| rng.normal() as f32 * 0.5).collect();
    let act: Vec<f32> = (0..slots * INTER)
        .map(|_| rng.normal() as f32 * 0.5)
        .collect();
    let x2_buf = ctx.upload(&f32_bytes(&x2)).expect("x2");
    let act_buf = ctx.upload(&f32_bytes(&act)).expect("act");
    let q8s = filled(&ctx, 0x55, q8_1_bytes(HIDDEN, slots).expect("q8s"));
    let q8act = filled(&ctx, 0x55, q8_1_bytes(INTER, slots).expect("q8act"));
    let nsm = ctx.mmq_device_facts().expect("facts").nsm;
    let fixup = filled(&ctx, 0, fixup_floats_max(nsm).expect("fixup") * 4);
    let ld_gu = 2 * INTER;
    let gu = filled(&ctx, 0, slots * ld_gu * 4);
    let down_out = filled(&ctx, 0, slots * HIDDEN * 4);
    let gu_up = gu
        .range(INTER * 4, gu.bytes() - INTER * 4)
        .expect("up half");
    let fixup_range = fixup.range(0, fixup.bytes()).expect("fixup");
    let gate_x = RoutedX {
        q8: q8s.range(0, q8s.bytes()).expect("q8s"),
        rows: x2_buf.range(0, x2_buf.bytes()).expect("x2"),
        row_stride: HIDDEN,
        per_slot: false,
    };
    // The scatter quantize runs for the stage because the up stack is vendored; the gate stack
    // on the dequantizing GEMM ignores it.
    routed_quantize(
        &ctx,
        &stream,
        &ids,
        &[&gate, &up],
        false,
        MmqPacking::CpuReference,
        HIDDEN,
        gate_x.rows,
        gate_x.q8,
    )
    .expect("scatter quantize");
    routed_gemm(
        &ctx,
        &stream,
        &gate,
        &dims,
        &ids,
        &gate_x,
        ld_gu,
        gu.range(0, gu.bytes()).expect("gate dst"),
        fixup_range,
    )
    .expect("gate on the dequantizing GEMM");
    routed_gemm(
        &ctx,
        &stream,
        &up,
        &dims,
        &ids,
        &gate_x,
        ld_gu,
        gu_up,
        fixup_range,
    )
    .expect("up on the vendored MMQ");
    let down_x = RoutedX {
        q8: q8act.range(0, q8act.bytes()).expect("q8act"),
        rows: act_buf.range(0, act_buf.bytes()).expect("act"),
        row_stride: INTER,
        per_slot: true,
    };
    // The down stack is on the dequantizing GEMM: no gather quantize runs and no q8_1 is read.
    routed_quantize(
        &ctx,
        &stream,
        &ids,
        &[&down],
        true,
        MmqPacking::CpuReference,
        INTER,
        down_x.rows,
        down_x.q8,
    )
    .expect("gather quantize is a no-op for a dequantizing stack");
    routed_gemm(
        &ctx,
        &stream,
        &down,
        &dims,
        &ids,
        &down_x,
        HIDDEN,
        down_out.range(0, down_out.bytes()).expect("down dst"),
        fixup_range,
    )
    .expect("down on the dequantizing GEMM");
    stream.synchronize().expect("drain");
    let gu_got = read_f32(&stream, &gu, slots * ld_gu);
    let down_got = read_f32(&stream, &down_out, slots * HIDDEN);
    let (mut worst_gate, mut worst_up, mut worst_down) = (0.0f64, 0.0f64, 0.0f64);
    for s in 0..slots {
        let e = expert_of[s];
        let x_row = &x2[(s / TOP_K) * HIDDEN..(s / TOP_K + 1) * HIDDEN];
        let a_row = &act[s * INTER..(s + 1) * INTER];
        let g = reference(&gate_w, HIDDEN, INTER, e, x_row);
        let u = reference(&up_w, HIDDEN, INTER, e, x_row);
        let d = reference(&down_w, INTER, HIDDEN, e, a_row);
        for (r, (dot, abs)) in g.iter().enumerate() {
            // f16 weights and activations on the tensor cores: a few units of 2^-11 of the
            // absolute sum bound the error.
            let err = (f64::from(gu_got[s * ld_gu + r]) - dot).abs() / (abs + 1e-9);
            worst_gate = worst_gate.max(err);
        }
        for (r, (dot, abs)) in u.iter().enumerate() {
            // The q8_1 activation rounds each value to 8 bits.
            let err = (f64::from(gu_got[s * ld_gu + INTER + r]) - dot).abs() / (abs + 1e-9);
            worst_up = worst_up.max(err);
        }
        for (r, (dot, abs)) in d.iter().enumerate() {
            let err = (f64::from(down_got[s * HIDDEN + r]) - dot).abs() / (abs + 1e-9);
            worst_down = worst_down.max(err);
        }
    }
    println!(
        "routed stage: worst error over sum|w x| gate {worst_gate:.2e} up {worst_up:.2e} down {worst_down:.2e}"
    );
    assert!(worst_gate < 2e-3, "gate (dequantizing GEMM) {worst_gate:e}");
    assert!(
        worst_up < 2e-2,
        "up (vendored MMQ after the on-demand quantize) {worst_up:e}"
    );
    assert!(worst_down < 2e-3, "down (slot rows) {worst_down:e}");
}
