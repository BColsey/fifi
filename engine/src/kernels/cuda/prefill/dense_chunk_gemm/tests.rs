//! The tensor-core gemm of a chunk gives the rows the column-invariant matvec gives on the same
//! quantized activations, up to the f32 order of the sums, at the shapes of the Bolt projections.

use super::*;
use crate::kernels::cuda::QuantTensor;
use crate::kernels::cuda::resident::add_rmsnorm_q8::tests::as_f32;
use crate::kernels::cuda::resident::matvec_q::tests::reference;
use crate::kernels::cuda::resident::matvec_q::{MatvecCall, QuantType};
use crate::kernels::cuda::resident::quantize_q8_1::{
    QuantizeCall, q8_1_bytes as decode_q8_1_bytes,
};
use crate::kernels::cuda::testkit;

#[test]
fn the_gemm_matches_the_invariant_matvec_at_the_bolt_shapes() {
    if !testkit::device_present("the_gemm_matches_the_invariant_matvec_at_the_bolt_shapes") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let gemm = DenseChunkGemm::new(&ctx, 10_240, 64).expect("scratch");
    let mut rng = testkit::Rng::new(9);
    for (rows, k, m) in [
        (2048usize, 2560usize, 33usize),
        (512, 2560, 64),
        (2560, 2048, 17),
        (10_240, 2560, 40),
        (2560, 10_240, 33),
    ] {
        let w_bytes = reference::build_weight(QuantType::Q80, rows, k, &mut rng);
        let w_dev = testkit::upload_bytes(&ctx, &w_bytes);
        let weight = {
            let tensor = QuantTensor {
                name: "test",
                ty: crate::weights::GgufType::Q80,
                n_rows: rows,
                k,
                source_len: w_bytes.len(),
            };
            ctx.upload_native(tensor, 1 << 20, &mut |at, part| {
                part.copy_from_slice(&w_bytes[at..at + part.len()]);
                Ok(())
            })
            .expect("native weight")
        };
        let x: Vec<f32> = (0..m * k).map(|_| (rng.normal() as f32) * 0.5).collect();
        let x_dev = testkit::upload_f32(&ctx, &x);
        let q8_bytes = decode_q8_1_bytes(k, m).expect("q8 bytes");
        let q8 = testkit::upload_filled(&ctx, 0, q8_bytes);
        ctx.quantize_q8_1(
            &stream,
            QuantizeCall {
                x: x_dev.range(0, m * k * 4).expect("x"),
                y: q8.range(0, q8_bytes).expect("q8"),
                k,
                ncols: m,
            },
        )
        .expect("quantize");
        let want = testkit::upload_filled(&ctx, 0, m * rows * 4);
        for first in (0..m).step_by(8) {
            let cols = 8.min(m - first);
            let row_bytes = decode_q8_1_bytes(k, 1).expect("row bytes");
            ctx.matvec_q_invariant(
                &stream,
                MatvecCall {
                    ty: QuantType::Q80,
                    w: w_dev.range(0, w_bytes.len()).expect("w"),
                    x_q8: q8.range(first * row_bytes, cols * row_bytes).expect("x q8"),
                    y: want.range(first * rows * 4, cols * rows * 4).expect("want"),
                    k,
                    rows,
                    ncols: cols,
                },
            )
            .expect("matvec");
        }
        let got = testkit::upload_filled(&ctx, 0, m * rows * 4);
        let want = as_f32(&testkit::read_bytes(&ctx, &want, m * rows * 4));
        let scale = want.iter().fold(0f32, |acc, v| acc.max(v.abs()));
        // The packing of the activation: the reference rule, and the vendored one for comparison.
        for packing in [MmqPacking::CpuReference, MmqPacking::Vendor] {
            ctx.mmq_dense_input(
                &stream,
                MmqDenseInputCall {
                    packing,
                    weight: &weight,
                    m,
                    row_stride: k,
                    x: x_dev.range(0, m * k * 4).expect("x"),
                    y: gemm.input.range(0, gemm.input.bytes()).expect("input"),
                },
            )
            .expect("input");
            gemm.project(
                &ctx,
                &stream,
                &weight,
                m,
                got.range(0, m * rows * 4).expect("got"),
            )
            .expect("project");
            stream.synchronize().expect("drain");
            let got = as_f32(&testkit::read_bytes(&ctx, &got, m * rows * 4));
            let worst = want
                .iter()
                .zip(&got)
                .fold(0f32, |acc, (a, b)| acc.max((a - b).abs()));
            eprintln!(
                "rows {rows} k {k} m {m} {packing:?}: max |dy| {worst:.3e} of max |y| {scale:.3e}"
            );
            if packing == MmqPacking::CpuReference {
                assert!(
                    worst <= scale * 1e-3,
                    "rows {rows} k {k} m {m}: {worst} over {scale}"
                );
            }
        }
    }
}
