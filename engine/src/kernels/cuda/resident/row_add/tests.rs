//! Sum and refusal tests of row_add. Device runs go through gpu_run.sh only.
use super::*;
use crate::kernels::cuda::{CudaContext, testkit};

/// The sum of two rows matches the host reference bit for bit.
#[test]
fn adds_two_rows_against_the_host_reference() {
    if !testkit::device_present("row_add basic") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.5 - 7.0).collect();
    let y: Vec<f32> = (0..64).map(|i| 3.25 - i as f32 * 0.25).collect();
    let x_dev = testkit::upload_f32(&ctx, &x);
    let y_dev = testkit::upload_f32(&ctx, &y);
    let out_dev = testkit::upload_filled(&ctx, 0xff, x.len() * 4);
    ctx.row_add(
        &stream,
        RowAddCall {
            out: out_dev.range(0, x.len() * 4).expect("out"),
            x: x_dev.range(0, x.len() * 4).expect("x"),
            y: y_dev.range(0, x.len() * 4).expect("y"),
            n: x.len(),
            flag: None,
            fused: false,
        },
    )
    .expect("row add");
    stream.synchronize().expect("drain");
    let got = testkit::floats(&testkit::read_bytes(&ctx, &out_dev, x.len() * 4));
    for (index, value) in got.iter().enumerate() {
        assert_eq!(*value, x[index] + y[index], "element={index}");
    }
}

/// The wrapper refuses an out that overlaps x before any native call.
#[test]
fn refuses_an_out_that_overlaps_an_input() {
    if !testkit::device_present("row_add alias refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let x_dev = testkit::upload_f32(&ctx, &[1.0f32; 32]);
    let y_dev = testkit::upload_f32(&ctx, &[2.0f32; 32]);
    let range = x_dev.range(0, 128).expect("x");
    let err = ctx
        .row_add(
            &stream,
            RowAddCall {
                out: range,
                x: range,
                y: y_dev.range(0, 128).expect("y"),
                n: 32,
                flag: None,
                fused: false,
            },
        )
        .expect_err("alias refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}

/// A fused call whose flag shares a byte with out, x or y is refused before any native
/// call although every other argument passes every guard; a clear flag is accepted.
#[test]
fn refuses_a_fused_flag_that_overlaps_a_tensor_and_accepts_a_clear_one() {
    if !testkit::device_present("row_add fused flag alias refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let bytes = 128;
    let x_dev = testkit::upload_f32(&ctx, &[1.0f32; 32]);
    let y_dev = testkit::upload_f32(&ctx, &[2.0f32; 32]);
    let out_dev = testkit::upload_filled(&ctx, 0xa5, bytes);
    let flag_dev = testkit::upload_bytes(&ctx, &8u32.to_le_bytes());
    for (name, flag) in [
        ("out", out_dev.range(64, 4).expect("flag in out")),
        ("x", x_dev.range(4, 4).expect("flag in x")),
        ("y", y_dev.range(124, 4).expect("flag in y")),
    ] {
        let err = ctx
            .row_add(
                &stream,
                RowAddCall {
                    out: out_dev.range(0, bytes).expect("out"),
                    x: x_dev.range(0, bytes).expect("x"),
                    y: y_dev.range(0, bytes).expect("y"),
                    n: 32,
                    flag: Some(flag),
                    fused: true,
                },
            )
            .expect_err(name);
        assert!(
            matches!(err, crate::EngError::BadRequest(_)),
            "{name}: {err:?}"
        );
    }
    ctx.row_add(
        &stream,
        RowAddCall {
            out: out_dev.range(0, bytes).expect("out"),
            x: x_dev.range(0, bytes).expect("x"),
            y: y_dev.range(0, bytes).expect("y"),
            n: 32,
            flag: Some(flag_dev.range(0, 4).expect("flag")),
            fused: true,
        },
    )
    .expect("a clear flag is accepted");
    stream.synchronize().expect("drain");
    assert_eq!(testkit::read_bytes(&ctx, &flag_dev, 4), 8u32.to_le_bytes());
}
