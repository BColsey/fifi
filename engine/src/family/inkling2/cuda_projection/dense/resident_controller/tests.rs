use super::{MAX_ROWS, Plan, ResidentController, TILE_BYTES, WORD};
use crate::{
    EngError,
    family::inkling2::cuda_projection::Resident,
    kernels::cuda::{
        CudaContext, DeviceRange, DeviceStream, device_present, resident::dense_gemv::DenseDtype,
    },
};

const EPS: (f32, f32) = (1e-6, 1e-5);

#[test]
fn plans_wide_tiles_and_bounds_input() {
    let plan = Plan::new([511, 2304, 256]).unwrap();
    assert!(plan.tile > 64);
    assert!(plan.tile < 511);
    assert!(plan.allocation()[1] <= TILE_BYTES);
    assert_eq!(plan.allocation()[2], 511 * (2304 + 3) * WORD);
    for shape in [
        [1, 2304, 256],
        [MAX_ROWS + 1, 2304, 256],
        [2, 2305, 256],
        [2, 2304, 0],
        [2, 4097, 256],
    ] {
        assert!(Plan::new(shape).is_none(), "{shape:?}");
    }
}

fn upload(
    ctx: &CudaContext,
    stream: &DeviceStream,
    shape: [usize; 2],
    f: impl Fn(usize, usize) -> f32,
) -> Resident {
    let [rows, cols] = shape;
    let mut raw = Vec::with_capacity(rows * cols * WORD);
    for i in 0..rows {
        for j in 0..cols {
            raw.extend_from_slice(&f(i, j).to_ne_bytes());
        }
    }
    let codes = ctx.allocate(raw.len()).unwrap();
    codes.write_at(stream, 0, &raw).unwrap();
    Resident { codes, shape }
}

fn read(ctx: &CudaContext, source: DeviceRange<'_>) -> Vec<f32> {
    // The consumer is on a different stream, as in the prefill router.
    let stream = ctx.stream().unwrap();
    let mut host = ctx.pinned_buf(source.bytes()).unwrap();
    ctx.copy_to_pinned(&stream, source, &mut host, 0).unwrap();
    host.synchronize(&stream).unwrap();
    let mut out = vec![0.0; source.bytes() / WORD];
    host.read_f32(0, &mut out).unwrap();
    out
}

fn scale(x: &[f32], eps: f32) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps)
        .sqrt()
        .recip()
}

#[test]
fn outputs_match_host_across_full_and_tail_tiles() {
    if !device_present("wide Inkling MLP controller") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let producer = ctx.stream().unwrap();
    let shape @ [rows, h, rank] = [129, 2304, 4];
    let mut scratch = ResidentController::allocate(&ctx, shape).unwrap().unwrap();
    assert!(rows > scratch.tile_rows() && !rows.is_multiple_of(scratch.tile_rows()));
    let x: Vec<Vec<_>> = (0..rows)
        .map(|i| {
            (0..h)
                .map(|j| ((i * 7 + j) % 97) as f32 / 49.0 - 1.0)
                .collect()
        })
        .collect();
    let msg: Vec<Vec<_>> = (0..rows)
        .map(|i| {
            (0..h)
                .map(|j| ((i + j * 3) % 79) as f32 / 43.0 - 1.0)
                .collect()
        })
        .collect();
    let gamma: Vec<_> = (0..h).map(|j| 0.75 + (j % 21) as f32 / 37.0).collect();
    let down_col = |i: usize| if i < 2 { i * 99 } else { h + i * 37 };
    let down = upload(&ctx, &producer, [rank, 2 * h], |i, j| {
        if j == down_col(i) { 0.5 } else { 0.0 }
    });
    let gain = upload(&ctx, &producer, [h, rank], |i, j| {
        if j == i % rank { 0.35 } else { 0.0 }
    });
    let forget = upload(&ctx, &producer, [h, rank], |i, j| {
        if j == (i + 1) % rank { -0.6 } else { 0.0 }
    });
    let weights = [&down, &gain, &forget].map(|w| (w, DenseDtype::F32));
    assert!(ResidentController::admits(shape, weights));
    let truncated = Resident {
        codes: ctx.allocate(WORD).unwrap(),
        shape: down.shape,
    };
    assert!(!ResidentController::admits(
        shape,
        [(&truncated, DenseDtype::F32), weights[1], weights[2]],
    ));
    for count in [rows, scratch.tile_rows() + 1, 3] {
        let normalized = scratch
            .run(&ctx, weights, (&x[..count], &msg[..count], &gamma), EPS)
            .unwrap();
        let got = read(&ctx, normalized);
        for (row, actual) in got.chunks_exact(h).enumerate() {
            let (x, msg) = (&x[row], &msg[row]);
            let inv = [scale(x, EPS.0), scale(msg, EPS.0)];
            let r: Vec<_> = (0..rank)
                .map(|i| {
                    let col = down_col(i);
                    let half = usize::from(col >= h);
                    let value = [x, msg][half][col % h];
                    (value * inv[half]) * 0.5
                })
                .collect();
            let expected: Vec<_> = (0..h)
                .map(|i| {
                    let gain = r[i % rank] * 0.35;
                    let forget = r[(i + 1) % rank] * -0.6;
                    (x[i] - forget.tanh() * x[i]) + gain.tanh() * msg[i]
                })
                .collect();
            let inv = scale(&expected, EPS.1);
            for (i, (&actual, value)) in actual.iter().zip(expected).enumerate() {
                let value = (value * inv) * gamma[i];
                assert!(
                    (actual - value).abs() <= 1.5e-5,
                    "row {row} element {i}: {actual} vs {value}"
                );
            }
        }
        assert_eq!(got.len(), count * h);
    }
}

#[test]
fn separate_gate_and_final_norm_health_and_clean_reuse() {
    if !device_present("wide Inkling MLP gate health") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let shape @ [rows, h, rank] = [2, 8, 1];
    let mut scratch = ResidentController::allocate(&ctx, shape).unwrap().unwrap();
    let down = upload(&ctx, &stream, [rank, 2 * h], |_, _| 0.0);
    let project = upload(&ctx, &stream, [h, rank], |_, _| 0.0);
    let weights = [&down, &project, &project].map(|w| (w, DenseDtype::F32));
    let mut msg = vec![vec![1.0; h]; rows];
    msg[1][0] = 2.0e19; // Gate sum overflows while apply and final norm remain finite.
    let input = vec![vec![1.0; h]; rows];
    let gamma = vec![1.0; h];
    let check_health = |result: Result<DeviceRange<'_>, EngError>| {
        assert!(matches!(result, Err(EngError::Checkpoint(message))
            if message == "Inkling 2 Mini: non-finite resident Inkling prefill controller"));
    };
    check_health(scratch.run(&ctx, weights, (&input, &msg, &gamma), EPS));
    msg[1][0] = 1.0;
    let large = vec![vec![4.0e18; h]; rows];
    check_health(scratch.run(&ctx, weights, (&large, &msg, &gamma), (EPS.0, f32::MAX)));
    let mut invalid_gamma = gamma.clone();
    invalid_gamma[h - 1] = f32::NAN;
    check_health(scratch.run(&ctx, weights, (&input, &msg, &invalid_gamma), EPS));
    let result = scratch
        .run(&ctx, weights, (&input, &msg, &gamma), EPS)
        .unwrap();
    assert!(
        read(&ctx, result)
            .iter()
            .all(|&v| (v - 1.0 / (1.0 + EPS.1).sqrt()).abs() < 1e-6)
    );
}
