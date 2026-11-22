use super::super::{DenseDtype, DensePlan};
use super::*;
use crate::{
    family::inkling2::cuda_projection::{Resident, upload_tensor},
    kernels::cuda::device_present,
};

#[test]
fn chained_attention_matches_host_and_checks_workspace_and_arithmetic() {
    if !device_present("Inkling CUDA chained controller attention") {
        return;
    }
    let context = CudaContext::new().unwrap();
    let hidden = 320;
    let rank = 32;
    let eps = 1e-5;
    let plan = DensePlan::new([25 * hidden, 2 * hidden], hidden, rank).unwrap();
    let step = DenseStep::new(&context, &plan).unwrap();
    let tensor = |rows, cols, seed| Resident {
        codes: upload_tensor(
            &context,
            &fixture(rows * cols, seed)
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        shape: [rows, cols],
    };
    let down = tensor(rank, 2 * hidden, 3);
    let gain = tensor(hidden, rank, 11);
    let forget = tensor(hidden, rank, 19);
    let controllers = [[
        (&down, DenseDtype::F32),
        (&gain, DenseDtype::F32),
        (&forget, DenseDtype::F32),
    ]; 3];
    let projection = tensor(12, hidden, 7);
    let proj = |input| {
        Some(ResidentProjection {
            weight: &projection,
            dtype: DenseDtype::F32,
            input,
        })
    };
    let weights = [proj(2), None, proj(0), proj(1)];
    let gates = std::array::from_fn::<_, 3, _>(|i| fixture(2 * hidden, 9 + 5 * i));
    let mut raw = std::array::from_fn::<_, 3, _>(|i| fixture(hidden, 1 + 7 * i));
    let messages = std::array::from_fn::<_, 3, _>(|i| fixture(hidden, 2 + 4 * i));
    let gamma = std::array::from_fn::<_, 3, _>(|i| fixture(hidden, 3 + 6 * i));
    let run = |step: &DenseStep, raw: &[Vec<f32>; 3]| {
        let inputs = std::array::from_fn(|i| ControlledAttentionInput {
            gate: &gates[i],
            input: &raw[i],
            message: &messages[i],
            gamma: &gamma[i],
        });
        step.controlled_attention(&context, inputs, controllers, weights, eps)
    };
    let actual = run(&step, &raw).unwrap().unwrap();
    let expected_inputs = std::array::from_fn(|i| {
        let coeffs = step
            .controller(&context, controllers[i], &gates[i])
            .unwrap();
        host_norm(&raw[i], &messages[i], &gamma[i], coeffs, eps)
    });
    let expected = step
        .project_group(
            &context,
            expected_inputs.each_ref().map(Vec::as_slice),
            weights,
        )
        .unwrap();
    for (got, want) in actual.iter().zip(expected) {
        assert_eq!(got.len(), want.len());
        for (&got, want) in got.iter().zip(want) {
            assert!((got - want).abs() < 1e-5, "{got} != {want}");
        }
    }
    let batch_raw: Vec<[Vec<f32>; 3]> = (0..3)
        .map(|row| std::array::from_fn(|i| fixture(hidden, 2 + 5 * row + 7 * i)))
        .collect();
    let rows = make_rows(&batch_raw, &gates, &messages, &gamma);
    let wide = DensePlan::new(
        [3 * (24 * hidden + 3 * rank + 3 + 36), 2 * hidden],
        hidden,
        rank,
    )
    .unwrap();
    let batch_step = DenseStep::new(&context, &wide).unwrap();
    let batch = batch_step
        .controlled_attention_batch(
            &context,
            &rows,
            controllers,
            weights,
            (eps, Arithmetic::Generic),
        )
        .unwrap()
        .unwrap();
    for (row, got) in batch_raw.iter().zip(batch) {
        let want = run(&step, row).unwrap().unwrap();
        for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
            assert!((g - w).abs() < 1e-5, "batch={g} single={w}");
        }
    }
    assert!(
        step.controlled_attention_batch(
            &context,
            &rows,
            controllers,
            weights,
            (eps, Arithmetic::CpuParity),
        )
        .unwrap()
        .is_none()
    );
    let small = DensePlan::new([24 * hidden - 32, 2 * hidden], hidden, rank).unwrap();
    assert!(
        run(&DenseStep::new(&context, &small).unwrap(), &raw)
            .unwrap()
            .is_none()
    );
    raw[1].fill(f32::MAX);
    assert_eq!(
        run(&step, &raw).unwrap_err(),
        bad("CUDA projection produced a nonfinite output")
    );
}

fn make_rows<'a>(
    values: &'a [[Vec<f32>; 3]],
    gates: &'a [Vec<f32>; 3],
    messages: &'a [Vec<f32>; 3],
    gamma: &'a [Vec<f32>; 3],
) -> Vec<[ControlledAttentionInput<'a>; 3]> {
    values
        .iter()
        .map(|raw| {
            std::array::from_fn(|i| ControlledAttentionInput {
                gate: &gates[i],
                input: &raw[i],
                message: &messages[i],
                gamma: &gamma[i],
            })
        })
        .collect()
}

fn fixture(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + seed) % 31) as f32 - 15.0) / 32.0)
        .collect()
}

fn host_norm(
    raw: &[f32],
    message: &[f32],
    gamma: &[f32],
    [gain, forget]: [Vec<f32>; 2],
    eps: f32,
) -> Vec<f32> {
    let mut out: Vec<_> = raw
        .iter()
        .enumerate()
        .map(|(i, &x)| (x - forget[i].tanh() * x) + gain[i].tanh() * message[i])
        .collect();
    let scale = (out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32 + eps)
        .sqrt()
        .recip();
    for (x, gamma) in out.iter_mut().zip(gamma) {
        *x = (*x * scale) * gamma;
    }
    out
}
