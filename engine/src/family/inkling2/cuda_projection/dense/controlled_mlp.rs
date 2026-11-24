//! Keep a one-row MLP controller on the router/expert stream.

use super::{ControlledAttentionInput, ControllerWeights, DenseStep, Slot, bad};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceRange, DeviceStream},
};

impl DenseStep {
    pub(in crate::family::inkling2::cuda_projection) fn run_controlled_routed_layer(
        &self,
        context: &CudaContext,
        input: ControlledAttentionInput<'_>,
        bias: &[f32],
        (weights, eps): (ControllerWeights<'_>, f32),
        enqueue: impl FnOnce(&DeviceStream, [DeviceRange<'_>; 4]) -> Result<(), EngError>,
    ) -> Result<Option<Vec<f32>>, EngError> {
        let width = input.input.len();
        if !(1..=4096).contains(&width) || !width.is_multiple_of(4) {
            return Ok(None);
        }
        let status_at = bias.len();
        // The router loads its input as float4. Keep every activation row aligned.
        let gate_at = (status_at + 1).next_multiple_of(4);
        let work_at = gate_at + 2 * width;
        let health_at = work_at + 6 * width;
        let Ok(readback) = self.range(Slot::Output, health_at, width + 1) else {
            return Ok(None);
        };
        if input.gate.len() != 2 * width
            || input.message.len() != width
            || input.gamma.len() != width
        {
            return Err(bad("invalid CUDA controlled MLP input shape"));
        }
        let upload = self.range(Slot::Output, 0, work_at + 3 * width)?;
        let norm = self.range(Slot::Output, work_at, 6 * width)?;
        let health = self.range(Slot::Output, health_at, 1)?;
        let bias_range = self.range(Slot::Output, 0, bias.len())?;
        let status = self.range(Slot::Output, status_at, 1)?;
        let x = self.range(Slot::Output, work_at + 5 * width, width)?;
        let y = self.range(Slot::Output, health_at + 1, width)?;
        let reduced = self.range(Slot::Reduced, 0, weights[0].0.shape[0])?;
        let calls = self.controller_calls(
            weights,
            [
                self.range(Slot::Output, gate_at, 2 * width)?,
                reduced,
                self.range(Slot::Output, work_at + 3 * width, width)?,
                self.range(Slot::Output, work_at + 4 * width, width)?,
            ],
        )?;
        let zeros = [0.0; 4];
        let rows = [
            bias,
            &zeros[..gate_at - status_at],
            input.gate,
            input.input,
            input.message,
            input.gamma,
        ];
        let mut output = self.run_rows(context, &self.stream, &rows, [upload, readback], || {
            context.dense_gemv_group(&self.stream, &calls)?;
            context.inkling_controller_norm_roles(&self.stream, norm, health, ([width, 1], eps))?;
            enqueue(&self.stream, [x, bias_range, status, y])
        })?;
        // run_rows validated both the controller's sentinel and the expert result.
        output.copy_within(1.., 0);
        output.truncate(width);
        Ok(Some(output))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{DensePlan, Resident};
    use super::*;
    use crate::{
        family::inkling2::cuda_projection::upload_tensor,
        kernels::cuda::{
            device_present,
            resident::{
                dense_gemv::DenseDtype,
                router::{RouterLogitsCall, RouterShape, router_packed_logits_bytes},
            },
        },
    };

    #[test]
    fn mlp_chain_keeps_normalized_row_on_device_and_reports_invalid_values() {
        if !device_present("Inkling CUDA controlled MLP") {
            return;
        }
        let context = CudaContext::new().unwrap();
        let width = 32;
        let step = DenseStep::new(&context, &DensePlan::new([512, 64], width, 1).unwrap()).unwrap();
        let weight = |shape: [usize; 2]| Resident {
            shape,
            codes: upload_tensor(&context, &vec![0; shape.iter().product::<usize>() * 4]).unwrap(),
        };
        let [down, gain, forget] = [weight([1, 64]), weight([32, 1]), weight([32, 1])];
        let weights = [&down, &gain, &forget].map(|w| (w, DenseDtype::F32));
        let shape = RouterShape {
            m: 1,
            hidden: width,
            n_expert: 8,
            top_k: 2,
        };
        let router = weight([shape.n_expert, width]);
        let bytes = router_packed_logits_bytes(shape).unwrap();
        let logits = context.allocate(bytes).unwrap();
        let echo = context.allocate(width * size_of::<f32>()).unwrap();
        let run = |input: &[f32]| {
            step.run_controlled_routed_layer(
                &context,
                ControlledAttentionInput {
                    gate: &[1.0; 64],
                    input,
                    message: &[2.0; 32],
                    gamma: &[0.5; 32],
                },
                &[0.0; 8],
                (weights, 1e-5),
                |stream, [x, _, _, y]| {
                    context.router_logits_rows(
                        stream,
                        RouterLogitsCall {
                            shape,
                            x,
                            w: router.codes.range(0, router.codes.bytes())?,
                            logits: logits.range(0, bytes)?,
                        },
                    )?;
                    let echo = echo.range(0, echo.bytes())?;
                    context.copy_device(stream, x, echo)?;
                    context.copy_device(stream, echo, y)
                },
            )
        };
        let input = [2.0; 32];
        let output = run(&input).unwrap().unwrap();
        let expected = 1.0 / (4.0_f32 + 1e-5).sqrt();
        assert!(output.iter().all(|&v| (v - expected).abs() <= f32::EPSILON));
        assert!(run(&[f32::MAX; 32]).is_err());
        assert_eq!(run(&input).unwrap().unwrap(), output);
    }
}
