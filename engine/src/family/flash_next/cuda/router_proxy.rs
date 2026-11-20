//! Decode-only measurement of future routers on the current layer's normalized MoE input.

use crate::kernels::cuda::CudaContext;
use crate::{
    EngError,
    family::exl3::launch::Launch,
    kernels::cuda::{
        DeviceBuffer, DeviceRange,
        resident::router::{
            RouterCall, RouterOut, RouterShape, RouterWeights, router_logits_bytes,
        },
    },
};
use std::time::Instant;

const WORD: usize = size_of::<i32>();
pub(super) const HORIZONS: usize = 2;
const MAX_PICKS: usize = 32;

pub(super) struct ActualRoute<'a> {
    pub(super) layer: usize,
    pub(super) position: usize,
    pub(super) picks: &'a [i32],
    pub(super) slots: &'a [i32],
}

pub(super) struct RouterProxy {
    shape: RouterShape,
    scratch: DeviceBuffer,
}

impl RouterProxy {
    pub(super) fn build(
        ctx: &CudaContext,
        hidden: usize,
        experts: usize,
    ) -> Result<Self, EngError> {
        let shape = RouterShape {
            m: 1,
            hidden,
            n_expert: experts,
            top_k: experts.min(MAX_PICKS),
        };
        let outputs = (HORIZONS + 1) * shape.top_k * WORD + WORD;
        Ok(Self {
            shape,
            scratch: ctx.allocate(outputs + router_logits_bytes(shape)?)?,
        })
    }

    /// Router matrices and the input are read-only. The true route and health flag
    /// keep their own buffers. Both predictions share one blocking readback.
    fn sample(
        &self,
        launch: &Launch<'_>,
        x: DeviceRange<'_>,
        future_routers: &[&DeviceBuffer],
    ) -> Result<Vec<Vec<i32>>, EngError> {
        if future_routers.is_empty() {
            return Ok(Vec::new());
        }
        let picks = self.shape.top_k * WORD;
        let status = (HORIZONS + 1) * picks;
        self.scratch.write_at(launch.stream, status, &[0; WORD])?;
        for (i, router) in future_routers.iter().enumerate() {
            launch.ctx.router(
                launch.stream,
                RouterCall {
                    shape: self.shape,
                    x,
                    weights: RouterWeights {
                        w: router.range(0, router.bytes())?,
                        w_shared: None,
                    },
                    logits: self
                        .scratch
                        .range(status + WORD, router_logits_bytes(self.shape)?)?,
                    out: RouterOut {
                        ids: self.scratch.range(i * picks, picks)?,
                        weights: self.scratch.range(HORIZONS * picks, picks)?,
                        shared_gate: None,
                        status: self.scratch.range(status, WORD)?,
                    },
                },
            )?;
        }
        let mut ranked = vec![0; future_routers.len() * picks];
        self.scratch.read_at(launch.stream, 0, &mut ranked)?;
        Ok(ranked
            .chunks_exact(picks)
            .map(|row| {
                row.as_chunks::<WORD>()
                    .0
                    .iter()
                    .copied()
                    .map(i32::from_le_bytes)
                    .collect()
            })
            .collect())
    }

    pub(super) fn record(
        &self,
        launch: &Launch<'_>,
        input: DeviceRange<'_>,
        actual: ActualRoute<'_>,
        future_routers: &[&DeviceBuffer],
    ) -> Result<(), EngError> {
        let started = Instant::now();
        let predictions = self.sample(launch, input, future_routers)?;
        let probe_ns = started.elapsed().as_nanos();
        let resident: Vec<u64> = actual
            .slots
            .chunks(64)
            .map(|slots| {
                slots
                    .iter()
                    .enumerate()
                    .fold(0, |mask, (i, &slot)| mask | (u64::from(slot >= 0) << i))
            })
            .collect();
        eprintln!(
            "flash-next router proxy: {}",
            serde_json::json!({
                "position": actual.position,
                "layer": actual.layer,
                "actual": actual.picks,
                "resident": resident,
                "predictions": predictions,
                "probe_ns": probe_ns,
            })
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::project::cuda::CudaProjector;

    #[test]
    #[ignore = "needs a CUDA device"]
    fn ranks_both_future_routers_without_changing_the_input() {
        assert!(crate::kernels::cuda::device_present(
            "Flash-Next router proxy"
        ));
        let projector = CudaProjector::new().unwrap();
        let ctx = projector.context();
        let stream = ctx.stream().unwrap();
        let launch = Launch {
            ctx,
            stream: &stream,
        };
        let write = |values: &[f32]| {
            let bytes: Vec<_> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            let buffer = ctx.allocate(bytes.len()).unwrap();
            buffer.write_at(&stream, 0, &bytes).unwrap();
            (buffer, bytes)
        };
        let probe = RouterProxy::build(ctx, 4, 4).unwrap();
        let (row, expected_input) = write(&[1.0, 0.0, 0.0, 0.0]);
        let (first, _) = write(&[
            3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let (second, _) = write(&[
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ]);
        let input = row.range(0, row.bytes()).unwrap();
        assert_eq!(
            probe.sample(&launch, input, &[&first, &second]).unwrap(),
            [vec![0, 1, 2, 3], vec![2, 0, 3, 1]]
        );
        assert_eq!(
            probe.sample(&launch, input, &[&second]).unwrap(),
            [vec![2, 0, 3, 1]]
        );
        assert!(probe.sample(&launch, input, &[]).unwrap().is_empty());
        let mut got = vec![0; row.bytes()];
        row.read_at(&stream, 0, &mut got).unwrap();
        assert_eq!(got, expected_input);
    }
}
