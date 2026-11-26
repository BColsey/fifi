//! Wide cross-pass MLP controller. Tile scratch is reused on one stream; the
//! finished activation slab outlives it and feeds the router and expert gather.

use super::{
    ControllerWeights, bad,
    controller_prefill::{admits_controller_weights, batch_controller_calls},
};
use crate::{
    EngError,
    family::inkling2::cuda_projection::DEVICE_HEADROOM,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream, PinnedBuf, resident::router::MAX_ROWS,
    },
};

const WORD: usize = size_of::<f32>();
const TILE_BYTES: usize = 8 << 20;

#[derive(Clone, Copy)]
struct Plan {
    rows: usize,
    width: usize,
    rank: usize,
    tile: usize,
}

impl Plan {
    fn new([rows, width, rank]: [usize; 3]) -> Option<Self> {
        if !(2..=MAX_ROWS).contains(&rows)
            || !(4..=4096).contains(&width)
            || !width.is_multiple_of(4)
            || !(1..=4096).contains(&rank)
        {
            return None;
        }
        // Workspace is gates + reduction + raw triples, coefficients and output.
        let tile = (TILE_BYTES / ((8 * width + rank) * WORD))
            .min(rows)
            .min(1536);
        Some(Self {
            rows,
            width,
            rank,
            tile,
        })
    }

    fn allocation(self) -> [usize; 3] {
        let Self {
            rows,
            width,
            rank,
            tile,
        } = self;
        [
            (2 * rows + 1) * width,
            (8 * width + rank) * tile,
            (width + 3) * rows,
        ]
        .map(|n| n * WORD)
    }
}

/// The import contains [input, message] pairs and one gamma row, not gates or
/// repeated weights. A separate slab holds normalized outputs followed by receipts.
pub(in crate::family::inkling2) struct ResidentController {
    plan: Plan,
    input: DeviceBuffer,
    scratch: DeviceBuffer,
    output: DeviceBuffer,
    transfer: PinnedBuf,
    stream: DeviceStream,
}

impl ResidentController {
    pub(in crate::family::inkling2::cuda_projection) fn admits(
        shape: [usize; 3],
        weights: ControllerWeights<'_>,
    ) -> bool {
        // Weight extent and column alignment do not depend on the last tile's row count.
        Plan::new(shape).is_some_and(|plan| admits_controller_weights(weights, plan.tile))
    }

    pub(in crate::family::inkling2) fn allocate(
        ctx: &CudaContext,
        shape: [usize; 3],
    ) -> Result<Option<Self>, EngError> {
        let Some(plan) = Plan::new(shape) else {
            return Ok(None);
        };
        let [input, scratch, output] = plan.allocation();
        if (ctx.memory_info()?.free as u64).saturating_sub((input + scratch + output) as u64)
            < DEVICE_HEADROOM
        {
            return Ok(None);
        }
        let allocate = || {
            Ok(Self {
                plan,
                stream: ctx.stream()?,
                input: ctx.allocate(input)?,
                scratch: ctx.allocate(scratch)?,
                output: ctx.allocate(output)?,
                transfer: ctx.pinned_buf(input)?,
            })
        };
        match allocate() {
            Ok(workspace) if ctx.memory_info()?.free as u64 >= DEVICE_HEADROOM => {
                Ok(Some(workspace))
            }
            Ok(_) | Err(EngError::MemoryPlan(_)) => Ok(None),
            Err(error) => Err(error),
        }
    }

    pub(in crate::family::inkling2) fn fits(&self, [rows, width, rank]: [usize; 3]) -> bool {
        (2..=self.plan.rows).contains(&rows) && [width, rank] == [self.plan.width, self.plan.rank]
    }

    pub(in crate::family::inkling2) fn tile_rows(&self) -> usize {
        self.plan.tile
    }

    pub(in crate::family::inkling2::cuda_projection) fn run(
        &mut self,
        ctx: &CudaContext,
        weights: ControllerWeights<'_>,
        (inputs, messages, gamma): (&[Vec<f32>], &[Vec<f32>], &[f32]),
        eps: (f32, f32),
    ) -> Result<DeviceRange<'_>, EngError> {
        let m = inputs.len();
        let h = self.plan.width;
        self.check(weights, m, eps)?;
        if messages.len() != m
            || gamma.len() != h
            || inputs.iter().chain(messages).any(|row| row.len() != h)
        {
            return Err(bad("invalid resident Inkling prefill controller shape"));
        }
        let upload = words(&self.input, 0, (2 * m + 1) * h)?;
        self.transfer.fill(upload.bytes(), |bytes| {
            let source = inputs
                .iter()
                .zip(messages)
                .flat_map(|(x, message)| x.iter().chain(message))
                .chain(gamma);
            for (dst, value) in bytes.as_chunks_mut::<WORD>().0.iter_mut().zip(source) {
                dst.copy_from_slice(&value.to_ne_bytes());
            }
            Ok(())
        })?;
        let queued = ctx.copy_from_pinned(&self.stream, &mut self.transfer, 0, upload);
        self.finish(ctx, weights, m, eps, queued)
    }

    /// Where a device producer writes the `rows` [input, message] pairs and the gamma row that
    /// [`Self::run_written`] reads.
    pub(in crate::family::inkling2) fn inputs(
        &self,
        rows: usize,
    ) -> Result<[DeviceRange<'_>; 2], EngError> {
        let h = self.plan.width;
        if !self.fits([rows, h, self.plan.rank]) {
            return Err(bad("invalid resident Inkling prefill controller rows"));
        }
        Ok([
            words(&self.input, 0, 2 * rows * h)?,
            words(&self.input, 2 * rows * h, h)?,
        ])
    }

    /// [`Self::run`] of the `rows` pairs a device producer wrote into [`Self::inputs`]; the
    /// producer's stream must have finished them.
    pub(in crate::family::inkling2::cuda_projection) fn run_written(
        &mut self,
        ctx: &CudaContext,
        weights: ControllerWeights<'_>,
        rows: usize,
        eps: (f32, f32),
    ) -> Result<DeviceRange<'_>, EngError> {
        self.check(weights, rows, eps)?;
        self.finish(ctx, weights, rows, eps, Ok(()))
    }

    fn check(
        &self,
        weights: ControllerWeights<'_>,
        rows: usize,
        (gate_eps, eps): (f32, f32),
    ) -> Result<(), EngError> {
        let Plan { width: h, rank, .. } = self.plan;
        if !self.fits([rows, h, rank])
            || weights.map(|(w, _)| w.shape) != [[rank, 2 * h], [h, rank], [h, rank]]
            || [gate_eps, eps].iter().any(|&e| !e.is_finite() || e <= 0.0)
        {
            return Err(bad("invalid resident Inkling prefill controller shape"));
        }
        Ok(())
    }

    /// Runs the tiles after `queued` (the input's upload, or nothing), then reads the health.
    fn finish(
        &mut self,
        ctx: &CudaContext,
        weights: ControllerWeights<'_>,
        m: usize,
        eps: (f32, f32),
        queued: Result<(), EngError>,
    ) -> Result<DeviceRange<'_>, EngError> {
        let h = self.plan.width;
        let health = words(&self.output, m * h, 3 * m)?;
        let queued = queued
            .and_then(|()| self.enqueue(ctx, weights, m, eps))
            .and_then(|()| ctx.copy_to_pinned(&self.stream, health, &mut self.transfer, 0));
        self.transfer
            .synchronize_after_enqueue(&self.stream, queued)?;
        let mut receipt = vec![0.0; 3 * m];
        self.transfer.read_f32(0, &mut receipt)?;
        if receipt.iter().any(|&v| v != 0.0) {
            return Err(bad("non-finite resident Inkling prefill controller"));
        }
        words(&self.output, 0, m * h)
    }

    fn enqueue(
        &self,
        ctx: &CudaContext,
        weights: ControllerWeights<'_>,
        rows: usize,
        (gate_eps, eps): (f32, f32),
    ) -> Result<(), EngError> {
        let Plan {
            width: h,
            rank,
            tile,
            ..
        } = self.plan;
        let gamma = words(&self.input, 2 * rows * h, h)?;
        for first in (0..rows).step_by(tile) {
            let m = (rows - first).min(tile);
            let plane = m * h;
            let scratch = |at, n| words(&self.scratch, at, n);
            let gate = scratch(0, 2 * plane)?;
            let reduced = scratch(2 * plane, m * rank)?;
            let work = 2 * plane + m * rank;
            let calls = batch_controller_calls(
                weights,
                m,
                [gate, reduced, scratch(work + 3 * plane, 2 * plane)?],
            )?;
            let pairs = words(&self.input, 2 * first * h, 2 * plane)?;
            let health = words(&self.output, rows * h + 2 * first, 2 * m)?;
            ctx.inkling_prepare_controller(
                &self.stream,
                [pairs, gamma, scratch(work, 3 * plane)?, gate, health],
                ([h, m], gate_eps),
            )?;
            ctx.dense_gemv_group(&self.stream, &calls)?;
            let health = words(&self.output, rows * h + 2 * rows + first, m)?;
            ctx.inkling_controller_norm_roles(
                &self.stream,
                scratch(work, 6 * plane)?,
                health,
                ([h, m], eps),
            )?;
            ctx.copy_device(
                &self.stream,
                scratch(work + 5 * plane, plane)?,
                words(&self.output, first * h, plane)?,
            )?;
        }
        Ok(())
    }
}

fn words(buffer: &DeviceBuffer, start: usize, count: usize) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(start * WORD, count * WORD)
}

#[cfg(test)]
mod tests;
