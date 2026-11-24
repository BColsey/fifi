//! Apply the Q/K/V cross-pass gates and weighted RMSNorm on the projection stream.

#[cfg(test)]
mod tests;

use crate::EngError;
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

pub(crate) use super::super::native::resident::inkling_controller::Arithmetic;
use super::super::native::resident::inkling_controller::{launch, prepare, rebuild_gates};

impl CudaContext {
    /// Rebuild gates from the host's exact RMS scales. The workspace is packed
    /// as gate outputs, [input, message, gamma] triples, then scale pairs.
    pub(crate) fn inkling_rebuild_gates(
        &self,
        stream: &DeviceStream,
        workspace: DeviceRange<'_>,
        shape: [usize; 2],
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let workspace = self.raw_range(&workspace)?;
        self.launch_on(stream, |raw| rebuild_gates(raw, workspace, shape))
    }

    /// Enqueue gate-half RMS normalization and broadcast the shared gamma into
    /// the controller's raw triples. The five ranges are `[pairs, gamma, triples,
    /// gate_pairs, half_health]`; health is written separately from controller apply.
    pub(crate) fn inkling_prepare_controller(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 5],
        shape_eps: ([usize; 2], f32),
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let [a, b, c, d, e] = ranges.each_ref().map(|r| self.raw_range(r));
        let raw = [a?, b?, c?, d?, e?];
        self.launch_on(stream, |stream| prepare(stream, raw, shape_eps))
    }

    /// The workspace packs three [input, message, gamma] triples, three
    /// [gain, forget] pairs and three normalized output rows, in that order.
    /// Health is three floats: zero on success or NaN on invalid arithmetic.
    pub(crate) fn inkling_controller_norm(
        &self,
        stream: &DeviceStream,
        workspace: DeviceRange<'_>,
        health: DeviceRange<'_>,
        (width, eps): (usize, f32),
    ) -> Result<(), EngError> {
        self.inkling_controller_norm_roles(stream, workspace, health, ([width, 3], eps))
    }

    /// Packs 1..=1536 independent roles: triples, pairs, then outputs.
    /// Always uses generic arithmetic. Health has one float per role;
    /// workspace and health must not overlap.
    pub(crate) fn inkling_controller_norm_roles(
        &self,
        stream: &DeviceStream,
        workspace: DeviceRange<'_>,
        health: DeviceRange<'_>,
        shape_eps: ([usize; 2], f32),
    ) -> Result<(), EngError> {
        self.inkling_controller_norm_with_arithmetic(
            stream,
            workspace,
            health,
            shape_eps,
            Arithmetic::Generic,
        )
    }

    /// Same packed layout and health contract; grouped prefill may opt into parity.
    pub(crate) fn inkling_controller_norm_with_arithmetic(
        &self,
        stream: &DeviceStream,
        workspace: DeviceRange<'_>,
        health: DeviceRange<'_>,
        (shape, eps): ([usize; 2], f32),
        arithmetic: Arithmetic,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let workspace = self.raw_range(&workspace)?;
        let health = self.raw_range(&health)?;
        self.launch_on(stream, |raw| {
            launch(raw, workspace, health, shape, eps, arithmetic)
        })
    }
}
