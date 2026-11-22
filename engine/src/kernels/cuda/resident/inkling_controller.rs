//! Apply the Q/K/V cross-pass gates and weighted RMSNorm on the projection stream.

#[cfg(test)]
mod tests;

use crate::EngError;
use crate::kernels::cuda::{
    CudaContext, DeviceRange, DeviceStream, native::resident::inkling_controller::launch,
};

impl CudaContext {
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
    /// Health has one float per role; workspace and health must not overlap.
    pub(crate) fn inkling_controller_norm_roles(
        &self,
        stream: &DeviceStream,
        workspace: DeviceRange<'_>,
        health: DeviceRange<'_>,
        (shape, eps): ([usize; 2], f32),
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let workspace = self.raw_range(&workspace)?;
        let health = self.raw_range(&health)?;
        self.launch_on(stream, |raw| launch(raw, workspace, health, shape, eps))
    }
}
