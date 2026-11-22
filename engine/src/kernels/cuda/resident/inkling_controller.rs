//! Apply the Q/K/V cross-pass gates and weighted RMSNorm on the projection stream.

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
        self.ensure_usable()?;
        let workspace = self.raw_range(&workspace)?;
        let health = self.raw_range(&health)?;
        self.launch_on(stream, |raw| launch(raw, workspace, health, width, eps))
    }
}
