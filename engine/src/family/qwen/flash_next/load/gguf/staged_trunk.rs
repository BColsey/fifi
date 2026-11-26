//! Provenance-bound, staged GGUF layer-trunk upload.
//!
//! Only this module can assemble the owned weights: every layer and the optional final mixer
//! and head are uploaded from the same open model. The expert bank can use the borrowed model
//! and the step can use these weights without accepting a caller-provided identity token.
//! External checkpoint mutation remains outside this guarantee. This is not production admission
//! or startup.

use std::ops::Range;

use super::GgufModel;
use crate::{
    EngError,
    family::project::cuda::CudaWeights,
    kernels::cuda::{CudaContext, DeviceStream},
};

pub(in super::super::super) struct GgufTrunk<'a> {
    model: &'a GgufModel,
    weights: CudaWeights,
    layers: Range<usize>,
}

fn check_layers(model: &GgufModel, layers: &Range<usize>) -> Result<(), EngError> {
    if layers.start >= layers.end || layers.end > model.config().layers() {
        return Err(EngError::BadRequest(
            "Flash-Next GGUF trunk: invalid layer window".into(),
        ));
    }
    Ok(())
}

impl<'a> GgufTrunk<'a> {
    /// The caller admits the whole layer window, any transient staging, and later expert and
    /// step allocations before calling. No captured or preassembled weights are accepted.
    pub(in super::super::super) fn upload(
        model: &'a GgufModel,
        ctx: &CudaContext,
        stream: &DeviceStream,
        layers: Range<usize>,
    ) -> Result<Self, EngError> {
        Self::upload_inner(model, ctx, stream, layers, false, |_, _| Ok(()))
    }

    /// Include the final mixer and head, uploaded from the same model as the layers.
    /// The caller must also admit those weights and their transient staging.
    pub(in super::super::super) fn upload_with_final(
        model: &'a GgufModel,
        ctx: &CudaContext,
        stream: &DeviceStream,
        layers: Range<usize>,
    ) -> Result<Self, EngError> {
        Self::upload_inner(model, ctx, stream, layers, true, |_, _| Ok(()))
    }

    /// Test-only staged upload: the caller checks each group before allocation and verifies
    /// its completed weight payload before the next group. `None` denotes the final head.
    #[cfg(test)]
    pub(in super::super::super) fn upload_with_final_guarded(
        model: &'a GgufModel,
        ctx: &CudaContext,
        stream: &DeviceStream,
        layers: Range<usize>,
        guard: impl FnMut(
            Option<usize>,
            Option<&[crate::family::project::cuda::CudaWeight]>,
        ) -> Result<(), EngError>,
    ) -> Result<Self, EngError> {
        Self::upload_inner(model, ctx, stream, layers, true, guard)
    }

    fn upload_inner(
        model: &'a GgufModel,
        ctx: &CudaContext,
        stream: &DeviceStream,
        layers: Range<usize>,
        with_final: bool,
        mut guard: impl FnMut(
            Option<usize>,
            Option<&[crate::family::project::cuda::CudaWeight]>,
        ) -> Result<(), EngError>,
    ) -> Result<Self, EngError> {
        check_layers(model, &layers)?;
        ctx.check_launch_stream(stream)?;
        let mut weights = Vec::new();
        for layer in layers.clone() {
            guard(Some(layer), None)?;
            let group = model.upload_layer_trunk(layer, ctx, stream)?;
            guard(Some(layer), Some(&group))?;
            weights.extend(group);
        }
        if with_final {
            guard(None, None)?;
            let group = model.upload_final_trunk(ctx, stream)?;
            guard(None, Some(&group))?;
            weights.extend(group);
        }
        let weights = CudaWeights::from_native(weights)?;
        Ok(Self {
            model,
            weights,
            layers,
        })
    }

    pub(in super::super::super) fn model(&self) -> &'a GgufModel {
        self.model
    }

    pub(in super::super::super) fn weights(&self) -> &CudaWeights {
        &self.weights
    }

    pub(in super::super::super) fn layers(&self) -> &Range<usize> {
        &self.layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen::flash_next::load::gguf::testkit::{Spec, write_flash};

    #[test]
    #[allow(clippy::reversed_empty_ranges)] // a reversed range is one of the invalid inputs under test
    fn layer_window_rejects_empty_inverted_and_outside_model() {
        let file = write_flash(&Spec::default());
        let model = GgufModel::open(&file.first).unwrap().unwrap();
        let count = model.config().layers();
        assert!(check_layers(&model, &(0..count)).is_ok());
        for invalid in [0..0, 2..1, count..count + 1, 0..count + 1] {
            assert!(
                check_layers(&model, &invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
    }
}
