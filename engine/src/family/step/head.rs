//! The output head of a step: the final norm of one hidden row and the projection onto the
//! vocabulary.

use super::rows::{Norm, Rows, StepIo, q8};
use crate::EngError;
use crate::kernels::cuda::{DeviceBuffer, DeviceRange, NativeWeight};

/// The weights of the head and the rows it works in.
#[derive(Clone, Copy)]
pub(crate) struct Head<'a> {
    pub(crate) norm: DeviceRange<'a>,
    pub(crate) eps: f32,
    pub(crate) weight: &'a NativeWeight,
    /// The normed row, and its q8_1 row.
    pub(crate) normed: Rows<'a>,
    /// One float per vocabulary entry.
    pub(crate) logits: &'a DeviceBuffer,
}

impl Head<'_> {
    /// The logits of the hidden row `x`, queued on the step's stream.
    pub(crate) fn logits(&self, io: &StepIo<'_>, x: DeviceRange<'_>) -> Result<(), EngError> {
        let d = self.weight.k();
        let Some(normed) = self.normed.f32 else {
            return Err(EngError::BadRequest(format!(
                "{}: the head needs f32 rows for its norm",
                io.family
            )));
        };
        let norm = Norm {
            d,
            eps: self.eps,
            x,
            delta: None,
            weight: self.norm.slice(0, d * 4)?,
        };
        io.norm(
            1,
            norm,
            normed.range(0, d * 4)?,
            self.normed.q8.range(0, q8(d, 1))?,
        )?;
        io.project(self.weight, self.normed, self.logits, 1)
    }
}
