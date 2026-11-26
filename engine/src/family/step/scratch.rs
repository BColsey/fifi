//! Where the scratch of a step comes from.
//!
//! A model declares its scratch once, as a function of an [`Alloc`]. Run with [`OnDevice`] the
//! declaration allocates the buffers; run with [`Count`] it only adds up their bytes, so the
//! memory plan of a load cannot drift from what the load then allocates.

use std::cell::Cell;

use super::rows::q8;
use crate::EngError;
use crate::kernels::cuda::{CudaContext, DeviceBuffer};

pub(crate) trait Alloc {
    type Buf;

    /// `bytes` bytes whose content the step writes before it reads.
    fn bytes(&self, bytes: usize) -> Result<Self::Buf, EngError>;

    /// `bytes` zero bytes.
    fn zeroed(&self, bytes: usize) -> Result<Self::Buf, EngError>;

    /// `count` f32 values.
    fn floats(&self, count: usize) -> Result<Self::Buf, EngError> {
        self.bytes(count * size_of::<f32>())
    }

    /// The q8_1 rows of `rows` rows of `width` values.
    fn q8(&self, width: usize, rows: usize) -> Result<Self::Buf, EngError> {
        self.bytes(q8(width, rows))
    }
}

/// Allocates on the device of a context.
pub(crate) struct OnDevice<'a>(pub(crate) &'a CudaContext);

impl Alloc for OnDevice<'_> {
    type Buf = DeviceBuffer;

    fn bytes(&self, bytes: usize) -> Result<DeviceBuffer, EngError> {
        self.0.allocate(bytes)
    }

    fn zeroed(&self, bytes: usize) -> Result<DeviceBuffer, EngError> {
        self.0.upload(&vec![0u8; bytes])
    }
}

/// Adds up the bytes a declaration asks for and allocates nothing.
#[derive(Default)]
pub(crate) struct Count(Cell<u64>);

impl Count {
    pub(crate) fn total(&self) -> u64 {
        self.0.get()
    }

    fn add(&self, bytes: usize) {
        self.0.set(self.0.get().saturating_add(bytes as u64));
    }
}

impl Alloc for Count {
    type Buf = ();

    fn bytes(&self, bytes: usize) -> Result<(), EngError> {
        self.add(bytes);
        Ok(())
    }

    fn zeroed(&self, bytes: usize) -> Result<(), EngError> {
        self.add(bytes);
        Ok(())
    }
}
