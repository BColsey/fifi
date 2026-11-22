//! Prepare one local Inkling attention row before resident relative attention.

use crate::EngError;
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream, native::resident};

#[derive(Clone, Copy)]
pub(crate) struct PrepShape {
    pub queries: usize,
    pub kv: usize,
    pub dims: usize,
}

impl PrepShape {
    pub(crate) const WIDTH: usize = 128;
    pub(crate) const TAPS: usize = 4;

    pub(crate) fn words(self) -> [usize; 10] {
        let Self { queries, kv, dims } = self;
        let q = queries * Self::WIDTH;
        let rel = queries * dims;
        let kv_words = kv * Self::WIDTH;
        let aux = 1 + q + rel + 2 * (2 * Self::TAPS - 1) * kv_words;
        [
            q,
            rel,
            kv_words,
            kv_words,
            kv_words,
            kv_words,
            queries,
            q,
            aux,
            1 + 2 * (queries + kv),
        ]
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PrepMath {
    pub qk_limit: f32,
    pub rel_limit: f32,
    pub eps: f32,
}

impl CudaContext {
    pub(crate) fn inkling_local_prep(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 10],
        shape: PrepShape,
        history: usize,
        math: PrepMath,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ranges = ranges.each_ref().map(|range| self.raw_range(range));
        let [a, b, c, d, e, f, g, h, i, j] = ranges;
        let ranges = [a?, b?, c?, d?, e?, f?, g?, h?, i?, j?];
        self.launch_on(stream, |raw| {
            resident::inkling_local_prep::launch(raw, ranges, shape, history, math)
        })
    }
}
