//! A checked, bit-exact grouped-to-tiled value-head permutation for one device row.
//! Queues one kernel without allocating or synchronizing. Source and destination must be
//! disjoint; invalid geometry and ranges are refused before a native call.
use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::head_tile::{HeadTileRaw, head_tile_launch};

#[derive(Clone, Copy)]
pub(crate) struct HeadTileShape {
    pub(crate) key_heads: usize,
    pub(crate) value_heads: usize,
    pub(crate) head_dim: usize,
}
pub(crate) struct HeadTileCall<'a> {
    pub(crate) src: DeviceRange<'a>,
    pub(crate) dst: DeviceRange<'a>,
    pub(crate) shape: HeadTileShape,
}

impl CudaContext {
    pub(crate) fn head_tile(
        &self,
        stream: &DeviceStream,
        call: HeadTileCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let s = call.shape;
        let bad = || EngError::BadRequest("head_tile: invalid geometry".into());
        if s.key_heads == 0
            || s.value_heads == 0
            || s.head_dim == 0
            || !s.value_heads.is_multiple_of(s.key_heads)
            || i32::try_from(s.key_heads).is_err()
            || i32::try_from(s.value_heads).is_err()
            || i32::try_from(s.head_dim).is_err()
        {
            return Err(bad());
        }
        let bytes = s
            .value_heads
            .checked_mul(s.head_dim)
            .filter(|&n| n <= 1 << 24)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(bad)?;
        let src = self.raw_range(&call.src)?;
        let dst = self.raw_range(&call.dst)?;
        require_fit("src", &call.src, bytes, 4)?;
        require_fit("dst", &call.dst, bytes, 4)?;
        require_apart(
            &Claim::of("dst", &call.dst, bytes),
            &Claim::of("src", &call.src, bytes),
        )?;
        let raw = HeadTileRaw {
            src,
            dst,
            key_heads: s.key_heads,
            value_heads: s.value_heads,
            head_dim: s.head_dim,
        };
        self.launch_on(stream, |stream_raw| head_tile_launch(stream_raw, &raw))
    }
}
#[cfg(test)]
mod tests;
