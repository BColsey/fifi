//! KV cache row acceptance guard and F16 publication, device side.
//!
//! [CudaContext::kv_f16_rows] queues the kernel on a [DeviceStream] and returns
//! when it is queued. It allocates nothing, copies nothing and never
//! synchronizes. The device twin of the host `f16_row` guard: a value is
//! accepted only when it is finite, has its low 16 bits clear and converts to a
//! finite F16 under round-to-nearest-even, and the stored half-word is that
//! conversion, the same bits `half::f16::from_f32` gives the host. Any reject
//! sets the sticky fault flag; the caller rejects the token with the host error
//! text once per token, before sampling.
//!
//! Argument errors return BadRequest before a native launch. Validation checks
//! backend usability, buffer ownership, and checked size arithmetic first: rows
//! in 1..=65536, columns in 1..=2^22, rows*columns <= 2^32. x must hold
//! rows*columns f32 values, the optional out rows*columns half-words, and flag
//! one u32, all at 4-byte alignment and pairwise disjoint. Stream ownership is
//! checked after range planning. A rejected call queues no work. Existing
//! terminal errors keep their error mapping.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::kv_f16_rows::{KvF16RowsRaw, kv_f16_rows_launch};

/// One kv f16 call: rows of `columns` f32 values in `x`; when `out` is present
/// the accepted half-words are written there, otherwise the call is a guard
/// pass only. `flag` is the sticky fault word the caller initializes.
pub(crate) struct KvF16RowsCall<'b> {
    pub(crate) x: DeviceRange<'b>,
    pub(crate) out: Option<DeviceRange<'b>>,
    pub(crate) flag: DeviceRange<'b>,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
}

impl CudaContext {
    /// Queues the KV f16 acceptance guard of call on stream, writing call.out
    /// when present.
    ///
    /// Returns BadRequest for a stream or buffer of another adapter, geometry
    /// outside the documented ranges, a short or misaligned range, or ranges
    /// that overlap.
    pub(crate) fn kv_f16_rows(
        &self,
        stream: &DeviceStream,
        call: KvF16RowsCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        if !(1..=65536).contains(&call.rows) || !(1..=(1usize << 22)).contains(&call.columns) {
            return Err(EngError::BadRequest(
                "KV F16 rows: invalid geometry or overflow".into(),
            ));
        }
        let values = call
            .rows
            .checked_mul(call.columns)
            .filter(|values| *values <= (1usize << 32))
            .ok_or_else(|| {
                EngError::BadRequest("KV F16 rows: invalid geometry or overflow".into())
            })?;
        let mut checks: Vec<(&str, &DeviceRange<'_>, usize)> =
            vec![("KV F16 in region", &call.x, values * 4)];
        if let Some(out) = &call.out {
            checks.push(("KV F16 out region", out, values * 2));
        }
        checks.push(("KV F16 flag region", &call.flag, 4));
        for (i, (name, range, need)) in checks.iter().enumerate() {
            self.raw_range(range)?;
            require_fit(name, range, *need, 4)?;
            for (name_j, range_j, need_j) in checks.iter().take(i) {
                require_apart(
                    &Claim::of(name, range, *need),
                    &Claim::of(name_j, range_j, *need_j),
                )?;
            }
        }
        let raw = KvF16RowsRaw {
            x: self.raw_range(&call.x)?,
            y: match &call.out {
                Some(out) => Some(self.raw_range(out)?),
                None => None,
            },
            flag: self.raw_range(&call.flag)?,
            rows: call.rows,
            columns: call.columns,
        };
        self.launch_on(stream, move |raw_stream| {
            kv_f16_rows_launch(raw_stream, &raw)
        })
    }
}

#[cfg(test)]
mod tests;
