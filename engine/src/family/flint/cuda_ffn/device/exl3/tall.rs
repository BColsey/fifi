//! Tall single-matrix EXL3 projections for the layer-major jobs pass.
//!
//! A run of at most [`TILE_MAX_ROWS`] rows goes through the rows entry (SIMT tile below 64
//! rows, tensor cores from 64), which decodes each weight tile once per 64 rows instead of once
//! per row. A taller piece is cut into the fewest even runs, so no run falls back to the
//! per-row kernels. A call of eight rows or fewer keeps the split-K small-row entry.

use super::bad;
use super::layer_plan::{SMALL_MAX_ROWS, TILE_MAX_ROWS};
use crate::{
    EngError,
    family::project::cuda::{Exl3MatrixView, exl3::Exl3Io},
    kernels::cuda::{CudaContext, DeviceRange, DeviceStream},
};

/// `rows` cut into the fewest runs of at most [`TILE_MAX_ROWS`], as even as possible, as
/// `(first, count)` pairs.
pub(super) fn row_slices(rows: usize) -> Vec<(usize, usize)> {
    if rows == 0 {
        return Vec::new();
    }
    let count = rows.div_ceil(TILE_MAX_ROWS);
    let (base, extra) = (rows / count, rows % count);
    let mut first = 0;
    (0..count)
        .map(|index| {
            let rows = base + usize::from(index < extra);
            let slice = (first, rows);
            first += rows;
            slice
        })
        .collect()
}

/// `count` rows of `width` F32 values of `range`, starting `first` rows in.
fn window<'a>(
    range: DeviceRange<'a>,
    (first, count): (usize, usize),
    width: usize,
) -> Result<DeviceRange<'a>, EngError> {
    let row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| bad("tall projection row width overflows"))?;
    let offset = first
        .checked_mul(row_bytes)
        .and_then(|bytes| bytes.checked_add(range.offset()))
        .ok_or_else(|| bad("tall projection offset overflows"))?;
    let bytes = count
        .checked_mul(row_bytes)
        .ok_or_else(|| bad("tall projection span overflows"))?;
    range.buffer().range(offset, bytes)
}

/// Projects `io.rows` rows of `io.x` through `weight`: the small-row entry up to eight rows, the
/// rows entry in even runs above. `io.xh`, `io.z` and `io.out` hold `io.rows` rows of their widths.
pub(in crate::family::flint::cuda_ffn::device) fn linear_rows(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weight: Exl3MatrixView<'_>,
    io: Exl3Io<'_>,
) -> Result<(), EngError> {
    if io.rows == 0 {
        return Err(bad("tall projection needs at least one row"));
    }
    if io.rows <= SMALL_MAX_ROWS {
        return ctx.exl3_linear(stream, weight.linear_call(&io));
    }
    let [k, n] = weight.shape;
    for (first, count) in row_slices(io.rows) {
        let part = Exl3Io {
            x: window(io.x, (first, count), k)?,
            xh: window(io.xh, (first, count), k)?,
            z: window(io.z, (first, count), n)?,
            out: window(io.out, (first, count), n)?,
            flag: io.flag,
            rows: count,
            fused: false,
        };
        ctx.exl3_linear_midrows(stream, weight.linear_call(&part))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_cover_the_rows_once_and_stay_on_the_rows_entry() {
        for rows in 1..=3000 {
            let slices = row_slices(rows);
            assert_eq!(slices.iter().map(|&(_, count)| count).sum::<usize>(), rows);
            let mut next = 0;
            for &(first, count) in &slices {
                assert_eq!(first, next, "runs are contiguous");
                assert!(count <= TILE_MAX_ROWS, "rows {rows}");
                next += count;
            }
            if rows > TILE_MAX_ROWS {
                assert!(slices.iter().all(|&(_, count)| count >= 72), "rows {rows}");
            }
        }
        assert_eq!(row_slices(0), []);
        assert_eq!(row_slices(1024).len(), 8);
        assert_eq!(row_slices(145), [(0, 73), (73, 72)]);
    }
}
