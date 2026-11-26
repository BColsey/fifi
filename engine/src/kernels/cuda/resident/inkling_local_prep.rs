//! Prepare one local Inkling attention row before resident relative attention.

use crate::EngError;
use crate::kernels::cuda::{
    CudaContext, DeviceRange, DeviceStream, PinnedBuf,
    context::{Claim, require_apart, require_fit},
    native::resident,
};

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

#[derive(Clone, Copy)]
pub(crate) struct OutputConvShape {
    pub rows: usize,
    pub width: usize,
    pub history: usize,
}

impl OutputConvShape {
    pub(crate) fn words(self) -> Option<[usize; 5]> {
        let Self {
            rows,
            width,
            history,
        } = self;
        let taps = PrepShape::TAPS;
        if !(1..=512).contains(&rows) || !(1..=4096).contains(&width) || history >= taps {
            return None;
        }
        let plane = rows * width;
        Some([plane, (taps - 1) * width, taps * width, plane, rows])
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DepthShape {
    pub rows: usize,
    pub width: usize,
    pub depth: usize,
}

impl DepthShape {
    pub(crate) fn words(self) -> Option<[usize; 4]> {
        let Self { rows, width, depth } = self;
        if !(1..=512).contains(&rows) || !(1..=4096).contains(&width) || !(1..=16).contains(&depth)
        {
            return None;
        }
        let plane = rows * width;
        Some([depth * plane, width, rows * depth, plane])
    }
}

fn validate_planes<const N: usize>(
    names: [&'static str; N],
    ranges: &[DeviceRange<'_>; N],
    words: [usize; N],
) -> Result<(), EngError> {
    for (i, (name, range)) in names.into_iter().zip(ranges).enumerate() {
        let bytes = words[i] * size_of::<f32>();
        if range.bytes() != bytes {
            return Err(EngError::BadRequest(format!(
                "wrong CUDA Inkling {name} extent"
            )));
        }
        require_fit(name, range, bytes, align_of::<f32>())?;
        let claim = Claim::of(name, range, bytes);
        for (prev, range) in names.into_iter().zip(&ranges[..i]) {
            require_apart(&claim, &Claim::of(prev, range, range.bytes()))?;
        }
    }
    Ok(())
}

impl CudaContext {
    /// Ordered FP32 RMSNorm followed by gamma; only the status rows are read back.
    pub(crate) fn inkling_weighted_norm(
        &self,
        stream: &DeviceStream,
        transfer: &mut PinnedBuf,
        ranges: [DeviceRange<'_>; 4],
        [rows, width]: [usize; 2],
        eps: f32,
    ) -> Result<(), EngError> {
        let [plane, ..] = (DepthShape {
            rows,
            width,
            depth: 1,
        })
        .words()
        .filter(|_| eps.is_finite() && eps > 0.0)
        .ok_or_else(|| {
            EngError::BadRequest("invalid CUDA Inkling norm geometry or epsilon".into())
        })?;
        validate_planes(
            ["input", "gamma", "output", "status"],
            &ranges,
            [plane, width, plane, rows],
        )?;
        let [a, b, c, d] = ranges.each_ref().map(|r| self.raw_range(r));
        let raw = [a?, b?, c?, d?];
        let queued = self
            .launch_on(stream, |stream| {
                resident::inkling_local_prep::weighted_norm(stream, raw, [rows, width], eps)
            })
            .and_then(|()| self.copy_to_pinned(stream, ranges[3], transfer, 0));
        transfer.synchronize_after_enqueue(stream, queued)?;
        let mut status = vec![0.0; rows];
        transfer.read_f32(0, &mut status)?;
        if status.iter().any(|&v| v != 0.0) {
            return Err(EngError::BadRequest(
                "non-finite CUDA Inkling MLP norm".into(),
            ));
        }
        Ok(())
    }

    /// Launch-only, single-row RMSNorm/gamma. The transaction owns the status
    /// word until its final fence; validation still precedes the launch.
    pub(crate) fn enqueue_row_weighted_norm(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 4],
        width: usize,
        eps: f32,
    ) -> Result<(), EngError> {
        if !(1..=4096).contains(&width) || !eps.is_finite() || eps <= 0. {
            return Err(EngError::BadRequest(
                "invalid resident row norm geometry".into(),
            ));
        }
        validate_planes(
            ["input", "gamma", "output", "status"],
            &ranges,
            [width, width, width, 1],
        )?;
        let [a, b, c, d] = ranges.each_ref().map(|r| self.raw_range(r));
        let raw = [a?, b?, c?, d?];
        self.launch_on(stream, |s| {
            resident::inkling_local_prep::weighted_norm(s, raw, [1, width], eps)
        })
    }

    /// Form the next raw residual and its normalized scoring plane from
    /// `[prior, convolved_attention, next_raw, next_normalized]` on one stream.
    /// At a block boundary the caller retains the sealed prior separately.
    pub(crate) fn inkling_depth_current(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 4],
        [rows, width]: [usize; 2],
        boundary: bool,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let [plane, ..] = (DepthShape {
            rows,
            width,
            depth: 1,
        })
        .words()
        .ok_or_else(|| EngError::BadRequest("invalid Inkling current depth shape".into()))?;
        validate_planes(
            ["prior", "attention", "raw", "normalized"],
            &ranges,
            [plane; 4],
        )?;
        let [a, b, c, d] = ranges.each_ref().map(|r| self.raw_range(r));
        let ranges = [a?, b?, c?, d?];
        self.launch_on(stream, |stream| {
            resident::inkling_local_prep::depth_current(stream, ranges, [rows, width], boundary)
        })
    }

    /// Keep normalized depth planes and the mixed output on device. The compact
    /// round trip preserves host `f32::exp` rounding without moving hidden rows.
    /// `planes` are in scoring order (newest sealed to oldest, then current).
    pub(crate) fn inkling_depth_mix(
        &self,
        stream: &DeviceStream,
        transfer: &mut PinnedBuf,
        [planes, query, compact, output]: [DeviceRange<'_>; 4],
        shape: DepthShape,
    ) -> Result<(), EngError> {
        let [_, _, scores, _] = shape
            .words()
            .ok_or_else(|| EngError::BadRequest("invalid CUDA Inkling depth shape".into()))?;
        let queued = self
            .inkling_depth_pass(stream, [planes, query, compact], shape, true)
            .and_then(|()| self.copy_to_pinned(stream, compact, transfer, 0));
        transfer.synchronize_after_enqueue(stream, queued)?;
        let mut values = vec![0.0; scores];
        transfer.read_f32(0, &mut values)?;
        for row in values.chunks_exact_mut(shape.depth) {
            if row.iter().any(|v| !v.is_finite()) {
                return Err(EngError::BadRequest(
                    "non-finite CUDA Inkling depth score".into(),
                ));
            }
            let peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for v in row {
                *v = (*v - peak).exp();
            }
        }
        let upload: Vec<_> = values.into_iter().map(f32::to_ne_bytes).collect();
        transfer.write(0, upload.as_flattened())?;
        let queued = self
            .copy_from_pinned(stream, transfer, 0, compact)
            .and_then(|()| {
                self.inkling_depth_pass(stream, [planes, compact, output], shape, false)
            });
        transfer.synchronize_after_enqueue(stream, queued)
    }

    fn inkling_depth_pass(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 3],
        shape: DepthShape,
        scores_only: bool,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let [a, b, c] = ranges.each_ref().map(|r| self.raw_range(r));
        let ranges = [a?, b?, c?];
        self.launch_on(stream, |stream| {
            resident::inkling_local_prep::depth_mix(stream, ranges, shape, scores_only)
        })
    }

    /// Apply ordered static taps on resident projected rows. The input plane is
    /// unchanged and its tail can seed the next tile or the first decode row.
    /// The caller initializes status, which the kernel only ORs into.
    pub(crate) fn inkling_output_conv(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 5],
        shape: OutputConvShape,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let [raw, past, weight, result, status] = ranges.each_ref().map(|r| self.raw_range(r));
        let ranges = [raw?, past?, weight?, result?, status?];
        self.launch_on(stream, |stream| {
            resident::inkling_local_prep::output_conv(stream, ranges, shape)
        })
    }

    pub(crate) fn inkling_effective_taps(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 3],
        width: usize,
    ) -> Result<(), EngError> {
        let [a, b, c] = ranges.each_ref().map(|r| self.raw_range(r));
        let rows = [a?, b?, c?];
        self.launch_on(stream, |s| {
            resident::inkling_local_prep::effective_taps(s, rows, width)
        })
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::{
        device_present,
        testkit::{read_bytes, upload_bytes},
    };

    fn bytes(values: &[f32]) -> Vec<[u8; 4]> {
        values.iter().map(|x| x.to_ne_bytes()).collect()
    }

    fn reference(planes: &[f32], query: &[f32], shape: DepthShape, row: usize) -> Vec<u32> {
        let DepthShape { rows, width, depth } = shape;
        let at = |d| (d * rows + row) * width;
        let scale = (width as f32).sqrt().recip();
        let dots: Vec<f32> = (0..depth)
            .map(|d| {
                planes[at(d)..][..width]
                    .iter()
                    .zip(query)
                    .map(|(x, q)| x * q)
                    .sum::<f32>()
                    * scale
            })
            .collect();
        let peak = dots.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut result = vec![0.0; width];
        let mut mass = 0.0;
        for (d, s) in dots.into_iter().enumerate() {
            let weight = (s - peak).exp();
            mass += weight;
            for (sum, x) in result.iter_mut().zip(&planes[at(d)..][..width]) {
                *sum += weight * x;
            }
        }
        result.into_iter().map(|x| (x / mass).to_bits()).collect()
    }

    /// `ordered_rms_scale` and the scaled output, op for op in scalar order.
    fn ordered_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<u32> {
        let squares = x.iter().fold(-0.0f32, |sum, v| sum + v * v);
        let scale = 1.0 / (squares / x.len() as f32 + eps).sqrt();
        x.iter()
            .zip(gamma)
            .map(|(v, g)| ((v * scale) * g).to_bits())
            .collect()
    }

    #[test]
    fn weighted_norm_matches_the_scalar_order_bit_for_bit() {
        if !device_present("Inkling CUDA weighted norm") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let mut seed = 41_u32;
        let mut generate = |n: usize| {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 8) as f32 * (1.0 / 8388608.0) - 1.0
                })
                .collect::<Vec<_>>()
        };
        for [rows, width] in [[1, 129], [1, 1536], [1, 2304], [1, 4096], [3, 2304]] {
            let shape = DepthShape {
                rows,
                width,
                depth: 1,
            };
            let [plane, ..] = shape.words().unwrap();
            let input = generate(plane);
            let gamma = generate(width);
            let x = upload_bytes(&ctx, bytes(&input).as_flattened());
            let g = upload_bytes(&ctx, bytes(&gamma).as_flattened());
            let y = ctx.allocate(plane * 4).unwrap();
            let status = ctx.allocate(rows * 4).unwrap();
            let mut transfer = ctx.pinned_buf(rows * 4).unwrap();
            let eps = 1.0e-6f32;
            ctx.inkling_weighted_norm(
                &stream,
                &mut transfer,
                [
                    x.range(0, x.bytes()).unwrap(),
                    g.range(0, g.bytes()).unwrap(),
                    y.range(0, y.bytes()).unwrap(),
                    status.range(0, status.bytes()).unwrap(),
                ],
                [rows, width],
                eps,
            )
            .unwrap();
            let actual = read_bytes(&ctx, &y, y.bytes());
            for (r, row) in actual.chunks_exact(width * 4).enumerate() {
                let row = row
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| u32::from_ne_bytes(*v));
                assert!(
                    row.eq(ordered_norm(&input[r * width..][..width], &gamma, eps)),
                    "{rows} rows, width {width}, row {r}"
                );
            }
        }
    }

    #[test]
    fn depth_mix_round_trips_only_compact_rows_and_matches_scalar_bits() {
        if !device_present("Inkling CUDA depth mix") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let mut seed = 15_u32;
        let mut generate = |n| {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 8) as f32 * (1.0 / 8388608.0) - 1.0
                })
                .collect::<Vec<_>>()
        };
        for [rows, width, depth] in [[1, 32, 1], [33, 129, 6], [511, 2304, 9]] {
            let shape = DepthShape { rows, width, depth };
            let [p, q, c, o] = shape.words().unwrap();
            let planes = generate(p);
            let query = generate(q);
            let p = upload_bytes(&ctx, bytes(&planes).as_flattened());
            let q = upload_bytes(&ctx, bytes(&query).as_flattened());
            let c = ctx.allocate(c * 4).unwrap();
            let o = ctx.allocate(o * 4).unwrap();
            let mut transfer = ctx.pinned_buf(c.bytes()).unwrap();
            ctx.inkling_depth_mix(
                &stream,
                &mut transfer,
                [
                    p.range(0, p.bytes()).unwrap(),
                    q.range(0, q.bytes()).unwrap(),
                    c.range(0, c.bytes()).unwrap(),
                    o.range(0, o.bytes()).unwrap(),
                ],
                shape,
            )
            .unwrap();
            let actual = read_bytes(&ctx, &o, o.bytes());
            for (r, row) in actual.chunks_exact(width * 4).enumerate() {
                let row = row
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| u32::from_ne_bytes(*v));
                assert!(
                    row.eq(reference(&planes, &query, shape, r)),
                    "{rows}x{width} depth {depth} row {r}"
                );
            }
        }
    }
}
