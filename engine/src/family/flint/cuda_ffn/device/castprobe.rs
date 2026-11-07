//! Test-only probe of what the F16 cast loses at the inputs of the tall single-matrix projections of the jobs pass.
//!
//! The rows entry rounds the rotated activations (`xh`, F32) to F16 before the tensor-core multiply. When armed, every tall
//! projection (`tall::linear_rows`) hands its `xh` to [`observe`] after the call; the probe reads it back and measures, per layer and
//! projection shape, how the cast treats it:
//!   * range: elements that overflow F16 (|x| of 65520 or more rounds to infinity), elements in the F16 subnormal range
//!     (2^-24 <= |x| < 2^-14, fewer mantissa bits) and elements that round to zero;
//!   * mantissa: the relative L2 error of the cast, split by those classes, against the error that 10 mantissa bits alone predict
//!     (sum over normal elements of ulp^2 / 12, round-to-nearest-even errors being uniform);
//!   * candidates on the same values: BF16, F16 after a per-row power-of-two scale, a hi/lo F16 split (x = hi + lo), and the same
//!     split with lo scaled by 2^11 (what a two-accumulator kernel would feed the tensor cores).
//!
//! The routed experts of the jobs pass round their input rows inside the job-table kernel and are not observed here.
//! Nothing in this module is compiled outside test builds.

use half::{bf16, f16};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use crate::{
    EngError,
    kernels::cuda::{DeviceRange, DeviceStream},
};

/// Sum of squares of the values and of the errors of each treatment, over the non-overflowing elements.
#[derive(Clone, Debug, Default)]
pub(crate) struct Stats {
    pub(crate) calls: u64,
    pub(crate) elements: u64,
    pub(crate) zeros: u64,
    /// Elements whose F16 cast is infinite (|x| >= 65520).
    pub(crate) overflow: u64,
    /// Elements whose F16 cast is a subnormal (2^-24 <= |x| < 2^-14, or rounds up into it).
    pub(crate) subnormal: u64,
    /// Non-zero elements whose F16 cast is zero.
    pub(crate) flushed: u64,
    pub(crate) max_abs: f64,
    pub(crate) energy: f64,
    /// Energy of the values by class: normal, subnormal, flushed.
    pub(crate) class_energy: [f64; 3],
    /// F16 cast error energy by class: normal, subnormal, flushed.
    pub(crate) class_error: [f64; 3],
    /// What 10 mantissa bits alone predict for the normal class: sum of ulp^2 / 12.
    pub(crate) predicted_normal: f64,
    pub(crate) error_bf16: f64,
    pub(crate) error_row_pow2: f64,
    pub(crate) error_hilo: f64,
    pub(crate) error_hilo_scaled: f64,
    /// Worst row of the F16 cast: relative L2 error of that row.
    pub(crate) worst_row: f64,
    /// Largest max|x| / rms(x) of a row.
    pub(crate) peak_over_rms: f64,
}

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static LAYER: Cell<usize> = const { Cell::new(0) };
    static STATS: RefCell<BTreeMap<(usize, usize, usize), Stats>> = const { RefCell::new(BTreeMap::new()) };
}

pub(crate) fn arm() {
    ARMED.with(|armed| armed.set(true));
    STATS.with(|stats| stats.borrow_mut().clear());
}

pub(crate) fn armed() -> bool {
    ARMED.with(Cell::get)
}

pub(crate) fn set_layer(layer: usize) {
    LAYER.with(|current| current.set(layer));
}

/// The statistics by (layer, k, n), draining the probe.
pub(crate) fn take() -> BTreeMap<(usize, usize, usize), Stats> {
    ARMED.with(|armed| armed.set(false));
    STATS.with(|stats| std::mem::take(&mut *stats.borrow_mut()))
}

fn read(stream: &DeviceStream, range: DeviceRange<'_>) -> Result<Vec<f32>, EngError> {
    let mut bytes = vec![0u8; range.bytes()];
    range.buffer().read_at(stream, range.offset(), &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
        .collect())
}

/// `FLINT_PROBE_ROW_STRIDE` (default 32): the probe measures one row in this many, to keep the host side to about a minute.
fn row_stride() -> usize {
    static STRIDE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *STRIDE.get_or_init(|| {
        std::env::var("FLINT_PROBE_ROW_STRIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&s| s >= 1)
            .unwrap_or(32)
    })
}

/// Reads `rows` rows of `width` rotated activations and adds their cast statistics under (layer, width, n).
pub(crate) fn observe(
    stream: &DeviceStream,
    xh: DeviceRange<'_>,
    rows: usize,
    width: usize,
    n: usize,
) -> Result<(), EngError> {
    stream.synchronize()?;
    let values = read(stream, xh)?;
    assert!(
        values.len() >= rows * width,
        "the probe read fewer values than rows x width"
    );
    let layer = LAYER.with(Cell::get);
    STATS.with(|all| {
        let mut all = all.borrow_mut();
        let stats = all.entry((layer, width, n)).or_default();
        stats.calls += 1;
        // Every `stride`-th row, the first one moving with the call so that all rows of a piece are sampled over the layers.
        let stride = row_stride();
        let first = (stats.calls as usize * 7 + layer) % stride;
        for (index, row) in values[..rows * width].chunks(width).enumerate() {
            if index % stride == first {
                add_row(stats, row);
            }
        }
    });
    Ok(())
}

/// F16 round-to-nearest-even, the cast of `__floats2half2_rn` (`half` 2.7.1 `f16::from_f32` rounds to nearest, ties to even,
/// and gives subnormals or zero below the normal range).
fn f16_round(x: f64) -> f64 {
    f64::from(f16::from_f32(x as f32).to_f32())
}

fn bf16_round(x: f64) -> f64 {
    f64::from(bf16::from_f32(x as f32).to_f32())
}

/// Unit in the last place of a normal F16 value of magnitude `a`.
fn f16_ulp(a: f64) -> f64 {
    2f64.powi(a.log2().floor() as i32 - 10)
}

const F16_MIN_NORMAL: f64 = 6.103_515_625e-5; // 2^-14
const F16_OVERFLOW: f64 = 65_520.0; // rounds to infinity from here

pub(crate) fn add_row(stats: &mut Stats, row: &[f32]) {
    let row_max = row.iter().fold(0f64, |m, &v| m.max(f64::from(v).abs()));
    let row_energy: f64 = row.iter().map(|&v| f64::from(v).powi(2)).sum();
    let rms = (row_energy / row.len() as f64).sqrt();
    if rms > 0.0 {
        stats.peak_over_rms = stats.peak_over_rms.max(row_max / rms);
    }
    // A per-row power-of-two scale that brings the row maximum into [2^14, 2^15).
    let scale = if row_max > 0.0 {
        2f64.powi(14 - row_max.log2().floor() as i32)
    } else {
        1.0
    };
    let mut row_error = 0f64;
    let mut row_ok_energy = 0f64;
    for &value in row {
        let x = f64::from(value);
        let a = x.abs();
        stats.elements += 1;
        stats.max_abs = stats.max_abs.max(a);
        if a == 0.0 {
            stats.zeros += 1;
            continue;
        }
        let hi = f16_round(x);
        if !hi.is_finite() {
            stats.overflow += 1;
            continue;
        }
        let x2 = x * x;
        stats.energy += x2;
        row_ok_energy += x2;
        let err = (hi - x).powi(2);
        let class = if hi == 0.0 {
            stats.flushed += 1;
            2
        } else if hi.abs() < F16_MIN_NORMAL {
            stats.subnormal += 1;
            1
        } else {
            stats.predicted_normal += f16_ulp(a).powi(2) / 12.0;
            0
        };
        stats.class_energy[class] += x2;
        stats.class_error[class] += err;
        row_error += err;
        stats.error_bf16 += (bf16_round(x) - x).powi(2);
        stats.error_row_pow2 += (f16_round(x * scale) / scale - x).powi(2);
        stats.error_hilo += (hi + f16_round(x - hi) - x).powi(2);
        stats.error_hilo_scaled += (hi + f16_round((x - hi) * 2048.0) / 2048.0 - x).powi(2);
        let _ = F16_OVERFLOW;
    }
    if row_ok_energy > 0.0 {
        stats.worst_row = stats.worst_row.max((row_error / row_ok_energy).sqrt());
    }
}

impl Stats {
    pub(crate) fn merge(&mut self, other: &Stats) {
        self.calls += other.calls;
        self.elements += other.elements;
        self.zeros += other.zeros;
        self.overflow += other.overflow;
        self.subnormal += other.subnormal;
        self.flushed += other.flushed;
        self.max_abs = self.max_abs.max(other.max_abs);
        self.energy += other.energy;
        for i in 0..3 {
            self.class_energy[i] += other.class_energy[i];
            self.class_error[i] += other.class_error[i];
        }
        self.predicted_normal += other.predicted_normal;
        self.error_bf16 += other.error_bf16;
        self.error_row_pow2 += other.error_row_pow2;
        self.error_hilo += other.error_hilo;
        self.error_hilo_scaled += other.error_hilo_scaled;
        self.worst_row = self.worst_row.max(other.worst_row);
        self.peak_over_rms = self.peak_over_rms.max(other.peak_over_rms);
    }

    fn rel(&self, error: f64) -> f64 {
        (error / self.energy.max(f64::MIN_POSITIVE)).sqrt()
    }

    /// One line of figures; the relative L2 errors are over the non-overflowing elements.
    pub(crate) fn line(&self) -> String {
        let f16_error: f64 = self.class_error.iter().sum();
        let share =
            |class: usize| 100.0 * self.class_error[class] / f16_error.max(f64::MIN_POSITIVE);
        format!(
            "calls {} elems {} zeros {} overflow {} subnormal {} flushed {} max|x| {:.3e} peak/rms {:.1} | F16 relL2 {:.3e} (mantissa-only prediction {:.3e}; \
             error energy: normal {:.1}% subnormal {:.1}% flushed {:.1}%) worst row {:.3e} | BF16 {:.3e} F16+row-pow2 {:.3e} hi/lo {:.3e} hi/lo*2^11 {:.3e}",
            self.calls,
            self.elements,
            self.zeros,
            self.overflow,
            self.subnormal,
            self.flushed,
            self.max_abs,
            self.peak_over_rms,
            self.rel(f16_error),
            self.rel(self.predicted_normal),
            share(0),
            share(1),
            share(2),
            self.worst_row,
            self.rel(self.error_bf16),
            self.rel(self.error_row_pow2),
            self.rel(self.error_hilo),
            self.rel(self.error_hilo_scaled),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cast_classes_and_errors_follow_the_f16_format() {
        let mut stats = Stats::default();
        // normal, subnormal, flushed to zero, overflow, zero, and a value on a rounding boundary of the normal range.
        add_row(&mut stats, &[1.0, 3.0e-5, 1.0e-9, 70_000.0, 0.0, 65_519.0]);
        assert_eq!(stats.elements, 6);
        assert_eq!(
            (stats.zeros, stats.overflow, stats.subnormal, stats.flushed),
            (1, 1, 1, 1)
        );
        // 1.0 and 65519 (rounds to 65504) are normal; 1.0 is exact.
        assert!(stats.class_error[0] > 0.0, "65519 rounds to 65504");
        assert!((stats.class_error[0] - (65_519.0f64 - 65_504.0).powi(2)).abs() < 1e-6);
        // a flushed element's whole value is the error
        assert!((stats.class_error[2] - 1.0e-18).abs() < 1e-24);
        // hi/lo scaled is exact for 1.0 and near-exact for the subnormal 3e-5 (its lo is representable after the 2^11 scale)
        assert!(stats.error_hilo_scaled < stats.class_error.iter().sum::<f64>());
    }

    #[test]
    fn a_unit_gaussian_row_matches_the_mantissa_prediction() {
        // a deterministic quasi-normal row: the F16 error of values near 1 is mantissa-only and close to ulp^2/12 per element
        let row: Vec<f32> = (0..4096)
            .map(|i| ((i * 2654435761u64 as usize) % 2001) as f32 / 1000.0 - 1.0 + 0.5003)
            .collect();
        let mut stats = Stats::default();
        add_row(&mut stats, &row);
        assert_eq!(stats.overflow, 0);
        let measured = stats.class_error[0];
        let predicted = stats.predicted_normal;
        assert!(
            measured > 0.25 * predicted && measured < 4.0 * predicted,
            "measured {measured:e} predicted {predicted:e}"
        );
        assert!(
            stats.error_hilo_scaled < 1e-3 * measured,
            "the scaled split removes the mantissa error"
        );
    }
}
