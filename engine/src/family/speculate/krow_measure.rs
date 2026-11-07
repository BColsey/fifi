//! The host arithmetic of the K-row measurement tests (`krow_alpha_device_tests`,
//! `krow_step_device_tests`): everything that needs no device and can be tested on the host.
//!
//! The measurements replace the estimated per-row costs of the K-row verifier design with numbers
//! from the real pack: the marginal cost of one more row through the rows kernel (alpha), the
//! phases of one captured step, the expert union of K consecutive tokens, and the routed-expert
//! chain of K rows. This file holds the definitions those numbers share, so a device run and a
//! host test read the same formulas:
//!
//! * alpha(R) = (t(R) - t(1)) / ((R - 1) * t(1)), where t(1) is the time of one single-row launch
//!   and t(R) the time of one launch at R rows. The time-weighted alpha of a set of roles weights
//!   each role by its own t(1) of one whole token (a sweep over every layer that has the role), so
//!   it is (sum of t(R) - t(1)) / ((R - 1) * sum of t(1)).
//! * the expert union of K tokens in one layer is the number of distinct ids among their K * top_k
//!   picks; windows slide one token at a time.

/// The row counts of the alpha measurement. 1 is the single-row (wide1) reference.
pub(crate) const ROWS: [usize; 4] = [1, 2, 4, 8];
/// The window sizes of the expert union.
pub(crate) const UNION_WINDOWS: [usize; 7] = [2, 3, 4, 5, 6, 7, 8];

/// alpha(R) = (t(R) - t(1)) / ((R - 1) * t(1)); `None` for R = 1 or a non-positive t(1).
pub(crate) fn alpha(t1: f64, t_rows: f64, rows: usize) -> Option<f64> {
    if rows < 2 || !t1.is_finite() || t1 <= 0.0 || !t_rows.is_finite() {
        return None;
    }
    Some((t_rows - t1) / ((rows - 1) as f64 * t1))
}

/// The time-weighted alpha of `roles` (each `(t(1), t(R))` of a whole-token sweep): the sum of the
/// marginal times over (R - 1) times the sum of the solo times.
pub(crate) fn weighted_alpha(roles: &[(f64, f64)], rows: usize) -> Option<f64> {
    let solo: f64 = roles.iter().map(|r| r.0).sum();
    let marginal: f64 = roles.iter().map(|r| r.1 - r.0).sum();
    if rows < 2 || !solo.is_finite() || solo <= 0.0 || !marginal.is_finite() {
        return None;
    }
    Some(marginal / ((rows - 1) as f64 * solo))
}

/// A float output that differs between the rows kernel and the single-row launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Mismatch {
    pub(crate) row: usize,
    pub(crate) col: usize,
    /// The element index in the flat `[rows][row_values]` output.
    pub(crate) index: usize,
    pub(crate) left_bits: u32,
    pub(crate) right_bits: u32,
}

/// The first differing 32-bit word of two equally long little-endian f32 buffers laid out as
/// `[rows][row_values]`, compared by bits (a signed zero or a NaN payload counts). `None` when
/// the buffers are equal; a length that is not a whole number of words or differs between the two
/// is reported as a mismatch at index `usize::MAX`.
pub(crate) fn first_word_difference(
    left: &[u8],
    right: &[u8],
    row_values: usize,
) -> Option<Mismatch> {
    if left.len() != right.len() || !left.len().is_multiple_of(4) || row_values == 0 {
        return Some(Mismatch {
            row: usize::MAX,
            col: usize::MAX,
            index: usize::MAX,
            left_bits: 0,
            right_bits: 0,
        });
    }
    let word = |bytes: &[u8], at: usize| {
        u32::from_le_bytes([
            bytes[4 * at],
            bytes[4 * at + 1],
            bytes[4 * at + 2],
            bytes[4 * at + 3],
        ])
    };
    (0..left.len() / 4).find_map(|index| {
        let (a, b) = (word(left, index), word(right, index));
        (a != b).then_some(Mismatch {
            row: index / row_values,
            col: index % row_values,
            index,
            left_bits: a,
            right_bits: b,
        })
    })
}

/// The sizes of the distinct experts among `k` consecutive tokens of one layer, for every start.
/// `picks` holds the `top_k` ids of each token in order. A negative id (a faulty router row) is a
/// refusal: the union of a window with an invalid pick says nothing.
pub(crate) fn union_sizes(picks: &[Vec<i32>], k: usize) -> Result<Vec<usize>, String> {
    if k == 0 {
        return Err("a window of zero tokens".into());
    }
    if let Some(bad) = picks.iter().flatten().find(|id| **id < 0) {
        return Err(format!("the router picked the invalid id {bad}"));
    }
    Ok(picks
        .windows(k)
        .map(|window| {
            let mut ids: Vec<i32> = window.iter().flatten().copied().collect();
            ids.sort_unstable();
            ids.dedup();
            ids.len()
        })
        .collect())
}

/// The p-th percentile (0..=100) by the nearest-rank rule on a copy of `values`; `None` if empty.
pub(crate) fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=100.0).contains(&p) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    Some(sorted[rank.min(sorted.len()) - 1])
}

pub(crate) fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// `items` rotated left by `round` places: each round starts the sequence at a different item, so
/// drift of the clocks or the temperature does not always hit the same measurement first.
pub(crate) fn rotated<T: Clone>(items: &[T], round: usize) -> Vec<T> {
    if items.is_empty() {
        return Vec::new();
    }
    let shift = round % items.len();
    items[shift..]
        .iter()
        .chain(&items[..shift])
        .cloned()
        .collect()
}

/// A small deterministic generator of roughly normal values (the sum of four uniforms, centred and
/// scaled to unit variance), seeded per measurement so a rerun feeds the same inputs.
pub(crate) struct Inputs(u64);

impl Inputs {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u32(&mut self) -> u32 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }

    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / 4_294_967_296.0
    }

    pub(crate) fn normal(&mut self) -> f32 {
        // Four uniforms on [0, 1) have mean 2 and variance 1/3.
        let sum: f32 = (0..4).map(|_| self.unit()).sum();
        (sum - 2.0) * 3.0_f32.sqrt()
    }

    pub(crate) fn values(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.normal()).collect()
    }
}

pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
