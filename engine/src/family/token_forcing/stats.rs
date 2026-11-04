//! The statistics of the two rejection rules: score intervals for a mismatch rate and for the
//! difference of two rates, a sequence-clustered interval as a diagnostic, and the exact
//! conditional binomial test that compares two event rates.

use crate::EngError;
use serde::Serialize;

/// The standard normal quantile at 0.95: a one-sided bound at 95% confidence.
pub(crate) const Z_ONE_SIDED_95: f64 = 1.644_853_626_951_472_2;
/// The standard normal quantile at 0.975: the two ends of a two-sided 95% interval.
pub(crate) const Z_TWO_SIDED_95: f64 = 1.959_963_984_540_054;

/// A closed interval.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct Interval {
    pub(crate) lower: f64,
    pub(crate) upper: f64,
}

/// The Wilson score interval of `successes / trials` at the normal quantile `z`.
pub(crate) fn wilson(successes: u64, trials: u64, z: f64) -> Result<Interval, EngError> {
    if trials == 0 {
        return Err(stats_error("a rate needs at least one trial"));
    }
    if successes > trials {
        return Err(stats_error("a rate has more successes than trials"));
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    Ok(Interval {
        lower: (center - half).max(0.0),
        upper: (center + half).min(1.0),
    })
}

/// Newcombe's hybrid score interval (his method 10) for `p1 - p2`, built from the two Wilson
/// intervals at the same quantile. It keeps its coverage at rates near zero, where the Wald
/// interval of a difference collapses, and mismatch rates of a sound system sit near zero.
pub(crate) fn newcombe_difference(
    first: (u64, u64),
    second: (u64, u64),
    z: f64,
) -> Result<Interval, EngError> {
    let p1 = first.0 as f64 / first.1.max(1) as f64;
    let p2 = second.0 as f64 / second.1.max(1) as f64;
    let w1 = wilson(first.0, first.1, z)?;
    let w2 = wilson(second.0, second.1, z)?;
    let d = p1 - p2;
    Ok(Interval {
        lower: d - ((p1 - w1.lower).powi(2) + (w2.upper - p2).powi(2)).sqrt(),
        upper: d + ((w1.upper - p1).powi(2) + (p2 - w2.lower).powi(2)).sqrt(),
    })
}

/// The ratio estimate of a mismatch rate over sequences, with its cluster-robust variance.
///
/// Tokens of one sequence are not independent draws: a repetitive output repeats its near ties.
/// Treating each sequence as one cluster gives the design-based variance
/// `k / (k - 1) * sum((m_i - p n_i)^2) / (sum n_i)^2` of the pooled rate `p`.
pub(crate) fn clustered_rate(clusters: &[(u64, u64)]) -> Result<(f64, f64), EngError> {
    if clusters.len() < 2 {
        return Err(stats_error("a clustered rate needs at least two sequences"));
    }
    let tokens: u64 = clusters.iter().map(|&(_, n)| n).sum();
    if tokens == 0 {
        return Err(stats_error("a clustered rate needs at least one token"));
    }
    let mismatches: u64 = clusters.iter().map(|&(m, _)| m).sum();
    let p = mismatches as f64 / tokens as f64;
    let k = clusters.len() as f64;
    let squares: f64 = clusters
        .iter()
        .map(|&(m, n)| (m as f64 - p * n as f64).powi(2))
        .sum();
    let variance = k / (k - 1.0) * squares / (tokens as f64).powi(2);
    Ok((p, variance))
}

/// The upper tail `P(X >= k)` of `X ~ Binomial(n, p)`, summed in log space so that a long
/// sequence of events does not underflow the terms.
pub(crate) fn binomial_upper_tail(k: u64, n: u64, p: f64) -> Result<f64, EngError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(stats_error("a binomial probability lies outside [0, 1]"));
    }
    if k > n {
        return Ok(0.0);
    }
    if k == 0 || p == 1.0 {
        return Ok(1.0);
    }
    if p == 0.0 {
        return Ok(0.0);
    }
    let log_p = p.ln();
    let log_q = (1.0 - p).ln();
    // log C(n, i), advanced by one factor per step.
    let mut log_choose = 0.0_f64;
    let mut terms = Vec::with_capacity((n - k + 1) as usize);
    for i in 0..=n {
        if i > 0 {
            log_choose += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        if i >= k {
            terms.push(log_choose + i as f64 * log_p + (n - i) as f64 * log_q);
        }
    }
    let top = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = terms.iter().map(|term| (term - top).exp()).sum();
    Ok((top + sum.ln()).exp().min(1.0))
}

/// The one-sided p-value that the first event rate exceeds the second: conditional on the total
/// number of events, the first count is `Binomial(e1 + e2, t1 / (t1 + t2))` when both rates are
/// equal, where `t` is the number of tokens each rate is measured over.
pub(crate) fn rate_excess_p_value(first: (u64, u64), second: (u64, u64)) -> Result<f64, EngError> {
    let (e1, t1) = first;
    let (e2, t2) = second;
    if t1 == 0 || t2 == 0 {
        return Err(stats_error("an event rate needs at least one token"));
    }
    let share = t1 as f64 / (t1 as f64 + t2 as f64);
    binomial_upper_tail(e1, e1 + e2, share)
}

fn stats_error(message: &str) -> EngError {
    EngError::BadRequest(format!("token forcing statistics: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected} by more than {tolerance}"
        );
    }

    #[test]
    fn wilson_matches_published_values() {
        // Newcombe (1998), table I: 81/263 gives 0.2553-0.3662 at 95%.
        let interval = wilson(81, 263, Z_TWO_SIDED_95).unwrap();
        close(interval.lower, 0.2553, 5e-5);
        close(interval.upper, 0.3662, 5e-5);
        // Zero successes keep a positive upper bound: 0/10 gives 0-0.2775.
        let zero = wilson(0, 10, Z_TWO_SIDED_95).unwrap();
        assert_eq!(zero.lower, 0.0);
        close(zero.upper, 0.2775, 5e-5);
    }

    #[test]
    fn newcombe_matches_published_values() {
        // Newcombe (1998), method 10, example (a): 56/70 - 48/80 gives 0.0524-0.3339.
        let interval = newcombe_difference((56, 70), (48, 80), Z_TWO_SIDED_95).unwrap();
        close(interval.lower, 0.0524, 5e-5);
        close(interval.upper, 0.3339, 5e-5);
        // Example (g): 0/10 - 0/20 gives -0.1611-0.2775.
        let zeros = newcombe_difference((0, 10), (0, 20), Z_TWO_SIDED_95).unwrap();
        close(zeros.lower, -0.1611, 5e-5);
        close(zeros.upper, 0.2775, 5e-5);
    }

    #[test]
    fn empty_and_impossible_rates_are_refused() {
        assert!(wilson(0, 0, Z_TWO_SIDED_95).is_err());
        assert!(wilson(3, 2, Z_TWO_SIDED_95).is_err());
        assert!(rate_excess_p_value((1, 0), (1, 5)).is_err());
        assert!(clustered_rate(&[(1, 10)]).is_err());
    }

    #[test]
    fn binomial_tail_matches_direct_sums() {
        // P(X >= 3), X ~ Binomial(5, 0.5) = 16/32.
        close(binomial_upper_tail(3, 5, 0.5).unwrap(), 0.5, 1e-12);
        // P(X >= 1), X ~ Binomial(10, 0.1) = 1 - 0.9^10.
        close(
            binomial_upper_tail(1, 10, 0.1).unwrap(),
            1.0 - 0.9_f64.powi(10),
            1e-12,
        );
        assert_eq!(binomial_upper_tail(0, 7, 0.3).unwrap(), 1.0);
        assert_eq!(binomial_upper_tail(8, 7, 0.3).unwrap(), 0.0);
        // 0.5^3000 underflows f64, so a direct sum would lose every term; the log-space sum
        // keeps both ends of a long count.
        close(binomial_upper_tail(1, 3000, 0.5).unwrap(), 1.0, 1e-12);
        let far = binomial_upper_tail(1700, 3000, 0.5).unwrap();
        assert!(far > 0.0 && far < 1e-12, "{far}");
    }

    #[test]
    fn event_excess_is_significant_only_with_enough_events() {
        // Equal exposure: three events against none is p = 1/8, not significant at 0.05.
        close(
            rate_excess_p_value((3, 1000), (0, 1000)).unwrap(),
            0.125,
            1e-12,
        );
        // Five against none is p = 1/32.
        close(
            rate_excess_p_value((5, 1000), (0, 1000)).unwrap(),
            1.0 / 32.0,
            1e-12,
        );
        // No events at all is no evidence.
        assert_eq!(rate_excess_p_value((0, 1000), (0, 1000)).unwrap(), 1.0);
    }

    #[test]
    fn clustered_variance_grows_with_uneven_sequences() {
        let even = clustered_rate(&[(5, 1000), (5, 1000), (5, 1000)]).unwrap();
        let uneven = clustered_rate(&[(15, 1000), (0, 1000), (0, 1000)]).unwrap();
        close(even.0, uneven.0, 1e-15);
        assert_eq!(even.1, 0.0);
        assert!(uneven.1 > 0.0);
    }
}
