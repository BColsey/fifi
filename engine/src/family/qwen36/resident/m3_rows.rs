//! One logits row to one scored position (M3, registration v4).
//!
//! Pure host math shared by every arm's device loop: the true-token
//! log-probability and the greedy argmax of one f32 logits row. Each arm is
//! scored against its OWN full distribution: normalization runs over every
//! head row the arm computes, padding rows included (the head width pads the
//! true vocabulary to the same width in both arms, so the effect is
//! identical, and the true id is always inside the true vocabulary —
//! disclosed in the registration). All sums run in f64. Non-finite logits
//! are a hard error in the runner; here NaN simply never ranks and NaN
//! propagates into the log-probability the runner checks.

use super::m3_quality::PositionRow;

/// `ln(sum(exp(row)))` in f64, stable at the max. An all-NaN or empty row
/// is a caller bug the runner rejects; this returns -inf / NaN faithfully.
pub(super) fn log_sum_exp(row: &[f32]) -> f64 {
    let max = row
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    if row.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    if !max.is_finite() {
        return f64::from(max);
    }
    let max = f64::from(max);
    max + row
        .iter()
        .map(|&v| (f64::from(v) - max).exp())
        .sum::<f64>()
        .ln()
}

/// The log-probability of `true_id` under the row: `logit - lse`. A `true_id`
/// outside the row is a layout bug, refused loudly.
pub(super) fn logprob_true(row: &[f32], true_id: u32) -> Result<f64, &'static str> {
    let logit = f64::from(
        row.get(true_id as usize)
            .copied()
            .ok_or("the true token id is outside the logits row")?,
    );
    Ok(logit - log_sum_exp(row))
}

/// The greedy argmax, ties to the lower id, NaN never ranks. An empty row or
/// an all-NaN row is refused (the runner treats it as a device failure).
pub(super) fn argmax_low_id(row: &[f32]) -> Result<u32, &'static str> {
    let mut best: Option<(u32, f32)> = None;
    for (id, &v) in row.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if v <= b => {}
            _ => best = Some((id as u32, v)),
        }
    }
    best.map(|(id, _)| id).ok_or("no finite entry in the logits row")
}

/// The scored-position row of one logits row: true-token log-probability
/// (f32 for storage; the scorer's bootstrap re-widens to f64) and the greedy
/// argmax equality on the true token.
pub(super) fn position_row(
    row: &[f32],
    true_id: u32,
    cluster: u32,
) -> Result<PositionRow, &'static str> {
    Ok(PositionRow {
        cluster,
        logprob_true: logprob_true(row, true_id)? as f32,
        top1_correct: argmax_low_id(row)? == true_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed golden rows: exps and sums small enough to verify by
    /// hand, checked against ln identities rather than restating the code.
    #[test]
    fn golden_rows_match_hand_computed_probabilities() {
        // exps [1, 2, 1], sum 4: P(1) = 1/2, P(0) = 1/4.
        let row = [0.0f32, 2.0f32.ln(), 0.0];
        let lp1 = logprob_true(&row, 1).unwrap();
        let lp0 = logprob_true(&row, 0).unwrap();
        // The input logits are f32 (ln 2 is rounded), so 1e-6, not 1e-12.
        assert!((lp1 - (0.5f64).ln()).abs() < 1e-6);
        assert!((lp0 - (0.25f64).ln()).abs() < 1e-6);
        assert_eq!(argmax_low_id(&row), Ok(1));
        // Probabilities sum to 1 across the row.
        let total: f64 = (0..3).map(|i| logprob_true(&row, i).unwrap().exp()).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    /// Ties break to the LOWER id, and NaN never ranks even at id 0.
    #[test]
    fn ties_break_low_and_nan_never_ranks() {
        assert_eq!(argmax_low_id(&[1.0, 1.0, 2.0]), Ok(2));
        assert_eq!(argmax_low_id(&[3.0, 3.0, 1.0]), Ok(0));
        assert_eq!(argmax_low_id(&[f32::NAN, 5.0, 5.0]), Ok(1));
        assert!(argmax_low_id(&[f32::NAN, f32::NAN]).is_err());
    }

    /// The padded-vocab rule: padding rows participate in normalization
    /// identically in both arms. exps [e,e,e,e,e,1,1,1] over width 8 with a
    /// true vocabulary of 5: P(true) = e / (5e + 3), never renormalized to
    /// the 5-row universe.
    #[test]
    fn padding_participates_in_normalization() {
        let row = [1.0f32; 5].iter().copied().chain([0.0f32; 3]).collect::<Vec<_>>();
        let lp = logprob_true(&row, 2).unwrap();
        let want = 1.0f64 - (5.0 * 1.0f64.exp() + 3.0).ln();
        assert!((lp - want).abs() < 1e-12, "got {lp}, want {want}");
        assert_eq!(argmax_low_id(&row), Ok(0));
    }

    /// A true id outside the row is refused, an empty row is refused, and a
    /// NaN row propagates NaN instead of inventing a value.
    #[test]
    fn layout_and_device_bugs_are_refused() {
        assert!(logprob_true(&[0.0, 0.0], 2).is_err());
        assert!(logprob_true(&[], 0).is_err());
        assert!(argmax_low_id(&[]).is_err());
        let nan = [f32::NAN, 0.0];
        assert!(logprob_true(&nan, 1).unwrap().is_nan());
        assert_eq!(argmax_low_id(&nan), Ok(1));
    }

    /// The runner's storage type round-trips the scored row: f32 storage is
    /// the registration's precision for per-position rows.
    #[test]
    fn position_row_carries_cluster_and_greedy_equality() {
        let row = [0.0f32, 2.0f32.ln(), 0.0];
        let good = position_row(&row, 1, 7).unwrap();
        assert_eq!(good.cluster, 7);
        assert!(good.top1_correct);
        assert!((good.logprob_true - (0.5f32).ln()).abs() < 1e-6);
        let bad = position_row(&row, 0, 8).unwrap();
        assert!(!bad.top1_correct);
    }
}
