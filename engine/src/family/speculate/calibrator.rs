//! Dynamic draft length: an online map from the drafter's confidence in a token to the chance that
//! the target accepts it, and the stop rule that reads it.
//!
//! A drafter that can score its own guesses (the MTP head knows the probability of its top
//! token) reports a score per draft. After each window the verdict labels the scores: the drafts
//! before the first mismatch were accepted, the first mismatch was rejected, and later drafts say
//! nothing because the target never judged them on their own merit. Each label lands in a bin of
//! the score; old labels fade, so the map follows the text. Before enough labels exist the map
//! has no opinion and the window keeps its full length.

const BINS: usize = 16;
/// The score of the lowest bin's lower edge; bins are one unit wide.
const FLOOR: f32 = -8.0;
/// Weight kept per round by every older label.
const DECAY: f32 = 0.995;
/// A bin with less decayed weight than this borrows the nearest bin that has enough.
const MIN_WEIGHT: f32 = 8.0;
/// Labels needed before the map answers at all.
const BURN_IN: u32 = 64;

/// The log-odds score of a draft whose top token had probability `p`.
pub(crate) fn confidence_score(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

fn bin_of(score: f32) -> usize {
    let index = (score - FLOOR).floor();
    if index > 0.0 {
        (index as usize).min(BINS - 1)
    } else {
        0
    }
}

#[derive(Clone, Copy, Default)]
struct Bin {
    hits: f32,
    total: f32,
}

/// Maps a draft score to the chance that the target accepts the draft, given that every earlier
/// draft of the window was accepted.
pub(crate) struct ConfidenceCalibrator {
    bins: [Bin; BINS],
    labels: u32,
}

impl Default for ConfidenceCalibrator {
    fn default() -> Self {
        Self {
            bins: [Bin::default(); BINS],
            labels: 0,
        }
    }
}

impl ConfidenceCalibrator {
    /// Learns from one window: `scores[j]` is the score of draft `j`, `accepted` how many drafts
    /// the target took (at most `scores.len()`).
    pub(crate) fn observe(&mut self, scores: &[f32], accepted: usize) {
        for bin in &mut self.bins {
            bin.hits *= DECAY;
            bin.total *= DECAY;
        }
        let judged = (accepted + 1).min(scores.len());
        for (j, &score) in scores[..judged].iter().enumerate() {
            let bin = &mut self.bins[bin_of(score)];
            bin.total += 1.0;
            if j < accepted {
                bin.hits += 1.0;
            }
            self.labels = self.labels.saturating_add(1);
        }
    }

    /// The chance that a draft with this score is accepted, or `None` while the map has too few
    /// labels to say.
    pub(crate) fn estimate(&self, score: f32) -> Option<f32> {
        if self.labels < BURN_IN {
            return None;
        }
        let home = bin_of(score);
        (0..BINS).find_map(|distance| {
            let below = home.checked_sub(distance);
            let above = (home + distance < BINS).then_some(home + distance);
            [below, above]
                .into_iter()
                .flatten()
                .map(|index| self.bins[index])
                .find(|bin| bin.total >= MIN_WEIGHT)
                .map(|bin| ((bin.hits + 0.5) / (bin.total + 1.0)).clamp(0.01, 0.99))
        })
    }
}

/// The stop rule of one window: a draft is verified while the product of the acceptance chances
/// of the drafts so far stays at or above the threshold.
pub(crate) struct DraftBudget {
    threshold: f32,
    product: f32,
}

impl DraftBudget {
    pub(crate) fn new(threshold: f32) -> Self {
        Self {
            threshold,
            product: 1.0,
        }
    }

    /// Decides whether the next draft, with this acceptance chance, is worth a row. An unknown
    /// chance keeps the draft. After a `false` the window ends: the draft is not verified.
    pub(crate) fn admit(&mut self, chance: Option<f32>) -> bool {
        let Some(chance) = chance else {
            return true;
        };
        self.product *= chance;
        self.product >= self.threshold
    }

    pub(crate) fn product(&self) -> f32 {
        self.product
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained(rounds: usize, scores: &[f32], accepted: usize) -> ConfidenceCalibrator {
        let mut calibrator = ConfidenceCalibrator::default();
        for _ in 0..rounds {
            calibrator.observe(scores, accepted);
        }
        calibrator
    }

    #[test]
    fn the_map_is_silent_until_the_burn_in_has_passed() {
        let mut calibrator = ConfidenceCalibrator::default();
        for _ in 0..31 {
            calibrator.observe(&[1.0, 1.0], 2);
        }
        assert_eq!(calibrator.estimate(1.0), None, "62 labels");
        calibrator.observe(&[1.0, 1.0], 2);
        assert!(calibrator.estimate(1.0).is_some(), "64 labels");
    }

    #[test]
    fn a_score_that_is_always_accepted_estimates_near_one() {
        let calibrator = trained(200, &[2.5], 1);
        let chance = calibrator.estimate(2.5).expect("estimate");
        assert!(chance > 0.95, "{chance}");
    }

    #[test]
    fn a_score_that_is_always_rejected_estimates_near_zero() {
        let calibrator = trained(200, &[-3.5], 0);
        let chance = calibrator.estimate(-3.5).expect("estimate");
        assert!(chance < 0.05, "{chance}");
    }

    #[test]
    fn drafts_after_the_first_mismatch_are_not_labelled() {
        // The second draft follows a rejected first one: only the first is judged.
        let mut calibrator = ConfidenceCalibrator::default();
        for _ in 0..100 {
            calibrator.observe(&[-3.5, 4.5], 0);
        }
        assert!(calibrator.estimate(-3.5).is_some());
        assert_eq!(
            calibrator.estimate(4.5),
            calibrator.estimate(-3.5),
            "borrowed, never labelled"
        );
    }

    #[test]
    fn an_empty_bin_borrows_the_nearest_populated_one() {
        let calibrator = trained(100, &[3.5], 1);
        let near = calibrator.estimate(3.5).expect("home");
        assert_eq!(calibrator.estimate(-6.5), Some(near));
    }

    #[test]
    fn old_labels_fade() {
        let mut calibrator = trained(100, &[1.5], 1);
        for _ in 0..2000 {
            calibrator.observe(&[1.5], 0);
        }
        let chance = calibrator.estimate(1.5).expect("estimate");
        assert!(chance < 0.05, "{chance}");
    }

    #[test]
    fn scores_outside_the_bins_land_in_the_end_bins() {
        assert_eq!(bin_of(-100.0), 0);
        assert_eq!(bin_of(f32::NAN), 0);
        assert_eq!(bin_of(100.0), BINS - 1);
        assert_eq!(bin_of(-8.0), 0);
        assert_eq!(bin_of(-6.5), 1);
    }

    #[test]
    fn the_score_is_the_log_odds() {
        assert!((confidence_score(0.5)).abs() < 1e-6);
        assert!((confidence_score(0.9) - 2.1972246).abs() < 1e-4);
        assert!(confidence_score(1.0).is_finite());
        assert!(confidence_score(0.0).is_finite());
    }

    #[test]
    fn the_budget_stops_when_the_product_falls_below_the_threshold() {
        let mut budget = DraftBudget::new(0.4);
        assert!(budget.admit(Some(0.9)));
        assert!(budget.admit(Some(0.8)), "0.72");
        assert!(budget.admit(Some(0.6)), "0.432");
        assert!(!budget.admit(Some(0.8)), "0.3456");
        assert!((budget.product() - 0.3456).abs() < 1e-5);
    }

    #[test]
    fn an_unknown_chance_keeps_the_draft_and_the_product() {
        let mut budget = DraftBudget::new(0.4);
        assert!(budget.admit(None));
        assert_eq!(budget.product(), 1.0);
    }
}
