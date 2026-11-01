//! The MTP drafter: a chain of steps of the model's own head.
//!
//! The head sees, per step, the token before the position it drafts and a hidden row: the
//! target's final hidden row for the first step, its own previous output for the next ones. The
//! device side (`MtpHead`) owns the hidden rows, the head's attention cache and the argmax; this
//! side owns the chain length. With dynamic length on, every step is scored by the probability
//! the head gave its token, a calibrator maps the score to the chance that the target accepts the
//! token, and the chain stops when the running product of those chances falls below the setting.

use super::{
    accept::Accepted,
    calibrator::{ConfidenceCalibrator, DraftBudget, confidence_score},
    driver::{Drafter, Proposal},
    window::Window,
};
use crate::EngError;

type Result<T> = std::result::Result<T, EngError>;

/// One step of the head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HeadStep {
    /// The head's argmax over the sampler vocabulary.
    pub(crate) token: u32,
    /// The probability the head gave `token`.
    pub(crate) probability: f32,
}

/// The device side of the head.
pub(crate) trait MtpHead {
    /// Starts a chain for the position after the pending token: the head's first input is the
    /// target's final hidden row of the last committed row.
    fn begin(&mut self) -> Result<()>;

    /// Drafts the token after `previous`, which is the pending token for the first step and the
    /// draft before it for the later ones.
    fn step(&mut self, previous: u32) -> Result<HeadStep>;

    /// Brings the head's own cache up to the window's verdict: the first `kept_rows` rows of the
    /// window stay, each paired with the target hidden row it produced.
    fn commit(&mut self, window: &Window, kept_rows: usize) -> Result<()>;
}

/// The settings the chain reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChainSettings {
    /// Most drafts of a chain.
    pub(crate) ceiling: usize,
    /// Stop the chain from the calibrated chances.
    pub(crate) dynamic: bool,
    /// The product of chances below which the chain stops.
    pub(crate) confidence: f32,
}

pub(crate) struct MtpDrafter<H> {
    head: H,
    settings: ChainSettings,
    calibrator: ConfidenceCalibrator,
}

impl<H: MtpHead> MtpDrafter<H> {
    pub(crate) fn new(head: H, settings: ChainSettings) -> Self {
        Self {
            head,
            settings,
            calibrator: ConfidenceCalibrator::default(),
        }
    }
}

impl<H: MtpHead> Drafter for MtpDrafter<H> {
    fn propose(&mut self, history: &[u32], room: usize) -> Result<Proposal> {
        let Some(&pending) = history.last() else {
            return Ok(Proposal::default());
        };
        let steps = room.min(self.settings.ceiling);
        let mut budget = DraftBudget::new(self.settings.confidence);
        let mut proposal = Proposal::default();
        if steps == 0 {
            return Ok(proposal);
        }
        self.head.begin()?;
        let mut previous = pending;
        for _ in 0..steps {
            let step = self.head.step(previous)?;
            let score = confidence_score(step.probability);
            if self.settings.dynamic && !budget.admit(self.calibrator.estimate(score)) {
                break;
            }
            proposal.tokens.push(step.token);
            proposal.scores.push(score);
            previous = step.token;
        }
        Ok(proposal)
    }

    fn observe(&mut self, window: &Window, accepted: &Accepted) -> Result<()> {
        if self.settings.dynamic {
            self.calibrator
                .observe(window.scores(), accepted.drafts_accepted());
        }
        self.head.commit(window, accepted.rows_kept())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A head that drafts `token = previous + 1` with scripted probabilities and records its calls.
    struct Fake {
        probabilities: Vec<f32>,
        begins: usize,
        steps: Vec<u32>,
        commits: Vec<usize>,
    }

    impl Fake {
        fn new(probabilities: &[f32]) -> Self {
            Self {
                probabilities: probabilities.to_vec(),
                begins: 0,
                steps: Vec::new(),
                commits: Vec::new(),
            }
        }
    }

    impl MtpHead for Fake {
        fn begin(&mut self) -> Result<()> {
            self.begins += 1;
            Ok(())
        }

        fn step(&mut self, previous: u32) -> Result<HeadStep> {
            let index = self.steps.len() % self.probabilities.len();
            self.steps.push(previous);
            Ok(HeadStep {
                token: previous + 1,
                probability: self.probabilities[index],
            })
        }

        fn commit(&mut self, _window: &Window, kept_rows: usize) -> Result<()> {
            self.commits.push(kept_rows);
            Ok(())
        }
    }

    fn settings(dynamic: bool) -> ChainSettings {
        ChainSettings {
            ceiling: 4,
            dynamic,
            confidence: 0.4,
        }
    }

    #[test]
    fn the_chain_feeds_each_draft_to_the_next_step() {
        let mut drafter = MtpDrafter::new(Fake::new(&[0.9]), settings(false));
        let proposal = drafter.propose(&[5, 6, 7], 8).expect("proposal");
        assert_eq!(proposal.tokens, [8, 9, 10, 11]);
        assert_eq!(proposal.scores.len(), 4);
        assert_eq!(drafter.head.steps, [7, 8, 9, 10]);
        assert_eq!(drafter.head.begins, 1);
    }

    #[test]
    fn the_chain_is_held_to_the_room_and_the_ceiling() {
        let mut drafter = MtpDrafter::new(Fake::new(&[0.9]), settings(false));
        assert_eq!(drafter.propose(&[1], 2).expect("proposal").tokens.len(), 2);
        assert_eq!(
            drafter.propose(&[1], 100).expect("proposal").tokens.len(),
            4
        );
        assert!(
            drafter
                .propose(&[1], 0)
                .expect("proposal")
                .tokens
                .is_empty()
        );
        assert!(drafter.propose(&[], 4).expect("proposal").tokens.is_empty());
    }

    #[test]
    fn a_verdict_reaches_the_head_and_the_calibrator() {
        let mut drafter = MtpDrafter::new(Fake::new(&[0.9]), settings(true));
        let proposal = drafter.propose(&[1], 4).expect("proposal");
        let window = Window::new(10, proposal.tokens, proposal.scores);
        let accepted = Accepted {
            tokens: vec![2, 3, 9],
            stopped: false,
        };
        drafter.observe(&window, &accepted).expect("observe");
        assert_eq!(drafter.head.commits, [3]);
    }

    /// Drives windows where a draft is accepted only when its probability is high, so the
    /// calibrator learns the split and the chain stops at the first low-probability draft.
    #[test]
    fn dynamic_length_learns_to_stop_before_a_draft_the_target_rejects() {
        // Steps alternate: confident, confident, doubtful, doubtful.
        let mut drafter = MtpDrafter::new(Fake::new(&[0.95, 0.95, 0.05, 0.05]), settings(true));
        let mut lengths = Vec::new();
        for round in 0..120 {
            drafter.head.steps.clear();
            let proposal = drafter.propose(&[1], 4).expect("proposal");
            lengths.push(proposal.tokens.len());
            // The target accepts the two confident drafts and rejects the first doubtful one.
            let accepted_drafts = proposal.tokens.len().min(2);
            let window = Window::new(round, proposal.tokens, proposal.scores);
            let accepted = Accepted {
                tokens: vec![0; accepted_drafts + 1],
                stopped: false,
            };
            drafter.observe(&window, &accepted).expect("observe");
        }
        assert_eq!(lengths[0], 4, "no opinion before the burn-in");
        assert_eq!(
            *lengths.last().expect("rounds"),
            2,
            "stops before the doubtful drafts"
        );
    }

    #[test]
    fn without_dynamic_length_the_calibrator_stays_silent() {
        let mut drafter = MtpDrafter::new(Fake::new(&[0.05]), settings(false));
        for round in 0..200 {
            let proposal = drafter.propose(&[1], 4).expect("proposal");
            assert_eq!(proposal.tokens.len(), 4);
            let window = Window::new(round, proposal.tokens, proposal.scores);
            let accepted = Accepted {
                tokens: vec![0],
                stopped: false,
            };
            drafter.observe(&window, &accepted).expect("observe");
        }
    }
}
