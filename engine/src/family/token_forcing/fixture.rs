//! A deterministic toy language model for the host tests of token forcing.
//!
//! Its logits mix three terms, each from a fixed hash: a sliding window over the last `WINDOW`
//! tokens, a long-range read of the token `LAG` positions back from a ring of `RING` slots, and a
//! bucket of the context position. A rounding seed adds a perturbation of at most `ROUNDING`,
//! which stands in for a legal change of reduction order: it can flip only near ties. A `Bug`
//! stands in for a defect of the kind the gate must catch.

use super::score::{ForcedModel, greedy_generate};
use crate::EngError;
use crate::family::session::TokenStepper;

pub(super) const VOCAB: usize = 48;
const WINDOW: usize = 4;
pub(super) const LAG: usize = 300;
pub(super) const RING: usize = 1024;
const BUCKET: usize = 64;
/// The largest legal rounding change, far below the 1/8 grid of delta.
const ROUNDING: f32 = 2e-3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Bug {
    None,
    /// From context position `from`, every `stride`-th id gets `bias`: SpecPilot's injected
    /// fault (bias 2 on every 7th id after 2,048 tokens).
    Bias {
        from: usize,
        stride: u32,
        bias: f32,
    },
    /// Once the ring has wrapped, the long-range read takes the slot after the right one.
    RingWrap,
    /// From context position `from`, the sliding window drops its oldest token.
    ShortWindow {
        from: usize,
    },
    /// From context position `from`, the position term reads the next bucket.
    PositionShift {
        from: usize,
    },
    /// A save and restore at context position `at` loses the `span` ring slots before it.
    RestoreLosesBlock {
        at: usize,
        span: usize,
    },
    /// Every `period`-th context position, one hashed id gets `boost`.
    RareSpike {
        period: usize,
        boost: f32,
    },
    /// Whenever the best id is a multiple of `modulus`, its logit and the runner-up's trade
    /// places, so the model always emits the wrong one of the two.
    PersistentSwap {
        modulus: u32,
    },
}

#[derive(Clone, Debug)]
pub(super) struct Toy {
    seed: u64,
    bug: Bug,
    history: Vec<u32>,
    ring: Vec<u32>,
}

impl Toy {
    /// The exact model: no rounding change, no bug.
    pub(super) fn exact() -> Self {
        Self::new(0, Bug::None)
    }

    /// A legal execution with rounding seed `seed` (non-zero).
    pub(super) fn rounded(seed: u64) -> Self {
        Self::new(seed, Bug::None)
    }

    pub(super) fn new(seed: u64, bug: Bug) -> Self {
        Self {
            seed,
            bug,
            history: Vec::new(),
            ring: vec![0; RING],
        }
    }

    fn row(&self) -> Vec<f32> {
        let n = self.history.len();
        let window_key = self.window_key(n);
        let far_key = self.far_token(n).map(|token| mix(0xfa, u64::from(token)));
        let bucket_key = mix(0xb0, self.bucket(n) as u64);
        let mut row: Vec<f32> = (0..VOCAB as u64)
            .map(|id| {
                let mut logit = 3.0 * unit(mix(window_key, id)) + 0.5 * unit(mix(bucket_key, id));
                if let Some(key) = far_key {
                    logit += 1.5 * unit(mix(key, id));
                }
                if self.seed != 0 {
                    logit += ROUNDING * unit(mix(mix(self.seed, n as u64), id));
                }
                logit
            })
            .collect();
        self.apply_logit_bugs(n, &mut row);
        row
    }

    fn window_key(&self, n: usize) -> u64 {
        let short = matches!(self.bug, Bug::ShortWindow { from } if n >= from);
        let width = if short { WINDOW - 1 } else { WINDOW };
        let window = &self.history[n.saturating_sub(width)..];
        window
            .iter()
            .fold(mix(0x51, window.len() as u64), |key, &token| {
                mix(key, u64::from(token))
            })
    }

    fn far_token(&self, n: usize) -> Option<u32> {
        let back = n.checked_sub(LAG)?;
        let wrapped = self.bug == Bug::RingWrap && n >= RING;
        let slot = if wrapped { back + 1 } else { back };
        Some(self.ring[slot % RING])
    }

    fn bucket(&self, n: usize) -> usize {
        let shifted = matches!(self.bug, Bug::PositionShift { from } if n >= from);
        n / BUCKET + usize::from(shifted)
    }

    fn apply_logit_bugs(&self, n: usize, row: &mut [f32]) {
        match self.bug {
            Bug::Bias { from, stride, bias } if n >= from => {
                for (id, logit) in row.iter_mut().enumerate() {
                    if (id as u32).is_multiple_of(stride) {
                        *logit += bias;
                    }
                }
            }
            Bug::RareSpike { period, boost } if n % period == period - 1 => {
                row[(mix(0x5b, n as u64) % VOCAB as u64) as usize] += boost;
            }
            Bug::PersistentSwap { modulus } => {
                let best = argmax(row);
                if (best as u32).is_multiple_of(modulus) {
                    let runner_up = (0..VOCAB)
                        .filter(|&other| other != best)
                        .reduce(|found, other| {
                            if row[other] > row[found] {
                                other
                            } else {
                                found
                            }
                        })
                        .expect("the toy vocabulary has more than one id");
                    row.swap(best, runner_up);
                }
            }
            _ => {}
        }
    }
}

impl TokenStepper for Toy {
    const OPERATION: &'static str = "token_forcing_toy";

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        let position = self.history.len();
        self.ring[position % RING] = token;
        self.history.push(token);
        if let Bug::RestoreLosesBlock { at, span } = self.bug
            && self.history.len() == at
        {
            for lost in at.saturating_sub(span)..at {
                self.ring[lost % RING] = 0;
            }
        }
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        self.feed(token)?;
        Ok(self.row())
    }

    fn fed(&self) -> usize {
        self.history.len()
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.history.clear();
        self.ring.fill(0);
        Ok(())
    }
}

/// Greedy generation of `tokens` ids through the model's prompt path and then its step path.
pub(super) fn generate(model: &mut dyn ForcedModel, prompt: &[u32], tokens: usize) -> Vec<u32> {
    greedy_generate(model, prompt, tokens, VOCAB).expect("the toy model generates")
}

/// A prompt of `len` hashed ids, distinct per `seed`.
pub(super) fn prompt(seed: u64, len: usize) -> Vec<u32> {
    (0..len as u64)
        .map(|index| (mix(mix(0x9e, seed), index) % VOCAB as u64) as u32)
        .collect()
}

/// The greedy choice with ties to the lower id, as in the sampler.
fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (index, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = index;
        }
    }
    best
}

/// splitmix64 of the two words.
fn mix(key: u64, value: u64) -> u64 {
    let mut z = key
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(value)
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A hash as a uniform value in [-1, 1).
fn unit(hash: u64) -> f32 {
    ((hash >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0) as f32
}
