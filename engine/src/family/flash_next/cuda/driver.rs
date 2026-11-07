//! The Flash-Next token driver: one token end to end, behind the shared `TokenStepper` seam.
//!
//! WHAT: [`TokenDriver`] runs the sequence of the host reference (`family/flash_next/step.rs`) over
//! two collaborators and owns the position, the poison flag and the fed count:
//!
//! 1. [`TokenFeeds::begin`]: the embedding row repeated over the branches (the initial stream) and
//!    the PLE embedding of the token, both host work (`feeds`, `embedding`, `ple_feed`).
//! 2. [`write_start`] to the device: the step parameters (token and position) and the stream.
//! 3. The layers before the PLE layer.
//! 4. The PLE block: the stream comes back to the host, [`TokenFeeds::ple_update`] returns the
//!    update, and the sum goes back to the device (the host reference adds it to the stream before
//!    the PLE layer's attention read).
//! 5. The remaining layers.
//! 6. With logits wanted: the final mixer, `lm_head` and the read-back of the logits row.
//! 7. The health check (status word and the sticky arithmetic flag) and the position advance.
//!
//! The device half is [`TokenDevice`] (the decode step and its MoE block on a card), the host half is
//! [`TokenFeeds`] (the pack's embedding and the PLE feed). The traits exist so that the sequence,
//! the poison rule and the executor on top of it run in host tests against fakes; the card
//! implementations are `device.rs` and `feeds_pack.rs`.
//!
//! The fed rule of `TokenStepper`: `fed` advances when, and only when, the model state consumed the
//! token. A failure anywhere after the first write to the card poisons the driver: every later
//! feed is refused until `reset`, as the host reference does after a failed step.
//!
//! Sampling is the shared one: [`TokenStepper::next_token`] reads the logits row and
//! `sample_logits` draws from the first `sampler_vocab` values, so the padded ids of the head
//! (above the tokenizer's vocabulary) are never drawn and a greedy request is argmax.

use std::ops::Range;

use super::feeds::{StartShape, StartSink, write_start};
use crate::{EngError, family::session::TokenStepper};

type Result<T> = std::result::Result<T, EngError>;

/// The sizes the driver checks and splits the layers by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DriverShape {
    /// Decoder layers the device runs.
    pub(super) layers: usize,
    /// The layer whose attention read follows the PLE update.
    pub(super) ple_layer: usize,
    /// Positions the K/V caches hold: the context limit of the driver.
    pub(super) positions: usize,
    /// The sizes of one token start.
    pub(super) start: StartShape,
}

/// The start vectors of a token.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StartVectors {
    /// `branches * hidden`: the embedding row once per branch.
    pub(super) stream: Vec<f32>,
    /// `ple_embed`: the PLE embedding of the token (its n-gram rows).
    pub(super) ple_embedding: Vec<f32>,
}

/// The host side of a token: the embedding table and the PLE feed.
pub(super) trait TokenFeeds {
    /// The start vectors of `token`. Reads the pack and the n-gram table; changes no state.
    fn begin(&mut self, token: u32) -> Result<StartVectors>;
    /// The PLE update for `token` given the stream at the PLE layer's input (`branches * hidden`).
    /// Advances the PLE history and ring: called once per fed token, after [`Self::begin`].
    fn ple_update(&mut self, token: u32, stream: &[f32]) -> Result<Vec<f32>>;
    /// Forgets the history and the ring: a new sequence.
    fn reset(&mut self);
}

/// The device side of a token. It is the sink of the token start.
pub(super) trait TokenDevice: StartSink {
    /// Clears the per-token flags. Called once, before the start is written.
    fn begin_token(&mut self) -> Result<()>;
    /// Queues layers `range` for the token at `pos`.
    fn layers(&mut self, range: Range<usize>, pos: usize) -> Result<()>;
    /// The stream (`branches * hidden`), read back behind the queued work.
    fn read_stream(&mut self) -> Result<Vec<f32>>;
    /// Replaces the stream.
    fn write_stream_back(&mut self, stream: &[f32]) -> Result<()>;
    /// Queues the final mixer and the head.
    fn head(&mut self) -> Result<()>;
    /// The logits row of the last head, behind the queued work.
    fn read_logits(&mut self) -> Result<Vec<f32>>;
    /// Reads the status word and the arithmetic flag; refuses a token that set either.
    fn check_health(&mut self) -> Result<()>;
    /// Zeroes the persistent state: a new sequence.
    fn reset(&mut self) -> Result<()>;
}

/// The token driver: a [`TokenStepper`] over a device and its feeds.
pub(super) struct TokenDriver<D: TokenDevice, F: TokenFeeds> {
    device: D,
    feeds: F,
    shape: DriverShape,
    /// Tokens the state has consumed: the next position.
    pos: usize,
    poisoned: bool,
}

impl<D: TokenDevice, F: TokenFeeds> TokenDriver<D, F> {
    pub(super) fn new(device: D, feeds: F, shape: DriverShape) -> Result<Self> {
        if shape.layers == 0 || shape.ple_layer >= shape.layers || shape.positions == 0 {
            return Err(EngError::BadRequest(format!(
                "Flash-Next driver: {} layers with the PLE layer at {} and {} positions",
                shape.layers, shape.ple_layer, shape.positions
            )));
        }
        Ok(Self {
            device,
            feeds,
            shape,
            pos: 0,
            poisoned: false,
        })
    }

    /// Positions the caches hold.
    pub(super) fn positions(&self) -> usize {
        self.shape.positions
    }

    #[cfg(test)]
    pub(super) fn device(&self) -> &D {
        &self.device
    }

    #[cfg(test)]
    pub(super) fn feeds(&self) -> &F {
        &self.feeds
    }

    /// One token. `want_logits` runs the head and returns the row.
    fn run_token(&mut self, token: u32, want_logits: bool) -> Result<Option<Vec<f32>>> {
        if self.poisoned {
            return Err(EngError::BadRequest(
                "Flash-Next driver: the state needs a reset after a failed step".into(),
            ));
        }
        if self.pos >= self.shape.positions {
            return Err(EngError::ContextLimit(format!(
                "Flash-Next driver: the caches hold {} positions",
                self.shape.positions
            )));
        }
        // A refused id changes nothing; `write_start` repeats the rule, but inside the queued part,
        // where a failure poisons the driver.
        if usize::try_from(token).is_ok_and(|id| id >= self.shape.start.sampler_vocab) {
            return Err(EngError::BadRequest(format!(
                "token id {token} is outside the sampler vocabulary 0..{}",
                self.shape.start.sampler_vocab
            )));
        }
        // Host reads only: a failure here leaves the state as it was.
        let start = self.feeds.begin(token)?;
        let outcome = self.queue(token, &start, want_logits);
        match outcome {
            Ok(logits) => {
                self.pos += 1;
                Ok(logits)
            }
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }

    /// Everything after the first write to the card.
    fn queue(
        &mut self,
        token: u32,
        start: &StartVectors,
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let pos = self.pos;
        let shape = self.shape;
        write_start(
            &mut self.device,
            &shape.start,
            token,
            pos,
            &start.stream,
            &start.ple_embedding,
        )?;
        self.device.begin_token()?;
        self.device.layers(0..shape.ple_layer, pos)?;
        let mut stream = self.device.read_stream()?;
        let update = self.feeds.ple_update(token, &stream)?;
        if update.len() != stream.len() {
            return Err(EngError::BadRequest(format!(
                "Flash-Next driver: the PLE update has {} values for a stream of {}",
                update.len(),
                stream.len()
            )));
        }
        for (value, delta) in stream.iter_mut().zip(&update) {
            *value += delta;
        }
        self.device.write_stream_back(&stream)?;
        self.device.layers(shape.ple_layer..shape.layers, pos)?;
        let logits = if want_logits {
            self.device.head()?;
            Some(self.device.read_logits()?)
        } else {
            None
        };
        self.device.check_health()?;
        Ok(logits)
    }
}

impl<D: TokenDevice, F: TokenFeeds> TokenStepper for TokenDriver<D, F> {
    const OPERATION: &'static str = "flash_next_cuda";

    fn feed(&mut self, token: u32) -> Result<()> {
        self.run_token(token, false).map(|_| ())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.run_token(token, true)?
            .ok_or_else(|| EngError::Backend {
                operation: Self::OPERATION,
                message: "the logits step returned no logits".to_string(),
            })
    }

    fn fed(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) -> Result<()> {
        self.device.reset()?;
        self.feeds.reset();
        self.pos = 0;
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(test)]
pub(super) mod fakes;
#[cfg(test)]
mod tests;
