//! Fakes of the two collaborators of the driver, for host tests of the sequence.
//!
//! The device keeps a stream of four floats; layer `l` at position `p` adds `l + 1 + p` to every
//! value; the logits row has [`LOGITS`] values and value `i` is `stream[0] * i`, so the greedy pick
//! of a positive stream is the last id of the row, which lies above the sampler vocabulary.

use std::ops::Range;

use super::{DriverShape, StartVectors, TokenDevice, TokenFeeds};
use crate::{
    EngError,
    family::flash_next::cuda::feeds::{StartShape, StartSink},
};

pub(in crate::family::flash_next::cuda) const LAYERS: usize = 4;
pub(in crate::family::flash_next::cuda) const PLE_LAYER: usize = 2;
pub(in crate::family::flash_next::cuda) const POSITIONS: usize = 6;
pub(in crate::family::flash_next::cuda) const STREAM: usize = 4;
pub(in crate::family::flash_next::cuda) const PLE_EMBED: usize = 3;
/// Width of the logits row, the head.
pub(in crate::family::flash_next::cuda) const LOGITS: usize = 8;
/// Ids the sampler may draw: the tokenizer's vocabulary.
pub(in crate::family::flash_next::cuda) const SAMPLER_VOCAB: usize = 6;

pub(in crate::family::flash_next::cuda) fn shape() -> DriverShape {
    DriverShape {
        layers: LAYERS,
        ple_layer: PLE_LAYER,
        positions: POSITIONS,
        start: StartShape {
            sampler_vocab: SAMPLER_VOCAB,
            stream: STREAM,
            ple_embed: PLE_EMBED,
        },
    }
}

#[derive(Default)]
pub(in crate::family::flash_next::cuda) struct FakeDevice {
    /// Every call, in order.
    pub(in crate::family::flash_next::cuda) log: Vec<String>,
    pub(in crate::family::flash_next::cuda) stream: Vec<f32>,
    /// The call that fails (its log name), once.
    pub(in crate::family::flash_next::cuda) fail: Option<&'static str>,
    pub(in crate::family::flash_next::cuda) resets: usize,
}

impl FakeDevice {
    fn step(&mut self, name: &'static str, detail: String) -> Result<(), EngError> {
        self.log
            .push(format!("{name} {detail}").trim_end().to_string());
        if self.fail == Some(name) {
            self.fail = None;
            return Err(EngError::Backend {
                operation: "fake_device",
                message: format!("{name} failed"),
            });
        }
        Ok(())
    }
}

impl StartSink for FakeDevice {
    fn write_step_params(&mut self, token: i32, pos: i32) -> Result<(), EngError> {
        self.step("params", format!("{token} {pos}"))
    }

    fn write_stream(&mut self, stream: &[f32]) -> Result<(), EngError> {
        self.stream = stream.to_vec();
        self.step("stream", String::new())
    }

    fn write_ple_embedding(&mut self, _embedding: &[f32]) -> Result<(), EngError> {
        self.step("ple_embedding", String::new())
    }
}

impl TokenDevice for FakeDevice {
    fn begin_token(&mut self) -> Result<(), EngError> {
        self.step("begin", String::new())
    }

    fn layers(&mut self, range: Range<usize>, pos: usize) -> Result<(), EngError> {
        for layer in range.clone() {
            for value in &mut self.stream {
                *value += (layer + 1 + pos) as f32;
            }
        }
        self.step("layers", format!("{}..{} @{pos}", range.start, range.end))
    }

    fn read_stream(&mut self) -> Result<Vec<f32>, EngError> {
        self.step("read_stream", String::new())?;
        Ok(self.stream.clone())
    }

    fn write_stream_back(&mut self, stream: &[f32]) -> Result<(), EngError> {
        self.stream = stream.to_vec();
        self.step("write_back", String::new())
    }

    fn head(&mut self) -> Result<(), EngError> {
        self.step("head", String::new())
    }

    fn read_logits(&mut self) -> Result<Vec<f32>, EngError> {
        self.step("read_logits", String::new())?;
        Ok((0..LOGITS).map(|i| self.stream[0] * i as f32).collect())
    }

    fn check_health(&mut self) -> Result<(), EngError> {
        self.step("health", String::new())
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.resets += 1;
        self.stream.clear();
        self.step("reset", String::new())
    }
}

#[derive(Default)]
pub(in crate::family::flash_next::cuda) struct FakeFeeds {
    /// The updates asked for: the token and the stream it saw.
    pub(in crate::family::flash_next::cuda) updates: Vec<(u32, Vec<f32>)>,
    pub(in crate::family::flash_next::cuda) resets: usize,
    /// The token whose `begin` fails.
    pub(in crate::family::flash_next::cuda) fail_begin: Option<u32>,
    /// Fails the PLE update.
    pub(in crate::family::flash_next::cuda) fail_update: bool,
    /// The width of the update (the stream's, unless a test breaks it).
    pub(in crate::family::flash_next::cuda) update_width: Option<usize>,
}

impl TokenFeeds for FakeFeeds {
    fn begin(&mut self, token: u32) -> Result<StartVectors, EngError> {
        if self.fail_begin == Some(token) {
            return Err(EngError::Checkpoint("the pack read failed".into()));
        }
        Ok(StartVectors {
            stream: vec![token as f32; STREAM],
            ple_embedding: vec![token as f32; PLE_EMBED],
        })
    }

    fn ple_update(&mut self, token: u32, stream: &[f32]) -> Result<Vec<f32>, EngError> {
        if self.fail_update {
            return Err(EngError::Checkpoint("the PLE step failed".into()));
        }
        self.updates.push((token, stream.to_vec()));
        Ok(vec![0.5; self.update_width.unwrap_or(stream.len())])
    }

    fn reset(&mut self) {
        self.resets += 1;
    }
}
