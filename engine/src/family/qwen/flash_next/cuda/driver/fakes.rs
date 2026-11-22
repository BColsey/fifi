//! Fakes of the two collaborators of the driver, for host tests of the sequence.
//!
//! The device keeps a stream of four floats; layer `l` at position `p` adds `l + 1 + p` to every
//! value; the logits row has [`LOGITS`] values and value `i` is `stream[0] * i`, so the greedy pick
//! of a positive stream is the last id of the row, which lies above the sampler vocabulary.

use std::ops::Range;

use super::super::driver::StateSlot;
use super::{DriverShape, StartVectors, TokenDevice, TokenFeeds, chunk::ChunkDevice};
use crate::family::qwen::flash_next::cuda::bank_source::DecodeMisses;
use crate::family::qwen::flash_next::cuda::ple_feed::PleKeyValue;
use crate::{
    EngError,
    family::qwen::flash_next::cuda::feeds::{StartShape, StartSink},
};

pub(in super::super) const LAYERS: usize = 4;
pub(in super::super) const PLE_LAYER: usize = 2;
pub(in super::super) const POSITIONS: usize = 6;
pub(in super::super) const STREAM: usize = 4;
pub(in super::super) const PLE_EMBED: usize = 3;
/// Width of the logits row, the head.
pub(in super::super) const LOGITS: usize = 8;
/// Ids the sampler may draw: the tokenizer's vocabulary.
pub(in super::super) const SAMPLER_VOCAB: usize = 6;

pub(in super::super) fn shape() -> DriverShape {
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
pub(in super::super) struct FakeDevice {
    /// Every call, in order.
    pub(in super::super) log: Vec<String>,
    pub(in super::super) stream: Vec<f32>,
    /// The fixed-state image a resume reads and installs (one blob; the fake has no slots).
    pub(in super::super) fixed: Vec<u8>,
    /// The call that fails (its log name), once.
    pub(in super::super) fail: Option<&'static str>,
    pub(in super::super) resets: usize,
    /// The rung's decode miss counter. When set, `layers` observes the
    /// card device's pattern — one count per layer, misses equal to the
    /// layer index — so driver-level tests see the turn line.
    pub(in super::super) decode_misses: Option<DecodeMisses>,
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
            if let Some(misses) = &self.decode_misses {
                misses.observe(pos as u64, layer, layer);
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
        // The card device ends an open decode step on reset; so does the fake.
        if let Some(misses) = &self.decode_misses {
            misses.discard_open();
        }
        self.step("reset", String::new())
    }

    fn rung_miss_line(&self) -> Option<String> {
        self.decode_misses.as_ref()?.line(4)
    }

    fn discard_decode_step(&self) {
        if let Some(misses) = &self.decode_misses {
            misses.discard_open();
        }
    }
}

impl super::StateIo for FakeDevice {
    fn read_fixed_state(&self) -> Result<Vec<(StateSlot, Vec<u8>)>, EngError> {
        Ok(vec![(StateSlot::PleRing, self.fixed.clone())])
    }

    fn fixed_state_bytes(&self) -> u64 {
        self.fixed.len() as u64
    }

    fn write_fixed_state(&mut self, parts: &[(StateSlot, Vec<u8>)]) -> Result<(), EngError> {
        let Some((_, bytes)) = parts.first() else {
            return Ok(());
        };
        self.fixed = bytes.clone();
        self.log.push("state_write".to_string());
        Ok(())
    }
}

/// The chunk fake: `m` rows of the stream; layer `l` at position `p` adds `l + 1 + p + row` to
/// every value of row `row`, so a row carries its own index and the PLE roundtrip is checkable
/// per row. The logits row follows the LAST row, as the head does.
#[derive(Default)]
pub(in super::super) struct FakeChunkDevice {
    /// Every call, in order.
    pub(in super::super) log: Vec<String>,
    pub(in super::super) stream: Vec<f32>,
    /// The device PLE pairs `ple_project_chunk` returns, one per row.
    pub(in super::super) ple_pairs: Option<Vec<PleKeyValue>>,
    /// The call that fails (its log name), once.
    pub(in super::super) fail: Option<&'static str>,
    pub(in super::super) resets: usize,
    /// The last state the resume seam wrote back, as read.
    pub(in super::super) state: Vec<(StateSlot, Vec<u8>)>,
    pub(in super::super) track_windows: bool,
    /// The rung's decode miss counter, as on the token fake.
    pub(in super::super) decode_misses: Option<DecodeMisses>,
    /// True inside the prefill window: the production prefill path never
    /// observes (SplitMoe::run_prefill), so the fake does not either.
    pub(in super::super) in_prefill: bool,
}

impl FakeChunkDevice {
    fn step(&mut self, name: &'static str, detail: String) -> Result<(), EngError> {
        self.log
            .push(format!("{name} {detail}").trim_end().to_string());
        if self.fail == Some(name) {
            self.fail = None;
            return Err(EngError::Backend {
                operation: "fake_chunk_device",
                message: format!("{name} failed"),
            });
        }
        Ok(())
    }
}

impl super::StateIo for FakeChunkDevice {
    fn read_fixed_state(&self) -> Result<Vec<(StateSlot, Vec<u8>)>, EngError> {
        // read takes &self, so no log step here — the log is only for the mutable calls.
        Ok(self.state.clone())
    }

    fn fixed_state_bytes(&self) -> u64 {
        self.state.iter().map(|(_, bytes)| bytes.len() as u64).sum()
    }

    fn write_fixed_state(&mut self, parts: &[(StateSlot, Vec<u8>)]) -> Result<(), EngError> {
        self.step("write_fixed_state", String::new())?;
        self.state = parts.to_vec();
        Ok(())
    }
}

impl ChunkDevice for FakeChunkDevice {
    fn write_chunk_params(&mut self, token0: i32, pos: i32, m: i32) -> Result<(), EngError> {
        self.step("chunk_params", format!("{token0} {pos} {m}"))
    }

    fn write_chunk_stream(&mut self, rows: &[f32]) -> Result<(), EngError> {
        self.stream = rows.to_vec();
        self.step("chunk_stream", String::new())
    }

    fn write_chunk_ple(&mut self, _rows: &[f32]) -> Result<(), EngError> {
        self.step("chunk_ple", String::new())
    }

    fn ple_project_chunk(&mut self) -> Result<Vec<PleKeyValue>, EngError> {
        self.step("chunk_ple_project", String::new())?;
        self.ple_pairs
            .clone()
            .ok_or_else(|| EngError::BadRequest("the fake has no PLE pairs".to_string()))
    }

    fn begin_chunk(&mut self) -> Result<(), EngError> {
        self.step("begin", String::new())
    }

    fn begin_prefill(&mut self) -> Result<(), EngError> {
        self.in_prefill = true;
        if self.track_windows {
            self.step("begin_prefill", String::new())?;
        }
        Ok(())
    }

    fn begin_prefill_logits(&mut self) -> Result<(), EngError> {
        if self.track_windows {
            self.step("begin_prefill_logits", String::new())?;
        }
        Ok(())
    }

    fn end_prefill(&mut self) -> Result<(), EngError> {
        self.in_prefill = false;
        if self.track_windows {
            self.step("end_prefill", String::new())?;
        }
        Ok(())
    }

    fn layers(&mut self, range: Range<usize>, pos: usize, m: usize) -> Result<(), EngError> {
        let width = self.stream.len() / m.max(1);
        for layer in range.clone() {
            for (row, value) in self.stream.chunks_mut(width).enumerate() {
                let add = (layer + 1 + pos + row) as f32;
                for cell in value.iter_mut() {
                    *cell += add;
                }
            }
            if let (true, Some(misses)) = (!self.in_prefill, &self.decode_misses) {
                misses.observe(pos as u64, layer, layer);
            }
        }
        self.step(
            "layers",
            format!("{}..{} @{pos} m{m}", range.start, range.end),
        )
    }

    fn read_stream(&mut self) -> Result<Vec<f32>, EngError> {
        self.step("read_stream", String::new())?;
        Ok(self.stream.clone())
    }

    fn write_stream_back(&mut self, rows: &[f32]) -> Result<(), EngError> {
        self.stream = rows.to_vec();
        self.step("write_back", String::new())
    }

    fn head(&mut self) -> Result<(), EngError> {
        self.step("head", String::new())
    }

    fn read_logits(&mut self) -> Result<Vec<f32>, EngError> {
        self.step("read_logits", String::new())?;
        let width = self.stream.len().max(1);
        let last = self.stream[width - STREAM..].to_vec();
        Ok((0..LOGITS).map(|i| last[0] * i as f32).collect())
    }

    fn check_health(&mut self) -> Result<(), EngError> {
        self.step("health", String::new())
    }

    /// Logs like every other call, so the boundary-order test can see the drain between the
    /// phases; the `fail` mechanism covers a failing drain by name.
    fn drain(&mut self) -> Result<(), EngError> {
        self.step("drain", String::new())
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.resets += 1;
        self.stream.clear();
        // The card device ends an open decode step on reset; so does the fake.
        if let Some(misses) = &self.decode_misses {
            misses.discard_open();
        }
        self.step("reset", String::new())
    }

    fn rung_miss_line(&self) -> Option<String> {
        self.decode_misses.as_ref()?.line(4)
    }

    fn discard_decode_step(&self) {
        if let Some(misses) = &self.decode_misses {
            misses.discard_open();
        }
    }
}

#[derive(Default)]
pub(in super::super) struct FakeFeeds {
    /// The updates asked for: the token and the stream it saw.
    pub(in super::super) updates: Vec<(u32, Vec<f32>)>,
    /// The first key of every device pair set (the device route hands them over in order).
    pub(in super::super) pairs: Vec<f32>,
    pub(in super::super) resets: usize,
    /// PLE updates seen, for `fail_update_after`.
    pub(in super::super) ple_calls: usize,
    /// The token whose `begin` fails.
    pub(in super::super) fail_begin: Option<u32>,
    /// Fails the PLE update.
    pub(in super::super) fail_update: bool,
    /// Fails the PLE update after this many successful calls (a mid-turn
    /// death that leaves earlier steps complete).
    pub(in super::super) fail_update_after: Option<usize>,
    /// The width of the update (the stream's, unless a test breaks it).
    pub(in super::super) update_width: Option<usize>,
    pub(in super::super) start_width: Option<usize>,
}

/// Exercise a real host feed through the driver with only the device replaced by a fake.
pub(in super::super::super) fn exercise_host_feed<F: TokenFeeds>(
    feeds: F,
    model: &crate::family::qwen::flash_next::step::ModelShape,
) {
    use crate::family::session::TokenStepper;
    let shape = DriverShape {
        layers: model.ple_layer + 1,
        ple_layer: model.ple_layer,
        positions: 8,
        start: StartShape {
            sampler_vocab: model.vocab,
            stream: model.hyper.width(),
            ple_embed: model.ple.embed(),
        },
    };
    let mut driver = super::TokenDriver::new(FakeDevice::default(), feeds, shape).unwrap();
    driver.feed(1).unwrap();
    assert_eq!(driver.fed(), 1);
    assert_eq!(driver.feeds.ple_state().unwrap().0, &[1]);
    driver.device.fail = Some("write_back");
    assert!(driver.feed(2).is_err());
    assert_eq!(driver.fed(), 1);
    assert!(
        driver.feed(2).is_err(),
        "failed device write poisons the driver"
    );
    driver.reset().unwrap();
    assert_eq!(driver.fed(), 0);
    assert!(driver.feeds.ple_state().unwrap().0.is_empty());
    driver.feed(2).unwrap();
    assert_eq!(driver.fed(), 1);
    assert_eq!(driver.feeds.ple_state().unwrap().0, &[2]);
}

impl TokenFeeds for FakeFeeds {
    fn begin(&mut self, token: u32) -> Result<StartVectors, EngError> {
        if self.fail_begin == Some(token) {
            return Err(EngError::Checkpoint("the pack read failed".into()));
        }
        Ok(StartVectors {
            stream: vec![token as f32; self.start_width.unwrap_or(STREAM)],
            ple_embedding: vec![token as f32; PLE_EMBED],
        })
    }

    fn ple_update(&mut self, token: u32, stream: &[f32]) -> Result<Vec<f32>, EngError> {
        let calls = self.ple_calls;
        self.ple_calls += 1;
        if self.fail_update || self.fail_update_after.is_some_and(|n| calls >= n) {
            return Err(EngError::Checkpoint("the PLE step failed".into()));
        }
        self.updates.push((token, stream.to_vec()));
        Ok(vec![0.5; self.update_width.unwrap_or(stream.len())])
    }

    fn set_ple_pair(&mut self, pair: PleKeyValue) -> Result<(), EngError> {
        self.pairs
            .push(pair.keys.first().copied().unwrap_or(f32::NAN));
        Ok(())
    }

    fn reset(&mut self) {
        self.resets += 1;
    }
}
