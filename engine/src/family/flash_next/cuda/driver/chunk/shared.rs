//! Flash-Next's chunk-family binding. The driver keeps token state; this wrapper only borrows it.

use super::{ChunkDevice, ChunkDriver, DriverShape, Result, TokenFeeds, advance_boundary, prof};
use crate::{
    EngError,
    family::{
        prefill::{
            chunk_family::ChunkFamily,
            steps::{PrefillStep, check_chunk_start, drive_steps},
        },
        project::exl3_jobs_plan::MAX_EXL3_ROWS,
    },
};
use std::{sync::atomic::Ordering, time::Instant};

pub(super) struct ChunkClock {
    at: Instant,
    profile: bool,
    drain_on: bool,
}

impl ChunkClock {
    pub(super) fn new() -> Self {
        let profile = prof::on();
        Self {
            at: Instant::now(),
            profile,
            drain_on: profile && prof::drain_tick(),
        }
    }

    pub(super) fn tick<D: ChunkDevice>(&mut self, device: &mut D, slot: usize) -> Result<()> {
        advance_boundary(
            device,
            slot,
            &mut self.at,
            self.profile,
            self.drain_on,
            prof::add,
        )
    }
}

struct FlashChunkCtx<'a> {
    tokens: &'a [u32],
    starts: &'a [super::super::StartVectors],
    ple_rows: Vec<f32>,
    clock: ChunkClock,
}

struct FlashChunkFamily<'a, D: ChunkDevice, F: TokenFeeds> {
    driver: &'a mut ChunkDriver<D, F>,
}

impl<'a, D: ChunkDevice, F: TokenFeeds> ChunkFamily for FlashChunkFamily<'a, D, F> {
    type Ctx = FlashChunkCtx<'a>;

    fn begin_window(&mut self, ctx: &mut Self::Ctx) -> Result<()> {
        self.driver.device.begin_prefill()?;
        self.driver.shared_window = true;
        ctx.clock.at = Instant::now();
        Ok(())
    }

    fn clear_chunk_status(&mut self, ctx: &mut Self::Ctx) -> Result<()> {
        self.driver.device.begin_chunk()?;
        ctx.clock.tick(&mut self.driver.device, 1)
    }

    fn write_chunk_inputs(&mut self, ctx: &mut Self::Ctx, p0: usize, m: usize) -> Result<()> {
        check_chunk_start(p0, self.driver.pos)?;
        if m != ctx.tokens.len() {
            return Err(refuse("the input row count differs from the chunk"));
        }
        ctx.ple_rows = self.driver.write_inputs(ctx.tokens, ctx.starts)?;
        Ok(())
    }

    fn run_layers(&mut self, ctx: &mut Self::Ctx, p0: usize, m: usize) -> Result<()> {
        if m != ctx.tokens.len() {
            return Err(refuse("the layer row count differs from the chunk"));
        }
        self.driver
            .queue_layers(p0, ctx.tokens, &ctx.ple_rows, &mut ctx.clock)
    }

    fn read_health(&mut self, _ctx: &mut Self::Ctx) -> Result<i32> {
        self.driver.device.check_health().map(|()| 0)
    }

    fn commit_chunk(
        &mut self,
        ctx: &mut Self::Ctx,
        p0: usize,
        m: usize,
        _status: i32,
    ) -> Result<()> {
        check_chunk_start(p0, self.driver.pos)?;
        if m != ctx.tokens.len() {
            return Err(refuse("the committed row count differs from the chunk"));
        }
        self.driver.pos = p0
            .checked_add(m)
            .ok_or_else(|| refuse("the position overflows"))?;
        Ok(())
    }

    fn bank_misses(&self) -> Option<u64> {
        self.driver.device.bank_misses()
    }
}

fn refuse(detail: &str) -> EngError {
    EngError::BadRequest(format!("Flash-Next shared prefill: {detail}"))
}

impl<D: ChunkDevice, F: TokenFeeds> ChunkDriver<D, F> {
    fn preflight_shared(&self, tokens: &[u32]) -> Result<()> {
        if self.poisoned {
            return Err(refuse("the state needs a reset after a failed step"));
        }
        let end = self
            .pos
            .checked_add(tokens.len())
            .ok_or_else(|| refuse("the position overflows"))?;
        if end > self.shape.positions {
            return Err(EngError::ContextLimit(format!(
                "Flash-Next shared prefill: {end} rows exceed {} positions",
                self.shape.positions
            )));
        }
        if i32::try_from(end).is_err() {
            return Err(refuse("the position does not fit the signed kernel index"));
        }
        if tokens.iter().any(|&token| {
            token as usize >= self.shape.start.sampler_vocab || i32::try_from(token).is_err()
        }) {
            return Err(refuse(
                "a token id is outside the sampler vocabulary or signed kernel index",
            ));
        }
        Ok(())
    }

    pub(super) fn feed_shared_rows(&mut self, tokens: &[u32]) -> Result<()> {
        self.preflight_shared(tokens)?;
        let cap = self.chunk_cap.max(1);
        if cap > MAX_EXL3_ROWS {
            return Err(refuse(
                format!("the shared path is not qualified above {MAX_EXL3_ROWS} rows").as_str(),
            ));
        }
        let start = self.pos;
        let steps = tokens
            .chunks(cap)
            .enumerate()
            .map(|(i, chunk)| PrefillStep::Chunk {
                p0: start + i * cap,
                rows: chunk.len(),
            })
            .collect();
        let mut first = true;
        drive_steps(start, tokens, steps, |_, rows| {
            self.run_shared_chunk(rows, false, first)?;
            first = false;
            Ok(())
        })?;
        Ok(())
    }

    pub(super) fn run_shared_chunk(
        &mut self,
        tokens: &[u32],
        want_logits: bool,
        open_window: bool,
    ) -> Result<Option<Vec<f32>>> {
        self.preflight_shared(tokens)?;
        let (p0, m) = (self.pos, tokens.len());
        if !(1..=MAX_EXL3_ROWS).contains(&m) {
            return Err(refuse(
                format!("a chunk must hold 1 to {MAX_EXL3_ROWS} rows").as_str(),
            ));
        }
        let began = Instant::now();
        let starts = self.feeds.begin_chunk(tokens)?;
        let DriverShape { start, .. } = self.shape;
        if starts.len() != m
            || starts.iter().any(|row| {
                row.stream.len() != start.stream || row.ple_embedding.len() != start.ple_embed
            })
        {
            return Err(refuse(
                "the host start vectors do not match the chunk shape",
            ));
        }
        if prof::on() {
            prof::add(0, began.elapsed());
            prof::ROWS.fetch_add(m as u64, Ordering::Relaxed);
        }
        if p0 == 0 {
            self.generation += 1;
        }
        let mut ctx = FlashChunkCtx {
            tokens,
            starts: &starts,
            ple_rows: Vec::new(),
            clock: ChunkClock::new(),
        };
        let mut family = FlashChunkFamily { driver: self };
        let result = (|| {
            if open_window {
                family.begin_window(&mut ctx)?;
            }
            family.write_chunk_inputs(&mut ctx, p0, m)?;
            family.clear_chunk_status(&mut ctx)?;
            family.run_layers(&mut ctx, p0, m)?;
            let logits = family.driver.queue_head(want_logits, &mut ctx.clock)?;
            let status = family.read_health(&mut ctx)?;
            family.commit_chunk(&mut ctx, p0, m, status)?;
            Ok(logits)
        })();
        if result.is_err() {
            family.driver.poisoned = true;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::{
        flash_next::cuda::driver::fakes::{FakeChunkDevice, FakeFeeds, shape},
        session::TokenStepper,
    };

    #[test]
    fn layer_queue_uses_the_walkers_position_without_publishing_it() {
        let mut driver =
            ChunkDriver::new(FakeChunkDevice::default(), FakeFeeds::default(), shape()).unwrap();
        driver.feed_rows(&[1]).unwrap();
        driver.device.log.clear();
        {
            let mut family = FlashChunkFamily {
                driver: &mut driver,
            };
            let mut ctx = FlashChunkCtx {
                tokens: &[2],
                starts: &[],
                ple_rows: Vec::new(),
                clock: ChunkClock::new(),
            };
            assert!(family.run_layers(&mut ctx, 7, 2).is_err());
            assert!(family.driver.device.log.is_empty());
            family.run_layers(&mut ctx, 7, 1).unwrap();
        }
        let shape = shape();
        assert!(
            driver
                .device
                .log
                .contains(&format!("layers 0..{} @7 m1", shape.ple_layer))
        );
        assert!(driver.device.log.contains(&format!(
            "layers {}..{} @7 m1",
            shape.ple_layer, shape.layers
        )));
        assert_eq!(driver.fed(), 1);
    }
}
