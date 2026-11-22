//! Which device sampler a request's settings take, and the arguments each one is given.
//!
//! A request whose `top_k` is within the ranked width draws from the shortlist the card ranked
//! (exact: nothing below the width can be drawn); greedy is the same draw at temperature 0. Any
//! other request (no `top_k`, or one wider than the ranking) draws from the whole row. Both use the
//! Philox uniform of the seed and the position, so a request replays the same tokens.

use crate::EngError;
use crate::family::sampler::Sampler;
use crate::kernels::cuda::resident::sample::{
    SampleCall, SampleParams, SampleShortlistCall, ShortlistDraw,
};
use crate::kernels::cuda::resident::shortlist::{self, ShortlistRowsCall};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

/// The sampler one token takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DrawRoute {
    /// A draw over the ranked shortlist of `width` candidates.
    Shortlist(ShortlistDraw),
    /// A draw over the whole row.
    FullRow(SampleParams),
}

/// The route of the next token of `sampler`, drawn at `position`, when the card ranks `width`
/// candidates per row.
pub(crate) fn route(sampler: &Sampler, position: u64, width: usize) -> DrawRoute {
    let params = sampler.params();
    let temperature = if params.is_greedy() {
        0.0
    } else {
        params.temperature() as f32
    };
    let top_k = params.top_k() as usize;
    if params.is_greedy() || (1..=width).contains(&top_k) {
        return DrawRoute::Shortlist(ShortlistDraw {
            temperature,
            top_k: if params.is_greedy() { 0 } else { top_k },
            top_p: if params.is_greedy() {
                1.0
            } else {
                params.top_p() as f32
            },
            seed: sampler.seed(),
            position,
        });
    }
    DrawRoute::FullRow(SampleParams {
        temperature,
        top_k,
        top_p: params.top_p() as f32,
        min_p: 0.0,
        seed: sampler.seed(),
        position,
    })
}

/// Bytes of the words of one row of a draw (token, log probability, status; padded to 16).
const WORD_BYTES: usize = 16;

/// The device buffers of a token draw for up to `rows` rows of `vocab` logits each, ranked at
/// `width` candidates: the partials and the records of the ranking, the words the sampler writes,
/// and the probability row of a draw over the whole row.
pub(crate) struct DrawBuffers {
    scratch: DeviceBuffer,
    records: DeviceBuffer,
    words: DeviceBuffer,
    probs: DeviceBuffer,
    rows: usize,
    vocab: usize,
    width: usize,
}

impl DrawBuffers {
    pub(crate) fn new(
        ctx: &CudaContext,
        rows: usize,
        vocab: usize,
        width: usize,
    ) -> Result<Self, EngError> {
        let ranking = shortlist::shape(width)?;
        Ok(Self {
            scratch: ctx.allocate(rows * ranking.result_offset)?,
            records: ctx.allocate(rows * ranking.part_bytes)?,
            words: ctx.allocate(rows * WORD_BYTES)?,
            probs: ctx.allocate(vocab * 4)?,
            rows,
            vocab,
            width,
        })
    }

    /// Ranks each of the `routes.len()` logits rows of `logits` and draws one token from each by
    /// its route, on `stream`. The token and the status of each row come back in one read.
    pub(crate) fn draw(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        logits: &DeviceBuffer,
        routes: &[DrawRoute],
    ) -> Result<Vec<u32>, EngError> {
        let m = routes.len();
        if m == 0 || m > self.rows {
            return Err(EngError::BadRequest(format!(
                "a draw takes 1 to {} rows, got {m}",
                self.rows
            )));
        }
        let (vocab, width) = (self.vocab, self.width);
        let ranking = shortlist::shape(width)?;
        self.words
            .write_at_enqueue(stream, 0, &vec![0u8; m * WORD_BYTES])?;
        ctx.shortlist_rows(
            stream,
            ShortlistRowsCall {
                k: width,
                m,
                n: vocab,
                ld: vocab,
                logits: logits.range(0, m * vocab * 4)?,
                scratch: self.scratch.range(0, m * ranking.result_offset)?,
                scratch_stride: ranking.result_offset,
                out: self.records.range(0, m * ranking.part_bytes)?,
            },
        )?;
        for (row, route) in routes.iter().enumerate() {
            let word = row * WORD_BYTES;
            let (token, logprob, status) = (
                self.words.range(word, 4)?,
                self.words.range(word + 4, 4)?,
                self.words.range(word + 8, 4)?,
            );
            match route {
                DrawRoute::Shortlist(settings) => {
                    let record = row * ranking.part_bytes;
                    ctx.sample_shortlist(
                        stream,
                        SampleShortlistCall {
                            token,
                            logprob,
                            status,
                            entries: self.records.range(record, width * 8)?,
                            flag: self.records.range(record + width * 8, 4)?,
                            count: width,
                            draw: *settings,
                        },
                    )?;
                }
                DrawRoute::FullRow(params) => {
                    ctx.sample(
                        stream,
                        SampleCall {
                            token,
                            logprob,
                            status,
                            probs: self.probs.range(0, vocab * 4)?,
                            logits: logits.range(row * vocab * 4, vocab * 4)?,
                            n: vocab,
                            params: *params,
                        },
                    )?;
                }
            }
        }
        let mut bytes = vec![0u8; m * WORD_BYTES];
        self.words.read_at(stream, 0, &mut bytes)?;
        bytes
            .as_chunks::<WORD_BYTES>()
            .0
            .iter()
            .map(|word| {
                let token = u32::from_le_bytes(word[0..4].try_into().expect("token word"));
                let status = u32::from_le_bytes(word[8..12].try_into().expect("status word"));
                if status != 0 {
                    return Err(EngError::Backend {
                        operation: "device draw",
                        message: format!(
                            "the device draw refused the row (status {status}: 2 empty, 4 non-finite)"
                        ),
                    });
                }
                Ok(token)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::sampler::SamplingParams;

    fn sampler(temperature: f64, top_p: f64, top_k: u32) -> Sampler {
        Sampler::new(SamplingParams::new(temperature, top_p, top_k).unwrap(), 7)
    }

    #[test]
    fn a_top_k_within_the_width_draws_from_the_shortlist() {
        let route = route(&sampler(1.0, 0.95, 64), 12, 64);
        assert!(matches!(route, DrawRoute::Shortlist(d) if d.top_k == 64 && d.position == 12));
        let route = super::route(&sampler(1.0, 0.95, 64), 12, 20);
        assert!(
            matches!(route, DrawRoute::FullRow(_)),
            "64 is wider than 20"
        );
    }

    #[test]
    fn greedy_is_a_shortlist_draw_at_temperature_zero() {
        let route = route(&Sampler::greedy(), 0, 20);
        assert!(matches!(route, DrawRoute::Shortlist(d) if d.temperature == 0.0));
    }

    #[test]
    fn no_top_k_draws_from_the_whole_row() {
        let route = route(&sampler(0.7, 1.0, 0), 3, 64);
        assert!(matches!(route, DrawRoute::FullRow(p) if p.top_k == 0 && p.seed == 7));
    }
}
