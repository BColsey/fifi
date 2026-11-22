//! Which device sampler a request's settings take, and the arguments each one is given.
//!
//! A request whose `top_k` is within the ranked width draws from the shortlist the card ranked
//! (exact: nothing below the width can be drawn); greedy is the same draw at temperature 0. Any
//! other request (no `top_k`, or one wider than the ranking) draws from the whole row. Both use the
//! Philox uniform of the seed and the position, so a request replays the same tokens.

use crate::EngError;
use crate::family::sampler::{Sampler, SamplingParams};
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

/// The `top_k` and `top_p` a device draw runs with for `params`, over rows of `vocab` logits, so
/// that the device keeps the tokens the host sampler keeps. A `top_k` that reaches the vocabulary
/// keeps every token (0). A `top_p` of zero, or one that casts to zero in f32, keeps the top
/// token alone, which is `top_k` 1; the device refuses a zero `top_p` mid-generation.
pub(crate) fn filters(params: SamplingParams, vocab: usize) -> (usize, f32) {
    let top_p = params.top_p() as f32;
    if top_p.is_nan() || top_p <= 0.0 {
        return (1, 1.0);
    }
    let top_k = params.top_k() as usize;
    (if top_k >= vocab { 0 } else { top_k }, top_p)
}

/// The sampling settings a device shortlist draw runs under: [`filters`] applied to `params`.
pub(crate) fn kept_params(params: SamplingParams, vocab: usize) -> SamplingParams {
    if params.is_greedy() {
        return params;
    }
    let (top_k, top_p) = filters(params, vocab);
    SamplingParams::new(
        params.temperature(),
        f64::from(top_p),
        u32::try_from(top_k).unwrap_or(u32::MAX),
    )
    .unwrap_or(params)
}

/// Draws the next token on the host from one logits row, for a request whose sampler records its
/// draws: only the host walk has the candidates to record.
pub(crate) fn host_draw(
    operation: &'static str,
    logits: &[f32],
    sampler: &mut Sampler,
    vocab: usize,
) -> Result<u32, EngError> {
    let visible = logits.get(..vocab).ok_or_else(|| EngError::Backend {
        operation,
        message: format!(
            "logits have {} values but the sampler vocabulary is {vocab}",
            logits.len()
        ),
    })?;
    sampler.sample(visible)
}

/// The route of the next token of `sampler`, drawn at `position`, when the card ranks `width`
/// candidates per row of `vocab` logits.
pub(crate) fn route(sampler: &Sampler, position: u64, width: usize, vocab: usize) -> DrawRoute {
    let params = sampler.params();
    let greedy = params.is_greedy();
    let temperature = if greedy {
        0.0
    } else {
        params.temperature() as f32
    };
    let (top_k, top_p) = filters(params, vocab);
    if greedy || (1..=width).contains(&top_k) {
        return DrawRoute::Shortlist(ShortlistDraw {
            temperature,
            top_k: if greedy { 0 } else { top_k },
            top_p: if greedy { 1.0 } else { top_p },
            seed: sampler.seed(),
            position,
        });
    }
    DrawRoute::FullRow(SampleParams {
        temperature,
        top_k,
        top_p,
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

    const VOCAB: usize = 262_144;

    fn sampler(temperature: f64, top_p: f64, top_k: u32) -> Sampler {
        Sampler::new(SamplingParams::new(temperature, top_p, top_k).unwrap(), 7)
    }

    fn at(top_k: u32, width: usize) -> DrawRoute {
        route(&sampler(1.0, 0.95, top_k), 12, width, VOCAB)
    }

    #[test]
    fn a_top_k_within_the_width_draws_from_the_shortlist() {
        let route = at(64, 64);
        assert!(matches!(route, DrawRoute::Shortlist(d) if d.top_k == 64 && d.position == 12));
        assert!(
            matches!(at(64, 20), DrawRoute::FullRow(_)),
            "64 is wider than 20"
        );
    }

    #[test]
    fn the_top_k_edges_of_the_width_and_the_vocabulary_take_the_route_the_host_keeps() {
        assert!(
            matches!(at(0, 64), DrawRoute::FullRow(p) if p.top_k == 0),
            "0 keeps every token"
        );
        assert!(matches!(at(64, 64), DrawRoute::Shortlist(d) if d.top_k == 64));
        assert!(
            matches!(at(65, 64), DrawRoute::FullRow(p) if p.top_k == 65),
            "65 is past the width"
        );
        let vocab = VOCAB as u32;
        assert!(
            matches!(at(vocab - 1, 64), DrawRoute::FullRow(p) if p.top_k == VOCAB - 1),
            "one below the vocabulary still cuts a token"
        );
        assert!(
            matches!(at(vocab, 64), DrawRoute::FullRow(p) if p.top_k == 0),
            "the vocabulary keeps every token"
        );
        assert!(matches!(at(vocab + 1, 64), DrawRoute::FullRow(p) if p.top_k == 0));
        assert!(matches!(at(1_000_000, 64), DrawRoute::FullRow(p) if p.top_k == 0));
    }

    #[test]
    fn a_zero_top_p_keeps_the_top_token_on_either_route() {
        for top_k in [0, 5, 64, 100, VOCAB as u32 + 1] {
            for top_p in [0.0, 1e-50] {
                let route = route(&sampler(1.0, top_p, top_k), 3, 64, VOCAB);
                assert!(
                    matches!(route, DrawRoute::Shortlist(d) if d.top_k == 1 && d.top_p == 1.0),
                    "top_k {top_k} top_p {top_p}: {route:?}"
                );
            }
        }
        assert_eq!(
            filters(SamplingParams::new(1.0, 0.0, 0).unwrap(), VOCAB),
            (1, 1.0)
        );
    }

    #[test]
    fn greedy_is_a_shortlist_draw_at_temperature_zero() {
        let route = route(&Sampler::greedy(), 0, 20, VOCAB);
        assert!(matches!(route, DrawRoute::Shortlist(d) if d.temperature == 0.0));
    }

    #[test]
    fn no_top_k_draws_from_the_whole_row() {
        let route = route(&sampler(0.7, 1.0, 0), 3, 64, VOCAB);
        assert!(matches!(route, DrawRoute::FullRow(p) if p.top_k == 0 && p.seed == 7));
    }
}
