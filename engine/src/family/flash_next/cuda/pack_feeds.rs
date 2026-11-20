//! The host feeds of a token from the real pack: [`PackFeeds`] is the [`TokenFeeds`] of the driver.
//!
//! The driver gathers its BF16 token embeddings and n-gram rows before writing anything to the
//! card. With the optional row pool, a multi-token chunk submits its n-gram reads before gathering
//! the BF16 embeddings and shares the decoded row block in the ordered PLE updates.

use std::{
    collections::VecDeque,
    rc::Rc,
    time::{Duration, Instant},
};

use super::{
    driver::{FeedProfile, StartVectors, TokenFeeds},
    embedding::stream_row,
    feeds::RowReader,
    ple_feed::{Gathered, PleFeed, PleKeyValue},
};
use crate::{EngError, weights::StreamedCheckpoint};

/// The gathered rows of the token being fed, with the token.
struct Pending<T> {
    token: u32,
    rows: T,
}

/// Takes the front pending rows, which must be `token`'s; a chunk consumes its rows in order.
fn take_pending<T>(pending: &mut VecDeque<Pending<T>>, token: u32) -> Result<T, EngError> {
    if let Some(front) = pending.pop_front_if(|front| front.token == token) {
        return Ok(front.rows);
    }
    Err(EngError::BadRequest(format!(
        "no n-gram rows were read for token {token}"
    )))
}

pub(super) struct PackFeeds {
    source: &'static StreamedCheckpoint,
    ple: PleFeed,
    hidden: usize,
    branches: usize,
    pending: VecDeque<Pending<Gathered>>,
    reader: Option<RowReader>,
    profile: Option<FeedProfile>,
}

fn time_feed<T>(total: Option<&mut Duration>, f: impl FnOnce() -> T) -> T {
    let Some(total) = total else {
        return f();
    };
    let began = Instant::now();
    let result = f();
    *total += began.elapsed();
    result
}

impl PackFeeds {
    pub(super) fn new(
        source: &'static StreamedCheckpoint,
        ple: PleFeed,
        hidden: usize,
        branches: usize,
        reader: Option<RowReader>,
    ) -> Self {
        Self {
            source,
            ple,
            hidden,
            branches,
            pending: VecDeque::new(),
            reader,
            profile: None,
        }
    }

    fn begin_bulk_chunk(
        &self,
        tokens: &[u32],
        reader: &RowReader,
        mut profile: Option<&mut FeedProfile>,
    ) -> Result<(Vec<StartVectors>, VecDeque<Pending<Gathered>>), EngError> {
        let (plan, read) = time_feed(profile.as_mut().map(|p| &mut p.bulk_plan_submit), || {
            let plan = self.ple.plan_chunk(tokens)?;
            let read = reader.submit(&plan.all())?;
            Ok::<_, EngError>((plan, read))
        })?;
        let streams: Result<Vec<_>, _> =
            time_feed(profile.as_mut().map(|p| &mut p.token_embedding), || {
                tokens
                    .iter()
                    .map(|&token| stream_row(self.source, token, self.hidden, self.branches))
                    .collect()
            });
        let block = time_feed(profile.as_mut().map(|p| &mut p.bulk_join), || read.wait());
        let streams = streams?;
        let block = Rc::new(block?);

        time_feed(profile.map(|p| &mut p.bulk_assembly), || {
            let mut starts = Vec::with_capacity(tokens.len());
            let mut pending = VecDeque::with_capacity(tokens.len());
            for ((&token, stream), ids) in tokens.iter().zip(streams).zip(plan.per_token) {
                let ple_embedding = block.embedding(&ids, 1.0)?;
                starts.push(StartVectors {
                    stream,
                    ple_embedding,
                });
                pending.push_back(Pending {
                    token,
                    rows: Gathered::Bulk {
                        ids,
                        block: Rc::clone(&block),
                    },
                });
            }
            Ok((starts, pending))
        })
    }
}

impl TokenFeeds for PackFeeds {
    fn begin(&mut self, token: u32) -> Result<StartVectors, EngError> {
        self.begin_chunk(&[token], false)
            .map(|mut starts| starts.pop().expect("one start"))
    }

    /// Row `i` plans against the history it will have when its turn comes (the tokens before it
    /// in the chunk), and every row set stays pending for the ordered update pass. All reads
    /// are host-only, so a failure leaves the feed where it was.
    fn begin_chunk(
        &mut self,
        tokens: &[u32],
        profile: bool,
    ) -> Result<Vec<StartVectors>, EngError> {
        let mut timing = profile.then(|| self.profile.unwrap_or_default());
        if let Some(reader) = self.reader.as_ref().filter(|_| tokens.len() > 1) {
            let (starts, pending) = self.begin_bulk_chunk(tokens, reader, timing.as_mut())?;
            if let Some(timing) = timing.as_mut() {
                timing.bulk_rows += tokens.len();
            }
            self.profile = timing;
            self.pending = pending;
            return Ok(starts);
        }

        let mut starts = Vec::with_capacity(tokens.len());
        let mut pending = VecDeque::with_capacity(tokens.len());
        for (i, &token) in tokens.iter().enumerate() {
            let stream = time_feed(timing.as_mut().map(|p| &mut p.token_embedding), || {
                stream_row(self.source, token, self.hidden, self.branches)
            })?;
            let (ple_embedding, rows) =
                time_feed(timing.as_mut().map(|p| &mut p.serial_ngram), || {
                    self.ple.chunk_embedding(&tokens[..i], token)
                })?;
            starts.push(StartVectors {
                stream,
                ple_embedding,
            });
            pending.push_back(Pending { token, rows });
        }
        if let Some(timing) = timing.as_mut() {
            timing.serial_rows += tokens.len();
        }
        self.profile = timing;
        self.pending = pending;
        Ok(starts)
    }

    fn take_feed_profile(&mut self) -> Option<FeedProfile> {
        self.profile.take()
    }

    fn ple_update(&mut self, token: u32, stream: &[f32]) -> Result<Vec<f32>, EngError> {
        let rows = take_pending(&mut self.pending, token)?;
        self.ple.update(token, &rows, stream)
    }

    fn set_ple_pair(&mut self, pair: PleKeyValue) -> Result<(), EngError> {
        self.ple.set_fixed(pair)
    }

    fn reset(&mut self) {
        self.profile = None;
        self.pending.clear();
        self.ple.reset();
    }

    fn ple_state(&self) -> Option<(&[u32], &[f32])> {
        Some(self.ple.state_parts())
    }

    fn ple_resume(&mut self, history: &[u32], ring: &[f32]) -> Result<(), EngError> {
        self.profile = None;
        self.pending.clear();
        self.ple.resume(history, ring)
    }
}

#[cfg(test)]
mod tests {
    use super::{Pending, VecDeque, take_pending};
    use crate::EngError;

    #[test]
    fn pending_rows_serve_their_own_token_once() {
        let mut pending = vec![Pending {
            token: 7,
            rows: "rows",
        }]
        .into();
        assert_eq!(take_pending(&mut pending, 7).unwrap(), "rows");
        assert!(matches!(
            take_pending(&mut pending, 7),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn rows_read_for_another_token_are_refused_and_kept() {
        let mut pending = vec![Pending {
            token: 7,
            rows: "rows",
        }]
        .into();
        assert!(matches!(
            take_pending(&mut pending, 8),
            Err(EngError::BadRequest(text)) if text.contains("token 8")
        ));
        // The queue keeps its row: a mismatched take is a caller bug, not a reason to drop
        // the row the real token still needs.
        assert_eq!(pending.len(), 1);
        assert_eq!(take_pending(&mut pending, 7).unwrap(), "rows");
        let mut empty: VecDeque<Pending<()>> = VecDeque::new();
        assert!(take_pending(&mut empty, 1).is_err());
    }
}
