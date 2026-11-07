//! The host feeds of a token from the real pack: [`PackFeeds`] is the [`TokenFeeds`] of the driver.
//!
//! WHAT: [`TokenFeeds::begin`] reads the token's BF16 embedding row (`embedding::stream_row`, one
//! positioned read of 5 KiB), plans and reads its 16 n-gram rows (`PleFeed::plan` and `gather`) and
//! keeps the gathered rows for the update. [`TokenFeeds::ple_update`] hands the stream at the PLE
//! layer's input to `PleFeed::update`, which is the host reference's PLE step over those rows.
//!
//! WHY the rows are read in `begin`: the driver reads them before it writes anything to the card, so
//! a failed read leaves the state as it was. Overlapping the read with the layers before the PLE
//! layer is a later change (the rows depend on the token history only).

use super::{
    driver::{StartVectors, TokenFeeds},
    embedding::stream_row,
    ple_feed::{Gathered, PleFeed},
};
use crate::{EngError, weights::StreamedCheckpoint};

/// The gathered rows of the token being fed, with the token.
struct Pending<T> {
    token: u32,
    rows: T,
}

/// Takes the pending rows for `token`; refuses rows read for another token or none at all.
fn take_pending<T>(pending: &mut Option<Pending<T>>, token: u32) -> Result<T, EngError> {
    pending
        .take()
        .filter(|pending| pending.token == token)
        .map(|pending| pending.rows)
        .ok_or_else(|| EngError::BadRequest(format!("no n-gram rows were read for token {token}")))
}

pub(super) struct PackFeeds {
    source: &'static StreamedCheckpoint,
    ple: PleFeed,
    hidden: usize,
    branches: usize,
    pending: Option<Pending<Gathered>>,
}

impl PackFeeds {
    pub(super) fn new(
        source: &'static StreamedCheckpoint,
        ple: PleFeed,
        hidden: usize,
        branches: usize,
    ) -> Self {
        Self {
            source,
            ple,
            hidden,
            branches,
            pending: None,
        }
    }
}

impl TokenFeeds for PackFeeds {
    fn begin(&mut self, token: u32) -> Result<StartVectors, EngError> {
        self.pending = None;
        let stream = stream_row(self.source, token, self.hidden, self.branches)?;
        let plan = self.ple.plan(token)?;
        let rows = self.ple.gather(&plan)?;
        let mut ple_embedding = Vec::new();
        for &(_, local) in &plan.rows {
            let values = rows.get(u64::from(local)).ok_or_else(|| {
                EngError::BadRequest(format!("n-gram row {local} was not gathered"))
            })?;
            ple_embedding.extend_from_slice(values);
        }
        self.pending = Some(Pending { token, rows });
        Ok(StartVectors {
            stream,
            ple_embedding,
        })
    }

    fn ple_update(&mut self, token: u32, stream: &[f32]) -> Result<Vec<f32>, EngError> {
        let rows = take_pending(&mut self.pending, token)?;
        self.ple.update(token, &rows, stream)
    }

    fn reset(&mut self) {
        self.pending = None;
        self.ple.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::{Pending, take_pending};
    use crate::EngError;

    #[test]
    fn pending_rows_serve_their_own_token_once() {
        let mut pending = Some(Pending {
            token: 7,
            rows: "rows",
        });
        assert_eq!(take_pending(&mut pending, 7).unwrap(), "rows");
        assert!(matches!(
            take_pending(&mut pending, 7),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn rows_read_for_another_token_are_refused_and_dropped() {
        let mut pending = Some(Pending {
            token: 7,
            rows: "rows",
        });
        assert!(matches!(
            take_pending(&mut pending, 8),
            Err(EngError::BadRequest(text)) if text.contains("token 8")
        ));
        assert!(pending.is_none());
        assert!(take_pending::<()>(&mut None, 1).is_err());
    }
}
