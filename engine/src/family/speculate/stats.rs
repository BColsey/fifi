//! Counters of the speculation of one engine, printed once when the engine ends.

/// Why a step ran one row although speculation was on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StandDown {
    /// The draft source had nothing to offer.
    NoDraft,
    /// The output limit or the cache left no room for a draft.
    NoRoom,
}

/// What the windows of an engine did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SpecStats {
    windows: u64,
    rows_fed: u64,
    drafted: u64,
    accepted: u64,
    emitted: u64,
    stand_down: [u64; 2],
}

impl SpecStats {
    /// A window of `rows` rows (the pending token included) that emitted `emitted` tokens.
    pub(crate) fn window(&mut self, rows: usize, emitted: usize) {
        let drafts = rows.saturating_sub(1) as u64;
        self.windows += 1;
        self.rows_fed += rows as u64;
        self.drafted += drafts;
        self.accepted += (emitted as u64).saturating_sub(1).min(drafts);
        self.emitted += emitted as u64;
    }

    /// A one-row step that speculation left alone.
    pub(crate) fn stand_down(&mut self, why: StandDown) {
        self.stand_down[why as usize] += 1;
        self.emitted += 1;
    }

    /// True once a window or a stand-down was counted.
    pub(crate) fn used(&self) -> bool {
        self.windows > 0 || self.stand_down.iter().any(|&count| count > 0)
    }

    /// Drafts accepted per draft verified, or `None` before any draft ran.
    pub(crate) fn acceptance(&self) -> Option<f64> {
        (self.drafted > 0).then(|| self.accepted as f64 / self.drafted as f64)
    }

    /// Tokens emitted per model step, windows and one-row steps alike.
    pub(crate) fn tokens_per_step(&self) -> Option<f64> {
        let steps = self.windows + self.stand_down.iter().sum::<u64>();
        (steps > 0).then(|| self.emitted as f64 / steps as f64)
    }

    /// The teardown line.
    pub(crate) fn line(&self) -> String {
        format!(
            "spec: windows={} rows_fed={} drafted={} accepted={} emitted={} \
             stand_down_no_draft={} stand_down_no_room={} \
             acceptance={} tokens_per_step={}",
            self.windows,
            self.rows_fed,
            self.drafted,
            self.accepted,
            self.emitted,
            self.stand_down[StandDown::NoDraft as usize],
            self.stand_down[StandDown::NoRoom as usize],
            self.acceptance()
                .map_or("-".to_owned(), |a| format!("{a:.3}")),
            self.tokens_per_step()
                .map_or("-".to_owned(), |t| format!("{t:.3}")),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_count_rows_drafts_and_accepts() {
        let mut stats = SpecStats::default();
        stats.window(4, 4);
        stats.window(3, 1);
        assert_eq!(
            (
                stats.windows,
                stats.rows_fed,
                stats.drafted,
                stats.accepted,
                stats.emitted
            ),
            (2, 7, 5, 3, 5)
        );
        assert_eq!(stats.acceptance(), Some(0.6));
        assert_eq!(stats.tokens_per_step(), Some(2.5));
    }

    #[test]
    fn a_stand_down_is_a_one_token_step_under_its_reason() {
        let mut stats = SpecStats::default();
        stats.stand_down(StandDown::NoDraft);
        stats.stand_down(StandDown::NoDraft);
        stats.stand_down(StandDown::NoRoom);
        assert_eq!(stats.stand_down, [2, 1]);
        assert_eq!(stats.tokens_per_step(), Some(1.0));
        assert_eq!(stats.acceptance(), None);
    }

    #[test]
    fn the_line_names_every_counter() {
        let mut stats = SpecStats::default();
        assert!(stats.line().contains("acceptance=- tokens_per_step=-"));
        stats.window(2, 2);
        let line = stats.line();
        for key in [
            "windows=1",
            "rows_fed=2",
            "drafted=1",
            "accepted=1",
            "emitted=2",
            "acceptance=1.000",
        ] {
            assert!(line.contains(key), "{key} in {line}");
        }
    }
}
