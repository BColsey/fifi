//! `window_logic` against a recording host. The fake records every primitive in call order and can
//! be told to fail one of them, so the order, the hooks and the failure rules of the real decision
//! code are tested on the host.

use super::*;

/// Calls a name at most once when told to: `fail` names the primitive that returns an error.
#[derive(Default)]
pub(super) struct Fake {
    /// Every primitive in call order.
    pub(super) log: Vec<String>,
    /// Committed rows.
    pub(super) tokens: usize,
    pub(super) device: bool,
    pub(super) window: Option<Window<usize, u32>>,
    /// The primitive that fails the next time it is called (once).
    pub(super) fail: Option<&'static str>,
}

impl Fake {
    pub(super) fn new(tokens: usize, device: bool) -> Self {
        Self {
            tokens,
            device,
            ..Self::default()
        }
    }

    fn note(&mut self, what: impl Into<String>) {
        self.log.push(what.into());
    }

    /// Records `name` and fails it when it is the armed failure.
    pub(super) fn step(&mut self, name: &'static str) -> Result<(), EngError> {
        self.note(name);
        if self.fail == Some(name) {
            self.fail = None;
            return Err(EngError::BadRequest(format!("fake: {name} failed")));
        }
        Ok(())
    }
}

impl WindowHost for Fake {
    type Saved = usize;
    type Row = u32;

    fn device_wanted(&self) -> bool {
        self.device
    }

    fn tokens(&self) -> usize {
        self.tokens
    }

    fn clear_window(&mut self) {
        self.note("clear_window");
        self.window = None;
    }

    fn store_window(&mut self, window: Window<usize, u32>) {
        self.note("store_window");
        self.window = Some(window);
    }

    fn take_window(&mut self) -> Option<Window<usize, u32>> {
        self.note("take_window");
        self.window.take()
    }

    fn export(&mut self) -> Result<usize, EngError> {
        self.step("export")?;
        Ok(self.tokens)
    }

    fn run_row(&mut self, token: u32) -> Result<(u32, u32), EngError> {
        self.note(format!("run_row {token}"));
        if self.fail == Some("run_row") {
            self.fail = None;
            return Err(EngError::BadRequest("fake: run_row failed".into()));
        }
        self.tokens += 1;
        Ok((token + 100, token + 200))
    }

    fn replay_row(&mut self, token: u32) -> Result<(), EngError> {
        self.note(format!("replay_row {token}"));
        if self.fail == Some("replay_row") {
            self.fail = None;
            return Err(EngError::BadRequest("fake: replay_row failed".into()));
        }
        self.tokens += 1;
        Ok(())
    }

    fn import(&mut self, saved: &usize) -> Result<(), EngError> {
        self.step("import")?;
        self.tokens = *saved;
        Ok(())
    }

    fn require_armed(&mut self) -> Result<(), EngError> {
        self.step("require_armed")
    }

    fn complete_prefix_save(&mut self) -> Result<(), EngError> {
        self.step("complete_prefix_save")
    }

    fn rollback_begin(&mut self, rows: usize) -> Result<(), EngError> {
        self.note(format!("rollback_begin {rows}"));
        if self.fail == Some("rollback_begin") {
            self.fail = None;
            return Err(EngError::BadRequest("fake: rollback_begin failed".into()));
        }
        Ok(())
    }

    fn capture(&mut self, row: usize) -> Result<(), EngError> {
        self.note(format!("capture {row}"));
        if self.fail == Some("capture") {
            self.fail = None;
            return Err(EngError::BadRequest("fake: capture failed".into()));
        }
        Ok(())
    }

    fn rollback_discard(&mut self) {
        self.note("rollback_discard");
    }

    fn rollback_poison(&mut self) {
        self.note("rollback_poison");
    }

    fn rollback_window_complete(&mut self) -> Result<(), EngError> {
        self.step("rollback_window_complete")
    }

    fn rollback_finish(&mut self, kept: usize, committed: usize) -> Result<bool, EngError> {
        self.note(format!("rollback_finish {kept} {committed}"));
        if self.fail == Some("rollback_finish") {
            self.fail = None;
            return Err(EngError::BadRequest("fake: rollback_finish failed".into()));
        }
        Ok(kept < 3)
    }

    fn trace_mark(&mut self, mark: Mark) {
        self.note(format!("{mark:?}"));
    }

    fn feed_fault(&mut self, row: usize) -> Result<(), EngError> {
        self.note(format!("feed_fault {row}"));
        Ok(())
    }

    fn replay_fault(&mut self, row: usize) -> Result<(), EngError> {
        self.note(format!("replay_fault {row}"));
        Ok(())
    }

    fn capture_fault(&mut self, row: usize) -> Result<(), EngError> {
        self.note(format!("capture_fault {row}"));
        Ok(())
    }

    fn restore_fault(&mut self) -> Result<(), EngError> {
        self.step("restore_fault")
    }
}

pub(super) fn commit(base: usize, kept: usize, rows: usize) -> Commit {
    Commit {
        committed_rows: base + kept,
        kept_rows: kept,
        discarded_rows: rows - kept,
        state_slot: kept - 1,
    }
}

fn ran(host: &Fake) -> Vec<&str> {
    host.log.iter().map(String::as_str).collect()
}

// The success-path pin. These sequences were read off the bodies of `run_window`,
// `run_window_device`, `commit_window` and `commit_window_device` as they were before the
// extraction; the extraction commit moves the bodies and these sequences must not change in any
// later Q2 commit (a Q2 commit may only add calls on error paths).

#[test]
fn pin_host_window_that_keeps_every_row() {
    let mut host = Fake::new(10, false);
    run_window(&mut host, &[1, 2, 3]).expect("run");
    assert_eq!(
        ran(&host),
        [
            "clear_window",
            "WindowStart",
            "export",
            "Exported",
            "feed_fault 0",
            "run_row 1",
            "RowDone",
            "feed_fault 1",
            "run_row 2",
            "RowDone",
            "feed_fault 2",
            "run_row 3",
            "RowDone",
            "store_window",
        ]
    );
    assert_eq!(host.tokens, 13);
    host.log.clear();
    commit_window(&mut host, &commit(10, 3, 3)).expect("commit");
    assert_eq!(ran(&host), ["take_window", "Dropped"]);
    assert_eq!(host.tokens, 13, "a full accept leaves the state as it is");
    assert!(host.window.is_none());
}

#[test]
fn pin_host_window_that_keeps_some_rows() {
    let mut host = Fake::new(10, false);
    run_window(&mut host, &[1, 2, 3]).expect("run");
    host.log.clear();
    commit_window(&mut host, &commit(10, 2, 3)).expect("commit");
    assert_eq!(
        ran(&host),
        [
            "take_window",
            "CommitStart",
            "import",
            "Imported",
            "replay_fault 0",
            "replay_row 1",
            "ReplayDone",
            "replay_fault 1",
            "replay_row 2",
            "ReplayDone",
            "Dropped",
        ]
    );
    assert_eq!(host.tokens, 12, "the state of two one-row steps");
}

#[test]
fn pin_device_window_that_keeps_every_row() {
    let mut host = Fake::new(10, true);
    run_window(&mut host, &[1, 2, 3]).expect("run");
    assert_eq!(
        ran(&host),
        [
            "clear_window",
            "WindowStart",
            "require_armed",
            "complete_prefix_save",
            "rollback_begin 3",
            "feed_fault 0",
            "run_row 1",
            "RowDone",
            "capture_fault 0",
            "capture 0",
            "Captured",
            "feed_fault 1",
            "run_row 2",
            "RowDone",
            "capture_fault 1",
            "capture 1",
            "Captured",
            "feed_fault 2",
            "run_row 3",
            "RowDone",
            "rollback_window_complete",
            "store_window",
        ]
    );
    host.log.clear();
    commit_window(&mut host, &commit(10, 3, 3)).expect("commit");
    assert_eq!(
        ran(&host),
        ["take_window", "CommitStart", "rollback_finish 3 13"]
    );
}

#[test]
fn pin_device_window_that_keeps_some_rows() {
    let mut host = Fake::new(10, true);
    run_window(&mut host, &[1, 2, 3]).expect("run");
    host.log.clear();
    commit_window(&mut host, &commit(10, 1, 3)).expect("commit");
    assert_eq!(
        ran(&host),
        [
            "take_window",
            "CommitStart",
            "restore_fault",
            "rollback_finish 1 11",
            "Imported",
        ]
    );
}

#[test]
fn a_commit_without_a_window_touches_nothing() {
    let mut host = Fake::new(10, false);
    assert!(commit_window(&mut host, &commit(10, 1, 2)).is_err());
    assert_eq!(ran(&host), ["take_window"]);
    assert_eq!(host.tokens, 10);
}

#[test]
fn a_device_window_fails_before_its_first_row_without_poisoning() {
    for early in ["require_armed", "complete_prefix_save", "rollback_begin"] {
        let mut host = Fake::new(10, true);
        host.fail = Some(early);
        assert!(run_window(&mut host, &[1, 2]).is_err(), "{early}");
        assert!(
            !host.log.iter().any(|call| call == "rollback_poison"),
            "{early}"
        );
    }
}

#[test]
fn a_device_window_that_fails_after_a_row_poisons_and_drops_the_window() {
    for failing in ["run_row", "capture"] {
        let mut host = Fake::new(10, true);
        host.fail = Some(failing);
        // A failing first row ran nothing, a failing capture follows a row that ran.
        let result = run_window(&mut host, &[1, 2, 3]);
        assert!(result.is_err());
        let poisoned = host.log.iter().any(|call| call == "rollback_poison");
        assert_eq!(poisoned, failing == "capture", "{failing}");
        assert!(host.log.iter().any(|call| call == "rollback_discard"));
        assert!(host.window.is_none());
    }
}

#[test]
fn a_host_window_that_fails_leaves_no_window() {
    let mut host = Fake::new(10, false);
    host.fail = Some("run_row");
    assert!(run_window(&mut host, &[1, 2]).is_err());
    assert!(host.window.is_none());
}
