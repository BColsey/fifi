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
    /// The state is poisoned.
    pub(super) poisoned: bool,
    /// The recovery gate.
    pub(super) gate: bool,
    /// A pinned-prefix save is pending.
    pub(super) pending: bool,
    /// How many times the pinned pool was invalidated.
    pub(super) invalidations: usize,
    /// Rows the KV cache holds.
    pub(super) capacity: usize,
    /// The row index (counted in `run_row` calls of the current window) that is refused.
    pub(super) fail_nth_row: Option<usize>,
    /// A refused row also poisons the state, as a step that fails inside its launch does.
    pub(super) poison_on_row_failure: bool,
    /// Rows fed by the current window so far.
    pub(super) fed_rows: usize,
}

impl Fake {
    pub(super) fn new(tokens: usize, device: bool) -> Self {
        Self {
            tokens,
            device,
            capacity: 4096,
            ..Self::default()
        }
    }

    fn note(&mut self, what: impl Into<String>) {
        self.log.push(what.into());
    }

    /// Records `name` and fails it when it is the armed failure.
    pub(super) fn step(&mut self, name: &'static str) -> Result<(), EngError> {
        self.note(name);
        self.hit(name)
    }

    /// Fails `name` when it is the armed failure (the call is recorded by the caller).
    pub(super) fn hit(&mut self, name: &'static str) -> Result<(), EngError> {
        if self.fail == Some(name) {
            self.fail = None;
            return Err(EngError::BadRequest(format!("fake: {name} failed")));
        }
        Ok(())
    }

    /// Whether `name` was recorded.
    pub(super) fn called(&self, name: &str) -> bool {
        self.log.iter().any(|call| call == name)
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
        self.hit("run_row")?;
        let nth = self.fed_rows;
        self.fed_rows += 1;
        if self.fail_nth_row == Some(nth) {
            self.fail_nth_row = None;
            self.poisoned |= self.poison_on_row_failure;
            return Err(EngError::BadRequest("fake: row refused".into()));
        }
        self.tokens += 1;
        Ok((token + 100, token + 200))
    }

    fn replay_row(&mut self, token: u32) -> Result<(), EngError> {
        self.note(format!("replay_row {token}"));
        self.hit("replay_row")?;
        self.tokens += 1;
        Ok(())
    }

    fn overwrite(&mut self, saved: &usize) -> Result<(), EngError> {
        self.step("import")?;
        self.tokens = *saved;
        Ok(())
    }

    fn window_awaits(&self) -> bool {
        self.window.is_some()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn is_valid(&self) -> bool {
        !self.poisoned
    }

    fn needs_reset(&self) -> bool {
        self.gate
    }

    fn clear_needs_reset(&mut self) {
        self.note("clear_needs_reset");
        self.gate = false;
    }

    fn set_needs_reset(&mut self) {
        self.note("set_needs_reset");
        self.gate = true;
    }

    fn poison(&mut self) {
        self.note("poison");
        self.poisoned = true;
    }

    fn drop_pending_prefix(&mut self) {
        self.note("drop_pending_prefix");
        self.pending = false;
    }

    fn invalidate_pool(&mut self) {
        self.note("invalidate_pool");
        self.invalidations += 1;
    }

    fn close_window(&mut self) {
        self.note("close_window");
        self.window = None;
    }

    fn check_import(&mut self, _saved: &usize) -> Result<(), EngError> {
        self.step("check_import")
    }

    fn resident_kv(&mut self) -> Result<(), EngError> {
        self.step("resident_kv")
    }

    fn require_armed(&mut self) -> Result<(), EngError> {
        self.step("require_armed")
    }

    fn complete_prefix_save(&mut self) -> Result<(), EngError> {
        self.note("complete_prefix_save");
        // The pending save is taken before it can fail.
        self.pending = false;
        self.hit("complete_prefix_save")
    }

    fn rollback_begin(&mut self, rows: usize) -> Result<(), EngError> {
        self.note(format!("rollback_begin {rows}"));
        self.hit("rollback_begin")?;
        Ok(())
    }

    fn capture(&mut self, row: usize) -> Result<(), EngError> {
        self.note(format!("capture {row}"));
        self.hit("capture")?;
        Ok(())
    }

    fn rollback_discard(&mut self) {
        self.note("rollback_discard");
    }

    fn rollback_window_complete(&mut self) -> Result<(), EngError> {
        self.step("rollback_window_complete")
    }

    fn rollback_finish(&mut self, kept: usize, committed: usize) -> Result<bool, EngError> {
        self.note(format!("rollback_finish {kept} {committed}"));
        self.hit("rollback_finish")?;
        Ok(kept < 3)
    }

    fn trace_mark(&mut self, mark: Mark) {
        if mark == Mark::WindowStart {
            self.fed_rows = 0;
        }
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
            "close_window",
            "complete_prefix_save",
            "check_import",
            "resident_kv",
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
        assert!(!host.called("poison"), "{early}");
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
        let poisoned = host.called("poison");
        assert_eq!(poisoned, failing == "capture", "{failing}");
        assert!(host.log.iter().any(|call| call == "rollback_discard"));
        assert!(host.window.is_none());
        assert_eq!(host.gate, failing == "capture", "{failing}");
    }
}

#[test]
fn a_host_window_that_fails_leaves_no_window() {
    let mut host = Fake::new(10, false);
    host.fail = Some("run_row");
    assert!(run_window(&mut host, &[1, 2]).is_err());
    assert!(host.window.is_none());
}

// Q2, the commit side: every failure of a host commit after the window was taken is a failed
// commit.

/// A host window of three rows from 10 committed rows, with a save pending.
fn host_window() -> Fake {
    let mut host = Fake::new(10, false);
    host.pending = true;
    run_window(&mut host, &[1, 2, 3]).expect("run");
    host.log.clear();
    host
}

/// What a failed commit must leave: poisoned, gated, no pending save, the pool invalidated once,
/// no stored window.
fn assert_failed_closed(host: &Fake, what: &str) {
    assert!(host.poisoned, "{what}: the state must be poisoned");
    assert!(host.gate, "{what}: the recovery gate must be set");
    assert!(!host.pending, "{what}: no pending save may survive");
    assert_eq!(host.invalidations, 1, "{what}: the pool is invalidated");
    assert!(host.window.is_none(), "{what}: the window must be closed");
}

#[test]
fn a_successful_commit_poisons_nothing_and_sets_no_gate() {
    for kept in 1..=3 {
        let mut host = host_window();
        commit_window(&mut host, &commit(10, kept, 3)).expect("commit");
        assert!(!host.poisoned && !host.gate, "kept {kept}");
        assert!(!host.called("poison") && !host.called("set_needs_reset"));
        assert_eq!(host.invalidations, 0);
        assert_eq!(host.tokens, 10 + kept);
    }
}

#[test]
fn every_failure_of_a_host_commit_after_the_window_was_taken_fails_it_closed() {
    for failing in [
        "complete_prefix_save",
        "check_import",
        "resident_kv",
        "import",
        "replay_row",
    ] {
        let mut host = host_window();
        host.fail = Some(failing);
        assert!(
            commit_window(&mut host, &commit(10, 2, 3)).is_err(),
            "{failing}"
        );
        assert_failed_closed(&host, failing);
    }
}

#[test]
fn a_commit_that_fails_before_begin_import_does_not_import() {
    for failing in ["complete_prefix_save", "check_import", "resident_kv"] {
        let mut host = host_window();
        host.fail = Some(failing);
        commit_window(&mut host, &commit(10, 2, 3)).expect_err(failing);
        assert!(!host.called("import"), "{failing}");
        assert!(!host.called("replay_row 1"), "{failing}");
    }
}

#[test]
fn a_commit_on_a_state_poisoned_since_the_window_stays_poisoned_and_imports_nothing() {
    let mut host = host_window();
    host.poisoned = true;
    commit_window(&mut host, &commit(10, 2, 3)).expect_err("a poisoned state");
    assert_failed_closed(&host, "poisoned between run and commit");
    assert!(!host.called("import") && !host.called("complete_prefix_save"));
}

#[test]
fn a_commit_with_the_recovery_gate_set_is_refused_and_imports_nothing() {
    let mut host = host_window();
    host.gate = true;
    commit_window(&mut host, &commit(10, 2, 3)).expect_err("the gate");
    assert!(host.poisoned && !host.called("import"));
}

#[test]
fn a_verdict_that_is_not_this_windows_poisons_the_state() {
    let wrong = [
        (
            "committed rows one too many",
            Commit {
                committed_rows: 13,
                ..commit(10, 2, 3)
            },
        ),
        (
            "committed rows one too few",
            Commit {
                committed_rows: 11,
                ..commit(10, 2, 3)
            },
        ),
        ("committed rows of another base", commit(9, 2, 3)),
        (
            "no row kept",
            Commit {
                kept_rows: 0,
                discarded_rows: 3,
                ..commit(10, 1, 3)
            },
        ),
        ("more rows kept than fed", commit(10, 4, 4)),
        (
            "discarded rows that do not add up",
            Commit {
                discarded_rows: 2,
                ..commit(10, 2, 3)
            },
        ),
    ];
    for (what, verdict) in wrong {
        let mut host = host_window();
        commit_window(&mut host, &verdict).expect_err(what);
        assert_failed_closed(&host, what);
        assert!(!host.called("import"), "{what}");
    }
}

#[test]
fn a_state_that_is_not_at_the_end_of_the_window_poisons_on_the_verdict() {
    let mut host = host_window();
    host.tokens = 12;
    commit_window(&mut host, &commit(10, 3, 3)).expect_err("a full accept at the wrong end");
    assert_failed_closed(&host, "tokens != base + rows");
}

#[test]
fn the_window_remembers_the_rows_committed_when_it_began() {
    let mut host = Fake::new(37, false);
    run_window(&mut host, &[1, 2]).expect("run");
    assert_eq!(host.window.as_ref().map(|window| window.base), Some(37));
    commit_window(&mut host, &commit(37, 1, 2)).expect("a verdict for base 37");
}

// Q2, the host run_window.

#[test]
fn a_window_over_a_window_that_awaits_its_verdict_poisons_and_is_refused() {
    for device in [false, true] {
        let mut host = Fake::new(10, device);
        host.pending = true;
        run_window(&mut host, &[1, 2]).expect("first window");
        host.log.clear();
        run_window(&mut host, &[5, 6]).expect_err("a leftover window");
        assert_failed_closed(&host, "leftover window");
        // Refused before everything else: no save finished, no export, no row, no replacement.
        for later in [
            "complete_prefix_save",
            "export",
            "run_row 5",
            "require_armed",
        ] {
            assert!(!host.called(later), "device {device}: {later}");
        }
    }
}

#[test]
fn a_window_with_the_recovery_gate_set_is_refused_without_touching_anything() {
    for device in [false, true] {
        let mut host = Fake::new(10, device);
        host.gate = true;
        run_window(&mut host, &[1, 2]).expect_err("the gate");
        for later in ["export", "run_row 1", "complete_prefix_save", "poison"] {
            assert!(!host.called(later), "device {device}: {later}");
        }
        assert_eq!(host.tokens, 10);
    }
}

#[test]
fn a_host_window_that_does_not_fit_the_cache_is_refused_before_the_export() {
    let mut host = Fake::new(10, false);
    host.capacity = 12;
    let error = run_window(&mut host, &[1, 2, 3]).expect_err("13 rows in 12");
    assert!(matches!(error, EngError::ContextLimit(_)));
    assert!(!host.called("export") && !host.called("poison") && !host.gate);
    assert_eq!(host.tokens, 10);
    // The edge: the last row exactly fills the cache.
    run_window(&mut host, &[1, 2]).expect("12 rows in 12");
}

#[test]
fn the_host_capacity_edge_is_exact() {
    for (capacity, fits) in [(12, true), (11, false), (13, true)] {
        let mut host = Fake::new(10, false);
        host.capacity = capacity;
        assert_eq!(run_window(&mut host, &[1, 2]).is_ok(), fits, "{capacity}");
    }
}

#[test]
fn a_host_row_that_fails_after_the_first_poisons_and_sets_the_gate() {
    // The fake fails the second run_row: the first already ran.
    let mut host = Fake::new(10, false);
    host.pending = true;
    host.fail_nth_row = Some(1);
    run_window(&mut host, &[1, 2, 3]).expect_err("row 1 refused");
    assert_failed_closed(&host, "row 1 refused before launch");
}

#[test]
fn a_host_row_that_fails_before_the_first_ran_leaves_the_state_valid() {
    let mut host = Fake::new(10, false);
    host.fail_nth_row = Some(0);
    run_window(&mut host, &[1, 2, 3]).expect_err("row 0 refused");
    assert!(!host.poisoned && !host.gate && !host.called("poison"));
    assert_eq!(host.tokens, 10);
}

#[test]
fn a_host_export_failure_leaves_the_state_valid() {
    let mut host = Fake::new(10, false);
    host.fail = Some("export");
    run_window(&mut host, &[1, 2]).expect_err("export");
    assert!(!host.poisoned && !host.gate && host.window.is_none());
}

#[test]
fn a_row_that_poisons_the_state_without_moving_it_still_sets_the_gate() {
    let mut host = Fake::new(10, false);
    host.poison_on_row_failure = true;
    host.fail_nth_row = Some(0);
    run_window(&mut host, &[1, 2]).expect_err("row 0 failed and poisoned");
    assert!(host.gate && host.poisoned);
}

// Q2, the consumers of the live state and the restoring entries.

#[test]
fn state_usable_refuses_the_gate_alone_and_an_awaiting_window_alone() {
    let mut host = Fake::new(10, false);
    state_usable(&host).expect("a fresh state is usable");
    host.gate = true;
    assert!(state_usable(&host).is_err(), "the gate alone");
    host.gate = false;
    run_window(&mut host, &[1, 2]).expect("run");
    assert!(state_usable(&host).is_err(), "an awaiting window alone");
    commit_window(&mut host, &commit(10, 2, 2)).expect("commit");
    state_usable(&host).expect("usable again after the verdict");
}

#[test]
fn a_step_through_the_gate_is_refused_and_through_an_awaiting_window_fails_closed() {
    let mut host = Fake::new(10, false);
    step_gate(&mut host).expect("a fresh state");
    host.gate = true;
    step_gate(&mut host).expect_err("the gate");
    assert!(!host.called("poison"), "the gate needs no second poison");

    let mut host = Fake::new(10, false);
    host.pending = true;
    run_window(&mut host, &[1, 2]).expect("run");
    step_gate(&mut host).expect_err("an awaiting window");
    assert_failed_closed(&host, "a step through an awaiting window");
}

#[test]
fn an_export_through_the_gate_or_an_awaiting_window_is_refused_without_a_change() {
    let mut host = Fake::new(10, false);
    export_gate(&host).expect("a fresh state");
    host.gate = true;
    export_gate(&host).expect_err("the gate");
    let mut host = Fake::new(10, false);
    run_window(&mut host, &[1, 2]).expect("run");
    export_gate(&host).expect_err("an awaiting window");
    assert!(
        host.window.is_some() && !host.poisoned,
        "an export only refuses"
    );
}

#[test]
fn every_restoring_entry_closes_an_open_window_and_is_refused_by_the_gate() {
    // The wrappers of the table in the specification all enter through `restore_gate` (the
    // pinned-prefix restore and the `PrefixState` restore) or `import_state` (which calls it).
    type Entry = fn(&mut Fake) -> Result<(), EngError>;
    let entries: [(&str, Entry); 2] = [
        ("restore_gate", restore_gate::<Fake>),
        ("import_state", |host| import_state(host, &7)),
    ];
    for (name, entry) in entries {
        let mut host = Fake::new(10, false);
        run_window(&mut host, &[1, 2]).expect("run");
        entry(&mut host).expect(name);
        assert!(host.window.is_none(), "{name}: the window must be closed");

        let mut host = Fake::new(10, false);
        run_window(&mut host, &[1, 2]).expect("run");
        host.gate = true;
        host.log.clear();
        entry(&mut host).expect_err(name);
        assert!(
            !host.called("close_window") && !host.called("import"),
            "{name}"
        );
        assert!(
            host.window.is_some(),
            "{name}: a refused restore changes nothing"
        );
    }
}

#[test]
fn the_window_is_closed_before_the_first_fallible_call_of_a_restore() {
    // A restore that fails at its first fallible call still closed the window.
    let mut host = Fake::new(10, false);
    run_window(&mut host, &[1, 2]).expect("run");
    host.fail = Some("complete_prefix_save");
    import_state(&mut host, &7).expect_err("the pending save fails");
    assert!(host.window.is_none(), "closed before the pending save ran");
    let closed = host.log.iter().position(|call| call == "close_window");
    let saved = host
        .log
        .iter()
        .position(|call| call == "complete_prefix_save");
    assert!(
        closed < saved,
        "close_window precedes complete_prefix_save: {:?}",
        host.log
    );
}

#[test]
fn an_import_over_a_stale_host_window_drops_it_and_imports_only_the_given_state() {
    // The stale window of an error path before a pageable prefix restore: its saved snapshot
    // (the state at 10 rows) must never be written back.
    let mut host = Fake::new(10, false);
    run_window(&mut host, &[1, 2, 3]).expect("run");
    assert_eq!(host.tokens, 13);
    import_state(&mut host, &40).expect("the import of the prefix state");
    assert!(host.window.is_none(), "the stale window is dropped");
    assert_eq!(host.tokens, 40, "only the given state was imported");
    assert_eq!(host.log.iter().filter(|call| *call == "import").count(), 1);
    // The dropped window cannot commit: there is nothing left to restore from.
    commit_window(&mut host, &commit(10, 1, 3)).expect_err("no window");
    assert_eq!(host.tokens, 40);
}

#[test]
fn an_import_over_a_stale_host_window_with_the_gate_set_is_refused() {
    let mut host = Fake::new(10, false);
    run_window(&mut host, &[1, 2, 3]).expect("run");
    host.gate = true;
    import_state(&mut host, &40).expect_err("the gate");
    assert_eq!(host.tokens, 13, "nothing imported");
    assert!(!host.called("import"));
}

#[test]
fn reset_is_never_wedged_by_an_awaiting_window_with_a_pending_save() {
    let mut host = Fake::new(10, false);
    host.pending = true;
    run_window(&mut host, &[1, 2]).expect("run");
    host.pending = true; // a save staged before the window and still pending
    host.log.clear();
    reset_prologue(&mut host).expect("reset is exempt from the check");
    reset_epilogue(&mut host);
    assert!(!host.pending && host.window.is_none() && !host.gate);
    assert!(
        !host.called("complete_prefix_save"),
        "dropped, not finished"
    );
    assert_eq!(host.invalidations, 1, "the pool is invalidated");
    // The same window runs again afterwards.
    host.tokens = 10;
    run_window(&mut host, &[1, 2]).expect("the same window");
    commit_window(&mut host, &commit(10, 2, 2)).expect("and commits");
}

#[test]
fn reset_is_never_wedged_by_the_gate_of_a_failed_commit() {
    let mut host = host_window();
    host.fail = Some("check_import");
    commit_window(&mut host, &commit(10, 2, 3)).expect_err("failed commit");
    assert!(host.gate && host.poisoned);
    host.pending = true;
    host.log.clear();
    reset_prologue(&mut host).expect("reset is exempt");
    host.poisoned = false; // the state reset zeroes the state
    reset_epilogue(&mut host);
    assert!(!host.gate, "the reset opens the gate");
    assert!(!host.called("complete_prefix_save"));
    host.tokens = 10;
    run_window(&mut host, &[1, 2, 3]).expect("the same window runs again");
    commit_window(&mut host, &commit(10, 2, 3)).expect("and commits");
}

#[test]
fn a_reset_on_a_usable_state_finishes_the_pending_save() {
    let mut host = Fake::new(10, false);
    host.pending = true;
    reset_prologue(&mut host).expect("reset");
    assert!(host.called("complete_prefix_save") && host.invalidations == 0);
}
