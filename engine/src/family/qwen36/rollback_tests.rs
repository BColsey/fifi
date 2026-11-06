use super::*;

/// Bytes of one Delta slot of Qwen3.6-35B-A3B: 30 layers of (2,097,152 recurrent + 98,304
/// convolution) bytes (`state_layout.rs`, pinned by `cuda/tests.rs`).
const SLOT: usize = 30 * (2_097_152 + 98_304);

fn plan(rows: usize) -> RollbackPlan {
    RollbackPlan::new(rows, SLOT).expect("plan")
}

fn facts() -> StateFacts {
    StateFacts {
        owner: true,
        valid: true,
        idle: true,
        resident_kv: true,
    }
}

/// Pins the constant of this file only; the pin against the real layout is native-only
/// (`cuda/tests.rs`, `DeltaPool`).
#[test]
fn the_slot_of_the_production_state_is_65_863_680_bytes() {
    assert_eq!(SLOT, 65_863_680);
}

#[test]
fn a_plan_holds_one_slot_per_row_but_the_last() {
    for (rows, slots, total) in [(2, 1, 65_863_680), (4, 3, 197_591_040), (8, 7, 461_045_760)] {
        let p = plan(rows);
        assert_eq!(p.slots(), slots);
        assert_eq!(p.total_bytes().unwrap(), total);
        assert_eq!(p.offset(slots - 1).unwrap(), (slots - 1) * SLOT);
        assert!(p.offset(slots).is_err(), "the slot after the last");
    }
}

#[test]
fn a_plan_refuses_row_counts_the_window_cannot_have() {
    for rows in [0, 1, 9, 16, 64] {
        assert!(RollbackPlan::new(rows, SLOT).is_err(), "{rows} rows");
    }
    assert!(RollbackPlan::new(4, 0).is_err(), "a slot of zero bytes");
}

#[test]
fn a_plan_whose_buffer_overflows_is_a_memory_plan_refusal() {
    let error = RollbackPlan::new(8, usize::MAX / 4).expect_err("overflow");
    assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
}

#[test]
fn a_capture_needs_an_open_window_and_follows_row_order() {
    let mut book = RollbackBook::new(plan(4));
    assert!(book.begin_capture(0).is_err(), "no window");
    book.begin_window(100, 4).unwrap();
    assert!(book.begin_capture(1).is_err(), "row 1 before row 0");
    assert_eq!(book.begin_capture(0).unwrap(), 0);
    book.note_captured(0).unwrap();
    assert!(book.note_captured(0).is_err(), "row 0 twice");
    book.note_captured(1).unwrap();
    book.note_captured(2).unwrap();
    assert!(
        book.begin_capture(3).is_err(),
        "the last row has no slot (an out-of-range write)"
    );
}

#[test]
fn a_verdict_that_keeps_a_rows_restores_slot_a_minus_one_at_base_plus_a() {
    for kept in 1..4 {
        let mut book = RollbackBook::new(plan(4));
        book.begin_window(100, 4).unwrap();
        for row in 0..3 {
            book.note_captured(row).unwrap();
        }
        let restore = book.finish_window(kept, 100 + kept, 104).unwrap();
        assert_eq!(
            restore,
            Some(Restore {
                slot: kept - 1,
                frontier: 100 + kept
            })
        );
        assert!(!book.is_open());
    }
}

#[test]
fn a_verdict_that_keeps_every_row_restores_nothing() {
    let mut book = RollbackBook::new(plan(4));
    book.begin_window(100, 4).unwrap();
    for row in 0..3 {
        book.note_captured(row).unwrap();
    }
    assert_eq!(book.finish_window(4, 104, 104).unwrap(), None);
}

#[test]
fn a_verdict_that_does_not_fit_the_window_is_refused_and_closes_it() {
    let cases = [
        (0, 100, 104, "keeps no row"),
        (5, 105, 104, "keeps more rows than fed"),
        (2, 103, 104, "commits another row count"),
        (2, 102, 103, "the state is not at the window end"),
    ];
    for (kept, committed, tokens, why) in cases {
        let mut book = RollbackBook::new(plan(4));
        book.begin_window(100, 4).unwrap();
        for row in 0..3 {
            book.note_captured(row).unwrap();
        }
        assert!(
            book.finish_window(kept, committed, tokens).is_err(),
            "{why}"
        );
        assert!(!book.is_open(), "{why}: a refused verdict leaves no window");
    }
}

#[test]
fn a_slot_that_was_never_captured_is_not_restored() {
    let mut book = RollbackBook::new(plan(4));
    book.begin_window(100, 4).unwrap();
    book.note_captured(0).unwrap();
    // The window died after row 1: slot 1 was never written.
    assert!(book.finish_window(2, 102, 104).is_err());
}

#[test]
fn a_new_window_never_inherits_the_slot_marks_of_the_last_one() {
    let mut book = RollbackBook::new(plan(4));
    book.begin_window(100, 4).unwrap();
    for row in 0..3 {
        book.note_captured(row).unwrap();
    }
    assert_eq!(book.finish_window(4, 104, 104).unwrap(), None);
    book.begin_window(104, 4).unwrap();
    assert!(
        book.finish_window(2, 106, 108).is_err(),
        "slot 1 holds the old window's state: it is not captured in this one"
    );
}

#[test]
fn a_window_cannot_open_over_an_open_one_and_discard_closes_it() {
    let mut book = RollbackBook::new(plan(4));
    book.begin_window(100, 4).unwrap();
    assert!(book.begin_window(100, 4).is_err());
    book.discard();
    assert!(!book.is_open());
    book.begin_window(100, 4).unwrap();
    book.discard();
    assert!(
        book.finish_window(1, 101, 104).is_err(),
        "nothing to commit"
    );
}

#[test]
fn a_window_longer_than_the_slots_serve_is_refused() {
    let mut book = RollbackBook::new(plan(4));
    assert!(book.begin_window(0, 5).is_err());
    assert!(book.begin_window(0, 0).is_err());
    // A one-row window has no slot to fill and keeps its row.
    assert!(book.begin_window(0, 1).is_ok());
    assert_eq!(book.finish_window(1, 1, 1).unwrap(), None);
    assert!(book.begin_window(0, 2).is_ok());
}

#[test]
fn the_copy_guard_refuses_a_poisoned_busy_foreign_or_paged_state() {
    assert!(facts().require_copyable("restore").is_ok());
    for (name, bad) in [
        (
            "poisoned",
            StateFacts {
                valid: false,
                ..facts()
            },
        ),
        (
            "pending steps",
            StateFacts {
                idle: false,
                ..facts()
            },
        ),
        (
            "another stream",
            StateFacts {
                owner: false,
                ..facts()
            },
        ),
        (
            "paged KV",
            StateFacts {
                resident_kv: false,
                ..facts()
            },
        ),
    ] {
        let error = bad.require_copyable("restore").expect_err(name);
        assert!(
            format!("{error:?}").contains("restore"),
            "{name}: {error:?}"
        );
    }
}

#[test]
fn slots_that_no_admission_charges_must_fit_the_load_reserve() {
    const MIB: usize = 1 << 20;
    let slots = plan(8).total_bytes().unwrap();
    let tip = SLOT;
    // The only uncharged case today: FP16 KV with chunk prefill off, so the legacy 1 GiB reserve.
    assert!(check_uncharged_slots(slots, tip, 1024 * MIB).is_ok());
    // A 256 MiB reserve (the FP16 fixed-admission figure) cannot cover 8-row slots.
    let error = check_uncharged_slots(slots, tip, 256 * MIB).expect_err("refused");
    assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
    // The edge: exactly the reserve passes, one byte more is refused.
    assert!(check_uncharged_slots(slots, tip, slots + tip).is_ok());
    assert!(check_uncharged_slots(slots, tip, slots + tip - 1).is_err());
    // Fewer rows fit a smaller reserve.
    assert!(check_uncharged_slots(plan(4).total_bytes().unwrap(), 0, 256 * MIB).is_ok());
    assert!(
        check_uncharged_slots(usize::MAX, 1, usize::MAX).is_err(),
        "overflow"
    );
}

#[test]
fn the_startup_line_names_the_cover_and_a_fallback_to_per_token_prefill() {
    let p = plan(8);
    let charged = startup_line(&p, Cover::Arena, Some(true)).unwrap();
    assert!(charged.contains("7 slots of 65863680 B"), "{charged}");
    assert!(
        charged.contains("charged to the prefill arena admission"),
        "{charged}"
    );
    assert!(charged.contains("prefill arena admitted"), "{charged}");
    let reserve = startup_line(&p, Cover::Reserve { headroom: 1 << 30 }, None).unwrap();
    assert!(
        reserve.contains("covered by the 1073741824 B load reserve"),
        "{reserve}"
    );
    assert!(
        reserve.contains("no prefill arena admission ran"),
        "{reserve}"
    );
    let fell_back = startup_line(&p, Cover::Arena, Some(false)).unwrap();
    assert!(fell_back.contains("per-token prefill"), "{fell_back}");
}

#[test]
fn a_window_awaits_its_verdict_only_after_every_row_ran_and_until_it_is_closed() {
    let mut book = RollbackBook::new(plan(4));
    assert!(!book.awaiting_verdict(), "no window");
    assert!(book.mark_complete().is_err(), "nothing to complete");
    book.begin_window(100, 4).unwrap();
    assert!(
        !book.awaiting_verdict(),
        "while its rows run the window is open, not awaiting: the rows are its own steps"
    );
    for row in 0..3 {
        book.note_captured(row).unwrap();
    }
    book.mark_complete().unwrap();
    assert!(book.awaiting_verdict());
    assert!(
        book.begin_window(100, 4).is_err(),
        "no second window over it"
    );
    assert_eq!(book.finish_window(4, 104, 104).unwrap(), None);
    assert!(!book.awaiting_verdict(), "a verdict closes it");
    book.begin_window(104, 2).unwrap();
    book.mark_complete().unwrap();
    book.discard();
    assert!(
        !book.awaiting_verdict(),
        "a reset, an import or a poison closes it"
    );
}

#[test]
fn a_window_that_does_not_fit_the_kv_capacity_is_refused_before_any_row() {
    assert!(
        check_window_fits(100, 4, 104).is_ok(),
        "the last cell is exactly the capacity"
    );
    let error = check_window_fits(100, 4, 103).expect_err("one cell short");
    assert!(matches!(error, EngError::ContextLimit(_)), "{error:?}");
    assert!(check_window_fits(0, 8, 7).is_err());
    assert!(
        check_window_fits(usize::MAX, 2, usize::MAX).is_err(),
        "overflow"
    );
}

#[test]
fn arming_is_refused_for_every_unsupported_state_and_forked_experts_by_name() {
    assert!(check_arming(ArmingFacts::default()).is_ok());
    for (name, facts, word) in [
        (
            "started",
            ArmingFacts {
                started: true,
                ..ArmingFacts::default()
            },
            "already run",
        ),
        (
            "armed",
            ArmingFacts {
                armed: true,
                ..ArmingFacts::default()
            },
            "already armed",
        ),
        (
            "tiered",
            ArmingFacts {
                tiered: true,
                ..ArmingFacts::default()
            },
            "tiered",
        ),
        (
            "sparse",
            ArmingFacts {
                sparse: true,
                ..ArmingFacts::default()
            },
            "sparse",
        ),
        (
            "paged",
            ArmingFacts {
                paged: true,
                ..ArmingFacts::default()
            },
            "paged",
        ),
        (
            "forked experts",
            ArmingFacts {
                forked_experts: true,
                ..ArmingFacts::default()
            },
            "forked experts",
        ),
    ] {
        let error = check_arming(facts).expect_err(name);
        let text = format!("{error:?}");
        assert!(text.contains(word), "{name}: {text}");
        assert!(text.contains("draft_rollback"), "{name}: {text}");
    }
}
