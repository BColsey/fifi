//! The automatic lower rung of a Qwen3.6 GGUF start: when the card cannot carry the whole
//! checkpoint, the start serves the tiered path (experts on the host, the hottest ones in
//! per-layer slots on the card) instead of refusing.
//!
//! The decision comes from the shared ladder (`expert_bank::bank_tier::choose_tier`): this
//! module only turns the start's figures into the ladder's inputs, applies the override rule
//! (a config that names tiered experts, host-paged KV or a device list is taken as given) and
//! words the one start line. The figures in that line are the ones the device admission quotes
//! when it refuses, so the two texts agree.

use super::settings::{Q36Settings, TieredSettings};
use crate::family::expert_bank::bank_tier::{CardBudget, TierBudget, TierChoice, choose_tier};

/// Host bytes the tiered path keeps beside the mapped experts: the CPU pool's scratch and the
/// process heap: the flat headroom the mapped-expert admission of the loader keeps free.
const HOST_OVERHEAD_BYTES: u64 = crate::family::mem::HOST_HEADROOM_BYTES;

/// The figures of one start's card admission, before any weight is uploaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CardFigures {
    /// Device bytes free now.
    pub(crate) free: u64,
    /// Bytes of every tensor of the resident load.
    pub(crate) tensors: u64,
    /// The part of `tensors` that is routed-expert stacks.
    pub(crate) expert_bytes: u64,
    /// The KV cache bytes the load reserves.
    pub(crate) kv_cache: u64,
    /// The work bytes the load reserves.
    pub(crate) work: u64,
    /// The headroom the resident admission keeps free.
    pub(crate) headroom: u64,
    /// The headroom the card-tiered load's bank admission keeps free (it runs with the fixed
    /// admission off, so it keeps the legacy headroom, not the resident one). The slots the
    /// ladder promises must fit it, or the bank admission refuses the plan the ladder made.
    pub(crate) tiered_headroom: u64,
    /// Device bytes of one expert slot in one layer.
    pub(crate) slot_bytes: u64,
    /// The bytes the weights must leave free once they are loaded: what the load takes beyond
    /// the planned tensors plus the free-after-load floor the start refuses below. The cache,
    /// the work and the headroom come after that check, so it bounds the weights alone.
    pub(crate) after_load_floor: u64,
    /// Cache bytes the banks can use after leaving room for the initial allocation and growth.
    pub(crate) kv_credit: u64,
    /// Device bytes of one session's part (KV planes, DeltaNet and convolution state) at its
    /// initial cells: what each session after the first keeps on the card beside the banks.
    pub(crate) session_part_bytes: u64,
    /// The sessions the start would like to serve beside the banks: the most the planner funds.
    pub(crate) sessions: usize,
    /// Whether the config named the count (`max_seqs`): it is then funded whole or refused by
    /// name, never lowered.
    pub(crate) sessions_named: bool,
    /// Layers of the model.
    pub(crate) layers: usize,
    /// Experts per layer.
    pub(crate) experts: usize,
    /// Experts one token picks per layer.
    pub(crate) top_k: usize,
}

impl CardFigures {
    /// The bytes the device admission compares to the free bytes.
    ///
    /// The larger of the load's own sum (tensors, cache, work, headroom) and the weights plus
    /// the floor the start checks right after the load: a card that holds the first but not the
    /// second is short, and the lower rung is taken before any weight is uploaded rather than a
    /// refusal after it.
    pub(crate) fn needed(&self) -> Option<u64> {
        self.needed_with_credit(0)
    }

    fn needed_with_credit(&self, credit: u64) -> Option<u64> {
        let planned = [
            self.kv_cache.saturating_sub(credit),
            self.work,
            self.headroom,
        ]
        .into_iter()
        .try_fold(self.tensors, u64::checked_add)?;
        let after_load = self.tensors.checked_add(self.after_load_floor)?;
        Some(planned.max(after_load))
    }
}

/// What a card-short start serves: the shortfall and the ladder's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CardShort {
    /// Bytes the resident load is short of the card.
    pub(crate) short_by: u64,
    /// Bytes the resident load needs (tensors, KV cache, work, headroom).
    pub(crate) needed: u64,
    /// Device bytes free when the start asked.
    pub(crate) free: u64,
    /// Bytes of routed experts the host serves.
    pub(crate) host_bytes: u64,
    /// Whether the host cannot hold those experts, so they are read from the file on a miss.
    pub(crate) from_disk: bool,
    /// The sessions the card funds beside the banks, at most the figures' target.
    pub(crate) sessions: usize,
    /// The slots per layer one session alone would have kept: what the sessions are charged
    /// against, so the line reports what the banks admitted gave up.
    pub(crate) slots_one: usize,
    /// The most slots per layer the card carries beside the rest of the load, with the tiered
    /// load's headroom kept free (an upper bound the bank sizing may lower).
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the tests read the ladder's promise")
    )]
    pub(crate) slots: usize,
}

impl CardShort {
    /// The tail of the start line that says what the sessions cost the banks.
    fn sessions_note(&self, admitted_slots: usize) -> String {
        if self.sessions <= 1 {
            return String::new();
        }
        format!(
            ", {} sessions funded beside the banks ({} slots/layer given up for them)",
            self.sessions,
            self.slots_one.saturating_sub(admitted_slots)
        )
    }

    /// The one start line, printed once the banks exist: `slots` is the count the banks hold
    /// and `context` the context limit the start serves.
    pub(crate) fn line(&self, slots: usize, context: usize) -> String {
        let experts = if self.from_disk {
            format!(
                "experts streamed from disk through the page cache (the host cannot hold their \
                 {} bytes)",
                self.host_bytes
            )
        } else {
            format!("experts on host ({} bytes)", self.host_bytes)
        };
        format!(
            "q36 cuda: card short by {} bytes (needs {} tensors+KV+headroom, has {} free): \
             serving tiered, {slots} slots/layer on card, {experts}, context {context}{}",
            self.short_by,
            self.needed,
            self.free,
            self.sessions_note(slots)
        )
    }
}

/// The ladder's verdict on one start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CardVerdict {
    /// The resident load fits the card: nothing changes.
    Fits,
    /// The card is short and card plus host carry the tiered path.
    Short(CardShort),
    /// The card is short and the tiered path cannot be carried either: the start keeps its
    /// device refusal, and `line` says why the lower rung was not taken.
    Refused { line: String },
}

/// Asks the ladder about one start. `host_available` is the host budget the start probed
/// (MemAvailable clamped by the enclosing cgroup).
pub(crate) fn decide(figures: &CardFigures, host_available: u64) -> CardVerdict {
    let Some(needed) = figures.needed() else {
        return CardVerdict::Fits;
    };
    let ladder = |sessions: usize| {
        let reserve =
            tiered_session_reserve(figures.session_part_bytes, sessions).unwrap_or(u64::MAX);
        choose_tier(&TierBudget {
            record_bytes: 0,
            layers: figures.layers,
            experts: figures.experts,
            slots: 0,
            max_spare_places: 0,
            min_spare_places: 0,
            overhead_bytes: HOST_OVERHEAD_BYTES,
            cache_floor: 0,
            ring_bytes: 0,
            available: host_available,
            card: Some(CardBudget {
                free: figures.free,
                demand: needed,
                expert_bytes: figures.expert_bytes,
                slot_bytes: figures.slot_bytes,
                min_slots: figures.top_k,
                carry_reserve: figures
                    .tiered_headroom
                    .saturating_sub(figures.headroom)
                    .saturating_add(reserve),
                kv_credit: needed.saturating_sub(
                    figures
                        .needed_with_credit(figures.kv_credit)
                        .unwrap_or(needed),
                ),
            }),
        })
    };
    let carried = |choice: &Result<TierChoice, crate::EngError>| match choice {
        Ok(TierChoice::CardTiered { slots, .. } | TierChoice::CardDisk { slots }) => Some(*slots),
        _ => None,
    };
    // One session first: what the card carries with its idle part and nothing else is the
    // baseline the sessions are funded against, and the refusal when even that is short.
    let one = ladder(1);
    let slots_one = carried(&one).unwrap_or(0);
    let slots_at = |sessions: usize| carried(&ladder(sessions));
    let funded = crate::family::card_fit::funded_sessions(
        figures.sessions.max(1),
        figures.sessions_named,
        &slots_at,
    );
    if figures.sessions_named && figures.sessions > 1 && funded.is_none() && slots_one > 0 {
        return CardVerdict::Refused {
            line: format!(
                "q36 cuda: card short by {} bytes (needs {needed} tensors+KV+headroom, has {} \
                 free): the card funds one session beside the minimum banks, not the {} that \
                 max_seqs names",
                needed.saturating_sub(figures.free),
                figures.free,
                figures.sessions
            ),
        };
    }
    let sessions = funded.unwrap_or(1);
    let chosen = if sessions > 1 { ladder(sessions) } else { one };
    let short = |host_bytes: u64, from_disk: bool, slots: usize| {
        CardVerdict::Short(CardShort {
            short_by: needed.saturating_sub(figures.free),
            needed,
            free: figures.free,
            host_bytes,
            from_disk,
            sessions,
            slots_one,
            slots,
        })
    };
    match chosen {
        Ok(TierChoice::CardTiered { host_bytes, slots }) => short(host_bytes, false, slots),
        Ok(TierChoice::CardDisk { slots }) => short(figures.expert_bytes, true, slots),
        Ok(_) => CardVerdict::Fits,
        Err(refusal) => CardVerdict::Refused {
            line: format!(
                "q36 cuda: card short by {} bytes (needs {needed} tensors+KV+headroom, has {} \
                 free): the tiered path is not available: {}",
                needed.saturating_sub(figures.free),
                figures.free,
                refusal.describe()
            ),
        },
    }
}

/// The idle parts the sessions executor builds beside its sessions' own: the part the step runs
/// while no session does.
const TIERED_IDLE_PARTS: usize = 1;

/// The device bytes a tiered start keeps beside its banks for `sessions` sessions: the planner
/// sizes the slots against it and the bank attach leaves exactly it, so the two cannot differ.
pub(crate) fn tiered_session_reserve(part_bytes: u64, sessions: usize) -> Option<u64> {
    crate::family::card_fit::session_reserve_bytes(part_bytes, sessions, TIERED_IDLE_PARTS)
}

/// The sessions a start with no named count asks the planner to fund beside the tiered banks:
/// the count the other arms derive for themselves. A target, not a ceiling: the planner takes
/// what the card funds up to it.
pub(crate) const TIERED_SESSION_TARGET: usize = super::settings::AUTO_MAX_SEQS;

/// Whether the config leaves the card-short choice to the engine. A config that names tiered
/// experts, host-paged KV or more than one device is taken as given and never re-tiered, as is
/// a start that asks for the device-first KV; only the single-device GGUF start with none of
/// them is automatic. The context limit is never changed, so a context the config names is
/// served as named.
pub(crate) fn automatic(settings: &Q36Settings, device_first: bool) -> bool {
    settings.cuda_devices == [0]
        && settings.tiered.is_none()
        && settings.kv_offload_host_bytes.is_none()
        && !device_first
}

/// The settings of the tiered start the lower rung serves: the `"tiered": {"slots": "auto"}`
/// member with its defaults and graph capture off (the tiered step is eager only). Everything
/// else is the caller's.
pub(crate) fn tiered_variant(settings: &Q36Settings) -> Q36Settings {
    Q36Settings {
        tiered: Some(TieredSettings::AUTO),
        capture: false,
        ..settings.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen::qwen36::settings::{Backend, Q36Env, Q36Options};
    use std::path::Path;

    const GIB: u64 = 1 << 30;

    /// The figures of the 3090 measurement: the Q4_K_M checkpoint on a card with 5,836 MiB
    /// free.
    fn figures(free: u64) -> CardFigures {
        CardFigures {
            free,
            tensors: 20_408_576_512,
            expert_bytes: 18_900_000_000,
            kv_cache: 671_088_640,
            work: 0,
            headroom: GIB,
            tiered_headroom: GIB,
            slot_bytes: 1_769_472,
            after_load_floor: 0,
            kv_credit: 0,
            session_part_bytes: 0,
            sessions: 1,
            sessions_named: false,
            layers: 40,
            experts: 256,
            top_k: 8,
        }
    }

    fn settings(body: &str) -> Q36Settings {
        let options: Q36Options = serde_json::from_str(body).expect("a valid qwen36 object");
        Q36Settings::new(
            Path::new("/m"),
            Path::new("/t"),
            Some(&options),
            &Q36Env::default(),
            Backend::Cuda,
        )
        .expect("valid settings")
    }

    #[test]
    fn the_needed_bytes_are_the_device_admissions_sum() {
        assert_eq!(figures(0).needed(), Some(22_153_406_976));
    }

    #[test]
    fn a_card_that_carries_the_load_changes_nothing() {
        for free in [22_153_406_976, 22_153_406_977, 24 * GIB] {
            assert_eq!(
                decide(&figures(free), 64 * GIB),
                CardVerdict::Fits,
                "{free}"
            );
        }
    }

    #[test]
    fn a_short_card_with_room_on_the_host_serves_tiered() {
        let free = 5_805_441_024;
        match decide(&figures(free), 64 * GIB) {
            CardVerdict::Short(short) => {
                assert_eq!(short.needed, 22_153_406_976);
                assert_eq!(short.free, free);
                assert_eq!(short.short_by, 22_153_406_976 - free);
                assert_eq!(short.host_bytes, 18_900_000_000);
                let line = short.line(40, 32_768);
                assert!(
                    line.starts_with(
                        "q36 cuda: card short by 16347965952 bytes (needs 22153406976 \
                         tensors+KV+headroom, has 5805441024 free): serving tiered, 40 \
                         slots/layer on card, experts on host"
                    ),
                    "{line}"
                );
                assert!(line.ends_with("context 32768"), "{line}");
            }
            other => panic!("expected the lower rung, got {other:?}"),
        }
    }

    #[test]
    fn a_card_too_small_for_even_the_minimum_banks_keeps_its_refusal() {
        // The non-expert load is 22,153,406,976 - 18,900,000,000 = 3,253,406,976; the
        // minimum banks are 40 layers x 8 slots x 1,769,472 = 566,231,040.
        let enough = 3_253_406_976 + 566_231_040;
        assert!(matches!(
            decide(&figures(enough), 64 * GIB),
            CardVerdict::Short(_)
        ));
        match decide(&figures(enough - 1), 64 * GIB) {
            CardVerdict::Refused { line } => {
                assert!(line.contains("the tiered path is not available"), "{line}");
                assert!(line.contains("needs 22153406976"), "{line}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_host_without_room_for_the_experts_streams_them_from_disk() {
        match decide(&figures(5_805_441_024), 10 * GIB) {
            CardVerdict::Short(short) => {
                assert!(short.from_disk);
                assert_eq!(short.host_bytes, 18_900_000_000);
                let line = short.line(40, 32_768);
                assert!(line.contains("streamed from disk"), "{line}");
                assert!(line.contains("18900000000 bytes"), "{line}");
            }
            other => panic!("expected the disk rung, got {other:?}"),
        }
    }

    #[test]
    fn a_host_without_room_for_its_own_overhead_names_both_figures() {
        match decide(&figures(5_805_441_024), HOST_OVERHEAD_BYTES - 1) {
            CardVerdict::Refused { line } => {
                assert!(line.contains("card short by 16347965952 bytes"), "{line}");
                assert!(line.contains("1073741823"), "{line}");
                assert!(line.contains("18900000000 expert bytes"), "{line}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert!(matches!(
            decide(&figures(5_805_441_024), HOST_OVERHEAD_BYTES),
            CardVerdict::Short(_)
        ));
    }

    /// A card that holds the load's own sum but not the weights plus the floor the start checks
    /// right after the load is short: the lower rung is taken before any upload, never a refusal
    /// after it. The measured start (Q4_K_M, 21.6 GB free) planned a 9,216-cell int8 cache and
    /// the load then refused with 909,967,360 bytes free against the 1 GiB floor.
    #[test]
    fn a_card_that_leaves_the_weights_below_the_floor_is_short_at_any_context() {
        let floor = crate::family::mem::DEVICE.load_unplanned + GIB;
        let at = |free: u64| CardFigures {
            after_load_floor: floor,
            kv_cache: 9_216 * 24_576,
            headroom: 256 << 20,
            ..figures(free)
        };
        let weights = 20_408_576_512u64;
        // The floor binds: the weights, the load's unplanned bytes and the floor, not the sum.
        assert_eq!(at(0).needed(), Some(weights + floor));
        // Exactly enough carries the whole load; one byte under is short.
        assert_eq!(decide(&at(weights + floor), 64 * GIB), CardVerdict::Fits);
        for free in [21_600_468_992, 21_300_000_000] {
            match decide(&at(free), 64 * GIB) {
                CardVerdict::Short(short) => {
                    assert_eq!(short.needed, weights + floor, "{free}");
                    assert_eq!(short.short_by, weights + floor - free, "{free}");
                    assert!(!short.from_disk, "{free}");
                }
                other => panic!("{free}: expected the lower rung, got {other:?}"),
            }
        }
        // Without the floor term the same card would have been called a fit.
        assert_eq!(
            decide(
                &CardFigures {
                    after_load_floor: 0,
                    ..at(21_600_468_992)
                },
                64 * GIB
            ),
            CardVerdict::Fits
        );
    }

    /// The sessions the card funds beside the banks: each takes its part's bytes out of the slots,
    /// the planner stops where the banks would lose more than a quarter, and the count never
    /// passes the target. Q4_K_M on a card with 21.3 GB free, 95 MiB of state per session.
    #[test]
    fn the_sessions_are_funded_from_the_slots_up_to_a_quarter_of_them() {
        let part = 95u64 << 20;
        let at = |sessions: usize| CardFigures {
            session_part_bytes: part,
            sessions,
            ..figures(21_300_000_000)
        };
        let mut seen = Vec::new();
        for target in 1..=8 {
            match decide(&at(target), 64 * GIB) {
                CardVerdict::Short(short) => {
                    seen.push((target, short.sessions, short.slots, short.slots_one));
                }
                other => panic!("target {target}: {other:?}"),
            }
        }
        let one = seen[0].2;
        for &(target, sessions, slots, slots_one) in &seen {
            assert!((1..=target).contains(&sessions), "{seen:?}");
            assert_eq!(
                slots_one, one,
                "the baseline is the one-session slots: {seen:?}"
            );
            assert!(slots * 4 >= one * 3, "a quarter at most: {seen:?}");
        }
        // The slots only shrink as sessions are funded, and one session gives up nothing.
        assert!(
            seen.windows(2).all(|pair| pair[1].2 <= pair[0].2),
            "{seen:?}"
        );
        // A target of four funds four here: each session costs far less than a quarter.
        assert_eq!(seen[3].1, 4, "{seen:?}");
        // The line reports what the banks admitted gave up, not what was planned.
        let line = match decide(&at(4), 64 * GIB) {
            CardVerdict::Short(short) => short.line(short.slots - 1, 32_768),
            other => panic!("{other:?}"),
        };
        let given_up = one - (seen[3].2 - 1);
        assert!(
            line.contains(&format!(
                "4 sessions funded beside the banks ({given_up} slots/layer given up for them)"
            )),
            "{line}"
        );
    }

    /// The planner and the bank attach leave the same reserve for the same inputs: one function.
    #[test]
    fn the_planner_reserve_is_the_attach_reserve() {
        let part = 95u64 << 20;
        for sessions in 1..=8 {
            let reserve = tiered_session_reserve(part, sessions).expect("fits u64");
            // The idle part is charged even for one session; the slack from the second on.
            let parts = sessions as u64;
            let slack = if sessions > 1 {
                crate::family::card_fit::SESSION_PLAN_SLACK_BYTES
            } else {
                0
            };
            assert_eq!(reserve, part * parts + slack, "{sessions} sessions");
        }
        // The slots the ladder gives up are the reserve spread over the layers' slot bytes.
        let at = |sessions: usize| CardFigures {
            session_part_bytes: part,
            sessions,
            sessions_named: true,
            ..figures(21_300_000_000)
        };
        let slots_of = |sessions: usize| match decide(&at(sessions), 64 * GIB) {
            CardVerdict::Short(short) => short.slots,
            other => panic!("{other:?}"),
        };
        let per_slot = 1_769_472u64 * 40;
        let extra =
            tiered_session_reserve(part, 4).unwrap() - tiered_session_reserve(part, 1).unwrap();
        let lost = slots_of(1) - slots_of(4);
        assert!(
            (lost as u64) * per_slot <= extra + per_slot && (lost as u64 + 1) * per_slot > extra,
            "{lost} slots lost for {extra} bytes"
        );
    }

    /// A card that carries only the minimum banks funds one session: the planner never takes the
    /// banks below a quarter, and never refuses for the sessions it could not fund.
    #[test]
    fn a_card_with_room_for_the_minimum_banks_funds_one_session() {
        let enough = 3_253_406_976 + 566_231_040 + (95 << 20);
        let tight = CardFigures {
            session_part_bytes: 95 << 20,
            sessions: 4,
            ..figures(enough)
        };
        match decide(&tight, 64 * GIB) {
            CardVerdict::Short(short) => {
                assert_eq!(short.sessions, 1);
                assert_eq!(short.slots, short.slots_one);
                assert!(!short.line(8, 32_768).contains("sessions funded"));
            }
            other => panic!("expected the lower rung, got {other:?}"),
        }
    }

    /// A named count is funded whole or refused by name: a card with room for the minimum banks
    /// and an idle part, but not for three sessions beside them, refuses `max_seqs` of 3.
    #[test]
    fn a_named_session_count_the_card_cannot_fund_is_refused_by_name() {
        let enough = 3_253_406_976 + 566_231_040 + (95 << 20);
        let named = CardFigures {
            session_part_bytes: 95 << 20,
            sessions: 3,
            sessions_named: true,
            ..figures(enough)
        };
        match decide(&named, 64 * GIB) {
            CardVerdict::Refused { line } => {
                assert!(line.contains("funds one session"), "{line}");
                assert!(line.contains("max_seqs names"), "{line}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        let roomy = CardFigures {
            free: 21_300_000_000,
            ..named
        };
        match decide(&roomy, 64 * GIB) {
            CardVerdict::Short(short) => assert_eq!(short.sessions, 3),
            other => panic!("{other:?}"),
        }
    }

    /// The cache the start builds at its ceiling and gives back before it serves belongs to the
    /// banks: the slots rise by the credit spread over the layers' slot bytes.
    #[test]
    fn the_cache_the_executor_gives_back_is_taken_by_the_banks() {
        let credit = 584_000_000u64;
        let without = |free: u64| match decide(&figures(free), 64 * GIB) {
            CardVerdict::Short(short) => short.slots,
            other => panic!("{other:?}"),
        };
        let with = |free: u64| match decide(
            &CardFigures {
                kv_credit: credit,
                ..figures(free)
            },
            64 * GIB,
        ) {
            CardVerdict::Short(short) => short.slots,
            other => panic!("{other:?}"),
        };
        let per_slot = 1_769_472u64 * 40;
        let gained = (with(19_000_000_000) - without(19_000_000_000)) as u64;
        assert!(
            gained * per_slot <= credit + per_slot && (gained + 1) * per_slot > credit,
            "{gained} slots for {credit} bytes"
        );
        assert!(gained >= 8, "{gained}");
        let floor = CardFigures {
            after_load_floor: 8 * GIB,
            ..figures(0)
        };
        assert_eq!(floor.needed_with_credit(credit), floor.needed());
        // The card is still short by the whole cache: the credit moves slots, not the verdict.
        assert!(matches!(
            decide(
                &CardFigures {
                    kv_credit: credit,
                    ..figures(5_805_441_024)
                },
                64 * GIB
            ),
            CardVerdict::Short(_)
        ));
    }

    /// The idle part is a planner charge: a card that holds the minimum banks but not the idle
    /// part beside them is refused by the planner, not by a start that dies late.
    #[test]
    fn the_idle_part_is_charged_to_the_planner_even_for_one_session() {
        let minimum = 3_253_406_976 + 566_231_040;
        let one = |free: u64| CardFigures {
            session_part_bytes: 95 << 20,
            sessions: 1,
            ..figures(free)
        };
        assert!(matches!(
            decide(&one(minimum + (95 << 20)), 64 * GIB),
            CardVerdict::Short(_)
        ));
        assert!(matches!(
            decide(&one(minimum + (95 << 20) - 1), 64 * GIB),
            CardVerdict::Refused { .. }
        ));
    }

    #[test]
    fn explicit_tiered_host_paged_kv_and_device_lists_are_never_re_tiered() {
        assert!(automatic(&settings("{}"), false));
        assert!(
            !automatic(&settings("{}"), true),
            "device-first KV is named"
        );
        assert!(!automatic(
            &settings(r#"{"capture":false,"tiered":{"slots":40}}"#),
            false
        ));
        assert!(!automatic(
            &settings(r#"{"capture":false,"tiered":{"slots":"auto"}}"#),
            false
        ));
        assert!(!automatic(
            &settings(r#"{"capture":false,"kv_dtype":"q8","kv_offload_host_bytes":4294967296}"#),
            false
        ));
    }

    #[test]
    fn the_context_limit_is_served_as_named() {
        let named = settings(r#"{"context_limit":65536}"#);
        let tiered = tiered_variant(&named);
        assert_eq!(tiered.context_limit, 65_536);
        assert_eq!(tiered.context_limit, named.context_limit);
    }

    #[test]
    fn the_tiered_variant_is_the_auto_member_with_capture_off() {
        let base = settings("{}");
        assert!(base.capture, "the CUDA arm captures by default");
        let tiered = tiered_variant(&base);
        assert_eq!(tiered.tiered, Some(TieredSettings::AUTO));
        assert!(!tiered.capture);
        // The same member the explicit config resolves to.
        let explicit = settings(r#"{"capture":false,"tiered":{"slots":"auto"}}"#);
        assert_eq!(tiered.tiered, explicit.tiered);
        let unchanged = Q36Settings {
            tiered: None,
            capture: true,
            ..tiered
        };
        assert_eq!(unchanged, base, "nothing else moves");
    }

    /// The bytes the carried slots and the tiered load's own non-expert bytes take, headroom
    /// included: what `admit_banks` compares to the free bytes.
    fn tiered_need(figures: &CardFigures, slots: usize, tiered_headroom: u64) -> u64 {
        let non_expert = figures.tensors - figures.expert_bytes + figures.kv_cache + figures.work;
        non_expert + tiered_headroom + slots as u64 * figures.slot_bytes * figures.layers as u64
    }

    /// The slots a card-tiered load needs fit its own admission: every slot the ladder carries
    /// leaves the tiered headroom (1 GiB) free. With the resident headroom (256 MiB) in the
    /// ladder, up to 768 MiB of slots were promised and then refused by the bank admission.
    #[test]
    fn the_ladder_promises_only_slots_the_tiered_admission_keeps_its_headroom_for() {
        let mut over_promised = 0;
        for free in (6u64..=12).flat_map(|gib| (0..8).map(move |step| gib * GIB + step * (GIB / 8)))
        {
            let resident = CardFigures {
                headroom: 256 << 20,
                tiered_headroom: 256 << 20,
                ..figures(free)
            };
            // The ladder as it was: told only the resident headroom.
            if let CardVerdict::Short(short) = decide(&resident, 64 * GIB) {
                over_promised += usize::from(tiered_need(&resident, short.slots, GIB) > free);
            }
            // The ladder now: told the headroom the tiered admission keeps.
            let told = CardFigures {
                tiered_headroom: GIB,
                ..resident
            };
            if let CardVerdict::Short(short) = decide(&told, 64 * GIB) {
                assert!(tiered_need(&told, short.slots, GIB) <= free, "{free}");
            }
        }
        assert!(
            over_promised > 0,
            "the old inputs promised slots the admission refuses"
        );
    }
}
