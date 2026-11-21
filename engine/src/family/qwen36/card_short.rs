//! The automatic lower rung of a Qwen3.6 GGUF start: when the card cannot carry the whole
//! checkpoint, the start serves the tiered path (experts on the host, the hottest ones in
//! per-layer slots on the card) instead of refusing.
//!
//! The decision comes from the shared ladder (`expert_bank::bank_tier::choose_tier`): this
//! module only turns the start's figures into the ladder's inputs, applies the override rule
//! (a config that names tiered experts, host-paged KV or a device list is taken as given) and
//! words the one start line. The figures in that line are the ones the device admission quotes
//! when it refuses, so the two texts agree.

use super::settings::{Q36Settings, TieredSettings, WeightFormat};
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
    /// Layers of the model.
    pub(crate) layers: usize,
    /// Experts per layer.
    pub(crate) experts: usize,
    /// Experts one token picks per layer.
    pub(crate) top_k: usize,
}

impl CardFigures {
    /// The bytes the device admission compares to the free bytes.
    pub(crate) fn needed(&self) -> Option<u64> {
        [self.kv_cache, self.work, self.headroom]
            .into_iter()
            .try_fold(self.tensors, u64::checked_add)
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
    /// The most slots per layer the card carries beside the rest of the load, with the tiered
    /// load's headroom kept free (an upper bound the bank sizing may lower).
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the ladder's promise; the tests check it against the bank admission"
        )
    )]
    pub(crate) slots: usize,
}

impl CardShort {
    /// The one start line, printed once the banks exist: `slots` is the count the banks hold
    /// and `context` the context limit the start serves.
    pub(crate) fn line(&self, slots: usize, context: usize) -> String {
        format!(
            "q36 cuda: card short by {} bytes (needs {} tensors+KV+headroom, has {} free): \
             serving tiered, {slots} slots/layer on card, experts on host ({} bytes), context \
             {context}",
            self.short_by, self.needed, self.free, self.host_bytes
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
    let budget = TierBudget {
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
            carry_reserve: figures.tiered_headroom.saturating_sub(figures.headroom),
        }),
    };
    match choose_tier(&budget) {
        Ok(TierChoice::CardTiered { host_bytes, slots }) => CardVerdict::Short(CardShort {
            short_by: needed.saturating_sub(figures.free),
            needed,
            free: figures.free,
            host_bytes,
            slots,
        }),
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

/// Whether the config leaves the card-short choice to the engine. A config that names tiered
/// experts, host-paged KV or more than one device is taken as given and never re-tiered, as is
/// a start that asks for the device-first KV; only the single-device GGUF start with none of
/// them is automatic. The context limit is never changed, so a context the config names is
/// served as named.
pub(crate) fn automatic(settings: &Q36Settings, device_first: bool) -> bool {
    settings.weights == WeightFormat::Gguf
        && settings.cuda_devices == [0]
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
    use crate::family::qwen36::settings::{Backend, Q36Env, Q36Options};
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
    fn a_host_without_room_for_the_experts_names_both_figures() {
        match decide(&figures(5_805_441_024), 10 * GIB) {
            CardVerdict::Refused { line } => {
                assert!(line.contains("card short by 16347965952 bytes"), "{line}");
                assert!(line.contains("10737418240"), "{line}");
                assert!(line.contains("18900000000 expert bytes"), "{line}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
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
        assert!(!automatic(&settings(r#"{"weight_format":"exl3"}"#), false));
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
