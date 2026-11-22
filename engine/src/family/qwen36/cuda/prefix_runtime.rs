//! The step's deferred prefix lifecycle. This is a child of step, so device owners stay private.
use super::super::{
    prefix::{CudaSavedPrefix, HOST_HEADROOM, PendingPrefix, entry_bytes},
    state::{
        device_tip::DeviceTip,
        pinned::{MAX_RETAINED, PinnedSnapshot, PrefixPinPool},
    },
};
use super::QwenResidentStep;
use crate::family::kv_disk::DiskOwner;
use crate::family::{executor::ExecutorOps, graph_state::book::Health};
use crate::{
    EngError,
    family::mem::HostFacts,
    family::qwen36::window_logic,
    family::session::{
        DecodeSession, PrefixIdentity, PrefixSlot, SaveKind, StoreLimits, shadow_block_lcp_enabled,
    },
};
use std::time::Instant;

const TIERED_EXTRA_BYTES_ENV: &str = "Q36_TIERED_PREFIX_EXTRA_BYTES";
const PIN_BYTES_ENV: &str = "Q36_CUDA_PINNED_PREFIX_BYTES";

fn byte_limit(name: &str, value: Option<&std::ffi::OsStr>) -> Result<Option<usize>, EngError> {
    let Some(value) = value else { return Ok(None) };
    let bytes = value
        .to_str()
        .and_then(|value| value.trim().parse().ok())
        .ok_or_else(|| EngError::BadRequest(format!("{name} must be an unsigned byte count")))?;
    Ok(Some(bytes))
}

// Production reads the admission budget instead; this parser stays as the pinned-prefix
// test seam for the malformed-figure refusals below.
#[cfg(test)]
fn mem_available(contents: &str) -> Option<usize> {
    let value = contents
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?;
    let mut fields = value.split_whitespace();
    let kib = fields.next()?.parse::<usize>().ok()?;
    if fields.next()? != "kB" || fields.next().is_some() {
        return None;
    }
    kib.checked_mul(1024)
}

/// The default pinned-pool byte cap: every retained entry up to `MAX_RETAINED` plus the staging
/// slot, and never more than half of the admission budget, which stays for the weights' host
/// tier and the pageable paths. No budget (a failed probe) plans no pool.
fn default_pin_bytes(checkpoint: usize, available: Option<usize>) -> usize {
    let full = checkpoint.saturating_mul(MAX_RETAINED + 1);
    full.min(available.unwrap_or(0) / 2)
}

/// The retained entries (`1..=MAX_RETAINED`) that `limit` bytes and the host budget can hold
/// beside one staging slot and `HOST_HEADROOM`, or 0 when not even one fits.
fn planned_entries(checkpoint: usize, limit: usize, available: Option<usize>) -> usize {
    if checkpoint == 0 {
        return 0;
    }
    let retained = (limit / checkpoint).saturating_sub(1).min(MAX_RETAINED);
    if retained == 0 {
        return 0;
    }
    for retained in (1..=retained).rev() {
        let required = checkpoint
            .checked_mul(retained + 1)
            .and_then(|bytes| bytes.checked_add(HOST_HEADROOM));
        if required.is_some_and(|bytes| available.is_some_and(|free| free >= bytes)) {
            return retained;
        }
    }
    0
}

/// The admission probe's available budget in bytes, shared by the pinned-prefix and
/// tiered-prefix planners. A failed probe logs `refused` (each planner states its own
/// fallback) and reads as no budget.
fn admission_budget(probe: &'static str, refused: &str) -> Option<usize> {
    HostFacts::read(probe, "Qwen36")
        .map(|admission| usize::try_from(admission.budget()).ok())
        .inspect_err(|error| {
            eprintln!("q36 cuda: {probe} host probe failed ({error:?}); {refused}")
        })
        .ok()
        .flatten()
}

/// The budget must cover the additional pageable snapshot payload and its token key at full
/// context, and host memory must fit both entries plus headroom because neither has been
/// allocated at startup. Returns the refusal cause for the one-entry log, or `None` when the
/// second entry fits.
fn second_tiered_entry_refusal(
    payload: usize,
    capacity: usize,
    extra_budget: usize,
    available: Option<usize>,
) -> Option<&'static str> {
    if payload == 0 {
        return Some("the checkpoint payload is zero");
    }
    let Some(key) = capacity
        .checked_add(1)
        .and_then(|n| n.checked_mul(size_of::<u32>()))
    else {
        return Some("the full-context token key overflows");
    };
    let Some(entry) = payload.checked_add(key) else {
        return Some("the entry size overflows");
    };
    if entry > extra_budget {
        return Some("the extra budget is below the entry size");
    }
    match entry
        .checked_mul(2)
        .and_then(|both| both.checked_add(HOST_HEADROOM))
    {
        Some(required) if available.is_some_and(|free| free >= required) => None,
        _ => Some("available host memory is below two entries plus headroom"),
    }
}

impl QwenResidentStep<'_> {
    /// Names the weights, tokenizer and template this load serves. Without all three nothing is
    /// written to disk and nothing is read from it.
    pub(crate) fn set_prefix_identity(&mut self, identity: crate::family::session::PrefixIdentity) {
        self.prefix_identity = identity;
    }

    pub(crate) fn prefix_identity(&self) -> crate::family::session::PrefixIdentity {
        self.prefix_identity.clone()
    }

    /// Import refills the report; freeze lookup facts afterward, before later saves replace it.
    pub(crate) fn record_prefix_lookup(&mut self, prompt: &[u32], covered: usize) {
        self.prefill_report
            .observe_prefix_lookup(&self.prefix, prompt, covered);
        self.prefill_report.prefix_lookup.tier = self.restore_tier(covered);
    }

    /// Called once after tiered banks are attached and serving has enabled prefix reuse.
    pub(crate) fn configure_tiered_prefix_store(&mut self) -> Result<(), EngError> {
        let Some(raw) = std::env::var_os(TIERED_EXTRA_BYTES_ENV) else {
            return Ok(());
        };
        if self.prefix_limits().entries == 0 {
            eprintln!(
                "q36 cuda: {TIERED_EXTRA_BYTES_ENV} is set but ignored: tiered prefix reuse is off"
            );
            return Ok(());
        }
        let budget = byte_limit(TIERED_EXTRA_BYTES_ENV, Some(&raw))?.unwrap_or_default();
        let layout = self.state.layout();
        let capacity = layout.kv.capacity;
        let payload = entry_bytes(layout, capacity)?;
        // The cgroup-aware admission budget, not the bare meminfo figure, for the same
        // reason as the pinned-prefix probe: a capped host must not plan a second entry
        // from memory it cannot use. A failed probe keeps the one-entry store (logged).
        let available = admission_budget("Qwen36 tiered prefix", "one entry stays");
        if let Some(cause) = second_tiered_entry_refusal(payload, capacity, budget, available) {
            eprintln!("q36 cuda: tiered prefix second entry refused: {cause}");
        } else {
            self.prefix = PrefixSlot::with_limits(StoreLimits {
                roots: 0,
                entries: 2,
            })
            .with_recent_longer_protection()
            .with_blocks_per_lineage();
        }
        eprintln!(
            "q36 cuda: tiered prefix store, {} entries, extra budget {budget} B",
            self.prefix.limits().entries
        );
        Ok(())
    }

    pub(crate) fn enable_pinned_prefix(&mut self) -> Result<(), EngError> {
        self.enable_pinned_prefix_with(None)
    }

    /// [`Self::enable_pinned_prefix`] with the byte cap of the pool given by the caller. `None`
    /// reads `Q36_CUDA_PINNED_PREFIX_BYTES` and falls back to the default cap.
    pub(crate) fn enable_pinned_prefix_with(&mut self, cap: Option<usize>) -> Result<(), EngError> {
        if self.started || self.is_tiered() || self.prefix.key_len().is_some() {
            return Err(EngError::BadRequest(
                "pinned prefix needs a fresh non-tiered step".into(),
            ));
        }
        self.prefix = PrefixSlot::with_limits(StoreLimits {
            roots: 0,
            entries: 0,
        });
        self.prefix.set_shadow_enabled(shadow_block_lcp_enabled());
        let checkpoint = PrefixPinPool::checkpoint_bytes(self.state.layout())?;
        // The admission budget (`MemAvailable` clamped by the cgroup limits), not the bare
        // meminfo figure: a cgroup-capped host must size its pins from the capped budget, and
        // there is no later guard here. A failed probe plans ZERO pinned entries (logged):
        // the probe refuses on exactly the cases the clamp exists for, so falling back to the
        // unclamped figure would repeat the bug this closes.
        let available = admission_budget("Qwen36 pinned prefix", "pins stay off");
        let limit = match cap {
            Some(bytes) => bytes,
            None => byte_limit(PIN_BYTES_ENV, std::env::var_os(PIN_BYTES_ENV).as_deref())?
                .unwrap_or_else(|| default_pin_bytes(checkpoint, available)),
        };
        let entries = planned_entries(checkpoint, limit, available);
        if entries == 0 {
            eprintln!(
                "q36 cuda: pinned prefix refused, checkpoint {checkpoint} B, cap {limit} B, host headroom {HOST_HEADROOM} B, budget {available:?}; retained entries 0, bytes 0"
            );
            return Ok(());
        }
        let pool = match PrefixPinPool::new(self.context, *self.state.layout(), entries) {
            Ok(pool) => pool,
            Err(EngError::MemoryPlan(_)) => {
                eprintln!(
                    "q36 cuda: pinned prefix allocation refused; retained entries 0, bytes 0"
                );
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let retained = pool.retained_entries();
        self.prefix = PrefixSlot::with_limits(StoreLimits {
            roots: 0,
            entries: retained,
        })
        .with_recent_longer_protection()
        .with_blocks_per_lineage();
        self.prefix.set_shadow_enabled(shadow_block_lcp_enabled());
        eprintln!(
            "q36 cuda: persistent pinned prefix, retained {retained}, staging 1, checkpoint {} B, total {} B; kill switches: Q36_CUDA_PINNED_PREFIX=0 (pageable), Q36_CUDA_STATE_ONLY_PREFIX=0 (pinned only)",
            pool.bytes() / (retained + 1),
            pool.bytes()
        );
        self.prefix_pins = Some(pool);
        Ok(())
    }

    /// The extra device slot is accounted by the constructor before admitting the prefill arena.
    pub(crate) fn enable_device_tip(&mut self) -> Result<usize, EngError> {
        if self.started || self.prefix_pins.is_none() || self.prefix.key_len().is_some() {
            return Err(EngError::BadRequest(
                "device tip needs a fresh pinned-prefix step".into(),
            ));
        }
        let tip = match DeviceTip::new(self.context, &self.stream, &self.state) {
            Ok(tip) => tip,
            Err(EngError::MemoryPlan(_)) => {
                eprintln!("q36 cuda: device tip allocation refused; pinned host fallback");
                return Ok(0);
            }
            Err(error) => return Err(error),
        };
        let bytes = tip.bytes();
        self.device_tip = Some(tip);
        eprintln!(
            "q36 cuda: state-only device tip, {bytes} B; disable: Q36_CUDA_STATE_ONLY_PREFIX=0"
        );
        Ok(bytes)
    }

    pub(crate) fn pinned_prefix_enabled(&self) -> bool {
        self.prefix_pins.is_some()
    }

    pub(crate) fn stage_pinned_prefix(
        &mut self,
        session: &DecodeSession,
        key: &[u32],
        kind: SaveKind,
        path: (super::super::prefix::CudaPrefillPath, PrefixIdentity),
    ) -> Result<bool, EngError> {
        self.state_usable()?;
        let fed = self.tokens();
        let frozen = session.prepare_prefix_save(key, fed)?;
        self.complete_prefix_save()?;
        if self.prefix.limits().entries == 0 {
            return Ok(false);
        }
        self.free_slot_for_save();
        let pool = self
            .prefix_pins
            .as_mut()
            .ok_or_else(|| EngError::BadRequest("pinned prefix was not enabled".into()))?;
        let recycled = if kind == SaveKind::Tip {
            self.prefix.take_tip_if(key, &path, |saved| match saved {
                CudaSavedPrefix::Pinned { state: saved, .. } => {
                    pool.can_recycle(&self.stream, &self.state, saved)
                }
                _ => false,
            })
        } else {
            None
        };
        let recycled = match recycled {
            Some(CudaSavedPrefix::Pinned { state: saved, .. }) => Some(saved),
            _ => None,
        };
        let mut state = match pool.begin(self.context, &self.stream, &self.state, recycled) {
            Ok(state) => state,
            Err(EngError::MemoryPlan(_)) => return Ok(false),
            Err(error) => return Err(error),
        };
        // The tip copies the live state: refused through a gated state or an awaiting window.
        self.state_usable()?;
        if let Some(device) = self.device_tip.as_mut() {
            let key = device.capture(self.context, &self.stream, &mut self.state)?;
            state.attach_device_tip(key)?;
        }
        self.pending_prefix = Some(PendingPrefix {
            session: frozen,
            fed,
            kind,
            path,
            state,
        });
        Ok(true)
    }

    pub(crate) fn complete_prefix_save(&mut self) -> Result<(), EngError> {
        self.state_usable()?;
        let Some(pending) = self.pending_prefix.take() else {
            return Ok(());
        };
        // A failed model step must never publish a partly advanced snapshot.
        if self.state.health() != Health::Valid {
            return Ok(());
        }
        let began = Instant::now();
        let pool = self
            .prefix_pins
            .as_mut()
            .ok_or_else(|| EngError::BadRequest("pending prefix lost its pool".into()))?;
        let state = pool.finish(self.context, &self.stream, &self.state, pending.state)?;
        let positions = state.tokens();
        let charge = entry_bytes(self.state.layout(), positions)?;
        let copied_kv_bytes = state.copied_kv_bytes();
        let mut ready = Some(CudaSavedPrefix::Pinned {
            state,
            payload_bytes: charge,
        });
        let saved = pending.session.save_prefix(
            &mut self.prefix,
            pending.session.history(),
            pending.path,
            pending.fed,
            pending.kind,
            || {
                ready
                    .take()
                    .ok_or_else(|| EngError::BadRequest("prefix already published".into()))
            },
        )?;
        if saved {
            eprintln!(
                "q36 cuda: prefix saved {positions} positions, {charge} B, {:.1} ms, KV copied {copied_kv_bytes} B",
                began.elapsed().as_secs_f64() * 1000.0
            );
            self.spill_after_save();
        }
        Ok(())
    }

    pub(crate) fn restore_pinned_prefix(&mut self, saved: &PinnedSnapshot) -> Result<(), EngError> {
        // Refused, before the first fallible call, while the recovery gate is set or a window is
        // unresolved: only a reset recovers it (DESIGN 17.10).
        window_logic::restore_gate(self)?;
        self.complete_prefix_save()?;
        let device = self
            .prefix_pins
            .as_mut()
            .ok_or_else(|| EngError::BadRequest("pinned snapshot lost its pool".into()))?
            .restore(
                self.context,
                &self.stream,
                &mut self.state,
                saved,
                self.device_tip.as_ref(),
            )?;
        self.restore_source = Some(if device {
            crate::sched::job::PrefixSource::Device
        } else {
            crate::sched::job::PrefixSource::Pinned
        });
        if self.device_tip.is_some() {
            eprintln!(
                "q36 cuda: prefix restore source {}",
                if device { "device" } else { "host" }
            );
        }
        self.refill_prefill_report();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::decode_step::state_layout::StateLayout;
    use crate::family::{
        project::cuda::WeightLayout,
        qwen36::cuda::{
            common_tests,
            dims::{KvMode, StepOptions},
        },
    };
    use std::ffi::OsStr;

    #[test]
    fn second_tiered_entry_requires_its_full_key_and_payload_and_host_headroom() {
        let capacity = 128;
        let payload = 1024;
        let entry = payload + (capacity + 1) * size_of::<u32>();
        let available = 2 * entry + HOST_HEADROOM;
        let fits = |budget: usize, free: Option<usize>| {
            second_tiered_entry_refusal(payload, capacity, budget, free).is_none()
        };
        assert!(fits(entry, Some(available)));
        // Each refusal names its own cause.
        assert_eq!(
            second_tiered_entry_refusal(payload, capacity, entry - 1, Some(available)),
            Some("the extra budget is below the entry size")
        );
        assert_eq!(
            second_tiered_entry_refusal(payload, capacity, entry, Some(available - 1)),
            Some("available host memory is below two entries plus headroom")
        );
        assert_eq!(
            second_tiered_entry_refusal(payload, capacity, entry, None),
            Some("available host memory is below two entries plus headroom")
        );
        assert_eq!(
            second_tiered_entry_refusal(0, capacity, entry, Some(available)),
            Some("the checkpoint payload is zero")
        );
        assert!(!fits(entry - 1, Some(available)));
        assert!(!fits(entry, Some(available - 1)));
        assert!(!fits(entry, None));
        for (bytes, tokens) in [
            (0, 1),
            (usize::MAX, 1),
            (usize::MAX / 2, 1),
            (1, usize::MAX),
        ] {
            assert!(
                second_tiered_entry_refusal(bytes, tokens, usize::MAX, Some(usize::MAX)).is_some()
            );
        }
    }

    #[test]
    fn pin_admission_keeps_one_retained_entry_when_two_do_not_fit_host_memory() {
        let size = 1024;
        let limit = 3 * size;
        for available in [HOST_HEADROOM + 2 * size, HOST_HEADROOM + limit - 1] {
            assert_eq!(planned_entries(size, limit, Some(available)), 1);
        }
        assert_eq!(
            planned_entries(size, limit, Some(HOST_HEADROOM + 2 * size - 1)),
            0
        );
    }

    #[test]
    fn default_pin_bytes_cover_the_cap_or_half_of_the_budget() {
        let size = 1024;
        let full = (MAX_RETAINED + 1) * size;
        let margin = 1usize << 20;
        for (available, expected) in [
            (None, 0),
            (Some(0), 0),
            (Some(1), 0),
            (Some(2 * size), size),
            (Some(2 * size + 1), size),
            (Some(2 * full - 1), full - 1),
            (Some(2 * full), full),
            (Some(2 * full + margin), full),
            (Some(usize::MAX), full),
        ] {
            assert_eq!(
                default_pin_bytes(size, available),
                expected,
                "{available:?}"
            );
        }
        assert_eq!(
            default_pin_bytes(usize::MAX, Some(usize::MAX)),
            usize::MAX / 2
        );
        assert_eq!(default_pin_bytes(0, Some(usize::MAX)), 0);
    }

    #[test]
    fn pin_admission_reserves_a_physical_staging_slot() {
        let size = 1024;
        let margin = HOST_HEADROOM;
        for (limit, available, expected) in [
            (0, Some(margin + 3 * size), 0),
            (2 * size - 1, Some(margin + 3 * size), 0),
            (2 * size, Some(margin + 2 * size), 1),
            (2 * size, Some(margin + 2 * size - 1), 0),
            (3 * size - 1, Some(margin + 3 * size), 1),
            (3 * size, Some(margin + 3 * size), 2),
            (8 * size, Some(margin + 3 * size), 2),
            (3 * size, Some(margin + 3 * size - 1), 1),
            (3 * size, None, 0),
        ] {
            assert_eq!(planned_entries(size, limit, available), expected);
        }
        for checkpoint in [0, usize::MAX, usize::MAX / 2] {
            assert_eq!(planned_entries(checkpoint, usize::MAX, Some(usize::MAX)), 0);
        }
        assert_eq!(
            planned_entries(usize::MAX / 3, usize::MAX, Some(usize::MAX)),
            1
        );
    }

    #[test]
    fn retained_entries_grow_with_the_budget_and_stop_at_the_cap() {
        const GIB: usize = 1 << 30;
        // The default rule feeding the planner, as `enable_pinned_prefix` composes them, for
        // a checkpoint of 1 GiB.
        let plan =
            |budget: Option<usize>| planned_entries(GIB, default_pin_bytes(GIB, budget), budget);
        for (budget, expected) in [
            (None, 0),
            (Some(0), 0),
            // Half of the budget must hold the staging slot and one retained entry.
            (Some(3 * GIB), 0),
            (Some(4 * GIB), 1),
            (Some(6 * GIB), 2),
            (Some(8 * GIB), 3),
            (Some(14 * GIB), 6),
            (Some(16 * GIB), 7),
            (Some(18 * GIB), MAX_RETAINED),
            (Some(20 * GIB), MAX_RETAINED),
            (Some(512 * GIB), MAX_RETAINED),
        ] {
            assert_eq!(plan(budget), expected, "budget {budget:?}");
        }
        // The served 35B checkpoint on a 50 GiB host reaches the cap; a 5 GiB checkpoint (a
        // 256K-token KV) is limited by the half-budget rule instead.
        let served = 736_952_320;
        let host = Some(50 * GIB);
        assert_eq!(
            planned_entries(served, default_pin_bytes(served, host), host),
            MAX_RETAINED
        );
        let long = 5 * GIB;
        assert_eq!(
            planned_entries(long, default_pin_bytes(long, host), host),
            4
        );
        // The headroom check still lowers an operator-sized plan when memory runs short.
        assert_eq!(planned_entries(GIB, 9 * GIB, Some(4 * GIB)), 2);
        assert_eq!(planned_entries(GIB, 9 * GIB, Some(GIB + GIB / 2)), 0);
    }

    #[test]
    fn the_operator_byte_cap_overrides_the_default_rule() {
        let size = 1usize << 20;
        let budget = Some(HOST_HEADROOM + 64 * size);
        // An exact cap picks the retained count whatever the default would have been.
        for (cap, expected) in [
            (0, 0),
            (size, 0),
            (2 * size, 1),
            (4 * size, 3),
            (9 * size, MAX_RETAINED),
            (40 * size, MAX_RETAINED),
        ] {
            assert_eq!(planned_entries(size, cap, budget), expected, "cap {cap}");
        }
        // The default for the same budget is the capped rule, not the override.
        assert_eq!(default_pin_bytes(size, budget), 9 * size);
        // The override is still bounded by what the host can hold.
        let scarce = Some(HOST_HEADROOM + 4 * size);
        assert_eq!(planned_entries(size, 40 * size, scarce), 3);
    }

    #[test]
    fn pin_admission_keeps_one_gib_available_beyond_the_pool() {
        let size = 1024;
        let headroom = 1usize << 30;
        for (retained, below) in [(1, 0), (2, 1)] {
            let bytes = (retained + 1) * size;
            assert_eq!(
                planned_entries(size, bytes, Some(headroom + bytes - 1)),
                below
            );
            assert_eq!(
                planned_entries(size, bytes, Some(headroom + bytes)),
                retained
            );
        }
    }

    #[test]
    fn pin_checkpoint_charge_is_the_allocated_layout_in_both_formats() {
        let mut saw_padding = false;
        for mode in [KvMode::Fp16, KvMode::Int8G32] {
            for slots in [1, 3] {
                let options = StepOptions {
                    kv_mode: mode,
                    context_tokens: 17,
                    delta_slots: slots,
                    forked_experts: false,
                    tiered: None,
                    layout: WeightLayout::Native,
                };
                let layout = StateLayout::new(&common_tests::dims(), &options).unwrap();
                let bytes = PrefixPinPool::checkpoint_bytes(&layout).unwrap();
                assert_eq!(bytes, layout.delta.total / slots + layout.kv.total);
                let payload = entry_bytes(&layout, 17).unwrap();
                assert!(bytes >= payload);
                saw_padding |= bytes > payload;
                assert!(bytes > entry_bytes(&layout, 1).unwrap());
                let mut invalid = layout;
                invalid.delta.slots = 0;
                assert!(PrefixPinPool::checkpoint_bytes(&invalid).is_err());
                invalid.delta.slots = 1;
                invalid.delta.total = usize::MAX;
                assert!(matches!(
                    PrefixPinPool::checkpoint_bytes(&invalid),
                    Err(EngError::MemoryPlan(_))
                ));
            }
        }
        assert!(
            saw_padding,
            "fixture must distinguish physical backing from exported payload"
        );
    }

    #[test]
    fn pin_byte_limits_reject_malformed_values_without_global_env_mutation() {
        assert_eq!(byte_limit(PIN_BYTES_ENV, None), Ok(None));
        assert_eq!(
            byte_limit(PIN_BYTES_ENV, Some(OsStr::new("0"))),
            Ok(Some(0))
        );
        assert_eq!(
            byte_limit(PIN_BYTES_ENV, Some(OsStr::new(" 1024 "))),
            Ok(Some(1024))
        );
        for value in ["", "-1", "1KiB", "garbage"] {
            assert!(byte_limit(PIN_BYTES_ENV, Some(OsStr::new(value))).is_err());
        }
        let overflow = format!("{}0", usize::MAX);
        assert!(byte_limit(PIN_BYTES_ENV, Some(OsStr::new(&overflow))).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(byte_limit(PIN_BYTES_ENV, Some(OsStr::from_bytes(&[0xff]))).is_err());
        }
    }

    #[test]
    fn unavailable_or_malformed_host_memory_refuses_retention() {
        assert_eq!(
            mem_available("MemTotal: 99 kB\nMemAvailable: 3072 kB\n"),
            Some(3 << 20)
        );
        for info in [
            "",
            "MemAvailable: -1 kB",
            "MemAvailable: 1 MB",
            "MemAvailable: 1 kB extra",
            "MemAvailable: nope kB",
        ] {
            assert_eq!(mem_available(info), None);
        }
        assert_eq!(
            mem_available(&format!("MemAvailable: {} kB", usize::MAX)),
            None
        );
    }
}
