//! The safe device-set probe: which visible devices exist, what they are, and
//! whether each ordered pair may directly access the other's memory.
//!
//! The raw entries live in `native::peer_probe`; every call here runs under
//! the process dispatch guard. Probing changes no persistent device state:
//! the free-memory figure of the profile query selects the device briefly
//! and restores the thread's previous device, so the selection cache stays
//! true. `enable_peer_access` is the one stateful call: it enables peer
//! mappings for the process and restores device 0 of the set as the thread's
//! current device through `context::select_device`, so the selection cache
//! there stays true.

use std::ffi::c_int;

use super::context::select_device;
use super::{INITIALIZE_OPERATION, TERMINAL, count_operation, ensure_usable, guard, native_error};
use crate::EngError;

/// The most devices one engine may span. The config surface enforces the same
/// bound; this is the backend's own statement of it.
pub(crate) const MAX_DEVICES: usize = 4;

/// `cudaErrorPeerAccessAlreadyEnabled` from `cuda_runtime_api.h`.
const PEER_ACCESS_ALREADY_ENABLED: c_int = 704;

/// What one probed device is and which set members it may access directly.
/// The memory figures and the name are the placement facts of the same
/// probe: one loop answers both, so the caller never mixes snapshots of the
/// same device taken at different times.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceFacts {
    /// The CUDA device ordinal.
    pub(crate) device: usize,
    pub(crate) cc_major: c_int,
    pub(crate) cc_minor: c_int,
    /// One entry per probed set member, in set order: `true` when this device
    /// may directly access that member's memory. The diagonal is `true` by
    /// definition and is not queried.
    pub(crate) can_access: Vec<bool>,
    /// The device name, as the runtime reports it.
    pub(crate) name: String,
    /// Total global memory of the device.
    pub(crate) total_bytes: u64,
    /// Free global memory at probe time, BEFORE any context of this process
    /// exists: the figure the placement planner budgets against.
    pub(crate) free_bytes: u64,
    /// The SM count, a hardware fact of the share model.
    pub(crate) multiprocessors: u32,
    /// Memory bus width in bits and memory clock in Hz: the two facts the
    /// bandwidth figure of the placement profile derives from (bus/8 x 2 x
    /// clock, the double-data-rate peak, a device property, not a benchmark).
    pub(crate) bus_width_bits: u32,
    pub(crate) memory_clock_hz: u64,
}

/// Checks a requested device list against the visible count and returns the
/// ordinals as `c_int`. Pure: the caller supplies the count.
///
/// An empty list, more than [`MAX_DEVICES`] entries, a duplicate, or a list
/// that is not strictly ascending is a `BadRequest`. An ordinal at or above
/// the visible count is a `Backend` failure naming both numbers.
pub(crate) fn check_device_list(devices: &[usize], visible: usize) -> Result<Vec<c_int>, EngError> {
    let bad = |message: String| EngError::BadRequest(format!("cuda devices: {message}"));
    if devices.is_empty() {
        return Err(bad("the list is empty".to_owned()));
    }
    if devices.len() > MAX_DEVICES {
        return Err(bad(format!(
            "{} devices exceed the maximum of {MAX_DEVICES}",
            devices.len()
        )));
    }
    if devices.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(bad(format!(
            "the list {devices:?} is not strictly ascending without duplicates"
        )));
    }
    devices
        .iter()
        .map(|&device| {
            if device >= visible {
                return Err(EngError::Backend {
                    operation: INITIALIZE_OPERATION,
                    message: format!("CUDA device {device} is not visible ({visible} devices)"),
                });
            }
            c_int::try_from(device).map_err(|_| EngError::Backend {
                operation: INITIALIZE_OPERATION,
                message: format!("CUDA device {device} does not fit a device ordinal"),
            })
        })
        .collect()
}

/// The verdict a probed set must pass before an engine may span it, as a
/// refusal text. Pure over [`DeviceFacts`]: one compute capability across the
/// set, and direct access for every ordered pair of distinct devices. The
/// first failing pair is named with the usual causes, because "peer access
/// off" alone sends the operator to three different settings.
pub(crate) fn check_peer_facts(facts: &[DeviceFacts]) -> Result<(), String> {
    let Some(first) = facts.first() else {
        return Err("the device set is empty".to_owned());
    };
    for fact in &facts[1..] {
        if (fact.cc_major, fact.cc_minor) != (first.cc_major, first.cc_minor) {
            return Err(format!(
                "device {} is compute capability {}.{} but device {} is {}.{}; \
                 a device set needs one compute capability",
                fact.device,
                fact.cc_major,
                fact.cc_minor,
                first.device,
                first.cc_major,
                first.cc_minor
            ));
        }
    }
    for (i, fact) in facts.iter().enumerate() {
        for (j, peer) in facts.iter().enumerate() {
            if i != j && !fact.can_access.get(j).copied().unwrap_or(false) {
                return Err(format!(
                    "device {} cannot directly access device {} \
                     (peer access off, IOMMU not in pass-through, or BAR1 too small)",
                    fact.device, peer.device
                ));
            }
        }
    }
    Ok(())
}

fn probe_error(status: c_int) -> EngError {
    native_error(&TERMINAL, count_operation(status), status)
}

/// Probes every device of `devices`: compute capability and the full ordered
/// peer-access matrix of the set. Makes no state change.
pub(crate) fn probe_device_set(devices: &[usize]) -> Result<Vec<DeviceFacts>, EngError> {
    let ordinals = check_device_list(devices, super::device_count()?)?;
    let _dispatch = guard::DispatchGuard::enter(&TERMINAL);
    ensure_usable(&TERMINAL)?;
    let mut facts = Vec::with_capacity(ordinals.len());
    for &device in &ordinals {
        let (cc_major, cc_minor) =
            super::native::peer_probe::device_cc(device).map_err(probe_error)?;
        let profile = super::native::peer_probe::device_profile(device).map_err(probe_error)?;
        let mut can_access = Vec::with_capacity(ordinals.len());
        for &peer in &ordinals {
            can_access.push(if peer == device {
                true
            } else {
                super::native::peer_probe::peer_can_access(device, peer).map_err(probe_error)?
            });
        }
        facts.push(DeviceFacts {
            device: usize::try_from(device).unwrap_or(0),
            cc_major,
            cc_minor,
            can_access,
            name: profile.name,
            total_bytes: profile.total_bytes,
            free_bytes: profile.free_bytes,
            multiprocessors: profile.multiprocessors,
            bus_width_bits: profile.bus_width_bits,
            memory_clock_hz: profile.memory_clock_hz,
        });
    }
    Ok(facts)
}

/// Enables direct peer access for every ordered pair of distinct devices in
/// the set. A pair the runtime reports as already enabled counts as enabled.
/// On return the thread's current device is the first set member, selected
/// through the context selection cache.
pub(crate) fn enable_peer_access(devices: &[usize]) -> Result<(), EngError> {
    let ordinals = check_device_list(devices, super::device_count()?)?;
    let _dispatch = guard::DispatchGuard::enter(&TERMINAL);
    ensure_usable(&TERMINAL)?;
    for &device in &ordinals {
        for &peer in &ordinals {
            if peer == device {
                continue;
            }
            match super::native::peer_probe::peer_enable(device, peer) {
                Ok(()) | Err(PEER_ACCESS_ALREADY_ENABLED) => {}
                Err(status) => return Err(probe_error(status)),
            }
            // The native enable left `device` current; record exactly that in
            // the selection cache before anything else selects a device.
            select_device(device)?;
        }
    }
    select_device(ordinals[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::testkit::device_present;

    #[test]
    fn the_device_list_rules_refuse_each_bad_shape() {
        let cases: [&[usize]; 5] = [&[], &[0, 0], &[1, 0], &[0, 2, 1], &[0, 1, 2, 3, 4]];
        for devices in cases {
            let error = check_device_list(devices, 8).expect_err("bad list must refuse");
            assert!(
                matches!(error, EngError::BadRequest(_)),
                "{devices:?}: {error:?}"
            );
        }
        assert_eq!(
            check_device_list(&[0, 2], 3).expect("ascending pair"),
            vec![0, 2]
        );
    }

    #[test]
    fn an_invisible_ordinal_names_both_numbers() {
        let error = check_device_list(&[0, 3], 2).expect_err("invisible device must refuse");
        match error {
            EngError::Backend { message, .. } => {
                assert!(message.contains("device 3"), "{message}");
                assert!(message.contains("(2 devices)"), "{message}");
            }
            other => panic!("want Backend, got {other:?}"),
        }
    }

    fn facts(device: usize, cc: (c_int, c_int), can_access: &[bool]) -> DeviceFacts {
        DeviceFacts {
            device,
            cc_major: cc.0,
            cc_minor: cc.1,
            can_access: can_access.to_vec(),
            name: format!("device {device}"),
            total_bytes: 24 * 1024 * 1024 * 1024,
            free_bytes: 20 * 1024 * 1024 * 1024,
            multiprocessors: 82,
            bus_width_bits: 384,
            memory_clock_hz: 9_500_000_000,
        }
    }

    #[test]
    fn peer_facts_refuse_a_mixed_capability_and_a_blocked_pair() {
        let mixed = [
            facts(0, (8, 6), &[true, true]),
            facts(1, (7, 0), &[true, true]),
        ];
        assert!(
            check_peer_facts(&mixed)
                .expect_err("mixed cc")
                .contains("compute capability")
        );
        let blocked = [
            facts(0, (8, 6), &[true, false]),
            facts(1, (8, 6), &[true, true]),
        ];
        let refusal = check_peer_facts(&blocked).expect_err("blocked pair");
        assert!(
            refusal.contains("device 0 cannot directly access device 1"),
            "{refusal}"
        );
        let full = [
            facts(0, (8, 6), &[true, true]),
            facts(2, (8, 6), &[true, true]),
        ];
        assert_eq!(check_peer_facts(&full), Ok(()));
        assert!(check_peer_facts(&[]).is_err());
    }

    #[test]
    fn probing_device_zero_reports_its_compute_capability() {
        if !device_present("probing_device_zero_reports_its_compute_capability") {
            return;
        }
        let facts = probe_device_set(&[0]).expect("probe of device 0");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].device, 0);
        assert!(
            facts[0].cc_major > 0,
            "cc {}.{}",
            facts[0].cc_major,
            facts[0].cc_minor
        );
        assert_eq!(facts[0].can_access, vec![true]);
        enable_peer_access(&[0]).expect("single-device enable is a no-op");
    }
}
