//! The plan of a prefix store whose entries are copies in host memory.
//!
//! A family whose saved prefixes live in host RAM asks this module at start how many entries the
//! host may hold, and asks the shared admission at every save whether one more entry fits.
//!
//! The count at start reads the one byte budget of the host store (`kv_disk::default_host_budget`:
//! the machine's memory less the OS floor and the bytes the process already holds), the figure
//! the disk tier spills against, so the plan and the tier cannot disagree on how much the host may
//! hold. The save reads [`HostFacts`] fresh and admits the entry as a payload (`admit_payload`:
//! the free memory above the OS floor, inside a binding cgroup the cgroup's remainder less the
//! process's own overhead). The budget is a ceiling on the store; the admission is the memory the
//! kernel will grant now, and under a cgroup limit it is the smaller of the two.
//!
//! The store is planned whenever the configuration allows it. A host too small or too busy to
//! fund an entry keeps the store with its saves refused by the admission, so reuse begins once
//! memory frees up and stops if the host tightens. A start never refuses over this: the store is
//! an optional gain.
//!
//! A host whose memory is unknown has no budget to size against: the store keeps one entry.

use crate::EngError;
use crate::family::mem::HostFacts;
use crate::family::session::StoreLimits;

/// The bytes of the largest entry of a store whose entries copy the cache cells of one session:
/// `per_position` bytes for each of `cells` positions. An unknown cost or an overflow saturates, so
/// a plan never funds it.
pub(crate) fn largest_entry_bytes(per_position: Option<u64>, cells: u64) -> u64 {
    per_position
        .and_then(|bytes| bytes.checked_mul(cells))
        .unwrap_or(u64::MAX)
}

/// Admits one save of `bytes` named `what` against the host facts a save reads now, by the rule the
/// start funds the store by. Facts that cannot be read proceed with a log line: an optional save
/// must not turn a host whose layout the reader cannot parse into a host where every save is
/// refused and reuse disappears silently.
pub(crate) fn admit_entry(
    what: &str,
    bytes: usize,
    host: Result<HostFacts, EngError>,
) -> Result<(), EngError> {
    match host {
        Ok(facts) => facts.admit_payload(bytes as u64, what),
        Err(error) => {
            eprintln!(
                "{what}: host facts unavailable ({}); proceeding without the check",
                error.describe()
            );
            Ok(())
        }
    }
}

/// What one start decided about its host-side prefix store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostStorePlan {
    /// The bytes of the largest entry the store can hold.
    pub(crate) entry_bytes: u64,
    /// The bytes of saved prefixes the host may hold (the shared host byte budget the disk tier
    /// spills against), or `None` when the machine's memory is unknown.
    pub(crate) budget: Option<u64>,
    /// How many entries the store keeps: the sessions that may hold distinct conversations, bounded
    /// by the entries the budget funds; at least one while the store is on.
    pub(crate) entries: usize,
    /// Whether the budget funds one largest entry. A store that is not funded stays planned and
    /// its saves are refused by [`admit_entry`] until the host can fund them.
    pub(crate) funded: bool,
    /// Whether the configuration turned the store off (`prefix_reuse` set to false).
    pub(crate) configured_off: bool,
}

impl HostStorePlan {
    /// The plan for a store the configuration turned off.
    pub(crate) fn off(entry_bytes: u64) -> Self {
        Self {
            entry_bytes,
            budget: None,
            entries: 0,
            funded: false,
            configured_off: true,
        }
    }

    /// The plan of one start: off when the configuration says so (the budget is not read), else
    /// decided from the byte budget `budget` reads.
    pub(crate) fn for_start(
        prefix_reuse: bool,
        entry_bytes: u64,
        budget: impl FnOnce() -> Option<u64>,
        sessions: usize,
    ) -> Self {
        if prefix_reuse {
            Self::plan(entry_bytes, budget(), sessions)
        } else {
            Self::off(entry_bytes)
        }
    }

    /// Decides from the byte budget; `sessions` is how many conversations the engine serves at
    /// once.
    pub(crate) fn plan(entry_bytes: u64, budget: Option<u64>, sessions: usize) -> Self {
        let affordable = budget.map(|bytes| {
            usize::try_from(bytes.checked_div(entry_bytes).unwrap_or(0)).unwrap_or(usize::MAX)
        });
        let entries = match affordable {
            // Nothing to size against: one entry, as a lone chat needs.
            None => 1,
            Some(fits) => sessions.min(fits).max(1),
        };
        Self {
            entry_bytes,
            budget,
            entries,
            funded: affordable.is_none_or(|fits| fits >= 1),
            configured_off: false,
        }
    }

    /// Whether the store is planned: every start that the configuration did not turn off.
    pub(crate) fn on(&self) -> bool {
        !self.configured_off
    }

    /// The store's limits: `roots` pinned roots and the planned entries when it is on, and no
    /// roots and no entries otherwise, which the worker reads as a store a prompt boundary cannot
    /// use.
    pub(crate) fn limits(&self, roots: usize) -> StoreLimits {
        if self.on() {
            StoreLimits {
                roots,
                entries: self.entries,
            }
        } else {
            StoreLimits {
                roots: 0,
                entries: 0,
            }
        }
    }

    /// The start line, in MiB.
    pub(crate) fn describe(&self) -> String {
        let mib = |bytes: u64| bytes >> 20;
        if self.configured_off {
            return "prefix reuse is off (the configuration sets prefix_reuse to false)".into();
        }
        match self.budget {
            None => format!(
                "prefix reuse is on: saved prefixes are host copies of up to {} MiB, {} kept; the host memory could not be read, so the saves are not sized against it",
                mib(self.entry_bytes),
                self.entries
            ),
            Some(budget) if self.funded => format!(
                "prefix reuse is on: saved prefixes are host copies of up to {} MiB, {} kept; the host may hold {} MiB of them; each save checks the free memory again",
                mib(self.entry_bytes),
                self.entries,
                mib(budget)
            ),
            Some(budget) => format!(
                "prefix reuse is on but saving waits: the host budget of {} MiB cannot fund a saved prefix of up to {} MiB; each save checks the host again and starts once it can",
                mib(budget),
                mib(self.entry_bytes)
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// The facts of a host with `total` GiB of memory and `available` GiB free, and the cgroup's
    /// remaining bytes when it has a limit.
    fn host(total: u64, available: u64, cgroup: Option<u64>) -> HostFacts {
        let meminfo = format!(
            "MemTotal:       {} kB\nMemAvailable:   {} kB\n",
            total * (1 << 20),
            available * (1 << 20)
        );
        HostFacts::from_meminfo("test", &meminfo, cgroup).expect("host")
    }

    #[test]
    fn a_budget_that_funds_the_entries_keeps_them() {
        let plan = HostStorePlan::plan(GIB, Some(12 * GIB), 4);
        assert!(plan.on() && plan.funded);
        assert_eq!(plan.entries, 4);
        assert_eq!(plan.limits(0).entries, 4);
        let line = plan.describe();
        assert!(line.starts_with("prefix reuse is on:"), "{line}");
        assert!(line.contains("1024 MiB, 4 kept"), "{line}");
        assert!(line.contains("12288 MiB of them"), "{line}");
    }

    #[test]
    fn the_entries_are_bounded_by_what_the_budget_funds() {
        // 12 GiB funds three 4 GiB entries however many sessions serve.
        let plan = HostStorePlan::plan(4 * GIB, Some(12 * GIB), 8);
        assert_eq!(plan.entries, 3);
        // A lone session keeps one entry on any host.
        assert_eq!(HostStorePlan::plan(GIB, Some(12 * GIB), 1).entries, 1);
    }

    #[test]
    fn a_budget_under_one_entry_keeps_the_store_and_waits() {
        let plan = HostStorePlan::plan(GIB, Some(GIB / 2), 4);
        assert!(plan.on());
        assert!(!plan.funded);
        assert_eq!(plan.limits(0).entries, 1, "the store is planned for later");
        let line = plan.describe();
        assert!(
            line.starts_with("prefix reuse is on but saving waits"),
            "{line}"
        );
        assert!(line.contains("512 MiB"), "{line}");
        assert!(line.contains("1024 MiB"), "{line}");
    }

    #[test]
    fn a_save_is_admitted_exactly_when_the_entry_fits_above_the_floor() {
        let floor = host(32, 20, None).os_floor();
        let at = |spare: u64| host(32, (floor + spare) >> 30, None);
        assert!(admit_entry("a saved prefix", (3 * GIB) as usize, Ok(at(3 * GIB))).is_ok());
        let error = admit_entry("a saved prefix", (3 * GIB) as usize, Ok(at(2 * GIB)))
            .expect_err("the floor is not spent");
        assert!(
            matches!(&error, EngError::MemoryPlan(text) if text.contains("a saved prefix")),
            "{error:?}"
        );
    }

    #[test]
    fn a_host_that_has_room_for_the_bytes_but_not_the_floor_refuses_the_entry() {
        // A 32 GiB host keeps 8 GiB for the OS: of 10 GiB available, 2 GiB can be spent, so a
        // 3 GiB entry is refused although 10 GiB is free (the old save compared the bytes with the
        // available memory alone and admitted it).
        let tight = host(32, 10, None);
        assert!(tight.budget() > 3 * GIB);
        assert!(admit_entry("a saved prefix", (3 * GIB) as usize, Ok(tight)).is_err());
        assert!(admit_entry("a saved prefix", (2 * GIB) as usize, Ok(host(32, 10, None))).is_ok());
    }

    #[test]
    fn a_cgroup_that_has_little_left_refuses_what_the_host_would_hold() {
        let boxed = host(32, 20, Some(1 << 29));
        assert!(admit_entry("a saved prefix", GIB as usize, Ok(boxed)).is_err());
        assert!(admit_entry("a saved prefix", GIB as usize, Ok(host(32, 20, None))).is_ok());
    }

    #[test]
    fn a_host_whose_memory_is_unknown_keeps_one_entry_and_an_unreadable_save_check_proceeds() {
        let plan = HostStorePlan::plan(GIB, None, 8);
        assert!(plan.on());
        assert_eq!(plan.entries, 1);
        assert!(plan.describe().contains("not sized against it"));
        let refusal = EngError::MemoryPlan("cgroup v1 layout".into());
        assert!(admit_entry("a saved prefix", GIB as usize, Err(refusal)).is_ok());
    }

    #[test]
    fn a_configuration_that_turns_the_store_off_keeps_nothing() {
        let plan = HostStorePlan::off(GIB);
        assert!(!plan.on());
        assert_eq!(plan.limits(2).entries, 0);
        assert_eq!(plan.limits(2).roots, 0);
        assert!(plan.describe().contains("prefix_reuse to false"));
        let off = HostStorePlan::for_start(false, GIB, || panic!("the host is not read"), 4);
        assert!(!off.on());
    }

    #[test]
    fn the_largest_entry_is_the_whole_cache_and_saturates_when_unknown() {
        assert_eq!(largest_entry_bytes(Some(30_464), 32_768), 30_464 * 32_768);
        assert_eq!(largest_entry_bytes(None, 32_768), u64::MAX);
        assert_eq!(largest_entry_bytes(Some(u64::MAX), 2), u64::MAX);
    }
}
