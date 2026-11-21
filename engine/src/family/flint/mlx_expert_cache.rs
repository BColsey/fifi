//! Per-layer packed expert banks for one-row disk streaming.

use super::{
    config::LAYERS,
    mlx_forward::grouped_experts,
    mlx_model::SwiGlu,
    weights::{ExpertPageLayout, FlintCheckpoint},
};
use crate::{
    EngError,
    kernels::{HostPages, MlxKernels, PageRange},
    tensor::{Dtype, ExecutionTensor},
};

const CACHE_MIB: &str = "FIFI_FLINT_MLX_EXPERT_CACHE_MIB";
const ROUTED_LAYERS: usize = LAYERS - 2;
const ROUTES: usize = 6;
const EXPERTS: usize = 256;
const INDEX_BYTES: usize = ROUTES * size_of::<u32>();

type Result<T> = std::result::Result<T, EngError>;
/// The (expert, slot) pairs a route still has to read.
type SlotWrites = Vec<(u32, usize)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    expert: u32,
    used: u64,
}

pub(super) struct LayerExpertCache {
    layout: ExpertPageLayout,
    pages: HostPages,
    slots: Vec<Option<Entry>>,
    tick: u64,
}

/// What the cache switch asks for, given the bytes left for it.
#[derive(Debug, PartialEq, Eq)]
enum Sizing {
    /// The switch is absent or 0.
    Off,
    /// The requested cache does not fit, and not even the smallest useful one does.
    Skip(String),
    /// Slots to allocate inside `usable` bytes; `reduced` when the budget, not the
    /// setting, set the size.
    Use {
        slots: usize,
        usable: u64,
        reduced: bool,
    },
}

fn minimum_cache_bytes(per_expert: u64, page: u64) -> u64 {
    per_expert * (ROUTED_LAYERS * ROUTES) as u64 + ROUTED_LAYERS as u64 * page
}

/// Read the MiB setting and fit it to the bytes available. A value that cannot
/// hold the smallest useful cache, or a budget that cannot hold it, skips the
/// cache; a budget below the setting shrinks the cache to fit. Only a value
/// that is not a MiB count is refused.
fn size_cache(
    setting: Option<&std::ffi::OsStr>,
    per_expert: u64,
    page: u64,
    available: u64,
) -> Result<Sizing> {
    let Some(setting) = setting else {
        return Ok(Sizing::Off);
    };
    let mib = setting
        .to_str()
        .and_then(|text| text.parse::<u64>().ok())
        .ok_or_else(|| EngError::MemoryPlan(format!("{CACHE_MIB} must be an integer MiB")))?;
    if mib == 0 {
        return Ok(Sizing::Off);
    }
    let cap = mib
        .checked_mul(1 << 20)
        .ok_or_else(|| EngError::MemoryPlan(format!("{CACHE_MIB} overflows bytes")))?;
    let minimum = minimum_cache_bytes(per_expert, page);
    if cap < minimum {
        return Ok(Sizing::Skip(format!(
            "{cap} bytes requested, and the smallest cache needs {minimum} for {ROUTES} slots per routed layer"
        )));
    }
    let usable = cap.min(available);
    if usable < minimum {
        return Ok(Sizing::Skip(format!(
            "{available} bytes remain after weights, staging and one prefill piece, and the smallest cache needs {minimum}"
        )));
    }
    let index_reserve = ROUTED_LAYERS as u64 * page;
    let slots = ((usable - index_reserve) / per_expert).min((ROUTED_LAYERS * EXPERTS) as u64);
    Ok(Sizing::Use {
        slots: slots as usize,
        usable,
        reduced: available < cap,
    })
}

/// Log that the switch is set but this start does not use it.
pub(super) fn note_ignored(reason: &str) {
    if let Some(value) = std::env::var_os(CACHE_MIB)
        && !value.is_empty()
        && value != "0"
    {
        eprintln!("flint: {CACHE_MIB} ignored: {reason}");
    }
}

fn build_layers(
    source: &FlintCheckpoint,
    slots: usize,
) -> Result<(Vec<Option<LayerExpertCache>>, u64)> {
    let mut layers: Vec<Option<LayerExpertCache>> = (0..LAYERS).map(|_| None).collect();
    let mut bytes = 0;
    for (index, layer) in layers.iter_mut().enumerate().skip(2) {
        let count = slots / ROUTED_LAYERS + usize::from(index - 2 < slots % ROUTED_LAYERS);
        let cache = LayerExpertCache::new(
            source,
            &format!("model.layers.{index}.mlp.switch_mlp"),
            count,
        )?;
        bytes += cache.pages.len() as u64;
        *layer = Some(cache);
    }
    Ok((layers, bytes))
}

/// Size and allocate the cache inside `available` bytes (what remains of the
/// device budget after resident weights, one-row staging and one prefill
/// piece).
pub(super) fn plan(
    source: &FlintCheckpoint,
    available: u64,
) -> Result<(Vec<Option<LayerExpertCache>>, u64)> {
    let empty = || (0..LAYERS).map(|_| None).collect();
    let page = HostPages::page_size() as u64;
    let setting = std::env::var_os(CACHE_MIB);
    let (slots, usable, reduced) = match size_cache(
        setting.as_deref(),
        source.bytes_per_routed_expert(),
        page,
        available,
    )? {
        Sizing::Off => return Ok((empty(), 0)),
        Sizing::Skip(reason) => {
            eprintln!("flint: {CACHE_MIB} ignored: {reason}");
            return Ok((empty(), 0));
        }
        Sizing::Use {
            slots,
            usable,
            reduced,
        } => (slots, usable, reduced),
    };
    match fit_allocation(slots, usable, ROUTED_LAYERS * ROUTES, |slots| {
        build_layers(source, slots)
    })? {
        Some((layers, bytes, slots, shrunk)) => {
            if reduced || shrunk {
                eprintln!("flint: {CACHE_MIB} reduced to {slots} slots to fit the device budget");
            }
            eprintln!("flint: MLX expert cache bytes={bytes} slots={slots}");
            Ok((layers, bytes))
        }
        None => {
            eprintln!(
                "flint: {CACHE_MIB} ignored: the smallest cache allocates more than the {usable} bytes available"
            );
            Ok((empty(), 0))
        }
    }
}

/// Allocate `slots` with `build`, which returns the allocation and its byte
/// size. Each layer's pages are page aligned, so the size can exceed the slot
/// arithmetic: shrink the slot count in proportion until the allocation fits
/// `usable`, and give up (`None`) when even `minimum` slots do not. The flag
/// says whether the slot count shrank.
fn fit_allocation<T>(
    mut slots: usize,
    usable: u64,
    minimum: usize,
    mut build: impl FnMut(usize) -> Result<(T, u64)>,
) -> Result<Option<(T, u64, usize, bool)>> {
    let mut shrunk = false;
    loop {
        let (built, bytes) = build(slots)?;
        if bytes <= usable {
            return Ok(Some((built, bytes, slots, shrunk)));
        }
        if slots <= minimum {
            return Ok(None);
        }
        shrunk = true;
        let fitted = (slots as u128 * u128::from(usable) / u128::from(bytes)) as usize;
        slots = fitted.min(slots - 1).max(minimum);
    }
}

/// Publish `next` only after `read` has filled every written slot. A failed
/// read can overwrite part of a victim, so the written slots are emptied
/// before the read and stay empty when it fails.
fn commit_reads(
    slots: &mut Vec<Option<Entry>>,
    next: Vec<Option<Entry>>,
    writes: &[(u32, usize)],
    read: impl FnOnce() -> Result<()>,
) -> Result<()> {
    for &(_, slot) in writes {
        slots[slot] = None;
    }
    read()?;
    *slots = next;
    Ok(())
}

fn route_slots(
    slots: &mut [Option<Entry>],
    experts: &[u32; ROUTES],
    tick: u64,
) -> Result<([u32; ROUTES], SlotWrites)> {
    let mut local = [0; ROUTES];
    let mut writes = Vec::with_capacity(ROUTES);
    for (rank, &expert) in experts.iter().enumerate() {
        let slot = match slots
            .iter()
            .position(|entry| entry.is_some_and(|e| e.expert == expert))
        {
            Some(slot) => slot,
            None => {
                // Protect the entire route, including hits later in rank order.
                let (slot, _) = slots
                    .iter()
                    .enumerate()
                    .filter(|(_, entry)| entry.is_none_or(|e| !experts.contains(&e.expert)))
                    .min_by_key(|(_, entry)| entry.map_or(0, |e| e.used))
                    .ok_or_else(|| {
                        EngError::BadRequest("Flint expert cache route has no free slot".into())
                    })?;
                writes.push((expert, slot));
                slot
            }
        };
        slots[slot] = Some(Entry { expert, used: tick });
        local[rank] = slot as u32;
    }
    Ok((local, writes))
}

impl LayerExpertCache {
    pub(super) fn new(source: &FlintCheckpoint, stem: &str, count: usize) -> Result<Self> {
        let layout = source.expert_page_layout(stem, count as u32)?;
        let pages = HostPages::try_new(layout.index_offset + INDEX_BYTES)?;
        Ok(Self {
            layout,
            pages,
            slots: vec![None; count],
            tick: 0,
        })
    }

    pub(super) fn evaluate(
        &mut self,
        k: &mut MlxKernels,
        source: &FlintCheckpoint,
        x: &ExecutionTensor,
        experts: &[u32; ROUTES],
    ) -> Result<ExecutionTensor> {
        let dst = self.pages.try_write().ok_or_else(|| EngError::Backend {
            operation: "Flint expert cache",
            message: "an earlier stage still holds expert cache pages".into(),
        })?;
        self.tick += 1;
        let mut next = self.slots.clone();
        let (local, writes) = route_slots(&mut next, experts, self.tick)?;
        let layout = &self.layout;
        commit_reads(&mut self.slots, next, &writes, || {
            source.read_expert_slots(layout, &writes, dst)
        })?;

        let offset = self.layout.index_offset;
        for (dst, id) in dst[offset..offset + INDEX_BYTES]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(local)
        {
            dst.copy_from_slice(&id.to_le_bytes());
        }
        let indices = PageRange {
            offset,
            dtype: Dtype::U32,
            shape: vec![ROUTES as u32],
        };
        let result = (|| {
            let (weights, ids) =
                SwiGlu::import_expert_pages(k, &self.pages, &self.layout.packed, indices)?;
            let local = k.bound_indices(&ids, self.slots.len() as u32)?;
            let rows = grouped_experts(k, &weights, x, &local)?;
            k.evaluate(&[&rows])?;
            Ok(rows)
        })();
        // The graph has completed, and only the evaluated output can escape.
        self.pages.release();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_protects_later_hits_and_evicts_oldest_unselected() {
        let mut slots = vec![None; 7];
        let (_, writes) = route_slots(&mut slots, &[10, 11, 12, 13, 14, 15], 1).unwrap();
        assert_eq!(writes.len(), 6);
        let (local, writes) = route_slots(&mut slots, &[16, 11, 12, 13, 14, 15], 2).unwrap();
        assert_eq!(local, [6, 1, 2, 3, 4, 5]);
        assert_eq!(writes, [(16, 6)]);
        let (local, writes) = route_slots(&mut slots, &[17, 11, 12, 13, 14, 15], 3).unwrap();
        assert_eq!(local, [0, 1, 2, 3, 4, 5]);
        assert_eq!(writes, [(17, 0)]);
    }

    const PER_EXPERT: u64 = 1_769_472;
    const PAGE: u64 = 16_384;

    fn minimum() -> u64 {
        minimum_cache_bytes(PER_EXPERT, PAGE)
    }

    fn os(text: &str) -> Option<&std::ffi::OsStr> {
        Some(std::ffi::OsStr::new(text))
    }

    #[test]
    fn the_cache_is_off_when_the_switch_is_absent_or_zero() {
        assert_eq!(
            size_cache(None, PER_EXPERT, PAGE, u64::MAX).unwrap(),
            Sizing::Off
        );
        assert_eq!(
            size_cache(os("0"), PER_EXPERT, PAGE, u64::MAX).unwrap(),
            Sizing::Off
        );
    }

    #[test]
    fn an_invalid_or_too_small_setting_is_refused() {
        for bad in ["", "1.5", "-1", "many"] {
            assert!(
                size_cache(os(bad), PER_EXPERT, PAGE, u64::MAX).is_err(),
                "{bad:?}"
            );
        }
        assert!(size_cache(os("18446744073709551615"), PER_EXPERT, PAGE, u64::MAX).is_err());
    }

    #[test]
    fn a_setting_below_the_smallest_cache_skips_it_like_a_small_budget() {
        for mib in ["1", "100"] {
            assert!(
                matches!(
                    size_cache(os(mib), PER_EXPERT, PAGE, u64::MAX).unwrap(),
                    Sizing::Skip(_)
                ),
                "{mib}"
            );
        }
    }

    #[test]
    fn the_allocation_shrinks_until_page_alignment_fits_and_gives_up_at_the_minimum() {
        // Each slot takes 1.25 units of room, so the arithmetic count overshoots the budget.
        let build = |slots: usize| Ok::<_, EngError>((slots, slots as u64 * 125 / 100));
        let fitted = fit_allocation(1000, 800, 10, build).unwrap().unwrap();
        assert!(fitted.1 <= 800 && fitted.0 < 1000 && fitted.3);
        assert!(fitted.0 >= 600, "shrinks in proportion, not to the floor");
        let exact = fit_allocation(100, 125, 10, build).unwrap().unwrap();
        assert_eq!((exact.0, exact.2, exact.3), (100, 100, false));
        assert!(fit_allocation(1000, 5, 10, build).unwrap().is_none());
        let failing =
            |_: usize| Err::<((), u64), _>(EngError::BadRequest("allocation failed".into()));
        assert!(fit_allocation(100, 125, 10, failing).is_err());
    }

    #[test]
    fn a_budget_below_the_setting_shrinks_the_cache_and_one_below_the_minimum_skips_it() {
        let wanted = size_cache(os("4096"), PER_EXPERT, PAGE, 8 << 30).unwrap();
        let Sizing::Use {
            slots,
            usable,
            reduced,
        } = wanted
        else {
            panic!("{wanted:?}")
        };
        assert!(!reduced);
        assert_eq!(usable, 4 << 30);
        let less = size_cache(os("4096"), PER_EXPERT, PAGE, 1 << 30).unwrap();
        let Sizing::Use {
            slots: fewer,
            usable,
            reduced,
        } = less
        else {
            panic!("{less:?}")
        };
        assert!(reduced && fewer < slots && usable == 1 << 30);
        assert!(fewer >= ROUTED_LAYERS * ROUTES);
        assert!(matches!(
            size_cache(os("4096"), PER_EXPERT, PAGE, minimum() - 1).unwrap(),
            Sizing::Skip(_)
        ));
        assert!(matches!(
            size_cache(os("4096"), PER_EXPERT, PAGE, minimum()).unwrap(),
            Sizing::Use { reduced: true, .. }
        ));
    }

    fn entry(expert: u32) -> Option<Entry> {
        Some(Entry { expert, used: 1 })
    }

    #[test]
    fn a_failed_read_leaves_the_written_slots_empty() {
        let mut slots = vec![entry(1), entry(2), entry(3)];
        let mut next = slots.clone();
        next[1] = entry(9);
        let failed = commit_reads(&mut slots, next, &[(9, 1)], || {
            Err(EngError::BadRequest("read failed".into()))
        });
        assert!(failed.is_err());
        assert_eq!(slots, [entry(1), None, entry(3)]);
    }

    #[test]
    fn a_complete_read_publishes_the_new_slots() {
        let mut slots = vec![entry(1), entry(2), entry(3)];
        let mut next = slots.clone();
        next[1] = entry(9);
        commit_reads(&mut slots, next.clone(), &[(9, 1)], || Ok(())).unwrap();
        assert_eq!(slots, next);
    }
}
