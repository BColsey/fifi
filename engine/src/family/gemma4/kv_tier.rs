//! The Bolt engine's use of the shared on-disk prefix tier. A Bolt prefix is a host copy of the
//! cache cells `0..fed` of every physical layer (`BoltSnapshot`); the tier writes it out in
//! chunks behind the request, and a miss in the host store looks the prompt up on disk first and
//! reads a hit back into a host copy that the store then holds as an ordinary entry. The family
//! supplies the layout (one object per physical layer), its identity and its store; the policy
//! (when to write, how long a stop may wait, how a damaged entry falls back) is the tier's.
use super::resident::bolt::BoltSnapshot;
use crate::{
    EngError,
    family::{
        host_admit::HostAdmission,
        kv_disk::{
            Capture, DiskTier, FamilyContract, Held, Hit, KvDisk, LoadIdentity, MIN_POSITIONS,
            ObjectRole, PUMP_EMIT_BYTES, Reason, Resolved, Run, SlabLayout, SlotBytes, Stored,
            drain_bound, host_memory_is_short, hurry_flag, restore_report, retry_while_removed,
            spill_candidates, spill_picks, start_line, stop_picks,
        },
        session::{PrefixIdentity, PrefixStore, SaveKind},
    },
    sched::job::{PrefixSource, PrefixTier},
};
use std::{rc::Rc, time::Instant};

/// The store the Bolt engine keeps its host copies in.
pub(super) type Store = PrefixStore<BoltSnapshot, PrefixIdentity>;

/// The bytes of each physical layer's part of a snapshot of the given number of cells.
pub(super) type LayerBytes<'a> = &'a dyn Fn(usize) -> Vec<usize>;

/// What the shared tier calls this engine in its log lines.
const TIER_LABEL: &str = "gemma4 bolt";

/// Names the layout of a snapshot's bytes: bump it when `BoltSnapshot` stores them differently.
const LAYOUT_TAG: &str = "bolt-kv8-snapshot-v1";

/// What a Bolt single-card CUDA load fixes about the entries it writes.
const CONTRACT: FamilyContract = FamilyContract {
    adapter: "gemma4.bolt",
    backend: "cuda.single-card",
    snapshot_schema: "gemma4.bolt.kv8-cells.v1",
    payload_abi: "kv8-planes-by-physical-layer.v1",
    position_policy: "absolute-rope-from-zero",
    render_mode: "chat",
};

/// Host memory the engine leaves to the system before it keeps a copy of its own: what the
/// start plan of the other stores keeps.
const HOST_RESERVE: u64 = 1 << 30;

/// Lookups one restore may spend on entries that turn out damaged.
const PROMOTE_ATTEMPTS: usize = 3;

/// The layout of an entry: one object per physical layer, each the layer's part of the snapshot,
/// back to back in the order the snapshot stores them.
pub(super) fn entry_slab(layer_bytes: &[usize]) -> SlabLayout {
    let mut at = 0;
    let objects = layer_bytes
        .iter()
        .map(|&bytes| {
            let run = Run { slab: at, bytes };
            at += bytes;
            vec![run]
        })
        .collect();
    SlabLayout {
        objects,
        roles: vec![ObjectRole::Pages; layer_bytes.len()],
    }
}

fn capture_of(kind: SaveKind) -> Capture {
    match kind {
        SaveKind::Tip => Capture::Tip,
        SaveKind::Root | SaveKind::Periodic | SaveKind::Block => Capture::Periodic,
    }
}

fn kind_of(capture: Capture) -> SaveKind {
    match capture {
        Capture::Tip => SaveKind::Tip,
        Capture::Periodic => SaveKind::Periodic,
    }
}

/// A stored snapshot as a spill: `None` when it is too short to repay a read or its key is not
/// the cells plus the pending token.
fn held_of(key: &[u32], kind: SaveKind, snapshot: &BoltSnapshot, reason: Reason) -> Option<Held> {
    let fed = snapshot.fed();
    if fed < MIN_POSITIONS || key.len() != fed + 1 {
        return None;
    }
    let slot: Rc<dyn SlotBytes> = snapshot.hold();
    Some(Held {
        reason,
        slot,
        key: key.to_vec(),
        capture: capture_of(kind),
        positions: fed,
        bytes: snapshot.bytes() as u64,
    })
}

/// What names a Bolt load to an entry kept outside the process.
pub(super) fn load_identity(
    weights: &str,
    layer_bytes: &[usize],
    attention_split: bool,
) -> LoadIdentity {
    let (tokenizer, template) = crate::tokenizer::gemma::pinned_digests();
    LoadIdentity {
        weights: weights.to_owned(),
        tokenizer: crate::weights::sha256_hex(&tokenizer),
        template: crate::weights::sha256_hex(&template),
        numeric_path: format!("bolt-kv8 attention-split {attention_split}"),
        family: CONTRACT,
        layout: format!(
            "{LAYOUT_TAG}; {} physical layers; bytes per cell by layer {layer_bytes:?}",
            layer_bytes.len()
        ),
    }
}

/// The disk tier of one Bolt engine.
pub(super) struct BoltDisk {
    pub(super) tier: DiskTier,
}

impl BoltDisk {
    /// Opens the tier at `resolved`, or logs why it stays off. A directory that cannot be used
    /// never fails the start.
    pub(super) fn open(resolved: &Resolved, origin: &str, load: &LoadIdentity) -> Option<Self> {
        match KvDisk::open(&resolved.dir, resolved.quota_bytes, load) {
            Ok((cache, report)) => {
                eprintln!("{}", start_line(TIER_LABEL, resolved, origin, &report, 1));
                Some(Self {
                    tier: DiskTier::new(TIER_LABEL, cache, 1),
                })
            }
            Err(error) => {
                eprintln!(
                    "{TIER_LABEL}: kv disk tier off: {} cannot be used: {error}",
                    resolved.dir.display()
                );
                None
            }
        }
    }

    pub(super) fn pending(&self) -> bool {
        self.tier.pending()
    }

    /// Copies up to `budget` bytes of pending saves to the writer.
    fn pump(&mut self, budget: usize, layer_bytes: LayerBytes<'_>) {
        self.tier
            .pump(&|cells| Ok(entry_slab(&layer_bytes(cells))), budget);
    }

    pub(super) fn pump_emit(&mut self, layer_bytes: LayerBytes<'_>) {
        self.pump(PUMP_EMIT_BYTES, layer_bytes);
    }

    /// After a save: when the store holds as many entries as the watermark and the host's memory
    /// is short, queue the entries the next saves would push out. With memory to spare the disk
    /// is not touched.
    pub(super) fn spill_over_watermark(&mut self, store: &Store) {
        let stored = store.entries_len() + store.roots_len();
        if spill_candidates(stored, self.tier.retained) == 0 {
            return;
        }
        let largest = store
            .snapshots()
            .map(|snapshot| snapshot.bytes() as u64)
            .max()
            .unwrap_or(0);
        let available = HostAdmission::probe("a Bolt kv disk spill", "a Bolt prefix snapshot")
            .ok()
            .map(|admission| admission.available_budget());
        if !host_memory_is_short(available, HOST_RESERVE, largest) {
            return;
        }
        let want = spill_candidates(stored, self.tier.retained);
        let entries: Vec<_> = store.entries_lru().collect();
        let view: Vec<Stored<'_>> = entries
            .iter()
            .map(|&(key, kind, _)| Stored {
                key,
                tip: kind == SaveKind::Tip,
            })
            .collect();
        for at in spill_picks(&view, want) {
            let (key, kind, snapshot) = entries[at];
            if let Some(held) = held_of(key, kind, snapshot, Reason::Watermark) {
                self.tier.enqueue(held);
            }
        }
    }

    /// A graceful stop: writes the live tips that exist only in memory, newest first, within
    /// the bound the tier derives from their bytes and the disk's write rate.
    pub(super) fn drain(&mut self, store: &Store, layer_bytes: LayerBytes<'_>) {
        let entries: Vec<_> = store.entries_lru().collect();
        let view: Vec<Stored<'_>> = entries
            .iter()
            .map(|&(key, kind, _)| Stored {
                key,
                tip: kind == SaveKind::Tip,
            })
            .collect();
        let tips: Vec<Held> = stop_picks(&view)
            .into_iter()
            .filter_map(|at| {
                let (key, kind, snapshot) = entries[at];
                held_of(key, kind, snapshot, Reason::Stop)
            })
            .collect();
        let left_in_memory = entries.len().saturating_sub(tips.len());
        self.tier.drain(
            tips,
            left_in_memory,
            &|cells| Ok(entry_slab(&layer_bytes(cells))),
            hurry_flag(),
            &drain_bound,
        );
    }

    /// Before a restore: when the disk holds a longer stored prefix of `prompt` than the host
    /// store does, reads it (verified in full) into host memory and stores it as an ordinary
    /// entry, so the restore that follows finds it. A miss changes nothing.
    pub(super) fn promote(
        &mut self,
        store: &mut Store,
        identity: &PrefixIdentity,
        prompt: &[u32],
        context_cells: usize,
        layer_bytes: LayerBytes<'_>,
    ) {
        self.tier.hit = None;
        retry_while_removed(PROMOTE_ATTEMPTS, || {
            self.promote_once(store, identity, prompt, context_cells, layer_bytes)
        });
    }

    /// One lookup and read. Returns true when the entry failed its integrity check and was
    /// removed, so another candidate may serve the prompt.
    fn promote_once(
        &mut self,
        store: &mut Store,
        identity: &PrefixIdentity,
        prompt: &[u32],
        context_cells: usize,
        layer_bytes: LayerBytes<'_>,
    ) -> bool {
        let tier = &mut self.tier;
        let Some(candidate) = tier.cache.lookup(prompt) else {
            return false;
        };
        if store
            .longest_candidate_len(prompt, identity)
            .is_some_and(|len| len >= candidate.key_len)
        {
            return false;
        }
        let began = Instant::now();
        let cells = candidate.key_len - 1;
        if cells > context_cells {
            eprintln!(
                "{TIER_LABEL}: kv disk entry of {cells} positions does not fit the {context_cells} position context"
            );
            return false;
        }
        let sizes = layer_bytes(cells);
        let total: usize = sizes.iter().sum();
        let available = HostAdmission::probe("a Bolt kv disk restore", "a Bolt prefix snapshot")
            .ok()
            .map(|admission| admission.available_budget());
        if available.is_some_and(|available| total as u64 > available) {
            tier.counters.missed += 1;
            eprintln!(
                "{TIER_LABEL}: kv disk restore skipped: the host budget cannot hold {total} B"
            );
            return false;
        }
        let mut data = Vec::new();
        if data.try_reserve_exact(total).is_err() {
            tier.counters.missed += 1;
            eprintln!("{TIER_LABEL}: kv disk restore skipped: {total} B do not fit in host memory");
            return false;
        }
        data.resize(total, 0u8);
        let shape = entry_slab(&sizes);
        let stats = match tier.read_into(&candidate, prompt, &shape, &mut |slab, part| {
            let end = slab
                .checked_add(part.len())
                .filter(|end| *end <= data.len())
                .ok_or_else(|| EngError::BadRequest("kv disk: a part past the snapshot".into()))?;
            data[slab..end].copy_from_slice(part);
            Ok(())
        }) {
            Ok(stats) => stats,
            Err(removed) => return removed,
        };
        let Some(&pending) = prompt[..candidate.key_len].last() else {
            return false;
        };
        let stored = store.save_loaded(
            &prompt[..candidate.key_len],
            pending,
            identity.clone(),
            kind_of(candidate.capture),
            BoltSnapshot::from_parts(cells, data),
        );
        match stored {
            Ok(true) => {
                tier.counters.restored += 1;
                tier.hit = Some(Hit {
                    key_len: candidate.key_len,
                    bytes: stats.bytes,
                    ms: stats.ms,
                });
                eprintln!(
                    "{TIER_LABEL}: kv disk read {cells} positions, {} B, {:.1} ms (verified), {:.1} ms with the lookup",
                    stats.bytes,
                    stats.ms,
                    began.elapsed().as_secs_f64() * 1000.0
                );
            }
            Ok(false) => {
                tier.counters.missed += 1;
                eprintln!(
                    "{TIER_LABEL}: kv disk entry read but the store refused it; the prompt is prefilled"
                );
            }
            Err(error) => {
                tier.counters.missed += 1;
                eprintln!("{TIER_LABEL}: kv disk entry could not be stored: {error:?}");
            }
        }
        false
    }
}

/// The tier facts of the restore that just covered `covered` tokens.
pub(super) fn restore_tier(disk: Option<&mut BoltDisk>, covered: usize) -> PrefixTier {
    restore_report(
        disk.map(|disk| &mut disk.tier),
        covered,
        None,
        PrefixSource::Pinned,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::session::{PrefixStore, StoreLimits};

    fn identity() -> PrefixIdentity {
        PrefixIdentity {
            weights: "w".into(),
            tokenizer: "t".into(),
            template: "p".into(),
        }
    }

    fn key(len: usize) -> Vec<u32> {
        (0..len as u32).collect()
    }

    #[test]
    fn an_entry_has_one_object_per_physical_layer_back_to_back() {
        let shape = entry_slab(&[100, 40, 60]);
        assert_eq!(shape.object_bytes(), vec![100, 40, 60]);
        assert_eq!(
            shape.flat(),
            vec![
                Run {
                    slab: 0,
                    bytes: 100
                },
                Run {
                    slab: 100,
                    bytes: 40
                },
                Run {
                    slab: 140,
                    bytes: 60
                },
            ]
        );
        assert!(shape.roles.iter().all(|role| *role == ObjectRole::Pages));
    }

    #[test]
    fn a_snapshot_is_a_spill_only_when_its_key_is_its_cells_plus_the_pending_token() {
        let snapshot = BoltSnapshot::from_parts(2_000, vec![7u8; 4_000]);
        let held = held_of(&key(2_001), SaveKind::Tip, &snapshot, Reason::Stop).unwrap();
        assert_eq!((held.positions, held.bytes), (2_000, 4_000));
        assert_eq!(held.capture, Capture::Tip);
        let mut one = [0u8; 4];
        held.slot.read(3_998, &mut one[..2]).unwrap();
        assert_eq!(&one[..2], &[7, 7]);
        // A key of the wrong length, or too few cells, is not offered.
        assert!(held_of(&key(2_000), SaveKind::Tip, &snapshot, Reason::Stop).is_none());
        let short = BoltSnapshot::from_parts(10, vec![0u8; 10]);
        assert!(held_of(&key(11), SaveKind::Tip, &short, Reason::Stop).is_none());
    }

    #[test]
    fn the_load_identity_names_the_weights_the_pinned_text_and_the_layout() {
        let one = load_identity("w1", &[8, 16], true);
        assert!(one.is_complete());
        let other_weights = load_identity("w2", &[8, 16], true);
        let other_layout = load_identity("w1", &[8, 24], true);
        let other_split = load_identity("w1", &[8, 16], false);
        let namespace = |load: &LoadIdentity| {
            format!("{:?}", (&load.weights, &load.layout, &load.numeric_path))
        };
        assert_ne!(namespace(&one), namespace(&other_weights));
        assert_ne!(namespace(&one), namespace(&other_layout));
        assert_ne!(namespace(&one), namespace(&other_split));
    }

    #[test]
    fn a_store_view_gives_the_stop_its_live_tips() {
        let mut store: Store = PrefixStore::with_limits(StoreLimits {
            roots: 0,
            entries: 1,
        });
        let id = identity();
        store
            .save_loaded(
                &key(2_001),
                2_000,
                id.clone(),
                SaveKind::Tip,
                BoltSnapshot::from_parts(2_000, vec![1u8; 64]),
            )
            .unwrap();
        let entries: Vec<_> = store.entries_lru().collect();
        let view: Vec<Stored<'_>> = entries
            .iter()
            .map(|&(key, kind, _)| Stored {
                key,
                tip: kind == SaveKind::Tip,
            })
            .collect();
        assert_eq!(stop_picks(&view), vec![0]);
    }
}
