//! The slot policies of the expert-bank seam: the recency-list LRU row and the
//! replace-on-miss policy the first bank serves. The geometry parameterizes experts,
//! slots and record size, and the policy keeps the three rules below (the experts of
//! a pass are never victims, slot 0 is not a victim in a pass with a miss, and the
//! slot row it publishes is the classifier's table).
//!
//! Known residue of the move: `dump_usage` still writes the profile in the layout-key
//! format and stderr label its first caller shipped (`model.language_model.layers.{}.mlp`,
//! the usage-dump prefix); the next promotion slice moves both to the caller. Beside the
//! trait-served policies sits the tick-stamp sticky LRU (`LayerSlots`), which runs its own
//! assign protocol and publishes its slot row without the `SlotPolicy` trait.

use super::{BankGeometry, PassPlan, SlotMove, SlotPolicy, SlotTable, bank_error};
use crate::EngError;

#[derive(Debug, Clone)]
struct LayerLru {
    /// Slot of every expert or -1.
    slot_of: Vec<i32>,
    /// Expert of every slot.
    resident: Vec<u16>,
    /// Slots, least recently used first.
    recency: Vec<u16>,
}

impl LayerLru {
    /// Runs the experts of `order` through the layer: a hit becomes the most recent, a miss takes
    /// the least recent slot whose expert `demanded` does not protect. Fails only when everything
    /// is protected, which the callers' capacity check rules out.
    fn apply(&mut self, order: &[u16], demanded: &[bool]) -> Result<PassPlan, EngError> {
        let mut plan = PassPlan {
            distinct: order.len(),
            ..PassPlan::default()
        };
        for &expert in order {
            let slot = match u16::try_from(self.slot_of[usize::from(expert)]) {
                Ok(slot) => {
                    plan.hits += 1;
                    slot
                }
                Err(_) => {
                    let slot = self
                        .recency
                        .iter()
                        .copied()
                        .find(|&slot| !demanded[usize::from(self.resident[usize::from(slot)])])
                        .ok_or_else(|| bank_error("no unprotected victim".into()))?;
                    let outgoing = self.resident[usize::from(slot)];
                    self.slot_of[usize::from(outgoing)] = -1;
                    self.slot_of[usize::from(expert)] = i32::from(slot);
                    self.resident[usize::from(slot)] = expert;
                    plan.moves.push(SlotMove {
                        slot,
                        incoming: expert,
                        outgoing,
                    });
                    slot
                }
            };
            let at = self
                .recency
                .iter()
                .position(|&other| other == slot)
                .ok_or_else(|| bank_error("invalid recency order".into()))?;
            self.recency.remove(at);
            self.recency.push(slot);
        }
        Ok(plan)
    }
}

/// Per-layer replace-on-miss LRU with the experts of the pass protected, sized by the
/// geometry. Every miss is admitted; there is no swap limit.
#[derive(Debug, Clone)]
pub(crate) struct ReplaceOnMissLru {
    geometry: BankGeometry,
    layers: Vec<LayerLru>,
    /// Per-layer pick counts, written to `usage_out` on drop so a run can produce the popularity
    /// profile of its own workload (the read side is the profile path the owning family
    /// configures, the `expert_stats` engine key). `None` when no write path is set.
    usage: Option<Vec<Vec<u64>>>,
    /// Where the dump goes: the profile-path override the owning family configures, or the
    /// engine's `expert_stats` path (which then also feeds the next start's fill).
    usage_out: Option<std::path::PathBuf>,
    /// Per-layer miss counts of the same dump: where the slot capacity actually binds.
    miss_counts: Vec<u64>,
}

impl ReplaceOnMissLru {
    /// Starts from a filled bank: `rows[layer][expert]` is the slot of the expert or -1, with exactly
    /// `slots` experts held (`TierPlan::hot_set`'s table). The least popular holder (by `counts`,
    /// ties to the higher id) is the first victim.
    pub(crate) fn from_fill(
        geometry: BankGeometry,
        rows: &[&[i32]],
        counts: &[Vec<u64>],
        usage_out: Option<&std::path::Path>,
    ) -> Result<Self, EngError> {
        if rows.len() != geometry.layers || counts.len() != geometry.layers {
            return Err(bank_error(format!(
                "{} rows and {} count rows for {} layers",
                rows.len(),
                counts.len(),
                geometry.layers
            )));
        }
        let mut layers = Vec::with_capacity(geometry.layers);
        for (layer, (row, count)) in rows.iter().zip(counts).enumerate() {
            if row.len() != geometry.experts || count.len() != geometry.experts {
                return Err(bank_error(format!(
                    "layer {layer}: a row of {} and counts of {} for {} experts",
                    row.len(),
                    count.len(),
                    geometry.experts
                )));
            }
            let mut resident = vec![u16::MAX; geometry.slots];
            for (expert, &slot) in row.iter().enumerate() {
                if slot < 0 {
                    continue;
                }
                let held = resident.get_mut(slot as usize).ok_or_else(|| {
                    bank_error(format!("layer {layer}: slot {slot} is outside the bank"))
                })?;
                if *held != u16::MAX {
                    return Err(bank_error(format!("layer {layer}: slot {slot} twice")));
                }
                *held = expert as u16;
            }
            if resident.contains(&u16::MAX) {
                return Err(bank_error(format!("layer {layer}: a slot is not filled")));
            }
            let mut recency: Vec<u16> = (0..geometry.slots as u16).collect();
            recency.sort_by_key(|&slot| {
                let expert = resident[usize::from(slot)];
                (count[usize::from(expert)], std::cmp::Reverse(expert))
            });
            layers.push(LayerLru {
                slot_of: row.to_vec(),
                resident,
                recency,
            });
        }
        let usage_out = usage_out.map(std::path::Path::to_path_buf);
        let usage = usage_out
            .as_ref()
            .map(|_| vec![vec![0u64; geometry.experts]; geometry.layers]);
        let miss_counts = vec![0u64; geometry.layers];
        Ok(Self {
            geometry,
            layers,
            usage,
            usage_out,
            miss_counts,
        })
    }

    /// Writes the collected pick counts in the `expert_stats` layer-key format the profile
    /// reader takes, so the file feeds straight back through the profile path.
    fn dump_usage(&self) {
        let Some(path) = self.usage_out.as_deref() else {
            return;
        };
        let Some(usage) = &self.usage else { return };
        let layers: serde_json::Map<String, serde_json::Value> = usage
            .iter()
            .enumerate()
            .map(|(layer, counts)| {
                (
                    format!("model.language_model.layers.{layer}.mlp"),
                    serde_json::json!(counts),
                )
            })
            .collect();
        let usage_misses = self.miss_counts.clone();
        let body =
            serde_json::json!({"schema": "fifi-usage/1", "layers": layers, "misses": usage_misses});
        // Write-and-rename: a crash mid-dump must not leave a truncated file the next start
        // refuses. The temp file sits next to the target so the rename stays on one filesystem.
        let temp = path.with_extension("json.tmp");
        match serde_json::to_vec(&body) {
            Ok(bytes) => {
                let written =
                    std::fs::write(&temp, bytes).and_then(|()| std::fs::rename(&temp, path));
                if let Err(e) = written {
                    std::fs::remove_file(&temp).ok();
                    eprintln!("flash-next cuda: expert usage dump {}: {e}", path.display());
                }
            }
            // The body is maps of u64 vectors: serialization cannot fail. Skip the write
            // instead of emitting an empty file a next start would refuse.
            Err(e) => eprintln!("flash-next cuda: expert usage dump {}: {e}", path.display()),
        }
    }

    fn layer(&self, layer: usize) -> Result<&LayerLru, EngError> {
        self.layers
            .get(layer)
            .ok_or_else(|| bank_error(format!("layer {layer} is outside the bank")))
    }

    fn check_expert(&self, id: i32) -> Result<u16, EngError> {
        usize::try_from(id)
            .ok()
            .filter(|&expert| expert < self.geometry.experts)
            .map(|expert| expert as u16)
            .ok_or_else(|| bank_error(format!("expert {id} is outside the layer")))
    }
}

impl Drop for ReplaceOnMissLru {
    fn drop(&mut self) {
        self.dump_usage();
    }
}

impl SlotTable for ReplaceOnMissLru {
    fn slot_row(&self, layer: usize) -> &[i32] {
        self.layers.get(layer).map_or(&[], |lru| &lru.slot_of)
    }

    fn expert_in_slot(&self, layer: usize, slot: usize) -> Option<u16> {
        self.layers.get(layer)?.resident.get(slot).copied()
    }
}

impl SlotPolicy for ReplaceOnMissLru {
    fn plan_pass(&mut self, layer: usize, picks: &[i32]) -> Result<PassPlan, EngError> {
        let current = self.layer(layer)?;
        let mut demanded = vec![false; self.geometry.experts];
        let mut order = Vec::with_capacity(picks.len());
        // Pick ids deferred to after `current`'s borrow ends, for the usage profile.
        let mut picked = Vec::with_capacity(picks.len());
        for &id in picks {
            if id == -1 {
                continue;
            }
            let expert = self.check_expert(id)?;
            picked.push(usize::from(expert));
            if !std::mem::replace(&mut demanded[usize::from(expert)], true) {
                order.push(expert);
            }
        }
        if order
            .iter()
            .any(|&expert| current.slot_of[usize::from(expert)] < 0)
        {
            // The filler: slot 0 is read by the resident pass while the copies run.
            demanded[usize::from(current.resident[0])] = true;
        }
        let protected = demanded.iter().filter(|&&flag| flag).count();
        if protected > self.geometry.slots {
            return Err(bank_error(format!(
                "{protected} protected experts exceed the {} slots",
                self.geometry.slots
            )));
        }
        let mut next = current.clone();
        let _ = current; // end the borrow: the write-back below takes the layer slot
        if let Some(usage) = &mut self.usage {
            for expert in picked {
                usage[layer][expert] += 1;
            }
        }
        let plan = next.apply(&order, &demanded)?;
        self.miss_counts[layer] += plan.moves.len() as u64;
        self.layers[layer] = next;
        Ok(plan)
    }

    fn admit(&mut self, layer: usize, experts_by_use: &[u16]) -> Result<PassPlan, EngError> {
        let current = self.layer(layer)?;
        let mut demanded = vec![false; self.geometry.experts];
        let mut newest_first = Vec::new();
        for &expert in experts_by_use.iter().rev() {
            self.check_expert(i32::from(expert))?;
            if newest_first.len() < self.geometry.slots
                && !std::mem::replace(&mut demanded[usize::from(expert)], true)
            {
                newest_first.push(expert);
            }
        }
        newest_first.reverse();
        let mut next = current.clone();
        let plan = next.apply(&newest_first, &demanded)?;
        self.layers[layer] = next;
        Ok(plan)
    }
}

// ---------------------------------------------------------------------------------------------
// The tick-stamp LRU with a sticky hot set.
// ---------------------------------------------------------------------------------------------

/// The least-recently-used slot table of one layer: which expert sits in which slot of the
/// layer's three banks. Pure — the device copies live one level up, so the policy is testable
/// without a GPU.
///
/// One slot index addresses member `i` of all three parts' device banks and of the store's
/// part stores alike, so the table is one array per layer, not per part.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerSlots {
    experts: usize,
    pub(in crate::family) slots: usize,
    /// How many of the layer's hottest experts (by decayed use count) stay pinned: a miss
    /// evicts a non-hot slot first, so repeatedly routed experts stop refilling. Zero is
    /// plain LRU. Clamped to `slots - 1`; ties at the threshold can still mark more experts
    /// hot than that, and the victim's fallback — not the clamp — guarantees progress.
    sticky: usize,
    /// Expert -> slot, [`NO_SLOT`] when absent.
    pub(in crate::family) slot_of: Vec<i32>,
    /// Slot -> expert, [`EMPTY`] when the slot never filled.
    pub(in crate::family) expert_of: Vec<u16>,
    last_used: Vec<u64>,
    /// Decayed use count per expert; the sticky set is its top-`sticky`. Halved every
    /// [`DECAY_CALLS`] calls so the set tracks a drifting workload.
    counts: Vec<u32>,
    /// Cached top-`sticky` membership, recomputed at each decay.
    hot: Vec<bool>,
    pub(in crate::family) tick: u64,
}

/// Calls between use-count halvings. 64 calls ≈ 8 tokens of one layer's routing at top-8:
/// the replay study on a captured trace (384-token greedy decode) put the adaptive
/// policy's best miss reduction (9.5% fewer than LRU at k=48) at this interval; longer
/// intervals only lag a drifting workload, shorter ones thrash the hot set.
pub(in crate::family) const DECAY_CALLS: u64 = 64;

/// The absent marker of the slot -> expert direction.
pub(in crate::family) const EMPTY: u16 = u16::MAX;
/// The absent marker of the expert -> slot direction: the slot row the shared seam
/// publishes (`SlotTable::slot_row`) is exactly this array.
pub(in crate::family) const NO_SLOT: i32 = -1;

impl LayerSlots {
    pub(in crate::family) fn new(experts: usize, slots: usize, sticky: usize) -> Self {
        let sticky = sticky.min(slots.saturating_sub(1));
        // The decay reads `sorted[self.sticky - 1]`; `decide()` keeps slots <= experts,
        // pinned here so a future caller change cannot move the panic boundary.
        debug_assert!(sticky <= experts);
        Self {
            experts,
            slots,
            sticky,
            slot_of: vec![NO_SLOT; experts],
            expert_of: vec![EMPTY; slots],
            last_used: vec![0; slots],
            counts: vec![0; experts],
            hot: vec![false; experts],
            tick: 0,
        }
    }

    /// Halves every use count and rebuilds the hot set from the survivors: the `sticky`
    /// largest counts, ties included. Called on the decay boundary only; with no sticky
    /// set the counts are dead weight and the halving is skipped.
    fn decay(&mut self) {
        if self.sticky == 0 {
            return;
        }
        for count in &mut self.counts {
            *count >>= 1;
        }
        let mut sorted = self.counts.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let threshold = sorted[self.sticky - 1];
        for (expert, &count) in self.counts.iter().enumerate() {
            self.hot[expert] = count >= threshold && count > 0;
        }
    }

    /// Maps `picks` (this layer's routed experts of one token, distinct) to slot indices,
    /// evicting for every miss the least-recently-used slot that no pick of THIS call uses —
    /// all picks must be resident together, because the launches of one block read them all.
    /// Returns the slots in pick order and, for each miss, the slot whose device bytes the
    /// caller must refill from the store.
    pub(in crate::family) fn assign(
        &mut self,
        picks: &[usize],
    ) -> Result<(Vec<usize>, Vec<usize>), EngError> {
        // Atomic, the seam's rule: an Err changes nothing. The pass runs on a scratch
        // clone (the table is KiBs — noise next to the three record copies one miss
        // issues) and only lands on Ok, so a mid-stream refusal leaves the live table
        // exactly as the call found it.
        let mut scratch = self.clone();
        let outcome = scratch.assign_steps(picks)?;
        *self = scratch;
        Ok(outcome)
    }

    fn assign_steps(&mut self, picks: &[usize]) -> Result<(Vec<usize>, Vec<usize>), EngError> {
        self.tick += 1;
        if self.tick.is_multiple_of(DECAY_CALLS) {
            self.decay();
        }
        let mut slots = Vec::with_capacity(picks.len());
        let mut misses = Vec::new();
        for &expert in picks {
            if expert >= self.experts {
                return Err(EngError::BadRequest(format!(
                    "EXL3 expert bank: pick {expert} exceeds the {} experts of the layer",
                    self.experts
                )));
            }
            if self.sticky > 0 {
                self.counts[expert] = self.counts[expert].saturating_add(1);
            }
            let slot = match self.slot_of[expert] {
                s if (0..self.slots as i32).contains(&s) => s as usize,
                _ => self.victim(&slots)?,
            };
            let miss = self.slot_of[expert] == NO_SLOT;
            if miss {
                let evicted = self.expert_of[slot];
                if evicted != EMPTY {
                    self.slot_of[usize::from(evicted)] = NO_SLOT;
                }
                self.expert_of[slot] = u16::try_from(expert).map_err(|_| {
                    EngError::BadRequest(format!(
                        "EXL3 expert bank: expert {expert} does not fit the slot table"
                    ))
                })?;
                self.slot_of[expert] = i32::try_from(slot).expect("slots fit i32");
                misses.push(slot);
            }
            slots.push(slot);
        }
        for &slot in &slots {
            self.last_used[slot] = self.tick;
        }
        Ok((slots, misses))
    }

    /// The next slot for a miss: an empty one, else the least-recently-used slot no pick of
    /// this call took. A slot holding a hot (sticky) expert is skipped unless every
    /// untouched slot is hot — pinning yields to progress, never to a refusal.
    fn victim(&self, taken: &[usize]) -> Result<usize, EngError> {
        if let Some(slot) = (0..self.slots).find(|&slot| self.expert_of[slot] == EMPTY) {
            return Ok(slot);
        }
        let all = |hot: bool| {
            (0..self.slots)
                .filter(|&slot| {
                    !taken.contains(&slot) && self.hot[usize::from(self.expert_of[slot])] == hot
                })
                .min_by_key(|&slot| self.last_used[slot])
        };
        all(false).or_else(|| all(true)).ok_or_else(|| {
            EngError::BadRequest(format!(
                "EXL3 expert bank: {} slots cannot hold the {} picks of one call \
                     (a decode token, or a prefill sub-block's rows together)",
                self.slots,
                taken.len() + 1
            ))
        })
    }

    /// The expert a slot holds (for the store read of a miss copy).
    pub(in crate::family) fn expert_of(&self, slot: usize) -> usize {
        usize::from(self.expert_of[slot])
    }

    /// The torn-refill response: drop `slots` from both directions of the table so the
    /// next call treats them as misses and refills. A failed record write leaves a
    /// slot's bytes untrustworthy while the table still names its incoming expert, and
    /// restoring the pre-call table would be wrong — an evicted expert's bytes may be
    /// half overwritten — so the slot simply reads absent in both directions.
    /// The seam's read of the expert-to-slot row (`SlotTable::slot_row`): exactly the
    /// `slot_of` array, `NO_SLOT` where a expert is absent.
    pub(in crate::family) fn slot_row(&self) -> &[i32] {
        self.slot_of.as_slice()
    }

    /// The seam's read of the slot-to-expert direction: `None` where the slot never filled.
    pub(in crate::family) fn expert_in_slot(&self, slot: usize) -> Option<u16> {
        let expert = self.expert_of.get(slot).copied()?;
        (expert != EMPTY).then_some(expert)
    }

    pub(in crate::family) fn invalidate(&mut self, slots: &[usize]) {
        for &slot in slots {
            let expert = self.expert_of[slot];
            if expert != EMPTY {
                self.slot_of[usize::from(expert)] = NO_SLOT;
            }
            self.expert_of[slot] = EMPTY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- the slot table's own tests, moved with the type ---

    /// The first picks fill the empty slots in order; a repeat of the same picks is all hits
    /// with the same slots and no misses.
    #[test]
    fn slot_table_fills_in_order_then_hits() {
        let mut table = LayerSlots::new(64, 8, 0);
        let picks = [3usize, 41, 7, 19];
        let (slots, misses) = table.assign(&picks).unwrap();
        assert_eq!(slots, [0, 1, 2, 3]);
        assert_eq!(misses, [0, 1, 2, 3]);
        let (slots, misses) = table.assign(&picks).unwrap();
        assert_eq!(slots, [0, 1, 2, 3]);
        assert!(misses.is_empty());
    }

    /// A miss evicts the least-recently-used slot no pick of the call uses: touching expert 3
    /// again makes expert 41's slot the victim, not 3's.
    #[test]
    fn slot_table_evicts_least_recently_used() {
        let mut table = LayerSlots::new(64, 8, 0);
        table.assign(&[3, 41, 5, 6, 7, 8, 9, 10]).unwrap();
        table.assign(&[3]).unwrap(); // touch 3; 41 is now the least-recently-used
        let (slots, misses) = table.assign(&[50]).unwrap();
        assert_eq!(slots, [1], "the victim is 41's slot");
        assert_eq!(misses, [1]);
        // 41 must miss again; it lands in some other slot.
        let (slots, misses) = table.assign(&[41]).unwrap();
        assert_eq!(misses.len(), 1);
        assert_ne!(
            slots[0], 1,
            "41 cannot return to the slot it was evicted from"
        );
    }

    /// Sticky pinning: after a hot set emerges (enough calls to cross the decay boundary),
    /// a cold sweep cannot evict it, while plain LRU loses the same experts and re-misses.
    #[test]
    fn slot_table_sticky_keeps_the_hot_set_through_a_cold_sweep() {
        fn sweep_misses(sticky: usize) -> usize {
            let mut table = LayerSlots::new(64, 8, sticky);
            // The hot set: experts 0-3, routed twice per call, long enough that the
            // repeated halvings reach the count steady state and promote them past any
            // cold count.
            for _ in 0..600 {
                table.assign(&[0, 1, 2, 3]).unwrap();
                table.assign(&[0, 1, 2, 3]).unwrap();
            }
            // The cold sweep: distinct experts the hot calls never touch.
            let mut misses = 0;
            for cold in 20..44 {
                misses += table.assign(&[cold]).unwrap().1.len();
            }
            // The hot set returns: sticky keeps it resident.
            misses += table.assign(&[0, 1, 2, 3]).unwrap().1.len();
            misses
        }
        let lru = sweep_misses(0);
        let sticky = sweep_misses(4);
        assert_eq!(lru, 24 + 4, "plain LRU re-misses the whole hot set");
        assert_eq!(sticky, 24, "the sweep misses, the hot set does not");
    }

    /// Pinning yields to progress: when ties at the decay threshold mark every resident
    /// expert hot (sticky=1, eight equally-counted residents), a new pick still resolves —
    /// the victim falls back to plain LRU instead of refusing.
    #[test]
    fn slot_table_sticky_falls_back_to_lru_when_every_slot_is_hot() {
        let mut table = LayerSlots::new(64, 8, 1);
        // Eight equal-count residents: the tie marks them all hot at the first decay.
        for _ in 0..128 {
            table.assign(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        }
        assert!(
            table.hot[..8].iter().all(|&hot| hot),
            "the tie marks all hot"
        );
        // A cold pick must still seat: the fallback evicts a hot slot rather than refusing.
        let (slots, misses) = table.assign(&[40]).unwrap();
        assert_eq!(misses.len(), 1);
        assert_eq!(slots.len(), 1);
    }

    /// A token whose picks are all new may replace every resident: the victims hold old
    /// experts, which no new pick uses. A token with more picks than slots is refused.
    #[test]
    fn slot_table_replaces_everything_then_refuses_overload() {
        let mut table = LayerSlots::new(64, 4, 0);
        table.assign(&[0, 1, 2, 3]).unwrap();
        let (slots, misses) = table.assign(&[10, 11, 12, 13]).unwrap();
        assert_eq!(misses.len(), 4);
        assert_eq!(slots.len(), 4);
        assert!((0..4).all(|slot| table.expert_of(slot) >= 10));
        let error = format!("{:?}", table.assign(&[20, 21, 22, 23, 24]).unwrap_err());
        assert!(error.contains("cannot hold"), "{error}");
    }

    /// A pick outside the layer's experts is refused by name.
    #[test]
    fn slot_table_refuses_out_of_range_picks() {
        let mut table = LayerSlots::new(64, 8, 0);
        let error = format!("{:?}", table.assign(&[64]).unwrap_err());
        assert!(error.contains("pick 64"), "{error}");
    }

    const EXPERTS: usize = 512;
    const SLOTS: usize = 48;
    const PICKS: usize = 10;
    const CALLS: usize = 200;
    const PLAIN_MISSES: usize = 1790;
    const STICKY_MISSES: usize = 1779;

    /// The pick stream: the xorshift64 generator the bank pick-replay goldens use, with
    /// this test's own seed, mapped per call to router faults (-1) and in-range experts,
    /// so the sticky policy serves a 512-expert, 10-pick route with no family module
    /// involved.
    fn call_picks(call: usize) -> Vec<i32> {
        let mut state = 0x5dee_ce6d_e87a_u64 ^ (call as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (0..PICKS)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let raw = (state >> 33) as i64 % 532 - 10;
                if raw < 0 {
                    -1
                } else if raw > 511 {
                    (raw - 512) as i32
                } else {
                    raw as i32
                }
            })
            .collect()
    }

    fn distinct_in_range(picks: &[i32]) -> Vec<usize> {
        let mut seen = vec![false; EXPERTS];
        let mut order = Vec::new();
        for &pick in picks {
            if pick < 0 || seen.get(pick as usize).is_none() {
                continue;
            }
            let expert = pick as usize;
            if !std::mem::replace(&mut seen[expert], true) {
                order.push(expert);
            }
        }
        order
    }

    /// Replays the wide top-k route through the sticky policy and returns the total
    /// misses with the final table. Deterministic: no clock, no randomness.
    fn replay(sticky: usize) -> (usize, LayerSlots) {
        let mut table = LayerSlots::new(EXPERTS, SLOTS, sticky);
        let mut misses = 0;
        for call in 0..CALLS {
            let picks = distinct_in_range(&call_picks(call));
            let (slots, missed) = table.assign(&picks).expect("the picks fit the slots");
            assert_eq!(slots.len(), picks.len());
            // Every pick landed on its own expert's slot, and the missed victims are
            // distinct slots bounded by the picks of the call.
            for (i, &slot) in slots.iter().enumerate() {
                assert_eq!(table.expert_in_slot(slot), Some(picks[i] as u16));
            }
            assert!(missed.len() <= picks.len());
            let mut distinct = missed.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(missed.len(), distinct.len(), "a victim slot repeats");
            misses += missed.len();
        }
        (misses, table)
    }

    #[test]
    fn the_sticky_policy_serves_a_wide_top_k_route() {
        let (plain, plain_table) = replay(0);
        let (sticky, sticky_table) = replay(8);
        // Pinned totals for this fixed stream: both arms miss (the route is wider than
        // the 48 slots), the sticky arm misses strictly less (the stream's extra weight on
        // experts 0..9 is exactly the hot set sticky=8 protects across the decays at
        // ticks 64/128/192), and both replays are deterministic.
        assert_eq!(plain, PLAIN_MISSES);
        assert_eq!(sticky, STICKY_MISSES);
        assert!(plain > 0);
        assert!(sticky > 0);
        assert!(sticky < plain);
        // Both arms serve every call and leave a consistent table: every held expert's
        // row names its slot, and every held slot's row names its expert.
        for table in [&plain_table, &sticky_table] {
            for (expert, &slot) in table.slot_row().iter().enumerate() {
                if slot >= 0 {
                    assert_eq!(table.expert_in_slot(slot as usize), Some(expert as u16));
                }
            }
            for slot in 0..SLOTS {
                if let Some(expert) = table.expert_in_slot(slot) {
                    assert_eq!(table.slot_row()[usize::from(expert)], slot as i32);
                }
            }
        }
    }
}
