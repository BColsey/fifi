//! The slot policies of the expert-bank seam: the recency-list LRU row and the
//! replace-on-miss policy the first bank serves. The geometry parameterizes experts,
//! slots and record size, and the policy keeps the three rules below (the experts of
//! a pass are never victims, slot 0 is not a victim in a pass with a miss, and the
//! slot row it publishes is the classifier's table).
//!
//! Known residue of the move: `dump_usage` still writes the profile in the layout-key
//! format and stderr label its first caller shipped (`model.language_model.layers.{}.mlp`,
//! the usage-dump prefix); the next promotion slice moves both to the caller.

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
