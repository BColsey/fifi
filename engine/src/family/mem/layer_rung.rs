//! The layer rung: the last rung of the one memory ladder, for any GGUF child, dense or MoE,
//! whose weights fit in neither the card nor the host.
//!
//! A decoder layer is one contiguous extent of the GGUF, so a layer that is not resident is read
//! in place from the file (no second copy on disk, no conversion pass), through a small pinned
//! chunk ring, into one of a few staging slots on the card. The plan therefore never needs disk
//! bytes beyond the model file that is already there, and it refuses only when the card cannot
//! hold what no placement can move off it: the non-layer tensors that run on the card, the
//! per-request state (the KV cache and work buffers), the reserve, and one layer beside them.
//!
//! The solver is pure arithmetic over a table of layer sizes read from the GGUF header, so any
//! family supplies its layer table and nothing else; the reserve and the state are the caller's
//! (the device policy and the context the start chose).

/// Staging slots on the card: one layer being visited while the next is copied in.
pub(crate) const STAGING_SLOTS: usize = 2;

/// The pinned host ring a streamed layer is read through, in chunks, on its way to a staging slot.
pub(crate) const STAGE_CHUNK_RING_BYTES: u64 = 64 << 20;

/// What the solver reads about a model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RungModel {
    /// Bytes of every decoder layer's tensors, in layer order (experts included: the unit is the
    /// whole layer).
    pub(crate) layers: Vec<u64>,
    /// Bytes of the non-layer tensors that run on the card (head, output norm, small tables).
    pub(crate) card_fixed_bytes: u64,
    /// Bytes of tables whose rows are gathered on the host and stay in the file (the per-layer
    /// token embedding): they take no card and no pinned memory.
    pub(crate) host_table_bytes: u64,
}

/// What the solver may spend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RungBudget {
    /// Device bytes free.
    pub(crate) card_free: u64,
    /// Device bytes the per-request state takes (KV cache and work buffers).
    pub(crate) state_bytes: u64,
    /// Device bytes kept free (headroom and runtime floor).
    pub(crate) reserve_bytes: u64,
    /// Host bytes a payload may spend (`HostMemory::payload_spare`).
    pub(crate) host_spare: u64,
}

/// Where a layer's weights live between visits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Home {
    /// Resident on the card.
    Card,
    /// Pinned in host memory, copied to a staging slot when visited.
    Pinned,
    /// Read from the GGUF into a staging slot when visited.
    Disk,
}

/// A layer plan that streams at least one layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LayerRung {
    pub(crate) homes: Vec<Home>,
    /// Staging slots on the card, each the size of the largest streamed layer.
    pub(crate) staging_slots: usize,
    pub(crate) staging_slot_bytes: u64,
    /// Device bytes of the resident layers.
    pub(crate) card_layer_bytes: u64,
    /// Host bytes pinned: the resident share of streamed layers and the chunk ring.
    pub(crate) pinned_bytes: u64,
    /// Bytes read from the GGUF in one pass over the layers.
    pub(crate) disk_bytes_per_pass: u64,
}

/// The ladder's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Placement {
    /// Every layer fits the card beside the rest: nothing streams.
    Resident,
    /// Some layers stream.
    Streamed(LayerRung),
}

/// Why no placement exists: bytes that cannot leave the card do not fit it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Refusal {
    pub(crate) what: &'static str,
    pub(crate) needed: u64,
    pub(crate) have: u64,
}

/// Places the layers of `model` on `budget`. Refuses only when the card cannot hold the non-layer
/// tensors, the state and the reserve, or one layer beside them; a host that holds nothing and a
/// disk with only the model file on it still plan.
pub(crate) fn place(model: &RungModel, budget: &RungBudget) -> Result<Placement, Refusal> {
    let base = model
        .card_fixed_bytes
        .saturating_add(budget.state_bytes)
        .saturating_add(budget.reserve_bytes);
    if base > budget.card_free {
        return Err(Refusal {
            what: "the card cannot hold the non-layer tensors, the state and the reserve",
            needed: base,
            have: budget.card_free,
        });
    }
    let room = budget.card_free - base;
    let total: u64 = model.layers.iter().sum();
    if total <= room {
        return Ok(Placement::Resident);
    }
    let mut order: Vec<usize> = (0..model.layers.len()).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(model.layers[index]));
    let sorted: Vec<u64> = order.iter().map(|&index| model.layers[index]).collect();
    // The largest k layers stay on the card; the staging slots are sized by the largest layer
    // that streams. Two slots overlap the copy with the visit; one slot is the fallback.
    let mut best: Option<(usize, usize, u64)> = None;
    for slots in [STAGING_SLOTS, 1] {
        let mut kept = 0u64;
        for (k, &largest_streamed) in sorted.iter().enumerate() {
            // `kept` is the sum of the k largest layers; layer k is the largest that streams.
            let need = kept.saturating_add(largest_streamed.saturating_mul(slots as u64));
            if need <= room && best.is_none_or(|(_, _, bytes)| kept > bytes) {
                best = Some((slots, k, kept));
            }
            kept = kept.saturating_add(largest_streamed);
        }
        if best.is_some() {
            break;
        }
    }
    let Some((slots, k, card_layer_bytes)) = best else {
        return Err(Refusal {
            what: "the card cannot hold one layer beside the non-layer tensors, the state and the reserve",
            needed: base.saturating_add(sorted.first().copied().unwrap_or(0)),
            have: budget.card_free,
        });
    };
    let mut homes = vec![Home::Card; model.layers.len()];
    let mut pinned_room = budget.host_spare.saturating_sub(STAGE_CHUNK_RING_BYTES);
    let mut pinned_layers = 0u64;
    let mut disk_bytes = 0u64;
    for &index in &order[k..] {
        let bytes = model.layers[index];
        if bytes <= pinned_room {
            homes[index] = Home::Pinned;
            pinned_room -= bytes;
            pinned_layers += bytes;
        } else {
            homes[index] = Home::Disk;
            disk_bytes += bytes;
        }
    }
    Ok(Placement::Streamed(LayerRung {
        homes,
        staging_slots: slots,
        staging_slot_bytes: sorted[k],
        card_layer_bytes,
        pinned_bytes: pinned_layers + STAGE_CHUNK_RING_BYTES,
        disk_bytes_per_pass: disk_bytes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;

    fn model(layers: &[u64]) -> RungModel {
        RungModel {
            layers: layers.iter().map(|&bytes| bytes * MIB).collect(),
            card_fixed_bytes: 500 * MIB,
            host_table_bytes: 0,
        }
    }

    fn budget(card_mib: u64, host_mib: u64) -> RungBudget {
        RungBudget {
            card_free: card_mib * MIB,
            state_bytes: 300 * MIB,
            reserve_bytes: 700 * MIB,
            host_spare: host_mib * MIB,
        }
    }

    #[test]
    fn a_model_that_fits_the_card_stays_resident() {
        let m = model(&[100; 10]);
        // 1000 of layers + 500 fixed + 300 state + 700 reserve = 2500.
        assert_eq!(place(&m, &budget(2500, 0)), Ok(Placement::Resident));
    }

    #[test]
    fn one_byte_short_streams_the_smallest_share_the_card_cannot_hold() {
        let m = model(&[100; 10]);
        let Ok(Placement::Streamed(rung)) = place(&m, &budget(2499, 4096)) else {
            panic!("a layer must stream");
        };
        // Room is 999 MiB: two 100 MiB slots leave 799, so 7 layers stay and 3 stream.
        assert_eq!(rung.staging_slots, 2);
        assert_eq!(rung.staging_slot_bytes, 100 * MIB);
        assert_eq!(rung.card_layer_bytes, 700 * MIB);
        let streamed = rung.homes.iter().filter(|h| **h != Home::Card).count();
        assert_eq!(streamed, 3);
    }

    #[test]
    fn the_host_holds_streamed_layers_largest_first_and_the_rest_read_from_disk() {
        let m = model(&[400, 300, 200, 100, 100, 100, 100, 100]);
        // Card room 1100 MiB: two 200 MiB slots leave 700 for the 400 and the 300 MiB layers.
        let b = budget(1500 + 1100, 64 + 450);
        let Ok(Placement::Streamed(rung)) = place(&m, &b) else {
            panic!("a layer must stream");
        };
        assert_eq!(rung.staging_slots, 2);
        let pinned: Vec<usize> = (0..8).filter(|&i| rung.homes[i] == Home::Pinned).collect();
        let disk: Vec<usize> = (0..8).filter(|&i| rung.homes[i] == Home::Disk).collect();
        assert!(!pinned.is_empty() && !disk.is_empty(), "{:?}", rung.homes);
        let largest_pinned = pinned.iter().map(|&i| m.layers[i]).max().unwrap();
        let largest_disk = disk.iter().map(|&i| m.layers[i]).max().unwrap();
        assert!(largest_pinned >= largest_disk, "{:?}", rung.homes);
        assert_eq!(
            rung.disk_bytes_per_pass,
            disk.iter().map(|&i| m.layers[i]).sum::<u64>()
        );
        assert_eq!(
            rung.pinned_bytes,
            pinned.iter().map(|&i| m.layers[i]).sum::<u64>() + STAGE_CHUNK_RING_BYTES
        );
    }

    #[test]
    fn a_host_with_nothing_to_spare_still_plans_every_streamed_layer_from_disk() {
        let m = model(&[400; 12]);
        let Ok(Placement::Streamed(rung)) = place(&m, &budget(3000, 0)) else {
            panic!("a layer must stream");
        };
        assert!(rung.homes.iter().all(|h| *h != Home::Pinned));
        assert!(rung.homes.contains(&Home::Disk));
    }

    #[test]
    fn a_card_that_holds_one_layer_but_not_two_uses_one_staging_slot() {
        let m = model(&[400; 6]);
        // room = 1100 + ... ; fixed+state+reserve = 1500; room 500: one 400 slot fits, two do not.
        let Ok(Placement::Streamed(rung)) = place(&m, &budget(2000, 0)) else {
            panic!("a layer must stream");
        };
        assert_eq!(rung.staging_slots, 1);
        assert_eq!(rung.card_layer_bytes, 0);
    }

    #[test]
    fn the_only_refusals_are_bytes_that_cannot_leave_the_card() {
        let m = model(&[400; 6]);
        // Fixed + state + reserve is 1500: a card below it refuses by name.
        let error = place(&m, &budget(1499, 4096)).unwrap_err();
        assert_eq!((error.needed, error.have), (1500 * MIB, 1499 * MIB));
        // One layer beside them: 1900.
        let error = place(&m, &budget(1899, 4096)).unwrap_err();
        assert_eq!((error.needed, error.have), (1900 * MIB, 1899 * MIB));
        assert!(place(&m, &budget(1900, 0)).is_ok());
    }

    /// Never refuses while the card holds the fixed bytes and one layer, whatever the host and
    /// the disk have: the grid of cards and hosts over a mixed-size model.
    #[test]
    fn a_grid_of_cards_and_hosts_never_refuses_above_the_fixed_floor() {
        let m = model(&[480, 470, 465, 300, 300, 120, 100, 100, 80, 50]);
        for card in (1980..6000).step_by(37) {
            for host in [0, 64, 512, 4096, 65_536] {
                let b = budget(card, host);
                let placed = place(&m, &b).unwrap_or_else(|e| panic!("{card}/{host}: {e:?}"));
                if let Placement::Streamed(rung) = placed {
                    // The plan fits the card it was given.
                    let on_card = rung.card_layer_bytes
                        + rung.staging_slots as u64 * rung.staging_slot_bytes
                        + m.card_fixed_bytes
                        + b.state_bytes
                        + b.reserve_bytes;
                    assert!(on_card <= b.card_free, "{card}/{host}");
                    // And pins no more than the host may spend beyond the ring.
                    assert!(
                        rung.pinned_bytes <= b.host_spare.max(STAGE_CHUNK_RING_BYTES),
                        "{card}/{host}"
                    );
                }
            }
        }
    }

    /// More card never puts fewer layers on it, while the plan keeps both staging slots.
    #[test]
    fn a_larger_card_keeps_at_least_as_many_bytes_resident() {
        let m = model(&[480, 470, 465, 300, 300, 120, 100, 100, 80, 50]);
        let mut last = 0;
        for card in (1980..6000).step_by(11) {
            let kept = match place(&m, &budget(card, 4096)).unwrap() {
                Placement::Resident => m.layers.iter().sum(),
                Placement::Streamed(rung) if rung.staging_slots == STAGING_SLOTS => {
                    rung.card_layer_bytes
                }
                Placement::Streamed(_) => continue,
            };
            assert!(kept >= last, "{card}");
            last = kept;
        }
        assert!(last > 0);
    }
}
