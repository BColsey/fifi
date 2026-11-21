//! Where the bytes of one Qwen3.6 prefix entry sit in a pinned slab: the DeltaNet state, then one
//! object per attention layer, as the layout the shared disk tier cuts into files. Pure host
//! arithmetic over a `StateLayout`.
use super::state_layout::StateLayout;
use crate::{
    EngError,
    family::kv_disk::{Run, SlabLayout},
};

/// The layout of an entry of `positions` committed positions: object 0 is the DeltaNet state;
/// each following object is one attention layer's runs of `prefix_ranges` back to back, so it
/// does not depend on the KV capacity of the process that wrote it.
pub(super) fn entry_slab(layout: &StateLayout, positions: usize) -> Result<SlabLayout, EngError> {
    let delta_bytes = layout
        .delta
        .total
        .checked_div(layout.delta.slots)
        .ok_or_else(|| EngError::BadRequest("kv disk: zero DeltaNet slots".into()))?;
    let mut objects = Vec::with_capacity(1 + layout.kv.layers);
    objects.push(vec![Run {
        slab: 0,
        bytes: delta_bytes,
    }]);
    for layer in 0..layout.kv.layers {
        let mut runs = Vec::new();
        for range in layout.kv.prefix_ranges(layer, positions)? {
            if range.bytes == 0 {
                continue;
            }
            runs.push(Run {
                slab: delta_bytes
                    .checked_add(range.device_offset)
                    .ok_or_else(|| EngError::BadRequest("kv disk: offset overflows".into()))?,
                bytes: range.bytes,
            });
        }
        objects.push(runs);
    }
    Ok(SlabLayout { objects })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entry_has_the_same_objects_at_every_capacity_and_every_run_fits_its_slab() {
        use crate::family::{
            project::cuda::WeightLayout,
            qwen36::cuda::{
                common_tests,
                dims::{KvMode, StepOptions},
            },
        };
        for mode in [KvMode::Fp16, KvMode::Int8G32] {
            let mut sizes = Vec::new();
            for capacity in [2_048usize, 4_096] {
                let options = StepOptions {
                    kv_mode: mode,
                    context_tokens: capacity,
                    delta_slots: 1,
                    forked_experts: false,
                    tiered: None,
                    layout: WeightLayout::Native,
                };
                let layout = StateLayout::new(&common_tests::dims(), &options).unwrap();
                let shape = entry_slab(&layout, 1_500).unwrap();
                let slab = layout.delta.total / layout.delta.slots + layout.kv.total;
                let mut runs = shape.flat();
                runs.sort_by_key(|run| run.slab);
                for pair in runs.windows(2) {
                    assert!(
                        pair[0].slab + pair[0].bytes <= pair[1].slab,
                        "{mode:?} overlap"
                    );
                }
                let last = runs.last().unwrap();
                assert!(last.slab + last.bytes <= slab, "{mode:?} escapes the slab");
                assert_eq!(shape.objects.len(), 1 + layout.kv.layers);
                sizes.push(shape.object_bytes());
            }
            assert_eq!(
                sizes[0], sizes[1],
                "{mode:?}: objects depend on the capacity"
            );
            assert!(sizes[0].iter().all(|&bytes| bytes > 0));
        }
    }
}
