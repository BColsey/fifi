//! Where the bytes of one prefix entry sit in a pinned slab, and how an entry is cut into the
//! objects of the disk cache. Pure host arithmetic over a `StateLayout`.
use super::state_layout::StateLayout;
use crate::EngError;

/// Receives the bytes that belong at a slab offset.
pub(super) type Put<'a> = dyn FnMut(usize, &[u8]) -> Result<(), EngError> + 'a;

/// A contiguous piece of an entry's slab: `bytes` at `slab` offset of the pinned slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Run {
    pub(super) slab: usize,
    pub(super) bytes: usize,
}

/// Where the bytes of one entry sit in a slab. An entry is the DeltaNet state followed by one
/// object per attention layer; a layer object is the runs of `prefix_ranges` back to back, so it
/// does not depend on the KV capacity of the process that wrote it.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct EntryShape {
    pub(super) delta: Run,
    pub(super) layers: Vec<Vec<Run>>,
}

impl EntryShape {
    pub(super) fn of(layout: &StateLayout, positions: usize) -> Result<Self, EngError> {
        let delta_bytes = layout
            .delta
            .total
            .checked_div(layout.delta.slots)
            .ok_or_else(|| EngError::BadRequest("kv disk: zero DeltaNet slots".into()))?;
        let mut layers = Vec::with_capacity(layout.kv.layers);
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
            layers.push(runs);
        }
        Ok(Self {
            delta: Run {
                slab: 0,
                bytes: delta_bytes,
            },
            layers,
        })
    }

    /// The byte length of every object, in file order.
    pub(super) fn object_bytes(&self) -> Vec<u64> {
        std::iter::once(self.delta.bytes as u64)
            .chain(
                self.layers
                    .iter()
                    .map(|runs| runs.iter().map(|run| run.bytes as u64).sum()),
            )
            .collect()
    }

    /// Every run in the order the objects are written.
    pub(super) fn flat(&self) -> Vec<Run> {
        std::iter::once(self.delta)
            .chain(self.layers.iter().flatten().copied())
            .collect()
    }

    /// Places `bytes`, which start `offset` bytes into object `object`, at their slab offsets.
    pub(super) fn scatter(
        &self,
        object: usize,
        mut offset: u64,
        mut bytes: &[u8],
        put: &mut Put<'_>,
    ) -> Result<(), EngError> {
        let single = [self.delta];
        let runs: &[Run] = match object {
            0 => &single,
            other => self
                .layers
                .get(other - 1)
                .ok_or_else(|| EngError::BadRequest("kv disk: object index".into()))?,
        };
        let mut start = 0u64;
        for run in runs {
            let end = start + run.bytes as u64;
            if offset < end && !bytes.is_empty() {
                let within = (offset - start) as usize;
                let take = (run.bytes - within).min(bytes.len());
                put(run.slab + within, &bytes[..take])?;
                bytes = &bytes[take..];
                offset += take as u64;
            }
            start = end;
        }
        if bytes.is_empty() {
            Ok(())
        } else {
            Err(EngError::BadRequest(
                "kv disk: bytes past the end of an object".into(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> EntryShape {
        EntryShape {
            delta: Run { slab: 0, bytes: 10 },
            layers: vec![
                vec![
                    Run {
                        slab: 100,
                        bytes: 6,
                    },
                    Run {
                        slab: 200,
                        bytes: 4,
                    },
                ],
                vec![Run {
                    slab: 300,
                    bytes: 8,
                }],
            ],
        }
    }

    #[test]
    fn object_sizes_and_flat_runs_follow_the_file_order() {
        let shape = shape();
        assert_eq!(shape.object_bytes(), vec![10, 10, 8]);
        let flat = shape.flat();
        assert_eq!(flat.len(), 4);
        assert_eq!(flat[0], Run { slab: 0, bytes: 10 });
        assert_eq!(
            flat[3],
            Run {
                slab: 300,
                bytes: 8
            }
        );
    }

    #[test]
    fn scatter_splits_a_chunk_across_runs_and_refuses_bytes_past_the_object() {
        let shape = shape();
        let placed: std::cell::RefCell<Vec<(usize, Vec<u8>)>> = Default::default();
        let mut put = |slab: usize, bytes: &[u8]| {
            placed.borrow_mut().push((slab, bytes.to_vec()));
            Ok(())
        };
        // Bytes 4..10 of layer 0's object cross from its first run into its second.
        shape.scatter(1, 4, &[1, 2, 3, 4, 5, 6], &mut put).unwrap();
        assert_eq!(
            *placed.borrow(),
            vec![(104, vec![1, 2]), (200, vec![3, 4, 5, 6])]
        );
        placed.borrow_mut().clear();
        shape.scatter(0, 0, &[9; 10], &mut put).unwrap();
        assert_eq!(*placed.borrow(), vec![(0, vec![9; 10])]);
        assert!(shape.scatter(2, 6, &[0; 3], &mut put).is_err());
        assert!(shape.scatter(3, 0, &[0], &mut put).is_err());
    }

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
                let shape = EntryShape::of(&layout, 1_500).unwrap();
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
                assert_eq!(shape.layers.len(), layout.kv.layers);
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
