//! Where the bytes of one prefix entry sit in a family's host slab, and how an entry is cut into
//! the objects of the disk cache. Pure arithmetic over byte offsets: a family builds the layout
//! of an entry from its own state geometry and the tier does the rest.

use crate::EngError;

/// Receives the bytes that belong at a slab offset.
pub(crate) type Put<'a> = dyn FnMut(usize, &[u8]) -> Result<(), EngError> + 'a;

/// A contiguous piece of an entry's slab: `bytes` at `slab` offset of the held slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Run {
    pub(crate) slab: usize,
    pub(crate) bytes: usize,
}

/// What one object of an entry holds, which is what the manifest names it by. The envelope fixes
/// the page size (256 rows), so a family names only the kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ObjectRole {
    /// State of constant size whatever the length (recurrent state): `kind` names it.
    FixedState { kind: &'static str },
    /// The rows `0..positions` of one attention layer's cache.
    Pages,
}

/// The objects of one entry, in file order, each the runs of the slab that make it up back to
/// back, and what each holds. An object does not depend on the capacity of the process that
/// wrote it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SlabLayout {
    pub(crate) objects: Vec<Vec<Run>>,
    pub(crate) roles: Vec<ObjectRole>,
}

impl SlabLayout {
    /// The byte length of every object, in file order.
    pub(crate) fn object_bytes(&self) -> Vec<u64> {
        self.objects
            .iter()
            .map(|runs| runs.iter().map(|run| run.bytes as u64).sum())
            .collect()
    }

    /// Every run in the order the objects are written.
    pub(crate) fn flat(&self) -> Vec<Run> {
        self.objects.iter().flatten().copied().collect()
    }

    /// Places `bytes`, which start `offset` bytes into object `object`, at their slab offsets.
    pub(crate) fn scatter(
        &self,
        object: usize,
        mut offset: u64,
        mut bytes: &[u8],
        put: &mut Put<'_>,
    ) -> Result<(), EngError> {
        let runs: &[Run] = self
            .objects
            .get(object)
            .ok_or_else(|| EngError::BadRequest("kv disk: object index".into()))?;
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

    fn layout() -> SlabLayout {
        SlabLayout {
            roles: vec![
                ObjectRole::FixedState { kind: "state" },
                ObjectRole::Pages,
                ObjectRole::Pages,
            ],
            objects: vec![
                vec![Run { slab: 0, bytes: 10 }],
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
        let layout = layout();
        assert_eq!(layout.object_bytes(), vec![10, 10, 8]);
        let flat = layout.flat();
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
        let layout = layout();
        let placed: std::cell::RefCell<Vec<(usize, Vec<u8>)>> = Default::default();
        let mut put = |slab: usize, bytes: &[u8]| {
            placed.borrow_mut().push((slab, bytes.to_vec()));
            Ok(())
        };
        // Bytes 4..10 of object 1 cross from its first run into its second.
        layout.scatter(1, 4, &[1, 2, 3, 4, 5, 6], &mut put).unwrap();
        assert_eq!(
            *placed.borrow(),
            vec![(104, vec![1, 2]), (200, vec![3, 4, 5, 6])]
        );
        placed.borrow_mut().clear();
        layout.scatter(0, 0, &[9; 10], &mut put).unwrap();
        assert_eq!(*placed.borrow(), vec![(0, vec![9; 10])]);
        assert!(layout.scatter(2, 6, &[0; 3], &mut put).is_err());
        assert!(layout.scatter(3, 0, &[0], &mut put).is_err());
    }
}
