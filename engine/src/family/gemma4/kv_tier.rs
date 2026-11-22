//! The Bolt engine's use of the shared on-disk prefix tier. A Bolt prefix is a host copy of the
//! cache cells `0..fed` of every physical layer (`BoltSnapshot`); the family tells the shared
//! host-store adapter (`kv_disk::HostFamily`) how many cells a snapshot covers, how an entry is
//! cut into one object per physical layer and how a snapshot is rebuilt from a verified read.
//! The policy (when to write, how long a stop may wait, how a damaged entry falls back) is the
//! tier's.
use super::resident::bolt::BoltSnapshot;
use crate::{
    EngError,
    family::kv_disk::{
        DiskTier, FamilyContract, HostFamily, LoadIdentity, ObjectRole, Run, SlabLayout,
        SnapshotView, place_in_vec, vec_landing,
    },
};

/// What the shared tier calls this engine in its log lines.
pub(super) const TIER_LABEL: &str = "gemma4 bolt";

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

/// The Bolt model as the host-store adapter sees it: the bytes of each physical layer's part of
/// a snapshot of any number of cells.
pub(super) struct BoltFamily<'a> {
    pub(super) layer_bytes: &'a dyn Fn(usize) -> Vec<usize>,
    /// The cells of a session of this load: the longest entry it can hold.
    pub(super) context_cells: usize,
}

impl HostFamily for BoltFamily<'_> {
    type Snapshot = BoltSnapshot;
    type Path = crate::family::session::PrefixIdentity;
    type Landing = Vec<u8>;

    fn view(&self, snapshot: &BoltSnapshot) -> Option<SnapshotView> {
        Some(SnapshotView {
            positions: snapshot.fed(),
            bytes: snapshot.bytes() as u64,
            slot: snapshot.hold(),
        })
    }

    fn layout(&self, positions: usize) -> Result<SlabLayout, EngError> {
        let cells = self.context_cells;
        if positions > cells {
            return Err(EngError::ContextLimit(format!(
                "a kv disk entry of {positions} positions does not fit the {cells} position context"
            )));
        }
        Ok(entry_slab(&(self.layer_bytes)(positions)))
    }

    fn begin_landing(
        &mut self,
        _tier: &mut DiskTier,
        _positions: usize,
        layout: &SlabLayout,
    ) -> Result<Vec<u8>, EngError> {
        vec_landing(TIER_LABEL, layout)
    }

    fn place(&self, landing: &mut Vec<u8>, slab: usize, part: &[u8]) -> Result<(), EngError> {
        place_in_vec(landing, slab, part)
    }

    fn finish_landing(
        &mut self,
        landing: Vec<u8>,
        positions: usize,
    ) -> Result<BoltSnapshot, EngError> {
        Ok(BoltSnapshot::from_parts(positions, landing))
    }
}

/// What names a Bolt load to an entry kept outside the process.
/// The numeric path of a load: whether the attention split is armed and, off the card the count
/// was tuned on, how many splits. A different count sums the softmax in a different order, so an
/// entry written under one count is not another's; the tuned card keeps the original name.
fn numeric_path(attention_splits: Option<usize>) -> String {
    match attention_splits {
        None => "bolt-kv8 attention-split false".to_owned(),
        Some(splits) if splits == super::resident::bolt::ATTN_SPLIT_MAX => {
            "bolt-kv8 attention-split true".to_owned()
        }
        Some(splits) => format!("bolt-kv8 attention-split true splits {splits}"),
    }
}

pub(super) fn load_identity(
    weights: &str,
    layer_bytes: &[usize],
    attention_splits: Option<usize>,
) -> LoadIdentity {
    let (tokenizer, template) = crate::tokenizer::gemma::pinned_digests();
    LoadIdentity {
        weights: weights.to_owned(),
        tokenizer: crate::weights::sha256_hex(&tokenizer),
        template: crate::weights::sha256_hex(&template),
        numeric_path: numeric_path(attention_splits),
        family: CONTRACT,
        layout: format!(
            "{LAYOUT_TAG}; {} physical layers; bytes per cell by layer {layer_bytes:?}",
            layer_bytes.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn the_load_identity_names_the_weights_the_pinned_text_and_the_layout() {
        let one = load_identity("w1", &[8, 16], Some(32));
        assert!(one.is_complete());
        let other_weights = load_identity("w2", &[8, 16], Some(32));
        let other_layout = load_identity("w1", &[8, 24], Some(32));
        let other_split = load_identity("w1", &[8, 16], None);
        let namespace = |load: &LoadIdentity| {
            format!("{:?}", (&load.weights, &load.layout, &load.numeric_path))
        };
        assert_ne!(namespace(&one), namespace(&other_weights));
        assert_ne!(namespace(&one), namespace(&other_layout));
        assert_ne!(namespace(&one), namespace(&other_split));
        // The tuned card keeps the name entries were written under; another count is another
        // numeric path.
        assert_eq!(one.numeric_path, "bolt-kv8 attention-split true");
        let small = load_identity("w1", &[8, 16], Some(8));
        assert_ne!(namespace(&one), namespace(&small));
    }
}
