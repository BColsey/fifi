//! The expert records of the real Flash-Next pack: the [`RecordSource`] of the expert bank.
//!
//! WHAT: [`PackRecordSource`] reads the nine components of each of an expert's three matrices
//! (`trellis`, `suh`, `svh` and the codebook words) from an opened pack, loads them as `Matrix`es
//! and packs them with `SlotBankLayout::pack_matrices` into the one record the bank stores
//! (`trellis | suh | svh` per part, scales widened to F32). The caller opens the pack once, through
//! `FlashPackFiles::open` (one `LocalGuard` over the directory, every shard and the side files),
//! and lends its `Checkpoint`; this module opens nothing. WHY a borrow: the trunk loader and the
//! n-gram feed read the same pack, so one open serves all of them. The bank reads the pack at build
//! only (the host store and the first fill of the slots); a miss copies from the pinned store.
//!
//! The layout (widths and rates) is derived from the pack itself: the gate matrix of expert 0 of
//! layer 0 gives `[hidden, inter]` and the rate, and `pack_matrices` refuses any other expert whose
//! shape or rate differs. Nothing here allocates device memory.
//!
//! Staged: the CUDA step (S6) is the first caller.

#![cfg_attr(not(test), allow(dead_code))]

use super::bank::RecordSource;
use crate::{
    EngError,
    family::{
        exl3_pack::{ExpertPart, Root, expert_name_in},
        flash_next::{load::model_shape, moe::slot_bank::SlotBankLayout},
    },
    weights::exl3::{COMPONENT_SUFFIXES, Checkpoint, Matrix},
};

/// Bytes one matrix read may take (the components of one matrix are well under 1 MiB at 3 bpw).
const MATRIX_READ_BUDGET: u64 = 1 << 30;

fn pack_error(detail: String) -> EngError {
    EngError::Checkpoint(format!("Flash-Next expert records: {detail}"))
}

/// The records of the routed experts of an opened pack.
pub(super) struct PackRecordSource<'a> {
    checkpoint: &'a Checkpoint,
    layout: SlotBankLayout,
    layers: usize,
    experts: usize,
}

impl<'a> PackRecordSource<'a> {
    /// Takes the layer and expert counts from `config.json` and the record layout from the first
    /// expert of the first layer.
    pub(super) fn new(checkpoint: &'a Checkpoint) -> Result<Self, EngError> {
        let (shape, kinds) = model_shape(&checkpoint.model_config)?;
        let (layers, experts) = (kinds.len(), shape.moe.experts());
        let gate = load_matrix(
            checkpoint,
            &expert_name_in(Root::Layers, 0, 0, ExpertPart::Gate),
        )?;
        let [hidden, inter] = gate.shape();
        let words = usize::from(gate.bitrate_twice()) * 8;
        let layout = SlotBankLayout::new(hidden, inter, [words; 3])?;
        Ok(Self {
            checkpoint,
            layout,
            layers,
            experts,
        })
    }

    pub(super) fn layout(&self) -> &SlotBankLayout {
        &self.layout
    }

    pub(super) fn layers(&self) -> usize {
        self.layers
    }

    pub(super) fn experts(&self) -> usize {
        self.experts
    }

    /// The record of `expert` in `layer`. Refuses an index outside the pack and an expert whose
    /// matrices differ in shape or rate from the layout.
    pub(super) fn read_record(&self, layer: usize, expert: usize) -> Result<Vec<u8>, EngError> {
        if layer >= self.layers || expert >= self.experts {
            return Err(pack_error(format!(
                "layer {layer} expert {expert} is outside {} layers of {} experts",
                self.layers, self.experts
            )));
        }
        let [gate, up, down] = ExpertPart::ALL.map(|part| {
            load_matrix(
                self.checkpoint,
                &expert_name_in(Root::Layers, layer, expert, part),
            )
        });
        self.layout.pack_matrices([&gate?, &up?, &down?])
    }
}

impl RecordSource for PackRecordSource<'_> {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        let record = self.read_record(layer, usize::from(expert))?;
        if out.len() != record.len() {
            return Err(pack_error(format!(
                "a record is {} bytes, the buffer has {}",
                record.len(),
                out.len()
            )));
        }
        out.copy_from_slice(&record);
        Ok(())
    }
}

/// One matrix of the pack by its tensor-name prefix: its nine components, read whole.
fn load_matrix(checkpoint: &Checkpoint, prefix: &str) -> Result<Matrix, EngError> {
    let names = COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
    let keys = names.each_ref().map(String::as_str);
    let selected = checkpoint.source.read_selected(&keys, MATRIX_READ_BUDGET)?;
    Matrix::load(&selected, prefix, &checkpoint.config)
}
