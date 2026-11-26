//! The family-neutral `(layer, role) -> tensor name` lookup and the per-kind layer
//! ordinals of the resident step. The Qwen3.6 constructor (`new` from `StepDims` and
//! the inventory) stays in `qwen36::cuda`; the paths re-export from there.

//! Pure host code. The inventory lists every expected tensor with its role and its GGUF name
//! `blk.<layer>.<suffix>`; this table groups them by layer once, so a launch asks for
//! `(layer, Role)` and gets the name to look up in the family's weights, and for the ordinal
//! of the layer among the layers of its kind. The suffix text stays private to the
//! inventory: the layer number is read from the name prefix.

use crate::{
    EngError,
    family::{AttentionKind, gguf_model::Role},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family) struct LayerEntry {
    pub(in crate::family) kind: AttentionKind,
    pub(in crate::family) ordinal: usize,
    pub(in crate::family) names: Vec<(Role, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family) struct LayerTable {
    pub(in crate::family) layers: Vec<LayerEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family) struct LayerSlot {
    pub(in crate::family) kind: AttentionKind,
    pub(in crate::family) ordinal: usize,
}

/// The layer number of `blk.<layer>.<suffix>`, or `None` for a global tensor.
pub(in crate::family) fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let (number, _) = rest.split_once('.')?;
    number.parse().ok()
}

impl LayerTable {
    pub(in crate::family) fn len(&self) -> usize {
        self.layers.len()
    }

    pub(in crate::family) fn slot(&self, layer: usize) -> Result<LayerSlot, EngError> {
        let entry = self.entry(layer)?;
        Ok(LayerSlot {
            kind: entry.kind,
            ordinal: entry.ordinal,
        })
    }

    pub(in crate::family) fn name(&self, layer: usize, role: Role) -> Result<&str, EngError> {
        self.entry(layer)?
            .names
            .iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, name)| name.as_str())
            .ok_or_else(|| {
                EngError::Checkpoint(format!("layer table: layer {layer} has no role {role:?}"))
            })
    }

    fn entry(&self, layer: usize) -> Result<&LayerEntry, EngError> {
        self.layers.get(layer).ok_or_else(|| {
            EngError::BadRequest(format!(
                "layer table: layer {layer} is outside {} layers",
                self.layers.len()
            ))
        })
    }
}
