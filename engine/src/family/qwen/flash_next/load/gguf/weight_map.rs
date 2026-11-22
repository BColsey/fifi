//! Flash-Next's GGUF tensors as a recipe's weight map.
//!
//! The recipe names a role of a stored layer; the map the loader built and checked against the
//! file answers with the tensor that holds it and the stored type of that tensor, which selects
//! the kernel. A role a layer does not have (a QSA indexer on a DeltaNet layer, the per-layer
//! embedding step outside its layer) is `None`. The expert stacks of different layers are
//! stored at different types in the same file, so the stored type is per layer.

use super::tensors::TensorMap;
use crate::family::recipe::{Global, Role, WeightMap};
use crate::weights::GgufType;

impl WeightMap for TensorMap {
    fn layer(&self, layer: u32, role: Role) -> Option<String> {
        self.get(role, Some(layer as usize))
            .map(|entry| entry.name.clone())
    }

    fn global(&self, role: Global) -> Option<String> {
        self.get(role, None).map(|entry| entry.name.clone())
    }

    fn storage(&self, layer: u32, role: Role) -> Option<GgufType> {
        self.get(role, Some(layer as usize)).map(|entry| entry.ty)
    }
}
