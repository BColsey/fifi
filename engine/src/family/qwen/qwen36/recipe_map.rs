//! Qwen3.6 as a recipe: its layer list and the names and stored types of its GGUF tensors.
//!
//! The spec already holds the layer pattern (a full-attention layer every fourth, the gated
//! DeltaNet in between, routed experts with a shared expert in every layer), so the recipe is the
//! shared one built from it. What is Qwen3.6's own is the naming convention of its GGUF files,
//! which the weight map answers from the tensor inventory the loader already checks a file
//! against: the map names a tensor only when the inventory lists it for that layer's mixer, so a
//! role a layer does not have is `None`. The stored type of a tensor comes from the file's header
//! through `types`, so a file that quantizes layers differently is read as it is.

use crate::family::{
    QWEN3_6_35B_A3B,
    gguf_model::{Inventory, Role as GgufRole},
    recipe::{Global, Recipe, Role, WeightMap},
};
use crate::{EngError, weights::GgufType};

/// The recipe of Qwen3.6-35B-A3B.
pub(crate) fn recipe() -> Result<Recipe, EngError> {
    Recipe::from_spec(&QWEN3_6_35B_A3B)
}

/// The GGUF role a recipe role is stored as, when Qwen3.6 has one.
fn layer_role(role: Role) -> Option<GgufRole> {
    Some(match role {
        Role::AttnNorm => GgufRole::AttnNorm,
        Role::QProj => GgufRole::AttnQ,
        Role::KProj => GgufRole::AttnK,
        Role::VProj => GgufRole::AttnV,
        Role::OProj => GgufRole::AttnOutput,
        Role::QNorm => GgufRole::AttnQNorm,
        Role::KNorm => GgufRole::AttnKNorm,
        Role::DeltaQkv => GgufRole::AttnQkv,
        Role::DeltaGate => GgufRole::AttnGate,
        Role::DeltaOut => GgufRole::SsmOut,
        Role::DeltaConv => GgufRole::SsmConv1d,
        Role::DeltaDecay => GgufRole::SsmA,
        Role::DeltaAlpha => GgufRole::SsmAlpha,
        Role::DeltaBeta => GgufRole::SsmBeta,
        Role::DeltaDtBias => GgufRole::SsmDtBias,
        Role::DeltaNorm => GgufRole::SsmNorm,
        Role::FfnNorm => GgufRole::PostAttentionNorm,
        Role::Router => GgufRole::FfnGateInp,
        Role::SharedRouterGate => GgufRole::FfnGateInpShexp,
        Role::ExpertGate => GgufRole::FfnGateExps,
        Role::ExpertUp => GgufRole::FfnUpExps,
        Role::ExpertDown => GgufRole::FfnDownExps,
        Role::SharedGate => GgufRole::FfnGateShexp,
        Role::SharedUp => GgufRole::FfnUpShexp,
        Role::SharedDown => GgufRole::FfnDownShexp,
        Role::IndexProj
        | Role::IndexKey
        | Role::IndexQNorm
        | Role::IndexKNorm
        | Role::DenseGate
        | Role::DenseUp
        | Role::DenseDown
        | Role::Hyper(..)
        | Role::PleKey
        | Role::PleValue
        | Role::PleNormKey
        | Role::PleNormQuery
        | Role::PleNormConv
        | Role::PleConv => return None,
    })
}

/// The tensor names and stored types of one Qwen3.6 GGUF file.
pub(crate) struct Qwen36GgufMap<'a> {
    inventory: &'a Inventory,
    /// The stored type of the tensor of a name, from the file's header.
    types: &'a dyn Fn(&str) -> Option<GgufType>,
}

impl<'a> Qwen36GgufMap<'a> {
    pub(crate) fn new(
        inventory: &'a Inventory,
        types: &'a dyn Fn(&str) -> Option<GgufType>,
    ) -> Self {
        Self { inventory, types }
    }

    fn named(&self, name: String) -> Option<String> {
        self.inventory.get(&name).map(|_| name)
    }
}

impl WeightMap for Qwen36GgufMap<'_> {
    fn layer(&self, layer: u32, role: Role) -> Option<String> {
        self.named(layer_role(role)?.tensor_name(Some(layer as usize)))
    }

    fn global(&self, role: Global) -> Option<String> {
        let role = match role {
            Global::Embedding => GgufRole::TokenEmbd,
            Global::FinalNorm => GgufRole::OutputNorm,
            Global::Head => GgufRole::Output,
            Global::HeadSite(_) | Global::NgramTable => return None,
        };
        self.named(role.tensor_name(None))
    }

    fn storage(&self, layer: u32, role: Role) -> Option<GgufType> {
        (self.types)(&self.layer(layer, role)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::{
        recipe::{Ffn, Site, SiteTensor},
        spec::AttentionKind,
    };
    use std::collections::HashSet;

    /// Every role a visit of this kind reads.
    fn roles_of(mixer: AttentionKind, ffn: Ffn) -> Vec<Role> {
        let mut roles = vec![Role::AttnNorm];
        roles.extend(match mixer {
            AttentionKind::Full => vec![
                Role::QProj,
                Role::KProj,
                Role::VProj,
                Role::OProj,
                Role::QNorm,
                Role::KNorm,
            ],
            _ => vec![
                Role::DeltaQkv,
                Role::DeltaGate,
                Role::DeltaOut,
                Role::DeltaConv,
                Role::DeltaDecay,
                Role::DeltaAlpha,
                Role::DeltaBeta,
                Role::DeltaDtBias,
                Role::DeltaNorm,
            ],
        });
        if let Ffn::Moe { .. } = ffn {
            roles.extend([
                Role::FfnNorm,
                Role::Router,
                Role::SharedRouterGate,
                Role::ExpertGate,
                Role::ExpertUp,
                Role::ExpertDown,
                Role::SharedGate,
                Role::SharedUp,
                Role::SharedDown,
            ]);
        }
        roles
    }

    #[test]
    fn the_recipe_names_every_tensor_of_the_inventory_once() {
        let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).unwrap();
        let types = |_: &str| Some(GgufType::Q4K);
        let map = Qwen36GgufMap::new(&inventory, &types);
        let recipe = recipe().unwrap();
        let mut reached = HashSet::new();
        for (index, visit) in recipe.visits.iter().enumerate() {
            for role in roles_of(visit.mixer, visit.ffn) {
                let name = map
                    .layer(visit.stored, role)
                    .unwrap_or_else(|| panic!("visit {index}: {role:?} has no tensor"));
                assert!(reached.insert(name.clone()), "{name} is named twice");
            }
        }
        for global in [Global::Embedding, Global::FinalNorm, Global::Head] {
            let name = map.global(global).expect("a global tensor");
            assert!(reached.insert(name));
        }
        let listed: HashSet<String> = inventory
            .tensors()
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect();
        assert_eq!(reached, listed, "the recipe and the inventory differ");
    }

    #[test]
    fn a_role_a_layer_does_not_have_is_none() {
        let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).unwrap();
        let types = |_: &str| None;
        let map = Qwen36GgufMap::new(&inventory, &types);
        // Layer 3 is full attention, layer 0 is DeltaNet.
        assert!(map.layer(3, Role::QProj).is_some());
        assert_eq!(map.layer(0, Role::QProj), None);
        assert!(map.layer(0, Role::DeltaQkv).is_some());
        assert_eq!(map.layer(3, Role::DeltaQkv), None);
        assert_eq!(map.layer(0, Role::DenseUp), None);
        assert_eq!(
            map.layer(0, Role::Hyper(Site::Mixer, SiteTensor::Down)),
            None
        );
        assert_eq!(map.global(Global::NgramTable), None);
        assert_eq!(
            map.layer(3, Role::ExpertDown).as_deref(),
            Some("blk.3.ffn_down_exps.weight")
        );
    }

    #[test]
    fn the_stored_type_of_a_role_comes_from_the_header_of_its_tensor() {
        let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).unwrap();
        // A file that holds the experts of the last layers at a higher precision.
        let types = |name: &str| {
            if name.starts_with("blk.3") && name.contains("exps") {
                Some(GgufType::Q80)
            } else {
                Some(GgufType::Q4K)
            }
        };
        let map = Qwen36GgufMap::new(&inventory, &types);
        assert_eq!(map.storage(3, Role::ExpertUp), Some(GgufType::Q80));
        assert_eq!(map.storage(2, Role::ExpertUp), Some(GgufType::Q4K));
        assert_eq!(map.storage(3, Role::QProj), Some(GgufType::Q4K));
        assert_eq!(map.storage(0, Role::QProj), None);
    }
}
