//! A model as a list of layer visits: the description a resident decode step runs.
//!
//! A family used to write its own stepper: the order of the mixer, the feed-forward, the
//! residual form and the head was in code, per family. A [`Recipe`] puts that order in data.
//! A model is a list of [`Visit`]s; a visit names the stored layer whose weights it reads, the
//! mixer (an attention kind or the gated DeltaNet), and the feed-forward (dense, routed experts
//! with an optional shared expert, or none). Before the visits sit the input stages (the token
//! embedding and, for a model with them, n-gram rows), after them the head and an optional
//! multi-token-prediction head. A looped model lists a stored layer more than once: the weights
//! are shared and each visit keeps its own state (K/V cache, DeltaNet state).
//!
//! Weights come through a [`WeightMap`]: the recipe names a [`Role`] of a stored layer and the
//! map says which tensor of the checkpoint holds it, so one step serves a checkpoint named in
//! any convention. A family supplies a [`FamilySpec`], its recipe (usually
//! [`Recipe::from_spec`], then its own adjustments) and a weight map; it writes a layer kernel
//! only when its math is new.
//!
//! Two seams are in the data on purpose. `window_rows` is the number of token rows a step takes:
//! 1 for plain decoding, more for a window that verifies drafted tokens in one pass, so
//! speculation is a step that already takes rows, not a second step. And the kernel of a matrix
//! is chosen by the type of its own tensor ([`WeightMap::storage`]), per stored layer, so a
//! checkpoint that quantizes layers differently (early layers at one precision, late layers at
//! another) needs no recipe change.
//!
//! This module is pure data and checks: no card types, so the rules are tested on the host.

use crate::EngError;
use crate::family::spec::{Act, AttentionKind, FamilySpec, HyperConnection, Router};
use crate::weights::GgufType;

/// The form of the residual stream through the visits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Residual {
    /// One stream of `hidden` values: `x + f(norm(x))` at every sublayer.
    Plain,
    /// `branches` streams mixed and applied by hyper-connection sites around each sublayer.
    Hyper(HyperConnection),
}

/// The feed-forward of a visit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Ffn {
    None,
    Dense {
        intermediate: u32,
        act: Act,
    },
    /// Routed experts, with a shared expert of `shared_inter` values when it is not zero.
    Moe {
        experts: u32,
        top_k: u32,
        inter: u32,
        shared_inter: u32,
        router: Router,
        act: Act,
    },
}

/// One visit of a stored layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Visit {
    /// The stored layer whose weights this visit reads.
    pub(crate) stored: u32,
    pub(crate) mixer: AttentionKind,
    pub(crate) ffn: Ffn,
}

/// What enters the first visit besides the token embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputRows {
    None,
    /// Rows hashed from the token history and folded in after `layer` (the n-gram memory and
    /// the per-layer embedding block of Flash-Next).
    Ngram {
        layer: u32,
    },
}

/// The output head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Head {
    /// Whether the output projection is the embedding table.
    pub(crate) tied: bool,
    /// The logit soft cap, when the model has one.
    pub(crate) softcap: Option<f32>,
}

/// A multi-token-prediction head: `layers` extra visits after the main stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MtpHead {
    pub(crate) layers: u32,
}

/// A model as data.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Recipe {
    pub(crate) hidden: u32,
    pub(crate) vocab: u32,
    pub(crate) residual: Residual,
    pub(crate) input: InputRows,
    pub(crate) visits: Vec<Visit>,
    pub(crate) head: Head,
    pub(crate) mtp: Option<MtpHead>,
    /// Token rows one step takes: 1 to decode, more to verify a drafted window.
    pub(crate) window_rows: u32,
}

impl Recipe {
    /// The recipe of a family that visits each of its layers once, in order.
    pub(crate) fn from_spec(spec: &FamilySpec) -> Result<Self, EngError> {
        if spec.layers_attention.len() != spec.layers as usize {
            return Err(refuse(format!(
                "{} lists {} attention kinds for {} layers",
                spec.name,
                spec.layers_attention.len(),
                spec.layers
            )));
        }
        let ffn = match spec.feed_forward {
            Some(dense) => Ffn::Dense {
                intermediate: dense.intermediate,
                act: spec.act,
            },
            None => Ffn::Moe {
                experts: spec.experts,
                top_k: spec.top_k,
                inter: spec.moe_inter,
                shared_inter: spec.shared_inter,
                router: spec.router,
                act: spec.act,
            },
        };
        let visits = spec
            .layers_attention
            .iter()
            .enumerate()
            .map(|(layer, &mixer)| Visit {
                stored: layer as u32,
                mixer,
                ffn,
            })
            .collect();
        let recipe = Self {
            hidden: spec.hidden,
            vocab: spec.vocab,
            residual: spec.hyper.map_or(Residual::Plain, Residual::Hyper),
            input: InputRows::None,
            visits,
            head: Head {
                tied: false,
                softcap: None,
            },
            mtp: spec.mtp.then_some(MtpHead { layers: 1 }),
            window_rows: 1,
        };
        recipe.validate()?;
        Ok(recipe)
    }

    /// The same model visiting its stored layers in `order` (a loop lists a layer twice).
    /// Each visit keeps the mixer and feed-forward of the stored layer it names.
    pub(crate) fn visiting(self, order: &[u32]) -> Result<Self, EngError> {
        let stored = self.stored_layers();
        let mut by_layer: Vec<Option<Visit>> = vec![None; stored];
        for visit in &self.visits {
            by_layer[visit.stored as usize].get_or_insert(*visit);
        }
        let visits = order
            .iter()
            .map(|&layer| {
                by_layer
                    .get(layer as usize)
                    .copied()
                    .flatten()
                    .ok_or_else(|| {
                        refuse(format!("a visit names stored layer {layer} of {stored}"))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let recipe = Self { visits, ..self };
        recipe.validate()?;
        Ok(recipe)
    }

    /// The same model with n-gram rows folded in after `layer`.
    pub(crate) fn with_ngram_rows(self, layer: u32) -> Result<Self, EngError> {
        let recipe = Self {
            input: InputRows::Ngram { layer },
            ..self
        };
        recipe.validate()?;
        Ok(recipe)
    }

    /// The same model with a step that takes `rows` token rows.
    pub(crate) fn with_window(self, rows: u32) -> Result<Self, EngError> {
        let recipe = Self {
            window_rows: rows,
            ..self
        };
        recipe.validate()?;
        Ok(recipe)
    }

    /// Stored layers the visits read: one more than the largest index.
    pub(crate) fn stored_layers(&self) -> usize {
        self.visits
            .iter()
            .map(|visit| visit.stored as usize + 1)
            .max()
            .unwrap_or(0)
    }

    /// Visits whose mixer keeps state of its own, in visit order: what a resident step allocates
    /// a K/V cache or a DeltaNet state for. A looped layer counts once per visit.
    pub(crate) fn stateful_visits(&self) -> impl Iterator<Item = (usize, AttentionKind)> + '_ {
        self.visits
            .iter()
            .enumerate()
            .map(|(index, visit)| (index, visit.mixer))
    }

    /// Visits of each kind of mixer, for the sizing of per-kind state.
    pub(crate) fn count(&self, kind: AttentionKind) -> usize {
        self.visits
            .iter()
            .filter(|visit| visit.mixer == kind)
            .count()
    }

    /// The checks every step relies on, so a step does not repeat them.
    pub(crate) fn validate(&self) -> Result<(), EngError> {
        if self.window_rows == 0 {
            return Err(refuse("a step takes at least one token row".into()));
        }
        if self.visits.is_empty() || self.hidden == 0 || self.vocab == 0 {
            return Err(refuse(
                "a recipe needs visits, a width and a vocabulary".into(),
            ));
        }
        let stored = self.stored_layers();
        for (index, visit) in self.visits.iter().enumerate() {
            if let Ffn::Moe {
                experts,
                top_k,
                inter,
                ..
            } = visit.ffn
                && (top_k == 0 || top_k > experts || inter == 0)
            {
                return Err(refuse(format!(
                    "visit {index}: {top_k} picks of {experts} experts of width {inter}"
                )));
            }
            if let Ffn::Dense {
                intermediate: 0, ..
            } = visit.ffn
            {
                return Err(refuse(format!(
                    "visit {index}: a dense feed-forward of width 0"
                )));
            }
        }
        // A stored layer is one block of weights: every visit that names it must agree on its
        // mixer and feed-forward, or the weights could not serve both.
        for layer in 0..stored as u32 {
            let mut named = self.visits.iter().filter(|visit| visit.stored == layer);
            if let Some(first) = named.next()
                && let Some(other) = named.find(|visit| **visit != *first)
            {
                return Err(refuse(format!(
                    "stored layer {layer} is visited as {first:?} and as {other:?}"
                )));
            }
        }
        if let InputRows::Ngram { layer } = self.input
            && layer as usize >= self.visits.len()
        {
            return Err(refuse(format!(
                "n-gram rows enter after visit {layer} of {}",
                self.visits.len()
            )));
        }
        if let Residual::Hyper(hyper) = self.residual
            && hyper.branches < 2
        {
            return Err(refuse(format!(
                "a hyper-connected stream needs 2 or more branches, got {}",
                hyper.branches
            )));
        }
        Ok(())
    }
}

/// The tensors of one stored layer a recipe can ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    AttnNorm,
    QProj,
    KProj,
    VProj,
    OProj,
    QNorm,
    KNorm,
    /// The gated DeltaNet projections, convolution and recurrence parameters.
    DeltaQkv,
    DeltaGate,
    DeltaOut,
    DeltaConv,
    DeltaDecay,
    /// The QSA indexer projection.
    IndexProj,
    FfnNorm,
    Router,
    SharedRouterGate,
    ExpertGate,
    ExpertUp,
    ExpertDown,
    SharedGate,
    SharedUp,
    SharedDown,
    DenseGate,
    DenseUp,
    DenseDown,
    /// The hyper-connection sites before the mixer and before the feed-forward.
    MixerSite,
    FfnSite,
}

/// The tensors outside the layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Global {
    Embedding,
    FinalNorm,
    Head,
    /// The hyper-connection mixer that feeds the head.
    HeadSite,
    NgramTable,
}

/// Which tensor of a checkpoint holds a role: the one place a naming convention lives.
pub(crate) trait WeightMap {
    /// The tensor of `role` in stored layer `layer`, or `None` when this checkpoint has none.
    fn layer(&self, layer: u32, role: Role) -> Option<String>;
    fn global(&self, role: Global) -> Option<String>;
    /// The stored type of the tensor of `role` in `layer`, which selects its kernel. The expert
    /// stacks of different layers may differ.
    fn storage(&self, layer: u32, role: Role) -> Option<GgufType>;
}

fn refuse(detail: String) -> EngError {
    EngError::Checkpoint(format!("recipe: {detail}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::spec::{QWEN3_6_35B_A3B, QWEN3_8_FLASH_NEXT};

    #[test]
    fn a_family_spec_gives_one_visit_per_layer_with_its_mixer_and_experts() {
        let recipe = Recipe::from_spec(&QWEN3_6_35B_A3B).unwrap();
        assert_eq!(recipe.visits.len(), 40);
        assert_eq!(recipe.stored_layers(), 40);
        assert_eq!(recipe.count(AttentionKind::Full), 10);
        assert_eq!(recipe.count(AttentionKind::DeltaNet), 30);
        assert_eq!(recipe.residual, Residual::Plain);
        assert!(matches!(
            recipe.visits[0].ffn,
            Ffn::Moe {
                experts: 256,
                top_k: 8,
                shared_inter: 512,
                ..
            }
        ));
    }

    #[test]
    fn flash_next_is_hyper_connected_with_qsa_layers_and_an_mtp_head() {
        let recipe = Recipe::from_spec(&QWEN3_8_FLASH_NEXT)
            .unwrap()
            .with_ngram_rows(1)
            .unwrap();
        assert_eq!(recipe.visits.len(), 48);
        assert!(matches!(recipe.residual, Residual::Hyper(h) if h.branches == 4));
        assert!(recipe.count(AttentionKind::Qsa) > 0);
        assert_eq!(recipe.mtp, Some(MtpHead { layers: 1 }));
        assert_eq!(recipe.input, InputRows::Ngram { layer: 1 });
    }

    #[test]
    fn a_looped_model_visits_each_stored_layer_twice_with_state_per_visit() {
        let recipe = Recipe::from_spec(&QWEN3_6_35B_A3B).unwrap();
        let order: Vec<u32> = (0..40).chain(0..40).collect();
        let looped = recipe.visiting(&order).unwrap();
        assert_eq!(looped.visits.len(), 80);
        assert_eq!(looped.stored_layers(), 40);
        assert_eq!(looped.stateful_visits().count(), 80);
        assert_eq!(looped.count(AttentionKind::Full), 20);
    }

    #[test]
    fn a_visit_of_a_layer_that_is_not_stored_is_refused_by_name() {
        let recipe = Recipe::from_spec(&QWEN3_6_35B_A3B).unwrap();
        let error = format!("{:?}", recipe.visiting(&[0, 40]).unwrap_err());
        assert!(error.contains("stored layer 40 of 40"), "{error}");
    }

    #[test]
    fn two_visits_of_one_stored_layer_must_agree_on_what_the_layer_is() {
        let mut recipe = Recipe::from_spec(&QWEN3_6_35B_A3B).unwrap();
        recipe.visits.push(Visit {
            mixer: AttentionKind::Full,
            ..recipe.visits[0]
        });
        let error = format!("{:?}", recipe.validate().unwrap_err());
        assert!(error.contains("stored layer 0 is visited as"), "{error}");
    }

    #[test]
    fn picks_above_the_expert_count_and_a_single_branch_are_refused() {
        let mut recipe = Recipe::from_spec(&QWEN3_6_35B_A3B).unwrap();
        recipe.visits[3].ffn = Ffn::Moe {
            experts: 4,
            top_k: 8,
            inter: 512,
            shared_inter: 0,
            router: Router::SoftmaxTopK { renorm: true },
            act: Act::Silu,
        };
        assert!(recipe.validate().is_err());
        let mut hyper = Recipe::from_spec(&QWEN3_8_FLASH_NEXT).unwrap();
        hyper.residual = Residual::Hyper(HyperConnection {
            branches: 1,
            rank: 8,
        });
        assert!(hyper.validate().is_err());
    }

    #[test]
    fn a_weight_map_names_the_tensor_of_a_role() {
        struct Hf;
        impl WeightMap for Hf {
            fn layer(&self, layer: u32, role: Role) -> Option<String> {
                match role {
                    Role::QProj => Some(format!("model.layers.{layer}.self_attn.q_proj.weight")),
                    _ => None,
                }
            }
            fn global(&self, role: Global) -> Option<String> {
                (role == Global::Embedding).then(|| "model.embed_tokens.weight".to_string())
            }
            fn storage(&self, layer: u32, role: Role) -> Option<GgufType> {
                // Early layers hold their experts at one precision, late layers at another.
                (role == Role::ExpertDown).then_some(if layer < 20 {
                    GgufType::Nvfp4
                } else {
                    GgufType::Q80
                })
            }
        }
        let map: &dyn WeightMap = &Hf;
        assert_eq!(
            map.layer(3, Role::QProj).as_deref(),
            Some("model.layers.3.self_attn.q_proj.weight")
        );
        assert_eq!(map.layer(3, Role::KProj), None);
        assert!(map.global(Global::Embedding).is_some());
        assert_eq!(map.storage(3, Role::ExpertDown), Some(GgufType::Nvfp4));
        assert_eq!(map.storage(30, Role::ExpertDown), Some(GgufType::Q80));
    }

    #[test]
    fn a_window_takes_one_or_more_rows_and_never_zero() {
        let recipe = Recipe::from_spec(&QWEN3_6_35B_A3B).unwrap();
        assert_eq!(recipe.window_rows, 1);
        assert_eq!(recipe.clone().with_window(4).unwrap().window_rows, 4);
        assert!(recipe.with_window(0).is_err());
    }
}
