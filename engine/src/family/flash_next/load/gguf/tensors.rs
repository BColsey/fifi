//! The tensors of a `qwen4exp` GGUF mapped to the weight structures the Flash-Next executor
//! consumes.
//!
//! [`TensorMap::build`] derives every tensor the model needs from the configuration (layer
//! kinds, widths, counts), looks each up by its GGUF name, and checks its dimensions. A missing
//! tensor or a wrong shape refuses the load naming the tensor. A tensor of the file that no
//! role claims is reported in [`TensorMap::unmapped`] and left unread. Every role has the form
//! the device holds it in ([`Form`], the counterpart of the EXL3 trunk's forms) and the
//! convention of the stored values ([`Convention`]) relative to what the EXL3 pack stores.
//!
//! The GGUF layout differs from the EXL3 pack in four places the executor must honour:
//! - Zero-centred norm gammas are stored as `1 + w`: the hyper-connection norms, the indexer
//!   norms, the PLE norms and the attention `q_norm` and `k_norm`. The EXL3 pack stores `w`.
//!   The kernels that add the 1 themselves (hyper-connection, indexer, the host PLE step) take
//!   `stored - 1`; the attention prep reads `1 + w` and takes the stored value as it is.
//! - The DeltaNet value heads are in tiled order (key head `h % key_heads`), where the pack
//!   keeps the grouped order. The scan kernel has both orders (`GdnKeyHeads`).
//! - `A_log` is stored as `-exp(A_log)`, the decay form the scan reads.
//! - The indexer projection is two tensors, queries then the key; the pack has one fused matrix.

use std::collections::HashSet;

use super::{
    super::Dims,
    config::{GgufConfig, is_float},
};
use crate::{
    EngError,
    weights::{GgufHeader, GgufType},
};

/// Where one of the three hyper-connection sites sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::family::flash_next) enum Site {
    /// The final mixer that feeds the head (no inject projection).
    Mixer,
    /// Before the token mixer of a layer.
    Attn,
    /// Before the MoE block of a layer.
    Mlp,
}

/// What a tensor is for: one field of the executor's weight structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::family::flash_next) enum Role {
    Embedding,
    Head,
    HyperNorm(Site),
    HyperDown(Site),
    HyperUp(Site),
    HyperInject(Site),
    DeltaQkv,
    DeltaZ,
    DeltaBeta,
    DeltaAlpha,
    DeltaOut,
    DeltaConv,
    DeltaDecay,
    DeltaDtBias,
    DeltaNorm,
    QsaQ,
    QsaK,
    QsaV,
    QsaO,
    QsaIndexQ,
    QsaIndexK,
    QsaQNorm,
    QsaKNorm,
    QsaIndexQNorm,
    QsaIndexKNorm,
    Router,
    SharedGate,
    SharedUp,
    SharedDown,
    SharedScalar,
    ExpertGate,
    ExpertUp,
    ExpertDown,
    PleKey,
    PleValue,
    PleNormKey,
    PleNormQuery,
    PleNormConv,
    PleConv,
    NgramTable,
}

/// Which memory a role's bytes serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::family::flash_next) enum Place {
    /// Uploaded once and resident on the card for the run.
    Device,
    /// Read on the host: the embedding rows of each token and the PLE vectors.
    Host,
    /// The routed experts, streamed from the file through the expert tiers.
    Experts,
    /// The n-gram table, read row by row from the file.
    Ngram,
}

/// How the device holds a tensor (the counterpart of the EXL3 trunk's `Form`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::flash_next) enum Form {
    /// The GGUF blocks as stored; the matvec kernel of the tensor's type reads them.
    Blocks,
    /// Widened to F16 at upload, for the kernels that read half-precision weights.
    Half,
    /// Widened to F32 at upload.
    Float,
    /// A norm gamma as `1 + w` in F32.
    NormPlusOne,
    /// `-exp(A_log)` in F32, the decay the scan reads.
    Decay,
    /// Not held on the device.
    None,
}

/// How the stored values relate to the EXL3 pack's stored values for the same role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::flash_next) enum Convention {
    AsStored,
    /// Stored as `1 + w`; the EXL3 pack stores `w`. A consumer that adds the 1 itself
    /// subtracts it at load.
    GammaOnePlusW,
}

/// One tensor of the file claimed by a role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::flash_next) struct Mapped {
    pub(in crate::family::flash_next) role: Role,
    pub(in crate::family::flash_next) layer: Option<usize>,
    pub(in crate::family::flash_next) name: String,
    pub(in crate::family::flash_next) ty: GgufType,
    pub(in crate::family::flash_next) dims: Vec<u64>,
    pub(in crate::family::flash_next) bytes: u64,
    pub(in crate::family::flash_next) shard: usize,
    pub(in crate::family::flash_next) place: Place,
    pub(in crate::family::flash_next) form: Form,
    pub(in crate::family::flash_next) convention: Convention,
}

impl Mapped {
    /// Bytes this tensor takes on the card in its device form (zero off the card).
    pub(in crate::family::flash_next) fn device_bytes(&self) -> u64 {
        let elements: u64 = self.dims.iter().product();
        match self.form {
            Form::Blocks => self.bytes,
            Form::Half => elements * 2,
            Form::Float | Form::NormPlusOne | Form::Decay => elements * 4,
            Form::None => 0,
        }
    }

    /// Whether a matvec kernel of this tensor's type reads it (block formats only).
    fn needs_kernel(&self) -> bool {
        matches!(self.form, Form::Blocks) || self.place == Place::Experts
    }
}

/// A tensor type some matrix of the model has and no CUDA decode kernel reads yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::flash_next) struct KernelGap {
    pub(in crate::family::flash_next) ty: GgufType,
    pub(in crate::family::flash_next) tensors: usize,
    pub(in crate::family::flash_next) bytes: u64,
    pub(in crate::family::flash_next) first: String,
}

/// Tensor types the CUDA decode path can read as stored blocks: the in-tree matvec kernels
/// (`QuantType` of `kernels::cuda::resident::matvec_q`). A native-CUDA test keeps this list
/// equal to what the weight upload accepts.
pub(in crate::family::flash_next) const DECODE_BLOCK_TYPES: [GgufType; 18] = [
    GgufType::Q4K,
    GgufType::Q50,
    GgufType::Q6K,
    GgufType::Q80,
    GgufType::Q5K,
    GgufType::Q40,
    GgufType::Q3K,
    GgufType::Iq4Xs,
    GgufType::Iq4Nl,
    GgufType::Iq2Xxs,
    GgufType::Iq2Xs,
    GgufType::Iq2S,
    GgufType::Iq3Xxs,
    GgufType::Iq3S,
    GgufType::Iq1S,
    GgufType::Iq1M,
    GgufType::Q20,
    GgufType::Q2K,
];

/// Tensor types the routed-expert kernel reads: the vendored matvec arms and the in-house
/// codebook kernels (every IQ type, Q2_0 and Q2_K). The in-house Q5_K, Q4_0 and Q3_K dense
/// kernels have no id-indexed expert form, so an expert matrix of those types is a gap although
/// a dense matrix of the same type is not.
pub(in crate::family::flash_next) const EXPERT_BLOCK_TYPES: [GgufType; 15] = [
    GgufType::Q4K,
    GgufType::Q50,
    GgufType::Q6K,
    GgufType::Q80,
    GgufType::Iq4Xs,
    GgufType::Iq4Nl,
    GgufType::Iq2Xxs,
    GgufType::Iq2Xs,
    GgufType::Iq2S,
    GgufType::Iq3Xxs,
    GgufType::Iq3S,
    GgufType::Iq1S,
    GgufType::Iq1M,
    GgufType::Q20,
    GgufType::Q2K,
];

/// What a role requires of its tensor's type.
#[derive(Clone, Copy)]
enum Kind {
    /// A weight matrix of any block or float type.
    Matrix,
    /// Floating-point values (F32, F16 or BF16).
    Vector,
}

struct Expect {
    role: Role,
    layer: Option<usize>,
    names: Vec<String>,
    dims: Vec<u64>,
    kind: Kind,
}

fn site_prefix(site: Site, layer: Option<usize>) -> String {
    match (site, layer) {
        (Site::Mixer, _) => "output_hc".to_string(),
        (Site::Attn, Some(layer)) => format!("blk.{layer}.hc_attn"),
        (Site::Mlp, Some(layer)) => format!("blk.{layer}.hc_ffn"),
        _ => unreachable!("a layer site has a layer"),
    }
}

/// Every tensor the model needs, in the order of the executor's structures.
fn expected(dims: &Dims, rows_needed: u64) -> Vec<Expect> {
    let d = |v: u32| u64::from(v);
    let (hidden, width, rank, branches) = (
        d(dims.hidden),
        d(dims.width),
        d(dims.rank),
        d(dims.branches),
    );
    let mut out = Vec::new();
    let mut add = |role, layer: Option<usize>, names: Vec<String>, shape: Vec<u64>, kind| {
        out.push(Expect {
            role,
            layer,
            names,
            dims: shape,
            kind,
        });
    };
    let one = |name: &str| vec![name.to_string()];
    add(
        Role::Embedding,
        None,
        one("token_embd.weight"),
        vec![hidden, d(dims.vocab)],
        Kind::Matrix,
    );
    // A model without its own output matrix reuses the embedding.
    add(
        Role::Head,
        None,
        vec!["output.weight".into(), "token_embd.weight".into()],
        vec![hidden, d(dims.vocab)],
        Kind::Matrix,
    );
    let hyper = |add: &mut dyn FnMut(Role, Option<usize>, Vec<String>, Vec<u64>, Kind),
                 site: Site,
                 layer: Option<usize>| {
        let prefix = site_prefix(site, layer);
        add(
            Role::HyperNorm(site),
            layer,
            vec![format!("{prefix}_norm.weight")],
            vec![width],
            Kind::Vector,
        );
        add(
            Role::HyperDown(site),
            layer,
            vec![format!("{prefix}_down.weight")],
            vec![width, rank],
            Kind::Matrix,
        );
        add(
            Role::HyperUp(site),
            layer,
            vec![format!("{prefix}_up.weight")],
            vec![rank, width],
            Kind::Matrix,
        );
        if site != Site::Mixer {
            add(
                Role::HyperInject(site),
                layer,
                vec![format!("{prefix}_inject.weight")],
                vec![width, branches],
                Kind::Matrix,
            );
        }
    };
    hyper(&mut add, Site::Mixer, None);
    let key_width = d(dims.key_heads) * d(dims.key_dim);
    let value_width = d(dims.value_heads) * d(dims.value_dim);
    let conv_dim = 2 * key_width + value_width;
    let q_width = d(dims.heads_q) * d(dims.head_dim);
    let kv_width = d(dims.heads_kv) * d(dims.head_dim);
    let (inter, shared, experts) = (d(dims.moe_inter), d(dims.shared_inter), d(dims.experts));
    for (layer, &attention) in dims.kinds.iter().enumerate() {
        let at = Some(layer);
        let blk = |suffix: &str| vec![format!("blk.{layer}.{suffix}")];
        hyper(&mut add, Site::Attn, at);
        if attention {
            add(
                Role::QsaQ,
                at,
                blk("attn_q.weight"),
                vec![hidden, 2 * q_width],
                Kind::Matrix,
            );
            add(
                Role::QsaK,
                at,
                blk("attn_k.weight"),
                vec![hidden, kv_width],
                Kind::Matrix,
            );
            add(
                Role::QsaV,
                at,
                blk("attn_v.weight"),
                vec![hidden, kv_width],
                Kind::Matrix,
            );
            add(
                Role::QsaO,
                at,
                blk("attn_output.weight"),
                vec![q_width, hidden],
                Kind::Matrix,
            );
            add(
                Role::QsaIndexQ,
                at,
                blk("indexer.q_proj.weight"),
                vec![hidden, d(dims.idx_heads) * d(dims.idx_dim)],
                Kind::Matrix,
            );
            add(
                Role::QsaIndexK,
                at,
                blk("indexer.k_proj.weight"),
                vec![hidden, d(dims.idx_dim)],
                Kind::Matrix,
            );
            add(
                Role::QsaQNorm,
                at,
                blk("attn_q_norm.weight"),
                vec![d(dims.head_dim)],
                Kind::Vector,
            );
            add(
                Role::QsaKNorm,
                at,
                blk("attn_k_norm.weight"),
                vec![d(dims.head_dim)],
                Kind::Vector,
            );
            add(
                Role::QsaIndexQNorm,
                at,
                blk("indexer.q_norm.weight"),
                vec![d(dims.idx_dim)],
                Kind::Vector,
            );
            add(
                Role::QsaIndexKNorm,
                at,
                blk("indexer.k_norm.weight"),
                vec![d(dims.idx_dim)],
                Kind::Vector,
            );
        } else {
            add(
                Role::DeltaQkv,
                at,
                blk("attn_qkv.weight"),
                vec![hidden, conv_dim],
                Kind::Matrix,
            );
            add(
                Role::DeltaZ,
                at,
                blk("attn_gate.weight"),
                vec![hidden, value_width],
                Kind::Matrix,
            );
            add(
                Role::DeltaBeta,
                at,
                blk("ssm_beta.weight"),
                vec![hidden, d(dims.value_heads)],
                Kind::Matrix,
            );
            add(
                Role::DeltaAlpha,
                at,
                blk("ssm_alpha.weight"),
                vec![hidden, d(dims.value_heads)],
                Kind::Matrix,
            );
            add(
                Role::DeltaOut,
                at,
                blk("ssm_out.weight"),
                vec![value_width, hidden],
                Kind::Matrix,
            );
            add(
                Role::DeltaConv,
                at,
                blk("ssm_conv1d.weight"),
                vec![d(dims.conv_kernel), conv_dim],
                Kind::Vector,
            );
            add(
                Role::DeltaDecay,
                at,
                blk("ssm_a"),
                vec![d(dims.value_heads)],
                Kind::Vector,
            );
            add(
                Role::DeltaDtBias,
                at,
                blk("ssm_dt.bias"),
                vec![d(dims.value_heads)],
                Kind::Vector,
            );
            add(
                Role::DeltaNorm,
                at,
                blk("ssm_norm.weight"),
                vec![d(dims.value_dim)],
                Kind::Vector,
            );
        }
        if layer == dims.ple_layer {
            let embed = d(dims.ple_embed);
            add(
                Role::PleKey,
                at,
                blk("ple_key.weight"),
                vec![embed, width],
                Kind::Matrix,
            );
            add(
                Role::PleValue,
                at,
                blk("ple_value.weight"),
                vec![embed, hidden],
                Kind::Matrix,
            );
            add(
                Role::PleNormKey,
                at,
                blk("ple_norm_key.weight"),
                vec![width],
                Kind::Vector,
            );
            add(
                Role::PleNormQuery,
                at,
                blk("ple_norm_query.weight"),
                vec![width],
                Kind::Vector,
            );
            add(
                Role::PleNormConv,
                at,
                blk("ple_norm_conv.weight"),
                vec![width],
                Kind::Vector,
            );
            add(
                Role::PleConv,
                at,
                blk("ple_conv1d.weight"),
                vec![d(dims.ple_conv), width],
                Kind::Vector,
            );
        }
        hyper(&mut add, Site::Mlp, at);
        add(
            Role::Router,
            at,
            blk("ffn_gate_inp.weight"),
            vec![hidden, experts],
            Kind::Vector,
        );
        add(
            Role::SharedScalar,
            at,
            blk("ffn_gate_inp_shexp.weight"),
            vec![hidden],
            Kind::Vector,
        );
        add(
            Role::SharedGate,
            at,
            blk("ffn_gate_shexp.weight"),
            vec![hidden, shared],
            Kind::Matrix,
        );
        add(
            Role::SharedUp,
            at,
            blk("ffn_up_shexp.weight"),
            vec![hidden, shared],
            Kind::Matrix,
        );
        add(
            Role::SharedDown,
            at,
            blk("ffn_down_shexp.weight"),
            vec![shared, hidden],
            Kind::Matrix,
        );
        add(
            Role::ExpertGate,
            at,
            blk("ffn_gate_exps.weight"),
            vec![hidden, inter, experts],
            Kind::Matrix,
        );
        add(
            Role::ExpertUp,
            at,
            blk("ffn_up_exps.weight"),
            vec![hidden, inter, experts],
            Kind::Matrix,
        );
        add(
            Role::ExpertDown,
            at,
            blk("ffn_down_exps.weight"),
            vec![inter, hidden, experts],
            Kind::Matrix,
        );
    }
    add(
        Role::NgramTable,
        None,
        vec!["per_layer_token_embd.weight".into()],
        vec![d(dims.ple_head_dim), rows_needed],
        Kind::Matrix,
    );
    out
}

/// The place, device form and value convention of a role.
fn traits_of(role: Role) -> (Place, Form, Convention) {
    use Convention::{AsStored, GammaOnePlusW};
    use Form::{Blocks, Decay, Float, Half, NormPlusOne};
    use Place::{Device, Experts, Host, Ngram};
    match role {
        Role::Embedding => (Host, Form::None, AsStored),
        Role::Head
        | Role::DeltaQkv
        | Role::DeltaZ
        | Role::DeltaOut
        | Role::QsaQ
        | Role::QsaK
        | Role::QsaV
        | Role::QsaO
        | Role::SharedGate
        | Role::SharedUp
        | Role::SharedDown
        | Role::PleKey
        | Role::PleValue => (Device, Blocks, AsStored),
        Role::HyperDown(_) | Role::HyperUp(_) | Role::HyperInject(_) => (Device, Half, AsStored),
        Role::QsaIndexQ | Role::QsaIndexK => (Device, Half, AsStored),
        Role::HyperNorm(_) => (Device, Float, GammaOnePlusW),
        Role::QsaIndexQNorm | Role::QsaIndexKNorm => (Device, Float, GammaOnePlusW),
        Role::QsaQNorm | Role::QsaKNorm => (Device, NormPlusOne, GammaOnePlusW),
        Role::DeltaDecay => (Device, Decay, AsStored),
        Role::DeltaBeta
        | Role::DeltaAlpha
        | Role::DeltaConv
        | Role::DeltaDtBias
        | Role::DeltaNorm
        | Role::Router
        | Role::SharedScalar => (Device, Float, AsStored),
        Role::PleNormKey | Role::PleNormQuery | Role::PleNormConv => {
            (Host, Form::None, GammaOnePlusW)
        }
        Role::PleConv => (Host, Form::None, AsStored),
        Role::ExpertGate | Role::ExpertUp | Role::ExpertDown => (Experts, Form::None, AsStored),
        Role::NgramTable => (Ngram, Form::None, AsStored),
    }
}

/// One tensor of the inventory: what [`TensorMap::build`] looks for.
#[cfg(test)]
pub(super) struct InventoryItem {
    pub(super) role: Role,
    pub(super) layer: Option<usize>,
    pub(super) name: String,
    pub(super) dims: Vec<u64>,
    pub(super) vector: bool,
}

/// Every tensor the model needs by its first-choice name (the tests write files from it).
#[cfg(test)]
pub(super) fn inventory(config: &GgufConfig) -> Vec<InventoryItem> {
    expected(&config.dims, config.ngram_rows_needed)
        .into_iter()
        .map(|want| InventoryItem {
            role: want.role,
            layer: want.layer,
            name: want.names[0].clone(),
            dims: want.dims,
            vector: matches!(want.kind, Kind::Vector),
        })
        .collect()
}

fn mismatch(detail: String) -> EngError {
    EngError::Checkpoint(format!("Flash-Next GGUF tensors: {detail}"))
}

/// The tensors of a file mapped to roles.
#[derive(Debug)]
pub(in crate::family::flash_next) struct TensorMap {
    entries: Vec<Mapped>,
    unmapped: Vec<String>,
    total: usize,
    mapped_names: usize,
    experts: u64,
}

impl TensorMap {
    /// Maps every tensor of `header` that a role of the model claims. A role whose tensor is
    /// missing, has other dimensions, or has a non-float type where floats are required
    /// refuses the whole map; the error lists up to five such tensors and the total.
    pub(in crate::family::flash_next) fn build(
        header: &GgufHeader,
        config: &GgufConfig,
    ) -> Result<Self, EngError> {
        let dims = &config.dims;
        let needed = expected(dims, config.ngram_rows_needed);
        let mut entries = Vec::with_capacity(needed.len());
        let mut problems = Vec::new();
        let mut claimed: HashSet<&str> = HashSet::new();
        for want in &needed {
            let found = want
                .names
                .iter()
                .find_map(|name| header.tensor(name).map(|tensor| (name, tensor)));
            let Some((name, tensor)) = found else {
                problems.push(format!("{} is missing", want.names[0]));
                continue;
            };
            let is_ngram = want.role == Role::NgramTable;
            // The n-gram table may hold more rows than the heads address.
            let dims_agree = if is_ngram {
                tensor.dims().len() == 2
                    && tensor.dims()[0] == want.dims[0]
                    && tensor.dims()[1] >= want.dims[1]
            } else {
                tensor.dims() == want.dims.as_slice()
            };
            if !dims_agree {
                problems.push(format!(
                    "{name} has dimensions {:?}, the model needs {:?}",
                    tensor.dims(),
                    want.dims
                ));
                continue;
            }
            if matches!(want.kind, Kind::Vector) && !is_float(tensor.ggml_type()) {
                problems.push(format!(
                    "{name} is {}, the model needs F32, F16 or BF16 values",
                    tensor.ggml_type().name()
                ));
                continue;
            }
            claimed.insert(name.as_str());
            let (place, form, convention) = traits_of(want.role);
            entries.push(Mapped {
                role: want.role,
                layer: want.layer,
                name: name.clone(),
                ty: tensor.ggml_type(),
                dims: tensor.dims().to_vec(),
                bytes: tensor.byte_len(),
                shard: tensor.shard(),
                place,
                form,
                convention,
            });
        }
        if !problems.is_empty() {
            let shown = problems
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ");
            return Err(mismatch(format!(
                "{} of the model's {} tensors are missing or do not fit: {shown}",
                problems.len(),
                needed.len()
            )));
        }
        let unmapped = header
            .tensors()
            .iter()
            .map(|tensor| tensor.name())
            .filter(|name| !claimed.contains(name))
            .map(str::to_owned)
            .collect();
        Ok(Self {
            entries,
            unmapped,
            total: header.tensors().len(),
            mapped_names: claimed.len(),
            experts: u64::from(dims.experts),
        })
    }

    /// Every claimed tensor, in the order of the executor's structures.
    pub(in crate::family::flash_next) fn entries(&self) -> &[Mapped] {
        &self.entries
    }

    /// Tensors in the file.
    pub(in crate::family::flash_next) fn total(&self) -> usize {
        self.total
    }

    /// Tensors of the file claimed by a role (a tied head and embedding count once).
    pub(in crate::family::flash_next) fn mapped(&self) -> usize {
        self.mapped_names
    }

    /// Names of file tensors no role claims.
    pub(in crate::family::flash_next) fn unmapped(&self) -> &[String] {
        &self.unmapped
    }

    /// The entry of `role` in `layer` (`None` for roles outside the layers).
    pub(in crate::family::flash_next) fn get(
        &self,
        role: Role,
        layer: Option<usize>,
    ) -> Option<&Mapped> {
        self.entries
            .iter()
            .find(|entry| entry.role == role && entry.layer == layer)
    }

    /// File bytes of the tensors served from `place`.
    pub(in crate::family::flash_next) fn file_bytes(&self, place: Place) -> u64 {
        self.entries
            .iter()
            .filter(|entry| entry.place == place)
            .map(|entry| entry.bytes)
            .sum()
    }

    /// Device bytes of the resident trunk in the forms the executor holds it in.
    pub(in crate::family::flash_next) fn device_bytes(&self) -> u64 {
        self.entries.iter().map(Mapped::device_bytes).sum()
    }

    /// Bytes of one expert's three matrices in `layer`, as stored in the file.
    pub(in crate::family::flash_next) fn expert_record_bytes(&self, layer: usize) -> u64 {
        [Role::ExpertGate, Role::ExpertUp, Role::ExpertDown]
            .into_iter()
            .filter_map(|role| self.get(role, Some(layer)))
            .map(|entry| entry.bytes / self.experts)
            .sum()
    }

    /// The tensor types that matrices of the model have and `supported` does not accept,
    /// each with its tensor count, bytes and first tensor, in order of first appearance.
    pub(in crate::family::flash_next) fn kernel_gaps(
        &self,
        supported: impl Fn(Place, GgufType) -> bool,
    ) -> Vec<KernelGap> {
        let mut gaps: Vec<KernelGap> = Vec::new();
        for entry in self.entries.iter().filter(|e| e.needs_kernel()) {
            if is_float(entry.ty) || supported(entry.place, entry.ty) {
                continue;
            }
            match gaps.iter_mut().find(|gap| gap.ty == entry.ty) {
                Some(gap) => {
                    gap.tensors += 1;
                    gap.bytes += entry.bytes;
                }
                None => gaps.push(KernelGap {
                    ty: entry.ty,
                    tensors: 1,
                    bytes: entry.bytes,
                    first: entry.name.clone(),
                }),
            }
        }
        gaps
    }
}

/// Whether the in-tree CUDA decode kernels read a matrix of `ty` served from `place` as stored
/// blocks: the dense matvec kernels for the trunk, the expert kernel for the routed experts.
pub(in crate::family::flash_next) fn has_decode_kernel(place: Place, ty: GgufType) -> bool {
    if place == Place::Experts {
        EXPERT_BLOCK_TYPES.contains(&ty)
    } else {
        DECODE_BLOCK_TYPES.contains(&ty)
    }
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod cuda_tests {
    use super::*;
    use crate::kernels::cuda::resident::matvec_q::QuantType;

    /// The GGUF type each matvec kernel format reads. The match has no wildcard: a kernel
    /// format added to `QuantType` stops this test compiling until the table and
    /// [`DECODE_BLOCK_TYPES`] are updated together.
    fn gguf_type(quant: QuantType) -> GgufType {
        match quant {
            QuantType::Q4K => GgufType::Q4K,
            QuantType::Q50 => GgufType::Q50,
            QuantType::Q6K => GgufType::Q6K,
            QuantType::Q80 => GgufType::Q80,
            QuantType::Q5K => GgufType::Q5K,
            QuantType::Q40 => GgufType::Q40,
            QuantType::Q3K => GgufType::Q3K,
            QuantType::Iq4Xs => GgufType::Iq4Xs,
            QuantType::Iq4Nl => GgufType::Iq4Nl,
            QuantType::Iq2S => GgufType::Iq2S,
            QuantType::Iq3Xxs => GgufType::Iq3Xxs,
            QuantType::Iq2Xxs => GgufType::Iq2Xxs,
            QuantType::Q20 => GgufType::Q20,
            QuantType::Iq2Xs => GgufType::Iq2Xs,
            QuantType::Iq3S => GgufType::Iq3S,
            QuantType::Q2K => GgufType::Q2K,
            QuantType::Iq1S => GgufType::Iq1S,
            QuantType::Iq1M => GgufType::Iq1M,
        }
    }

    /// Every matvec kernel format.
    const ALL_KERNELS: [QuantType; 18] = [
        QuantType::Q4K,
        QuantType::Q50,
        QuantType::Q6K,
        QuantType::Q80,
        QuantType::Q5K,
        QuantType::Q40,
        QuantType::Q3K,
        QuantType::Iq4Xs,
        QuantType::Iq4Nl,
        QuantType::Iq2S,
        QuantType::Iq3Xxs,
        QuantType::Iq2Xxs,
        QuantType::Q20,
        QuantType::Iq2Xs,
        QuantType::Iq3S,
        QuantType::Q2K,
        QuantType::Iq1S,
        QuantType::Iq1M,
    ];

    #[test]
    fn the_expert_block_types_are_the_vendored_arms_and_the_codebook_kernels() {
        // The vendored arms carry a `ggml_type`; the in-house dense-only formats (Q5_K, Q4_0,
        // Q3_K) have no id-indexed expert form; every codebook format has one. Same table, no
        // wildcard.
        let expert = |quant: QuantType| match quant {
            QuantType::Q4K | QuantType::Q50 | QuantType::Q6K | QuantType::Q80 => true,
            QuantType::Q5K | QuantType::Q40 | QuantType::Q3K => false,
            _ => quant.is_codebook(),
        };
        for quant in ALL_KERNELS {
            assert_eq!(
                EXPERT_BLOCK_TYPES.contains(&gguf_type(quant)),
                expert(quant),
                "{quant:?}"
            );
        }
    }

    #[test]
    fn the_decode_block_types_are_the_matvec_kernel_formats() {
        let kernels = ALL_KERNELS;
        let mut from_kernels: Vec<u32> = kernels.map(|q| gguf_type(q).id()).to_vec();
        let mut listed: Vec<u32> = DECODE_BLOCK_TYPES.map(GgufType::id).to_vec();
        from_kernels.sort_unstable();
        listed.sort_unstable();
        assert_eq!(listed, from_kernels);
        for quant in kernels {
            let ty = gguf_type(quant);
            assert_eq!(ty.block_elems() as usize, quant.block_size(), "{ty:?}");
            assert_eq!(ty.block_bytes() as usize, quant.block_bytes(), "{ty:?}");
        }
    }
}
