//! The stock-pack tensor names of Qwen3.6 and a header-only inspection of a pack against the
//! model shape.
//!
//! [`inspect_pack`] reads no payload: it walks every tensor the loader will read, checks its
//! stored dtype and shape against the model, requires the routed experts of a projection to
//! share one rate (a bank has one stride), and sums the device bytes the load will hold. A pack
//! of another size, rate, codebook or layout is refused here, before the first upload, and the
//! same walk runs on a CPU host for any pack on disk.

use std::collections::BTreeMap;

use crate::{
    EngError,
    family::{AttentionKind, qwen36::host::HostShape},
    tensor::Dtype,
    weights::StreamedCheckpoint,
};

/// Root of every layer tensor name in a stock pack.
const LAYER_ROOT: &str = "model.language_model.layers";
/// Pack name of the embedding table.
pub(super) const EMBED_NAME: &str = "model.language_model.embed_tokens.weight";
/// Pack name of the output norm.
pub(super) const NORM_NAME: &str = "model.language_model.norm.weight";
/// Pack prefix of the output head.
pub(super) const HEAD_PREFIX: &str = "lm_head";

/// Root of every tensor name of the MTP head's layers in a stock pack.
const MTP_LAYER_ROOT: &str = "mtp.layers";

/// Where the layers of a walk live in the pack: the model's own layers or the MTP head's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Root {
    Layers,
    Mtp,
}

impl Root {
    /// The full pack name of `suffix` in `layer`.
    pub(super) fn name(self, layer: usize, suffix: &str) -> String {
        let root = match self {
            Self::Layers => LAYER_ROOT,
            Self::Mtp => MTP_LAYER_ROOT,
        };
        format!("{root}.{layer}.{suffix}")
    }
}

/// The full pack name of `suffix` in `layer`.
pub(super) fn layer_name(layer: usize, suffix: &str) -> String {
    Root::Layers.name(layer, suffix)
}

/// The router weight of a layer under `root`: a plain (unquantized) tensor of expert scores.
pub(super) fn router_name_in(root: Root, layer: usize) -> String {
    root.name(layer, "mlp.gate.weight")
}

/// The sigmoid gate of the shared expert under `root`: a plain tensor of one row.
pub(super) fn shared_gate_name_in(root: Root, layer: usize) -> String {
    root.name(layer, "mlp.shared_expert_gate.weight")
}

/// One projection of one routed expert under `root`.
pub(super) fn expert_name_in(root: Root, layer: usize, expert: usize, part: ExpertPart) -> String {
    root.name(layer, &format!("mlp.experts.{expert}.{}", part.suffix()))
}

/// One projection of the shared expert under `root`.
pub(super) fn shared_name_in(root: Root, layer: usize, part: ExpertPart) -> String {
    root.name(layer, &format!("mlp.shared_expert.{}", part.suffix()))
}

/// The router weight of a layer of the model.
#[cfg(test)]
pub(super) fn router_name(layer: usize) -> String {
    router_name_in(Root::Layers, layer)
}

/// The sigmoid gate of the shared expert of a layer of the model.
#[cfg(test)]
pub(super) fn shared_gate_name(layer: usize) -> String {
    shared_gate_name_in(Root::Layers, layer)
}

/// One projection of one routed expert of a layer of the model.
#[cfg(test)]
pub(super) fn expert_name(layer: usize, expert: usize, part: ExpertPart) -> String {
    expert_name_in(Root::Layers, layer, expert, part)
}

/// One projection of the shared expert of a layer of the model.
#[cfg(test)]
pub(super) fn shared_name(layer: usize, part: ExpertPart) -> String {
    shared_name_in(Root::Layers, layer, part)
}

/// The three projections every expert owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExpertPart {
    Gate,
    Up,
    Down,
}

impl ExpertPart {
    pub(super) const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    fn suffix(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }
}

/// What a header-only walk of a pack found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PackReport {
    /// Device bytes the load holds: trellises as stored, scales and plain vectors widened to
    /// F32, the embedding as stored.
    pub(super) device_bytes: u64,
    /// Quantized matrices (a routed expert counts one per projection).
    pub(super) matrices: usize,
    /// Matrices per rate, keyed by twice the bits per weight.
    pub(super) rates: BTreeMap<u8, usize>,
    /// The codebook the matrices carry.
    pub(super) codebook: Marker,
}

/// The codebook marker tensor a matrix carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Marker {
    Mul1,
    Mcg,
}

fn refuse(detail: String) -> EngError {
    EngError::Checkpoint(format!("EXL3 pack: {detail}"))
}

/// The walk: the source and the running totals.
pub(super) struct Inspector<'a> {
    source: &'a StreamedCheckpoint,
    root: Root,
    bytes: u64,
    matrices: usize,
    rates: BTreeMap<u8, usize>,
    marker: Option<Marker>,
}

impl<'a> Inspector<'a> {
    /// A walk over the layers under `root` of `source`.
    pub(super) fn new(source: &'a StreamedCheckpoint, root: Root) -> Self {
        Self {
            source,
            root,
            bytes: 0,
            matrices: 0,
            rates: BTreeMap::new(),
            marker: None,
        }
    }

    /// What the walk found so far.
    pub(super) fn report(self) -> Result<PackReport, EngError> {
        Ok(PackReport {
            device_bytes: self.bytes,
            matrices: self.matrices,
            rates: self.rates,
            codebook: self
                .marker
                .ok_or_else(|| refuse("the pack has no matrix".into()))?,
        })
    }

    fn stored(&self, name: &str, want: &[u32]) -> Result<Dtype, EngError> {
        let (dtype, shape) = self
            .source
            .stored(name)
            .ok_or_else(|| refuse(format!("missing tensor {name}")))?;
        if shape != want {
            return Err(refuse(format!(
                "{name} is {shape:?}, the model needs {want:?}"
            )));
        }
        Ok(dtype)
    }

    fn add(&mut self, bytes: u64) {
        self.bytes += bytes;
    }

    /// A plain tensor of `shape`, half width as stored and F32 on the device, or kept as stored.
    /// Returns the stored dtype.
    pub(super) fn plain(
        &mut self,
        name: &str,
        shape: &[u32],
        keep: bool,
    ) -> Result<Dtype, EngError> {
        let dtype = self.stored(name, shape)?;
        if !matches!(dtype, Dtype::F16 | Dtype::Bf16 | Dtype::F32) {
            return Err(refuse(format!("{name} has dtype {dtype:?}")));
        }
        let values = shape.iter().map(|&d| u64::from(d)).product::<u64>();
        self.add(values * if keep { dtype.byte_width() } else { 4 });
        Ok(dtype)
    }

    /// The quantized matrix under `prefix`, `k` inputs by `n` outputs, at some rate; returns the
    /// rate (twice the bits per weight).
    pub(super) fn matrix(&mut self, prefix: &str, k: usize, n: usize) -> Result<u8, EngError> {
        let dim = |value: usize| {
            u32::try_from(value)
                .map_err(|_| refuse(format!("{prefix}: dimension {value} exceeds u32")))
        };
        if !k.is_multiple_of(128) || !n.is_multiple_of(128) {
            return Err(refuse(format!(
                "{prefix}: {k} by {n} is not a multiple of 128 on both axes"
            )));
        }
        let (tiles_k, tiles_n) = (dim(k / 16)?, dim(n / 16)?);
        let name = format!("{prefix}.trellis");
        let (dtype, shape) = self
            .source
            .stored(&name)
            .ok_or_else(|| refuse(format!("missing tensor {name}")))?;
        let words = match shape.as_slice() {
            [a, b, words] if (*a, *b) == (tiles_k, tiles_n) && dtype == Dtype::I16 => *words,
            _ => {
                return Err(refuse(format!(
                    "{name} is {dtype:?} {shape:?}, the model needs I16 [{tiles_k}, {tiles_n}, w]"
                )));
            }
        };
        if words % 8 != 0 || !(16..=128).contains(&words) {
            return Err(refuse(format!("{name} has {words} words per tile")));
        }
        for (suffix, len) in [("suh", k), ("svh", n)] {
            let name = format!("{prefix}.{suffix}");
            if self.stored(&name, &[dim(len)?])? != Dtype::F16 {
                return Err(refuse(format!("{name} is not F16")));
            }
        }
        for legacy in ["su", "sv", "scale"] {
            if self.source.contains(&format!("{prefix}.{legacy}")) {
                return Err(refuse(format!(
                    "{prefix} stores the old component {legacy}"
                )));
            }
        }
        self.mark(prefix)?;
        let trellis = u64::from(tiles_k) * u64::from(tiles_n) * u64::from(words) * 2;
        self.add(trellis + (k as u64 + n as u64) * 4);
        self.matrices += 1;
        let rate = (words / 8) as u8;
        *self.rates.entry(rate).or_default() += 1;
        Ok(rate)
    }

    /// The marker tensor of the matrix: the whole pack carries one codebook.
    fn mark(&mut self, prefix: &str) -> Result<(), EngError> {
        let found = [("mul1", Marker::Mul1), ("mcg", Marker::Mcg)]
            .into_iter()
            .filter(|(suffix, _)| self.source.contains(&format!("{prefix}.{suffix}")))
            .map(|(_, marker)| marker)
            .collect::<Vec<_>>();
        let [marker] = found.as_slice() else {
            return Err(refuse(format!(
                "{prefix} carries {} codebook markers",
                found.len()
            )));
        };
        if *self.marker.get_or_insert(*marker) != *marker {
            return Err(refuse(format!(
                "{prefix} has another codebook than the pack"
            )));
        }
        Ok(())
    }
}

/// Walks the pack headers against `shape`. See the module doc.
pub(super) fn inspect_pack(
    source: &StreamedCheckpoint,
    shape: &HostShape,
) -> Result<PackReport, EngError> {
    let mut walk = Inspector::new(source, Root::Layers);
    let dim = |value: usize| u32::try_from(value).unwrap_or(u32::MAX);
    let h = shape.hidden;
    // The step gathers rows from the table as BF16 (`exl3_global`); another width would load
    // every layer before failing there.
    let embed = walk.plain(EMBED_NAME, &[dim(shape.vocab), dim(h)], true)?;
    if embed != Dtype::Bf16 {
        return Err(refuse(format!(
            "{EMBED_NAME} is {embed:?}, the step needs Bf16"
        )));
    }
    walk.plain(NORM_NAME, &[dim(h)], false)?;
    walk.matrix(HEAD_PREFIX, h, shape.vocab)?;
    for (layer, kind) in shape.kinds.iter().enumerate() {
        walk.plain(
            &walk.root.name(layer, "input_layernorm.weight"),
            &[dim(h)],
            false,
        )?;
        walk.plain(
            &walk.root.name(layer, "post_attention_layernorm.weight"),
            &[dim(h)],
            false,
        )?;
        match kind {
            AttentionKind::Full => walk.attention(layer, shape)?,
            AttentionKind::DeltaNet => walk.delta(layer, shape)?,
            other => return Err(refuse(format!("layer {layer} is {other:?}"))),
        }
        walk.moe(layer, shape)?;
    }
    Ok(PackReport {
        device_bytes: walk.bytes,
        matrices: walk.matrices,
        rates: walk.rates,
        codebook: walk
            .marker
            .ok_or_else(|| refuse("the pack has no matrix".into()))?,
    })
}

impl Inspector<'_> {
    pub(super) fn attention(&mut self, layer: usize, shape: &HostShape) -> Result<(), EngError> {
        let (h, a) = (shape.hidden, &shape.attn);
        let (q, kv) = (a.heads_q * a.head_dim, a.heads_kv * a.head_dim);
        for (role, k, n) in [
            ("q_proj", h, 2 * q),
            ("k_proj", h, kv),
            ("v_proj", h, kv),
            ("o_proj", q, h),
        ] {
            self.matrix(&self.root.name(layer, &format!("self_attn.{role}")), k, n)?;
        }
        for role in ["q_norm", "k_norm"] {
            let name = self.root.name(layer, &format!("self_attn.{role}.weight"));
            self.plain(&name, &[a.head_dim as u32], false)?;
        }
        Ok(())
    }

    fn delta(&mut self, layer: usize, shape: &HostShape) -> Result<(), EngError> {
        let (h, d) = (shape.hidden, &shape.delta);
        let (z, heads) = (d.value_heads * d.value_dim, d.value_heads as u32);
        for (role, k, n) in [
            ("in_proj_qkv", h, d.conv_dim()),
            ("in_proj_z", h, z),
            ("out_proj", z, h),
        ] {
            self.matrix(&self.root.name(layer, &format!("linear_attn.{role}")), k, n)?;
        }
        for role in ["in_proj_a", "in_proj_b"] {
            let name = self.root.name(layer, &format!("linear_attn.{role}.weight"));
            self.plain(&name, &[heads, h as u32], false)?;
        }
        for (role, dims) in [
            ("A_log", vec![heads]),
            ("dt_bias", vec![heads]),
            (
                "conv1d.weight",
                vec![d.conv_dim() as u32, 1, d.conv_kernel as u32],
            ),
            ("norm.weight", vec![d.value_dim as u32]),
        ] {
            self.plain(
                &self.root.name(layer, &format!("linear_attn.{role}")),
                &dims,
                false,
            )?;
        }
        Ok(())
    }

    pub(super) fn moe(&mut self, layer: usize, shape: &HostShape) -> Result<(), EngError> {
        let (h, m) = (shape.hidden, &shape.moe);
        self.plain(
            &router_name_in(self.root, layer),
            &[m.experts as u32, h as u32],
            false,
        )?;
        self.plain(
            &shared_gate_name_in(self.root, layer),
            &[1, h as u32],
            false,
        )?;
        for part in ExpertPart::ALL {
            let (k, n) = match part {
                ExpertPart::Down => (m.inter, h),
                _ => (h, m.inter),
            };
            let mut bank_rate = None;
            for expert in 0..m.experts {
                let rate = self.matrix(&expert_name_in(self.root, layer, expert, part), k, n)?;
                if *bank_rate.get_or_insert(rate) != rate {
                    return Err(refuse(format!(
                        "layer {layer} {part:?} experts differ in rate: a bank has one stride"
                    )));
                }
            }
            let (k, n) = match part {
                ExpertPart::Down => (m.shared_inter, h),
                _ => (h, m.shared_inter),
            };
            self.matrix(&shared_name_in(self.root, layer, part), k, n)?;
        }
        Ok(())
    }
}
