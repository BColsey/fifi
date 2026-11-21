//! Where the routed experts of a mixture-of-experts GGUF live, and reads of one expert's bytes.
//!
//! A GGUF stores the experts of a layer as three stacked tensors (gate, up and down) whose last
//! dimension indexes the experts, so one expert is one contiguous slice of each tensor. This
//! module turns a parsed header into the validated placement of every slice: which shard file
//! holds it (a split set keeps different layers in different files), at which offset, how many
//! bytes, and of which quantization type. The type may differ from layer to layer, so the size
//! of an expert record is per layer.
//!
//! [`GgufExpertIndex`] does no I/O. [`GgufRecordStore`] reads records through the checkpoint's
//! positioned reads and is an [`ExpertRecordStore`], the seam the expert tiers read through.
//! Families supply the tensor names, the extents and the types they accept; nothing about a
//! model is fixed here.

use std::ops::Range;

use super::{
    expert_store::{ExpertRecordStore, RecordSite, site_outside_store},
    gguf::{GgufHeader, GgufTensor, GgufType},
    gguf_weights::GgufWeights,
};
use crate::EngError;

/// Names of the stacked expert tensors of a layer; the literal `{layer}` is replaced by the layer
/// number. The gate is absent for experts that have only an up and a down projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertTensorNames {
    pub gate: Option<String>,
    pub up: String,
    pub down: String,
}

impl ExpertTensorNames {
    /// The common GGUF convention: `blk.{layer}.ffn_gate_exps.weight`, `..._up_exps...` and
    /// `..._down_exps...`.
    pub fn standard() -> Self {
        Self {
            gate: Some(String::from("blk.{layer}.ffn_gate_exps.weight")),
            up: String::from("blk.{layer}.ffn_up_exps.weight"),
            down: String::from("blk.{layer}.ffn_down_exps.weight"),
        }
    }

    /// The three stacks of a layer named by a common stem: `{stem}_gate_exps.weight`, the up
    /// stack and the down stack of the same spelling.
    pub fn with_stem(stem: &str) -> Self {
        Self {
            gate: Some(format!("{stem}_gate_exps.weight")),
            up: format!("{stem}_up_exps.weight"),
            down: format!("{stem}_down_exps.weight"),
        }
    }

    /// The convention without a gate stack: `..._up_exps...` and `..._down_exps...` only.
    pub fn up_down() -> Self {
        Self {
            gate: None,
            ..Self::standard()
        }
    }

    pub fn for_layer(&self, layer: usize) -> Self {
        let fill = |name: &str| name.replace("{layer}", &layer.to_string());
        Self {
            gate: self.gate.as_deref().map(fill),
            up: fill(&self.up),
            down: fill(&self.down),
        }
    }
}

/// Typed failure of building or querying an index; every refusal names the tensor or the
/// arguments that lost, and nothing is half-indexed.
#[derive(Debug, PartialEq, Eq)]
pub enum LocatorError {
    MissingTensor {
        /// The looked-up name, `{layer}` already replaced.
        name: String,
    },
    /// The tensor exists but is unusable: a type the caller does not accept, the wrong rank, a
    /// byte length that disagrees with its dims, a range beyond its file, or an overlap with a
    /// sibling of the same layer; `why` says which.
    BadTensor { name: String, why: String },
    /// A tensor's dims differ from the expected `[ne0, ne1, ne2]`, or the arguments are
    /// impossible.
    ShapeMismatch(String),
    /// A query named a layer outside `0..layers` or an expert outside `0..experts`.
    OutOfRange { layer: usize, expert: usize },
    /// A checked offset or size computation overflowed; nothing was indexed.
    Overflow,
}

impl From<LocatorError> for EngError {
    fn from(error: LocatorError) -> Self {
        Self::Checkpoint(format!("GGUF experts: {error}"))
    }
}

impl std::fmt::Display for LocatorError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTensor { name } => write!(out, "tensor {name} is missing"),
            Self::BadTensor { name, why } => write!(out, "tensor {name}: {why}"),
            Self::ShapeMismatch(detail) => write!(out, "{detail}"),
            Self::OutOfRange { layer, expert } => {
                write!(out, "layer {layer} expert {expert} is outside the index")
            }
            Self::Overflow => write!(out, "an offset or size overflows"),
        }
    }
}

/// One tensor's stack of experts.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Stack {
    name: String,
    ty: GgufType,
    shard: usize,
    /// Offset of the tensor's first byte in its shard file.
    offset: u64,
    /// Bytes of one expert's slice.
    per_expert: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LayerStacks {
    gate: Option<Stack>,
    up: Stack,
    down: Stack,
}

/// One expert's slice of one stacked tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertSlice<'a> {
    /// The stacked tensor's name.
    pub name: &'a str,
    pub ty: GgufType,
    /// Zero-based shard file that holds the tensor (0 outside a split set).
    pub shard: usize,
    /// Byte range of the slice in its shard file.
    pub file_range: Range<u64>,
    /// Byte offset of the slice inside its tensor.
    pub tensor_offset: u64,
}

/// The slices of one expert: gate (when the experts have one), up and down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertSlices<'a> {
    pub gate: Option<ExpertSlice<'a>>,
    pub up: ExpertSlice<'a>,
    pub down: ExpertSlice<'a>,
}

impl<'a> ExpertSlices<'a> {
    /// The slices present, in record order: gate, up, down.
    pub fn parts(&self) -> impl Iterator<Item = &ExpertSlice<'a>> {
        self.gate.iter().chain([&self.up, &self.down])
    }

    /// The gate, up and down slices, refused with the tensor named when the experts have no gate.
    pub fn three(&self) -> Result<[&ExpertSlice<'a>; 3], LocatorError> {
        Ok([self.require_gate()?, &self.up, &self.down])
    }

    /// The gate slice, refused with the tensor named when the experts have none.
    pub fn require_gate(&self) -> Result<&ExpertSlice<'a>, LocatorError> {
        self.gate.as_ref().ok_or_else(|| {
            LocatorError::ShapeMismatch(format!(
                "the experts of {} have no gate stack, which this reader needs",
                self.up.name
            ))
        })
    }
}

/// The validated placement of every expert of every requested layer.
#[derive(Debug, Clone)]
pub struct GgufExpertIndex {
    layers: Vec<LayerStacks>,
    experts: usize,
}

fn slice_of(s: &Stack, expert: usize) -> Result<ExpertSlice<'_>, LocatorError> {
    let inside = widen(expert)?
        .checked_mul(s.per_expert)
        .ok_or(LocatorError::Overflow)?;
    let start = s.offset.checked_add(inside).ok_or(LocatorError::Overflow)?;
    let end = start
        .checked_add(s.per_expert)
        .ok_or(LocatorError::Overflow)?;
    Ok(ExpertSlice {
        name: &s.name,
        ty: s.ty,
        shard: s.shard,
        file_range: start..end,
        tensor_offset: inside,
    })
}

fn widen(n: usize) -> Result<u64, LocatorError> {
    u64::try_from(n).map_err(|_| LocatorError::Overflow)
}

impl GgufExpertIndex {
    /// The index of one layer's stacks in a single file: the set of a model that stores one
    /// layer per file or a header read for one layer's tensors.
    pub fn single_file(
        header: &GgufHeader,
        names: &ExpertTensorNames,
        experts: usize,
        shape: (usize, usize),
        accepted: &[GgufType],
    ) -> Result<Self, LocatorError> {
        Self::new(
            header,
            &[header.file_len()],
            names,
            1,
            experts,
            shape,
            Some(accepted),
        )
    }

    /// The gate, up and down slices of one expert, refused with the tensor named when the
    /// experts have no gate.
    pub fn three_slices(
        &self,
        layer: usize,
        expert: usize,
    ) -> Result<[ExpertSlice<'_>; 3], LocatorError> {
        let slices = self.slices(layer, expert)?;
        let [gate, up, down] = slices.three()?;
        Ok([gate.clone(), up.clone(), down.clone()])
    }

    /// Validates and indexes `layers` layers. Every named tensor must exist, have a type in
    /// `accepted` (any type when `None`) and three dimensions: gate (when named) and up `[hidden, inter, experts]`, down
    /// `[inter, hidden, experts]` (GGUF order, the first the contiguous row length), a byte
    /// length of `experts` slices, a range inside its shard's file (`shard_lens`, one length per
    /// shard), and the tensors of a layer must be disjoint. All arithmetic is checked.
    pub fn new(
        header: &GgufHeader,
        shard_lens: &[u64],
        names: &ExpertTensorNames,
        layers: usize,
        experts: usize,
        (hidden, inter): (usize, usize),
        accepted: Option<&[GgufType]>,
    ) -> Result<Self, LocatorError> {
        if layers == 0 || experts == 0 || hidden == 0 || inter == 0 {
            return Err(LocatorError::ShapeMismatch(format!(
                "locator: layers {layers}, experts {experts} and the {hidden}x{inter} expert \
                 shape must all be nonzero"
            )));
        }
        let (hidden, inter, experts64) = (widen(hidden)?, widen(inter)?, widen(experts)?);
        let side = [hidden, inter, experts64];
        let down = [inter, hidden, experts64];
        let mut out = Vec::with_capacity(layers);
        for layer in 0..layers {
            let at = names.for_layer(layer);
            let stack = |name: &str, want: [u64; 3]| {
                let tensor = header
                    .tensor(name)
                    .ok_or_else(|| LocatorError::MissingTensor {
                        name: name.to_owned(),
                    })?;
                Self::stack(tensor, name, want, experts64, shard_lens, accepted)
            };
            let entry = LayerStacks {
                gate: at
                    .gate
                    .as_deref()
                    .map(|name| stack(name, side))
                    .transpose()?,
                up: stack(&at.up, side)?,
                down: stack(&at.down, down)?,
            };
            Self::check_disjoint(&entry, experts64)?;
            out.push(entry);
        }
        Ok(Self {
            layers: out,
            experts,
        })
    }

    fn stack(
        tensor: &GgufTensor,
        name: &str,
        want: [u64; 3],
        experts: u64,
        shard_lens: &[u64],
        accepted: Option<&[GgufType]>,
    ) -> Result<Stack, LocatorError> {
        let ty = tensor.ggml_type();
        if let Some(accepted) = accepted.filter(|list| !list.contains(&ty)) {
            let list: Vec<&str> = accepted.iter().map(|t| t.name()).collect();
            return Err(LocatorError::BadTensor {
                name: name.to_owned(),
                why: format!("expected type {}, found {}", list.join(" or "), ty.name()),
            });
        }
        let dims = tensor.dims();
        if dims.len() != 3 {
            return Err(LocatorError::BadTensor {
                name: name.to_owned(),
                why: format!("expected 3 dimensions, found {}", dims.len()),
            });
        }
        let got = [dims[0], dims[1], dims[2]];
        if got != want {
            return Err(LocatorError::ShapeMismatch(format!(
                "tensor {name} has dims {got:?}, expected {want:?} \
                 (ne0 is the contiguous row length)"
            )));
        }
        let per_expert = want[0]
            .checked_mul(want[1])
            .and_then(|v| v.checked_div(ty.block_elems()))
            .and_then(|v| v.checked_mul(ty.block_bytes()))
            .ok_or(LocatorError::Overflow)?;
        let expected_len = per_expert
            .checked_mul(experts)
            .ok_or(LocatorError::Overflow)?;
        if tensor.byte_len() != expected_len {
            return Err(LocatorError::BadTensor {
                name: name.to_owned(),
                why: format!(
                    "byte length {} disagrees with per-expert {per_expert} x {experts} experts",
                    tensor.byte_len()
                ),
            });
        }
        let end = tensor
            .abs_offset()
            .checked_add(tensor.byte_len())
            .ok_or(LocatorError::Overflow)?;
        let file_len =
            shard_lens
                .get(tensor.shard())
                .copied()
                .ok_or_else(|| LocatorError::BadTensor {
                    name: name.to_owned(),
                    why: format!("is in shard {}, which has no length", tensor.shard()),
                })?;
        if end > file_len {
            return Err(LocatorError::BadTensor {
                name: name.to_owned(),
                why: format!(
                    "range {}..{end} ends beyond the file length {file_len}",
                    tensor.abs_offset()
                ),
            });
        }
        Ok(Stack {
            name: name.to_owned(),
            ty,
            shard: tensor.shard(),
            offset: tensor.abs_offset(),
            per_expert,
        })
    }

    /// The three tensors of a layer must not overlap each other when they share a file (the
    /// parser already refuses overlap globally; this keeps the promise local).
    fn check_disjoint(entry: &LayerStacks, experts: u64) -> Result<(), LocatorError> {
        let placed: Vec<&Stack> = entry.gate.iter().chain([&entry.up, &entry.down]).collect();
        for pair in placed.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if a.shard != b.shard {
                continue;
            }
            let end = |s: &Stack| {
                s.per_expert
                    .checked_mul(experts)
                    .and_then(|len| s.offset.checked_add(len))
                    .ok_or(LocatorError::Overflow)
            };
            let (end_a, end_b) = (end(a)?, end(b)?);
            if a.offset < end_b && b.offset < end_a {
                return Err(LocatorError::BadTensor {
                    name: b.name.clone(),
                    why: format!("overlaps tensor {} of the same layer", a.name),
                });
            }
        }
        Ok(())
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Bytes of one expert (the parts present: gate, up, down) in `layer`: the size of that
    /// layer's records.
    pub fn record_bytes(&self, layer: usize) -> Result<u64, LocatorError> {
        let entry = self.layer(layer, 0)?;
        entry
            .gate
            .iter()
            .chain([&entry.up, &entry.down])
            .try_fold(0u64, |sum, stack| {
                sum.checked_add(stack.per_expert)
                    .ok_or(LocatorError::Overflow)
            })
    }

    /// The largest record of any layer: what a slot or a staging buffer that serves every layer
    /// must hold.
    pub fn max_record_bytes(&self) -> Result<u64, LocatorError> {
        (0..self.layers.len()).try_fold(0u64, |most, layer| Ok(most.max(self.record_bytes(layer)?)))
    }

    /// Bytes of the routed experts of all layers.
    pub fn total_bytes(&self) -> Result<u64, LocatorError> {
        let per_expert = (0..self.layers.len())
            .map(|layer| self.record_bytes(layer))
            .try_fold(0u64, |sum, bytes| {
                sum.checked_add(bytes?).ok_or(LocatorError::Overflow)
            })?;
        per_expert
            .checked_mul(widen(self.experts)?)
            .ok_or(LocatorError::Overflow)
    }

    fn layer(&self, layer: usize, expert: usize) -> Result<&LayerStacks, LocatorError> {
        self.layers
            .get(layer)
            .ok_or(LocatorError::OutOfRange { layer, expert })
    }

    /// The slices of expert `expert` of layer `layer`.
    pub fn slices(&self, layer: usize, expert: usize) -> Result<ExpertSlices<'_>, LocatorError> {
        if expert >= self.experts {
            return Err(LocatorError::OutOfRange { layer, expert });
        }
        let entry = self.layer(layer, expert)?;
        let slice = |s| slice_of(s, expert);
        Ok(ExpertSlices {
            gate: entry.gate.as_ref().map(slice).transpose()?,
            up: slice(&entry.up)?,
            down: slice(&entry.down)?,
        })
    }

    /// The types of `layer`'s stacks present, in record order: gate, up, down.
    pub fn types(&self, layer: usize) -> Result<Vec<GgufType>, LocatorError> {
        let entry = self.layer(layer, 0)?;
        Ok(entry
            .gate
            .iter()
            .chain([&entry.up, &entry.down])
            .map(|stack| stack.ty)
            .collect())
    }

    /// Reads one expert's record into `out`: the slices present (gate, up, down) back to back. `out`
    /// must be exactly [`Self::record_bytes`] of the layer. The index holds each slice's shard
    /// and file range, so the reads address the files directly (no tensor lookup by name), and
    /// each shard's length is checked once for the record.
    #[cfg_attr(not(test), expect(dead_code, reason = "no caller outside the tests"))]
    pub(crate) fn read_record(
        &self,
        weights: &GgufWeights,
        layer: usize,
        expert: usize,
        out: &mut [u8],
    ) -> Result<(), EngError> {
        let slices = self
            .slices(layer, expert)
            .map_err(|error| EngError::BadRequest(format!("expert record: {error}")))?;
        let want = self
            .record_bytes(layer)
            .map_err(|error| EngError::BadRequest(format!("expert record: {error}")))?;
        if out.len() as u64 != want {
            return Err(EngError::BadRequest(format!(
                "expert record: a buffer of {} bytes for a record of {want}",
                out.len()
            )));
        }
        let mut rest = out;
        let mut reads = Vec::with_capacity(3);
        for slice in slices.parts() {
            let (head, tail) =
                rest.split_at_mut((slice.file_range.end - slice.file_range.start) as usize);
            reads.push((slice.shard, slice.file_range.start, head));
            rest = tail;
        }
        weights.read_file_ranges(&mut reads)
    }
}

/// An [`ExpertRecordStore`] over a GGUF checkpoint: one positioned read per slice, no pooling.
/// Records are the layer's own size; [`ExpertRecordStore::record_bytes`] is the largest, which is
/// what a slot that serves every layer holds.
#[derive(Debug)]
#[cfg_attr(not(test), expect(dead_code, reason = "no caller outside the tests"))]
pub(crate) struct GgufRecordStore {
    weights: std::sync::Arc<GgufWeights>,
    index: GgufExpertIndex,
    /// The largest record, computed once.
    max_record: u64,
}

#[cfg_attr(not(test), expect(dead_code, reason = "no caller outside the tests"))]
impl GgufRecordStore {
    pub(crate) fn new(
        weights: std::sync::Arc<GgufWeights>,
        index: GgufExpertIndex,
    ) -> Result<Self, EngError> {
        let max_record = index
            .max_record_bytes()
            .map_err(|error| EngError::Checkpoint(format!("expert records: {error}")))?;
        Ok(Self {
            weights,
            index,
            max_record,
        })
    }
}

impl ExpertRecordStore for GgufRecordStore {
    type Lease = Vec<u8>;

    fn read_records(&self, sites: &[RecordSite]) -> Result<Vec<Vec<u8>>, EngError> {
        sites
            .iter()
            .map(|&site| {
                let bytes = self
                    .record_bytes_at(site.layer)
                    .map_err(|_| site_outside_store(site))?;
                if usize::from(site.expert) >= self.index.experts() {
                    return Err(site_outside_store(site));
                }
                let mut record = vec![0u8; bytes as usize];
                self.index.read_record(
                    &self.weights,
                    site.layer,
                    usize::from(site.expert),
                    &mut record,
                )?;
                Ok(record)
            })
            .collect()
    }

    fn record_bytes(&self) -> u64 {
        self.max_record
    }

    fn record_bytes_at(&self, layer: usize) -> Result<u64, EngError> {
        self.index
            .record_bytes(layer)
            .map_err(|_| EngError::BadRequest(format!("layer {layer} is outside the store")))
    }

    fn max_batch(&self) -> usize {
        usize::MAX
    }
}

/// A small GGUF writer for the tests of the index and of the readers built on it.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::weights::GgufLimits;
    use std::path::Path;

    pub(crate) const LIMITS: GgufLimits = GgufLimits {
        max_header_bytes: 1 << 20,
        max_tensor_bytes: 1 << 24,
    };
    const ALIGN: u64 = 32;

    pub(crate) fn string(text: &str) -> Vec<u8> {
        let mut out = (text.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(text.as_bytes());
        out
    }

    pub(crate) fn kv_u16(key: &str, value: u16) -> Vec<u8> {
        let mut out = string(key);
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
        out
    }

    pub(crate) struct T {
        pub(crate) name: String,
        pub(crate) dims: Vec<u64>,
        pub(crate) ty: GgufType,
        pub(crate) payload: Vec<u8>,
    }

    /// Writes one GGUF file of `tensors` with `kvs` and returns its length.
    pub(crate) fn write(path: &Path, kvs: &[Vec<u8>], tensors: &[T]) -> u64 {
        let mut head = b"GGUF".to_vec();
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        head.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for kv in kvs {
            head.extend_from_slice(kv);
        }
        let mut at = 0u64;
        let mut offsets = Vec::new();
        for t in tensors {
            head.extend_from_slice(&string(&t.name));
            head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                head.extend_from_slice(&d.to_le_bytes());
            }
            head.extend_from_slice(&t.ty.id().to_le_bytes());
            head.extend_from_slice(&at.to_le_bytes());
            offsets.push(at);
            at = (at + t.payload.len() as u64).div_ceil(ALIGN) * ALIGN;
        }
        let start = (head.len() as u64).div_ceil(ALIGN) * ALIGN;
        head.resize(start as usize, 0);
        let mut file = head;
        for (t, offset) in tensors.iter().zip(&offsets) {
            file.resize((start + offset) as usize, 0);
            file.extend_from_slice(&t.payload);
        }
        std::fs::write(path, &file).expect("write");
        file.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{LIMITS, T, kv_u16, write};
    use super::*;
    use crate::weights::GgufWeights;

    const HIDDEN: usize = 64;
    const INTER: usize = 32;
    const EXPERTS: usize = 3;

    /// Layer `l` is `types[l]` (gate, up, down). Expert `e` of a stack holds bytes
    /// `(layer * 16 + part * 4 + e) as u8` repeated, so a read of the wrong slice shows.
    fn layer_tensors(layer: usize, ty: GgufType) -> Vec<T> {
        let per = HIDDEN * INTER / ty.block_elems() as usize * ty.block_bytes() as usize;
        let mut out = Vec::new();
        for (part, name) in ["gate", "up", "down"].into_iter().enumerate() {
            let dims = if part == 2 {
                vec![INTER as u64, HIDDEN as u64, EXPERTS as u64]
            } else {
                vec![HIDDEN as u64, INTER as u64, EXPERTS as u64]
            };
            let mut payload = Vec::new();
            for e in 0..EXPERTS {
                payload.extend(std::iter::repeat_n((layer * 16 + part * 4 + e) as u8, per));
            }
            out.push(T {
                name: format!("blk.{layer}.ffn_{name}_exps.weight"),
                dims,
                ty,
                payload,
            });
        }
        out
    }

    fn open_one(types: &[GgufType]) -> (tempfile::TempDir, GgufWeights) {
        let dir = tempfile::tempdir().expect("tempdir");
        let tensors: Vec<T> = types
            .iter()
            .enumerate()
            .flat_map(|(l, &ty)| layer_tensors(l, ty))
            .collect();
        let path = dir.path().join("m.gguf");
        write(&path, &[], &tensors);
        let weights = GgufWeights::open_any_type(&path, LIMITS, None).expect("open");
        (dir, weights)
    }

    fn index(
        weights: &GgufWeights,
        accepted: &[GgufType],
    ) -> Result<GgufExpertIndex, LocatorError> {
        let lens: Vec<u64> = weights
            .paths()
            .iter()
            .map(|p| std::fs::metadata(p).expect("stat").len())
            .collect();
        GgufExpertIndex::new(
            weights.header(),
            &lens,
            &ExpertTensorNames::standard(),
            weights.header().tensors().len() / 3,
            EXPERTS,
            (HIDDEN, INTER),
            Some(accepted),
        )
    }

    /// A set whose experts have an up and a down stack only: the record is those two slices
    /// back to back, the gate slice is absent and the types list two entries.
    #[test]
    fn a_set_without_a_gate_stack_indexes_and_reads_up_and_down() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tensors: Vec<T> = [GgufType::Q80, GgufType::Q40]
            .iter()
            .enumerate()
            .flat_map(|(layer, &ty)| layer_tensors(layer, ty))
            .filter(|tensor| !tensor.name.contains("_gate_"))
            .collect();
        assert_eq!(tensors.len(), 4);
        let path = dir.path().join("m.gguf");
        write(&path, &[], &tensors);
        let weights = GgufWeights::open_any_type(&path, LIMITS, None).expect("open");
        let len = std::fs::metadata(&path).expect("stat").len();
        let index = GgufExpertIndex::new(
            weights.header(),
            &[len],
            &ExpertTensorNames::up_down(),
            2,
            EXPERTS,
            (HIDDEN, INTER),
            Some(&[GgufType::Q80, GgufType::Q40]),
        )
        .expect("index");
        let q80 = 2 * (HIDDEN * INTER / 32 * 34) as u64;
        let q40 = 2 * (HIDDEN * INTER / 32 * 18) as u64;
        assert_eq!(index.record_bytes(0).unwrap(), q80);
        assert_eq!(index.record_bytes(1).unwrap(), q40);
        assert_eq!(index.max_record_bytes().unwrap(), q80);
        assert_eq!(index.total_bytes().unwrap(), (q80 + q40) * EXPERTS as u64);
        assert_eq!(index.types(1).unwrap(), [GgufType::Q40, GgufType::Q40]);
        let slices = index.slices(0, 1).unwrap();
        assert!(slices.gate.is_none());
        assert_eq!(slices.parts().count(), 2);
        assert!(slices.require_gate().is_err());
        for (layer, bytes) in [(0usize, q80), (1, q40)] {
            for expert in 0..EXPERTS {
                let mut record = vec![0u8; bytes as usize];
                index
                    .read_record(&weights, layer, expert, &mut record)
                    .expect("read");
                let half = record.len() / 2;
                // Up is part 1 and down part 2 of the writer's byte pattern.
                assert!(
                    record[..half]
                        .iter()
                        .all(|&b| b as usize == layer * 16 + 4 + expert)
                );
                assert!(
                    record[half..]
                        .iter()
                        .all(|&b| b as usize == layer * 16 + 8 + expert)
                );
            }
        }
        // The names with a gate refuse the set by the missing gate tensor.
        let with_gate = GgufExpertIndex::new(
            weights.header(),
            &[len],
            &ExpertTensorNames::standard(),
            2,
            EXPERTS,
            (HIDDEN, INTER),
            None,
        );
        assert_eq!(
            with_gate.expect_err("no gate tensor"),
            LocatorError::MissingTensor {
                name: "blk.0.ffn_gate_exps.weight".into()
            }
        );
    }

    #[test]
    fn records_have_each_layers_own_size_and_read_back_their_own_slices() {
        let (_dir, weights) = open_one(&[GgufType::Q80, GgufType::Q40]);
        let index = index(&weights, &[GgufType::Q80, GgufType::Q40]).expect("index");
        let q80 = 3 * (HIDDEN * INTER / 32 * 34) as u64;
        let q40 = 3 * (HIDDEN * INTER / 32 * 18) as u64;
        assert_eq!(index.record_bytes(0).unwrap(), q80);
        assert_eq!(index.record_bytes(1).unwrap(), q40);
        assert_eq!(index.max_record_bytes().unwrap(), q80);
        assert_eq!(index.total_bytes().unwrap(), (q80 + q40) * EXPERTS as u64);
        assert_eq!(index.types(1).unwrap(), [GgufType::Q40; 3]);
        for (layer, bytes) in [(0usize, q80), (1, q40)] {
            for expert in 0..EXPERTS {
                let mut record = vec![0u8; bytes as usize];
                index
                    .read_record(&weights, layer, expert, &mut record)
                    .expect("read");
                let per = bytes as usize / 3;
                for part in 0..3 {
                    let want = (layer * 16 + part * 4 + expert) as u8;
                    assert!(
                        record[part * per..(part + 1) * per]
                            .iter()
                            .all(|&b| b == want),
                        "layer {layer} expert {expert} part {part}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_store_serves_batches_in_order_and_reports_the_largest_record() {
        let (_dir, weights) = open_one(&[GgufType::Q80, GgufType::Q40]);
        let index = index(&weights, &[GgufType::Q80, GgufType::Q40]).expect("index");
        let store = GgufRecordStore::new(std::sync::Arc::new(weights), index).expect("store");
        assert_eq!(store.record_bytes(), store.record_bytes_at(0).unwrap());
        assert!(store.record_bytes_at(1).unwrap() < store.record_bytes_at(0).unwrap());
        assert!(store.record_bytes_at(2).is_err());
        let sites = [
            RecordSite {
                layer: 1,
                expert: 2,
            },
            RecordSite {
                layer: 0,
                expert: 0,
            },
            RecordSite {
                layer: 1,
                expert: 2,
            },
        ];
        let leases = store.read_records(&sites).expect("batch");
        assert_eq!(leases.len(), 3);
        assert_eq!(leases[0].len() as u64, store.record_bytes_at(1).unwrap());
        assert_eq!(leases[1].len() as u64, store.record_bytes_at(0).unwrap());
        assert_eq!(leases[0][0], (16 + 2) as u8);
        assert_eq!(leases[1][0], 0);
        assert_eq!(leases[0], leases[2]);
        let outside = store
            .read_records(&[RecordSite {
                layer: 2,
                expert: 0,
            }])
            .expect_err("no such layer");
        assert!(matches!(outside, EngError::BadRequest(_)));
        let outside = store
            .read_records(&[RecordSite {
                layer: 0,
                expert: 3,
            }])
            .expect_err("no such expert");
        assert!(matches!(outside, EngError::BadRequest(_)));
    }

    #[test]
    fn a_type_the_caller_does_not_accept_is_named() {
        let (_dir, weights) = open_one(&[GgufType::Q80]);
        let error = index(&weights, &[GgufType::Q4K]).expect_err("refused");
        assert_eq!(
            error,
            LocatorError::BadTensor {
                name: "blk.0.ffn_gate_exps.weight".into(),
                why: "expected type Q4_K, found Q8_0".into()
            }
        );
    }

    #[test]
    fn a_missing_tensor_and_wrong_dims_are_named() {
        let (_dir, weights) = open_one(&[GgufType::Q80]);
        let lens = [std::fs::metadata(&weights.paths()[0]).expect("stat").len()];
        let names = ExpertTensorNames::standard();
        let missing = GgufExpertIndex::new(
            weights.header(),
            &lens,
            &names,
            2,
            EXPERTS,
            (HIDDEN, INTER),
            Some(&[GgufType::Q80]),
        )
        .expect_err("layer 1 is absent");
        assert_eq!(
            missing,
            LocatorError::MissingTensor {
                name: "blk.1.ffn_gate_exps.weight".into()
            }
        );
        let wrong = GgufExpertIndex::new(
            weights.header(),
            &lens,
            &names,
            1,
            EXPERTS,
            (INTER, HIDDEN),
            Some(&[GgufType::Q80]),
        )
        .expect_err("transposed extents");
        assert!(matches!(wrong, LocatorError::ShapeMismatch(_)), "{wrong:?}");
    }

    #[test]
    fn a_split_set_indexes_each_stack_in_its_own_shard() {
        let dir = tempfile::tempdir().expect("tempdir");
        let name = |n: u16| dir.path().join(format!("e-{n:05}-of-00003.gguf"));
        let split = |no: u16| {
            vec![
                kv_u16("split.no", no),
                kv_u16("split.count", 3),
                kv_u16("split.tensors.count", 6),
            ]
        };
        write(&name(1), &split(0), &[]);
        write(&name(2), &split(1), &layer_tensors(0, GgufType::Q80));
        write(&name(3), &split(2), &layer_tensors(1, GgufType::Q40));
        let weights = GgufWeights::open_any_type(&name(1), LIMITS, None).expect("open the set");
        let index = index(&weights, &[GgufType::Q80, GgufType::Q40]).expect("index");
        assert_eq!(index.slices(0, 1).unwrap().require_gate().unwrap().shard, 1);
        assert_eq!(index.slices(1, 1).unwrap().down.shard, 2);
        // The file range starts where the tensor's own offset in its shard plus the expert's
        // slice says, computed here from the header alone.
        for (layer, part, shard) in [(0usize, "gate", 1usize), (1, "down", 2)] {
            let tensor = weights
                .header()
                .tensor(&format!("blk.{layer}.ffn_{part}_exps.weight"))
                .expect("tensor");
            assert_eq!(tensor.shard(), shard);
            let per = tensor.byte_len() / EXPERTS as u64;
            for expert in 0..EXPERTS {
                let slices = index.slices(layer, expert).unwrap();
                let slice = if part == "gate" {
                    slices.require_gate().unwrap()
                } else {
                    &slices.down
                };
                assert_eq!(
                    slice.file_range.start,
                    tensor.abs_offset() + expert as u64 * per,
                    "layer {layer} expert {expert}"
                );
                assert_eq!(slice.file_range.end - slice.file_range.start, per);
            }
        }
        let mut record = vec![0u8; index.record_bytes(1).unwrap() as usize];
        index
            .read_record(&weights, 1, 2, &mut record)
            .expect("read");
        assert_eq!(record[0], (16 + 2) as u8);
        // A shard shorter than the tensor's range is refused by its own length, not shard 1's.
        let short = GgufExpertIndex::new(
            weights.header(),
            &[1 << 20, 10, 1 << 20],
            &ExpertTensorNames::standard(),
            2,
            EXPERTS,
            (HIDDEN, INTER),
            Some(&[GgufType::Q80, GgufType::Q40]),
        )
        .expect_err("short shard");
        assert!(
            matches!(&short, LocatorError::BadTensor { why, .. } if why.contains("beyond the file length 10")),
            "{short:?}"
        );
    }

    #[test]
    fn a_wrongly_sized_buffer_and_an_outside_query_are_refused() {
        let (_dir, weights) = open_one(&[GgufType::Q80]);
        let index = index(&weights, &[GgufType::Q80]).expect("index");
        assert!(matches!(
            index.read_record(&weights, 0, 0, &mut [0u8; 5]),
            Err(EngError::BadRequest(_))
        ));
        assert_eq!(
            index.slices(0, EXPERTS).unwrap_err(),
            LocatorError::OutOfRange {
                layer: 0,
                expert: EXPERTS
            }
        );
        assert_eq!(
            index.slices(5, 0).unwrap_err(),
            LocatorError::OutOfRange {
                layer: 5,
                expert: 0
            }
        );
    }
}
