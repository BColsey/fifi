//! Checked openers for the Gemma 4 Q8_0 GGUF checkpoints.
//!
//! open_humanizer_q8 and open_bolt_q8 each open one single-file GGUF through
//! the shared reader, check the gemma4 architecture, the scalar metadata, the
//! per-layer attention arrays, and the exact tensor inventory. They never map
//! the file and claim no execution.

use std::path::Path;

use crate::{
    EngError,
    weights::{GgufLimits, GgufType, GgufWeights, MetaValue},
};

/// general.architecture of the llama.cpp conversion of Gemma 4.
const ARCHITECTURE: &str = "gemma4";
/// Reader limits: a 100-MB header cap and a 1-GiB tensor cap.
const LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 100_000_000,
    // Humanizer's largest tensor stays under 1 GiB.
    max_tensor_bytes: 1 << 30,
};
const BOLT_LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 100_000_000,
    // The canonical spelling's merged per_layer_token_embd table holds
    // PLE * layers rows of the vocabulary: 2_994_733_056 bytes, so the cap
    // sits just above it (the legacy per-tensor ceiling was ~712 MiB).
    max_tensor_bytes: 3 << 30,
};
/// The positions the Humanizer keeps per head; a window is at most that many.
pub(super) const HUMANIZER_CELLS: u64 = 8192;
const LAYERS: usize = 48;
const HIDDEN: u64 = 3840;
const FEED_FORWARD: u64 = 15360;
const QUERY_HEADS: u64 = 16;
const VOCAB: u64 = 262_144;
const LOCAL_KV_HEADS: u64 = 8;
const LOCAL_HEAD: u64 = 256;
const GLOBAL_KV_HEADS: u64 = 1;
const GLOBAL_HEAD: u64 = 512;
const GGUF_I32: u32 = 5;
const GGUF_BOOL: u32 = 7;
const HUMANIZER_INT_SCALARS: [(&str, u64); 11] = [
    ("gemma4.block_count", LAYERS as u64),
    ("gemma4.embedding_length", HIDDEN),
    ("gemma4.feed_forward_length", FEED_FORWARD),
    ("gemma4.attention.head_count", QUERY_HEADS),
    ("gemma4.attention.key_length", GLOBAL_HEAD),
    ("gemma4.attention.key_length_swa", LOCAL_HEAD),
    ("gemma4.attention.value_length", GLOBAL_HEAD),
    ("gemma4.attention.value_length_swa", LOCAL_HEAD),
    ("gemma4.attention.shared_kv_layers", 0),
    ("gemma4.rope.dimension_count", 512),
    ("gemma4.rope.dimension_count_swa", 256),
];

const FLOAT_SCALARS: [(&str, f64); 4] = [
    ("gemma4.rope.freq_base", 1_000_000.0),
    ("gemma4.rope.freq_base_swa", 10_000.0),
    ("gemma4.final_logit_softcapping", 30.0),
    ("gemma4.attention.layer_norm_rms_epsilon", 1.0e-6),
];

/// Opens a Humanizer-geometry Q8_0 GGUF and checks its metadata and tensors.
///
/// Opener rejections are EngError::Checkpoint with the prefix gguf model: and
/// name the offending metadata key or tensor. Failures of GgufWeights::open
/// pass through unchanged, so a parse or header-hash failure keeps its own
/// gguf: prefix and a tensor over the limit stays EngError::MemoryPlan.
pub fn open_humanizer_q8(path: &Path) -> Result<(), EngError> {
    open_humanizer_weights(path).map(|_| ())
}

#[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
/// A checked Humanizer file: the reader and the sliding window its header declares.
pub(crate) struct HumanizerFile {
    pub(crate) weights: GgufWeights,
    pub(crate) window: usize,
}

/// Opens a Gemma 4 Humanizer-geometry checkpoint and returns the checked reader and the window
/// its header names, for the resident loader. The file is accepted by what its header says, not by
/// which file it is: the layer counts, widths, head shapes and the five-local, one-full pattern
/// must be the ones the resident code is built for, and the window is read from the header.
/// Applies exactly the checks of open_humanizer_q8.
pub(crate) fn open_humanizer_weights(path: &Path) -> Result<HumanizerFile, EngError> {
    let weights = GgufWeights::open(path, LIMITS, None)?;
    weights.require_single_file("the Gemma 4 engine")?;
    let header = weights.header();
    check_architecture(header)?;
    check_int_scalars(header, &HUMANIZER_INT_SCALARS)?;
    check_float_scalars(header)?;
    check_bool_scalar(header, "tokenizer.ggml.add_bos_token", true)?;
    let window = read_window(header, HUMANIZER_CELLS)?;
    let kv = read_i32_array(
        &weights,
        header,
        "gemma4.attention.head_count_kv",
        LAYERS,
        GGUF_I32,
    )?;
    let pattern = read_bool_array(
        &weights,
        header,
        "gemma4.attention.sliding_window_pattern",
        LAYERS,
        GGUF_BOOL,
    )?;
    check_layer_pattern(&kv, &pattern)?;
    check_inventory(header, LAYERS, expected_inventory(&kv))?;
    Ok(HumanizerFile { weights, window })
}

/// The sliding window of the header: present, positive and at most `most`.
fn read_window(header: &crate::weights::GgufHeader, most: u64) -> Result<usize, EngError> {
    let key = "gemma4.attention.sliding_window";
    let window = uint_meta(header, key)?;
    if window == 0 || window > most {
        return Err(reject(format!(
            "metadata key {key} is {window}, expected 1 to {most}"
        )));
    }
    Ok(window as usize)
}

/// How the file stores one expected tensor.
#[derive(Clone, Copy)]
pub(super) enum Stored {
    /// Exactly this type (the norms and scales, which every file keeps in F32).
    Exact(GgufType),
    /// A matrix: any type a native kernel reads, which is how the Q8_0 original and its K-quant
    /// and IQ requants share one inventory.
    Matrix,
}

impl Stored {
    /// The type a test fixture writes for this tensor: the exact one, or Q8_0 for a matrix.
    #[cfg(test)]
    pub(super) fn fixture(self) -> GgufType {
        match self {
            Self::Exact(ty) => ty,
            Self::Matrix => GgufType::Q80,
        }
    }
}

/// One expected tensor: how it is stored and its dimensions in file order.
pub(super) struct Expected {
    pub(super) name: String,
    pub(super) ty: Stored,
    pub(super) dims: Vec<u64>,
}

/// Builds the exact 667-tensor inventory driven by the checked KV-head array.
pub(super) fn expected_inventory(kv: &[i32]) -> Vec<Expected> {
    let mut expected = Vec::with_capacity(667);
    expected.push(Expected {
        name: "token_embd.weight".to_owned(),
        ty: Stored::Matrix,
        dims: vec![HIDDEN, VOCAB],
    });
    expected.push(Expected {
        name: "output_norm.weight".to_owned(),
        ty: Stored::Exact(GgufType::F32),
        dims: vec![HIDDEN],
    });
    expected.push(Expected {
        name: "rope_freqs.weight".to_owned(),
        ty: Stored::Exact(GgufType::F32),
        dims: vec![256],
    });
    for (layer, &heads) in kv.iter().enumerate() {
        let local = heads == LOCAL_KV_HEADS as i32;
        let head = if local { LOCAL_HEAD } else { GLOBAL_HEAD };
        let kv_heads = if local {
            LOCAL_KV_HEADS
        } else {
            GLOBAL_KV_HEADS
        };
        let base = format!("blk.{layer}.");
        let norm = |suffix: &str, width: u64| Expected {
            name: format!("{base}{suffix}"),
            ty: Stored::Exact(GgufType::F32),
            dims: vec![width],
        };
        let matrix = |suffix: &str, rows: u64, cols: u64| Expected {
            name: format!("{base}{suffix}"),
            ty: Stored::Matrix,
            dims: vec![rows, cols],
        };
        expected.push(norm("attn_norm.weight", HIDDEN));
        expected.push(matrix("attn_q.weight", HIDDEN, QUERY_HEADS * head));
        expected.push(norm("attn_q_norm.weight", head));
        expected.push(matrix("attn_k.weight", HIDDEN, kv_heads * head));
        expected.push(norm("attn_k_norm.weight", head));
        if local {
            expected.push(matrix("attn_v.weight", HIDDEN, kv_heads * head));
        }
        expected.push(matrix("attn_output.weight", QUERY_HEADS * head, HIDDEN));
        expected.push(norm("post_attention_norm.weight", HIDDEN));
        expected.push(norm("ffn_norm.weight", HIDDEN));
        expected.push(matrix("ffn_gate.weight", HIDDEN, FEED_FORWARD));
        expected.push(matrix("ffn_up.weight", HIDDEN, FEED_FORWARD));
        expected.push(matrix("ffn_down.weight", FEED_FORWARD, HIDDEN));
        expected.push(norm("post_ffw_norm.weight", HIDDEN));
        expected.push(norm("layer_output_scale.weight", 1));
    }
    debug_assert_eq!(expected.len(), 667);
    expected
}

/// Whether an unlisted tensor changes what the model computes if the recipe ignores it: an untied
/// output head, any bias, or any further tensor of a layer the recipe runs.
fn changes_the_result(name: &str, layers: usize) -> bool {
    let in_a_layer = name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split('.').next())
        .and_then(|index| index.parse::<usize>().ok())
        .is_some_and(|index| index < layers);
    name == "output.weight" || name.ends_with(".bias") || in_a_layer
}

fn check_inventory(
    header: &crate::weights::GgufHeader,
    layers: usize,
    mut expected: Vec<Expected>,
) -> Result<(), EngError> {
    // The missing-tensor refusal below names the first absent entry in this
    // order; sorting keeps that name stable regardless of inventory build order.
    expected.sort_by(|left, right| left.name.cmp(&right.name));
    let table: std::collections::HashMap<&str, (Stored, &[u64])> = expected
        .iter()
        .map(|item| (item.name.as_str(), (item.ty, item.dims.as_slice())))
        .collect();
    for tensor in header.tensors() {
        // A tensor the recipe does not read and that cannot change the result (a file's own extras)
        // does not stop it from running; one that would change it is refused by name.
        let Some((ty, dims)) = table.get(tensor.name()) else {
            if changes_the_result(tensor.name(), layers) {
                return Err(reject(format!("unexpected tensor {}", tensor.name())));
            }
            continue;
        };
        let stored = tensor.ggml_type();
        match ty {
            Stored::Matrix if !stored.has_native_kernel() => {
                return Err(reject(format!(
                    "tensor {} has type {}, expected a matrix type with a native kernel",
                    tensor.name(),
                    stored.name()
                )));
            }
            Stored::Exact(want) if stored != *want => {
                return Err(reject(format!(
                    "tensor {} has type {}, expected {}",
                    tensor.name(),
                    stored.name(),
                    want.name()
                )));
            }
            _ => {}
        }
        if tensor.dims() != *dims {
            return Err(reject(format!(
                "tensor {} has dims {:?}, expected {:?}",
                tensor.name(),
                tensor.dims(),
                dims
            )));
        }
    }
    for item in &expected {
        if header.tensor(&item.name).is_none() {
            return Err(reject(format!("missing tensor {}", item.name)));
        }
    }
    Ok(())
}

/// Checks the five-local, one-full pattern against both per-layer arrays.
fn check_layer_pattern(kv: &[i32], pattern: &[u8]) -> Result<(), EngError> {
    for (layer, (kv_heads, sliding)) in kv.iter().zip(pattern).enumerate() {
        let local = layer % 6 != 5;
        let want_kv = if local {
            LOCAL_KV_HEADS as i32
        } else {
            GLOBAL_KV_HEADS as i32
        };
        let want_pattern = u8::from(local);
        if *kv_heads != want_kv {
            return Err(reject(format!(
                "metadata key gemma4.attention.head_count_kv is {kv_heads} at layer {layer}, expected {want_kv}"
            )));
        }
        if *sliding != want_pattern {
            return Err(reject(format!(
                "metadata key gemma4.attention.sliding_window_pattern is {sliding} at layer {layer}, expected {want_pattern}"
            )));
        }
    }
    Ok(())
}

fn check_architecture(header: &crate::weights::GgufHeader) -> Result<(), EngError> {
    match header.meta("general.architecture") {
        Some(MetaValue::String(found)) if found == ARCHITECTURE => Ok(()),
        Some(MetaValue::String(found)) => Err(reject(format!(
            "metadata key general.architecture is {found:?}, expected {ARCHITECTURE:?}"
        ))),
        Some(_) => Err(reject("metadata key general.architecture is not a string")),
        None => Err(reject("metadata key general.architecture is missing")),
    }
}

fn check_int_scalars(
    header: &crate::weights::GgufHeader,
    scalars: &[(&str, u64)],
) -> Result<(), EngError> {
    for (key, want) in scalars {
        let found = uint_meta(header, key)?;
        if found != *want {
            return Err(reject(format!(
                "metadata key {key} is {found}, expected {want}"
            )));
        }
    }
    Ok(())
}

fn uint_meta(header: &crate::weights::GgufHeader, key: &str) -> Result<u64, EngError> {
    header.meta_u64(key).ok_or_else(|| {
        reject(format!(
            "metadata key {key} is missing or not an unsigned integer"
        ))
    })
}

/// Checks each float scalar against the expected value in its stored precision.
fn check_float_scalars(header: &crate::weights::GgufHeader) -> Result<(), EngError> {
    for (key, want) in FLOAT_SCALARS {
        let (found, want) = match header.meta(key) {
            Some(MetaValue::F32(value)) => (f64::from(*value), f64::from(want as f32)),
            Some(MetaValue::F64(value)) => (*value, want),
            Some(_) => return Err(reject(format!("metadata key {key} is not a float"))),
            None => return Err(reject(format!("metadata key {key} is missing"))),
        };
        if found != want {
            return Err(reject(format!(
                "metadata key {key} is {found}, expected {want}"
            )));
        }
    }
    Ok(())
}

fn check_bool_scalar(
    header: &crate::weights::GgufHeader,
    key: &str,
    want: bool,
) -> Result<(), EngError> {
    match header.meta(key) {
        Some(MetaValue::Bool(found)) if *found == want => Ok(()),
        Some(MetaValue::Bool(found)) => Err(reject(format!(
            "metadata key {key} is {found}, expected {want}"
        ))),
        Some(_) => Err(reject(format!("metadata key {key} is not a bool"))),
        None => Err(reject(format!("metadata key {key} is missing"))),
    }
}

fn read_i32_array(
    weights: &GgufWeights,
    header: &crate::weights::GgufHeader,
    key: &str,
    len: usize,
    elem_type: u32,
) -> Result<Vec<i32>, EngError> {
    let bytes = read_array_bytes(weights, header, key, len, elem_type, len * 4)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| i32::from_le_bytes(*chunk))
        .collect())
}

fn read_bool_array(
    weights: &GgufWeights,
    header: &crate::weights::GgufHeader,
    key: &str,
    len: usize,
    elem_type: u32,
) -> Result<Vec<u8>, EngError> {
    read_array_bytes(weights, header, key, len, elem_type, len)
}

fn read_array_bytes(
    weights: &GgufWeights,
    header: &crate::weights::GgufHeader,
    key: &str,
    len: usize,
    elem_type: u32,
    byte_len: usize,
) -> Result<Vec<u8>, EngError> {
    let meta = match header.meta(key) {
        Some(MetaValue::Array(meta)) => meta,
        Some(_) => return Err(reject(format!("metadata key {key} is not an array"))),
        None => return Err(reject(format!("metadata key {key} is missing"))),
    };
    if meta.elem_type != elem_type {
        return Err(reject(format!(
            "metadata key {key} has element type {}, expected {elem_type}",
            meta.elem_type
        )));
    }
    if meta.len != len as u64 {
        return Err(reject(format!(
            "metadata key {key} has {} elements, expected {len}",
            meta.len
        )));
    }
    let mut buf = vec![0; byte_len];
    weights.read_header_range(meta.byte_range.start, &mut buf)?;
    Ok(buf)
}

fn reject(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("gguf model: {}", message.into()))
}

/// Checked geometry of the Nemotron 3.5 Bolt E4B conversion: 42 layers of
/// which the last 18 share the K/V of the last physical layer of their kind
/// (source 22 sliding, 23 full), two KV heads everywhere, separate V
/// projections on the 24 physical layers.
const BOLT_LAYERS: usize = 42;
const BOLT_SHARED: usize = 18;
const BOLT_HIDDEN: u64 = 2560;
const BOLT_FEED_FORWARD: u64 = 10_240;
const BOLT_QUERY_HEADS: u64 = 8;
const BOLT_VOCAB: u64 = 262_144;
const BOLT_KV_HEADS: u64 = 2;
const BOLT_LOCAL_HEAD: u64 = 256;
const BOLT_GLOBAL_HEAD: u64 = 512;
/// Width of the per-layer input the E4B checkpoint feeds every decoder layer.
const BOLT_PER_LAYER_INPUT: u64 = 256;
/// The window the Bolt resident code is built for; it takes the header's window once bolt.rs does.
pub(super) const BOLT_WINDOW: u64 = 512;
const BOLT_PHYSICAL_LAYERS: usize = BOLT_LAYERS - BOLT_SHARED;

const BOLT_INT_SCALARS: [(&str, u64); 12] = [
    ("gemma4.block_count", BOLT_LAYERS as u64),
    ("gemma4.embedding_length", BOLT_HIDDEN),
    ("gemma4.feed_forward_length", BOLT_FEED_FORWARD),
    ("gemma4.attention.head_count", BOLT_QUERY_HEADS),
    ("gemma4.attention.key_length", BOLT_GLOBAL_HEAD),
    ("gemma4.attention.key_length_swa", BOLT_LOCAL_HEAD),
    ("gemma4.attention.value_length", BOLT_GLOBAL_HEAD),
    ("gemma4.attention.value_length_swa", BOLT_LOCAL_HEAD),
    ("gemma4.attention.sliding_window", BOLT_WINDOW),
    ("gemma4.attention.shared_kv_layers", BOLT_SHARED as u64),
    ("gemma4.rope.dimension_count", BOLT_GLOBAL_HEAD),
    ("gemma4.rope.dimension_count_swa", BOLT_LOCAL_HEAD),
];

/// Which of the two accepted Bolt GGUF spellings a file uses: the legacy
/// engine-private names the in-house converter wrote, or the canonical
/// llama.cpp gemma4 reference names. The sharing map never depends on the
/// spelling: it derives from `gemma4.attention.shared_kv_layers` and the
/// sliding pattern, so both spellings resolve to the same per-layer KV
/// sources.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BoltSpelling {
    Canonical,
    Legacy,
}

impl BoltSpelling {
    /// The model-level per-layer projection onto the residual stream.
    pub(crate) fn ple_model_proj(self) -> &'static str {
        match self {
            BoltSpelling::Canonical => "per_layer_model_proj.weight",
            BoltSpelling::Legacy => "per_layer_model_projection.weight",
        }
    }

    /// The norm over the per-layer embedding width.
    pub(crate) fn ple_proj_norm(self) -> &'static str {
        match self {
            BoltSpelling::Canonical => "per_layer_proj_norm.weight",
            BoltSpelling::Legacy => "per_layer_projection_norm.weight",
        }
    }

    /// The suffix that gates the per-layer embedding into the layer.
    pub(crate) fn ple_gate(self) -> &'static str {
        match self {
            BoltSpelling::Canonical => "inp_gate.weight",
            BoltSpelling::Legacy => "per_layer_gate.weight",
        }
    }

    /// The suffix of the per-layer projection back to the residual stream.
    pub(crate) fn ple_proj(self) -> &'static str {
        match self {
            BoltSpelling::Canonical => "proj.weight",
            BoltSpelling::Legacy => "per_layer_proj.weight",
        }
    }

    /// The suffix of the post-norm after the per-layer projection.
    pub(crate) fn ple_post_norm(self) -> &'static str {
        match self {
            BoltSpelling::Canonical => "post_norm.weight",
            BoltSpelling::Legacy => "post_per_layer_input_norm.weight",
        }
    }

    /// The merged [PLE * layers, vocab] per-layer token table of the
    /// canonical spelling; the legacy spelling has no model-level table.
    pub(crate) fn per_layer_token_embd(self) -> Option<&'static str> {
        match self {
            BoltSpelling::Canonical => Some("per_layer_token_embd.weight"),
            BoltSpelling::Legacy => None,
        }
    }

    /// The per-layer slice of the per-layer token table: `None` when the
    /// spelling carries one table per layer instead of the merged one.
    /// Only the CUDA resident load path asks for it.
    #[cfg(feature = "native-cuda")]
    pub(crate) fn ple_table(self, layer: usize) -> Option<String> {
        match self {
            BoltSpelling::Canonical => None,
            BoltSpelling::Legacy => Some(format!("blk.{layer}.per_layer_embd.weight")),
        }
    }
}

/// Opens the converted Bolt Q8_0 GGUF and checks its metadata and tensors.
/// Applies the checks of open_humanizer_q8 against the Bolt geometry: two KV
/// heads on the 24 physical layers and none on the 18 shared layers.
pub fn open_bolt_q8(path: &Path) -> Result<(), EngError> {
    open_bolt_weights(path).map(|_| ())
}

/// Opens the converted Bolt checkpoint and returns the checked reader plus
/// the spelling its tensors use, for the resident loader. Applies exactly
/// the checks of open_bolt_q8.
pub(crate) fn open_bolt_weights(path: &Path) -> Result<(GgufWeights, BoltSpelling), EngError> {
    let weights = GgufWeights::open(path, BOLT_LIMITS, None)?;
    weights.require_single_file("the Bolt engine")?;
    let header = weights.header();
    check_architecture(header)?;
    check_int_scalars(header, &BOLT_INT_SCALARS)?;
    check_float_scalars(header)?;
    check_bool_scalar(header, "tokenizer.ggml.add_bos_token", true)?;
    let kv = read_i32_array(
        &weights,
        header,
        "gemma4.attention.head_count_kv",
        BOLT_LAYERS,
        GGUF_I32,
    )?;
    let pattern = read_bool_array(
        &weights,
        header,
        "gemma4.attention.sliding_window_pattern",
        BOLT_LAYERS,
        GGUF_BOOL,
    )?;
    check_bolt_layer_pattern(&kv, &pattern)?;
    let spelling = bolt_spelling(header)?;
    check_inventory(header, BOLT_LAYERS, bolt_expected_inventory(spelling))?;
    Ok((weights, spelling))
}

/// Extracts one layer's PLE rows from the canonical merged per_layer_token_embd payload of type
/// `ty`. The reference layout is token-major: each of the `vocab` rows carries every layer's PLE
/// block contiguous, so a layer's slice is `ple` values' blocks at a per-row `stride` offset.
#[cfg(any(test, feature = "native-cuda"))]
pub(crate) fn ple_rows_from_merged(
    merged: &[u8],
    ty: GgufType,
    layer: usize,
    ple: usize,
    layers: usize,
    vocab: usize,
) -> Result<Vec<u8>, EngError> {
    let (elems, bytes) = (ty.block_elems() as usize, ty.block_bytes() as usize);
    if !ple.is_multiple_of(elems) {
        return Err(reject(format!(
            "the per-layer table of {} has rows of {ple} values, not whole blocks",
            ty.name()
        )));
    }
    let width = ple / elems * bytes;
    let stride = width * layers;
    if merged.len() != stride * vocab || layer >= layers {
        return Err(reject(format!(
            "the merged per-layer table holds {} bytes, expected {} for layer {layer} of {layers}",
            merged.len(),
            stride * vocab
        )));
    }
    let mut out = Vec::with_capacity(vocab * width);
    for token in 0..vocab {
        let at = token * stride + layer * width;
        out.extend_from_slice(&merged[at..at + width]);
    }
    Ok(out)
}

/// Resolves which tensor spelling the file carries. A file with both
/// spellings is ambiguous and refused; a file with neither is missing its
/// per-layer machinery entirely.
fn bolt_spelling(header: &crate::weights::GgufHeader) -> Result<BoltSpelling, EngError> {
    let canonical = header.tensor("per_layer_model_proj.weight").is_some();
    let legacy = header.tensor("per_layer_model_projection.weight").is_some();
    // Run before inventory: a legacy file with the canonical merged table
    // appended would otherwise fail as a plain unexpected tensor.
    if header.tensor("per_layer_token_embd.weight").is_some()
        && header.tensor("blk.0.per_layer_embd.weight").is_some()
    {
        return Err(reject(
            "per-layer embedding tables and per_layer_token_embd.weight are both present",
        ));
    }
    match (canonical, legacy) {
        (true, false) => Ok(BoltSpelling::Canonical),
        (false, true) => Ok(BoltSpelling::Legacy),
        (false, false) => Err(reject(
            "missing tensor per_layer_model_proj.weight (and its legacy spelling)",
        )),
        (true, true) => Err(reject(
            "tensors per_layer_model_proj.weight and per_layer_model_projection.weight are both present",
        )),
    }
}

/// Builds the exact Bolt inventory for one spelling: every layer carries
/// its own query, output, norms, feed-forward, per-layer input gate,
/// projection and post-norm, and the layer output scale; the 24 physical
/// layers additionally carry K/V projections and the K norm; the 18 shared
/// layers carry neither K nor V. The per-layer tables are per-layer tensors
/// in the legacy spelling (707 tensors) and rows of one merged model-level
/// per_layer_token_embd in the canonical spelling (666 tensors). The
/// model-level per-layer machinery - the 10752-wide projection, its norm -
/// sits at the top level.
fn bolt_expected_inventory(spelling: BoltSpelling) -> Vec<Expected> {
    let mut expected = Vec::with_capacity(707);
    expected.push(Expected {
        name: "token_embd.weight".to_owned(),
        ty: Stored::Matrix,
        dims: vec![BOLT_HIDDEN, BOLT_VOCAB],
    });
    expected.push(Expected {
        name: "output_norm.weight".to_owned(),
        ty: Stored::Exact(GgufType::F32),
        dims: vec![BOLT_HIDDEN],
    });
    expected.push(Expected {
        name: "rope_freqs.weight".to_owned(),
        ty: Stored::Exact(GgufType::F32),
        dims: vec![BOLT_GLOBAL_HEAD / 2],
    });
    expected.push(Expected {
        name: spelling.ple_model_proj().to_owned(),
        ty: Stored::Matrix,
        dims: vec![BOLT_HIDDEN, BOLT_PER_LAYER_INPUT * BOLT_LAYERS as u64],
    });
    expected.push(Expected {
        name: spelling.ple_proj_norm().to_owned(),
        ty: Stored::Exact(GgufType::F32),
        dims: vec![BOLT_PER_LAYER_INPUT],
    });
    if let Some(name) = spelling.per_layer_token_embd() {
        expected.push(Expected {
            name: name.to_owned(),
            ty: Stored::Matrix,
            dims: vec![BOLT_PER_LAYER_INPUT * BOLT_LAYERS as u64, BOLT_VOCAB],
        });
    }
    for layer in 0..BOLT_LAYERS {
        let local = layer % 6 != 5;
        let head = if local {
            BOLT_LOCAL_HEAD
        } else {
            BOLT_GLOBAL_HEAD
        };
        let physical = layer < BOLT_PHYSICAL_LAYERS;
        let base = format!("blk.{layer}.");
        let norm = |suffix: &str, width: u64| Expected {
            name: format!("{base}{suffix}"),
            ty: Stored::Exact(GgufType::F32),
            dims: vec![width],
        };
        let matrix = |suffix: &str, rows: u64, cols: u64| Expected {
            name: format!("{base}{suffix}"),
            ty: Stored::Matrix,
            dims: vec![rows, cols],
        };
        expected.push(norm("attn_norm.weight", BOLT_HIDDEN));
        expected.push(matrix(
            "attn_q.weight",
            BOLT_HIDDEN,
            BOLT_QUERY_HEADS * head,
        ));
        expected.push(norm("attn_q_norm.weight", head));
        if physical {
            expected.push(matrix("attn_k.weight", BOLT_HIDDEN, BOLT_KV_HEADS * head));
            expected.push(norm("attn_k_norm.weight", head));
            expected.push(matrix("attn_v.weight", BOLT_HIDDEN, BOLT_KV_HEADS * head));
        }
        expected.push(matrix(
            "attn_output.weight",
            BOLT_QUERY_HEADS * head,
            BOLT_HIDDEN,
        ));
        expected.push(norm("post_attention_norm.weight", BOLT_HIDDEN));
        expected.push(norm("ffn_norm.weight", BOLT_HIDDEN));
        expected.push(matrix("ffn_gate.weight", BOLT_HIDDEN, BOLT_FEED_FORWARD));
        expected.push(matrix("ffn_up.weight", BOLT_HIDDEN, BOLT_FEED_FORWARD));
        expected.push(matrix("ffn_down.weight", BOLT_FEED_FORWARD, BOLT_HIDDEN));
        expected.push(norm("post_ffw_norm.weight", BOLT_HIDDEN));
        expected.push(matrix(
            spelling.ple_gate(),
            BOLT_HIDDEN,
            BOLT_PER_LAYER_INPUT,
        ));
        expected.push(matrix(
            spelling.ple_proj(),
            BOLT_PER_LAYER_INPUT,
            BOLT_HIDDEN,
        ));
        expected.push(norm(spelling.ple_post_norm(), BOLT_HIDDEN));
        if spelling == BoltSpelling::Legacy {
            expected.push(matrix(
                "per_layer_embd.weight",
                BOLT_PER_LAYER_INPUT,
                BOLT_VOCAB,
            ));
        }
        expected.push(norm("layer_output_scale.weight", 1));
    }
    debug_assert_eq!(
        expected.len(),
        match spelling {
            BoltSpelling::Canonical => 707 - BOLT_LAYERS + 1,
            BoltSpelling::Legacy => 707,
        }
    );
    expected
}

/// Checks the five-sliding, one-full pattern and the KV head array. The
/// canonical spelling writes the two KV heads on every layer (a zero head
/// count made the reference fork SIGFPE in ggml_flash_attn_ext); the legacy
/// spelling wrote zeros on the 18 shared layers. Both spellings resolve to
/// the same sharing map because sharing derives from
/// `gemma4.attention.shared_kv_layers`, never from this array: the array
/// must simply be one of the two consistent spellings, not a per-layer mix.
fn check_bolt_layer_pattern(kv: &[i32], pattern: &[u8]) -> Result<(), EngError> {
    let canonical = kv.iter().all(|heads| *heads == BOLT_KV_HEADS as i32);
    for (layer, (kv_heads, sliding)) in kv.iter().zip(pattern).enumerate() {
        let local = layer % 6 != 5;
        let physical = layer < BOLT_PHYSICAL_LAYERS;
        let want_legacy = if physical { BOLT_KV_HEADS as i32 } else { 0 };
        if *kv_heads != want_legacy && !canonical {
            return Err(reject(format!(
                "metadata key gemma4.attention.head_count_kv is {kv_heads} at layer {layer}, expected {want_legacy} or the all-{} canonical spelling",
                BOLT_KV_HEADS
            )));
        }
        let want_pattern = u8::from(local);
        if *sliding != want_pattern {
            return Err(reject(format!(
                "metadata key gemma4.attention.sliding_window_pattern is {sliding} at layer {layer}, expected {want_pattern}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::spec::NEMOTRON_BOLT_E4B;

    #[test]
    fn the_gemma_and_bolt_openers_refuse_a_split_set() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first = crate::weights::write_two_shard_set(dir.path(), "gemma");
        for error in [
            open_humanizer_weights(&first)
                .map(|_| ())
                .expect_err("a split set is refused"),
            open_bolt_weights(&first)
                .map(|_| ())
                .expect_err("a split set is refused"),
        ] {
            assert!(
                error.describe().contains("does not take a split GGUF set"),
                "{}",
                error.describe()
            );
        }
    }

    /// The reader's geometry constants are the descriptor's, not a second
    /// source of truth: a descriptor change fails here until the reader agrees.
    #[test]
    fn bolt_geometry_constants_match_the_descriptor() {
        let spec = NEMOTRON_BOLT_E4B;
        assert_eq!(BOLT_LAYERS as u32, spec.layers);
        assert_eq!(BOLT_SHARED as u32, spec.kv_shared_layers);
        assert_eq!(BOLT_PHYSICAL_LAYERS as u32, spec.first_shared_layer());
        assert_eq!(BOLT_HIDDEN, u64::from(spec.hidden));
        assert_eq!(
            BOLT_FEED_FORWARD,
            u64::from(spec.feed_forward.expect("dense").intermediate)
        );
        assert_eq!(BOLT_QUERY_HEADS, u64::from(spec.heads_q));
        assert_eq!(BOLT_KV_HEADS, u64::from(spec.heads_kv));
        assert_eq!(BOLT_VOCAB, u64::from(spec.vocab));
        assert_eq!(BOLT_LOCAL_HEAD, u64::from(spec.head_dim));
        assert_eq!(BOLT_GLOBAL_HEAD, u64::from(spec.full_head_dim()));
    }
}
