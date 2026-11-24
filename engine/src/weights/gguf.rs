//! GGUF v3 header and tensor table reader (slice G1).
//!
//! The reader consumes only the header bytes: the fixed prefix, the metadata
//! entries, and the tensor table. It never reads tensor payloads. Every
//! malformed input fails closed with `EngError::Checkpoint`.
//!
//! The layout follows the GGUF format specification. All integers are
//! little-endian. Tensor offsets are relative to the data section, which starts
//! at the header end rounded up to `general.alignment` (default 32).
//!
//! Deliberate differences from the llama.cpp reader (`gguf.cpp`):
//! - Metadata arrays may nest up to four levels deep. Deeper nesting fails.
//! - Tensor data offsets may leave gaps and may appear in any table order.
//!   Only overlap, misalignment, and out-of-file ranges fail.
//! - The data start is the padded header end, even when the file has no tensors. A file
//!   with no tensors may end short of it.
//! - A tensor dimension of zero fails.
//! - Metadata keys may be up to 65,535 bytes. Tensor names follow the format
//!   limit and must be shorter than 64 bytes.

use std::{
    collections::{HashMap, HashSet},
    io::{ErrorKind, Read},
    ops::Range,
};

use sha2::{Digest, Sha256};

use crate::EngError;

const MAGIC: [u8; 4] = *b"GGUF";
const SUPPORTED_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_KEY_BYTES: u64 = 65_535;
/// The format limit is `GGML_MAX_NAME` (64) including the terminator.
const MAX_TENSOR_NAME_BYTES: u64 = 63;
const STRING_CHUNK: u64 = 1 << 16;
const MAX_TENSOR_DIMS: u32 = 4;
/// Deepest array-in-array nesting accepted. The top-level array has depth 0.
const MAX_ARRAY_DEPTH: u32 = 4;
/// Smallest encoded metadata entry: key length, one key byte, type, one value byte.
const MIN_KV_BYTES: u64 = 8 + 1 + 4 + 1;
/// Smallest encoded tensor entry: name, one dimension, type, and offset.
const MIN_TENSOR_BYTES: u64 = 8 + 1 + 4 + 8 + 4 + 8;
const SKIP_CHUNK: usize = 8192;
const HASH_CHUNK: usize = 1 << 20;

fn bad<T>(message: impl Into<String>) -> Result<T, EngError> {
    Err(EngError::Checkpoint(format!("gguf: {}", message.into())))
}

/// A ggml tensor storage type that this reader can size.
///
/// Type ids and block geometry follow `ggml/include/ggml.h` and
/// `ggml/src/ggml-common.h` at llama.cpp commit `fb4b2737`.
/// NVFP4 also follows the Flint reference at `a37cc461`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufType {
    F32,
    F16,
    Bf16,
    F64,
    I8,
    I16,
    I32,
    I64,
    Q40,
    Q41,
    Q50,
    Q51,
    Q80,
    Mxfp4,
    Nvfp4,
    Q20,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq1S,
    Iq1M,
    Iq4Nl,
    Iq4Xs,
}

impl GgufType {
    pub fn from_id(id: u32) -> Result<Self, EngError> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q40,
            3 => Self::Q41,
            6 => Self::Q50,
            7 => Self::Q51,
            8 => Self::Q80,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            39 => Self::Mxfp4,
            40 => Self::Nvfp4,
            42 => Self::Q20,
            _ => return bad(format!("unknown or unsupported tensor type id {id}")),
        })
    }

    pub fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q40 => 2,
            Self::Q41 => 3,
            Self::Q50 => 6,
            Self::Q51 => 7,
            Self::Q80 => 8,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Iq2Xxs => 16,
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq1S => 19,
            Self::Iq4Nl => 20,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::Iq1M => 29,
            Self::Bf16 => 30,
            Self::Mxfp4 => 39,
            Self::Nvfp4 => 40,
            Self::Q20 => 42,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Q40 => "Q4_0",
            Self::Q41 => "Q4_1",
            Self::Q50 => "Q5_0",
            Self::Q51 => "Q5_1",
            Self::Q80 => "Q8_0",
            Self::Mxfp4 => "MXFP4",
            Self::Nvfp4 => "NVFP4",
            Self::Q20 => "Q2_0",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq2S => "IQ2_S",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq3S => "IQ3_S",
            Self::Iq1S => "IQ1_S",
            Self::Iq1M => "IQ1_M",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq4Xs => "IQ4_XS",
        }
    }

    /// True for the quantized types the native CUDA kernels read as stored blocks (decode matvec,
    /// expert matvec and dequantizing or vendored prefill GEMM). The one list every loader and
    /// planner checks; `kernels::cuda::plan` pins its block geometry to it.
    pub const fn has_native_kernel(self) -> bool {
        matches!(
            self,
            Self::Q4K
                | Self::Q6K
                | Self::Q80
                | Self::Q50
                | Self::Q5K
                | Self::Q40
                | Self::Q3K
                | Self::Iq4Nl
                | Self::Q20
                | Self::Iq2Xxs
                | Self::Iq2Xs
                | Self::Iq2S
                | Self::Iq3Xxs
                | Self::Iq3S
                | Self::Iq4Xs
                | Self::Q2K
                | Self::Iq1S
                | Self::Iq1M
        )
    }

    /// Every type with a native kernel, in id order.
    pub fn native_types() -> Vec<Self> {
        (0..64)
            .filter_map(|id| Self::from_id(id).ok())
            .filter(|ty| ty.has_native_kernel())
            .collect()
    }

    /// Weights per block. A row length must be a multiple of this value.
    pub fn block_elems(self) -> u64 {
        match self {
            Self::F32
            | Self::F16
            | Self::Bf16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => 1,
            Self::Q40
            | Self::Q41
            | Self::Q50
            | Self::Q51
            | Self::Q80
            | Self::Iq4Nl
            | Self::Mxfp4 => 32,
            Self::Q20 | Self::Nvfp4 => 64,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq2S
            | Self::Iq3Xxs
            | Self::Iq3S
            | Self::Iq1S
            | Self::Iq1M
            | Self::Iq4Xs => 256,
        }
    }

    /// Stored bytes per block.
    pub fn block_bytes(self) -> u64 {
        match self {
            Self::I8 => 1,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::Q40 | Self::Q20 | Self::Iq4Nl => 18,
            Self::Q41 => 20,
            Self::Q50 => 22,
            Self::Q51 => 24,
            Self::Q80 => 34,
            Self::Mxfp4 => 17,
            Self::Nvfp4 => 36,
            Self::Q2K => 84,
            Self::Q3K | Self::Iq3S => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq2S => 82,
            Self::Iq3Xxs => 98,
            Self::Iq1S => 50,
            Self::Iq1M => 56,
            Self::Iq4Xs => 136,
        }
    }
}

/// A metadata value. Arrays are validated in full but not retained.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(ArrayMeta),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayMeta {
    /// GGUF metadata type id of every element.
    pub elem_type: u32,
    pub len: u64,
    /// File byte range of the element data, after the element type and count.
    pub byte_range: Range<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitInfo {
    /// `split.count`: the number of shards.
    pub count: u64,
    /// `split.no`: this shard's zero-based index.
    pub no: u64,
    /// `split.tensors.count`: the tensor total over all shards, when present.
    pub tensors_count: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensor {
    name: String,
    dims: Vec<u64>,
    ggml_type: GgufType,
    data_offset: u64,
    abs_offset: u64,
    byte_len: u64,
    element_count: u64,
    shard: usize,
}

/// The largest tensor, in stored bytes, that the loaders widen to F32 when a file keeps it as F16
/// or BF16.
pub const SMALL_VECTOR_BYTES: u64 = 1 << 20;

impl GgufTensor {
    /// Whether the tensor is a norm, bias or gate vector that a file stores in half precision:
    /// one dimension, F16 or BF16, under [`SMALL_VECTOR_BYTES`]. Every loader reads such a tensor
    /// as F32 values; the large matrices are not vectors of this kind.
    pub fn is_small_half_vector(&self) -> bool {
        self.dims.len() == 1
            && matches!(self.ggml_type, GgufType::F16 | GgufType::Bf16)
            && self.byte_len < SMALL_VECTOR_BYTES
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    /// Dimensions in GGUF order: the first entry is the contiguous row length.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }
    pub fn ggml_type(&self) -> GgufType {
        self.ggml_type
    }
    /// Offset relative to the data section start.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }
    /// Offset from the start of the file that holds the tensor. In a split set that is
    /// the tensor's own shard file; see [`Self::shard`].
    pub fn abs_offset(&self) -> u64 {
        self.abs_offset
    }
    /// Zero-based index of the shard file that holds the tensor (always 0 outside a split set).
    pub fn shard(&self) -> usize {
        self.shard
    }
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
    pub fn element_count(&self) -> u64 {
        self.element_count
    }
    /// Absolute byte range in the file.
    pub fn abs_range(&self) -> Range<u64> {
        self.abs_offset..self.abs_offset + self.byte_len
    }
}

/// A parsed and checked GGUF header and tensor table.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    version: u32,
    alignment: u64,
    data_start: u64,
    header_len: u64,
    file_len: u64,
    header_sha256: [u8; 32],
    metadata: Vec<(String, MetaValue)>,
    tensors: Vec<GgufTensor>,
    by_name: HashMap<String, usize>,
    split: Option<SplitInfo>,
}

struct Src<'a, R: ?Sized> {
    inner: &'a mut R,
    pos: u64,
    max_header: u64,
    file_len: u64,
    hasher: Sha256,
}

impl<R: Read + ?Sized> Src<'_, R> {
    fn remaining(&self) -> u64 {
        self.max_header.min(self.file_len) - self.pos
    }

    fn fill(&mut self, buf: &mut [u8]) -> Result<(), EngError> {
        let end = self.pos.checked_add(buf.len() as u64);
        let Some(end) = end else {
            return bad("header offset overflow");
        };
        if end > self.max_header {
            return bad("header exceeds the header byte limit");
        }
        if end > self.file_len {
            return bad("truncated: header exceeds the file length");
        }
        self.inner.read_exact(buf).map_err(|error| {
            if error.kind() == ErrorKind::UnexpectedEof {
                EngError::Checkpoint("gguf: truncated header".into())
            } else {
                EngError::Checkpoint(format!("gguf: header read: {error}"))
            }
        })?;
        self.hasher.update(&*buf);
        self.pos = end;
        Ok(())
    }

    fn u32(&mut self) -> Result<u32, EngError> {
        let mut b = [0; 4];
        self.fill(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64, EngError> {
        let mut b = [0; 8];
        self.fill(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], EngError> {
        let mut b = [0; N];
        self.fill(&mut b)?;
        Ok(b)
    }

    fn string(&mut self, max_len: u64, what: &str) -> Result<String, EngError> {
        let len = self.u64()?;
        if len > max_len {
            return bad(format!("{what} is longer than {max_len} bytes"));
        }
        if len > self.remaining() {
            return bad(format!("{what} length exceeds the remaining header"));
        }
        // Read in bounded chunks so a declared length never forces one large allocation.
        let mut bytes = Vec::new();
        let mut left = len;
        while left > 0 {
            let n = left.min(STRING_CHUNK) as usize;
            let at = bytes.len();
            bytes.resize(at + n, 0);
            self.fill(&mut bytes[at..])?;
            left -= n as u64;
        }
        String::from_utf8(bytes).or_else(|_| bad(format!("{what} is not valid UTF-8")))
    }

    fn name(&mut self, what: &str, max_len: u64) -> Result<String, EngError> {
        let name = self.string(max_len, what)?;
        if name.is_empty() {
            return bad(format!("{what} is empty"));
        }
        Ok(name)
    }

    fn skip(&mut self, mut len: u64, check_bool: bool) -> Result<(), EngError> {
        let mut buf = [0u8; SKIP_CHUNK];
        while len > 0 {
            let n = len.min(SKIP_CHUNK as u64) as usize;
            self.fill(&mut buf[..n])?;
            if check_bool && buf[..n].iter().any(|&b| b > 1) {
                return bad("boolean array element is not 0 or 1");
            }
            len -= n as u64;
        }
        Ok(())
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<MetaValue, EngError> {
        Ok(match ty {
            0 => MetaValue::U8(self.array::<1>()?[0]),
            1 => MetaValue::I8(self.array::<1>()?[0] as i8),
            2 => MetaValue::U16(u16::from_le_bytes(self.array()?)),
            3 => MetaValue::I16(i16::from_le_bytes(self.array()?)),
            4 => MetaValue::U32(u32::from_le_bytes(self.array()?)),
            5 => MetaValue::I32(i32::from_le_bytes(self.array()?)),
            6 => MetaValue::F32(f32::from_le_bytes(self.array()?)),
            7 => match self.array::<1>()?[0] {
                0 => MetaValue::Bool(false),
                1 => MetaValue::Bool(true),
                other => return bad(format!("boolean value {other} is not 0 or 1")),
            },
            8 => MetaValue::String(self.string(self.max_header, "string value")?),
            9 => MetaValue::Array(self.array_value(depth)?),
            10 => MetaValue::U64(u64::from_le_bytes(self.array()?)),
            11 => MetaValue::I64(i64::from_le_bytes(self.array()?)),
            12 => MetaValue::F64(f64::from_le_bytes(self.array()?)),
            other => return bad(format!("unknown metadata value type {other}")),
        })
    }

    fn array_value(&mut self, depth: u32) -> Result<ArrayMeta, EngError> {
        if depth >= MAX_ARRAY_DEPTH {
            return bad("metadata arrays are nested too deeply");
        }
        let elem_type = self.u32()?;
        let len = self.u64()?;
        // Smallest encoded element size, which bounds the count by the header size.
        let (min_size, fixed) = match elem_type {
            0 | 1 | 7 => (1, true),
            2 | 3 => (2, true),
            4..=6 => (4, true),
            10..=12 => (8, true),
            8 => (8, false),
            9 => (12, false),
            other => return bad(format!("unknown array element type {other}")),
        };
        match len.checked_mul(min_size) {
            Some(bytes) if bytes <= self.remaining() => {}
            _ => return bad("array length exceeds the remaining header"),
        }
        let start = self.pos;
        if fixed {
            self.skip(len * min_size, elem_type == 7)?;
        } else {
            for _ in 0..len {
                if elem_type == 8 {
                    self.string(self.max_header, "array string")?;
                } else {
                    self.array_value(depth + 1)?;
                }
            }
        }
        Ok(ArrayMeta {
            elem_type,
            len,
            byte_range: start..self.pos,
        })
    }
}

fn nonneg_int(value: &MetaValue, key: &str) -> Result<u64, EngError> {
    let signed = |v: i64| u64::try_from(v).or_else(|_| bad(format!("{key} is negative")));
    match *value {
        MetaValue::U8(v) => Ok(v.into()),
        MetaValue::U16(v) => Ok(v.into()),
        MetaValue::U32(v) => Ok(v.into()),
        MetaValue::U64(v) => Ok(v),
        MetaValue::I8(v) => signed(v.into()),
        MetaValue::I16(v) => signed(v.into()),
        MetaValue::I32(v) => signed(v.into()),
        MetaValue::I64(v) => signed(v),
        _ => bad(format!("{key} is not an integer")),
    }
}

fn find<'a>(metadata: &'a [(String, MetaValue)], key: &str) -> Option<&'a MetaValue> {
    metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn parse_tensor<R: Read + ?Sized>(src: &mut Src<'_, R>) -> Result<GgufTensor, EngError> {
    let name = src.name("tensor name", MAX_TENSOR_NAME_BYTES)?;
    let n_dims = src.u32()?;
    if !(1..=MAX_TENSOR_DIMS).contains(&n_dims) {
        return bad(format!("tensor {name} has {n_dims} dimensions"));
    }
    let mut dims = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        let dim = src.u64()?;
        if dim == 0 {
            return bad(format!("tensor {name} has a zero dimension"));
        }
        dims.push(dim);
    }
    let ggml_type = GgufType::from_id(src.u32()?)?;
    let data_offset = src.u64()?;
    let block = ggml_type.block_elems();
    if dims[0] % block != 0 {
        return bad(format!(
            "tensor {name} row length {} is not a multiple of the {} block size {block}",
            dims[0],
            ggml_type.name()
        ));
    }
    let mut rows = 1u64;
    for &dim in &dims[1..] {
        rows = rows
            .checked_mul(dim)
            .map_or_else(|| bad(format!("tensor {name} size overflow")), Ok)?;
    }
    let element_count = dims[0]
        .checked_mul(rows)
        .map_or_else(|| bad(format!("tensor {name} element count overflows")), Ok)?;
    let byte_len = (dims[0] / block)
        .checked_mul(rows)
        .and_then(|blocks| blocks.checked_mul(ggml_type.block_bytes()))
        .map_or_else(|| bad(format!("tensor {name} byte size overflows")), Ok)?;
    Ok(GgufTensor {
        name,
        dims,
        ggml_type,
        data_offset,
        abs_offset: 0,
        byte_len,
        element_count,
        shard: 0,
    })
}

fn parse_split(
    metadata: &[(String, MetaValue)],
    tensor_count: usize,
) -> Result<Option<SplitInfo>, EngError> {
    let count = find(metadata, "split.count");
    let no = find(metadata, "split.no");
    let total = find(metadata, "split.tensors.count");
    let (count, no) = match (count, no, total) {
        (None, None, None) => return Ok(None),
        (Some(count), Some(no), _) => (
            nonneg_int(count, "split.count")?,
            nonneg_int(no, "split.no")?,
        ),
        _ => return bad("split.count and split.no must appear together"),
    };
    if count == 0 || no >= count {
        return bad(format!("split.no {no} is outside split.count {count}"));
    }
    let tensors_count = total
        .map(|v| nonneg_int(v, "split.tensors.count"))
        .transpose()?;
    if let Some(total) = tensors_count {
        let own = tensor_count as u64;
        if total < own || (count == 1 && total != own) {
            return bad(format!(
                "split.tensors.count {total} disagrees with {own} tensors in this shard"
            ));
        }
    }
    Ok(Some(SplitInfo {
        count,
        no,
        tensors_count,
    }))
}

impl GgufHeader {
    /// Parses the header and tensor table from `reader`.
    ///
    /// `file_len` is the full file length. `max_header_bytes` bounds the header
    /// size, including counts and lengths declared inside it. The reader stops
    /// at the end of the tensor table and never reads tensor payloads.
    pub fn parse<R: Read + ?Sized>(
        reader: &mut R,
        file_len: u64,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        if max_header_bytes == 0 {
            return Err(EngError::BadRequest(
                "header byte limit must be nonzero".into(),
            ));
        }
        let mut src = Src {
            inner: reader,
            pos: 0,
            max_header: max_header_bytes,
            file_len,
            hasher: Sha256::new(),
        };
        let magic: [u8; 4] = src.array()?;
        if magic != MAGIC {
            return bad("bad magic");
        }
        let version = src.u32()?;
        if version != SUPPORTED_VERSION {
            return bad(format!("unsupported version {version}"));
        }
        let tensor_count = src.u64()?;
        let kv_count = src.u64()?;
        let fits = |count: u64, min: u64, src: &Src<'_, R>| {
            count
                .checked_mul(min)
                .is_some_and(|bytes| bytes <= src.remaining())
        };
        if !fits(tensor_count, MIN_TENSOR_BYTES, &src) {
            return bad("tensor count exceeds the remaining header");
        }
        if !fits(kv_count, MIN_KV_BYTES, &src) {
            return bad("metadata count exceeds the remaining header");
        }

        let mut metadata = Vec::with_capacity(kv_count.min(1024) as usize);
        let mut seen = HashSet::new();
        for _ in 0..kv_count {
            let key = src.name("metadata key", MAX_KEY_BYTES)?;
            if !seen.insert(key.clone()) {
                return bad(format!("duplicate metadata key {key}"));
            }
            let ty = src.u32()?;
            let value = src.value(ty, 0)?;
            metadata.push((key, value));
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(4096) as usize);
        let mut by_name = HashMap::with_capacity(tensor_count.min(4096) as usize);
        for _ in 0..tensor_count {
            let tensor = parse_tensor(&mut src)?;
            if by_name.insert(tensor.name.clone(), tensors.len()).is_some() {
                return bad(format!("duplicate tensor name {}", tensor.name));
            }
            tensors.push(tensor);
        }

        let header_len = src.pos;
        let header_sha256: [u8; 32] = src.hasher.finalize().into();

        let alignment = match find(&metadata, "general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(MetaValue::U32(a)) if a.is_power_of_two() => u64::from(*a),
            Some(_) => return bad("general.alignment is not a nonzero power-of-two u32"),
        };
        let data_start = header_len
            .checked_add(alignment - 1)
            .map(|v| v / alignment * alignment)
            .map_or_else(|| bad("data start overflows"), Ok)?;
        // A file without tensors (shard 1 of a split set holds only metadata) may end at its
        // header, short of the padded data start: its data section is empty.
        if data_start > file_len && !tensors.is_empty() {
            return bad("the data section starts beyond the end of the file");
        }
        for tensor in &mut tensors {
            if tensor.data_offset % alignment != 0 {
                return bad(format!(
                    "tensor {} offset {} is not aligned to {alignment}",
                    tensor.name, tensor.data_offset
                ));
            }
            let start = data_start.checked_add(tensor.data_offset).map_or_else(
                || bad(format!("tensor {} offset overflows", tensor.name)),
                Ok,
            )?;
            let end = start.checked_add(tensor.byte_len).map_or_else(
                || bad(format!("tensor {} range overflows", tensor.name)),
                Ok,
            )?;
            if end > file_len {
                return bad(format!(
                    "tensor {} extends beyond the end of the file",
                    tensor.name
                ));
            }
            tensor.abs_offset = start;
        }
        let mut order: Vec<usize> = (0..tensors.len()).collect();
        order.sort_by_key(|&i| tensors[i].abs_offset);
        for pair in order.windows(2) {
            let (a, b) = (&tensors[pair[0]], &tensors[pair[1]]);
            if a.abs_offset + a.byte_len > b.abs_offset {
                return bad(format!("tensors {} and {} overlap", a.name, b.name));
            }
        }

        let split = parse_split(&metadata, tensors.len())?;
        Ok(Self {
            version,
            alignment,
            data_start,
            header_len,
            file_len,
            header_sha256,
            metadata,
            tensors,
            by_name,
            split,
        })
    }

    /// Joins the headers of one split set into a single view: the metadata of shard 1 and
    /// every shard's tensors, in shard order. Each tensor keeps its own shard's absolute
    /// offset and records the shard in [`GgufTensor::shard`]. `headers` must be ordered by
    /// shard number and have passed [`check_split_set`]; the first header supplies the
    /// version, data start, header length, file length and header digest of the result.
    ///
    /// A later shard may carry only split keys. Any other key it holds must have the value
    /// shard 1 has (arrays compare by element type and length, since their contents are not
    /// kept), and its `general.alignment`, when present, must be a valid alignment.
    pub fn merge_split(headers: Vec<GgufHeader>) -> Result<Self, EngError> {
        let refs: Vec<&GgufHeader> = headers.iter().collect();
        check_split_set(&refs)?;
        for (index, header) in headers.iter().enumerate() {
            let no = header.split().map_or(u64::MAX, |info| info.no);
            if no != index as u64 {
                return bad(format!(
                    "shard headers are out of order: position {index} holds split.no {no}"
                ));
            }
        }
        let mut headers = headers.into_iter();
        let Some(mut merged) = headers.next() else {
            return bad("split set is empty");
        };
        for (offset, header) in headers.enumerate() {
            let shard = offset + 1;
            for (key, value) in header.metadata() {
                if key.starts_with("split.") {
                    continue;
                }
                match find(&merged.metadata, key) {
                    Some(first) if meta_agrees(first, value) => {}
                    Some(_) => {
                        return bad(format!(
                            "shard {} of the split set disagrees with shard 1 on metadata key {key}",
                            shard + 1
                        ));
                    }
                    None => {
                        return bad(format!(
                            "shard {} of the split set has metadata key {key}, which shard 1 lacks",
                            shard + 1
                        ));
                    }
                }
            }
            for mut tensor in header.tensors {
                tensor.shard = shard;
                merged
                    .by_name
                    .insert(tensor.name.clone(), merged.tensors.len());
                merged.tensors.push(tensor);
            }
        }
        Ok(merged)
    }

    /// Parses a header from a byte slice that holds at least the header bytes.
    pub fn from_bytes(
        bytes: &[u8],
        file_len: u64,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        let mut reader = bytes;
        Self::parse(&mut reader, file_len, max_header_bytes)
    }

    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn alignment(&self) -> u64 {
        self.alignment
    }
    /// File offset where the data section starts.
    pub fn data_start(&self) -> u64 {
        self.data_start
    }
    /// Bytes from the file start to the end of the tensor table, before padding.
    pub fn header_len(&self) -> u64 {
        self.header_len
    }
    pub fn file_len(&self) -> u64 {
        self.file_len
    }
    /// SHA-256 over the first `header_len` bytes of the file.
    pub fn header_sha256(&self) -> [u8; 32] {
        self.header_sha256
    }
    pub fn metadata(&self) -> &[(String, MetaValue)] {
        &self.metadata
    }
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        find(&self.metadata, key)
    }
    /// An unsigned integer key stored as u32 or u64; `None` when the key is absent or holds
    /// another type.
    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        match self.meta(key)? {
            MetaValue::U32(value) => Some(u64::from(*value)),
            MetaValue::U64(value) => Some(*value),
            _ => None,
        }
    }
    /// An unsigned integer key that fits u32; `None` when absent, of another type or too large.
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.meta_u64(key)
            .and_then(|value| u32::try_from(value).ok())
    }
    pub fn tensors(&self) -> &[GgufTensor] {
        &self.tensors
    }
    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.by_name.get(name).map(|&index| &self.tensors[index])
    }
    pub fn split(&self) -> Option<SplitInfo> {
        self.split
    }

    /// Checks the header hash against a recorded lowercase or uppercase hex digest.
    pub fn check_header_sha256(&self, expected_hex: &str) -> Result<(), EngError> {
        let expected = parse_hex_digest(expected_hex)?;
        if expected != self.header_sha256 {
            return bad(format!(
                "header sha256 {} differs from the recorded {}",
                sha256_hex(&self.header_sha256),
                expected_hex.to_ascii_lowercase()
            ));
        }
        Ok(())
    }
}

/// Whether a later shard's value for a key equals shard 1's. Array contents are not
/// retained, so arrays agree when their element type and length do.
fn meta_agrees(first: &MetaValue, other: &MetaValue) -> bool {
    match (first, other) {
        (MetaValue::Array(a), MetaValue::Array(b)) => a.elem_type == b.elem_type && a.len == b.len,
        _ => first == other,
    }
}

/// Checks that parsed shard headers form one complete, consistent split set.
///
/// Every header needs split metadata. The shards must agree on `split.count`,
/// cover each index once, agree on any `split.tensors.count`, and share no
/// tensor name. The shard tensor counts must add up to that total.
pub fn check_split_set(headers: &[&GgufHeader]) -> Result<(), EngError> {
    if headers.is_empty() {
        return bad("split set is empty");
    }
    let infos: Vec<SplitInfo> = headers
        .iter()
        .map(|h| {
            h.split()
                .map_or_else(|| bad("shard has no split metadata"), Ok)
        })
        .collect::<Result<_, _>>()?;
    let count = infos[0].count;
    if infos.iter().any(|i| i.count != count) {
        return bad("shards disagree on split.count");
    }
    if headers.len() as u64 != count {
        return bad(format!(
            "split set has {} shards but split.count is {count}",
            headers.len()
        ));
    }
    let mut numbers: Vec<u64> = infos.iter().map(|i| i.no).collect();
    numbers.sort_unstable();
    if numbers.iter().enumerate().any(|(i, &n)| n != i as u64) {
        return bad("split.no values do not cover each shard once");
    }
    let totals: HashSet<Option<u64>> = infos.iter().map(|i| i.tensors_count).collect();
    if totals.len() != 1 {
        return bad("shards disagree on split.tensors.count");
    }
    let mut names = HashSet::new();
    let mut sum = 0u64;
    for header in headers {
        for tensor in header.tensors() {
            if !names.insert(tensor.name()) {
                return bad(format!(
                    "tensor {} appears in more than one shard",
                    tensor.name()
                ));
            }
            sum += 1;
        }
    }
    if let Some(Some(total)) = totals.into_iter().next()
        && total != sum
    {
        return bad(format!(
            "split.tensors.count {total} differs from {sum} listed tensors"
        ));
    }
    Ok(())
}

/// Formats a digest as 64 lowercase hex digits.
pub fn sha256_hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write;
    digest.iter().fold(String::with_capacity(64), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// Bounded content sample for load identities: the length of the file's data
/// region, then its first and last 64 KiB, enter `hasher` — so a same-size
/// replacement of the region cannot collide with (name, size) alone. A region
/// under 128 KiB is covered end to end; a longer one leaves the middle
/// unsampled. Load cost stays bounded at 128 KiB per weight file (a
/// 200-shard pack reads about 25 MiB).
pub fn hash_sampled_edges(
    hasher: &mut sha2::Sha256,
    file: &std::path::Path,
    data_start: u64,
    len: u64,
) -> Result<(), std::io::Error> {
    use sha2::Digest;
    use std::os::unix::fs::FileExt;

    const EDGE: u64 = 64 * 1024;
    let region = len.saturating_sub(data_start);
    hasher.update(region.to_le_bytes());
    let head = region.min(EDGE);
    let mut buf = vec![0u8; head as usize];
    let file = std::fs::File::open(file)?;
    file.read_exact_at(&mut buf, data_start)?;
    hasher.update(&buf);
    let tail = region.saturating_sub(head).min(EDGE);
    if tail > 0 {
        buf.clear();
        buf.resize(tail as usize, 0);
        file.read_exact_at(&mut buf, data_start + region - tail)?;
        hasher.update(&buf);
    }
    Ok(())
}

fn parse_hex_digest(hex: &str) -> Result<[u8; 32], EngError> {
    let bytes = hex.as_bytes();
    if bytes.len() != 64 || !bytes.iter().all(u8::is_ascii_hexdigit) {
        return bad("recorded sha256 is not 64 hex digits");
    }
    let nibble = |c: u8| (c as char).to_digit(16).unwrap_or(0) as u8;
    let mut out = [0u8; 32];
    for (i, pair) in bytes.chunks(2).enumerate() {
        out[i] = nibble(pair[0]) << 4 | nibble(pair[1]);
    }
    Ok(out)
}

/// Streams `reader` through SHA-256. Returns the digest and the byte count.
pub fn sha256_stream<R: Read + ?Sized>(reader: &mut R) -> Result<([u8; 32], u64), EngError> {
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_CHUNK];
    let mut total = 0u64;
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                hasher.update(&buf[..n]);
                total += n as u64;
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return bad(format!("hash read: {error}")),
        }
    }
    Ok((hasher.finalize().into(), total))
}

/// Streams `reader` and checks its length and SHA-256 against recorded values.
pub fn verify_sha256<R: Read + ?Sized>(
    reader: &mut R,
    expected_len: u64,
    expected_hex: &str,
) -> Result<(), EngError> {
    let expected = parse_hex_digest(expected_hex)?;
    let (digest, len) = sha256_stream(reader)?;
    if len != expected_len {
        return bad(format!("hashed {len} bytes but expected {expected_len}"));
    }
    if digest != expected {
        return bad(format!(
            "sha256 {} differs from the recorded {}",
            sha256_hex(&digest),
            expected_hex.to_ascii_lowercase()
        ));
    }
    Ok(())
}

/// Files below this size are parsed on every call: a header cache is for model files, and a small
/// file may be rewritten under the same size and modification time.
const SHARED_HEADER_MIN_FILE_BYTES: u64 = 1 << 20;

/// Parsed headers kept at once.
const SHARED_HEADER_ENTRIES: usize = 4;

type SharedHeaderEntry = (
    std::path::PathBuf,
    u64,
    std::time::SystemTime,
    std::sync::Arc<GgufHeader>,
);

/// The parsed header of the GGUF at `path`, shared by every reader in the process. A start reads
/// the architecture, the sampling defaults, the tokenizer and the capability facts from one header
/// instead of parsing it again for each. An entry is keyed by path, length and modification time;
/// `max_header_bytes` bounds a parse that is not served from the cache.
pub(crate) fn shared_header(
    path: &std::path::Path,
    max_header_bytes: u64,
) -> Result<std::sync::Arc<GgufHeader>, EngError> {
    use std::sync::{Arc, Mutex};
    static CACHE: Mutex<Vec<SharedHeaderEntry>> = Mutex::new(Vec::new());
    let unreadable = |error: std::io::Error| {
        EngError::Checkpoint(format!("model file {}: {error}", path.display()))
    };
    let file = std::fs::File::open(path).map_err(unreadable)?;
    let meta = file.metadata().map_err(unreadable)?;
    let (len, modified) = (meta.len(), meta.modified().map_err(unreadable)?);
    let cacheable = len >= SHARED_HEADER_MIN_FILE_BYTES;
    if cacheable
        && let Ok(cache) = CACHE.lock()
        && let Some((_, _, _, header)) = cache
            .iter()
            .find(|(p, l, m, _)| p == path && *l == len && *m == modified)
    {
        return Ok(Arc::clone(header));
    }
    let header = Arc::new(GgufHeader::parse(
        &mut std::io::BufReader::new(file),
        len,
        max_header_bytes,
    )?);
    if cacheable && let Ok(mut cache) = CACHE.lock() {
        cache.retain(|(p, _, _, _)| p != path);
        if cache.len() >= SHARED_HEADER_ENTRIES {
            cache.remove(0);
        }
        cache.push((path.to_path_buf(), len, modified, Arc::clone(&header)));
    }
    Ok(header)
}
