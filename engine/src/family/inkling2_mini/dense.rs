//! Text weight access for the official Inkling 2 Mini safetensor layout.

use std::{cell::OnceCell, collections::BTreeMap, fs::File, ops::Range, path::Path, sync::Arc};

use half::bf16;

const PARALLEL_BF16_ROWS: usize = 100_000;
const MAX_BF16_WORKERS: usize = 4;
const ATTN_WORKERS_ENV: &str = "FIFI_INKLING_ATTN_CPU_WORKERS";
const QKV_WEIGHTS: [&str; 3] = ["wq_du.weight", "wk_dv.weight", "wv_dv.weight"];
use safetensors::{Dtype, tensor::TensorInfo};

use super::timing::DenseIo;
use super::{
    CORE_START, HEADER_LIMIT, SHARD, checkpoint_error, gguf, gguf_dense, parse_json, read_exact_at,
    read_stored_header, validate_config, weight_plan,
};
use crate::{EngError, family::mapped_file::MappedFile, storage_guard::LocalGuard};

pub struct DenseWeights(Backend);

enum Backend {
    Safetensors(SafetensorDense),
    Gguf(gguf_dense::Dense),
}

impl DenseWeights {
    pub fn open(dir: &Path) -> Result<Self, EngError> {
        Ok(Self(Backend::Safetensors(SafetensorDense::open(dir)?)))
    }

    pub fn from_gguf(source: Arc<gguf::Weights>) -> Result<Self, EngError> {
        Self::from_gguf_with_mapping(source, 0, 1).map(|(dense, _)| dense)
    }

    pub(super) fn from_gguf_with_mapping(
        source: Arc<gguf::Weights>,
        mapping_budget: u64,
        attention_workers: usize,
    ) -> Result<(Self, u64), EngError> {
        let (dense, mapped_bytes) =
            gguf_dense::Dense::new(source, mapping_budget, attention_workers)?;
        Ok((Self(Backend::Gguf(dense)), mapped_bytes))
    }

    #[cfg(all(
        test,
        feature = "native-mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub(super) fn has_mapped_gguf_projection(&self) -> bool {
        match &self.0 {
            Backend::Gguf(dense) => dense.has_mapping(),
            Backend::Safetensors(_) => false,
        }
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn io_snapshot(&self) -> super::timing::DenseIoStats {
        match &self.0 {
            Backend::Safetensors(w) => w.io.as_ref().map(DenseIo::snapshot).unwrap_or_default(),
            Backend::Gguf(w) => w.io_snapshot(),
        }
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], EngError> {
        match &self.0 {
            Backend::Safetensors(w) => w.shape(name),
            Backend::Gguf(w) => w.shape(name),
        }
    }

    pub fn vector(&self, name: &str) -> Result<Vec<f32>, EngError> {
        match &self.0 {
            Backend::Safetensors(w) => w.vector(name),
            Backend::Gguf(w) => w.vector(name),
        }
    }

    pub fn bf16(&self, name: &str) -> Result<Vec<bf16>, EngError> {
        match &self.0 {
            Backend::Safetensors(w) => w.bf16(name),
            Backend::Gguf(w) => Ok(w.vector(name)?.into_iter().map(bf16::from_f32).collect()),
        }
    }

    pub fn row(&self, name: &str, row: usize) -> Result<Vec<f32>, EngError> {
        match &self.0 {
            Backend::Safetensors(w) => w.row(name, row),
            Backend::Gguf(w) => w.row(name, row),
        }
    }

    pub fn matvec(&self, name: &str, input: &[f32]) -> Result<Vec<f32>, EngError> {
        match &self.0 {
            Backend::Safetensors(w) => w.matvec(name, input),
            Backend::Gguf(w) => w.matvec_rows(name, input, 0..w.shape(name)?[0]),
        }
    }

    pub fn matvec_rows(
        &self,
        name: &str,
        input: &[f32],
        rows: Range<usize>,
    ) -> Result<Vec<f32>, EngError> {
        match &self.0 {
            Backend::Safetensors(w) => w.matvec_rows(name, input, rows),
            Backend::Gguf(w) => w.matvec_rows(name, input, rows),
        }
    }

    pub fn loop_query(&self, layer: usize, stream: usize) -> Result<Vec<f32>, EngError> {
        match &self.0 {
            Backend::Safetensors(w) => {
                let (suffix, offset) = match stream {
                    0 => ("", 0),
                    1 => ("_kv", 0),
                    2 => ("_kv", 1),
                    3 => ("", 1),
                    _ => return Err(checkpoint_error("invalid Inkling loop query stream")),
                };
                w.row(
                    &format!("model.llm.middle_cycle_cross_pass{suffix}_queries"),
                    2 * (layer - CORE_START) + offset,
                )
            }
            Backend::Gguf(w) => w.loop_query(layer, stream),
        }
    }
}

enum DenseShard {
    Buffered(File),
    Mapped(MappedFile),
}

struct SafetensorDense {
    shard: DenseShard,
    len: u64,
    data_start: u64,
    entries: BTreeMap<String, DenseEntry>,
    attention_workers: usize,
    io: Option<DenseIo>,
}

struct DenseEntry {
    info: TensorInfo,
    bytes: OnceCell<Box<[u8]>>,
}

impl SafetensorDense {
    /// Validates the official checkpoint metadata. Text matrices are loaded on first
    /// projection and retain their stored precision. Embedding lookup reads one row.
    pub fn open(dir: &Path) -> Result<Self, EngError> {
        let guard = LocalGuard::new()?;
        let config = dir.join("config.json");
        let index = dir.join("model.safetensors.index.json");
        let shard = dir.join(SHARD);
        for path in [&config, &index, &shard] {
            guard.check(path, "model weights")?;
        }
        validate_config(&std::fs::read(config).map_err(checkpoint_error)?)?;
        let index = parse_json(&std::fs::read(index).map_err(checkpoint_error)?)?;
        let mut shard = File::open(shard).map_err(checkpoint_error)?;
        let metadata = shard.metadata().map_err(checkpoint_error)?;
        let len = metadata.len();
        let (data_start, entries) = read_stored_header(&mut shard, len, HEADER_LIMIT)?;
        let entries: BTreeMap<_, _> = entries.into_iter().collect();
        weight_plan(&index, &entries)?;
        let entries = entries
            .into_iter()
            .filter(|(name, info)| {
                name.starts_with("model.llm.")
                    && !name.contains(".experts.")
                    && matches!(info.dtype, Dtype::BF16 | Dtype::F32)
            })
            .map(|(name, info)| {
                (
                    name,
                    DenseEntry {
                        info,
                        bytes: OnceCell::new(),
                    },
                )
            })
            .collect();
        let shard = if std::env::var_os("FIFI_INKLING_MMAP_DENSE").is_some_and(|value| value == "1")
        {
            let modified = metadata.modified().map_err(checkpoint_error)?;
            DenseShard::Mapped(
                MappedFile::from_file(shard, len, modified).map_err(checkpoint_error)?,
            )
        } else {
            DenseShard::Buffered(shard)
        };
        let attention_workers = attention_cpu_workers();
        Ok(Self {
            shard,
            len,
            data_start,
            entries,
            attention_workers,
            io: DenseIo::start_if_enabled(),
        })
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], EngError> {
        Ok(&self.entry(name)?.info.shape)
    }

    /// Reads flattened BF16 or F32 weights as F32, including small tensors of rank above one.
    pub fn vector(&self, name: &str) -> Result<Vec<f32>, EngError> {
        let entry = self.entry(name)?;
        self.with_bytes(name, entry, full_range(entry), true, |bytes| {
            widen(entry.info.dtype, bytes)
        })
    }

    /// Reads flattened BF16 weights at their stored precision.
    pub fn bf16(&self, name: &str) -> Result<Vec<bf16>, EngError> {
        let entry = self.entry(name)?;
        if entry.info.dtype != Dtype::BF16 {
            return Err(checkpoint_error(format!(
                "dense weight {name} must be BF16"
            )));
        }
        self.with_bytes(name, entry, full_range(entry), true, |bytes| {
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&word| bf16::from_bits(u16::from_le_bytes(word)))
                .collect()
        })
    }

    pub fn row(&self, name: &str, row: usize) -> Result<Vec<f32>, EngError> {
        let entry = self.entry(name)?;
        let rows = self.dimensions(name, entry)?.0;
        if row >= rows {
            return Err(EngError::BadRequest(format!(
                "Inkling dense row {row} exceeds {name}"
            )));
        }
        let bytes = row_range(&entry.info, row..row + 1);
        self.with_bytes(name, entry, bytes, false, |bytes| {
            widen(entry.info.dtype, bytes)
        })
    }

    pub fn matvec(&self, name: &str, input: &[f32]) -> Result<Vec<f32>, EngError> {
        let entry = self.entry(name)?;
        let (rows, _) = self.dimensions(name, entry)?;
        self.matvec_rows(name, input, 0..rows)
    }

    /// Projects only the requested output rows, for example the unpadded output vocabulary.
    pub fn matvec_rows(
        &self,
        name: &str,
        input: &[f32],
        rows: Range<usize>,
    ) -> Result<Vec<f32>, EngError> {
        let entry = self.entry(name)?;
        let (height, width) = self.dimensions(name, entry)?;
        if input.len() != width || rows.is_empty() || rows.end > height {
            return Err(EngError::BadRequest(format!(
                "Inkling dense projection {name} expects input width {width} and rows within {height}"
            )));
        }
        let row_size = width * (entry.info.dtype.bitsize() / 8);
        let selected = row_range(&entry.info, rows);
        self.with_bytes(name, entry, selected, true, |resident| {
            match entry.info.dtype {
                Dtype::BF16 => {
                    let workers = if resident.len() / row_size >= PARALLEL_BF16_ROWS {
                        bf16_worker_limit()
                    } else if is_attention_qkv(name) {
                        self.attention_workers
                    } else {
                        1
                    };
                    bf16_rows(resident, input, workers)
                }
                Dtype::F32 => resident
                    .chunks_exact(row_size)
                    .map(|row| dot_f32_le(row, input))
                    .collect(),
                _ => unreachable!("only BF16 and F32 dense tensors are loaded"),
            }
        })
    }

    fn entry(&self, name: &str) -> Result<&DenseEntry, EngError> {
        self.entries
            .get(name)
            .ok_or_else(|| checkpoint_error(format!("missing dense text weight {name}")))
    }

    fn dimensions(&self, name: &str, entry: &DenseEntry) -> Result<(usize, usize), EngError> {
        match entry.info.shape.as_slice() {
            &[rows, cols] if rows != 0 && cols != 0 => Ok((rows, cols)),
            _ => Err(checkpoint_error(format!(
                "dense weight {name} must be a nonempty matrix"
            ))),
        }
    }

    fn with_bytes<T>(
        &self,
        name: &str,
        entry: &DenseEntry,
        range: Range<usize>,
        retain: bool,
        use_bytes: impl FnOnce(&[u8]) -> T,
    ) -> Result<T, EngError> {
        match &self.shard {
            DenseShard::Mapped(shard) => {
                shard.check_unchanged().map_err(checkpoint_error)?;
                let invalid = || {
                    checkpoint_error(format!("dense weight {name} mapped range is out of bounds"))
                };
                if range.end > full_range(entry).end {
                    return Err(invalid());
                }
                let tensor = usize::try_from(self.data_start)
                    .ok()
                    .and_then(|start| start.checked_add(entry.info.data_offsets.0))
                    .ok_or_else(invalid)?;
                let start = tensor.checked_add(range.start).ok_or_else(invalid)?;
                let end = tensor.checked_add(range.end).ok_or_else(invalid)?;
                let bytes = shard.as_ref().get(start..end).ok_or_else(invalid)?;
                let result = use_bytes(bytes);
                if let Some(io) = &self.io {
                    io.mapped_access(bytes.len());
                }
                shard.check_unchanged().map_err(checkpoint_error)?;
                Ok(result)
            }
            DenseShard::Buffered(shard) => {
                if let Some(bytes) = entry.bytes.get() {
                    return Ok(use_bytes(&bytes[range]));
                }
                if retain {
                    let bytes = self.read(shard, name, entry, full_range(entry))?;
                    let bytes = entry.bytes.get_or_init(|| bytes);
                    Ok(use_bytes(&bytes[range]))
                } else {
                    let bytes = self.read(shard, name, entry, range)?;
                    Ok(use_bytes(&bytes))
                }
            }
        }
    }

    fn read(
        &self,
        shard: &File,
        name: &str,
        entry: &DenseEntry,
        range: Range<usize>,
    ) -> Result<Box<[u8]>, EngError> {
        let check_len = || {
            let observed = shard.metadata().map_err(checkpoint_error)?.len();
            if observed != self.len {
                return Err(checkpoint_error(format!(
                    "shard length changed from {} to {observed}",
                    self.len
                )));
            }
            Ok(())
        };
        check_len()?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(range.len()).map_err(|_| {
            EngError::MemoryPlan(format!(
                "Inkling dense weight {name} does not fit in host memory"
            ))
        })?;
        bytes.resize(range.len(), 0);
        let offset = self.data_start + entry.info.data_offsets.0 as u64 + range.start as u64;
        read_exact_at(shard, &mut bytes, offset).map_err(|error| {
            checkpoint_error(format!("dense weight {name} read at {offset}: {error}"))
        })?;
        if let Some(io) = &self.io {
            io.file_read(bytes.len());
        }
        check_len()?;
        Ok(bytes.into_boxed_slice())
    }
}

pub(super) fn attention_cpu_workers() -> usize {
    if std::env::var_os(ATTN_WORKERS_ENV).is_some_and(|value| value == "4") {
        bf16_worker_limit()
    } else {
        1
    }
}

fn bf16_worker_limit() -> usize {
    std::thread::available_parallelism().map_or(1, |count| count.get().min(MAX_BF16_WORKERS))
}

fn is_attention_qkv(name: &str) -> bool {
    name.starts_with("model.llm.layers.")
        && name
            .rsplit_once(".attn.")
            .is_some_and(|(_, projection)| QKV_WEIGHTS.contains(&projection))
}

fn bf16_rows(weights: &[u8], input: &[f32], workers: usize) -> Vec<f32> {
    if workers == 1 {
        return weights
            .chunks_exact(input.len() * 2)
            .map(|row| dot_bf16(row, input))
            .collect();
    }
    parallel_bf16_rows(weights, input, workers)
}

// Split the resident BF16 rows without repacking or replicating the output matrix.
fn parallel_bf16_rows(weights: &[u8], input: &[f32], workers: usize) -> Vec<f32> {
    let row_bytes = input.len() * 2;
    let rows = weights.len() / row_bytes;
    let mut logits = vec![0.; rows];
    let rows_per_worker = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        for (out, packed) in logits
            .chunks_mut(rows_per_worker)
            .zip(weights.chunks(rows_per_worker * row_bytes))
        {
            scope.spawn(move || {
                for (logit, row) in out.iter_mut().zip(packed.chunks_exact(row_bytes)) {
                    *logit = dot_bf16(row, input);
                }
            });
        }
    });
    logits
}

fn full_range(entry: &DenseEntry) -> Range<usize> {
    let (start, end) = entry.info.data_offsets;
    0..end - start
}

fn row_range(info: &TensorInfo, rows: Range<usize>) -> Range<usize> {
    let row_size = info.shape[1] * (info.dtype.bitsize() / 8);
    rows.start * row_size..rows.end * row_size
}

fn widen(dtype: Dtype, bytes: &[u8]) -> Vec<f32> {
    match dtype {
        Dtype::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&word| widen_bf16(word))
            .collect(),
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| f32::from_le_bytes(word))
            .collect(),
        _ => unreachable!("only BF16 and F32 dense tensors are loaded"),
    }
}

pub(super) fn dot_f32_le(weights: &[u8], input: &[f32]) -> f32 {
    weights
        .as_chunks::<4>()
        .0
        .iter()
        .zip(input)
        .map(|(&word, &value)| f32::from_le_bytes(word) * value)
        .sum()
}

// GGUF's reference reduces in scalar order, including BF16 rows.
pub(super) fn dot_bf16_le_scalar(weights: &[u8], input: &[f32]) -> f32 {
    weights
        .as_chunks::<2>()
        .0
        .iter()
        .zip(input)
        .map(|(&word, &value)| widen_bf16(word) * value)
        .sum()
}

fn widen_bf16(word: [u8; 2]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes(word)) << 16)
}

#[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
fn dot_bf16(weights: &[u8], input: &[f32]) -> f32 {
    dot_bf16_le_scalar(weights, input)
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[allow(unsafe_code)]
fn dot_bf16(weights: &[u8], input: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    // SAFETY: NEON and floating-point SIMD are part of this target's ABI.
    // Each load reads a complete chunk. Byte loads also support unaligned weights.
    unsafe {
        let mut even = vdupq_n_f32(0.0);
        let mut odd = vdupq_n_f32(0.0);
        let (weights, weight_tail) = weights.as_chunks::<16>();
        let (input, input_tail) = input.as_chunks::<8>();
        for (words, values) in weights.iter().zip(input) {
            let packed = vreinterpretq_u16_u8(vld1q_u8(words.as_ptr()));
            let low = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_low_u16(packed))));
            let high = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_high_u16(packed))));
            even = vfmaq_f32(even, low, vld1q_f32(values.as_ptr()));
            odd = vfmaq_f32(odd, high, vld1q_f32(values.as_ptr().add(4)));
        }
        let mut sum = vaddvq_f32(vaddq_f32(even, odd));
        for (&word, &value) in weight_tail.as_chunks::<2>().0.iter().zip(input_tail) {
            sum += widen_bf16(word) * value;
        }
        sum
    }
}

#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod projection_bench;

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn parallel_bf16_rows_preserve_order_and_serial_dot_bits_with_a_short_final_chunk() {
        let input = [1.25, -0.25, 2.0, 0.75, -0.5, 3.0, -2.0, 0.5, -1.0];
        let weights: Vec<_> = (0..7 * input.len())
            .flat_map(|i| {
                bf16::from_f32((i as f32 - 30.0) / 16.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let serial: Vec<_> = weights
            .chunks_exact(input.len() * 2)
            .map(|row| dot_bf16(row, &input).to_bits())
            .collect();
        let parallel: Vec<_> = parallel_bf16_rows(&weights, &input, 4)
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(serial, parallel);
    }

    #[test]
    fn only_publisher_attention_qkv_projections_use_the_attention_worker_setting() {
        for projection in QKV_WEIGHTS {
            assert!(is_attention_qkv(&format!(
                "model.llm.layers.5.attn.{projection}"
            )));
        }
        for name in [
            "model.llm.layers.5.attn.wo_ud.weight",
            "model.llm.layers.5.mlp.gate.weight",
            "model.llm.unembed.weight",
            "blk.5.attn_q.weight",
        ] {
            assert!(!is_attention_qkv(name), "{name}");
        }
    }

    #[test]
    fn dense_row_reads_and_bf16_projection_keep_packed_weights() {
        let mut shard = tempfile::tempfile().unwrap();
        let first = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let second = [2.0, 1.0, 0.0, -1.0, -2.0, 0.0, 1.0, 2.0, 3.0];
        let bytes: Vec<_> = first
            .into_iter()
            .chain(second)
            .flat_map(|value| bf16::from_f32(value).to_bits().to_le_bytes())
            .collect();
        shard.write_all(&[0; 11]).unwrap();
        shard.write_all(&bytes).unwrap();
        let dense = SafetensorDense {
            shard: DenseShard::Buffered(shard),
            len: 11 + bytes.len() as u64,
            data_start: 11,
            attention_workers: 1,
            io: Some(DenseIo::default()),
            entries: BTreeMap::from([(
                "matrix".into(),
                DenseEntry {
                    info: TensorInfo {
                        dtype: Dtype::BF16,
                        shape: vec![2, 9],
                        data_offsets: (0, bytes.len()),
                    },
                    bytes: OnceCell::new(),
                },
            )]),
        };

        assert_eq!(dense.shape("matrix").unwrap(), [2, 9]);
        assert_eq!(dense.row("matrix", 1).unwrap(), second);
        assert!(dense.row("matrix", 2).is_err());
        let io = dense.io.as_ref().unwrap();
        assert_eq!(io.snapshot().file_ranges, 1);
        assert_eq!(io.snapshot().file_bytes, 18);
        assert!(dense.entry("matrix").unwrap().bytes.get().is_none());
        assert_eq!(dense.matvec_rows("matrix", &[1.0; 9], 1..2).unwrap(), [6.0],);
        assert_eq!(dense.matvec("matrix", &[1.0; 9]).unwrap(), [45.0, 6.0]);
        assert!(dense.matvec("matrix", &[1.0; 8]).is_err());
        assert_eq!(io.snapshot().file_ranges, 2);
        assert_eq!(io.snapshot().file_bytes, (18 + bytes.len()) as u64);
        assert_eq!(io.snapshot().mapped_accesses, 0);
        assert_eq!(
            &**dense.entry("matrix").unwrap().bytes.get().unwrap(),
            &bytes
        );
    }
    #[test]
    fn mapped_dense_projects_exact_bytes_and_rejects_a_changed_shard() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let values = [1.0, 2.0, 4.0, -1.0];
        let bytes: Vec<_> = values
            .into_iter()
            .flat_map(|value| bf16::from_f32(value).to_bits().to_le_bytes())
            .collect();
        file.write_all(&[0; 11]).unwrap();
        file.write_all(&bytes).unwrap();
        let shard = File::open(file.path()).unwrap();
        let metadata = shard.metadata().unwrap();
        let len = metadata.len();
        let mapped = MappedFile::from_file(shard, len, metadata.modified().unwrap()).unwrap();
        let dense = SafetensorDense {
            shard: DenseShard::Mapped(mapped),
            len,
            data_start: 11,
            attention_workers: 1,
            io: Some(DenseIo::default()),
            entries: BTreeMap::from([(
                "matrix".into(),
                DenseEntry {
                    info: TensorInfo {
                        dtype: Dtype::BF16,
                        shape: vec![2, 2],
                        data_offsets: (0, bytes.len()),
                    },
                    bytes: OnceCell::new(),
                },
            )]),
        };
        assert_eq!(dense.row("matrix", 1).unwrap(), [4.0, -1.0]);
        assert_eq!(dense.vector("matrix").unwrap(), values);
        assert_eq!(dense.bf16("matrix").unwrap(), values.map(bf16::from_f32));
        assert_eq!(dense.matvec("matrix", &[2.0, 3.0]).unwrap(), [8.0, 5.0]);
        assert_eq!(
            dense.matvec_rows("matrix", &[2.0, 3.0], 1..2).unwrap(),
            [5.0]
        );
        let io = dense.io.as_ref().unwrap();
        let stats = io.snapshot();
        assert_eq!(stats.file_ranges, 0);
        assert_eq!(stats.file_bytes, 0);
        assert_eq!(stats.mapped_accesses, 5);
        assert_eq!(stats.mapped_span_bytes, 32);
        let entry = dense.entry("matrix").unwrap();
        assert!(entry.bytes.get().is_none());
        assert!(
            dense
                .with_bytes("matrix", entry, 0..bytes.len() + 1, false, |_| ())
                .is_err()
        );

        // The output of an operation that observes mutation is discarded.
        assert!(
            dense
                .with_bytes("matrix", entry, 0..2, false, |_| {
                    file.write_all(&[0]).unwrap();
                })
                .is_err()
        );
        file.as_file().set_len(0).unwrap();
        assert!(dense.row("matrix", 0).is_err());
        assert!(dense.matvec("matrix", &[2.0, 3.0]).is_err());
    }
}
