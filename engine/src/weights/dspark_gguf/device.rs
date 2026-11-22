//! CUDA projections and attention with a cache of committed target features.

use std::sync::Arc;

use half::bf16;

use super::{
    DsparkBlock, DsparkConfig, DsparkGguf, DsparkMatrix, MarkovMatrix,
    forward::{DraftAttention, DraftMath, KvRows, Matrix},
};
use crate::{
    EngError,
    kernels::cuda::{
        CHUNK_LIMIT, CudaContext, DeviceBuffer, DeviceStream,
        resident::{
            dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape, dense_gemv_bytes},
            draft_attention_bf16::{CommittedKv, DraftAttentionCall, DraftKv},
        },
    },
};

struct Projection {
    dtype: DenseDtype,
    rows: usize,
    cols: usize,
    weights: DeviceBuffer,
}

struct DeviceKv {
    keys: DeviceBuffer,
    values: DeviceBuffer,
}

impl DeviceKv {
    fn new(cuda: &CudaContext, bytes: usize) -> Result<Self, EngError> {
        Ok(Self {
            keys: cuda.allocate(bytes)?,
            values: cuda.allocate(bytes)?,
        })
    }

    fn ranges(&self) -> Result<DraftKv<'_>, EngError> {
        Ok(DraftKv {
            keys: self.keys.range(0, self.keys.bytes())?,
            values: self.values.range(0, self.values.bytes())?,
        })
    }

    fn write(
        &self,
        stream: &DeviceStream,
        slot: usize,
        row_bytes: usize,
        rows: &KvRows,
        source: std::ops::Range<usize>,
    ) -> Result<(), EngError> {
        if !source.is_empty() {
            let offset = slot * row_bytes;
            write_bf16(&self.keys, stream, offset, &rows.keys[source.clone()])?;
            write_bf16(&self.values, stream, offset, &rows.values[source])?;
        }
        Ok(())
    }
}

struct Layer {
    weights: Vec<Projection>,
    cache: DeviceKv,
    capacity: usize,
    through: usize,
}

struct Scratch {
    x: DeviceBuffer,
    y: DeviceBuffer,
    query: DeviceBuffer,
    block: DeviceKv,
    attended: DeviceBuffer,
    tile_rows: usize,
}

struct DevicePlan {
    capacities: Vec<usize>,
    q_bytes: usize,
    kv_row_bytes: usize,
    block_bytes: usize,
    projection_bytes: usize,
    tile_rows: usize,
    bytes: usize,
}

impl DevicePlan {
    fn new(model: &DsparkGguf, context: usize, markov: bool) -> Result<Self, EngError> {
        let config = model.config();
        if config.head_width != 128 || config.query_heads > 64 || config.block_size > 65_535 {
            return Err(EngError::BadRequest(
                "CUDA DSpark requires 128-wide heads and supported attention geometry".into(),
            ));
        }
        let context = context
            .saturating_sub(1)
            .min(config.context_length - config.block_size)
            .max(1);
        if context >= 1 << 30 {
            return Err(EngError::MemoryPlan(
                "CUDA DSpark context is too large".into(),
            ));
        }
        let capacities: Vec<_> = config
            .local_layers
            .iter()
            .map(|&local| {
                if local {
                    context.min(config.sliding_window)
                } else {
                    context
                }
            })
            .collect();
        let kv_row_bytes = 2 * config.kv_heads * config.head_width;
        let q_bytes = 2 * config.query_heads * config.head_width * config.block_size;
        let block_bytes = kv_row_bytes * config.block_size;
        let mut width = 0;
        let mut weight_bytes = 0u128;
        let layers = model
            .layers
            .iter()
            .flat_map(|layer| Matrix::ALL.into_iter().map(|slot| slot.weight(layer)));
        let recurrent = markov.then(|| [&model.markov.joint, &model.markov.output]);
        for weights in layers.chain(recurrent.into_iter().flatten()) {
            let cols = weights.inputs;
            let rows = weights.bytes.len() / weights.row_bytes;
            dense_gemv_bytes(DenseGemvShape::packed(1, rows, cols), dtype(weights)?)?;
            width = width.max(cols).max(rows);
            weight_bytes += weights.bytes.len() as u128;
        }
        // Batch the first prompt's committed rows too. Keep each transfer
        // within the shared copy chunk limit without capping prompt length.
        let row_bytes = size_of::<f32>() * width;
        let tile_rows = config
            .block_size
            .max(context)
            .min(DenseGemvShape::MAX_ACTIVATION_ROWS)
            .min(CHUNK_LIMIT / row_bytes);
        let projection_bytes = tile_rows * row_bytes;
        let cache_rows: u128 = capacities.iter().map(|&n| n as u128).sum();
        let bytes = weight_bytes
            + 2 * (cache_rows * kv_row_bytes as u128
                + (q_bytes + block_bytes + projection_bytes) as u128);
        let bytes = usize::try_from(bytes)
            .map_err(|_| EngError::MemoryPlan("CUDA DSpark allocation size overflows".into()))?;
        Ok(Self {
            capacities,
            q_bytes,
            kv_row_bytes,
            block_bytes,
            projection_bytes,
            tile_rows,
            bytes,
        })
    }
}

pub(crate) struct DeviceDraftOptions {
    pub(crate) context: usize,
    pub(crate) markov: bool,
    pub(crate) reserve_floor: fn(u64) -> u64,
}

pub(crate) struct DeviceDraft {
    layers: Vec<Layer>,
    markov: Option<[Projection; 2]>,
    scratch: Scratch,
    stream: DeviceStream,
    lineage: Option<Arc<()>>,
}

impl DeviceDraft {
    pub(crate) fn load(
        cuda: &CudaContext,
        model: &DsparkGguf,
        DeviceDraftOptions {
            context,
            markov,
            reserve_floor,
        }: DeviceDraftOptions,
    ) -> Result<Self, EngError> {
        let plan = DevicePlan::new(model, context, markov)?;
        let before = cuda.memory_info()?.free as u64;
        let available = before.saturating_sub(reserve_floor(plan.bytes as u64));
        if before < reserve_floor(0) || plan.bytes as u64 > available {
            return Err(EngError::MemoryPlan(format!(
                "CUDA DSpark needs {} device bytes after target placement and headroom, {available} available",
                plan.bytes
            )));
        }
        let layers = model
            .layers
            .iter()
            .zip(plan.capacities)
            .map(|(layer, capacity)| {
                let weights = Matrix::ALL
                    .iter()
                    .map(|&slot| upload(cuda, slot.weight(layer)))
                    .collect::<Result<_, _>>()?;
                Ok(Layer {
                    weights,
                    cache: DeviceKv::new(cuda, capacity * plan.kv_row_bytes)?,
                    capacity,
                    through: 0,
                })
            })
            .collect::<Result<_, EngError>>()?;
        let markov = markov
            .then(|| -> Result<_, EngError> {
                Ok([
                    upload(cuda, &model.markov.joint)?,
                    upload(cuda, &model.markov.output)?,
                ])
            })
            .transpose()?;
        let scratch = Scratch {
            x: cuda.allocate(plan.projection_bytes)?,
            y: cuda.allocate(plan.projection_bytes)?,
            query: cuda.allocate(plan.q_bytes)?,
            block: DeviceKv::new(cuda, plan.block_bytes)?,
            attended: cuda.allocate(plan.q_bytes)?,
            tile_rows: plan.tile_rows,
        };
        let stream = cuda.stream()?;
        let free = cuda.memory_info()?.free as u64;
        let charged = before.saturating_sub(free);
        if free < reserve_floor(charged) {
            return Err(EngError::MemoryPlan(format!(
                "CUDA DSpark has {free} free device bytes after allocation, below the state reserve floor",
            )));
        }
        eprintln!("Inkling CUDA DSpark draft resident bytes {}", plan.bytes);
        eprintln!("Inkling CUDA DSpark draft reserve charged {charged} bytes, free {free}");
        if markov.is_some() {
            eprintln!("Inkling CUDA DSpark Markov projections enabled");
        }
        Ok(Self {
            layers,
            markov,
            scratch,
            stream,
            lineage: None,
        })
    }

    pub(crate) fn project_markov(
        &self,
        cuda: &CudaContext,
        kind: MarkovMatrix,
        input: &[f32],
    ) -> Result<Option<Vec<f32>>, EngError> {
        let Some([joint, output]) = &self.markov else {
            return Ok(None);
        };
        let weights = match kind {
            MarkovMatrix::Joint => joint,
            MarkovMatrix::Output => output,
        };
        if input.len() != weights.cols {
            return Err(EngError::BadRequest(
                "invalid DSpark Markov projection width".into(),
            ));
        }
        let mut rows = project(cuda, &self.stream, &self.scratch, weights, &[input])?;
        Ok(Some(rows.pop().expect("one DSpark projection row")))
    }

    pub(crate) fn forward(
        &mut self,
        cuda: &CudaContext,
        model: &DsparkGguf,
        embedding: &[f32],
        features: &[Arc<[f32]>],
        lineage: &Arc<()>,
    ) -> Result<DsparkBlock, EngError> {
        if self
            .lineage
            .as_ref()
            .is_none_or(|old| !Arc::ptr_eq(old, lineage))
            || self
                .layers
                .iter()
                .any(|layer| layer.through > features.len())
        {
            self.invalidate();
            self.lineage = Some(Arc::clone(lineage));
        }
        let result = model.forward_with(
            embedding,
            features,
            features.len(),
            &mut DeviceMath { cuda, draft: self },
        );
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    fn invalidate(&mut self) {
        for layer in &mut self.layers {
            layer.through = 0;
        }
    }
}

struct DeviceMath<'a> {
    cuda: &'a CudaContext,
    draft: &'a mut DeviceDraft,
}

impl DraftMath for DeviceMath<'_> {
    fn project(
        &mut self,
        layer: usize,
        matrix: Matrix,
        _: &DsparkMatrix,
        rows: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let projection = &self.draft.layers[layer].weights[matrix as usize];
        let scratch = &self.draft.scratch;
        let mut output = Vec::with_capacity(rows.len());
        for tile in rows.chunks(scratch.tile_rows) {
            output.extend(project(
                self.cuda,
                &self.draft.stream,
                scratch,
                projection,
                tile,
            )?);
        }
        Ok(output)
    }

    fn cached_until(&self, layer: usize, start: usize, end: usize) -> usize {
        self.draft.layers[layer].through.clamp(start, end)
    }

    fn attention(
        &mut self,
        layer: usize,
        input: DraftAttention<'_>,
        config: &DsparkConfig,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let stream = &self.draft.stream;
        let scratch = &self.draft.scratch;
        let layer = &mut self.draft.layers[layer];
        let len = input.end - input.start;
        if len > layer.capacity {
            return Err(EngError::BadRequest(
                "committed context exceeds the CUDA DSpark KV capacity".into(),
            ));
        }
        let row_bytes = 2 * config.kv_heads * config.head_width;
        let slot = input.new_start % layer.capacity;
        let count = input.end - input.new_start;
        let first_write = count.min(layer.capacity - slot);
        layer
            .cache
            .write(stream, slot, row_bytes, &input.committed, 0..first_write)?;
        layer
            .cache
            .write(stream, 0, row_bytes, &input.committed, first_write..count)?;
        write_bf16(&scratch.query, stream, 0, input.query)?;
        scratch
            .block
            .write(stream, 0, row_bytes, &input.draft, 0..input.query.len())?;
        let query_width = config.query_heads * config.head_width;
        let bytes = input.query.len() * query_width * 2;
        self.cuda.draft_attention_bf16(
            stream,
            DraftAttentionCall {
                query_heads: config.query_heads,
                kv_heads: config.kv_heads,
                rows: input.query.len(),
                query: scratch.query.range(0, bytes)?,
                draft: scratch.block.ranges()?,
                committed: CommittedKv {
                    kv: layer.cache.ranges()?,
                    capacity: layer.capacity,
                    first: input.start % layer.capacity,
                    len,
                },
                output: scratch.attended.range(0, bytes)?,
            },
        )?;
        let mut result = vec![0; bytes];
        scratch.attended.read_at(stream, 0, &mut result)?;
        layer.through = input.end;
        Ok(result
            .chunks_exact(query_width * 2)
            .map(|row| {
                row.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&x| bf16::from_bits(u16::from_le_bytes(x)).to_f32())
                    .collect()
            })
            .collect())
    }
}

fn project(
    cuda: &CudaContext,
    stream: &DeviceStream,
    scratch: &Scratch,
    projection: &Projection,
    rows: &[impl AsRef<[f32]>],
) -> Result<Vec<Vec<f32>>, EngError> {
    let shape = DenseGemvShape::packed(rows.len(), projection.rows, projection.cols);
    let input: Vec<_> = rows
        .iter()
        .flat_map(AsRef::as_ref)
        .flat_map(|x| x.to_le_bytes())
        .collect();
    scratch.x.write_at_enqueue(stream, 0, &input)?;
    let mut output = vec![0; rows.len() * shape.rows * 4];
    cuda.dense_gemv(
        stream,
        DenseGemvCall {
            shape,
            dtype: projection.dtype,
            w: projection.weights.range(0, projection.weights.bytes())?,
            x: scratch.x.range(0, input.len())?,
            y: scratch.y.range(0, output.len())?,
        },
    )?;
    scratch.y.read_at(stream, 0, &mut output)?;
    Ok(output
        .chunks_exact(shape.rows * 4)
        .map(|row| {
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|&x| f32::from_le_bytes(x))
                .collect()
        })
        .collect())
}

fn write_bf16(
    buffer: &DeviceBuffer,
    stream: &DeviceStream,
    offset: usize,
    rows: &[Vec<f32>],
) -> Result<(), EngError> {
    let bytes: Vec<_> = rows
        .iter()
        .flatten()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
        .collect();
    buffer.write_at_enqueue(stream, offset, &bytes)
}

fn dtype(weights: &DsparkMatrix) -> Result<DenseDtype, EngError> {
    DenseDtype::from_gguf(weights.ty)
        .ok_or_else(|| EngError::Checkpoint("unsupported CUDA DSpark matrix dtype".into()))
}

fn upload(cuda: &CudaContext, weights: &DsparkMatrix) -> Result<Projection, EngError> {
    let buffer = cuda.allocate(weights.bytes.len())?;
    for (index, chunk) in weights.bytes.chunks(CHUNK_LIMIT).enumerate() {
        cuda.upload_at(&buffer, index * CHUNK_LIMIT, chunk)?;
    }
    Ok(Projection {
        dtype: dtype(weights)?,
        rows: weights.bytes.len() / weights.row_bytes,
        cols: weights.inputs,
        weights: buffer,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    #[ignore = "requires CUDA and the published Inkling DSpark GGUF"]
    fn published_sidecar_keeps_only_committed_kv_and_invalidates_on_restore() {
        let path = std::env::var_os("FIFI_TEST_INKLING_DSPARK_GGUF")
            .expect("set FIFI_TEST_INKLING_DSPARK_GGUF to the published Q8_0 or BF16 sidecar");
        let mut model = DsparkGguf::open(Path::new(&path)).unwrap();
        // Exercise wraparound without rebuilding a thousand committed rows.
        model.config.sliding_window = 2;
        let config = model.config();
        let probe = |seed, i| (((i + seed) % 19) as f32 - 9.0) / 19.0;
        let anchor: Vec<_> = (0..config.width).map(|i| probe(0, i)).collect();
        let taps = |seed| -> Arc<[f32]> {
            (0..config.width * config.target_layers.len())
                .map(|i| probe(seed, i))
                .collect()
        };
        let cuda = CudaContext::new().unwrap();
        let mut draft = DeviceDraft::load(
            &cuda,
            &model,
            DeviceDraftOptions {
                context: 128,
                markov: false,
                reserve_floor: |_| crate::expert_tier::VRAM_HEADROOM_BYTES,
            },
        )
        .unwrap();
        let lineage = Arc::new(());
        let mut features = vec![taps(0)];
        draft
            .forward(&cuda, &model, &anchor, &features, &lineage)
            .unwrap();

        features.push(taps(5));
        draft
            .forward(&cuda, &model, &anchor, &features, &lineage)
            .unwrap();
        features.push(taps(7));
        let warm = draft
            .forward(&cuda, &model, &anchor, &features, &lineage)
            .unwrap();
        let before = draft.stream.enqueued();
        let cached = draft
            .forward(&cuda, &model, &anchor, &features, &lineage)
            .unwrap();
        let cached_work = draft.stream.enqueued() - before;
        assert_eq!(warm.hidden, cached.hidden);
        assert_eq!(warm.recurrent_context, cached.recurrent_context);
        assert!(
            draft
                .layers
                .iter()
                .all(|layer| layer.through == features.len())
        );

        let uncached_lineage = Arc::new(());
        let before = draft.stream.enqueued();
        let uncached = draft
            .forward(&cuda, &model, &anchor, &features, &uncached_lineage)
            .unwrap();
        assert_eq!(warm.hidden, uncached.hidden);
        assert!(cached_work < draft.stream.enqueued() - before);

        features[1] = taps(11);
        let restored = draft
            .forward(&cuda, &model, &anchor, &features, &Arc::new(()))
            .unwrap();
        let reference = model.forward(&anchor, &features, features.len()).unwrap();
        assert_ne!(restored.hidden, warm.hidden);
        assert_eq!(restored.recurrent_context, reference.recurrent_context);
        let squared_error: f64 = restored
            .hidden
            .iter()
            .flatten()
            .zip(reference.hidden.iter().flatten())
            .map(|(&actual, &expected)| f64::from(actual - expected).powi(2))
            .sum();
        let squared_signal: f64 = reference
            .hidden
            .iter()
            .flatten()
            .map(|&x| f64::from(x).powi(2))
            .sum();
        let relative_rms = (squared_error / squared_signal).sqrt();
        eprintln!("CUDA DSpark draft restored block relative RMS error {relative_rms}");
        assert!(relative_rms < 0.025);
    }
}
