//! GGUF attention with resident, per-visit K/V and reusable staging.
//!
//! Host K/V rows still back prefix snapshots. The device ring is valid only for the
//! current continuation, so restore invalidates its cursor even at the same position.

use std::{cell::RefCell, collections::VecDeque, sync::Arc};

use crate::{
    EngError,
    expert_tier::VRAM_HEADROOM_BYTES,
    family::inkling2::{Attention, LayerVisit, TextSpec, gguf_weights::GgufTextWeights},
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceStream, PinnedBuf,
        resident::relative_attention::{HEAD_DIM, RelativeAttentionCall, RelativeAttentionShape},
    },
};

#[cfg(test)]
use super::DEVICE_HEADROOM;
use super::{super::state_headroom, bad};

#[derive(Clone, Copy)]
pub(in crate::family::inkling2) struct Config {
    pub context: usize,
    pub relative_dims: usize,
    pub relative_extent: usize,
}

#[derive(Clone, Copy)]
struct Visit {
    shape: RelativeAttentionShape,
    table_at: usize,
    table_bytes: usize,
    cache_at: usize,
    cache_bytes: usize,
}

pub(in crate::family::inkling2) struct Plan {
    visits: Vec<Visit>,
    tables: Vec<u8>,
    scratch_bytes: usize,
}

pub(in crate::family::inkling2) struct Row<'a> {
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub relative: &'a [f32],
    pub sinks: &'a [f32],
    pub gates: &'a [f32],
}

pub(in crate::family::inkling2) struct History<'a> {
    pub position: usize,
    pub keys: &'a VecDeque<Arc<Vec<f32>>>,
    pub values: &'a VecDeque<Arc<Vec<f32>>>,
}

pub(in crate::family::inkling2) struct ResidentAttention {
    visits: Vec<Visit>,
    arena: DeviceBuffer,
    scratch_at: usize,
    projection_headroom: u64,
    stream: DeviceStream,
    state: RefCell<Runtime>,
}

struct Runtime {
    transfer: PinnedBuf,
    encoded: Vec<u8>,
    next_position: Vec<usize>,
}

impl Plan {
    pub(in crate::family::inkling2) fn new(
        weights: &GgufTextWeights,
        spec: &TextSpec,
        config: Config,
    ) -> Result<Option<Self>, EngError> {
        let table_words = config
            .relative_dims
            .checked_mul(config.relative_extent)
            .ok_or_else(|| bad("CUDA relative attention table overflows"))?;
        let table_bytes = f32_bytes(table_words)?;
        let mut tables = Vec::new();
        for layer in 0..spec.stored_layers() {
            let table = weights.vec(
                &format!("model.llm.layers.{layer}.attn.rel_logits_proj.proj"),
                table_words,
            )?;
            extend_words(&mut tables, &table);
        }

        let mut visits = Vec::new();
        let mut cache_at = tables.len();
        let mut scratch_bytes = 0;
        for visit in spec.visits() {
            let Some(shape) = shape(spec, config, visit) else {
                return Ok(None);
            };
            let Some(
                [
                    out,
                    cache_bytes,
                    _,
                    status,
                    q,
                    k,
                    v,
                    relative,
                    _,
                    sinks,
                    gates,
                    pos,
                ],
            ) = shape.region_bytes()
            else {
                return Ok(None);
            };
            scratch_bytes =
                scratch_bytes.max(q + k + v + relative + sinks + gates + pos + status + out);
            visits.push(Visit {
                shape,
                table_at: visit.weight_layer * table_bytes,
                table_bytes,
                cache_at,
                cache_bytes,
            });
            cache_at = cache_at
                .checked_add(cache_bytes)
                .and_then(|at| at.checked_add(cache_bytes))
                .ok_or_else(|| bad("CUDA relative attention cache overflows"))?;
        }
        Ok(Some(Self {
            visits,
            tables,
            scratch_bytes,
        }))
    }

    pub(super) fn load(
        self,
        context: &CudaContext,
        charge_reserve: bool,
    ) -> Result<Option<ResidentAttention>, EngError> {
        let last = self
            .visits
            .last()
            .ok_or_else(|| bad("CUDA attention requires at least one visit"))?;
        let scratch_at = last.cache_at + 2 * last.cache_bytes;
        let bytes = scratch_at
            .checked_add(self.scratch_bytes)
            .ok_or_else(|| bad("CUDA relative attention arena overflows"))?;
        let before = context.memory_info()?.free as u64;
        let proposed_credit = if charge_reserve { bytes as u64 } else { 0 };
        let spare = before.saturating_sub(state_headroom(proposed_credit));
        if before < VRAM_HEADROOM_BYTES || bytes as u64 > spare {
            eprintln!(
                "Inkling CUDA relative attention not admitted: needs {bytes} bytes, {spare} available"
            );
            return Ok(None);
        }
        let arena = context.allocate(bytes)?;
        let after = context.memory_info()?.free as u64;
        let device_bytes = before.saturating_sub(after);
        // Only this allocation's requested bytes count against the reserve.
        // CUDA padding still consumes the physical free-memory margin.
        let credit = device_bytes.min(proposed_credit);
        let projection_headroom = state_headroom(credit);
        if after < projection_headroom {
            eprintln!(
                "Inkling CUDA relative attention not admitted after device allocation rounding"
            );
            return Ok(None);
        }
        let stream = context.stream()?;
        arena.write_at(&stream, 0, &self.tables)?;
        let transfer = context.pinned_buf(self.scratch_bytes)?;
        if charge_reserve {
            let free = context.memory_info()?.free as u64;
            if free < projection_headroom {
                eprintln!("Inkling CUDA relative attention not admitted after staging setup");
                return Ok(None);
            }
            eprintln!(
                "Inkling CUDA relative attention reserve: before {before}, allocated {device_bytes}, credited {credit}, free {free}, projection floor {projection_headroom} bytes"
            );
        }
        let next_position = vec![0; self.visits.len()];
        eprintln!(
            "Inkling CUDA relative attention admitted {} visits, {} device bytes",
            self.visits.len(),
            bytes
        );
        Ok(Some(ResidentAttention {
            visits: self.visits,
            arena,
            scratch_at,
            projection_headroom,
            stream,
            state: RefCell::new(Runtime {
                transfer,
                encoded: Vec::with_capacity(self.scratch_bytes),
                next_position,
            }),
        }))
    }
}

fn shape(spec: &TextSpec, config: Config, visit: LayerVisit) -> Option<RelativeAttentionShape> {
    let (query_heads, kv_heads, width, capacity) = match visit.attention {
        Attention::Local => (
            spec.swa_num_attention_heads,
            spec.swa_num_key_value_heads,
            spec.swa_head_dim,
            config.context.min(spec.sliding_window_size as usize),
        ),
        Attention::Global => (
            spec.num_attention_heads,
            spec.num_key_value_heads,
            spec.head_dim,
            config.context,
        ),
    };
    (width == HEAD_DIM as u64).then_some(RelativeAttentionShape {
        query_heads: query_heads as usize,
        kv_heads: kv_heads as usize,
        relative_dims: config.relative_dims,
        relative_extent: config.relative_extent,
        capacity,
    })
}

impl ResidentAttention {
    pub(super) fn projection_headroom(&self) -> u64 {
        self.projection_headroom
    }

    pub(in crate::family::inkling2) fn invalidate(&self) {
        self.state.borrow_mut().next_position.fill(0);
    }

    pub(in crate::family::inkling2) fn step(
        &self,
        context: &CudaContext,
        state_slot: usize,
        row: Row<'_>,
        history: History<'_>,
    ) -> Result<Vec<f32>, EngError> {
        let visit = self.visits[state_slot];
        let mut state = self.state.borrow_mut();
        if state.next_position[state_slot] != history.position {
            self.reseed(visit, &history, &mut state.encoded)?;
        }

        let ordered = row.encode(visit.shape, history.position, &mut state.encoded)?;
        let input_bytes = state.encoded.len();
        let output_bytes = f32_bytes(row.query.len())?;
        let Runtime {
            transfer, encoded, ..
        } = &mut *state;
        transfer.write(0, encoded)?;
        let [q, k, v, relative, sinks, gates, pos, status] =
            ordered.map(|offset| offset + self.scratch_at);
        let out = self.scratch_at + input_bytes;
        let call = RelativeAttentionCall {
            shape: visit.shape,
            q: self.arena.range(q, k - q)?,
            k: self.arena.range(k, v - k)?,
            v: self.arena.range(v, relative - v)?,
            relative: self.arena.range(relative, sinks - relative)?,
            table: self.arena.range(visit.table_at, visit.table_bytes)?,
            sinks: self.arena.range(sinks, gates - sinks)?,
            gates: self.arena.range(gates, pos - gates)?,
            pos: self.arena.range(pos, size_of::<u32>())?,
            status: self.arena.range(status, size_of::<u32>())?,
            out: self.arena.range(out, output_bytes)?,
            k_cache: self.arena.range(visit.cache_at, visit.cache_bytes)?,
            v_cache: self
                .arena
                .range(visit.cache_at + visit.cache_bytes, visit.cache_bytes)?,
        };
        let queued = (|| {
            context.copy_from_pinned(
                &self.stream,
                transfer,
                0,
                self.arena.range(self.scratch_at, input_bytes)?,
            )?;
            context.relative_attention(&self.stream, call)?;
            context.copy_to_pinned(
                &self.stream,
                self.arena.range(status, size_of::<u32>() + output_bytes)?,
                transfer,
                0,
            )
        })();
        // Finish any accepted transfer before shared staging can be used again.
        let complete = transfer.synchronize(&self.stream);
        queued?;
        complete?;
        let mut health = [0; size_of::<u32>()];
        transfer.read(0, &mut health)?;
        if u32::from_ne_bytes(health) != 0 {
            return Err(bad("invalid CUDA relative attention state"));
        }
        let mut result = vec![0.0; row.query.len()];
        transfer.read_f32(size_of::<u32>(), &mut result)?;
        if !result.iter().all(|v| v.is_finite()) {
            return Err(bad("non-finite CUDA relative attention output"));
        }
        state.next_position[state_slot] = history.position + 1;
        Ok(result)
    }

    fn reseed(
        &self,
        visit: Visit,
        history: &History<'_>,
        bytes: &mut Vec<u8>,
    ) -> Result<(), EngError> {
        let capacity = visit.shape.capacity;
        let rows = capacity.min(history.position);
        if rows == 0 {
            return Ok(());
        }
        let row_bytes = visit.cache_bytes / capacity;
        let slot = (history.position - rows) % capacity;
        let splits = ring_splits(slot, rows, capacity);
        for (plane, values) in [history.keys, history.values].into_iter().enumerate() {
            if values.len() < rows {
                return Err(bad("missing host rows for CUDA relative attention restore"));
            }
            bytes.clear();
            for row in values.iter().take(rows).rev() {
                if f32_bytes(row.len())? != row_bytes {
                    return Err(bad("invalid host row for CUDA relative attention restore"));
                }
                extend_words(bytes, row);
            }
            let base = visit.cache_at + plane * visit.cache_bytes;
            let mut read = 0;
            for (at, count) in splits {
                if count == 0 {
                    continue;
                }
                let len = count * row_bytes;
                self.arena.write_at(
                    &self.stream,
                    base + at * row_bytes,
                    &bytes[read..read + len],
                )?;
                read += len;
            }
        }
        Ok(())
    }
}

impl Row<'_> {
    fn encode(
        &self,
        shape: RelativeAttentionShape,
        position: usize,
        bytes: &mut Vec<u8>,
    ) -> Result<[usize; 8], EngError> {
        let slices = [
            self.query,
            self.key,
            self.value,
            self.relative,
            self.sinks,
            self.gates,
        ];
        let q = shape.query_heads * HEAD_DIM;
        let kv = shape.kv_heads * HEAD_DIM;
        let sizes = [
            q,
            kv,
            kv,
            shape.query_heads * shape.relative_dims,
            shape.query_heads,
            q,
        ];
        if slices.each_ref().map(|slice| slice.len()) != sizes {
            return Err(bad("incorrect CUDA relative attention input shape"));
        }
        bytes.clear();
        let mut offsets = [0; 8];
        for (offset, slice) in offsets.iter_mut().zip(slices) {
            *offset = bytes.len();
            extend_words(bytes, slice);
        }
        offsets[6] = bytes.len();
        bytes.extend_from_slice(
            &u32::try_from(position)
                .map_err(|_| bad("CUDA relative attention position overflows"))?
                .to_ne_bytes(),
        );
        offsets[7] = bytes.len();
        bytes.extend_from_slice(&0_u32.to_ne_bytes());
        Ok(offsets)
    }
}

fn ring_splits(first_slot: usize, rows: usize, capacity: usize) -> [(usize, usize); 2] {
    let until_wrap = rows.min(capacity - first_slot);
    [(first_slot, until_wrap), (0, rows - until_wrap)]
}

fn extend_words(bytes: &mut Vec<u8>, words: &[f32]) {
    bytes.extend(words.iter().flat_map(|word| word.to_ne_bytes()));
}

fn f32_bytes(words: usize) -> Result<usize, EngError> {
    words
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| bad("CUDA relative attention byte count overflows"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_credit_preserves_workspace_headroom() {
        assert_eq!(state_headroom(0), VRAM_HEADROOM_BYTES);
        assert_eq!(state_headroom(32 << 20), VRAM_HEADROOM_BYTES - (32 << 20));
        assert_eq!(state_headroom(VRAM_HEADROOM_BYTES), DEVICE_HEADROOM);
    }

    #[test]
    fn restore_remaps_visible_rows_across_the_ring_boundary() {
        assert_eq!(ring_splits(0, 2, 4), [(0, 2), (0, 0)]);
        assert_eq!(ring_splits(3, 4, 4), [(3, 1), (0, 3)]);
        assert_eq!(ring_splits(2, 3, 4), [(2, 2), (0, 1)]);
    }

    #[test]
    fn restored_attention_replaces_a_divergent_ring_at_the_same_position() {
        if !crate::kernels::cuda::device_present("Inkling CUDA restored relative attention") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let shape = RelativeAttentionShape {
            query_heads: 1,
            kv_heads: 1,
            relative_dims: 1,
            relative_extent: 1,
            capacity: 3,
        };
        let [
            out,
            cache_bytes,
            _,
            status,
            q,
            k,
            v,
            relative,
            table_bytes,
            sinks,
            gates,
            pos,
        ] = shape.region_bytes().unwrap();
        let attention = Plan {
            visits: vec![Visit {
                shape,
                table_at: 0,
                table_bytes,
                cache_at: table_bytes,
                cache_bytes,
            }],
            tables: vec![0; table_bytes],
            scratch_bytes: q + k + v + relative + sinks + gates + pos + status + out,
        }
        .load(&ctx, false)
        .unwrap()
        .unwrap();
        let zero = [0.0; HEAD_DIM];
        let run = |position, value, history: &[f32]| {
            let keys = history.iter().map(|_| Arc::new(zero.to_vec())).collect();
            let values = history
                .iter()
                .map(|&v| Arc::new(vec![v; HEAD_DIM]))
                .collect();
            attention
                .step(
                    &ctx,
                    0,
                    Row {
                        query: &zero,
                        key: &zero,
                        value: &[value; HEAD_DIM],
                        relative: &[0.0],
                        sinks: &[0.0],
                        gates: &zero,
                    },
                    History {
                        position,
                        keys: &keys,
                        values: &values,
                    },
                )
                .unwrap()
        };
        assert_eq!(run(0, 2.0, &[]), [0.5; HEAD_DIM]);
        assert_eq!(run(1, 4.0, &[2.0]), [1.0; HEAD_DIM]);
        attention.invalidate();
        assert_eq!(run(2, 2.0, &[40.0, 20.0]), [7.75; HEAD_DIM]);
        assert_eq!(run(3, 6.0, &[2.0, 40.0, 20.0]), [6.0; HEAD_DIM]);
    }
}
