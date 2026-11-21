//! Where the experts of a MoE layer live inside a GGUF file, host only.
//!
//! WHAT: [`ExpertLocator`] turns one parsed [`GgufHeader`] into the absolute
//! byte ranges of single experts inside the three stacked per-layer expert
//! tensors of a Q4_K GGUF conversion, into pool jobs over any shared region
//! holding the whole file, and into the per-expert byte count the budget
//! planners need. The placement and its validation are the shared
//! [`GgufExpertIndex`] (`weights::gguf_experts`); this module adds the Q4_K
//! shape rule and the mapped-region jobs. WHY: the CPU tier reads each routed
//! expert straight from the memory-mapped file, so it needs one validated,
//! I/O-free mapping from `(layer, expert)` to bytes, built once at load. This
//! module does no file I/O and creates no mapping (a file-backed mmap needs
//! `unsafe`, a separate decision): the caller supplies any `Arc<dyn
//! MappedRegion>`. Nothing about Qwen3.6 is hard-coded; every number arrives as
//! an argument. Used by `TieredHost` and the tiered load.

use std::ops::Range;
use std::sync::Arc;

use super::cpu_pool::{ExpertJob, MappedRegion};
use crate::kernels::expert_cpu::{ExpertShape, Q4K_BLOCK_VALUES};
pub use crate::weights::{ExpertTensorNames, LocatorError};
use crate::weights::{GgufExpertIndex, GgufHeader, GgufType};

/// The byte ranges of one expert inside the file: absolute, half-open `usize`
/// offsets, ready to index any region that holds the whole file at its offsets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertRanges {
    /// Gate projection bytes of the expert: `inter` rows over `hidden` inputs.
    pub gate: Range<usize>,
    /// Up projection bytes of the expert: `inter` rows over `hidden` inputs.
    pub up: Range<usize>,
    /// Down projection bytes of the expert: `hidden` rows over `inter` inputs.
    pub down: Range<usize>,
}

/// The validated placement of every expert of every requested layer inside one
/// GGUF file. Built once at load from a parsed header; every later
/// `ranges`/`job` call is pure arithmetic over the stored spans.
#[derive(Debug)]
pub struct ExpertLocator {
    index: GgufExpertIndex,
    expert_bytes: u64,
    shape: ExpertShape,
}

/// The routed-expert refusal's shared-expert twin: every layer's three shexp
/// tensors (`blk.{l}.ffn_{gate,up,down}_shexp.weight`, llama.cpp names) must
/// exist and be Q4_K. WHY: the tiered chunk facts assert a Q4_K shared expert
/// without reading the tensor, so the attach that builds the locator also proves
/// that fact; anything else refuses the load before the banks exist.
pub fn require_q4k_shexp(header: &GgufHeader, layers: usize) -> Result<(), LocatorError> {
    for layer in 0..layers {
        for part in ["gate", "up", "down"] {
            let name = format!("blk.{layer}.ffn_{part}_shexp.weight");
            let tensor = header
                .tensor(&name)
                .ok_or_else(|| LocatorError::MissingTensor { name: name.clone() })?;
            if tensor.ggml_type() != GgufType::Q4K {
                return Err(LocatorError::BadTensor {
                    name,
                    why: format!(
                        "shared expert expected type Q4_K, found {}",
                        tensor.ggml_type().name()
                    ),
                });
            }
        }
    }
    Ok(())
}

impl ExpertLocator {
    /// Validates and indexes `layers` layers: every named tensor must exist,
    /// be Q4_K and 3-dimensional, with gate/up dims `[hidden, inter, experts]`
    /// and down dims `[inter, hidden, experts]` (ggml order, `ne0` the row
    /// length) matching `shape` and `ne2 == experts`, `byte_len == per_expert *
    /// experts`, its whole range inside `header.file_len()`, and the three
    /// tensors of a layer disjoint. WHY: one up-front refusal keeps every
    /// per-token lookup a panic-free subtraction. All arithmetic is checked.
    pub fn new(
        header: &GgufHeader,
        names: &ExpertTensorNames,
        layers: usize,
        experts: usize,
        shape: ExpertShape,
    ) -> Result<ExpertLocator, LocatorError> {
        if layers == 0 || experts == 0 {
            return Err(LocatorError::ShapeMismatch(format!(
                "locator: layers {layers} and experts {experts} must both be nonzero"
            )));
        }
        if shape.hidden == 0
            || shape.inter == 0
            || !shape.hidden.is_multiple_of(Q4K_BLOCK_VALUES)
            || !shape.inter.is_multiple_of(Q4K_BLOCK_VALUES)
        {
            return Err(LocatorError::ShapeMismatch(format!(
                "locator: expert shape {}x{} must have nonzero extents that are multiples \
                 of {Q4K_BLOCK_VALUES}",
                shape.hidden, shape.inter
            )));
        }
        if names.gate.is_none() {
            return Err(LocatorError::ShapeMismatch(
                "locator: the expert kernels need a gate stack, and the names give none".into(),
            ));
        }
        // One file: the mapped region holds the whole file at its offsets, so a tensor in a
        // later shard of a split set cannot be addressed by absolute offset.
        for layer in 0..layers {
            let at = names.for_layer(layer);
            for name in at.gate.iter().chain([&at.up, &at.down]) {
                if let Some(shard) = header
                    .tensor(name)
                    .map(|tensor| tensor.shard())
                    .filter(|&shard| shard > 0)
                {
                    return Err(LocatorError::BadTensor {
                        name: name.clone(),
                        why: format!(
                            "the locator maps one file; tensor {name} is in shard {}",
                            shard + 1
                        ),
                    });
                }
            }
        }
        let index = GgufExpertIndex::new(
            header,
            &[header.file_len()],
            names,
            layers,
            experts,
            (shape.hidden, shape.inter),
            Some(&[GgufType::Q4K]),
        )?;
        let expert_bytes = index.record_bytes(0)?;
        Ok(ExpertLocator {
            index,
            expert_bytes,
            shape,
        })
    }

    /// The layer count this locator was built for; valid layer indices are
    /// `0..layers()`.
    pub fn layers(&self) -> usize {
        self.index.layers()
    }

    /// The expert count per layer; valid expert indices are `0..experts()`.
    pub fn experts(&self) -> usize {
        self.index.experts()
    }

    /// The expert shape every indexed tensor was validated against.
    pub fn shape(&self) -> ExpertShape {
        self.shape
    }

    /// Bytes of one expert (gate + up + down) — the number
    /// [`ExpertSize::bytes`](super::budget::ExpertSize::bytes) wants. Units:
    /// bytes, `u64` so the budget planners consume it directly.
    pub fn expert_bytes(&self) -> u64 {
        self.expert_bytes
    }

    /// The absolute byte ranges of expert `expert` of layer `layer`:
    /// `abs_offset + expert * per_expert .. + per_expert` per stacked tensor,
    /// checked at every step; the ranges index a region holding the file.
    pub fn ranges(&self, layer: usize, expert: usize) -> Result<ExpertRanges, LocatorError> {
        let slices = self.index.slices(layer, expert)?;
        let range = |r: &Range<u64>| -> Result<Range<usize>, LocatorError> {
            Ok(
                usize::try_from(r.start).map_err(|_| LocatorError::Overflow)?
                    ..usize::try_from(r.end).map_err(|_| LocatorError::Overflow)?,
            )
        };
        Ok(ExpertRanges {
            gate: range(&slices.require_gate()?.file_range)?,
            up: range(&slices.up.file_range)?,
            down: range(&slices.down.file_range)?,
        })
    }

    /// Builds the pool job for `(layer, expert)` over `region`, which must hold
    /// the WHOLE file at its offsets (the ranges are absolute file offsets);
    /// `weight` is the router weight carried into the job unchanged.
    pub fn job(
        &self,
        region: Arc<dyn MappedRegion>,
        layer: usize,
        expert: usize,
        weight: f32,
    ) -> Result<ExpertJob, LocatorError> {
        let ranges = self.ranges(layer, expert)?;
        Ok(ExpertJob {
            region,
            gate: ranges.gate,
            up: ranges.up,
            down: ranges.down,
            weight,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{ExpertLocator, ExpertTensorNames, LocatorError};
    use crate::family::qwen36::tiered::cpu_pool::{CpuPool, MappedRegion};
    use crate::family::qwen36::tiered::{PolicyMode, TieredConfig};
    use crate::family::tier_budget::ExpertSize;
    use crate::kernels::expert_cpu::{
        ExpertScratch, ExpertShape, ExpertSlices, Q4K_BLOCK_BYTES, Q4K_BLOCK_VALUES, expert_forward,
    };
    use crate::weights::{GgufHeader, GgufType};
    use std::ops::Range;
    use std::sync::Arc;
    use std::time::Duration;

    /// Tiny deterministic LCG (Knuth MMIX constants); tests never touch the OS.
    pub(crate) struct Lcg(pub(crate) u64);

    impl Lcg {
        pub(crate) fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        pub(crate) fn f32_between(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (self.next() as f32 / (1u64 << 53) as f32)
        }
    }

    /// Moderate f16 scale patterns; they keep expert outputs finite and small.
    const F16_BITS: [u16; 4] = [0x3500, 0x3c00, 0x4000, 0xbc00];

    /// `blocks` random Q4_K blocks with moderate, finite scales.
    fn random_q4k(rng: &mut Lcg, blocks: usize) -> Vec<u8> {
        let mut w = vec![0u8; blocks * Q4K_BLOCK_BYTES];
        for blk in w.as_chunks_mut::<Q4K_BLOCK_BYTES>().0 {
            blk[4..].iter_mut().for_each(|b| *b = rng.next() as u8);
            let d = F16_BITS[(rng.next() % 4) as usize].to_le_bytes();
            blk[0..2].copy_from_slice(&d);
            blk[2..4].copy_from_slice(&d);
        }
        w
    }

    /// Byte length of a `rows x cols` Q4_K slice: the task formula.
    fn slice_bytes(rows: usize, cols: usize) -> usize {
        rows * cols / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES
    }

    /// One tensor table entry of the synthetic GGUF: name, dims, ggml type id
    /// and payload, placed at `data_offset` relative to the data section start
    /// (the alignment of the format, 32 here, divides every offset).
    #[derive(Clone)]
    struct Entry {
        name: String,
        dims: [u64; 3],
        type_id: u32,
        data_offset: u64,
        payload: Vec<u8>,
    }

    /// Serializes a minimal GGUF v3 (no metadata, so the alignment stays the
    /// default 32): magic, version, counts, the tensor table, padding, then
    /// each payload at `data_start + data_offset`. Returns the file bytes and
    /// the data section start.
    fn write_gguf(entries: &[Entry]) -> (Vec<u8>, usize) {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        for e in entries {
            out.extend_from_slice(&(e.name.len() as u64).to_le_bytes());
            out.extend_from_slice(e.name.as_bytes());
            out.extend_from_slice(&3u32.to_le_bytes());
            for d in e.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&e.type_id.to_le_bytes());
            out.extend_from_slice(&e.data_offset.to_le_bytes());
        }
        while out.len() % 32 != 0 {
            out.push(0);
        }
        let data_start = out.len();
        for e in entries {
            let at = data_start + usize::try_from(e.data_offset).unwrap();
            let end = at + e.payload.len();
            if out.len() < end {
                out.resize(end, 0);
            }
            out[at..end].copy_from_slice(&e.payload);
        }
        (out, data_start)
    }

    /// The standard tensor set of `layers` layers (llama.cpp names, Q4_K,
    /// task-formula payload lengths) with random payloads and unset offsets.
    fn expert_entries(
        rng: &mut Lcg,
        layers: usize,
        experts: usize,
        shape: ExpertShape,
    ) -> Vec<Entry> {
        let names = ExpertTensorNames::standard();
        let blocks = shape.hidden * shape.inter / Q4K_BLOCK_VALUES * experts;
        let (h, i, e) = (shape.hidden as u64, shape.inter as u64, experts as u64);
        let mut entries = Vec::new();
        for layer in 0..layers {
            let nl = names.for_layer(layer);
            for (name, dims) in [
                (nl.gate.expect("gate"), [h, i, e]),
                (nl.up, [h, i, e]),
                (nl.down, [i, h, e]),
            ] {
                entries.push(Entry {
                    name,
                    dims,
                    type_id: 12, // Q4_K
                    data_offset: 0,
                    payload: random_q4k(rng, blocks),
                });
            }
        }
        entries
    }

    /// Assigns sequential 32-aligned data offsets in the given (table) order.
    fn place_sequential(entries: &mut [Entry]) {
        let mut cursor = 0u64;
        for e in entries.iter_mut() {
            e.data_offset = cursor;
            cursor = (cursor + e.payload.len() as u64).div_ceil(32) * 32;
        }
    }

    /// A good synthetic expert file: the standard entries, but the table is
    /// written in reversed order so the file layout differs from the
    /// layer-major reading order and the locator must follow the table.
    /// Returns the file and each layer's `[gate, up, down]` data offsets.
    pub(crate) fn expert_file(
        rng: &mut Lcg,
        layers: usize,
        experts: usize,
        shape: ExpertShape,
    ) -> ((Vec<u8>, usize), Vec<[u64; 3]>) {
        let mut table = expert_entries(rng, layers, experts, shape);
        table.reverse();
        place_sequential(&mut table);
        let names = ExpertTensorNames::standard();
        let offsets = (0..layers)
            .map(|layer| {
                let nl = names.for_layer(layer);
                let at = |n: &str| table.iter().find(|e| e.name == n).unwrap().data_offset;
                [
                    at(nl.gate.as_deref().expect("gate")),
                    at(&nl.up),
                    at(&nl.down),
                ]
            })
            .collect();
        (write_gguf(&table), offsets)
    }

    /// The byte length the parser derives from dims and type, so a mutated
    /// entry still parses (its payload is resized to match).
    fn derived_len(dims: &[u64; 3], type_id: u32) -> usize {
        let t = GgufType::from_id(type_id).unwrap();
        (dims[0] / t.block_elems() * dims[1] * dims[2] * t.block_bytes()) as usize
    }

    /// Parses a synthetic file; its own length is the file length.
    pub(crate) fn parse(bytes: &[u8]) -> GgufHeader {
        let len = bytes.len() as u64;
        GgufHeader::from_bytes(bytes, len, len).unwrap()
    }

    /// The expected expert slice at `off` (a data offset), cut by hand.
    fn hand_range(data_start: usize, off: u64, expert: usize, per: usize) -> Range<usize> {
        let at = data_start + usize::try_from(off).unwrap() + expert * per;
        at..at + per
    }

    /// One layer's shexp trio: Q4_K everywhere passes; a wrong type names the
    /// tensor; a missing tensor names what the load needed.
    #[test]
    fn shexp_refusal_names_the_offending_tensor() {
        let mut rng = Lcg(0x514D_3252_0000_0030);
        let shape = ExpertShape {
            hidden: 256,
            inter: 256,
        };
        let mut entries = expert_entries(&mut rng, 1, 4, shape);
        for part in ["gate", "up", "down"] {
            entries.push(Entry {
                name: format!("blk.0.ffn_{part}_shexp.weight"),
                dims: [256, 256, 1],
                type_id: 12, // Q4_K
                data_offset: 0,
                payload: random_q4k(&mut rng, 256 * 256 / 32),
            });
        }
        place_sequential(&mut entries);
        let (bytes, _) = write_gguf(&entries);
        let header = parse(&bytes);
        super::require_q4k_shexp(&header, 1).unwrap();
        assert_eq!(
            super::require_q4k_shexp(&header, 2).unwrap_err(),
            LocatorError::MissingTensor {
                name: "blk.1.ffn_gate_shexp.weight".to_owned()
            }
        );

        let last = entries.last_mut().unwrap();
        last.type_id = 0; // F32: unblocked, so give it a 4-byte body to match.
        last.dims = [1, 1, 1];
        last.payload = vec![0u8; 4];
        let (bytes, _) = write_gguf(&entries);
        let header = parse(&bytes);
        match super::require_q4k_shexp(&header, 1).unwrap_err() {
            LocatorError::BadTensor { name, why } => {
                assert!(name.contains("ffn_down_shexp"));
                assert!(why.contains("Q4_K"));
            }
            other => panic!("wrong error: {other:?}"),
        }

        entries.pop();
        let (bytes, _) = write_gguf(&entries);
        let header = parse(&bytes);
        match super::require_q4k_shexp(&header, 1).unwrap_err() {
            LocatorError::MissingTensor { name } => assert!(name.contains("ffn_down_shexp")),
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn standard_names_are_the_gguf_convention() {
        // The literal spellings of the stacked expert tensors; `{layer}` stays
        // a placeholder until for_layer replaces it.
        let n = ExpertTensorNames::standard();
        assert_eq!(n.gate.as_deref(), Some("blk.{layer}.ffn_gate_exps.weight"));
        assert_eq!(n.up, "blk.{layer}.ffn_up_exps.weight");
        assert_eq!(n.down, "blk.{layer}.ffn_down_exps.weight");
    }

    #[test]
    fn locator_names_replace_the_layer_placeholder() {
        let l7 = ExpertTensorNames::standard().for_layer(7);
        assert_eq!(l7.gate.as_deref(), Some("blk.7.ffn_gate_exps.weight"));
        assert_eq!(l7.up, "blk.7.ffn_up_exps.weight");
        assert_eq!(l7.down, "blk.7.ffn_down_exps.weight");
        let custom = ExpertTensorNames {
            gate: Some(String::from("{layer}/g/{layer}")),
            up: String::from("{layer}/u"),
            down: String::from("{layer}/d"),
        };
        assert_eq!(custom.for_layer(3).gate.as_deref(), Some("3/g/3"));
        assert_eq!(custom.for_layer(3).up, "3/u");
        assert_eq!(custom.gate.as_deref(), Some("{layer}/g/{layer}"));
    }

    #[test]
    fn the_synthetic_writer_round_trips_through_the_header_parser() {
        // Binds the test writer to the real reader: every entry is found by
        // name and its dims, type, absolute offset and byte length round-trip.
        let shape = ExpertShape {
            hidden: 256,
            inter: 256,
        };
        let mut table = expert_entries(&mut Lcg(0x0FF_5EED_0000_0000), 2, 3, shape);
        place_sequential(&mut table);
        let (bytes, data_start) = write_gguf(&table);
        let len = bytes.len() as u64;
        let header = GgufHeader::from_bytes(&bytes, len, len).unwrap();
        assert_eq!(header.tensors().len(), 6);
        for e in &table {
            let t = header.tensor(&e.name).unwrap();
            assert_eq!(t.dims(), e.dims.as_slice(), "{}", e.name);
            assert_eq!(t.ggml_type(), GgufType::Q4K);
            assert_eq!(t.abs_offset(), data_start as u64 + e.data_offset);
            assert_eq!(t.byte_len() as usize, e.payload.len());
        }
    }

    #[test]
    fn locator_ranges_refuses_an_out_of_range_layer_or_expert() {
        // A 2-layer, 5-expert locator echoes the out-of-range arguments; the
        // legal boundaries still answer.
        let shape = ExpertShape {
            hidden: 256,
            inter: 256,
        };
        let (file, _) = expert_file(&mut Lcg(0x10C_A7E2_0000_0001), 2, 5, shape);
        let header = parse(&file.0);
        let locator =
            ExpertLocator::new(&header, &ExpertTensorNames::standard(), 2, 5, shape).unwrap();
        assert_eq!(
            locator.ranges(2, 0),
            Err(LocatorError::OutOfRange {
                layer: 2,
                expert: 0
            })
        );
        assert_eq!(
            locator.ranges(0, 5),
            Err(LocatorError::OutOfRange {
                layer: 0,
                expert: 5
            })
        );
        // `ExpertJob` carries an `Arc<dyn MappedRegion>`, which has neither
        // `Debug` nor `PartialEq`, so this one is matched by pattern.
        assert!(matches!(
            locator.job(Arc::new(Vec::new()), 9, 9, 0.5),
            Err(LocatorError::OutOfRange {
                layer: 9,
                expert: 9
            })
        ));
        assert!(locator.ranges(1, 4).is_ok());
    }

    #[test]
    fn locator_finds_every_expert_range_of_a_synthetic_gguf() {
        // 3 layers, 5 experts, 256/256: every range of every tensor equals the
        // independently computed offset (from the writer's records, not the
        // locator) and carries the shape's byte length.
        let shape = ExpertShape {
            hidden: 256,
            inter: 256,
        };
        let ((bytes, data_start), offsets) =
            expert_file(&mut Lcg(0xFA7E_5EED_0000_0002), 3, 5, shape);
        let header = parse(&bytes);
        let locator =
            ExpertLocator::new(&header, &ExpertTensorNames::standard(), 3, 5, shape).unwrap();
        assert_eq!((locator.layers(), locator.experts()), (3, 5));
        let per = slice_bytes(shape.hidden, shape.inter);
        for (layer, layer_offsets) in offsets.iter().enumerate() {
            for expert in 0..5 {
                let r = locator.ranges(layer, expert).unwrap();
                for (got, off) in [
                    (&r.gate, layer_offsets[0]),
                    (&r.up, layer_offsets[1]),
                    (&r.down, layer_offsets[2]),
                ] {
                    assert_eq!(
                        *got,
                        hand_range(data_start, off, expert, per),
                        "layer {layer} expert {expert}"
                    );
                    assert_eq!(got.len(), per);
                }
            }
        }
    }

    #[test]
    fn locator_job_runs_through_the_cpu_pool_and_matches_a_direct_forward() {
        // The jobs the locator builds over a region holding the whole file,
        // run on a 2-thread pool, equal the ordered serial sum of
        // expert_forward over the same slices cut by hand.
        let shape = ExpertShape {
            hidden: 256,
            inter: 256,
        };
        let mut rng = Lcg(0xB10C_8EED_0000_0003);
        let ((bytes, data_start), offsets) = expert_file(&mut rng, 2, 3, shape);
        let header = parse(&bytes);
        let locator =
            ExpertLocator::new(&header, &ExpertTensorNames::standard(), 2, 3, shape).unwrap();
        let per = slice_bytes(shape.hidden, shape.inter);
        let region: Arc<dyn MappedRegion> = Arc::new(bytes.clone());
        let hidden: Vec<f32> = (0..shape.hidden)
            .map(|_| rng.f32_between(-1.0, 1.0))
            .collect();
        let weights: Vec<f32> = (0..3).map(|_| rng.f32_between(-0.5, 0.5)).collect();
        let jobs: Vec<_> = (0..3)
            .map(|e| locator.job(Arc::clone(&region), 1, e, weights[e]).unwrap())
            .collect();
        let pool = CpuPool::new(2, shape).unwrap();
        let got = pool
            .run_layer(jobs, &hidden, Duration::from_secs(60))
            .unwrap()
            .unwrap();
        // Serial reference over slices cut by hand at the recorded offsets.
        let mut scratch = ExpertScratch::new(shape);
        let mut out = vec![0.0f32; shape.hidden];
        let mut want = vec![0.0f32; shape.hidden];
        for e in 0..3 {
            let slices = ExpertSlices {
                gate: &bytes[hand_range(data_start, offsets[1][0], e, per)],
                up: &bytes[hand_range(data_start, offsets[1][1], e, per)],
                down: &bytes[hand_range(data_start, offsets[1][2], e, per)],
            };
            expert_forward(shape, &slices, &hidden, &mut scratch, &mut out).unwrap();
            for (s, v) in want.iter_mut().zip(&out) {
                *s += weights[e] * v;
            }
        }
        // The pool's contract is exact bits, not a tolerance.
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn locator_over_a_mapped_file_feeds_the_q8k_pool_like_the_in_memory_file() {
        // The whole host path: a synthetic GGUF written to disk, mapped read-only
        // by `MappedFile`, indexed by the locator and run on a 3-thread Q8_K pool
        // gives the bits of the same jobs over the file's bytes held in memory.
        use crate::family::mapped_file::MappedFile;
        use crate::family::qwen36::tiered::cpu_pool::ExpertKernel;
        use crate::kernels::expert_cpu_q8k::DotBackend;
        use std::io::Write;

        let shape = ExpertShape {
            hidden: 256,
            inter: 256,
        };
        let mut rng = Lcg(0xE2E0_5EED_0000_0004);
        let ((bytes, _), _) = expert_file(&mut rng, 2, 4, shape);
        let header = parse(&bytes);
        let locator =
            ExpertLocator::new(&header, &ExpertTensorNames::standard(), 2, 4, shape).unwrap();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        let mapped = MappedFile::open(file.path(), header.file_len()).unwrap();
        mapped.check_unchanged().unwrap();
        let on_disk: Arc<dyn MappedRegion> = Arc::new(mapped);
        let in_memory: Arc<dyn MappedRegion> = Arc::new(bytes.clone());
        let hidden: Vec<f32> = (0..shape.hidden)
            .map(|_| rng.f32_between(-1.0, 1.0))
            .collect();
        let weights: Vec<f32> = (0..4).map(|_| rng.f32_between(0.05, 0.4)).collect();
        let pool = CpuPool::with_kernel(3, shape, ExpertKernel::Q8k(DotBackend::detect())).unwrap();
        let mut sums = Vec::new();
        for region in [on_disk, in_memory] {
            let jobs: Vec<_> = (0..4)
                .map(|e| locator.job(Arc::clone(&region), 1, e, weights[e]).unwrap())
                .collect();
            let sum = pool
                .run_layer(jobs, &hidden, Duration::from_secs(60))
                .unwrap()
                .unwrap();
            sums.push(sum.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        assert_eq!(sums[0], sums[1]);
        assert!(sums[0].iter().any(|&b| f32::from_bits(b) != 0.0));
    }

    #[test]
    fn locator_expert_bytes_feeds_the_budget() {
        // The per-expert byte count is exactly three slices and the budget
        // planners accept it.
        let shape = ExpertShape {
            hidden: 256,
            inter: 256,
        };
        let (file, _) = expert_file(&mut Lcg(0x0EED_0CC0_0000_0004), 3, 5, shape);
        let header = parse(&file.0);
        let locator =
            ExpertLocator::new(&header, &ExpertTensorNames::standard(), 3, 5, shape).unwrap();
        let per = slice_bytes(shape.hidden, shape.inter) as u64;
        assert_eq!(locator.expert_bytes(), 3 * per);
        let cfg = TieredConfig {
            layers: locator.layers(),
            experts: locator.experts(),
            top_k: 1,
            bank_slots_per_layer: 1,
            mode: PolicyMode::Static,
            swaps_per_token: 0.25,
            decay: 0.999,
        };
        cfg.check().unwrap();
        let size = ExpertSize::from_config(&cfg, locator.expert_bytes());
        assert_eq!((size.bytes, size.layers), (3 * per, 3));
    }

    #[test]
    fn locator_refuses_a_missing_tensor_a_wrong_type_wrong_dims_and_a_short_file() {
        // hidden != inter so a transposition is visible; entries are
        // layer-major [gate, up, down], so layer 1's down is index 5.
        let shape = ExpertShape {
            hidden: 512,
            inter: 256,
        };
        let build = |mut table: Vec<Entry>| -> (Vec<u8>, usize) {
            place_sequential(&mut table);
            write_gguf(&table)
        };
        let locate = |file: &(Vec<u8>, usize)| {
            let header = parse(&file.0);
            ExpertLocator::new(&header, &ExpertTensorNames::standard(), 2, 4, shape)
        };
        let fresh = |seed: u64| expert_entries(&mut Lcg(seed), 2, 4, shape);

        // A missing down tensor of layer 1.
        let mut table = fresh(0x1BAD_C0DE_0000_0005);
        table.remove(5);
        match locate(&build(table)) {
            Err(LocatorError::MissingTensor { name }) => {
                assert_eq!(name, "blk.1.ffn_down_exps.weight")
            }
            other => panic!("expected MissingTensor, got {other:?}"),
        }

        // A gate stored as Q8_0.
        let mut table = fresh(0x2BAD_C0DE_0000_0006);
        table[0].type_id = 8; // Q8_0
        table[0].payload = vec![0; derived_len(&table[0].dims, table[0].type_id)];
        match locate(&build(table)) {
            Err(LocatorError::BadTensor { name, why }) => {
                assert_eq!(name, "blk.0.ffn_gate_exps.weight");
                assert!(why.contains("Q8_0") && why.contains("Q4_K"), "{why}")
            }
            other => panic!("expected BadTensor, got {other:?}"),
        }

        // Transposed gate dims ([inter, hidden, experts] instead).
        let mut table = fresh(0x3BAD_C0DE_0000_0007);
        table[0].dims = [256, 512, 4];
        table[0].payload = vec![0; derived_len(&table[0].dims, table[0].type_id)];
        match locate(&build(table)) {
            Err(LocatorError::ShapeMismatch(why)) => {
                assert!(why.contains("blk.0.ffn_gate_exps.weight"), "{why}");
                assert!(why.contains("[256, 512, 4]"), "{why}");
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }

        // ne2 != experts: dims [512, 256, 6] with experts = 4. This is also
        // the only way byte_len can disagree with per_expert * experts through
        // a parsed header (the parser derives byte_len from the dims), so the
        // locator's byte_len check is defensive once the dims match.
        let mut table = fresh(0x4BAD_C0DE_0000_0008);
        table[0].dims = [512, 256, 6];
        table[0].payload = vec![0; derived_len(&table[0].dims, table[0].type_id)];
        let file = build(table);
        let header = parse(&file.0);
        let gate = header.tensor("blk.0.ffn_gate_exps.weight").unwrap();
        let per = (512 * 256 / 256 * 144) as u64;
        assert_ne!(gate.byte_len(), per * 4, "the byte lengths must disagree");
        match ExpertLocator::new(&header, &ExpertTensorNames::standard(), 2, 4, shape) {
            Err(LocatorError::ShapeMismatch(why)) => {
                assert!(why.contains("[512, 256, 6]"), "{why}");
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }

        // A short file: the parser itself refuses a tensor range beyond
        // file_len, so the locator never sees one (its check is defensive).
        let (bytes, _) = build(fresh(0x5BAD_C0DE_0000_0009));
        let short = bytes.len() as u64 - 1;
        assert!(GgufHeader::from_bytes(&bytes, short, short).is_err());

        // Overlapping gate/up of one layer: the parser refuses the overlap
        // before the locator can index it (its check is defensive).
        let mut table = fresh(0x6BAD_C0DE_0000_000A);
        place_sequential(&mut table);
        table[1].data_offset = table[0].data_offset;
        let (bytes, _) = write_gguf(&table);
        let len = bytes.len() as u64;
        assert!(GgufHeader::from_bytes(&bytes, len, len).is_err());
    }

    /// A split set whose expert stacks are in the second file: the locator maps one file and
    /// refuses by tensor and shard.
    #[test]
    fn a_tensor_in_a_later_shard_is_refused_by_name_and_shard() {
        use crate::weights::GgufWeights;
        use crate::weights::gguf_experts::test_support::{LIMITS, T, kv_u16, write};
        let dir = tempfile::tempdir().expect("tempdir");
        let name = |n: u16| dir.path().join(format!("q-{n:05}-of-00002.gguf"));
        let kvs = |no: u16| {
            vec![
                kv_u16("split.no", no),
                kv_u16("split.count", 2),
                kv_u16("split.tensors.count", 3),
            ]
        };
        let per = 256 * 256 / 256 * 144;
        let stack = |part: &str| T {
            name: format!("blk.0.ffn_{part}_exps.weight"),
            dims: vec![256, 256, 2],
            ty: GgufType::Q4K,
            payload: vec![0; 2 * per],
        };
        write(&name(1), &kvs(0), &[]);
        write(
            &name(2),
            &kvs(1),
            &[stack("gate"), stack("up"), stack("down")],
        );
        let weights = GgufWeights::open_any_type(&name(1), LIMITS, None).expect("open the set");
        let shape = ExpertShape {
            hidden: 256,
            inter: 256,
        };
        let error = ExpertLocator::new(
            weights.header(),
            &ExpertTensorNames::standard(),
            1,
            2,
            shape,
        )
        .expect_err("a later shard");
        assert_eq!(
            error,
            LocatorError::BadTensor {
                name: "blk.0.ffn_gate_exps.weight".into(),
                why: "the locator maps one file; tensor blk.0.ffn_gate_exps.weight is in shard 2"
                    .into()
            }
        );
    }
}
