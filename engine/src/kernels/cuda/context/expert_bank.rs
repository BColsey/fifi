//! The per-layer expert bank of the tiered decode.
//!
//! WHAT: an [`ExpertBank`] is three zero-initialised native-layout weights (gate, up, down)
//! of `slots` experts each (the file's own record type per stack), the stack the existing
//! `matvec_expert` launches read with
//! `n_expert = slots` and the slot index as the expert id. [`ExpertBank::fill`] copies one
//! expert's three slices into a slot with blocking writes. WHY: the tiered step keeps only the
//! hot experts of a layer in VRAM; the existing launches stay untouched, so the bank has to
//! look exactly like an expert stack, and it has to be finite everywhere (a missed pick reads
//! slot 0 with weight 0.0, and `0 * Inf` would be NaN). Built by the tiered load.
//!
//! `slots` must be at least `top_k`: the launch needs `n_used <= n_expert`. Slot 0 doubles as
//! the dummy target of a missed pick but may also hold a real expert: duplicate ids in one
//! launch are legal and write distinct output rows, so `slots = 8` gives eight resident slots.

use crate::EngError;
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::weights::GgufType;

use super::{CudaContext, DeviceStream, NativeWeight, QuantTensor};

/// Source bytes the zero upload materialises at once.
const ZERO_CHUNK_BYTES: usize = 8 << 20;
/// The widest pick list the expert launch takes (`n_used` is 1 to 8).
const MAX_TOP_K: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExpertBankGeometry {
    /// Model hidden size: columns of gate and up, rows of down.
    pub(crate) hidden: usize,
    /// Per-expert intermediate size: rows of gate and up, columns of down.
    pub(crate) inter: usize,
    /// Experts picked per token and layer (1 to 8).
    pub(crate) top_k: usize,
    /// The file's record type of the gate, up and down stacks of this layer. The three need not
    /// agree: a file may keep a layer's down projection in a wider type than its gate and up.
    pub(crate) quant: [GgufType; 3],
}

/// Three native stacks of `slots` experts and the geometry they were built for.
pub(crate) struct ExpertBank {
    gate: NativeWeight,
    up: NativeWeight,
    down: NativeWeight,
    slots: usize,
    geometry: ExpertBankGeometry,
}

fn refuse(what: String) -> EngError {
    EngError::BadRequest(format!("expert bank: {what}"))
}

impl ExpertBankGeometry {
    /// A geometry whose three stacks are all Q4_K.
    #[cfg(test)]
    pub(crate) const fn q4k(hidden: usize, inter: usize, top_k: usize) -> Self {
        Self {
            hidden,
            inter,
            top_k,
            quant: [GgufType::Q4K; 3],
        }
    }

    /// `base` with the types of the layer whose three records are largest among `quants`: the
    /// geometry that sizes a plan for a file whose layers mix types.
    pub(crate) fn widest(base: Self, quants: &[[GgufType; 3]]) -> Result<Self, EngError> {
        let mut best: Option<(usize, Self)> = None;
        for quant in quants {
            let geometry = Self {
                quant: *quant,
                ..base
            };
            let bytes = geometry.slice_bytes()?.into_iter().sum::<usize>();
            if best.is_none_or(|(most, _)| bytes > most) {
                best = Some((bytes, geometry));
            }
        }
        best.map(|(_, geometry)| geometry)
            .ok_or_else(|| refuse("the model has no layers".to_owned()))
    }

    /// The kernel type of the stack `which` (0 gate, 1 up, 2 down), refused when the file's type
    /// has no native kernel.
    fn stack_type(&self, which: usize) -> Result<QuantType, EngError> {
        let ty = self.quant[which];
        QuantType::from_gguf(ty).ok_or_else(|| {
            refuse(format!(
                "the expert type {} has no native kernel",
                ty.name()
            ))
        })
    }

    /// Bytes of one expert's gate, up and down slices: `inter` rows of `hidden` values for the
    /// first two and `hidden` rows of `inter` values for the third, in each stack's own blocks.
    pub(crate) fn slice_bytes(&self) -> Result<[usize; 3], EngError> {
        let bytes = |which: usize, rows: usize, k: usize| {
            self.stack_type(which)?
                .weight_bytes(rows, k)
                .ok_or_else(|| {
                    refuse(format!(
                        "{rows} rows of {k} values overflow or are off the block grid"
                    ))
                })
        };
        Ok([
            bytes(0, self.inter, self.hidden)?,
            bytes(1, self.inter, self.hidden)?,
            bytes(2, self.hidden, self.inter)?,
        ])
    }

    /// The refusals shared by sizing and building, before anything is allocated.
    fn check(&self, slots: usize) -> Result<(), EngError> {
        if self.hidden == 0
            || self.inter == 0
            || !self.hidden.is_multiple_of(4)
            || !self.inter.is_multiple_of(4)
        {
            return Err(refuse(format!(
                "hidden {} and inter {} must be nonzero multiples of 4",
                self.hidden, self.inter
            )));
        }
        if !(1..=MAX_TOP_K).contains(&self.top_k) {
            return Err(refuse(format!(
                "top_k {} is outside 1..={MAX_TOP_K}",
                self.top_k
            )));
        }
        if slots < self.top_k {
            return Err(refuse(format!(
                "{slots} slots are fewer than top_k {}: the expert launch needs n_used <= n_expert",
                self.top_k
            )));
        }
        self.slice_bytes().map(|_| ())
    }
}

impl ExpertBank {
    /// Device bytes of a bank of `slots` slots: three stacks. A pure checked function, so
    /// admission can size a bank before allocating it.
    pub(crate) fn bytes_for(geometry: ExpertBankGeometry, slots: usize) -> Result<u64, EngError> {
        geometry.check(slots)?;
        let [gate, up, down] = geometry.slice_bytes()?;
        let total = gate
            .checked_add(up)
            .and_then(|both| both.checked_add(down))
            .and_then(|three| three.checked_mul(slots))
            .ok_or_else(|| refuse(format!("{slots} slots overflow the byte count")))?;
        u64::try_from(total).map_err(|_| refuse(String::from("the byte count does not fit u64")))
    }

    /// Builds a zero-initialised bank of `slots` slots. Refuses before any allocation when the
    /// geometry is invalid or `slots < top_k`; the zero blocks are uploaded chunk by chunk, so
    /// the host never holds the bank.
    pub(crate) fn new(
        ctx: &CudaContext,
        geometry: ExpertBankGeometry,
        slots: usize,
    ) -> Result<Self, EngError> {
        Self::bytes_for(geometry, slots)?;
        let stack =
            |name: &str, which: usize, rows: usize, k: usize| -> Result<NativeWeight, EngError> {
                let n_rows = rows
                    .checked_mul(slots)
                    .ok_or_else(|| refuse(format!("{slots} slots of {rows} rows overflow")))?;
                let source_len = geometry
                    .stack_type(which)?
                    .weight_bytes(n_rows, k)
                    .ok_or_else(|| refuse(format!("{n_rows} rows of {k} values overflow")))?;
                let tensor = QuantTensor {
                    name,
                    ty: geometry.quant[which],
                    n_rows,
                    k,
                    source_len,
                };
                let mut zeros = |_offset: usize, out: &mut [u8]| {
                    out.fill(0);
                    Ok(())
                };
                Ok(ctx
                    .upload_native(tensor, ZERO_CHUNK_BYTES, &mut zeros)?
                    .with_experts(Some(slots)))
            };
        Ok(Self {
            gate: stack("expert bank gate", 0, geometry.inter, geometry.hidden)?,
            up: stack("expert bank up", 1, geometry.inter, geometry.hidden)?,
            down: stack("expert bank down", 2, geometry.hidden, geometry.inter)?,
            slots,
            geometry,
        })
    }

    /// Slots per layer. Tests only: the production callers know the count they built the bank with.
    #[cfg(test)]
    pub(crate) fn slots(&self) -> usize {
        self.slots
    }

    /// Slots per layer, for the tiered runtime's chunk arm (MR #20): the grouped MMQ
    /// launch's `n_experts` is the bank depth, so the chunk code asks the bank.
    pub(crate) fn slots_of(&self) -> usize {
        self.slots
    }

    /// The gate stack: `slots` experts of `inter` rows of `hidden` values.
    pub(crate) fn gate(&self) -> &NativeWeight {
        &self.gate
    }

    /// The up stack, laid out like the gate stack.
    pub(crate) fn up(&self) -> &NativeWeight {
        &self.up
    }

    /// The down stack: `slots` experts of `hidden` rows of `inter` values.
    pub(crate) fn down(&self) -> &NativeWeight {
        &self.down
    }

    pub(crate) fn slice_bytes(&self) -> Result<[usize; 3], EngError> {
        self.geometry.slice_bytes()
    }

    /// Queues the copy of one expert into `slot` (three copies, gate, up, down) behind the work
    /// queued on `stream` and returns without waiting for it. The slices must stay valid and
    /// unchanged until `stream` completes the copies when they are page-locked memory (a pageable
    /// source is consumed before the call returns). Refuses before any copy for a slot out of
    /// range or a slice of the wrong length. An error from a copy itself can leave the slot half
    /// written: the caller treats it as a failed fill and never publishes the slot.
    pub(crate) fn fill(
        &self,
        stream: &DeviceStream,
        slot: usize,
        gate: &[u8],
        up: &[u8],
        down: &[u8],
    ) -> Result<(), EngError> {
        let [gate_bytes, up_bytes, down_bytes] = self.slice_bytes()?;
        if slot >= self.slots {
            return Err(refuse(format!("slot {slot} is outside 0..{}", self.slots)));
        }
        for (what, got, want) in [
            ("gate", gate.len(), gate_bytes),
            ("up", up.len(), up_bytes),
            ("down", down.len(), down_bytes),
        ] {
            if got != want {
                return Err(refuse(format!(
                    "the {what} slice has {got} bytes, a slot holds {want}"
                )));
            }
        }
        self.gate
            .buffer()
            .write_at_enqueue(stream, slot * gate_bytes, gate)?;
        self.up
            .buffer()
            .write_at_enqueue(stream, slot * up_bytes, up)?;
        self.down
            .buffer()
            .write_at_enqueue(stream, slot * down_bytes, down)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::DeviceBuffer;
    use crate::kernels::cuda::resident::matvec_expert::ExpertMatvecCall;
    use crate::kernels::cuda::resident::matvec_q::tests::reference;
    use crate::kernels::cuda::resident::quantize_q8_1::{QuantizeCall, q8_1_bytes};
    use crate::kernels::cuda::testkit;

    const REAL: ExpertBankGeometry = ExpertBankGeometry::q4k(2048, 512, 8);
    const SLICE: usize = 512 * 8 * 144; // 589,824 bytes: gate, up and down are all this size

    /// Every copy of a fill is an enqueue: a blocking write would drain the stream the fill
    /// is meant to leave running. The count is textual: it skips whole-line comments.
    #[test]
    fn fill_queues_its_three_copies_and_blocks_on_none() {
        let source = include_str!("expert_bank.rs");
        let start = source
            .find("pub(crate) fn fill(")
            .expect("expert_bank.rs defines fill");
        let body = &source[start..];
        let end = body
            .find("\n    }\n")
            .expect("fill ends at its closing brace");
        let code: String = body[..end]
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect();
        assert_eq!(code.matches(".write_at_enqueue(").count(), 3, "{code}");
        assert_eq!(code.matches(".write_at(").count(), 0, "{code}");
    }

    #[test]
    fn bank_sizes_follow_the_geometry() {
        assert_eq!(REAL.slice_bytes().unwrap(), [SLICE; 3]);
        assert_eq!(
            ExpertBank::bytes_for(REAL, 8).unwrap(),
            3 * 8 * SLICE as u64
        );
        assert_eq!(
            ExpertBank::bytes_for(REAL, 61).unwrap(),
            3 * 61 * SLICE as u64
        );
        let wide = ExpertBankGeometry::q4k(2048, 1024, 4);
        assert_eq!(
            wide.slice_bytes().unwrap(),
            [1024 * 8 * 144, 1024 * 8 * 144, 2048 * 4 * 144]
        );
        // Each stack sizes by its own type: a Q6_K down beside Q5_K gate and up.
        let mixed = ExpertBankGeometry {
            quant: [GgufType::Q5K, GgufType::Q5K, GgufType::Q6K],
            ..wide
        };
        assert_eq!(
            mixed.slice_bytes().unwrap(),
            [1024 * 8 * 176, 1024 * 8 * 176, 2048 * 4 * 210]
        );
        // An absurd slot count is a typed refusal, never a wrapped size.
        assert!(ExpertBank::bytes_for(REAL, usize::MAX / 2).is_err());
    }

    #[test]
    fn bank_refuses_bad_geometry_and_slots_before_any_work() {
        let bad = |g: ExpertBankGeometry, slots: usize| {
            format!("{:?}", ExpertBank::bytes_for(g, slots).unwrap_err())
        };
        assert!(bad(REAL, 7).contains("fewer than top_k 8"));
        assert!(bad(ExpertBankGeometry { top_k: 0, ..REAL }, 8).contains("top_k 0"));
        assert!(bad(ExpertBankGeometry { top_k: 9, ..REAL }, 9).contains("top_k 9"));
        assert!(
            bad(ExpertBankGeometry { hidden: 0, ..REAL }, 8).contains("nonzero multiples of 4")
        );
        assert!(bad(ExpertBankGeometry { inter: 6, ..REAL }, 8).contains("nonzero multiples of 4"));
        // Off the Q4_K block grid (256 values): 2052 is a multiple of 4 but not of 256.
        assert!(
            bad(
                ExpertBankGeometry {
                    hidden: 2052,
                    ..REAL
                },
                8
            )
            .contains("block grid")
        );
    }

    /// The slot layout of a launch: eight picks of `rows` rows, packed.
    fn launch(
        ctx: &CudaContext,
        stream: &DeviceStream,
        stack: &DeviceBuffer,
        (k, rows): (usize, usize),
        ids: &[i32; 8],
        n_expert: usize,
        x_q8: &DeviceBuffer,
    ) -> Vec<u8> {
        let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let ids_dev = testkit::upload_bytes(ctx, &id_bytes);
        let y = testkit::upload_filled(ctx, 0xAA, 8 * rows * 4);
        let w_bytes = QuantType::Q4K.weight_bytes(rows * n_expert, k).unwrap();
        ctx.matvec_expert(
            stream,
            ExpertMatvecCall {
                ty: QuantType::Q4K,
                w_stack: stack.range(0, w_bytes).unwrap(),
                ids: ids_dev.range(0, 32).unwrap(),
                x_q8: x_q8.range(0, q8_1_bytes(k, 1).unwrap()).unwrap(),
                x_per_slot: false,
                y: y.range(0, 8 * rows * 4).unwrap(),
                y_slot_stride: rows,
                k,
                rows,
                n_expert,
                n_used: 8,
                glu_gate: None,
            },
        )
        .unwrap();
        stream.synchronize().unwrap();
        testkit::read_bytes(ctx, &y, 8 * rows * 4)
    }

    fn quantized_row(
        ctx: &CudaContext,
        stream: &DeviceStream,
        k: usize,
        rng: &mut testkit::Rng,
    ) -> DeviceBuffer {
        let x: Vec<f32> = (0..k).map(|_| rng.normal() as f32 * 0.05).collect();
        let x_dev = testkit::upload_f32(ctx, &x);
        let bytes = q8_1_bytes(k, 1).unwrap();
        let out = testkit::upload_filled(ctx, 0xAA, bytes);
        let call = QuantizeCall {
            x: x_dev.range(0, k * 4).unwrap(),
            y: out.range(0, bytes).unwrap(),
            k,
            ncols: 1,
        };
        ctx.quantize_q8_1(stream, call).unwrap();
        stream.synchronize().unwrap();
        out
    }

    #[test]
    fn a_zero_bank_gives_finite_zero_rows() {
        if !testkit::device_present("a_zero_bank_gives_finite_zero_rows") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let bank = ExpertBank::new(&ctx, REAL, 8).unwrap();
        let x = quantized_row(&ctx, &stream, 2048, &mut testkit::Rng::new(0xB4AC_0001));
        let rows = launch(
            &ctx,
            &stream,
            bank.gate().buffer(),
            (2048, 512),
            &[0, 1, 2, 3, 4, 5, 6, 7],
            8,
            &x,
        );
        assert!(
            rows.iter().all(|&b| b == 0),
            "a zero bank must give exact zeros"
        );
        assert_eq!(bank.gate().experts(), Some(8));
        assert_eq!(bank.slots(), 8);
        assert_eq!(bank.down().n_rows(), 8 * 2048);
    }

    #[test]
    fn a_filled_slot_matches_the_same_expert_in_a_full_stack_bitwise() {
        if !testkit::device_present("a_filled_slot_matches_the_same_expert_in_a_full_stack_bitwise")
        {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let mut rng = testkit::Rng::new(0xB4AC_0002);
        // A full stack of 16 random experts (gate and down), and a bank whose slot i holds
        // expert (3i + 1) % 16.
        let full_gate = reference::build_weight(QuantType::Q4K, 16 * 512, 2048, &mut rng);
        let full_down = reference::build_weight(QuantType::Q4K, 16 * 2048, 512, &mut rng);
        let gate_dev = testkit::upload_bytes(&ctx, &full_gate);
        let down_dev = testkit::upload_bytes(&ctx, &full_down);
        let bank = ExpertBank::new(&ctx, REAL, 8).unwrap();
        let picks: [usize; 8] = std::array::from_fn(|i| (3 * i + 1) % 16);
        for (slot, &expert) in picks.iter().enumerate() {
            let at = expert * SLICE;
            bank.fill(
                &stream,
                slot,
                &full_gate[at..at + SLICE],
                &full_gate[at..at + SLICE],
                &full_down[at..at + SLICE],
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let x_gate = quantized_row(&ctx, &stream, 2048, &mut rng);
        let x_down = quantized_row(&ctx, &stream, 512, &mut rng);
        let via_full_ids: [i32; 8] = picks.map(|e| e as i32);
        let via_bank_ids: [i32; 8] = std::array::from_fn(|i| i as i32);
        let want_gate = launch(
            &ctx,
            &stream,
            &gate_dev,
            (2048, 512),
            &via_full_ids,
            16,
            &x_gate,
        );
        let got_gate = launch(
            &ctx,
            &stream,
            bank.gate().buffer(),
            (2048, 512),
            &via_bank_ids,
            8,
            &x_gate,
        );
        assert_eq!(
            testkit::first_difference(&want_gate, &got_gate),
            None,
            "gate rows differ"
        );
        let want_down = launch(
            &ctx,
            &stream,
            &down_dev,
            (512, 2048),
            &via_full_ids,
            16,
            &x_down,
        );
        let got_down = launch(
            &ctx,
            &stream,
            bank.down().buffer(),
            (512, 2048),
            &via_bank_ids,
            8,
            &x_down,
        );
        assert_eq!(
            testkit::first_difference(&want_down, &got_down),
            None,
            "down rows differ"
        );
    }

    #[test]
    fn a_fill_touches_only_its_slot_and_refusals_copy_nothing() {
        if !testkit::device_present("a_fill_touches_only_its_slot_and_refusals_copy_nothing") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let bank = ExpertBank::new(&ctx, REAL, 8).unwrap();
        let total = 8 * SLICE;
        let (a, b, c) = (
            vec![0x11u8; SLICE],
            vec![0x22u8; SLICE],
            vec![0x33u8; SLICE],
        );
        // Refusals first: a wrong length and a slot out of range leave the bank all zero.
        assert!(bank.fill(&stream, 2, &a[..SLICE - 1], &b, &c).is_err());
        assert!(bank.fill(&stream, 8, &a, &b, &c).is_err());
        for stack in [bank.gate(), bank.up(), bank.down()] {
            assert!(
                testkit::read_bytes(&ctx, stack.buffer(), total)
                    .iter()
                    .all(|&x| x == 0)
            );
        }
        bank.fill(&stream, 2, &a, &b, &c).unwrap();
        stream.synchronize().unwrap();
        for (stack, byte) in [
            (bank.gate(), 0x11u8),
            (bank.up(), 0x22),
            (bank.down(), 0x33),
        ] {
            let bytes = testkit::read_bytes(&ctx, stack.buffer(), total);
            assert!(bytes[2 * SLICE..3 * SLICE].iter().all(|&x| x == byte));
            assert!(
                bytes[..2 * SLICE]
                    .iter()
                    .chain(&bytes[3 * SLICE..])
                    .all(|&x| x == 0)
            );
        }
    }

    /// 0 x Inf = NaN is the only way the dummy slot of a missed pick can poison a combine: a bank
    /// with only some slots filled must give finite rows from EVERY slot, filled or zero.
    #[test]
    fn a_partly_filled_bank_is_finite_in_every_slot() {
        if !testkit::device_present("a_partly_filled_bank_is_finite_in_every_slot") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let mut rng = testkit::Rng::new(0xB4AC_0004);
        let bank = ExpertBank::new(&ctx, REAL, 16).unwrap();
        let filled = [0usize, 3, 9];
        for &slot in &filled {
            let gate = reference::build_weight(QuantType::Q4K, 512, 2048, &mut rng);
            let up = reference::build_weight(QuantType::Q4K, 512, 2048, &mut rng);
            let down = reference::build_weight(QuantType::Q4K, 2048, 512, &mut rng);
            bank.fill(&stream, slot, &gate, &up, &down).unwrap();
        }
        stream.synchronize().unwrap();
        let x = quantized_row(&ctx, &stream, 2048, &mut rng);
        for first in [0i32, 8] {
            let ids: [i32; 8] = std::array::from_fn(|i| first + i as i32);
            let bytes = launch(
                &ctx,
                &stream,
                bank.gate().buffer(),
                (2048, 512),
                &ids,
                16,
                &x,
            );
            let (words, _) = bytes.as_chunks::<4>();
            for (pick, row) in words.chunks(512).enumerate() {
                let slot = first as usize + pick;
                let values: Vec<f32> = row.iter().map(|w| f32::from_le_bytes(*w)).collect();
                assert!(
                    values.iter().all(|v| v.is_finite()),
                    "slot {slot} gave a non-finite row"
                );
                let zero = values.iter().all(|v| *v == 0.0);
                assert_eq!(
                    zero,
                    !filled.contains(&slot),
                    "slot {slot}: only zero slots give zero rows"
                );
            }
        }
    }
}
