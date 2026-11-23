//! Device tests of a grown state: the sequence a session holds is the same sequence in the larger
//! planes. A wrong copy would serve wrong text, so the test compares the whole committed state
//! (the DeltaNet state and every layer's cache prefix) of the state before and after growth, for
//! both cache forms, and that the grown state takes a further import and growth.

use super::super::dims::{KvMode, StepDims, StepOptions};
use super::ResidentState;
use super::export::{DeltaExport, ExportedState, KvExport, plane_sizes};
use crate::EngError;
use crate::family::decode_step::state_layout::{Plane, StateLayout};
use crate::family::project::cuda::WeightLayout;
use crate::family::qwen::qwen36::host::HostShape;
use crate::family::{QWEN3_6_35B_A3B, gguf_model::Inventory};
use crate::kernels::cuda::{CudaContext, device_present};

fn dims() -> Result<StepDims, EngError> {
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64)?;
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    StepDims::new(&shape, &inventory)
}

fn options(mode: KvMode, context_tokens: usize) -> StepOptions {
    StepOptions {
        kv_mode: mode,
        context_tokens,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    }
}

/// A deterministic stream of bytes and small floats.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn byte(&mut self) -> u8 {
        self.next() as u8
    }

    fn float(&mut self) -> f32 {
        (self.next() % 20_001) as f32 / 10_000.0 - 1.0
    }
}

/// A committed state of `tokens` positions filled with noise: finite DeltaNet values, and cache
/// bytes that are finite halves (Fp16) or codes and scales (Int8G32) of any pattern.
fn noisy(
    state: &ResidentState,
    tokens: usize,
    noise: &mut Noise,
) -> Result<ExportedState, EngError> {
    let layout = state.layout();
    let delta = (0..layout.delta.layers)
        .map(|_| DeltaExport {
            recurrent: (0..layout.delta.rec_values)
                .map(|_| noise.float())
                .collect(),
            conv: (0..layout.delta.conv_values)
                .map(|_| noise.float())
                .collect(),
        })
        .collect();
    let mut kv = Vec::new();
    for ordinal in 0..layout.kv.layers {
        let planes = plane_sizes(&layout.kv, ordinal, tokens)?
            .into_iter()
            .map(|len| {
                (0..len)
                    .map(|at| {
                        let byte = noise.byte();
                        // A half is two little-endian bytes: clear the top exponent bit of the high one.
                        if layout.kv.mode == KvMode::Fp16 && at % 2 == 1 {
                            byte & 0x3f
                        } else {
                            byte
                        }
                    })
                    .collect()
            })
            .collect();
        kv.push(KvExport { planes });
    }
    Ok(ExportedState {
        identity: state.identity.clone(),
        tokens,
        delta,
        kv,
    })
}

/// Growth keeps the committed state, for one cache form: import a noisy state at `tokens`
/// positions into a small state, grow it twice (the second time past the first growth's cells),
/// and the export is what was imported each time.
fn grown_phase(mode: KvMode, tokens: usize) -> Result<(), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let d = dims()?;
    let mut state = ResidentState::allocate(&ctx, &stream, &d, &options(mode, 1024))?;
    let mut noise = Noise(0x6d52_0501 + tokens as u64);
    let committed = noisy(&state, tokens, &mut noise)?;
    state.import_state(&stream, &committed)?;
    assert_eq!(
        state.export_state(&stream)?,
        committed,
        "the import round trips"
    );
    let mut at = 1024;
    for cells in [3072, 6144] {
        let grown = state.grown(&ctx, &stream, &d, &options(mode, cells))?;
        assert_eq!(grown.layout().kv.capacity, cells);
        assert_eq!(grown.tokens(), tokens);
        let kept = grown.export_state(&stream)?;
        assert!(
            kept == committed,
            "{mode:?}: the state grown from {at} to {cells} cells differs from the committed one"
        );
        state = grown;
        at = cells;
    }
    // The grown state is a state like any other: it takes a new import over it.
    let other = noisy(&state, tokens / 2, &mut noise)?;
    state.import_state(&stream, &other)?;
    assert_eq!(state.export_state(&stream)?, other);
    // And a size that does not hold its tokens is refused, the state left as it was.
    let refused = state.grown(&ctx, &stream, &d, &options(mode, tokens / 4));
    assert!(matches!(refused, Err(EngError::BadRequest(_))));
    assert_eq!(state.export_state(&stream)?, other);
    Ok(())
}

#[test]
fn a_grown_fp16_state_holds_the_state_it_grew_from() {
    if !device_present("a grown fp16 state") {
        return;
    }
    grown_phase(KvMode::Fp16, 900).expect("fp16 growth");
}

#[test]
fn a_grown_int8_state_holds_the_state_it_grew_from() {
    if !device_present("a grown int8 state") {
        return;
    }
    grown_phase(KvMode::Int8G32, 1000).expect("int8 growth");
}

/// The value stamped at one coordinate of the cache, from the coordinate alone.
fn stamp(layer: usize, plane: usize, head: usize, pos: usize, at: usize) -> u8 {
    let mixed = (layer * 31 + plane * 17 + head * 13 + pos * 7 + at * 3) as u64;
    (mixed.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 56) as u8
}

/// The byte offsets of one cache cell of `plane` in the buffer of `state`, written out from the
/// form of the planes (head-major cells of `D` codes and `D / 32` half scales for Int8G32;
/// position-major rows of `heads * D` halves for Fp16) and the aligned plane starts of the pool.
/// It uses neither the prefix runs nor the export that the growth and the tests above share.
fn cell_bytes(
    d: &StepDims,
    state: &ResidentState,
    layer: usize,
    plane: usize,
    head: usize,
    pos: usize,
) -> std::ops::Range<usize> {
    let kv = &state.layout().kv;
    let attn = &d.shape.attn;
    let (heads, width) = (attn.heads_kv, attn.head_dim);
    let which = kv.planes()[plane];
    let (start, _) = kv.plane_extent(layer, which).expect("plane extent");
    let cells = kv.capacity;
    let (first, len) = match (kv.mode, which) {
        (KvMode::Fp16, _) => (start + (pos * heads + head) * width * 2, width * 2),
        (KvMode::Int8G32, Plane::K | Plane::V) => (start + (head * cells + pos) * width, width),
        (KvMode::Int8G32, _) => {
            let scales = width / 32 * 2;
            (start + (head * cells + pos) * scales, scales)
        }
    };
    first..first + len
}

/// Stamps every cell of a state's cache and DeltaNet buffer from its coordinates, writing the
/// device buffers directly, and marks `tokens` positions committed.
fn stamp_state(
    d: &StepDims,
    state: &mut ResidentState,
    stream: &crate::kernels::cuda::DeviceStream,
    tokens: usize,
) -> Result<Vec<u8>, EngError> {
    let attn = &d.shape.attn;
    let mut kv = vec![0u8; state.resident_kv()?.bytes()];
    for layer in 0..state.layout().kv.layers {
        for plane in 0..state.layout().kv.planes().len() {
            for head in 0..attn.heads_kv {
                for pos in 0..tokens {
                    for (at, byte) in cell_bytes(d, state, layer, plane, head, pos).enumerate() {
                        let value = stamp(layer, plane, head, pos, at);
                        // Fp16 cells are halves: keep the top exponent bit of each clear.
                        kv[byte] = if state.layout().kv.mode == KvMode::Fp16 && at % 2 == 1 {
                            value & 0x3f
                        } else {
                            value
                        };
                    }
                }
            }
        }
    }
    state.resident_kv()?.write_at(stream, 0, &kv)?;
    let delta: Vec<u8> = (0..state.layout().delta.total)
        .map(|at| stamp(99, 0, 0, at, 0) & 0x3f)
        .collect();
    state.delta.write_at(stream, 0, &delta)?;
    state.book.finish_overwrite(tokens);
    Ok(delta)
}

/// Growth puts every committed cell where the grown layout keeps it: stamp each cell of each
/// layer, plane and head from its coordinates in the device buffer of a small state, grow it
/// twice, and read each cell back from the larger buffer at the offset the layout of the larger
/// state gives it. This does not go through `export_state`, so a wrong export order and a wrong
/// copy of it cannot agree with each other.
fn cells_are_where_the_layout_keeps_them(mode: KvMode, tokens: usize) -> Result<(), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let d = dims()?;
    let mut state = ResidentState::allocate(&ctx, &stream, &d, &options(mode, 1024))?;
    let delta = stamp_state(&d, &mut state, &stream, tokens)?;
    for cells in [3072, 6144] {
        state = state.grown(&ctx, &stream, &d, &options(mode, cells))?;
        let mut back = vec![0u8; delta.len()];
        state.delta.read_at(&stream, 0, &mut back)?;
        assert!(
            back == delta,
            "{mode:?}: the DeltaNet state moved in the growth to {cells}"
        );
        for layer in 0..state.layout().kv.layers {
            for plane in 0..state.layout().kv.planes().len() {
                // Only the committed cells are read: the planes past them were never written.
                // An Fp16 plane holds its rows of every head together; an Int8G32 plane holds
                // the cells of one head together.
                let groups: Vec<(usize, usize)> = if mode == KvMode::Fp16 {
                    vec![(0, d.shape.attn.heads_kv - 1)]
                } else {
                    (0..d.shape.attn.heads_kv)
                        .map(|head| (head, head))
                        .collect()
                };
                for (first, last) in groups {
                    let from = cell_bytes(&d, &state, layer, plane, first, 0).start;
                    let to = cell_bytes(&d, &state, layer, plane, last, tokens - 1).end;
                    let mut run = vec![0u8; to - from];
                    state.resident_kv()?.read_at(&stream, from, &mut run)?;
                    for head in first..=last {
                        for pos in 0..tokens {
                            let cell = cell_bytes(&d, &state, layer, plane, head, pos);
                            for (at, byte) in cell.enumerate() {
                                let want = stamp(layer, plane, head, pos, at);
                                let want = if mode == KvMode::Fp16 && at % 2 == 1 {
                                    want & 0x3f
                                } else {
                                    want
                                };
                                assert_eq!(
                                    run[byte - from],
                                    want,
                                    "{mode:?} to {cells}: layer {layer} plane {plane} head {head} position {pos} byte {at}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn the_cells_of_a_grown_fp16_state_are_where_the_layout_keeps_them() {
    if !device_present("fp16 cells after growth") {
        return;
    }
    cells_are_where_the_layout_keeps_them(KvMode::Fp16, 900).expect("fp16 cells");
}

#[test]
fn the_cells_of_a_grown_int8_state_are_where_the_layout_keeps_them() {
    if !device_present("int8 cells after growth") {
        return;
    }
    cells_are_where_the_layout_keeps_them(KvMode::Int8G32, 1000).expect("int8 cells");
}

/// A growth the card cannot hold is a refusal that leaves the state as it was. The size asked for
/// is twice the free memory of the card, so the allocation inside the growth fails on the
/// device; the state still reads back whole and grows to a size the card holds.
fn growth_past_the_card(mode: KvMode, tokens: usize) -> Result<(), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let d = dims()?;
    let mut state = ResidentState::allocate(&ctx, &stream, &d, &options(mode, 1024))?;
    let delta = stamp_state(&d, &mut state, &stream, tokens)?;
    let mut before = vec![0u8; state.resident_kv()?.bytes()];
    state.resident_kv()?.read_at(&stream, 0, &mut before)?;

    let per_cell = (StateLayout::new(&d, &options(mode, 2048))?.bytes()?
        - StateLayout::new(&d, &options(mode, 1024))?.bytes()?)
        / 1024;
    let free = ctx.memory_info()?.free;
    let cells = free * 2 / per_cell;
    let refused = state.grown(&ctx, &stream, &d, &options(mode, cells));
    let Err(error) = refused else {
        panic!(
            "{mode:?}: {cells} cells ({} bytes) fit in {free} free bytes",
            cells * per_cell
        );
    };
    eprintln!("{mode:?}: growth past the card refused: {error:?}");
    assert!(
        !matches!(error, EngError::BadRequest(_) | EngError::ContextLimit(_)),
        "the refusal comes from the device allocation, not a check before it: {error:?}"
    );
    // The state is as it was: tokens, bytes and the ability to grow.
    assert_eq!(state.tokens(), tokens);
    let mut after = vec![0u8; before.len()];
    state.resident_kv()?.read_at(&stream, 0, &mut after)?;
    assert!(
        after == before,
        "{mode:?}: a refused growth changed the cache"
    );
    let mut back = vec![0u8; delta.len()];
    state.delta.read_at(&stream, 0, &mut back)?;
    assert!(
        back == delta,
        "{mode:?}: a refused growth changed the DeltaNet state"
    );
    let grown = state.grown(&ctx, &stream, &d, &options(mode, 3072))?;
    assert_eq!(grown.tokens(), tokens);
    assert_eq!(grown.layout().kv.capacity, 3072);
    Ok(())
}

#[test]
fn a_growth_the_card_cannot_hold_is_refused_and_leaves_the_fp16_state_as_it_was() {
    if !device_present("fp16 growth past the card") {
        return;
    }
    growth_past_the_card(KvMode::Fp16, 900).expect("fp16 refusal");
}

#[test]
fn a_growth_the_card_cannot_hold_is_refused_and_leaves_the_int8_state_as_it_was() {
    if !device_present("int8 growth past the card") {
        return;
    }
    growth_past_the_card(KvMode::Int8G32, 1000).expect("int8 refusal");
}
