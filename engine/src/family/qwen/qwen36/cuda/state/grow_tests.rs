//! Device tests of a grown state: the sequence a session holds is the same sequence in the larger
//! planes. A wrong copy would serve wrong text, so the test compares the whole committed state
//! (the DeltaNet state and every layer's cache prefix) of the state before and after growth, for
//! both cache forms, and that the grown state takes a further import and growth.

use super::super::dims::{KvMode, StepDims, StepOptions};
use super::ResidentState;
use super::export::{DeltaExport, ExportedState, KvExport, plane_sizes};
use crate::EngError;
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
