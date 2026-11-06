//! Device tests of the rollback slots, on a bare `ResidentState` (no weights): they COMPILE on the
//! host build of the CUDA container and RUN only on a device, under the exclusive GPU lease. None
//! has run. A "row" here writes a recognizable value into the Delta slot 0 and into the KV cell of
//! its own position, the way a step writes them, so a wrong slot, a wrong frontier, a missing
//! component and a stale cell all change the exported state.
use super::*;
use crate::family::{
    project::cuda::WeightLayout,
    qwen36::cuda::{
        common_tests,
        dims::{KvMode, StepOptions},
        state::{device_tip::DeviceTip, export::ExportedState, pinned::PrefixPinPool},
    },
};
use crate::kernels::cuda::resident::step_params::StepParams;

const CONTEXT: usize = 32;

fn state_in(ctx: &CudaContext, stream: &DeviceStream, mode: KvMode) -> ResidentState {
    let options = StepOptions {
        kv_mode: mode,
        context_tokens: CONTEXT,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    };
    ResidentState::allocate(ctx, stream, &common_tests::dims(), &options).unwrap()
}

/// One committed row of value `value`: a start block, an enqueue, the Delta slot 0 and the KV cell
/// of the row's position written with `value`, a commit.
fn row(state: &mut ResidentState, stream: &DeviceStream, value: u8) {
    let pos = state.tokens();
    state
        .write_step_params(stream, &StepParams::new(1, pos as i32).unwrap())
        .unwrap();
    state.enqueue_rows(1).unwrap();
    let bytes = slot_bytes_of(state).unwrap();
    state
        .delta
        .write_at(stream, 0, &vec![value; bytes])
        .unwrap();
    for layer in 0..state.layout.kv.layers {
        let old = state.layout.kv.prefix_ranges(layer, pos).unwrap();
        let new = state.layout.kv.prefix_ranges(layer, pos + 1).unwrap();
        for (old, new) in old.iter().zip(&new) {
            let cell = new.bytes - old.bytes;
            state
                .resident_kv()
                .unwrap()
                .write_at(stream, new.device_offset + old.bytes, &vec![value; cell])
                .unwrap();
        }
    }
    state.commit_step(1, 0).unwrap();
}

/// The states after each row of a run: the reference a rollback must reproduce.
fn reference(
    ctx: &CudaContext,
    stream: &DeviceStream,
    mode: KvMode,
    base_rows: usize,
    values: &[u8],
) -> Vec<ExportedState> {
    let mut state = state_in(ctx, stream, mode);
    for _ in 0..base_rows {
        row(&mut state, stream, 9);
    }
    values
        .iter()
        .map(|&value| {
            row(&mut state, stream, value);
            state.export_state(stream).unwrap()
        })
        .collect()
}

#[test]
#[ignore = "requires native CUDA; run explicitly under the GPU lease"]
fn a_window_rolls_back_to_every_kept_prefix_in_both_kv_modes() {
    let ctx = CudaContext::new().expect("native CUDA context");
    let stream = ctx.stream().unwrap();
    let values = [1u8, 2, 3, 4];
    for mode in [KvMode::Fp16, KvMode::Int8G32] {
        let wanted = reference(&ctx, &stream, mode, 3, &values);
        for kept in 1..=values.len() {
            let mut state = state_in(&ctx, &stream, mode);
            for _ in 0..3 {
                row(&mut state, &stream, 9);
            }
            let mut slots = RollbackSlots::new(&ctx, &stream, &state, values.len()).unwrap();
            assert_eq!(slots.plan().slots(), values.len() - 1);
            assert_eq!(
                slots.bytes(),
                (values.len() - 1) * slot_bytes_of(&state).unwrap()
            );
            slots.begin_window(&stream, &state, values.len()).unwrap();
            for (at, &value) in values.iter().enumerate() {
                row(&mut state, &stream, value);
                if at + 1 < values.len() {
                    slots.capture(&ctx, &stream, &state, at).unwrap();
                }
            }
            let restored = slots
                .finish(&ctx, &stream, &mut state, kept, 3 + kept)
                .unwrap();
            assert_eq!(restored, kept < values.len(), "{mode:?} kept {kept}");
            assert_eq!(state.tokens(), 3 + kept);
            assert!(!slots.is_open());
            assert_eq!(
                state.export_state(&stream).unwrap(),
                wanted[kept - 1],
                "{mode:?}: kept {kept} of {}",
                values.len()
            );
            // The state goes on: the next row starts at the frontier, over the stale cells.
            row(&mut state, &stream, 7);
            assert_eq!(state.tokens(), 3 + kept + 1);
        }
    }
}

#[test]
#[ignore = "requires native CUDA; run explicitly under the GPU lease"]
fn repeated_windows_reuse_the_slots_without_inheriting_their_contents() {
    let ctx = CudaContext::new().expect("native CUDA context");
    let stream = ctx.stream().unwrap();
    let mut state = state_in(&ctx, &stream, KvMode::Fp16);
    row(&mut state, &stream, 9);
    let mut slots = RollbackSlots::new(&ctx, &stream, &state, 4).unwrap();
    for (round, kept) in [1usize, 1, 3, 2, 4, 1].into_iter().enumerate() {
        let base = state.tokens();
        slots.begin_window(&stream, &state, 4).unwrap();
        for at in 0..4 {
            row(&mut state, &stream, (10 + round * 4 + at) as u8);
            if at < 3 {
                slots.capture(&ctx, &stream, &state, at).unwrap();
            }
        }
        slots
            .finish(&ctx, &stream, &mut state, kept, base + kept)
            .unwrap();
        let mut got = vec![0u8; 16];
        state.delta.read_at(&stream, 0, &mut got).unwrap();
        assert!(
            got.iter().all(|&b| b == (10 + round * 4 + kept - 1) as u8),
            "round {round}: the Delta state is that of row {}",
            kept - 1
        );
    }
}

#[test]
#[ignore = "requires native CUDA; run explicitly under the GPU lease"]
fn a_slot_buffer_the_card_cannot_hold_is_a_memory_plan_refusal_and_leaves_the_state() {
    let ctx = CudaContext::new().expect("native CUDA context");
    let stream = ctx.stream().unwrap();
    let mut state = state_in(&ctx, &stream, KvMode::Fp16);
    row(&mut state, &stream, 9);
    let before = state.export_state(&stream).unwrap();
    let plan = RollbackPlan::new(8, 1usize << 40).unwrap();
    match RollbackSlots::new_sized(&ctx, &stream, &state, plan) {
        Err(EngError::MemoryPlan(message)) => {
            assert!(message.contains("device rollback"), "{message}");
        }
        Err(other) => panic!("a MemoryPlan refusal was expected: {other:?}"),
        Ok(_) => panic!("7 TiB of slots cannot be allocated"),
    }
    assert_eq!(state.health(), Health::Valid);
    assert_eq!(state.export_state(&stream).unwrap(), before);
    row(&mut state, &stream, 9);
}

#[test]
#[ignore = "requires native CUDA; run explicitly under the GPU lease"]
fn a_poisoned_busy_or_foreign_state_is_refused_and_a_poisoned_one_is_not_healed() {
    let ctx = CudaContext::new().expect("native CUDA context");
    let stream = ctx.stream().unwrap();
    let mut state = state_in(&ctx, &stream, KvMode::Fp16);
    row(&mut state, &stream, 9);
    let mut slots = RollbackSlots::new(&ctx, &stream, &state, 4).unwrap();
    slots.begin_window(&stream, &state, 4).unwrap();
    row(&mut state, &stream, 1);
    slots.capture(&ctx, &stream, &state, 0).unwrap();
    // Work pending: the device words belong to the device.
    state
        .write_step_params(&stream, &StepParams::new(1, state.tokens() as i32).unwrap())
        .unwrap();
    state.enqueue_rows(1).unwrap();
    assert!(slots.capture(&ctx, &stream, &state, 1).is_err(), "busy");
    state.cancel_enqueue().unwrap();
    // Poisoned between the window and its commit.
    row(&mut state, &stream, 2);
    row(&mut state, &stream, 3);
    state.poison();
    assert!(slots.capture(&ctx, &stream, &state, 1).is_err(), "poisoned");
    let committed = slots.finish(&ctx, &stream, &mut state, 2, 3);
    assert!(committed.is_err(), "a poisoned state is not restored");
    assert_eq!(state.health(), Health::Poisoned, "and it is not healed");
    assert!(state.export_state(&stream).is_err());
    // A reset or an import is the way back.
    state.reset(&stream).unwrap();
    assert_eq!(state.health(), Health::Valid);
}

#[test]
#[ignore = "requires native CUDA; run explicitly under the GPU lease"]
fn a_verdict_that_is_not_this_windows_poisons_and_a_slot_never_captured_is_not_restored() {
    let ctx = CudaContext::new().expect("native CUDA context");
    let stream = ctx.stream().unwrap();
    let mut state = state_in(&ctx, &stream, KvMode::Fp16);
    row(&mut state, &stream, 9);
    let mut slots = RollbackSlots::new(&ctx, &stream, &state, 4).unwrap();
    // Committed rows that do not match base + kept.
    slots.begin_window(&stream, &state, 4).unwrap();
    for at in 0..4 {
        row(&mut state, &stream, 1 + at as u8);
        if at < 3 {
            slots.capture(&ctx, &stream, &state, at).unwrap();
        }
    }
    assert!(slots.finish(&ctx, &stream, &mut state, 2, 99).is_err());
    assert_eq!(state.health(), Health::Poisoned);
    state.reset(&stream).unwrap();
    // A window whose captures stopped early.
    row(&mut state, &stream, 9);
    slots.begin_window(&stream, &state, 4).unwrap();
    for at in 0..4 {
        row(&mut state, &stream, 1 + at as u8);
        if at < 1 {
            slots.capture(&ctx, &stream, &state, at).unwrap();
        }
    }
    assert!(slots.finish(&ctx, &stream, &mut state, 3, 1 + 3).is_err());
    assert_eq!(state.health(), Health::Poisoned);
}

#[test]
#[ignore = "requires native CUDA; run explicitly under the GPU lease"]
fn a_rolled_back_window_leaves_the_device_tip_below_it_valid() {
    let ctx = CudaContext::new().expect("native CUDA context");
    let stream = ctx.stream().unwrap();
    let mut state = state_in(&ctx, &stream, KvMode::Fp16);
    for _ in 0..3 {
        row(&mut state, &stream, 9);
    }
    let mut tip = DeviceTip::new(&ctx, &stream, &state).unwrap();
    let key = tip.capture(&ctx, &stream, &mut state).unwrap();
    let mut slots = RollbackSlots::new(&ctx, &stream, &state, 4).unwrap();
    slots.begin_window(&stream, &state, 4).unwrap();
    for at in 0..4 {
        row(&mut state, &stream, 20 + at as u8);
        if at < 3 {
            slots.capture(&ctx, &stream, &state, at).unwrap();
        }
    }
    slots.finish(&ctx, &stream, &mut state, 1, 3 + 1).unwrap();
    assert!(
        tip.restore(&ctx, &stream, &mut state, &key).unwrap(),
        "the tip still leases the unchanged prefix"
    );
    assert_eq!(state.tokens(), 3);
    let mut got = vec![0u8; tip.bytes()];
    state.delta.read_at(&stream, 0, &mut got).unwrap();
    assert!(
        got.iter().all(|&b| b == 9),
        "the tip restores its own state"
    );
}

#[test]
#[ignore = "requires native CUDA; run explicitly under the GPU lease"]
fn a_rejected_window_leaves_the_pinned_prefix_pool_able_to_finish_recycle_and_restore() {
    let ctx = CudaContext::new().expect("native CUDA context");
    let stream = ctx.stream().unwrap();
    for mode in [KvMode::Fp16, KvMode::Int8G32] {
        let mut state = state_in(&ctx, &stream, mode);
        for _ in 0..3 {
            row(&mut state, &stream, 9);
        }
        let mut pool = PrefixPinPool::new(&ctx, state.layout, 2).unwrap();
        // A save staged at 3 rows, its KV copy still deferred.
        let pending = pool.begin(&ctx, &stream, &state, None).unwrap();
        let first = pool.finish(&ctx, &stream, &state, pending).unwrap();
        assert_eq!(first.tokens(), 3);
        // A window that keeps one row: the device path writes no KV byte below row 3 and does not
        // invalidate the pool.
        let mut slots = RollbackSlots::new(&ctx, &stream, &state, 4).unwrap();
        slots.begin_window(&stream, &state, 4).unwrap();
        for at in 0..4 {
            row(&mut state, &stream, 20 + at as u8);
            if at < 3 {
                slots.capture(&ctx, &stream, &state, at).unwrap();
            }
        }
        slots.finish(&ctx, &stream, &mut state, 1, 3 + 1).unwrap();
        assert!(
            pool.can_recycle(&stream, &state, &first),
            "{mode:?}: the rollback keeps the pool's live source"
        );
        // The recycled save at 4 rows copies only the new KV and restores the one-row state.
        let pending = pool.begin(&ctx, &stream, &state, Some(first)).unwrap();
        let second = pool.finish(&ctx, &stream, &state, pending).unwrap();
        assert_eq!(second.tokens(), 4);
        let want = reference(&ctx, &stream, mode, 3, &[20]).remove(0);
        state.reset(&stream).unwrap();
        pool.restore(&ctx, &stream, &mut state, &second, None)
            .unwrap();
        assert_eq!(state.export_state(&stream).unwrap(), want, "{mode:?}");
    }
}

#[test]
#[ignore = "requires native CUDA; run explicitly under the GPU lease"]
fn a_host_paged_kv_state_refuses_the_slots_and_is_left_alone() {
    use super::super::paged_prefix_tests::{options, snapshot_budget};
    let ctx = CudaContext::new().expect("native CUDA context");
    let stream = ctx.stream().unwrap();
    let opts = options();
    let (_bytes, limit) = snapshot_budget(&common_tests::dims(), &opts).unwrap();
    let state = ResidentState::allocate_paged(&ctx, &stream, &common_tests::dims(), &opts, limit)
        .expect("a paged state");
    match RollbackSlots::new(&ctx, &stream, &state, 4) {
        Err(EngError::BadRequest(message)) => {
            assert!(message.contains("paged"), "{message}");
        }
        Err(other) => panic!("a refusal naming the paged KV was expected: {other:?}"),
        Ok(_) => panic!("a paged KV cache has no resident plane to truncate"),
    }
    assert_eq!(state.health(), Health::Valid);
}
