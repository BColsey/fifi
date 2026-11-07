//! Device test of the Flash-Next arena plans: allocate the state arena and the decode step arena at
//! a 32K context on the GPU, check every tensor is a checked view of the planned size, and free
//! both. Needs a GPU and about 0.9 GiB (state 0.9 GiB at 32K, step arena a few MiB). No weights.

use super::{
    state::{FlashGeometry, StatePlan, StateSlot},
    state_tests::real,
    step_plan::{Slot, StepOptions, StepPlan},
};
use crate::{
    family::flash_next::moe::context_budget::KvFormat,
    kernels::cuda::{CudaContext, device_present},
};

const POSITIONS: usize = 32_768;

#[test]
#[ignore = "needs a GPU (about 0.9 GiB)"]
fn the_state_and_step_arenas_allocate_at_32k_and_free_cleanly() {
    assert!(
        device_present("Flash-Next arena plans"),
        "a GPU is required"
    );
    let (_, geometry): (_, FlashGeometry) = real();
    let ctx = CudaContext::new().expect("context");
    let state_plan = StatePlan::new(&geometry, POSITIONS, KvFormat::F16).expect("state plan");
    let step_plan =
        StepPlan::new(&geometry, 1, POSITIONS, &StepOptions::default()).expect("step plan");
    {
        let state = ctx.arena(&state_plan.arena_tensors()).expect("state arena");
        let step = ctx.arena(&step_plan.arena_tensors()).expect("step arena");
        assert_eq!(state.bytes(), state_plan.arena_bytes());
        assert_eq!(step.bytes(), step_plan.arena_bytes());
        for &(slot, bytes) in state_plan.slots() {
            let index = state_plan.index(slot).expect("planned");
            assert_eq!(
                state.tensor(index).expect("view").bytes(),
                bytes,
                "{slot:?}"
            );
        }
        for planned in step_plan.slots() {
            let index = step_plan.index(planned.slot).expect("planned");
            assert_eq!(
                step.tensor(index).expect("view").bytes(),
                planned.bytes,
                "{:?}",
                planned.slot
            );
        }
        // The words block and the stream are views of the planned size.
        let last = state_plan.index(StateSlot::Words).expect("words");
        let words = state.tensor(last).expect("words view");
        let stream = step
            .tensor(step_plan.index(Slot::Stream).expect("stream"))
            .expect("view");
        assert!(words.bytes() >= 768 && stream.bytes() == 40 * 1024);
    }
    // Both arenas are dropped here; a third allocation of the same size must succeed again.
    let again = ctx
        .arena(&state_plan.arena_tensors())
        .expect("state arena again");
    assert_eq!(again.bytes(), state_plan.arena_bytes());
}
