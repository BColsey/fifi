//! Bounded staging for grouped resident controller → attention projections.

use super::{CudaContext, DensePlan, DenseStep, EngError, controller_residency};

pub(super) const TILE_ROWS: usize = 8;
const MAX_BYTES: usize = 4 << 20;

pub(super) fn enabled() -> bool {
    std::env::var("FIFI_INKLING_CUDA_PREFILL_CONTROLLER_WORKSPACE").as_deref() == Ok("1")
}

pub(super) fn admit(
    context: &CudaContext,
    [rank, hidden, max_output]: [usize; 3],
    expert_bytes: u64,
) -> Result<Option<DenseStep>, EngError> {
    let Some(plan) = plan([rank, hidden, max_output])? else {
        return Ok(None);
    };
    let Some(floor) = controller_residency::workspace_floor(expert_bytes) else {
        return Ok(None);
    };
    if !fits(context.memory_info()?.free as u64, plan.device_bytes, floor) {
        return Ok(None);
    }
    let step = match DenseStep::new(context, &plan) {
        Ok(step) => step,
        Err(EngError::MemoryPlan(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    // Check the physical charge as allocations may be rounded by CUDA.
    if (context.memory_info()?.free as u64) < floor {
        return Ok(None);
    }
    Ok(Some(step))
}

fn fits(free: u64, bytes: usize, floor: u64) -> bool {
    free.checked_sub(bytes as u64)
        .is_some_and(|after| after >= floor)
}

fn plan([rank, hidden, max_output]: [usize; 3]) -> Result<Option<DensePlan>, EngError> {
    if !(1..=4096).contains(&hidden) || !(1..=4096).contains(&rank) || max_output == 0 {
        return Ok(None);
    }
    let Some(words) = DenseStep::controlled_attention_row_words([rank, hidden], max_output)
        .and_then(|n| n.checked_mul(TILE_ROWS))
        .filter(|&n| n <= MAX_BYTES / size_of::<f32>())
    else {
        return Ok(None);
    };
    let plan = DensePlan::new([words, 2 * hidden], hidden, rank)?;
    Ok((plan.device_bytes <= MAX_BYTES).then_some(plan))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_workspace_bounds_shape_and_reserve() {
        let [rank, hidden, output] = [64, 2304, 8 * 3072];
        let budget = plan([rank, hidden, output]).unwrap().unwrap().device_bytes;
        let words = TILE_ROWS * (24 * hidden + 3 * rank + 3 + output);
        assert!(budget >= 4 * words && budget <= MAX_BYTES);
        for shape in [
            [0, hidden, output],
            [rank, hidden, 0],
            [rank, hidden, usize::MAX],
            [rank, 4097, output],
            [rank, hidden, MAX_BYTES],
        ] {
            assert!(plan(shape).unwrap().is_none(), "{shape:?}");
        }
        let floor = controller_residency::workspace_floor(212459524).unwrap();
        let exact = floor + budget as u64;
        assert!(fits(exact, budget, floor));
        assert!(!fits(exact - 1, budget, floor));
        assert!(!fits(0, budget, floor));
    }
}
