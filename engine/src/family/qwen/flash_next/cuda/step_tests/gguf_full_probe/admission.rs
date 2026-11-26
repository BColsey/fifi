//! Conservative checks for this ignored, single-device experiment only.
//! Not production admission: the weight upper bounds reflect the current uploader and
//! reserves are deliberately large; allocator/driver peaks remain subject to the live guard.
use super::*;
use crate::{
    family::qwen::flash_next::{
        cuda::{
            state::{FlashGeometry, StatePlan},
            step_plan::{StepOptions, StepPlan},
        },
        load::gguf::staged_budget::{routed_budget, upload_group_mapped, upload_group_upper},
        moe::context_budget::KvFormat,
    },
    kernels::cuda::{CHUNK_LIMIT, resident::quantize_q8_1::q8_1_bytes},
};

const DEVICE_MARGIN: usize = 2 * 1024 * 1024 * 1024;
const HOST_MARGIN: usize = 8 * 1024 * 1024 * 1024;

fn bad(message: &str) -> EngError {
    EngError::MemoryPlan(format!("GGUF full probe: {message}"))
}

fn sum(values: &[usize]) -> Result<usize, EngError> {
    values
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(*b))
        .ok_or_else(|| bad("size overflow"))
}

fn host_available() -> Result<usize, EngError> {
    let text =
        std::fs::read_to_string("/proc/meminfo").map_err(|_| bad("cannot read host memory"))?;
    let line = text
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))
        .ok_or_else(|| bad("MemAvailable missing"))?;
    let kib = line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| bad("MemAvailable invalid"))?
        .parse::<usize>()
        .map_err(|_| bad("MemAvailable invalid"))?;
    let mut available = kib
        .checked_mul(1024)
        .ok_or_else(|| bad("host memory overflows"))?;
    let cgroup =
        std::fs::read_to_string("/proc/self/cgroup").map_err(|_| bad("cannot read cgroup"))?;
    let relative = cgroup
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .ok_or_else(|| bad("no unified cgroup"))?;
    let path = std::path::Path::new(relative);
    if path.components().any(|c| {
        matches!(
            c,
            std::path::Component::ParentDir | std::path::Component::Prefix(_)
        )
    }) {
        return Err(bad("invalid cgroup path"));
    }
    let root = std::path::Path::new("/sys/fs/cgroup");
    let mut current = root.join(
        path.strip_prefix("/")
            .map_err(|_| bad("cgroup path not absolute"))?,
    );
    loop {
        let max_path = current.join("memory.max");
        if !max_path.exists() && current != root {
            return Err(bad("cgroup memory controller path does not resolve"));
        }
        if max_path.exists() {
            let max = std::fs::read_to_string(max_path)
                .map_err(|_| bad("cannot read cgroup memory.max"))?;
            if max.trim() != "max" {
                let limit = max
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| bad("invalid cgroup memory.max"))?;
                let usage = std::fs::read_to_string(current.join("memory.current"))
                    .map_err(|_| bad("cannot read cgroup memory.current"))?
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| bad("invalid cgroup memory.current"))?;
                available = available.min(limit.saturating_sub(usage));
            }
        }
        if current == root {
            break;
        }
        if !current.pop() || !current.starts_with(root) {
            return Err(bad("invalid cgroup ancestry"));
        }
    }
    Ok(available)
}

pub(super) struct ProbeBudget {
    layer_bounds: Vec<usize>,
    layer_mapped: Vec<usize>,
    final_bound: usize,
    final_mapped: usize,
    state_and_step: usize,
    bank: usize,
    pinned: usize,
    scratch: usize,
}

impl ProbeBudget {
    pub(super) fn check_host(model: &GgufModel) -> Result<(), EngError> {
        let shape = model.config().model_shape()?;
        let slots = shape
            .moe
            .top_k()
            .checked_add(1)
            .ok_or_else(|| bad("slots overflow"))?;
        let routed = routed_budget(model, 0..model.config().layers(), slots)?;
        let required = sum(&[HOST_MARGIN, routed.pinned_records, 2 * CHUNK_LIMIT])?;
        let available = host_available()?;
        eprintln!("full probe host preflight available={available} required={required}");
        if available < required {
            return Err(bad(
                "insufficient host memory after pinned and upload-ring reserve",
            ));
        }
        Ok(())
    }

    pub(super) fn new(
        model: &GgufModel,
        ctx: &CudaContext,
        positions: usize,
    ) -> Result<Self, EngError> {
        let shape = model.config().model_shape()?;
        let layers = model.config().layers();
        let slots = shape
            .moe
            .top_k()
            .checked_add(1)
            .ok_or_else(|| bad("slots overflow"))?;
        let routed = routed_budget(model, 0..layers, slots)?;
        let geometry = FlashGeometry::new(&shape, model.config().kinds())?;
        // The QSA caches as the step this budget admits stores them (`GgufStep::new` reads the
        // setting): with f32 caches the state arena is larger.
        let state = StatePlan::with_qsa_cache(
            &geometry,
            positions,
            KvFormat::F16,
            super::super::super::step::gguf_qsa_cache_storage()?,
        )?;
        let step = StepPlan::gguf_decode(
            &geometry,
            positions,
            &StepOptions::for_sm_count(ctx.sm_count()),
            model.map().max_quantized_input_width()?,
        )?;
        let zeros = shape
            .hyper
            .hidden()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| bad("zero buffer overflow"))?;
        // The step uploads its activation pack beside the arenas and the zero buffer.
        let pack = super::super::super::step::activation_pack_bytes(&geometry)?;
        let state_and_step = sum(&[state.arena_bytes(), step.arena_bytes(), zeros, pack])?;
        let ids = shape
            .moe
            .top_k()
            .checked_mul(size_of::<i32>())
            .ok_or_else(|| bad("slot scratch overflow"))?;
        let qx = q8_1_bytes(shape.hyper.hidden(), 1).ok_or_else(|| bad("Q8 input overflow"))?;
        let qa = q8_1_bytes(shape.moe.inter(), shape.moe.top_k())
            .ok_or_else(|| bad("Q8 activation overflow"))?;
        let scratch = sum(&[ids, qx, qa])?;
        let layer_bounds = (0..layers)
            .map(|layer| upload_group_upper(model, Some(layer)))
            .collect::<Result<Vec<_>, _>>()?;
        let layer_mapped = (0..layers)
            .map(|layer| upload_group_mapped(model, Some(layer)))
            .collect::<Result<Vec<_>, _>>()?;
        let final_bound = upload_group_upper(model, None)?;
        let final_mapped = upload_group_mapped(model, None)?;
        let host_needed = sum(&[HOST_MARGIN, routed.pinned_records, 2 * CHUNK_LIMIT])?;
        let host = host_available()?;
        if host < host_needed {
            return Err(bad(
                "insufficient host memory after pinned and upload-ring reserve",
            ));
        }
        eprintln!(
            "full probe preflight: host_available={host} host_needed={host_needed} layers={layers} slots={slots} state_step_bytes={state_and_step} bank_bytes={} bank_scratch_bytes={scratch} pinned_bytes={} final_upper_bytes={final_bound} max_layer_upper_bytes={} device_margin_bytes={DEVICE_MARGIN}",
            routed.stacks,
            routed.pinned_records,
            layer_bounds.iter().max().unwrap()
        );
        let budget = Self {
            layer_bounds,
            layer_mapped,
            final_bound,
            final_mapped,
            state_and_step,
            bank: routed.stacks,
            pinned: routed.pinned_records,
            scratch,
        };
        let mapped = sum(&budget.layer_mapped)?;
        Self::check_free(
            ctx,
            sum(&[budget.remaining()?, mapped, final_mapped])?,
            "initial-mapped-estimate",
        )?;
        Ok(budget)
    }

    fn group(&self, layer: Option<usize>) -> Result<usize, EngError> {
        match layer {
            Some(i) => self
                .layer_bounds
                .get(i)
                .copied()
                .ok_or_else(|| bad("invalid layer guard")),
            None => Ok(self.final_bound),
        }
    }

    fn remaining(&self) -> Result<usize, EngError> {
        sum(&[self.state_and_step, self.bank, self.scratch, DEVICE_MARGIN])
    }

    fn check_free(ctx: &CudaContext, required: usize, phase: &str) -> Result<(), EngError> {
        let free = ctx.memory_info()?.free;
        eprintln!("full probe memory phase={phase} free={free} required={required}");
        if free < required {
            return Err(bad(
                "insufficient current device memory for next phase and reserved owners",
            ));
        }
        Ok(())
    }

    pub(super) fn group_guard(
        &self,
        ctx: &CudaContext,
        layer: Option<usize>,
        weights: Option<&[CudaWeight]>,
    ) -> Result<(), EngError> {
        let upper = self.group(layer)?;
        let reserve = self.remaining()?;
        let (mapped, final_reserve, future_layers) = match layer {
            Some(i) => (
                self.layer_mapped[i],
                self.final_bound,
                sum(&self.layer_mapped[i + 1..])?,
            ),
            None => (self.final_mapped, 0, 0),
        };
        if let Some(weights) = weights {
            let actual = weights
                .iter()
                .try_fold(0usize, |a, w| a.checked_add(w.device_bytes()))
                .ok_or_else(|| bad("weight bytes overflow"))?;
            if actual > upper || actual > mapped {
                return Err(bad(
                    "staged upload exceeded group payload bound or mapped estimate",
                ));
            }
            eprintln!(
                "full probe uploaded group={layer:?} actual_bytes={actual} mapped_bytes={mapped} payload_upper={upper}"
            );
            Self::check_free(
                ctx,
                sum(&[reserve, final_reserve, future_layers])?,
                "after-group",
            )
        } else {
            Self::check_free(
                ctx,
                sum(&[reserve, final_reserve, future_layers, upper])?,
                "before-group",
            )
        }
    }

    pub(super) fn before_step(&self, ctx: &CudaContext) -> Result<(), EngError> {
        Self::check_free(ctx, self.remaining()?, "before-step")
    }
    pub(super) fn before_bank(&self, model: &GgufModel, ctx: &CudaContext) -> Result<(), EngError> {
        Self::check_host(model)?;
        Self::check_free(
            ctx,
            sum(&[self.bank, self.scratch, DEVICE_MARGIN])?,
            "before-bank",
        )
    }
    pub(super) fn before_execution(&self, ctx: &CudaContext) -> Result<(), EngError> {
        Self::check_free(ctx, DEVICE_MARGIN, "before-execution")
    }
    pub(super) fn pinned_bytes(&self) -> usize {
        self.pinned
    }
    pub(super) fn bank_payload(&self) -> Result<usize, EngError> {
        sum(&[self.bank, self.scratch])
    }
    pub(super) fn step_payload(&self) -> usize {
        self.state_and_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn budget_addition_checks_overflow() {
        assert_eq!(sum(&[1, 2, 3]).unwrap(), 6);
        assert!(sum(&[usize::MAX, 1]).is_err());
    }
}
