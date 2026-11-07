//! Pure host planning of the opt-in EXL3 MoE block that queues the shared expert inside the
//! routed launches (`FIFI_EXL3_SHARED_IN_ROUTED`): the switch, the pairs of each launch, and the
//! byte layout of the working buffer. No device, no allocation. The CUDA side
//! (`cuda/exl3_moe/merge.rs`, which holds the layout contract in its module doc) runs exactly the
//! plan built here, and the host tests (`exl3_merge_plan_tests.rs`) pin it.

use std::ffi::OsStr;

use crate::EngError;

const F32: usize = size_of::<f32>();

/// The sizes of one MoE block the plan needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) struct PlanDims {
    /// Token rows; the merged block takes one.
    pub(in crate::family) rows: usize,
    pub(in crate::family) hidden: usize,
    /// Experts picked per row.
    pub(in crate::family) top_k: usize,
    /// Width of one routed expert.
    pub(in crate::family) inter: usize,
    /// Width of the shared expert.
    pub(in crate::family) shared_inter: usize,
}

/// The environment switch: unset or `0` keeps the reference MoE block, `1` puts the shared expert
/// in the routed launches. Process-wide, read once, strict: any other value stops the session
/// start. Same bytes as the reference block (device-tested), fewer launches.
pub(in crate::family) const SHARED_IN_ROUTED_VAR: &str = "FIFI_EXL3_SHARED_IN_ROUTED";

/// The most pairs of the merged gate/up launch: the 20-pair kernels' capacity.
const PAIR_LIMIT: usize = 20;

/// Parses [`SHARED_IN_ROUTED_VAR`]: unset is off.
pub(in crate::family) fn shared_in_routed_from_env(value: Option<&OsStr>) -> Result<bool, String> {
    let Some(value) = value else {
        return Ok(false);
    };
    match value.to_str().map(str::trim) {
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err(format!(
            "{SHARED_IN_ROUTED_VAR} must be unset, \"0\" or \"1\""
        )),
    }
}

/// The switch from the environment, or the refusal of a bad value. Read once per process.
pub(in crate::family) fn shared_in_routed_explicit() -> Result<bool, EngError> {
    static EXPLICIT: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    EXPLICIT
        .get_or_init(|| {
            shared_in_routed_from_env(std::env::var_os(SHARED_IN_ROUTED_VAR).as_deref())
        })
        .clone()
        .map_err(EngError::BadRequest)
}

/// The load-line suffix: empty when the switch is off, so the default line is unchanged.
pub(in crate::family) fn load_suffix(on: bool) -> &'static str {
    if on { " exl3_shared_in_routed=on" } else { "" }
}

/// What a MoE block must satisfy for the shared expert to ride in the routed launches.
pub(in crate::family) fn check_dims(dims: &PlanDims) -> Result<(), String> {
    if dims.rows != 1 {
        return Err(format!("the block takes one row, got {}", dims.rows));
    }
    if dims.top_k == 0 || 2 * dims.top_k + 2 > PAIR_LIMIT {
        return Err(format!(
            "{} picks need {} gate and up pairs in one launch, at most {PAIR_LIMIT} fit",
            dims.top_k,
            2 * dims.top_k + 2
        ));
    }
    if dims.shared_inter != dims.inter {
        return Err(format!(
            "the shared expert is {} wide, the routed experts {}: the pairs of one launch share k and n",
            dims.shared_inter, dims.inter
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) enum Proj {
    Gate,
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) enum Member {
    /// The expert the router wrote in `ids[slot]`.
    Routed(u32),
    /// The shared expert: a bank of one member, its ids word is the zero word.
    Shared,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) enum Input {
    /// The normalized row of the block.
    NormRow,
    /// Row `r` of the swiglu output.
    ActRow(usize),
}

/// One pair of a launch. Its scratch row is its index in the launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) struct PlannedPair {
    pub(in crate::family) proj: Proj,
    pub(in crate::family) member: Member,
    pub(in crate::family) input: Input,
    /// Row of the launch's output area the pair writes (`gu` half-row or `down` row).
    pub(in crate::family) out_row: usize,
}

/// The three steps of the merged block, besides the router and the combine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::family) struct BlockPlan {
    pub(in crate::family) gate_up: Vec<PlannedPair>,
    pub(in crate::family) swiglu_rows: usize,
    pub(in crate::family) down: Vec<PlannedPair>,
}

/// The kernels of one grouped or id-indexed launch: input rotate, projection, output rotate.
#[cfg(test)]
const KERNELS_PER_LAUNCH: usize = 3;

#[cfg(test)]
impl BlockPlan {
    /// Kernel launches of the block besides the router and the combine.
    pub(in crate::family) fn kernel_launches(&self) -> usize {
        // The gate/up group and the down group, and the one swiglu between them.
        2 * KERNELS_PER_LAUNCH + 1
    }
}

/// The merged block of a layer that picks `top_k` experts.
pub(in crate::family) fn block_plan(top_k: usize) -> BlockPlan {
    let mut gate_up = Vec::with_capacity(2 * top_k + 2);
    for slot in 0..top_k {
        for (half, proj) in [Proj::Gate, Proj::Up].into_iter().enumerate() {
            gate_up.push(PlannedPair {
                proj,
                member: Member::Routed(slot as u32),
                input: Input::NormRow,
                out_row: 2 * slot + half,
            });
        }
    }
    for (half, proj) in [Proj::Gate, Proj::Up].into_iter().enumerate() {
        gate_up.push(PlannedPair {
            proj,
            member: Member::Shared,
            input: Input::NormRow,
            out_row: 2 * top_k + half,
        });
    }
    let mut down: Vec<PlannedPair> = (0..top_k)
        .map(|slot| PlannedPair {
            proj: Proj::Down,
            member: Member::Routed(slot as u32),
            input: Input::ActRow(slot),
            out_row: slot,
        })
        .collect();
    down.push(PlannedPair {
        proj: Proj::Down,
        member: Member::Shared,
        input: Input::ActRow(top_k),
        out_row: top_k,
    });
    BlockPlan {
        gate_up,
        swiglu_rows: top_k + 1,
        down,
    }
}

/// Kernel launches of the reference block besides the router and the combine: routed gate/up,
/// swiglu, routed down, shared gate/up, swiglu, shared down.
#[cfg(test)]
pub(in crate::family) const fn reference_kernel_launches() -> usize {
    4 * KERNELS_PER_LAUNCH + 2
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) struct Span {
    pub(in crate::family) offset: usize,
    pub(in crate::family) bytes: usize,
}

/// Byte placement of the merged working buffer; every span starts on a 256-byte boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) struct MergeLayout {
    /// Rotated inputs: row `p` of `k` values for pair `p` of the running launch.
    pub(in crate::family) xh: Span,
    /// Split-K partials: row `p` of `n` values.
    pub(in crate::family) z: Span,
    /// `[top_k + 1][gate | up]`, each half `i` wide.
    pub(in crate::family) gu: Span,
    /// `[top_k + 1][i]`.
    pub(in crate::family) act: Span,
    /// `[top_k + 1][h]`.
    pub(in crate::family) down: Span,
    /// Four bytes holding 0, the `ids` word of the shared pairs; nothing writes it.
    pub(in crate::family) zero: Span,
    pub(in crate::family) total: usize,
}

impl MergeLayout {
    pub(in crate::family) fn new(dims: &PlanDims) -> Result<Self, String> {
        check_dims(dims)?;
        let (h, i, k) = (dims.hidden, dims.inter, dims.top_k);
        let pairs = 2 * k + 2;
        let floats = |count: usize| -> Result<usize, String> {
            count
                .checked_mul(F32)
                .ok_or_else(|| "a merged span overflows".to_owned())
        };
        let sizes = [
            floats((pairs * h).max((k + 1) * i))?,
            floats((pairs * i).max((k + 1) * h))?,
            floats((k + 1) * 2 * i)?,
            floats((k + 1) * i)?,
            floats((k + 1) * h)?,
            4,
        ];
        let mut offset = 0usize;
        let mut spans = [Span {
            offset: 0,
            bytes: 0,
        }; 6];
        for (span, bytes) in spans.iter_mut().zip(sizes) {
            *span = Span { offset, bytes };
            offset += bytes.next_multiple_of(256);
        }
        let [xh, z, gu, act, down, zero] = spans;
        Ok(Self {
            xh,
            z,
            gu,
            act,
            down,
            zero,
            total: offset,
        })
    }
}
