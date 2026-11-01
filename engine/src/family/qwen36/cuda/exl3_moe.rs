//! The MoE block of ONE layer of the EXL3 step at ONE token (`m = 1`).
//!
//! The router writes the ids, the pick weights and the shared gate on the device, as in the GGUF
//! block. The block has two ways to turn the ids into launches. Built without graph capture it
//! reads the eight ids back, which waits for the stream once per layer, and queues grouped
//! launches over the picked members' ranges. Built with graph capture it queues id-indexed
//! launches that read `ids[slot]` on the device when the kernels run, so it makes no host read and
//! can be recorded in a graph. Both queue the same groups: the gate and up projections of every pick (and of the shared expert) in one group,
//! the SwiGLU, the down projections in one group, and the combine. Every projection reads the F32
//! row and keeps F32 intermediates; the sticky arithmetic flag is read by the step once per token.
//!
//! The order of the pairs of the gate and up group is the layout of `ExpertGu`: pick `s` owns
//! the pairs `2 s` (gate) and `2 s + 1` (up), so each pick's `[gate | up]` row is contiguous for
//! the SwiGLU. The grouped launch requires the rotated inputs, the partials and the outputs of its
//! pairs contiguous in group order; `Scratch` places them.

use super::{
    layer_table::LayerTable,
    moe::{Launch, MoeDims, RouteBufs},
};
use crate::{
    EngError,
    family::{
        gguf_model::Role,
        project::cuda::{
            CudaWeight, CudaWeights,
            exl3::{Exl3Bank, Exl3Io, Exl3Weight},
        },
    },
    kernels::cuda::{
        DeviceBuffer, DeviceRange,
        prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape},
        resident::{
            moe_combine::{CombineCall, CombineOut, CombineShape, SharedTerm},
            router::{RouterCall, RouterOut, RouterShape, RouterWeights},
        },
    },
};

const F32: usize = size_of::<f32>();
const I32: usize = size_of::<i32>();

/// The weights of the MoE block of one layer of an EXL3 load.
pub(super) struct Exl3MoeWeights<'w> {
    /// The raw F32 `[n_expert][hidden]` router matrix.
    pub(super) router: &'w DeviceBuffer,
    /// The raw F32 `[hidden]` shared-gate vector.
    pub(super) router_shared: &'w DeviceBuffer,
    pub(super) gate: &'w Exl3Bank,
    pub(super) up: &'w Exl3Bank,
    pub(super) down: &'w Exl3Bank,
    pub(super) shared_gate: &'w Exl3Weight,
    pub(super) shared_up: &'w Exl3Weight,
    pub(super) shared_down: &'w Exl3Weight,
}

impl<'w> Exl3MoeWeights<'w> {
    /// Resolves layer `layer` in `weights`, checking every shape against `moe`.
    pub(super) fn resolve(
        weights: &'w CudaWeights,
        table: &LayerTable,
        layer: usize,
        moe: &MoeDims,
    ) -> Result<Self, EngError> {
        let name = |role| table.name(layer, role);
        let raw = |role, values: usize| -> Result<&'w DeviceBuffer, EngError> {
            match weights.get(name(role)?) {
                Some(CudaWeight::Raw { buffer, .. }) if buffer.bytes() == values * F32 => {
                    Ok(buffer)
                }
                _ => Err(refuse(format!(
                    "layer {layer} {role:?} is not a raw F32 tensor of {values} values"
                ))),
            }
        };
        let bank = |role, shape: [usize; 2]| -> Result<&'w Exl3Bank, EngError> {
            match weights.get(name(role)?) {
                Some(CudaWeight::Exl3Bank { bank, .. })
                    if bank.count() == moe.n_expert && bank.shape() == shape =>
                {
                    Ok(bank)
                }
                _ => Err(refuse(format!(
                    "layer {layer} {role:?} is not a bank of {} experts of {shape:?}",
                    moe.n_expert
                ))),
            }
        };
        let single = |role, shape: [usize; 2]| -> Result<&'w Exl3Weight, EngError> {
            match weights.get(name(role)?) {
                Some(CudaWeight::Exl3 { weight, .. }) if weight.shape == shape => Ok(weight),
                _ => Err(refuse(format!(
                    "layer {layer} {role:?} is not an EXL3 matrix of {shape:?}"
                ))),
            }
        };
        let (h, i, s) = (moe.hidden, moe.inter, moe.shared_inter);
        Ok(Self {
            router: raw(Role::FfnGateInp, moe.n_expert * h)?,
            router_shared: raw(Role::FfnGateInpShexp, h)?,
            gate: bank(Role::FfnGateExps, [h, i])?,
            up: bank(Role::FfnUpExps, [h, i])?,
            down: bank(Role::FfnDownExps, [i, h])?,
            shared_gate: single(Role::FfnGateShexp, [h, s])?,
            shared_up: single(Role::FfnUpShexp, [h, s])?,
            shared_down: single(Role::FfnDownShexp, [s, h])?,
        })
    }
}

/// The arena ranges of the block: the routing ranges of the GGUF block, the intermediates and
/// the shared projection scratch.
pub(super) struct Exl3MoeBufs<'b> {
    pub(super) route: RouteBufs<'b>,
    /// f32 `[top_k][gate | up]`.
    pub(super) expert_gu: DeviceRange<'b>,
    /// f32 `[top_k][inter]`: the SwiGLU of the picks.
    pub(super) expert_act: DeviceRange<'b>,
    /// f32 `[top_k][hidden]`: the down outputs in pick order.
    pub(super) expert_down: DeviceRange<'b>,
    /// f32 `[gate | up]` of the shared expert.
    pub(super) shared_gu: DeviceRange<'b>,
    /// f32 `[shared_inter]`.
    pub(super) shared_act: DeviceRange<'b>,
    pub(super) shared_out: DeviceRange<'b>,
    /// The rotated inputs, the split-K partials and the arithmetic flag of the projections.
    pub(super) scratch: Scratch<'b>,
}

/// The shared scratch of the EXL3 projections (see `Slot::Exl3Xh`, `Exl3Z`, `Exl3Flag`).
#[derive(Clone, Copy)]
pub(super) struct Scratch<'b> {
    pub(super) xh: DeviceRange<'b>,
    pub(super) z: DeviceRange<'b>,
    pub(super) flag: DeviceRange<'b>,
}

fn refuse(detail: String) -> EngError {
    EngError::BadRequest(format!("EXL3 moe block: {detail}"))
}

/// Row `index` of `width` bytes of `range`.
fn row<'b>(
    range: &DeviceRange<'b>,
    width: usize,
    index: usize,
) -> Result<DeviceRange<'b>, EngError> {
    range.buffer().range(range.offset() + index * width, width)
}

/// The whole allocation behind a weight buffer, as a router weight range.
fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

/// The ids of the picks, read back once the router has run; a faulty router row writes -1.
fn read_picks(
    launch: &Launch<'_>,
    ids: &DeviceRange<'_>,
    dims: &MoeDims,
) -> Result<Vec<usize>, EngError> {
    let mut raw = vec![0u8; dims.top_k * I32];
    ids.buffer()
        .read_at(launch.stream, ids.offset(), &mut raw)?;
    raw.as_chunks::<I32>()
        .0
        .iter()
        .map(|bytes| {
            usize::try_from(i32::from_le_bytes(*bytes))
                .ok()
                .filter(|&id| id < dims.n_expert)
                .ok_or_else(|| {
                    EngError::ModelOutput(
                        "EXL3 moe block: the router picked no valid expert (non-finite logits)"
                            .into(),
                    )
                })
        })
        .collect()
}

/// One pair of a grouped launch: a projection, the row it reads and where the pair's rows go.
struct Pair<'a> {
    view: crate::family::project::cuda::Exl3MatrixView<'a>,
    x: DeviceRange<'a>,
    out: DeviceRange<'a>,
}

/// Queues one grouped launch of `pairs`, all of one shape: pair `p` gets row `p` of the scratch
/// partial and rotated-input areas.
fn grouped(launch: &Launch<'_>, pairs: &[Pair<'_>], scratch: &Scratch<'_>) -> Result<(), EngError> {
    let [k, n] = pairs
        .first()
        .map(|pair| pair.view.shape)
        .ok_or_else(|| refuse("a grouped launch without pairs".into()))?;
    let mut calls = Vec::with_capacity(pairs.len());
    for (index, pair) in pairs.iter().enumerate() {
        let io = Exl3Io {
            x: pair.x,
            xh: row(&scratch.xh, k * F32, index)?,
            z: row(&scratch.z, n * F32, index)?,
            out: pair.out,
            flag: scratch.flag,
            rows: 1,
            fused: false,
        };
        calls.push(pair.view.linear_call(&io));
    }
    launch.ctx.exl3_grouped_step(launch.stream, &calls)
}

/// One pair of an id-indexed launch: the bank, the pick slot whose id chooses the member, the row
/// the pair reads and where its rows go.
struct IndexedPair<'a> {
    bank: &'a Exl3Bank,
    slot: u32,
    x: DeviceRange<'a>,
    out: DeviceRange<'a>,
}

/// Queues one id-indexed launch of `pairs`, all of one shape: pair `p` gets row `p` of the scratch
/// partial and rotated-input areas, and its member is `ids[slot]` read on the device.
fn grouped_indexed(
    launch: &Launch<'_>,
    pairs: &[IndexedPair<'_>],
    ids: DeviceRange<'_>,
    scratch: &Scratch<'_>,
) -> Result<(), EngError> {
    let [k, n] = pairs
        .first()
        .map(|pair| pair.bank.shape())
        .ok_or_else(|| refuse("an indexed launch without pairs".into()))?;
    let mut calls = Vec::with_capacity(pairs.len());
    for (index, pair) in pairs.iter().enumerate() {
        let io = Exl3Io {
            x: pair.x,
            xh: row(&scratch.xh, k * F32, index)?,
            z: row(&scratch.z, n * F32, index)?,
            out: pair.out,
            flag: scratch.flag,
            rows: 1,
            fused: false,
        };
        calls.push(pair.bank.span()?.indexed_call(&io, ids, pair.slot));
    }
    launch
        .ctx
        .exl3_grouped_indexed(launch.stream, &calls, launch.ctx.exl3_indexed_cols(n)?)
}

/// The router: ids, pick weights and the shared gate on the device.
fn route(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let route = &bufs.route;
    launch.ctx.router(
        launch.stream,
        RouterCall {
            shape: RouterShape {
                m: dims.rows,
                hidden: dims.hidden,
                n_expert: dims.n_expert,
                top_k: dims.top_k,
            },
            x: route.norm_row,
            weights: RouterWeights {
                w: whole(weights.router)?,
                w_shared: Some(whole(weights.router_shared)?),
            },
            logits: route.logits,
            out: RouterOut {
                ids: route.ids,
                weights: route.weights,
                shared_gate: Some(route.shared_gate),
                status: route.status,
            },
        },
    )
}

/// The SwiGLU of `rows` rows of `[gate | up]` into `rows` rows of `width` values.
fn swiglu(
    launch: &Launch<'_>,
    gu: DeviceRange<'_>,
    act: DeviceRange<'_>,
    (rows, width): (usize, usize),
) -> Result<(), EngError> {
    launch.ctx.swiglu_rows(
        launch.stream,
        SwigluRowsCall {
            shape: SwigluRowsShape {
                rows,
                n_ff: width,
                up_off: width,
                ld_in: 2 * width,
                ld_out: width,
            },
            gu,
            act,
        },
    )
}

/// The routed experts: gate and up of every pick in one group, the SwiGLU, the down group.
fn routed(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    (picks, dims): (&[usize], &MoeDims),
) -> Result<(), EngError> {
    let (h, i) = (dims.hidden, dims.inter);
    let mut group = Vec::with_capacity(2 * picks.len());
    for (slot, &expert) in picks.iter().enumerate() {
        for (half, bank) in [weights.gate, weights.up].into_iter().enumerate() {
            group.push(Pair {
                view: bank.view(expert)?,
                x: bufs.route.norm_row,
                out: row(&bufs.expert_gu, i * F32, 2 * slot + half)?,
            });
        }
    }
    grouped(launch, &group, &bufs.scratch)?;
    swiglu(launch, bufs.expert_gu, bufs.expert_act, (dims.top_k, i))?;
    let mut group = Vec::with_capacity(picks.len());
    for (slot, &expert) in picks.iter().enumerate() {
        group.push(Pair {
            view: weights.down.view(expert)?,
            x: row(&bufs.expert_act, i * F32, slot)?,
            out: row(&bufs.expert_down, h * F32, slot)?,
        });
    }
    grouped(launch, &group, &bufs.scratch)
}

/// The routed experts with the picks read on the device: the same groups and the same arithmetic
/// as [`routed`], but the member of every pair comes from the router's id array when the kernels
/// run, so the block waits for nothing and can be captured in a graph.
fn routed_on_device(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let (h, i) = (dims.hidden, dims.inter);
    let slots = u32::try_from(dims.top_k).map_err(|_| refuse("top_k does not fit u32".into()))?;
    let ids = bufs.route.ids;
    let mut group = Vec::with_capacity(2 * dims.top_k);
    for slot in 0..slots {
        for (half, bank) in [weights.gate, weights.up].into_iter().enumerate() {
            group.push(IndexedPair {
                bank,
                slot,
                x: bufs.route.norm_row,
                out: row(&bufs.expert_gu, i * F32, 2 * slot as usize + half)?,
            });
        }
    }
    grouped_indexed(launch, &group, ids, &bufs.scratch)?;
    swiglu(launch, bufs.expert_gu, bufs.expert_act, (dims.top_k, i))?;
    let mut group = Vec::with_capacity(dims.top_k);
    for slot in 0..slots {
        group.push(IndexedPair {
            bank: weights.down,
            slot,
            x: row(&bufs.expert_act, i * F32, slot as usize)?,
            out: row(&bufs.expert_down, h * F32, slot as usize)?,
        });
    }
    grouped_indexed(launch, &group, ids, &bufs.scratch)
}

/// The shared expert: gate and up in one group, the SwiGLU, the down projection.
fn shared(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let s = dims.shared_inter;
    let mut group = Vec::with_capacity(2);
    for (half, weight) in [weights.shared_gate, weights.shared_up]
        .into_iter()
        .enumerate()
    {
        group.push(Pair {
            view: weight.view()?,
            x: bufs.route.norm_row,
            out: row(&bufs.shared_gu, s * F32, half)?,
        });
    }
    grouped(launch, &group, &bufs.scratch)?;
    swiglu(launch, bufs.shared_gu, bufs.shared_act, (1, s))?;
    let down = [Pair {
        view: weights.shared_down.view()?,
        x: bufs.shared_act,
        out: bufs.shared_out,
    }];
    grouped(launch, &down, &bufs.scratch)
}

/// `resid_out = resid_in + (picks in pick order + gate * shared)`, as the GGUF block combines.
fn combine(launch: &Launch<'_>, bufs: &Exl3MoeBufs<'_>, dims: &MoeDims) -> Result<(), EngError> {
    launch.ctx.moe_combine(
        launch.stream,
        CombineCall {
            shape: CombineShape {
                m: dims.rows,
                n: dims.hidden,
                top_k: dims.top_k,
                ld_down: dims.hidden,
            },
            res: bufs.route.resid_in,
            down: bufs.expert_down,
            weights: bufs.route.weights,
            shared: Some(SharedTerm {
                gate: bufs.route.shared_gate,
                row: bufs.shared_out,
            }),
            out: CombineOut::To(bufs.route.resid_out),
        },
    )
}

/// The block of one layer on the one stream. With `device_ids` the picks stay on the device and
/// the block makes no host read; otherwise it waits for the stream once, to read the ids.
pub(super) fn run_exl3_moe(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
    device_ids: bool,
) -> Result<(), EngError> {
    if dims.rows != 1 {
        return Err(refuse(format!(
            "the block takes one row, got {}",
            dims.rows
        )));
    }
    route(launch, weights, bufs, dims)?;
    if device_ids {
        routed_on_device(launch, weights, bufs, dims)?;
    } else {
        let picks = read_picks(launch, &bufs.route.ids, dims)?;
        routed(launch, weights, bufs, (&picks, dims))?;
    }
    shared(launch, weights, bufs, dims)?;
    combine(launch, bufs, dims)
}
