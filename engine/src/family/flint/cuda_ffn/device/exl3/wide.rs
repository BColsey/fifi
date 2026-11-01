//! Device side of the layer-major routed pass: the expert-sorted staging
//! faces, the per-tile scratch, and the tile pipeline that reads them.
//!
//! One tile runs gate, up, the pointwise chain and down for a run of one
//! expert's rows. Every value is published at the same points as the
//! per-block walk, and each row's arithmetic depends only on that row, so a
//! row computed here equals the same row computed in a narrower group except
//! for the matrix kernel the tile height selects.

use super::jobs::{self, MemberPlan};
use super::layer_plan::ExpertVisit;
use super::{ExpertTiles, RowClass, TILE_MAX_ROWS, Tile, TileWalk, bad, expert_shape};
use crate::{
    EngError,
    family::{
        flint::exl3_resident::FlintExpertBank,
        project::cuda::{
            Exl3JobsIo, Exl3MatrixView,
            exl3::{Exl3Io, Exl3Members},
        },
    },
    kernels::cuda::{
        ArenaTensor, CudaContext, DeviceArena, DeviceRange, DeviceStream, plan_arena,
        prefill::row_ops::{RowMulCall, RowShape, RowUnaryCall},
        resident::{exl3::Exl3Job, row_scale::Bf16PublishCall},
    },
};

#[derive(Clone, Copy)]
#[repr(usize)]
pub(super) enum Slot {
    /// Pair rows in expert-major order: the gathered inputs, overwritten tile
    /// by tile with that tile's expert outputs.
    Sorted,
    /// Expert outputs back in (row, rank) order.
    Paired,
    Xh,
    Z,
    Out,
    Gate,
    Up,
    SigmoidRaw,
    Sigmoid,
    SiluRaw,
    Silu,
    ActivatedRaw,
    Activated,
    /// The device job table of one launch batch.
    Jobs,
}

const SLOTS: usize = Slot::Jobs as usize + 1;

/// Most jobs one launch may carry.
const JOBS_MAX: usize = 4096;

/// One allocation holding the two pair faces and the scratch of a single
/// tile or launch batch. The faces grow with the piece; the scratch never does.
pub(crate) struct PairFaces {
    arena: DeviceArena,
    /// Sized for the job-table pass (batch scratch) rather than for single tiles.
    jobs: bool,
}

fn tensors(
    pairs: usize,
    hidden: usize,
    intermediate: usize,
    scratch_rows: usize,
) -> Result<[ArenaTensor; SLOTS], EngError> {
    let bytes = |rows: usize, columns: usize| {
        rows.checked_mul(columns)
            .and_then(|cells| cells.checked_mul(4))
            .ok_or_else(|| bad("layer-major staging bytes overflow"))
    };
    let wide = hidden.max(intermediate);
    let mut sizes = [bytes(scratch_rows, intermediate)?; SLOTS];
    sizes[Slot::Sorted as usize] = bytes(pairs, hidden)?;
    sizes[Slot::Paired as usize] = bytes(pairs, hidden)?;
    for slot in [Slot::Xh, Slot::Z, Slot::Out] {
        sizes[slot as usize] = bytes(scratch_rows, wide)?;
    }
    sizes[Slot::Jobs as usize] = bytes(scratch_rows.min(JOBS_MAX), size_of::<Exl3Job>() / 4 + 1)?;
    Ok(sizes.map(|bytes| ArenaTensor {
        bytes,
        first: 0,
        last: 0,
    }))
}

/// Rows of the per-launch scratch: one tile, or one batch of the job-table pass.
fn scratch_rows(jobs: bool) -> usize {
    if jobs {
        jobs::BATCH_ROWS
    } else {
        TILE_MAX_ROWS
    }
}

impl PairFaces {
    pub(crate) fn new(
        ctx: &CudaContext,
        pairs: usize,
        hidden: usize,
        intermediate: usize,
        jobs: bool,
    ) -> Result<Self, EngError> {
        Ok(Self {
            arena: ctx.arena(&tensors(pairs, hidden, intermediate, scratch_rows(jobs))?)?,
            jobs,
        })
    }

    /// Whether these faces serve the job-table pass.
    pub(super) fn jobs(&self) -> bool {
        self.jobs
    }

    /// Bytes of the allocation for `pairs` routed pairs.
    pub(crate) fn bytes(
        pairs: usize,
        hidden: usize,
        intermediate: usize,
        jobs: bool,
    ) -> Result<u64, EngError> {
        u64::try_from(plan_arena(&tensors(pairs, hidden, intermediate, scratch_rows(jobs))?)?.total)
            .map_err(|_| bad("layer-major staging charge overflows"))
    }

    /// `shape` rows of `slot`, starting `first` rows in.
    pub(super) fn rows(
        &self,
        slot: Slot,
        first: usize,
        shape: RowShape,
    ) -> Result<DeviceRange<'_>, EngError> {
        let base = self.arena.tensor(slot as usize)?;
        let bytes = shape
            .bytes()
            .ok_or_else(|| bad("layer-major staging shape overflows"))?;
        let offset = first
            .checked_mul(shape.columns)
            .and_then(|cells| cells.checked_mul(4))
            .ok_or_else(|| bad("layer-major staging offset overflows"))?;
        if offset
            .checked_add(bytes)
            .is_none_or(|end| end > base.bytes())
        {
            return Err(bad("layer-major staging span exceeds its slot"));
        }
        base.buffer().range(base.offset() + offset, bytes)
    }
}

/// Runs the plan's rounds on the stream the weights were staged on. Staging a
/// later round overwrites slots an earlier round read; both sit on this one
/// stream, so the copies queue behind that round's kernels and nothing waits
/// on the host between rounds.
pub(super) struct WideCuda<'a> {
    pub(super) ctx: &'a CudaContext,
    pub(super) stream: &'a DeviceStream,
    pub(super) experts: &'a FlintExpertBank<'a>,
    pub(super) flag: DeviceRange<'a>,
    pub(super) faces: &'a PairFaces,
    /// Runs each round as job-table launches instead of tile by tile.
    pub(super) jobs: bool,
}

impl TileWalk for WideCuda<'_> {
    fn round(
        &mut self,
        ids: &[i32],
        visit: &mut dyn FnMut(&mut dyn ExpertTiles) -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        let (ctx, stream, flag, faces, jobs) =
            (self.ctx, self.stream, self.flag, self.faces, self.jobs);
        self.experts.with_prepared(ctx, stream, ids, |experts| {
            #[cfg(test)]
            super::super::stage_clock::mark(stream, "moe: bank prepare and copies");
            visit(&mut WideTiles {
                ctx,
                stream,
                flag,
                faces,
                experts,
                jobs,
            })
        })
    }
}

struct WideTiles<'a, 'w> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    flag: DeviceRange<'a>,
    faces: &'a PairFaces,
    experts: &'a FlintExpertBank<'w>,
    jobs: bool,
}

impl WideTiles<'_, '_> {
    /// Projects `x` through `weight` into `destination`, published. The
    /// matrix entry follows the tile height: small rows take the split-K
    /// kernels, taller tiles the opt-in rows entry (SIMT tile below 64 rows,
    /// tensor cores from 64).
    fn project(
        &self,
        weight: Exl3MatrixView<'_>,
        x: DeviceRange<'_>,
        destination: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        let [k, n] = weight.shape;
        let row_bytes = k
            .checked_mul(4)
            .ok_or_else(|| bad("tile width overflows"))?;
        if x.bytes() == 0 || !x.bytes().is_multiple_of(row_bytes) {
            return Err(bad("tile input has incomplete rows"));
        }
        let rows = x.bytes() / row_bytes;
        let class = RowClass::of(rows)?;
        let output = RowShape { rows, columns: n };
        let faces = self.faces;
        let (out, flag) = (faces.rows(Slot::Out, 0, output)?, self.flag);
        let call = weight.linear_call(&Exl3Io {
            x,
            xh: faces.rows(Slot::Xh, 0, RowShape { rows, columns: k })?,
            z: faces.rows(Slot::Z, 0, output)?,
            out,
            flag,
            rows,
            fused: false,
        });
        match class {
            RowClass::Small => self.ctx.exl3_linear(self.stream, call)?,
            RowClass::Tile | RowClass::Tensor => self.ctx.exl3_linear_midrows(self.stream, call)?,
        }
        self.publish(out, destination, output)
    }

    fn publish(
        &self,
        x: DeviceRange<'_>,
        out: DeviceRange<'_>,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.ctx.publish_bf16_rows(
            self.stream,
            Bf16PublishCall {
                x,
                out,
                flag: self.flag,
                rows: shape.rows,
                columns: shape.columns,
            },
        )
    }

    fn multiply(
        &self,
        [x, rhs]: [DeviceRange<'_>; 2],
        out: DeviceRange<'_>,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.ctx.row_multiply(
            self.stream,
            RowMulCall {
                x,
                rhs,
                out,
                flag: self.flag,
                shape,
                rhs_columns: shape.columns,
                fused: false,
            },
        )
    }
}

impl ExpertTiles for WideTiles<'_, '_> {
    fn round(&mut self, visits: &[ExpertVisit]) -> Result<(), EngError> {
        if self.jobs {
            return self.run_jobs(visits);
        }
        for visit in visits {
            for tile in &visit.tiles {
                self.tile(visit.id, *tile)?;
            }
        }
        Ok(())
    }

    fn tile(&mut self, expert: usize, tile: Tile) -> Result<(), EngError> {
        let weights = self.experts.get(expert)?;
        let (hidden, intermediate) = expert_shape(weights)?;
        let rows = tile.rows;
        let (input, wide) = (
            RowShape {
                rows,
                columns: hidden,
            },
            RowShape {
                rows,
                columns: intermediate,
            },
        );
        let faces = self.faces;
        let slot = |slot| faces.rows(slot, 0, wide);
        let x = faces.rows(Slot::Sorted, tile.first, input)?;
        self.project(weights.gate, x, slot(Slot::Gate)?)?;
        self.project(weights.up, x, slot(Slot::Up)?)?;
        self.ctx.row_sigmoid(
            self.stream,
            RowUnaryCall {
                x: slot(Slot::Gate)?,
                out: slot(Slot::SigmoidRaw)?,
                flag: self.flag,
                shape: wide,
            },
        )?;
        self.publish(slot(Slot::SigmoidRaw)?, slot(Slot::Sigmoid)?, wide)?;
        for (lhs, rhs, raw, out) in [
            (Slot::Gate, Slot::Sigmoid, Slot::SiluRaw, Slot::Silu),
            (Slot::Silu, Slot::Up, Slot::ActivatedRaw, Slot::Activated),
        ] {
            self.multiply([slot(lhs)?, slot(rhs)?], slot(raw)?, wide)?;
            self.publish(slot(raw)?, slot(out)?, wide)?;
        }
        // The gate and up kernels have consumed this tile's inputs, so the
        // outputs take their place in the sorted face.
        self.project(weights.down, slot(Slot::Activated)?, x)
    }
}

/// The three projections of one round's experts as bank members.
struct RoundMembers<'a> {
    gate: Exl3Members<'a>,
    up: Exl3Members<'a>,
    down: Exl3Members<'a>,
    member: Vec<u32>,
}

/// The views of one projection of every expert of a round as one bank spanning the round
/// (`plan`): member `i` starts `i * stride` past the lowest expert's component. The experts
/// must agree on shape, bit rate and codebook and live in one buffer per component, which the
/// addressing assumes.
fn members<'a>(
    plan: &MemberPlan,
    views: &[Exl3MatrixView<'a>],
) -> Result<Exl3Members<'a>, EngError> {
    let first = views.first().ok_or_else(|| bad("an empty round"))?;
    let same = |view: &Exl3MatrixView<'_>| {
        view.shape == first.shape
            && view.bits_twice == first.bits_twice
            && view.book == first.book
            && std::ptr::eq(view.trellis.buffer(), first.trellis.buffer())
            && std::ptr::eq(view.suh.buffer(), first.suh.buffer())
            && std::ptr::eq(view.svh.buffer(), first.svh.buffer())
            && view.trellis.bytes() == first.trellis.bytes()
            && view.suh.bytes() == first.suh.bytes()
            && view.svh.bytes() == first.svh.bytes()
    };
    if !views.iter().all(same) {
        return Err(bad(
            "experts of a round differ in shape, rate, book or buffer",
        ));
    }
    let span = |range: DeviceRange<'a>, c: usize| -> Result<DeviceRange<'a>, EngError> {
        let bytes = plan.count.saturating_sub(1) * plan.stride[c] + range.bytes();
        range.buffer().range(plan.base[c], bytes)
    };
    Ok(Exl3Members {
        trellis: span(first.trellis, 0)?,
        suh: span(first.suh, 1)?,
        svh: span(first.svh, 2)?,
        trellis_stride: plan.stride[0] / 2,
        scale_stride: plan.stride[1] / 4,
        members: plan.count,
        shape: first.shape,
        bits_twice: first.bits_twice,
        book: first.book,
    })
}

impl WideTiles<'_, '_> {
    /// Gate, up and down as bank members of this round, with each expert's member index.
    fn round_members(&self, visits: &[ExpertVisit]) -> Result<RoundMembers<'_>, EngError> {
        let weights: Vec<_> = visits
            .iter()
            .map(|visit| self.experts.get(visit.id))
            .collect::<Result<_, _>>()?;
        let roles: [Vec<Exl3MatrixView<'_>>; 3] = [
            weights.iter().map(|w| w.gate).collect(),
            weights.iter().map(|w| w.up).collect(),
            weights.iter().map(|w| w.down).collect(),
        ];
        let plans = roles.each_ref().map(|views| {
            let offsets: Vec<[usize; 3]> = views
                .iter()
                .map(|v| [v.trellis.offset(), v.suh.offset(), v.svh.offset()])
                .collect();
            jobs::plan_members(&offsets)
        });
        let [gate, up, down] = plans;
        let (gate, up, down) = (gate?, up?, down?);
        if gate.member != up.member || gate.member != down.member || gate.count != up.count {
            return Err(bad(
                "expert matrices of a round sit in different bank slots",
            ));
        }
        Ok(RoundMembers {
            gate: members(&gate, &roles[0])?,
            up: members(&up, &roles[1])?,
            down: members(&down, &roles[2])?,
            member: gate.member,
        })
    }

    /// One routed projection over a batch: the job-table launches, then the publication of the
    /// raw rows into `destination`.
    fn project_jobs(
        &self,
        bank: &Exl3Members<'_>,
        batch: &jobs::Batch,
        x: DeviceRange<'_>,
        destination: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        let [k, n] = bank.shape;
        let faces = self.faces;
        let out = faces.rows(
            Slot::Out,
            0,
            RowShape {
                rows: batch.rows,
                columns: n,
            },
        )?;
        let table = faces.rows(
            Slot::Jobs,
            0,
            RowShape {
                rows: batch.jobs.len(),
                columns: size_of::<Exl3Job>() / 4 + 1,
            },
        )?;
        let io = Exl3JobsIo {
            x,
            xh: faces.rows(
                Slot::Xh,
                0,
                RowShape {
                    rows: batch.rows,
                    columns: k,
                },
            )?,
            z: faces.rows(
                Slot::Z,
                0,
                RowShape {
                    rows: batch.rows,
                    columns: n,
                },
            )?,
            out,
            flag: self.flag,
            jobs: table,
            jobs_count: batch.jobs.len(),
            max_rows: batch
                .jobs
                .iter()
                .map(|job| job.rows as usize)
                .max()
                .unwrap_or(1),
            total_rows: batch.rows,
            out_ld: n,
            out_col: 0,
        };
        self.ctx.exl3_jobs(self.stream, bank.jobs_call(&io))?;
        self.publish(
            out,
            destination,
            RowShape {
                rows: batch.rows,
                columns: n,
            },
        )
    }

    /// Runs the round: per batch of rows, gate and up as job tables, the pointwise chain over the
    /// whole batch, and down back into the sorted face.
    fn run_jobs(&self, visits: &[ExpertVisit]) -> Result<(), EngError> {
        let Some(first) = visits.first() else {
            return Ok(());
        };
        let (hidden, intermediate) = expert_shape(self.experts.get(first.id)?)?;
        let bank = self.round_members(visits)?;
        let faces = self.faces;
        for batch in jobs::plan_batches(visits, &bank.member)? {
            let table = Exl3Job::table(&batch.jobs);
            let at = faces.rows(
                Slot::Jobs,
                0,
                RowShape {
                    rows: batch.jobs.len(),
                    columns: size_of::<Exl3Job>() / 4 + 1,
                },
            )?;
            at.buffer().write_at(self.stream, at.offset(), &table)?;
            let wide = RowShape {
                rows: batch.rows,
                columns: intermediate,
            };
            let sorted = faces.rows(
                Slot::Sorted,
                batch.first,
                RowShape {
                    rows: batch.rows,
                    columns: hidden,
                },
            )?;
            let slot = |slot| faces.rows(slot, 0, wide);
            self.project_jobs(&bank.gate, &batch, sorted, slot(Slot::Gate)?)?;
            self.project_jobs(&bank.up, &batch, sorted, slot(Slot::Up)?)?;
            #[cfg(test)]
            super::super::stage_clock::mark(self.stream, "moe: gate and up jobs");
            self.ctx.row_sigmoid(
                self.stream,
                RowUnaryCall {
                    x: slot(Slot::Gate)?,
                    out: slot(Slot::SigmoidRaw)?,
                    flag: self.flag,
                    shape: wide,
                },
            )?;
            self.publish(slot(Slot::SigmoidRaw)?, slot(Slot::Sigmoid)?, wide)?;
            for (lhs, rhs, raw, out) in [
                (Slot::Gate, Slot::Sigmoid, Slot::SiluRaw, Slot::Silu),
                (Slot::Silu, Slot::Up, Slot::ActivatedRaw, Slot::Activated),
            ] {
                self.multiply([slot(lhs)?, slot(rhs)?], slot(raw)?, wide)?;
                self.publish(slot(raw)?, slot(out)?, wide)?;
            }
            // The gate and up kernels have consumed the batch's inputs, so the outputs take
            // their place in the sorted face.
            #[cfg(test)]
            super::super::stage_clock::mark(self.stream, "moe: pointwise chain");
            self.project_jobs(&bank.down, &batch, slot(Slot::Activated)?, sorted)?;
            #[cfg(test)]
            super::super::stage_clock::mark(self.stream, "moe: down jobs");
        }
        Ok(())
    }
}
