//! Bounded-row EXL3 dispatch through the shared projection and completion seams.

use super::{
    FfnBlock, FfnShape, Frame, RowGatherCall, RowMulCall, RowUnaryCall, TOP_K, bad, route_reusing,
};

mod cpu_pairs;
mod jobs;
mod layer_plan;
mod routed;
mod tall;
mod wide;

pub(super) use tall::linear_rows;

#[cfg(test)]
mod prefill_control {
    use super::CudaContext;
    use std::{cell::RefCell, marker::PhantomData, rc::Rc};

    #[derive(Debug, serde::Serialize)]
    pub(crate) struct Partition {
        pub(crate) rows: Vec<usize>,
        pub(crate) split: Option<usize>,
    }

    struct Capture {
        regroup: bool,
        partitions: Vec<Partition>,
    }

    thread_local! {
        static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) };
    }

    pub(crate) struct Scope(PhantomData<Rc<()>>);

    impl CudaContext {
        pub(crate) fn test_prefill_control_scope(regroup: bool) -> Scope {
            CAPTURE.with(|slot| {
                assert!(slot.borrow().is_none(), "nested prefill control scope");
                *slot.borrow_mut() = Some(Capture {
                    regroup,
                    partitions: Vec::new(),
                });
            });
            Scope(PhantomData)
        }
    }

    impl Scope {
        pub(crate) fn finish(self) -> Vec<Partition> {
            CAPTURE.with(|slot| slot.borrow_mut().take().unwrap().partitions)
        }
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            CAPTURE.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }

    pub(super) fn partition(rows: impl Iterator<Item = usize>) -> Option<usize> {
        CAPTURE.with(|slot| {
            let mut slot = slot.borrow_mut();
            let capture = slot.as_mut()?;
            let rows: Vec<_> = rows.collect();
            let split = (capture.regroup && !rows.is_empty()).then(|| rows.len().div_ceil(2));
            capture.partitions.push(Partition { rows, split });
            split
        })
    }

    #[test]
    fn regroup_scope_records_halves_and_restores_on_unwind() {
        assert_eq!(partition([1, 2, 3].into_iter()), None);
        let scope = CudaContext::test_prefill_control_scope(true);
        assert_eq!(partition([1, 2, 3].into_iter()), Some(2));
        assert_eq!(partition([8].into_iter()), Some(1));
        assert_eq!(partition([].into_iter()), None);
        let records = scope.finish();
        assert_eq!(records[0].rows, [1, 2, 3]);
        assert_eq!(records[0].split, Some(2));
        assert_eq!(partition([1, 2, 3].into_iter()), None);
        let failure = std::panic::catch_unwind(|| {
            let _scope = CudaContext::test_prefill_control_scope(true);
            panic!("expected scope unwind");
        });
        assert!(failure.is_err());
        assert_eq!(partition([1, 2, 3].into_iter()), None);
        let scope = CudaContext::test_prefill_control_scope(false);
        assert_eq!(partition([1, 2, 3].into_iter()), None);
        assert_eq!(scope.finish()[0].split, None);
    }
}
use crate::family::flint::cuda_ffn::plan::Grouping;
use crate::family::flint::settings::PREFILL_MAX_ROWS;
use crate::{
    EngError,
    family::{
        flint::exl3_resident::{FlintExpertBank, FlintExpertView, FlintFfnView, PickBank},
        project::cuda::{Exl3MatrixView, exl3::Exl3Io},
    },
    kernels::cuda::{
        ArenaTensor, CudaContext, DeviceArena, DeviceBuffer, DeviceRange, DeviceStream,
        prefill::row_ops::RowShape,
        resident::exl3::{Exl3IndexedCall, Exl3LinearCall},
    },
};
use layer_plan::{ExpertTiles, LayerPlan, RowClass, TILE_MAX_ROWS, Tile, TileWalk, walk};
pub(crate) use wide::PairFaces;
use wide::{Slot as Face, WideCuda};

/// Resident experts per layer that one prefill piece of `rows` rows can demand at once: the
/// smallest bank that can serve the piece.
pub(crate) fn cache_prefill_floor(rows: usize) -> Result<usize, EngError> {
    rows.checked_mul(TOP_K)
        .filter(|&pairs| pairs > 0)
        .ok_or_else(|| bad("EXL3 expert cache requires a nonzero row count"))
}

pub(super) fn row_shape(bytes: usize, columns: usize) -> Result<RowShape, EngError> {
    row_shape_to(bytes, columns, PREFILL_MAX_ROWS)
}

/// [`row_shape`] for a pass that may hold up to `cap` rows: the token-major walk caps at eight, a
/// layer-major jobs piece at its piece height.
pub(super) fn row_shape_to(bytes: usize, columns: usize, cap: usize) -> Result<RowShape, EngError> {
    let row_bytes = columns
        .checked_mul(4)
        .filter(|&n| n != 0)
        .ok_or_else(|| bad("EXL3 row width overflows"))?;
    let rows = bytes / row_bytes;
    if !bytes.is_multiple_of(row_bytes) || !(1..=cap).contains(&rows) {
        return Err(bad(&format!("EXL3 requires 1..={cap} complete F32 rows")));
    }
    Ok(RowShape { rows, columns })
}

/// Returns completed raw F32; the caller chooses any BF16 publication boundary.
pub(crate) fn project(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weight: Exl3MatrixView<'_>,
    x: DeviceRange<'_>,
) -> Result<DeviceBuffer, EngError> {
    project_to(ctx, stream, weight, x, PREFILL_MAX_ROWS)
}

/// [`project`] for a pass of up to `cap` rows. Up to eight rows take the checked small-row
/// entry; a taller run goes through [`linear_rows`] and the frame's own completion check.
pub(super) fn project_to(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weight: Exl3MatrixView<'_>,
    x: DeviceRange<'_>,
    cap: usize,
) -> Result<DeviceBuffer, EngError> {
    ctx.ensure_usable()?;
    let [k, n] = weight.shape;
    let input = row_shape_to(x.bytes(), k, cap)?;
    let output = RowShape {
        rows: input.rows,
        columns: n,
    };
    let mut frame = Frame::new(ctx, stream)?;
    let xh = frame.allocate(input)?;
    let z = frame.allocate(output)?;
    let out = frame.allocate(output)?;
    let io = Exl3Io {
        x,
        xh: frame.range(xh)?,
        z: frame.range(z)?,
        out: frame.range(out)?,
        flag: frame.flag()?,
        rows: input.rows,
        fused: false,
    };
    if input.rows <= PREFILL_MAX_ROWS {
        weight.project(ctx, stream, io)?;
    } else {
        linear_rows(ctx, stream, weight, io)?;
    }
    frame.finish(out)
}

fn expert_shape(weight: FlintExpertView<'_>) -> Result<(usize, usize), EngError> {
    let [hidden, intermediate] = weight.gate.shape;
    if weight.up.shape != [hidden, intermediate] || weight.down.shape != [intermediate, hidden] {
        return Err(bad("EXL3 gate/up/down matrix shapes disagree"));
    }
    Ok((hidden, intermediate))
}

/// The FFN block of `bytes` of F32 rows for a pass of up to `cap` rows.
fn block_to(weights: &FlintFfnView<'_>, bytes: usize, cap: usize) -> Result<FfnBlock, EngError> {
    let shape = match weights {
        FlintFfnView::Dense(weight) => {
            let (hidden, intermediate) = expert_shape(*weight)?;
            FfnShape::Dense {
                hidden,
                intermediate,
            }
        }
        FlintFfnView::Moe {
            experts, shared, ..
        } => {
            let (hidden, intermediate) = experts.shape(0)?;
            let (shared_hidden, shared_intermediate) = expert_shape(*shared)?;
            if hidden != shared_hidden {
                return Err(bad("EXL3 shared expert hidden width disagrees"));
            }
            FfnShape::Moe {
                hidden,
                intermediate,
                shared_intermediate,
                experts: experts.len(),
            }
        }
    };
    let hidden = match shape {
        FfnShape::Dense { hidden, .. } | FfnShape::Moe { hidden, .. } => hidden,
    };
    FfnBlock::new(row_shape_to(bytes, hidden, cap)?.rows, shape)
}

pub(super) struct GroupedScratch {
    layout: routed::Layout,
    arena: DeviceArena,
}

impl GroupedScratch {
    pub(super) fn new(
        ctx: &CudaContext,
        rows: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, EngError> {
        let layout = routed::Layout::new(rows.min(routed::MAX_ROWS), hidden, intermediate)?;
        let arena = ctx.arena(&layout.tensors)?;
        Ok(Self { layout, arena })
    }

    pub(super) fn bytes(rows: usize, hidden: usize, intermediate: usize) -> Result<u64, EngError> {
        routed::Layout::new(rows.min(routed::MAX_ROWS), hidden, intermediate)?.bytes()
    }

    fn loan(
        &self,
        slot: routed::Slot,
        first: usize,
        shape: RowShape,
    ) -> Result<routed::Loan, EngError> {
        self.layout.loan(slot, first, shape)
    }

    fn range(&self, loan: routed::Loan) -> Result<DeviceRange<'_>, EngError> {
        let slot = self.arena.tensor(loan.slot as usize)?;
        if loan
            .offset
            .checked_add(loan.bytes)
            .is_none_or(|end| end > slot.bytes())
        {
            return Err(bad("routed loan exceeds its persistent arena slot"));
        }
        let offset = slot
            .offset()
            .checked_add(loan.offset)
            .ok_or_else(|| bad("routed arena offset overflows"))?;
        slot.buffer().range(offset, loan.bytes)
    }
}

pub(super) struct Scratch(DeviceArena);

impl Scratch {
    #[cfg(test)]
    pub(super) fn new(ctx: &CudaContext, weights: &FlintFfnView<'_>) -> Result<Self, EngError> {
        Self::for_rows(ctx, weights, 1)
    }

    pub(super) fn for_rows(
        ctx: &CudaContext,
        weights: &FlintFfnView<'_>,
        rows: usize,
    ) -> Result<Self, EngError> {
        let mut width = 0usize;
        let mut include = |(hidden, intermediate): (usize, usize)| {
            width = width.max(hidden).max(intermediate);
        };
        match weights {
            FlintFfnView::Dense(weight) => include(expert_shape(*weight)?),
            FlintFfnView::Moe {
                experts, shared, ..
            } => {
                for id in 0..experts.len() {
                    include(experts.shape(id)?);
                }
                include(expert_shape(*shared)?);
            }
        }
        let bytes = width
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| EngError::MemoryPlan("EXL3 FFN scratch bytes overflow".to_owned()))?;
        let tensors = [ArenaTensor {
            bytes,
            first: 0,
            last: 0,
        }; 3];
        Ok(Self(ctx.arena(&tensors)?))
    }

    fn range(&self, index: usize, bytes: usize) -> Result<DeviceRange<'_>, EngError> {
        let slot = self.0.tensor(index)?;
        if bytes > slot.bytes() {
            return Err(bad("EXL3 projection exceeds its scratch slot"));
        }
        slot.buffer().range(slot.offset(), bytes)
    }

    fn ranges<'a>(
        &'a mut self,
        weight: Exl3MatrixView<'_>,
        x: DeviceRange<'_>,
        cap: usize,
    ) -> Result<[DeviceRange<'a>; 3], EngError> {
        let [k, n] = weight.shape;
        let input = row_shape_to(x.bytes(), k, cap)?;
        let bytes = RowShape {
            rows: input.rows,
            columns: n,
        }
        .bytes()
        .ok_or_else(|| bad("EXL3 projection output bytes overflow"))?;
        Ok([
            self.range(0, x.bytes())?,
            self.range(1, bytes)?,
            self.range(2, bytes)?,
        ])
    }

    #[cfg(test)]
    pub(super) fn project<'a>(
        &'a mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        weight: Exl3MatrixView<'_>,
        x: DeviceRange<'_>,
    ) -> Result<DeviceRange<'a>, EngError> {
        ctx.ensure_usable()?;
        let [xh, z, out] = self.ranges(weight, x, PREFILL_MAX_ROWS)?;
        let frame = Frame::new(ctx, stream)?;
        weight.project(
            ctx,
            stream,
            Exl3Io {
                x,
                xh,
                z,
                out,
                flag: frame.flag()?,
                rows: 1,
                fused: false,
            },
        )?;
        frame.complete()?;
        Ok(out)
    }
}

impl Frame<'_, '_> {
    pub(super) fn project_bf16(
        &mut self,
        weight: Exl3MatrixView<'_>,
        x: usize,
        scratch: &mut Scratch,
    ) -> Result<usize, EngError> {
        self.ctx.ensure_usable()?;
        let rows = row_shape_to(self.range(x)?.bytes(), weight.shape[0], self.rows_cap)?.rows;
        // Producer fusion stores the published roundtrip straight into owned storage; the rows
        // entry of a tall run has no fused arm, so a tall projection publishes below.
        if self.fused_publish() && rows <= PREFILL_MAX_ROWS {
            let shape = RowShape {
                rows,
                columns: weight.shape[1],
            };
            let published = self.allocate(shape)?;
            let [xh, z, _] = scratch.ranges(weight, self.range(x)?, self.rows_cap)?;
            self.ctx.exl3_linear(
                self.stream,
                weight.linear_call(&Exl3Io {
                    x: self.range(x)?,
                    xh,
                    z,
                    out: self.range(published)?,
                    flag: self.flag()?,
                    rows,
                    fused: true,
                }),
            )?;
            return Ok(published);
        }
        let [xh, z, out] = scratch.ranges(weight, self.range(x)?, self.rows_cap)?;
        tall::linear_rows(
            self.ctx,
            self.stream,
            weight,
            Exl3Io {
                x: self.range(x)?,
                xh,
                z,
                out,
                flag: self.flag()?,
                rows,
                fused: false,
            },
        )?;
        // Publish into owned storage before the mutable scratch loan can end.
        self.input(
            out,
            RowShape {
                rows,
                columns: weight.shape[1],
            },
        )
    }
}

fn expert(
    frame: &mut Frame<'_, '_>,
    weight: FlintExpertView<'_>,
    x: usize,
    scratch: &mut Scratch,
) -> Result<usize, EngError> {
    let (hidden, intermediate) = expert_shape(weight)?;
    let shape = RowShape {
        rows: row_shape_to(frame.range(x)?.bytes(), hidden, frame.rows_cap)?.rows,
        columns: intermediate,
    };
    let gate = frame.project_bf16(weight.gate, x, scratch)?;
    let up = frame.project_bf16(weight.up, x, scratch)?;
    let sig = frame.sigmoid(gate, shape)?;
    let silu = frame.multiply(gate, sig, shape, shape.columns)?;
    let hidden = frame.multiply(silu, up, shape, shape.columns)?;
    frame.project_bf16(weight.down, hidden, scratch)
}

#[derive(Clone, Copy)]
enum RoutedStorage {
    Owned(usize),
    Workspace(routed::Slot),
}

#[derive(Clone, Copy)]
struct RoutedTensor {
    storage: RoutedStorage,
    offset: usize,
    bytes: usize,
}

impl RoutedTensor {
    fn owned(frame: &Frame<'_, '_>, index: usize) -> Result<Self, EngError> {
        let (_, bytes) = frame
            .buffers
            .get(index)
            .ok_or_else(|| bad("routed tensor is not live"))?;
        Ok(Self {
            storage: RoutedStorage::Owned(index),
            offset: 0,
            bytes: *bytes,
        })
    }

    fn owned_index(self, frame: &Frame<'_, '_>) -> Result<usize, EngError> {
        if let RoutedStorage::Owned(index) = self.storage
            && self.offset == 0
            && frame
                .buffers
                .get(index)
                .is_some_and(|(_, bytes)| *bytes == self.bytes)
        {
            return Ok(index);
        }
        Err(bad("routed output is not a complete owned tensor"))
    }

    fn range<'a>(
        self,
        frame: &'a Frame<'_, '_>,
        grouped: Option<&'a GroupedScratch>,
    ) -> Result<DeviceRange<'a>, EngError> {
        match self.storage {
            RoutedStorage::Owned(index) => {
                let (buffer, bytes) = frame
                    .buffers
                    .get(index)
                    .ok_or_else(|| bad("routed tensor is not live"))?;
                if self
                    .offset
                    .checked_add(self.bytes)
                    .is_none_or(|end| end > *bytes)
                {
                    return Err(bad("routed tensor exceeds its owning buffer"));
                }
                buffer.range(self.offset, self.bytes)
            }
            RoutedStorage::Workspace(slot) => grouped
                .ok_or_else(|| bad("grouped routed workspace is missing"))?
                .range(routed::Loan {
                    slot,
                    offset: self.offset,
                    bytes: self.bytes,
                }),
        }
    }

    fn view(self, offset: usize, bytes: usize) -> Result<Self, EngError> {
        if bytes == 0 || offset.checked_add(bytes).is_none_or(|end| end > self.bytes) {
            return Err(bad("routed view exceeds its logical tensor"));
        }
        let offset = self
            .offset
            .checked_add(offset)
            .ok_or_else(|| bad("routed offset overflows"))?;
        Ok(Self {
            storage: self.storage,
            offset,
            bytes,
        })
    }
}

pub(super) struct RoutedCuda<'a, 'ctx, 'stream, 'weights> {
    pub(super) frame: &'a mut Frame<'ctx, 'stream>,
    pub(super) scratch: &'a mut Scratch,
    pub(super) grouped: Option<&'a GroupedScratch>,
    pub(super) experts: &'a FlintExpertBank<'weights>,
    pub(super) packed_pointwise: bool,
    pub(super) grouped_gather: bool,
    /// The experts addressed by the ids the router left on the device: the bank and the frame
    /// buffer holding the ids. Grouped launches then read the picks on the device.
    pub(super) picks: Option<(PickBank<'a>, usize)>,
}

impl<'a> routed::Effects for RoutedCuda<'a, '_, '_, '_> {
    type Tensor = RoutedTensor;
    type Matrix = Exl3MatrixView<'a>;

    fn packed_pointwise(&self) -> bool {
        self.packed_pointwise
    }

    fn allocate(&mut self, shape: RowShape) -> Result<RoutedTensor, EngError> {
        let index = self.frame.allocate(shape)?;
        RoutedTensor::owned(self.frame, index)
    }

    fn view(&mut self, span: routed::Span<RoutedTensor>) -> Result<RoutedTensor, EngError> {
        span.tensor.view(span.offset, span.bytes)
    }

    fn stage(
        &mut self,
        slot: routed::Slot,
        first: usize,
        shape: RowShape,
    ) -> Result<RoutedTensor, EngError> {
        let loan = self.workspace()?.loan(slot, first, shape)?;
        Ok(RoutedTensor {
            storage: RoutedStorage::Workspace(loan.slot),
            offset: loan.offset,
            bytes: loan.bytes,
        })
    }

    fn grouped(
        &mut self,
        calls: &[routed::Projection<Self::Matrix, RoutedTensor>],
    ) -> Result<(), EngError> {
        let workspace = self.workspace()?;
        let mut first = 0;
        let mut args = Vec::with_capacity(calls.len());
        for call in calls {
            let [k, n] = call.weight.shape;
            let x = self.range(call.x)?;
            let input = row_shape(x.bytes(), k)?;
            let output = RowShape {
                rows: input.rows,
                columns: n,
            };
            args.push(call.weight.linear_call(&Exl3Io {
                x,
                xh: workspace.range(workspace.loan(routed::Slot::Xh, first, input)?)?,
                z: workspace.range(workspace.loan(routed::Slot::Z, first, output)?)?,
                out: workspace.range(workspace.loan(routed::Slot::Out, first, output)?)?,
                flag: self.frame.flag()?,
                rows: input.rows,
                fused: false,
            }));
            first = first
                .checked_add(input.rows)
                .ok_or_else(|| bad("grouped rows overflow"))?;
        }
        #[cfg(test)]
        if let Some(split) = prefill_control::partition(args.iter().map(|call| call.rows)) {
            self.frame
                .ctx
                .exl3_grouped(self.frame.stream, &args[..split], self.grouped_gather)?;
            if split < args.len() {
                self.frame.ctx.exl3_grouped(
                    self.frame.stream,
                    &args[split..],
                    self.grouped_gather,
                )?;
            }
            return Ok(());
        }
        self.launch_grouped(args)
    }

    fn indices(&mut self, values: &[i32]) -> Result<RoutedTensor, EngError> {
        let index = self.frame.upload_indices(values)?;
        RoutedTensor::owned(self.frame, index)
    }

    fn expert(&mut self, id: usize) -> Result<routed::Expert<Self::Matrix>, EngError> {
        let weight = self.host_expert(id)?;
        let (hidden, intermediate) = expert_shape(weight)?;
        Ok(routed::Expert {
            matrices: [weight.gate, weight.up, weight.down],
            hidden,
            intermediate,
        })
    }

    fn device_picks(&self) -> bool {
        self.picks.is_some()
    }

    fn pick_expert(&mut self) -> Result<routed::Expert<Self::Matrix>, EngError> {
        if self.picks.is_none() {
            return Err(bad("the routed picks are not on the device"));
        }
        // Every pick is described by expert 0 of the bank; the launch swaps in the member the
        // device names for pick `p` (`launch_grouped`).
        let weight = self.experts.get(0)?;
        let (hidden, intermediate) = expert_shape(weight)?;
        Ok(routed::Expert {
            matrices: [weight.gate, weight.up, weight.down],
            hidden,
            intermediate,
        })
    }

    fn copy(
        &mut self,
        source: routed::Span<RoutedTensor>,
        out: routed::Span<RoutedTensor>,
    ) -> Result<(), EngError> {
        if source.bytes != out.bytes {
            return Err(bad("routed copy lengths disagree"));
        }
        self.frame.ctx.copy_device(
            self.frame.stream,
            self.range(source.tensor.view(source.offset, source.bytes)?)?,
            self.range(out.tensor.view(out.offset, out.bytes)?)?,
        )
    }

    fn gather(
        &mut self,
        x: RoutedTensor,
        indices: RoutedTensor,
        out: RoutedTensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        let row_bytes = shape
            .columns
            .checked_mul(4)
            .filter(|&n| n != 0)
            .ok_or_else(|| bad("routed gather row width overflows"))?;
        if !x.bytes.is_multiple_of(row_bytes) {
            return Err(bad("routed gather source has incomplete rows"));
        }
        self.frame.ctx.row_gather(
            self.frame.stream,
            RowGatherCall {
                x: self.range(x)?,
                indices: self.range(indices)?,
                out: self.range(out)?,
                flag: self.frame.flag()?,
                shape,
                source_rows: x.bytes / row_bytes,
            },
        )
    }

    fn project(&mut self, weight: Self::Matrix, x: RoutedTensor) -> Result<(), EngError> {
        self.frame.ctx.ensure_usable()?;
        let rows = row_shape(x.bytes, weight.shape[0])?.rows;
        let [xh, z, out] =
            self.scratch
                .ranges(weight, x.range(self.frame, self.grouped)?, PREFILL_MAX_ROWS)?;
        self.frame.ctx.exl3_linear(
            self.frame.stream,
            weight.linear_call(&Exl3Io {
                x: x.range(self.frame, self.grouped)?,
                xh,
                z,
                out,
                flag: self.frame.flag()?,
                rows,
                fused: false,
            }),
        )
    }

    fn publish_projection(&mut self, out: RoutedTensor, shape: RowShape) -> Result<(), EngError> {
        let bytes = shape
            .bytes()
            .ok_or_else(|| bad("routed publication shape overflows"))?;
        #[cfg(test)]
        super::count_publish(super::PublishLevel::Routed);
        self.frame
            .input_into(self.scratch.range(2, bytes)?, self.range(out)?, shape)
    }

    fn sigmoid(
        &mut self,
        x: RoutedTensor,
        out: RoutedTensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.frame.ctx.row_sigmoid(
            self.frame.stream,
            RowUnaryCall {
                x: self.range(x)?,
                out: self.range(out)?,
                flag: self.frame.flag()?,
                shape,
            },
        )
    }

    fn multiply(
        &mut self,
        [x, rhs]: [RoutedTensor; 2],
        out: RoutedTensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.frame.ctx.row_multiply(
            self.frame.stream,
            RowMulCall {
                x: self.range(x)?,
                rhs: self.range(rhs)?,
                out: self.range(out)?,
                flag: self.frame.flag()?,
                shape,
                rhs_columns: shape.columns,
                fused: false,
            },
        )
    }

    fn publish(
        &mut self,
        x: RoutedTensor,
        out: RoutedTensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        #[cfg(test)]
        super::count_publish(super::PublishLevel::Routed);
        self.frame
            .input_into(self.range(x)?, self.range(out)?, shape)
    }

    fn prefers_fused_publish(&self) -> bool {
        crate::family::flint::settings::fused_publish_armed()
    }

    fn project_published(
        &mut self,
        weight: Self::Matrix,
        x: RoutedTensor,
        out: RoutedTensor,
        _shape: RowShape,
    ) -> Result<bool, EngError> {
        if !self.prefers_fused_publish() {
            self.project(weight, x)?;
            return Ok(false);
        }
        self.frame.ctx.ensure_usable()?;
        let rows = row_shape(x.bytes, weight.shape[0])?.rows;
        let [xh, z, _] =
            self.scratch
                .ranges(weight, x.range(self.frame, self.grouped)?, PREFILL_MAX_ROWS)?;
        self.frame.ctx.exl3_linear(
            self.frame.stream,
            weight.linear_call(&Exl3Io {
                x: x.range(self.frame, self.grouped)?,
                xh,
                z,
                out: out.range(self.frame, self.grouped)?,
                flag: self.frame.flag()?,
                rows,
                fused: true,
            }),
        )?;
        Ok(true)
    }

    fn grouped_published(
        &mut self,
        calls: &[routed::Projection<Self::Matrix, RoutedTensor>],
        outs: &[RoutedTensor],
    ) -> Result<bool, EngError> {
        if !self.prefers_fused_publish() || outs.is_empty() || calls.len() != outs.len() {
            self.grouped(calls)?;
            return Ok(false);
        }
        let workspace = self.workspace()?;
        let mut first = 0;
        let mut args = Vec::with_capacity(calls.len());
        for (call, out) in calls.iter().zip(outs) {
            let [k, n] = call.weight.shape;
            let x = self.range(call.x)?;
            let input = row_shape(x.bytes(), k)?;
            let output = RowShape {
                rows: input.rows,
                columns: n,
            };
            let out_range = out.range(self.frame, self.grouped)?;
            let bytes = output
                .bytes()
                .ok_or_else(|| bad("grouped routed output overflows"))?;
            if out_range.bytes() < bytes {
                return Err(bad("grouped routed destination is too small"));
            }
            args.push(call.weight.linear_call(&Exl3Io {
                x,
                xh: workspace.range(workspace.loan(routed::Slot::Xh, first, input)?)?,
                z: workspace.range(workspace.loan(routed::Slot::Z, first, output)?)?,
                out: out_range.buffer().range(out_range.offset(), bytes)?,
                flag: self.frame.flag()?,
                rows: input.rows,
                fused: true,
            }));
            first = first
                .checked_add(input.rows)
                .ok_or_else(|| bad("grouped rows overflow"))?;
        }
        self.launch_grouped(args)?;
        Ok(true)
    }

    fn multiply_published(
        &mut self,
        x: [RoutedTensor; 2],
        out: RoutedTensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        if !self.prefers_fused_publish() {
            let raw = self.allocate(shape)?;
            self.multiply(x, raw, shape)?;
            return self.publish(raw, out, shape);
        }
        self.frame.ctx.row_multiply(
            self.frame.stream,
            RowMulCall {
                x: self.range(x[0])?,
                rhs: self.range(x[1])?,
                out: self.range(out)?,
                flag: self.frame.flag()?,
                shape,
                rhs_columns: shape.columns,
                fused: true,
            },
        )
    }
}

impl<'a> RoutedCuda<'a, '_, '_, '_> {
    /// One expert by id for the host-routed walk. With the picks on the device no id is known on
    /// the host, so asking for one is a bug that must not read the wrong weights.
    pub(super) fn host_expert(&self, id: usize) -> Result<FlintExpertView<'a>, EngError> {
        if self.picks.is_some() {
            return Err(bad(
                "a host expert lookup is not available while the picks are on the device",
            ));
        }
        self.experts.get(id)
    }

    /// Launches one grouped set of routed projections: from the host-built member list, or, with
    /// the picks on the device, as an id-indexed launch whose call `p` reads pick `p`.
    fn launch_grouped(&self, args: Vec<Exl3LinearCall<'_>>) -> Result<(), EngError> {
        let ctx = self.frame.ctx;
        let Some((bank, ids_index)) = &self.picks else {
            return ctx.exl3_grouped(self.frame.stream, &args, self.grouped_gather);
        };
        let first = args
            .first()
            .ok_or_else(|| bad("an indexed routed launch needs calls"))?;
        let role = bank
            .roles
            .iter()
            .find(|role| role.trellis.offset() == first.trellis.offset())
            .ok_or_else(|| bad("routed projection is not a role of the pick bank"))?;
        debug_assert!(
            args.iter()
                .all(|call| call.trellis.offset() == first.trellis.offset()),
            "an indexed routed launch reads one projection role"
        );
        let ids = self.frame.buffers[*ids_index].0.range(0, TOP_K * 4)?;
        let cols = ctx.exl3_indexed_cols(first.n)?;
        let calls: Vec<Exl3IndexedCall<'_>> = args
            .into_iter()
            .enumerate()
            .map(|(slot, mut call)| {
                call.trellis = role.trellis;
                call.suh = role.suh;
                call.svh = role.svh;
                call.split_k = 2;
                Exl3IndexedCall {
                    call,
                    ids,
                    slot: slot as u32,
                    members: bank.members,
                    trellis_stride: 0,
                    scale_stride: 0,
                    offsets: Some(bank.table),
                }
            })
            .collect();
        ctx.exl3_grouped_indexed(self.frame.stream, &calls, cols)
    }

    fn workspace(&self) -> Result<&GroupedScratch, EngError> {
        self.grouped
            .ok_or_else(|| bad("grouped routed workspace is missing"))
    }

    fn range(&self, tensor: RoutedTensor) -> Result<DeviceRange<'_>, EngError> {
        tensor.range(self.frame, self.grouped)
    }
}

pub(super) struct ScratchLoans<'a> {
    pub(super) router: Option<DeviceRange<'a>>,
    pub(super) grouped: Option<&'a GroupedScratch>,
    /// Token-health arm: the FFN's status word in the per-token ring. When
    /// present the root frame is parked on it and `Frame::finish` leaves its
    /// health word here for the token-end gate instead of a per-frame
    /// status read.
    pub(super) ffn_flag: Option<DeviceRange<'a>>,
    /// Keep the router's picks on the device for a single-row step over a resident load
    /// (`FIFI_FLINT_DEVICE_PICKS`).
    pub(super) device_picks: bool,
}

fn moe(
    mut frame: Frame<'_, '_>,
    weights: &FlintFfnView<'_>,
    x: usize,
    block: FfnBlock,
    scratch: &mut Scratch,
    loans: ScratchLoans<'_>,
) -> Result<DeviceBuffer, EngError> {
    let FlintFfnView::Moe {
        router,
        experts,
        shared,
        routing_weight_multiplier,
    } = weights
    else {
        return Err(bad("expected EXL3 MoE weights"));
    };
    let mut picks_bank = if loans.device_picks
        && block.rows == 1
        && crate::family::flint::cpu_moe::active().is_none()
    {
        experts.pick_bank(frame.ctx)?
    } else {
        None
    };
    let routed = route_reusing(
        &mut frame,
        *router,
        x,
        block,
        loans.router,
        picks_bank.is_some(),
    )?;
    let served = cpu_pairs::serve(&mut frame, experts, x, block, &routed.ids)?;
    if served.as_ref().is_some_and(cpu_pairs::Served::all)
        && let Some(served) = served
    {
        let (ctx, stream) = (frame.ctx, frame.stream);
        let down = served.into_down(&mut frame)?;
        let output = finish_moe(
            frame,
            scratch,
            (*shared, *routing_weight_multiplier),
            (x, routed.gates, down),
            block,
        )?;
        cpu_pairs::adapt(ctx, stream, experts, &routed.ids)?;
        return Ok(output);
    }
    let ctx = frame.ctx;
    let stream = frame.stream;
    let packed_pointwise = crate::family::flint::settings::packed_pointwise_enabled();
    let grouped_gather = crate::family::flint::settings::grouped_gather_enabled();
    let (ids, grouping) = served
        .as_ref()
        .and_then(cpu_pairs::Served::device)
        .unwrap_or((routed.ids.as_slice(), &routed.grouping));
    let output = experts.with_prepared(ctx, stream, ids, |experts| {
        let input = RoutedTensor::owned(&frame, x)?;
        let down = routed::run(
            &mut RoutedCuda {
                frame: &mut frame,
                scratch,
                grouped: loans.grouped,
                experts,
                packed_pointwise,
                grouped_gather,
                picks: picks_bank.take().zip(routed.device_ids),
            },
            block,
            ids,
            grouping,
            input,
        )?;
        let down = down.owned_index(&frame)?;
        if let Some(served) = &served {
            served.overlay(&mut frame, down)?;
        }
        finish_moe(
            frame,
            scratch,
            (*shared, *routing_weight_multiplier),
            (x, routed.gates, down),
            block,
        )
    })?;
    cpu_pairs::adapt(ctx, stream, experts, &routed.ids)?;
    Ok(output)
}

/// The shared expert, the scaled gates and the rank combine after the routed pair rows `down`
/// exist, in the order the per-block walk always used.
fn finish_moe(
    mut frame: Frame<'_, '_>,
    scratch: &mut Scratch,
    (shared, routing_weight_multiplier): (FlintExpertView<'_>, f32),
    (x, gates, down): (usize, usize, usize),
    block: FfnBlock,
) -> Result<DeviceBuffer, EngError> {
    let shared = expert(&mut frame, shared, x, scratch)?;
    let shape = RowShape {
        rows: block.rows,
        columns: TOP_K,
    };
    let scaled = frame.allocate(shape)?;
    let scales = routing_weight_multiplier.to_ne_bytes().repeat(block.rows);
    let scale = frame.ctx.upload(&scales)?;
    let scale = frame.keep(scale, scales.len());
    // The stored routed up weights stay /128; scale normalized gates before publication.
    frame.ctx.row_multiply(
        frame.stream,
        RowMulCall {
            x: frame.range(gates)?,
            rhs: frame.range(scale)?,
            out: frame.range(scaled)?,
            flag: frame.flag()?,
            shape,
            rhs_columns: 1,
            fused: false,
        },
    )?;
    let output = frame.combine(down, scaled, shared, block)?;
    frame.finish(output)
}

/// Rows of one layer-major piece: the router and the pair gathers are
/// admitted up to 4096 rows per call.
pub(crate) const LAYER_MAX_ROWS: usize = 4096;

/// Routing geometry for a whole piece. The block constructor refuses more
/// than 2048 rows because it also guards the shared-expert row add; routing
/// and the pair gathers have no such limit, and the shared expert here still
/// runs in the narrow blocks.
fn layer_block(rows: usize, hidden: usize, experts: usize) -> Result<FfnBlock, EngError> {
    if !(1..=LAYER_MAX_ROWS).contains(&rows) {
        return Err(bad("layer-major piece rows are outside 1..=4096"));
    }
    Ok(FfnBlock {
        rows,
        hidden,
        experts,
        pairs: rows * TOP_K,
    })
}

/// One MoE layer's routed experts over every row of a piece, and what each
/// narrow block still owes afterwards.
pub(super) struct RoutedLayer<'a, 'w> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    weights: &'a FlintFfnView<'w>,
    faces: &'a PairFaces,
    scratch: &'a mut Scratch,
    /// Scaled routing gates for every row, completed and health-checked.
    gates: DeviceBuffer,
    /// Most rows one block of the piece may carry in [`Self::finish_block`]: the piece itself.
    rows_cap: usize,
}

impl<'a, 'w> RoutedLayer<'a, 'w> {
    /// `norm` holds the published post-attention rows of the whole piece.
    /// Routes all rows at once, gathers the pairs expert-major, runs the
    /// plan's rounds on the stream, and scatters the expert outputs into the
    /// paired face in (row, rank) order. The round-level sync-free schedule is
    /// what the token-major walk cannot offer: it sees at most eight rows.
    pub(super) fn run(
        ctx: &'a CudaContext,
        stream: &'a DeviceStream,
        weights: &'a FlintFfnView<'w>,
        faces: &'a PairFaces,
        scratch: &'a mut Scratch,
        norm: DeviceRange<'_>,
        router_scratch: DeviceRange<'_>,
    ) -> Result<Self, EngError> {
        ctx.ensure_usable()?;
        let FlintFfnView::Moe {
            router,
            experts,
            routing_weight_multiplier,
            ..
        } = weights
        else {
            return Err(bad("expected EXL3 MoE weights"));
        };
        let (hidden, _) = experts.shape(0)?;
        let row_bytes = hidden * 4;
        if norm.bytes() == 0 || !norm.bytes().is_multiple_of(row_bytes) {
            return Err(bad("layer-major norm has incomplete rows"));
        }
        let rows = norm.bytes() / row_bytes;
        let block = layer_block(rows, hidden, experts.len())?;
        let mut frame = Frame::new(ctx, stream)?;
        let x = frame.input(
            norm,
            RowShape {
                rows,
                columns: hidden,
            },
        )?;
        let routed = route_reusing(&mut frame, *router, x, block, Some(router_scratch), false)?;
        #[cfg(test)]
        super::stage_clock::mark(stream, "moe: router and id read");
        // Experts already on the device go first, so only the last round copies weights.
        let grouping = match experts.resident()? {
            Some((_, resident)) => Grouping::ranked(&routed.ids, rows, experts.len(), &resident)?,
            None => routed.grouping,
        };
        let plan = LayerPlan::new(&routed.ids, &grouping, experts.round_capacity())?;
        let pairs = RowShape {
            rows: block.pairs,
            columns: hidden,
        };
        let tokens = frame.upload_indices(&grouping.tokens)?;
        let inverse = frame.upload_indices(&grouping.inverse)?;
        let sorted = faces.rows(Face::Sorted, 0, pairs)?;
        ctx.row_gather(
            stream,
            RowGatherCall {
                x: frame.range(x)?,
                indices: frame.range(tokens)?,
                out: sorted,
                flag: frame.flag()?,
                shape: pairs,
                source_rows: rows,
            },
        )?;
        #[cfg(test)]
        super::stage_clock::mark(stream, "moe: gather rows");
        walk(
            &mut WideCuda {
                ctx,
                stream,
                experts,
                flag: frame.flag()?,
                faces,
                jobs: faces.jobs(),
            },
            &plan,
        )?;
        // Leave the cache as a token-by-token walk would: the experts of the latest rows are the
        // most recent, which is what the decode that follows evicts against.
        experts.touch_recent(&plan.recency())?;
        #[cfg(test)]
        super::stage_clock::mark(stream, "moe: walk tail");
        ctx.row_gather(
            stream,
            RowGatherCall {
                x: sorted,
                indices: frame.range(inverse)?,
                out: faces.rows(Face::Paired, 0, pairs)?,
                flag: frame.flag()?,
                shape: pairs,
                source_rows: block.pairs,
            },
        )?;
        let picks = RowShape {
            rows,
            columns: TOP_K,
        };
        let scaled = frame.allocate(picks)?;
        let scales = routing_weight_multiplier.to_ne_bytes().repeat(rows);
        let scale = frame.ctx.upload(&scales)?;
        let scale = frame.keep(scale, scales.len());
        // The stored routed up weights stay /128; scale normalized gates before publication.
        ctx.row_multiply(
            stream,
            RowMulCall {
                x: frame.range(routed.gates)?,
                rhs: frame.range(scale)?,
                out: frame.range(scaled)?,
                flag: frame.flag()?,
                shape: picks,
                rhs_columns: 1,
                fused: false,
            },
        )?;
        let gates = frame.finish(scaled)?;
        #[cfg(test)]
        super::stage_clock::mark(stream, "moe: gather back and gates");
        Ok(Self {
            ctx,
            stream,
            weights,
            faces,
            scratch,
            gates,
            rows_cap: rows,
        })
    }

    /// What a block owes after its routed rows exist: the shared expert and
    /// the rank combine, in the same frame order as the per-block walk.
    /// `x` is the block's published post-attention rows and `first_row` its
    /// position in the piece.
    pub(super) fn finish_block(
        &mut self,
        first_row: usize,
        x: DeviceRange<'_>,
    ) -> Result<DeviceBuffer, EngError> {
        let FlintFfnView::Moe { shared, .. } = self.weights else {
            return Err(bad("expected EXL3 MoE weights"));
        };
        let block = block_to(self.weights, x.bytes(), self.rows_cap)?;
        let first_pair = first_row * TOP_K;
        let mut frame = Frame::new(self.ctx, self.stream)?.tall(self.rows_cap);
        let x = frame.input(
            x,
            RowShape {
                rows: block.rows,
                columns: block.hidden,
            },
        )?;
        let shared = expert(&mut frame, *shared, x, self.scratch)?;
        #[cfg(test)]
        super::stage_clock::mark(self.stream, "shared expert: projections");
        let pairs = RowShape {
            rows: block.pairs,
            columns: block.hidden,
        };
        let down = adopt(
            &mut frame,
            self.faces.rows(Face::Paired, first_pair, pairs)?,
            pairs,
        )?;
        let picks = RowShape {
            rows: block.rows,
            columns: TOP_K,
        };
        let gates = adopt(
            &mut frame,
            self.gates.range(first_pair * 4, block.pairs * 4)?,
            picks,
        )?;
        #[cfg(test)]
        super::stage_clock::mark(self.stream, "shared expert: pair rows adopted");
        let output = frame.combine(down, gates, shared, block)?;
        #[cfg(test)]
        super::stage_clock::mark(self.stream, "shared expert: combine");
        frame.finish(output)
    }
}

/// Copies `source` into a frame-owned buffer so the frame's own combine can
/// consume rows that live in layer-wide staging.
fn adopt(
    frame: &mut Frame<'_, '_>,
    source: DeviceRange<'_>,
    shape: RowShape,
) -> Result<usize, EngError> {
    let index = frame.allocate(shape)?;
    frame
        .ctx
        .copy_device(frame.stream, source, frame.range(index)?)?;
    Ok(index)
}

/// Test-only entry point with an owned router publication.
#[cfg(test)]
pub(crate) fn forward(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &FlintFfnView<'_>,
    x: DeviceRange<'_>,
) -> Result<DeviceBuffer, EngError> {
    forward_with(ctx, stream, weights, x, false)
}

/// [`forward`] with the choice of keeping the router's picks on the device.
#[cfg(test)]
pub(crate) fn forward_with(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &FlintFfnView<'_>,
    x: DeviceRange<'_>,
    device_picks: bool,
) -> Result<DeviceBuffer, EngError> {
    let block = block_to(weights, x.bytes(), PREFILL_MAX_ROWS)?;
    let grouped = match weights {
        FlintFfnView::Moe { experts, .. } if block.rows <= routed::MAX_ROWS => {
            let (hidden, intermediate) = experts.shape(0)?;
            Some(GroupedScratch::new(ctx, block.rows, hidden, intermediate)?)
        }
        _ => None,
    };
    forward_reusing(
        ctx,
        stream,
        weights,
        x,
        ScratchLoans {
            router: None,
            grouped: grouped.as_ref(),
            ffn_flag: None,
            device_picks,
        },
    )
}

/// Returns completed BF16-in-F32 FFN output, without adding the residual.
pub(super) fn forward_reusing(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &FlintFfnView<'_>,
    x: DeviceRange<'_>,
    loans: ScratchLoans<'_>,
) -> Result<DeviceBuffer, EngError> {
    forward_reusing_to(ctx, stream, weights, x, loans, PREFILL_MAX_ROWS)
}

/// [`forward_reusing`] for blocks of up to `cap` rows. A block above eight rows is a dense
/// layer's feed-forward of a jobs piece; a MoE layer keeps the eight-row block geometry.
pub(super) fn forward_reusing_to(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &FlintFfnView<'_>,
    x: DeviceRange<'_>,
    loans: ScratchLoans<'_>,
    cap: usize,
) -> Result<DeviceBuffer, EngError> {
    ctx.ensure_usable()?;
    let block = block_to(weights, x.bytes(), cap)?;
    let shape = RowShape {
        rows: block.rows,
        columns: block.hidden,
    };
    // Root-frame cleanup must finish before its borrowed scratch is released.
    let mut scratch = Scratch::for_rows(ctx, weights, block.rows)?;
    let mut frame = match loans.ffn_flag {
        Some(flag) => Frame::parked(ctx, stream, flag)?,
        None => Frame::new(ctx, stream)?,
    }
    .tall(cap);
    let x = frame.input(x, shape)?;
    match weights {
        FlintFfnView::Dense(weight) => {
            let output = expert(&mut frame, *weight, x, &mut scratch)?;
            frame.finish(output)
        }
        FlintFfnView::Moe { .. } => moe(frame, weights, x, block, &mut scratch, loans),
    }
}
