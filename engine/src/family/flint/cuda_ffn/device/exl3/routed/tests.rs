use super::super::{
    CudaContext, DeviceStream, FlintFfnView, Frame, GroupedScratch, RoutedCuda, RoutedTensor,
    Scratch, expert as scalar_expert,
};
use super::*;
use crate::family::flint::{
    cuda_ffn::plan::{FfnShape, TOP_K},
    exl3_resident::{FlintResidentLimits, load_flint_resident},
};
use crate::family::project::cuda::Reserve;

const H: usize = 2048;
const I: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Legacy,
    Grouped,
    Packed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backing {
    Owned(usize),
    Workspace,
}

struct Buffer {
    bytes: usize,
    indices: Option<Vec<i32>>,
    backing: Backing,
    offset: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct ObservedOp {
    kind: &'static str,
    core: usize,
    input: [Option<(usize, usize)>; 2],
    out: (usize, usize),
    shape: [usize; 2],
}

/// Symbolic spans and host index metadata only; no tensor values or health state.
struct Observer {
    layout: Layout,
    buffers: Vec<Buffer>,
    allocations: Vec<RowShape>,
    experts: Vec<usize>,
    projections: Vec<[usize; 3]>,
    emissions: Vec<[u32; 7]>,
    noncore: usize,
    operations: Vec<ObservedOp>,
    packed: bool,
    raw: Option<RowShape>,
    /// Armed observers take the producer-fusion paths; `fused_sites` counts the
    /// publishes that a fusing producer absorbed, by kind.
    fused: bool,
    fused_sites: Vec<&'static str>,
}

impl Observer {
    fn new(rows: usize) -> Self {
        Self {
            layout: Layout::new(rows, H, I).unwrap(),
            buffers: vec![Buffer {
                bytes: rows * H * 4,
                indices: None,
                backing: Backing::Owned(0),
                offset: 0,
            }],
            allocations: Vec::new(),
            experts: Vec::new(),
            projections: Vec::new(),
            emissions: Vec::new(),
            noncore: 0,
            operations: Vec::new(),
            packed: false,
            raw: None,
            fused: false,
            fused_sites: Vec::new(),
        }
    }

    fn check(&self, tensor: usize, shape: RowShape) {
        assert_eq!(self.buffers[tensor].bytes, shape.bytes().unwrap());
    }

    fn apart(&self, a: usize, b: usize) {
        let (a, b) = (&self.buffers[a], &self.buffers[b]);
        assert!(
            a.backing != b.backing
                || a.offset + a.bytes <= b.offset
                || b.offset + b.bytes <= a.offset
        );
    }

    fn bounds(&self, span: Span<usize>) {
        assert!(span.offset.checked_add(span.bytes).unwrap() <= self.buffers[span.tensor].bytes);
    }

    fn record(
        &mut self,
        kind: &'static str,
        input: [Option<usize>; 2],
        out: usize,
        shape: RowShape,
    ) {
        if !self.packed {
            return;
        }
        let span = |tensor: usize| {
            let buffer = &self.buffers[tensor];
            assert_eq!(buffer.backing, Backing::Workspace);
            (buffer.offset, buffer.bytes)
        };
        let input = input.map(|tensor| tensor.map(span));
        let out = span(out);
        self.operations.push(ObservedOp {
            kind,
            core: self.emissions.len(),
            input,
            out,
            shape: [shape.rows, shape.columns],
        });
    }
}

impl Effects for Observer {
    type Tensor = usize;
    type Matrix = (usize, [usize; 2]);

    fn packed_pointwise(&self) -> bool {
        self.packed
    }

    fn allocate(&mut self, shape: RowShape) -> Result<usize, EngError> {
        self.allocations.push(shape);
        let index = self.buffers.len();
        self.buffers.push(Buffer {
            bytes: shape.bytes().unwrap(),
            indices: None,
            backing: Backing::Owned(index),
            offset: 0,
        });
        Ok(index)
    }

    fn view(&mut self, span: Span<usize>) -> Result<usize, EngError> {
        self.bounds(span);
        let source = &self.buffers[span.tensor];
        let buffer = Buffer {
            bytes: span.bytes,
            indices: None,
            backing: source.backing,
            offset: source.offset.checked_add(span.offset).unwrap(),
        };
        let index = self.buffers.len();
        self.buffers.push(buffer);
        Ok(index)
    }

    fn stage(&mut self, slot: Slot, first: usize, shape: RowShape) -> Result<usize, EngError> {
        let loan = self.layout.loan(slot, first, shape)?;
        let plan = plan_arena(&self.layout.tensors)?;
        let index = self.buffers.len();
        self.buffers.push(Buffer {
            bytes: loan.bytes,
            indices: None,
            backing: Backing::Workspace,
            offset: plan.offsets[slot as usize] + loan.offset,
        });
        Ok(index)
    }

    fn grouped(&mut self, calls: &[Projection<Self::Matrix, usize>]) -> Result<(), EngError> {
        assert!(!calls.is_empty());
        let (role, [k, n]) = calls[0].weight;
        assert_eq!(
            role,
            self.emissions.len() / 3,
            "gate/up/down projection order"
        );
        let mut first = 0;
        for call in calls {
            assert_eq!(call.weight, (role, [k, n]));
            let bytes = self.buffers[call.x].bytes;
            assert!(bytes.is_multiple_of(k * 4));
            let rows = bytes / (k * 4);
            assert!((1..=8).contains(&rows));
            self.layout
                .loan(Slot::Xh, first, RowShape { rows, columns: k })?;
            self.layout
                .loan(Slot::Z, first, RowShape { rows, columns: n })?;
            self.layout
                .loan(Slot::Out, first, RowShape { rows, columns: n })?;
            self.projections.push([rows, k, n]);
            first += rows;
        }
        let (emissions, complete) = CudaContext::test_exl3_grouped_emissions([first, k, n], 0)?;
        assert!(complete);
        self.emissions.extend(emissions);
        Ok(())
    }

    fn indices(&mut self, values: &[i32]) -> Result<usize, EngError> {
        let index = self.buffers.len();
        self.buffers.push(Buffer {
            bytes: values.len() * 4,
            indices: Some(values.to_vec()),
            backing: Backing::Owned(index),
            offset: 0,
        });
        Ok(index)
    }

    fn expert(&mut self, id: usize) -> Result<Expert<Self::Matrix>, EngError> {
        assert!(id < 256);
        self.experts.push(id);
        Ok(Expert {
            matrices: [(0, [H, I]), (1, [H, I]), (2, [I, H])],
            hidden: H,
            intermediate: I,
        })
    }

    fn copy(&mut self, source: Span<usize>, out: Span<usize>) -> Result<(), EngError> {
        self.bounds(source);
        self.bounds(out);
        assert_eq!(source.bytes, out.bytes);
        assert_ne!(
            self.buffers[source.tensor].backing,
            self.buffers[out.tensor].backing
        );
        Ok(())
    }

    fn gather(
        &mut self,
        x: usize,
        indices: usize,
        out: usize,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.check(out, shape);
        let rows = self.buffers[x].bytes / (shape.columns * 4);
        let indices = self.buffers[indices].indices.as_ref().unwrap();
        assert_eq!(indices.len(), shape.rows);
        assert!(
            indices
                .iter()
                .all(|&index| index >= 0 && (index as usize) < rows)
        );
        Ok(())
    }

    fn project(&mut self, (_, [k, n]): Self::Matrix, x: usize) -> Result<(), EngError> {
        let bytes = self.buffers[x].bytes;
        assert!(bytes.is_multiple_of(k * 4));
        let rows = bytes / (k * 4);
        let (emissions, complete) = CudaContext::test_exl3_emissions([rows, k, n], 2, false, 0)?;
        assert!(complete);
        self.projections.push([rows, k, n]);
        self.emissions.extend(emissions);
        self.raw = Some(RowShape { rows, columns: n });
        Ok(())
    }

    fn publish_projection(&mut self, out: usize, shape: RowShape) -> Result<(), EngError> {
        let raw = self.raw.take().unwrap();
        assert_eq!((raw.rows, raw.columns), (shape.rows, shape.columns));
        self.check(out, shape);
        self.noncore += 1;
        Ok(())
    }

    fn prefers_fused_publish(&self) -> bool {
        self.fused
    }

    fn project_published(
        &mut self,
        weight: Self::Matrix,
        x: usize,
        out: usize,
        shape: RowShape,
    ) -> Result<bool, EngError> {
        self.project(weight, x)?;
        // The fusing producer wrote the published values: nothing is left raw.
        let raw = self.raw.take().unwrap();
        assert_eq!((raw.rows, raw.columns), (shape.rows, shape.columns));
        self.check(out, shape);
        self.fused_sites.push("project");
        Ok(true)
    }

    fn grouped_published(
        &mut self,
        calls: &[Projection<Self::Matrix, usize>],
        outs: &[usize],
    ) -> Result<bool, EngError> {
        assert_eq!(calls.len(), outs.len());
        self.grouped(calls)?;
        for (call, &out) in calls.iter().zip(outs) {
            let (_, [k, n]) = call.weight;
            let rows = self.buffers[call.x].bytes / (k * 4);
            self.check(out, RowShape { rows, columns: n });
            self.fused_sites.push("grouped");
        }
        Ok(true)
    }

    fn multiply_published(
        &mut self,
        x: [usize; 2],
        out: usize,
        shape: RowShape,
    ) -> Result<(), EngError> {
        for input in x {
            self.check(input, shape);
            self.apart(input, out);
        }
        self.check(out, shape);
        self.noncore += 1;
        self.fused_sites.push("multiply");
        Ok(())
    }

    fn sigmoid(&mut self, x: usize, out: usize, shape: RowShape) -> Result<(), EngError> {
        self.check(x, shape);
        self.check(out, shape);
        self.apart(x, out);
        self.record("sigmoid", [Some(x), None], out, shape);
        self.noncore += 1;
        Ok(())
    }

    fn multiply(&mut self, x: [usize; 2], out: usize, shape: RowShape) -> Result<(), EngError> {
        for input in x {
            self.check(input, shape);
            self.apart(input, out);
        }
        self.check(out, shape);
        self.record("multiply", x.map(Some), out, shape);
        self.noncore += 1;
        Ok(())
    }

    fn publish(&mut self, x: usize, out: usize, shape: RowShape) -> Result<(), EngError> {
        self.check(x, shape);
        self.check(out, shape);
        self.apart(x, out);
        self.record("publish", [Some(x), None], out, shape);
        self.noncore += 1;
        Ok(())
    }
}

fn block(rows: usize) -> FfnBlock {
    FfnBlock::new(
        rows,
        FfnShape::Moe {
            hidden: H,
            intermediate: I,
            shared_intermediate: 1024,
            experts: 256,
        },
    )
    .unwrap()
}

fn observe(ids: &[i32], mode: Mode) -> Observer {
    observe_armed(ids, mode, false)
}

fn observe_armed(ids: &[i32], mode: Mode, fused: bool) -> Observer {
    let rows = ids.len() / 6;
    let grouping = Grouping::new(ids, rows, 256).unwrap();
    let mut observer = Observer::new(rows);
    observer.packed = mode == Mode::Packed;
    observer.fused = fused;
    let output = if mode != Mode::Legacy {
        run(&mut observer, block(rows), ids, &grouping, 0)
    } else {
        legacy(&mut observer, block(rows), ids, &grouping, 0)
    }
    .unwrap();
    observer.check(
        output,
        RowShape {
            rows: rows * 6,
            columns: H,
        },
    );
    observer
}

fn check_packed(ids: &[i32], baseline: &Observer) {
    use Slot::*;
    let packed = observe(ids, Mode::Packed);
    assert_eq!(packed.noncore, 9);
    let arena = plan_arena(&packed.layout.tensors).unwrap();
    let expected = [
        ("publish", 3, [Some(Out), None], Gate, I),
        ("publish", 6, [Some(Out), None], Up, I),
        ("sigmoid", 6, [Some(Gate), None], SigmoidRaw, I),
        ("publish", 6, [Some(SigmoidRaw), None], Sigmoid, I),
        ("multiply", 6, [Some(Gate), Some(Sigmoid)], SiluRaw, I),
        ("publish", 6, [Some(SiluRaw), None], Silu, I),
        ("multiply", 6, [Some(Silu), Some(Up)], ActivatedRaw, I),
        ("publish", 6, [Some(ActivatedRaw), None], Activated, I),
        ("publish", 9, [Some(Out), None], Down, H),
    ]
    .map(|(kind, core, input, out, columns)| {
        let span = |slot: Slot| (arena.offsets[slot as usize], ids.len() * columns * 4);
        ObservedOp {
            kind,
            core,
            input: input.map(|slot| slot.map(span)),
            out: span(out),
            shape: [ids.len(), columns],
        }
    });
    assert_eq!(packed.operations, expected);
    assert_eq!(packed.projections, baseline.projections);
    assert_eq!(packed.emissions, baseline.emissions);
    let shapes = |observer: &Observer| {
        observer
            .allocations
            .iter()
            .map(|shape| [shape.rows, shape.columns])
            .collect::<Vec<_>>()
    };
    assert_eq!(shapes(&packed), shapes(baseline));
}

#[test]
fn legacy_routed_walk_records_real_requests_for_singletons_repeats_and_tails() {
    for rows in [1, 7, 8] {
        let ids = [255, 0, 73, 4, 111, 2].repeat(rows);
        let observer = observe(&ids, Mode::Legacy);
        assert_eq!(
            observer.experts,
            if rows == 1 {
                ids.iter().map(|&id| id as usize).collect::<Vec<_>>()
            } else {
                vec![0, 2, 4, 73, 111, 255]
            }
        );
        assert_eq!(observer.projections.len(), 18);
        assert_eq!(observer.emissions.len(), 54);
        assert_eq!(observer.noncore, 54);
        assert!(observer.projections.iter().all(|shape| shape[0] == rows));
        assert_eq!(observer.allocations.len(), if rows == 1 { 55 } else { 63 });
        let candidate = observe(&ids, Mode::Grouped);
        check_packed(&ids, &candidate);
        assert_eq!(candidate.experts, observer.experts);
        assert_eq!(candidate.emissions.len(), 9);
        assert_eq!(candidate.noncore, 54);
        assert_eq!(candidate.allocations.len(), if rows == 1 { 1 } else { 2 });
        assert!(candidate.projections.iter().all(|shape| shape[0] == rows));
        assert_eq!(candidate.projections[..6], [[rows, H, I]; 6]);
        assert_eq!(candidate.projections[6..12], [[rows, H, I]; 6]);
        assert_eq!(candidate.projections[12..], [[rows, I, H]; 6]);
    }
    let ids: Vec<i32> = (0..48).collect();
    let observer = observe(&ids, Mode::Legacy);
    assert_eq!(observer.experts.len(), 48);
    assert_eq!(observer.projections.len(), 144);
    assert_eq!(observer.emissions.len(), 432);
    assert_eq!(observer.noncore, 432);
    assert_eq!(observer.allocations.len(), 483);
    assert_eq!(
        observer
            .allocations
            .iter()
            .map(|shape| shape.bytes().unwrap())
            .sum::<usize>(),
        2_752_512
    );
}

/// The armed walk removes publishes and nothing else: the compute requests,
/// projection shapes and expert visits are exactly the unarmed walk's, and the
/// publishes absorbed by fusing producers are exactly the ones that vanish.
#[test]
fn fused_publish_arm_removes_only_the_publishes_it_absorbs() {
    let mut cases: Vec<Vec<i32>> = [1, 7, 8]
        .map(|rows| [255, 0, 73, 4, 111, 2].repeat(rows))
        .into();
    cases.push((0..48).collect());
    for ids in &cases {
        // Legacy per-expert walk: 3 projection publishes and 2 multiply
        // publishes per visit vanish; the sigmoid publish stays.
        let (plain, fused) = (
            observe_armed(ids, Mode::Legacy, false),
            observe_armed(ids, Mode::Legacy, true),
        );
        let visits = plain.experts.len();
        assert_eq!(fused.experts, plain.experts);
        assert_eq!(fused.projections, plain.projections);
        assert_eq!(fused.emissions, plain.emissions);
        assert_eq!(plain.noncore, 9 * visits);
        assert_eq!(fused.noncore, 4 * visits);
        assert_eq!(
            fused.allocations.len(),
            plain.allocations.len() - 2 * visits
        );
        assert_eq!(fused.fused_sites.len(), 5 * visits);
        assert!(plain.fused_sites.is_empty());
        // Grouped walks: one publish per group per role vanishes.
        for mode in [Mode::Grouped, Mode::Packed] {
            let (plain, fused) = (
                observe_armed(ids, mode, false),
                observe_armed(ids, mode, true),
            );
            let groups = plain.experts.len();
            assert_eq!(fused.experts, plain.experts);
            assert_eq!(fused.projections, plain.projections);
            assert_eq!(fused.emissions, plain.emissions);
            assert_eq!(fused.allocations.len(), plain.allocations.len());
            assert!(fused.fused_sites.iter().all(|&site| site == "grouped"));
            if mode == Mode::Packed {
                // One packed publish per role replaces the per-group loops.
                assert_eq!(plain.noncore - fused.noncore, 3);
                assert_eq!(fused.fused_sites.len(), 3 * groups);
                // Only the pointwise chain is left: the Out-to-Gate/Up/Down publishes are gone.
                assert_eq!(plain.operations.len(), 9);
                assert_eq!(fused.operations, plain.operations[2..8]);
            } else {
                assert_eq!(plain.noncore - fused.noncore, 3 * groups);
                assert_eq!(fused.fused_sites.len(), 3 * groups);
            }
        }
    }
}

/// An armed impl that opts into fusion but keeps every `*_published` hook at its trait
/// default: the documented fallback. Each required method delegates to the Observer.
struct Fallback(Observer);

impl Effects for Fallback {
    type Tensor = usize;
    type Matrix = (usize, [usize; 2]);

    fn packed_pointwise(&self) -> bool {
        self.0.packed_pointwise()
    }
    fn allocate(&mut self, shape: RowShape) -> Result<usize, EngError> {
        self.0.allocate(shape)
    }
    fn view(&mut self, span: Span<usize>) -> Result<usize, EngError> {
        self.0.view(span)
    }
    fn stage(&mut self, slot: Slot, first: usize, shape: RowShape) -> Result<usize, EngError> {
        self.0.stage(slot, first, shape)
    }
    fn grouped(&mut self, calls: &[Projection<Self::Matrix, usize>]) -> Result<(), EngError> {
        self.0.grouped(calls)
    }
    fn indices(&mut self, values: &[i32]) -> Result<usize, EngError> {
        self.0.indices(values)
    }
    fn expert(&mut self, id: usize) -> Result<Expert<Self::Matrix>, EngError> {
        self.0.expert(id)
    }
    fn copy(&mut self, source: Span<usize>, out: Span<usize>) -> Result<(), EngError> {
        self.0.copy(source, out)
    }
    fn gather(
        &mut self,
        x: usize,
        indices: usize,
        out: usize,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.0.gather(x, indices, out, shape)
    }
    fn project(&mut self, weight: Self::Matrix, x: usize) -> Result<(), EngError> {
        self.0.project(weight, x)
    }
    fn publish_projection(&mut self, out: usize, shape: RowShape) -> Result<(), EngError> {
        self.0.publish_projection(out, shape)
    }
    fn prefers_fused_publish(&self) -> bool {
        true
    }
    fn sigmoid(&mut self, x: usize, out: usize, shape: RowShape) -> Result<(), EngError> {
        self.0.sigmoid(x, out, shape)
    }
    fn multiply(&mut self, x: [usize; 2], out: usize, shape: RowShape) -> Result<(), EngError> {
        self.0.multiply(x, out, shape)
    }
    fn publish(&mut self, x: usize, out: usize, shape: RowShape) -> Result<(), EngError> {
        self.0.publish(x, out, shape)
    }
}

fn observe_fallback(ids: &[i32], mode: Mode) -> Observer {
    let rows = ids.len() / 6;
    let grouping = Grouping::new(ids, rows, 256).unwrap();
    let mut fallback = Fallback(Observer::new(rows));
    fallback.0.packed = mode == Mode::Packed;
    let output = if mode != Mode::Legacy {
        run(&mut fallback, block(rows), ids, &grouping, 0)
    } else {
        legacy(&mut fallback, block(rows), ids, &grouping, 0)
    }
    .unwrap();
    fallback.0.check(
        output,
        RowShape {
            rows: rows * 6,
            columns: H,
        },
    );
    fallback.0
}

/// An impl that prefers fusion but declines it at every hook enqueues each projection
/// exactly once and publishes exactly as the unarmed walk does.
#[test]
fn fused_publish_fallback_enqueues_each_projection_once() {
    for ids in [
        [255, 0, 73, 4, 111, 2].to_vec(),
        [255, 0, 73, 4, 111, 2].repeat(8),
        (0..48).collect::<Vec<i32>>(),
    ] {
        for mode in [Mode::Legacy, Mode::Grouped, Mode::Packed] {
            let (plain, fallback) = (observe(&ids, mode), observe_fallback(&ids, mode));
            assert_eq!(fallback.emissions, plain.emissions, "{mode:?}");
            assert_eq!(fallback.projections, plain.projections, "{mode:?}");
            assert_eq!(fallback.experts, plain.experts, "{mode:?}");
            assert_eq!(fallback.noncore, plain.noncore, "{mode:?}");
            assert_eq!(
                fallback.allocations.len(),
                plain.allocations.len(),
                "{mode:?}"
            );
            assert!(fallback.fused_sites.is_empty(), "{mode:?}");
        }
    }
}

#[test]
fn mixed_groups_keep_each_original_local_row_count() {
    let mut tokens: [Vec<i32>; 8] = std::array::from_fn(|_| Vec::new());
    let mut cursor = 0;
    for (id, rows) in [255, 0, 73, 4, 111, 2, 197, 90, 12, 33]
        .into_iter()
        .zip([1, 2, 3, 4, 5, 6, 7, 8, 8, 4])
    {
        for _ in 0..rows {
            tokens[cursor % 8].push(id);
            cursor += 1;
        }
    }
    assert!(tokens.iter().all(|token| token.len() == 6));
    let ids: Vec<_> = tokens.into_iter().flatten().collect();
    let observer = observe(&ids, Mode::Legacy);
    let rows: Vec<_> = observer
        .projections
        .as_chunks::<3>()
        .0
        .iter()
        .map(|group| group[0][0])
        .collect();
    assert_eq!(rows, [2, 6, 4, 8, 4, 3, 8, 5, 7, 1]);
    assert_eq!(observer.emissions.len(), 90);
    assert_eq!(observer.noncore, 90);
    assert_eq!(observer.allocations.len(), 103);
    let candidate = observe(&ids, Mode::Grouped);
    check_packed(&ids, &candidate);
    assert_eq!(candidate.experts, observer.experts);
    for phase in candidate.projections.as_chunks::<10>().0 {
        assert_eq!(phase.iter().map(|shape| shape[0]).collect::<Vec<_>>(), rows);
    }
    assert_eq!(candidate.emissions.len(), 9);
    assert_eq!(candidate.noncore, 90);
    assert_eq!(candidate.allocations.len(), 2);
}

#[test]
fn grouped_routed_host_core_budget() {
    let ids: Vec<i32> = (0..48).collect();
    let observer = observe(&ids, Mode::Grouped);
    check_packed(&ids, &observer);
    assert_eq!(
        observer.emissions.len(),
        9,
        "scheduled requests only; not device qualification"
    );
    assert_eq!(observer.noncore, 432);
    assert_eq!(observer.allocations.len(), 2);
    assert_eq!(
        observer
            .allocations
            .iter()
            .map(|shape| shape.bytes().unwrap())
            .sum::<usize>(),
        786_432
    );
}

// Frozen 8f53601f routed loop: retain original scalar expert/BF16 methods as oracle.
fn original_routed(
    effects: &mut RoutedCuda<'_, '_, '_, '_>,
    block: FfnBlock,
    ids: &[i32],
    grouping: &Grouping,
    x: usize,
) -> Result<usize, EngError> {
    let _scope = CudaContext::test_exl3_routed_scope();
    let frame = &mut *effects.frame;
    let scratch = &mut *effects.scratch;
    let experts = effects.experts;
    let down = frame.allocate(RowShape {
        rows: block.pairs,
        columns: block.hidden,
    })?;
    let row_bytes = block.hidden * 4;
    if block.rows == 1 {
        for (rank, &id) in ids.iter().enumerate() {
            let output = scalar_expert(frame, experts.get(id as usize)?, x, scratch)?;
            frame.ctx.copy_device(
                frame.stream,
                frame.range(output)?,
                frame.buffers[down].0.range(rank * row_bytes, row_bytes)?,
            )?;
        }
        return Ok(down);
    }
    let tokens = frame.upload_indices(&grouping.tokens)?;
    let pairs = RowShape {
        rows: block.pairs,
        columns: block.hidden,
    };
    let gathered = frame.gather(x, tokens, pairs, block.rows)?;
    let mut start = 0;
    while start < block.pairs {
        let id = ids[grouping.order[start]];
        let mut end = start + 1;
        while end < block.pairs && ids[grouping.order[end]] == id {
            end += 1;
        }
        let rows = end - start;
        let input = frame.allocate(RowShape {
            rows,
            columns: block.hidden,
        })?;
        frame.ctx.copy_device(
            frame.stream,
            frame.buffers[gathered]
                .0
                .range(start * row_bytes, rows * row_bytes)?,
            frame.range(input)?,
        )?;
        let output = scalar_expert(frame, experts.get(id as usize)?, input, scratch)?;
        frame.ctx.copy_device(
            frame.stream,
            frame.range(output)?,
            frame.buffers[down]
                .0
                .range(start * row_bytes, rows * row_bytes)?,
        )?;
        start = end;
    }
    let inverse = frame.upload_indices(&grouping.inverse)?;
    frame.gather(down, inverse, pairs, block.pairs)
}

/// Counts the pointwise loop's sigmoid passes by row count: the packed
/// path emits ONE pass at the full pair count, the per-group path one pass
/// per expert run. Delegates everything else untouched.
struct PointwiseProbe<'p, E> {
    effects: E,
    calls: &'p mut Vec<u32>,
}

impl<E: super::Effects> super::Effects for PointwiseProbe<'_, E> {
    type Tensor = E::Tensor;
    type Matrix = E::Matrix;

    fn packed_pointwise(&self) -> bool {
        self.effects.packed_pointwise()
    }
    fn allocate(&mut self, shape: RowShape) -> Result<Self::Tensor, EngError> {
        self.effects.allocate(shape)
    }
    fn view(&mut self, span: Span<Self::Tensor>) -> Result<Self::Tensor, EngError> {
        self.effects.view(span)
    }
    fn stage(
        &mut self,
        slot: Slot,
        first: usize,
        shape: RowShape,
    ) -> Result<Self::Tensor, EngError> {
        self.effects.stage(slot, first, shape)
    }
    fn grouped(
        &mut self,
        calls: &[Projection<Self::Matrix, Self::Tensor>],
    ) -> Result<(), EngError> {
        self.effects.grouped(calls)
    }
    fn indices(&mut self, values: &[i32]) -> Result<Self::Tensor, EngError> {
        self.effects.indices(values)
    }
    fn expert(&mut self, id: usize) -> Result<Expert<Self::Matrix>, EngError> {
        self.effects.expert(id)
    }
    fn copy(
        &mut self,
        source: Span<Self::Tensor>,
        out: Span<Self::Tensor>,
    ) -> Result<(), EngError> {
        self.effects.copy(source, out)
    }
    fn gather(
        &mut self,
        x: Self::Tensor,
        indices: Self::Tensor,
        out: Self::Tensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.effects.gather(x, indices, out, shape)
    }
    fn project(&mut self, weight: Self::Matrix, x: Self::Tensor) -> Result<(), EngError> {
        self.effects.project(weight, x)
    }
    fn publish_projection(&mut self, out: Self::Tensor, shape: RowShape) -> Result<(), EngError> {
        self.effects.publish_projection(out, shape)
    }
    fn sigmoid(
        &mut self,
        x: Self::Tensor,
        out: Self::Tensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.calls.push(shape.rows as u32);
        self.effects.sigmoid(x, out, shape)
    }
    fn multiply(
        &mut self,
        x: [Self::Tensor; 2],
        out: Self::Tensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.effects.multiply(x, out, shape)
    }
    fn publish(
        &mut self,
        x: Self::Tensor,
        out: Self::Tensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.effects.publish(x, out, shape)
    }
}

#[allow(clippy::too_many_arguments)]
fn device_run(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &FlintFfnView<'_>,
    mode: Mode,
    ids: &[i32],
    input: &[u8],
    grouped: Option<&GroupedScratch>,
    gather: bool,
    pointwise_calls: Option<&mut Vec<u32>>,
) -> (Vec<u8>, u64) {
    let FlintFfnView::Moe { experts, .. } = weights else {
        panic!("real layer must be routed")
    };
    assert_eq!(experts.len(), 256);
    let rows = ids.len() / 6;
    assert_eq!(input.len(), rows * H * 4);
    let uploaded = ctx.upload(input).unwrap();
    let mut frame = Frame::new(ctx, stream).unwrap();
    let x = frame
        .input(
            uploaded.range(0, input.len()).unwrap(),
            RowShape { rows, columns: H },
        )
        .unwrap();
    let mut scratch = Scratch::for_rows(ctx, weights, 8).unwrap();
    let grouping = Grouping::new(ids, rows, 256).unwrap();
    let owned_grouped = (mode != Mode::Legacy && grouped.is_none())
        .then(|| GroupedScratch::new(ctx, 8, H, I).unwrap());
    let grouped = grouped.or(owned_grouped.as_ref());
    let (down, actual) = CudaContext::test_routed_exl3_launches(|| {
        let mut effects = RoutedCuda {
            frame: &mut frame,
            scratch: &mut scratch,
            grouped,
            experts,
            packed_pointwise: mode == Mode::Packed,
            grouped_gather: gather,
            picks: None,
        };
        if mode != Mode::Legacy {
            match pointwise_calls {
                None => {
                    let input = RoutedTensor::owned(effects.frame, x)?;
                    run(&mut effects, block(rows), ids, &grouping, input)
                        .and_then(|down| down.owned_index(effects.frame))
                }
                Some(calls) => {
                    let mut probe = PointwiseProbe { effects, calls };
                    let input = RoutedTensor::owned(probe.effects.frame, x)?;
                    run(&mut probe, block(rows), ids, &grouping, input)
                        .and_then(|down| down.owned_index(probe.effects.frame))
                }
            }
        } else {
            original_routed(&mut effects, block(rows), ids, &grouping, x)
        }
    });
    if mode != Mode::Legacy {
        let expected = if gather { "gather" } else { "shuffle" };
        assert_eq!(
            CudaContext::test_routed_grouped_variant(),
            Some(expected),
            "leg claiming gather={gather} must actually run that variant"
        );
    }
    let output = frame.finish(down.unwrap()).unwrap();
    let mut bytes = vec![0; ids.len() * H * 4];
    output.read_at(stream, 0, &mut bytes).unwrap();
    (bytes, actual)
}

#[test]
#[ignore = "requires COO GPU slot and the admitted real full EXL3 pack; intentional device RED"]
fn grouped_routed_actual_core_budget_on_real_bank() {
    let root =
        std::path::PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("real EXL3 pack"));
    let ctx = CudaContext::new().expect("device RED requires CUDA; do not skip");
    let stream = ctx.stream().unwrap();
    let free = ctx.memory_info().unwrap().free as u64;
    let model = load_flint_resident(
        &ctx,
        &stream,
        &root,
        FlintResidentLimits {
            max_staging_bytes: 2 << 30,
            max_weight_bytes: free,
            reserve: Reserve {
                kv_cache: 0,
                work: 512 << 20,
                headroom: 1 << 30,
            },
        },
    )
    .unwrap();
    let layer = model.layer(2).unwrap();
    let ids: Vec<i32> = (0..48).collect();
    let input: Vec<u8> = vec![1.0_f32 / 128.0; 8 * H]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
    let (legacy_bits, legacy_launches) = device_run(
        &ctx,
        &stream,
        &layer.ffn,
        Mode::Legacy,
        &ids,
        &input,
        None,
        false,
        None,
    );
    assert_eq!(
        legacy_launches, 432,
        "counter must observe actual legacy native enqueues"
    );
    let (candidate_bits, actual) = device_run(
        &ctx,
        &stream,
        &layer.ffn,
        Mode::Grouped,
        &ids,
        &input,
        None,
        true,
        None,
    );
    assert_eq!(
        candidate_bits, legacy_bits,
        "same bank/input, BF16-in-F32 output bits"
    );
    let (rollback_bits, _) = device_run(
        &ctx,
        &stream,
        &layer.ffn,
        Mode::Grouped,
        &ids,
        &input,
        None,
        false,
        None,
    );
    assert_eq!(
        rollback_bits, legacy_bits,
        "rollback path must stay bit-identical too"
    );
    println!(
        "real admitted layer2 forced G48: legacy_actual={legacy_launches}, candidate_actual={actual}, target=9; no performance claim"
    );
    assert_eq!(actual, 9, "actual successful routed-core native launches");
    for rows in 1..=8u32 {
        let ids: Vec<i32> = [255, 0, 73, 4, 111, 2]
            .iter()
            .cycle()
            .take(rows as usize * 6)
            .copied()
            .collect();
        let input: Vec<u8> = (0..rows as usize * H)
            .map(|i| (((i * 17 + i / H * 31) % 257) as f32 - 128.0) / 1024.0)
            .flat_map(f32::to_ne_bytes)
            .collect();
        let (legacy_bits, _) = device_run(
            &ctx,
            &stream,
            &layer.ffn,
            Mode::Legacy,
            &ids,
            &input,
            None,
            false,
            None,
        );
        let (before, old_core) = device_run(
            &ctx,
            &stream,
            &layer.ffn,
            Mode::Grouped,
            &ids,
            &input,
            None,
            true,
            None,
        );
        let mut default_calls = Vec::new();
        let (after, packed_core) = device_run(
            &ctx,
            &stream,
            &layer.ffn,
            Mode::Packed,
            &ids,
            &input,
            None,
            true,
            Some(&mut default_calls),
        );
        assert_eq!(
            before, legacy_bits,
            "gather-on grouped bits for rows {rows}"
        );
        assert_eq!((old_core, packed_core), (9, 9));
        assert!(
            after == before,
            "packed output bytes differ for rows {rows}"
        );
        assert_eq!(
            default_calls,
            vec![rows * TOP_K as u32],
            "the shipped default must run the packed pass"
        );
        // The gather row's documented rollback lands here: packed
        // pointwise with the shuffle gather, and the per-group loop with
        // the shuffle gather (both switches rolled back), must each match
        // the legacy bits. The probe leg proves this really is the packed
        // path - ONE sigmoid pass at the full pair count - and not the
        // per-group loop wearing the same output.
        let mut packed_calls = Vec::new();
        let (shuffled, shuffled_core) = device_run(
            &ctx,
            &stream,
            &layer.ffn,
            Mode::Packed,
            &ids,
            &input,
            None,
            false,
            Some(&mut packed_calls),
        );
        assert_eq!(shuffled_core, 9);
        assert_eq!(
            packed_calls,
            vec![rows * TOP_K as u32],
            "packed pointwise must be one pass over all pairs"
        );
        assert!(
            shuffled == legacy_bits,
            "packed with shuffle gather differs for rows {rows}"
        );
        let mut grouped_calls = Vec::new();
        let (both_off, both_off_core) = device_run(
            &ctx,
            &stream,
            &layer.ffn,
            Mode::Grouped,
            &ids,
            &input,
            None,
            false,
            Some(&mut grouped_calls),
        );
        assert_eq!(both_off_core, 9);
        assert!(
            both_off == legacy_bits,
            "per-group with shuffle gather differs for rows {rows}"
        );
        // The sweep's ids repeat six experts every row; the grouping sorts
        // pairs by (id, pair) and merges same-expert runs, so the per-group
        // path is six passes of `rows` - one per expert run, never one
        // pass over all pairs like the packed arm above.
        assert_eq!(
            grouped_calls,
            vec![rows; TOP_K],
            "per-group path: one pass per expert run"
        );
    }
    for ids in [
        (0..48).rev().collect::<Vec<i32>>(),
        (0..48).map(|i| (i * 5 + i / 6) % 11).collect::<Vec<i32>>(),
    ] {
        let input: Vec<u8> = (0..ids.len() / 6 * H)
            .map(|i| (((i * 17 + i / H * 31) % 257) as f32 - 128.0) / 1024.0)
            .flat_map(f32::to_ne_bytes)
            .collect();
        let (legacy_bits, _) = device_run(
            &ctx,
            &stream,
            &layer.ffn,
            Mode::Legacy,
            &ids,
            &input,
            None,
            false,
            None,
        );
        let (before, old_core) = device_run(
            &ctx,
            &stream,
            &layer.ffn,
            Mode::Grouped,
            &ids,
            &input,
            None,
            true,
            None,
        );
        let mut arbitrary_calls = Vec::new();
        let (after, packed_core) = device_run(
            &ctx,
            &stream,
            &layer.ffn,
            Mode::Packed,
            &ids,
            &input,
            None,
            true,
            Some(&mut arbitrary_calls),
        );
        assert_eq!(before, legacy_bits, "gather-on grouped bits for {ids:?}");
        assert_eq!((old_core, packed_core), (9, 9));
        assert!(after == before, "packed output bytes differ for {ids:?}");
        assert_eq!(
            arbitrary_calls,
            vec![ids.len() as u32],
            "arbitrary ids must still run one packed pass over all pairs"
        );
    }

    let retained = GroupedScratch::new(&ctx, 8, H, I).unwrap();
    let arena_bytes = usize::try_from(retained.layout.bytes().unwrap()).unwrap();
    let poison = ctx.upload(&vec![0x5a; arena_bytes]).unwrap();
    let backing = retained.arena.tensor(0).unwrap();
    let target = backing.buffer().range(0, arena_bytes).unwrap();
    ctx.copy_device(&stream, poison.range(0, arena_bytes).unwrap(), target)
        .unwrap();
    // Preserve live stale values while reseeding inactive tails in one retained arena.
    for layer_index in [2, 3] {
        let layer = model.layer(layer_index).unwrap();
        for (leg, rows) in [8, 1, 7, 8].into_iter().enumerate() {
            let ids: Vec<i32> = (0..rows * 6)
                .map(|i| ((i * 5 + i / 6 + layer_index) % 11) as i32)
                .collect();
            let input: Vec<u8> = (0..rows * H)
                .map(|i| {
                    (((i * 17 + i / H * 31 + leg * 29 + layer_index * 37) % 257) as f32 - 128.0)
                        / 1024.0
                })
                .flat_map(f32::to_ne_bytes)
                .collect();
            let (before, old_core) = device_run(
                &ctx,
                &stream,
                &layer.ffn,
                Mode::Grouped,
                &ids,
                &input,
                None,
                true,
                None,
            );
            for (index, spec) in retained.layout.tensors.iter().enumerate() {
                let start = spec.bytes * rows / 8;
                let bytes = spec.bytes - start;
                if bytes == 0 {
                    continue;
                }
                let slot = retained.arena.tensor(index).unwrap();
                let target = slot.buffer().range(slot.offset() + start, bytes).unwrap();
                ctx.copy_device(&stream, poison.range(0, bytes).unwrap(), target)
                    .unwrap();
            }
            let mut retained_calls = Vec::new();
            let (after, packed_core) = device_run(
                &ctx,
                &stream,
                &layer.ffn,
                Mode::Packed,
                &ids,
                &input,
                Some(&retained),
                true,
                Some(&mut retained_calls),
            );
            assert_eq!((old_core, packed_core), (9, 9));
            assert!(
                after == before,
                "retained layer{layer_index} rows{rows} differs"
            );
            assert_eq!(
                retained_calls,
                vec![(rows * TOP_K) as u32],
                "retained layers must run the packed pass"
            );
            for (index, spec) in retained.layout.tensors.iter().enumerate() {
                let start = spec.bytes * rows / 8;
                let bytes = spec.bytes - start;
                if bytes == 0 {
                    continue;
                }
                let slot = retained.arena.tensor(index).unwrap();
                let mut tail = vec![0; bytes];
                slot.buffer()
                    .read_at(&stream, slot.offset() + start, &mut tail)
                    .unwrap();
                assert!(
                    tail.iter().all(|&byte| byte == 0x5a),
                    "layer{layer_index} rows{rows} slot{index} tail"
                );
            }
        }
    }
}

#[test]
#[ignore = "GPU lease: times the grouped MoE step over every layer. CEILING of the EXL3 matmul path, not a model speed"]
fn grouped_decode_step_ceiling() {
    let root =
        std::path::PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("real EXL3 pack"));
    let ctx = CudaContext::new().expect("device");
    let stream = ctx.stream().unwrap();
    let free = ctx.memory_info().unwrap().free as u64;
    let model = load_flint_resident(
        &ctx,
        &stream,
        &root,
        FlintResidentLimits {
            max_staging_bytes: 2 << 30,
            max_weight_bytes: free,
            reserve: Reserve {
                kv_cache: 0,
                work: 512 << 20,
                headroom: 1 << 30,
            },
        },
    )
    .unwrap();
    let layer_count = model.layer_count();
    assert!(layer_count > 0, "fixture pack must have layers");
    let (ids, grouping, one_row) = ceiling_step_args();
    let input: Vec<u8> = (0..H)
        .map(|i| (((i * 17) % 257) as f32 - 128.0) / 1024.0)
        .flat_map(f32::to_ne_bytes)
        .collect();
    let uploaded = ctx.upload(&input).unwrap();
    // Per-layer work areas are prepared untimed; a decode step reuses them.
    struct Prepared<'a, 'c, 's> {
        ffn: FlintFfnView<'a>,
        frame: Frame<'c, 's>,
        scratch: Scratch,
        grouped: GroupedScratch,
        x: usize,
    }
    let mut layers: Vec<Prepared<'_, '_, '_>> = Vec::with_capacity(layer_count);
    let mut moe_layers = 0usize;
    for index in 0..layer_count {
        let layer = model.layer(index).unwrap();
        let FlintFfnView::Moe { experts, .. } = &layer.ffn else {
            continue;
        };
        assert_eq!(experts.len(), 256);
        moe_layers += 1;
        let mut frame = Frame::new(&ctx, &stream).unwrap();
        let x = frame
            .input(
                uploaded.range(0, input.len()).unwrap(),
                RowShape {
                    rows: 1,
                    columns: H,
                },
            )
            .unwrap();
        let scratch = Scratch::for_rows(&ctx, &layer.ffn, 1).unwrap();
        let grouped = GroupedScratch::new(&ctx, 1, H, I).unwrap();
        layers.push(Prepared {
            ffn: layer.ffn,
            frame,
            scratch,
            grouped,
            x,
        });
    }
    assert_eq!(
        moe_layers, 47,
        "the admitted fixture routes 47 layers; a different pack changes the budget"
    );
    // Untimed verification leg: observation must name the gather variant, and
    // the launch census counts the routed-core launches per whole pack. Off
    // during timed passes.
    let (_, launches) = CudaContext::test_routed_exl3_launches(|| {
        let _scope = CudaContext::test_exl3_routed_scope();
        for prepared in &mut layers {
            let FlintFfnView::Moe { experts, .. } = &prepared.ffn else {
                unreachable!("prepared layers are routed");
            };
            let mut effects = RoutedCuda {
                frame: &mut prepared.frame,
                scratch: &mut prepared.scratch,
                grouped: Some(&prepared.grouped),
                experts,
                packed_pointwise: false,
                grouped_gather: true,
                picks: None,
            };
            let input = RoutedTensor::owned(effects.frame, prepared.x).unwrap();
            run(&mut effects, one_row, &ids, &grouping, input).unwrap();
        }
    });
    assert_eq!(
        CudaContext::test_routed_grouped_variant(),
        Some("gather"),
        "the ceiling leg claims the gather variant and must run it"
    );
    assert_eq!(
        launches,
        47 * 9,
        "nine grouped routed-core launches per layer over all forty-seven"
    );
    println!("grouped_decode_ceiling census: {launches} launches per pass, 9 per layer");
    // Two arms share every prepared layer: the fused packed-pointwise path
    // this tree ships by default, and the per-group loop a rollback lands
    // on. Interleaving them per pass keeps clocks common between the arms.
    let arms = [("pointwise_grouped", false), ("pointwise_packed", true)];
    let mut pass_ms = [[0f32; 2]; 5];
    for (pass, slot) in pass_ms.iter_mut().enumerate() {
        // Alternate which arm runs first each pass (grouped, packed / packed,
        // grouped / ...): a fixed order would let cache and clock drift bias
        // one arm.
        let order = [(0usize, 1usize), (1, 0)][pass % 2];
        for arm in [order.0, order.1] {
            let (name, packed) = arms[arm];
            let begin = ctx.event_timing().unwrap();
            let end = ctx.event_timing().unwrap();
            stream.synchronize().unwrap();
            stream.record_timing(&begin).unwrap();
            for prepared in &mut layers {
                let FlintFfnView::Moe { experts, .. } = &prepared.ffn else {
                    unreachable!("prepared layers are routed");
                };
                let mut effects = RoutedCuda {
                    frame: &mut prepared.frame,
                    scratch: &mut prepared.scratch,
                    grouped: Some(&prepared.grouped),
                    experts,
                    packed_pointwise: packed,
                    grouped_gather: true,
                    picks: None,
                };
                let input = RoutedTensor::owned(effects.frame, prepared.x).unwrap();
                run(&mut effects, one_row, &ids, &grouping, input).unwrap();
            }
            stream.record_timing(&end).unwrap();
            // Both records must complete before the elapsed query: a query on
            // unfinished events fails not-ready.
            stream.synchronize().unwrap();
            let ms = begin.elapsed_ms_since(&end).unwrap();
            slot[arm] = ms;
            // Outside the timed interval: every prepared layer must have
            // completed a routed call, and the sticky health word must be
            // clean - the kernels zero non-finite values silently, so an
            // unhealthy pass must fail the timing run, not feed it.
            for prepared in &layers {
                prepared.frame.complete().unwrap();
            }
            println!(
                "grouped_decode_ceiling {name} pass {pass}: {ms:.3} ms for {moe_layers} routed layers at 1 row top-6"
            );
        }
    }
    for (arm, (name, _)) in arms.iter().enumerate() {
        let warm = &pass_ms[1..];
        let mean = warm.iter().map(|s| s[arm]).sum::<f32>() / warm.len() as f32;
        let best = warm.iter().map(|s| s[arm]).fold(f32::INFINITY, f32::min);
        println!(
            "grouped_decode_ceiling summary {name}: warm passes {:?} mean {mean:.3} ms best {best:.3} ms per {moe_layers}-layer grouped MoE step at 1 row top-6; CEILING of the EXL3 matmul path, not a model speed",
            warm.iter().map(|s| s[arm]).collect::<Vec<_>>()
        );
    }
}

/// The single argument set the device ceiling run and this host dry run
/// share: one decode row, the fixture's six distinct expert picks, the block
/// metadata and the routing grouping built from them.
fn ceiling_step_args() -> (Vec<i32>, Grouping, FfnBlock) {
    let ids: Vec<i32> = [255, 0, 73, 4, 111, 2].to_vec();
    assert_eq!(
        ids.len(),
        TOP_K,
        "one decode row with the fixture's expert picks"
    );
    let grouping = Grouping::new(&ids, 1, 256).expect("top-6 ids cover one row");
    (ids, grouping, block(1))
}

#[test]
fn ceiling_args_cover_one_row() {
    // Host-only dry run of the exact arguments the device ceiling test passes
    // to the coverage check that rejected a top-8 list on this top-6 fixture.
    let (ids, grouping, block) = ceiling_step_args();
    assert_eq!(ids.len(), TOP_K);
    assert_eq!(
        ids.len(),
        ids.iter().collect::<std::collections::BTreeSet<_>>().len()
    );
    assert_eq!(block.rows, 1);
    assert_eq!(block.pairs, TOP_K);
    assert_eq!(grouping.tokens.len(), TOP_K);
    assert_eq!(grouping.order.len(), TOP_K);
    assert_eq!(grouping.inverse.len(), TOP_K);
}

#[test]
#[ignore = "requires a CUDA device and the real EXL3 pack; measurement only"]
fn grouped_project_scaling_microbench() {
    // Does one grouped projection launch move more pairs nearly free
    // (latency-bound; batching wins) or does its time scale with pairs
    // (throughput-bound; the read path itself must widen)? Per row count the
    // serving-shape work areas are prepared untimed, then three event-timed
    // repeats bracket only the routed step (completion and readback sit
    // outside the interval). Launch observation and the pointwise probe stay
    // armed inside the timed legs (the per-rep assertions need them).
    // Without and with the wrap, per-row first-repeat step_ms, rows
    // 1/2/4/8: 0.120/0.189/0.298/0.520 vs 0.119/0.191/0.297/0.535 ms
    // (deltas -0.001/+0.002/-0.001/+0.015); the repeat ranges at rows=8 are
    // 0.520-0.562 without and 0.520-0.560 with the wrap.
    let root =
        std::path::PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("real EXL3 pack"));
    let ctx = CudaContext::new().expect("device");
    let stream = ctx.stream().unwrap();
    let free = ctx.memory_info().unwrap().free as u64;
    let model = load_flint_resident(
        &ctx,
        &stream,
        &root,
        FlintResidentLimits {
            max_staging_bytes: 2 << 30,
            max_weight_bytes: free,
            reserve: Reserve {
                kv_cache: 0,
                work: 512 << 20,
                headroom: 1 << 30,
            },
        },
    )
    .unwrap();
    let layer = model.layer(2).unwrap();
    let FlintFfnView::Moe { experts, .. } = &layer.ffn else {
        panic!("real layer must be routed");
    };
    for rows in [1u32, 2, 4, 8] {
        let ids: Vec<i32> = [255, 0, 73, 4, 111, 2]
            .iter()
            .cycle()
            .take(rows as usize * 6)
            .copied()
            .collect();
        let input: Vec<u8> = (0..rows as usize * H)
            .map(|i| (((i * 17 + i / H * 31) % 257) as f32 - 128.0) / 1024.0)
            .flat_map(f32::to_ne_bytes)
            .collect();
        // Probe leg, untimed: one packed pass over all pairs plus the nine
        // routed-core launches, so the timed legs below cannot silently
        // measure the per-group path.
        let mut probe_calls = Vec::new();
        let (_, actual) = device_run(
            &ctx,
            &stream,
            &layer.ffn,
            Mode::Packed,
            &ids,
            &input,
            None,
            true,
            Some(&mut probe_calls),
        );
        assert_eq!(actual, 9, "nine routed-core launches on the probe leg");
        assert_eq!(
            probe_calls,
            vec![rows * 6],
            "one packed pass over all pairs"
        );
        // Serving-shape work areas, prepared once per row count, untimed.
        let uploaded = ctx.upload(&input).unwrap();
        let mut frame = Frame::new(&ctx, &stream).unwrap();
        let x = frame
            .input(
                uploaded.range(0, input.len()).unwrap(),
                RowShape {
                    rows: rows as usize,
                    columns: H,
                },
            )
            .unwrap();
        let mut scratch = Scratch::for_rows(&ctx, &layer.ffn, 8).unwrap();
        let grouped = GroupedScratch::new(&ctx, 8, H, I).unwrap();
        let grouping = Grouping::new(&ids, rows as usize, 256).unwrap();
        let mut samples = Vec::new();
        let mut timed_down = None;
        for _ in 0..3 {
            let mut timed_calls = Vec::new();
            let begin = ctx.event_timing().unwrap();
            let end = ctx.event_timing().unwrap();
            stream.synchronize().unwrap();
            stream.record_timing(&begin).unwrap();
            let (down, timed_launches) = CudaContext::test_routed_exl3_launches(|| {
                let effects = RoutedCuda {
                    frame: &mut frame,
                    scratch: &mut scratch,
                    grouped: Some(&grouped),
                    experts,
                    packed_pointwise: true,
                    grouped_gather: true,
                    picks: None,
                };
                // The probe wraps the TIMED effects: without
                // it, flipping only the flags above would leave the probe
                // leg's assertions satisfied while the timed legs silently
                // measure the per-group path.
                let mut probe = PointwiseProbe {
                    effects,
                    calls: &mut timed_calls,
                };
                let input = RoutedTensor::owned(probe.effects.frame, x)?;
                run(&mut probe, block(rows as usize), &ids, &grouping, input)
                    .and_then(|down| down.owned_index(probe.effects.frame))
            });
            let down = down.unwrap();
            stream.record_timing(&end).unwrap();
            stream.synchronize().unwrap();
            let ms = begin.elapsed_ms_since(&end).unwrap();
            assert_eq!(
                timed_launches, 9,
                "nine routed-core launches on the timed leg"
            );
            assert_eq!(
                timed_calls,
                vec![rows * 6],
                "one packed pass over all pairs on the timed leg"
            );
            assert!(
                ms.is_finite() && ms > 0.0,
                "a timing sample must be finite and positive"
            );
            samples.push(ms);
            timed_down = Some(down);
        }
        // Outside every timed interval: one completion and readback. The
        // repeats reuse the grouped workspace and retain separate outputs
        // until completion; the probe leg above pinned the pass structure,
        // so this leg only takes timings from it.
        frame.finish(timed_down.unwrap()).unwrap();
        println!(
            "microbench rows={rows} pairs={} step_ms=[{:.3} {:.3} {:.3}]",
            rows * 6,
            samples[0],
            samples[1],
            samples[2]
        );
    }
}

/// What the token-major walk hands each expert: every eight-row block of the
/// piece run through the real host walk, reduced to the (row, rank) pairs per
/// expert. The recorded visits of the walk itself are checked against the
/// grouping runs they must follow, so this model cannot drift from the code.
fn token_major_pairs(ids: &[i32]) -> std::collections::BTreeMap<usize, Vec<(usize, usize)>> {
    let mut pairs: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
    for (index, block_ids) in ids.chunks(MAX_ROWS * TOP_K).enumerate() {
        let rows = block_ids.len() / TOP_K;
        let grouping = Grouping::new(block_ids, rows, 256).unwrap();
        let mut runs: Vec<(usize, Vec<usize>)> = Vec::new();
        for (position, &pair) in grouping.order.iter().enumerate() {
            let expert = block_ids[pair] as usize;
            match runs.last_mut() {
                Some((last, members)) if *last == expert => members.push(position),
                _ => runs.push((expert, vec![position])),
            }
        }
        let observer = observe(block_ids, Mode::Grouped);
        let mut walked: Vec<_> = observer
            .experts
            .iter()
            .copied()
            .zip(observer.projections.iter().map(|shape| shape[0]))
            .collect();
        let mut modelled: Vec<_> = runs
            .iter()
            .map(|(expert, members)| (*expert, members.len()))
            .collect();
        walked.sort_unstable();
        modelled.sort_unstable();
        assert_eq!(walked[..modelled.len()], modelled, "block {index}");
        for (expert, members) in runs {
            for position in members {
                let pair = grouping.order[position];
                pairs
                    .entry(expert)
                    .or_default()
                    .push((index * MAX_ROWS + pair / TOP_K, pair % TOP_K));
            }
        }
    }
    pairs.values_mut().for_each(|rows| rows.sort_unstable());
    pairs
}

/// The layer-major plan reduced to the same (row, rank) pairs per expert.
fn layer_major_pairs(
    ids: &[i32],
    capacity: usize,
) -> (
    std::collections::BTreeMap<usize, Vec<(usize, usize)>>,
    super::super::layer_plan::tests::Recorder,
) {
    use super::super::layer_plan::{LayerPlan, tests::Recorder, walk};
    let rows = ids.len() / TOP_K;
    let grouping = Grouping::new(ids, rows, 256).unwrap();
    let plan = LayerPlan::new(ids, &grouping, capacity).unwrap();
    let mut recorder = Recorder::default();
    walk(&mut recorder, &plan).unwrap();
    let mut pairs: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
    for &(_, expert, tile) in &recorder.tiles {
        for position in tile.first..tile.first + tile.rows {
            let pair = grouping.order[position];
            pairs
                .entry(expert)
                .or_default()
                .push((pair / TOP_K, pair % TOP_K));
        }
    }
    pairs.values_mut().for_each(|rows| rows.sort_unstable());
    (pairs, recorder)
}

/// The layer-major walk is the token-major walk regrouped: every expert sees
/// the same (row, rank) pairs, and nothing else changes except how many
/// times each expert is entered and how tall its tiles are.
#[test]
fn layer_major_walk_visits_the_same_experts_with_the_same_rows_only_regrouped() {
    use super::super::layer_plan::{TILE_MAX_ROWS, tests::routing};
    for rows in [1, 7, 8, 9, 63, 64, 65, 200, 1000, 4000] {
        let ids = routing(rows, 0xfeed + rows as u64);
        let legacy = token_major_pairs(&ids);
        let (armed, recorder) = layer_major_pairs(&ids, 224);
        assert_eq!(armed, legacy, "rows {rows}");
        // Regrouping only coarsens: no expert is entered more often than in
        // the narrow walk, and an expert with more than one tile owes that to
        // the tensor-core row cap alone.
        let entered = |expert: usize| {
            recorder
                .tiles
                .iter()
                .filter(|&&(_, id, _)| id == expert)
                .count()
        };
        for (&expert, members) in &armed {
            assert_eq!(entered(expert), members.len().div_ceil(TILE_MAX_ROWS));
            assert!(entered(expert) <= rows.div_ceil(MAX_ROWS), "rows {rows}");
        }
        // Every expert is resident in exactly one round of at most 224.
        let staged: Vec<i32> = recorder.rounds.iter().flatten().copied().collect();
        let mut distinct = staged.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(staged.len(), distinct.len(), "rows {rows}: staged twice");
        assert!(recorder.rounds.iter().all(|round| round.len() <= 224));
        assert_eq!(
            distinct,
            armed.keys().map(|&id| id as i32).collect::<Vec<_>>()
        );
    }
}
