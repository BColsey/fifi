//! Host-only capture-topology proof for the rows==1 routed walk
//! (qualification/flint-p0p4, P1 of DESIGN.md §0.1).
//!
//! These tests drive [`run`] against a recording double, so they execute no
//! device call even when the `native-cuda` feature compiles them. They pin
//! three facts the capture feasibility verdict rests on:
//!
//! 1. At `rows == 1` the walk's operation sequence (kinds, shapes, group
//!    counts, and every launch argument except expert identity) is invariant
//!    under the router's choices.
//! 2. In the host-routed mode the expert identity is the only per-token datum
//!    that reaches the grouped launches: same fixed topology, different
//!    weight descriptors per token. A captured graph would bake those
//!    descriptors, which is the concrete capture hazard.
//! 3. In the device-picks mode (`FIFI_FLINT_DEVICE_PICKS`) every launch names
//!    the canonical bank descriptor and the host walk never sees an id, so
//!    the whole sequence is replay-safe as-is.
//!
//! The walk also gathers and uploads no route indices at `rows == 1`, which
//! the tests pin so a future change that adds per-token uploads fails loudly
//! instead of silently breaking capture compatibility.

use super::*;
use crate::family::flint::cuda_ffn::plan::{FfnShape, TOP_K};

const H: usize = 2048;
const I: usize = 512;
const EXPERTS: usize = 256;

/// The shared deterministic router from the layer-plan tests: six distinct
/// experts per row, skewed toward low ids.
use super::super::layer_plan::tests::routing;

/// One recorded walk operation. `Grouped::weights` carries the weight
/// descriptors the launches named, which is exactly the data a captured graph
/// would bake in.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Op {
    Grouped {
        weights: Vec<usize>,
        rows_per_call: Vec<usize>,
        columns: usize,
    },
    Indices {
        count: usize,
    },
    ExpertId(usize),
    PickDescriptor,
    Stage {
        slot: Slot,
        first: usize,
        rows: usize,
        columns: usize,
    },
    Sigmoid {
        rows: usize,
        columns: usize,
    },
    Multiply {
        rows: usize,
        columns: usize,
    },
    Publish {
        rows: usize,
        columns: usize,
    },
    Copy {
        bytes: usize,
    },
    Gather {
        rows: usize,
    },
    Project {
        weight: usize,
        columns: usize,
    },
    PublishProjection {
        rows: usize,
        columns: usize,
    },
}

/// Recording double: `Tensor` is a buffer id whose byte size is tracked,
/// `Matrix` is `(weight descriptor, role, [k, n])`. Production decode runs
/// with packed pointwise armed, so the double reports it armed too.
struct Recorder {
    picks: bool,
    next: usize,
    buffers: Vec<usize>,
    ops: Vec<Op>,
}

impl Recorder {
    fn new(picks: bool) -> Self {
        Self {
            picks,
            next: 1,
            buffers: vec![H * 4],
            ops: Vec::new(),
        }
    }

    fn fresh(&mut self, bytes: usize) -> usize {
        let id = self.next;
        self.next += 1;
        self.buffers.push(bytes);
        id
    }
}

type Matrix = (usize, usize, [usize; 2]);

impl Effects for Recorder {
    type Tensor = usize;
    type Matrix = Matrix;

    fn packed_pointwise(&self) -> bool {
        true
    }

    fn allocate(&mut self, shape: RowShape) -> Result<Self::Tensor, EngError> {
        let bytes = shape
            .bytes()
            .ok_or_else(|| bad("allocation overflows a usize"))?;
        Ok(self.fresh(bytes))
    }

    fn view(&mut self, span: Span<Self::Tensor>) -> Result<Self::Tensor, EngError> {
        Ok(span.tensor)
    }

    fn stage(
        &mut self,
        slot: Slot,
        first: usize,
        shape: RowShape,
    ) -> Result<Self::Tensor, EngError> {
        self.ops.push(Op::Stage {
            slot,
            first,
            rows: shape.rows,
            columns: shape.columns,
        });
        let bytes = shape
            .bytes()
            .ok_or_else(|| bad("staging overflows a usize"))?;
        Ok(self.fresh(bytes))
    }

    fn grouped(
        &mut self,
        calls: &[Projection<Self::Matrix, Self::Tensor>],
    ) -> Result<(), EngError> {
        let mut weights = Vec::with_capacity(calls.len());
        let mut rows_per_call = Vec::with_capacity(calls.len());
        let mut columns = None;
        for call in calls {
            let [k, n] = call.weight.2;
            let stride = k
                .checked_mul(4)
                .filter(|stride| *stride != 0)
                .ok_or_else(|| bad("grouped input width overflows"))?;
            if self.buffers[call.x] % stride != 0 {
                return Err(bad("grouped input is not row aligned"));
            }
            weights.push(call.weight.0);
            rows_per_call.push(self.buffers[call.x] / stride);
            columns = Some(match columns {
                None => n,
                Some(seen) if seen == n => n,
                Some(_) => return Err(bad("grouped launch mixes output widths")),
            });
        }
        self.ops.push(Op::Grouped {
            weights,
            rows_per_call,
            columns: columns.unwrap_or(0),
        });
        Ok(())
    }

    fn indices(&mut self, values: &[i32]) -> Result<Self::Tensor, EngError> {
        self.ops.push(Op::Indices {
            count: values.len(),
        });
        Ok(self.fresh(values.len() * 4))
    }

    fn expert(&mut self, id: usize) -> Result<Expert<Self::Matrix>, EngError> {
        self.ops.push(Op::ExpertId(id));
        Ok(Expert {
            matrices: [(id, 0, [H, I]), (id, 1, [H, I]), (id, 2, [I, H])],
            hidden: H,
            intermediate: I,
        })
    }

    fn device_picks(&self) -> bool {
        self.picks
    }

    fn pick_expert(&mut self) -> Result<Expert<Self::Matrix>, EngError> {
        self.ops.push(Op::PickDescriptor);
        Ok(Expert {
            matrices: [(0, 0, [H, I]), (0, 1, [H, I]), (0, 2, [I, H])],
            hidden: H,
            intermediate: I,
        })
    }

    fn copy(
        &mut self,
        source: Span<Self::Tensor>,
        out: Span<Self::Tensor>,
    ) -> Result<(), EngError> {
        assert_eq!(source.bytes, out.bytes, "copy spans disagree");
        self.ops.push(Op::Copy {
            bytes: source.bytes,
        });
        Ok(())
    }

    fn gather(
        &mut self,
        _x: Self::Tensor,
        _indices: Self::Tensor,
        _out: Self::Tensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.ops.push(Op::Gather { rows: shape.rows });
        Ok(())
    }

    fn project(&mut self, weight: Self::Matrix, _x: Self::Tensor) -> Result<(), EngError> {
        self.ops.push(Op::Project {
            weight: weight.0,
            columns: weight.2[1],
        });
        Ok(())
    }

    fn publish_projection(&mut self, _out: Self::Tensor, shape: RowShape) -> Result<(), EngError> {
        self.ops.push(Op::PublishProjection {
            rows: shape.rows,
            columns: shape.columns,
        });
        Ok(())
    }

    fn sigmoid(
        &mut self,
        _x: Self::Tensor,
        _out: Self::Tensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.ops.push(Op::Sigmoid {
            rows: shape.rows,
            columns: shape.columns,
        });
        Ok(())
    }

    fn multiply(
        &mut self,
        _x: [Self::Tensor; 2],
        _out: Self::Tensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.ops.push(Op::Multiply {
            rows: shape.rows,
            columns: shape.columns,
        });
        Ok(())
    }

    fn publish(
        &mut self,
        _x: Self::Tensor,
        _out: Self::Tensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.ops.push(Op::Publish {
            rows: shape.rows,
            columns: shape.columns,
        });
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
            experts: EXPERTS,
        },
    )
    .unwrap()
}

fn walk(picks: bool, rows: usize, seed: u64) -> Vec<Op> {
    let ids = routing(rows, seed);
    let grouping = Grouping::new(&ids, rows, EXPERTS).unwrap();
    let mut recorder = Recorder::new(picks);
    run(&mut recorder, block(rows), &ids, &grouping, 0).unwrap();
    recorder.ops
}

/// The topology: every identity datum (expert ids, pick descriptors) masked
/// to one marker, so only launch structure remains.
fn masked(ops: &[Op]) -> Vec<Op> {
    ops.iter()
        .map(|op| match op {
            Op::ExpertId(_) => Op::ExpertId(0),
            Op::PickDescriptor => Op::ExpertId(0),
            Op::Grouped {
                weights,
                rows_per_call,
                columns,
            } => Op::Grouped {
                weights: vec![0; weights.len()],
                rows_per_call: rows_per_call.clone(),
                columns: *columns,
            },
            other => other.clone(),
        })
        .collect()
}

fn first_grouped(ops: &[Op]) -> &Op {
    ops.iter()
        .find(|op| matches!(op, Op::Grouped { .. }))
        .expect("the walk always launches a grouped projection")
}

fn grouped_count(ops: &[Op]) -> usize {
    ops.iter()
        .filter(|op| matches!(op, Op::Grouped { .. }))
        .count()
}

#[test]
fn rows1_topology_is_invariant_under_the_routers_choices() {
    let reference = masked(&walk(false, 1, 1));
    assert!(!reference.is_empty());
    for seed in 2..=32u64 {
        assert_eq!(
            masked(&walk(false, 1, seed)),
            reference,
            "topology changed with the route, seed {seed}"
        );
    }
    // Three grouped launches per layer walk: gate, up, down.
    let ops = walk(false, 1, 1);
    assert_eq!(grouped_count(&ops), 3, "gate, up and down projections");
    for op in ops.iter().filter(|op| matches!(op, Op::Grouped { .. })) {
        let Op::Grouped {
            weights,
            rows_per_call,
            ..
        } = op
        else {
            unreachable!();
        };
        assert_eq!(weights.len(), TOP_K);
        assert!(rows_per_call.iter().all(|&rows| rows == 1));
    }
}

#[test]
fn rows1_host_mode_identity_is_the_only_per_token_launch_datum() {
    let ids = routing(1, 9);
    let grouping = Grouping::new(&ids, 1, EXPERTS).unwrap();
    let mut recorder = Recorder::new(false);
    run(&mut recorder, block(1), &ids, &grouping, 0).unwrap();
    let ops = recorder.ops;

    // The host names each routed expert exactly once per layer walk...
    let named: Vec<usize> = ops
        .iter()
        .filter_map(|op| match op {
            Op::ExpertId(id) => Some(*id),
            _ => None,
        })
        .collect();
    assert_eq!(named, ids.iter().map(|&id| id as usize).collect::<Vec<_>>());
    // ...and the first grouped launch (gate) names the same six descriptors
    // in pair order.
    let Op::Grouped { weights, .. } = first_grouped(&ops) else {
        unreachable!();
    };
    assert_eq!(
        *weights,
        ids.iter().map(|&id| id as usize).collect::<Vec<_>>()
    );

    // A different route keeps the masked topology but moves the descriptors:
    // the same launch sequence with different pointers, which is exactly what
    // a graph capture would bake.
    let other_ids = routing(1, 10);
    let other_grouping = Grouping::new(&other_ids, 1, EXPERTS).unwrap();
    let mut other = Recorder::new(false);
    run(&mut other, block(1), &other_ids, &other_grouping, 0).unwrap();
    assert_ne!(other.ops, ops);
    assert_eq!(masked(&other.ops), masked(&ops));
}

#[test]
fn rows1_picks_mode_is_replay_safe_with_the_same_topology() {
    let host = walk(false, 1, 21);
    let picks = walk(true, 1, 21);
    // Same launch structure in both modes...
    assert_eq!(masked(&picks), masked(&host));
    // ...but the picks mode never asks the host for an id, never uploads
    // indices, and every launch names the canonical bank descriptor (member
    // 0); the device's id buffer selects the member, so replay is exact.
    assert!(
        picks
            .iter()
            .all(|op| !matches!(op, Op::ExpertId(_)) && !matches!(op, Op::Indices { .. }))
    );
    assert_eq!(
        picks
            .iter()
            .filter(|op| matches!(op, Op::PickDescriptor))
            .count(),
        TOP_K
    );
    for op in picks.iter().filter(|op| matches!(op, Op::Grouped { .. })) {
        let Op::Grouped { weights, .. } = op else {
            unreachable!();
        };
        assert!(weights.iter().all(|&weight| weight == 0));
    }
}

#[test]
fn rows1_walk_uploads_no_route_indices_or_gathers() {
    for picks in [false, true] {
        let ops = walk(picks, 1, 5);
        assert!(
            ops.iter().all(|op| !matches!(op, Op::Gather { .. })),
            "decode gather appeared, picks {picks}"
        );
        assert!(
            ops.iter().all(|op| !matches!(op, Op::Indices { .. })),
            "decode index upload appeared, picks {picks}"
        );
    }
    // The multi-row walk does gather and upload indices; the claim above is
    // deliberately decode-only.
    let ops = walk(false, 8, 6);
    assert!(ops.iter().any(|op| matches!(op, Op::Gather { .. })));
    assert!(ops.iter().any(|op| matches!(op, Op::Indices { .. })));
}
