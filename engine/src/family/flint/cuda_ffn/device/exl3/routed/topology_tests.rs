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
//!
//! Production decode defaults to BOTH publish paths armed (packed pointwise
//! and fused publish: `settings.rs` selects them `default_on_selected`), so
//! every topology claim below runs under both configurations — the
//! production default and the fused-publish rollback
//! (`FIFI_FLINT_CUDA_FUSED_PUBLISH=0`).

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
    /// The fused grouped projection: one launch publishing straight into
    /// the per-group destinations, replacing the publish loop.
    GroupedPublished {
        weights: Vec<usize>,
        columns: usize,
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
/// `Matrix` is `(weight descriptor, role, [k, n])`. `fused` mirrors the
/// production `default_on_selected` fused-publish setting; `false` is the
/// `FIFI_FLINT_CUDA_FUSED_PUBLISH=0` rollback walk.
struct Recorder {
    picks: bool,
    fused: bool,
    next: usize,
    buffers: Vec<usize>,
    ops: Vec<Op>,
}

impl Recorder {
    fn new(picks: bool) -> Self {
        Self {
            picks,
            fused: true,
            next: 1,
            buffers: vec![H * 4],
            ops: Vec::new(),
        }
    }

    fn rollback(picks: bool) -> Self {
        Self {
            fused: false,
            ..Self::new(picks)
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

    fn prefers_fused_publish(&self) -> bool {
        self.fused
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
        // Rows are derived from the whole-buffer byte size, which is exact
        // for every rows==1 assertion here (each call's input is one whole
        // row); multi-row walks feed gathered views whose span offsets this
        // double ignores, so do not reuse rows_per_call there.
        let mut weights = Vec::with_capacity(calls.len());
        let mut rows_per_call = Vec::with_capacity(calls.len());
        let mut columns = None;
        for call in calls {
            let [k, n] = call.weight.2;
            let stride = k
                .checked_mul(4)
                .filter(|stride| *stride != 0)
                .ok_or_else(|| bad("grouped input width overflows"))?;
            if !self.buffers[call.x].is_multiple_of(stride) {
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

    fn grouped_published(
        &mut self,
        calls: &[Projection<Self::Matrix, Self::Tensor>],
        _outs: &[Self::Tensor],
    ) -> Result<bool, EngError> {
        if !self.fused {
            return Ok(false);
        }
        let weights = calls.iter().map(|call| call.weight.0).collect();
        let columns = calls.first().map(|call| call.weight.2[1]).unwrap_or(0);
        self.ops.push(Op::GroupedPublished { weights, columns });
        Ok(true)
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
    walk_with(&Recorder::new(picks), rows, seed)
}

fn walk_rollback(picks: bool, rows: usize, seed: u64) -> Vec<Op> {
    walk_with(&Recorder::rollback(picks), rows, seed)
}

fn walk_with(template: &Recorder, rows: usize, seed: u64) -> Vec<Op> {
    let ids = routing(rows, seed);
    let grouping = Grouping::new(&ids, rows, EXPERTS).unwrap();
    let mut recorder = Recorder {
        picks: template.picks,
        fused: template.fused,
        next: 1,
        buffers: vec![H * 4],
        ops: Vec::new(),
    };
    run(&mut recorder, block(rows), &ids, &grouping, 0).unwrap();
    recorder.ops
}

/// Every (production, rollback) publish configuration.
type Walk = fn(bool, usize, u64) -> Vec<Op>;
const CONFIGS: [(&str, Walk); 2] = [("production", walk), ("rollback", walk_rollback)];

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
            Op::GroupedPublished { weights, columns } => Op::GroupedPublished {
                weights: vec![0; weights.len()],
                columns: *columns,
            },
            other => other.clone(),
        })
        .collect()
}

/// The identity-bearing launches of a walk, in order: plain grouped in the
/// rollback configuration, fused grouped in the production one.
fn identity_launches(ops: &[Op]) -> Vec<Vec<usize>> {
    ops.iter()
        .filter_map(|op| match op {
            Op::Grouped { weights, .. } | Op::GroupedPublished { weights, .. } => {
                Some(weights.clone())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn rows1_topology_is_invariant_under_the_routers_choices() {
    // Odd seeds only: routing() forces `seed | 1`, so even seeds would repeat
    // the previous route. Seed 1 plus these sixteen gives seventeen distinct
    // routes.
    for (config, walk_fn) in CONFIGS {
        let reference = masked(&walk_fn(false, 1, 1));
        assert!(!reference.is_empty());
        for seed in (3..=33u64).step_by(2) {
            assert_eq!(
                masked(&walk_fn(false, 1, seed)),
                reference,
                "topology changed with the route, {config}, seed {seed}"
            );
        }
        // One launch per projection role (gate, up, down), six calls each,
        // one row per call — and the two configurations must genuinely be
        // different sequences, so both are pinned, not one twice.
        let ops = walk_fn(false, 1, 1);
        let launches = identity_launches(&ops);
        assert_eq!(launches.len(), 3, "{config}: gate, up and down");
        for weights in &launches {
            assert_eq!(weights.len(), TOP_K, "{config}");
        }
        let grouped = ops.iter().filter(|op| matches!(op, Op::Grouped { .. }));
        for op in grouped {
            let Op::Grouped { rows_per_call, .. } = op else {
                unreachable!();
            };
            assert!(rows_per_call.iter().all(|&rows| rows == 1), "{config}");
        }
    }
    let production = walk(false, 1, 1);
    let rollback = walk_rollback(false, 1, 1);
    assert_ne!(
        masked(&production),
        masked(&rollback),
        "the fused and rollback publish paths must differ"
    );
    assert!(
        production
            .iter()
            .any(|op| matches!(op, Op::GroupedPublished { .. }))
    );
    assert!(
        !rollback
            .iter()
            .any(|op| matches!(op, Op::GroupedPublished { .. }))
    );
}

#[test]
fn rows1_host_mode_identity_is_the_only_per_token_launch_datum() {
    for (config, walk_fn) in CONFIGS {
        let ops = walk_fn(false, 1, 9);
        let ids = routing(1, 9);
        // The host names each routed expert exactly once per layer walk...
        let named: Vec<usize> = ops
            .iter()
            .filter_map(|op| match op {
                Op::ExpertId(id) => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(
            named,
            ids.iter().map(|&id| id as usize).collect::<Vec<_>>(),
            "{config}"
        );
        // ...and every identity-bearing launch (gate, up, down) names the
        // same six descriptors in pair order.
        let expected = ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        for weights in identity_launches(&ops) {
            assert_eq!(weights, expected, "{config}");
        }
        // A different route keeps the masked topology but moves the
        // descriptors: the same launch sequence with different pointers,
        // which is exactly what a graph capture would bake.
        let other = walk_fn(false, 1, 11);
        assert_ne!(other, ops, "{config}: routes must differ");
        assert_eq!(masked(&other), masked(&ops), "{config}");
    }
}

#[test]
fn rows1_picks_mode_is_replay_safe_with_the_same_topology() {
    for (config, walk_fn) in CONFIGS {
        let host = walk_fn(false, 1, 21);
        let picks = walk_fn(true, 1, 21);
        // Same launch structure in both modes...
        assert_eq!(masked(&picks), masked(&host), "{config}");
        // ...but the picks mode never asks the host for an id, never uploads
        // indices, and every launch names the canonical bank descriptor
        // (member 0); the device's id buffer selects the member, so replay
        // is exact.
        assert!(picks.iter().all(|op| !matches!(op, Op::ExpertId(_))));
        assert_eq!(
            picks
                .iter()
                .filter(|op| matches!(op, Op::PickDescriptor))
                .count(),
            TOP_K,
            "{config}"
        );
        for weights in identity_launches(&picks) {
            assert!(weights.iter().all(|&weight| weight == 0), "{config}");
        }
    }
}

#[test]
fn rows1_walk_uploads_no_route_indices_or_gathers() {
    for (config, walk_fn) in CONFIGS {
        for picks in [false, true] {
            let ops = walk_fn(picks, 1, 5);
            assert!(
                ops.iter().all(|op| !matches!(op, Op::Gather { .. })),
                "decode gather appeared, {config}, picks {picks}"
            );
            assert!(
                ops.iter().all(|op| !matches!(op, Op::Indices { .. })),
                "decode index upload appeared, {config}, picks {picks}"
            );
        }
    }
    // The multi-row walk does gather and upload indices; the claim above is
    // deliberately decode-only.
    let ops = walk(false, 8, 6);
    assert!(ops.iter().any(|op| matches!(op, Op::Gather { .. })));
    assert!(ops.iter().any(|op| matches!(op, Op::Indices { .. })));
}
