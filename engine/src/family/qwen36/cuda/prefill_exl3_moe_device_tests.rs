//! The chunk MoE block of an EXL3 pack against fp64.
//!
//! `FIFI_EXL3_CHECKPOINT` names the pack directory. The test loads the MoE tensors of one layer,
//! runs the router and the chunk block over `ROWS` deterministic normalized rows and residuals,
//! and compares the block output with a host evaluation: every row's picked experts and the
//! shared expert through the fp64 reference of the stored matrices, SiLU in fp64, the picks and
//! weights as the device router wrote them. The router itself is compared by its picks only: a
//! row whose top experts tie to F32 rounding may flip.
//!
//! The chunk is tall enough that the busiest experts take tensor-core tiles (64 rows or more)
//! and the least busy take the split-K tail (8 rows or fewer), so the block crosses every tile
//! kernel of the rows entry.

use std::path::Path;

use super::{Bufs, run};
use crate::{
    family::{
        project::cuda::{CudaProjector, CudaWeights, Reserve},
        qwen36::{
            cuda::{
                common_tests::{dims, inventory},
                exl3_load_device_tests::{matrix_of, relative_l2, values_of},
                exl3_moe::Exl3MoeWeights,
                exl3_moe_source::moe_weights,
                exl3_source::PackReader,
                layer_table::LayerTable,
                moe::{Launch, MoeDims},
                prefill::{RouteIo, run_chunk_route},
                prefill_plan::exl3_max_jobs,
            },
            exl3_pack::{ExpertPart, expert_name, shared_name},
        },
    },
    kernels::cuda::{
        DeviceBuffer, device_present,
        resident::router::{RouterShape, router_logits_bytes},
    },
    weights::exl3::{Checkpoint, Matrix},
};

use super::super::{
    F32,
    device_tests::{Scratch, filled, read, upload, whole},
};

/// Rows of the chunk: 1,024 rows * 8 picks over 256 experts is 32 rows per expert on average.
const ROWS: usize = 1024;
/// Relative L2 of the block output against fp64 (measured 7.3e-6).
const BOUND: f64 = 1e-4;
/// The least share of rows whose picks must equal the fp64 router's.
const PICK_AGREEMENT: f64 = 0.99;

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn project(matrix: &Matrix, input: &[f64], rows: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * matrix.shape()[1]];
    matrix.project_fp64_into(input, rows, &mut out).unwrap();
    out
}

/// Row `t` of the normalized input: a deterministic spread that changes with the row.
fn row_of(t: usize, hidden: usize) -> Vec<f32> {
    (0..hidden)
        .map(|i| (((i * 7919 + t * 104_729 + t * i * 31) % 2003) as f32 - 1001.0) / 700.0)
        .collect()
}

fn resid_of(t: usize, hidden: usize) -> Vec<f32> {
    (0..hidden)
        .map(|i| (((i * 31 + t * 17) % 97) as f32 - 48.0) / 50.0)
        .collect()
}

/// `down(silu(gate x) * up x)` for `rows` rows of `x` (f64, `[rows][hidden]`).
fn expert(parts: &[Matrix; 3], x: &[f64], rows: usize) -> Vec<f64> {
    let (gate, up) = (project(&parts[0], x, rows), project(&parts[1], x, rows));
    let act: Vec<f64> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
    project(&parts[2], &act, rows)
}

/// The fp64 evaluation of the block given the device's picks, weights and shared gates.
fn reference(
    checkpoint: &Checkpoint,
    (layer, rows_n): (usize, usize),
    moe: &MoeDims,
    device: &Picks,
) -> Vec<f64> {
    let h = moe.hidden;
    let load = |prefix: String| matrix_of(checkpoint, &prefix);
    let rows: Vec<Vec<f32>> = (0..rows_n).map(|t| row_of(t, h)).collect();
    let mut out: Vec<f64> = (0..rows_n)
        .flat_map(|t| resid_of(t, h).into_iter().map(f64::from))
        .collect();
    for e in 0..moe.n_expert {
        let pairs: Vec<(usize, usize)> = (0..rows_n * moe.top_k)
            .filter(|&p| device.ids[p] == e)
            .map(|p| (p / moe.top_k, p))
            .collect();
        if pairs.is_empty() {
            continue;
        }
        let parts = ExpertPart::ALL.map(|part| load(expert_name(layer, e, part)));
        let x: Vec<f64> = pairs
            .iter()
            .flat_map(|&(t, _)| rows[t].iter().map(|&v| f64::from(v)))
            .collect();
        let down = expert(&parts, &x, pairs.len());
        for (i, &(t, p)) in pairs.iter().enumerate() {
            let weight = f64::from(device.weights[p]);
            for c in 0..h {
                out[t * h + c] += weight * down[i * h + c];
            }
        }
    }
    let shared = ExpertPart::ALL.map(|part| load(shared_name(layer, part)));
    let x: Vec<f64> = rows.iter().flatten().map(|&v| f64::from(v)).collect();
    let down = expert(&shared, &x, rows_n);
    for t in 0..rows_n {
        let gate = f64::from(device.shared_gate[t]);
        for c in 0..h {
            out[t * h + c] += gate * down[t * h + c];
        }
    }
    out
}

struct Picks {
    ids: Vec<usize>,
    weights: Vec<f32>,
    shared_gate: Vec<f32>,
}

/// The picks of the forced case: every row picks expert 5 (more than two tiles of rows), half the
/// rows pick expert 9 and the other half expert 10 (more than one tile each), and the other five
/// ranks are distinct experts that spread with the row.
fn forced_picks(rows: usize, top_k: usize, n_expert: usize) -> Vec<i32> {
    let mut picks = Vec::with_capacity(rows * top_k);
    for t in 0..rows {
        let mut row = vec![5, if t % 2 == 0 { 9 } else { 10 }];
        let mut next = 17 + 3 * t;
        while row.len() < top_k {
            next += 7;
            let expert = (next % n_expert) as i32;
            if !row.contains(&expert) {
                row.push(expert);
            }
        }
        picks.extend(row);
    }
    picks
}

#[test]
#[ignore = "requires admitted GPU memory (4 GiB) and the pinned EXL3 pack in FIFI_EXL3_CHECKPOINT"]
fn the_chunk_moe_block_matches_fp64_at_every_tile_kernel() {
    moe_case(ROWS, false);
}

/// Forced picks (the router is bypassed) put 320 rows on one expert and 160 on two others, so the
/// tile loop of an expert runs past one and two full tiles, with the first row of each tile inside
/// the expert's rows. Mutation: a tile loop that restarts at the expert's first row, or one that
/// drops the tail tile, fails the fp64 comparison.
#[test]
#[ignore = "requires admitted GPU memory (4 GiB) and the pinned EXL3 pack in FIFI_EXL3_CHECKPOINT"]
fn one_expert_with_more_than_two_tiles_of_rows_matches_fp64() {
    moe_case(FORCED_ROWS, true);
}

/// Rows of the forced case: 320 rows on expert 5 make a 144-row, a 144-row and a 32-row tile.
const FORCED_ROWS: usize = 320;

fn moe_case(rows: usize, forced: bool) {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    assert!(device_present("EXL3 chunk moe"), "a GPU is required");
    let checkpoint = Checkpoint::open(Path::new(&dir)).expect("open the pack");
    let d = dims();
    let table = LayerTable::new(&d, &inventory()).unwrap();
    let moe = MoeDims::new(&d, rows);
    let projector = CudaProjector::new().expect("projector");
    let layer = 3;
    let reader = PackReader::new(&projector, &checkpoint, Reserve::default());
    let loaded = moe_weights(&reader, &table, layer, &moe).expect("moe tensors");
    let weights = CudaWeights::from_exl3(loaded).unwrap();
    let block = Exl3MoeWeights::resolve(&weights, &table, layer, &moe).unwrap();
    let (h, i, s, k) = (moe.hidden, moe.inter, moe.shared_inter, moe.top_k);
    let pairs = rows * k;
    let x2: Vec<f32> = (0..rows).flat_map(|t| row_of(t, h)).collect();
    let resid: Vec<f32> = (0..rows).flat_map(|t| resid_of(t, h)).collect();
    let logits = router_logits_bytes(RouterShape {
        m: rows,
        hidden: h,
        n_expert: moe.n_expert,
        top_k: k,
    })
    .unwrap();
    let widest = h.max(i).max(s);
    let (ids_init, weights_init, gate_init) = if forced {
        let ids: Vec<u8> = forced_picks(rows, k, moe.n_expert)
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect();
        (
            projector.context().upload(&ids).unwrap(),
            upload(&projector, &vec![1.0 / k as f32; pairs]),
            upload(&projector, &vec![0.5; rows]),
        )
    } else {
        (
            filled(&projector, 0, pairs * 4),
            filled(&projector, 0, pairs * F32),
            filled(&projector, 0, rows * F32),
        )
    };
    let owned: Vec<DeviceBuffer> = vec![
        upload(&projector, &x2),    // 0 x2
        ids_init,                   // 1 ids
        weights_init,               // 2 weights
        gate_init,                  // 3 shared gate
        upload(&projector, &resid), // 4 hidden
        filled(
            &projector,
            0,
            4 * (2 * pairs + 4 * exl3_max_jobs(moe.n_expert, pairs)),
        ), // 5 perm
        filled(&projector, 0, 4),   // 6 (spare)
        filled(&projector, 0, pairs * 2 * i * F32), // 7 gu
        filled(&projector, 0, pairs * i * F32), // 8 act
        filled(&projector, 0, pairs * h * F32), // 9 paired
        filled(&projector, 0, rows * 2 * s * F32), // 10 shared gu
        filled(&projector, 0, rows * s * F32), // 11 shared act
        filled(&projector, 0, rows * h * F32), // 12 shared out
        filled(&projector, 0, logits), // 13 logits
        filled(&projector, 0, 4),   // 14 status
        filled(&projector, 0, pairs * h.max(i) * F32), // 15 expert xh
        filled(&projector, 0, pairs * h.max(i) * F32), // 16 expert z
    ];
    let scratch = Scratch::new(&projector, rows, (widest, widest));
    let r = |index: usize| whole(&owned[index]);
    let ctx = projector.context();
    let stream = ctx.stream().unwrap();
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let projector_view = scratch.chunk(&launch, rows);
    if !forced {
        run_chunk_route(
            &launch,
            &RouteIo {
                m: rows,
                dims: moe,
                x2: r(0),
                logits: r(13),
                ids: r(1),
                weights: r(2),
                shared_gate: r(3),
                status: r(14),
                router: block.router,
                router_shared: block.router_shared,
            },
        )
        .unwrap();
    }
    let bufs = Bufs {
        x2: r(0),
        ids: r(1),
        weights: r(2),
        shared_gate: r(3),
        hidden: r(4),
        perm: r(5),
        gu: r(7),
        act: r(8),
        paired: r(9),
        expert_xh: r(15),
        expert_z: r(16),
        shared_gu: r(10),
        shared_act: r(11),
        shared_out: r(12),
    };
    run(&projector_view, &block, &bufs, &moe).expect("chunk block");
    stream.synchronize().unwrap();
    let ids: Vec<usize> = read(&projector, &owned[1], pairs)
        .iter()
        .map(|v| v.to_bits() as usize)
        .collect();
    let picks = Picks {
        ids,
        weights: read(&projector, &owned[2], pairs),
        shared_gate: read(&projector, &owned[3], rows),
    };
    let got = read(&projector, &owned[4], rows * h);
    let want = reference(&checkpoint, (layer, rows), &moe, &picks);
    let error = relative_l2(&got, &want);
    let busiest = (0..moe.n_expert)
        .map(|e| picks.ids.iter().filter(|&&p| p == e).count())
        .max()
        .unwrap();
    println!("layer {layer}: {rows} rows, busiest expert {busiest} rows, relative L2 {error:.3e}");
    assert!(
        busiest >= if forced { 289 } else { 64 },
        "the chunk reaches the tensor-core tiles (forced: past two full tiles)"
    );
    assert!(error < BOUND, "block output: {error}");
    assert_eq!(scratch.flag_word(&projector), 0, "flag");
    assert_eq!(read(&projector, &owned[14], 1)[0].to_bits(), 0, "status");
    if forced {
        return;
    }
    // The router: the picks of the fp64 router on the same rows.
    let router = values_of(
        &checkpoint,
        &crate::family::qwen36::exl3_pack::router_name(layer),
    );
    let mut agree = 0;
    for t in 0..rows {
        let x = row_of(t, h);
        let mut scores: Vec<(f64, usize)> = router
            .chunks_exact(h)
            .enumerate()
            .map(|(e, w)| {
                (
                    w.iter()
                        .zip(&x)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum(),
                    e,
                )
            })
            .collect();
        scores.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut want: Vec<usize> = scores[..k].iter().map(|s| s.1).collect();
        let mut have = picks.ids[t * k..(t + 1) * k].to_vec();
        want.sort_unstable();
        have.sort_unstable();
        agree += usize::from(want == have);
    }
    assert!(
        agree as f64 >= PICK_AGREEMENT * rows as f64,
        "router picks agree on {agree} of {rows} rows"
    );
}
