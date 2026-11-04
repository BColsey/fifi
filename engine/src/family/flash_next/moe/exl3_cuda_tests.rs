//! The EXL3 MoE block of the CUDA step (`qwen36::cuda::exl3_moe`, shared with Flash-Next) on one
//! real layer of the Qwen3.8 Flash-Next EXL3 pack, against an fp64 evaluation of the same stored
//! matrices and against this module's host `route`.
//!
//! `FIFI_FLASH_NEXT_EXL3_PACK` names a local pack directory; a copy that holds the headers of
//! every shard and the payload of the chosen layer's MoE tensors is enough.
//! `FIFI_FLASH_NEXT_EXL3_MOE_LAYER` picks the layer (3 by default). Before any upload the header
//! walk checks every tensor of the pack against `config.json` (names, dtypes, shapes, one rate
//! per expert bank, one codebook), and the config's MoE shape must be Flash-Next's: 512 experts,
//! 10 picks, weights renormalized over the picks (the device router's rule).
//!
//! The block feeds a hyper-connection apply in Flash-Next, not a residual add, so it runs on a
//! zero residual and its output is the weighted picks plus the gated shared expert. Each of three
//! input rows runs the block twice: with the picks read back (one grouped launch of the 20 gate
//! and up pairs) and with the picks read on the device (id-indexed launches of 16 and 4 pairs).
//! Both must write the same bytes. Against fp64: the ids in rank order, each pick weight and the
//! shared gate within `2^-17` (the device router's class R1, as in `cuda_tests`), the block output
//! within a relative L2 of 2e-5 (the bound of the Qwen3.6 block's real-pack test). The host `route` (f32) must pick the same ids, the
//! arithmetic flag and the router status must stay clear, and some pick over the three rows must
//! lie in the upper half of the 512-expert banks.

use std::path::Path;

use super::route;
use crate::{
    EngError,
    family::{
        exl3_pack::{
            ExpertPart, Root, expert_name_in, router_name_in, shared_gate_name_in, shared_name_in,
        },
        flash_next::{
            load::{exl3_pack::inspect_flash_pack, model_shape},
            stored::host_f32,
        },
        gguf_model::Role,
        project::cuda::{CudaProjector, CudaWeight, CudaWeights, Reserve},
        qwen36::cuda::{
            exl3_moe::{Exl3MoeBufs, Exl3MoeWeights, Scratch, run_exl3_moe},
            moe::{Launch, MoeDims, RouteBufs},
        },
    },
    kernels::cuda::{
        DeviceBuffer, device_present,
        resident::router::{RouterShape, router_logits_bytes},
    },
    weights::{
        GgufType,
        exl3::{COMPONENT_SUFFIXES, Checkpoint, Matrix},
    },
};

const F32: usize = 4;
const OUTPUT_BOUND: f64 = 2e-5;
/// The bound on a pick weight and on the shared gate, `2^-17` (the router's class R1).
const WEIGHT_BOUND: f64 = 1.0 / 131_072.0;
/// The input rows the block runs on.
const ROWS: usize = 3;

fn le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The stored components of the matrix under `prefix`.
fn matrix(checkpoint: &Checkpoint, prefix: &str) -> Result<Matrix, EngError> {
    let names = COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
    let keys = names.each_ref().map(String::as_str);
    let selected = checkpoint.source.read_selected(&keys, 1 << 30)?;
    Matrix::load(&selected, prefix, &checkpoint.config)
}

/// The key of the bank of every routed expert's `part` of `layer` in the uploaded set.
fn bank_key(layer: usize, part: ExpertPart) -> String {
    format!("layer {layer} routed {part:?} bank")
}

/// The key of each role the shared block resolves.
fn keys(layer: usize) -> Vec<(Role, String)> {
    let root = Root::Layers;
    let mut keys = vec![
        (Role::FfnGateInp, router_name_in(root, layer)),
        (Role::FfnGateInpShexp, shared_gate_name_in(root, layer)),
    ];
    for (bank, shared, part) in [
        (Role::FfnGateExps, Role::FfnGateShexp, ExpertPart::Gate),
        (Role::FfnUpExps, Role::FfnUpShexp, ExpertPart::Up),
        (Role::FfnDownExps, Role::FfnDownShexp, ExpertPart::Down),
    ] {
        keys.push((bank, bank_key(layer, part)));
        keys.push((shared, shared_name_in(root, layer, part)));
    }
    keys
}

/// The host copies of the router matrix and the shared-gate row, widened to F32.
struct Router {
    matrix: Vec<f32>,
    gate: Vec<f32>,
}

impl Router {
    fn read(checkpoint: &Checkpoint, layer: usize, moe: &MoeDims) -> Result<Self, EngError> {
        let root = Root::Layers;
        Ok(Self {
            matrix: host_f32(
                &checkpoint.source,
                &router_name_in(root, layer),
                moe.n_expert * moe.hidden,
            )?,
            gate: host_f32(
                &checkpoint.source,
                &shared_gate_name_in(root, layer),
                moe.hidden,
            )?,
        })
    }
}

/// Uploads the block of `layer`: the router and the shared gate as F32, each routed projection
/// as one bank of every expert in id order, and the shared expert's three matrices.
fn upload(
    projector: &CudaProjector,
    checkpoint: &Checkpoint,
    (layer, router): (usize, &Router),
    moe: &MoeDims,
) -> Result<CudaWeights, EngError> {
    let root = Root::Layers;
    let reserve = Reserve::default();
    let mut weights = Vec::with_capacity(8);
    for (name, values) in [
        (router_name_in(root, layer), &router.matrix),
        (shared_gate_name_in(root, layer), &router.gate),
    ] {
        let buffer = projector.context().upload(&le_bytes(values))?;
        weights.push(CudaWeight::Raw {
            name,
            ty: GgufType::F32,
            buffer,
        });
    }
    for part in ExpertPart::ALL {
        let bank = projector.upload_exl3_bank(moe.n_expert, reserve, |expert| {
            matrix(checkpoint, &expert_name_in(root, layer, expert, part))
        })?;
        weights.push(CudaWeight::Exl3Bank {
            name: bank_key(layer, part),
            bank,
        });
        let name = shared_name_in(root, layer, part);
        let weight = projector.upload_exl3(&matrix(checkpoint, &name)?, reserve)?;
        weights.push(CudaWeight::Exl3 { name, weight });
    }
    CudaWeights::from_exl3(weights)
}

/// Input row `seed`: values in about [-1.4, 1.4], a different pattern per seed.
fn input(hidden: usize, seed: usize) -> Vec<f32> {
    (0..hidden)
        .map(|i| (((i + 613 * seed) * 7919 % 2003) as f32 - 1001.0) / 700.0)
        .collect()
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn project(matrix: &Matrix, input: &[f64]) -> Result<Vec<f64>, EngError> {
    let mut out = vec![0.0; matrix.shape()[1]];
    matrix.project_fp64_into(input, 1, &mut out)?;
    Ok(out)
}

/// One expert on `x`: `down(silu(gate x) * up x)`.
fn expert(parts: &[Matrix], x: &[f64]) -> Result<Vec<f64>, EngError> {
    let (gate, up) = (project(&parts[0], x)?, project(&parts[1], x)?);
    let act: Vec<f64> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
    project(&parts[2], &act)
}

struct Reference {
    ids: Vec<usize>,
    weights: Vec<f64>,
    gate: f64,
    /// The probability gap between the last pick and the best expert left out.
    margin: f64,
    out: Vec<f64>,
}

/// The routing in fp64: softmax over all logits, the top `top_k` by probability then by the
/// lower id, the weights renormalized over the picks, the sigmoid of the shared gate.
fn route_fp64(router: &Router, x: &[f64], top_k: usize) -> (Vec<usize>, Vec<f64>, f64, f64) {
    let dot = |row: &[f32]| -> f64 { row.iter().zip(x).map(|(w, x)| f64::from(*w) * x).sum() };
    let logits: Vec<f64> = router.matrix.chunks_exact(x.len()).map(dot).collect();
    let top = logits.iter().copied().fold(f64::MIN, f64::max);
    let exp: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
    let total: f64 = exp.iter().sum();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| exp[b].total_cmp(&exp[a]).then(a.cmp(&b)));
    let margin = (exp[order[top_k - 1]] - exp[order[top_k]]) / total;
    order.truncate(top_k);
    let picked: f64 = order.iter().map(|&e| exp[e]).sum();
    let weights = order.iter().map(|&e| exp[e] / picked).collect();
    let gate = 1.0 / (1.0 + (-dot(&router.gate)).exp());
    (order, weights, gate, margin)
}

/// The block in fp64 on `row`: every pick's and the shared expert's projections through the fp64
/// reference of the stored matrices.
fn reference(
    checkpoint: &Checkpoint,
    (layer, router): (usize, &Router),
    moe: &MoeDims,
    row: &[f32],
) -> Result<Reference, EngError> {
    let root = Root::Layers;
    let x: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
    let (ids, weights, gate, margin) = route_fp64(router, &x, moe.top_k);
    let mut out = vec![0.0f64; moe.hidden];
    for (&id, weight) in ids.iter().zip(&weights) {
        let parts = ExpertPart::ALL
            .map(|part| matrix(checkpoint, &expert_name_in(root, layer, id, part)))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        for (o, v) in out.iter_mut().zip(expert(&parts, &x)?) {
            *o += weight * v;
        }
    }
    let shared = ExpertPart::ALL
        .map(|part| matrix(checkpoint, &shared_name_in(root, layer, part)))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    for (o, v) in out.iter_mut().zip(expert(&shared, &x)?) {
        *o += gate * v;
    }
    Ok(Reference {
        ids,
        weights,
        gate,
        margin,
        out,
    })
}

struct Device {
    ids: Vec<usize>,
    weights: Vec<f32>,
    gate: f32,
    /// The block output's bytes, compared exactly between the two pick paths.
    out: Vec<u8>,
    flag: u32,
    status: u32,
}

fn read(projector: &CudaProjector, buffer: &DeviceBuffer) -> Result<Vec<u8>, EngError> {
    let stream = projector.context().stream()?;
    let mut bytes = vec![0; buffer.bytes()];
    buffer.read_at(&stream, 0, &mut bytes)?;
    Ok(bytes)
}

fn word(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes[..4].try_into().expect("a word"))
}

/// Runs the block once on `row` over fresh zeroed buffers and reads its results.
fn run_block(
    projector: &CudaProjector,
    block: &Exl3MoeWeights<'_>,
    (moe, row): (&MoeDims, &[f32]),
    device_ids: bool,
) -> Result<Device, EngError> {
    let ctx = projector.context();
    let zeros = |bytes: usize| ctx.upload(&vec![0u8; bytes]);
    let (h, k) = (moe.hidden * F32, moe.top_k);
    let logits = router_logits_bytes(RouterShape {
        m: 1,
        hidden: moe.hidden,
        n_expert: moe.n_expert,
        top_k: k,
    })?;
    // Index: resid_in, resid_out, row, logits, ids, weights, gate, status, gu, act, down,
    // shared_gu, shared_act, shared_out, xh, z, flag.
    let owned = [
        zeros(h)?,
        zeros(h)?,
        ctx.upload(&le_bytes(row))?,
        zeros(logits)?,
        zeros(k * 4)?,
        zeros(k * F32)?,
        zeros(F32)?,
        zeros(4)?,
        zeros(k * 2 * moe.inter * F32)?,
        zeros(k * moe.inter * F32)?,
        zeros(k * h)?,
        zeros(2 * moe.shared_inter * F32)?,
        zeros(moe.shared_inter * F32)?,
        zeros(h)?,
        zeros(2 * k * moe.hidden * F32)?,
        zeros(2 * k * moe.inter.max(moe.hidden) * F32)?,
        zeros(4)?,
    ];
    let r = |i: usize| owned[i].range(0, owned[i].bytes());
    let bufs = Exl3MoeBufs {
        route: RouteBufs {
            resid_in: r(0)?,
            resid_out: r(1)?,
            norm_row: r(2)?,
            logits: r(3)?,
            ids: r(4)?,
            weights: r(5)?,
            shared_gate: r(6)?,
            status: r(7)?,
        },
        expert_gu: r(8)?,
        expert_act: r(9)?,
        expert_down: r(10)?,
        shared_gu: r(11)?,
        shared_act: r(12)?,
        shared_out: r(13)?,
        scratch: Scratch {
            xh: r(14)?,
            z: r(15)?,
            flag: r(16)?,
        },
    };
    let stream = ctx.stream()?;
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    run_exl3_moe(&launch, block, &bufs, moe, device_ids)?;
    stream.synchronize()?;
    let ids = read(projector, &owned[4])?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b) as usize)
        .collect();
    Ok(Device {
        ids,
        weights: floats(&read(projector, &owned[5])?),
        gate: floats(&read(projector, &owned[6])?)[0],
        out: read(projector, &owned[1])?,
        flag: word(&read(projector, &owned[16])?),
        status: word(&read(projector, &owned[7])?),
    })
}

fn relative_l2(actual: &[f32], expected: &[f64]) -> f64 {
    let (mut error, mut norm) = (0.0f64, 0.0f64);
    for (a, e) in actual.iter().zip(expected) {
        error += (f64::from(*a) - e).powi(2);
        norm += e * e;
    }
    (error / norm).sqrt()
}

fn apart(actual: f32, expected: f64) -> f64 {
    (f64::from(actual) - expected).abs()
}

/// Runs both pick paths on input `seed` and checks them against fp64 and the host `route`;
/// returns the picks.
fn check_row(
    projector: &CudaProjector,
    checkpoint: &Checkpoint,
    (layer, router, block): (usize, &Router, &Exl3MoeWeights<'_>),
    (moe, shape, seed): (&MoeDims, &super::MoeShape, usize),
) -> Vec<usize> {
    let row = input(moe.hidden, seed);
    let want = reference(checkpoint, (layer, router), moe, &row).expect("fp64 reference");
    let got = run_block(projector, block, (moe, &row), false).expect("block, read-back picks");
    let on_device = run_block(projector, block, (moe, &row), true).expect("block, device picks");
    let tag = format!("layer {layer} row {seed}");
    assert_eq!(on_device.ids, got.ids, "{tag}: device picks, the ids");
    assert_eq!(
        on_device.out, got.out,
        "{tag}: device picks, the output bytes"
    );
    for run in [&got, &on_device] {
        assert_eq!(run.flag, 0, "{tag}: arithmetic flag");
        assert_eq!(run.status, 0, "{tag}: router status");
    }
    let host: Vec<usize> = route(shape, &router.matrix, &row)
        .expect("host route")
        .into_iter()
        .map(|(id, _)| id as usize)
        .collect();
    assert_eq!(host, want.ids, "{tag}: the host route against fp64");
    assert_eq!(got.ids, want.ids, "{tag}: the device picks against fp64");
    for (g, w) in got.weights.iter().zip(&want.weights) {
        assert!(
            apart(*g, *w) < WEIGHT_BOUND,
            "{tag}: weight {g} against {w}"
        );
    }
    assert!(
        apart(got.gate, want.gate) < WEIGHT_BOUND,
        "{tag}: shared gate"
    );
    let error = relative_l2(&floats(&got.out), &want.out);
    println!(
        "{tag}: picks {:?}, rank-10 margin {:.3e}, shared gate {:.4}, relative L2 {error:.3e}",
        got.ids, want.margin, want.gate
    );
    assert!(error < OUTPUT_BOUND, "{tag}: relative L2 {error}");
    got.ids
}

#[test]
#[ignore = "requires a GPU (2 GiB) and a local Flash-Next EXL3 pack with the layer's MoE payload in FIFI_FLASH_NEXT_EXL3_PACK"]
fn a_real_flash_next_moe_layer_matches_the_fp64_reference() {
    let dir = std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK");
    let layer = std::env::var("FIFI_FLASH_NEXT_EXL3_MOE_LAYER")
        .map_or(3, |value| value.parse().expect("a layer number"));
    assert!(
        device_present("real Flash-Next EXL3 moe block"),
        "a GPU is required"
    );
    let checkpoint = Checkpoint::open(Path::new(&dir)).expect("open the pack");
    inspect_flash_pack(&checkpoint.source, &checkpoint.model_config)
        .expect("the pack matches its config");
    let (shape, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let shape = shape.moe;
    assert_eq!(
        (shape.experts, shape.top_k),
        (512, 10),
        "Flash-Next routing"
    );
    assert!(
        shape.renorm,
        "the device router renormalizes over the picks"
    );
    let moe = MoeDims {
        rows: 1,
        hidden: shape.hidden,
        n_expert: shape.experts,
        top_k: shape.top_k,
        inter: shape.inter,
        shared_inter: shape.shared_inter,
    };
    let projector = CudaProjector::new().expect("projector");
    let router = Router::read(&checkpoint, layer, &moe).expect("router");
    let weights = upload(&projector, &checkpoint, (layer, &router), &moe).expect("upload");
    let keys = keys(layer);
    let block = Exl3MoeWeights::resolve_by(&weights, layer, &moe, |role| {
        keys.iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, key)| key.as_str())
            .ok_or_else(|| EngError::BadRequest(format!("no key for {role:?}")))
    })
    .expect("the block resolves");
    let mut picks = Vec::new();
    for seed in 0..ROWS {
        let context = (layer, &router, &block);
        picks.extend(check_row(
            &projector,
            &checkpoint,
            context,
            (&moe, &shape, seed),
        ));
    }
    assert!(
        picks.iter().any(|&id| id >= moe.n_expert / 2),
        "no pick in the upper half of the banks: {picks:?}"
    );
}
