//! Layer 3 of the real Qwen3.8 Flash-Next 3.05 bpw pack on the host: the first real-weight tests of
//! the expert tier. `FIFI_FLASH_NEXT_EXL3_PACK` names a local pack directory (a sparse copy that holds
//! the headers of every shard and the payload of layer 3's MoE tensors is enough);
//! `FIFI_FLASH_NEXT_EXL3_MOE_LAYER` picks another layer (3 by default). No GPU.
//!
//! - The real router at top-k 10: the host `route`, a popularity profile, the tier plan's hot set,
//!   and the classifier oracle on routes outside the profile.
//! - The CPU cold-expert kernel (`ExpertCpu`, AVX-512) on real routed experts at 3 bpw and on the
//!   shared expert at 5 bpw, against an fp64 evaluation of the same stored matrices.
//! - The pack's trunk bytes against the plan's 3.59 GiB.

use std::path::Path;
use std::time::Duration;

use half::f16;

use super::{
    route,
    tier_plan::{RouteCounts, SlotTable, TierInputs, TierPlan, TierShape, position_bytes},
};
use crate::{
    family::{
        exl3_pack::{ExpertPart, Root, expert_name_in, router_name_in, shared_name_in},
        flash_next::{
            load::{exl3_pack::inspect_flash_pack, model_shape},
            stored::host_f32,
        },
    },
    kernels::{
        exl3_cpu::{Exl3Isa, Exl3Projection, ExpertCpu, ExpertPick, ExpertView},
        nvfp4_cpu::WaitPolicy,
    },
    tensor::Dtype,
    weights::exl3::{COMPONENT_SUFFIXES, Checkpoint, Matrix},
};

const GIB: u64 = 1 << 30;
/// Relative L2 bound of the CPU kernel against fp64 on one expert block. The fp64 oracle takes the
/// codebook weights rounded to binary16 once (the device kernels' rule); the CPU kernel evaluates the
/// exact affine value, so the two differ by that rounding: about 2^-12 per weight, which averages to a
/// few times 1e-4 of the output (measured 3.5e-4 on a real 3 bpw expert). The bound is 1e-3.
const CPU_BOUND: f64 = 1e-3;

pub(super) fn open() -> (Checkpoint, usize) {
    let dir = std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK");
    let layer = std::env::var("FIFI_FLASH_NEXT_EXL3_MOE_LAYER")
        .map_or(3, |value| value.parse().expect("a layer number"));
    (
        Checkpoint::open(Path::new(&dir)).expect("open the pack"),
        layer,
    )
}

/// Input row `seed`: values in about [-1.4, 1.4], a different pattern per seed.
pub(super) fn input(hidden: usize, seed: usize) -> Vec<f32> {
    (0..hidden)
        .map(|i| (((i + 613 * seed) * 7919 % 2003) as f32 - 1001.0) / 700.0)
        .collect()
}

fn matrix(checkpoint: &Checkpoint, prefix: &str) -> Matrix {
    let names = COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
    let keys = names.each_ref().map(String::as_str);
    let selected = checkpoint
        .source
        .read_selected(&keys, 1 << 30)
        .unwrap_or_else(|error| panic!("{prefix}: {error:?}"));
    Matrix::load(&selected, prefix, &checkpoint.config).expect("a matrix")
}

/// The plan on the 3090 at `positions` of f16 K/V, with the trunk the header walk found.
pub(super) fn plan_at(tier: TierShape, trunk: u64, positions: u64) -> TierPlan {
    let bytes = position_bytes(12, 2, 256, 2, (128, 4)).expect("position bytes");
    let inputs = TierInputs::rtx3090(
        tier,
        (trunk, 36 * 3 * (1 << 20)),
        crate::family::qwen36::tiered::budget::KvPlan {
            positions,
            bytes_per_position: bytes,
        },
    );
    TierPlan::new(&inputs).expect("the plan")
}

/// The tier shape and the trunk bytes of the pack.
pub(super) fn pack_sizes(checkpoint: &Checkpoint) -> (TierShape, u64, usize) {
    let report = inspect_flash_pack(&checkpoint.source, &checkpoint.model_config).expect("walk");
    let (shape, kinds) = model_shape(&checkpoint.model_config).expect("the model shape");
    let moe = shape.moe;
    let tier = TierShape::from_pack(kinds.len(), moe.experts, moe.top_k, report.expert_bytes)
        .expect("the pack's experts share one size");
    let embedding = shape.vocab as u64 * moe.hidden as u64 * 2;
    (
        tier,
        report.pack.device_bytes - report.expert_bytes - embedding,
        moe.hidden,
    )
}

#[test]
#[ignore = "reads layer 3 of the real pack in FIFI_FLASH_NEXT_EXL3_PACK (about 60 MiB of the router and headers)"]
fn the_real_router_picks_ten_and_the_tier_classifies_routes_outside_its_profile() {
    let (checkpoint, layer) = open();
    let report = inspect_flash_pack(&checkpoint.source, &checkpoint.model_config).expect("walk");
    let (shape, kinds) = model_shape(&checkpoint.model_config).expect("the model shape");
    let moe = shape.moe;
    assert_eq!((moe.experts, moe.top_k, moe.renorm), (512, 10, true));
    let tier = TierShape::from_pack(kinds.len(), moe.experts, moe.top_k, report.expert_bytes)
        .expect("the pack's experts share one size");
    assert_eq!(tier.expert_bytes, 1_881_600);

    // The trunk the plan assumes: everything on the device but the experts and the embedding
    // (which the host gathers), against the plan doc's 3.59 GiB.
    let embedding = shape.vocab as u64 * moe.hidden as u64 * 2;
    let trunk = report.pack.device_bytes - report.expert_bytes - embedding;
    eprintln!("trunk bytes {trunk} ({:.3} GiB)", trunk as f64 / GIB as f64);
    assert!(
        trunk.abs_diff((3.59 * GIB as f64) as u64) < GIB / 50,
        "trunk {trunk}"
    );

    let router = host_f32(
        &checkpoint.source,
        &router_name_in(Root::Layers, layer),
        moe.experts * moe.hidden,
    )
    .expect("the real router");
    let routes: Vec<Vec<(u32, f32)>> = (0..128)
        .map(|seed| route(&moe, &router, &input(moe.hidden, seed)).expect("route"))
        .collect();
    for (seed, picks) in routes.iter().enumerate() {
        let mut ids: Vec<u32> = picks.iter().map(|p| p.0).collect();
        assert_eq!(ids.len(), 10, "row {seed}");
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 10, "row {seed}: ten distinct experts");
        assert!(ids.iter().all(|&id| (id as usize) < moe.experts));
        let sum: f32 = picks.iter().map(|p| p.1).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "row {seed}: renormalized weights sum to {sum}"
        );
    }

    let plan = plan_at(tier, trunk, 131_072);
    eprintln!(
        "slots per layer {}, bank {} bytes, cold {} bytes",
        plan.slots_per_layer(),
        plan.bank_bytes(),
        plan.cold_bytes()
    );
    assert!((170..=190).contains(&plan.slots_per_layer()));

    // Profile on the first 96 routes of this layer, test on the last 32.
    let mut counts = RouteCounts::new(kinds.len(), moe.experts);
    for picks in &routes[..96] {
        let ids: Vec<u32> = picks.iter().map(|p| p.0).collect();
        counts.record(layer, &ids).unwrap();
    }
    let hot = plan.hot_set(counts.counts()).expect("the hot set");
    let table: &SlotTable = &hot.table;
    let (mut hits, mut picks_seen) = (0usize, 0usize);
    for picks in &routes[96..] {
        let ids: Vec<i32> = picks.iter().map(|p| p.0 as i32).collect();
        let weights: Vec<f32> = picks.iter().map(|p| p.1).collect();
        let got = table.classify(layer, &ids, &weights).expect("classify");
        assert_eq!(got.word, 0);
        assert_eq!(got.slot_ids.len(), 10);
        for (index, &id) in ids.iter().enumerate() {
            picks_seen += 1;
            match table.slot_of(layer, id as usize) {
                Some(slot) => {
                    hits += 1;
                    assert_eq!(got.slot_ids[index], slot as i32);
                    assert_eq!(got.weights[index].to_bits(), weights[index].to_bits());
                }
                None => {
                    assert_eq!(got.slot_ids[index], 0);
                    assert_eq!(got.weights[index].to_bits(), 0.0f32.to_bits());
                }
            }
        }
        let cold = ids
            .iter()
            .filter(|&&id| table.slot_of(layer, id as usize).is_none())
            .count();
        assert_eq!(got.misses.len(), cold);
    }
    eprintln!("hot-set hits {hits} of {picks_seen} picks on routes outside the profile");
    assert!(hits > 0, "a profile that never hits is no profile");
}

/// One expert's three matrices, widened for the CPU kernel.
struct Stored {
    parts: [Matrix; 3],
    bytes: [[Vec<u8>; 3]; 3],
}

fn widened(tensor: &crate::tensor::Tensor) -> Vec<u8> {
    let bytes = tensor.host_bytes().expect("host bytes");
    match tensor.dtype() {
        Dtype::F32 => bytes.to_vec(),
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|b| f16::from_le_bytes(*b).to_f32().to_le_bytes())
            .collect(),
        other => panic!("scale stored as {other:?}"),
    }
}

fn stored(checkpoint: &Checkpoint, prefixes: [String; 3]) -> Stored {
    let parts = prefixes.map(|prefix| matrix(checkpoint, &prefix));
    let bytes = [0, 1, 2].map(|i| {
        [
            parts[i].trellis().host_bytes().expect("trellis").to_vec(),
            widened(parts[i].suh()),
            widened(parts[i].svh()),
        ]
    });
    Stored { parts, bytes }
}

impl Stored {
    fn view(&self) -> ExpertView<'_> {
        let projection = |i: usize| {
            let m = &self.parts[i];
            Exl3Projection::new(
                m.shape(),
                usize::from(m.bitrate_twice()) * 8,
                &self.bytes[i][0],
                &self.bytes[i][1],
                &self.bytes[i][2],
            )
            .expect("a CPU projection")
        };
        ExpertView {
            gate: projection(0),
            up: projection(1),
            down: projection(2),
        }
    }

    /// `down(silu(gate x) * up x)` in fp64.
    fn fp64(&self, x: &[f64]) -> Vec<f64> {
        let project = |m: &Matrix, input: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; m.shape()[1]];
            m.project_fp64_into(input, 1, &mut out)
                .expect("fp64 projection");
            out
        };
        let (gate, up) = (project(&self.parts[0], x), project(&self.parts[1], x));
        let act: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        project(&self.parts[2], &act)
    }
}

fn relative_l2(got: &[f32], want: &[f64]) -> f64 {
    let diff: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).powi(2))
        .sum();
    let norm: f64 = want.iter().map(|w| w * w).sum();
    (diff / norm).sqrt()
}

#[test]
#[ignore = "reads real experts of layer 3 in FIFI_FLASH_NEXT_EXL3_PACK (about 12 MiB) and needs AVX-512"]
fn the_cpu_kernel_matches_fp64_on_real_experts_and_the_shared_expert() {
    if Exl3Isa::detect() != Exl3Isa::Avx512 {
        eprintln!("skipped: no AVX-512 on this host");
        return;
    }
    let (checkpoint, layer) = open();
    let (shape, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let hidden = shape.moe.hidden;
    let root = Root::Layers;
    let cpu = ExpertCpu::new(
        4,
        WaitPolicy::SpinThenPark(Duration::from_micros(500)),
        Some(Exl3Isa::Avx512),
    )
    .expect("the CPU kernel");
    let mut cases: Vec<(String, Stored)> = [0usize, 255, 511]
        .into_iter()
        .map(|expert| {
            let prefixes = ExpertPart::ALL.map(|part| expert_name_in(root, layer, expert, part));
            (
                format!("expert {expert} (3 bpw)"),
                stored(&checkpoint, prefixes),
            )
        })
        .collect();
    let shared = ExpertPart::ALL.map(|part| shared_name_in(root, layer, part));
    cases.push(("shared expert (5 bpw)".into(), stored(&checkpoint, shared)));
    for seed in 0..3 {
        let x = input(hidden, seed);
        let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        for (name, expert) in &cases {
            let mut out = vec![0.0f32; hidden];
            cpu.forward(
                &x,
                &[ExpertPick {
                    expert: expert.view(),
                    weight: 1.0,
                }],
                &mut out,
            )
            .expect("the CPU forward");
            let error = relative_l2(&out, &expert.fp64(&x64));
            eprintln!("{name}, row {seed}: relative L2 {error:.3e}");
            assert!(error <= CPU_BOUND, "{name}, row {seed}: {error:.3e}");
        }
    }
    // Two experts weighted and summed in pick order, as the tier's cold path combines them.
    let x = input(hidden, 7);
    let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let weights = [0.625f32, 0.375];
    let mut out = vec![0.0f32; hidden];
    cpu.forward(
        &x,
        &[
            ExpertPick {
                expert: cases[0].1.view(),
                weight: weights[0],
            },
            ExpertPick {
                expert: cases[1].1.view(),
                weight: weights[1],
            },
        ],
        &mut out,
    )
    .expect("the CPU forward");
    let want: Vec<f64> = cases[0]
        .1
        .fp64(&x64)
        .iter()
        .zip(cases[1].1.fp64(&x64))
        .map(|(a, b)| f64::from(weights[0]) * a + f64::from(weights[1]) * b)
        .collect();
    let error = relative_l2(&out, &want);
    eprintln!("two weighted picks: relative L2 {error:.3e}");
    assert!(error <= CPU_BOUND, "{error:.3e}");
}
