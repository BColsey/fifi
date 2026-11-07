//! The split MoE block on two real layers of the full pack, against the all-resident block
//! (`qwen36::cuda::exl3_moe::run_exl3_moe` over a bank of every expert) and against an fp64
//! evaluation of the stored matrices.
//!
//! `FIFI_FLASH_NEXT_EXL3_PACK` names the local pack directory (the full 3.05 bpw pack, or a copy
//! with the headers of every shard and the payload of the layers' MoE tensors);
//! `FIFI_FLASH_NEXT_EXL3_MOE_LAYER` picks the first layer (3 by default), the second is the next
//! one. The bank has two rows (bank layer 0 is the first pack layer, bank layer 1 the second) with
//! 16 slots each, so the block runs on a layer other than 0, with a trunk other than the first.
//! The routed experts of the all-resident runs are uploaded as separate banks.
//!
//! Cases, each on the real router's picks for a fixed input row of each layer:
//! - a bank that holds six of the row's ten picks: a mixed pass with four misses, then the same row
//!   again with no miss;
//! - a bank that holds none of the row's picks: ten misses, the resident pass is all filler, then
//!   the same row again.
//!
//! Each case runs with the finite-filler check on and again, on a fresh bank, with it off. With it
//! off the host does not read the stream between the resident pass and the bank's copies, so the
//! copies overlap the resident pass: the default path.
//!
//! Checked: a layer without a miss writes the bytes of the all-resident block; a layer with misses
//! is within a relative L2 of 1e-6 of it (the sums are associated in two groups, every product is
//! the same); both are within the block's bound of fp64; the bank holds every pick afterwards.

use std::path::PathBuf;

use super::{
    bank::{
        BankGeometry, ExpertBankState, HostMemory, HostMode, HostPinPlan, HostTerms, RecordSource,
        ReplaceOnMissLru, device::DeviceBank,
    },
    bank_source::PackRecordSource,
    moe_block::{SplitMoe, SplitMoeTrunk},
};
use crate::{
    EngError,
    family::{
        flash_next::{
            load::{exl3_pack::inspect_flash_pack, model_shape, pack_files::FlashPackFiles},
            moe::exl3_cuda_tests::{
                OUTPUT_BOUND, Reference, Router, floats, input, keys, reference, relative_l2,
                upload,
            },
        },
        gguf_model::Role,
        project::cuda::{CudaProjector, CudaWeight, CudaWeights},
        qwen36::cuda::{
            exl3_moe::{Exl3MoeBufs, Exl3MoeWeights, Scratch, run_exl3_moe},
            moe::{Launch, MoeDims, RouteBufs},
        },
    },
    kernels::cuda::{
        DeviceBuffer, device_present,
        resident::router::{RouterShape, router_logits_bytes},
    },
    weights::exl3::Checkpoint,
};

const F32: usize = 4;
const SLOTS: usize = 16;
/// The two groupings of the sums differ only by association: a handful of float roundings.
const SPLIT_BOUND: f64 = 1e-6;

fn pack_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK"))
}

fn moe_layer() -> usize {
    std::env::var("FIFI_FLASH_NEXT_EXL3_MOE_LAYER")
        .map_or(3, |value| value.parse().expect("a layer number"))
}

/// Bank layer `i` holds pack layer `layers[i]`.
struct Layers<'a, 'b> {
    inner: &'b PackRecordSource<'a>,
    layers: Vec<usize>,
}

impl RecordSource for Layers<'_, '_> {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        self.inner.record(self.layers[layer], expert, out)
    }
}

/// The buffers of one block call, zeroed, with `row` as the normalized input.
struct Rig {
    owned: Vec<DeviceBuffer>,
}

impl Rig {
    fn new(projector: &CudaProjector, moe: &MoeDims, row: &[f32]) -> Result<Self, EngError> {
        let ctx = projector.context();
        let zeros = |bytes: usize| ctx.upload(&vec![0u8; bytes]);
        let (h, k) = (moe.hidden * F32, moe.top_k);
        let logits = router_logits_bytes(RouterShape {
            m: 1,
            hidden: moe.hidden,
            n_expert: moe.n_expert,
            top_k: k,
        })?;
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let owned = vec![
            zeros(h)?,
            zeros(h)?,
            ctx.upload(&bytes)?,
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
        Ok(Self { owned })
    }

    fn bufs(&self) -> Result<Exl3MoeBufs<'_>, EngError> {
        let r = |i: usize| self.owned[i].range(0, self.owned[i].bytes());
        Ok(Exl3MoeBufs {
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
        })
    }

    fn out_bytes(&self, projector: &CudaProjector) -> Vec<u8> {
        let stream = projector.context().stream().expect("stream");
        let mut bytes = vec![0u8; self.owned[1].bytes()];
        self.owned[1]
            .read_at(&stream, 0, &mut bytes)
            .expect("read the output");
        bytes
    }
}

fn raw<'w>(weights: &'w CudaWeights, name: &str) -> &'w DeviceBuffer {
    match weights.get(name) {
        Some(CudaWeight::Raw { buffer, .. }) => buffer,
        _ => panic!("{name} is not a raw tensor"),
    }
}

fn exl3<'w>(
    weights: &'w CudaWeights,
    name: &str,
) -> &'w crate::family::project::cuda::exl3::Exl3Weight {
    match weights.get(name) {
        Some(CudaWeight::Exl3 { weight, .. }) => weight,
        _ => panic!("{name} is not an EXL3 matrix"),
    }
}

/// A bank with one row per entry of `held` (slot `s` of row `r` holds `held[r][s]`) over the pack
/// layers `layers`.
fn build_bank(
    projector: &CudaProjector,
    source: &PackRecordSource<'_>,
    layers: &[usize],
    held: &[Vec<u16>],
) -> DeviceBank {
    let record = source.layout().record_bytes();
    let slots = held[0].len();
    let geometry = BankGeometry::new(layers.len(), 512, 10, slots, record).expect("geometry");
    let rows: Vec<Vec<i32>> = held
        .iter()
        .map(|held| {
            let mut row = vec![-1i32; 512];
            for (slot, &expert) in held.iter().enumerate() {
                row[usize::from(expert)] = slot as i32;
            }
            row
        })
        .collect();
    let row_refs: Vec<&[i32]> = rows.iter().map(Vec::as_slice).collect();
    let counts: Vec<Vec<u64>> = layers.iter().map(|_| (0..512u64).rev().collect()).collect();
    let policy = ReplaceOnMissLru::from_fill(geometry, &row_refs, &counts).expect("the fill");
    let state = ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).expect("state");
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &HostMemory {
            total: 62 << 30,
            available: 55 << 30,
        },
        &HostTerms::DEFAULT,
    )
    .expect("the host plan");
    DeviceBank::build(
        projector.context(),
        state,
        &Layers {
            inner: source,
            layers: layers.to_vec(),
        },
        &plan,
    )
    .expect("build the bank")
}

/// `count` experts that are not in `avoid`, lowest ids first.
fn others(avoid: &[usize], count: usize) -> Vec<u16> {
    (0..512u16)
        .filter(|&e| !avoid.contains(&usize::from(e)))
        .take(count)
        .collect()
}

/// One pack layer's uploads and its references on its own input row.
struct Prepared {
    row: Vec<f32>,
    weights: CudaWeights,
    keys: Vec<(Role, String)>,
    want: Reference,
    resident: Vec<u8>,
}

impl Prepared {
    fn key(&self, role: Role) -> &str {
        self.keys
            .iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, key)| key.as_str())
            .unwrap_or_else(|| panic!("no key for {role:?}"))
    }

    fn trunk(&self) -> SplitMoeTrunk<'_> {
        SplitMoeTrunk {
            router: raw(&self.weights, self.key(Role::FfnGateInp)),
            router_shared: raw(&self.weights, self.key(Role::FfnGateInpShexp)),
            shared_gate: exl3(&self.weights, self.key(Role::FfnGateShexp)),
            shared_up: exl3(&self.weights, self.key(Role::FfnUpShexp)),
            shared_down: exl3(&self.weights, self.key(Role::FfnDownShexp)),
        }
    }

    fn picks(&self) -> &[usize] {
        &self.want.ids
    }
}

fn prepare(
    projector: &CudaProjector,
    checkpoint: &Checkpoint,
    (layer, seed): (usize, usize),
    moe: &MoeDims,
    launch: &Launch<'_>,
) -> Prepared {
    let router = Router::read(checkpoint, layer, moe).expect("router");
    let weights = upload(projector, checkpoint, (layer, &router), moe).expect("upload");
    let keys = keys(layer);
    let row = input(moe.hidden, seed);
    let want = reference(checkpoint, (layer, &router), moe, &row).expect("fp64 reference");
    // The all-resident output on the same row, one launch over every expert.
    let resident = {
        let block = Exl3MoeWeights::resolve_by(&weights, layer, moe, |role| {
            keys.iter()
                .find(|(candidate, _)| *candidate == role)
                .map(|(_, key)| key.as_str())
                .ok_or_else(|| EngError::BadRequest(format!("no key for {role:?}")))
        })
        .expect("the all-resident block resolves");
        let rig = Rig::new(projector, moe, &row).expect("rig");
        run_exl3_moe(launch, &block, &rig.bufs().expect("bufs"), moe, true).expect("block");
        launch.stream.synchronize().expect("sync");
        rig.out_bytes(projector)
    };
    let error = relative_l2(&floats(&resident), &want.out);
    assert!(error < OUTPUT_BOUND, "layer {layer} all-resident {error}");
    Prepared {
        row,
        weights,
        keys,
        want,
        resident,
    }
}

#[test]
#[ignore = "needs a GPU (about 4 GiB), about 2 GiB of pinned host memory per bank and FIFI_FLASH_NEXT_EXL3_PACK"]
fn a_real_layer_split_over_a_partial_bank_matches_the_all_resident_block_and_fp64() {
    assert!(
        device_present("Flash-Next split MoE block"),
        "a GPU is required"
    );
    let first = moe_layer();
    let layers = vec![first, first + 1];
    let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
    let checkpoint = &files.checkpoint;
    inspect_flash_pack(&checkpoint.source, &checkpoint.model_config)
        .expect("the pack matches its config");
    let (model, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let shape = model.moe;
    let moe = MoeDims {
        rows: 1,
        hidden: model.hyper.hidden(),
        n_expert: shape.experts(),
        top_k: shape.top_k(),
        inter: shape.inter(),
        shared_inter: shape.shared_inter(),
    };
    assert_eq!((moe.n_expert, moe.top_k), (512, 10));
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let prepared: Vec<Prepared> = layers
        .iter()
        .enumerate()
        .map(|(index, &layer)| prepare(&projector, checkpoint, (layer, index), &moe, &launch))
        .collect();
    let source = PackRecordSource::new(checkpoint).expect("the record source");
    let layout = source.layout().clone();

    for verify in [true, false] {
        for case in ["mixed", "forced miss"] {
            // Per bank layer: the experts its slots hold and the misses its row will cause.
            let held: Vec<Vec<u16>> = prepared
                .iter()
                .map(|p| match case {
                    // Six of the ten picks held, ten other experts: four misses.
                    "mixed" => {
                        let mut held: Vec<u16> =
                            p.picks().iter().take(6).map(|&e| e as u16).collect();
                        held.extend(others(p.picks(), SLOTS - 6));
                        held
                    }
                    // None held: ten misses, the resident pass is all filler.
                    _ => others(p.picks(), SLOTS),
                })
                .collect();
            let expected_misses: Vec<usize> = prepared
                .iter()
                .zip(&held)
                .map(|(p, held)| {
                    p.picks()
                        .iter()
                        .filter(|&&pick| !held.contains(&(pick as u16)))
                        .count()
                })
                .collect();
            let bank = build_bank(&projector, &source, &layers, &held);
            let trunks = prepared.iter().map(Prepared::trunk).collect();
            let mut split = SplitMoe::new(ctx, trunks, bank, &layout, &moe).expect("split");
            split.verify_finite_filler(verify);
            let mut errors = Vec::new();
            for run in 0..2 {
                // Bank layer 1 first, then layer 0.
                for bank_layer in [1usize, 0] {
                    let p = &prepared[bank_layer];
                    let expect = if run == 0 {
                        expected_misses[bank_layer]
                    } else {
                        0
                    };
                    let rig = Rig::new(&projector, &moe, &p.row).expect("rig");
                    let before = split.bank().state().stats().misses;
                    split
                        .run(&launch, bank_layer, &rig.bufs().expect("bufs"), &moe)
                        .expect("the split block");
                    stream.synchronize().expect("sync");
                    let got = rig.out_bytes(&projector);
                    let misses = (split.bank().state().stats().misses - before) as usize;
                    let tag = format!(
                        "{case}, filler check {}, run {run}, bank layer {bank_layer} (pack layer {})",
                        if verify { "on" } else { "off" },
                        layers[bank_layer]
                    );
                    assert_eq!(misses, expect, "{tag}: misses");
                    let resident_error = relative_l2(&floats(&got), &floats_f64(&p.resident));
                    let fp64_error = relative_l2(&floats(&got), &p.want.out);
                    println!(
                        "{tag}: {misses} misses, vs all-resident {resident_error:.3e}, vs fp64 \
                         {fp64_error:.3e}"
                    );
                    assert!(fp64_error < OUTPUT_BOUND, "{tag}: fp64 {fp64_error}");
                    if expect == 0 {
                        assert_eq!(got, p.resident, "{tag}: no miss writes the resident bytes");
                    } else {
                        assert!(resident_error < SPLIT_BOUND, "{tag}: {resident_error}");
                    }
                    errors.push(resident_error);
                }
            }
            for (bank_layer, p) in prepared.iter().enumerate() {
                for &pick in p.picks() {
                    assert!(
                        split.bank().state().slot_row(bank_layer)[pick] >= 0,
                        "{case}: pick {pick} of bank layer {bank_layer} is in the bank"
                    );
                }
            }
        }
    }
}

fn floats_f64(bytes: &[u8]) -> Vec<f64> {
    floats(bytes).into_iter().map(f64::from).collect()
}
