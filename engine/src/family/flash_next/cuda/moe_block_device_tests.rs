//! The split MoE block on two real layers of the full pack, against the all-resident block
//! (`family::exl3::moe::run_exl3_moe` over a bank of every expert) and against an fp64
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
//! The default path runs with no host read between the resident pass and the bank's copies, so
//! the copies overlap the resident pass.
//!
//! Checked: every run — misses or none — writes the BYTES of the all-resident block (one
//! rank-order combine, so the output does not depend on the slot count, the miss split or the
//! LRU state); all are within the block's bound of fp64; the bank holds every pick afterwards.
//! The slot-count independence is pinned directly by
//! [`the_split_output_is_identical_at_two_slot_counts`] (195 and 200 slots, byte for byte).
//!
//! The prefill ring has its own tests: `a_chunk_over_the_prefill_ring_matches_the_all_resident_block`
//! (49 rows over a funded ring, RING_BOUND against the all-resident block and fp64) and
//! [`each_projection_through_the_jobs_family_and_the_indexed_family_on_one_row`] (the launch-family
//! A/B behind that bound). `FIFI_FLASH_NEXT_RING_HELD` (test-only, read by the ring test) picks how
//! many of ROW 0's picks the bank holds — 0 the all-ring arm, 10 the all-bank arm, 6 the default
//! mix — rows 1.. keep the default split in every arm.
//!
//! The wide oracle runs read three test-only variables: `FIFI_RING_ORACLE_ROWS` sets the
//! row count (required by the wide oracle; the measured legs are 49, 1,024 and
//! 2,048 rows — the cap lifted below admits 2,048),
//! `FIFI_RING_REF_CACHE` names a
//! directory where the fp64 references are cached by input hash — computed host-side before any
//! device work, so a warm oracle leg holds the card for its launches only — and
//! `FIFI_RING_CONTROL_STRIDE` runs the same-family control on every Nth row (default 1: every
//! row; a stride also keeps the last row and the two rows either side of the midpoint).
//! The oracle covers 1,024 rows under the shared 16,384-row cap (swiglu_rows
//! MAX_ROWS); this commit lifts that cap to 32,768, so 2,048 prompt rows (up to
//! 18,779 expert rows per group) are admitted. At
//! 1,024 rows the all-resident reference is launched row by row, which costs about 19 minutes
//! of card for the comparisons alone — measured 2026-11-18, every-row control would add ~62
//! min per layer at ~1.8 s a row, hence the stride (131 rows per layer at stride 8: 231 s and
//! 228 s, max 0.000e0 on both layers). Batching that reference into grouped launches is a
//! follow-up, not done here. `prime_ring_reference_cache` fills or verifies that cache
//! without a GPU.

use crate::family::exl3::launch::{Launch, MoeDims, RouteBufs};
use std::path::PathBuf;

use super::{
    bank::{HostMemory, HostPinPlan, HostTerms, RecordSource, device::DeviceBank},
    bank_source::PackRecordSource,
    moe_block::{SplitMoe, SplitMoeTrunk},
};
use crate::family::expert_bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru};
use crate::{
    EngError,
    family::{
        exl3::moe::{Exl3MoeBufs, Exl3MoeWeights, Scratch, run_exl3_moe},
        flash_next::{
            load::{exl3_pack::inspect_flash_pack, model_shape, pack_files::FlashPackFiles},
            moe::exl3_cuda_tests::{
                OUTPUT_BOUND, Reference, Router, floats, input, keys, reference, relative_l2,
                relative_l2_f64, upload,
            },
        },
        gguf_model::Role,
        project::cuda::{CudaProjector, CudaWeight, CudaWeights},
    },
    kernels::cuda::{
        DeviceBuffer, device_present,
        resident::router::{RouterShape, router_logits_bytes},
    },
    weights::exl3::Checkpoint,
};

const F32: usize = 4;
const SLOTS: usize = 16;
/// The union-admit chunk path's flat launch groups the rows' pairs differently than the
/// per-row launches; the difference is a handful of float roundings.
const SPLIT_BOUND: f64 = 1e-6;
/// The ring oracle's tolerance is a chosen ceiling, not a derived invariant: 2^-11
/// sits about 1.2x above the largest deviation measured on this pack — 3.846e-4
/// relative L2 at 49 rows (max over both layers) and 4.038e-4 at 1,024 rows; the
/// maximum grows with the row count, as a max over draws does. The deviation is identical with the fused pair launch on and off
/// and with all, none or some of row 0's picks held in the bank
/// (FIFI_FLASH_NEXT_RING_HELD arms those variants; the knob moves row 0 only), so sum
/// order and that source split are excluded. The cause is the launch family, not the
/// ring: the job-table tile kernels stage activations as F16 operands (`exl3.cu`'s
/// jobs_input_rotate rounds on write, JobsTile multiplies the halves) while the fp64
/// and all-resident references carry no such rounding — the per-projection split is
/// measured by [`each_projection_through_the_jobs_family_and_the_indexed_family_on_one_row`]
/// at 1.9-3.9e-4 (bounded at GATE_UP_BOUND/DOWN_BOUND). 2^-11 = 4.88e-4, the F16 unit
/// roundoff, is adopted as the ceiling: about 1.2 times the 1,024-row maximum, so a
/// width change needs no re-measurement, while a real ring defect (a wrong expert
/// record, an off-by-one slot: 1e-1 to 1e-3 scale) still fails by orders of magnitude.
/// The same-family control inside the oracle (the ring against an all-resident launch
/// of the SAME jobs family, bit-identical) is what proves the ring adds nothing;
/// this bound only fixes the launch-family difference. The per-layer mean is guarded
/// separately at MEAN_BOUND.
/// The ceiling catches any real defect class the measurement campaign has shown
/// (a wrong slot row, a torn scale, an off-by-one slot record: 1e-1 to 1e-3 scale,
/// orders of magnitude above it) while absorbing the rounding-scale floor the
/// measurements sit on; it would NOT catch a defect that lands inside ~20% of the
/// measured maximum — that is what the bit-identity control and the mean guard
/// are for.
const RING_BOUND: f64 = 1.0 / 2048.0; // 2^-11, a chosen ceiling (see above)

/// The per-layer mean guard: measured means are 3.219e-4 and 3.290e-4 at 49 rows
/// and 3.212e-4 and 3.321e-4 at 1,024 rows (both layers, both widths); 3.6e-4
/// catches a uniform uplift (one more rounding stage) while sitting above those.
const MEAN_BOUND: f64 = 3.6e-4;

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
    let policy = ReplaceOnMissLru::from_fill(geometry, &row_refs, &counts, None).expect("the fill");
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
        None,
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
        run_exl3_moe(
            launch,
            &block,
            &rig.bufs().expect("bufs"),
            moe,
            true,
            None,
            false,
        )
        .expect("block");
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

    for case in ["mixed", "one miss", "forced miss"] {
        // Per bank layer: the experts its slots hold and the misses its row will cause.
        let held: Vec<Vec<u16>> = prepared
            .iter()
            .map(|p| match case {
                // Six of the ten picks held, ten other experts: four misses.
                "mixed" => {
                    let mut held: Vec<u16> = p.picks().iter().take(6).map(|&e| e as u16).collect();
                    held.extend(others(p.picks(), SLOTS - 6));
                    held
                }
                // Nine of the ten picks held: exactly one miss, the narrowest compact
                // miss pass (width 1 where the filler launch always carried 10).
                "one miss" => {
                    let mut held: Vec<u16> = p.picks().iter().take(9).map(|&e| e as u16).collect();
                    held.extend(others(p.picks(), SLOTS - 9));
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
                    "{case}, run {run}, bank layer {bank_layer} (pack layer {})",
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
                // One rank-order combine: the split writes the all-resident bytes at any
                // miss split, so the output does not depend on the slot count.
                assert_eq!(
                    got, p.resident,
                    "{tag}: the split writes the resident bytes"
                );
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

/// The split output must not depend on the bank geometry OR the miss split: the same rows over
/// a 195-slot bank holding SIX of the ten picks (four misses) and a 200-slot bank holding NINE
/// (one miss) write identical bytes — different miss sets, not just different eviction — and
/// those bytes are the all-resident block's, the association prefill sums in.
#[test]
#[ignore = "needs a GPU (about 4 GiB), about 2 GiB of pinned host memory per bank and FIFI_FLASH_NEXT_EXL3_PACK"]
fn the_split_output_is_identical_at_two_slot_counts() {
    assert!(
        device_present("Flash-Next split MoE block"),
        "a GPU is required"
    );
    let first = moe_layer();
    let layers = vec![first, first + 1];
    let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
    let checkpoint = &files.checkpoint;
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

    // Four outputs per slot count: run 0 (mixed misses) and run 1 (all picks now resident),
    // bank layer 1 then 0, in the same order at both counts. The two counts hold DIFFERENT
    // miss sets: six of ten picks at 195 slots (four misses), nine of ten at 200 (one miss) —
    // the old two-group sum gave those different associations, so this test fails on it.
    let mut by_slots: Vec<Vec<Vec<u8>>> = Vec::new();
    for (slots, held_picks) in [(195usize, 6usize), (200, 9)] {
        let held: Vec<Vec<u16>> = prepared
            .iter()
            .map(|p| {
                let mut held: Vec<u16> = p
                    .picks()
                    .iter()
                    .take(held_picks)
                    .map(|&e| e as u16)
                    .collect();
                held.extend(others(p.picks(), slots - held_picks));
                held
            })
            .collect();
        let bank = build_bank(&projector, &source, &layers, &held);
        let trunks = prepared.iter().map(Prepared::trunk).collect();
        let mut split = SplitMoe::new(ctx, trunks, bank, &layout, &moe).expect("split");
        let mut outs = Vec::new();
        for run in 0..2 {
            for bank_layer in [1usize, 0] {
                let p = &prepared[bank_layer];
                let rig = Rig::new(&projector, &moe, &p.row).expect("rig");
                let before = split.bank().state().stats().misses;
                split
                    .run(&launch, bank_layer, &rig.bufs().expect("bufs"), &moe)
                    .expect("the split block");
                stream.synchronize().expect("sync");
                let misses = (split.bank().state().stats().misses - before) as usize;
                assert_eq!(
                    misses,
                    if run == 0 { 10 - held_picks } else { 0 },
                    "slots {slots}, run {run}, bank layer {bank_layer}"
                );
                outs.push(rig.out_bytes(&projector));
            }
        }
        by_slots.push(outs);
    }
    for (i, (a, b)) in by_slots[0].iter().zip(&by_slots[1]).enumerate() {
        assert_eq!(a, b, "output {i} differs between 195 and 200 slots");
    }
    for (i, out) in by_slots[0].iter().enumerate() {
        let bank_layer = if i % 2 == 0 { 1 } else { 0 };
        assert_eq!(
            out, &prepared[bank_layer].resident,
            "output {i} is the all-resident block's bytes"
        );
    }
}

fn floats_f64(bytes: &[u8]) -> Vec<f64> {
    floats(bytes).into_iter().map(f64::from).collect()
}

/// The buffers of one multi-row block call, zeroed, with `rows` as the normalized inputs. A
/// source's gate/up launch stages one `xh` row per pair and the whole chunk can be one
/// source's, so the scratch staging is sized `m·k` rows (the F32 byte sizing also covers
/// the F16 staging the job-table family writes into the same ranges).
struct RigRows {
    owned: Vec<DeviceBuffer>,
    hidden: usize,
}

impl RigRows {
    fn new(projector: &CudaProjector, moe: &MoeDims, rows: &[Vec<f32>]) -> Result<Self, EngError> {
        let m = rows.len();
        let ctx = projector.context();
        let zeros = |bytes: usize| ctx.upload(&vec![0u8; bytes]);
        let (h, k) = (moe.hidden * F32, moe.top_k);
        let one = router_logits_bytes(RouterShape {
            m: 1,
            hidden: moe.hidden,
            n_expert: moe.n_expert,
            top_k: k,
        })?;
        let mut norm = Vec::with_capacity(m * h);
        for row in rows {
            norm.extend(row.iter().flat_map(|v| v.to_le_bytes()));
        }
        let owned = vec![
            zeros(m * h)?,
            zeros(m * h)?,
            ctx.upload(&norm)?,
            zeros(m * one)?,
            zeros(m * k * 4)?,
            zeros(m * k * F32)?,
            zeros(m * F32)?,
            zeros(4)?,
            zeros(m * k * 2 * moe.inter * F32)?,
            zeros(m * k * moe.inter * F32)?,
            zeros(m * k * h)?,
            zeros(m * 2 * moe.shared_inter * F32)?,
            zeros(m * moe.shared_inter * F32)?,
            zeros(m * h)?,
            // A source's gate/up launch stages one xh row a pair, and the whole chunk can be
            // one source's: m·k rows, like the down windows through z.
            zeros(m * k * moe.hidden * F32)?,
            zeros(m * k * moe.inter.max(moe.hidden) * F32)?,
            zeros(4)?,
        ];
        Ok(Self {
            owned,
            hidden: moe.hidden * F32,
        })
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

    fn out_row(&self, projector: &CudaProjector, r: usize) -> Vec<u8> {
        let stream = projector.context().stream().expect("stream");
        let mut bytes = vec![0u8; self.owned[1].bytes()];
        self.owned[1]
            .read_at(&stream, 0, &mut bytes)
            .expect("read the outputs");
        bytes[r * self.hidden..][..self.hidden].to_vec()
    }

    /// `expert_gu`, `expert_down` (whole chunk) and the routed ids, read back.
    fn expert_bytes(&self, projector: &CudaProjector) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let stream = projector.context().stream().expect("stream");
        let read = |i: usize| {
            let mut bytes = vec![0u8; self.owned[i].bytes()];
            self.owned[i]
                .read_at(&stream, 0, &mut bytes)
                .expect("read the expert buffers");
            bytes
        };
        (read(8), read(10), read(4))
    }
}

#[test]
#[ignore = "needs a GPU (about 3 GiB), pinned host memory, FIFI_FLASH_NEXT_EXL3_PACK and FIFI_RING_ORACLE_ROWS for the row count"]
fn the_shared_prefill_ring_matches_the_oracle_for_full_tiles_tails_and_one_row() {
    // One invocation per row set: FIFI_RING_ORACLE_ROWS selects the full-tile, tail,
    // one-row or widened-bound window this run exercises.
    let rows: usize = std::env::var("FIFI_RING_ORACLE_ROWS")
        .expect("FIFI_RING_ORACLE_ROWS selects this oracle's row count")
        .parse()
        .expect("a row count");
    ring_chunk_pass(rows, Some(6));
}

#[test]
#[ignore = "needs a GPU (about 3 GiB), pinned host memory and FIFI_FLASH_NEXT_EXL3_PACK"]
fn the_shared_prefill_ring_handles_an_empty_bank_or_ring_pair_subset() {
    for held in [0, 10] {
        ring_chunk_pass(1, Some(held));
    }
}

fn threaded_references(
    checkpoint: &Checkpoint,
    routed: (usize, &Router),
    one_row: &MoeDims,
    row_inputs: &[Vec<f32>],
) -> Vec<Reference> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let size = row_inputs.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = row_inputs
            .chunks(size)
            .map(|rows| {
                scope.spawn(move || {
                    rows.iter()
                        .map(|row| {
                            reference(checkpoint, routed, one_row, row).expect("fp64 reference")
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("the reference thread"))
            .collect()
    })
}

/// The pack's identity for the cache key: every file under the pack directory with its
/// size and mtime, sorted by path. A rebuilt or edited pack changes the key even when
/// the directory path does not.
fn pack_identity(hasher: &mut std::collections::hash_map::DefaultHasher) {
    fn walk(dir: &std::path::Path, depth: usize, hasher: &mut impl std::hash::Hasher) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut paths: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            let Ok(meta) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            hasher.write(path.file_name().unwrap_or_default().as_encoded_bytes());
            hasher.write(&meta.len().to_le_bytes());
            if let Some(secs) = meta
                .modified()
                .ok()
                .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
            {
                hasher.write(&secs.as_secs().to_le_bytes());
            }
            if path.is_dir() && depth < 3 {
                walk(&path, depth + 1, hasher);
            }
        }
    }
    walk(&pack_dir(), 0, hasher);
}

/// One layer's cached-reference file, keyed by everything the fixture feeds the fp64
/// computation: the tag (bump it when the seeding formula changes), the pack's path AND
/// per-file identity, the MoE shapes, the layer's POSITION in the bank pair (rows are
/// seeded `seed + 7 * (index + 1)`, so the layer number alone is not the input), the
/// layer number and the row count. `DefaultHasher` is stable within one toolchain; a
/// toolchain change only misses the cache (a safe recompute).
fn ring_ref_cache_path(
    dir: &std::path::Path,
    dims: &MoeDims,
    index: usize,
    layer: usize,
    m: usize,
) -> PathBuf {
    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hasher.write(b"ring-refs-v2");
    hasher.write(pack_dir().to_string_lossy().as_bytes());
    pack_identity(&mut hasher);
    for field in [
        dims.hidden,
        dims.n_expert,
        dims.top_k,
        dims.inter,
        dims.shared_inter,
    ] {
        hasher.write(&(field as u64).to_le_bytes());
    }
    hasher.write(&(index as u64).to_le_bytes());
    hasher.write(&(layer as u64).to_le_bytes());
    hasher.write(&(m as u64).to_le_bytes());
    hasher.write(&((7 * (index + 1)) as u64).to_le_bytes());
    dir.join(format!("ring-refs-{:016x}.json", hasher.finish()))
}

/// The chunk fixture's fp64 references, computed on the host before any device work
/// (~0.23 s per row per layer on this pack) and cached under `FIFI_RING_REF_CACHE` when
/// set, so a warm oracle leg holds the card for its launches only. A cache hit is
/// length-checked against the fixture (`m` rows, `top_k` picks, `hidden` outputs); a
/// corrupt, truncated or stale file fails the check and recomputes. Writes are atomic
/// (temp file + rename), so a concurrent leg never reads a half-written file.
fn ring_host_references(
    files: &FlashPackFiles,
    dims: &MoeDims,
    layers: &[usize],
    m: usize,
) -> Vec<Vec<Reference>> {
    let cache_dir = std::env::var_os("FIFI_RING_REF_CACHE").map(PathBuf::from);
    let checkpoint = &files.checkpoint;
    let mut all = Vec::with_capacity(layers.len());
    for (index, &layer) in layers.iter().enumerate() {
        let router = Router::read(checkpoint, layer, dims).expect("router");
        let row_inputs: Vec<Vec<f32>> = (0..m)
            .map(|seed| input(dims.hidden, seed + 7 * (index + 1)))
            .collect();
        let path = cache_dir
            .as_ref()
            .map(|dir| ring_ref_cache_path(dir, dims, index, layer, m));
        let cached = path.as_ref().and_then(|path| {
            let wants: Vec<Reference> =
                serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
            (wants.len() == m
                && wants
                    .iter()
                    .all(|w| w.ids.len() == dims.top_k && w.out.len() == dims.hidden))
            .then_some(wants)
        });
        match cached {
            Some(wants) => {
                println!("layer {layer}: fp64 references from cache ({m} rows)");
                all.push(wants);
            }
            None => {
                let started = std::time::Instant::now();
                let computed = threaded_references(checkpoint, (layer, &router), dims, &row_inputs);
                println!(
                    "layer {layer}: fp64 references computed in {:.1?} ({m} rows)",
                    started.elapsed()
                );
                if let Some(path) = path.as_ref() {
                    let dir = path.parent().expect("the cache file's directory");
                    std::fs::create_dir_all(dir).expect("create the reference cache dir");
                    let tmp = dir.join(format!(
                        ".{}.tmp{}",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        std::process::id()
                    ));
                    std::fs::write(
                        &tmp,
                        serde_json::to_string(&computed).expect("serialize the references"),
                    )
                    .expect("write the reference cache's temp file");
                    std::fs::rename(&tmp, path).expect("publish the reference cache atomically");
                }
                all.push(computed);
            }
        }
    }
    all
}

/// One bank layer's fixture inside [`ring_chunk_pass`]: the uploads, the chunk's rows,
/// and the per-row references (fp64 from the host phase, all-resident from the card).
struct Arm {
    row_inputs: Vec<Vec<f32>>,
    weights: CudaWeights,
    keys: Vec<(Role, String)>,
    wants: Vec<Reference>,
    residents: Vec<Vec<u8>>,
}

/// The bank layers' trunks over the arms' uploads (raw router weights, exl3 shared
/// experts). Used by the main split and, per row, by the same-family control's split.
fn arm_trunks(arms: &[Arm]) -> Vec<SplitMoeTrunk<'_>> {
    arms.iter()
        .map(|arm| {
            let key = |role: Role| {
                arm.keys
                    .iter()
                    .find(|(candidate, _)| *candidate == role)
                    .map(|(_, key)| key.as_str())
                    .unwrap_or_else(|| panic!("no key for {role:?}"))
            };
            SplitMoeTrunk {
                router: raw(&arm.weights, key(Role::FfnGateInp)),
                router_shared: raw(&arm.weights, key(Role::FfnGateInpShexp)),
                shared_gate: exl3(&arm.weights, key(Role::FfnGateShexp)),
                shared_up: exl3(&arm.weights, key(Role::FfnUpShexp)),
                shared_down: exl3(&arm.weights, key(Role::FfnDownShexp)),
            }
        })
        .collect()
}

fn ring_chunk_pass(m: usize, held: Option<usize>) {
    let first = moe_layer();
    let layers = vec![first, first + 1];
    let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
    let checkpoint = &files.checkpoint;
    inspect_flash_pack(&checkpoint.source, &checkpoint.model_config)
        .expect("the pack matches its config");
    let (model, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let shape = model.moe;
    let one_row = MoeDims {
        rows: 1,
        hidden: model.hyper.hidden(),
        n_expert: shape.experts(),
        top_k: shape.top_k(),
        inter: shape.inter(),
        shared_inter: shape.shared_inter(),
    };
    // The legacy oracle uses 49 rows; the promoted-store arms also cover the full tile,
    // a short tail and the active prefill's trailing one-row window.
    let chunk = MoeDims { rows: m, ..one_row };

    // Host phase first, no device taken: see `ring_host_references` — with a primed
    // cache the card phase is uploads and launches only.
    let cached_wants = ring_host_references(&files, &one_row, &layers, m);

    assert!(
        device_present("Flash-Next split MoE block"),
        "a GPU is required"
    );
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let source = PackRecordSource::new(checkpoint).expect("the record source");
    let layout = source.layout().clone();

    // Per bank layer: uploads, the chunk's rows, and per-row fp64 and all-resident references.
    let mut arms: Vec<Arm> = Vec::with_capacity(layers.len());
    for ((index, &layer), wants) in layers.iter().enumerate().zip(cached_wants) {
        let router = Router::read(checkpoint, layer, &one_row).expect("router");
        let weights = upload(&projector, checkpoint, (layer, &router), &one_row).expect("upload");
        let keys = keys(layer);
        let row_inputs: Vec<Vec<f32>> = (0..m)
            .map(|seed| input(one_row.hidden, seed + 7 * (index + 1)))
            .collect();
        let block = Exl3MoeWeights::resolve_by(&weights, layer, &one_row, |role| {
            keys.iter()
                .find(|(candidate, _)| *candidate == role)
                .map(|(_, key)| key.as_str())
                .ok_or_else(|| EngError::BadRequest(format!("no key for {role:?}")))
        })
        .expect("the all-resident block resolves");
        // The all-resident reference is the oracle: one launch per row, exactly as before
        // the widening. Its fp64 sibling moved to the host phase above (cached); only the
        // launches remain here, timed on their own.
        let started = std::time::Instant::now();
        let mut residents = Vec::with_capacity(m);
        for row in &row_inputs {
            let rig = Rig::new(&projector, &one_row, row).expect("rig");
            run_exl3_moe(
                &launch,
                &block,
                &rig.bufs().expect("bufs"),
                &one_row,
                true,
                None,
                false,
            )
            .expect("the all-resident block");
            launch.stream.synchronize().expect("sync");
            residents.push(rig.out_bytes(&projector));
        }
        println!(
            "layer {layer} over {m} rows: all-resident launches {:.1?}",
            started.elapsed()
        );
        arms.push(Arm {
            row_inputs,
            weights,
            keys,
            wants,
            residents,
        });
    }

    // The bank holds the first `held_of` of row 0's picks per layer, padded with other
    // experts: the default 6 leaves row 0 mixed; the ring-tolerance measurements hold every
    // pick of row 0 (FIFI_FLASH_NEXT_RING_HELD=10, the all-bank arm) or none (0, the
    // all-ring arm). Rows 1.. keep the default split in every arm.
    let held_of: usize = held.unwrap_or_else(|| {
        std::env::var("FIFI_FLASH_NEXT_RING_HELD")
            .map_or(6, |v| v.parse().expect("a held-pick count"))
    });
    let held: Vec<Vec<u16>> = arms
        .iter()
        .map(|arm| {
            let mut held: Vec<u16> = arm.wants[0]
                .ids
                .iter()
                .take(held_of)
                .map(|&e| e as u16)
                .collect();
            held.extend(others(&arm.wants[0].ids, SLOTS - held.len()));
            held
        })
        .collect();
    let bank = build_bank(&projector, &source, &layers, &held);
    let trunks = arm_trunks(&arms);
    let mut split = SplitMoe::new(ctx, trunks, bank, &layout, &chunk).expect("split");
    let streamed = shape.experts() - SLOTS;
    let slots = split.plan_ring(ctx, streamed, 1).expect("fund the ring");
    assert!(slots >= streamed, "the ring holds one full streamed layer");
    split.begin_prefill().unwrap();

    // One chunk per run, every layer of it in order — the driver cadence the ring's free
    // cursor assumes. Run 1's row outputs are kept byte for byte: against run 0 (both runs
    // identical) and for the same-family control after the loop.
    let mut first_run: Vec<Vec<Vec<u8>>> = vec![Vec::new(); arms.len()];
    let mut last_run: Vec<Vec<Vec<u8>>> = vec![Vec::new(); arms.len()];
    for run in 0..2 {
        for (bank_layer, arm) in arms.iter().enumerate() {
            if bank_layer == 0 {
                split.begin_chunk();
            }
            let rig = RigRows::new(&projector, &chunk, &arm.row_inputs).expect("rig");
            let before = split.bank().state().stats().clone();
            let frozen = split.bank().state().slot_row(bank_layer).to_vec();
            shared_prefill_pass(
                &mut split,
                &launch,
                bank_layer,
                &rig.bufs().expect("bufs"),
                &chunk,
            );
            stream.synchronize().expect("sync");
            let tag = format!(
                "bank layer {bank_layer} (pack layer {}), run {run}",
                layers[bank_layer]
            );
            // Streaming changes neither the slot mapping nor any decode/admission counter.
            assert_eq!(
                split.bank().state().stats(),
                &before,
                "{tag}: bank counters"
            );
            assert_eq!(
                split.bank().state().slot_row(bank_layer),
                frozen,
                "{tag}: frozen bank"
            );
            let mut fp64_errors = Vec::with_capacity(m);
            let mut resident_errors = Vec::with_capacity(m);
            for r in 0..m {
                let got = rig.out_row(&projector, r);
                if run == 0 {
                    first_run[bank_layer].push(got.clone());
                } else {
                    assert_eq!(
                        &got, &first_run[bank_layer][r],
                        "{tag}, row {r}: run 1 differs from run 0 (determinism)"
                    );
                    last_run[bank_layer].push(got.clone());
                }
                let resident_error = relative_l2(&floats(&got), &floats_f64(&arm.residents[r]));
                let fp64_error = relative_l2(&floats(&got), &arm.wants[r].out);
                println!(
                    "{tag}, row {r}: vs all-resident {resident_error:.3e}, vs fp64 {fp64_error:.3e}"
                );
                assert!(fp64_error < RING_BOUND, "{tag}, row {r}: fp64 {fp64_error}");
                assert!(
                    resident_error < RING_BOUND,
                    "{tag}, row {r}: all-resident {resident_error}"
                );
                // The all-resident reference is itself within ~3e-7 of fp64 (the indexed
                // family's class), so the two error series must agree; a disagreement is
                // a broken reference, not tolerance.
                assert!(
                    (fp64_error - resident_error).abs() < 1e-5,
                    "{tag}, row {r}: the references disagree \
                     (fp64 {fp64_error} vs all-resident {resident_error})"
                );
                fp64_errors.push(fp64_error);
                resident_errors.push(resident_error);
            }
            // The run criteria: per-row max under RING_BOUND (asserted above) plus a
            // per-layer mean guard — a uniform uplift (one more rounding stage) moves the
            // mean while leaving most rows under the max bound.
            let stats = |errors: &[f64]| {
                let max = errors.iter().copied().fold(f64::MIN, f64::max);
                let mean = errors.iter().sum::<f64>() / errors.len() as f64;
                let var = errors.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>()
                    / errors.len() as f64;
                (
                    max,
                    mean,
                    var.sqrt(),
                    errors.iter().filter(|&&e| e > 3.5e-4).count(),
                    errors.iter().filter(|&&e| e > 4.0e-4).count(),
                )
            };
            let (fmax, fmean, fsigma, f35, f40) = stats(&fp64_errors);
            let (rmax, rmean, _, _, _) = stats(&resident_errors);
            println!(
                "{tag}: fp64 max {fmax:.3e} mean {fmean:.3e} sigma {fsigma:.3e} \
                 rows>3.5e-4 {f35} rows>4.0e-4 {f40}; all-resident max {rmax:.3e} mean {rmean:.3e}"
            );
            assert!(
                fmean < MEAN_BOUND,
                "{tag}: fp64 mean {fmean} exceeds the mean guard {MEAN_BOUND}"
            );
            assert!(
                rmean < MEAN_BOUND,
                "{tag}: all-resident mean {rmean} exceeds the mean guard {MEAN_BOUND}"
            );
        }
    }
    split.end_prefill().expect("hand the ring's stores back");

    // Same-family control: every Nth row (FIFI_RING_CONTROL_STRIDE, default 1: every row)
    // again through the SAME jobs family, but with a bank holding that row's own ten
    // picks — all-resident, no ring traffic for the row — while the main arm streamed
    // most of its experts through the ring. The family's F16 arithmetic is identical in
    // both arms, so anything above the indexed family's 1e-6 scale is the ring's, not
    // the launch family's. This is the test that proves the ring adds nothing; the
    // RING_BOUND comparisons above only fix the family difference.
    let stride: usize = std::env::var("FIFI_RING_CONTROL_STRIDE")
        .ok()
        .map(|v| v.parse().expect("a stride"))
        .unwrap_or(1)
        .max(1);
    assert!(
        stride <= m,
        "FIFI_RING_CONTROL_STRIDE {stride} exceeds the row count {m}: \
         a control that skips every row is not a control"
    );
    let one_row_chunk = MoeDims { rows: 1, ..chunk };
    let mut control_max = 0.0f64;
    // Stride rows plus the boundaries the stride can miss: the last row and the two
    // rows either side of the midpoint (at 1,024 rows: 0, 511, 512, 513 and 1,023).
    let control_rows: Vec<usize> = (0..m)
        .filter(|&r| r % stride == 0 || r + 1 == m || r.abs_diff(m / 2) <= 1)
        .collect();
    for (bank_layer, arm) in arms.iter().enumerate() {
        let started = std::time::Instant::now();
        let mut layer_max = 0.0f64;
        for r in control_rows.iter().copied() {
            // The control bank holds row r's ten picks in BOTH bank layers.
            let held: Vec<Vec<u16>> = arms
                .iter()
                .map(|a| {
                    let mut picks: Vec<u16> = a.wants[r].ids.iter().map(|&e| e as u16).collect();
                    picks.extend(others(&a.wants[r].ids, SLOTS - picks.len()));
                    picks
                })
                .collect();
            let bank = build_bank(&projector, &source, &layers, &held);
            let mut control = SplitMoe::new(ctx, arm_trunks(&arms), bank, &layout, &one_row_chunk)
                .expect("the control split");
            control
                .plan_ring(ctx, shape.experts() - SLOTS, 1)
                .expect("fund the control's ring");
            control.begin_prefill().unwrap();
            control.begin_chunk();
            let rig = RigRows::new(&projector, &one_row_chunk, &[arm.row_inputs[r].clone()])
                .expect("rig");
            // The ring frees layers strictly in order (0, 1, ...), so a control that
            // runs only bank layer b must first walk the lower layers' one-row passes
            // to free them; their outputs are discarded, only b's row is compared.
            for l in 0..=bank_layer {
                shared_prefill_pass(
                    &mut control,
                    &launch,
                    l,
                    &rig.bufs().expect("bufs"),
                    &one_row_chunk,
                );
            }
            stream.synchronize().expect("sync");
            let got = rig.out_row(&projector, 0);
            let error = relative_l2(&floats(&got), &floats_f64(&last_run[bank_layer][r]));
            assert!(
                error == 0.0,
                "bank layer {bank_layer} (pack layer {}), row {r}: \
                 same-family control error {error} — the ring must be bit-identical \
                 to the resident pass, not merely small",
                layers[bank_layer]
            );
            layer_max = layer_max.max(error);
            control
                .end_prefill()
                .expect("hand the control's stores back");
        }
        control_max = control_max.max(layer_max);
        println!(
            "bank layer {bank_layer} control over {} rows: max {layer_max:.3e} in {:.1?}",
            control_rows.len(),
            started.elapsed()
        );
    }
    println!(
        "same-family control: max over both layers {control_max:.3e} (bit-identity, 0.0 required)"
    );
}

/// Fills or verifies the fp64 reference cache without a GPU: `FIFI_RING_REF_CACHE`
/// names the directory, `FIFI_RING_ORACLE_ROWS` the row count (1,024 by default, the
/// oracle's measurement width). Cold, it computes and writes the cache; then, in both
/// cases, it reads the files back and asserts they round-trip to exactly what the host
/// phase produced — so a prime run is a checked statement, never a silent pass.
#[test]
#[ignore = "set FIFI_RING_REF_CACHE (and optionally FIFI_RING_ORACLE_ROWS) to prime the fp64 reference cache"]
fn prime_ring_reference_cache() {
    let cache_dir = PathBuf::from(
        std::env::var_os("FIFI_RING_REF_CACHE").expect("FIFI_RING_REF_CACHE names the cache dir"),
    );
    let rows: usize = std::env::var("FIFI_RING_ORACLE_ROWS")
        .ok()
        .map(|v| v.parse().expect("a row count"))
        .unwrap_or(1024);
    let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
    let first = moe_layer();
    let layers = vec![first, first + 1];
    let (model, _) = model_shape(&files.checkpoint.model_config).expect("the model shape");
    let shape = model.moe;
    let dims = MoeDims {
        rows: 1,
        hidden: model.hyper.hidden(),
        n_expert: shape.experts(),
        top_k: shape.top_k(),
        inter: shape.inter(),
        shared_inter: shape.shared_inter(),
    };
    let wants = ring_host_references(&files, &dims, &layers, rows);
    for (index, &layer) in layers.iter().enumerate() {
        let path = ring_ref_cache_path(&cache_dir, &dims, index, layer, rows);
        let loaded: Vec<Reference> = serde_json::from_str(
            &std::fs::read_to_string(&path).expect("the cache file after priming"),
        )
        .expect("deserialize the cache file");
        assert_eq!(
            loaded.len(),
            rows,
            "layer {layer}: the cache holds {rows} rows"
        );
        assert_eq!(
            loaded, wants[index],
            "layer {layer}: the cache file round-trips to what the host phase produced"
        );
    }
    println!(
        "prime: {} layers x {rows} rows verified under {}",
        layers.len(),
        cache_dir.display()
    );
}

/// Prints the CPU cost of the fp64 references for a row count (no GPU).
#[test]
#[ignore = "set FIFI_RING_REF_TIMING to print the fp64 references' host cost"]
fn ring_reference_cpu_cost() {
    let Ok(rows) = std::env::var("FIFI_RING_REF_TIMING") else {
        return;
    };
    let rows: usize = rows.parse().expect("a row count");
    let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
    let checkpoint = &files.checkpoint;
    let (model, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let shape = model.moe;
    let one_row = MoeDims {
        rows: 1,
        hidden: model.hyper.hidden(),
        n_expert: shape.experts(),
        top_k: shape.top_k(),
        inter: shape.inter(),
        shared_inter: shape.shared_inter(),
    };
    let layer = moe_layer();
    let router = Router::read(checkpoint, layer, &one_row).expect("router");
    let row_inputs: Vec<Vec<f32>> = (0..rows)
        .map(|seed| input(one_row.hidden, seed + 7))
        .collect();
    let start = std::time::Instant::now();
    let wants = threaded_references(checkpoint, (layer, &router), &one_row, &row_inputs);
    let elapsed = start.elapsed();
    assert_eq!(wants.len(), rows);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!(
        "FIFI_RING_REF_TIMING rows={rows} elapsed={elapsed:.1?} per_row={:.3}s threads={threads}",
        elapsed.as_secs_f64() / rows as f64
    );
}

/// The job-table family's per-projection bounds against fp64, split by projection and
/// measured on this pack (Qwen3.8-Flash-Next exl3 3.05bpw h5 ng5, pack layer 3 — the
/// `FIFI_FLASH_NEXT_EXL3_MOE_LAYER` default — its ten picks of one deterministic row, seed 9;
/// 3090): gate/up 1.9-2.2e-4 and down 2.9-3.9e-4. Both bounds are empirical ceilings
/// (the down bound keeps 1.3x headroom over its 3.9e-4 maximum), not figures derived
/// from the F16 unit roundoff: a projection of an F16-rounded operand has no
/// `at most u` theorem, and the chained roundings of the block can and do land above u.
const GATE_UP_BOUND: f64 = 3e-4;
const DOWN_BOUND: f64 = 5e-4;
/// The flat indexed family reads F32 operands: 1.5-3.0e-7 against fp64 on the same row
/// (same provenance), three orders under the jobs family, so the two cannot be confused.
const INDEXED_BOUND: f64 = 1e-6;

/// The launch families the ring tolerance splits between, on one row and one expert each:
/// the flat all-resident block runs the indexed launches (F32 operands), the funded ring runs
/// the job-table tile kernels (F16-staged activations, `exl3.cu`'s jobs_input_rotate and
/// JobsTile), and every projection of the same routed expert is asserted across the two
/// families and against fp64, both twin rows each. Measured on the provenance above:
/// indexed 1.5-3.0e-7 vs fp64, jobs 1.9-2.2e-4 (gate/up) and 2.9-3.9e-4 (down), and
/// jobs-vs-indexed equals jobs-vs-fp64 because the indexed family sits at 1e-7 — the ring
/// branch's whole-block deviation is the job-table family's F16 operand arithmetic, not a
/// defect of the ring branch. The dtype statement rests on the kernel source (the staging
/// write at `exl3.cu`'s jobs_input_rotate rounds to F16; the indexed launches read F32) and
/// on those asserted error classes; no byte-level staging comparison is made. The Qwen3.6
/// chunk prefill test runs these same job-table kernels; its measured print (layer 3, 1024
/// rows, the Qwen3.6 4.00 bpw pack, the host's RTX 3090) is `whole row 8.183e-6; moe term
/// vs residual norm 2.450e-2; delta only 3.341e-4` — the MoE term is 2.45% of the residual
/// stream's norm, and the delta-only figure sits in the 1.9-3.9e-4 job-table class above:
/// the whole-row figure is consistent with that F16-operand error diluted by the residual
/// stream. Margins: down 1.28x over 3.9e-4, gate/up 1.36x over 2.2e-4, indexed 3.4x over
/// 2.9e-7.
#[test]
#[ignore = "needs a GPU (about 3 GiB of ring and bank, about 2 GiB of pinned host memory per bank and FIFI_FLASH_NEXT_EXL3_PACK"]
fn each_projection_through_the_jobs_family_and_the_indexed_family_on_one_row() {
    assert!(
        device_present("Flash-Next split MoE block"),
        "a GPU is required"
    );
    let first = moe_layer();
    let layers = vec![first];
    let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
    let checkpoint = &files.checkpoint;
    inspect_flash_pack(&checkpoint.source, &checkpoint.model_config)
        .expect("the pack matches its config");
    let (model, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let shape = model.moe;
    let one_row = MoeDims {
        rows: 1,
        hidden: model.hyper.hidden(),
        n_expert: shape.experts(),
        top_k: shape.top_k(),
        inter: shape.inter(),
        shared_inter: shape.shared_inter(),
    };
    // The split block takes 2 to 64 rows: the row is run twice, identically, so every
    // pick's projection appears once per twin row in the jobs segment; the scan below
    // asserts both twins, so a defect in one twin alone fails.
    let two_rows = MoeDims { rows: 2, ..one_row };
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let source = PackRecordSource::new(checkpoint).expect("the record source");
    let layout = source.layout().clone();

    // The flat arm: the all-resident block, one row, indexed launches for every pick.
    let router = Router::read(checkpoint, first, &one_row).expect("router");
    let weights = upload(&projector, checkpoint, (first, &router), &one_row).expect("upload");
    let keys = keys(first);
    let row = input(one_row.hidden, 9);
    let want = reference(checkpoint, (first, &router), &one_row, &row).expect("fp64 reference");
    let block = Exl3MoeWeights::resolve_by(&weights, first, &one_row, |role| {
        keys.iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, key)| key.as_str())
            .ok_or_else(|| EngError::BadRequest(format!("no key for {role:?}")))
    })
    .expect("the all-resident block resolves");
    let flat_rig = RigRows::new(&projector, &two_rows, &[row.clone(), row.clone()]).expect("rig");
    // The flat qwen36 block is per-row: one call computes row 0's picks and fills the gu/down
    // slots at offset 0 (`norm_row` is that row's h bytes).
    run_exl3_moe(
        &launch,
        &block,
        &flat_rig.bufs().expect("bufs"),
        &one_row,
        true,
        None,
        false,
    )
    .expect("the all-resident block");
    launch.stream.synchronize().expect("sync");
    let (flat_gu, flat_down, flat_ids) = flat_rig.expert_bytes(&projector);

    // The jobs arm: a bank with none of the row's picks resident, the ring funded, the same
    // row through the job-table tile kernels.
    let held = others(&want.ids, SLOTS);
    let bank = build_bank(&projector, &source, &layers, &[held]);
    let trunk = {
        let key = |role: Role| {
            keys.iter()
                .find(|(candidate, _)| *candidate == role)
                .map(|(_, key)| key.as_str())
                .unwrap_or_else(|| panic!("no key for {role:?}"))
        };
        SplitMoeTrunk {
            router: raw(&weights, key(Role::FfnGateInp)),
            router_shared: raw(&weights, key(Role::FfnGateInpShexp)),
            shared_gate: exl3(&weights, key(Role::FfnGateShexp)),
            shared_up: exl3(&weights, key(Role::FfnUpShexp)),
            shared_down: exl3(&weights, key(Role::FfnDownShexp)),
        }
    };
    let mut split = SplitMoe::new(ctx, vec![trunk], bank, &layout, &two_rows).expect("split");
    let streamed = shape.experts() - SLOTS;
    let slots = split.plan_ring(ctx, streamed, 1).expect("fund the ring");
    assert!(slots >= streamed, "the ring holds one full streamed layer");
    split.begin_prefill().unwrap();
    split.begin_chunk();
    let jobs_rig = RigRows::new(&projector, &two_rows, &[row.clone(), row]).expect("rig");
    shared_prefill_pass(
        &mut split,
        &launch,
        0,
        &jobs_rig.bufs().expect("bufs"),
        &two_rows,
    );
    stream.synchronize().expect("sync");
    let (jobs_gu, jobs_down, jobs_ids) = jobs_rig.expert_bytes(&projector);

    // The flat block writes the routed ids of row 0 only; both arms route the same logits
    // through the same route kernel, so row 0's picks must agree.
    let (k, i, h) = (one_row.top_k, one_row.inter, one_row.hidden);
    assert_eq!(
        &flat_ids[..k * 4],
        &jobs_ids[..k * 4],
        "both arms route to the same picks"
    );
    // No byte-level staging comparison between the arms is made; the families' difference
    // is established by the asserted error classes below. The dtype statement rests on the
    // kernel source (jobs_input_rotate rounds its operands to F16 where it stores the
    // rotated rows; the indexed launches
    // read the F32 staging) and on the per-projection error classes asserted below: the
    // indexed family reproduces fp64 at 1e-7 while the jobs family sits at 1e-4.
    let (flat_gu, flat_down) = (floats_f64(&flat_gu), floats_f64(&flat_down));
    let (jobs_gu, jobs_down) = (floats_f64(&jobs_gu), floats_f64(&jobs_down));
    // The jobs segment's pair order follows the ring's slot order, not the routing order, so
    // each fp64 pick is matched by scanning the segment: every segment row within the
    // family bound IS this pick (every other expert's projection is orders off — no
    // ambiguity at these gaps), and the row is run twice, identically, so the pick appears
    // once per twin row: exactly two rows must match, and a defect in one twin alone fails.
    let twins = |rows: &[f64], want: &[f64], bound: f64| -> Vec<f64> {
        rows.chunks(want.len())
            .map(|got| relative_l2_f64(got, want))
            .filter(|error| *error < bound)
            .collect()
    };
    let max_of = |v: &[f64]| v.iter().copied().fold(f64::MIN, f64::max);
    for p in 0..k {
        let fg = &flat_gu[p * 2 * i..][..2 * i];
        let wg = &want.gu[p * 2 * i..][..2 * i];
        let fd = &flat_down[p * h..][..h];
        let wd = &want.down[p * h..][..h];
        // The indexed family against fp64, asserted at its own F32 class.
        let (gu_indexed, down_indexed) = (relative_l2_f64(fg, wg), relative_l2_f64(fd, wd));
        assert!(
            gu_indexed < INDEXED_BOUND,
            "pick {p}: indexed gate/up vs fp64 {gu_indexed:.3e}"
        );
        assert!(
            down_indexed < INDEXED_BOUND,
            "pick {p}: indexed down vs fp64 {down_indexed:.3e}"
        );
        // The jobs family against fp64 and against the indexed row, both twins asserted.
        let gu_twins = twins(&jobs_gu, wg, GATE_UP_BOUND);
        let down_twins = twins(&jobs_down, wd, DOWN_BOUND);
        assert_eq!(
            gu_twins.len(),
            2,
            "pick {p}: both gate/up twin rows match fp64 (errors {:?})",
            gu_twins
        );
        assert_eq!(
            down_twins.len(),
            2,
            "pick {p}: both down twin rows match fp64 (errors {:?})",
            down_twins
        );
        let jg = twins(&jobs_gu, fg, GATE_UP_BOUND);
        let jd = twins(&jobs_down, fd, DOWN_BOUND);
        assert_eq!(
            jg.len(),
            2,
            "pick {p}: both gate/up twin rows match the indexed row (errors {:?})",
            jg
        );
        assert_eq!(
            jd.len(),
            2,
            "pick {p}: both down twin rows match the indexed row (errors {:?})",
            jd
        );
        println!(
            "pick {p} gate/up: indexed vs fp64 {gu_indexed:.3e}; jobs vs fp64 {:.3e} (both twins); jobs vs indexed {:.3e}; down: indexed vs fp64 {down_indexed:.3e}; jobs vs fp64 {:.3e}; jobs vs indexed {:.3e}",
            max_of(&gu_twins),
            max_of(&jg),
            max_of(&down_twins),
            max_of(&jd),
        );
    }
}

// Dead since 5590d49ed (2026-11-10, on main) dropped its attributes in a rename; restored
#[test]
#[ignore = "needs a GPU (about 3 GiB), about 3 GiB of pinned host memory and FIFI_FLASH_NEXT_EXL3_PACK"]
fn the_shared_prefill_pass_resolves_misses_and_a_trailing_one_row_window() {
    // One invocation per row set: FIFI_UNION_ORACLE_ROWS runs exactly that row
    // count (the widened-bound windows each need their own card lease); unset
    // runs the full ladder in one process.
    let ladder: Vec<usize> = std::env::var("FIFI_UNION_ORACLE_ROWS")
        .map(|value| vec![value.parse().expect("a row count")])
        .unwrap_or_else(|_| vec![1, 3, 65, 256]);
    for rows in ladder {
        chunk_union_pass(rows);
    }
}

fn shared_prefill_pass(
    split: &mut SplitMoe<'_>,
    launch: &Launch<'_>,
    layer: usize,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) {
    use crate::family::{
        exl3::prefill::{Exl3Scratch, Projector, write_dense_jobs},
        project::exl3_jobs_plan::exl3_job_bytes,
    };
    let (dense_bytes, perm_bytes) =
        exl3_job_bytes(dims.rows, 256, dims.n_expert, dims.top_k).expect("job sizes");
    let ctx = launch.ctx;
    let y = ctx
        .upload(&vec![
            0u8;
            dims.rows * dims.hidden.max(dims.shared_inter) * F32
        ])
        .expect("dense scratch");
    let jobs = ctx.upload(&vec![0u8; dense_bytes]).expect("dense jobs");
    let guarded = ctx
        .upload(&vec![0xA5; perm_bytes + 16])
        .expect("guarded permutation");
    let scratch = Exl3Scratch {
        xh: bufs.scratch.xh,
        z: bufs.scratch.z,
        y: y.range(0, y.bytes()).expect("y range"),
        flag: bufs.scratch.flag,
        jobs: jobs.range(0, jobs.bytes()).expect("jobs range"),
    };
    write_dense_jobs(launch, &scratch, dims.rows, 256).expect("write dense jobs");
    let projector = Projector {
        launch,
        scratch,
        rows: dims.rows,
        tile_rows: 256,
    };
    let row_before = split.bank().state().slot_row(layer).to_vec();
    let short = guarded.range(8, perm_bytes - 1).expect("short permutation");
    let error = split
        .run_prefill(&projector, layer, bufs, dims, short)
        .expect_err("refuse before routing or admitting");
    assert!(
        matches!(error, crate::EngError::BadRequest(message) if message.contains("permutation range"))
    );
    assert_eq!(split.bank().state().slot_row(layer), row_before);
    // The in-place combine must receive the residual, not bytes left in the output face.
    bufs.route
        .resid_out
        .buffer()
        .write_at(
            launch.stream,
            bufs.route.resid_out.offset(),
            &vec![0xFF; dims.rows * dims.hidden * F32],
        )
        .expect("poison output");
    split
        .run_prefill(
            &projector,
            layer,
            bufs,
            dims,
            guarded.range(8, perm_bytes).expect("permutation range"),
        )
        .expect("the shared banked pass");
    launch.stream.synchronize().expect("drain shared pass");
    let mut bytes = vec![0u8; guarded.bytes()];
    guarded
        .read_at(launch.stream, 0, &mut bytes)
        .expect("read guards");
    assert_eq!(&bytes[..8], &[0xA5; 8]);
    assert_eq!(&bytes[8 + perm_bytes..], &[0xA5; 8]);
}

fn chunk_union_pass(m: usize) {
    assert!(
        device_present("Flash-Next split MoE block"),
        "a GPU is required"
    );
    let layer = moe_layer();
    let layers = vec![layer];
    let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
    let checkpoint = &files.checkpoint;
    inspect_flash_pack(&checkpoint.source, &checkpoint.model_config)
        .expect("the pack matches its config");
    let (model, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let shape = model.moe;
    let one_row = MoeDims {
        rows: 1,
        hidden: model.hyper.hidden(),
        n_expert: shape.experts(),
        top_k: shape.top_k(),
        inter: shape.inter(),
        shared_inter: shape.shared_inter(),
    };
    assert_eq!((one_row.n_expert, one_row.top_k), (512, 10));
    let chunk = MoeDims { rows: m, ..one_row };
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let source = PackRecordSource::new(checkpoint).expect("the record source");
    let layout = source.layout().clone();

    let router = Router::read(checkpoint, layer, &one_row).expect("router");
    let weights = upload(&projector, checkpoint, (layer, &router), &one_row).expect("upload");
    let keys = keys(layer);
    let row_inputs: Vec<Vec<f32>> = (0..m)
        .map(|seed| input(one_row.hidden, seed + 31))
        .collect();
    let block = Exl3MoeWeights::resolve_by(&weights, layer, &one_row, |role| {
        keys.iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, key)| key.as_str())
            .ok_or_else(|| EngError::BadRequest(format!("no key for {role:?}")))
    })
    .expect("the all-resident block resolves");
    // The fp64 references dominate the wall (the oracle prints 587.7s threaded for 256
    // rows against 212.0ms of launches): compute them on every core, exactly as the ring
    // oracle does, or a 256-row window cannot fit a card lease.
    let wants = threaded_references(checkpoint, (layer, &router), &one_row, &row_inputs);
    let mut residents = Vec::with_capacity(m);
    for row in &row_inputs {
        let rig = Rig::new(&projector, &one_row, row).expect("rig");
        run_exl3_moe(
            &launch,
            &block,
            &rig.bufs().expect("bufs"),
            &one_row,
            true,
            None,
            false,
        )
        .expect("the all-resident block");
        launch.stream.synchronize().expect("sync");
        residents.push(rig.out_bytes(&projector));
    }

    // The fill holds all 10 of row 0's picks of 20 slots; the chunk's union (at most 20
    // distinct) always fits, so the union admit covers the chunk, never the per-row tail.
    let mut held: Vec<u16> = wants[0].ids.iter().take(10).map(|&e| e as u16).collect();
    held.extend(others(&wants[0].ids, 10));
    let bank = build_bank(&projector, &source, &layers, &[held]);
    let key = |role: Role| {
        keys.iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, key)| key.as_str())
            .unwrap_or_else(|| panic!("no key for {role:?}"))
    };
    let trunk = SplitMoeTrunk {
        router: raw(&weights, key(Role::FfnGateInp)),
        router_shared: raw(&weights, key(Role::FfnGateInpShexp)),
        shared_gate: exl3(&weights, key(Role::FfnGateShexp)),
        shared_up: exl3(&weights, key(Role::FfnUpShexp)),
        shared_down: exl3(&weights, key(Role::FfnDownShexp)),
    };
    let mut split = SplitMoe::new(ctx, vec![trunk], bank, &layout, &chunk).expect("split");

    for run in 0..2 {
        let rig = RigRows::new(&projector, &chunk, &row_inputs).expect("rig");
        let passes = split.bank().state().stats().passes;
        shared_prefill_pass(&mut split, &launch, 0, &rig.bufs().expect("bufs"), &chunk);
        stream.synchronize().expect("sync");
        let tag = format!("run {run}");
        // The admit is not decode traffic and never moves the pass counter; a per-row `step`
        // would. This is the assertion that the union-admit path ran, not the tail.
        assert_eq!(
            split.bank().state().stats().passes,
            passes,
            "{tag}: no per-row bank steps"
        );
        // The promoted pass uses the FP16 job family.
        let (resident_bound, oracle_bound) = (RING_BOUND, RING_BOUND);
        for r in 0..m {
            let got = rig.out_row(&projector, r);
            assert!(
                relative_l2(&floats(&got), &floats_f64(&residents[r])) < resident_bound,
                "{tag}, row {r}: the all-resident bound"
            );
            assert!(
                relative_l2(&floats(&got), &wants[r].out) < oracle_bound,
                "{tag}, row {r}: the fp64 bound"
            );
        }
        let last_window = (m - 1) / 2 * 2;
        for want in &wants[last_window..] {
            for &pick in &want.ids {
                assert!(
                    split.bank().state().slot_row(0)[pick] >= 0,
                    "{tag}: pick {pick} of the final window is in the bank"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU (about 3 GiB), about 3 GiB of pinned host memory and FIFI_FLASH_NEXT_EXL3_PACK"]
fn the_prefill_ring_and_union_paths_cross_both_ways_without_a_copy_conflict() {
    assert!(
        device_present("Flash-Next split MoE block"),
        "a GPU is required"
    );
    let layer = moe_layer();
    let layers = vec![layer];
    let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
    let checkpoint = &files.checkpoint;
    inspect_flash_pack(&checkpoint.source, &checkpoint.model_config)
        .expect("the pack matches its config");
    let (model, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let shape = model.moe;
    let one_row = MoeDims {
        rows: 1,
        hidden: model.hyper.hidden(),
        n_expert: shape.experts(),
        top_k: shape.top_k(),
        inter: shape.inter(),
        shared_inter: shape.shared_inter(),
    };
    // A whole chunk at or above the threshold streams the ring; a smaller piece takes the
    // bank's union admit. Reopening the window after a ring chunk leaves the ring's
    // stream-ahead copies pending on the copy stream, so the next union piece must drain
    // them before the bank admits on its own stream (ring -> union); the reverse order
    // (union -> ring) is safe because the bank synchronizes its own admit copies first.
    const THRESHOLD: usize = 8;
    const BIG: usize = 16;
    const SMALL: usize = 2;
    let big = MoeDims {
        rows: BIG,
        ..one_row
    };
    let small = MoeDims {
        rows: SMALL,
        ..one_row
    };
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let source = PackRecordSource::new(checkpoint).expect("the record source");
    let layout = source.layout().clone();
    let router = Router::read(checkpoint, layer, &one_row).expect("router");
    let weights = upload(&projector, checkpoint, (layer, &router), &one_row).expect("upload");
    let keys = keys(layer);
    let row_inputs: Vec<Vec<f32>> = (0..BIG)
        .map(|seed| input(one_row.hidden, seed + 31))
        .collect();
    let block = Exl3MoeWeights::resolve_by(&weights, layer, &one_row, |role| {
        keys.iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, key)| key.as_str())
            .ok_or_else(|| EngError::BadRequest(format!("no key for {role:?}")))
    })
    .expect("the all-resident block resolves");
    let wants = threaded_references(checkpoint, (layer, &router), &one_row, &row_inputs);
    let mut residents = Vec::with_capacity(BIG);
    for row in &row_inputs {
        let rig = Rig::new(&projector, &one_row, row).expect("rig");
        run_exl3_moe(
            &launch,
            &block,
            &rig.bufs().expect("bufs"),
            &one_row,
            true,
            None,
            false,
        )
        .expect("the all-resident block");
        launch.stream.synchronize().expect("sync");
        residents.push(rig.out_bytes(&projector));
    }
    let mut held: Vec<u16> = wants[0].ids.iter().take(10).map(|&e| e as u16).collect();
    held.extend(others(&wants[0].ids, 10));
    let bank = build_bank(&projector, &source, &layers, &[held]);
    let key = |role: Role| {
        keys.iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, key)| key.as_str())
            .unwrap_or_else(|| panic!("no key for {role:?}"))
    };
    let trunk = SplitMoeTrunk {
        router: raw(&weights, key(Role::FfnGateInp)),
        router_shared: raw(&weights, key(Role::FfnGateInpShexp)),
        shared_gate: exl3(&weights, key(Role::FfnGateShexp)),
        shared_up: exl3(&weights, key(Role::FfnUpShexp)),
        shared_down: exl3(&weights, key(Role::FfnDownShexp)),
    };
    let mut split = SplitMoe::new(ctx, vec![trunk], bank, &layout, &big).expect("split");
    let streamed = shape.experts() - SLOTS;
    let slots = split
        .plan_ring(ctx, streamed, THRESHOLD)
        .expect("fund the ring");
    assert!(slots >= streamed, "the ring holds one full streamed layer");
    // Each window reopens (begin_prefill) and runs one chunk; no end_prefill between them,
    // so the ring's copies stay pending across the ring -> union crossing.
    let windows = [
        (&big, BIG, "ring window"),
        (&small, SMALL, "union after ring"),
        (&big, BIG, "ring after union"),
    ];
    for (dims, rows, tag) in windows {
        split.begin_prefill().unwrap();
        split.begin_chunk();
        let rig = RigRows::new(&projector, dims, &row_inputs[..rows]).expect("rig");
        let admit_before = split.bank().prefill_report();
        shared_prefill_pass(&mut split, &launch, 0, &rig.bufs().expect("bufs"), dims);
        stream.synchronize().expect("sync");
        if tag == "union after ring" {
            // The union window must really admit (not a zero-op early return): its two rows'
            // union outgrows the 16-slot bank, so the admit copies. That copy on the bank
            // stream is what the reopen drain made safe against the ring's pending copies.
            let admit_after = split.bank().prefill_report();
            assert!(
                admit_after.passes > admit_before.passes,
                "union after ring must run a bank admit"
            );
            assert!(
                admit_after.misses > admit_before.misses,
                "union after ring must copy (exercise the reopen drain)"
            );
        }
        for r in 0..rows {
            let got = rig.out_row(&projector, r);
            let resident_error = relative_l2(&floats(&got), &floats_f64(&residents[r]));
            let fp64_error = relative_l2(&floats(&got), &wants[r].out);
            assert!(
                resident_error < RING_BOUND,
                "{tag}, row {r}: all-resident {resident_error}"
            );
            assert!(fp64_error < RING_BOUND, "{tag}, row {r}: fp64 {fp64_error}");
        }
    }
    split.end_prefill().expect("hand the ring's stores back");
}
