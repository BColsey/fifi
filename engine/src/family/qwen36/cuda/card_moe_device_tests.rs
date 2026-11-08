//! The distributed MoE block: one model-dims MoE layer on `CardStack`, two
//! logical contexts, against the single-device reference over the same Q4_K
//! sources and the same token rows - through the production launchers
//! (`moe::launch_op` behind `run_card_moe_block`, `run_moe_block` for the
//! reference), never a private re-wiring.
//!
//! The fixture is the REAL 35B MoE geometry (hidden 2048, 256 experts,
//! top_k 8, expert inter 512, shared inter 512): one rng stream lays every
//! expert's rows, so each expert carries its own block scales - the router
//! picks 8 of the 256, and every picked expert's activation distribution is
//! its own.
//!
//! Stage evidence (the acceptance table):
//! * router + `RouteBroadcast`: ids, pick weights and the shared gate are
//!   EXACT (bit-equal) against the reference on BOTH cards;
//! * gate/up projections: each card's `expert_gu`/`shared_gu` row slice is
//!   EXACT against the reference - same pack, same weight bytes;
//! * `SwigluExperts`/`SwigluShared` Q8_1 pack at the card's narrowed inter:
//!   the pack's own blocks (codes and scales) are EXACT against the
//!   corresponding block range of the reference pack - quantization groups
//!   preserved, per-card scales correct;
//! * `MoeSum` + down/combine: `expert_down` per pick slot, `shared_out` and
//!   the combined residual are NUMERICAL comparisons (split sums), with the
//!   reference's own amplitudes as the bound;
//! * three tokens over the SAME buffers: fresh inputs each token, so a pack
//!   or partial not rewritten, or an accumulating sum, cannot pass.
//!
//! The reference shares card 0's context (the process admits only two).
use super::card_block::run_card_moe_block;
use super::card_step::CardStack;
use super::common_tests::dims as step_dims;
use super::dims::{KvMode, StepOptions};
use super::mixer_weights::DenseWeight;
use super::moe::{
    CardMoeOp, ExpertBufs, Launch, MoeBuffers, MoeDims, MoeLayerWeights, Packs, RouteBufs,
    StackWeights, moe_sequence_card, run_moe_block,
};
use super::native_device_rig::{Rng, device_present, f32_buf, q4k_rows, q8_1_bytes, zero_buf};
use super::shard::ShardTable;
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::family::project::cuda::served_admission;
use crate::kernels::cuda::NativeWeight;
use crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};
use crate::weights::GgufType;

const F32: usize = size_of::<f32>();
const Q4K_BLOCK: usize = 144;
const Q4K_VALUES: usize = 256;
/// The split-sum envelope of the mixer rigs: the per-card partial matvecs
/// differ from the fused one by reduction order only.
const SPLIT_SUM_RELATIVE: f32 = 1e-3;
const TOKENS: usize = 3;

fn options() -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    }
}

/// The whole fixture of the rig at the model's MoE geometry.
struct Sources {
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    shared_gate: Vec<u8>,
    shared_up: Vec<u8>,
    shared_down: Vec<u8>,
    router: Vec<f32>,
    router_shared: Vec<f32>,
    norms: [Vec<f32>; TOKENS],
    resids: [Vec<f32>; TOKENS],
}

fn sources(dims: &MoeDims, seed: u64) -> Sources {
    let mut rng = Rng::new(seed);
    let expert_rows =
        |rng: &mut Rng, rows_per: usize, k: usize| q4k_rows(dims.n_expert * rows_per, k, rng);
    let dense_rows = |rng: &mut Rng, rows: usize, k: usize| q4k_rows(rows, k, rng);
    let row = |rng: &mut Rng, n: usize| {
        (0..n)
            .map(|_| (rng.below(200) as f32 - 100.0) / 400.0)
            .collect()
    };
    let norms = std::array::from_fn(|_| row(&mut rng, dims.hidden));
    let resids = std::array::from_fn(|_| row(&mut rng, dims.hidden));
    Sources {
        gate: expert_rows(&mut rng, dims.inter, dims.hidden),
        up: expert_rows(&mut rng, dims.inter, dims.hidden),
        down: expert_rows(&mut rng, dims.hidden, dims.inter),
        shared_gate: dense_rows(&mut rng, dims.shared_inter, dims.hidden),
        shared_up: dense_rows(&mut rng, dims.shared_inter, dims.hidden),
        shared_down: dense_rows(&mut rng, dims.hidden, dims.shared_inter),
        router: row(&mut rng, dims.n_expert * dims.hidden),
        router_shared: row(&mut rng, dims.hidden),
        norms,
        resids,
    }
}

/// The rows `range` of every expert's block of a Q4_K expert stack: the card
/// owns the same row range of EVERY expert.
fn expert_row_slice(
    source: &[u8],
    experts: usize,
    rows_per: usize,
    k: usize,
    range: std::ops::Range<usize>,
) -> Vec<u8> {
    let per_row = k / Q4K_VALUES * Q4K_BLOCK;
    let mut out = Vec::with_capacity(experts * range.len() * per_row);
    for expert in 0..experts {
        let at = (expert * rows_per + range.start) * per_row;
        out.extend_from_slice(&source[at..at + range.len() * per_row]);
    }
    out
}

/// The columns `range` of every row of every expert's block of a Q4_K expert
/// stack; the cut must sit on Q4_K row blocks.
fn expert_col_slice(
    source: &[u8],
    experts: usize,
    rows_per: usize,
    k: usize,
    range: std::ops::Range<usize>,
) -> Vec<u8> {
    assert!(
        range.start.is_multiple_of(Q4K_VALUES) && range.end.is_multiple_of(Q4K_VALUES),
        "the column cut must sit on Q4_K row blocks"
    );
    let blocks = k / Q4K_VALUES;
    let take = (range.end - range.start) / Q4K_VALUES;
    let mut out = Vec::with_capacity(experts * rows_per * take * Q4K_BLOCK);
    for row in 0..experts * rows_per {
        let at = (row * blocks + range.start / Q4K_VALUES) * Q4K_BLOCK;
        out.extend_from_slice(&source[at..at + take * Q4K_BLOCK]);
    }
    out
}

fn native_weight(
    ctx: &CudaContext,
    source: &[u8],
    rows: usize,
    k: usize,
    experts: Option<usize>,
) -> Result<NativeWeight, EngError> {
    let tensor = crate::kernels::cuda::QuantTensor {
        name: "rig",
        ty: GgufType::Q4K,
        n_rows: rows,
        k,
        source_len: source.len(),
    };
    let mut reader = |at: usize, buf: &mut [u8]| {
        buf.copy_from_slice(&source[at..at + buf.len()]);
        Ok(())
    };
    let weight = ctx.upload_native(tensor, 1 << 16, &mut reader)?;
    Ok(weight.with_experts(experts))
}

/// The owned buffers of one run (reference or card): the route words, the
/// q8_1 packs, the intermediates and the residual rows.
struct RunBufs {
    norm: DeviceBuffer,
    resid: DeviceBuffer,
    resid_out: DeviceBuffer,
    logits: DeviceBuffer,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    shared_gate: DeviceBuffer,
    status: DeviceBuffer,
    packs: Vec<DeviceBuffer>,
    expert_gu: DeviceBuffer,
    expert_down: DeviceBuffer,
    shared_gu: DeviceBuffer,
    shared_out: DeviceBuffer,
}

fn run_bufs(ctx: &CudaContext, dims: &MoeDims) -> Result<RunBufs, EngError> {
    let q8 = |k: usize, rows: usize| -> Result<DeviceBuffer, EngError> {
        zero_buf(
            ctx,
            q8_1_bytes(k, rows).ok_or_else(|| EngError::Checkpoint("q8".into()))?,
        )
    };
    Ok(RunBufs {
        norm: zero_buf(ctx, dims.hidden * F32)?,
        resid: zero_buf(ctx, dims.hidden * F32)?,
        resid_out: zero_buf(ctx, dims.hidden * F32)?,
        logits: zero_buf(ctx, (dims.n_expert + 1) * F32)?,
        ids: zero_buf(ctx, dims.top_k * 4)?,
        weights: zero_buf(ctx, dims.top_k * F32)?,
        shared_gate: zero_buf(ctx, F32)?,
        status: zero_buf(ctx, 4)?,
        packs: vec![
            q8(dims.hidden, 1)?,
            q8(dims.inter, dims.top_k)?,
            q8(dims.shared_inter, 1)?,
        ],
        expert_gu: zero_buf(ctx, dims.top_k * dims.inter * 2 * F32)?,
        expert_down: zero_buf(ctx, (dims.top_k + 1) * dims.hidden * F32)?,
        shared_gu: zero_buf(ctx, dims.shared_inter * 2 * F32)?,
        shared_out: zero_buf(ctx, dims.hidden * F32)?,
    })
}

fn range(buf: &DeviceBuffer) -> Result<crate::kernels::cuda::DeviceRange<'_>, EngError> {
    buf.range(0, buf.bytes())
}

fn moe_bufs(b: &RunBufs, combined: bool) -> Result<MoeBuffers<'_>, EngError> {
    let row_bytes = b.resid.bytes();
    let down_bytes = (b.ids.bytes() / 4) * row_bytes;
    let shared_out = if combined {
        b.expert_down.range(down_bytes, row_bytes)?
    } else {
        range(&b.shared_out)?
    };
    Ok(MoeBuffers {
        route: RouteBufs {
            resid_in: range(&b.resid)?,
            resid_out: range(&b.resid_out)?,
            norm_row: range(&b.norm)?,
            logits: range(&b.logits)?,
            ids: range(&b.ids)?,
            weights: range(&b.weights)?,
            shared_gate: range(&b.shared_gate)?,
            status: range(&b.status)?,
        },
        expert: ExpertBufs {
            packs: Packs::Q8 {
                moe: range(&b.packs[0])?,
                down: range(&b.packs[1])?,
                shared: range(&b.packs[2])?,
            },
            expert_gu: range(&b.expert_gu)?,
            expert_down: b.expert_down.range(0, down_bytes)?,
            shared_gu: range(&b.shared_gu)?,
            shared_out,
        },
    })
}

fn produce(ctx: &CudaContext, stream: &DeviceStream, b: &RunBufs) -> Result<(), EngError> {
    ctx.quantize_q8_1(
        stream,
        QuantizeCall {
            x: range(&b.norm)?,
            y: range(&b.packs[0])?,
            k: b.norm.bytes() / F32,
            ncols: 1,
        },
    )
}

/// One token's read-back, plain data for the host asserts.
#[derive(Debug, PartialEq)]
struct TokenRun {
    ids: Vec<i32>,
    weights: Vec<f32>,
    shared_gate: Vec<f32>,
    expert_gu: Vec<f32>,
    expert_down: Vec<f32>,
    shared_gu: Vec<f32>,
    shared_out: Vec<f32>,
    resid_out: Vec<f32>,
    down_pack: Vec<u8>,
    shared_pack: Vec<u8>,
}

fn read_run(stream: &DeviceStream, b: &RunBufs, combined: bool) -> Result<TokenRun, EngError> {
    let read_f32 = |buf: &DeviceBuffer| -> Result<Vec<f32>, EngError> {
        let mut bytes = vec![0u8; buf.bytes()];
        buf.read_at(stream, 0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };
    let read_i32 = |buf: &DeviceBuffer| -> Result<Vec<i32>, EngError> {
        let mut bytes = vec![0u8; buf.bytes()];
        buf.read_at(stream, 0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect())
    };
    let mut down_pack = vec![0u8; b.packs[1].bytes()];
    b.packs[1].read_at(stream, 0, &mut down_pack)?;
    let mut shared_pack = vec![0u8; b.packs[2].bytes()];
    b.packs[2].read_at(stream, 0, &mut shared_pack)?;
    let mut expert_down = read_f32(&b.expert_down)?;
    let shared_tail = expert_down.split_off((b.ids.bytes() / 4) * (b.resid.bytes() / F32));
    let shared_out = if combined {
        shared_tail
    } else {
        read_f32(&b.shared_out)?
    };
    Ok(TokenRun {
        ids: read_i32(&b.ids)?,
        weights: read_f32(&b.weights)?,
        shared_gate: read_f32(&b.shared_gate)?,
        expert_gu: read_f32(&b.expert_gu)?,
        expert_down,
        shared_gu: read_f32(&b.shared_gu)?,
        shared_out,
        resid_out: read_f32(&b.resid_out)?,
        down_pack,
        shared_pack,
    })
}

/// The owned weights of one side: the stacks and denses of its width. The
/// raw router rows are borrowed from the reference (only card 0 reads them).
struct OwnedWeights {
    stacks: Vec<NativeWeight>,
    denses: Vec<NativeWeight>,
}

impl OwnedWeights {
    fn moe<'a>(
        &'a self,
        router: &'a DeviceBuffer,
        router_shared: &'a DeviceBuffer,
    ) -> MoeLayerWeights<'a> {
        MoeLayerWeights {
            router,
            router_shared,
            gate: StackWeights::Native(&self.stacks[0]),
            up: StackWeights::Native(&self.stacks[1]),
            down: StackWeights::Native(&self.stacks[2]),
            shared_gate: DenseWeight::Native(&self.denses[0]),
            shared_up: DenseWeight::Native(&self.denses[1]),
            shared_down: DenseWeight::Native(&self.denses[2]),
        }
    }
}

/// The whole comparison, `?` inside; asserts live in the caller.
struct Evidence {
    tokens: Vec<(TokenRun, Vec<TokenRun>)>,
    inter: Vec<std::ops::Range<usize>>,
    shared_inter: Vec<std::ops::Range<usize>>,
    dims: MoeDims,
    card_dims: Vec<MoeDims>,
    words: Vec<u32>,
}

fn run_rig(
    devices: [usize; 2],
    compare_fused: bool,
    compare_shared: bool,
) -> Result<Evidence, EngError> {
    let d = step_dims();
    let table = ShardTable::new(&d.shape, 2, 32).expect("table");
    let full = MoeDims::new(&d, 1);
    let card_dims = vec![
        MoeDims::for_card(&d, &table, 0, 1)?,
        MoeDims::for_card(&d, &table, 1, 1)?,
    ];
    let inter: Vec<std::ops::Range<usize>> = (0..2)
        .map(|c| table.expert_inter(c))
        .collect::<Result<_, _>>()?;
    let shared_inter: Vec<std::ops::Range<usize>> = (0..2)
        .map(|c| table.shared_inter(c))
        .collect::<Result<_, _>>()?;

    let s = sources(&full, 0x50E_FEED);
    let admission = served_admission();
    let (mut stack, _states) = if compare_shared {
        CardStack::new_for_fused_shared_sum_test(&d, &options(), &devices, &admission)?
    } else if compare_fused {
        CardStack::new_for_fused_expert_sum_test(&d, &options(), &devices, &admission)?
    } else {
        CardStack::new(&d, &options(), &devices, &admission)?
    };
    stack.enable_broadcast_sync_for_test();
    let ctx0 = &stack.contexts()[0];

    // The reference: full widths, card 0's context. Its router rows serve
    // every side (only card 0's router op reads them).
    let router = f32_buf(ctx0, &s.router)?;
    let router_shared = f32_buf(ctx0, &s.router_shared)?;
    let reference = OwnedWeights {
        stacks: vec![
            native_weight(
                ctx0,
                &s.gate,
                full.n_expert * full.inter,
                full.hidden,
                Some(full.n_expert),
            )?,
            native_weight(
                ctx0,
                &s.up,
                full.n_expert * full.inter,
                full.hidden,
                Some(full.n_expert),
            )?,
            native_weight(
                ctx0,
                &s.down,
                full.n_expert * full.hidden,
                full.inter,
                Some(full.n_expert),
            )?,
        ],
        denses: vec![
            native_weight(ctx0, &s.shared_gate, full.shared_inter, full.hidden, None)?,
            native_weight(ctx0, &s.shared_up, full.shared_inter, full.hidden, None)?,
            native_weight(ctx0, &s.shared_down, full.hidden, full.shared_inter, None)?,
        ],
    };
    let ref_bufs = run_bufs(ctx0, &full)?;

    // The cards: expert row slices for gate/up, column slices for down, the
    // shared expert the same; the router rows borrowed from the reference.
    let mut card_weights = Vec::with_capacity(2);
    let mut card_bufs = Vec::with_capacity(2);
    for card in 0..2 {
        let ctx = &stack.contexts()[card];
        let dims = &card_dims[card];
        let gate = expert_row_slice(
            &s.gate,
            full.n_expert,
            full.inter,
            full.hidden,
            inter[card].clone(),
        );
        let up = expert_row_slice(
            &s.up,
            full.n_expert,
            full.inter,
            full.hidden,
            inter[card].clone(),
        );
        let down = expert_col_slice(
            &s.down,
            full.n_expert,
            full.hidden,
            full.inter,
            inter[card].clone(),
        );
        let sg = expert_row_slice(
            &s.shared_gate,
            1,
            full.shared_inter,
            full.hidden,
            shared_inter[card].clone(),
        );
        let su = expert_row_slice(
            &s.shared_up,
            1,
            full.shared_inter,
            full.hidden,
            shared_inter[card].clone(),
        );
        let sd = expert_col_slice(
            &s.shared_down,
            1,
            full.hidden,
            full.shared_inter,
            shared_inter[card].clone(),
        );
        card_weights.push(OwnedWeights {
            stacks: vec![
                native_weight(
                    ctx,
                    &gate,
                    full.n_expert * dims.inter,
                    full.hidden,
                    Some(full.n_expert),
                )?,
                native_weight(
                    ctx,
                    &up,
                    full.n_expert * dims.inter,
                    full.hidden,
                    Some(full.n_expert),
                )?,
                native_weight(
                    ctx,
                    &down,
                    full.n_expert * full.hidden,
                    dims.inter,
                    Some(full.n_expert),
                )?,
            ],
            denses: vec![
                native_weight(ctx, &sg, dims.shared_inter, full.hidden, None)?,
                native_weight(ctx, &su, dims.shared_inter, full.hidden, None)?,
                native_weight(ctx, &sd, full.hidden, dims.shared_inter, None)?,
            ],
        });
        card_bufs.push(run_bufs(ctx, dims)?);
    }

    let ops = [moe_sequence_card(true), moe_sequence_card(false)];
    let weight_sets: Vec<MoeLayerWeights<'_>> = card_weights
        .iter()
        .map(|w| w.moe(&router, &router_shared))
        .collect();
    let op_lists: Vec<&[CardMoeOp]> = ops.iter().map(|o| o.as_slice()).collect();

    let rounds = if compare_fused || compare_shared {
        8
    } else {
        1
    };
    let mut tokens = Vec::with_capacity(TOKENS * rounds);
    for step in 0..TOKENS * rounds {
        let t = step % TOKENS;
        // The reference first, on card 0's stream: write, pack, run, sync.
        {
            let stream = &stack.streams()[0];
            ref_bufs.norm.write_at(stream, 0, &f32_bytes(&s.norms[t]))?;
            ref_bufs
                .resid
                .write_at(stream, 0, &f32_bytes(&s.resids[t]))?;
            produce(ctx0, stream, &ref_bufs)?;
            run_moe_block(
                &Launch { ctx: ctx0, stream },
                &reference.moe(&router, &router_shared),
                &moe_bufs(&ref_bufs, false)?,
                &full,
            )?;
            stream.synchronize()?;
        }
        let reference_run = read_run(&stack.streams()[0], &ref_bufs, false)?;
        let mut card_runs = Vec::with_capacity(2);
        for ((ctx, stream), bufs) in stack.contexts().iter().zip(stack.streams()).zip(&card_bufs) {
            bufs.norm.write_at(stream, 0, &f32_bytes(&s.norms[t]))?;
            bufs.resid.write_at(stream, 0, &f32_bytes(&s.resids[t]))?;
            produce(ctx, stream, bufs)?;
        }
        let buf_sets: Vec<MoeBuffers<'_>> = card_bufs
            .iter()
            .map(|b| moe_bufs(b, false))
            .collect::<Result<_, _>>()?;
        let separate_start = stack.ring().next_sequence_for_test();
        run_card_moe_block(&stack, &weight_sets, &buf_sets, &op_lists)?;
        let separate_count = stack.ring().next_sequence_for_test() - separate_start;
        for (stream, bufs) in stack.streams().iter().zip(&card_bufs) {
            stream.synchronize()?;
            card_runs.push(read_run(stream, bufs, false)?);
        }
        if compare_fused || compare_shared {
            if compare_shared {
                stack.set_fused_shared_sum_for_test(true);
            } else {
                stack.set_fused_expert_sum_for_test(true);
            }
            for ((ctx, stream), bufs) in
                stack.contexts().iter().zip(stack.streams()).zip(&card_bufs)
            {
                bufs.norm.write_at(stream, 0, &f32_bytes(&s.norms[t]))?;
                bufs.resid.write_at(stream, 0, &f32_bytes(&s.resids[t]))?;
                produce(ctx, stream, bufs)?;
            }
            let fused_buf_sets: Vec<MoeBuffers<'_>> = card_bufs
                .iter()
                .map(|b| moe_bufs(b, compare_shared))
                .collect::<Result<_, _>>()?;
            let fused_start = stack.ring().next_sequence_for_test();
            run_card_moe_block(&stack, &weight_sets, &fused_buf_sets, &op_lists)?;
            let fused_count = stack.ring().next_sequence_for_test() - fused_start;
            assert_eq!(
                separate_count,
                fused_count
                    + if compare_shared {
                        1
                    } else {
                        u32::try_from(full.top_k - 1).unwrap()
                    },
                "the fused path must remove the expected exchanges"
            );
            if step == 0 {
                println!("MoE ring exchanges: separate {separate_count}, fused {fused_count}");
            }
            for (card, (stream, bufs)) in stack.streams().iter().zip(&card_bufs).enumerate() {
                stream.synchronize()?;
                let fused = read_run(stream, bufs, compare_shared)?;
                assert_eq!(fused, card_runs[card], "step {step} card {card}");
                for (label, got, want) in [
                    (
                        "expert_down",
                        &fused.expert_down,
                        &card_runs[card].expert_down,
                    ),
                    ("shared_out", &fused.shared_out, &card_runs[card].shared_out),
                    ("resid_out", &fused.resid_out, &card_runs[card].resid_out),
                ] {
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(got),
                        bits(want),
                        "step {step} card {card} {label} bits"
                    );
                }
            }
            if compare_shared {
                stack.set_fused_shared_sum_for_test(false);
            } else {
                stack.set_fused_expert_sum_for_test(false);
            }
        }
        tokens.push((reference_run, card_runs));
    }
    let mut words = Vec::with_capacity(2);
    for card in 0..2 {
        let mut word = [0u8; 4];
        stack.shards()[card]
            .ring_word()
            .read_at(&stack.streams()[card], 0, &mut word)?;
        words.push(u32::from_le_bytes(word));
    }
    Ok(Evidence {
        tokens,
        inter,
        shared_inter,
        dims: full,
        card_dims,
        words,
    })
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

#[test]
fn a_moe_block_runs_distributed_and_matches_the_reference() {
    const NAME: &str = "a_moe_block_runs_distributed_and_matches_the_reference";
    if !device_present(NAME) {
        return;
    }
    let ev = run_rig([0, 0], false, false).expect("the rig");
    let dims = &ev.dims;
    let gu_stride = |d: &MoeDims| d.inter * 2;
    let pack_row = |d: &MoeDims| q8_1_bytes(d.inter, 1).expect("q8 row");
    for (t, (reference, cards)) in ev.tokens.iter().enumerate() {
        for (card, run) in cards.iter().enumerate() {
            let cd = &ev.card_dims[card];
            // The router's broadcast: ids, weights and the shared gate exact.
            assert_eq!(
                run.ids, reference.ids,
                "{NAME}: t{t} card {card}: the picked experts differ"
            );
            assert_eq!(
                run.weights, reference.weights,
                "{NAME}: t{t} card {card}: the pick weights differ"
            );
            assert_eq!(
                run.shared_gate, reference.shared_gate,
                "{NAME}: t{t} card {card}: the shared gate differs"
            );
            // The gate/up projections: each HALF of the card's slot row
            // (gate then up, concatenated at the narrowed inter) is exact
            // against the corresponding half-range of the reference slot.
            for (half, name) in [(0, "gate"), (1, "up")] {
                for slot in 0..dims.top_k {
                    let at = half * cd.inter;
                    let got = &run.expert_gu
                        [slot * gu_stride(cd) + at..slot * gu_stride(cd) + at + cd.inter];
                    let from = slot * gu_stride(dims) + half * dims.inter + ev.inter[card].start;
                    let want = &reference.expert_gu[from..from + cd.inter];
                    assert_eq!(
                        got, want,
                        "{NAME}: t{t} card {card} slot {slot}: the {name}/up half differs"
                    );
                }
                let at = half * cd.shared_inter;
                let got = &run.shared_gu[at..at + cd.shared_inter];
                let from = half * dims.shared_inter + ev.shared_inter[card].start;
                let want = &reference.shared_gu[from..from + cd.shared_inter];
                assert_eq!(
                    got, want,
                    "{NAME}: t{t} card {card}: the shared {name}/up half differs"
                );
            }
            // The Q8_1 packs at the narrowed inter: the card's own blocks
            // (codes and scales) exact against the reference's block range.
            let take = (ev.inter[card].end - ev.inter[card].start) / 32 * 36;
            for slot in 0..dims.top_k {
                let from = slot * pack_row(dims) + ev.inter[card].start / 32 * 36;
                let want = &reference.down_pack[from..from + take];
                let got = &run.down_pack[slot * pack_row(cd)..slot * pack_row(cd) + take];
                assert_eq!(
                    got, want,
                    "{NAME}: t{t} card {card} slot {slot}: the q8_1 pack blocks differ"
                );
            }
            let shared_take = (ev.shared_inter[card].end - ev.shared_inter[card].start) / 32 * 36;
            let from = ev.shared_inter[card].start / 32 * 36;
            let want = &reference.shared_pack[from..from + shared_take];
            let got = &run.shared_pack[..shared_take];
            assert_eq!(
                got, want,
                "{NAME}: t{t} card {card}: the shared q8_1 pack blocks differ"
            );
            // The split sums: expert down per slot, shared out, residual.
            for slot in 0..dims.top_k {
                let at = slot * dims.hidden;
                let got = &run.expert_down[at..at + dims.hidden];
                let want = &reference.expert_down[at..at + dims.hidden];
                let diff = max_abs_diff(got, want);
                let bound = SPLIT_SUM_RELATIVE * max_abs(want);
                println!(
                    "{NAME}: t{t} card {card} slot {slot} expert_down diff {diff:.3e} bound {bound:.3e}"
                );
                assert!(
                    diff <= bound,
                    "{NAME}: t{t} card {card} slot {slot}: expert_down differs by {diff} > {bound}"
                );
            }
            let diff = max_abs_diff(&run.shared_out, &reference.shared_out);
            let bound = SPLIT_SUM_RELATIVE * max_abs(&reference.shared_out);
            println!("{NAME}: t{t} card {card} shared_out diff {diff:.3e} bound {bound:.3e}");
            assert!(
                diff <= bound,
                "{NAME}: t{t} card {card}: shared_out differs by {diff} > {bound}"
            );
            let diff = max_abs_diff(&run.resid_out, &reference.resid_out);
            let bound = SPLIT_SUM_RELATIVE * max_abs(&reference.resid_out);
            println!("{NAME}: t{t} card {card} resid_out diff {diff:.3e} bound {bound:.3e}");
            assert!(
                diff <= bound,
                "{NAME}: t{t} card {card}: resid_out differs by {diff} > {bound}"
            );
        }
        // The two cards' combined residuals agree (identical sums up to the
        // ring's addition order), and the picks are 8 distinct experts.
        let cross = max_abs_diff(&cards[0].resid_out, &cards[1].resid_out);
        println!("{NAME}: t{t} resid_out cross-card diff {cross:.3e}");
        assert!(
            cross <= SPLIT_SUM_RELATIVE * max_abs(&reference.resid_out),
            "{NAME}: t{t}: the cards' residuals disagree"
        );
        let mut sorted = reference.ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), dims.top_k, "{NAME}: t{t}: duplicate picks");
    }
    for (card, word) in ev.words.iter().enumerate() {
        assert_eq!(*word, 0, "{NAME}: card {card}: ring not clean");
    }
}

#[test]
#[ignore = "requires two peer-enabled CUDA devices; run alone under the GPU lease"]
fn fused_expert_sums_match_separate_sums_on_peer_gpus() {
    assert!(device_present(
        "fused_expert_sums_match_separate_sums_on_peer_gpus"
    ));
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe GPUs");
    crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer access");
    crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peers");
    let evidence = run_rig(devices, true, false).expect("fused and separate sums");
    assert_eq!(evidence.tokens.len(), TOKENS * 8);
    assert_eq!(evidence.words, vec![0, 0]);
}

#[test]
#[ignore = "requires two peer-enabled CUDA devices; run alone under the GPU lease"]
fn fused_shared_sum_matches_separate_sum_on_peer_gpus() {
    assert!(device_present(
        "fused_shared_sum_matches_separate_sum_on_peer_gpus"
    ));
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe GPUs");
    crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer access");
    crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peers");
    let evidence = run_rig(devices, false, true).expect("combined shared and separate sums");
    assert_eq!(evidence.tokens.len(), TOKENS * 8);
    assert_eq!(evidence.words, vec![0, 0]);
}
