//! Device tests of the sparse-attention decode wrapper: attention over the SELECTED cells
//! with the production Q8 arithmetic. The oracle is INDEPENDENT of the CUDA logic - a plain
//! f64 loop over the same dequantized values (`f32(f16 scale bits) * int8 code`, exact f16 to
//! f32 conversion), softmax and weighted V sum in f64 - so agreement (contract tolerance
//! 1e-3 absolute on outputs of magnitude O(1)) is a property of the kernel, not of a copy of
//! it. The GQA mapping is pinned structurally: the oracle writes `oracle[(kvh * G + g) * D +
//! d]` and the comparison reads the kernel's `out` at exactly that offset.
//!
//! Selection content mirrors the `sparse_topk` output format: ascending top-K picks in
//! `[sink, kv_len - recent)`, then sinks `[0, sink)`, then the recent window
//! `[kv_len - recent, kv_len)`. Paging and prefix-restore identity are orchestration
//! properties (phase 4d); here the indices ARE physical cells by construction.
//!
//! Runs on the device only: without one, each test returns early (`device_present` honors
//! INFERENCE_CUDA_REQUIRE_DEVICE, so the device window runs them strictly).

use crate::kernels::cuda::context::{CudaContext, DeviceStream};
use crate::kernels::cuda::resident::sparse_decode::{SparseDecodeData, SparseDecodeShape};
use crate::kernels::cuda::testkit::{self, Rng, f16_bits, f16_to_f32};

const KV_HEADS: usize = 2;
const PER_KV_HEAD: usize = 4; // G: query heads per KV head, the production GQA ratio
const N_HEAD: usize = KV_HEADS * PER_KV_HEAD;
const D: usize = 128;
const CELLS: usize = 256;
const TOP_K: usize = 56;
const SINK: usize = 4;
const RECENT: usize = 8;
const ROW: usize = TOP_K + SINK + RECENT; // 68: two 64-tiles, so boundaries are covered
const KV_LEN: usize = 200; // < CELLS: room above the visible window
const SCALE: f32 = 0.088_388_35; // 1/sqrt(128), a production-style attention scale
const TOLERANCE: f32 = 1e-3; // the prespecified subset-oracle tolerance

struct Fixture {
    q: Vec<f32>,        // [N_HEAD][D]
    k_codes: Vec<i8>,   // [KV_HEADS][CELLS][D]
    k_scales: Vec<u16>, // [KV_HEADS][CELLS][D / 32]
    v_codes: Vec<i8>,
    v_scales: Vec<u16>,
    indices: Vec<u32>, // [KV_HEADS][ROW]
    counts: Vec<u32>,  // [KV_HEADS]
    /// Optional gated-attention proj row [N_HEAD][2D]: sigmoid of the D half gates out.
    proj: Option<Vec<f32>>,
}

/// Deterministic fixture: standard-normal q, int8 codes against |N(0,1)| * 4/127 fp16 scales
/// (the spread the production decode tests use, so dequantized values span ~N(0,1)), and a
/// random ascending pick set per head disjoint from sinks and the recent window.
fn fixture(rng: &mut Rng, pick_len: usize) -> Fixture {
    debug_assert!(pick_len <= TOP_K);
    let mut q = Vec::with_capacity(N_HEAD * D);
    for _ in 0..N_HEAD * D {
        q.push(rng.normal() as f32);
    }
    let codes = |rng: &mut Rng, plane: &mut Vec<i8>| {
        for _ in 0..KV_HEADS * CELLS * D {
            plane.push((rng.below(255) as i64 - 127) as i8);
        }
    };
    let scales = |rng: &mut Rng, plane: &mut Vec<u16>| {
        for _ in 0..KV_HEADS * CELLS * (D / 32) {
            plane.push(f16_bits((rng.normal().abs() * (4.0 / 127.0)) as f32));
        }
    };
    let (mut k_codes, mut v_codes, mut k_scales, mut v_scales) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    codes(rng, &mut k_codes);
    codes(rng, &mut v_codes);
    scales(rng, &mut k_scales);
    scales(rng, &mut v_scales);

    let mut indices = Vec::with_capacity(KV_HEADS * ROW);
    let mut counts = Vec::with_capacity(KV_HEADS);
    for _ in 0..KV_HEADS {
        // Ascending picks inside the candidate range, via a boolean occupancy mask.
        let mut occupied = vec![false; CELLS];
        let mut picked = 0usize;
        while picked < pick_len {
            let cell = rng.below((KV_LEN - RECENT - SINK) as u64) as usize + SINK;
            if !occupied[cell] {
                occupied[cell] = true;
                picked += 1;
            }
        }
        let mut row_indices: Vec<u32> = occupied[SINK..KV_LEN - RECENT]
            .iter()
            .enumerate()
            .filter(|&(_, &is_picked)| is_picked)
            .map(|(offset, _)| (SINK + offset) as u32)
            .collect();
        row_indices.extend((0..SINK).map(|cell| cell as u32));
        row_indices.extend((KV_LEN - RECENT..KV_LEN).map(|cell| cell as u32));
        // Row-strided layout, padded to ROW: readers index at kvh * ROW.
        row_indices.resize(ROW, 0);
        indices.extend_from_slice(&row_indices);
        counts.push((pick_len + SINK + RECENT) as u32);
    }
    Fixture {
        q,
        k_codes,
        k_scales,
        v_codes,
        v_scales,
        indices,
        counts,
        proj: None,
    }
}

impl Fixture {
    fn dequant_k(&self, kvh: usize, cell: usize) -> Vec<f64> {
        let groups = D / 32;
        (0..D)
            .map(|d| {
                let s = f16_to_f32(self.k_scales[(kvh * CELLS + cell) * groups + d / 32]);
                s as f64 * self.k_codes[(kvh * CELLS + cell) * D + d] as f64
            })
            .collect()
    }

    fn dequant_v(&self, kvh: usize, cell: usize, d: usize) -> f64 {
        let groups = D / 32;
        let s = f16_to_f32(self.v_scales[(kvh * CELLS + cell) * groups + d / 32]);
        s as f64 * self.v_codes[(kvh * CELLS + cell) * D + d] as f64
    }

    /// The independent subset oracle: f64 scores against the dequantized keys, a stable f64
    /// softmax over the head's selected multiset, and a weighted sum of dequantized V rows.
    fn oracle(&self) -> Vec<f32> {
        let mut out = vec![0f32; N_HEAD * D];
        for kvh in 0..KV_HEADS {
            let count = self.counts[kvh] as usize;
            let sel = &self.indices[kvh * ROW..][..count];
            for g in 0..PER_KV_HEAD {
                let h = kvh * PER_KV_HEAD + g;
                let scores: Vec<f64> = sel
                    .iter()
                    .map(|&cell| {
                        let key = self.dequant_k(kvh, cell as usize);
                        let mut dot = 0f64;
                        for (qv, kv) in self.q[h * D..][..D].iter().zip(&key) {
                            dot += *qv as f64 * kv;
                        }
                        dot * SCALE as f64
                    })
                    .collect();
                let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let l: f64 = weights.iter().sum();
                for d in 0..D {
                    let mut acc = 0f64;
                    for (j, &cell) in sel.iter().enumerate() {
                        acc += weights[j] * self.dequant_v(kvh, cell as usize, d);
                    }
                    // The production gated-attention output: sigmoid of the proj D half.
                    let sig = match self.proj.as_ref() {
                        None => 1.0,
                        Some(proj) => 1.0 / (1.0 + (-proj[h * 2 * D + D + d] as f64).exp()),
                    };
                    out[h * D + d] = ((acc / l) * sig) as f32;
                }
            }
        }
        out
    }
}

fn shape() -> SparseDecodeShape {
    SparseDecodeShape {
        n_kv_head: KV_HEADS,
        n_head: N_HEAD,
        head_dim: D,
        cells: CELLS,
        row: ROW,
        scale: SCALE,
    }
}

fn run(ctx: &CudaContext, stream: &DeviceStream, f: &Fixture) -> Vec<f32> {
    let out_bytes = N_HEAD * D * 4;
    let q = testkit::upload_f32(ctx, &f.q);
    let k_codes = testkit::upload_bytes(
        ctx,
        &f.k_codes.iter().map(|c| *c as u8).collect::<Vec<u8>>(),
    );
    let k_scales = testkit::upload_bytes(
        ctx,
        &f.k_scales
            .iter()
            .flat_map(|b| b.to_ne_bytes().into_iter())
            .collect::<Vec<_>>(),
    );
    let v_codes = testkit::upload_bytes(
        ctx,
        &f.v_codes.iter().map(|c| *c as u8).collect::<Vec<u8>>(),
    );
    let v_scales = testkit::upload_bytes(
        ctx,
        &f.v_scales
            .iter()
            .flat_map(|b| b.to_ne_bytes().into_iter())
            .collect::<Vec<_>>(),
    );
    let indices = testkit::upload_bytes(
        ctx,
        &f.indices
            .iter()
            .flat_map(|b| b.to_ne_bytes().into_iter())
            .collect::<Vec<_>>(),
    );
    let counts = testkit::upload_bytes(
        ctx,
        &f.counts
            .iter()
            .flat_map(|b| b.to_ne_bytes().into_iter())
            .collect::<Vec<_>>(),
    );
    let out = testkit::upload_filled(ctx, 0xff, out_bytes);
    let status = testkit::upload_filled(ctx, 0x00, 4);
    let proj = f.proj.as_ref().map(|rows| testkit::upload_f32(ctx, rows));
    let gate = proj
        .as_ref()
        .map(|buf| buf.range(0, N_HEAD * 2 * D * 4).unwrap());
    ctx.sparse_decode(
        stream,
        shape(),
        SparseDecodeData {
            q: q.range(0, N_HEAD * D * 4).unwrap(),
            k_codes: k_codes.range(0, KV_HEADS * CELLS * D).unwrap(),
            k_scales: k_scales.range(0, KV_HEADS * CELLS * D / 32 * 2).unwrap(),
            v_codes: v_codes.range(0, KV_HEADS * CELLS * D).unwrap(),
            v_scales: v_scales.range(0, KV_HEADS * CELLS * D / 32 * 2).unwrap(),
            indices: indices.range(0, KV_HEADS * ROW * 4).unwrap(),
            counts: counts.range(0, KV_HEADS * 4).unwrap(),
            out: out.range(0, out_bytes).unwrap(),
            proj: gate,
            status: status.range(0, 4).unwrap(),
        },
    )
    .expect("decode queued");
    stream.synchronize().expect("drain");
    let status_word = u32::from_ne_bytes(
        testkit::read_bytes(ctx, &status, 4)[..4]
            .try_into()
            .unwrap(),
    );
    assert_eq!(
        status_word, 0,
        "a valid selection must not raise a status mask"
    );
    testkit::floats(&testkit::read_bytes(ctx, &out, out_bytes))
}

#[test]
fn decode_matches_the_independent_subset_reference() {
    if !testkit::device_present("sparse_decode subset reference") {
        return;
    }
    let mut rng = Rng::new(0x5eed_dec0de);
    let f = fixture(&mut rng, TOP_K);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let got = run(&ctx, &stream, &f);
    let want = f.oracle();
    let mut worst = 0f32;
    for h in 0..N_HEAD {
        for d in 0..D {
            let (g, w) = (got[h * D + d], want[h * D + d]);
            let diff = (g - w).abs();
            assert!(
                diff < TOLERANCE,
                "head {h} dim {d}: kernel {g} vs oracle {w} (diff {diff})"
            );
            worst = worst.max(diff);
        }
    }
    eprintln!("subset oracle worst abs diff: {worst}");
}

#[test]
fn decode_matches_the_reference_across_tile_boundaries() {
    if !testkit::device_present("sparse_decode tile boundaries") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    // 0 and 1 (near-empty), 20 (one tile), 52 (count exactly 64), 56 (the full row, 68).
    for (pick_len, seed) in [(0usize, 11u64), (1, 12), (20, 13), (52, 14), (56, 15)] {
        let mut rng = Rng::new(seed);
        let f = fixture(&mut rng, pick_len);
        let got = run(&ctx, &stream, &f);
        let want = f.oracle();
        for h in 0..N_HEAD {
            for d in 0..D {
                let diff = (got[h * D + d] - want[h * D + d]).abs();
                assert!(
                    diff < TOLERANCE,
                    "pick_len {pick_len} head {h} dim {d}: kernel {} vs oracle {}",
                    got[h * D + d],
                    want[h * D + d]
                );
            }
        }
    }
}

#[test]
fn duplicate_indices_are_a_multiset_by_contract() {
    if !testkit::device_present("sparse_decode duplicates") {
        return;
    }
    let mut rng = Rng::new(0xd00d);
    let mut f = fixture(&mut rng, TOP_K);
    // Duplicate one picked cell over another pick (the same token twice): the kernel attends
    // to the multiset - the token's mass doubles - which the selection contract forbids but
    // the decode documents. The oracle over the same multiset must agree.
    let kvh = 0usize;
    f.indices[kvh * ROW + 1] = f.indices[kvh * ROW];
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let got = run(&ctx, &stream, &f);
    let want = f.oracle();
    for h in 0..N_HEAD {
        for d in 0..D {
            assert!(
                (got[h * D + d] - want[h * D + d]).abs() < TOLERANCE,
                "duplicate multiset head {h} dim {d}"
            );
        }
    }
}

#[test]
fn refused_selections_write_nothing_and_raise_mask_4() {
    if !testkit::device_present("sparse_decode refusals") {
        return;
    }
    let mut rng = Rng::new(0xfeed);
    let f = fixture(&mut rng, TOP_K);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let out_bytes = N_HEAD * D * 4;
    for (label, refusing_head, mutate) in [
        (
            "count above row",
            0usize,
            Box::new(move |f: &mut Fixture| {
                f.counts[0] = ROW as u32 + 1;
            }) as Box<dyn Fn(&mut Fixture)>,
        ),
        (
            "count overflowing u32",
            0,
            Box::new(|f: &mut Fixture| {
                f.counts[0] = u32::MAX;
            }),
        ),
        (
            "index at cells",
            0,
            Box::new(|f: &mut Fixture| {
                f.indices[0] = CELLS as u32;
            }),
        ),
        (
            "index above cells",
            1,
            Box::new(|f: &mut Fixture| {
                f.indices[KV_HEADS * ROW - 1] = u32::MAX;
            }),
        ),
    ] {
        let mut f = f.clone_fixture();
        mutate(&mut f);
        let q = testkit::upload_f32(&ctx, &f.q);
        let k_codes = testkit::upload_bytes(
            &ctx,
            &f.k_codes.iter().map(|c| *c as u8).collect::<Vec<u8>>(),
        );
        let k_scales = testkit::upload_bytes(
            &ctx,
            &f.k_scales
                .iter()
                .flat_map(|b| b.to_ne_bytes().into_iter())
                .collect::<Vec<_>>(),
        );
        let v_codes = testkit::upload_bytes(
            &ctx,
            &f.v_codes.iter().map(|c| *c as u8).collect::<Vec<u8>>(),
        );
        let v_scales = testkit::upload_bytes(
            &ctx,
            &f.v_scales
                .iter()
                .flat_map(|b| b.to_ne_bytes().into_iter())
                .collect::<Vec<_>>(),
        );
        let indices = testkit::upload_bytes(
            &ctx,
            &f.indices
                .iter()
                .flat_map(|b| b.to_ne_bytes().into_iter())
                .collect::<Vec<_>>(),
        );
        let counts = testkit::upload_bytes(
            &ctx,
            &f.counts
                .iter()
                .flat_map(|b| b.to_ne_bytes().into_iter())
                .collect::<Vec<_>>(),
        );
        let out = testkit::upload_filled(&ctx, 0xff, out_bytes);
        let status = testkit::upload_filled(&ctx, 0x00, 4);
        ctx.sparse_decode(
            &stream,
            shape(),
            SparseDecodeData {
                q: q.range(0, N_HEAD * D * 4).unwrap(),
                k_codes: k_codes.range(0, KV_HEADS * CELLS * D).unwrap(),
                k_scales: k_scales.range(0, KV_HEADS * CELLS * D / 32 * 2).unwrap(),
                v_codes: v_codes.range(0, KV_HEADS * CELLS * D).unwrap(),
                v_scales: v_scales.range(0, KV_HEADS * CELLS * D / 32 * 2).unwrap(),
                indices: indices.range(0, KV_HEADS * ROW * 4).unwrap(),
                counts: counts.range(0, KV_HEADS * 4).unwrap(),
                out: out.range(0, out_bytes).unwrap(),
                proj: None,
                status: status.range(0, 4).unwrap(),
            },
        )
        .expect("queued; the device refuses it");
        stream.synchronize().expect("drain");
        // Refusal is per head: the refusing head's slice of `out` stays untouched (a valid
        // head's block may have written its slice before the mask is observed; a set mask
        // always means unusable results and the dense fallback).
        let head_bytes = PER_KV_HEAD * D * 4;
        let got = testkit::read_bytes(&ctx, &out, out_bytes);
        let from = refusing_head * head_bytes;
        assert_eq!(
            &got[from..from + head_bytes],
            &vec![0xffu8; head_bytes][..],
            "{label}: the refusing head must not write out"
        );
        let word = u32::from_ne_bytes(
            testkit::read_bytes(&ctx, &status, 4)[..4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(word & 4, 4, "{label}: mask 4");
        // The orchestration consults this helper after synchronize: an invalid
        // head is a refusal (dense fallback), never an empty selection.
        assert!(super::status_refused(word), "{label}: refusal consumed");
    }
}

#[test]
fn zero_count_writes_zeros_not_nan() {
    if !testkit::device_present("sparse_decode zero count") {
        return;
    }
    let mut rng = Rng::new(0x0c0);
    let mut f = fixture(&mut rng, TOP_K);
    f.counts[1] = 0; // head 1 attends to nothing: zeros, never NaN (l == 0)
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let got = run(&ctx, &stream, &f);
    // An all-zero-count head under a ZERO status word is the legitimate empty
    // selection; the refusal bit must never be set for it (status_refused is
    // what the orchestration consults after synchronize).
    for d in 0..D {
        let at = PER_KV_HEAD * D + d; // head 0 of kv head 1: out[(1 * G) * D + d]
        assert!(
            got[at] == 0.0,
            "zero-count head wrote {} at dim {d}",
            got[at]
        );
    }
}

#[test]
fn butterfly_lanes_reach_their_own_heads_scores() {
    if !testkit::device_present("sparse_decode reduce8 regression") {
        return;
    }
    // Regression for the wrong three-step reduce8 transcription (caught by the
    // f64 oracle on the first device run): orthogonal one-hot K rows make each
    // head's softmax exactly one-hot, so out[h] must equal head h's own V row
    // EXACTLY. A lane-reduction that mixes heads puts the argmax on another
    // head's cell and this fails while naming the head.
    let f = {
        let mut rng = Rng::new(0xb17);
        let mut f = fixture(&mut rng, TOP_K);
        // q[h] = 60 * e_h: heads 0..7 (both KV heads), distinct basis dims.
        for (h, chunk) in f.q.chunks_mut(D).enumerate() {
            chunk.fill(0.0);
            chunk[h] = 60.0;
        }
        let unit_scale = |plane: &mut Vec<u16>| {
            plane.clear();
            plane.resize(plane.capacity(), f16_bits(1.0));
        };
        // K[kvh][j] = 60 * e_{j mod 128}, unit scales: dot(q[h], k[j]) is
        // exactly 3600 for the head's own cell and exactly 0 elsewhere, so the
        // scaled gap (318) underflows fast-math expf to exactly 0.
        for kvh in 0..KV_HEADS {
            for j in 0..CELLS {
                let base = (kvh * CELLS + j) * D;
                f.k_codes[base..base + D].fill(0);
                f.k_codes[base + (j % D)] = 60;
            }
        }
        unit_scale(&mut f.k_scales);
        // V[j][d] = ((j*13 + d*3) % 41) - 20 with unit scales: dequant is exact.
        for kvh in 0..KV_HEADS {
            for j in 0..CELLS {
                let base = (kvh * CELLS + j) * D;
                for d in 0..D {
                    f.v_codes[base + d] = ((j * 13 + d * 3) % 41) as i8 - 20;
                }
            }
        }
        unit_scale(&mut f.v_scales);
        // Select cells [0, ROW) ascending on every head: no wrap aliases, every
        // head's own cell is in the selection, count = ROW.
        for kvh in 0..KV_HEADS {
            f.indices[kvh * ROW..kvh * ROW + ROW]
                .iter_mut()
                .enumerate()
                .for_each(|(j, slot)| *slot = j as u32);
            f.counts[kvh] = ROW as u32;
        }
        f
    };
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let got = run(&ctx, &stream, &f);
    for h in 0..N_HEAD {
        // Head h's own cell is h itself in its KV head's plane: q[h] = 60*e_h and
        // K[kvh][j] = 60*e_j, so the argmax is unique (cells [0, ROW) carry no
        // wrap alias, and the scaled gap underflows expf to exactly zero).
        let kvh = h / PER_KV_HEAD;
        let cell = h;
        let base = (kvh * CELLS + cell) * D;
        for d in 0..D {
            let want = f.v_codes[base + d] as f32; // unit scale: dequant is exact
            assert_eq!(
                got[h * D + d],
                want,
                "head {h} must equal V[cell {cell}] exactly at dim {d} \
                 (a lane-reduction that mixes heads puts the argmax elsewhere)"
            );
        }
    }
}

#[test]
fn nan_q_stays_isolated_to_its_head() {
    if !testkit::device_present("sparse_decode NaN q") {
        return;
    }
    // Under the production -use_fast_math, expf of a NaN score need not produce a NaN output,
    // so "NaN in, NaN out" is NOT pinned. What the contract needs instead: a NaN query head
    // must not disturb any other head's result.
    let mut rng = Rng::new(0x2ad0);
    let mut f = fixture(&mut rng, TOP_K);
    f.q[3 * D] = f32::NAN; // head 3, dim 0
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let got = run(&ctx, &stream, &f);
    let want = f.oracle();
    for h in 0..N_HEAD {
        if h == 3 {
            continue;
        }
        for d in 0..D {
            let (g, w) = (got[h * D + d], want[h * D + d]);
            // NaN on both sides is agreement (a NaN score may reach the output via the
            // softmax corrections); only a real diff fails.
            let agrees = (g.is_nan() && w.is_nan()) || (g - w).abs() < TOLERANCE;
            assert!(
                agrees,
                "head {h} disturbed by the NaN head: dim {d}, kernel {g} vs oracle {w}"
            );
        }
    }
}

impl Fixture {
    fn clone_fixture(&self) -> Fixture {
        Fixture {
            q: self.q.clone(),
            k_codes: self.k_codes.clone(),
            k_scales: self.k_scales.clone(),
            v_codes: self.v_codes.clone(),
            v_scales: self.v_scales.clone(),
            indices: self.indices.clone(),
            counts: self.counts.clone(),
            proj: self.proj.clone(),
        }
    }
}

#[test]
fn gated_output_matches_the_oracle_and_differs_from_ungated() {
    if !testkit::device_present("sparse_decode gated proj") {
        return;
    }
    // The production gated-attention tail: out *= sigmoid(proj[h][D..2D]). The
    // gated run must match the f64 oracle with the gate, and differ from the
    // ungated run wherever the gate is not ~1.
    let mut rng = Rng::new(0x9a7);
    let mut f = fixture(&mut rng, TOP_K);
    let mut proj = vec![0f32; N_HEAD * 2 * D];
    for v in proj.iter_mut() {
        *v = rng.normal() as f32;
    }
    f.proj = Some(proj);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let gated = run(&ctx, &stream, &f);
    let want = f.oracle();
    for i in 0..N_HEAD * D {
        assert!(
            (gated[i] - want[i]).abs() < TOLERANCE,
            "gated out {i}: kernel {} vs oracle {}",
            gated[i],
            want[i]
        );
    }
    f.proj = None;
    let ungated = run(&ctx, &stream, &f);
    let mut distinct = 0usize;
    for i in 0..N_HEAD * D {
        if (gated[i] - ungated[i]).abs() > TOLERANCE {
            distinct += 1;
        }
    }
    assert!(
        distinct > N_HEAD * D / 4,
        "the gate must materially change the output (only {distinct} of {} dims differ)",
        N_HEAD * D
    );
}
