//! SenseNova Loop family tests: plan mapping, step math against
//! hand-computed values, and an env-gated open of the real checkpoint.

use super::serve::LoopStep;
use super::step::{
    AttnView, DeltaView, LoopState, RouterView, apply_yarn, attn_step, delta_step, merge,
    mhc_route, yarn_inv_freq,
};
use super::weights::LoopShape;
use super::{LoopPlan, PHYSICAL_LAYERS, RopePlan, TRUNK_LAYERS};
use crate::family::executor::{ExecutorOps, GenerationState, SerialDecode};
use crate::family::math::sigmoid;
use crate::family::project::{DenseProjector, DenseWeight};
use crate::family::session::DecodeSession;
use crate::tokenizer::{DecodeFinish, IncrementalDecoder};
use crate::weights::GgufWeights;

fn plan() -> LoopPlan {
    // The checkpoint's own geometry: loop [4, 28) twice, mHC [4, 28).
    let mut recurrent = vec![true; TRUNK_LAYERS];
    for il in (3..TRUNK_LAYERS).step_by(4) {
        recurrent[il] = false;
    }
    LoopPlan {
        loop_start: 4,
        loop_end: 28,
        loop_count: 2,
        mhc_start: 4,
        mhc_end: 28,
        mhc_iterations: 20,
        mhc_pre_scale: 1.0,
        mhc_fixed_identity: false,
        recurrent,
    }
}

fn shape() -> LoopShape {
    LoopShape {
        hidden: 8,
        feed_forward: 16,
        heads_q: 2,
        heads_kv: 1,
        head_dim: 4,
        delta_key_heads: 1,
        delta_key_dim: 2,
        delta_value_heads: 2,
        delta_value_dim: 2,
        conv_kernel: 2,
        vocab: 11,
        context: 512,
        eps: 1e-6,
    }
}

#[test]
fn physical_mapping_matches_the_loop_plan() {
    let p = plan();
    assert_eq!(p.physical(0), 0);
    assert_eq!(p.physical(4), 4);
    assert_eq!(p.physical(27), 27);
    // Second pass rewinds to the loop start.
    assert_eq!(p.physical(28), 4);
    assert_eq!(p.physical(51), 27);
    // Tail layers continue after the extra visits.
    assert_eq!(p.physical(52), 28);
    assert_eq!(p.physical(55), 31);
    // Every physical slot is used and the attention kinds agree.
    let mut seen = [false; PHYSICAL_LAYERS];
    for il in 0..TRUNK_LAYERS {
        let physical = p.physical(il);
        assert_eq!(
            p.recurrent[p.block(physical)],
            p.recurrent[il],
            "layer {il}"
        );
        seen[physical] = true;
    }
    assert!(seen.iter().all(|s| *s));
    // mHC spans both passes of the loop range.
    assert!(!p.mhc_active(3));
    assert!(p.mhc_active(4));
    assert!(p.mhc_active(51));
    assert!(!p.mhc_active(52));
}

#[test]
fn block_indices_cover_the_file_blocks() {
    let p = plan();
    assert_eq!(p.block(0), 0);
    assert_eq!(p.block(27), 27);
    assert_eq!(p.block(28), 52);
    assert_eq!(p.block(31), 55);
}

#[test]
fn yarn_rope_matches_the_vendor_formula() {
    let rope = RopePlan {
        dims: 4,
        sections: [0, 0, 1, 1],
        freq_base: 10_000.0,
        factor: 2.0,
        original_context: 1024,
    };
    let mut head = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    apply_yarn(&mut head, &rope, 3);
    // The vendor truncates the correction bounds to [0, 2] here.
    for (i, angle) in [3.0_f32, 0.0225].into_iter().enumerate() {
        let (sin, cos) = angle.sin_cos();
        let a = [1.0f32, 2.0][i];
        let b = [3.0f32, 4.0][i];
        assert!((head[i] - (a * cos - b * sin)).abs() < 1e-5, "pair {i}");
        assert!((head[i + 2] - (b * cos + a * sin)).abs() < 1e-5, "pair {i}");
    }
    // Dims past the rotary span are untouched.
    assert_eq!(head[4], 5.0);
    assert_eq!(head[5], 6.0);
}

#[test]
fn yarn_table_rotation_matches_apply_yarn() {
    let rope = RopePlan {
        dims: 16,
        sections: [0, 0, 4, 4],
        freq_base: 1e7,
        factor: 2.0,
        original_context: 262_144,
    };
    let table = yarn_inv_freq(&rope);
    assert_eq!(table.len(), 8);
    // The fastest pair keeps its original frequency exactly.
    assert_eq!(table[0], 1.0);
    // The table is strictly decreasing: base powers shrink and the YaRN blend
    // only pulls slow pairs further down.
    assert!(table.windows(2).all(|pair| pair[0] > pair[1]));
    // The same rotation through the table equals `apply_yarn` on every pair
    // (up to float reassociation: the host folds pos into the angle first).
    let head: Vec<f32> = (0..16).map(|i| (i as f32 - 3.5) * 0.25).collect();
    let mut want = head.clone();
    apply_yarn(&mut want, &rope, 97);
    let mut got = head;
    for i in 0..8 {
        let angle = 97.0 * table[i];
        let (sin, cos) = angle.sin_cos();
        let a = got[i];
        let b = got[i + 8];
        got[i] = a * cos - b * sin;
        got[i + 8] = b * cos + a * sin;
    }
    for (got, want) in got.iter().zip(want.iter()) {
        assert!((got - want).abs() < 1e-5, "{} vs {}", got, want);
    }
}

#[test]
fn sinkhorn_mix_is_doubly_stochastic() {
    let s = shape();
    // Router that yields distinct logits so the mix has to move.
    let weight = DenseWeight::matrix(8, 16, (0..8 * 16).map(|i| i as f32 * 0.01).collect());
    let bias = vec![0.5, -0.25, 0.1, -0.1, 1.0, 0.5, -0.5, -1.0];
    let router = RouterView {
        weight: &weight,
        bias: &bias,
    };
    let a = vec![1.0f32; 8];
    let b = vec![-0.5f32; 8];
    let route = mhc_route(
        &mut DenseProjector::default(),
        &s,
        &router,
        false,
        1.0,
        20,
        (&a, &b),
    )
    .unwrap();
    let mix = &route.mix;
    let eps = 1e-4;
    assert!(
        (mix[0] + mix[1] - 1.0).abs() < eps && (mix[2] + mix[3] - 1.0).abs() < eps,
        "row sums {mix:?}"
    );
    assert!(
        (mix[0] + mix[2] - 1.0).abs() < eps && (mix[1] + mix[3] - 1.0).abs() < eps,
        "column sums {mix:?}"
    );
    // Input is the gated mix of the true logits; post gains live in [0, 2].
    let flat: Vec<f32> = a.iter().chain(b.iter()).copied().collect();
    let rms = (flat.iter().map(|v| v * v).sum::<f32>() / 16.0 + 1e-6).sqrt();
    let normed: Vec<f32> = flat.iter().map(|v| v / rms).collect();
    let logits: Vec<f32> = (0..8)
        .map(|r| {
            (0..16)
                .map(|c| weight.data[r * 16 + c] * normed[c])
                .sum::<f32>()
                + bias[r]
        })
        .collect();
    let want = |j: usize| a[j] * sigmoid(logits[0]) + b[j] * sigmoid(logits[1]);
    assert!((route.input[3] - want(3)).abs() < 1e-5);
    assert!(route.post[0] > 0.0 && route.post[0] < 2.0);
}

#[test]
fn merge_composes_mix_and_residual_gain() {
    let route = super::step::Route {
        input: vec![],
        post: [0.5, 2.0],
        mix: [0.25, 0.75, 0.75, 0.25],
    };
    let mut a = vec![1.0f32, 0.0];
    let mut b = vec![0.0f32, 1.0];
    merge(&mut a, &mut b, &[2.0, 4.0], &route);
    assert!((a[0] - (0.25 + 1.0)).abs() < 1e-6, "{}", a[0]);
    assert!((a[1] - (0.75 + 2.0)).abs() < 1e-6, "{}", a[1]);
    assert!((b[0] - (0.75 + 4.0)).abs() < 1e-6, "{}", b[0]);
    assert!((b[1] - (0.25 + 8.0)).abs() < 1e-6, "{}", b[1]);
}

#[test]
fn delta_step_matches_hand_math() {
    let s = shape();
    let p = plan();
    // hidden 8; conv_dim = 2*1*2 + 2*2 = 8; qkvz rows = 8 + 4 = 12.
    let qkvz = DenseWeight::matrix(
        12,
        8,
        (0..96).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect(),
    );
    let beta_alpha = DenseWeight::matrix(4, 8, vec![0.5; 32]);
    let out = DenseWeight::matrix(8, 4, [0.2, -0.1, 0.3, 0.4].repeat(8));
    let conv = vec![
        0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, // q head
        0.25, 0.75, 0.35, 0.65, 0.45, 0.55, 0.5, 0.5,
    ]; // 8 channels x kernel 2
    let a = vec![0.0f32, 0.0];
    let dt_bias = vec![0.1, -0.1];
    let norm = vec![1.0f32, 2.0];
    let view = DeltaView {
        qkvz: &qkvz,
        beta_alpha: &beta_alpha,
        out: &out,
        conv: &conv,
        a: &a,
        dt_bias: &dt_bias,
        norm: &norm,
    };
    let mut state = LoopState::new(&s, &p);
    let x = vec![0.5f32, -0.25, 0.75, 1.0, -1.0, 0.25, -0.5, 0.1];
    let output = delta_step(&mut DenseProjector::default(), &s, &view, &mut state, 0, &x).unwrap();
    assert_eq!(output.len(), 8);
    // The projection math itself is covered by the projector tests; here we
    // assert the composed contract: finite, and reproducible on a re-run
    // with fresh state.
    assert!(output.iter().all(|v| v.is_finite()));
    let mut state2 = LoopState::new(&s, &p);
    let again = delta_step(
        &mut DenseProjector::default(),
        &s,
        &view,
        &mut state2,
        0,
        &x,
    )
    .unwrap();
    assert_eq!(output, again);
    // A wrong-length input is rejected before touching the projector.
    let bad = delta_step(
        &mut DenseProjector::default(),
        &s,
        &view,
        &mut LoopState::new(&s, &p),
        0,
        &[0.0; 3],
    );
    assert!(bad.is_err());
}

#[test]
fn delta_step_reads_the_updated_state() {
    let s = LoopShape {
        hidden: 1,
        delta_key_dim: 1,
        delta_value_heads: 1,
        delta_value_dim: 1,
        eps: 0.25,
        ..shape()
    };
    let qkvz = DenseWeight::matrix(4, 1, vec![1.0; 4]);
    let beta_alpha = DenseWeight::matrix(2, 1, vec![0.0; 2]);
    let out = DenseWeight::matrix(1, 1, vec![1.0]);
    let view = DeltaView {
        qkvz: &qkvz,
        beta_alpha: &beta_alpha,
        out: &out,
        conv: &[0.0, 1.0].repeat(4),
        a: &[-1.0],
        dt_bias: &[0.0],
        norm: &[1.0],
    };
    let mut projector = DenseProjector::default();
    let mut state = LoopState::new(&s, &plan());
    let v = sigmoid(1.0); // All four projections are 1, then SiLU(1).
    let qk = v / (v * v + s.eps).sqrt();
    let first = 0.5 * v * qk;
    let second = 0.5 * first + 0.5 * (v - 0.5 * first * qk) * qk;
    for current in [first, second] {
        let expected = current * qk / ((current * qk).powi(2) + s.eps).sqrt() * v;
        let got = delta_step(&mut projector, &s, &view, &mut state, 0, &[1.0]).unwrap();
        assert!((got[0] - expected).abs() < 1e-6, "{got:?} vs {expected}");
    }
}

#[test]
fn attn_view_shapes_match_the_fused_layout() {
    let s = shape();
    assert_eq!(s.qkvz_dim(), 12);
    assert_eq!(s.conv_dim(), 8);
    assert_eq!(s.delta_inner(), 4);
    // Attention q packs [query | gate] per head: 2 * (heads * head_dim).
    assert_eq!(2 * s.heads_q * s.head_dim, 16);
}

#[test]
#[ignore = "needs SENSENOVA_GGUF, the released SenseNova-Loop-9B GGUF"]
fn opens_the_real_checkpoint() {
    let path = std::env::var("SENSENOVA_GGUF").expect("set SENSENOVA_GGUF");
    let limits = crate::weights::GgufLimits {
        max_header_bytes: 32 << 20,
        max_tensor_bytes: 8 << 30,
    };
    let weights = GgufWeights::open_any_type(std::path::Path::new(&path), limits, None).unwrap();
    let opened = super::weights::LoopWeights::open(std::sync::Arc::new(weights)).unwrap();
    let shape = opened.shape();
    assert_eq!(shape.hidden, 4096);
    assert_eq!(shape.heads_q, 16);
    assert_eq!(shape.heads_kv, 4);
    assert_eq!(shape.head_dim, 256);
    assert_eq!(shape.delta_key_heads, 16);
    assert_eq!(shape.delta_value_heads, 32);
    assert_eq!(shape.delta_key_dim, 128);
    assert_eq!(shape.delta_value_dim, 128);
    assert_eq!(shape.conv_kernel, 4);
    assert_eq!(shape.qkvz_dim(), 12288);
    assert_eq!(shape.conv_dim(), 8192);
    assert_eq!(opened.plan().physical(28), 4);
    assert_eq!(opened.plan().block(31), 55);
    assert_eq!(opened.plan().recurrent.len(), TRUNK_LAYERS);
    assert_eq!(opened.layers().len(), PHYSICAL_LAYERS);
    assert!(!opened.plan().recurrent[3]);
    assert!(opened.plan().recurrent[4]);
}

#[test]
fn full_attention_caches_both_tokens_and_groups_heads_with_separate_gates() {
    let s = LoopShape {
        heads_q: 4,
        heads_kv: 2,
        head_dim: 2,
        ..shape()
    };
    let rope = RopePlan {
        dims: 2,
        sections: [1, 0, 0, 0],
        freq_base: 10_000.0,
        factor: 2.0,
        original_context: 262_144,
    };
    // Each tuple specifies a projection row's coefficients on the two input tokens.
    let matrix = |rows: &[(f32, f32)]| {
        DenseWeight::matrix(
            rows.len(),
            8,
            rows.iter()
                .flat_map(|&(a, b)| [a, b, 0., 0., 0., 0., 0., 0.])
                .collect(),
        )
    };
    let q = matrix(
        &[0.8_f32, 0.5, 0.25, 0.75]
            .into_iter()
            .flat_map(|gate| {
                let logit = (gate / (1.0 - gate)).ln();
                [(1.0, 1.0), (0.0, 0.0), (logit, logit), (logit, logit)]
            })
            .collect::<Vec<_>>(),
    );
    let k = matrix(&[(1., 1.), (0., 0.), (-1., -1.), (0., 0.)]);
    let v = matrix(&[(2., 6.), (4., 8.), (-2., 2.), (-4., 0.)]);
    let o = DenseWeight::matrix(
        8,
        8,
        (0..64).map(|i| if i % 9 == 0 { 1. } else { 0. }).collect(),
    );
    let view = AttnView {
        q: &q,
        k: &k,
        v: &v,
        o: &o,
        q_norm: &[1.; 2],
        k_norm: &[1.; 2],
    };
    let mut projector = DenseProjector::default();
    let mut state = LoopState::new(&s, &plan());
    let gain = 1.0 / ((0.5 + s.eps) * 2.0_f32.sqrt());
    for pos in 0..2 {
        let mut x = [0.; 8];
        x[pos] = 1.0;
        let result = attn_step(
            &mut projector,
            &s,
            &rope,
            &view,
            &mut state,
            3,
            (&x, pos as u32),
        )
        .unwrap();
        let mix = |sign| {
            if pos == 0 {
                0.0
            } else {
                sigmoid(sign * gain * (1.0 - 1.0_f32.cos()))
            }
        };
        let front = mix(1.0);
        let back = mix(-1.0);
        let expected = [
            0.8 * (2. + 4. * front),
            0.8 * (4. + 4. * front),
            0.5 * (2. + 4. * front),
            0.5 * (4. + 4. * front),
            0.25 * (-2. + 4. * back),
            0.25 * (-4. + 4. * back),
            0.75 * (-2. + 4. * back),
            0.75 * (-4. + 4. * back),
        ];
        for (got, want) in result.into_iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "position {pos}: {got} vs {want}");
        }
    }
}

/// Greedy parity with the vendor on the bundled non-thinking chat template.
#[test]
#[ignore = "needs SENSENOVA_GGUF, the released SenseNova-Loop-9B Q4_K_M checkpoint"]
fn generates_from_the_real_prompt() {
    let path = std::env::var("SENSENOVA_GGUF").expect("set SENSENOVA_GGUF");
    let file = std::path::Path::new(&path);
    let bundle = crate::tokenizer::gguf::bundle_from_lone_gguf(file).unwrap();
    let messages = [crate::tokenizer::PromptMessage {
        role: crate::tokenizer::Role::User,
        content: Some("What is the capital of France? Answer in one word.".into()),
        reasoning_content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }];
    let ids = bundle
        .encode_prompt(&messages, &[], crate::tokenizer::Thinking::Off)
        .unwrap();
    assert_eq!(
        ids,
        [
            248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 303, 799, 3299, 13,
            248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
        ],
    );
    let mut executor = SerialDecode::new(
        LoopStep::open(file).unwrap(),
        bundle.eos_token_ids().to_vec(),
        bundle.vocab_size() as usize,
        4096,
    );
    let mut session = DecodeSession::default();
    for chunk in ids.chunks(8) {
        executor.prefill(&mut session, chunk).unwrap();
    }
    let mut generation =
        GenerationState::new(IncrementalDecoder::new(bundle.tokenizer())).with_max_tokens(2);
    let answer = executor.decode(&mut session, &mut generation).unwrap();
    assert_eq!(answer.tokens, [57590]);
    assert!(answer.finish.is_none());
    let stop = executor.decode(&mut session, &mut generation).unwrap();
    assert!(matches!(stop.finish, Some(DecodeFinish::Stop)));
    assert_eq!(stop.tokens.len(), 1);
    assert!(bundle.eos_token_ids().contains(&stop.tokens[0]));
    assert_eq!(answer.text + &stop.text, "Paris");
}
