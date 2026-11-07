//! The rest of layer 3 of the real Qwen3.8 Flash-Next 3.05 bpw pack on the host: the QSA attention
//! block and both hyper-connection sites (the 58,230,548 bytes that the MoE tests of
//! `real_layer_tests` do not read). `FIFI_FLASH_NEXT_EXL3_PACK` names a local pack directory that
//! holds the headers of every shard and the payload of layer 3; `FIFI_FLASH_NEXT_EXL3_MOE_LAYER`
//! picks another QSA layer. No GPU.
//!
//! - Every tensor of the range loads: a matrix through `Matrix::load` at the rate the header walk
//!   found, a plain tensor with finite values.
//! - The CPU kernel projects the five EXL3 attention matrices like the fp64 oracle that evaluates the
//!   codebook exactly, and sits one binary16 rounding away from the oracle the device kernels follow.
//! - The host QSA block runs ten positions on the real weights; its output is the reference the
//!   CUDA step is compared with.
//! - The host hyper-connection read and apply agree with an independent fp64 reading of the same
//!   stored weights at both sites.

use std::time::Duration;

use super::{
    super::{
        hyper::{self, HyperShape, HyperWeights},
        qsa::{self, QsaCache, QsaWeights},
        stored::host_f32,
    },
    real_layer_tests::{input, matrix, open, relative_l2, relative_l2_f64, widened},
};
use crate::{
    family::{
        flash_next::load::model_shape,
        project::{DenseProjector, DenseWeight},
    },
    kernels::{
        exl3_cpu::{Exl3Isa, Exl3Projection, ExpertCpu},
        nvfp4_cpu::WaitPolicy,
    },
    weights::exl3::Matrix,
};

/// Relative L2 bound of the CPU kernel against the exact fp64 oracle on one projection (f32
/// accumulation over at most 6,144 terms).
const EXACT_BOUND: f64 = 1e-5;

/// Rounding every codebook weight to binary16 once costs one projection a relative rms error of
/// `2^-11 * sqrt(1/6)` (`real_layer_tests::expected_rounding_error` derives it; an expert chains
/// three projections and multiplies by `sqrt(3)`, a single projection does not).
fn expected_single_projection_rounding() -> f64 {
    (1.0 / 2048.0) * (1.0f64 / 6.0).sqrt()
}

/// The names of layer `layer`'s attention matrices, with the rate the pack stores them at (twice the
/// bits per weight): the gated query, key, value and output projections at 5 bpw and the
/// block indexer's projection at 3 bpw.
fn attention_matrices(layer: usize) -> [(String, u8); 5] {
    let at = |role: &str| format!("model.language_model.layers.{layer}.self_attn.{role}");
    [
        (at("q_proj"), 10),
        (at("k_proj"), 10),
        (at("v_proj"), 10),
        (at("o_proj"), 10),
        (at("indexer.index_qk_proj"), 6),
    ]
}

/// A reconstructed EXL3 matrix as the `[out, in]` f32 weight of the test projector, from the exact
/// (unrounded) fp64 reconstruction.
fn dense_exact(matrix: &Matrix) -> DenseWeight {
    let [k, n] = matrix.shape();
    let mut wide = vec![0.0f64; k * n];
    matrix
        .reconstruct_fp64_exact_into(&mut wide)
        .expect("the fp64 reconstruction");
    let mut data = vec![0.0f32; k * n];
    for input in 0..k {
        for output in 0..n {
            data[output * k + input] = wide[input * n + output] as f32;
        }
    }
    DenseWeight::matrix(n, k, data)
}

fn dense_plain(
    checkpoint: &crate::weights::exl3::Checkpoint,
    name: &str,
    rows: usize,
    cols: usize,
) -> DenseWeight {
    DenseWeight::matrix(
        rows,
        cols,
        host_f32(&checkpoint.source, name, rows * cols).unwrap_or_else(|e| panic!("{name}: {e:?}")),
    )
}

fn finite_and_bounded(values: &[f32], bound: f32, what: &str) {
    assert!(
        values.iter().all(|v| v.is_finite() && v.abs() <= bound),
        "{what}: a value is not finite or exceeds {bound}"
    );
}

#[test]
#[ignore = "reads the attention block and the hyper-connection sites of layer 3 of the real pack in FIFI_FLASH_NEXT_EXL3_PACK (58 MiB)"]
fn every_tensor_of_the_rest_of_layer_three_loads_at_the_expected_rate() {
    let (checkpoint, layer) = open();
    let (shape, kinds) = model_shape(&checkpoint.model_config).expect("the model shape");
    assert!(kinds[layer], "layer {layer} is not a QSA layer");
    let dims = shape.qsa.dims();

    let mut matrix_bytes = 0usize;
    for (prefix, bits_twice) in attention_matrices(layer) {
        let m = matrix(&checkpoint, &prefix);
        assert_eq!(m.bitrate_twice(), bits_twice, "{prefix}: rate");
        matrix_bytes += m.trellis().host_bytes().expect("trellis").len();
        for scales in [m.suh(), m.svh()] {
            let scales = widened(scales);
            let values: Vec<f32> = scales
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            finite_and_bounded(&values, 1e3, &prefix);
        }
    }
    eprintln!("attention trellis bytes {matrix_bytes}");

    let at = |suffix: &str| format!("model.language_model.layers.{layer}.{suffix}");
    for (suffix, len) in [
        ("self_attn.q_norm.weight", dims.head_dim),
        ("self_attn.k_norm.weight", dims.head_dim),
        ("self_attn.indexer.q_layernorm.weight", dims.idx_dim),
        ("self_attn.indexer.k_layernorm.weight", dims.idx_dim),
    ] {
        let values = host_f32(&checkpoint.source, &at(suffix), len as usize).expect(suffix);
        finite_and_bounded(&values, 100.0, suffix);
    }

    let (width, rank, branches) = (
        shape.hyper.width(),
        shape.hyper.rank(),
        shape.hyper.branches(),
    );
    for site in ["attn_hyper_connection", "mlp_hyper_connection"] {
        for (suffix, len) in [
            ("hc_norm.weight", width),
            ("input_mix_weight_down.weight", rank * width),
            ("input_mix_weight_up.weight", width * rank),
            ("block_inject_weight.weight", branches * width),
        ] {
            let name = at(&format!("{site}.{suffix}"));
            let values = host_f32(&checkpoint.source, &name, len).expect(&name);
            finite_and_bounded(&values, 1e3, &name);
            assert!(values.iter().any(|&v| v != 0.0), "{name} is all zero");
        }
    }
}

#[test]
#[ignore = "reads the attention matrices of layer 3 of the real pack in FIFI_FLASH_NEXT_EXL3_PACK (about 34 MiB) and needs AVX-512"]
fn the_cpu_kernel_projects_the_real_attention_matrices_like_the_exact_oracle() {
    if Exl3Isa::detect() != Exl3Isa::Avx512 {
        eprintln!("skipped: no AVX-512 on this host");
        return;
    }
    let (checkpoint, layer) = open();
    let cpu = ExpertCpu::new(
        4,
        WaitPolicy::SpinThenPark(Duration::from_micros(500)),
        Some(Exl3Isa::Avx512),
    )
    .expect("the CPU kernel");
    let expected = expected_single_projection_rounding();
    for (prefix, _) in attention_matrices(layer) {
        let m = matrix(&checkpoint, &prefix);
        let [k, n] = m.shape();
        let (trellis, suh, svh) = (
            m.trellis().host_bytes().expect("trellis").to_vec(),
            widened(m.suh()),
            widened(m.svh()),
        );
        let projection = Exl3Projection::new(
            m.shape(),
            usize::from(m.bitrate_twice()) * 8,
            &trellis,
            &suh,
            &svh,
        )
        .expect("a CPU projection");
        for seed in 0..2 {
            let x = input(k, seed);
            let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
            let mut out = vec![0.0f32; n];
            cpu.project(&projection, &x, &mut out).expect("project");
            let (mut exact, mut rounded) = (vec![0.0f64; n], vec![0.0f64; n]);
            m.project_fp64_exact_into(&x64, 1, &mut exact)
                .expect("exact oracle");
            m.project_fp64_into(&x64, 1, &mut rounded)
                .expect("rounded oracle");
            let (to_exact, to_rounded) = (relative_l2(&out, &exact), relative_l2(&out, &rounded));
            let oracles = relative_l2_f64(&rounded, &exact);
            eprintln!(
                "{prefix} [{k} x {n}], row {seed}: against the exact oracle {to_exact:.3e}, against \
                 the binary16-rounded oracle {to_rounded:.3e} (oracles differ by {oracles:.3e}; \
                 expected {expected:.3e})"
            );
            assert!(to_exact <= EXACT_BOUND, "{prefix}: {to_exact:.3e}");
            assert!(
                (expected / 2.0..=2.0 * expected).contains(&oracles),
                "{prefix}: the oracles differ by {oracles:.3e}"
            );
            assert!(
                (expected / 2.0..=2.0 * expected).contains(&to_rounded),
                "{prefix}: {to_rounded:.3e} against the rounded oracle"
            );
        }
    }
}

#[test]
#[ignore = "reads the attention block of layer 3 of the real pack in FIFI_FLASH_NEXT_EXL3_PACK (about 34 MiB, reconstructs 60 M weights in fp64)"]
fn the_host_qsa_block_runs_ten_positions_on_the_real_weights() {
    let (checkpoint, layer) = open();
    let (shape, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let dims = shape.qsa.dims();
    let hidden = dims.hidden as usize;
    let at = |suffix: &str| format!("model.language_model.layers.{layer}.self_attn.{suffix}");
    let [q, k, v, o, idx] = attention_matrices(layer).map(|(prefix, _)| {
        let dense = dense_exact(&matrix(&checkpoint, &prefix));
        eprintln!("{prefix}: [{} x {}] reconstructed", dense.rows, dense.cols);
        dense
    });
    let norm = |suffix: &str, len: u32| {
        host_f32(&checkpoint.source, &at(suffix), len as usize).expect(suffix)
    };
    let (q_norm, k_norm) = (
        norm("q_norm.weight", dims.head_dim),
        norm("k_norm.weight", dims.head_dim),
    );
    let (idx_q_norm, idx_k_norm) = (
        norm("indexer.q_layernorm.weight", dims.idx_dim),
        norm("indexer.k_layernorm.weight", dims.idx_dim),
    );
    let weights = QsaWeights {
        q: &q,
        k: &k,
        v: &v,
        o: &o,
        idx: &idx,
        q_norm: &q_norm,
        k_norm: &k_norm,
        idx_q_norm: &idx_q_norm,
        idx_k_norm: &idx_k_norm,
    };
    let mut projector = DenseProjector::default();
    let mut cache = QsaCache::new();
    let mut energy = 0.0f64;
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for pos in 0..10u32 {
        let x = input(hidden, pos as usize);
        let out = qsa::step(&mut projector, &shape.qsa, &weights, &mut cache, pos, &x)
            .expect("the QSA step");
        assert_eq!(out.len(), hidden);
        finite_and_bounded(&out, 1e4, "QSA output");
        let rms = (out.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / hidden as f64).sqrt();
        eprintln!(
            "position {pos}: output rms {rms:.4e}, first {:?}",
            &out[..3]
        );
        assert!(rms > 1e-6, "position {pos}: the block output vanished");
        energy += rms;
        outputs.push(out);
    }
    assert_eq!(cache.len(), 10);
    assert_eq!(projector.calls, 10 * 5);
    eprintln!("sum of output rms over the ten positions {energy:.6e}");
    // The reference values the CUDA block is compared with, measured on the sparse pack's layer 3
    // with the exact fp64 reconstruction (2026-11-07). They pin the block against drift of the
    // host code, the pack range or the test inputs; the tolerance is far above f32 summation noise.
    // At ten positions the indexer holds two completed blocks and the budget takes all of them, so
    // this run does not exercise the ranking.
    assert!(
        (energy - 5.117_429).abs() < 1e-3 * 5.117_429,
        "sum of output rms {energy}"
    );
    for (pos, first) in [
        (0usize, [-0.092_134_11f32, 0.357_506_96, -0.891_631_6]),
        (9, [-1.051_901_6, 0.052_120_093, 0.167_305_6]),
    ] {
        for (got, want) in outputs[pos][..3].iter().zip(first) {
            assert!(
                (got - want).abs() < 1e-3,
                "position {pos}: {got} against the pinned {want}"
            );
        }
    }
}

/// A hyper-connection site read in fp64 from the stored weights, written from the definition and
/// sharing no code with `hyper::read`.
fn fp64_read(
    stream: &[f32],
    site: &Site,
    (branches, hidden, rank): (usize, usize, usize),
    eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    let width = branches * hidden;
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut n = vec![0.0f64; width];
    for b in 0..branches {
        let x = &stream[b * hidden..(b + 1) * hidden];
        let mean = x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / hidden as f64;
        let denom = (mean + eps).sqrt();
        for (d, &value) in x.iter().enumerate() {
            let i = b * hidden + d;
            n[i] = f64::from(value) * (1.0 + f64::from(site.norm[i])) / denom;
        }
    }
    let matvec = |w: &DenseWeight, x: &[f64]| -> Vec<f64> {
        w.data
            .chunks_exact(w.cols)
            .map(|row| row.iter().zip(x).map(|(a, b)| f64::from(*a) * b).sum())
            .collect()
    };
    let low: Vec<f64> = matvec(&site.down, &n)
        .iter()
        .map(|v| {
            let s = v / branches as f64;
            s * sigmoid(s)
        })
        .collect();
    assert_eq!(low.len(), rank);
    let mix = matvec(&site.up, &low);
    let mixed: Vec<f64> = (0..hidden)
        .map(|d| {
            (0..branches)
                .map(|b| sigmoid(mix[b * hidden + d]) * n[b * hidden + d])
                .sum::<f64>()
                / branches as f64
        })
        .collect();
    let gate = matvec(&site.inject, &n)
        .iter()
        .map(|v| 2.0 * sigmoid(v / branches as f64))
        .collect();
    (mixed, gate)
}

struct Site {
    norm: Vec<f32>,
    down: DenseWeight,
    up: DenseWeight,
    inject: DenseWeight,
}

fn site(checkpoint: &crate::weights::exl3::Checkpoint, layer: usize, which: &str) -> Site {
    let (shape, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let (width, rank, branches) = (
        shape.hyper.width(),
        shape.hyper.rank(),
        shape.hyper.branches(),
    );
    let at = |suffix: &str| format!("model.language_model.layers.{layer}.{which}.{suffix}");
    Site {
        norm: host_f32(&checkpoint.source, &at("hc_norm.weight"), width).expect("hc_norm"),
        down: dense_plain(checkpoint, &at("input_mix_weight_down.weight"), rank, width),
        up: dense_plain(checkpoint, &at("input_mix_weight_up.weight"), width, rank),
        inject: dense_plain(
            checkpoint,
            &at("block_inject_weight.weight"),
            branches,
            width,
        ),
    }
}

#[test]
#[ignore = "reads both hyper-connection sites of layer 3 of the real pack in FIFI_FLASH_NEXT_EXL3_PACK (26 MiB)"]
fn the_host_hyper_connection_matches_an_independent_fp64_reading_at_both_sites() {
    let (checkpoint, layer) = open();
    let (shape, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let hyper_shape: HyperShape = shape.hyper;
    let (branches, hidden, rank) = (
        hyper_shape.branches(),
        hyper_shape.hidden(),
        hyper_shape.rank(),
    );
    // Four branches that differ, as the stream does after the first layers.
    let mut stream: Vec<f32> = Vec::with_capacity(hyper_shape.width());
    for b in 0..branches {
        let row = input(hidden, 11 + b);
        stream.extend(row.iter().map(|v| v * (1.0 + 0.5 * b as f32)));
    }
    let mut projector = DenseProjector::default();
    for which in ["attn_hyper_connection", "mlp_hyper_connection"] {
        let s = site(&checkpoint, layer, which);
        let weights = HyperWeights {
            norm: &s.norm,
            down: &s.down,
            up: &s.up,
            inject: Some(&s.inject),
        };
        let read = hyper::read(&mut projector, &hyper_shape, &weights, &stream).expect("read");
        let gate = read.gate.clone().expect("the inject gate");
        let (mixed64, gate64) = fp64_read(
            &stream,
            &s,
            (branches, hidden, rank),
            f64::from(hyper_shape.eps()),
        );
        let to_mixed = relative_l2(&read.mixed, &mixed64);
        let to_gate = relative_l2(&gate, &gate64);
        eprintln!("{which}: mixed {to_mixed:.3e}, gate {to_gate:.3e} against fp64");
        assert!(to_mixed <= 1e-4, "{which}: mixed {to_mixed:.3e}");
        assert!(to_gate <= 1e-5, "{which}: gate {to_gate:.3e}");
        assert!(
            gate.iter().all(|g| *g > 0.0 && *g < 2.0),
            "{which}: the gate {gate:?} leaves (0, 2)"
        );
        finite_and_bounded(&read.mixed, 1e3, which);
        let block: Vec<f32> = read.mixed.iter().map(|v| 0.5 * v).collect();
        let mut after = stream.clone();
        hyper::apply(&hyper_shape, &mut after, &gate, &block).expect("apply");
        for (b, &g) in gate.iter().enumerate() {
            for (d, &value) in block.iter().enumerate() {
                let i = b * hidden + d;
                let want = stream[i] + g * value;
                assert_eq!(after[i].to_bits(), want.to_bits(), "{which} apply {i}");
            }
        }
    }
}
