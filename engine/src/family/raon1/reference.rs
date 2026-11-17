//! Host reference numerics for Raon-1.
//!
//! Scalar numerics: the hyperparameter reader, the checkpoint's quantized
//! dot (`fork_mm`), the activation round-trip quantizers, and the RDA
//! attention fold over dequantized tensors read from the checkpoint opener.
//! The graph-walk commit composes these into a forward pass; this one pins
//! the numeric contracts that are easy to get wrong, commented where they
//! are used.

use crate::EngError;
use crate::family::raon1::{Raon1Model, Raon1Spec};
use crate::weights::GgufType;
use crate::weights::MetaValue;

/// Hyperparameters the fork's raon1 loader reads from the checkpoint header.
/// Read, not hardcoded: the opener pins geometry only.
pub struct Hyper {
    pub streams: usize,
    pub norm_eps: f32,
    pub latent_eps: f32,
    pub hc_eps: f32,
    pub hc_magnitude: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub used_experts: usize,
    pub weights_norm: bool,
    pub gate_lower_bound: f32,
    pub rope_dim: usize,
}

/// The smallest weight sum `weights_norm` divides by, clamped the way ggml
/// does (smallest positive normal F16).
pub const WEIGHT_SUM_MIN: f32 = 6.103515_6e-5; // 2^-14, smallest positive normal F16

fn reject(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("raon1 reference: {}", message.into()))
}

fn meta_f32(model: &Raon1Model, key: &str) -> Result<f32, EngError> {
    match model.header().meta(key) {
        Some(MetaValue::F32(v)) => Ok(*v),
        _ => Err(reject(format!("metadata key {key} is missing or not f32"))),
    }
}

fn meta_u32(model: &Raon1Model, key: &str) -> Result<u32, EngError> {
    match model.header().meta(key) {
        Some(MetaValue::U32(v)) => Ok(*v),
        _ => Err(reject(format!("metadata key {key} is missing or not u32"))),
    }
}

fn meta_bool(model: &Raon1Model, key: &str) -> Result<bool, EngError> {
    match model.header().meta(key) {
        Some(MetaValue::Bool(v)) => Ok(*v),
        _ => Err(reject(format!("metadata key {key} is missing or not bool"))),
    }
}

/// Reads the Raon-1 hyperparameters the reference needs, rejecting values the
/// reference does not implement (gating func 2 = sigmoid, weight scale 1.0).
pub fn hyper(model: &Raon1Model, spec: Raon1Spec) -> Result<Hyper, EngError> {
    if meta_u32(model, "raon1.expert_gating_func")? != 2 {
        return Err(reject("expert_gating_func is not 2 (sigmoid)"));
    }
    let scale = meta_f32(model, "raon1.expert_weights_scale")?;
    if scale != 1.0 {
        return Err(reject(format!("expert_weights_scale {scale} is not 1.0")));
    }
    let rope_dim = meta_u32(model, "raon1.rope.dimension_count")? as usize;
    // Geometry the MLA path indexes silently if it disagrees: the shared KV
    // projection must be exactly the latent plus the rope dims, and the
    // per-head absorbed key width the key width minus the rope part.
    if u64::try_from(rope_dim).is_err() || spec.kv_a_width != spec.kv_latent + rope_dim as u64 {
        return Err(reject(format!(
            "rope.dimension_count {rope_dim} disagrees with kv_a_width {} minus kv_latent {}",
            spec.kv_a_width, spec.kv_latent
        )));
    }
    if spec.k_head != spec.key_mla - rope_dim as u64 {
        return Err(reject(format!(
            "k_head {} disagrees with key_mla {} minus rope_dim {rope_dim}",
            spec.k_head, spec.key_mla
        )));
    }
    // hc_site reads scale[0..=3]; fewer streams would slice out of range.
    if (spec.streams as usize) < 4 {
        return Err(reject(format!(
            "hyper_connection.count {} is below the 4 streams hc_site indexes",
            spec.streams
        )));
    }
    Ok(Hyper {
        streams: spec.streams as usize,
        norm_eps: meta_f32(model, "raon1.attention.layer_norm_rms_epsilon")?,
        latent_eps: meta_f32(model, "raon1.attention.latent_norm_rms_epsilon")?,
        hc_eps: meta_f32(model, "raon1.hyper_connection.epsilon")?,
        hc_magnitude: meta_f32(model, "raon1.hyper_connection.magnitude")?,
        situ_beta: meta_f32(model, "raon1.activation.situ_beta")?,
        situ_linear_beta: meta_f32(model, "raon1.activation.situ_linear_beta")?,
        used_experts: meta_u32(model, "raon1.expert_used_count")? as usize,
        weights_norm: meta_bool(model, "raon1.expert_weights_norm")?,
        gate_lower_bound: meta_f32(model, "raon1.kda.gate_lower_bound")?,
        rope_dim: meta_u32(model, "raon1.rope.dimension_count")? as usize,
    })
}

/// `y[o, t] = sum_k w[o, k] * x[k, t]`, ggml layout (fastest dim first) for
/// both operands and result. This is `ggml_mul_mat(w, x)` in the fork.
/// Accumulation is F64: the fork's F32 SIMD dot carries ~1e-6-of-range noise
/// on cancellation-heavy rows, and the exact value is the better oracle.
fn mm(w: &[f32], x: &[f32], k: usize, n: usize, nt: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * nt];
    for t in 0..nt {
        let xc = &x[t * k..(t + 1) * k];
        for o in 0..n {
            let wrow = &w[o * k..(o + 1) * k];
            let mut acc = 0.0f64;
            for i in 0..k {
                acc += wrow[i] as f64 * xc[i] as f64;
            }
            y[o + t * n] = acc as f32;
        }
    }
    y
}

/// Single-column `mm` (one token), the MoE path: `out[o] = w[o, :] . x`.
fn mv(w: &[f32], x: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (o, slot) in out.iter_mut().enumerate() {
        let wrow = &w[o * k..(o + 1) * k];
        let mut acc = 0.0f64;
        for i in 0..k {
            acc += wrow[i] as f64 * x[i] as f64;
        }
        *slot = acc as f32;
    }
    out
}

/// `nearest_int` as ggml-quants.c defines it: the 12582912.0f bit trick,
/// round-half-even in the mantissa it lands on.
fn ggml_nearest_int(f: f32) -> i32 {
    (f32::to_bits(f + 12582912.0) & 0x007f_ffff) as i32 - 0x0040_0000
}

/// f32 -> f16 -> f32 through the `half` crate (round-to-nearest-even), the
/// rounding `block_q8_0` applies to its scale and the F16 vec-dot applies to
/// an F32 activation.
pub(crate) fn f16_roundtrip(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

/// Roundtrip-quantizes one contraction-length activation column the way the
/// fork's CPU mul_mat does before a quantized dot: Q8_K blocks of 256 for the
/// K-quants, Q8_0 blocks of 32 for the plain quants, an F16 roundtrip for F16
/// weights (the CPU path's vec_dot_type for F16 is F16). Q8_K follows
/// `quantize_row_q8_K_ref`; Q8_0 follows `quantize_row_q8_0_ref`, including
/// its detail that the codes use the reciprocal of the unrounded scale while
/// the stored (f16-rounded) scale dequantizes them.
fn quantize_activation_column(ty: GgufType, x: &[f32]) -> Vec<f32> {
    let mut out = match ty {
        GgufType::F16 => x.iter().map(|&v| f16_roundtrip(v)).collect(),
        _ => x.to_vec(),
    };
    match ty {
        GgufType::Q2K | GgufType::Q3K | GgufType::Q4K | GgufType::Q5K | GgufType::Q6K => {
            assert_eq!(x.len() % 256, 0, "Q8_K needs whole 256 blocks");
            let qkk = 256; // Q8_K block size
            for blk in out.chunks_exact_mut(qkk) {
                let (mut max, mut amax) = (0.0f32, 0.0f32);
                for &v in blk.iter() {
                    let ax = v.abs();
                    if ax > amax {
                        amax = ax;
                        max = v;
                    }
                }
                if amax == 0.0 {
                    blk.fill(0.0);
                    continue;
                }
                let iscale = -127.0 / max;
                for v in blk.iter_mut() {
                    let q = ggml_nearest_int(iscale * *v).min(127);
                    *v = q as f32 / iscale;
                }
            }
        }
        GgufType::Q40 | GgufType::Q41 | GgufType::Q50 | GgufType::Q51 | GgufType::Q80 => {
            assert_eq!(x.len() % 32, 0, "Q8_0 needs whole 32 blocks");
            let qk8 = 32; // Q8_0 block size
            for blk in out.chunks_exact_mut(qk8) {
                let mut amax = 0.0f32;
                for &v in blk.iter() {
                    amax = amax.max(v.abs());
                }
                let d = amax / 127.0;
                if d == 0.0 {
                    blk.fill(0.0);
                    continue;
                }
                // ggml takes the reciprocal of the unrounded d for the codes
                // and rounds d to f16 only for the stored scale; taking it
                // from the rounded scale mis-rounds borderline elements by
                // one code step.
                let dr = f16_roundtrip(d);
                let id = 1.0 / d;
                for v in blk.iter_mut() {
                    *v = dr * (*v * id).round();
                }
            }
        }
        _ => {}
    }
    out
}

/// `mm` against a checkpoint tensor, reproducing the fork's CPU mul_mat
/// numerics: the activation is roundtrip-quantized to the weight's vec-dot
/// type first. Measured against the fork's `fixtures1` stage dumps: the
/// quantized dot matches the fork to ~1e-7 of stage range on Q6_K weights
/// (`the_dense_ffn_gate_element_pins_the_quantized_dot` in `tests.rs`).
pub fn fork_mm(
    model: &Raon1Model,
    name: &str,
    x: &[f32],
    k: usize,
    n: usize,
    nt: usize,
) -> Result<Vec<f32>, EngError> {
    let (w, ty) = model.read_f32_typed(name)?;
    let mut a = Vec::with_capacity(x.len());
    for t in 0..nt {
        let col = &x[t * k..(t + 1) * k];
        a.extend(quantize_activation_column(ty, col));
    }
    Ok(mm(&w, &a, k, n, nt))
}

/// Single-token `fork_mm` against an already-read expert weight slice.
pub fn fork_mv(w: &[f32], ty: GgufType, x: &[f32], k: usize, n: usize) -> Vec<f32> {
    let a = quantize_activation_column(ty, x);
    mv(w, &a, k, n)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softplus(x: f32) -> f32 {
    // ggml's softplus: logf(1 + expf(x)), overflow semantics included: x > 88
    // overflows to inf, which propagates through the RDA fold as NaN. Kept
    // for parity; a numerically stable form belongs to the kernel port, not
    // to this oracle.
    (1.0 + x.exp()).ln()
}

/// SiTU: `tanh(gate/beta)*beta * sigmoid(gate)`, then the up projection
/// passes through `tanh(up/lb)*lb` when `lb > 0` (fork `build_situ`).
pub fn situ(gate: &[f32], up: &[f32], beta: f32, lb: f32) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| {
            let act = (g / beta).tanh() * beta * sigmoid(g);
            let upq = if lb > 0.0 { (u / lb).tanh() * lb } else { u };
            act * upq
        })
        .collect()
}

/// `x / sqrt(mean(x^2) + eps)` over columns of `d` elements, scaled by `w`
/// (`ggml_rms_norm` + mul, fork `build_norm` RMS).
fn rms_norm_scaled(x: &[f32], w: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for (col, yc) in y.chunks_exact_mut(d).enumerate() {
        let xc = &x[col * d..(col + 1) * d];
        let mut sum = 0.0f64;
        for &v in xc {
            sum += v as f64 * v as f64;
        }
        let scale = 1.0 / (sum / d as f64 + eps as f64).sqrt() as f32;
        for (slot, (&v, &wv)) in yc.iter_mut().zip(xc.iter().zip(w)) {
            *slot = v * scale * wv;
        }
    }
    y
}

/// `x / max(||x||, eps)` over columns of `d` elements (fork `ggml_l2_norm`).
fn l2_norm_cols(x: &mut [f32], d: usize, eps: f32) {
    for col in x.chunks_exact_mut(d) {
        let mut sum = 0.0f64;
        for &v in &*col {
            sum += v as f64 * v as f64;
        }
        let scale = 1.0 / sum.sqrt().max(eps as f64) as f32;
        for slot in col.iter_mut() {
            *slot *= scale;
        }
    }
}

/// `build_rda_conv` for one q/k/v third. `x` is `[d_inner, nt]`.
pub fn rda_conv(
    model: &Raon1Model,
    layer: usize,
    third: &str,
    x: &[f32],
    din: usize,
) -> Result<Vec<f32>, EngError> {
    let nt = x.len() / din;
    let w = model.read_f32(&format!("blk.{layer}.ssm_conv1d_{third}.weight"))?;
    let mut y = vec![0.0f32; x.len()];
    for c in 0..din {
        // Weight layout [conv_kernel, 1, d_inner]: tap j of channel c at c*4 + j.
        let taps = [w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]];
        for t in 0..nt {
            let tap = |back: usize| -> f32 {
                if t < back {
                    0.0
                } else {
                    x[c + (t - back) * din]
                }
            };
            let acc = taps[0] * tap(3) + taps[1] * tap(2) + taps[2] * tap(1) + taps[3] * tap(0);
            y[c + t * din] = silu(acc);
        }
    }
    Ok(y)
}

/// The RDA (gated delta-rule) layer with KRAFTON's per-channel decay.
#[allow(clippy::too_many_arguments)]
pub fn rda_layer(
    model: &Raon1Model,
    spec: Raon1Spec,
    hy: &Hyper,
    cur: &[f32],
    layer: usize,
    emit: &mut dyn FnMut(&str, &[f32]),
) -> Result<Vec<f32>, EngError> {
    let h = spec.hidden as usize;
    let heads = spec.heads as usize;
    let hd = spec.head_dim as usize;
    let din = heads * hd;
    let nt = cur.len() / h;

    let w = |name: &str| model.read_f32(&format!("blk.{layer}.{name}"));
    let xq = fork_mm(
        model,
        &format!("blk.{layer}.attn_q.weight"),
        cur,
        h,
        din,
        nt,
    )?;
    let xk = fork_mm(
        model,
        &format!("blk.{layer}.attn_k.weight"),
        cur,
        h,
        din,
        nt,
    )?;
    let xv = fork_mm(
        model,
        &format!("blk.{layer}.attn_v.weight"),
        cur,
        h,
        din,
        nt,
    )?;
    let mut qc = rda_conv(model, layer, "q", &xq, din)?;
    let mut kc = rda_conv(model, layer, "k", &xk, din)?;
    let vc = rda_conv(model, layer, "v", &xv, din)?;
    emit(&format!("rda_q_conv-{layer}"), &qc);
    emit(&format!("rda_k_conv-{layer}"), &kc);
    emit(&format!("rda_v_conv-{layer}"), &vc);

    // g1 = lower_bound * sigmoid(-(A * (f_b(f_a(cur)) + dt_b))), A per head.
    let tsr = spec.time_step_rank as usize;
    let fa = fork_mm(
        model,
        &format!("blk.{layer}.ssm_f_a.weight"),
        cur,
        h,
        tsr,
        nt,
    )?;
    let mut g1 = fork_mm(
        model,
        &format!("blk.{layer}.ssm_f_b.weight"),
        &fa,
        tsr,
        din,
        nt,
    )?;
    let dt_b = w("ssm_dt.bias")?;
    let ssm_a = w("ssm_a")?;
    for t in 0..nt {
        for c in 0..din {
            let idx = c + t * din;
            let raw = (g1[idx] + dt_b[c]) * ssm_a[c / hd];
            g1[idx] = hy.gate_lower_bound * sigmoid(-raw);
        }
    }

    // Erase/write gates fold into V; the scan's beta is sigmoid(beta).
    let beta = fork_mm(
        model,
        &format!("blk.{layer}.ssm_beta.weight"),
        cur,
        h,
        heads,
        nt,
    )?;
    let delta = fork_mm(
        model,
        &format!("blk.{layer}.ssm_rda.weight"),
        cur,
        h,
        heads,
        nt,
    )?;
    let mut vs = vc.clone();
    for t in 0..nt {
        for head in 0..heads {
            let bi = beta[head + t * heads];
            let de = bi + delta[head + t * heads];
            let fold = (softplus(-bi) - softplus(-de)).exp();
            for d in 0..hd {
                vs[d + head * hd + t * din] *= fold;
            }
        }
    }
    emit(&format!("rda_v_scaled-{layer}"), &vs);
    emit(&format!("rda_g1-{layer}"), &g1);
    let bscan: Vec<f32> = beta.iter().map(|&b| sigmoid(b)).collect();
    emit(&format!("rda_beta-{layer}"), &bscan);

    l2_norm_cols(&mut qc, hd, 1e-6);
    l2_norm_cols(&mut kc, hd, 1e-6);
    // The 1e-6 l2 eps is the fork's hard-coded value (raon1.cpp's F.normalize
    // eps), not a checkpoint field; nothing here reads it from the header.

    // Sequential delta-rule scan (the fork chunks at 16, same law):
    // per-channel decay on the k-index rows, delta write k⊗δ, then the
    // read q^T·S after the write — exactly the fork's GDN kernel order.
    // The state matrix s is f32 on purpose: it matches the fork's state
    // dtype, so scan divergence measured here is what the kernel port
    // would see too.
    let qs = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; din * nt];
    for head in 0..heads {
        // s[u + w*hd] = S[u][w]: u is the k-dim index, w the v-dim index.
        let mut s = vec![0.0f32; hd * hd];
        let mut dvec = vec![0.0f32; hd];
        for t in 0..nt {
            for u in 0..hd {
                let decay = g1[u + head * hd + t * din].exp();
                for w in 0..hd {
                    s[u + w * hd] *= decay;
                }
            }
            for w in 0..hd {
                let mut sk = 0.0f64;
                for u in 0..hd {
                    sk += s[u + w * hd] as f64 * kc[u + head * hd + t * din] as f64;
                }
                dvec[w] = ((vs[w + head * hd + t * din] as f64 - sk)
                    * bscan[head + t * heads] as f64) as f32;
            }
            for u in 0..hd {
                let ku = kc[u + head * hd + t * din];
                for w in 0..hd {
                    s[u + w * hd] += ku * dvec[w];
                }
            }
            for w in 0..hd {
                let mut acc = 0.0f64;
                for u in 0..hd {
                    acc += s[u + w * hd] as f64 * qc[u + head * hd + t * din] as f64;
                }
                out[w + head * hd + t * din] = (acc * qs as f64) as f32;
            }
        }
    }
    emit(&format!("rda_scan_out-{layer}"), &out);

    let o_norm = rms_norm_scaled(&out, &w("ssm_norm.weight")?, hd, hy.norm_eps);
    emit(&format!("rda_normed-{layer}"), &o_norm);
    let g2 = fork_mm(model, &format!("blk.{layer}.ssm_g.weight"), cur, h, din, nt)?;
    emit(&format!("rda_g2-{layer}"), &g2);
    let mut gated = vec![0.0f32; din * nt];
    for (slot, (&o, &g)) in gated.iter_mut().zip(o_norm.iter().zip(&g2)) {
        *slot = o * sigmoid(g);
    }
    let out_mm = fork_mm(
        model,
        &format!("blk.{layer}.attn_output.weight"),
        &gated,
        din,
        h,
        nt,
    )?;
    emit(&format!("rda_out-{layer}"), &out_mm);
    Ok(out_mm)
}
