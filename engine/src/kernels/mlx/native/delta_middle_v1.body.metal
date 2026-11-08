// delta_middle_v1 -- ONE kernel for the whole decode-step DeltaNet pre-recurrence chain of
// resident/delta.rs::step_projected (one slot, B = 1): causal conv (ring + 3 taps + last)
// -> silu -> split q|k|v -> the q/k head norms -> the alpha/beta gates -> the shifted ring.
// The decode step is dispatch-bound at batch 1: this replaces ~15 traced kernels per layer
// (two multiplies, the tap sum, the add, the silu, the kept slice + concat, the three q/k/v
// slices, the two rms_norm kernels, the four-op decay chain and the beta sigmoid) with ONE
// launch. Structure follows gdn_prep_v1 of the fused prefill lane; the head counts are
// template args so the small test shapes run the same kernel.
//
// Rounding mirrors the unfused op sequence: every product and every add is rounded to F32
// separately (fp contract off), the tap sum keeps the (t0 + t1) + t2 order of the MLX sum,
// and sigmoid / logaddexp / exp use the same helpers as MLX's kernels. Only the sum of
// squares over a head is reduced in simdgroup order, like gdn_prep.
//
// Inputs: u_in [CD] activation dtype (qkv projection row), a_in [VH] and b_in [VH] activation
// dtype, ring [CD,3] f32, taps [CD,3] f32, last [CD] f32, neg_exp_a_log [VH] f32,
// dt_bias [VH] f32, and the f32 scalars qk_eps (= KD * 1e-6) and sqrt_kd.
// Outputs: q_out [1,1,KH,KD] f32 (normed, 1/sqrt(KD) scaled), k_out [1,1,KH,KD] f32,
// v_out [1,1,VH,VD] f32, g_out [1,1,VH] f32 (alpha), beta_out [1,1,VH] f32,
// ring_out [CD,3] f32.
// Launch: threadgroup (32,1,1) = one simdgroup per head; grid (heads*32, 1, 1) threads over
// heads = 2*KH + VH: q heads 0..KH-1 and k heads KH..2KH-1 (KD channels each), then VH value
// heads (VD channels). The head channels tile the CD columns of u and y, so every channel is
// convolved exactly once. KD and VD stay <= 128: each lane holds at most 4 elements.
const uint lane = thread_index_in_simdgroup;
const int hd = int(threadgroup_position_in_grid.x);
constexpr int CD = 2 * KH * KD + VH * VD;
const bool is_q = hd < KH;
const bool is_k = (hd >= KH) && (hd < 2 * KH);
const int base = is_q ? hd * KD : is_k ? KH * KD + (hd - KH) * KD
                                       : 2 * KH * KD + (hd - 2 * KH) * VD;
const int width = (hd < 2 * KH) ? KD : VD;
float yv[4];
float ss = 0.0f;
for (int i = int(lane); i < width; i += 32) {
  const int ch = base + i;
  const float x = float(u_in[ch]);
  const float now = last[ch] * x;
  const float p0 = taps[ch * 3 + 0] * ring[ch * 3 + 0];
  const float p1 = taps[ch * 3 + 1] * ring[ch * 3 + 1];
  const float p2 = taps[ch * 3 + 2] * ring[ch * 3 + 2];
  const float val = now + ((p0 + p1) + p2);
  const float y = val * mlx_sigmoid_f(val);
  ring_out[ch * 3 + 0] = ring[ch * 3 + 1];
  ring_out[ch * 3 + 1] = ring[ch * 3 + 2];
  ring_out[ch * 3 + 2] = x;
  if (hd < 2 * KH) {
    yv[i / 32] = y;
    ss += y * y;
  } else {
    v_out[(hd - 2 * KH) * VD + i] = y;
    if (lane == 0) {
      const int j = hd - 2 * KH;
      const float pre = float(a_in[j]) + dt_bias[j];
      const float soft = mlx_logaddexp_f(pre, 0.0f);
      g_out[j] = metal::precise::exp(neg_exp_a_log[j] * soft);
      beta_out[j] = mlx_sigmoid_f(float(b_in[j]));
    }
  }
}
if (hd < 2 * KH) {
  ss = simd_sum(ss) + qk_eps;
  float den = metal::precise::sqrt(ss);
  if (is_q) den = den * sqrt_kd;
  for (int k = 0; k < 4; ++k) {
    const int i = int(lane) + 32 * k;
    if (i < width) {
      device float* o = is_q ? q_out + hd * KD + i : k_out + (hd - KH) * KD + i;
      o[0] = yv[k] / den;
    }
  }
}
