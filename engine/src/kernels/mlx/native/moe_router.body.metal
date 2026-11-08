// router_topk -- the device MoE router of one row in ONE kernel: the stable top-k selection
// (largest probability first, lowest index first on equal values -- exactly the stable argsort
// of the negated row), the renormalized gates, in rank order. Replaces negate, argsort, slice,
// clamp, take, the k-sum, the divide and the reshape of the op chain (eight traced kernels).
// The softmax and the health sum over the whole row stay MLX ops: the softmax feeds the kernel
// bit-exact F32 values and the health reduction keeps its own accumulation order.
// Inputs: probs [B, E] F32 (row-contiguous), e (int), k (int), renorm (int).  Outputs:
//   ids_out   [B, k] U32 in rank order, already below E (no clamp follows),
//   gates_out [B, k] (outer shape [B, k, 1, 1] where the caller wants rank 4) F32.
// Rounding mirrors the unfused graph: the top values are the exact F32 softmax outputs, the
// renorm denominator is a sequential sum from 0 in rank order (the order moe_combine validated
// against MLX's small-row reduce), the divide is one IEEE F32 division.
// Launch: threadgroup (E, 1, 1) = one row per group, grid (E, B, 1).  E must be a nonzero
// multiple of 32 (the simdgroup width) and at most 1024; k at most E.  NaN never wins a round,
// so a row with fewer than k finite entries selects the finite ones first, as the chain does.
const uint row = uint(thread_position_in_grid.y);
const uint e = uint(thread_position_in_grid.x);
const uint lane = uint(thread_index_in_simdgroup);
const uint sg = uint(simdgroup_index_in_threadgroup);
const uint nsg = uint(threads_per_threadgroup.x) / 32u;
// K_MAX bounds the threadgroup scratch; the launch contract keeps k <= K_MAX.
threadgroup float red_v[32];
threadgroup uint red_i[32];
threadgroup float topv[K_MAX];
threadgroup uint topi[K_MAX];
float myv = probs[(long)row * e_count + e];
const uint k_count = uint(k);
for (uint kk = 0; kk < k_count; ++kk) {
  float v = myv;
  uint i = e;
  // simdgroup argmax butterfly: strictly larger value, equal value by lower index.
  for (uint off = 16u; off > 0u; off >>= 1u) {
    const float ov = simd_shuffle(v, lane ^ off);
    const uint oi = simd_shuffle(i, lane ^ off);
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
  }
  if (lane == 0u) { red_v[sg] = v; red_i[sg] = i; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (e == 0u) {
    float bv = red_v[0];
    uint bi = red_i[0];
    for (uint m = 1u; m < nsg; ++m) {
      if (red_v[m] > bv || (red_v[m] == bv && red_i[m] < bi)) { bv = red_v[m]; bi = red_i[m]; }
    }
    topv[kk] = bv;
    topi[kk] = bi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (e == topi[kk]) myv = -1e30f;  // every softmax value is >= 0, so this always masks.
}
if (e < k_count) {
  float g = topv[e];
  if (renorm != 0) {
    float s = 0.0f;
    for (uint m = 0u; m < k_count; ++m) s += topv[m];
    g = g / s;
  }
  gates_out[(long)row * k_count + e] = g;
  ids_out[(long)row * k_count + e] = topi[e];
}
