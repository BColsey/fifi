//! The grouped (per-expert) dequantizing GEMM behind the ids of an [`MmidIds`]: the weight is the
//! file's own blocks of one of the in-house dequantizing types (Q3_K, Q4_0, Q5_K, IQ4_XS, every
//! codebook type and Q2_K), the activation is f32 rows, and no q8_1 quantize pass feeds it.
//!
//! The ids are taken only as an [`MmidIds`], like the vendored grouped launch, so the kernel
//! never trusts caller-built id values: its constructors prove that `ids_dst` is a permutation of
//! the token slots and that the expert bounds rise over them.

use super::{MmidIds, MmidShape, MmidWeight, rule};
use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::gemm_dq::{
    GemmDqMmidRaw, gemm_dq_mmid_launch, gemm_dq_mmid_type,
};
use crate::kernels::cuda::resident::matvec_q::QuantType;

/// One grouped dequantizing GEMM. All ranges are device ranges of this context.
///
/// - `weight`: the expert stack, `weight.n` rows of `weight.k` values per expert, experts
///   `weight.stride_bytes` apart; its type is one [`gemm_dq_mmid_type`] accepts.
/// - `x`: f32 rows of `x_row_stride >= k` floats, 4-byte aligned, read only. With `x_per_slot`
///   false they are the token rows (slot `s` reads row `s / top_k`: the gate and up input);
///   with it true row `s` itself (the down input, in token-slot order).
/// - `y`: f32 slot rows of `ld >= n` floats, `(m * top_k - 1) * ld + n` floats, 4-byte aligned:
///   row `t * top_k + r` receives pick `r` of token `t`, like the vendored launch's `dst`.
pub(crate) struct DqMmidCall<'b, 'i> {
    pub(crate) shape: MmidShape,
    pub(crate) weight: MmidWeight<'b>,
    pub(crate) ids: &'i MmidIds<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) x_row_stride: usize,
    pub(crate) x_per_slot: bool,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) ld: usize,
}

impl CudaContext {
    /// Queues one grouped dequantizing GEMM of `call` on `stream` and returns when it is
    /// queued; `y` holds the products after the stream drained. The ids are read in stream
    /// order, so their constructor's work must precede this launch on `stream`.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a type
    /// the GEMM does not serve, ids made for another shape, a `k` that is not a positive
    /// multiple of the type's block size and of 64, strides below the row length, a range that
    /// is too short or misaligned, and `y` overlapping another range of the call.
    pub(crate) fn gemm_dq_mmid(
        &self,
        stream: &DeviceStream,
        call: DqMmidCall<'_, '_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let weight = &call.weight;
        let ty = weight.quant;
        if !gemm_dq_mmid_type(ty) {
            return Err(rule(
                "type",
                &format!("the grouped dequantizing GEMM has no {ty:?} decoder"),
            ));
        }
        if call.ids.shape != call.shape {
            let detail = format!("ids of {:?} for a {:?} launch", call.ids.shape, call.shape);
            return Err(rule("ids shape", &detail));
        }
        super::check_ids_shape(&call.shape)?;
        let (k, n) = (weight.k, weight.n);
        if k == 0 || !k.is_multiple_of(ty.block_size().max(64)) {
            return Err(rule(
                "K",
                &format!(
                    "K is {k}, not a positive multiple of {}",
                    ty.block_size().max(64)
                ),
            ));
        }
        let expert_bytes = ty
            .weight_bytes(n, k)
            .ok_or_else(|| rule("stride", "the expert size overflows"))?;
        if n == 0 || weight.stride_bytes < expert_bytes {
            return Err(rule(
                "stride",
                &format!(
                    "{} bytes between experts, below the {expert_bytes} of {n} rows",
                    weight.stride_bytes
                ),
            ));
        }
        if call.x_row_stride < k || call.ld < n {
            return Err(rule(
                "stride",
                &format!(
                    "x row stride {} for K {k}, y row stride {} for N {n}",
                    call.x_row_stride, call.ld
                ),
            ));
        }
        let total_cols = call.shape.ne_get_rows();
        let x_rows = if call.x_per_slot {
            total_cols
        } else {
            call.shape.m
        };
        let need_w = (call.shape.n_experts - 1)
            .checked_mul(weight.stride_bytes)
            .and_then(|head| head.checked_add(expert_bytes))
            .ok_or_else(|| rule("size", "the weight stack size overflows"))?;
        let need_x = (x_rows - 1)
            .checked_mul(call.x_row_stride)
            .and_then(|head| head.checked_add(k))
            .and_then(|floats| floats.checked_mul(4))
            .ok_or_else(|| rule("size", "the activation size overflows"))?;
        let need_y = (total_cols - 1)
            .checked_mul(call.ld)
            .and_then(|head| head.checked_add(n))
            .and_then(|floats| floats.checked_mul(4))
            .ok_or_else(|| rule("size", "the output size overflows"))?;
        let w_align = if ty == QuantType::Q5K { 16 } else { 2 };
        require_fit("w", &weight.base, need_w, w_align)?;
        require_fit("x", &call.x, need_x, 4)?;
        require_fit("y", &call.y, need_y, 4)?;
        let written = Claim::of("y", &call.y, need_y);
        require_apart(&written, &Claim::of("w", &weight.base, need_w))?;
        require_apart(&written, &Claim::of("x", &call.x, need_x))?;
        let ids = &call.ids.out;
        let ids_dst_bytes = total_cols * 4;
        let bounds_bytes = (call.shape.n_experts + 1) * 4;
        require_apart(&written, &Claim::of("ids_dst", &ids.ids_dst, ids_dst_bytes))?;
        require_apart(
            &written,
            &Claim::of("expert_bounds", &ids.expert_bounds, bounds_bytes),
        )?;
        let raw = GemmDqMmidRaw {
            ty,
            w: self.raw_range(&weight.base)?,
            x: self.raw_range(&call.x)?,
            y: self.raw_range(&call.y)?,
            ids_dst: self.raw_range(&ids.ids_dst)?,
            expert_bounds: self.raw_range(&ids.expert_bounds)?,
            k,
            rows: n,
            n_experts: call.shape.n_experts,
            total_cols,
            x_div: if call.x_per_slot { 1 } else { call.shape.top_k },
            x_row_stride: call.x_row_stride,
            y_row_stride: call.ld,
            expert_stride: weight.stride_bytes,
        };
        self.launch_on(stream, move |s| gemm_dq_mmid_launch(s, &raw))
    }
}
