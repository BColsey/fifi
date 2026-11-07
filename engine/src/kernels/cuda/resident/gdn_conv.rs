//! The DeltaNet causal conv step of the resident decode: kernel 4, depthwise, over the first
//! `channels` columns of each projection row.
//!
//! The kernel is the vendored `conv_silu_k` (xavier-qwen); this module adds a launch shim, this
//! wrapper and its tests, and no kernel. The ggml comparison of `conv_silu_k` found no shape
//! match.
//!
//! [`CudaContext::gdn_conv`] queues one kernel on a [`DeviceStream`] and returns when it is
//! queued. It allocates nothing, copies nothing and never synchronizes; a fault in the queued work
//! surfaces at the next synchronize of the stream. Per channel `c` and token `t`, with `x =
//! proj[t][c]` and `(s0, s1, s2)` the three previous inputs of the channel, oldest first:
//! `out[t][c] = silu(w[c][0]*s0 + w[c][1]*s1 + w[c][2]*s2 + w[c][3]*x)`, then the window shifts
//! by one. Tap 0 multiplies the oldest input, as in the GGUF tensor `ssm_conv1d.weight`.
//!
//! State layout. The state is planar f32 `[3][channels]` on the device, row 0 the oldest input.
//! ggml keeps the same ring as `[channel][tap]`, the transpose of this layout: a state import or
//! export (tag `qwen36-gguf-xq-1` keeps the device order) must transpose it. The state is read and
//! written in place.
//!
//! Sequence start. The state must be zero at the start of a sequence (llama.cpp starts from zeros).
//! This wrapper does not zero it, because it is launch-only; zeroing is the caller's (a `write_at`
//! or a memset on the stream).
//!
//! What the caller guarantees: the zero state at a sequence start, `proj` rows written by an
//! earlier launch on the same stream (one stream order) before this call, and `w` as stored by the
//! GGUF upload, f32 `[channels][4]`. The kernel reads no position, length, slot or status word, so
//! the call takes no `StepParams` range.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a status from the
//! native layer other than an allocation failure latches the backend terminal. In order: the
//! backend is usable; `m`, `channels` and `ld` are inside their limits and `ld >= channels`; each
//! range holds what the call uses at the offset alignment the kernel needs (`proj`, `state` and
//! `out` 4, `weights` 16); `state`, which the kernel reads and writes, overlaps none of the other
//! ranges, and `out` overlaps neither `proj` nor `weights` (`proj` and `weights` are read only
//! and may share bytes); the four ranges belong to this context. A refused call changes nothing.

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::gdn_conv::{
    GDN_CONV_MAX_CHANNELS, GDN_CONV_MAX_LD, GDN_CONV_MAX_M, GdnConvRaw, GdnConvSnapRaw,
    gdn_conv_launch, gdn_conv_snap_launch,
};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

const FLOAT: usize = size_of::<f32>();

/// The geometry of one conv launch: `m` tokens, `channels` conv channels, and `ld` floats between
/// two rows of the projection (`ld >= channels`; the first `channels` columns of a row are read).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConvShape {
    pub(crate) m: usize,
    pub(crate) channels: usize,
    pub(crate) ld: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConvBytes {
    pub(crate) proj: usize,
    pub(crate) weights: usize,
    pub(crate) state: usize,
    pub(crate) out: usize,
}

/// Where a K-row window stores the state after each of its rows but the last (patch
/// GDN-snap-stride): token `t < m - 1` writes its state at byte `t * stride` of `range`, in the
/// layout of the live state (the planar `[3][channels]` window for the conv, `[n_v_heads][128][128]`
/// for the delta). `stride` is the byte distance between two rollback slots, so the range starts at
/// the first slot's copy of this layer's state and reaches into the later slots.
pub(crate) struct SnapRows<'b> {
    pub(crate) range: DeviceRange<'b>,
    pub(crate) stride: usize,
}

/// The four device ranges of one conv launch and its geometry. `proj` is f32 `[m][ld]`, `weights`
/// f32 `[channels][4]` (16-byte aligned), `state` f32 `[3][channels]` (read and written) and `out`
/// f32 `[m][channels]`.
pub(crate) struct ConvCall<'b> {
    pub(crate) shape: ConvShape,
    pub(crate) proj: DeviceRange<'b>,
    pub(crate) weights: DeviceRange<'b>,
    pub(crate) state: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
}

pub(crate) fn conv_bytes(shape: ConvShape) -> Result<ConvBytes, EngError> {
    let overflow = || EngError::BadRequest(format!("the CUDA conv shape {shape:?} overflows"));
    let reach = shape
        .m
        .checked_sub(1)
        .and_then(|rows| rows.checked_mul(shape.ld))
        .and_then(|floats| floats.checked_add(shape.channels))
        .ok_or_else(overflow)?;
    Ok(ConvBytes {
        proj: reach.checked_mul(FLOAT).ok_or_else(overflow)?,
        weights: shape.channels.checked_mul(4 * FLOAT).ok_or_else(overflow)?,
        state: shape.channels.checked_mul(3 * FLOAT).ok_or_else(overflow)?,
        out: shape
            .m
            .checked_mul(shape.channels)
            .and_then(|floats| floats.checked_mul(FLOAT))
            .ok_or_else(overflow)?,
    })
}

fn check_shape(shape: ConvShape) -> Result<(), EngError> {
    let ConvShape { m, channels, ld } = shape;
    if !(1..=GDN_CONV_MAX_M).contains(&m) {
        return Err(EngError::BadRequest(format!(
            "the CUDA conv launch takes 1 to {GDN_CONV_MAX_M} tokens, got {m}"
        )));
    }
    if !(1..=GDN_CONV_MAX_CHANNELS).contains(&channels) {
        return Err(EngError::BadRequest(format!(
            "the CUDA conv launch takes 1 to {GDN_CONV_MAX_CHANNELS} channels, got {channels}"
        )));
    }
    if !(channels..=GDN_CONV_MAX_LD).contains(&ld) {
        return Err(EngError::BadRequest(format!(
            "the CUDA conv launch takes ld from {channels} (the channels) to {GDN_CONV_MAX_LD}, got {ld}"
        )));
    }
    Ok(())
}

impl CudaContext {
    /// Queues one conv step of `call.shape.m` tokens on `stream`: reads the first `channels`
    /// columns of each `proj` row, writes `silu(conv)` to `out` and advances `state` in place.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for a shape outside the
    /// limits, for a range that is short or misaligned, and for a `state` or `out` that overlaps
    /// another range. The state must be zero at the start of a sequence; the caller zeroes it.
    pub(crate) fn gdn_conv(
        &self,
        stream: &DeviceStream,
        call: ConvCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_gdn_conv(&call)?;
        self.launch_on(stream, move |stream_raw| gdn_conv_launch(stream_raw, raw))
    }

    fn plan_gdn_conv<'b>(&self, call: &ConvCall<'b>) -> Result<GdnConvRaw<'b>, EngError> {
        let shape = call.shape;
        check_shape(shape)?;
        let need = conv_bytes(shape)?;
        require_fit("proj", &call.proj, need.proj, 4)?;
        require_fit("weights", &call.weights, need.weights, 16)?;
        require_fit("state", &call.state, need.state, 4)?;
        require_fit("out", &call.out, need.out, 4)?;
        let state = Claim::of("state", &call.state, need.state);
        let out = Claim::of("out", &call.out, need.out);
        let proj = Claim::of("proj", &call.proj, need.proj);
        let weights = Claim::of("weights", &call.weights, need.weights);
        require_apart(&state, &out)?;
        require_apart(&state, &proj)?;
        require_apart(&state, &weights)?;
        require_apart(&out, &proj)?;
        require_apart(&out, &weights)?;
        Ok(GdnConvRaw {
            proj: self.raw_range(&call.proj)?,
            w: self.raw_range(&call.weights)?,
            state: self.raw_range(&call.state)?,
            out: self.raw_range(&call.out)?,
            m: shape.m,
            channels: shape.channels,
            ld: shape.ld,
        })
    }
}

/// Rule of the strided snapshots, shared with the delta wrapper: `m >= 2`, a stride that is a
/// multiple of 4 and holds one snapshot of `row` bytes. Returns the bytes the snapshots reach,
/// `(m - 2) * stride + row`.
pub(in crate::kernels::cuda::resident) fn snap_reach(
    m: usize,
    stride: usize,
    row: usize,
) -> Result<usize, EngError> {
    let rule = |detail: String| EngError::BadRequest(format!("gdn snap: {detail}"));
    if m < 2 {
        return Err(rule(format!("needs at least 2 tokens, got {m}")));
    }
    if !stride.is_multiple_of(4) || stride < row {
        return Err(rule(format!(
            "the stride must be a multiple of 4 and at least one snapshot ({row} bytes), got {stride}"
        )));
    }
    (m - 2)
        .checked_mul(stride)
        .and_then(|bytes| bytes.checked_add(row))
        .ok_or_else(|| rule("the snapshot extent overflows".to_owned()))
}

impl CudaContext {
    /// [`Self::gdn_conv`] that also stores the state after each token but the last at
    /// `snap.stride` bytes between tokens, so the state after row `t` of a K-row window lands in
    /// rollback slot `t`. The snapshot stores are all it adds: `out` and `state` are the bytes of
    /// the plain call (the `kSnap` instantiation of `conv_silu_k`; the plain call keeps the
    /// original one).
    ///
    /// Refused as `BadRequest`, besides the rules of the plain call: fewer than 2 tokens, a stride
    /// that is not a multiple of 4 or does not hold one `[3][channels]` window, a `snap` range
    /// that is short, misaligned or overlaps another range of the call.
    pub(crate) fn gdn_conv_snap(
        &self,
        stream: &DeviceStream,
        call: ConvCall<'_>,
        snap: SnapRows<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_gdn_conv(&call)?;
        let row =
            call.shape.channels.checked_mul(3 * FLOAT).ok_or_else(|| {
                EngError::BadRequest("the CUDA conv snap row overflows".to_owned())
            })?;
        let reach = snap_reach(call.shape.m, snap.stride, row)?;
        require_fit("snap", &snap.range, reach, 4)?;
        let need = conv_bytes(call.shape)?;
        let claim = Claim::of("snap", &snap.range, reach);
        require_apart(&claim, &Claim::of("state", &call.state, need.state))?;
        require_apart(&claim, &Claim::of("out", &call.out, need.out))?;
        require_apart(&claim, &Claim::of("proj", &call.proj, need.proj))?;
        require_apart(&claim, &Claim::of("weights", &call.weights, need.weights))?;
        let snap_raw = GdnConvSnapRaw {
            range: self.raw_range(&snap.range)?,
            stride: snap.stride,
        };
        #[cfg(test)]
        self.note_row_kernel(crate::kernels::row_launches::RowKernel::ConvSnap);
        self.launch_on(stream, move |stream_raw| {
            gdn_conv_snap_launch(stream_raw, raw, snap_raw)
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod snap_tests;
#[cfg(all(test, feature = "native-cuda"))]
mod tests;
