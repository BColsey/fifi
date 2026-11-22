//! The per-layer-embedding block of a decode step on the card: the decode of IQ4_NL embedding
//! rows, the gate of the projected rows against the residual stream, and the causal convolution
//! with its ring. Any model with a per-layer or n-gram embedding table uses these three launches
//! around its two projections (`dense_gemv`); the arithmetic is the reference of
//! `family/flash_next/ple.rs`, which checks each launch in the tests.
//!
//! Each method queues one kernel on a [DeviceStream] and returns; none allocates, copies or
//! synchronizes. Argument errors return BadRequest before a native launch: sizes, alignment and
//! (for ranges a kernel writes) disjointness from every other range of the call.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::ple_block::{
    PleConvRaw, PleGateRaw, PleRowsRaw, ple_conv_launch, ple_gate_launch, ple_rows_launch,
};

/// Bytes of one IQ4_NL block of 32 values.
const IQ4_BLOCK_BYTES: usize = 18;
const IQ4_BLOCK_VALUES: usize = 32;

/// `rows` rows of `head_dim` IQ4_NL values decoded to f32 and multiplied by `scale`.
pub(crate) struct PleRowsCall<'b> {
    pub(crate) out: DeviceRange<'b>,
    pub(crate) raw: DeviceRange<'b>,
    pub(crate) scale: f32,
    pub(crate) rows: usize,
    pub(crate) head_dim: usize,
}

/// The gate of one token over `branches` branches of `hidden` values.
pub(crate) struct PleGateCall<'b> {
    pub(crate) keys: DeviceRange<'b>,
    pub(crate) hyper: DeviceRange<'b>,
    pub(crate) value: DeviceRange<'b>,
    pub(crate) norm_key: DeviceRange<'b>,
    pub(crate) norm_query: DeviceRange<'b>,
    pub(crate) norm_conv: DeviceRange<'b>,
    pub(crate) gated: DeviceRange<'b>,
    pub(crate) normed: DeviceRange<'b>,
    pub(crate) eps: f32,
    pub(crate) branches: usize,
    pub(crate) hidden: usize,
}

/// The convolution of one token over `width` channels; `ring` is read and written in place.
pub(crate) struct PleConvCall<'b> {
    pub(crate) gated: DeviceRange<'b>,
    pub(crate) normed: DeviceRange<'b>,
    pub(crate) conv: DeviceRange<'b>,
    pub(crate) ring: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
    pub(crate) width: usize,
    pub(crate) conv_kernel: usize,
    pub(crate) ngram: usize,
}

fn bad(what: &str) -> EngError {
    EngError::BadRequest(format!("ple block: {what}"))
}

fn product(factors: &[usize]) -> Result<usize, EngError> {
    factors
        .iter()
        .try_fold(1usize, |acc, &f| acc.checked_mul(f))
        .ok_or_else(|| bad("a size overflows"))
}

/// Every pair of the named claims must share no byte.
fn all_apart(claims: &[(&str, &DeviceRange<'_>, usize)]) -> Result<(), EngError> {
    for (i, (name_a, range_a, bytes_a)) in claims.iter().enumerate() {
        for (name_b, range_b, bytes_b) in &claims[..i] {
            require_apart(
                &Claim::of(name_a, range_a, *bytes_a),
                &Claim::of(name_b, range_b, *bytes_b),
            )?;
        }
    }
    Ok(())
}

impl CudaContext {
    pub(crate) fn ple_rows(
        &self,
        stream: &DeviceStream,
        call: PleRowsCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        if !call.head_dim.is_multiple_of(IQ4_BLOCK_VALUES)
            || !(32..=8192).contains(&call.head_dim)
            || !(1..=65_536).contains(&call.rows)
            || !call.scale.is_finite()
        {
            return Err(bad(
                "rows need a head size of 32 to 8192 in whole blocks, 1 to 65536 rows and a finite scale",
            ));
        }
        let out_bytes = product(&[call.rows, call.head_dim, 4])?;
        let raw_bytes = product(&[call.rows, call.head_dim / IQ4_BLOCK_VALUES, IQ4_BLOCK_BYTES])?;
        require_fit("out", &call.out, out_bytes, 4)?;
        require_fit("raw", &call.raw, raw_bytes, 2)?;
        all_apart(&[("out", &call.out, out_bytes), ("raw", &call.raw, raw_bytes)])?;
        let raw = PleRowsRaw {
            out: self.raw_range(&call.out)?,
            raw: self.raw_range(&call.raw)?,
            scale: call.scale,
            rows: call.rows,
            head_dim: call.head_dim,
        };
        self.launch_on(stream, |stream_raw| ple_rows_launch(stream_raw, &raw))
    }

    pub(crate) fn ple_gate(
        &self,
        stream: &DeviceStream,
        call: PleGateCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        if !(1..=64).contains(&call.branches)
            || !(1..=(1 << 20)).contains(&call.hidden)
            || !call.eps.is_finite()
            || call.eps < 0.0
        {
            return Err(bad(
                "the gate needs 1 to 64 branches, a width of 1 to 2^20 and a finite non-negative epsilon",
            ));
        }
        let width = product(&[call.branches, call.hidden, 4])?;
        let value = product(&[call.hidden, 4])?;
        let wide = [
            ("keys", &call.keys),
            ("hyper", &call.hyper),
            ("norm_key", &call.norm_key),
            ("norm_query", &call.norm_query),
            ("norm_conv", &call.norm_conv),
            ("gated", &call.gated),
            ("normed", &call.normed),
        ];
        for (name, range) in wide {
            require_fit(name, range, width, 4)?;
        }
        require_fit("value", &call.value, value, 4)?;
        // The two written ranges share no byte with anything; the read ranges may overlap.
        let others = [
            ("keys", &call.keys, width),
            ("hyper", &call.hyper, width),
            ("value", &call.value, value),
            ("norm_key", &call.norm_key, width),
            ("norm_query", &call.norm_query, width),
            ("norm_conv", &call.norm_conv, width),
        ];
        for (name, range, bytes) in others {
            for (written, wr) in [("gated", &call.gated), ("normed", &call.normed)] {
                require_apart(
                    &Claim::of(written, wr, width),
                    &Claim::of(name, range, bytes),
                )?;
            }
        }
        require_apart(
            &Claim::of("gated", &call.gated, width),
            &Claim::of("normed", &call.normed, width),
        )?;
        let raw = PleGateRaw {
            keys: self.raw_range(&call.keys)?,
            hyper: self.raw_range(&call.hyper)?,
            value: self.raw_range(&call.value)?,
            norm_key: self.raw_range(&call.norm_key)?,
            norm_query: self.raw_range(&call.norm_query)?,
            norm_conv: self.raw_range(&call.norm_conv)?,
            gated: self.raw_range(&call.gated)?,
            normed: self.raw_range(&call.normed)?,
            eps: call.eps,
            branches: call.branches,
            hidden: call.hidden,
        };
        self.launch_on(stream, |stream_raw| ple_gate_launch(stream_raw, &raw))
    }

    pub(crate) fn ple_conv(
        &self,
        stream: &DeviceStream,
        call: PleConvCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        if !(1..=(1 << 22)).contains(&call.width)
            || !(2..=16).contains(&call.conv_kernel)
            || !(1..=16).contains(&call.ngram)
        {
            return Err(bad(
                "the convolution needs 1 to 2^22 channels, a kernel of 2 to 16 and an n-gram of 1 to 16",
            ));
        }
        let width = product(&[call.width, 4])?;
        let conv = product(&[call.width, call.conv_kernel, 4])?;
        let ring = product(&[call.width, call.conv_kernel - 1, call.ngram, 4])?;
        require_fit("gated", &call.gated, width, 4)?;
        require_fit("normed", &call.normed, width, 4)?;
        require_fit("out", &call.out, width, 4)?;
        require_fit("conv", &call.conv, conv, 4)?;
        require_fit("ring", &call.ring, ring, 4)?;
        all_apart(&[
            ("gated", &call.gated, width),
            ("normed", &call.normed, width),
            ("conv", &call.conv, conv),
            ("ring", &call.ring, ring),
            ("out", &call.out, width),
        ])?;
        let raw = PleConvRaw {
            gated: self.raw_range(&call.gated)?,
            normed: self.raw_range(&call.normed)?,
            conv: self.raw_range(&call.conv)?,
            ring: self.raw_range(&call.ring)?,
            out: self.raw_range(&call.out)?,
            width: call.width,
            conv_kernel: call.conv_kernel,
            ngram: call.ngram,
        };
        self.launch_on(stream, |stream_raw| ple_conv_launch(stream_raw, &raw))
    }
}

#[cfg(test)]
mod tests;
