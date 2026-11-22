//! Many rows in one launch: the DeltaNet conv, the DeltaNet recurrence, the attention
//! preparation, the KV8 append and the KV8 decode attention for `n` rows, each row the single token
//! of a sequence of its own. The decode window of several sequences runs these entries once per
//! layer instead of once per sequence per layer.
//!
//! Row `r` of a launch reads row `r` of every per-row input (`proj`, `conv`, `q`, ... laid out as
//! `n` rows of the single-token shape) and writes row `r` of every per-row output. What differs
//! per row lives in a device table that this module builds from proved ranges:
//!
//! - [`StateRows`]: `n` recurrent or conv state ranges of one layer, one per row.
//! - [`KvRows`]: `n` records, each the KV8 planes of one layer of one row's sequence and that
//!   sequence's 32-byte `StepParams` block (position at byte 4, `kv_len` at byte 8, status at
//!   byte 16). The words are read when the kernels run, so one captured graph serves every
//!   position of every row.
//!
//! A table is built once per window composition and layer (the addresses of a sequence's state do
//! not move) and holds its ranges by lifetime, so it cannot outlive the buffers it names. Building
//! one proves each range fits the geometry it is for, is aligned, and that the ranges the kernels
//! write do not overlap each other; a call proves them apart from its own outputs.
//!
//! Each entry computes row `r` as the single-token entry of the same name computes it for that
//! row alone: `n` rows in one launch carry the bytes of `n` launches of one row (the device tests
//! compare them bit for bit). A row whose position or `kv_len` is out of range is refused alone, as
//! in the single-token launch: its `q` or `out` row is NaN, none of its plane bytes is written and
//! mask 4 reaches its own status word. Every refusal of a malformed call is a `BadRequest` before
//! any native call.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the many-rows decode window")
)]

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::multi_row::{
    AppendMultiRaw, ConvMultiRaw, DecodeMultiRaw, DeltaMultiRaw, KV_ROW_BYTES, KvRowRaw,
    MULTI_ROW_MAX, PrepMultiRaw, STATE_ENTRY_BYTES, STEP_BYTES, attn_decode_q8_multi_launch,
    attn_prep_multi_launch, gdn_conv_multi_launch, gdn_delta_multi_launch, kv8_append_multi_launch,
};
use crate::kernels::cuda::resident::attn_decode::{AttnSpan, AttnSplit};
use crate::kernels::cuda::resident::attn_decode_q8::{AttnQ8Shape, GROUP, attn_q8_part_bytes};
use crate::kernels::cuda::resident::attn_prep::{HEAD_DIM, RopeParams, check_rope};
use crate::kernels::cuda::resident::gdn_conv::ConvShape;
use crate::kernels::cuda::resident::gdn_delta::{
    DeltaShape, DeltaWeights, GDN_EPS, check_counts, check_rows, delta_bytes,
};
use crate::kernels::cuda::resident::kv8_append::{GROUP as QUANT_GROUP, Kv8CodeScale, Kv8Shape};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("multi_row: {tag}: {detail}"))
}

fn check_rows_count(n: usize) -> Result<(), EngError> {
    if (1..=MULTI_ROW_MAX).contains(&n) {
        Ok(())
    } else {
        Err(rule(
            "rows",
            &format!("a launch takes 1 to {MULTI_ROW_MAX} rows, got {n}"),
        ))
    }
}

/// The addresses of `n` state ranges of one layer, one per row, in a device table.
pub(crate) struct StateRows<'b> {
    table: DeviceBuffer,
    ranges: Vec<DeviceRange<'b>>,
}

impl<'b> StateRows<'b> {
    /// Rows in the table.
    pub(crate) fn len(&self) -> usize {
        self.ranges.len()
    }
}

/// The KV8 planes of one layer and the step block of one row's sequence (see the module doc).
#[derive(Clone, Copy)]
pub(crate) struct KvRow<'b> {
    pub(crate) k_codes: DeviceRange<'b>,
    pub(crate) k_scales: DeviceRange<'b>,
    pub(crate) v_codes: DeviceRange<'b>,
    pub(crate) v_scales: DeviceRange<'b>,
    pub(crate) step: DeviceRange<'b>,
}

/// The records of `n` rows in a device table.
pub(crate) struct KvRows<'b> {
    table: DeviceBuffer,
    rows: Vec<KvRow<'b>>,
}

impl<'b> KvRows<'b> {
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }
}

impl CudaContext {
    /// Writes the table of `ranges`, one state range per row. Refused as `BadRequest` for no
    /// rows, more than 256, a range that is not 4-byte aligned or that belongs to another context,
    /// and two ranges that share a byte (rows write their own state).
    pub(crate) fn state_rows<'b>(
        &self,
        ranges: &[DeviceRange<'b>],
    ) -> Result<StateRows<'b>, EngError> {
        check_rows_count(ranges.len())?;
        let mut bytes = Vec::with_capacity(ranges.len() * STATE_ENTRY_BYTES);
        for range in ranges {
            require_fit("state", range, 4, 4)?;
            bytes.extend_from_slice(&self.raw_range(range)?.address().addr().to_le_bytes());
        }
        let claims: Vec<Claim<'_>> = ranges
            .iter()
            .map(|range| Claim::of("state", range, range.bytes()))
            .collect();
        for (i, claim) in claims.iter().enumerate() {
            for other in &claims[i + 1..] {
                require_apart(claim, other)?;
            }
        }
        Ok(StateRows {
            table: self.upload(&bytes)?,
            ranges: ranges.to_vec(),
        })
    }

    /// Writes the table of `rows`, one KV record per row. Refused as `BadRequest` for no rows,
    /// more than 256, a step block under 32 bytes or a plane that is not 4-byte aligned or belongs
    /// to another context.
    pub(crate) fn kv_rows<'b>(&self, rows: &[KvRow<'b>]) -> Result<KvRows<'b>, EngError> {
        check_rows_count(rows.len())?;
        let mut bytes = Vec::with_capacity(rows.len() * KV_ROW_BYTES);
        for row in rows {
            let addr = |range: &DeviceRange<'b>, align: usize, name: &str| {
                if !range.offset().is_multiple_of(align) {
                    return Err(rule(
                        name,
                        &format!("the range is not {align}-byte aligned"),
                    ));
                }
                Ok(self.raw_range(range)?.address().addr() as u64)
            };
            require_fit("step", &row.step, STEP_BYTES, 4)?;
            let raw = KvRowRaw {
                k_codes: addr(&row.k_codes, 4, "k_codes")?,
                k_scales: addr(&row.k_scales, 4, "k_scales")?,
                v_codes: addr(&row.v_codes, 4, "v_codes")?,
                v_scales: addr(&row.v_scales, 4, "v_scales")?,
                step: addr(&row.step, 4, "step")?,
                reserved: [0; 3],
            };
            for word in [
                raw.k_codes,
                raw.k_scales,
                raw.v_codes,
                raw.v_scales,
                raw.step,
            ]
            .into_iter()
            .chain(raw.reserved)
            {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        Ok(KvRows {
            table: self.upload(&bytes)?,
            rows: rows.to_vec(),
        })
    }
}

/// The geometry the planes of every row of a call must hold, and the check that they do.
fn check_planes(rows: &KvRows<'_>, shape: &Kv8Shape, write: bool) -> Result<(), EngError> {
    for row in &rows.rows {
        require_fit("k_codes", &row.k_codes, shape.code_bytes(), 4)?;
        require_fit("v_codes", &row.v_codes, shape.code_bytes(), 4)?;
        require_fit("k_scales", &row.k_scales, shape.scale_bytes(), 4)?;
        require_fit("v_scales", &row.v_scales, shape.scale_bytes(), 4)?;
    }
    if !write {
        return Ok(());
    }
    // Rows append to their own planes: no plane byte of one row is a plane byte of another row,
    // or of the same row's other plane.
    let mut claims = Vec::with_capacity(rows.rows.len() * 4);
    for row in &rows.rows {
        claims.push(Claim::of("k_codes", &row.k_codes, shape.code_bytes()));
        claims.push(Claim::of("v_codes", &row.v_codes, shape.code_bytes()));
        claims.push(Claim::of("k_scales", &row.k_scales, shape.scale_bytes()));
        claims.push(Claim::of("v_scales", &row.v_scales, shape.scale_bytes()));
    }
    for (i, claim) in claims.iter().enumerate() {
        for other in &claims[i + 1..] {
            require_apart(claim, other)?;
        }
    }
    Ok(())
}

/// The conv of `states.len()` rows. `shape.m` is the row count and must equal it.
pub(crate) struct ConvMultiCall<'b> {
    pub(crate) shape: ConvShape,
    pub(crate) proj: DeviceRange<'b>,
    pub(crate) weights: DeviceRange<'b>,
    pub(crate) states: &'b StateRows<'b>,
    pub(crate) out: DeviceRange<'b>,
}

/// The delta recurrence of `states.len()` rows. `shape.m` is the row count and must equal it.
pub(crate) struct DeltaMultiCall<'b> {
    pub(crate) shape: DeltaShape,
    pub(crate) eps: f32,
    pub(crate) conv: DeviceRange<'b>,
    pub(crate) proj: DeviceRange<'b>,
    pub(crate) weights: DeltaWeights<'b>,
    pub(crate) states: &'b StateRows<'b>,
    pub(crate) out: DeviceRange<'b>,
}

/// The attention preparation of `rows.len()` rows (the f32 rows instance): `q` is f32
/// `[n][n_head][256]`, `k_rows` and `v_rows` f32 `[n][n_kv_head][256]`, all 16-byte aligned.
/// `shape.cells` bounds each row's position word.
pub(crate) struct PrepMultiCall<'b> {
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) n_rot: usize,
    pub(crate) cells: usize,
    pub(crate) ld: usize,
    pub(crate) rope: RopeParams,
    pub(crate) proj: DeviceRange<'b>,
    pub(crate) q_norm: DeviceRange<'b>,
    pub(crate) k_norm: DeviceRange<'b>,
    pub(crate) k_rows: DeviceRange<'b>,
    pub(crate) v_rows: DeviceRange<'b>,
    pub(crate) q: DeviceRange<'b>,
    pub(crate) rows: &'b KvRows<'b>,
}

/// The KV8 append of `rows.len()` rows: `k` and `v` are f32 `[n][n_kv_head][head_dim]`.
pub(crate) struct AppendMultiCall<'b> {
    pub(crate) shape: Kv8Shape,
    pub(crate) k: DeviceRange<'b>,
    pub(crate) v: DeviceRange<'b>,
    pub(crate) rows: &'b KvRows<'b>,
}

/// The KV8 decode attention of `rows.len()` rows. `shape.m` is the row count and must equal it.
pub(crate) struct DecodeMultiCall<'b> {
    pub(crate) shape: AttnQ8Shape,
    pub(crate) span: AttnSpan,
    pub(crate) split: AttnSplit,
    pub(crate) scale: f32,
    pub(crate) q: DeviceRange<'b>,
    pub(crate) proj: Option<DeviceRange<'b>>,
    pub(crate) rows: &'b KvRows<'b>,
    pub(crate) part: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
}

fn same_rows(what: &str, shape_rows: usize, table_rows: usize) -> Result<(), EngError> {
    if shape_rows == table_rows {
        Ok(())
    } else {
        Err(rule(
            "rows",
            &format!("{what}: the shape has {shape_rows} rows, the table {table_rows}"),
        ))
    }
}

/// Each state range is apart from every range in `others` (a state is written in place).
fn states_apart(states: &StateRows<'_>, need: usize, others: &[Claim<'_>]) -> Result<(), EngError> {
    for range in &states.ranges {
        require_fit("state", range, need, 4)?;
        let claim = Claim::of("state", range, need);
        for other in others {
            require_apart(&claim, other)?;
        }
    }
    Ok(())
}

impl CudaContext {
    /// One conv step for each row of `call.states`, on `stream`.
    pub(crate) fn gdn_conv_multi(
        &self,
        stream: &DeviceStream,
        call: ConvMultiCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let n = call.states.len();
        same_rows("conv", call.shape.m, n)?;
        let need = crate::kernels::cuda::resident::gdn_conv::conv_bytes(call.shape)?;
        if !(1..=65536).contains(&call.shape.channels)
            || !(call.shape.channels..=(1 << 20)).contains(&call.shape.ld)
        {
            return Err(rule("shape", "channels or ld is outside the conv limits"));
        }
        require_fit("proj", &call.proj, need.proj, 4)?;
        require_fit("weights", &call.weights, need.weights, 16)?;
        require_fit("out", &call.out, need.out, 4)?;
        let out = Claim::of("out", &call.out, need.out);
        require_apart(&out, &Claim::of("proj", &call.proj, need.proj))?;
        require_apart(&out, &Claim::of("weights", &call.weights, need.weights))?;
        states_apart(
            call.states,
            3 * call.shape.channels * 4,
            &[
                out,
                Claim::of("proj", &call.proj, need.proj),
                Claim::of("weights", &call.weights, need.weights),
            ],
        )?;
        let raw = ConvMultiRaw {
            proj: self.raw_range(&call.proj)?,
            w: self.raw_range(&call.weights)?,
            states: self.raw_range(&call.states.table.range(0, n * STATE_ENTRY_BYTES)?)?,
            out: self.raw_range(&call.out)?,
            n,
            ld: call.shape.ld,
            channels: call.shape.channels,
        };
        self.launch_on(stream, move |raw_stream| {
            gdn_conv_multi_launch(raw_stream, raw)
        })
    }

    /// One delta step for each row of `call.states`, on `stream`.
    pub(crate) fn gdn_delta_multi(
        &self,
        stream: &DeviceStream,
        call: DeltaMultiCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let n = call.states.len();
        same_rows("delta", call.shape.m, n)?;
        check_counts(&call.shape, call.eps)?;
        check_rows(&call.shape)?;
        let need = delta_bytes(call.shape)?;
        require_fit("conv", &call.conv, need.conv, 4)?;
        require_fit("proj", &call.proj, need.proj, 4)?;
        require_fit("a", &call.weights.a, need.a, 4)?;
        require_fit("dt_bias", &call.weights.dt_bias, need.dt_bias, 4)?;
        require_fit("norm_w", &call.weights.norm_w, need.norm_w, 4)?;
        require_fit("out", &call.out, need.out, 4)?;
        let out = Claim::of("out", &call.out, need.out);
        let reads = [
            Claim::of("conv", &call.conv, need.conv),
            Claim::of("proj", &call.proj, need.proj),
            Claim::of("a", &call.weights.a, need.a),
            Claim::of("dt_bias", &call.weights.dt_bias, need.dt_bias),
            Claim::of("norm_w", &call.weights.norm_w, need.norm_w),
        ];
        for read in &reads {
            require_apart(&out, read)?;
        }
        let mut others = vec![out];
        others.extend(reads);
        states_apart(call.states, need.state, &others)?;
        let raw = DeltaMultiRaw {
            conv: self.raw_range(&call.conv)?,
            proj: self.raw_range(&call.proj)?,
            a: self.raw_range(&call.weights.a)?,
            dt_bias: self.raw_range(&call.weights.dt_bias)?,
            norm_w: self.raw_range(&call.weights.norm_w)?,
            states: self.raw_range(&call.states.table.range(0, n * STATE_ENTRY_BYTES)?)?,
            out: self.raw_range(&call.out)?,
            n,
            conv_ld: call.shape.conv_ld,
            ld: call.shape.ld,
            offsets: [call.shape.z_off, call.shape.beta_off, call.shape.alpha_off],
            n_k_heads: call.shape.n_k_heads,
            n_v_heads: call.shape.n_v_heads,
            eps: GDN_EPS,
        };
        self.launch_on(stream, move |raw_stream| {
            gdn_delta_multi_launch(raw_stream, raw)
        })
    }

    /// The attention preparation of each row of `call.rows`, on `stream`.
    pub(crate) fn attn_prep_multi(
        &self,
        stream: &DeviceStream,
        call: PrepMultiCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let n = call.rows.len();
        check_rope(&call.rope)?;
        let heads_ok = (1..=256).contains(&call.n_head) && (1..=64).contains(&call.n_kv_head);
        let rot_ok = (2..=HEAD_DIM).contains(&call.n_rot) && call.n_rot.is_power_of_two();
        let row_w = (call.n_head * 2 + call.n_kv_head * 2) * HEAD_DIM;
        if !(heads_ok && rot_ok && call.ld >= row_w && (1..(1 << 30)).contains(&call.cells)) {
            return Err(rule(
                "shape",
                "heads, n_rot, ld or cells are outside the limits",
            ));
        }
        let f = size_of::<f32>();
        require_fit("proj", &call.proj, ((n - 1) * call.ld + row_w) * f, 4)?;
        require_fit("q_norm", &call.q_norm, HEAD_DIM * f, 4)?;
        require_fit("k_norm", &call.k_norm, HEAD_DIM * f, 4)?;
        let q_bytes = n * call.n_head * HEAD_DIM * f;
        let kv_bytes = n * call.n_kv_head * HEAD_DIM * f;
        require_fit("q", &call.q, q_bytes, 16)?;
        require_fit("k_rows", &call.k_rows, kv_bytes, 16)?;
        require_fit("v_rows", &call.v_rows, kv_bytes, 16)?;
        let writes = [
            Claim::of("q", &call.q, q_bytes),
            Claim::of("k_rows", &call.k_rows, kv_bytes),
            Claim::of("v_rows", &call.v_rows, kv_bytes),
        ];
        let reads = [
            Claim::of("proj", &call.proj, ((n - 1) * call.ld + row_w) * f),
            Claim::of("q_norm", &call.q_norm, HEAD_DIM * f),
            Claim::of("k_norm", &call.k_norm, HEAD_DIM * f),
        ];
        for (i, write) in writes.iter().enumerate() {
            for read in &reads {
                require_apart(write, read)?;
            }
            for other in &writes[i + 1..] {
                require_apart(write, other)?;
            }
        }
        let raw = PrepMultiRaw {
            proj: self.raw_range(&call.proj)?,
            q_norm: self.raw_range(&call.q_norm)?,
            k_norm: self.raw_range(&call.k_norm)?,
            k_rows: self.raw_range(&call.k_rows)?,
            v_rows: self.raw_range(&call.v_rows)?,
            q: self.raw_range(&call.q)?,
            table: self.raw_range(&call.rows.table.range(0, n * KV_ROW_BYTES)?)?,
            n,
            ld: call.ld,
            heads: [call.n_head, call.n_kv_head],
            n_rot: call.n_rot,
            cells: call.cells,
            eps: call.rope.eps,
            rope_base: call.rope.rope_base,
        };
        self.launch_on(stream, move |raw_stream| {
            attn_prep_multi_launch(raw_stream, raw)
        })
    }

    /// The KV8 append of each row of `call.rows`, on `stream`.
    pub(crate) fn kv8_append_multi(
        &self,
        stream: &DeviceStream,
        call: AppendMultiCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let n = call.rows.len();
        let shape = Kv8Shape {
            rows: n,
            ..call.shape
        };
        let geometry_ok = (1..=64).contains(&shape.n_kv_head)
            && (32..=1024).contains(&shape.head_dim)
            && shape.head_dim.is_multiple_of(QUANT_GROUP)
            && (1..(1 << 30)).contains(&shape.cells);
        if !geometry_ok {
            return Err(rule(
                "shape",
                "kv heads, head_dim or cells are outside the limits",
            ));
        }
        require_fit("k", &call.k, shape.row_bytes(), 4)?;
        require_fit("v", &call.v, shape.row_bytes(), 4)?;
        check_planes(call.rows, &shape, true)?;
        let raw = AppendMultiRaw {
            k: self.raw_range(&call.k)?,
            v: self.raw_range(&call.v)?,
            table: self.raw_range(&call.rows.table.range(0, n * KV_ROW_BYTES)?)?,
            n,
            n_kv_head: shape.n_kv_head,
            head_dim: shape.head_dim,
            cells: shape.cells,
            code_scale: match shape.code_scale {
                Kv8CodeScale::StoredHalf => 1,
                Kv8CodeScale::UnroundedF32 => 2,
            },
        };
        self.launch_on(stream, move |raw_stream| {
            kv8_append_multi_launch(raw_stream, raw)
        })
    }

    /// The KV8 decode attention of each row of `call.rows`, on `stream`.
    pub(crate) fn attn_decode_q8_multi(
        &self,
        stream: &DeviceStream,
        call: DecodeMultiCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let n = call.rows.len();
        let shape = &call.shape;
        same_rows("decode", shape.m, n)?;
        let shape_ok = shape.n_head == shape.n_kv_head * GROUP
            && (1..=64).contains(&shape.n_kv_head)
            && (shape.head_dim == 128 || shape.head_dim == 256)
            && shape.cells >= 1
            && shape.cells < (1 << 30)
            && (1..=128).contains(&call.split.max_splits)
            && call.split.min_keys >= 64
            && call.split.min_keys.is_multiple_of(64)
            && call.split.min_keys <= (1 << 20)
            && call.scale > 0.0
            && call.scale < 1e30;
        let span_ok = call.span.window >= 0
            && call.span.ring >= 0
            && (call.span.ring == 0
                || (call.span.window > 0
                    && call.span.ring <= shape.cells as i64
                    && call.span.ring >= call.span.window));
        if !(shape_ok && span_ok) {
            return Err(rule(
                "shape",
                "the geometry, split plan or span is outside the limits",
            ));
        }
        let f = size_of::<f32>();
        let width = shape.n_head * shape.head_dim;
        require_fit("q", &call.q, n * width * f, 16)?;
        require_fit("out", &call.out, n * width * f, 4)?;
        let part_bytes =
            attn_q8_part_bytes(n, shape.n_kv_head, shape.head_dim, call.split.max_splits)?;
        require_fit("part", &call.part, part_bytes, 16)?;
        let plane = Kv8Shape {
            rows: n,
            n_kv_head: shape.n_kv_head,
            head_dim: shape.head_dim,
            cells: shape.cells,
            code_scale: Kv8CodeScale::StoredHalf,
        };
        check_planes(call.rows, &plane, false)?;
        let proj_bytes = match &call.proj {
            Some(proj) => {
                if shape.ld < shape.n_head * 2 * shape.head_dim {
                    return Err(rule("shape", "ld does not hold the gate columns"));
                }
                let bytes = ((n - 1) * shape.ld + shape.n_head * 2 * shape.head_dim) * f;
                require_fit("proj", proj, bytes, 4)?;
                bytes
            }
            None => 0,
        };
        let out = Claim::of("out", &call.out, n * width * f);
        let part = Claim::of("part", &call.part, part_bytes);
        require_apart(&out, &part)?;
        require_apart(&out, &Claim::of("q", &call.q, n * width * f))?;
        require_apart(&part, &Claim::of("q", &call.q, n * width * f))?;
        if let Some(proj) = &call.proj {
            let claim = Claim::of("proj", proj, proj_bytes);
            require_apart(&out, &claim)?;
            require_apart(&part, &claim)?;
        }
        let raw = DecodeMultiRaw {
            q: self.raw_range(&call.q)?,
            proj: call.proj.as_ref().map(|p| self.raw_range(p)).transpose()?,
            table: self.raw_range(&call.rows.table.range(0, n * KV_ROW_BYTES)?)?,
            part: self.raw_range(&call.part)?,
            out: self.raw_range(&call.out)?,
            n,
            ld: shape.ld,
            heads: [shape.n_head, shape.n_kv_head],
            head_dim: shape.head_dim,
            cells: shape.cells,
            split: [call.split.max_splits, call.split.min_keys],
            span: [call.span.window as usize, call.span.ring as usize],
            scale: call.scale,
        };
        self.launch_on(stream, move |raw_stream| {
            attn_decode_q8_multi_launch(raw_stream, raw)
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
