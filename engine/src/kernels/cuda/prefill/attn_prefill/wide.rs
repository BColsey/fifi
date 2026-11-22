//! The head_dim 512 prefill attention as one caller for every family whose global layers have
//! that head dimension over Q8 affine KV planes (`kv8_chunk` / `kv8_append` layout): the plane
//! views, the fixed-stride split plan, the partial buffer, the device gate and the override.
//! A family supplies its geometry and its plane buffers; it keeps its own row path for the
//! chunks this caller does not take.
//!
//! The kernel packs four query heads of a kv head with eight rows into one tile. A layer with
//! `n_head / 4` kv heads is read as it is; a layer with a single kv head and `n_head` a multiple
//! of four is read as `n_head / 4` virtual kv heads whose planes have head stride 0, so every
//! virtual head reads the one real head.

use super::{
    AttnKvView, AttnPrefillCall, AttnPrefillShape, CPF_ATTN_Q_TILE, CPF_ATTN_WIDE_HEAD_DIM,
    CpfAttnGeometry, CpfAttnPlan, CpfAttnSpan, cpf_attn_part_bytes, cpf_attn_wide_split,
    kv8_plane_view, rule,
};
use crate::EngError;
use crate::kernels::cuda::DeviceBuffer;
use crate::kernels::cuda::context::{CudaContext, DeviceRange, DeviceStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The environment variable that overrides the wide prefill attention: `0` or `off` keeps a
/// family on its row path. Unset (or `1`, `on`) leaves the choice to the device gate.
pub(crate) const WIDE_PREFILL_VAR: &str = "FIFI_WIDE_PREFILL";

/// Query heads of one kv head the kernel packs into a tile.
const GROUP: usize = 4;

/// What a family's global layers look like to the wide kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WideGeometry {
    /// Query heads of a global layer.
    pub(crate) n_head: usize,
    /// Kv heads of a global layer's planes: `n_head / 4`, or 1.
    pub(crate) n_kv_head: usize,
    /// Cells of each kv head's plane.
    pub(crate) cells: usize,
    /// The score scale of the family's attention.
    pub(crate) scale: f32,
}

impl WideGeometry {
    /// The kv heads the kernel sees: `n_head / 4`, each reading its own plane or, for a single
    /// real kv head, the shared one.
    fn virtual_kv(self) -> Result<usize, EngError> {
        let virtual_kv = self.n_head / GROUP;
        if self.n_head == 0 || !self.n_head.is_multiple_of(GROUP) {
            return Err(rule(
                "wide geometry",
                &format!("{} query heads are not a multiple of {GROUP}", self.n_head),
            ));
        }
        if self.n_kv_head != virtual_kv && self.n_kv_head != 1 {
            return Err(rule(
                "wide geometry",
                &format!(
                    "{} query heads over {} kv heads: the kernel needs {virtual_kv} kv heads or one",
                    self.n_head, self.n_kv_head
                ),
            ));
        }
        if self.cells == 0 {
            return Err(rule("wide geometry", "no cells"));
        }
        Ok(virtual_kv)
    }

    fn kernel_geometry(self, rows: usize) -> Result<CpfAttnGeometry, EngError> {
        Ok(CpfAttnGeometry {
            m: rows,
            n_head: self.n_head,
            n_kv_head: self.virtual_kv()?,
            head_dim: CPF_ATTN_WIDE_HEAD_DIM,
        })
    }

    /// The admission charge of the partial buffer: its bytes when the device gate passed, else 0.
    pub(crate) fn charge(self, passed: bool) -> Option<u64> {
        if passed {
            self.part_bytes().ok().map(|bytes| bytes as u64)
        } else {
            Some(0)
        }
    }

    /// Bytes of the partial buffer of one launch: a query tile of rows over `cells` cells. A call of
    /// more rows runs as consecutive launches of one tile each, so the buffer does not grow with
    /// the rows.
    pub(crate) fn part_bytes(self) -> Result<usize, EngError> {
        let split = cpf_attn_wide_split(self.cells)?;
        cpf_attn_part_bytes(
            self.kernel_geometry(CPF_ATTN_Q_TILE)?,
            CpfAttnPlan {
                q_tile: CPF_ATTN_Q_TILE,
                max_splits: split.max_splits,
                min_keys: split.min_keys,
            },
        )
    }
}

/// What the load decided about the wide prefill attention, and why.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum WideGate {
    /// The device runs the kernel and nothing turned it off.
    On,
    /// The environment turned it off.
    EnvOff,
    /// The device or the build cannot run the kernel.
    DeviceCannot,
    /// The device query failed; the family stays on its row path.
    QueryFailed(String),
    /// The family's geometry is not one the kernel serves.
    Geometry(String),
}

impl WideGate {
    pub(crate) fn is_on(&self) -> bool {
        *self == Self::On
    }

    /// One line for the start log.
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::On => "on (the device runs the head_dim 512 kernel)".to_owned(),
            Self::EnvOff => format!("off ({WIDE_PREFILL_VAR} turned it off)"),
            Self::DeviceCannot => "off (this device cannot run the head_dim 512 kernel)".to_owned(),
            Self::QueryFailed(why) => format!("off (the device query failed: {why})"),
            Self::Geometry(why) => format!("off ({why})"),
        }
    }
}

/// Whether a start with no override takes the wide path. `FIFI_WIDE_PREFILL=0` or `off`
/// refuses the path, and a card that fails the gate falls back to row attention.
pub(crate) const WIDE_PREFILL_DEFAULT_ON: bool = true;

/// The override text: `Some(false)` for `0` and `off`, `Some(true)` for `1` and `on`, `None` when
/// unset; anything else is an error, so a typo does not silently change the path.
pub(crate) fn wide_prefill_override_from(raw: Option<&str>) -> Result<Option<bool>, EngError> {
    match raw {
        None => Ok(None),
        Some("1") | Some("on") => Ok(Some(true)),
        Some("0") | Some("off") => Ok(Some(false)),
        Some(other) => Err(EngError::BadRequest(format!(
            "{WIDE_PREFILL_VAR} accepts 1, on, 0 or off; got {other:?}"
        ))),
    }
}

/// Whether the path starts on: the override, else [`WIDE_PREFILL_DEFAULT_ON`].
pub(crate) fn wide_prefill_start_on_from(raw: Option<&str>) -> Result<bool, EngError> {
    Ok(wide_prefill_override_from(raw)?.unwrap_or(WIDE_PREFILL_DEFAULT_ON))
}

/// The start state from the environment.
pub(crate) fn wide_prefill_start_on() -> Result<bool, EngError> {
    wide_prefill_start_on_from(std::env::var(WIDE_PREFILL_VAR).ok().as_deref())
}

/// The gate from its three facts: the override, the geometry check and the device answer.
pub(crate) fn wide_gate_from(
    allowed: bool,
    geometry: Result<(), EngError>,
    supported: Result<bool, EngError>,
) -> WideGate {
    if !allowed {
        return WideGate::EnvOff;
    }
    if let Err(error) = geometry {
        return WideGate::Geometry(error.describe());
    }
    match supported {
        Ok(true) => WideGate::On,
        Ok(false) => WideGate::DeviceCannot,
        Err(error) => WideGate::QueryFailed(error.describe()),
    }
}

/// Reads the override and the device and decides, once, at load. A device query that fails is
/// reported in the gate and leaves the family on its row path; only a malformed override is an
/// error.
pub(crate) fn wide_prefill_gate(
    ctx: &CudaContext,
    geometry: WideGeometry,
) -> Result<WideGate, EngError> {
    let allowed =
        wide_prefill_override_from(std::env::var(WIDE_PREFILL_VAR).ok().as_deref())? != Some(false);
    let checked = geometry.part_bytes().map(|_| ());
    Ok(wide_gate_from(
        allowed,
        checked,
        ctx.attn_prefill_wide_supported(),
    ))
}

/// The planes of one layer and the rows of one call.
pub(crate) struct WideCall<'a> {
    /// K codes, K scales, V codes and V scales.
    pub(crate) planes: [&'a DeviceBuffer; 4],
    /// f32 `[rows][n_head][512]`, after norm and rope.
    pub(crate) q: DeviceRange<'a>,
    /// f32 `[rows][n_head * 512]`.
    pub(crate) out: DeviceRange<'a>,
    pub(crate) rows: usize,
    /// The cell of the first row; its keys are cells `0 ..= p0 + row`.
    pub(crate) p0: usize,
    /// The cells of each head's plane when this sequence's planes are not the geometry's size: a
    /// session sized to its request. The key split stays the geometry's, so a row's output does
    /// not depend on how many cells its session holds.
    cells: Option<usize>,
}

impl<'a> WideCall<'a> {
    pub(crate) fn new(
        planes: [&'a DeviceBuffer; 4],
        q: DeviceRange<'a>,
        out: DeviceRange<'a>,
        rows: usize,
        p0: usize,
    ) -> Self {
        Self {
            planes,
            q,
            out,
            rows,
            p0,
            cells: None,
        }
    }

    /// The call over planes of `cells` cells per head.
    pub(crate) fn with_cells(mut self, cells: usize) -> Self {
        self.cells = Some(cells);
        self
    }
}

/// What a family's load decided about the wide prefill attention: the gate, the start state and
/// the line it logged.
pub(crate) struct WideLoad {
    gate: WideGate,
    start_on: bool,
}

impl WideLoad {
    /// Reads the override and the device against the largest geometry the family can plan, and
    /// logs the result under `family`.
    pub(crate) fn read(
        ctx: &CudaContext,
        family: &str,
        largest: WideGeometry,
    ) -> Result<Self, EngError> {
        let gate = wide_prefill_gate(ctx, largest)?;
        let start_on = wide_prefill_start_on()?;
        eprintln!(
            "{family}: wide prefill attention for the global layers {}, starts {}",
            gate.describe(),
            if start_on { "on" } else { "off" }
        );
        Ok(Self { gate, start_on })
    }

    /// Whether the device gate passed, so the family builds the attention and charges its buffer.
    pub(crate) fn passed(&self) -> bool {
        self.gate.is_on()
    }

    /// Builds the attention for the geometry the family settled on, when the gate passed.
    pub(crate) fn build(
        &self,
        ctx: &CudaContext,
        geometry: WideGeometry,
    ) -> Result<Option<WideAttention>, EngError> {
        if !self.passed() {
            return Ok(None);
        }
        WideAttention::new(ctx, geometry, self.start_on).map(Some)
    }
}

impl WideGeometry {
    /// The geometry of a family whose attention scores are unscaled (the norm carries the scale).
    pub(crate) fn unscaled(n_head: usize, n_kv_head: usize, cells: usize) -> Self {
        Self {
            n_head,
            n_kv_head,
            cells,
            scale: 1.0,
        }
    }
}

/// The wide prefill attention of one model: its partial buffer, the switch a comparison flips,
/// and a launch counter.
pub(crate) struct WideAttention {
    part: DeviceBuffer,
    geometry: WideGeometry,
    on: AtomicBool,
    launches: AtomicU64,
}

impl WideAttention {
    /// Allocates the partial buffer; `on` is the start state. The caller has seen the gate pass.
    pub(crate) fn new(
        ctx: &CudaContext,
        geometry: WideGeometry,
        on: bool,
    ) -> Result<Self, EngError> {
        Ok(Self {
            part: ctx.allocate(geometry.part_bytes()?)?,
            geometry,
            on: AtomicBool::new(on),
            launches: AtomicU64::new(0),
        })
    }

    /// Whether calls should take this path now.
    pub(crate) fn is_on(&self) -> bool {
        self.on.load(Ordering::Relaxed)
    }

    /// Turns the path on or off for this model, for a comparison on one load.
    pub(crate) fn set_on(&self, on: bool) {
        self.on.store(on, Ordering::Relaxed);
    }

    /// How many launches this model has queued.
    pub(crate) fn launches(&self) -> u64 {
        self.launches.load(Ordering::Relaxed)
    }

    /// Runs the attention when `wide` holds one that is on and `take` is true, and says so;
    /// `false` leaves the family on its own path.
    pub(crate) fn try_attend(
        wide: Option<&Self>,
        take: bool,
        ctx: &CudaContext,
        stream: &DeviceStream,
        call: WideCall<'_>,
    ) -> Result<bool, EngError> {
        match wide.filter(|wide| take && wide.is_on()) {
            Some(wide) => wide.attend(ctx, stream, call).map(|()| true),
            None => Ok(false),
        }
    }

    /// The gate, the start state and the build in one step, for a family whose cells are fixed.
    pub(crate) fn load(
        ctx: &CudaContext,
        family: &str,
        geometry: WideGeometry,
    ) -> Result<Option<Self>, EngError> {
        WideLoad::read(ctx, family, geometry)?.build(ctx, geometry)
    }

    fn view<'b>(
        &self,
        codes: &'b DeviceBuffer,
        scales: &'b DeviceBuffer,
        cells: usize,
    ) -> Result<AttnKvView<'b>, EngError> {
        let g = self.geometry;
        let mut view = kv8_plane_view(
            codes.range(0, codes.bytes())?,
            scales.range(0, scales.bytes())?,
            cells,
            CPF_ATTN_WIDE_HEAD_DIM,
        );
        if g.n_kv_head != g.virtual_kv()? {
            // One real head read by every virtual head: stride 0 over both planes.
            view.layout.payload_strides.head_stride_bytes = 0;
            if let Some(plane) = view.layout.scales.as_mut() {
                plane.strides.head_stride_bytes = 0;
            }
            let extents = view
                .layout
                .extents(cells, g.virtual_kv()?, CPF_ATTN_WIDE_HEAD_DIM)?;
            view.payload = codes.range(0, extents.payload_bytes)?;
            view.scales = Some(scales.range(0, extents.scale_bytes.unwrap_or(0))?);
        }
        Ok(view)
    }

    /// Queues the attention of `rows` consecutive rows of one sequence over its KV8 planes in one
    /// launch. The rows' own cells were appended on this stream just before. A row's output is a
    /// function of its position and the cache alone: the key splits are a fixed stride over the
    /// cache, so any grouping of the rows into calls gives the same bytes.
    pub(crate) fn attend(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        call: WideCall<'_>,
    ) -> Result<(), EngError> {
        let g = self.geometry;
        if call.rows == 0 {
            return Err(rule("wide rows", "no rows"));
        }
        // One launch takes one query tile of rows; the rows of a chunk are independent, so the
        // chunk runs as consecutive launches with the bytes of one launch.
        let row_bytes = g.n_head * CPF_ATTN_WIDE_HEAD_DIM * 4;
        fn part<'a>(
            range: DeviceRange<'a>,
            at: usize,
            bytes: usize,
        ) -> Result<DeviceRange<'a>, EngError> {
            range.buffer().range(range.offset() + at, bytes)
        }
        let mut done = 0;
        while done < call.rows {
            let rows = (call.rows - done).min(CPF_ATTN_Q_TILE);
            self.attend_tile(
                ctx,
                stream,
                WideCall {
                    planes: call.planes,
                    q: part(call.q, done * row_bytes, rows * row_bytes)?,
                    out: part(call.out, done * row_bytes, rows * row_bytes)?,
                    rows,
                    p0: call.p0 + done,
                },
            )?;
            done += rows;
        }
        Ok(())
    }

    fn attend_tile(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        call: WideCall<'_>,
    ) -> Result<(), EngError> {
        let g = self.geometry;
        let [k_codes, k_scales, v_codes, v_scales] = call.planes;
        let cells = call.cells.unwrap_or(g.cells);
        let views = [
            self.view(k_codes, k_scales, cells)?,
            self.view(v_codes, v_scales, cells)?,
        ];
        let split = cpf_attn_wide_split(g.cells)?;
        let part_bytes = cpf_attn_part_bytes(
            g.kernel_geometry(call.rows)?,
            CpfAttnPlan {
                q_tile: CPF_ATTN_Q_TILE,
                max_splits: split.max_splits,
                min_keys: split.min_keys,
            },
        )?;
        self.launches.fetch_add(1, Ordering::Relaxed);
        ctx.attn_prefill(
            stream,
            AttnPrefillCall {
                storage: views[0].layout.storage,
                shape: AttnPrefillShape {
                    m: call.rows,
                    n_head: g.n_head,
                    n_kv_head: g.virtual_kv()?,
                    head_dim: CPF_ATTN_WIDE_HEAD_DIM,
                    cache_rows: cells,
                    ld: 0,
                    p0: i64::try_from(call.p0)
                        .map_err(|_| rule("wide rows", "the first position overflows"))?,
                },
                span: CpfAttnSpan::FULL,
                split,
                q_tile: CPF_ATTN_Q_TILE,
                scale: g.scale,
                q: call.q,
                proj: None,
                k_cache: views[0].payload,
                v_cache: views[1].payload,
                kv_views: Some(views),
                part: self.part.range(0, part_bytes)?,
                out: call.out,
                status: None,
                sinks: None,
            },
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::prefill::attn_prefill::{
        CPF_ATTN_PART_CAP, cpf_attn_wide_window_split,
    };

    #[test]
    fn the_stride_plan_of_a_32k_cache_has_128_splits_of_256_keys() {
        for cells in [32_767, 32_768] {
            let plan = cpf_attn_wide_split(cells).expect("plan");
            assert_eq!((plan.min_keys, plan.max_splits), (256, 128), "{cells}");
        }
        let plan = cpf_attn_wide_split(4_096).expect("plan");
        assert_eq!((plan.min_keys, plan.max_splits), (128, 32));
        let plan = cpf_attn_wide_split(129).expect("plan");
        assert_eq!((plan.min_keys, plan.max_splits), (128, 2));
        assert!(cpf_attn_wide_split(0).is_err());
    }

    #[test]
    fn a_window_plan_holds_the_window_a_tile_and_the_rounding_stride() {
        let plan = cpf_attn_wide_window_split(512).expect("plan");
        assert_eq!(plan.min_keys, 128);
        assert!(plan.max_splits * plan.min_keys >= 512 + 8 + 127);
        // One split fewer would not reach the last visible key of the worst tile.
        assert!((plan.max_splits - 1) * plan.min_keys < 512 + 8 + 127);
    }

    fn geometry(n_head: usize, n_kv_head: usize) -> WideGeometry {
        WideGeometry {
            n_head,
            n_kv_head,
            cells: 32_767,
            scale: 1.0,
        }
    }

    #[test]
    fn the_partial_buffer_of_the_bolt_and_the_gemma_geometries_fits_the_cap() {
        // 8 heads over 2 kv heads, and 16 heads over one kv head read as four.
        assert_eq!(geometry(8, 2).part_bytes().expect("bolt"), 16_842_752);
        let gemma = geometry(16, 1).part_bytes().expect("gemma");
        assert_eq!(gemma, 33_685_504);
        assert!(gemma < CPF_ATTN_PART_CAP);
    }

    #[test]
    fn a_geometry_the_kernel_does_not_serve_is_refused() {
        assert!(geometry(6, 1).part_bytes().is_err());
        assert!(geometry(16, 2).part_bytes().is_err());
        assert!(geometry(0, 0).part_bytes().is_err());
    }

    #[test]
    fn a_start_takes_the_override_and_otherwise_the_held_default() {
        assert_eq!(
            wide_prefill_start_on_from(None).expect("unset"),
            WIDE_PREFILL_DEFAULT_ON
        );
        for on in [Some("1"), Some("on")] {
            assert!(wide_prefill_start_on_from(on).expect("on"));
        }
        for off in [Some("0"), Some("off")] {
            assert!(!wide_prefill_start_on_from(off).expect("off"));
            assert_eq!(wide_prefill_override_from(off).expect("off"), Some(false));
        }
        assert_eq!(wide_prefill_override_from(None).expect("unset"), None);
        assert!(wide_prefill_start_on_from(Some("yes")).is_err());
        assert!(wide_prefill_start_on_from(Some("")).is_err());
    }

    #[test]
    fn the_gate_is_off_for_the_override_the_geometry_and_the_device_and_never_fails_the_load() {
        let fail = || EngError::BadRequest("no device".to_owned());
        assert_eq!(wide_gate_from(true, Ok(()), Ok(true)), WideGate::On);
        assert_eq!(wide_gate_from(false, Ok(()), Ok(true)), WideGate::EnvOff);
        assert_eq!(
            wide_gate_from(true, Ok(()), Ok(false)),
            WideGate::DeviceCannot
        );
        assert!(matches!(
            wide_gate_from(true, Err(fail()), Ok(true)),
            WideGate::Geometry(_)
        ));
        let failed = wide_gate_from(true, Ok(()), Err(fail()));
        assert!(matches!(&failed, WideGate::QueryFailed(why) if why.contains("no device")));
        for gate in [failed, WideGate::DeviceCannot, WideGate::EnvOff] {
            assert!(!gate.is_on());
            assert!(gate.describe().starts_with("off ("));
        }
        assert!(WideGate::On.describe().starts_with("on ("));
    }
}
