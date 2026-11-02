//! Routed picks of one MoE layer served by the host CPU (see `flint::cpu_moe`).
//!
//! The layer walk asks [`serve`] once per layer after routing. When the configuration gives the
//! CPU some of the layer's picks, their expert outputs are computed here from the normed
//! activation rows read back from the device, and [`Served`] carries them to the two places the
//! walk needs them: the ids the device pass still runs, and the pair rows the device output
//! receives afterwards. Everything is stream-ordered behind the work already queued, and the
//! rows keep the published (bf16-rounded) precision the device expert outputs have.

use half::bf16;

use super::{CudaContext, DeviceStream, FfnBlock, FlintExpertBank, Frame, TOP_K, bad};
use crate::{EngError, family::flint::cpu_moe, family::flint::cuda_ffn::plan::Grouping};

/// Picks the CPU serves for one layer and their computed rows.
pub(super) struct Served {
    /// Indices into the layer's `ids` (row-major, `TOP_K` per row), ascending.
    pairs: Vec<usize>,
    /// `pairs.len()` rows of `hidden` unweighted expert outputs.
    rows: Vec<f32>,
    /// Ids and grouping for the device pass when it still has resident picks to run.
    device: Option<(Vec<i32>, Grouping)>,
    hidden: usize,
}

impl Served {
    /// Whether the CPU serves every pick, so the device pass is skipped.
    pub(super) fn all(&self) -> bool {
        self.device.is_none()
    }

    /// The ids and grouping the device pass runs.
    pub(super) fn device(&self) -> Option<(&[i32], &Grouping)> {
        self.device.as_ref().map(|(ids, g)| (ids.as_slice(), g))
    }

    fn bytes(&self) -> Vec<u8> {
        self.rows
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_f32().to_le_bytes())
            .collect()
    }

    /// The paired output when the CPU served every pick: one buffer of `pairs` rows.
    pub(super) fn into_down(self, frame: &mut Frame<'_, '_>) -> Result<usize, EngError> {
        let bytes = self.bytes();
        let buffer = frame.ctx.upload(&bytes)?;
        Ok(frame.keep(buffer, bytes.len()))
    }

    /// Copies the CPU rows into the paired device output `down` at their pair positions.
    pub(super) fn overlay(&self, frame: &mut Frame<'_, '_>, down: usize) -> Result<(), EngError> {
        let bytes = self.bytes();
        let source = frame.ctx.upload(&bytes)?;
        let source = frame.keep(source, bytes.len());
        let row_bytes = self.hidden * 4;
        for (i, &pair) in self.pairs.iter().enumerate() {
            frame.ctx.copy_device(
                frame.stream,
                frame.buffers[source].0.range(i * row_bytes, row_bytes)?,
                frame.buffers[down].0.range(pair * row_bytes, row_bytes)?,
            )?;
        }
        Ok(())
    }
}

/// Computes the picks of this layer the CPU serves, or `None` when it serves none.
pub(super) fn serve(
    frame: &mut Frame<'_, '_>,
    experts: &FlintExpertBank<'_>,
    x: usize,
    block: FfnBlock,
    ids: &[i32],
) -> Result<Option<Served>, EngError> {
    let Some(planned) = plan(experts, block, ids)? else {
        return Ok(None);
    };
    let rows = compute(frame, experts, x, block, ids, planned.pairs())?;
    Ok(Some(Served::from_rows(planned, rows, block.hidden)))
}

/// The host-only half of [`serve`]: which pair indices the CPU serves this layer and what the
/// device pass runs for the rest. No device access, no readback.
pub(super) fn plan(
    experts: &FlintExpertBank<'_>,
    block: FfnBlock,
    ids: &[i32],
) -> Result<Option<Planned>, EngError> {
    let Some(cpu) = cpu_moe::active() else {
        return Ok(None);
    };
    let Some((ordinal, resident)) = experts.resident()? else {
        return Err(bad("CPU experts need the pinned host expert store"));
    };
    let pairs = cpu.cpu_pairs(ordinal, ids, Some(&resident));
    if pairs.is_empty() {
        return Ok(None);
    }
    let device = if pairs.len() == ids.len() {
        None
    } else {
        let device_ids = cpu_moe::device_ids(ids, TOP_K, &pairs, &resident)
            .ok_or_else(|| bad("no free resident expert to stand in for a CPU pick"))?;
        let grouping = Grouping::new(&device_ids, block.rows, block.experts)?;
        Some((device_ids, grouping))
    };
    Ok(Some(Planned { pairs, device }))
}

/// The layer split [`plan`] decided: CPU pair indices, and the device pass when it still has
/// resident picks to run.
pub(super) struct Planned {
    pairs: Vec<usize>,
    device: Option<(Vec<i32>, Grouping)>,
}

impl Planned {
    pub(super) fn pairs(&self) -> &[usize] {
        &self.pairs
    }

    pub(super) fn device(&self) -> Option<(&[i32], &Grouping)> {
        self.device.as_ref().map(|(ids, g)| (ids.as_slice(), g))
    }

    /// `(row, expert id)` jobs for the CPU, in pair order.
    pub(super) fn row_pairs(&self, ids: &[i32]) -> Vec<(usize, usize)> {
        self.pairs
            .iter()
            .map(|&pair| (pair / TOP_K, ids[pair] as usize))
            .collect()
    }
}

impl Served {
    /// The [`Served`] a finished CPU computation of `planned` produces: `rows.len()` must be
    /// `planned.pairs().len() * hidden`.
    pub(super) fn from_rows(planned: Planned, rows: Vec<f32>, hidden: usize) -> Self {
        debug_assert_eq!(rows.len(), planned.pairs.len() * hidden);
        Served {
            pairs: planned.pairs,
            rows,
            device: planned.device,
            hidden,
        }
    }
}

fn compute(
    frame: &mut Frame<'_, '_>,
    experts: &FlintExpertBank<'_>,
    x: usize,
    block: FfnBlock,
    ids: &[i32],
    pairs: &[usize],
) -> Result<Vec<f32>, EngError> {
    let Some(cpu) = cpu_moe::active() else {
        return Err(bad("a planned CPU pass lost its CpuMoe"));
    };
    let jobs: Vec<(usize, usize)> = pairs
        .iter()
        .map(|&pair| (pair / TOP_K, ids[pair] as usize))
        .collect();
    let mut raw = vec![0u8; block.rows * block.hidden * 4];
    frame.buffers[x].0.read_at(frame.stream, 0, &mut raw)?;
    let xs: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let mut rows = vec![0.0f32; pairs.len() * block.hidden];
    experts.cpu_experts(|source| cpu.compute(source, &xs, &jobs, &mut rows))?;
    if rows.iter().any(|v| !v.is_finite()) {
        return Err(bad("a CPU expert produced a non-finite value"));
    }
    Ok(rows)
}

/// Feeds the layer's picks to the hot-set policy and applies the swap it proposes, behind the
/// work already queued on the stream. A no-op unless split mode is on.
pub(super) fn adapt(
    ctx: &CudaContext,
    stream: &DeviceStream,
    experts: &FlintExpertBank<'_>,
    ids: &[i32],
) -> Result<(), EngError> {
    let Some(cpu) = cpu_moe::active() else {
        return Ok(());
    };
    let Some((ordinal, resident)) = experts.resident()? else {
        return Ok(());
    };
    if let Some(swap) = cpu.observe(ordinal, ids, &resident) {
        experts.swap(ctx, stream, (swap.incoming as u16, swap.outgoing as u16))?;
    }
    Ok(())
}
