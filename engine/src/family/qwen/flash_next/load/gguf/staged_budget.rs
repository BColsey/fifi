//! Partial, allocation-free budget for the staged persistent routed owner.
//! This is not full-model admission: trunk, head, state, arena, policy heap and
//! transient allocations are not included.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "staged for GGUF decode admission")
)]

use std::ops::Range;

#[cfg(test)]
use super::tensors::{Form, Place};
use super::{
    GgufModel,
    expert_layout::{ExpertLayout, ExpertStacks},
};
use crate::EngError;

#[derive(Debug, PartialEq, Eq)]
pub(in super::super::super) struct RoutedBudget {
    pub stacks: usize,
    pub pinned_records: usize,
}

fn add_layer(total: &mut RoutedBudget, stack: usize, record: usize) -> Result<(), EngError> {
    let overflow = || EngError::BadRequest("GGUF staged routed budget overflow".into());
    let stacks = total.stacks.checked_add(stack).ok_or_else(overflow)?;
    let pinned_records = total
        .pinned_records
        .checked_add(record)
        .ok_or_else(overflow)?;
    *total = RoutedBudget {
        stacks,
        pinned_records,
    };
    Ok(())
}

/// Exact persistent stack and pinned-record payloads for the selected window.
/// Scratch and other allocations must be admitted separately.
pub(in super::super::super) fn routed_budget(
    model: &GgufModel,
    layers: Range<usize>,
    slots: usize,
) -> Result<RoutedBudget, EngError> {
    if layers.start >= layers.end || layers.end > model.config().layers() {
        return Err(EngError::BadRequest(
            "GGUF staged routed budget: invalid layer window".into(),
        ));
    }
    let dims = &model.config().dims;
    if slots <= dims.top_k as usize || slots > dims.experts as usize {
        return Err(EngError::BadRequest(
            "GGUF staged routed budget: invalid slot count".into(),
        ));
    }
    let mut total = RoutedBudget {
        stacks: 0,
        pinned_records: 0,
    };
    for layer in layers {
        let layout = ExpertLayout::for_layer(model, layer)?;
        let stack = ExpertStacks::new(&layout, slots)?;
        add_layer(&mut total, stack.bytes, layout.bytes)?;
    }
    Ok(total)
}

/// Checked mapped-form estimate for one group. This must be verified against actual uploads;
/// it is not itself a bound on driver overhead or future uploader changes.
#[cfg(test)]
pub(in super::super::super) fn upload_group_mapped(
    model: &GgufModel,
    layer: Option<usize>,
) -> Result<usize, EngError> {
    if layer.is_some_and(|layer| layer >= model.config().layers()) {
        return Err(EngError::BadRequest(
            "invalid GGUF mapped group layer".into(),
        ));
    }
    let overflow = || EngError::MemoryPlan("GGUF mapped group size overflows".into());
    let mut total = 0usize;
    for entry in model
        .map()
        .entries()
        .iter()
        .filter(|entry| entry.place == Place::Device && entry.layer == layer)
    {
        let elements = entry
            .dims
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
            .ok_or_else(overflow)?;
        let bytes = match entry.form {
            Form::Blocks => usize::try_from(entry.bytes).map_err(|_| overflow())?,
            Form::Half => elements.checked_mul(2).ok_or_else(overflow)?,
            Form::Float | Form::NormPlusOne | Form::Decay => {
                elements.checked_mul(4).ok_or_else(overflow)?
            }
            Form::None => 0,
        };
        total = total.checked_add(bytes).ok_or_else(overflow)?;
    }
    if total == 0 {
        return Err(EngError::BadRequest("empty GGUF mapped group".into()));
    }
    Ok(total)
}

/// Conservative one-group device payload for the current staged uploader: charge every
/// device-mapped entry in the group at the larger of its stored bytes and an F32 expansion.
/// PLE entries are deliberately included even though the staged trunk does not upload them.
/// This is not a bound on allocator overhead or temporary workspaces.
#[cfg(test)]
pub(in super::super::super) fn upload_group_upper(
    model: &GgufModel,
    layer: Option<usize>,
) -> Result<usize, EngError> {
    let overflow = || EngError::MemoryPlan("GGUF staged upload bound overflows".into());
    if layer.is_some_and(|layer| layer >= model.config().layers()) {
        return Err(EngError::BadRequest(
            "invalid GGUF upload bound layer".into(),
        ));
    }
    let mut entries = 0usize;
    let mut total = 0usize;
    for entry in model
        .map()
        .entries()
        .iter()
        .filter(|entry| entry.place == Place::Device && entry.layer == layer)
    {
        entries += 1;
        let elements = entry
            .dims
            .iter()
            .try_fold(1usize, |value, &dim| {
                value.checked_mul(usize::try_from(dim).ok()?)
            })
            .filter(|&value| value > 0)
            .ok_or_else(overflow)?;
        let expanded = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(overflow)?;
        let stored = usize::try_from(entry.bytes).map_err(|_| overflow())?;
        total = total
            .checked_add(expanded.max(stored))
            .ok_or_else(overflow)?;
    }
    if entries == 0 {
        return Err(EngError::BadRequest("empty GGUF upload group".into()));
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen::flash_next::load::gguf::{
        expert_layout::ExpertAllocation,
        testkit::{Spec, write_flash},
    };

    #[test]
    fn model_windows_sum_exact_stacks_and_each_pinned_record() {
        let file = write_flash(&Spec::default());
        let model = GgufModel::open(&file.first).unwrap().unwrap();
        let layers = model.config().layers();
        let slots = model.config().dims.top_k as usize + 1;
        let allocation = ExpertAllocation::for_model(&model, slots).unwrap();
        let full = routed_budget(&model, 0..layers, slots).unwrap();
        assert_eq!(full.stacks, allocation.device_bytes);
        let records: Vec<_> = (0..layers)
            .map(|layer| ExpertLayout::for_layer(&model, layer).unwrap().bytes)
            .collect();
        assert_eq!(full.pinned_records, records.iter().sum::<usize>());
        assert!(full.pinned_records > allocation.staging_bytes);
        let partial = routed_budget(&model, 1..3, slots).unwrap();
        assert_eq!(
            partial.stacks,
            allocation.layer_bytes[1..3].iter().sum::<usize>()
        );
        assert_eq!(partial.pinned_records, records[1..3].iter().sum::<usize>());
        assert!(routed_budget(&model, 0..0, slots).is_err());
        assert!(routed_budget(&model, 0..layers + 1, slots).is_err());
        assert!(routed_budget(&model, 0..1, slots - 1).is_err());
        assert!(routed_budget(&model, 0..1, model.config().dims.experts as usize + 1).is_err());
        for layer in 0..layers {
            let mapped: u64 = model
                .map()
                .entries()
                .iter()
                .filter(|entry| entry.place == Place::Device && entry.layer == Some(layer))
                .map(|entry| entry.device_bytes())
                .sum();
            assert_eq!(
                upload_group_mapped(&model, Some(layer)).unwrap() as u64,
                mapped
            );
            assert!(upload_group_upper(&model, Some(layer)).unwrap() as u64 >= mapped);
        }
        assert!(
            upload_group_upper(&model, None).unwrap() as u64
                >= model
                    .map()
                    .entries()
                    .iter()
                    .filter(|entry| entry.place == Place::Device && entry.layer.is_none())
                    .map(|entry| entry.device_bytes())
                    .sum::<u64>()
        );
        assert!(upload_group_upper(&model, Some(layers)).is_err());
    }

    #[test]
    fn sums_records_and_checks_overflow() {
        let mut total = RoutedBudget {
            stacks: 0,
            pinned_records: 0,
        };
        add_layer(&mut total, 20, 7).unwrap();
        add_layer(&mut total, 30, 11).unwrap();
        assert_eq!(
            total,
            RoutedBudget {
                stacks: 50,
                pinned_records: 18
            }
        );
        assert!(add_layer(&mut total, usize::MAX, 1).is_err());
        assert_eq!(total.stacks, 50);
    }
}
