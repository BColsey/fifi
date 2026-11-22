//! Stored-byte layouts for GGUF expert records. These are not CUDA bank layouts:
//! a device consumer must separately account for its alignment and dispatch requirements.

use std::ops::Range;

use super::GgufModel;
use crate::{EngError, family::recipe::Role, weights::GgufType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super::super) struct ExpertPart {
    pub(in super::super::super) role: Role,
    pub(in super::super::super) ty: GgufType,
    pub(in super::super::super) bytes: Range<usize>,
}

/// A packed host record, with no implied CUDA alignment or allocation policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super::super) struct ExpertLayout {
    pub(in super::super::super) parts: [ExpertPart; 3],
    pub(in super::super::super) bytes: usize,
}

impl ExpertLayout {
    pub(in super::super::super) fn for_layer(
        model: &GgufModel,
        layer: usize,
    ) -> Result<Self, EngError> {
        let bad = || {
            EngError::MemoryPlan(format!(
                "Flash-Next GGUF expert layout invalid at layer {layer}"
            ))
        };
        let mut offset = 0usize;
        let mut parts = Vec::with_capacity(3);
        for role in [Role::ExpertGate, Role::ExpertUp, Role::ExpertDown] {
            let tensor = model.map().get(role, Some(layer)).ok_or_else(bad)?;
            let [_, outputs, experts] = tensor.dims.as_slice() else {
                return Err(bad());
            };
            if *experts == 0 {
                return Err(bad());
            }
            let (first, len) = tensor.row_span(Some(0), 0..*outputs)?;
            let (last, last_len) = tensor.row_span(Some(experts - 1), 0..*outputs)?;
            if first != 0 || len != last_len || last.checked_add(len as u64) != Some(tensor.bytes) {
                return Err(bad());
            }
            let end = offset.checked_add(len).ok_or_else(bad)?;
            parts.push(ExpertPart {
                role,
                ty: tensor.ty,
                bytes: offset..end,
            });
            offset = end;
        }
        Ok(Self {
            parts: parts.try_into().map_err(|_| bad())?,
            bytes: offset,
        })
    }
}
