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

    /// Fill a packed host record. Validation precedes reads; an I/O error can still leave
    /// an already-read prefix in `out`, so callers must discard the whole record on error.
    pub(in super::super::super) fn read(
        &self,
        model: &GgufModel,
        layer: usize,
        expert: u64,
        out: &mut [u8],
    ) -> Result<(), EngError> {
        let bad = || EngError::BadRequest("invalid GGUF expert record request".into());
        if out.len() != self.bytes || *self != Self::for_layer(model, layer)? {
            return Err(bad());
        }
        let mut rows = [0u64; 3];
        for (i, part) in self.parts.iter().enumerate() {
            let tensor = model.map().get(part.role, Some(layer)).ok_or_else(bad)?;
            let [_, outputs, _] = tensor.dims.as_slice() else {
                return Err(bad());
            };
            let (_, bytes) = tensor.row_span(Some(expert), 0..*outputs)?;
            if bytes != part.bytes.len() {
                return Err(bad());
            }
            rows[i] = *outputs;
        }
        for (part, outputs) in self.parts.iter().zip(rows) {
            model.read_expert_rows(
                layer,
                part.role,
                expert,
                0..outputs,
                &mut out[part.bytes.clone()],
            )?;
        }
        Ok(())
    }
}
