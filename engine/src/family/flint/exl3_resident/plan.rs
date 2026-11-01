use std::{collections::HashMap, path::Path};

use super::{FfnRoles, FlintResidentLimits, LayerRoles, ResidentPlan};
use crate::{
    EngError,
    weights::exl3::{
        Checkpoint,
        flint_pack::{Geometry, Kind, Limits, Plan, RoutedScale, audit},
    },
};

pub(super) fn plan_resident(
    directory: &Path,
    checkpoint: &Checkpoint,
    geometry: &Geometry,
    limits: FlintResidentLimits,
) -> Result<ResidentPlan, EngError> {
    let layout = geometry.plan();
    let (stored_payload_bytes, converted_payload_bytes) = payload_bytes(&layout)?;
    if converted_payload_bytes > limits.max_weight_bytes {
        return Err(EngError::MemoryPlan(format!(
            "Flint EXL3 converted payload requires {converted_payload_bytes} bytes, budget {}",
            limits.max_weight_bytes
        )));
    }
    let mut audit_limits = Limits::default();
    audit_limits.batch_bytes = audit_limits.batch_bytes.min(limits.max_staging_bytes);
    audit_limits.matrix_bytes = audit_limits.matrix_bytes.min(limits.max_staging_bytes);
    audit(directory, checkpoint, geometry, &audit_limits)?;
    let factor = RoutedScale::from_model_config(&checkpoint.model_config)?.factor();
    let names = Names::new(&layout);
    let layers = (0..geometry.layers)
        .map(|layer| names.layer(geometry, layer, factor))
        .collect::<Result<_, _>>()?;
    let embedding = names.plain("model.embed_tokens.weight")?;
    let final_norm = names.plain("model.norm.weight")?;
    let head = names.matrix("lm_head")?;
    Ok(ResidentPlan {
        layout,
        layers,
        embedding,
        final_norm,
        head,
        stored_payload_bytes,
        converted_payload_bytes,
    })
}

fn payload_bytes(layout: &Plan) -> Result<(u64, u64), EngError> {
    let overflow = || EngError::MemoryPlan("Flint EXL3 payload byte count overflow".into());
    let (mut stored, mut converted) = (0u64, 0u64);
    for spec in layout.specs() {
        let raw = spec.bytes();
        let device = match spec.kind {
            Kind::Mul1 => 0,
            Kind::Trellis => raw,
            _ => raw.checked_mul(2).ok_or_else(overflow)?,
        };
        stored = stored.checked_add(raw).ok_or_else(overflow)?;
        converted = converted.checked_add(device).ok_or_else(overflow)?;
    }
    Ok((stored, converted))
}

// Index the existing schema once; expert banks must not perform quadratic name searches.
struct Names<'a> {
    matrices: HashMap<&'a str, usize>,
    plain: HashMap<&'a str, usize>,
}

impl<'a> Names<'a> {
    fn new(layout: &'a Plan) -> Self {
        Self {
            matrices: layout
                .modules
                .iter()
                .enumerate()
                .map(|(index, module)| (module.prefix.as_str(), index))
                .collect(),
            plain: layout
                .plain
                .iter()
                .enumerate()
                .map(|(index, tensor)| (tensor.name.as_str(), index))
                .collect(),
        }
    }

    fn matrix(&self, name: &str) -> Result<usize, EngError> {
        required(&self.matrices, name)
    }

    fn plain(&self, name: &str) -> Result<usize, EngError> {
        required(&self.plain, name)
    }

    fn expert(&self, prefix: &str) -> Result<[usize; 3], EngError> {
        Ok([
            self.matrix(&format!("{prefix}.gate_proj"))?,
            self.matrix(&format!("{prefix}.up_proj"))?,
            self.matrix(&format!("{prefix}.down_proj"))?,
        ])
    }

    fn ffn(&self, geometry: &Geometry, layer: usize, factor: f32) -> Result<FfnRoles, EngError> {
        let prefix = format!("model.layers.{layer}.mlp");
        if geometry.dense_layers.contains(&layer) {
            return Ok(FfnRoles::Dense(self.expert(&prefix)?));
        }
        let experts = (0..geometry.experts)
            .map(|expert| self.expert(&format!("{prefix}.experts.{expert}")))
            .collect::<Result<_, _>>()?;
        Ok(FfnRoles::Moe {
            router: self.plain(&format!("{prefix}.gate.weight"))?,
            experts,
            shared: self.expert(&format!("{prefix}.shared_expert"))?,
            routing_weight_multiplier: factor,
        })
    }

    fn layer(
        &self,
        geometry: &Geometry,
        layer: usize,
        factor: f32,
    ) -> Result<LayerRoles, EngError> {
        let prefix = format!("model.layers.{layer}");
        Ok(LayerRoles {
            qkvo: [
                self.matrix(&format!("{prefix}.self_attn.q_proj"))?,
                self.matrix(&format!("{prefix}.self_attn.k_proj"))?,
                self.matrix(&format!("{prefix}.self_attn.v_proj"))?,
                self.matrix(&format!("{prefix}.self_attn.o_proj"))?,
            ],
            norms: [
                self.plain(&format!("{prefix}.input_layernorm.weight"))?,
                self.plain(&format!("{prefix}.post_attention_layernorm.weight"))?,
                self.plain(&format!("{prefix}.self_attn.q_norm.weight"))?,
                self.plain(&format!("{prefix}.self_attn.k_norm.weight"))?,
            ],
            sinks: self.plain(&format!("{prefix}.self_attn.softmax_offset"))?,
            ffn: self.ffn(geometry, layer, factor)?,
        })
    }
}

fn required(names: &HashMap<&str, usize>, name: &str) -> Result<usize, EngError> {
    names
        .get(name)
        .copied()
        .ok_or_else(|| EngError::Checkpoint(format!("Flint EXL3 resident role {name} is missing")))
}

/// The offset-table entry of a member that is not on the device (`u32::MAX`, as the launch reads it).
pub(super) const ABSENT_MEMBER: u32 = u32::MAX;

/// Checks the host copy of a member offset table before it is written to the device: every
/// present offset is a multiple of 4 bytes and leaves one member's `needs` (trellis bytes, suh
/// floats, svh floats) inside the bank ranges `bank` (trellis bytes, suh bytes, svh bytes)
/// measured from member 0, where a single offset moves all three.
pub(super) fn check_member_offsets(
    offsets: &[u32],
    needs: [usize; 3],
    bank: [usize; 3],
) -> Result<(), EngError> {
    if offsets.is_empty() {
        return Err(offsets_error("member offset table is empty"));
    }
    for &offset in offsets.iter().filter(|&&offset| offset != ABSENT_MEMBER) {
        let offset = offset as usize;
        if !offset.is_multiple_of(4) {
            return Err(offsets_error(
                "a member offset is not a multiple of 4 bytes",
            ));
        }
        let fits = [
            (needs[0], bank[0]),
            (needs[1] * 4, bank[1]),
            (needs[2] * 4, bank[2]),
        ]
        .into_iter()
        .all(|(need, range)| offset.checked_add(need).is_some_and(|end| end <= range));
        if !fits {
            return Err(offsets_error("a member leaves the bank ranges"));
        }
    }
    Ok(())
}

fn offsets_error(message: &str) -> crate::EngError {
    crate::EngError::BadRequest(format!("Flint expert offsets: {message}"))
}

#[cfg(test)]
mod offset_tests {
    use super::*;

    #[test]
    fn member_offsets_must_be_aligned_present_or_absent_and_inside_the_bank() {
        // Three members of 100 bytes each: a 64-byte trellis then 4 + 4 floats (32 bytes).
        let needs = [64, 4, 4];
        let bank = [264, 232, 232];
        assert!(check_member_offsets(&[0, 100, 200], needs, bank).is_ok());
        assert!(check_member_offsets(&[200, ABSENT_MEMBER, 0], needs, bank).is_ok());
        assert!(check_member_offsets(&[ABSENT_MEMBER], needs, bank).is_ok());
        assert!(check_member_offsets(&[], needs, bank).is_err());
        assert!(
            check_member_offsets(&[0, 102], needs, bank).is_err(),
            "unaligned"
        );
        assert!(
            check_member_offsets(&[204], needs, bank).is_err(),
            "trellis leaves the range"
        );
        // At 216 the trellis fits (216 + 64 <= 300) but the suh floats end at 216 + 16 = 232,
        // one word past a 228-byte range.
        assert!(check_member_offsets(&[216], needs, [300, 228, 228]).is_err());
        assert!(check_member_offsets(&[216], needs, [300, 232, 232]).is_ok());
        assert!(check_member_offsets(&[u32::MAX - 1], needs, bank).is_err());
    }
}
