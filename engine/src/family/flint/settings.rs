//! Explicit options of the first CUDA serving profile; legacy MLX entries omit them.

use super::kv_plan::{KvPlan, KvStorage};
use crate::EngError;
use serde::Deserialize;

// Above eight rows EXL3 changes kernels and reduction order; this is a numeric cap.
pub(crate) const PREFILL_MAX_ROWS: usize = 8;

pub(crate) fn default_chunk_size() -> u32 {
    8
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlintOptions {
    pub(crate) context_limit: u32,
    pub(crate) kv_profile: String,
    #[serde(default = "default_chunk_size")]
    pub(crate) prefill_chunk_size: u32,
    /// Routed experts of the first N MoE layers run on the CPU (999 = all); default off.
    #[serde(default)]
    #[allow(dead_code, reason = "parsed for the config check; no start reads it")]
    pub(crate) cpu_moe_offload_layers: u32,
    /// Experts per layer served on the CPU instead of the device; default off, exclusive with
    /// `cpu_moe_offload_layers`.
    #[serde(default)]
    #[allow(dead_code, reason = "parsed for the config check; no start reads it")]
    pub(crate) cpu_moe_split_experts: u32,
    /// CPU expert worker threads; 0 picks the default.
    #[serde(default)]
    #[allow(dead_code, reason = "parsed for the config check; no start reads it")]
    pub(crate) cpu_moe_threads: u32,
}

impl FlintOptions {
    pub(crate) fn plan(&self) -> Result<KvPlan, EngError> {
        self.plan_rows(self.prefill_chunk_size as usize)
    }

    /// [`Self::plan`] with rings sized for appends of `rows` rows, which a layer-major jobs pass
    /// makes in one piece. `rows` is at least the prefill chunk size.
    pub(crate) fn plan_rows(&self, rows: usize) -> Result<KvPlan, EngError> {
        let (k_storage, v_storage) = match self.kv_profile.as_str() {
            "f16" => (KvStorage::F16, KvStorage::F16),
            "q8" => (KvStorage::Q8G32, KvStorage::Q8G32),
            // K-only q8, V f16: the mixed discriminator profile.
            "q8k" => (KvStorage::Q8G32, KvStorage::F16),
            _ => {
                return Err(EngError::BadRequest(
                    "Flint CUDA requires explicit kv_profile f16, q8 or q8k".into(),
                ));
            }
        };
        if !(1..=PREFILL_MAX_ROWS).contains(&(self.prefill_chunk_size as usize)) {
            return Err(EngError::BadRequest(
                "Flint CUDA prefill_chunk_size must be in 1..=8".into(),
            ));
        }
        KvPlan::new(
            self.context_limit as usize,
            rows.max(self.prefill_chunk_size as usize),
            true,
        )
        .map(|plan| plan.with_kv_storage(k_storage, v_storage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_q8_profile_is_opt_in_and_charges_payload_and_scales() {
        use crate::family::flint::kv_plan::KvStorage;

        let q8: FlintOptions =
            serde_json::from_str(r#"{"context_limit":4096,"kv_profile":"q8"}"#).unwrap();
        assert_eq!(q8.prefill_chunk_size, 8);
        let plan = q8.plan().unwrap();
        assert_eq!(plan.storage(), KvStorage::Q8G32);
        assert_eq!(plan.storage().row_bytes(), 1088);
        assert_eq!(plan.bytes(plan.storage().row_bytes()).unwrap(), 82_191_872);
        assert_eq!(plan.cache_rows(0).unwrap(), 4096);
        assert_eq!(plan.cache_rows(2).unwrap(), 520);

        let f16: FlintOptions =
            serde_json::from_str(r#"{"context_limit":4096,"kv_profile":"f16"}"#).unwrap();
        let control = f16.plan().unwrap();
        assert_eq!(control.storage(), KvStorage::F16);
        assert_eq!(
            control.bytes(control.storage().row_bytes()).unwrap(),
            154_714_112
        );
    }

    #[test]
    fn cuda_options_reject_missing_or_unknown_profile_keys() {
        for text in [
            r#"{"context_limit":128}"#,
            r#"{"context_limit":128,"kv_profile":"f16","typo":1}"#,
        ] {
            assert!(serde_json::from_str::<FlintOptions>(text).is_err());
        }
    }
}
