use crate::{EngError, weights::header::parse_header_json};

use super::Codebook;

#[derive(Debug, Clone)]
pub struct Config {
    version: String,
    pub(super) codebook: Option<Codebook>,
    expert_interm_div: u32,
}

impl Config {
    /// Conversion averages do not describe individual matrices.
    pub fn from_json(bytes: &[u8]) -> Result<Self, EngError> {
        if bytes.len() > 1_048_576 {
            return Err(EngError::MemoryPlan("EXL3 config exceeds 1 MiB".into()));
        }
        let root = parse_header_json(bytes, "EXL3 config")?;
        let config = root
            .get("quantization_config")
            .and_then(|v| v.as_object())
            .ok_or_else(|| invalid("quantization_config must be an object"))?;
        if config.get("quant_method").and_then(|v| v.as_str()) != Some("exl3") {
            return Err(invalid("quant_method must be exl3"));
        }
        let version = config
            .get("version")
            .and_then(|v| v.as_str())
            .ok_or_else(|| invalid("version is required"))?;
        let parts: Vec<_> = version.split('.').collect();
        // 1.4 to 1.6 are the layouts the reference decoder was checked against; 0.0.29 is the
        // 4.00-bpw Qwen3.6 pack, whose trellis, scale and marker tensors have the same layout
        // and is admitted by its exact version.
        if !matches!(parts.as_slice(), ["1", "4" | "5" | "6", patch]
            if !patch.is_empty() && patch.bytes().all(|v| v.is_ascii_digit()))
            && version != "0.0.29"
        {
            return Err(invalid(&format!("unsupported version {version:?}")));
        }
        let codebook = match config.get("codebook") {
            None => None,
            Some(value) => Some(match value.as_str() {
                Some("3inst") => Codebook::ThreeInst,
                Some("mcg") => Codebook::Mcg,
                Some("mul1") => Codebook::Mul1,
                _ => return Err(invalid("unknown codebook")),
            }),
        };
        let expert_interm_div = match config.get("expert_interm_div") {
            None => 1,
            Some(v) => match v.as_u64() {
                Some(v @ (1 | 128)) => v as u32,
                _ => return Err(invalid("unsupported expert_interm_div")),
            },
        };
        Ok(Self {
            version: version.into(),
            codebook,
            expert_interm_div,
        })
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    /// Family integration must retain this metadata; it is not folded into stored W.
    pub fn expert_interm_div(&self) -> u32 {
        self.expert_interm_div
    }
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("EXL3 config: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(version: &str, codebook: &str) -> Vec<u8> {
        format!(
            r#"{{"quantization_config":{{"quant_method":"exl3","version":"{version}","codebook":"{codebook}"}}}}"#
        )
        .into_bytes()
    }

    #[test]
    fn the_versions_of_the_checked_layouts_are_admitted_with_their_codebooks() {
        for (version, codebook, book) in [
            ("1.4.9", "mul1", Codebook::Mul1),
            ("1.5.0", "3inst", Codebook::ThreeInst),
            ("1.6.2", "mcg", Codebook::Mcg),
            ("0.0.29", "mcg", Codebook::Mcg),
        ] {
            let parsed = Config::from_json(&config(version, codebook)).unwrap();
            assert_eq!(parsed.version(), version);
            assert_eq!(parsed.codebook, Some(book));
        }
    }

    #[test]
    fn every_other_version_is_refused_by_name() {
        for version in [
            "0.0.28", "0.0.30", "0.0.290", "0.1.29", "1.3.9", "1.7.0", "2.0.0", "1.4", "1.4.",
            "1.4.x", "0.0.29.1", "",
        ] {
            let error = Config::from_json(&config(version, "mcg")).unwrap_err();
            assert!(
                matches!(&error, EngError::Checkpoint(m) if m.contains("unsupported version")),
                "{version:?}: {error:?}"
            );
        }
    }
}
