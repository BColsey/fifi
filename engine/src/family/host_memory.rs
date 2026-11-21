//! Host-memory planning shared by the MLX families: the share of unified memory a start may
//! use, the bytes of a checkpoint's routed experts read from its shard headers, and whether a
//! model, a session reserve and a saved prefix fit a budget. Pure functions, compiled on every
//! host so the settings of every family and the tests read the same numbers.

/// The bytes of a checkpoint's routed expert tensors, from the shard headers: all layers
/// together, and the largest layer, which is what one decode or prefill stage reads when the
/// experts stream from disk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExpertBytes {
    pub(crate) total_bytes: u64,
    pub(crate) stage_bytes: u64,
}

/// The resident bytes of a model whose routed experts stream: everything but the expert
/// tensors, plus one stage of experts.
pub(crate) fn staged_model_bytes(model_bytes: u64, experts: ExpertBytes) -> u64 {
    model_bytes
        .saturating_sub(experts.total_bytes)
        .saturating_add(experts.stage_bytes)
}

/// The share of unified memory a model and its sessions may use: the memory minus an OS floor
/// of one sixth, at least 4 GiB, rounded up to whole MiB.
pub(crate) fn host_budget(memory_bytes: u64) -> u64 {
    const MIB: u64 = 1 << 20;
    let floor = (memory_bytes / 6)
        .div_ceil(MIB)
        .saturating_mul(MIB)
        .max(4 << 30);
    memory_bytes.saturating_sub(floor)
}

/// The budget an MLX start with no device budget of its own sizes itself from: the host budget
/// of `memory_bytes` when a resident model of `model_bytes` does not fit it, and `None` when it
/// fits, so a start that fits is not sized and behaves as it always did. Every MLX family's
/// start reads this one rule.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(crate) fn implied_cap(model_bytes: u64, memory_bytes: u64) -> Option<u64> {
    let budget = host_budget(memory_bytes);
    (model_bytes > budget).then_some(budget)
}

#[cfg(target_os = "macos")]
pub(crate) fn physical_memory_bytes() -> Option<u64> {
    // The absolute path: a PATH lookup fails under a restricted PATH or sandbox.
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .filter(|out| out.status.success())?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn physical_memory_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib = text
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))?
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse::<u64>()
        .ok()?;
    kib.checked_mul(1024)
}

/// The layer index of a routed expert tensor name (`...layers.N...switch_mlp...`), which Qwen3.6
/// and Flint checkpoints share.
fn expert_layer(name: &str) -> Option<u32> {
    let (before, _) = name.split_once(".switch_mlp.")?;
    before
        .rsplit_once(".layers.")?
        .1
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// Reads the routed expert bytes of every top-level shard under `model_dir` (`None` when no
/// shard header is readable or the checkpoint has no expert stacks).
pub(crate) fn expert_bytes_in(model_dir: &std::path::Path) -> Option<ExpertBytes> {
    let mut per_layer: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut shards = 0u32;
    for entry in std::fs::read_dir(model_dir).ok()? {
        let path = entry.ok()?.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            expert_bytes_of_shard(&path, &mut per_layer)?;
            shards += 1;
        }
    }
    let total_bytes = per_layer.values().sum::<u64>();
    (shards > 0 && total_bytes > 0).then(|| ExpertBytes {
        total_bytes,
        stage_bytes: per_layer.values().copied().max().unwrap_or(0),
    })
}

fn expert_bytes_of_shard(
    path: &std::path::Path,
    per_layer: &mut std::collections::BTreeMap<u32, u64>,
) -> Option<()> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length).ok()?;
    let length = usize::try_from(u64::from_le_bytes(length)).ok()?;
    if length > 64 << 20 {
        return None;
    }
    let mut header = vec![0u8; length];
    file.read_exact(&mut header).ok()?;
    let header: serde_json::Value = serde_json::from_slice(&header).ok()?;
    for (name, tensor) in header.as_object()? {
        let Some(layer) = expert_layer(name) else {
            continue;
        };
        let offsets = tensor.get("data_offsets")?.as_array()?;
        let (start, end) = (offsets.first()?.as_u64()?, offsets.get(1)?.as_u64()?);
        *per_layer.entry(layer).or_default() += end.checked_sub(start)?;
    }
    Some(())
}

/// The summed length of the top-level `*.safetensors` files, as the loader opens them.
pub(crate) fn shard_bytes(model_dir: &std::path::Path) -> Option<u64> {
    let mut total = 0u64;
    let mut shards = 0u32;
    for entry in std::fs::read_dir(model_dir).ok()? {
        let path = entry.ok()?.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            total = total.checked_add(std::fs::metadata(&path).ok()?.len())?;
            shards += 1;
        }
    }
    (shards > 0).then_some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_bytes_come_from_the_shard_headers_per_layer() {
        let dir = std::env::temp_dir().join(format!("mlx-budget-experts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = serde_json::json!({
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": {"dtype": "U32", "shape": [1], "data_offsets": [0, 100]},
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": {"dtype": "U32", "shape": [1], "data_offsets": [100, 150]},
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight": {"dtype": "U32", "shape": [1], "data_offsets": [150, 400]},
            "language_model.model.layers.1.self_attn.q_proj.weight": {"dtype": "U32", "shape": [1], "data_offsets": [400, 900]},
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(dir.join("a.safetensors"), bytes).unwrap();
        let experts = expert_bytes_in(&dir).unwrap();
        assert_eq!((experts.total_bytes, experts.stage_bytes), (400, 250));
        assert_eq!(staged_model_bytes(1000, experts), 1000 - 400 + 250);
        std::fs::write(dir.join("a.safetensors"), [0u8; 4]).unwrap();
        assert_eq!(expert_bytes_in(&dir), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_host_budget_leaves_a_sixth_and_at_least_four_gib() {
        const GIB: u64 = 1 << 30;
        assert_eq!(host_budget(36 * GIB), 30 * GIB);
        assert_eq!(host_budget(16 * GIB), 12 * GIB);
        assert_eq!(host_budget(24 * GIB), 20 * GIB);
    }

    #[test]
    fn a_start_sizes_itself_from_the_host_only_when_the_resident_model_does_not_fit() {
        const GIB: u64 = 1 << 30;
        // The shard bytes of the Qwen3.6 MLX 4-bit checkpoint (19.0 GiB) and of the Flint INT4
        // checkpoint (21.05 GiB) against the host budgets of 16, 24, 36 and 64 GiB hosts
        // (12, 20, 30 and 53.3 GiB). Qwen3.6 on a 24 GiB host fits by its shards alone; its host
        // plan adds the session reserve and streams the experts from `mlx_staged_cap` instead.
        for (name, model, sized) in [
            ("qwen36", 20_401_094_656u64, [true, false, false, false]),
            ("flint", 22_600_000_000, [true, true, false, false]),
            ("large", 80 * GIB, [true, true, true, true]),
        ] {
            for (memory, want) in [16, 24, 36, 64].into_iter().zip(sized) {
                assert_eq!(
                    implied_cap(model, memory * GIB).is_some(),
                    want,
                    "{name} on {memory} GiB"
                );
            }
        }
        assert_eq!(implied_cap(22 * GIB, 24 * GIB), Some(20 * GIB));
    }
}
