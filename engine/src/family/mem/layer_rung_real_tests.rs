//! The layer rung over the real GGUF headers in `$HOME/models`: the plan of each model on each
//! card and host, against what the generic card-short rule did before, with the count of starts
//! that refused. Reads headers only (never a tensor payload), so it runs in seconds; it is
//! ignored by default because the files exist on the development host alone.

use super::HostMemory;
use super::layer_rung::{Home, Placement, RungBudget, RungModel, place};
use crate::family::qwen::qwen36::card_short::{CardFigures, CardVerdict, decide};
use crate::weights::{GgufHeader, MetaValue};
use std::{fs::File, io::BufReader, path::PathBuf};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// The context every plan is sized at, and the device reserve: the 1 GiB legacy headroom plus the
/// 384 MiB runtime floor, with 256 MiB of work buffers beside the KV cache.
const CONTEXT: u64 = 4096;
const RESERVE: u64 = GIB + 384 * MIB;
const WORK: u64 = 256 * MIB;

struct Loaded {
    label: &'static str,
    model: RungModel,
    expert_bytes: u64,
    experts: usize,
    expert_layers: usize,
    top_k: usize,
    kv_bytes_per_position: u64,
}

fn models() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("models")
}

fn shards(dir: &str, prefix: &str) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(models().join(dir))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension().is_some_and(|e| e == "gguf")
                && path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(prefix)
        })
        .collect();
    found.sort();
    found
}

fn load(label: &'static str, paths: &[PathBuf], kv_bytes_per_position: u64) -> Loaded {
    let mut layers: Vec<u64> = Vec::new();
    let mut exps: Vec<u64> = Vec::new();
    let (mut fixed, mut host_table, mut experts, mut top_k) = (0u64, 0u64, 0usize, 0usize);
    let headers: Vec<GgufHeader> = paths
        .iter()
        .map(|path| {
            let file = File::open(path).unwrap();
            GgufHeader::parse(
                &mut BufReader::new(&file),
                file.metadata().unwrap().len(),
                512 << 20,
            )
            .unwrap()
        })
        .collect();
    // A split set carries the metadata in its first shard only.
    let arch = headers
        .iter()
        .find_map(|header| match header.meta("general.architecture") {
            Some(MetaValue::String(arch)) => Some(arch.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{label}: no architecture"));
    if let Some(MetaValue::U32(count)) = headers
        .iter()
        .find_map(|header| header.meta(&format!("{arch}.expert_used_count")))
    {
        top_k = *count as usize;
    }
    for header in &headers {
        for tensor in header.tensors() {
            let bytes = tensor.byte_len();
            let name = tensor.name();
            if let Some(rest) = name.strip_prefix("blk.") {
                let index: usize = rest.split('.').next().unwrap().parse().unwrap();
                if layers.len() <= index {
                    layers.resize(index + 1, 0);
                    exps.resize(index + 1, 0);
                }
                layers[index] += bytes;
                if name.contains("_exps") {
                    exps[index] += bytes;
                    experts = experts.max(*tensor.dims().last().unwrap() as usize);
                }
            } else if name == "per_layer_token_embd.weight" {
                host_table += bytes;
            } else {
                fixed += bytes;
            }
        }
    }
    Loaded {
        label,
        expert_bytes: exps.iter().sum(),
        expert_layers: exps.iter().filter(|&&bytes| bytes > 0).count(),
        experts,
        top_k,
        kv_bytes_per_position,
        model: RungModel {
            layers,
            card_fixed_bytes: fixed,
            host_table_bytes: host_table,
        },
    }
}

fn corpus() -> Vec<Loaded> {
    let q = |name: &str| vec![models().join("qwen3.6").join(name)];
    let b = |name: &str| vec![models().join("nemotron-bolt-e4b").join(name)];
    vec![
        load("qwen3.6 Q4_K_M", &q("Qwen3.6-35B-A3B-Q4_K_M.gguf"), 40_960),
        load(
            "qwen3.6 UD-IQ3_XXS",
            &q("iq/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"),
            40_960,
        ),
        load(
            "qwen3.6 UD-IQ4_XS",
            &q("iq/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"),
            40_960,
        ),
        load(
            "flash-next UD-IQ3_XXS",
            &shards(
                "unsloth-Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS",
                "Qwen3.8-Flash-Next-UD-IQ3_XXS",
            ),
            26_112,
        ),
        load(
            "nemotron NVFP4",
            &shards("nemotron35-super-vl-gguf", "NVIDIA-Nemotron"),
            8_192,
        ),
        load(
            "bolt q4_k",
            &b("bolt-e4b-q4_k-from-q8-canonical.gguf"),
            24_576,
        ),
        load(
            "bolt q4_k_pure",
            &b("bolt-e4b-q4_k-pure-from-q8-canonical.gguf"),
            24_576,
        ),
        load(
            "bolt q4_k_m",
            &b("bolt-e4b-q4_k_m-from-q8-canonical.gguf"),
            24_576,
        ),
        load("bolt q8_0", &b("bolt-e4b-q8_0-fix2.gguf"), 24_576),
    ]
}

/// What the generic card-short rule did: the resident load, or for a MoE the expert-tier ladder
/// (experts on the host, banks on the card). `true` is a refusal.
fn before_refuses(m: &Loaded, card_free: u64, host_budget: u64) -> bool {
    let state = m.kv_bytes_per_position * CONTEXT + WORK;
    let tensors = m.model.card_fixed_bytes + m.model.layers.iter().sum::<u64>();
    if m.expert_bytes == 0 || m.experts == 0 {
        return tensors + state + RESERVE > card_free;
    }
    let figures = CardFigures {
        free: card_free,
        tensors,
        expert_bytes: m.expert_bytes,
        kv_cache: state,
        work: 0,
        headroom: RESERVE,
        tiered_headroom: RESERVE,
        slot_bytes: m.expert_bytes / m.experts as u64 / m.expert_layers as u64,
        after_load_floor: 0,
        layers: m.expert_layers,
        experts: m.experts,
        top_k: m.top_k.max(1),
    };
    matches!(decide(&figures, host_budget), CardVerdict::Refused { .. })
}

#[test]
#[ignore = "reads the real GGUF headers under $HOME/models"]
fn the_layer_rung_places_every_real_gguf_on_every_card_and_host() {
    let cards = [("4050 6 GB", 5_500 * MIB), ("3090 24 GB", 23_416 * MIB)];
    let hosts = [16u64, 32, 62];
    let corpus = corpus();
    let (mut cells, mut before, mut after) = (0, 0, 0);
    eprintln!("model | card | host GiB | before | after");
    for m in &corpus {
        for (card_name, free) in cards {
            for total in hosts {
                let host = HostMemory::from_pair(total * GIB, (total - 2) * GIB);
                let b = RungBudget {
                    card_free: free,
                    state_bytes: m.kv_bytes_per_position * CONTEXT + WORK,
                    reserve_bytes: RESERVE,
                    host_spare: host.payload_spare(),
                };
                let refused_before = before_refuses(m, free, host.available);
                let placed = place(&m.model, &b);
                cells += 1;
                before += usize::from(refused_before);
                after += usize::from(placed.is_err());
                let summary = match &placed {
                    Ok(Placement::Resident) => "resident".to_string(),
                    Ok(Placement::Streamed(r)) => {
                        let count = |home| r.homes.iter().filter(|h| **h == home).count();
                        format!(
                            "{} card / {} pinned / {} disk layers, {} slots, {:.1} GiB read per pass",
                            count(Home::Card),
                            count(Home::Pinned),
                            count(Home::Disk),
                            r.staging_slots,
                            r.disk_bytes_per_pass as f64 / GIB as f64
                        )
                    }
                    Err(refusal) => format!("REFUSED: {}", refusal.what),
                };
                eprintln!(
                    "{} | {card_name} | {total} | {} | {summary}",
                    m.label,
                    if refused_before { "REFUSED" } else { "starts" }
                );
                if let Err(refusal) = &placed {
                    // The only refusal is bytes that cannot leave the card.
                    assert!(refusal.needed > refusal.have, "{}: {refusal:?}", m.label);
                }
            }
        }
    }
    eprintln!("refusals before {before} of {cells}, after {after} of {cells}");
    assert!(after <= before);
}
