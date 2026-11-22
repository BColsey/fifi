//! The one load-time tuner: a pure decider over candidate values a
//! family supplies from its own qualified machinery. It names no family
//! and does no resource arithmetic of its own: the family builds each
//! decision's candidates with the deciders already on main (the
//! settings, the tier plan, the prefill byte cost), the tuner picks,
//! names the source of every value, applies explicit overrides and the
//! all-off mode, and prints the plan beside the values the start
//! actually uses. No start reads a plan value yet; a decision commit
//! adopts one value at a time.

/// Per-device facts of one accelerator card.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct DeviceFacts {
    pub(crate) vram_free_bytes: u64,
    pub(crate) vram_total_bytes: u64,
    pub(crate) sm_count: u32,
    pub(crate) compute_capability: (u32, u32),
    /// PCIe link (generation, width); (0, 0) when sysfs does not say.
    pub(crate) pcie_link: (u32, u32),
}

/// The host's memory topology: CPU-only, discrete accelerator cards, or
/// one unified memory pool. The unified form leaves room for the MLX
/// budget object without merging it here.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MemoryTopology {
    // The CUDA start only ever builds the Discrete form; the other two
    // are exercised by tests until an MLX start constructs Unified.
    #[allow(dead_code)]
    CpuOnly { host_budget_bytes: u64 },
    Discrete {
        host_budget_bytes: u64,
        devices: Vec<DeviceFacts>,
    },
    #[allow(dead_code)]
    Unified {
        pool_bytes: u64,
        working_set_limit_bytes: Option<u64>,
    },
}

/// One hardware fact set, probed once at load.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HardwareFacts {
    pub(crate) topology: MemoryTopology,
    pub(crate) cpu_threads: u32,
    pub(crate) cpu_isa: String,
    /// A short random-read probe of the pack file, bytes per second at
    /// queue depth 1 and 16; `None` until a decision needs the figure.
    pub(crate) drive: Option<(u64, u64)>,
}

impl HardwareFacts {
    /// Threads and best SIMD ISA of this host, shared by every topology.
    // The only caller today is the CUDA start's tuner line; keep it
    // compiled everywhere so the container and host agree on its bytes.
    #[cfg_attr(not(feature = "native-cuda"), expect(dead_code))]
    pub(crate) fn host_side() -> (u32, String) {
        (
            std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1),
            cpu_isa(),
        )
    }

    /// The PCIe link of a device from sysfs, given a validated PCI bus
    /// id ("0000:01:00.0") and the sysfs root, so a test can inject one.
    /// Generation comes from `max_link_speed` — a card idles at Gen1
    /// (ASPM), so the current speed understates the link; width is the
    /// current width. An unreadable sysfs is (0, 0), a fact like any
    /// other.
    pub(crate) fn pcie_link_of(bus_id: &str, sysfs_root: &std::path::Path) -> (u32, u32) {
        let valid = {
            let bytes = bus_id.as_bytes();
            bytes.len() == 12
                && bytes[..4].iter().all(|b| b.is_ascii_hexdigit())
                && bytes[..4]
                    .iter()
                    .all(|b| b.is_ascii_digit() || b.is_ascii_lowercase())
                && bytes[4] == b':'
                && bytes[5..7].iter().all(|b| b.is_ascii_hexdigit())
                && bytes[7] == b':'
                && bytes[8..10].iter().all(|b| b.is_ascii_hexdigit())
                && bytes[10] == b'.'
                && bytes[11..].iter().all(|b| b.is_ascii_hexdigit())
        };
        if !valid {
            return (0, 0);
        }
        let device_dir = sysfs_root.join(bus_id);
        let link_gen = std::fs::read_to_string(device_dir.join("max_link_speed"))
            .ok()
            .and_then(|text| link_generation(&text))
            .unwrap_or(0);
        let width = std::fs::read_to_string(device_dir.join("current_link_width"))
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .unwrap_or(0);
        (link_gen, width)
    }

    /// A bounded random-read probe of one file: bytes per second at
    /// queue depth 1 and 16. Bounded in bytes per depth (`byte_cap`) and
    /// wall time (`budget_seconds`); zero when the file cannot be opened
    /// or is smaller than one read. Test-only for now: the ring decision
    /// will move it to the load path with the cache-bypass and thread
    /// caps that a start needs.
    #[cfg(test)]
    pub(crate) fn drive_probe(
        path: &std::path::Path,
        budget_seconds: u64,
        read_bytes: usize,
        byte_cap: u64,
    ) -> (u64, u64) {
        use std::io::{Read, Seek};
        fn bytes_per_s(
            path: &std::path::Path,
            threads: usize,
            budget_seconds: u64,
            read_bytes: usize,
            byte_cap: u64,
        ) -> u64 {
            let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            if len <= read_bytes as u64 || threads == 0 || budget_seconds == 0 || byte_cap == 0 {
                return 0;
            }
            let per_thread_cap = byte_cap / threads as u64;
            let deadline =
                std::time::Instant::now() + std::time::Duration::from_secs(budget_seconds);
            std::thread::scope(|scope| {
                let counters: Vec<_> = (0..threads)
                    .map(|seed| {
                        scope.spawn(move || {
                            let Ok(mut file) = std::fs::File::open(path) else {
                                return 0u64;
                            };
                            let mut buffer = vec![0u8; read_bytes];
                            let mut state: u64 = 0x9e3779b97f4a7c15 ^ (seed as u64 + 1);
                            let mut done: u64 = 0;
                            while done < per_thread_cap && std::time::Instant::now() < deadline {
                                state ^= state << 13;
                                state ^= state >> 7;
                                state ^= state << 17;
                                let offset = state % (len - read_bytes as u64);
                                if file.seek(std::io::SeekFrom::Start(offset)).is_err() {
                                    break;
                                }
                                match file.read(&mut buffer) {
                                    Ok(n) if n > 0 => done += n as u64,
                                    _ => break,
                                }
                            }
                            done
                        })
                    })
                    .collect();
                counters
                    .into_iter()
                    .map(|counter| counter.join().unwrap_or(0))
                    .sum::<u64>()
            }) / budget_seconds.max(1)
        }
        (
            bytes_per_s(path, 1, budget_seconds, read_bytes, byte_cap),
            bytes_per_s(path, 16, budget_seconds, read_bytes, byte_cap),
        )
    }
}

/// Sysfs link speeds are GT/s per lane; a generation's base rate covers
/// the reading. Returns 0 for an unparsable value.
fn link_generation(text: &str) -> Option<u32> {
    let gts: f64 = text.trim().split(' ').next()?.parse().ok()?;
    [32.0, 16.0, 8.0, 5.0, 2.5]
        .iter()
        .position(|rate| gts >= *rate)
        .map(|index| (5 - index) as u32)
}

#[cfg_attr(not(feature = "native-cuda"), expect(dead_code))]
fn cpu_isa() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return "avx512".into();
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return "avx2".into();
        }
        "sse2".into()
    }
    #[cfg(target_arch = "aarch64")]
    {
        "neon".into()
    }
    #[cfg(all(not(target_arch = "x86_64"), not(target_arch = "aarch64")))]
    {
        "scalar".into()
    }
}

/// The multiprocessor count the decode attention split counts were measured on (an RTX 3090).
#[cfg_attr(
    not(feature = "native-cuda"),
    allow(dead_code, reason = "only the CUDA starts size their attention splits")
)]
pub(crate) const SPLIT_TUNED_SM_COUNT: u32 = 82;

/// The kernels' ceiling on the splits of one attention launch.
#[cfg_attr(
    not(feature = "native-cuda"),
    allow(dead_code, reason = "only the CUDA starts size their attention splits")
)]
pub(crate) const SPLIT_CEILING: usize = 128;

/// The splits of a decode attention launch on a device of `sm_count` multiprocessors, from the
/// count `tuned` on the 82-SM card: the blocks each multiprocessor runs stay what the tuning
/// measured, so a smaller card does not over-split into partial rows it must combine and a larger
/// one fills its multiprocessors. The tuned card keeps its count exactly. A device that reports
/// no count keeps the tuned one. Never below one nor above [`SPLIT_CEILING`].
#[cfg_attr(
    not(feature = "native-cuda"),
    allow(dead_code, reason = "only the CUDA starts size their attention splits")
)]
pub(crate) fn attention_splits(tuned: usize, sm_count: u32) -> usize {
    if sm_count == 0 {
        return tuned.clamp(1, SPLIT_CEILING);
    }
    let scaled = (tuned as u64 * u64::from(sm_count)).div_ceil(u64::from(SPLIT_TUNED_SM_COUNT));
    usize::try_from(scaled)
        .unwrap_or(SPLIT_CEILING)
        .clamp(1, SPLIT_CEILING)
}

/// One decision the family puts to the tuner: named candidates ordered
/// best-first (the family derives each with its own qualified machinery
/// and says why in the source string), plus the value the start uses
/// today and where that came from.
#[derive(Clone, Debug)]
pub(crate) struct Decision {
    pub(crate) name: &'static str,
    pub(crate) candidates: Vec<(u64, String)>,
    pub(crate) in_use: (u64, &'static str),
}

/// The one all-off switch: `Off` keeps every decision at its in-use
/// value; the plan then states the start's own configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum TuningMode {
    #[default]
    Auto,
    Off,
}

/// The explicit settings that override any candidate, by decision name.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct TuningOverrides {
    pub(crate) values: Vec<(&'static str, u64)>,
}

#[derive(Clone, Debug)]
pub(crate) struct TuningInput<'a> {
    pub(crate) mode: TuningMode,
    pub(crate) overrides: &'a TuningOverrides,
    pub(crate) facts: &'a HardwareFacts,
    pub(crate) decisions: &'a [Decision],
}

/// What the tuner decided for one decision and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Choice {
    pub(crate) name: &'static str,
    pub(crate) value: u64,
    /// The fact behind the value: a candidate's source, "setting", the
    /// in-use source under mode off, or "no candidate".
    pub(crate) source: String,
}

/// The pure decider. An override wins over every candidate and is named;
/// mode off keeps the in-use value; an empty candidate list keeps the
/// in-use value rather than inventing one.
pub(crate) fn decide(input: &TuningInput<'_>) -> Vec<Choice> {
    input
        .decisions
        .iter()
        .map(|decision| {
            let in_use = (decision.in_use.0, decision.in_use.1.to_string());
            if let Some((_, value)) = input
                .overrides
                .values
                .iter()
                .find(|(name, _)| *name == decision.name)
            {
                return Choice {
                    name: decision.name,
                    value: *value,
                    source: "setting".into(),
                };
            }
            match input.mode {
                TuningMode::Off => Choice {
                    name: decision.name,
                    value: in_use.0,
                    source: format!("mode off ({})", in_use.1),
                },
                TuningMode::Auto => match decision.candidates.first() {
                    Some((value, source)) => Choice {
                        name: decision.name,
                        value: *value,
                        source: source.clone(),
                    },
                    None => Choice {
                        name: decision.name,
                        value: in_use.0,
                        source: format!("no candidate ({})", in_use.1),
                    },
                },
            }
        })
        .collect()
}

/// The one `tuning plan` log line: every choice with its plan value and
/// source beside the value in use, the facts, and the override state.
/// It states the plan is NOT applied, because no start reads a plan
/// value yet. `overrides_known` is false until the settings layer hands
/// the tuner its explicit keys; the line says so instead of claiming
/// "none".
pub(crate) fn log_line(
    choices: &[Choice],
    input: &TuningInput<'_>,
    overrides_known: bool,
) -> String {
    use std::fmt::Write;
    let mut line = String::from("tuning plan (not applied):");
    for (index, choice) in choices.iter().enumerate() {
        let in_use = input
            .decisions
            .iter()
            .find(|decision| decision.name == choice.name)
            .map(|decision| decision.in_use)
            .unwrap_or((0, ""));
        let _ = write!(
            line,
            "{} {}: plan {} [{}] in use {} [{}]",
            if index == 0 { "" } else { " |" },
            choice.name,
            choice.value,
            choice.source,
            in_use.0,
            in_use.1
        );
    }
    match &input.facts.topology {
        MemoryTopology::CpuOnly { host_budget_bytes } => {
            let _ = write!(line, " | host={host_budget_bytes} no device");
        }
        MemoryTopology::Discrete {
            host_budget_bytes,
            devices,
        } => {
            let _ = write!(
                line,
                " | host={host_budget_bytes} devices={}",
                devices.len()
            );
            for device in devices {
                let pcie = format!("{}x{}", device.pcie_link.0, device.pcie_link.1);
                let _ = write!(
                    line,
                    " card[free={} total={} sm={} cc={}.{} pcie={}]",
                    device.vram_free_bytes,
                    device.vram_total_bytes,
                    device.sm_count,
                    device.compute_capability.0,
                    device.compute_capability.1,
                    pcie,
                );
            }
        }
        MemoryTopology::Unified {
            pool_bytes,
            working_set_limit_bytes,
        } => {
            let _ = write!(
                line,
                " | unified pool={pool_bytes} working_set_limit={:?}",
                working_set_limit_bytes
            );
        }
    }
    match input.facts.drive {
        Some((qd1, qd16)) => {
            let _ = write!(line, " | drive={qd1}/{qd16} B/s");
        }
        None => {
            line.push_str(" | drive=unmeasured");
        }
    }
    let _ = write!(
        line,
        " | threads={} isa={}",
        input.facts.cpu_threads, input.facts.cpu_isa
    );
    if !overrides_known {
        line.push_str(" | overrides: unknown (settings layer not wired)");
    } else if input.overrides.values.is_empty() {
        line.push_str(" | overrides: none");
    } else {
        let names: Vec<&str> = input.overrides.values.iter().map(|(n, _)| *n).collect();
        let _ = write!(line, " | overrides: {}", names.join(","));
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts_one_card() -> HardwareFacts {
        HardwareFacts {
            topology: MemoryTopology::Discrete {
                host_budget_bytes: 32_000_000_000,
                devices: vec![DeviceFacts {
                    vram_free_bytes: 22_500_000_000,
                    vram_total_bytes: 24_000_000_000,
                    sm_count: 82,
                    compute_capability: (8, 6),
                    pcie_link: (4, 16),
                }],
            },
            cpu_threads: 24,
            cpu_isa: "avx2".into(),
            drive: None,
        }
    }

    fn decisions() -> Vec<Decision> {
        vec![
            Decision {
                name: "piece",
                candidates: vec![(16, "default (no measured basis)".into())],
                in_use: (16, "default"),
            },
            Decision {
                name: "rows",
                candidates: vec![(1024, "arena fits within the floor".into())],
                in_use: (256, "default"),
            },
            Decision {
                name: "ring",
                candidates: vec![(1_600_000_000, "carve rule, whole records".into())],
                in_use: (0, "off"),
            },
        ]
    }

    /// Test files live under the build directory, never `/tmp`: unique
    /// per run, cleaned by the test, ignored by git.
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let base = std::path::PathBuf::from(
            std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".into()),
        )
        .join("tuning-tests")
        .join(format!("{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&base).expect("create the scratch dir");
        base
    }

    #[test]
    fn the_first_candidate_wins_in_auto_mode() {
        let overrides = TuningOverrides::default();
        let facts = facts_one_card();
        let input = TuningInput {
            mode: TuningMode::Auto,
            overrides: &overrides,
            facts: &facts,
            decisions: &decisions(),
        };
        let choices = decide(&input);
        assert_eq!(choices[0].value, 16);
        assert_eq!(choices[1].value, 1024);
        assert_eq!(choices[2].value, 1_600_000_000);
    }

    #[test]
    fn mode_off_keeps_the_in_use_values() {
        let overrides = TuningOverrides::default();
        let facts = facts_one_card();
        let input = TuningInput {
            mode: TuningMode::Off,
            overrides: &overrides,
            facts: &facts,
            decisions: &decisions(),
        };
        let choices = decide(&input);
        assert_eq!(choices[0].value, 16);
        assert_eq!(choices[1].value, 256);
        assert_eq!(choices[2].value, 0);
        assert!(choices[2].source.starts_with("mode off"));
        // With the switch set the line still prints, and it says so.
        let line = log_line(&choices, &input, false);
        assert!(
            line.contains("mode off"),
            "the line must state the mode: {line}"
        );
    }

    #[test]
    fn an_override_wins_over_every_candidate() {
        let overrides = TuningOverrides {
            values: vec![("rows", 512)],
        };
        let facts = facts_one_card();
        let input = TuningInput {
            mode: TuningMode::Auto,
            overrides: &overrides,
            facts: &facts,
            decisions: &decisions(),
        };
        let choices = decide(&input);
        let rows = choices.iter().find(|c| c.name == "rows").unwrap();
        assert_eq!(rows.value, 512);
        assert_eq!(rows.source, "setting");
    }

    #[test]
    fn a_decision_without_candidates_keeps_its_in_use_value() {
        let dense = vec![Decision {
            name: "ring",
            candidates: vec![], // a dense model offers no ring candidate
            in_use: (0, "off"),
        }];
        let overrides = TuningOverrides::default();
        let facts = facts_one_card();
        let input = TuningInput {
            mode: TuningMode::Auto,
            overrides: &overrides,
            facts: &facts,
            decisions: &dense,
        };
        let choices = decide(&input);
        assert_eq!(choices[0].value, 0);
        assert!(choices[0].source.starts_with("no candidate"));
    }

    #[test]
    fn the_line_states_it_is_not_applied_and_shows_in_use_values() {
        let overrides = TuningOverrides::default();
        let facts = facts_one_card();
        let input = TuningInput {
            mode: TuningMode::Auto,
            overrides: &overrides,
            facts: &facts,
            decisions: &decisions(),
        };
        let line = log_line(&decide(&input), &input, false);
        assert!(line.starts_with("tuning plan (not applied):"), "{line}");
        assert!(
            line.contains("rows: plan 1024 [arena fits within the floor] in use 256 [default]"),
            "{line}"
        );
        assert!(
            line.contains("ring: plan 1600000000 [carve rule, whole records] in use 0 [off]"),
            "{line}"
        );
        assert!(line.contains("drive=unmeasured"), "{line}");
        assert!(
            line.ends_with("overrides: unknown (settings layer not wired)"),
            "{line}"
        );
    }

    #[test]
    fn a_two_card_topology_prints_both_devices() {
        let mut facts = facts_one_card();
        if let MemoryTopology::Discrete { devices, .. } = &mut facts.topology {
            devices.push(DeviceFacts {
                vram_free_bytes: 10_000_000_000,
                vram_total_bytes: 11_000_000_000,
                sm_count: 70,
                compute_capability: (8, 6),
                pcie_link: (3, 8),
            });
        }
        let overrides = TuningOverrides::default();
        let input = TuningInput {
            mode: TuningMode::Auto,
            overrides: &overrides,
            facts: &facts,
            decisions: &decisions(),
        };
        let line = log_line(&decide(&input), &input, true);
        assert!(line.contains("devices=2"), "{line}");
        assert!(line.contains("overrides: none"), "{line}");
    }

    #[test]
    fn a_cpu_only_and_a_unified_topology_print_their_own_facts() {
        let cpu_only = HardwareFacts {
            topology: MemoryTopology::CpuOnly {
                host_budget_bytes: 1_000,
            },
            cpu_threads: 4,
            cpu_isa: "sse2".into(),
            drive: None,
        };
        let unified = HardwareFacts {
            topology: MemoryTopology::Unified {
                pool_bytes: 36_000_000_000,
                working_set_limit_bytes: Some(18_000_000_000),
            },
            cpu_threads: 10,
            cpu_isa: "neon".into(),
            drive: None,
        };
        let overrides = TuningOverrides::default();
        for facts in [&cpu_only, &unified] {
            let input = TuningInput {
                mode: TuningMode::Auto,
                overrides: &overrides,
                facts,
                decisions: &decisions(),
            };
            let line = log_line(&decide(&input), &input, true);
            assert!(line.contains("threads="), "{line}");
        }
        let overrides = TuningOverrides::default();
        let input = TuningInput {
            mode: TuningMode::Auto,
            overrides: &overrides,
            facts: &cpu_only,
            decisions: &decisions(),
        };
        assert!(
            log_line(&decide(&input), &input, true).contains("no device"),
            "cpu-only line"
        );
        let input = TuningInput {
            mode: TuningMode::Auto,
            overrides: &overrides,
            facts: &unified,
            decisions: &decisions(),
        };
        let line = log_line(&decide(&input), &input, true);
        assert!(line.contains("unified pool=36000000000"), "{line}");
        assert!(
            line.contains("working_set_limit=Some(18000000000)"),
            "{line}"
        );
    }

    #[test]
    fn pcie_link_parses_an_injected_sysfs_and_rejects_bad_bus_ids() {
        let dir = scratch_dir("pcie");
        let device = dir.join("0000:01:00.0");
        std::fs::create_dir_all(&device).expect("create the device dir");
        std::fs::write(device.join("max_link_speed"), "16 GT/s\n").expect("write speed");
        std::fs::write(device.join("current_link_width"), "16\n").expect("write width");
        assert_eq!(
            HardwareFacts::pcie_link_of("0000:01:00.0", &dir),
            (4, 16),
            "Gen4 from 16 GT/s, width 16"
        );
        assert_eq!(
            HardwareFacts::pcie_link_of("0000:01:00.0", &dir.join("absent")),
            (0, 0)
        );
        // Uppercase hex and a wrong shape are rejected as facts of (0, 0).
        assert_eq!(HardwareFacts::pcie_link_of("0000:01:00.A", &dir), (0, 0));
        assert_eq!(HardwareFacts::pcie_link_of("01:00.0", &dir), (0, 0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_drive_probe_reports_zero_for_unprobeable_inputs() {
        let dir = scratch_dir("drive");
        let tiny = dir.join("tiny");
        std::fs::write(&tiny, [0u8; 64]).expect("write the probe file");
        assert_eq!(
            HardwareFacts::drive_probe(&tiny, 1, 1 << 20, 1 << 20),
            (0, 0)
        );
        // A zero byte cap or budget reads nothing.
        let four = dir.join("four");
        let mut payload = vec![0u8; 3 << 20];
        payload[0] = 1;
        std::fs::write(&four, &payload).expect("write the probe file");
        assert_eq!(HardwareFacts::drive_probe(&four, 1, 1 << 20, 0), (0, 0));
        assert_eq!(
            HardwareFacts::drive_probe(&four, 0, 1 << 20, 1 << 20),
            (0, 0)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_drive_probe_reads_at_most_its_byte_cap() {
        // A two-megabyte cap bounds the single-threaded depth to two
        // reads of one megabyte, whatever the budget allows.
        let dir = scratch_dir("drive-cap");
        let file = dir.join("capped");
        let payload = vec![1u8; 4 << 20];
        std::fs::write(&file, &payload).expect("write the probe file");
        let (qd1, _) = HardwareFacts::drive_probe(&file, 5, 1 << 20, 2 << 20);
        assert!(qd1 <= (2 << 20), "qd1 respects the byte cap: {qd1}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn attention_splits_keep_the_tuned_card_and_follow_the_multiprocessor_count() {
        // The tuned card keeps its counts exactly, whatever they were tuned to.
        for tuned in [1, 4, 16, 32, 64, 128] {
            assert_eq!(attention_splits(tuned, SPLIT_TUNED_SM_COUNT), tuned);
        }
        // Blocks per multiprocessor stay: a 20-SM laptop card runs a quarter of the splits, a
        // 170-SM card about twice.
        assert_eq!(attention_splits(32, 20), 8);
        assert_eq!(attention_splits(16, 20), 4);
        assert_eq!(attention_splits(32, 170), 67);
        assert_eq!(attention_splits(16, 170), 34);
        // No count reported keeps the tuning; the result stays within the kernels' bounds.
        assert_eq!(attention_splits(32, 0), 32);
        assert_eq!(attention_splits(32, 1), 1);
        assert_eq!(attention_splits(128, 1_000), SPLIT_CEILING);
        let mut last = 0;
        for sms in 1..400 {
            let splits = attention_splits(32, sms);
            assert!(splits >= last && (1..=SPLIT_CEILING).contains(&splits));
            last = splits;
        }
    }
}
