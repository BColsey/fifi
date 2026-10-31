//! Shared pure-host storage of the M3 harness (registration v5): the
//! frozen layout with sha verification against the REGISTERED constants,
//! the per-window sidecar chunk, and the compare step that feeds
//! `m3_quality::Comparison::evaluate`.
//!
//! No device types here: the CUDA leg (`cuda::m3_run`) writes sidecars
//! through this module, and `m3_compare` runs on any host from two sidecar
//! sets. The sidecar chunk is a 67-byte header (magic `M3R\x02`, window,
//! lo, score_lo, hi, n_rows as u64s, the 3-byte arm id — gguf, kv-q8,
//! chunked — and the 20-byte git build sha) + `(logprob_true f32,
//! top1_correct u8, top1_id u32)` per scored position + the sha256 of
//! everything before it: every chunk self-describes the arm and build that
//! produced it.

use super::m3_quality::{Comparison, Margins, PositionRow, ProbeRow};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Proven constants of the pinned model (see cuda::m3_run): the bootstrap
/// seed and draw count are the frozen registration numbers.
const N_CLUSTERS: u32 = 1_526;
const DRAWS: usize = 10_000;
const SEED: u64 = 20_261_031;

/// Registered digests and counts of the frozen v5 layout (registration
/// §1). A layout whose digests differ aborts the harness: these are the
/// numbers the review signed off on.
const LAYOUT_SHA: &str = "86030447e2b4298e8eb383edeb4d58bfca2076663090f4c339dd8b4e7e268683";
const IDS_SHA: &str = "d904346c1c6fa5b91d48c441bada957f651394894bf152a33ea64df9c7f7b2a7";
const CLUSTERS_SHA: &str = "4a03ae50343c2072c43c3e880b548ca7f8bb574550d3f6457c146c5e8bb8030a";
const N_WINDOWS: usize = 58;
const N_RETRIEVAL: usize = 174;
const N_COPY: usize = 387;

/// Python `json.dumps(value, sort_keys=True)` byte-for-byte, for the
/// layout document: keys sorted, ", " item and ": " key separators, ints,
/// bools, null and ASCII-escaped strings only. A float aborts loudly (the
/// canonical writer covers the layout's int-only schema) instead of
/// hashing to a wrong recipe silently.
fn python_recipe(v: &Value) -> Result<String, String> {
    let esc = |s: &str| -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\u{08}' => out.push_str("\\b"),
                '\t' => out.push_str("\\t"),
                '\n' => out.push_str("\\n"),
                '\u{0c}' => out.push_str("\\f"),
                '\r' => out.push_str("\\r"),
                c if (c as u32) < 0x20 || (c as u32) >= 0x7f => {
                    let cp = c as u32;
                    if cp <= 0xffff {
                        out.push_str(&format!("\\u{cp:04x}"));
                    } else {
                        let s = cp - 0x1_0000;
                        out.push_str(&format!(
                            "\\u{:04x}\\u{:04x}",
                            0xd800 + (s >> 10),
                            0xdc00 + (s & 0x3ff)
                        ));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
        out
    };
    Ok(match v {
        Value::Null => "null".to_owned(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            if n.as_u64().is_none() && n.as_i64().is_none() {
                return Err(err(
                    "the recipe carries a float; canonical writer covers ints",
                ));
            }
            n.to_string()
        }
        Value::String(s) => esc(s),
        Value::Array(a) => {
            let parts = a.iter().map(python_recipe).collect::<Result<Vec<_>, _>>()?;
            format!("[{}]", parts.join(", "))
        }
        Value::Object(o) => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            let parts = keys
                .into_iter()
                .map(|k| Ok::<_, String>(format!("{}: {}", esc(k), python_recipe(&o[k])?)))
                .collect::<Result<Vec<_>, _>>()?;
            format!("{{{}}}", parts.join(", "))
        }
    })
}

fn err(text: &str) -> String {
    format!("m3 store: {text}")
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The frozen layout: ids, per-id document cluster, window geometry, and
/// probe spans (kind-split: retrieval vs copy).
pub(in crate::family::qwen36) struct Layout {
    pub(in crate::family::qwen36) ids: Vec<u32>,
    pub(in crate::family::qwen36) clusters: Vec<u16>,
    /// (lo, score_lo, hi) per window.
    pub(in crate::family::qwen36) windows: Vec<(usize, usize, usize)>,
    /// (span_lo, span_hi, cluster, is_copy).
    pub(in crate::family::qwen36) probes: Vec<(usize, usize, u32, bool)>,
}

fn u64_of(v: &Value, key: &str) -> Result<u64, String> {
    v[key]
        .as_u64()
        .ok_or_else(|| err(&format!("{key} is not a u64")))
}

/// Loads the layout directory and re-verifies every registered digest the
/// way check_v4.py does: the layout sha recipe, and the ids and cluster
/// bins against the layout's own fields.
pub(in crate::family::qwen36) fn load_layout(dir: &Path) -> Result<Layout, String> {
    let text = std::fs::read_to_string(dir.join("m3_layout_v5.json"))
        .map_err(|e| err(&format!("the layout json does not read: {e}")))?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| err(&format!("layout json: {e}")))?;
    let mut recipe = doc.clone();
    recipe
        .as_object_mut()
        .ok_or_else(|| err("layout is not an object"))?
        .remove("layout_sha256");
    let recipe = python_recipe(&recipe)?;
    if sha256_hex(recipe.as_bytes()) != doc["layout_sha256"].as_str().unwrap_or("") {
        return Err(err("the layout sha recipe does not reproduce"));
    }
    let ids_bin = std::fs::read(dir.join("m3_corpus_ids_v5.bin"))
        .map_err(|e| err(&format!("the ids bin does not read: {e}")))?;
    if sha256_hex(&ids_bin) != doc["ids_sha256"].as_str().unwrap_or("") {
        return Err(err("the ids bin sha does not match the layout"));
    }
    let cluster_bin = std::fs::read(dir.join("m3_pos_clusters_v5.bin"))
        .map_err(|e| err(&format!("the cluster bin does not read: {e}")))?;
    if sha256_hex(&cluster_bin) != doc["pos_clusters_sha256"].as_str().unwrap_or("") {
        return Err(err("the cluster bin sha does not match the layout"));
    }
    for (name, got, want) in [
        (
            "layout_sha256",
            doc["layout_sha256"].as_str().unwrap_or(""),
            LAYOUT_SHA,
        ),
        (
            "ids_sha256",
            doc["ids_sha256"].as_str().unwrap_or(""),
            IDS_SHA,
        ),
        (
            "pos_clusters_sha256",
            doc["pos_clusters_sha256"].as_str().unwrap_or(""),
            CLUSTERS_SHA,
        ),
    ] {
        if got != want {
            return Err(err(&format!(
                "{name} {got} is not the REGISTERED digest {want}"
            )));
        }
    }
    let n = u64_of(&doc, "n_ids")? as usize;
    if ids_bin.len() != n * 4 || cluster_bin.len() != n * 2 {
        return Err(err("a bin length disagrees with n_ids"));
    }
    let ids: Vec<u32> = ids_bin
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let clusters: Vec<u16> = cluster_bin
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    let windows = doc["windows"]
        .as_array()
        .ok_or_else(|| err("windows is not an array"))?
        .iter()
        .map(|w| {
            Ok::<_, String>((
                u64_of(w, "lo")? as usize,
                u64_of(w, "score_lo")? as usize,
                u64_of(w, "hi")? as usize,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut probes = Vec::new();
    for (key, is_copy) in [("retrieval_probes", false), ("copy_probes", true)] {
        for p in doc[key].as_array().ok_or_else(|| err(key))? {
            let span = p["span_ids"]
                .as_array()
                .ok_or_else(|| err("span_ids is not an array"))?;
            probes.push((
                span.first()
                    .and_then(Value::as_u64)
                    .ok_or_else(|| err("span_ids[0] is not a u64"))? as usize,
                span.get(1)
                    .and_then(Value::as_u64)
                    .ok_or_else(|| err("span_ids[1] is not a u64"))? as usize,
                u64_of(p, "doc")? as u32,
                is_copy,
            ));
        }
    }
    let counts = (
        probes.iter().filter(|p| !p.3).count(),
        probes.iter().filter(|p| p.3).count(),
    );
    if windows.len() != N_WINDOWS || counts != (N_RETRIEVAL, N_COPY) {
        return Err(err(&format!(
            "windows/retrieval/copy = {}/{}/{} are not the REGISTERED \
             {N_WINDOWS}/{N_RETRIEVAL}/{N_COPY}",
            windows.len(),
            counts.0,
            counts.1
        )));
    }
    Ok(Layout {
        ids,
        clusters,
        windows,
        probes,
    })
}

/// One scored row as stored: logprob f32, argmax-correct flag, argmax id.
pub(in crate::family::qwen36) type Row = (f32, bool, u32);

/// The registered digests, for the leg's arm manifest.
pub(in crate::family::qwen36) fn registered_layout_sha() -> &'static str {
    LAYOUT_SHA
}

pub(in crate::family::qwen36) fn registered_ids_sha() -> &'static str {
    IDS_SHA
}

/// The frozen arm-4 trace input: i64 LE ids, sha and count registered.
pub(in crate::family::qwen36) const TRACE_IDS: usize = 1119;
const TRACE_SHA: &str = "0c04651f0c9d7a85a7dee46c3ae5007518aa13bd13497991dbe27cd145858a7a";

pub(in crate::family::qwen36) fn registered_trace_sha() -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        let hex = |c: u8| (c as char).to_digit(16).expect("hex digit") as u8;
        *slot = (hex(TRACE_SHA.as_bytes()[i * 2]) << 4) | hex(TRACE_SHA.as_bytes()[i * 2 + 1]);
    }
    out
}

/// The sidecar bytes of one window: header + rows + trailing sha256.
pub(in crate::family::qwen36) fn sidecar_bytes(
    window: usize,
    (lo, score_lo, hi): (usize, usize, usize),
    arm: [u8; 3],
    build_sha: [u8; 20],
    rows: &[Row],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(67 + rows.len() * 9 + 32);
    out.extend_from_slice(b"M3R\x02");
    for field in [
        window as u64,
        lo as u64,
        score_lo as u64,
        hi as u64,
        rows.len() as u64,
    ] {
        out.extend_from_slice(&field.to_le_bytes());
    }
    out.extend_from_slice(&arm);
    out.extend_from_slice(&build_sha);
    for (logprob, correct, top1) in rows {
        out.extend_from_slice(&logprob.to_le_bytes());
        out.push(u8::from(*correct));
        out.extend_from_slice(&top1.to_le_bytes());
    }
    out.extend_from_slice(&Sha256::digest(&out));
    out
}

/// The chunk's arm identity: the 3-byte arm id and the 20-byte git build
/// sha it was produced from.
pub(in crate::family::qwen36) type ArmIdentity = ([u8; 3], [u8; 20]);

/// Reads and verifies one window chunk: sha over the body, header geometry
/// against the layout, then the arm id and build sha it carries. Returns
/// the rows in target order with the chunk's arm identity.
pub(in crate::family::qwen36) fn read_sidecar(
    path: &Path,
    window: usize,
    want: (usize, usize, usize),
) -> Result<(Vec<Row>, ArmIdentity), String> {
    let bytes = std::fs::read(path).map_err(|e| err(&format!("{}: {e}", path.display())))?;
    if bytes.len() < 67 + 32 {
        return Err(err(&format!("{}: too short", path.display())));
    }
    let (body, digest) = bytes.split_at(bytes.len() - 32);
    if &Sha256::digest(body)[..] != digest {
        return Err(err(&format!("{}: sha mismatch", path.display())));
    }
    if &body[..4] != b"M3R\x02" {
        return Err(err(&format!("{}: wrong magic", path.display())));
    }
    let fields: Vec<u64> = (0..5)
        .map(|i| u64::from_le_bytes(body[4 + i * 8..12 + i * 8].try_into().unwrap()))
        .collect();
    let got = (fields[1] as usize, fields[2] as usize, fields[3] as usize);
    if fields[0] as usize != window || got != want {
        return Err(err(&format!(
            "{}: header {got:?} disagrees with layout {want:?}",
            path.display()
        )));
    }
    let n = fields[4] as usize;
    if body.len() != 67 + n * 9 {
        return Err(err(&format!("{}: row count", path.display())));
    }
    let mut arm = [0u8; 3];
    arm.copy_from_slice(&body[44..47]);
    let mut build = [0u8; 20];
    build.copy_from_slice(&body[47..67]);
    let rows = body[67..]
        .as_chunks::<9>()
        .0
        .iter()
        .map(|c| {
            (
                f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                c[4] != 0,
                u32::from_le_bytes([c[5], c[6], c[7], c[8]]),
            )
        })
        .collect();
    Ok((rows, (arm, build)))
}

/// All position and probe rows of one arm, rebuilt from its sidecar set.
struct ArmRows {
    identity: ArmIdentity,
    positions: Vec<PositionRow>,
    retrieval: Vec<ProbeRow>,
    copy: Vec<ProbeRow>,
}

fn arm_rows(dir: &Path, layout: &Layout) -> Result<ArmRows, String> {
    let mut positions = Vec::new();
    // Per-window rows cached once; probe spans index into them by absolute
    // id (windows tile the stream, so score_lo..hi covers every target).
    // Every chunk must carry the SAME arm id and build sha: a sidecar set
    // mixed across arms or builds does not describe one arm.
    let mut window_rows = Vec::with_capacity(layout.windows.len());
    let mut identity: Option<ArmIdentity> = None;
    for (w, &(lo, score_lo, hi)) in layout.windows.iter().enumerate() {
        let (rows, (arm, build)) =
            read_sidecar(&dir.join(format!("w{w:03}.m3row")), w, (lo, score_lo, hi))?;
        match identity {
            None => identity = Some((arm, build)),
            Some((want_arm, want_build)) => {
                if (arm, build) != (want_arm, want_build) {
                    return Err(err(&format!(
                        "window {w}: sidecar arm/build {:?}/{:?} disagrees with {:?}/{:?}",
                        arm, build, want_arm, want_build
                    )));
                }
            }
        }
        for (i, (logprob, correct, _)) in rows.iter().enumerate() {
            let p = score_lo + i;
            if !logprob.is_finite() {
                return Err(err(&format!(
                    "window {w} position {p}: non-finite logprob (crash rule: the arm fails)"
                )));
            }
            positions.push(PositionRow {
                cluster: u32::from(layout.clusters[p]),
                logprob_true: *logprob,
                top1_correct: *correct,
            });
        }
        window_rows.push((score_lo, rows));
    }
    // A probe span id is exact iff the arm's argmax matched the true id at
    // that position; span ids always lie inside their window's scored run.
    let exact_at = |p: usize| -> Option<bool> {
        window_rows
            .iter()
            .find(|(score_lo, rows)| p >= *score_lo && p - *score_lo < rows.len())
            .map(|(score_lo, rows)| rows[p - score_lo].1)
    };
    let mut retrieval = Vec::new();
    let mut copy = Vec::new();
    for &(span_lo, span_hi, cluster, is_copy) in &layout.probes {
        let mut exact = true;
        for p in span_lo..span_hi {
            exact &= exact_at(p)
                .ok_or_else(|| err(&format!("probe position {p} lies in no scored window")))?;
        }
        let row = ProbeRow { cluster, exact };
        if is_copy {
            copy.push(row);
        } else {
            retrieval.push(row);
        }
    }
    let identity = identity.expect("the layout has at least one window");
    Ok(ArmRows {
        identity,
        positions,
        retrieval,
        copy,
    })
}

/// The frozen comparisons of two sidecar sets. `M3_COMPARE=P1:cand:ref`
/// or `O1:cand:ref` (plus M3_LAYOUT_DIR): the pair's expected sidecar
/// identities are registered HERE, so a swapped or mismatched pair aborts
/// before any math, and the two arms must be scored on the SAME build.
/// Writes the four margins and the verdict to stdout and panics on any
/// incomplete or mismatching arm (crash rule).
#[test]
#[ignore = "needs two complete arm sidecar sets: M3_COMPARE=P1|O1:cand_dir:ref_dir M3_LAYOUT_DIR=dir"]
fn m3_compare() {
    let layout_dir = crate::test_paths::required("M3_LAYOUT_DIR");
    let (pair, cand_dir, ref_dir) = {
        let spec = std::env::var("M3_COMPARE").expect("M3_COMPARE=P1|O1:cand:ref");
        let parts: Vec<&str> = spec.split(':').collect();
        assert!(parts.len() == 3, "spec is PAIR:cand_dir:ref_dir");
        (
            parts[0].to_string(),
            PathBuf::from(parts[1]),
            PathBuf::from(parts[2]),
        )
    };
    // [gguf, kv-q8, chunked]: P1 = arm3 vs arm2 (q8); O1 = arm2' vs arm1 (f16).
    let (want_cand, want_ref, label) = match pair.as_str() {
        "P1" => (
            [0, 1, 1],
            [0, 1, 0],
            "P1: chunked prefill at q8 must not degrade quality",
        ),
        "O1" => (
            [0, 0, 1],
            [1, 0, 1],
            "O1: EXL3 f16 chunked vs GGUF f16 chunked",
        ),
        other => panic!("unknown comparison {other}: P1 or O1"),
    };
    let layout = load_layout(&layout_dir).expect("layout");
    let cand = arm_rows(&cand_dir, &layout).expect("candidate arm");
    let reference = arm_rows(&ref_dir, &layout).expect("reference arm");
    assert_eq!(
        cand.identity.0, want_cand,
        "candidate sidecar carries the wrong arm"
    );
    assert_eq!(
        reference.identity.0, want_ref,
        "reference sidecar carries the wrong arm"
    );
    assert_eq!(
        cand.identity.1, reference.identity.1,
        "the two arms were scored on different builds"
    );
    let comparison = Comparison::evaluate(
        &cand.positions,
        &reference.positions,
        &cand.copy,
        &reference.copy,
        &cand.retrieval,
        &reference.retrieval,
        N_CLUSTERS,
        DRAWS,
        SEED,
    );
    let verdict = comparison.passes(&Margins::frozen());
    println!(
        "m3 comparison {label}: cand arm {want_cand:?} ref arm {want_ref:?} build {:02x?}: {comparison:?} verdict={verdict}",
        cand.identity.1
    );
    assert!(verdict, "the frozen margins failed: {comparison:?}");
}

/// The arm-4 gate: the two trace legs' digest files (one sha256 per
/// position's full f32 row) must be byte-identical. The legs write a
/// `trace_identity.txt` next to the digests; the compare first asserts the
/// pair really is (gather off, gather on) of the SAME build, trace input
/// and count — a mismatched pair aborts before the bitwise compare.
/// `#[ignore]`d host entry: M3_TRACE_COMPARE=dir_a:dir_b.
#[test]
#[ignore = "two trace leg outputs: M3_TRACE_COMPARE=arm2_dir:arm4_dir"]
fn m3_trace_compare() {
    let spec = std::env::var("M3_TRACE_COMPARE").expect("M3_TRACE_COMPARE=dir_a:dir_b");
    let parts: Vec<&str> = spec.split(':').collect();
    let identity_of = |dir: &str| -> std::collections::BTreeMap<String, String> {
        let text = std::fs::read_to_string(Path::new(dir).join("trace_identity.txt"))
            .expect("trace_identity.txt");
        text.lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    };
    let a = identity_of(parts[0]);
    let b = identity_of(parts[1]);
    for key in ["build_sha256", "trace_sha256", "ids"] {
        assert_eq!(
            a.get(key),
            b.get(key),
            "the trace pair disagrees on {key}: {:?} vs {:?}",
            a.get(key),
            b.get(key)
        );
    }
    // Normalize the gather vocabulary: identity files written before the
    // canonical-writer fix spell the two states "off" and "1"; later ones
    // write "false" and "true". Both spell one on/off pair.
    let norm = |v: Option<&str>| -> Option<String> {
        match v {
            Some("off") | Some("false") => Some("off".into()),
            Some("1") | Some("true") | Some("on") => Some("on".into()),
            other => other.map(str::to_string),
        }
    };
    let mut gathers = [
        norm(a.get("gather").map(String::as_str)),
        norm(b.get("gather").map(String::as_str)),
    ];
    gathers.sort_unstable();
    assert_eq!(
        gathers,
        [Some("off".to_string()), Some("on".to_string())],
        "the pair must be one gather=off leg and one gather=on leg"
    );
    let a = std::fs::read(Path::new(parts[0]).join("trace_rows.sha256"))
        .expect("arm a trace_rows.sha256");
    let b = std::fs::read(Path::new(parts[1]).join("trace_rows.sha256"))
        .expect("arm b trace_rows.sha256");
    assert_eq!(a.len(), b.len(), "the trace legs scored different lengths");
    assert!(
        a.len().is_multiple_of(32) && !a.is_empty(),
        "not a digest file"
    );
    for (i, (x, y)) in a
        .as_chunks::<32>()
        .0
        .iter()
        .zip(b.as_chunks::<32>().0)
        .enumerate()
    {
        assert_eq!(x, y, "trace position {i}: row bits differ");
    }
    println!("m3 trace: {} positions bitwise-identical", a.len() / 32);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frozen v5 layout files load and match every REGISTERED digest and
    /// count. Runs in the gate whenever M3_LAYOUT_DIR resolves (env or
    /// ~/.config/fifi/test-paths.json). Unset FAILS LOUDLY (v5 review SF): a
    /// silent skip here would let the gate pass without ever loading the
    /// registered layout.
    #[test]
    fn frozen_layout_loads_with_registered_shas() {
        let dir = crate::test_paths::required("M3_LAYOUT_DIR");
        let layout = load_layout(&dir).expect("the frozen layout loads");
        assert_eq!(layout.windows.len(), N_WINDOWS);
        assert_eq!(layout.probes.iter().filter(|p| !p.3).count(), N_RETRIEVAL);
        assert_eq!(layout.probes.iter().filter(|p| p.3).count(), N_COPY);
    }

    /// Power statement (registration §power): the minimum uniform exactness
    /// loss each probe gate detects against a perfect reference, through the
    /// REAL frozen bootstrap on the REAL v5 cluster universe. A uniform loss
    /// of r marks each probe inexact independently with a deterministic LCG;
    /// the detection threshold is the smallest grid rate where the verdict
    /// flips. Must be monotone, and the smaller probe set (retrieval) cannot
    /// detect a finer loss than the larger (copy).
    #[test]
    fn power_statement_against_perfect_reference() {
        let dir = crate::test_paths::required("M3_LAYOUT_DIR");
        let layout = load_layout(&dir).expect("layout");
        let positions: Vec<PositionRow> = layout
            .windows
            .iter()
            .flat_map(|&(_, score_lo, hi)| {
                (score_lo..hi).map(|p| PositionRow {
                    cluster: u32::from(layout.clusters[p]),
                    logprob_true: -0.5,
                    top1_correct: true,
                })
            })
            .collect();
        let copy_ref = layout
            .probes
            .iter()
            .filter(|p| p.3)
            .map(|p| ProbeRow {
                cluster: p.2,
                exact: true,
            })
            .collect::<Vec<_>>();
        let retr_ref = layout
            .probes
            .iter()
            .filter(|p| !p.3)
            .map(|p| ProbeRow {
                cluster: p.2,
                exact: true,
            })
            .collect::<Vec<_>>();
        let flip = |rows: &[ProbeRow], rate: f64| -> Vec<ProbeRow> {
            let mut s = 0x853c_49e6_748f_ea9du64;
            rows.iter()
                .map(|r| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    // s>>33 spans [0, 2^31): dividing by 2^31 (NOT u32::MAX,
                    // which halves the span and doubles the flip rate) maps
                    // u onto [0,1) so `rate` is the true flip probability.
                    let u = (s >> 33) as f64 / f64::from(1u32 << 31);
                    ProbeRow {
                        exact: u >= rate,
                        ..*r
                    }
                })
                .collect()
        };
        let rates: Vec<f64> = (0..=40).map(|i| i as f64 * 0.0025).collect(); // 0% .. 10%
        let mut detection = [None, None]; // (copy, retrieval)
        for (gate, is_copy) in [(0, true), (1, false)] {
            let mut verdicts = Vec::with_capacity(rates.len());
            for &rate in &rates {
                let (cand_copy, cand_retr) = if is_copy {
                    (flip(&copy_ref, rate), retr_ref.clone())
                } else {
                    (copy_ref.clone(), flip(&retr_ref, rate))
                };
                let comparison = Comparison::evaluate(
                    &positions, &positions, &cand_copy, &copy_ref, &cand_retr, &retr_ref,
                    N_CLUSTERS, DRAWS, SEED,
                );
                verdicts.push(comparison.passes(&Margins::frozen()));
            }
            assert!(verdicts[0], "the gate fails at zero loss");
            assert!(
                verdicts.windows(2).all(|w| w[0] || !w[1]),
                "detection is not monotone: {verdicts:?}"
            );
            detection[gate] = rates
                .iter()
                .zip(&verdicts)
                .find(|&(_, p)| !*p)
                .map(|(r, _)| *r);
        }
        let (copy_det, retr_det) = (
            detection[0].expect("copy gate never failed"),
            detection[1].expect("retrieval gate never failed"),
        );
        assert!(
            copy_det <= retr_det,
            "copy ({}) coarser than retrieval ({copy_det} <= {retr_det} failed)",
            copy_det
        );
        assert!(copy_det <= 0.05, "the copy gate misses a 5% uniform loss");
        println!(
            "m3 power: copy n={} detects a uniform loss >= {:.2}%; retrieval n={} >= {:.2}% (n_clusters {N_CLUSTERS}, B {DRAWS})",
            copy_ref.len(),
            100.0 * copy_det,
            retr_ref.len(),
            100.0 * retr_det
        );
    }

    /// The sidecar round-trips and read_sidecar rejects tampering, a wrong
    /// window, and wrong geometry.
    #[test]
    fn sidecar_round_trips_and_rejects_tampering() {
        let rows: Vec<Row> = vec![(-0.5, true, 7), (-1.25, false, 3)];
        let bytes = sidecar_bytes(9, (100, 356, 400), [0, 1, 0], [7u8; 20], &rows);
        let back = read_sidecar(
            std::path::Path::new("/dev/null-does-not-exist"),
            9,
            (100, 356, 400),
        );
        assert!(back.is_err());
        let path = std::env::temp_dir().join("m3_store_selfcheck");
        std::fs::write(&path, &bytes).unwrap();
        let (rows, (arm, build)) = read_sidecar(&path, 9, (100, 356, 400)).unwrap();
        assert_eq!(rows, vec![(-0.5f32, true, 7u32), (-1.25, false, 3)]);
        assert_eq!((arm, build), ([0, 1, 0], [7u8; 20]));
        assert!(read_sidecar(&path, 8, (100, 356, 400)).is_err());
        assert!(read_sidecar(&path, 9, (101, 356, 400)).is_err());
        let mut tampered = bytes.clone();
        tampered[67] ^= 1; // a row byte: the digest must reject it
        std::fs::write(&path, &tampered).unwrap();
        assert!(read_sidecar(&path, 9, (100, 356, 400)).is_err());
        let mut wrong_arm = bytes.clone();
        wrong_arm[44] ^= 1; // the arm id byte, digest recomputed below
        let len = wrong_arm.len();
        let digest = Sha256::digest(&wrong_arm[..len - 32]);
        wrong_arm[len - 32..].copy_from_slice(&digest);
        std::fs::write(&path, &wrong_arm).unwrap();
        // the digest passes but the returned identity must say what it carries
        let (rows, (arm, build)) = read_sidecar(&path, 9, (100, 356, 400)).unwrap();
        assert_eq!(rows.len(), 2);
        assert_ne!((arm, build), ([0, 1, 0], [7u8; 20]));
        std::fs::remove_file(&path).ok();
    }
}
