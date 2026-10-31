//! Shared pure-host storage of the M3 harness (registration v4): the
//! frozen layout with sha verification, the per-window sidecar chunk, and
//! the compare step that feeds `m3_quality::Comparison::evaluate`.
//!
//! No device types here: the CUDA leg (`cuda::m3_run`) writes sidecars
//! through this module, and `m3_compare` runs on any host from two sidecar
//! sets. The sidecar chunk is a 44-byte header (magic, window, lo, score_lo,
//! hi, n_rows as u64s) + `(logprob_true f32, top1_correct u8, top1_id u32)`
//! per scored position + the sha256 of everything before it.

use super::m3_quality::{Comparison, Margins, PositionRow, ProbeRow};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Proven constants of the pinned model (see cuda::m3_run): the bootstrap
/// seed and draw count are the frozen registration numbers.
const N_CLUSTERS: u32 = 1_526;
const DRAWS: usize = 10_000;
const SEED: u64 = 20_261_031;

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
    let text = std::fs::read_to_string(dir.join("m3_layout_v4.json"))
        .map_err(|e| err(&format!("the layout json does not read: {e}")))?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| err(&format!("layout json: {e}")))?;
    let mut recipe = doc.clone();
    recipe["layout_sha256"].take();
    let recipe = serde_json::to_string(&recipe).map_err(|e| err(&format!("layout recipe: {e}")))?;
    if sha256_hex(recipe.as_bytes()) != doc["layout_sha256"].as_str().unwrap_or("") {
        return Err(err("the layout sha recipe does not reproduce"));
    }
    let ids_bin = std::fs::read(dir.join("m3_corpus_ids_v4.bin"))
        .map_err(|e| err(&format!("the ids bin does not read: {e}")))?;
    if sha256_hex(&ids_bin) != doc["ids_sha256"].as_str().unwrap_or("") {
        return Err(err("the ids bin sha does not match the layout"));
    }
    let cluster_bin = std::fs::read(dir.join("m3_pos_clusters_v4.bin"))
        .map_err(|e| err(&format!("the cluster bin does not read: {e}")))?;
    if sha256_hex(&cluster_bin) != doc["pos_clusters_sha256"].as_str().unwrap_or("") {
        return Err(err("the cluster bin sha does not match the layout"));
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
            probes.push((
                u64_of(&p["span_ids"], "0")? as usize,
                u64_of(&p["span_ids"], "1")? as usize,
                u64_of(p, "cluster")? as u32,
                is_copy,
            ));
        }
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

/// The sidecar bytes of one window: header + rows + trailing sha256.
pub(in crate::family::qwen36) fn sidecar_bytes(
    window: usize,
    (lo, score_lo, hi): (usize, usize, usize),
    rows: &[Row],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(44 + rows.len() * 9 + 32);
    out.extend_from_slice(b"M3R\x01");
    for field in [window as u64, lo as u64, score_lo as u64, hi as u64, rows.len() as u64] {
        out.extend_from_slice(&field.to_le_bytes());
    }
    for (logprob, correct, top1) in rows {
        out.extend_from_slice(&logprob.to_le_bytes());
        out.push(u8::from(*correct));
        out.extend_from_slice(&top1.to_le_bytes());
    }
    out.extend_from_slice(&Sha256::digest(&out));
    out
}

/// Reads and verifies one window chunk: sha over the body, then header
/// geometry against the layout. Returns the rows in target order.
pub(in crate::family::qwen36) fn read_sidecar(
    path: &Path,
    window: usize,
    want: (usize, usize, usize),
) -> Result<Vec<Row>, String> {
    let bytes = std::fs::read(path).map_err(|e| err(&format!("{}: {e}", path.display())))?;
    if bytes.len() < 44 + 32 {
        return Err(err(&format!("{}: too short", path.display())));
    }
    let (body, digest) = bytes.split_at(bytes.len() - 32);
    if &Sha256::digest(body)[..] != digest {
        return Err(err(&format!("{}: sha mismatch", path.display())));
    }
    if &body[..4] != b"M3R\x01" {
        return Err(err(&format!("{}: wrong magic", path.display())));
    }
    let fields: Vec<u64> = (0..5)
        .map(|i| u64::from_le_bytes(body[4 + i * 8..12 + i * 8].try_into().unwrap()))
        .collect();
    let got = (
        fields[1] as usize,
        fields[2] as usize,
        fields[3] as usize,
    );
    if fields[0] as usize != window || got != want {
        return Err(err(&format!(
            "{}: header {got:?} disagrees with layout {want:?}",
            path.display()
        )));
    }
    let n = fields[4] as usize;
    if body.len() != 44 + n * 9 {
        return Err(err(&format!("{}: row count", path.display())));
    }
    Ok(body[44..]
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
        .collect())
}

/// All position and probe rows of one arm, rebuilt from its sidecar set.
struct ArmRows {
    positions: Vec<PositionRow>,
    retrieval: Vec<ProbeRow>,
    copy: Vec<ProbeRow>,
}

fn arm_rows(dir: &Path, layout: &Layout) -> Result<ArmRows, String> {
    let mut positions = Vec::new();
    // Per-window rows cached once; probe spans index into them by absolute
    // id (windows tile the stream, so score_lo..hi covers every target).
    let mut window_rows = Vec::with_capacity(layout.windows.len());
    for (w, &(lo, score_lo, hi)) in layout.windows.iter().enumerate() {
        let rows = read_sidecar(
            &dir.join(format!("w{w:03}.m3row")),
            w,
            (lo, score_lo, hi),
        )?;
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
            .find(|(score_lo, rows)| {
                p >= *score_lo && p - *score_lo < rows.len()
            })
            .map(|(score_lo, rows)| rows[p - score_lo].1)
    };
    let mut retrieval = Vec::new();
    let mut copy = Vec::new();
    for &(span_lo, span_hi, cluster, is_copy) in &layout.probes {
        let mut exact = true;
        for p in span_lo..span_hi {
            exact &= exact_at(p).ok_or_else(|| {
                err(&format!("probe position {p} lies in no scored window"))
            })?;
        }
        let row = ProbeRow { cluster, exact };
        if is_copy {
            copy.push(row);
        } else {
            retrieval.push(row);
        }
    }
    Ok(ArmRows {
        positions,
        retrieval,
        copy,
    })
}

/// The frozen comparison of two sidecar sets: env M3_COMPARE="cand:ref"
/// plus M3_LAYOUT_DIR. Writes the four margins and the verdict to stdout
/// and panics on any incomplete or mismatching arm (crash rule).
#[test]
#[ignore = "needs two complete arm sidecar sets: M3_COMPARE=cand_dir:ref_dir M3_LAYOUT_DIR=dir"]
fn m3_compare() {
    let layout_dir = PathBuf::from(
        std::env::var("M3_LAYOUT_DIR").expect("M3_LAYOUT_DIR"),
    );
    let (cand_dir, ref_dir) = {
        let spec = std::env::var("M3_COMPARE").expect("M3_COMPARE=cand:ref");
        let parts: Vec<&str> = spec.split(':').collect();
        (PathBuf::from(parts[0]), PathBuf::from(parts[1]))
    };
    let layout = load_layout(&layout_dir).expect("layout");
    let cand = arm_rows(&cand_dir, &layout).expect("candidate arm");
    let reference = arm_rows(&ref_dir, &layout).expect("reference arm");
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
    println!("m3 comparison: {comparison:?} verdict={verdict}");
    assert!(verdict, "the frozen margins failed: {comparison:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sidecar round-trips and read_sidecar rejects tampering, a wrong
    /// window, and wrong geometry.
    #[test]
    fn sidecar_round_trips_and_rejects_tampering() {
        let rows: Vec<Row> = vec![(-0.5, true, 7), (-1.25, false, 3)];
        let bytes = sidecar_bytes(9, (100, 356, 400), &rows);
        let back = read_sidecar(std::path::Path::new("/dev/null-does-not-exist"), 9, (100, 356, 400));
        assert!(back.is_err());
        let path = std::env::temp_dir().join("m3_store_selfcheck");
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(
            read_sidecar(&path, 9, (100, 356, 400)).unwrap(),
            vec![(-0.5f32, true, 7u32), (-1.25, false, 3u32)]
        );
        assert!(read_sidecar(&path, 8, (100, 356, 400)).is_err());
        assert!(read_sidecar(&path, 9, (101, 356, 400)).is_err());
        let mut tampered = bytes.clone();
        tampered[44] ^= 1;
        std::fs::write(&path, &tampered).unwrap();
        assert!(read_sidecar(&path, 9, (100, 356, 400)).is_err());
        std::fs::remove_file(&path).ok();
    }
}
