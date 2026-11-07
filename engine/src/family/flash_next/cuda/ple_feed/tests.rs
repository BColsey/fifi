//! The PLE feed against the recorded real rows of the committed n-gram fixture and an independent
//! fp64 evaluation of the step.
//!
//! - Rows: the sparse n-gram file with the real header and the 80 real rows of a five-token prompt
//!   (`tests/fixtures/flash_next_exl3_ngram`). Hashing the prompt must land on the recorded global
//!   rows, and the rows read back must decode to the recorded bits.
//! - Numbers: the update of five tokens on small synthetic projections must agree with an fp64
//!   reimplementation of the step written from the design (not from `ple::step`), history and
//!   convolution ring included.
//! - Real pack (ignored): `FIFI_FLASH_NEXT_EXL3_PACK` names the full local pack; the same rows come
//!   from the 30 GiB table, and the real PLE projections of layer 1 run through the feed.

use std::path::Path;

use super::{
    Gathered, HostMatrix, HostProjection, NgramRows, PleFeed, PleKeyValue, PleProjection,
    PleVectors,
};
use crate::{
    EngError,
    family::flash_next::ple::{PleDims, PleShape},
    storage_guard::LocalGuard,
    weights::exl3::{
        ngram::ROW_DIM,
        ngram_file::{
            NgramFile,
            tests::{DECODED, KEY, LAYOUT, manifest_integers, write_table},
        },
    },
};

const PROMPT: [u32; 5] = [760, 6511, 314, 9338, 369];
const EOS: u32 = 248_044;
const BRANCHES: usize = 4;
const CONV_KERNEL: usize = 4;
const EPS: f32 = 1e-6;

fn dims(hidden: u32) -> PleDims {
    PleDims {
        branches: BRANCHES as u32,
        hidden,
        ngram: 3,
        heads_per_ngram: 8,
        head_dim: ROW_DIM as u32,
        conv_kernel: CONV_KERNEL as u32,
        eos: EOS,
        eps: EPS,
    }
}

/// Deterministic values in about [-scale, scale].
fn values(count: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 7919 + seed * 613) % 2003) as f32 - 1001.0) / 1001.0 * scale)
        .collect()
}

fn host(shape: &PleShape) -> (PleVectors, HostProjection) {
    let d = shape.dims();
    let (hidden, width, embed) = (
        d.hidden as usize,
        (d.hidden * d.branches) as usize,
        shape.embed(),
    );
    let projection = HostProjection::new(
        HostMatrix::new(width, embed, values(width * embed, 1, 0.05)).unwrap(),
        HostMatrix::new(hidden, embed, values(hidden * embed, 2, 0.05)).unwrap(),
        3,
    );
    let vectors = PleVectors {
        norm_key: values(width, 3, 0.5),
        norm_query: values(width, 4, 0.5),
        norm_conv: values(width, 5, 0.5),
        conv: values(width * CONV_KERNEL, 6, 0.5),
    };
    (vectors, projection)
}

fn feed_on(dir: &Path, hidden: u32) -> PleFeed {
    let path = dir.join("ngram_embedding.safetensors");
    write_table(&path);
    let file = NgramFile::open(&path, &LocalGuard::new().unwrap(), KEY, LAYOUT).unwrap();
    let shape = PleShape::new(dims(hidden)).unwrap();
    let (vectors, projection) = host(&shape);
    PleFeed::new(
        shape,
        NgramRows::new(file).unwrap(),
        vectors,
        Box::new(projection),
    )
}

/// The 80 recorded rows as f32, 16 per prompt token.
fn recorded(token_index: usize, head: usize) -> Vec<f32> {
    let at = (token_index * 16 + head) * ROW_DIM * 4;
    DECODED[at..at + ROW_DIM * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

/// Feeds the prompt and returns the global rows and decoded values of each token's plan.
fn feed_prompt(feed: &mut PleFeed, hidden: usize) -> Vec<(Vec<u64>, Gathered, Vec<f32>)> {
    let width = BRANCHES * hidden;
    let mut out = Vec::new();
    for (index, &token) in PROMPT.iter().enumerate() {
        let plan = feed.plan(token).unwrap();
        let globals = plan.rows.iter().map(|&(_, row)| u64::from(row)).collect();
        let gathered = feed.gather(&plan).unwrap();
        let stream = values(width, 40 + index, 1.0);
        let update = feed.update(token, &gathered, &stream).unwrap();
        out.push((globals, gathered, update));
    }
    out
}

#[test]
fn the_prompt_hashes_to_the_recorded_rows_and_they_decode_to_the_recorded_bits() {
    let dir = tempfile::tempdir().unwrap();
    let mut feed = feed_on(dir.path(), 8);
    let recorded_rows = manifest_integers("global_rows");
    for (index, (globals, gathered, _)) in feed_prompt(&mut feed, 8).iter().enumerate() {
        let want: Vec<u64> = recorded_rows[index * 16..(index + 1) * 16]
            .iter()
            .map(|&r| r as u64)
            .collect();
        assert_eq!(globals, &want, "token {index}: hashed rows");
        assert_eq!(gathered.len(), 16);
        for (head, &row) in want.iter().enumerate() {
            let got = gathered.get(row).expect("gathered");
            let exact = recorded(index, head);
            assert!(
                got.iter()
                    .zip(&exact)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "token {index} head {head}: decoded bits"
            );
        }
    }
    // Two tokens of history after five tokens.
    assert_eq!(feed.state.history(), &PROMPT[3..]);
}

// ---- fp64 reference ------------------------------------------------------------------------

fn rms0(x: &[f64], weight: &[f64]) -> Vec<f64> {
    let mean = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let denom = (mean + f64::from(EPS)).sqrt();
    x.iter()
        .zip(weight)
        .map(|(v, w)| v * (1.0 + w) / denom)
        .collect()
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// The PLE step of the design, from the formulas: the gated key/query score per branch, the value
/// gated by it, RMS over the branch, then a causal convolution of kernel 4 with dilation `ngram`
/// whose taps read the normed values `(taps - k) * ngram` steps back (zeros before the start).
struct Reference {
    normed: Vec<Vec<f64>>,
}

impl Reference {
    fn step(
        &mut self,
        (host, projection): (&PleVectors, &HostProjection),
        d: &PleDims,
        emb: &[f64],
        stream: &[f64],
    ) -> Vec<f64> {
        let (hidden, width) = (d.hidden as usize, (d.hidden * d.branches) as usize);
        let matvec = |w: &HostMatrix| -> Vec<f64> {
            (0..w.rows)
                .map(|r| {
                    (0..w.cols)
                        .map(|c| f64::from(w.data[r * w.cols + c]) * emb[c])
                        .sum()
                })
                .collect()
        };
        let (keys, value) = (matvec(&projection.key), matvec(&projection.value));
        let (nk, nq, nc) = (
            widen(&host.norm_key),
            widen(&host.norm_query),
            widen(&host.norm_conv),
        );
        let mut gated = vec![0.0; width];
        let mut normed = vec![0.0; width];
        for b in 0..d.branches as usize {
            let r = b * hidden..(b + 1) * hidden;
            let kn = rms0(&keys[r.clone()], &nk[r.clone()]);
            let qn = rms0(&stream[r.clone()], &nq[r.clone()]);
            let score =
                kn.iter().zip(&qn).map(|(a, b)| a * b).sum::<f64>() / (hidden as f64).sqrt();
            let shaped = score.abs().max(1e-6).sqrt().copysign(score);
            let gate = 1.0 / (1.0 + (-shaped).exp());
            for (g, v) in gated[r.clone()].iter_mut().zip(&value) {
                *g = gate * v;
            }
            normed[r.clone()].copy_from_slice(&rms0(&gated[r.clone()], &nc[r]));
        }
        let kernel = widen(&host.conv);
        let (taps, dilation) = (CONV_KERNEL - 1, d.ngram as usize);
        let out = (0..width)
            .map(|c| {
                let k = &kernel[c * CONV_KERNEL..(c + 1) * CONV_KERNEL];
                let mut acc = k[taps] * normed[c];
                for (tap, w) in k[..taps].iter().enumerate() {
                    let back = (taps - tap) * dilation;
                    if self.normed.len() >= back {
                        acc += w * self.normed[self.normed.len() - back][c];
                    }
                }
                gated[c] + acc / (1.0 + (-acc).exp())
            })
            .collect();
        self.normed.push(normed);
        out
    }
}

fn relative_l2(got: &[f32], want: &[f64]) -> f64 {
    let diff: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).powi(2))
        .sum();
    let norm: f64 = want.iter().map(|w| w * w).sum();
    (diff / norm).sqrt()
}

#[test]
fn five_updates_agree_with_the_fp64_step_including_history_and_ring() {
    let hidden = 8usize;
    let dir = tempfile::tempdir().unwrap();
    let mut feed = feed_on(dir.path(), hidden as u32);
    let d = feed.shape.dims();
    let mut reference = Reference { normed: Vec::new() };
    let (vectors, projection) = host(&feed.shape);
    let steps = feed_prompt(&mut feed, hidden);
    for (index, (_, gathered, update)) in steps.iter().enumerate() {
        let emb: Vec<f64> = (0..16)
            .flat_map(|head| widen(&recorded(index, head)))
            .collect();
        assert_eq!(emb.len(), feed.shape.embed());
        let stream = widen(&values(BRANCHES * hidden, 40 + index, 1.0));
        let want = reference.step((&vectors, &projection), &d, &emb, &stream);
        let error = relative_l2(update, &want);
        assert!(error < 2e-6, "token {index}: relative L2 {error}");
        assert!(update.iter().all(|v| v.is_finite()));
        assert_eq!(gathered.len(), 16);
    }
}

#[test]
fn a_failed_update_leaves_the_history_and_ring_alone() {
    let hidden = 8usize;
    let (dir_a, dir_b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut feed = feed_on(dir_a.path(), hidden as u32);
    let mut fresh = feed_on(dir_b.path(), hidden as u32);
    let plan = feed.plan(PROMPT[0]).unwrap();
    let gathered = feed.gather(&plan).unwrap();
    let stream = values(BRANCHES * hidden, 40, 1.0);
    // A short stream, then rows that were not gathered for this token.
    assert!(feed.update(PROMPT[0], &gathered, &stream[1..]).is_err());
    let empty = Gathered { rows: Vec::new() };
    assert!(matches!(
        feed.update(PROMPT[0], &empty, &stream),
        Err(EngError::BadRequest(_))
    ));
    assert!(feed.state.history().is_empty());
    let got = feed.update(PROMPT[0], &gathered, &stream).unwrap();
    let plan = fresh.plan(PROMPT[0]).unwrap();
    let want = fresh
        .update(PROMPT[0], &fresh.gather(&plan).unwrap(), &stream)
        .unwrap();
    assert_eq!(got, want);
}

#[test]
fn end_of_sequence_clears_the_history_and_reset_restarts_the_sequence() {
    let hidden = 8usize;
    let dir = tempfile::tempdir().unwrap();
    let mut feed = feed_on(dir.path(), hidden as u32);
    let stream = values(BRANCHES * hidden, 7, 1.0);
    let plan = feed.plan(PROMPT[0]).unwrap();
    let gathered = feed.gather(&plan).unwrap();
    let first = feed.update(PROMPT[0], &gathered, &stream).unwrap();
    assert_eq!(feed.state.history(), &PROMPT[..1]);
    // The end-of-sequence token empties the history. The sparse table holds holes at its rows
    // (scale bits zero), which decode to the head bias: the gather still yields 16 rows.
    let plan = feed.plan(EOS).unwrap();
    let gathered_eos = feed.gather(&plan).unwrap();
    assert_eq!(gathered_eos.len(), 16);
    feed.update(EOS, &gathered_eos, &stream).unwrap();
    assert!(feed.state.history().is_empty());
    feed.reset();
    let again = feed.update(PROMPT[0], &gathered, &stream).unwrap();
    assert_eq!(first, again, "a reset feed repeats the first token");
}

#[test]
#[ignore = "opens the full local Flash-Next pack in FIFI_FLASH_NEXT_EXL3_PACK (30 GiB n-gram table)"]
fn the_real_table_gives_the_recorded_rows_and_the_real_projections_run() {
    use crate::family::flash_next::load::pack_files::FlashPackFiles;
    let dir = std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK");
    let files = FlashPackFiles::open(Path::new(&dir)).expect("open the pack");
    let (shape, _) = crate::family::flash_next::load::model_shape(&files.checkpoint.model_config)
        .expect("model shape");
    let layer = shape.ple_layer;
    let (vectors, projection) =
        PleVectors::read(&files.checkpoint.source, layer, &shape.ple, 8).expect("PLE tensors");
    let rows = NgramRows::new(files.ngram).expect("n-gram rows");
    let hidden = shape.ple.dims().hidden as usize;
    let mut feed = PleFeed::new(shape.ple, rows, vectors, Box::new(projection));
    let recorded_rows = manifest_integers("global_rows");
    for (index, &token) in PROMPT.iter().enumerate() {
        let plan = feed.plan(token).unwrap();
        let started = std::time::Instant::now();
        let gathered = feed.gather(&plan).unwrap();
        eprintln!("token {token}: 16 rows in {:?}", started.elapsed());
        for head in 0..16 {
            let row = recorded_rows[index * 16 + head] as u64;
            let got = gathered.get(row).expect("the recorded row");
            let exact = recorded(index, head);
            assert!(
                got.iter()
                    .zip(&exact)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        let stream = values(BRANCHES * hidden, 90 + index, 1.0);
        let started = std::time::Instant::now();
        let update = feed.update(token, &gathered, &stream).unwrap();
        eprintln!("token {token}: update in {:?}", started.elapsed());
        assert_eq!(update.len(), BRANCHES * hidden);
        assert!(update.iter().all(|v| v.is_finite()));
    }
}

/// A route that counts its calls and can be made to fail; the step must call it once per update
/// (the keys and the value of one embedding come from one `project`), and a failure must leave the
/// state alone.
struct Counting {
    inner: HostProjection,
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    fail: bool,
}

impl PleProjection for Counting {
    fn project(&mut self, embedding: &[f32]) -> Result<PleKeyValue, EngError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.fail {
            return Err(EngError::BadRequest("route failed".into()));
        }
        self.inner.project(embedding)
    }
}

#[test]
fn the_projection_route_is_called_once_per_update_and_its_failure_is_not_applied() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ngram_embedding.safetensors");
    write_table(&path);
    let build = |fail: bool, calls: &std::sync::Arc<std::sync::atomic::AtomicUsize>| {
        let file = NgramFile::open(&path, &LocalGuard::new().unwrap(), KEY, LAYOUT).unwrap();
        let shape = PleShape::new(dims(8)).unwrap();
        let (vectors, inner) = host(&shape);
        let route = Counting {
            inner,
            calls: calls.clone(),
            fail,
        };
        PleFeed::new(
            shape,
            NgramRows::new(file).unwrap(),
            vectors,
            Box::new(route),
        )
    };
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut feed = build(false, &calls);
    let mut reference = feed_on(dir.path(), 8);
    let stream = values(BRANCHES * 8, 11, 1.0);
    let plan = feed.plan(PROMPT[0]).unwrap();
    let gathered = feed.gather(&plan).unwrap();
    let got = feed.update(PROMPT[0], &gathered, &stream).unwrap();
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    let plan = reference.plan(PROMPT[0]).unwrap();
    let want = reference
        .update(PROMPT[0], &reference.gather(&plan).unwrap(), &stream)
        .unwrap();
    assert_eq!(got, want, "a route that delegates gives the host numbers");

    let failing_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut failing = build(true, &failing_calls);
    let plan = failing.plan(PROMPT[0]).unwrap();
    let gathered = failing.gather(&plan).unwrap();
    assert!(failing.update(PROMPT[0], &gathered, &stream).is_err());
    assert!(failing.state.history().is_empty());
}
