//! Device tests of the K-row shortlist: record `r` of one K-row launch against the record a
//! single-row launch gives row `r`, byte for byte (the selection is comparisons only, so the two
//! must be the same bytes), on rows built to hit the ties, the signed zeros, the short vocabularies
//! and the non-finite certificate, with stale scratch and output, a padded row stride and a
//! repeated launch. A host without a device skips them.

use super::*;
use crate::kernels::cuda::resident::step_params::ARGMAX_PART_BYTES;
use crate::kernels::cuda::testkit::{self, Rng};

const STALE: u8 = 0xA5;

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        Self { ctx, stream }
    }

    /// The record the single-row launch gives `row`.
    fn single(&self, row: &[f32]) -> [u8; RESULT_BYTES] {
        let input = testkit::upload_f32(&self.ctx, row);
        let scratch = testkit::upload_filled(&self.ctx, STALE, 3072);
        self.ctx
            .shortlist(
                &self.stream,
                ShortlistCall {
                    logits: input.range(0, row.len() * 4).unwrap(),
                    scratch: scratch.range(0, 3072).unwrap(),
                    n: row.len(),
                },
            )
            .expect("single-row shortlist");
        testkit::read_bytes(&self.ctx, &scratch, 3072)[RESULT_OFFSET..RESULT_OFFSET + RESULT_BYTES]
            .try_into()
            .expect("a record")
    }

    /// One K-row launch over `rows` stored `ld` floats apart (the padding is NaN, so a read past a
    /// row's `n` values would show). Scratch and output start stale. Returns the records.
    fn rows(&self, rows: &[Vec<f32>], ld: usize, stride: usize) -> Vec<[u8; RESULT_BYTES]> {
        let (m, n) = (rows.len(), rows[0].len());
        let mut flat = vec![f32::NAN; (m - 1) * ld + n];
        for (r, row) in rows.iter().enumerate() {
            flat[r * ld..r * ld + n].copy_from_slice(row);
        }
        let logits = testkit::upload_f32(&self.ctx, &flat);
        let scratch = testkit::upload_filled(&self.ctx, STALE, m * stride);
        let out = testkit::upload_filled(&self.ctx, STALE, m * RESULT_BYTES);
        self.ctx
            .shortlist_rows(
                &self.stream,
                ShortlistRowsCall {
                    m,
                    n,
                    ld,
                    logits: logits.range(0, flat.len() * 4).unwrap(),
                    scratch: scratch.range(0, m * stride).unwrap(),
                    scratch_stride: stride,
                    out: out.range(0, m * RESULT_BYTES).unwrap(),
                },
            )
            .expect("K-row shortlist");
        let bytes = testkit::read_bytes(&self.ctx, &out, m * RESULT_BYTES);
        bytes.as_chunks::<RESULT_BYTES>().0.to_vec()
    }
}

/// The number the header gives `INFERENCE_CUDA_SHORTLIST_<name>`.
fn header_value(name: &str) -> usize {
    const HEADER: &str = include_str!("../../native/resident/shortlist.h");
    let key = format!("INFERENCE_CUDA_SHORTLIST_{name} =");
    let line = HEADER
        .lines()
        .find(|line| line.trim_start().starts_with(&key))
        .unwrap_or_else(|| panic!("{key} is not in the header"));
    line.split('=')
        .nth(1)
        .and_then(|value| value.trim().trim_end_matches(',').parse().ok())
        .unwrap_or_else(|| panic!("no number in `{line}`"))
}

/// The sizes the Rust side mirrors from the header (and the K-row limit the kernel and the Rust
/// refusal both use) are the header's numbers. A host test: it needs no device.
#[test]
fn the_rust_constants_are_those_of_the_header() {
    use crate::kernels::cuda::native::resident::shortlist as raw;
    assert_eq!(header_value("K"), K);
    assert_eq!(header_value("PART_BYTES"), RESULT_BYTES);
    assert_eq!(header_value("RESULT_OFFSET"), RESULT_OFFSET);
    assert_eq!(header_value("SCRATCH_BYTES"), SCRATCH_BYTES);
    assert_eq!(header_value("MAX_ROWS"), raw::MAX_ROWS);
    assert_eq!(header_value("TILES") * RESULT_BYTES, PARTS_BYTES);
}

fn random_row(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * 3.0) as f32).collect()
}

/// Rows with exact ties across tiles (a tile is a sixteenth of the row), ties inside the 20th
/// place, signed zeros, few distinct values, a descending and an ascending ramp.
fn hard_rows(n: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    let last = n - 1;
    let tied = |at: &[usize]| -> Vec<f32> {
        let mut row: Vec<f32> = (0..n).map(|i| -2.0 - (i % 11) as f32 * 0.125).collect();
        for &i in at {
            row[i.min(last)] = 4.5;
        }
        row
    };
    let mut rows = vec![
        random_row(rng, n),
        tied(&[0, n / 16, n / 16 + 1, n / 2, last]),
        // 25 equal maxima: the 20 lowest ids win and ids 20.. are cut.
        tied(&(0..25).map(|i| i * 3).collect::<Vec<_>>()),
        vec![0.5; n],
        (0..n).map(|i| (i % 3) as f32).collect(),
        (0..n).map(|i| -(i as f32)).collect(),
        (0..n).map(|i| i as f32).collect(),
        (0..n)
            .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
            .collect(),
    ];
    let mut sparse = vec![-1.0f32; n];
    sparse[last / 3] = 1.0;
    rows.push(sparse);
    rows
}

/// The record a row must have by the plain statement of the order (larger logit first, then the
/// lower id; the bits of the logit kept): the 20 best entries, not the single-row kernel's answer,
/// so that a tie order reversed in both launches is not invisible.
fn reference(row: &[f32]) -> Vec<(u32, u32)> {
    let mut all: Vec<(u32, f32)> = row
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (i as u32, v))
        .collect();
    all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    all.truncate(K);
    all.into_iter().map(|(id, v)| (id, v.to_bits())).collect()
}

fn entries(record: &[u8; RESULT_BYTES], n: usize) -> Vec<(u32, u32)> {
    (0..n.min(K))
        .map(|at| {
            (
                u32::from_le_bytes(record[at * 8 + 4..at * 8 + 8].try_into().unwrap()),
                u32::from_le_bytes(record[at * 8..at * 8 + 4].try_into().unwrap()),
            )
        })
        .collect()
}

/// Record `r` of a K-row launch is the single-row record of row `r`, for 1 to 8 rows (and 16),
/// every row length that matters (shorter than 20 values, exactly 20, one past it, a tile edge, the
/// real vocabulary), a tight and two padded row strides, and a larger scratch stride.
#[test]
fn each_k_row_record_is_the_single_row_record_of_its_row() {
    if !testkit::device_present("K-row shortlist against single-row launches") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x5c10_0001);
    for n in [1, 3, 19, 20, 21, 257, 4_096, 248_077] {
        let pool = hard_rows(n, &mut rng);
        let singles: Vec<[u8; RESULT_BYTES]> = pool.iter().map(|row| rig.single(row)).collect();
        for m in [1usize, 2, 3, 4, 5, 8, 16] {
            let picks: Vec<usize> = (0..m).map(|r| (r * 5 + m) % pool.len()).collect();
            let rows: Vec<Vec<f32>> = picks.iter().map(|&p| pool[p].clone()).collect();
            for (ld, stride) in [(n, 3072), (n + 243, 3072), (n + 4096, 4096)] {
                let records = rig.rows(&rows, ld, stride);
                for (r, &p) in picks.iter().enumerate() {
                    assert_eq!(
                        records[r], singles[p],
                        "n {n} m {m} ld {ld} stride {stride} row {r} (pool row {p})"
                    );
                    assert_eq!(
                        entries(&records[r], n),
                        reference(&pool[p]),
                        "n {n} m {m} row {r} (pool row {p}) against the statement of the order"
                    );
                }
            }
        }
    }
}

/// Distinct rows in an order that cannot be mistaken: record `r` is row `r`'s, not the record of
/// row `r - 1`, row 0 or the last row (the failure a launch with a wrong row index shows).
#[test]
fn k_row_records_follow_their_rows_in_order() {
    if !testkit::device_present("K-row shortlist row order") {
        return;
    }
    let rig = Rig::new();
    let n = 248_077;
    let rows: Vec<Vec<f32>> = (0..8)
        .map(|r| {
            let mut row = vec![-1.0f32; n];
            row[1_000 * (r + 1)] = 10.0 + r as f32;
            row
        })
        .collect();
    let records = rig.rows(&rows, 248_320, 3072);
    for (r, record) in records.iter().enumerate() {
        let id = u32::from_le_bytes(record[4..8].try_into().unwrap());
        let logit = f32::from_le_bytes(record[..4].try_into().unwrap());
        assert_eq!(
            (id as usize, logit),
            (1_000 * (r + 1), 10.0 + r as f32),
            "row {r}"
        );
        assert_eq!(record[..], rig.single(&rows[r])[..], "row {r}");
    }
}

/// A non-finite value sets the non-finite word of its own row's record only; the other records are
/// the single-row records of their rows.
#[test]
fn a_non_finite_row_flags_its_own_record_only() {
    if !testkit::device_present("K-row shortlist non-finite certificate") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x5c10_0002);
    let n = 4_099;
    for bad in [0usize, 2, 5] {
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut rows: Vec<Vec<f32>> = (0..6).map(|_| random_row(&mut rng, n)).collect();
            let at = rows[bad].len() - 2;
            rows[bad][at] = poison;
            let records = rig.rows(&rows, n, 3072);
            for (r, record) in records.iter().enumerate() {
                let flag = u32::from_le_bytes(record[160..164].try_into().unwrap());
                assert_eq!(flag != 0, r == bad, "bad row {bad} {poison}: row {r}");
                if r != bad {
                    assert_eq!(
                        *record,
                        rig.single(&rows[r]),
                        "bad row {bad} {poison}: row {r}"
                    );
                }
                assert_eq!(
                    record[164..176],
                    [0u8; 12],
                    "pad words are stored zero: row {r}"
                );
            }
        }
    }
}

/// 100 launches of 8 rows with ties give the same 8 records every time and the single-row ones
/// (no ordering of the concurrent blocks reaches a record).
#[test]
fn a_hundred_launches_give_the_same_records() {
    if !testkit::device_present("K-row shortlist repeated launches") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x5c10_0003);
    let n = 248_077;
    let pool = hard_rows(n, &mut rng);
    let rows: Vec<Vec<f32>> = pool[..8].to_vec();
    let singles: Vec<_> = rows.iter().map(|row| rig.single(row)).collect();
    for launch in 0..100 {
        assert_eq!(
            rig.rows(&rows, 248_320, ARGMAX_PART_BYTES),
            singles,
            "launch {launch}"
        );
    }
}

/// Refusals decided before any launch: no rows, too many rows, a vocabulary of zero, a row stride
/// below the row length, a scratch stride that is too small or not a multiple of 16, and a scratch
/// or output that overlaps the logits.
#[test]
fn k_row_shortlist_refuses_bad_shapes_before_any_launch() {
    if !testkit::device_present("K-row shortlist refusals") {
        return;
    }
    let rig = Rig::new();
    let m = 2usize;
    let logits = testkit::upload_f32(&rig.ctx, &vec![1.0f32; 64]);
    let scratch = testkit::upload_filled(&rig.ctx, STALE, 4 * 3072);
    let out = testkit::upload_filled(&rig.ctx, STALE, 4 * RESULT_BYTES);
    let call = |m: usize, n: usize, ld: usize, stride: usize| ShortlistRowsCall {
        m,
        n,
        ld,
        logits: logits.range(0, 64 * 4).unwrap(),
        scratch: scratch.range(0, 4 * 3072).unwrap(),
        scratch_stride: stride,
        out: out.range(0, 4 * RESULT_BYTES).unwrap(),
    };
    rig.ctx
        .shortlist_rows(&rig.stream, call(m, 32, 32, 3072))
        .expect("the control launch is accepted");
    for (what, bad) in [
        ("no rows", call(0, 32, 32, 3072)),
        ("17 rows", call(17, 32, 32, 3072)),
        ("no values", call(m, 0, 32, 3072)),
        ("stride below the row", call(m, 32, 31, 3072)),
        ("scratch stride below the partials", call(m, 32, 32, 2800)),
        ("scratch stride not a multiple of 16", call(m, 32, 32, 3080)),
        ("rows past the logits", call(m, 64, 64, 3072)),
    ] {
        assert!(rig.ctx.shortlist_rows(&rig.stream, bad).is_err(), "{what}");
    }
    // A scratch that starts inside the logits overlaps them: one buffer holds both.
    let shared = testkit::upload_f32(&rig.ctx, &vec![1.0f32; 2 * 3072]);
    let overlap = ShortlistRowsCall {
        m,
        n: 32,
        ld: 32,
        logits: shared.range(0, 64 * 4).unwrap(),
        scratch: shared.range(0, m * 3072).unwrap(),
        scratch_stride: 3072,
        out: out.range(0, 4 * RESULT_BYTES).unwrap(),
    };
    assert!(rig.ctx.shortlist_rows(&rig.stream, overlap).is_err());
}
