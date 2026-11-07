//! Tests of the row-batched argmax and the window status fold (unit test 8 of the K-row design).
//!
//! The device tests compare `argmax_rows` of m rows with m separate launches of the single-row
//! `argmax` on the same rows, pair for pair (token and status bits), and run the status rules the
//! design relies on: each pair carries only its own row's bits, the step parameters are not
//! written by `argmax_rows`, the fold ORs the pairs and the EXL3 flag into the status word exactly
//! once and leaves the pairs alone. A host without a device skips them. The tests without a device
//! (the table decoder, the sizes, the refusals that are decided before any native call) run
//! anywhere the native tree builds.

use super::{
    ArgmaxRowsCall, MAX_ARGMAX_ROWS, WindowFoldCall, WindowResult, argmax_rows_part_bytes,
    window_result_bytes,
};
use crate::EngError;
use crate::kernels::cuda::resident::step_params::{
    ARGMAX_PART_BYTES, ArgmaxCall, STATUS_EXL3_ARITHMETIC, STATUS_NON_FINITE_LOGITS,
    STATUS_ROUTER_FAULT, STEP_PARAMS_BYTES, StepParams,
};
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

const VOCAB: usize = 248_320;
const NAN: f32 = f32::NAN;

#[test]
fn the_table_decodes_pairs_the_window_word_and_a_zero_pad() {
    let words: [i32; 8] = [7, 0, 9, 2, -1, 0, 18, 0];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let decoded = WindowResult::from_bytes(&bytes, 3).expect("table of 3 rows");
    assert_eq!(decoded.tokens, [7, 9, -1]);
    assert_eq!(decoded.bits, [0, 2, 0]);
    assert_eq!(decoded.window, 18);
    assert_eq!(window_result_bytes(3), bytes.len());
    // A short or long table, and a table whose pad word is not zero, are not a table.
    assert_eq!(WindowResult::from_bytes(&bytes[..bytes.len() - 4], 3), None);
    assert_eq!(WindowResult::from_bytes(&bytes, 2), None);
    let mut dirty = bytes.clone();
    dirty[28] = 1;
    assert_eq!(WindowResult::from_bytes(&dirty, 3), None);
}

#[test]
fn the_sizes_are_one_set_of_partials_per_row_and_a_pair_per_row_plus_the_window_word() {
    for m in 1..=MAX_ARGMAX_ROWS {
        assert_eq!(argmax_rows_part_bytes(m), m * ARGMAX_PART_BYTES);
        assert_eq!(window_result_bytes(m), 8 * m + 8);
    }
    assert_eq!(MAX_ARGMAX_ROWS, 16);
}

/// The scratch of a shortlist launch shares the argmax partials slot of the plan
/// (`Slot::ArgmaxPart`); a K-row plan sizes it as `rows * ARGMAX_PART_BYTES`, so each row's
/// shortlist scratch (one per row, step 2) fits its own set of partials.
#[test]
fn a_shortlist_scratch_fits_one_set_of_partials() {
    use crate::kernels::cuda::resident::shortlist::SCRATCH_BYTES;
    const { assert!(SCRATCH_BYTES <= ARGMAX_PART_BYTES) };
}

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

    fn params(&self, status: i32) -> DeviceBuffer {
        let block = StepParams {
            token_id: 0x1234,
            pos: 10,
            kv_len: 11,
            kv_slot: 10,
            status,
            pad: [0x2222_2222, 0x3333_3333, 0x4444_4444],
        };
        testkit::upload_bytes(&self.ctx, &block.to_bytes())
    }

    fn read_params(&self, params: &DeviceBuffer) -> [u8; STEP_PARAMS_BYTES] {
        testkit::read_bytes(&self.ctx, params, STEP_PARAMS_BYTES)
            .try_into()
            .expect("32 bytes")
    }

    /// The single-row kernel on one row: the pair it gives.
    fn single(&self, row: &[f32]) -> (i32, i32) {
        let logits = testkit::upload_f32(&self.ctx, row);
        let part = testkit::upload_filled(&self.ctx, 0xa5, ARGMAX_PART_BYTES);
        let params = self.params(0);
        let result = testkit::upload_filled(&self.ctx, 0x77, 8);
        self.ctx
            .argmax(
                &self.stream,
                ArgmaxCall {
                    n: row.len(),
                    logits: logits.range(0, row.len() * 4).unwrap(),
                    part: part.range(0, ARGMAX_PART_BYTES).unwrap(),
                    params: params.range(0, STEP_PARAMS_BYTES).unwrap(),
                    result: Some(result.range(0, 8).unwrap()),
                },
            )
            .expect("single-row argmax");
        let bytes = testkit::read_bytes(&self.ctx, &result, 8);
        let word = |i: usize| i32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        (word(0), word(1))
    }

    /// `argmax_rows` on the rows (stride `ld` floats, padding NaN so that a read past a row's
    /// `n` values would show), then the fold with the given status and flag word. Returns the
    /// table and the step parameter bytes after both.
    fn rows(
        &self,
        rows: &[Vec<f32>],
        ld: usize,
        status: i32,
        flag: u32,
    ) -> (WindowResult, [u8; STEP_PARAMS_BYTES], u32) {
        let (m, n) = (rows.len(), rows[0].len());
        let mut flat = vec![NAN; (m - 1) * ld + n];
        for (r, row) in rows.iter().enumerate() {
            flat[r * ld..r * ld + n].copy_from_slice(row);
        }
        let logits = testkit::upload_f32(&self.ctx, &flat);
        let part = testkit::upload_filled(&self.ctx, 0xa5, argmax_rows_part_bytes(m));
        let table = testkit::upload_filled(&self.ctx, 0x77, window_result_bytes(m));
        let params = self.params(status);
        let flag_buf = testkit::upload_bytes(&self.ctx, &flag.to_le_bytes());
        self.ctx
            .argmax_rows(
                &self.stream,
                ArgmaxRowsCall {
                    m,
                    n,
                    ld,
                    logits: logits.range(0, flat.len() * 4).unwrap(),
                    part: part.range(0, argmax_rows_part_bytes(m)).unwrap(),
                    result: table.range(0, 8 * m).unwrap(),
                },
            )
            .expect("argmax_rows");
        let after_rows = self.read_params(&params);
        assert_eq!(
            after_rows,
            StepParams {
                token_id: 0x1234,
                pos: 10,
                kv_len: 11,
                kv_slot: 10,
                status,
                pad: [0x2222_2222, 0x3333_3333, 0x4444_4444],
            }
            .to_bytes(),
            "argmax_rows must not write the step parameters"
        );
        self.ctx
            .window_status_fold(
                &self.stream,
                WindowFoldCall {
                    m,
                    params: params.range(0, STEP_PARAMS_BYTES).unwrap(),
                    flag: flag_buf.range(0, 4).unwrap(),
                    result: table.range(0, window_result_bytes(m)).unwrap(),
                },
            )
            .expect("window_status_fold");
        let bytes = testkit::read_bytes(&self.ctx, &table, window_result_bytes(m));
        let decoded = WindowResult::from_bytes(&bytes, m).expect("a whole table");
        let flag_after = u32::from_le_bytes(
            testkit::read_bytes(&self.ctx, &flag_buf, 4)
                .try_into()
                .unwrap(),
        );
        (decoded, self.read_params(&params), flag_after)
    }
}

fn status_of(bytes: &[u8; STEP_PARAMS_BYTES]) -> i32 {
    StepParams::from_bytes(*bytes).status
}

fn random_row(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * 3.0) as f32).collect()
}

/// A row whose maximum appears at every index of `at` (an exact tie), all else below.
fn tied_row(n: usize, at: &[usize]) -> Vec<f32> {
    let mut row: Vec<f32> = (0..n).map(|i| -2.0 - (i % 11) as f32 * 0.125).collect();
    for &i in at {
        row[i] = 4.5;
    }
    row
}

/// Rows built to hit the reduction at its seams: ties across blocks (a block is 256 threads and a
/// grid-stride loop, so indexes 255/256, 65535/65536 and the last one sit in different threads,
/// blocks and loop trips), NaN (alone, and the only value), all equal, infinities.
fn hard_rows(n: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    let last = n - 1;
    let mut rows = vec![
        tied_row(n, &[0, 255, 256]),
        tied_row(n, &[last, last - 1, 65_535.min(last), 65_536.min(last)]),
        tied_row(n, &[3, 65_536.min(last) + 3, last]),
        vec![0.5; n],
        random_row(rng, n),
    ];
    let mut with_nan = random_row(rng, n);
    with_nan[n / 3] = NAN;
    rows.push(with_nan);
    rows.push(vec![NAN; n]);
    let mut only_inf = random_row(rng, n);
    only_inf[7] = f32::INFINITY;
    only_inf[n / 2] = f32::INFINITY;
    rows.push(only_inf);
    rows.push(vec![f32::NEG_INFINITY; n]);
    let mut nan_over_tie = tied_row(n, &[10, 300]);
    nan_over_tie[5] = NAN;
    rows.push(nan_over_tie);
    rows
}

/// Unit test 8a: every row's pair equals the pair of the single-row kernel on that row, at the
/// real vocabulary width and at a short one, for 1 to 16 rows and for a padded row stride.
#[test]
fn argmax_rows_pairs_equal_the_single_row_kernel_on_every_row() {
    if !testkit::device_present("argmax_rows pairs against the single-row kernel") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4b52_0001);
    for n in [VOCAB, 1_000, 257] {
        let pool = hard_rows(n, &mut rng);
        let singles: Vec<(i32, i32)> = pool.iter().map(|row| rig.single(row)).collect();
        for m in [1usize, 2, 3, 4, 5, 8, 16] {
            let picks: Vec<usize> = (0..m).map(|r| (r * 3 + m) % pool.len()).collect();
            let rows: Vec<Vec<f32>> = picks.iter().map(|&p| pool[p].clone()).collect();
            for ld in [n, n + 5, n + 4096] {
                let (table, _, _) = rig.rows(&rows, ld, 0, 0);
                for (r, &p) in picks.iter().enumerate() {
                    assert_eq!(
                        (table.tokens[r], table.bits[r]),
                        singles[p],
                        "n {n} m {m} ld {ld} row {r} (pool row {p})"
                    );
                }
            }
        }
    }
}

/// Unit test 8b: ties go to the lowest index, a NaN never wins, nothing is comparable gives -1, in
/// the rows kernel as the rule says (the host model, not only the single-row kernel).
#[test]
fn argmax_rows_follow_the_tie_and_nan_rules() {
    if !testkit::device_present("argmax_rows tie and NaN rules") {
        return;
    }
    let rig = Rig::new();
    let n = VOCAB;
    let rows = vec![
        tied_row(n, &[200_000, 12, 70_000]),
        vec![1.0; n],
        {
            let mut r = vec![-1.0; n];
            r[9] = NAN;
            r[10] = -0.5;
            r
        },
        vec![NAN; n],
    ];
    let (table, _, _) = rig.rows(&rows, n, 0, 0);
    assert_eq!(table.tokens, [12, 0, 10, -1]);
    assert_eq!(
        table.bits,
        [0, 0, STATUS_NON_FINITE_LOGITS, STATUS_NON_FINITE_LOGITS]
    );
    assert_eq!(table.window, STATUS_NON_FINITE_LOGITS);
}

/// Unit test 8c, the row-local bits under concurrency (mutant M17): one NaN row among eight, 100
/// windows. Only that row's pair carries the bit, in every window; the window word carries it once.
/// A pair that stored the return value of `atomicOr` on the shared status word would show the bit
/// in some earlier-scheduled row in some window.
#[test]
fn the_status_bits_stay_in_the_row_that_has_them_over_100_windows() {
    if !testkit::device_present("argmax_rows row-local bits") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4b52_0002);
    let n = VOCAB;
    for bad in [0usize, 3, 7] {
        let mut rows: Vec<Vec<f32>> = (0..8).map(|_| random_row(&mut rng, n)).collect();
        rows[bad][n - 1] = NAN;
        for window in 0..100 {
            let (table, params, _) = rig.rows(&rows, n, 0, 0);
            let expected: Vec<i32> = (0..8)
                .map(|r| {
                    if r == bad {
                        STATUS_NON_FINITE_LOGITS
                    } else {
                        0
                    }
                })
                .collect();
            assert_eq!(table.bits, expected, "bad row {bad}, window {window}");
            assert_eq!(
                table.window, STATUS_NON_FINITE_LOGITS,
                "bad row {bad}, window {window}"
            );
            assert_eq!(
                status_of(&params),
                STATUS_NON_FINITE_LOGITS,
                "window {window}"
            );
        }
    }
}

/// The fold: it ORs the pairs and the EXL3 flag into the status once, keeps what the earlier
/// kernels wrote (router fault, here), zeroes the flag word, and leaves every pair alone.
#[test]
fn the_fold_ors_the_pairs_and_the_flag_into_the_status_once_and_leaves_the_pairs() {
    if !testkit::device_present("window_status_fold") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4b52_0003);
    let n = 4096;
    let clean: Vec<Vec<f32>> = (0..4).map(|_| random_row(&mut rng, n)).collect();
    let mut one_nan = clean.clone();
    one_nan[2][11] = f32::INFINITY;

    // Nothing set: the window word and the status stay 0.
    let (table, params, flag) = rig.rows(&clean, n, 0, 0);
    assert_eq!((table.window, status_of(&params), flag), (0, 0, 0));
    assert_eq!(table.bits, [0, 0, 0, 0]);

    // The flag alone: mask 16 in the word and the status, in no pair; the flag word is zeroed.
    let (table, params, flag) = rig.rows(&clean, n, 0, 1);
    assert_eq!(table.window, STATUS_EXL3_ARITHMETIC);
    assert_eq!(status_of(&params), STATUS_EXL3_ARITHMETIC);
    assert_eq!(table.bits, [0, 0, 0, 0]);
    assert_eq!(flag, 0, "the fold zeroes the flag word");

    // Only bit 0 of the flag word means the arithmetic flag.
    let (table, _, _) = rig.rows(&clean, n, 0, 2);
    assert_eq!(table.window, 0);

    // A status written by an earlier kernel (router fault) arrives in the word, in no pair; a
    // non-finite row adds mask 2 to its own pair, the word and the status.
    let (table, params, _) = rig.rows(&one_nan, n, STATUS_ROUTER_FAULT, 1);
    assert_eq!(table.bits, [0, 0, STATUS_NON_FINITE_LOGITS, 0]);
    let all = STATUS_ROUTER_FAULT | STATUS_NON_FINITE_LOGITS | STATUS_EXL3_ARITHMETIC;
    assert_eq!(table.window, all);
    assert_eq!(status_of(&params), all);
}

/// Every window rewrites the whole table, so a stale table (a window of other rows, a larger one)
/// can never be read for the new one.
#[test]
fn every_launch_rewrites_every_pair_and_the_window_word() {
    if !testkit::device_present("argmax_rows rewrites the table") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4b52_0004);
    let n = 2048;
    let rows: Vec<Vec<f32>> = (0..3).map(|_| random_row(&mut rng, n)).collect();
    // The table starts as 0x77 bytes (rig.rows); after the launch no byte of it is left.
    let (table, _, _) = rig.rows(&rows, n, 0, 0);
    for row in 0..3 {
        assert_ne!(table.tokens[row], 0x7777_7777);
        assert_eq!(table.bits[row], 0);
    }
    assert_eq!(table.window, 0);
}

fn refusal(result: Result<(), EngError>, needle: &str, case: &str) {
    match result {
        Err(EngError::BadRequest(message)) if message.contains(needle) => {}
        other => panic!("{case}: expected BadRequest containing {needle:?}, got {other:?}"),
    }
}

/// Every refusal is a `BadRequest` before any native call, by the text of its error.
#[test]
fn the_wrappers_refuse_bad_arguments_before_any_native_call() {
    if !testkit::device_present("argmax_rows refusals") {
        return;
    }
    let rig = Rig::new();
    let n = 64;
    let logits = testkit::upload_f32(&rig.ctx, &vec![1.0; 8 * n]);
    let part = testkit::upload_filled(&rig.ctx, 0, argmax_rows_part_bytes(8));
    let table = testkit::upload_filled(&rig.ctx, 0, window_result_bytes(8) + 16);
    let params = rig.params(0);
    let flag = testkit::upload_bytes(&rig.ctx, &0u32.to_le_bytes());
    let call = |m: usize, ld: usize, n: usize| ArgmaxRowsCall {
        m,
        n,
        ld,
        logits: logits.range(0, 8 * 64 * 4).unwrap(),
        part: part.range(0, argmax_rows_part_bytes(8)).unwrap(),
        result: table.range(0, 8 * 8).unwrap(),
    };
    refusal(
        rig.ctx.argmax_rows(&rig.stream, call(0, n, n)),
        "1 to 16 rows",
        "m = 0",
    );
    refusal(
        rig.ctx.argmax_rows(&rig.stream, call(17, n, n)),
        "1 to 16 rows",
        "m = 17",
    );
    refusal(
        rig.ctx.argmax_rows(&rig.stream, call(2, n, 0)),
        "values per row",
        "n = 0",
    );
    refusal(
        rig.ctx.argmax_rows(&rig.stream, call(2, 32, n)),
        "row stride",
        "ld < n",
    );
    // Short logits: 8 rows of 64 at stride 128 need 7 * 128 + 64 floats.
    refusal(
        rig.ctx.argmax_rows(&rig.stream, call(8, 128, n)),
        "logits",
        "short logits",
    );
    // Short scratch: one set of partials for two rows.
    let mut short_part = call(2, n, n);
    short_part.part = part.range(0, ARGMAX_PART_BYTES).unwrap();
    refusal(
        rig.ctx.argmax_rows(&rig.stream, short_part),
        "part",
        "short scratch",
    );
    // A result table at a 4-byte offset is not 8-byte aligned.
    let mut misaligned = call(2, n, n);
    misaligned.result = table.range(4, 16).unwrap();
    refusal(
        rig.ctx.argmax_rows(&rig.stream, misaligned),
        "result",
        "misaligned table",
    );
    // The result may not overlap the logits.
    let mut overlap = call(2, n, n);
    overlap.result = logits.range(0, 16).unwrap();
    refusal(
        rig.ctx.argmax_rows(&rig.stream, overlap),
        "overlap",
        "result over logits",
    );

    let fold = |m: usize, bytes: usize| WindowFoldCall {
        m,
        params: params.range(0, STEP_PARAMS_BYTES).unwrap(),
        flag: flag.range(0, 4).unwrap(),
        result: table.range(0, bytes).unwrap(),
    };
    refusal(
        rig.ctx.window_status_fold(&rig.stream, fold(0, 16)),
        "1 to 16 rows",
        "fold m = 0",
    );
    // The pairs alone (8 m bytes) are too short for the table: the window word needs 8 more.
    refusal(
        rig.ctx.window_status_fold(&rig.stream, fold(4, 32)),
        "result",
        "table without w",
    );
}
