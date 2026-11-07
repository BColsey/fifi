//! Device tests of `verify_greedy`, the device-side comparison of a greedy K-row window: the
//! accepted run and the bonus token it stores against a host reference, on synthetic tables and on
//! the whole chain from logits (`argmax_rows`, then `verify_greedy`) where exact ties decide the
//! tokens. A host without a device skips them.

use super::{
    ArgmaxRowsCall, VerifyGreedyCall, WindowReport, argmax_rows_part_bytes, window_report_bytes,
};
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceStream};

const STALE: u8 = 0x77;

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

/// The accepted run and the bonus of a window, from the plain statement of the rule: the leading
/// rows `i < m - 1` whose token (at least 0: a row without a token never extends the run) equals the
/// next window id, and the token of the row after them.
fn reference(tokens: &[i32], ids: &[i32]) -> (i32, i32) {
    let mut run = 0;
    while run + 1 < tokens.len() && tokens[run] >= 0 && tokens[run] == ids[run + 1] {
        run += 1;
    }
    (run as i32, tokens[run])
}

fn words(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|word| word.to_le_bytes()).collect()
}

impl Rig {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        Self { ctx, stream }
    }

    /// Runs `verify_greedy` on a table of `m` pairs (token, bits 0) with the window word `window`,
    /// a zero pad, and stale bytes where the verify words go. Returns the report bytes read back.
    fn verify(&self, tokens: &[i32], ids: &[i32], window: i32) -> Vec<u8> {
        let m = tokens.len();
        let mut table = Vec::new();
        for &token in tokens {
            table.extend(words(&[token, 0]));
        }
        table.extend(words(&[window, 0]));
        table.extend([STALE; 8]);
        let result = testkit::upload_bytes(&self.ctx, &table);
        let ids_buf = testkit::upload_bytes(&self.ctx, &words(ids));
        self.ctx
            .verify_greedy(
                &self.stream,
                VerifyGreedyCall {
                    m,
                    ids: ids_buf.range(0, 4 * m).unwrap(),
                    result: result.range(0, window_report_bytes(m)).unwrap(),
                },
            )
            .expect("verify_greedy");
        testkit::read_bytes(&self.ctx, &result, window_report_bytes(m))
    }

    /// The chain: `argmax_rows` on the rows (stride `n`), then `verify_greedy` against `ids`.
    fn chain(&self, rows: &[Vec<f32>], ids: &[i32]) -> WindowReport {
        let (m, n) = (rows.len(), rows[0].len());
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let logits = testkit::upload_f32(&self.ctx, &flat);
        let part = testkit::upload_filled(&self.ctx, 0xa5, argmax_rows_part_bytes(m));
        // The table: pairs unwritten (the argmax writes them), window word and pad zero, stale
        // verify words.
        let mut table = vec![0u8; 8 * m + 8];
        table.extend([STALE; 8]);
        let result = testkit::upload_bytes(&self.ctx, &table);
        let ids_buf = testkit::upload_bytes(&self.ctx, &words(ids));
        self.ctx
            .argmax_rows(
                &self.stream,
                ArgmaxRowsCall {
                    m,
                    n,
                    ld: n,
                    logits: logits.range(0, flat.len() * 4).unwrap(),
                    part: part.range(0, argmax_rows_part_bytes(m)).unwrap(),
                    result: result.range(0, 8 * m).unwrap(),
                },
            )
            .expect("argmax_rows");
        self.ctx
            .verify_greedy(
                &self.stream,
                VerifyGreedyCall {
                    m,
                    ids: ids_buf.range(0, 4 * m).unwrap(),
                    result: result.range(0, window_report_bytes(m)).unwrap(),
                },
            )
            .expect("verify_greedy");
        let bytes = testkit::read_bytes(&self.ctx, &result, window_report_bytes(m));
        WindowReport::from_bytes(&bytes, m, false).expect("a whole report")
    }
}

fn decode(bytes: &[u8], m: usize) -> WindowReport {
    WindowReport::from_bytes(bytes, m, false).expect("a whole report")
}

/// All accepted, rejected at the first row, kept in the middle, a mismatch at the last draft:
/// the run and the bonus are the host reference's, for 1 to 8 rows, and the pairs and the window
/// word are left as they were.
#[test]
fn verify_greedy_gives_the_accepted_run_and_the_bonus_of_the_host_rule() {
    if !testkit::device_present("verify_greedy against the host rule") {
        return;
    }
    let rig = Rig::new();
    for m in 1..=8usize {
        let tokens: Vec<i32> = (0..m as i32).map(|i| 100 + 7 * i).collect();
        let mut cases: Vec<Vec<i32>> = Vec::new();
        // ids[i + 1] == tokens[i] for every draft: all accepted.
        let all: Vec<i32> = std::iter::once(5)
            .chain(tokens.iter().copied())
            .take(m)
            .collect();
        cases.push(all.clone());
        for stop in 0..m.saturating_sub(1) {
            // The first mismatch is draft `stop`; later drafts match again and must not count.
            let mut ids = all.clone();
            ids[stop + 1] += 1;
            cases.push(ids);
        }
        for ids in cases {
            let bytes = rig.verify(&tokens, &ids, 18);
            let report = decode(&bytes, m);
            let (run, bonus) = reference(&tokens, &ids);
            assert_eq!(
                (report.accepted, report.bonus),
                (run, bonus),
                "m {m} ids {ids:?}"
            );
            assert_eq!(report.table.tokens, tokens, "pairs untouched, m {m}");
            assert_eq!(report.table.window, 18, "window word untouched, m {m}");
            // The run is a count of rows, the bonus is the token of the row after the run.
            assert!((run as usize) < m);
            assert_eq!(bonus, tokens[run as usize]);
        }
    }
}

/// Named cases at 8 rows: reject-first (run 0, bonus is row 0's token), keep-middle, all accepted
/// (run 7, bonus is the last row's token), and the first mismatch ends the run although later
/// drafts match.
#[test]
fn verify_greedy_reject_first_keep_middle_and_all_accepted_at_eight_rows() {
    if !testkit::device_present("verify_greedy named cases") {
        return;
    }
    let rig = Rig::new();
    let tokens = [11, 12, 13, 14, 15, 16, 17, 18];
    let accepted = |bytes: Vec<u8>| {
        let report = decode(&bytes, 8);
        (report.accepted, report.bonus)
    };
    // ids[i + 1] is the draft after row i; ids[0] is the pending token and never compared.
    assert_eq!(
        accepted(rig.verify(&tokens, &[0, 99, 12, 13, 14, 15, 16, 17], 0)),
        (0, 11),
        "reject first"
    );
    assert_eq!(
        accepted(rig.verify(&tokens, &[0, 11, 12, 13, 14, 15, 16, 17], 0)),
        (7, 18),
        "all accepted"
    );
    assert_eq!(
        accepted(rig.verify(&tokens, &[0, 11, 12, 13, 99, 15, 16, 17], 0)),
        (3, 14),
        "keep middle: the mismatch at draft 3 ends the run, drafts 4.. match again and do not count"
    );
    assert_eq!(
        accepted(rig.verify(&tokens, &[0, 11, 12, 13, 14, 15, 16, 99], 0)),
        (6, 17),
        "the last draft is the one that mismatches"
    );
}

/// A row with no comparable value (token -1) never equals a window id and ends the run; the
/// bonus is then -1 and the host's token check refuses it.
#[test]
fn verify_greedy_stops_at_a_row_without_a_token() {
    if !testkit::device_present("verify_greedy with a -1 token") {
        return;
    }
    let rig = Rig::new();
    let report = decode(&rig.verify(&[4, -1, 6, 7], &[0, 4, -1, 6], 2), 4);
    assert_eq!(
        report.accepted, 1,
        "row 1 has no token: it never matches draft 2"
    );
    assert_eq!(report.bonus, -1);
}

/// 600 random windows over a small alphabet (so matches are common and the first mismatch lands
/// everywhere), 1 to 8 rows, with -1 among the tokens and among the window ids (so that a row
/// without a token meets an id holding the same bits): the same answer as the host rule.
#[test]
fn verify_greedy_matches_the_host_rule_on_random_windows() {
    if !testkit::device_present("verify_greedy random windows") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x5635_0001);
    for case in 0..600 {
        let m = 1 + rng.below(8) as usize;
        let tokens: Vec<i32> = (0..m).map(|_| rng.below(4) as i32 - 1).collect();
        let ids: Vec<i32> = (0..m).map(|_| rng.below(4) as i32 - 1).collect();
        let report = decode(&rig.verify(&tokens, &ids, 0), m);
        let (run, bonus) = reference(&tokens, &ids);
        assert_eq!(
            (report.accepted, report.bonus),
            (run, bonus),
            "case {case}: {tokens:?} {ids:?}"
        );
    }
}

/// The whole chain with exact ties: the argmax takes the lowest index of a tie, so a window id
/// equal to the lowest tied index is accepted and one equal to a higher tied index is not. A
/// reversed tie order (the highest index wins) turns every accepted draft here into a rejection.
#[test]
fn the_chain_from_logits_takes_the_first_index_of_a_tie() {
    if !testkit::device_present("argmax_rows then verify_greedy on ties") {
        return;
    }
    let rig = Rig::new();
    let n = 1_000;
    let tied = |at: &[usize]| -> Vec<f32> {
        let mut row: Vec<f32> = (0..n).map(|i| -2.0 - (i % 11) as f32 * 0.125).collect();
        for &i in at {
            row[i] = 4.5;
        }
        row
    };
    // Row r ties at the indices below; its argmax token is the first of them.
    let rows = vec![
        tied(&[40, 41, 900]),
        tied(&[255, 256, 999]),
        tied(&[7, 8]),
        tied(&[500, 501, 502]),
    ];
    let first = [40, 255, 7, 500];
    // Every draft is the first tied index of the row before it: all accepted, bonus is row 3's.
    let ids: Vec<i32> = std::iter::once(1)
        .chain(first.iter().copied())
        .take(4)
        .collect();
    let report = rig.chain(&rows, &ids);
    assert_eq!(report.table.tokens, first);
    assert_eq!((report.accepted, report.bonus), (3, 500));
    // A draft equal to the SECOND tied index of row 1 is a mismatch there.
    let mut ids = ids.clone();
    ids[2] = 256;
    let report = rig.chain(&rows, &ids);
    assert_eq!((report.accepted, report.bonus), (1, 255));
}
