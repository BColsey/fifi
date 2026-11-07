//! The M3 quality gate: teacher-forced EXL3-vs-GGUF and chunked-vs-one-token
//! margins over the frozen M3 layout (registration v4).
//!
//! The pure core in this file is metric math only: paired differences, the
//! document-clustered bootstrap, and the one-sided bounds. It carries Linux
//! unit tests like `decode_prefill_kl`'s core; the model arms (the GGUF and
//! EXL3 loaders, the teacher-forced loops, the streamed sidecar writer) live
//! in the harness module added with the v4 registration.
//!
//! Frozen numbers (registration §3): NLL UB +0.02 nats, top-1 LB -0.5 pp,
//! copy-probe LB -1.0 pp, retrieval LB -2.0 pp; bootstrap B=10,000, seed
//! 20261031 (each bound re-seeds it, correlating the gates' draws — valid,
//! disclosed), one-sided 95% percentile, document clusters as the resample
//! unit. Every row of every arm is teacher-forced on the identical frozen id
//! stream; arms differ in weights, KV dtype, and prefill path per the
//! per-comparison pins.

/// The four gate margins of one comparison. All are candidate-minus-reference
/// differences; the NLL bound is a ceiling, the rest are floors (negative
/// floors tolerate a small drop, matching the registered quality standard
/// of 30 Oct 2026, recorded in `doc/explanation/status.md`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Margins {
    pub nll_ub: f64,
    pub top1_lb: f64,
    pub copy_lb: f64,
    pub retrieval_lb: f64,
}

impl Margins {
    /// The frozen registration values, in the statistic's units: NLL in
    /// nats, the three share floors in FRACTIONS (−0.5 pp, −1.0 pp, −2.0 pp).
    pub(in crate::family::qwen36) fn frozen() -> Self {
        Margins {
            nll_ub: 0.02,
            top1_lb: -0.005,
            copy_lb: -0.01,
            retrieval_lb: -0.02,
        }
    }
}

/// One scored position of one arm, as the streamed sidecar hands it to the
/// core. `logprob_true` is the natural log-probability of the frozen true
/// token (always <= 0; the receipt refuses a positive one), `top1_correct`
/// is the greedy argmax equality on the true token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::family::qwen36) struct PositionRow {
    pub cluster: u32,
    pub logprob_true: f32,
    pub top1_correct: bool,
}

/// One probe of one arm: all-token exact (copy continuations and retrieval
/// marker spans alike; v6 probes are all-token-exact by definition).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::family::qwen36) struct ProbeRow {
    pub cluster: u32,
    pub exact: bool,
}

/// Deterministic xorshift64* stream: the bootstrap must reproduce bit-for-bit
/// from the frozen seed on every host.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A resample index in `[0, n)`; rejection keeps the uniform law exact.
    fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0 && n < (1 << 32), "cluster count out of range");
        let zone = (u64::MAX / n) * n;
        loop {
            let v = self.next_u64();
            if v < zone {
                return v % n;
            }
        }
    }
}

/// The paired, document-clustered bootstrap bound of `metric(candidate) -
/// metric(reference)`. `metric` maps one arm's rows to its per-row statistic
/// (a mean NLL, an accuracy share); the bootstrap resamples clusters with
/// replacement and re-extracts the paired difference each draw. `upper`
/// picks the side: the one-sided 95% percentile of the draw distribution,
/// upper for the NLL ceiling, lower for the accuracy floors.
fn bootstrap_bound(
    candidate: &[f64],
    reference: &[f64],
    clusters: &[u32],
    n_clusters: u32,
    draws: usize,
    seed: u64,
    upper: bool,
) -> f64 {
    assert_eq!(candidate.len(), reference.len());
    assert_eq!(candidate.len(), clusters.len());
    assert!(n_clusters > 0);
    // Per-cluster sums, so one draw costs a walk over clusters, not rows.
    let mut cand_sum = vec![0.0_f64; n_clusters as usize];
    let mut ref_sum = vec![0.0_f64; n_clusters as usize];
    let mut counts = vec![0u64; n_clusters as usize];
    for ((c, r), k) in candidate.iter().zip(reference).zip(clusters) {
        cand_sum[*k as usize] += c;
        ref_sum[*k as usize] += r;
        counts[*k as usize] += 1;
    }
    let mut diffs = Vec::with_capacity(draws);
    let mut rng = Rng(seed | 1);
    for _ in 0..draws {
        let (mut cd, mut rd, mut nc) = (0.0_f64, 0.0_f64, 0u64);
        for _ in 0..n_clusters {
            let k = rng.below(n_clusters as u64) as usize;
            cd += cand_sum[k];
            rd += ref_sum[k];
            nc += counts[k];
        }
        diffs.push((cd - rd) / nc as f64);
    }
    diffs.sort_by(|a, b| a.total_cmp(b));
    // Nearest-rank 95%: index floor(0.95 * B) of the sorted draws.
    let idx = (0.95 * draws as f64).floor() as usize;
    if upper {
        diffs[idx.min(draws - 1)]
    } else {
        diffs[draws - 1 - idx.min(draws - 1)]
    }
}

/// The NLL rows of one arm: the per-row statistic is the true-token NLL
/// (negated logprob, higher = worse), so the paired difference of means is
/// the mean NLL delta and the +0.02-nat ceiling reads correctly. A positive
/// logprob row is refused (the receipt semantics: true log-probabilities are
/// always <= 0).
fn nll_rows(rows: &[PositionRow]) -> Result<Vec<f64>, String> {
    rows.iter()
        .map(|row| {
            if row.logprob_true > 0.0 || !row.logprob_true.is_finite() {
                Err(format!(
                    "row with logprob {:?}: not a true log-probability",
                    row.logprob_true
                ))
            } else {
                Ok(-(row.logprob_true as f64))
            }
        })
        .collect()
}

/// The top-1 rows: 1.0 where the arm's greedy pick matched the true token.
fn top1_rows(rows: &[PositionRow]) -> Vec<f64> {
    rows.iter()
        .map(|row| f64::from(row.top1_correct as u8))
        .collect()
}

/// The probe rows: 1.0 where the probe was all-token exact.
fn probe_rows(rows: &[ProbeRow]) -> Vec<f64> {
    rows.iter().map(|row| f64::from(row.exact as u8)).collect()
}

/// The full verdict of one comparison from both arms' frozen rows. Pure: the
/// device arms produce the rows, this decides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Comparison {
    pub nll_ub: f64,
    pub top1_lb: f64,
    pub probe_lb: f64,
    pub retrieval_lb: f64,
}

impl Comparison {
    /// Grouped args would hide the cand/ref pairing the registration pins;
    /// the seven data args ARE the frozen protocol.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::family::qwen36) fn evaluate(
        cand_pos: &[PositionRow],
        ref_pos: &[PositionRow],
        cand_copy: &[ProbeRow],
        ref_copy: &[ProbeRow],
        cand_retr: &[ProbeRow],
        ref_retr: &[ProbeRow],
        n_clusters: u32,
        draws: usize,
        seed: u64,
    ) -> Self {
        let clusters: Vec<u32> = cand_pos.iter().map(|row| row.cluster).collect();
        // Paired rows: the reference arm must carry the SAME positions in
        // the SAME order; cluster equality per index is the load-bearing
        // check (the bootstrap zips by index and weights by candidate
        // clusters only).
        assert!(
            ref_pos.len() == cand_pos.len()
                && ref_pos
                    .iter()
                    .zip(cand_pos)
                    .all(|(r, c)| r.cluster == c.cluster),
            "reference rows are not aligned with candidate rows"
        );
        for (c_rows, r_rows) in [(cand_copy, ref_copy), (cand_retr, ref_retr)] {
            assert!(
                c_rows.len() == r_rows.len()
                    && c_rows
                        .iter()
                        .zip(r_rows)
                        .all(|(c, r)| c.cluster == r.cluster),
                "reference probe rows are not aligned with candidate rows"
            );
        }
        let nll = bootstrap_bound(
            &nll_rows(cand_pos).expect("the sidecar verified the logprob signs"),
            &nll_rows(ref_pos).expect("the sidecar verified the logprob signs"),
            &clusters,
            n_clusters,
            draws,
            seed,
            true,
        );
        let top1 = bootstrap_bound(
            &top1_rows(cand_pos),
            &top1_rows(ref_pos),
            &clusters,
            n_clusters,
            draws,
            seed,
            false,
        );
        let probe_bound = |cand: &[ProbeRow], refr: &[ProbeRow]| {
            // Resample only the probe-bearing clusters (registration §3):
            // compacted universe, so every draw contains events and the
            // pooled statistic is always defined (no 0/0 draws).
            let mut universe: Vec<u32> = cand.iter().chain(refr).map(|row| row.cluster).collect();
            universe.sort_unstable();
            universe.dedup();
            let c: Vec<u32> = cand
                .iter()
                .map(|row| universe.binary_search(&row.cluster).expect("compact") as u32)
                .collect();
            bootstrap_bound(
                &probe_rows(cand),
                &probe_rows(refr),
                &c,
                universe.len() as u32,
                draws,
                seed,
                false,
            )
        };
        Comparison {
            nll_ub: nll,
            top1_lb: top1,
            probe_lb: probe_bound(cand_copy, ref_copy),
            retrieval_lb: probe_bound(cand_retr, ref_retr),
        }
    }

    /// All four gates must hold (registration §3); conjunctions need no alpha
    /// correction, and no gate is dropped after results.
    pub(in crate::family::qwen36) fn passes(&self, margins: &Margins) -> bool {
        self.nll_ub <= margins.nll_ub
            && self.top1_lb >= margins.top1_lb
            && self.probe_lb >= margins.copy_lb
            && self.retrieval_lb >= margins.retrieval_lb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two tiny arms over three clusters; the helper builds rows with real
    /// log-prob semantics (negative logprobs) and per-row correctness flags.
    fn rows(logprobs: &[f64], correct: &[bool], cluster_of: usize) -> Vec<PositionRow> {
        logprobs
            .iter()
            .zip(correct)
            .enumerate()
            .map(|(i, (lp, ok))| PositionRow {
                cluster: ((i / cluster_of) % 3) as u32,
                logprob_true: *lp as f32,
                top1_correct: *ok,
            })
            .collect()
    }

    #[test]
    fn identical_arms_pass_with_zero_bounds() {
        let pos = rows(&[-1.0, -2.0, -1.5, -0.7, -1.2, -1.8], &[true; 6], 2);
        let probes = (0..6)
            .map(|i| ProbeRow {
                cluster: (i % 3) as u32,
                exact: i % 2 == 0,
            })
            .collect::<Vec<_>>();
        let cmp = Comparison::evaluate(
            &pos, &pos, &probes, &probes, &probes, &probes, 3, 200, 20261031,
        );
        assert!(cmp.passes(&Margins::frozen()), "{cmp:?}");
        assert!(cmp.nll_ub.abs() < 1e-9 && cmp.top1_lb.abs() < 1e-9);
    }

    /// Direction cases with real semantics (the sign lesson): a candidate
    /// uniformly worse by a clear margin FAILS, and uniformly better PASSES;
    /// each failure direction is proven on its own.
    #[test]
    fn direction_cases_hold_on_both_sides() {
        // Baseline: 12 positions, 4 per cluster, logprobs near -1.
        let base_lp = [-1.0; 12];
        let base_ok = [true; 12];
        let reference = rows(&base_lp, &base_ok, 4);
        let clusters = 3;
        let probes_ref = vec![
            ProbeRow {
                cluster: 0,
                exact: true
            };
            12
        ];

        // Worse by 0.05 nats everywhere: the NLL bound alone must fail.
        let worse_lp = [-1.05; 12];
        let worse = rows(&worse_lp, &base_ok, 4);
        let nll_bad = Comparison::evaluate(
            &worse,
            &reference,
            &probes_ref,
            &probes_ref,
            &probes_ref,
            &probes_ref,
            clusters,
            500,
            20261031,
        );
        assert!(nll_bad.nll_ub > 0.02, "{nll_bad:?}");
        assert!(!nll_bad.passes(&Margins::frozen()), "{nll_bad:?}");

        // Better by 0.01 nats everywhere: NLL passes with room.
        let better_lp = [-0.99; 12];
        let better = rows(&better_lp, &base_ok, 4);
        let nll_good = Comparison::evaluate(
            &better,
            &reference,
            &probes_ref,
            &probes_ref,
            &probes_ref,
            &probes_ref,
            clusters,
            500,
            20261031,
        );
        assert!(nll_good.nll_ub < 0.02, "{nll_good:?}");
        assert!(nll_good.passes(&Margins::frozen()), "{nll_good:?}");

        // Top-1: two of twelve wrong (16.7 pp drop) must fail the -0.5 floor
        // while the NLL rows stay identical.
        let mut flip_ok = base_ok;
        flip_ok[2] = false;
        flip_ok[8] = false;
        let flipped = rows(&base_lp, &flip_ok, 4);
        let top1_bad = Comparison::evaluate(
            &flipped,
            &reference,
            &probes_ref,
            &probes_ref,
            &probes_ref,
            &probes_ref,
            clusters,
            500,
            20261031,
        );
        assert!(!top1_bad.passes(&Margins::frozen()), "{top1_bad:?}");

        // Probes: 3 of 12 copy probes lost (25 pp) must fail the -1.0 floor.
        let mut lost = probes_ref.clone();
        lost[1].exact = false;
        lost[5].exact = false;
        lost[9].exact = false;
        let probe_bad = Comparison::evaluate(
            &reference,
            &reference,
            &lost,
            &probes_ref,
            &probes_ref,
            &probes_ref,
            clusters,
            500,
            20261031,
        );
        assert!(!probe_bad.passes(&Margins::frozen()), "{probe_bad:?}");
    }

    #[test]
    fn positive_logprob_row_is_refused() {
        let bad = rows(&[-1.0, 0.4], &[true, true], 1);
        assert!(nll_rows(&bad).is_err());
    }

    /// The registered resampling stream: xorshift64* from seed 20261031 (|1),
    /// the first outputs and the first rejection-sampled indices are pinned
    /// to values computed by an independent implementation, so a changed
    /// shift, multiplier, seed handling or rejection zone fails here by
    /// design rather than by accident.
    #[test]
    fn rng_stream_is_pinned_to_the_registered_xorshift64star() {
        let mut raw = Rng(20_261_031 | 1);
        let words: Vec<u64> = (0..5).map(|_| raw.next_u64()).collect();
        assert_eq!(
            words,
            [
                0xe45f_0253_810d_29e8,
                0x073c_734b_64e0_a991,
                0x121b_3213_968f_6640,
                0xf1a3_689a_3c58_3388,
                0x4c69_fada_2e97_c313,
            ]
        );
        let mut r1526 = Rng(20_261_031 | 1);
        let idx: Vec<u64> = (0..8).map(|_| r1526.below(1526)).collect();
        assert_eq!(idx, [746, 755, 420, 784, 779, 1272, 220, 636]);
        let mut r40 = Rng(20_261_031 | 1);
        let idx: Vec<u64> = (0..8).map(|_| r40.below(40)).collect();
        assert_eq!(idx, [8, 25, 8, 32, 35, 24, 36, 18]);
    }

    /// 40 clusters of 1-3 rows with dyadic values (k/64), so every sum is
    /// exact and the golden bounds below are bit-for-bit reproducible.
    fn golden_fixture() -> (Vec<f64>, Vec<f64>, Vec<u32>) {
        let (mut cand, mut refr, mut clusters) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..40u32 {
            for j in 0..(1 + i % 3) {
                cand.push(f64::from((7 * i + 3 * j) % 13) / 64.0);
                refr.push(f64::from((5 * i + j) % 11) / 64.0);
                clusters.push(i);
            }
        }
        (cand, refr, clusters)
    }

    /// Golden bounds of the paired clustered bootstrap (500 draws, seed
    /// 20261031) from an independent implementation: upper AND lower side
    /// and the second seed (seed+2) are pinned, so a flipped side, a moved
    /// seed, a changed percentile level (0.90 gives 0.0271.. / 0.00257..),
    /// a changed multiplier or shift each change a hard-coded value. The
    /// point difference 0.01443829... lies strictly between the two bounds.
    #[test]
    fn bootstrap_bounds_match_the_golden_values_on_both_sides() {
        let (a, b, clusters) = golden_fixture();
        assert_eq!(a.len(), 79);
        let bound =
            |seed: u64, upper: bool| bootstrap_bound(&a, &b, &clusters, 40, 500, seed, upper);
        assert_eq!(bound(20_261_031, true), 0.03125);
        assert_eq!(bound(20_261_031, false), -0.0011432926829268292);
        // seed+2 moves BOTH bounds.
        assert_eq!(bound(20_261_033, true), 0.031433823529411764);
        assert_eq!(bound(20_261_033, false), -0.003043831168831169);
        // The stream starts from `seed | 1`: an EVEN seed gives the same
        // bounds as the odd seed one above it, so a dropped `| 1` changes
        // these values (an odd seed alone cannot show it).
        assert_eq!(bound(20_261_030, true), 0.03125);
        assert_eq!(bound(20_261_030, false), -0.0011432926829268292);
        assert_eq!(bound(20_261_032, true), 0.031433823529411764);
        assert_eq!(bound(20_261_032, false), -0.003043831168831169);
        // Side semantics: the lower bound is below the point difference and
        // the upper bound above it, never swapped.
        let point = 0.0144382911392405;
        assert!(bound(20_261_031, false) < point && point < bound(20_261_031, true));
    }

    /// The bootstrap is deterministic from the seed: the same seed twice is
    /// bit-identical; the golden test above pins the values themselves.
    #[test]
    fn bootstrap_is_deterministic_from_the_seed() {
        let (a, b, clusters) = golden_fixture();
        for upper in [true, false] {
            let first = bootstrap_bound(&a, &b, &clusters, 40, 500, 20_261_031, upper);
            let second = bootstrap_bound(&a, &b, &clusters, 40, 500, 20_261_031, upper);
            assert_eq!(first.to_bits(), second.to_bits());
        }
    }

    /// The four registered margins, exactly, by the names in the code.
    #[test]
    fn frozen_margins_are_exactly_the_registered_values() {
        assert_eq!(
            Margins::frozen(),
            Margins {
                nll_ub: 0.02,
                top1_lb: -0.005,
                copy_lb: -0.01,
                retrieval_lb: -0.02,
            }
        );
    }

    fn comparison(nll_ub: f64, top1_lb: f64, probe_lb: f64, retrieval_lb: f64) -> Comparison {
        Comparison {
            nll_ub,
            top1_lb,
            probe_lb,
            retrieval_lb,
        }
    }

    /// `passes` gate by gate: all four holding passes (also exactly at the
    /// margins), and each single failing bound alone makes it false. A
    /// `passes` that dropped one gate (for example the retrieval floor,
    /// which decides the registered P1 verdict) fails one of these cases.
    #[test]
    fn passes_requires_each_of_the_four_gates() {
        let m = Margins::frozen();
        assert!(comparison(0.0, 0.0, 0.0, 0.0).passes(&m));
        // Exactly at every margin still passes (<= for the ceiling, >= for floors).
        assert!(comparison(0.02, -0.005, -0.01, -0.02).passes(&m));
        // Only the NLL ceiling fails.
        assert!(!comparison(0.0201, 0.0, 0.0, 0.0).passes(&m));
        // Only the top-1 floor fails (just past -0.005).
        assert!(!comparison(0.0, -0.0051, 0.0, 0.0).passes(&m));
        // Only the copy floor fails (just past -0.01).
        assert!(!comparison(0.0, 0.0, -0.0101, 0.0).passes(&m));
        // Only the retrieval floor fails (just past -0.02; the registered
        // P1 retrieval bound was -0.034091).
        assert!(!comparison(0.0, 0.0, 0.0, -0.0201).passes(&m));
        assert!(!comparison(0.000271, -0.000272, -0.005051, -0.034091).passes(&m));
    }

    fn probes_in(clusters: &[u32]) -> Vec<ProbeRow> {
        clusters
            .iter()
            .map(|&cluster| ProbeRow {
                cluster,
                exact: true,
            })
            .collect()
    }

    fn evaluate_rows(
        cand_pos: &[PositionRow],
        ref_pos: &[PositionRow],
        cand_copy: &[ProbeRow],
        ref_copy: &[ProbeRow],
        cand_retr: &[ProbeRow],
        ref_retr: &[ProbeRow],
    ) -> Comparison {
        Comparison::evaluate(
            cand_pos, ref_pos, cand_copy, ref_copy, cand_retr, ref_retr, 3, 50, 20261031,
        )
    }

    /// Candidate and reference rows must carry the same clusters in the same
    /// order. Same length with different clusters, and a shorter reference,
    /// are each refused for positions.
    #[test]
    #[should_panic(expected = "reference rows are not aligned with candidate rows")]
    fn misaligned_position_clusters_are_refused() {
        let cand = rows(&[-1.0; 6], &[true; 6], 2); // clusters 0,0,1,1,2,2
        let refr = rows(&[-1.0; 6], &[true; 6], 3); // clusters 0,0,0,1,1,1
        let p = probes_in(&[0, 1, 2]);
        evaluate_rows(&cand, &refr, &p, &p, &p, &p);
    }

    #[test]
    #[should_panic(expected = "reference rows are not aligned with candidate rows")]
    fn short_reference_positions_are_refused() {
        let cand = rows(&[-1.0; 6], &[true; 6], 2);
        let refr = rows(&[-1.0; 5], &[true; 5], 2); // same clusters, one row short
        let p = probes_in(&[0, 1, 2]);
        evaluate_rows(&cand, &refr, &p, &p, &p, &p);
    }

    #[test]
    #[should_panic(expected = "reference probe rows are not aligned with candidate rows")]
    fn misaligned_copy_probe_clusters_are_refused() {
        let pos = rows(&[-1.0; 6], &[true; 6], 2);
        let ok = probes_in(&[0, 1, 2]);
        evaluate_rows(
            &pos,
            &pos,
            &probes_in(&[0, 1, 2]),
            &probes_in(&[0, 2, 1]),
            &ok,
            &ok,
        );
    }

    #[test]
    #[should_panic(expected = "reference probe rows are not aligned with candidate rows")]
    fn short_retrieval_probe_reference_is_refused() {
        let pos = rows(&[-1.0; 6], &[true; 6], 2);
        let ok = probes_in(&[0, 1, 2]);
        evaluate_rows(
            &pos,
            &pos,
            &ok,
            &ok,
            &probes_in(&[0, 1, 2]),
            &probes_in(&[0, 1]),
        );
    }

    #[test]
    #[should_panic(expected = "reference probe rows are not aligned with candidate rows")]
    fn misaligned_retrieval_probe_clusters_are_refused() {
        let pos = rows(&[-1.0; 6], &[true; 6], 2);
        let ok = probes_in(&[0, 1, 2]);
        evaluate_rows(
            &pos,
            &pos,
            &ok,
            &ok,
            &probes_in(&[0, 1, 2]),
            &probes_in(&[2, 1, 0]),
        );
    }

    /// Probes occupying only a SPARSE subset of the position clusters: the
    /// probe universe must be the probe-bearing clusters (v4: 169/169 of
    /// 1,526), so no draw can be event-free. 2 of 5 clusters here; the
    /// bounds must be finite, and losing 3 of 6 events must fail the floor.
    #[test]
    fn sparse_probe_universe_stays_finite_and_directional() {
        // Positions over 5 clusters (12 rows, cluster = i/5 % 5).
        let pos = rows(&[-1.0; 12], &[true; 12], 5);
        // 6 probe rows living only in clusters 1 and 3.
        let probes = (0..6)
            .map(|i| ProbeRow {
                cluster: if i < 3 { 1 } else { 3 },
                exact: true,
            })
            .collect::<Vec<_>>();
        let cmp = Comparison::evaluate(
            &pos, &pos, &probes, &probes, &probes, &probes, 5, 500, 20261031,
        );
        assert!(
            cmp.probe_lb.is_finite() && cmp.probe_lb.abs() < 1e-9,
            "{cmp:?}"
        );
        assert!(cmp.passes(&Margins::frozen()), "{cmp:?}");
        let mut lost = probes.clone();
        lost[1].exact = false;
        lost[4].exact = false;
        lost[5].exact = false;
        let bad = Comparison::evaluate(
            &pos, &pos, &lost, &probes, &probes, &probes, 5, 500, 20261031,
        );
        assert!(
            bad.probe_lb < -0.01 && !bad.passes(&Margins::frozen()),
            "{bad:?}"
        );
    }
}
