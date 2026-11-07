use std::time::{Duration, Instant};

use super::*;
use crate::weights::exl3::Codebook;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| (self.next() >> 24) as u8).collect()
    }
}

/// Every stored words-per-tile value: integer bit rates 1 to 8 and the half-bit rates.
const ALL_WORDS: [usize; 11] = [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128];

fn pool_isa() -> Vec<Exl3Isa> {
    [Exl3Isa::Scalar, Exl3Isa::Avx2, Exl3Isa::Avx512]
        .into_iter()
        .filter(|tier| tier.is_supported())
        .collect()
}

#[test]
fn scalar_tile_sums_reproduce_the_reference_codebook_weights() {
    let (alpha, beta) = (
        f16::from_bits(0x1eee).to_f64(),
        f16::from_bits(0xc931).to_f64(),
    );
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for words in ALL_WORDS {
        let tile = rng.bytes(words * 2);
        let units: Vec<u16> = tile
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect();
        let mut reference = [0u16; 256];
        Codebook::Mul1.decode_tile(&units, &mut reference).unwrap();
        for b in 0..2 {
            for q in 0..16 {
                let mut a = [0.0f32; 32];
                a[b * 16 + q] = 1.0;
                let mut acc = [0.0f32; TILE];
                tile_sums_scalar(&tile, &a, &mut acc);
                for c in 0..8 {
                    let col = c + 8 * ((q >> 2) & 1);
                    let sum = f64::from(acc[col]);
                    let expected = f16::from_f64(alpha * (1024.0 + sum) + beta).to_bits();
                    let row = lane_row(q, b);
                    assert_eq!(
                        reference[row * 16 + col],
                        expected,
                        "words={words} b={b} q={q} c={c}"
                    );
                }
            }
        }
    }
}

fn random_projection(
    rng: &mut Rng,
    k: usize,
    n: usize,
    words: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let trellis = rng.bytes((k / 16) * (n / 16) * words * 2);
    let scale = |rng: &mut Rng, len: usize| -> Vec<u8> {
        (0..len)
            .flat_map(|_| {
                f16::from_f32(rng.unit() * 0.75 + 1.0)
                    .to_f32()
                    .to_le_bytes()
            })
            .collect()
    };
    let (suh, svh) = (scale(rng, k), scale(rng, n));
    (trellis, suh, svh)
}

/// Every SIMD tier against the scalar column sums for one shape and stored rate: one-hot
/// activations (each sum is a single exact integer product, so the tiers must agree bit for bit)
/// at the first, second and last k tiles, then random activations under the rounding tolerance.
fn assert_tiers_match_scalar(rng: &mut Rng, (k, n): (usize, usize), words: usize) {
    let table: Vec<AtomicU32> = (0..2 * k).map(|_| AtomicU32::new(0)).collect();
    let (trellis, suh, svh) = random_projection(rng, k, n, words);
    let proj = Exl3Projection::new([k, n], words, &trellis, &suh, &svh).unwrap();
    let hot = (0..64).chain(2 * k - 32..2 * k);
    // The first, a middle and the last tile column; the random activations below cover all.
    let sampled = [0, n / TILE / 2, n / TILE - 1];
    for tier in pool_isa().into_iter().filter(|t| *t != Exl3Isa::Scalar) {
        for hot in hot.clone() {
            table.iter().for_each(|a| a.store(0, Ordering::Relaxed));
            store(&table, hot, 1.0);
            for nt in sampled {
                assert_eq!(
                    column_sums(tier, &proj, &table, nt),
                    column_sums(Exl3Isa::Scalar, &proj, &table, nt),
                    "{tier:?} [{k},{n}] words={words} hot={hot} nt={nt}"
                );
            }
        }
        for i in 0..table.len() {
            store(&table, i, rng.unit() * 4.0);
        }
        for nt in 0..n / TILE {
            let got = column_sums(tier, &proj, &table, nt);
            let want = column_sums(Exl3Isa::Scalar, &proj, &table, nt);
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (g, w) in got.iter().zip(want) {
                assert!(
                    (g - w).abs() <= 1e-5 * scale,
                    "{tier:?} [{k},{n}] words={words} nt={nt}: {g} vs {w}"
                );
            }
        }
    }
}

#[test]
fn every_tier_agrees_with_the_scalar_column_sums() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for words in ALL_WORDS {
        assert_tiers_match_scalar(&mut rng, (256, 384), words);
    }
}

/// Shapes with one k tile row of eight, odd counts of 128-wide blocks on either side (3 and 5
/// blocks leave a single tile column pair for the last pool chunk), and the tall down-projection
/// shape, for every stored rate.
#[test]
fn every_tier_agrees_with_the_scalar_on_odd_and_tall_shapes() {
    let mut rng = Rng(0x0f0f_1e1e_2d2d_3c3c);
    for shape in [(128, 128), (384, 128), (128, 640), (640, 256), (768, 384)] {
        for words in ALL_WORDS {
            assert_tiers_match_scalar(&mut rng, shape, words);
        }
    }
}

#[test]
fn results_do_not_depend_on_the_thread_count_or_tier_beyond_rounding() {
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    let (h, f) = (256, 128);
    let parts: Vec<_> = (0..3)
        .map(|_| {
            [
                random_projection(&mut rng, h, f, 64),
                random_projection(&mut rng, h, f, 64),
                random_projection(&mut rng, f, h, 64),
            ]
        })
        .collect();
    let experts: Vec<ExpertView<'_>> = parts
        .iter()
        .map(|[g, u, d]| ExpertView {
            gate: Exl3Projection::new([h, f], 64, &g.0, &g.1, &g.2).unwrap(),
            up: Exl3Projection::new([h, f], 64, &u.0, &u.1, &u.2).unwrap(),
            down: Exl3Projection::new([f, h], 64, &d.0, &d.1, &d.2).unwrap(),
        })
        .collect();
    let picks: Vec<_> = experts
        .iter()
        .zip([0.5f32, 0.25, 1.5])
        .map(|(&expert, weight)| ExpertPick { expert, weight })
        .collect();
    let x: Vec<f32> = (0..h).map(|_| rng.unit()).collect();
    let run_with = |threads: usize, isa: Exl3Isa, wait: WaitPolicy| {
        let cpu = ExpertCpu::new(threads, wait, Some(isa)).unwrap();
        let mut out = vec![0.0f32; h];
        cpu.forward(&x, &picks, &mut out).unwrap();
        out
    };
    let run = |threads: usize, isa: Exl3Isa| run_with(threads, isa, WaitPolicy::Park);
    let base = run(1, Exl3Isa::Scalar);
    assert!(base.iter().all(|v| v.is_finite()) && base.iter().any(|v| *v != 0.0));
    for tier in pool_isa() {
        let single = run(1, tier);
        for threads in [2, 5, 12] {
            assert_eq!(run(threads, tier), single, "{tier:?} threads={threads}");
        }
        for wait in [
            WaitPolicy::Spin,
            WaitPolicy::SpinThenPark(std::time::Duration::from_micros(50)),
        ] {
            assert_eq!(run_with(3, tier, wait), single, "{tier:?} {wait:?}");
        }
        let scale = base.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (a, b) in single.iter().zip(&base) {
            assert!((a - b).abs() <= 1e-4 * scale, "{tier:?}: {a} vs {b}");
        }
    }
}

#[test]
fn expert_pairs_serve_each_row_like_a_single_row_forward() {
    let mut rng = Rng(0x5555_aaaa_1234_9876);
    let (h, f) = (256, 128);
    let parts: Vec<_> = (0..3)
        .map(|_| {
            [
                random_projection(&mut rng, h, f, 64),
                random_projection(&mut rng, h, f, 64),
                random_projection(&mut rng, f, h, 64),
            ]
        })
        .collect();
    let experts: Vec<ExpertView<'_>> = parts
        .iter()
        .map(|[g, u, d]| ExpertView {
            gate: Exl3Projection::new([h, f], 64, &g.0, &g.1, &g.2).unwrap(),
            up: Exl3Projection::new([h, f], 64, &u.0, &u.1, &u.2).unwrap(),
            down: Exl3Projection::new([f, h], 64, &d.0, &d.1, &d.2).unwrap(),
        })
        .collect();
    let xs: Vec<f32> = (0..2 * h).map(|_| rng.unit()).collect();
    // (expert, row): an expert shared by both rows, and rows in a different order than experts.
    let pairs = [(2usize, 1usize), (0, 0), (2, 0), (1, 1)];
    let jobs: Vec<_> = pairs
        .iter()
        .map(|&(e, row)| PairJob {
            expert: experts[e],
            row,
        })
        .collect();
    for tier in pool_isa() {
        let cpu = ExpertCpu::new(3, WaitPolicy::Park, Some(tier)).unwrap();
        let mut got = vec![0.0f32; jobs.len() * h];
        cpu.expert_pairs(&xs, &jobs, &mut got).unwrap();
        for (i, &(e, row)) in pairs.iter().enumerate() {
            let mut want = vec![0.0f32; h];
            let pick = ExpertPick {
                expert: experts[e],
                weight: 1.0,
            };
            cpu.forward(&xs[row * h..(row + 1) * h], &[pick], &mut want)
                .unwrap();
            assert_eq!(&got[i * h..(i + 1) * h], &want[..], "{tier:?} pair {i}");
        }
        let bad_row = [PairJob {
            expert: experts[0],
            row: 2,
        }];
        assert!(cpu.expert_pairs(&xs, &bad_row, &mut vec![0.0; h]).is_err());
    }
}

#[test]
fn published_rounding_follows_the_device_pipeline_without_drifting() {
    let mut rng = Rng(0x7777_1111_2222_3333);
    let (h, f) = (256, 128);
    let parts: Vec<_> = (0..2)
        .map(|_| {
            [
                random_projection(&mut rng, h, f, 64),
                random_projection(&mut rng, h, f, 64),
                random_projection(&mut rng, f, h, 64),
            ]
        })
        .collect();
    let jobs: Vec<_> = parts
        .iter()
        .map(|[g, u, d]| PairJob {
            expert: ExpertView {
                gate: Exl3Projection::new([h, f], 64, &g.0, &g.1, &g.2).unwrap(),
                up: Exl3Projection::new([h, f], 64, &u.0, &u.1, &u.2).unwrap(),
                down: Exl3Projection::new([f, h], 64, &d.0, &d.1, &d.2).unwrap(),
            },
            row: 0,
        })
        .collect();
    let x: Vec<f32> = (0..h).map(|_| rng.unit()).collect();
    let run = |threads: usize, rounded: bool| {
        let cpu = ExpertCpu::new(threads, WaitPolicy::Park, None)
            .unwrap()
            .with_published_rounding(rounded);
        let mut out = vec![0.0f32; jobs.len() * h];
        cpu.expert_pairs(&x, &jobs, &mut out).unwrap();
        out
    };
    let exact = run(1, false);
    let rounded = run(1, true);
    assert_eq!(rounded, run(4, true), "rounding must not depend on threads");
    assert!(rounded.iter().all(|v| bf16::from_f32(*v).to_f32() == *v));
    assert_ne!(rounded, exact);
    let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    let diff: Vec<f32> = rounded.iter().zip(&exact).map(|(a, b)| a - b).collect();
    // A few bf16 roundings in a row move the result by a few 1e-3 of its size.
    assert!(
        rms(&diff) < 2e-2 * rms(&exact),
        "{}",
        rms(&diff) / rms(&exact)
    );
}

#[test]
fn malformed_projections_and_spans_are_refused() {
    let mut rng = Rng(7);
    let (trellis, suh, svh) = random_projection(&mut rng, 128, 128, 64);
    assert!(Exl3Projection::new([128, 128], 64, &trellis, &suh, &svh).is_ok());
    assert!(Exl3Projection::new([128, 128], 60, &trellis, &suh, &svh).is_err());
    assert!(Exl3Projection::new([128, 128], 64, &trellis[1..], &suh, &svh).is_err());
    assert!(Exl3Projection::new([128, 128], 64, &trellis, &suh[1..], &svh).is_err());
    assert!(Exl3Projection::new([64, 128], 64, &trellis, &suh, &svh).is_err());
    let proj = Exl3Projection::new([128, 128], 64, &trellis, &suh, &svh).unwrap();
    let cpu = ExpertCpu::new(2, WaitPolicy::Park, None).unwrap();
    assert!(cpu.project(&proj, &[0.0; 127], &mut [0.0; 128]).is_err());
    assert!(cpu.project(&proj, &[0.0; 128], &mut [0.0; 127]).is_err());
}

/// Three experts of `words` words per tile and the `h x f` shape, owned.
fn random_experts(rng: &mut Rng, (h, f): (usize, usize), words: usize) -> Vec<[Parts; 3]> {
    (0..3)
        .map(|_| {
            [
                random_projection(rng, h, f, words),
                random_projection(rng, h, f, words),
                random_projection(rng, f, h, words),
            ]
        })
        .collect()
}

type Parts = (Vec<u8>, Vec<u8>, Vec<u8>);

fn expert_views(parts: &[[Parts; 3]], (h, f): (usize, usize), words: usize) -> Vec<ExpertView<'_>> {
    parts
        .iter()
        .map(|[g, u, d]| ExpertView {
            gate: Exl3Projection::new([h, f], words, &g.0, &g.1, &g.2).unwrap(),
            up: Exl3Projection::new([h, f], words, &u.0, &u.1, &u.2).unwrap(),
            down: Exl3Projection::new([f, h], words, &d.0, &d.1, &d.2).unwrap(),
        })
        .collect()
}

/// Whole experts for every stored rate: one row on 1, 2, 5 and 12 threads gives the same bits
/// within a tier, the tiers agree with the scalar reference under the rounding tolerance, and
/// `expert_pairs` over several rows and a shared expert equals the single-row forward bit for bit.
#[test]
fn every_rate_is_thread_count_independent_and_agrees_across_tiers() {
    let mut rng = Rng(0xa5a5_5a5a_c3c3_3c3c);
    let shape = (384, 256);
    let h = shape.0;
    for words in ALL_WORDS {
        let parts = random_experts(&mut rng, shape, words);
        let experts = expert_views(&parts, shape, words);
        let picks: Vec<_> = experts
            .iter()
            .zip([0.75f32, 1.25, 0.5])
            .map(|(&expert, weight)| ExpertPick { expert, weight })
            .collect();
        let xs: Vec<f32> = (0..2 * h).map(|_| rng.unit()).collect();
        let forward = |threads: usize, tier: Exl3Isa| {
            let cpu = ExpertCpu::new(threads, WaitPolicy::Park, Some(tier)).unwrap();
            let mut out = vec![0.0f32; h];
            cpu.forward(&xs[..h], &picks, &mut out).unwrap();
            out
        };
        let base = forward(1, Exl3Isa::Scalar);
        assert!(base.iter().all(|v| v.is_finite()) && base.iter().any(|v| *v != 0.0));
        let scale = base.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for tier in pool_isa() {
            let single = forward(1, tier);
            for threads in [2, 5, 12] {
                assert_eq!(
                    forward(threads, tier),
                    single,
                    "{tier:?} words={words} threads={threads}"
                );
            }
            for (a, b) in single.iter().zip(&base) {
                assert!(
                    (a - b).abs() <= 1e-4 * scale,
                    "{tier:?} words={words}: {a} vs {b}"
                );
            }
            let pairs = [(2usize, 1usize), (0, 0), (2, 0), (1, 1)];
            let jobs: Vec<_> = pairs
                .iter()
                .map(|&(e, row)| PairJob {
                    expert: experts[e],
                    row,
                })
                .collect();
            let cpu = ExpertCpu::new(3, WaitPolicy::Park, Some(tier)).unwrap();
            let mut got = vec![0.0f32; jobs.len() * h];
            cpu.expert_pairs(&xs, &jobs, &mut got).unwrap();
            for (i, &(e, row)) in pairs.iter().enumerate() {
                let pick = ExpertPick {
                    expert: experts[e],
                    weight: 1.0,
                };
                let mut want = vec![0.0f32; h];
                cpu.forward(&xs[row * h..(row + 1) * h], &[pick], &mut want)
                    .unwrap();
                assert_eq!(
                    &got[i * h..(i + 1) * h],
                    &want[..],
                    "{tier:?} words={words} pair {i}"
                );
            }
        }
    }
}

/// The experts of a layer are combined in pick order: `forward` equals the sequential f32
/// accumulation of `weight * output` over the picks, bit for bit, on every tier.
#[test]
fn picked_experts_are_combined_in_pick_order() {
    let mut rng = Rng(0x2468_ace0_1357_9bdf);
    let shape = (256, 128);
    let h = shape.0;
    let parts = random_experts(&mut rng, shape, 64);
    let experts = expert_views(&parts, shape, 64);
    let weights = [0.3f32, 1.7, 0.45];
    let x: Vec<f32> = (0..h).map(|_| rng.unit()).collect();
    for tier in pool_isa() {
        let cpu = ExpertCpu::new(4, WaitPolicy::Park, Some(tier)).unwrap();
        let jobs: Vec<_> = experts
            .iter()
            .map(|&expert| PairJob { expert, row: 0 })
            .collect();
        let mut each = vec![0.0f32; jobs.len() * h];
        cpu.expert_pairs(&x, &jobs, &mut each).unwrap();
        let mut want = vec![0.0f32; h];
        for (y, weight) in each.chunks_exact(h).zip(weights) {
            for (acc, value) in want.iter_mut().zip(y) {
                *acc += weight * value;
            }
        }
        let picks: Vec<_> = experts
            .iter()
            .zip(weights)
            .map(|(&expert, weight)| ExpertPick { expert, weight })
            .collect();
        let mut got = vec![0.0f32; h];
        cpu.forward(&x, &picks, &mut got).unwrap();
        assert_eq!(got, want, "{tier:?}");
    }
}

/// Names the tiers the other tests loop over, so a run on a CPU without a tier shows it in the
/// log (`--nocapture`) instead of passing those loops without a SIMD tier.
#[test]
fn report_the_tiers_under_test() {
    let tiers = pool_isa();
    println!("exl3 cpu tiers under test: {tiers:?}");
    assert!(tiers.contains(&Exl3Isa::Scalar));
    assert_eq!(tiers.contains(&Exl3Isa::Avx2), IsaSupport::host().avx2);
    assert_eq!(tiers.contains(&Exl3Isa::Avx512), IsaSupport::host().avx512);
}

fn os(text: &str) -> std::ffi::OsString {
    text.into()
}

#[test]
fn the_automatic_order_is_avx512_then_avx2_then_scalar() {
    for (avx512, avx2, want) in [
        (true, true, Exl3Isa::Avx512),
        (true, false, Exl3Isa::Avx512),
        (false, true, Exl3Isa::Avx2),
        (false, false, Exl3Isa::Scalar),
    ] {
        let support = IsaSupport { avx512, avx2 };
        assert_eq!(support.best(), want, "avx512={avx512} avx2={avx2}");
        assert_eq!(resolve_isa(None, support), Ok(want));
        assert_eq!(resolve_isa(Some(&os("auto")), support), Ok(want));
    }
}

/// On a host with AVX-512 the default is the AVX-512 tier, as before the AVX2 tier existed, and
/// it produces the bits of an explicitly requested engine of that tier. An exported
/// `FIFI_EXL3_CPU_ISA` moves the expected tier to what the setting names, so it cannot make this
/// test pass without checking; the child-process test below pins the unset case in any
/// environment.
#[test]
fn an_avx512_host_keeps_the_avx512_default() {
    if !IsaSupport::host().avx512 {
        eprintln!("skipped: the host has no AVX-512");
        return;
    }
    let want = Exl3Isa::from_env().expect("a valid setting");
    if std::env::var_os(ISA_ENV).is_none() {
        assert_eq!(want, Exl3Isa::Avx512);
        assert_eq!(Exl3Isa::detect(), Exl3Isa::Avx512);
    }
    let mut rng = Rng(0x3141_5926_5358_9793);
    let shape = (256, 128);
    let parts = random_experts(&mut rng, shape, 64);
    let experts = expert_views(&parts, shape, 64);
    let picks: Vec<_> = experts
        .iter()
        .map(|&expert| ExpertPick {
            expert,
            weight: 1.0,
        })
        .collect();
    let x: Vec<f32> = (0..shape.0).map(|_| rng.unit()).collect();
    let run = |isa: Option<Exl3Isa>| {
        let cpu = ExpertCpu::new(2, WaitPolicy::Park, isa).unwrap();
        let mut out = vec![0.0f32; shape.0];
        cpu.forward(&x, &picks, &mut out).unwrap();
        (cpu.isa(), out)
    };
    let (default_isa, default_out) = run(None);
    let (_, forced_out) = run(Some(want));
    assert_eq!(default_isa, want);
    assert_eq!(default_out, forced_out);
}

/// What the child below prints before its result.
const CHILD_MARK: &str = "exl3-cpu-isa-child: ";

/// Entry point for `the_environment_setting_reaches_the_engine`: reports what
/// `ExpertCpu::new(.., None)` makes of the environment this process was started with. It asserts
/// nothing and is ignored, so only that test runs it.
#[test]
#[ignore = "run by a parent test in a child process with FIFI_EXL3_CPU_ISA set"]
fn child_reports_the_engine_tier_for_its_environment() {
    let line = match ExpertCpu::new(1, WaitPolicy::Park, None) {
        Ok(cpu) => format!("isa {:?}", cpu.isa()),
        Err(EngError::BadRequest(message)) => format!("err {message}"),
        Err(other) => format!("other {other:?}"),
    };
    println!("{CHILD_MARK}{line}");
}

/// Runs this test binary again, with `FIFI_EXL3_CPU_ISA` set to `setting` (removed for `None`) in
/// the child's environment only, and returns the child's report.
fn child_result(setting: Option<&str>) -> String {
    let crate_relative = module_path!().split_once("::").expect("a crate path").1;
    let name = format!("{crate_relative}::child_reports_the_engine_tier_for_its_environment");
    let mut command = std::process::Command::new(std::env::current_exe().expect("this binary"));
    command.args([
        "--ignored",
        "--exact",
        &name,
        "--nocapture",
        "--test-threads=1",
    ]);
    match setting {
        Some(value) => command.env(ISA_ENV, value),
        None => command.env_remove(ISA_ENV),
    };
    let output = command.output().expect("run the test binary");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "child failed: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
        .lines()
        .find_map(|line| line.split_once(CHILD_MARK).map(|(_, report)| report))
        .unwrap_or_else(|| panic!("the child reported nothing: {stdout}"))
        .to_owned()
}

/// `ExpertCpu::new(.., None)` follows `FIFI_EXL3_CPU_ISA` end to end: a forced tier is used, an
/// unsupported or unknown one fails with an error that names the variable, and `auto` or an unset
/// variable gives the best tier. The environment is set in child processes only.
#[test]
fn the_environment_setting_reaches_the_engine() {
    let host = IsaSupport::host();
    let best = format!("isa {:?}", host.best());
    assert_eq!(child_result(None), best);
    assert_eq!(child_result(Some("auto")), best);
    assert_eq!(child_result(Some("scalar")), "isa Scalar");
    for (value, tier, supported) in [
        ("avx2", Exl3Isa::Avx2, host.avx2),
        ("avx512", Exl3Isa::Avx512, host.avx512),
    ] {
        let got = child_result(Some(value));
        if supported {
            assert_eq!(got, format!("isa {tier:?}"), "{value}");
        } else {
            assert!(
                got.starts_with("err ") && got.contains(ISA_ENV),
                "{value}: {got}"
            );
        }
    }
    let bogus = child_result(Some("bogus"));
    assert!(
        bogus.starts_with("err ") && bogus.contains(ISA_ENV),
        "{bogus}"
    );
}

#[test]
fn the_setting_names_a_tier_or_is_refused() {
    use std::os::unix::ffi::OsStrExt;
    let all = IsaSupport {
        avx512: true,
        avx2: true,
    };
    for (text, want) in [
        ("avx512", Exl3Isa::Avx512),
        ("avx2", Exl3Isa::Avx2),
        ("scalar", Exl3Isa::Scalar),
        (" AVX2 ", Exl3Isa::Avx2),
        ("Scalar", Exl3Isa::Scalar),
    ] {
        assert_eq!(resolve_isa(Some(&os(text)), all), Ok(want), "{text:?}");
    }
    let bad = [
        os(""),
        os(" "),
        os("avx-512"),
        os("avx"),
        os("sse2"),
        os("1"),
        std::ffi::OsStr::from_bytes(b"avx2\xff").to_owned(),
    ];
    for value in &bad {
        let err = resolve_isa(Some(value), all).unwrap_err();
        assert!(
            matches!(&err, EngError::BadRequest(m) if m.contains(ISA_ENV)),
            "{value:?}: {err:?}"
        );
    }
}

/// A forced tier the CPU lacks is an error that names the setting; scalar is always available.
#[test]
fn a_forced_tier_the_cpu_lacks_is_refused_not_replaced() {
    for support in [
        IsaSupport {
            avx512: false,
            avx2: true,
        },
        IsaSupport {
            avx512: false,
            avx2: false,
        },
    ] {
        let err = resolve_isa(Some(&os("avx512")), support).unwrap_err();
        assert!(
            matches!(&err, EngError::BadRequest(m) if m.contains(ISA_ENV) && m.contains("Avx512")),
            "{err:?}"
        );
        assert_eq!(
            resolve_isa(Some(&os("scalar")), support),
            Ok(Exl3Isa::Scalar)
        );
    }
    let none = IsaSupport {
        avx512: false,
        avx2: false,
    };
    let err = resolve_isa(Some(&os("avx2")), none).unwrap_err();
    assert!(
        matches!(&err, EngError::BadRequest(m) if m.contains("Avx2")),
        "{err:?}"
    );
    assert!(!none.allows(Exl3Isa::Avx2) && !none.allows(Exl3Isa::Avx512));
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

/// Loads the three projections of one real expert of the Flint 4.0 pack as owned buffers.
struct RealExpert {
    parts: [RealProjection; 3],
}

struct RealProjection {
    shape: [usize; 2],
    words: usize,
    trellis: Vec<u8>,
    suh: Vec<u8>,
    svh: Vec<u8>,
}

impl RealExpert {
    fn load(pack: &std::path::Path, layer: usize, expert: usize) -> Self {
        use crate::weights::exl3::{COMPONENT_SUFFIXES, Checkpoint, Matrix};
        let checkpoint = Checkpoint::open(pack).unwrap();
        let widen = |t: &crate::tensor::Tensor| -> Vec<u8> {
            t.host_bytes()
                .unwrap()
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|b| {
                    f16::from_bits(u16::from_le_bytes(*b))
                        .to_f32()
                        .to_le_bytes()
                })
                .collect()
        };
        let parts = ["gate_proj", "up_proj", "down_proj"].map(|role| {
            let prefix = format!("model.layers.{layer}.mlp.experts.{expert}.{role}");
            let names = COMPONENT_SUFFIXES.map(|s| format!("{prefix}.{s}"));
            let keys: Vec<&str> = names.iter().map(String::as_str).collect();
            let selected = checkpoint.source.read_selected(&keys, 1 << 30).unwrap();
            let m = Matrix::load(&selected, &prefix, &checkpoint.config).unwrap();
            RealProjection {
                shape: m.shape(),
                words: m.trellis().shape()[2] as usize,
                trellis: m.trellis().host_bytes().unwrap().to_vec(),
                suh: widen(m.suh()),
                svh: widen(m.svh()),
            }
        });
        Self { parts }
    }

    fn view(&self) -> ExpertView<'_> {
        let p = |i: usize| {
            let r = &self.parts[i];
            Exl3Projection::new(r.shape, r.words, &r.trellis, &r.suh, &r.svh).unwrap()
        };
        ExpertView {
            gate: p(0),
            up: p(1),
            down: p(2),
        }
    }
}

fn real_pack() -> Option<std::path::PathBuf> {
    std::env::var_os("FLINT_EXL3_PACK").map(Into::into)
}

/// Run: `FLINT_EXL3_PACK=<dir> cargo test --release --lib real_flint_expert_latency -- --ignored --nocapture`.
#[test]
#[ignore = "needs the 20 GB Flint EXL3 pack"]
fn real_flint_expert_latency() {
    let pack = real_pack().expect("set FLINT_EXL3_PACK");
    // Six distinct experts, so a layer's worth of weights (about 10 MB) leaves the caches.
    let experts: Vec<RealExpert> = (0..6)
        .map(|e| RealExpert::load(&pack, 20, 40 + e))
        .collect();
    let x: Vec<f32> = {
        let mut rng = Rng(42);
        (0..2048).map(|_| rng.unit()).collect()
    };
    for tier in pool_isa() {
        for threads in [1usize, 6, 12] {
            let cpu = ExpertCpu::new(threads, WaitPolicy::Spin, Some(tier)).unwrap();
            for count in [1usize, 6] {
                let picks: Vec<_> = experts[..count]
                    .iter()
                    .map(|e| ExpertPick {
                        expert: e.view(),
                        weight: 1.0,
                    })
                    .collect();
                let mut out = vec![0.0f32; 2048];
                let mut samples = Vec::new();
                for round in 0..220 {
                    let start = Instant::now();
                    cpu.forward(&x, &picks, &mut out).unwrap();
                    if round >= 20 {
                        samples.push(start.elapsed());
                    }
                }
                let t = median(samples);
                let weights = (count * 1_603_584) as f64;
                println!(
                    "{tier:?} threads={threads:2} experts={count}: median {:8.1} us/layer, {:7.1} us/expert, {:5.1} GB/s",
                    t.as_secs_f64() * 1e6,
                    t.as_secs_f64() * 1e6 / count as f64,
                    weights / t.as_secs_f64() / 1e9
                );
            }
        }
    }
}

/// Random weights in the shape of a Flint 4.0 expert (2048 x 512, three projections) at every
/// stored rate, six experts per layer. The rounds cycle through `FIFI_EXL3_SWEEP_LAYERS` layers
/// (default 12, 72 experts, more than the caches hold); 1 keeps the six experts cache resident as
/// `real_flint_expert_latency` does. Times the SIMD tiers only; the scalar tier needs about 20 ms
/// per expert.
///
/// Run: `cargo test --release --lib flint_shape_rate_latency -- --ignored --nocapture`.
#[test]
#[ignore = "timing run"]
fn flint_shape_rate_latency() {
    let (h, f) = (2048, 512);
    let mut rng = Rng(0x1357_9bdf_2468_ace0);
    let x: Vec<f32> = (0..h).map(|_| rng.unit()).collect();
    let layer_count: usize = match std::env::var_os("FIFI_EXL3_SWEEP_LAYERS") {
        None => 12,
        Some(value) => value
            .to_str()
            .and_then(|text| text.trim().parse().ok())
            .filter(|&count| count > 0)
            .unwrap_or_else(|| {
                panic!("FIFI_EXL3_SWEEP_LAYERS must be a positive whole number, got {value:?}")
            }),
    };
    for words in ALL_WORDS {
        let parts: Vec<_> = (0..layer_count)
            .map(|_| {
                let owned = random_experts(&mut rng, (h, f), words);
                // Six experts per layer: two sets of three.
                let second = random_experts(&mut rng, (h, f), words);
                (owned, second)
            })
            .collect();
        let layers: Vec<Vec<ExpertPick<'_>>> = parts
            .iter()
            .map(|(a, b)| {
                expert_views(a, (h, f), words)
                    .into_iter()
                    .chain(expert_views(b, (h, f), words))
                    .map(|expert| ExpertPick {
                        expert,
                        weight: 1.0,
                    })
                    .collect()
            })
            .collect();
        for tier in pool_isa().into_iter().filter(|t| *t != Exl3Isa::Scalar) {
            for threads in [1usize, 6] {
                let cpu = ExpertCpu::new(threads, WaitPolicy::Spin, Some(tier)).unwrap();
                let mut out = vec![0.0f32; h];
                let mut samples = Vec::new();
                for round in 0..260 {
                    let start = Instant::now();
                    cpu.forward(&x, &layers[round % layers.len()], &mut out)
                        .unwrap();
                    if round >= 20 {
                        samples.push(start.elapsed());
                    }
                }
                let t = median(samples);
                println!(
                    "{tier:?} words={words:3} threads={threads:2}: median {:8.1} us/layer, {:7.1} us/expert",
                    t.as_secs_f64() * 1e6,
                    t.as_secs_f64() * 1e6 / 6.0
                );
            }
        }
    }
}
