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
    let mut tiers = vec![Exl3Isa::Scalar];
    if Exl3Isa::Avx512.is_supported() {
        tiers.push(Exl3Isa::Avx512);
    }
    tiers
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

#[test]
fn every_tier_agrees_with_the_scalar_column_sums() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let (k, n) = (256, 384);
    let table: Vec<AtomicU32> = (0..2 * k).map(|_| AtomicU32::new(0)).collect();
    for words in ALL_WORDS {
        let (trellis, suh, svh) = random_projection(&mut rng, k, n, words);
        let proj = Exl3Projection::new([k, n], words, &trellis, &suh, &svh).unwrap();
        let tiers: Vec<_> = pool_isa()
            .into_iter()
            .filter(|t| *t != Exl3Isa::Scalar)
            .collect();
        // One-hot activations make every sum a single exact integer product. The first, second
        // and last k tiles cover all 32 lane vectors and the last tile of the buffer.
        let hot = (0..64).chain(2 * k - 32..2 * k);
        for tier in &tiers {
            for hot in hot.clone() {
                table.iter().for_each(|a| a.store(0, Ordering::Relaxed));
                store(&table, hot, 1.0);
                for nt in 0..n / TILE {
                    assert_eq!(
                        column_sums(*tier, &proj, &table, nt),
                        column_sums(Exl3Isa::Scalar, &proj, &table, nt),
                        "{tier:?} words={words} hot={hot} nt={nt}"
                    );
                }
            }
            for i in 0..table.len() {
                store(&table, i, rng.unit() * 4.0);
            }
            for nt in 0..n / TILE {
                let got = column_sums(*tier, &proj, &table, nt);
                let want = column_sums(Exl3Isa::Scalar, &proj, &table, nt);
                let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                for (g, w) in got.iter().zip(want) {
                    assert!(
                        (g - w).abs() <= 1e-5 * scale,
                        "{tier:?} words={words} nt={nt}: {g} vs {w}"
                    );
                }
            }
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
