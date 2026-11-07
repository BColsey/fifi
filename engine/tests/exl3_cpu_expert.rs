//! The CPU EXL3 expert kernel against the ideal fp64 projection of the stored matrices.
//!
//! Bounds are the kernel-parity contract: relative RMS error below 2.5e-3 and largest error
//! below 4e-3 of the reference RMS. The kernel keeps activations in f32 and does not round the
//! decoded weight to binary16, so the expected error is far below both.

use std::collections::HashMap;

use half::f16;
use inference_engine::{
    kernels::{
        exl3_cpu::{Exl3Isa, Exl3Projection, ExpertCpu, ExpertPick, ExpertView},
        nvfp4_cpu::WaitPolicy,
    },
    tensor::{Dtype, Tensor},
    weights::{
        Tier, WeightSource,
        exl3::{Config, Matrix},
    },
};

struct Components(HashMap<String, Tensor>);

impl WeightSource for Components {
    fn tensor(&self, key: &str) -> Option<Tensor> {
        self.0.get(key).cloned()
    }
    fn resident_bytes(&self) -> u64 {
        self.0.values().map(Tensor::byte_len).sum()
    }
    fn tier(&self) -> Tier {
        Tier::Resident
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0
    }
}

fn tensor(dtype: Dtype, shape: Vec<u32>, bytes: Vec<u8>) -> Tensor {
    Tensor::try_from_host(dtype, shape, bytes).unwrap()
}

/// A random stored matrix: any trellis bit pattern is a valid mul1 matrix.
fn matrix(rng: &mut Rng, k: u32, n: u32, words: u32) -> Matrix {
    let trellis: Vec<u8> = (0..(k / 16) * (n / 16) * words * 2)
        .map(|_| (rng.next() >> 24) as u8)
        .collect();
    let scales = |rng: &mut Rng, len: u32| {
        let bytes = (0..len)
            .flat_map(|_| {
                f16::from_f64(1.0 + 0.75 * rng.unit())
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        tensor(Dtype::F16, vec![len], bytes)
    };
    let source = Components(
        [
            (
                "m.trellis".into(),
                tensor(Dtype::I16, vec![k / 16, n / 16, words], trellis),
            ),
            ("m.suh".into(), scales(rng, k)),
            ("m.svh".into(), scales(rng, n)),
            (
                "m.mul1".into(),
                tensor(Dtype::I32, vec![], 0x83dc_d12du32.to_le_bytes().to_vec()),
            ),
        ]
        .into(),
    );
    let config = Config::from_json(br#"{"quantization_config":{"quant_method":"exl3","version":"1.6.2","codebook":"mul1","bits":4.0,"out_scales":"auto","expert_interm_div":128}}"#).unwrap();
    Matrix::load(&source, "m", &config).unwrap()
}

/// Owned kernel inputs of one stored matrix.
struct Parts {
    shape: [usize; 2],
    words: usize,
    trellis: Vec<u8>,
    suh: Vec<u8>,
    svh: Vec<u8>,
}

impl Parts {
    fn of(m: &Matrix) -> Self {
        let widen = |t: &Tensor| -> Vec<u8> {
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
        Self {
            shape: m.shape(),
            words: m.trellis().shape()[2] as usize,
            trellis: m.trellis().host_bytes().unwrap().to_vec(),
            suh: widen(m.suh()),
            svh: widen(m.svh()),
        }
    }

    fn view(&self) -> Exl3Projection<'_> {
        Exl3Projection::new(self.shape, self.words, &self.trellis, &self.suh, &self.svh).unwrap()
    }
}

/// (relative RMS error, largest error over the reference RMS)
fn errors(reference: &[f64], actual: &[f32]) -> (f64, f64) {
    assert_eq!(reference.len(), actual.len());
    let rms = (reference.iter().map(|v| v * v).sum::<f64>() / reference.len() as f64).sqrt();
    let diffs: Vec<f64> = reference
        .iter()
        .zip(actual)
        .map(|(r, a)| (r - f64::from(*a)).abs())
        .collect();
    let err_rms = (diffs.iter().map(|d| d * d).sum::<f64>() / diffs.len() as f64).sqrt();
    let max = diffs.iter().fold(0.0f64, |m, d| m.max(*d));
    (err_rms / rms, max / rms)
}

fn tiers() -> Vec<Exl3Isa> {
    [Exl3Isa::Scalar, Exl3Isa::Avx2, Exl3Isa::Avx512]
        .into_iter()
        .filter(|tier| tier.is_supported())
        .collect()
}

fn check(label: &str, reference: &[f64], actual: &[f32]) {
    let (rel_rms, max) = errors(reference, actual);
    println!("{label}: rel RMS {rel_rms:.3e}, max/RMS {max:.3e}");
    assert!(rel_rms < 2.5e-3, "{label}: rel RMS {rel_rms}");
    assert!(max < 4e-3, "{label}: max {max}");
}

#[test]
fn projections_match_the_fp64_oracle_for_every_bit_rate_and_tier() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    for (k, n, words) in [
        (256u32, 128u32, 64u32),
        (128, 384, 64),
        (256, 256, 24),
        (128, 128, 128),
        (128, 128, 16),
        (256, 128, 32),
        (128, 256, 40),
        (128, 128, 48),
        (128, 128, 56),
        (128, 256, 80),
        (256, 128, 96),
        (128, 128, 112),
        (384, 640, 96),
        (640, 128, 24),
        (128, 384, 16),
    ] {
        let m = matrix(&mut rng, k, n, words);
        let parts = Parts::of(&m);
        let x: Vec<f64> = (0..k).map(|_| rng.unit() * 3.0).collect();
        let mut reference = vec![0.0; n as usize];
        m.project_fp64_into(&x, 1, &mut reference).unwrap();
        let x32: Vec<f32> = x.iter().map(|v| *v as f32).collect();
        for tier in tiers() {
            let cpu = ExpertCpu::new(3, WaitPolicy::Park, Some(tier)).unwrap();
            let mut out = vec![0.0f32; n as usize];
            cpu.project(&parts.view(), &x32, &mut out).unwrap();
            check(
                &format!("{tier:?} [{k},{n}] words={words}"),
                &reference,
                &out,
            );
        }
    }
}

#[test]
fn routed_experts_match_the_fp64_oracle() {
    let mut rng = Rng(0xfeed_face_0bad_f00d);
    let (h, f) = (384u32, 256u32);
    let experts: Vec<[Matrix; 3]> = (0..4)
        .map(|_| {
            [
                matrix(&mut rng, h, f, 64),
                matrix(&mut rng, h, f, 64),
                matrix(&mut rng, f, h, 64),
            ]
        })
        .collect();
    let weights = [0.4f32, 1.25, 0.05, 0.9];
    let x: Vec<f64> = (0..h).map(|_| rng.unit() * 2.0).collect();
    let mut reference = vec![0.0f64; h as usize];
    for (e, [gate, up, down]) in experts.iter().enumerate() {
        let (mut g, mut u) = (vec![0.0; f as usize], vec![0.0; f as usize]);
        gate.project_fp64_into(&x, 1, &mut g).unwrap();
        up.project_fp64_into(&x, 1, &mut u).unwrap();
        let mid: Vec<f64> = g
            .iter()
            .zip(&u)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let mut y = vec![0.0; h as usize];
        down.project_fp64_into(&mid, 1, &mut y).unwrap();
        for (acc, y) in reference.iter_mut().zip(&y) {
            *acc += f64::from(weights[e]) * y;
        }
    }
    let owned: Vec<[Parts; 3]> = experts
        .iter()
        .map(|[g, u, d]| [Parts::of(g), Parts::of(u), Parts::of(d)])
        .collect();
    let picks: Vec<ExpertPick<'_>> = owned
        .iter()
        .zip(weights)
        .map(|([g, u, d], weight)| ExpertPick {
            expert: ExpertView {
                gate: g.view(),
                up: u.view(),
                down: d.view(),
            },
            weight,
        })
        .collect();
    let x32: Vec<f32> = x.iter().map(|v| *v as f32).collect();
    for tier in tiers() {
        for threads in [1usize, 4] {
            let cpu = ExpertCpu::new(threads, WaitPolicy::Park, Some(tier)).unwrap();
            let mut out = vec![0.0f32; h as usize];
            cpu.forward(&x32, &picks, &mut out).unwrap();
            check(&format!("{tier:?} x{threads} experts"), &reference, &out);
        }
    }
}
