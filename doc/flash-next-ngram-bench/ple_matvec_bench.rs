// Host cost of the PLE key/value projections: [12800, 2560] x [2560], F32 vs F16 storage, N threads.
// rustc -O -C target-cpu=native ple_matvec_bench.rs ; taskset -c 16-23 ./ple_matvec_bench
use std::arch::x86_64::*;
use std::time::Instant;
const ROWS: usize = 12800;
const COLS: usize = 2560;

fn dot_f32(w: &[f32], x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (a, b) in w.chunks_exact(8).zip(x.chunks_exact(8)) {
        for i in 0..8 { acc[i] += a[i] * b[i]; }
    }
    acc.iter().sum()
}
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f16(w: &[u16], x: &[f32]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for i in (0..w.len()).step_by(8) {
        let h = _mm_loadu_si128(w.as_ptr().add(i) as *const __m128i);
        acc = _mm256_fmadd_ps(_mm256_cvtph_ps(h), _mm256_loadu_ps(x.as_ptr().add(i)), acc);
    }
    let mut t = [0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), acc);
    t.iter().sum()
}
fn run<F: Fn(usize, &mut [f32]) + Sync>(threads: usize, f: F) -> Vec<f32> {
    let mut out = vec![0f32; ROWS];
    let chunk = ROWS.div_ceil(threads);
    std::thread::scope(|s| {
        for (i, part) in out.chunks_mut(chunk).enumerate() {
            let f = &f;
            s.spawn(move || f(i * chunk, part));
        }
    });
    out
}
fn main() {
    let w32: Vec<f32> = (0..ROWS * COLS).map(|i| ((i % 977) as f32) * 1e-3 - 0.4).collect();
    let w16: Vec<u16> = w32.iter().map(|&v| unsafe { _mm_extract_epi16(_mm256_cvtps_ph::<0>(_mm256_set1_ps(v)), 0) as u16 }).collect();
    let x: Vec<f32> = (0..COLS).map(|i| (i % 13) as f32 * 0.1).collect();
    for threads in [1usize, 2, 4, 8, 16] {
        for name in ["f32", "f16"] {
            let mut t = vec![];
            for _ in 0..40 {
                let s = Instant::now();
                let o = run(threads, |base, part| {
                    for (k, slot) in part.iter_mut().enumerate() {
                        let r = base + k;
                        *slot = if name == "f32" { dot_f32(&w32[r * COLS..(r + 1) * COLS], &x) }
                                else { unsafe { dot_f16(&w16[r * COLS..(r + 1) * COLS], &x) } };
                    }
                });
                std::hint::black_box(o);
                t.push(s.elapsed().as_secs_f64() * 1e3);
            }
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("{name} threads={threads:<2} p50={:.2} ms p99={:.2} ms min={:.2} ms", t[20], t[39], t[0]);
        }
    }
}
