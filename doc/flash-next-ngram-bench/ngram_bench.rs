// Standalone n-gram row read-latency benchmark (std only, buffered pread, no profiling).
// Build: rustc -O -C target-cpu=native ngram_bench.rs -o ngram_bench
// Run:   taskset -c 16-23 ./ngram_bench <ngram_embedding.safetensors> <section>
// The table is the contiguous run of shard trellis tensors; rows are 102 bytes at random positions.
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}
const FADV_RANDOM: i32 = 1;
const FADV_DONTNEED: i32 = 4;

const ROW_BYTES: u64 = 102;
const ROWS: u64 = 320_001_536;
const HEADER: u64 = 8 + 21_328;
const FIRST_SHARD: u64 = 5_400;
const BASE: u64 = HEADER + FIRST_SHARD;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn row(&mut self) -> u64 {
        self.next() % ROWS
    }
}

fn offset(row: u64) -> u64 {
    BASE + row * ROW_BYTES
}

fn read_row(f: &File, row: u64) -> u8 {
    let mut b = [0u8; ROW_BYTES as usize];
    f.read_exact_at(&mut b, offset(row)).unwrap();
    b[0] ^ b[101]
}

fn drop_rows(f: &File, rows: &[u64]) {
    // posix_fadvise DONTNEED on our own 4 KiB pages only.
    let fd = f.as_raw_fd();
    for &r in rows {
        let o = offset(r);
        let page = o & !4095;
        let end = (o + ROW_BYTES + 4095) & !4095;
        unsafe { posix_fadvise(fd, page as i64, (end - page) as i64, FADV_DONTNEED) };
    }
}

fn pct(v: &mut Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn report(label: &str, v: &mut Vec<f64>, extra: &str) {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    println!(
        "{label:<46} n={:<7} p50={:>9.1} p99={:>9.1} max={:>10.1} mean={:>9.1} us {extra}",
        v.len(),
        pct(v, 0.5),
        pct(v, 0.99),
        v.last().unwrap(),
        mean
    );
}

// Persistent pool: each worker takes (rows, reply) jobs.
type Job = (Vec<u64>, mpsc::Sender<u8>);
struct Pool {
    tx: Vec<mpsc::Sender<Job>>,
}
impl Pool {
    fn new(f: Arc<File>, n: usize) -> Pool {
        let mut tx = Vec::new();
        for _ in 0..n {
            let (t, r) = mpsc::channel::<Job>();
            let f = f.clone();
            std::thread::spawn(move || {
                while let Ok((rows, reply)) = r.recv() {
                    let mut acc = 0u8;
                    for row in rows {
                        acc ^= read_row(&f, row);
                    }
                    let _ = reply.send(acc);
                }
            });
            tx.push(t);
        }
        Pool { tx }
    }
    // Split rows over the workers, wait for all.
    fn run(&self, rows: &[u64]) {
        let (rt, rr) = mpsc::channel();
        let n = self.tx.len();
        let per = rows.len().div_ceil(n).max(1);
        let mut jobs = 0;
        for (i, part) in rows.chunks(per).enumerate() {
            self.tx[i % n].send((part.to_vec(), rt.clone())).unwrap();
            jobs += 1;
        }
        for _ in 0..jobs {
            rr.recv().unwrap();
        }
    }
}

fn qd_sweep(f: &Arc<File>, qd: usize, reads: usize, seed: u64) {
    // Each thread reads blocks of 256 fresh random rows, then the same block again at once: the
    // second read is the warm figure, taken while the pages are certainly resident.
    let mut rng = Rng(seed);
    let rows: Vec<u64> = (0..reads).map(|_| rng.row()).collect();
    let per = reads / qd;
    let t0 = Instant::now();
    let lat = Arc::new(Mutex::new((Vec::with_capacity(reads), Vec::with_capacity(reads))));
    let mut hs = vec![];
    for t in 0..qd {
        let f = f.clone();
        let part = rows[t * per..(t + 1) * per].to_vec();
        let lat = lat.clone();
        hs.push(std::thread::spawn(move || {
            let (mut cold, mut warm) = (Vec::new(), Vec::new());
            for block in part.chunks(256) {
                for &r in block {
                    let s = Instant::now();
                    read_row(&f, r);
                    cold.push(s.elapsed().as_secs_f64() * 1e6);
                }
                for &r in block {
                    let s = Instant::now();
                    read_row(&f, r);
                    warm.push(s.elapsed().as_secs_f64() * 1e6);
                }
            }
            let mut g = lat.lock().unwrap();
            g.0.extend(cold);
            g.1.extend(warm);
        }));
    }
    for h in hs {
        h.join().unwrap();
    }
    let wall = t0.elapsed().as_secs_f64();
    let (mut cold, mut warm) = Arc::try_unwrap(lat).unwrap().into_inner().unwrap();
    let iops = (cold.len() + warm.len()) as f64 / wall;
    report(&format!("row read QD{qd} cold"), &mut cold, &format!("(mixed iops={iops:.0})"));
    report(&format!("row read QD{qd} warm"), &mut warm, "");
    drop_rows(f, &rows);
}

fn token_batches(f: &Arc<File>, threads: usize, tokens: usize, seed: u64) {
    let mut rng = Rng(seed);
    let toks: Vec<Vec<u64>> = (0..tokens).map(|_| (0..16).map(|_| rng.row()).collect()).collect();
    let pool = if threads > 1 { Some(Pool::new(f.clone(), threads)) } else { None };
    let (mut cold, mut warm) = (Vec::with_capacity(tokens), Vec::with_capacity(tokens));
    for t in &toks {
        for out in [&mut cold, &mut warm] {
            let s = Instant::now();
            match &pool {
                None => {
                    for &r in t {
                        read_row(f, r);
                    }
                }
                Some(p) => p.run(t),
            }
            out.push(s.elapsed().as_secs_f64() * 1e6);
        }
    }
    report(&format!("token (16 rows) threads={threads} cold"), &mut cold, "");
    report(&format!("token (16 rows) threads={threads} warm"), &mut warm, "");
    let all: Vec<u64> = toks.concat();
    drop_rows(f, &all);
}

fn chunk(f: &Arc<File>, threads: usize, tokens: usize, seed: u64) {
    let mut rng = Rng(seed);
    let mut rows: Vec<u64> = (0..tokens * 16).map(|_| rng.row()).collect();
    let raw = rows.len();
    rows.sort_unstable();
    rows.dedup();
    let uniq = rows.len();
    let pool = Pool::new(f.clone(), threads);
    for pass in ["cold", "warm"] {
        let s = Instant::now();
        pool.run(&rows);
        let ms = s.elapsed().as_secs_f64() * 1e3;
        println!(
            "chunk {tokens:>5} tokens ({raw} rows, {uniq} uniq, sorted) threads={threads} {pass}: {ms:.1} ms = {:.0} rows/s",
            uniq as f64 / (ms / 1e3)
        );
    }
    drop_rows(f, &rows);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let f = Arc::new(File::open(&args[1]).unwrap());
    if args.get(3).map(|s| s.as_str()) != Some("default-advice") {
        unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, FADV_RANDOM) };
        println!("advice: POSIX_FADV_RANDOM");
    } else {
        println!("advice: default (read-around on)");
    }
    let seed: u64 = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0x9E3779B97F4A7C15);
    match args[2].as_str() {
        "qd" => {
            qd_sweep(&f, 1, 20_000, seed ^ 1);
            qd_sweep(&f, 4, 40_000, seed ^ 2);
            qd_sweep(&f, 16, 100_000, seed ^ 3);
            qd_sweep(&f, 32, 100_000, seed ^ 4);
        }
        "token" => {
            token_batches(&f, 1, 2_000, seed ^ 5);
            token_batches(&f, 4, 2_000, seed ^ 6);
            token_batches(&f, 8, 2_000, seed ^ 7);
            token_batches(&f, 16, 2_000, seed ^ 8);
        }
        "chunk" => {
            chunk(&f, 16, 256, seed ^ 9);
            chunk(&f, 32, 256, seed ^ 10);
            chunk(&f, 16, 4096, seed ^ 11);
            chunk(&f, 32, 4096, seed ^ 12);
        }
        "qd1-short" => qd_sweep(&f, 1, 5_000, seed ^ 13),
        _ => panic!("section: qd | token | chunk | qd1-short"),
    }
}
