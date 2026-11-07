//! CPU kernel for the routed experts of EXL3 mixture-of-experts layers (mul1 codebook).
//!
//! WHAT: [`ExpertCpu::expert_pairs`] computes, for each (expert, activation row) pair, the
//! unweighted expert output; [`ExpertCpu::forward`] computes
//! `sum_e w_e * down_e(silu(gate_e(x)) * up_e(x))` for one activation row and the picked
//! experts, with every projection read straight from the stored EXL3 trellis bytes. [`ExpertCpu::project`] runs one stored matrix the same way.
//!
//! WHY: a cold expert costs the host about 6 KB of activations to compute where copying it to
//! the device costs 1.6 MB of PCIe traffic. The work is DRAM-bound, so the kernel spends its
//! effort on keeping every core's weight stream busy and on cheap state decoding.
//!
//! One projection of a matrix `W = diag(suh) H W' H diag(svh)` runs as: scale the input by
//! `suh`, apply the normalized 128-point Hadamard transform per block, multiply by the decoded
//! `W'` (stored as 16x16 trellis tiles), apply the Hadamard transform and `svh` to the result.
//! This is the device kernel's order. The decoded mul1 weight is
//! `alpha * (1024 + S) + beta`, with `S` the byte sum of `state * 0x83DCD12D` and the two
//! binary16 constants of the codebook, so a projection only needs `sum_k a_k S(k, col)` and the
//! plain sum of the rotated activations.
//!
//! Numerics: activations stay f32 (no activation quantization), sums are f32, and the weight is
//! the exact affine value where the device kernel rounds it to binary16 once. Results do not
//! depend on the thread count: every output element is produced by one chunk in a fixed order,
//! and the experts are combined in pick order. The SIMD and scalar column kernels agree to f32
//! rounding.
//!
//! ISA tiers: the scalar reference works for every mul1 bit rate and is the fallback on CPUs
//! without AVX2; the AVX2 + FMA tier (`exl3_cpu_avx2`) and the AVX-512 VBMI/VNNI tier
//! (`exl3_cpu_avx512`) each handle every stored rate (1 to 8 bits, including the half-bit rates).
//! The tier is chosen at run time: AVX-512 if the CPU has it, else AVX2, else scalar.
//!
//! Threading: [`ExpertCpu`] owns an [`Exl3Pool`]; a layer runs as five short phases (rotate,
//! project gate/up, activate and rotate, project down, rotate and scale), each a pool call.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use half::{bf16, f16};

use crate::EngError;
#[cfg(target_arch = "x86_64")]
use crate::kernels::exl3_cpu_avx2 as avx2;
#[cfg(target_arch = "x86_64")]
use crate::kernels::exl3_cpu_avx512 as avx512;
use crate::kernels::exl3_cpu_pool::Exl3Pool;
use crate::kernels::nvfp4_cpu::WaitPolicy;

/// Input channels sharing one Hadamard block.
const BLOCK: usize = 128;
/// Side of a trellis tile.
const TILE: usize = 16;
const MUL1: u32 = 0x83dc_d12d;
/// Tile columns one pool chunk projects: 2 columns read 256 bytes per k step, 32 KB in all for
/// a 2048-deep matrix.
const COLUMN_TILES_PER_CHUNK: usize = 2;

/// Instruction set tier of the column kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exl3Isa {
    Scalar,
    /// AVX2 and FMA.
    Avx2,
    /// AVX-512 F/BW/VL/VBMI/VNNI.
    Avx512,
}

/// The tiers a CPU can run, kept apart from the host query so the choice can be tested for CPUs
/// other than the one the tests run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IsaSupport {
    avx512: bool,
    avx2: bool,
}

impl IsaSupport {
    fn host() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx512: avx512::supported(),
                avx2: avx2::supported(),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                avx512: false,
                avx2: false,
            }
        }
    }

    fn allows(self, isa: Exl3Isa) -> bool {
        match isa {
            Exl3Isa::Scalar => true,
            Exl3Isa::Avx2 => self.avx2,
            Exl3Isa::Avx512 => self.avx512,
        }
    }

    /// The automatic order: AVX-512, then AVX2, then scalar.
    fn best(self) -> Exl3Isa {
        [Exl3Isa::Avx512, Exl3Isa::Avx2]
            .into_iter()
            .find(|&isa| self.allows(isa))
            .unwrap_or(Exl3Isa::Scalar)
    }
}

impl Exl3Isa {
    /// The best tier this CPU supports.
    pub fn detect() -> Self {
        IsaSupport::host().best()
    }

    pub fn is_supported(self) -> bool {
        IsaSupport::host().allows(self)
    }
}

/// One stored mul1 matrix, borrowed from host memory.
#[derive(Clone, Copy)]
pub struct Exl3Projection<'a> {
    k: usize,
    n: usize,
    words: usize,
    trellis: &'a [u8],
    suh: &'a [u8],
    svh: &'a [u8],
}

impl<'a> Exl3Projection<'a> {
    /// `trellis` is the `[k/16, n/16, words]` u16 tensor as little-endian bytes; `suh` and `svh`
    /// are the widened (f32, little-endian) input and output scale vectors, as the device-ready
    /// expert packs store them, so a pinned pack is used without a copy.
    pub fn new(
        shape: [usize; 2],
        words: usize,
        trellis: &'a [u8],
        suh: &'a [u8],
        svh: &'a [u8],
    ) -> Result<Self, EngError> {
        let [k, n] = shape;
        let bad = |what: &str| Err(EngError::Checkpoint(format!("EXL3 CPU projection: {what}")));
        if k == 0 || n == 0 || !k.is_multiple_of(BLOCK) || !n.is_multiple_of(BLOCK) {
            return bad("dimensions must be nonzero multiples of 128");
        }
        let integer = words.is_multiple_of(16) && (16..=128).contains(&words);
        if !integer && !matches!(words, 24 | 40 | 56) {
            return bad("unsupported words per tile");
        }
        let tiles = (k / TILE) * (n / TILE);
        if tiles.checked_mul(words * 2) != Some(trellis.len()) {
            return bad("trellis length does not match the shape");
        }
        if suh.len() != 4 * k || svh.len() != 4 * n {
            return bad("scale vector lengths do not match the shape");
        }
        Ok(Self {
            k,
            n,
            words,
            trellis,
            suh,
            svh,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.k, self.n]
    }

    fn suh_at(&self, i: usize) -> f32 {
        scale_at(self.suh, i)
    }

    fn svh_at(&self, i: usize) -> f32 {
        scale_at(self.svh, i)
    }

    fn k_tiles(&self) -> usize {
        self.k / TILE
    }

    fn n_tiles(&self) -> usize {
        self.n / TILE
    }

    fn tile(&self, kt: usize, nt: usize) -> &'a [u8] {
        let bytes = self.words * 2;
        let at = (kt * self.n_tiles() + nt) * bytes;
        &self.trellis[at..at + bytes]
    }
}

/// The three projections of one routed expert.
#[derive(Clone, Copy)]
pub struct ExpertView<'a> {
    pub gate: Exl3Projection<'a>,
    pub up: Exl3Projection<'a>,
    pub down: Exl3Projection<'a>,
}

impl ExpertView<'_> {
    fn hidden(&self) -> usize {
        self.gate.k
    }

    fn inner(&self) -> usize {
        self.gate.n
    }

    fn check_shapes(&self) -> Result<(), EngError> {
        let (h, f) = (self.hidden(), self.inner());
        if self.up.shape() != [h, f] || self.down.shape() != [f, h] {
            return Err(EngError::BadRequest(
                "EXL3 CPU expert projections have inconsistent shapes".into(),
            ));
        }
        Ok(())
    }
}

/// A routed expert and the weight its output is scaled by.
#[derive(Clone, Copy)]
pub struct ExpertPick<'a> {
    pub expert: ExpertView<'a>,
    pub weight: f32,
}

/// One expert applied to one row of a batch of activation rows.
#[derive(Clone, Copy)]
pub struct PairJob<'a> {
    pub expert: ExpertView<'a>,
    pub row: usize,
}

fn scale_at(bytes: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([
        bytes[4 * i],
        bytes[4 * i + 1],
        bytes[4 * i + 2],
        bytes[4 * i + 3],
    ])
}

/// The binary16 constants of the mul1 codebook as `weight = alpha * S + gamma`.
fn affine() -> (f32, f32) {
    let alpha = f16::from_bits(0x1eee).to_f32();
    let beta = f16::from_bits(0xc931).to_f32();
    (alpha, 1024.0 * alpha + beta)
}

fn ring_byte(tile: &[u8], m: usize) -> u32 {
    let m = m % tile.len();
    u32::from(tile[4 * (m / 4) + 3 - m % 4])
}

/// The 16-bit trellis state of position `p`: the ring is read most significant bit first from
/// little-endian 32-bit words, and the window of position `p` ends at bit `(p + 1) * bits / 2`.
fn trellis_state(tile: &[u8], p: usize) -> u32 {
    let bits_twice = tile.len() / 16;
    let ring_bits = tile.len() * 8;
    let start = ((p + 1) * bits_twice / 2 + ring_bits - 16) % ring_bits;
    let byte = start / 8;
    let word = (ring_byte(tile, byte) << 24)
        | (ring_byte(tile, byte + 1) << 16)
        | (ring_byte(tile, byte + 2) << 8)
        | ring_byte(tile, byte + 3);
    (word >> (16 - start % 8)) & 0xffff
}

fn byte_sum(x: u32) -> u32 {
    x.to_le_bytes().into_iter().map(u32::from).sum()
}

/// Row of lane `q` in the activation vector of group parity `b`, for tile position
/// `p = 16 g + q`, `b = g & 1`.
fn lane_row(q: usize, b: usize) -> usize {
    (q & 1) + 8 * ((q >> 1) & 1) + 2 * ((q >> 3) & 1) + 4 * b
}

/// Scalar reference: adds `a_lane * S` of every position of one tile to its column's sum.
/// `a` holds the two 16-lane activation vectors of the tile (see [`lane_row`]).
fn tile_sums_scalar(tile: &[u8], a: &[f32; 32], acc: &mut [f32; TILE]) {
    for p in 0..TILE * TILE {
        let sum = byte_sum(trellis_state(tile, p).wrapping_mul(MUL1));
        let col = (p >> 5) + 8 * ((p >> 2) & 1);
        acc[col] += a[((p >> 4) & 1) * 16 + (p & 15)] * sum as f32;
    }
}

fn load(buf: &[AtomicU32], i: usize) -> f32 {
    f32::from_bits(buf[i].load(Ordering::Relaxed))
}

fn store(buf: &[AtomicU32], i: usize, v: f32) {
    buf[i].store(v.to_bits(), Ordering::Relaxed);
}

fn load_block(buf: &[AtomicU32], start: usize) -> [f32; BLOCK] {
    std::array::from_fn(|l| load(buf, start + l))
}

fn store_block(buf: &[AtomicU32], start: usize, v: &[f32; BLOCK]) {
    for (l, &value) in v.iter().enumerate() {
        store(buf, start + l, value);
    }
}

/// Normalized 128-point Walsh-Hadamard transform in the device kernel's butterfly order.
fn hadamard128(v: &mut [f32; BLOCK]) {
    let mut step = 1;
    while step < BLOCK {
        for base in (0..BLOCK).step_by(2 * step) {
            for i in base..base + step {
                let (lo, hi) = (v[i], v[i + step]);
                v[i] = lo + hi;
                v[i + step] = lo - hi;
            }
        }
        step *= 2;
    }
    let scale = 1.0 / (BLOCK as f32).sqrt();
    v.iter_mut().for_each(|x| *x *= scale);
}

/// `value` rounded to bf16 when `on`, as the device kernels publish their intermediates.
fn publish(value: f32, on: bool) -> f32 {
    if on {
        bf16::from_f32(value).to_f32()
    } else {
        value
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Writes the rotated block `block` into the activation table: per k tile, the two 16-lane
/// vectors the column kernels consume.
fn store_activation_block(table: &[AtomicU32], block: usize, v: &[f32; BLOCK]) {
    for tile in 0..BLOCK / TILE {
        for b in 0..2 {
            for q in 0..16 {
                let at = ((block * (BLOCK / TILE) + tile) * 2 + b) * 16 + q;
                store(table, at, v[tile * TILE + lane_row(q, b)]);
            }
        }
    }
}

/// Phase 0: scales and rotates input block `j`; fills the activation table and the block sum.
fn rotate_in(
    proj: &Exl3Projection<'_>,
    x: &[f32],
    j: usize,
    table: &[AtomicU32],
    sums: &[AtomicU32],
) {
    let range = j * BLOCK..(j + 1) * BLOCK;
    let mut v: [f32; BLOCK] =
        std::array::from_fn(|l| x[range.start + l] * proj.suh_at(range.start + l));
    hadamard128(&mut v);
    store(sums, j, v.iter().sum());
    store_activation_block(table, j, &v);
}

#[cfg(target_arch = "x86_64")]
fn simd_column_sums(
    isa: Exl3Isa,
    proj: &Exl3Projection<'_>,
    table: &[AtomicU32],
    nt: usize,
    out: &mut [f32; TILE],
) -> bool {
    let shape = (proj.n_tiles(), proj.k_tiles());
    match isa {
        Exl3Isa::Scalar => return false,
        Exl3Isa::Avx2 => avx2::column_tile(proj.trellis, proj.words / 8, shape, nt, table, out),
        Exl3Isa::Avx512 => avx512::column_tile(proj.trellis, proj.words / 8, shape, nt, table, out),
    }
    true
}

#[cfg(not(target_arch = "x86_64"))]
fn simd_column_sums(
    _isa: Exl3Isa,
    _proj: &Exl3Projection<'_>,
    _table: &[AtomicU32],
    _nt: usize,
    _out: &mut [f32; TILE],
) -> bool {
    false
}

/// `sum_k a_k S(k, col)` for the 16 columns of tile column `nt`.
fn column_sums(
    isa: Exl3Isa,
    proj: &Exl3Projection<'_>,
    table: &[AtomicU32],
    nt: usize,
) -> [f32; TILE] {
    let mut out = [0.0; TILE];
    if simd_column_sums(isa, proj, table, nt, &mut out) {
        return out;
    }
    for kt in 0..proj.k_tiles() {
        let a: [f32; 32] = std::array::from_fn(|i| load(table, kt * 32 + i));
        tile_sums_scalar(proj.tile(kt, nt), &a, &mut out);
    }
    out
}

/// Phase 1/3: projects `COLUMN_TILES_PER_CHUNK` tile columns into the raw (unrotated) output.
fn project_chunk(
    isa: Exl3Isa,
    proj: &Exl3Projection<'_>,
    chunk: usize,
    input: (&[AtomicU32], &[AtomicU32]),
    z: &[AtomicU32],
) {
    let (table, sums) = input;
    let (alpha, gamma) = affine();
    let total: f32 = (0..sums.len()).map(|j| load(sums, j)).sum();
    for nt in chunk * COLUMN_TILES_PER_CHUNK..(chunk + 1) * COLUMN_TILES_PER_CHUNK {
        let acc = column_sums(isa, proj, table, nt);
        for (c, &value) in acc.iter().enumerate() {
            store(z, nt * TILE + c, alpha * value + gamma * total);
        }
    }
}

/// A grow-only buffer of f32 values the pool chunks write with disjoint index ranges.
#[derive(Default)]
struct SharedF32(Vec<AtomicU32>);

impl SharedF32 {
    fn ensure(&mut self, len: usize) {
        if self.0.len() < len {
            self.0.resize_with(len, || AtomicU32::new(0));
        }
    }

    fn slice(&self, start: usize, len: usize) -> &[AtomicU32] {
        &self.0[start..start + len]
    }
}

#[derive(Default)]
struct Scratch {
    table_in: SharedF32,
    sums_in: SharedF32,
    z_gate_up: SharedF32,
    table_mid: SharedF32,
    sums_mid: SharedF32,
    z_down: SharedF32,
    out: SharedF32,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The CPU expert engine: a worker pool, the chosen ISA tier and reusable scratch.
pub struct ExpertCpu {
    pool: Exl3Pool,
    isa: Exl3Isa,
    published: bool,
    scratch: Mutex<Scratch>,
}

impl ExpertCpu {
    /// `threads` counts the calling thread. `isa` forces a tier; `None` picks the best supported.
    pub fn new(threads: usize, wait: WaitPolicy, isa: Option<Exl3Isa>) -> Result<Self, EngError> {
        let isa = isa.unwrap_or_else(Exl3Isa::detect);
        if !isa.is_supported() {
            return Err(EngError::BadRequest(format!(
                "EXL3 CPU kernel tier {isa:?} is not supported by this CPU"
            )));
        }
        Ok(Self {
            pool: Exl3Pool::new(threads, wait)?,
            isa,
            published: false,
            scratch: Mutex::new(Scratch::default()),
        })
    }

    /// Rounds the expert's intermediate values to bf16 at the points the device kernels publish
    /// them (gate and up outputs, the sigmoid, both products, the expert output), so the CPU
    /// result follows the device pipeline instead of staying in f32 throughout.
    pub fn with_published_rounding(mut self, on: bool) -> Self {
        self.published = on;
        self
    }

    pub fn isa(&self) -> Exl3Isa {
        self.isa
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// One stored matrix: `out = x * W`, with `x` of length `k` and `out` of length `n`.
    pub fn project(
        &self,
        proj: &Exl3Projection<'_>,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), EngError> {
        let (k, n) = (proj.k, proj.n);
        if x.len() != k || out.len() != n {
            return Err(EngError::BadRequest(
                "EXL3 CPU projection spans do not match the matrix".into(),
            ));
        }
        let mut guard = lock(&self.scratch);
        let sc = &mut *guard;
        sc.table_in.ensure(2 * k);
        sc.sums_in.ensure(k / BLOCK);
        sc.z_gate_up.ensure(n);
        let (table, sums, z) = (
            sc.table_in.slice(0, 2 * k),
            sc.sums_in.slice(0, k / BLOCK),
            sc.z_gate_up.slice(0, n),
        );
        self.pool
            .run(k / BLOCK, |j| rotate_in(proj, x, j, table, sums));
        let isa = self.isa;
        let chunks = proj.n_tiles() / COLUMN_TILES_PER_CHUNK;
        self.pool
            .run(chunks, |i| project_chunk(isa, proj, i, (table, sums), z));
        self.pool.run(n / BLOCK, |j| {
            let mut v = load_block(z, j * BLOCK);
            hadamard128(&mut v);
            v.iter_mut()
                .enumerate()
                .for_each(|(l, value)| *value *= proj.svh_at(j * BLOCK + l));
            store_block(z, j * BLOCK, &v);
        });
        for (i, value) in out.iter_mut().enumerate() {
            *value = load(z, i);
        }
        Ok(())
    }

    /// `out = sum_e weight_e * down_e(silu(gate_e(x)) * up_e(x))`, accumulated in pick order.
    pub fn forward(
        &self,
        x: &[f32],
        picks: &[ExpertPick<'_>],
        out: &mut [f32],
    ) -> Result<(), EngError> {
        let h = x.len();
        if out.len() != h {
            return Err(EngError::BadRequest(
                "EXL3 CPU expert spans do not match the hidden size".into(),
            ));
        }
        let jobs: Vec<PairJob<'_>> = picks
            .iter()
            .map(|pick| PairJob {
                expert: pick.expert,
                row: 0,
            })
            .collect();
        let mut each = vec![0.0f32; jobs.len() * h];
        self.expert_pairs(x, &jobs, &mut each)?;
        out.fill(0.0);
        for (pick, y) in picks.iter().zip(each.chunks_exact(h.max(1))) {
            for (acc, value) in out.iter_mut().zip(y) {
                *acc += pick.weight * value;
            }
        }
        Ok(())
    }

    /// For every job, `down(silu(gate(x_row)) * up(x_row))` without a routing weight, written
    /// job-major to `out` (`jobs.len() * hidden` values). `xs` holds the activation rows.
    pub fn expert_pairs(
        &self,
        xs: &[f32],
        jobs: &[PairJob<'_>],
        out: &mut [f32],
    ) -> Result<(), EngError> {
        let Some(first) = jobs.first() else {
            return Ok(());
        };
        let (h, f) = (first.expert.hidden(), first.expert.inner());
        let rows = xs.len() / h;
        if xs.len() != rows * h || out.len() != jobs.len() * h {
            return Err(EngError::BadRequest(
                "EXL3 CPU expert spans do not match the hidden size".into(),
            ));
        }
        for job in jobs {
            job.expert.check_shapes()?;
            if job.expert.hidden() != h || job.expert.inner() != f || job.row >= rows {
                return Err(EngError::BadRequest(
                    "EXL3 CPU experts of one call must share their shapes and name a valid row"
                        .into(),
                ));
            }
        }
        let mut guard = lock(&self.scratch);
        let layout = Layout::new(jobs.len(), h, f);
        layout.reserve(&mut guard);
        self.run_phases(xs, jobs, &guard, &layout);
        for (i, value) in out.iter_mut().enumerate() {
            *value = load(guard.out.slice(0, jobs.len() * h), i);
        }
        Ok(())
    }

    fn run_phases(&self, xs: &[f32], jobs: &[PairJob<'_>], sc: &Scratch, l: &Layout) {
        let (isa, pool, published) = (self.isa, &self.pool, self.published);
        let (e_n, h, f, bh, bf) = (
            l.experts,
            l.hidden,
            l.inner,
            l.hidden / BLOCK,
            l.inner / BLOCK,
        );
        let gate_up = |e: usize, m: usize| {
            if m == 0 {
                &jobs[e].expert.gate
            } else {
                &jobs[e].expert.up
            }
        };
        let input = |s: usize| {
            (
                sc.table_in.slice(s * 2 * h, 2 * h),
                sc.sums_in.slice(s * bh, bh),
            )
        };
        pool.run(e_n * 2 * bh, |i| {
            let (s, j) = (i / bh, i % bh);
            let (table, sums) = input(s);
            let x = &xs[jobs[s / 2].row * h..][..h];
            rotate_in(gate_up(s / 2, s % 2), x, j, table, sums);
        });
        let per = (f / TILE) / COLUMN_TILES_PER_CHUNK;
        pool.run(e_n * 2 * per, |i| {
            let s = i / per;
            let z = sc.z_gate_up.slice(s * f, f);
            project_chunk(isa, gate_up(s / 2, s % 2), i % per, input(s), z);
        });
        pool.run(e_n * bf, |i| {
            let (e, j) = (i / bf, i % bf);
            let expert = &jobs[e].expert;
            let z = |m: usize| load_block(sc.z_gate_up.slice((e * 2 + m) * f, f), j * BLOCK);
            let (mut g, mut u) = (z(0), z(1));
            hadamard128(&mut g);
            hadamard128(&mut u);
            let mut mid: [f32; BLOCK] = std::array::from_fn(|l| {
                let c = j * BLOCK + l;
                let gate = publish(g[l] * expert.gate.svh_at(c), published);
                let up = publish(u[l] * expert.up.svh_at(c), published);
                let activated = if published {
                    // The device publishes the sigmoid, the gate product and the up product.
                    let sigmoid = publish(1.0 / (1.0 + (-gate).exp()), true);
                    let silu = publish(gate * sigmoid, true);
                    publish(silu * up, true)
                } else {
                    silu(gate) * up
                };
                activated * expert.down.suh_at(c)
            });
            hadamard128(&mut mid);
            store(sc.sums_mid.slice(e * bf, bf), j, mid.iter().sum());
            store_activation_block(sc.table_mid.slice(e * 2 * f, 2 * f), j, &mid);
        });
        let per = (h / TILE) / COLUMN_TILES_PER_CHUNK;
        pool.run(e_n * per, |i| {
            let e = i / per;
            let input = (
                sc.table_mid.slice(e * 2 * f, 2 * f),
                sc.sums_mid.slice(e * bf, bf),
            );
            let z = sc.z_down.slice(e * h, h);
            project_chunk(isa, &jobs[e].expert.down, i % per, input, z);
        });
        pool.run(e_n * bh, |i| {
            let (e, j) = (i / bh, i % bh);
            let mut v = load_block(sc.z_down.slice(e * h, h), j * BLOCK);
            hadamard128(&mut v);
            for (l, value) in v.iter_mut().enumerate() {
                *value = publish(
                    *value * jobs[e].expert.down.svh_at(j * BLOCK + l),
                    published,
                );
            }
            store_block(sc.out.slice(e * h, h), j * BLOCK, &v);
        });
    }
}

/// Buffer sizes of one `expert_pairs` call.
struct Layout {
    experts: usize,
    hidden: usize,
    inner: usize,
}

impl Layout {
    fn new(experts: usize, hidden: usize, inner: usize) -> Self {
        Self {
            experts,
            hidden,
            inner,
        }
    }

    fn reserve(&self, sc: &mut Scratch) {
        let (e, h, f) = (self.experts, self.hidden, self.inner);
        sc.table_in.ensure(e * 2 * 2 * h);
        sc.sums_in.ensure(e * 2 * (h / BLOCK));
        sc.z_gate_up.ensure(e * 2 * f);
        sc.table_mid.ensure(e * 2 * f);
        sc.sums_mid.ensure(e * (f / BLOCK));
        sc.z_down.ensure(e * h);
        sc.out.ensure(e * h);
    }
}

#[cfg(test)]
#[path = "exl3_cpu_tests.rs"]
mod tests;
