#![allow(unsafe_code)]
//! AVX-512 (VBMI + VNNI) column-tile kernel for mul1 EXL3 matrices of every stored bit rate.
//!
//! Scope: the intrinsics half of `exl3_cpu`. It computes, for one 16-column tile column of a
//! stored matrix, `acc[col] = sum_k a_k * S(k, col)`, where `S` is the byte sum of the mul1
//! state product (`state * 0x83DCD12D`) and `a_k` the rotated activation. The caller turns
//! `acc` into the projection with the fp16 affine constants of the codebook. The safe scalar
//! reference in `exl3_cpu` computes the same sums and is the fallback on other CPUs.
//!
//! Per 16 weights (one zmm): the 16 trellis states are cut out of the tile with one
//! cross-register byte permute (two groups of 16 per permute; ring wrap and the byte order of
//! the stored little-endian words are folded into the index table; a 4-bit tile is exactly two
//! registers, other rates load a 64-byte window per permute, the wrap through a second one) and one
//! `vpmultishiftqb` per group, which places each 16-bit state in a dword with the upper half
//! zeroed by the write mask. A dword multiply by the mul1 constant and `vpdpbusd` against
//! a vector of ones then give the byte sum `S` of each state in an i32 lane; it is converted to
//! f32 and fused-multiply-added with the activation of that lane's row. Activations stay f32,
//! so the only departure from the device kernel is that the weight is not rounded to fp16.
//!
//! Lane layout. Positions in a tile are numbered `p = 16 g + q` (group `g`, lane `q`) and map to
//! `row = (q&1) + 8 (q>>1 &1) + 2 (q>>3 &1) + 4 (g&1)`, `col = (g>>1) + 8 (q>>2 &1)`.
//! One accumulator therefore serves column `c` (lane bit 2 clear) and `c + 8` (set) for the
//! group pair `(2c, 2c+1)`; the activation table `a` holds, per k tile, the two 16-lane vectors
//! for `g&1 = 0` and `1`. The final reduction over the lanes of each accumulator is plain
//! scalar code.
//!
//! Unsafe scope: three one-line wrappers around unaligned vector loads and stores of arrays whose
//! length the type fixes, and the safe-to-unsafe call into the target-feature kernel behind the
//! runtime feature check. Confined to this file by the `unsafe_is_confined_to_the_avx2_module`
//! test. On other targets this module is empty and `exl3_cpu` uses the scalar reference.

#[cfg(target_arch = "x86_64")]
pub(crate) use imp::{column_tile, supported};

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::*;
    use std::sync::OnceLock;
    use std::sync::atomic::AtomicU32;

    const MUL1: i32 = 0x83dc_d12d_u32 as i32;
    /// Dword bytes 0 and 1 of every state are kept; bytes 2 and 3 are zeroed.
    const KEEP_LOW_HALVES: u64 = 0x3333_3333_3333_3333;
    /// `bits_twice` whose tile (16 * bits_twice bytes) is exactly two registers.
    const WHOLE_TILE: usize = 8;

    pub(crate) fn supported() -> bool {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512vnni")
    }

    struct Tables {
        /// Source byte for every byte of the permuted vector, per group pair: below 64 selects
        /// from the first window, from 64 on from the second.
        idx: [[u8; 64]; 8],
        /// Multishift controls for the first and the second group of a pair.
        ctrl: [[u8; 64]; 2],
        /// Tile offsets of the two 64-byte windows each group pair reads.
        window: [(usize, usize); 8],
    }

    /// Byte of the stored tile that holds ring byte `m`: the stored words are little-endian u32
    /// while the ring is read most significant bit first.
    fn stored_index(ring_byte: usize, tile_bytes: usize) -> usize {
        let m = ring_byte % tile_bytes;
        4 * (m / 4) + 3 - m % 4
    }

    /// First and second window offsets for group pair `pair`. Every ring byte the pair reads
    /// spans at most 40 bytes, so one window holds all of them unless the span wraps past the
    /// end of the ring; the second window, at the tile start, then holds the wrapped part.
    fn windows(bits_twice: usize, pair: usize) -> (usize, usize) {
        let tile_bytes = 16 * bits_twice;
        if bits_twice == WHOLE_TILE {
            return (0, 64);
        }
        if tile_bytes <= 64 {
            return (0, 0);
        }
        let ring_bits = tile_bytes * 8;
        let start = ((32 * pair + 1) * bits_twice / 2 + ring_bits - 16) % ring_bits;
        ((start / 8 / 4 * 4).min(tile_bytes - 64), 0)
    }

    fn build_tables(bits_twice: usize) -> Tables {
        let tile_bytes = 16 * bits_twice;
        let ring_bits = tile_bytes * 8;
        let start = |p: usize| ((p + 1) * bits_twice / 2 + ring_bits - 16) % ring_bits;
        let mut idx = [[0u8; 64]; 8];
        let mut ctrl = [[0u8; 64]; 2];
        let mut window = [(0usize, 0usize); 8];
        let mut first_pair = [[(0usize, 0usize); 8]; 2];
        for (pair, table) in idx.iter_mut().enumerate() {
            let (first, second) = windows(bits_twice, pair);
            window[pair] = (first, second);
            for half in 0..2 {
                for lane_pair in 0..8 {
                    let p0 = 32 * pair + 16 * half + 2 * lane_pair;
                    let (s0, s1) = (start(p0), start(p0 + 1));
                    let (byte0, offset0) = (s0 / 8, s0 % 8);
                    let offset1 = (s1 + ring_bits - 8 * byte0) % ring_bits;
                    assert!(offset1 + 16 <= 32, "two states must share one 32-bit read");
                    for j in 0..4 {
                        let stored = stored_index(byte0 + 3 - j, tile_bytes);
                        table[8 * lane_pair + 4 * half + j] =
                            if (first..first + 64).contains(&stored) {
                                (stored - first) as u8
                            } else {
                                assert!(
                                    (second..second + 64).contains(&stored),
                                    "byte outside windows"
                                );
                                (64 + stored - second) as u8
                            };
                    }
                    if pair == 0 {
                        first_pair[half][lane_pair] = (offset0, offset1);
                    }
                    // The shifts depend on the position inside the pair only.
                    assert_eq!(first_pair[half][lane_pair], (offset0, offset1));
                    let base = 32 * half;
                    let bytes = &mut ctrl[half][8 * lane_pair..8 * lane_pair + 8];
                    bytes[0] = (base + 16 - offset0) as u8;
                    bytes[1] = bytes[0] + 8;
                    bytes[4] = (base + 16 - offset1) as u8;
                    bytes[5] = bytes[4] + 8;
                }
            }
        }
        Tables { idx, ctrl, window }
    }

    fn tables(bits_twice: usize) -> &'static Tables {
        static TABLES: [OnceLock<Tables>; 17] = [const { OnceLock::new() }; 17];
        TABLES[bits_twice].get_or_init(|| build_tables(bits_twice))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn load_bytes(bytes: &[u8; 64]) -> __m512i {
        // SAFETY: the reference is a valid 64-byte region and the load is unaligned.
        unsafe { _mm512_loadu_si512(bytes.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn load_activations(a: &[AtomicU32; 16]) -> __m512 {
        // SAFETY: 16 contiguous words are valid for reads; no thread writes the activation
        // SAFETY: table while a projection phase runs (the pool joins between phases), so the
        // SAFETY: plain vector load cannot race an atomic store, and `AtomicU32` has the layout
        // SAFETY: of `u32`, whose bit pattern is the stored f32.
        unsafe { _mm512_loadu_ps(a.as_ptr().cast::<f32>()) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn store_lanes(out: &mut [f32; 16], v: __m512) {
        // SAFETY: the reference is a valid 16-float region and the store is unaligned.
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), v) }
    }

    /// The 64 bytes of `trellis` at `start`, zero padded past its end (the last tile of a small
    /// tile size reads beyond the buffer).
    fn window<'a>(trellis: &'a [u8], start: usize, spill: &'a mut [u8; 64]) -> &'a [u8; 64] {
        if let Some(bytes) = trellis.get(start..start + 64) {
            return bytes.try_into().unwrap();
        }
        spill.fill(0);
        let rest = &trellis[start..];
        spill[..rest.len()].copy_from_slice(rest);
        spill
    }

    /// Accumulates `sum_k a_k S(k, col)` over all `k_tiles` of tile column `nt` for tiles of
    /// `16 * bits_twice` bytes.
    ///
    /// `trellis` holds the `[k_tiles, n_tiles, words]` u16 words as bytes; `a` is the activation
    /// table, 32 floats per k tile. Slice lengths are validated by the caller and re-checked by
    /// the indexing here, so a mistake panics instead of reading out of bounds.
    pub(crate) fn column_tile(
        trellis: &[u8],
        bits_twice: usize,
        shape: (usize, usize),
        nt: usize,
        a: &[AtomicU32],
        out: &mut [f32; 16],
    ) {
        assert!(supported(), "AVX-512 VBMI and VNNI are required");
        assert!((2..=16).contains(&bits_twice), "unsupported tile size");
        // SAFETY: the required target features were detected at runtime just above.
        unsafe {
            if bits_twice == WHOLE_TILE {
                column_tile_inner::<true>(trellis, bits_twice, shape, nt, a, out)
            } else {
                column_tile_inner::<false>(trellis, bits_twice, shape, nt, a, out)
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    fn column_tile_inner<const WHOLE: bool>(
        trellis: &[u8],
        bits_twice: usize,
        (n_tiles, k_tiles): (usize, usize),
        nt: usize,
        a: &[AtomicU32],
        out: &mut [f32; 16],
    ) {
        let t = tables(bits_twice);
        let tile_bytes = 16 * bits_twice;
        let ctrl = [load_bytes(&t.ctrl[0]), load_bytes(&t.ctrl[1])];
        let mul = _mm512_set1_epi32(MUL1);
        let ones = _mm512_set1_epi8(1);
        let zero = _mm512_setzero_si512();
        let mut acc = [_mm512_setzero_ps(); 8];
        let (mut spill_a, mut spill_b) = ([0u8; 64], [0u8; 64]);
        for kt in 0..k_tiles {
            let at = (kt * n_tiles + nt) * tile_bytes;
            let av = &a[kt * 32..kt * 32 + 32];
            let a0 = load_activations(av[..16].try_into().unwrap());
            let a1 = load_activations(av[16..].try_into().unwrap());
            let whole = if WHOLE {
                let (low, high) = trellis[at..at + tile_bytes].split_at(64);
                Some((
                    load_bytes(low.try_into().unwrap()),
                    load_bytes(high.try_into().unwrap()),
                ))
            } else {
                None
            };
            for (c, acc) in acc.iter_mut().enumerate() {
                let (low, high) = match whole {
                    Some(pair) => pair,
                    None => {
                        let (first, second) = t.window[c];
                        (
                            load_bytes(window(trellis, at + first, &mut spill_a)),
                            load_bytes(window(trellis, at + second, &mut spill_b)),
                        )
                    }
                };
                let states = _mm512_permutex2var_epi8(low, load_bytes(&t.idx[c]), high);
                let first = _mm512_maskz_multishift_epi64_epi8(KEEP_LOW_HALVES, ctrl[0], states);
                let second = _mm512_maskz_multishift_epi64_epi8(KEEP_LOW_HALVES, ctrl[1], states);
                let sum0 = _mm512_dpbusd_epi32(zero, _mm512_mullo_epi32(first, mul), ones);
                let sum1 = _mm512_dpbusd_epi32(zero, _mm512_mullo_epi32(second, mul), ones);
                *acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum0), a0, *acc);
                *acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum1), a1, *acc);
            }
        }
        *out = [0.0; 16];
        let mut lanes = [0.0f32; 16];
        for (c, acc) in acc.into_iter().enumerate() {
            store_lanes(&mut lanes, acc);
            for (q, lane) in lanes.iter().enumerate() {
                out[c + 8 * ((q >> 2) & 1)] += lane;
            }
        }
    }
}
