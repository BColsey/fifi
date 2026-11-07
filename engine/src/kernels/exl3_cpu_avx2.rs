#![allow(unsafe_code)]
//! AVX2 + FMA column-tile kernel for mul1 EXL3 matrices of every stored bit rate.
//!
//! Scope: the intrinsics half of the second `exl3_cpu` tier, for CPUs without AVX-512 VBMI/VNNI.
//! It computes, for one 16-column tile column of a stored matrix, `acc[col] = sum_k a_k * S(k, col)`,
//! where `S` is the byte sum of the mul1 state product (`state * 0x83DCD12D`) and `a_k` the
//! rotated activation. The caller turns `acc` into the projection with the fp16 affine constants
//! of the codebook. The safe scalar reference in `exl3_cpu` computes the same sums.
//!
//! Per 8 weights (one ymm): a 16-byte window of the stored tile is broadcast to both 128-bit
//! halves, one `vpshufb` gathers three ring bytes per dword (the stored little-endian words
//! hold the ring most significant bit first, so the byte order of the ring is folded into the
//! index table), `vpsrlvd` by `8 - bit offset` and a 16-bit mask cut the trellis state out, and a
//! dword multiply by the mul1 constant (`vpmulld`, a wrapping 32-bit multiply like the scalar
//! `wrapping_mul`) followed by `vpmaddubsw` and `vpmaddwd` against ones gives the byte sum `S` of
//! each state in an i32 lane. It is converted to f32 and fused-multiply-added
//! with the activation of that lane's row. Activations stay f32, so, as in the AVX-512 tier, the
//! only departure from the device kernel is that the weight is not rounded to fp16. The decode
//! (`decode_sums`, window to byte sums) does not touch an activation and the FMA step
//! (`accumulate`) takes one activation row, so a mode that serves several rows per weight pass
//! can decode a group once and call `accumulate` per row.
//!
//! Every stored rate (1 to 8 bits, including the half-bit rates) runs on this one path. Eight
//! positions span at most 12 ring bytes, which are at most four stored words, so one 16-byte
//! window holds them. A group of eight whose ring bytes wrap past the end of the tile reads a
//! second window from the other end and merges the two shuffles with an OR; the tables mark those
//! groups. No rate is routed to the scalar reference.
//!
//! Lane layout. Positions in a tile are numbered `p = 8 h + j` (group `h`, lane `j`) and map to
//! `row = (p&1) + 8 (p>>1 &1) + 2 (p>>3 &1) + 4 (p>>4 &1)`, `col = (p>>5) + 8 (p>>2 &1)`. One
//! accumulator therefore serves column `c` (lanes 0 to 3) and `c + 8` (lanes 4 to 7) for the
//! groups `4c .. 4c+3`; the activation table `a` holds 32 floats per k tile, and group `h` reads
//! the eight starting at `8 (h & 3)`. The final reduction over the lanes of each accumulator is a
//! fixed shuffle-add sequence, so a result never depends on the thread that computed it.
//!
//! Unsafe scope: three one-line wrappers around unaligned vector loads of arrays whose length the
//! type fixes, and the safe-to-unsafe call into the target-feature kernel behind the runtime
//! feature check. Confined to this file by the `unsafe_is_confined_to_the_avx2_module` test.
//! On other targets this module is empty and `exl3_cpu` uses the scalar reference.

#[cfg(target_arch = "x86_64")]
pub(crate) use imp::{column_tile, supported};

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::*;
    use std::sync::OnceLock;
    use std::sync::atomic::AtomicU32;

    const MUL1: i32 = 0x83dc_d12d_u32 as i32;
    /// Positions one vector covers.
    const LANES: usize = 8;
    /// Vectors per 16 x 16 tile.
    const GROUPS: usize = 256 / LANES;
    /// Bytes of the stored tile one shuffle reads.
    const WINDOW: usize = 16;
    /// `vpshufb` index that produces a zero byte.
    const ZERO: u8 = 0x80;

    pub(crate) fn supported() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    /// Gather and shift controls of one group of eight positions.
    #[repr(C, align(32))]
    struct Group {
        /// Shuffle indices for the first and the second window (the second is all zero bytes
        /// unless `two_windows`).
        idx: [[u8; 32]; 2],
        /// The right shift of each dword, `8 - bit offset` of the state in its three ring bytes.
        shift: [u8; 32],
        /// Tile offsets of the two windows.
        base: [usize; 2],
        two_windows: bool,
    }

    struct Tables {
        groups: [Group; GROUPS],
    }

    /// Byte of the stored tile that holds ring byte `m`: the stored words are little-endian u32
    /// while the ring is read most significant bit first.
    fn stored_index(ring_byte: usize, tile_bytes: usize) -> usize {
        let m = ring_byte % tile_bytes;
        4 * (m / 4) + 3 - m % 4
    }

    /// Source bytes and shifts for group `group` of a tile of `16 * bits_twice` bytes.
    fn build_group(group: usize, bits_twice: usize) -> Group {
        let tile_bytes = 16 * bits_twice;
        let ring_bits = 8 * tile_bytes;
        let start = |p: usize| ((p + 1) * bits_twice / 2 + ring_bits - 16) % ring_bits;
        // (stored byte, destination byte of the shuffle output)
        let mut wanted = Vec::new();
        let mut shift = [0u8; 32];
        for lane in 0..LANES {
            let s = start(LANES * group + lane);
            let (ring, offset) = (s / 8, s % 8);
            // Dword bytes, least significant first: ring bytes +2, +1, +0 and a zero byte, so
            // the dword is the 24-bit big-endian value of the three ring bytes. A state at a
            // whole-byte offset ends in the second byte, so the third is not needed there.
            wanted.push((stored_index(ring, tile_bytes), 4 * lane + 2));
            wanted.push((stored_index(ring + 1, tile_bytes), 4 * lane + 1));
            if offset > 0 {
                wanted.push((stored_index(ring + 2, tile_bytes), 4 * lane));
            }
            shift[4 * lane] = (8 - offset) as u8;
        }
        let mut sources: Vec<usize> = wanted.iter().map(|w| w.0).collect();
        sources.sort_unstable();
        sources.dedup();
        let fits = |part: &[usize]| part[part.len() - 1] - part[0] < WINDOW;
        let clamp = |first: usize| first.min(tile_bytes - WINDOW);
        let mut base = [clamp(sources[0]); 2];
        let mut two_windows = false;
        if !fits(&sources) {
            // The ring wraps inside this group: the bytes form a cluster at the end of the tile
            // and one at its start, split at the largest gap between neighbours.
            let split = (1..sources.len())
                .max_by_key(|&i| sources[i] - sources[i - 1])
                .expect("a group reads several bytes");
            let (low, high) = sources.split_at(split);
            assert!(
                fits(low) && fits(high),
                "group bytes need more than two windows"
            );
            base = [clamp(high[0]), clamp(low[0])];
            two_windows = true;
        }
        let mut idx = [[ZERO; 32]; 2];
        for (stored, out) in wanted {
            let window = (0..2)
                .find(|&w| (base[w]..base[w] + WINDOW).contains(&stored) && (w == 0 || two_windows))
                .expect("byte outside the windows");
            idx[window][out] = (stored - base[window]) as u8;
        }
        Group {
            idx,
            shift,
            base,
            two_windows,
        }
    }

    fn build_tables(bits_twice: usize) -> Tables {
        Tables {
            groups: std::array::from_fn(|group| build_group(group, bits_twice)),
        }
    }

    fn tables(bits_twice: usize) -> &'static Tables {
        static TABLES: [OnceLock<Tables>; 17] = [const { OnceLock::new() }; 17];
        TABLES[bits_twice].get_or_init(|| build_tables(bits_twice))
    }

    #[inline]
    fn load_window(bytes: &[u8; WINDOW]) -> __m128i {
        // SAFETY: the reference is a valid 16-byte region and the load is unaligned.
        unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    fn load_bytes(bytes: &[u8; 32]) -> __m256i {
        // SAFETY: the reference is a valid 32-byte region and the load is unaligned.
        unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    fn load_activations(a: &[AtomicU32; LANES]) -> __m256 {
        // SAFETY: 8 contiguous words are valid for reads; no thread writes the activation
        // SAFETY: table while a projection phase runs (the pool joins between phases), so the
        // SAFETY: plain vector load cannot race an atomic store, and `AtomicU32` has the layout
        // SAFETY: of `u32`, whose bit pattern is the stored f32.
        unsafe { _mm256_loadu_ps(a.as_ptr().cast::<f32>()) }
    }

    fn window(tile: &[u8], base: usize) -> &[u8; WINDOW] {
        tile[base..base + WINDOW].try_into().unwrap()
    }

    /// Sum of the four lanes of `v`.
    #[inline]
    #[target_feature(enable = "sse")]
    fn sum_lanes(v: __m128) -> f32 {
        let pairs = _mm_add_ps(v, _mm_movehl_ps(v, v));
        _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps::<1>(pairs, pairs)))
    }

    /// Constants of the byte-sum step.
    #[derive(Clone, Copy)]
    struct Consts {
        mul: __m256i,
        ones_bytes: __m256i,
        ones_words: __m256i,
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn consts() -> Consts {
        Consts {
            mul: _mm256_set1_epi32(MUL1),
            ones_bytes: _mm256_set1_epi8(1),
            ones_words: _mm256_set1_epi16(1),
        }
    }

    /// The 16-bit trellis states of the eight positions of `group`, one per dword.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn states(tile: &[u8], group: &Group) -> __m256i {
        let first = _mm256_broadcastsi128_si256(load_window(window(tile, group.base[0])));
        let mut bytes = _mm256_shuffle_epi8(first, load_bytes(&group.idx[0]));
        if group.two_windows {
            let second = _mm256_broadcastsi128_si256(load_window(window(tile, group.base[1])));
            bytes = _mm256_or_si256(
                bytes,
                _mm256_shuffle_epi8(second, load_bytes(&group.idx[1])),
            );
        }
        let shifted = _mm256_srlv_epi32(bytes, load_bytes(&group.shift));
        _mm256_and_si256(shifted, _mm256_set1_epi32(0xffff))
    }

    /// The weight decode of one group: `S`, the byte sum of `state * 0x83DCD12D`, of each of the
    /// eight positions as an i32 lane. It does not involve an activation, so a caller that serves
    /// several activation rows decodes a group once and applies [`accumulate`] per row.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn decode_sums(tile: &[u8], group: &Group, k: Consts) -> __m256i {
        let product = _mm256_mullo_epi32(states(tile, group), k.mul);
        _mm256_madd_epi16(_mm256_maddubs_epi16(product, k.ones_bytes), k.ones_words)
    }

    /// Adds `S * a` of the eight positions to the accumulator, one activation row.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn accumulate(acc: __m256, sums: __m256i, activation: __m256) -> __m256 {
        _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums), activation, acc)
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
        assert!(supported(), "AVX2 and FMA are required");
        assert!((2..=16).contains(&bits_twice), "unsupported tile size");
        // SAFETY: the required target features were detected at runtime just above.
        unsafe { column_tile_inner(trellis, bits_twice, shape, nt, a, out) }
    }

    #[target_feature(enable = "avx2,fma")]
    fn column_tile_inner(
        trellis: &[u8],
        bits_twice: usize,
        (n_tiles, k_tiles): (usize, usize),
        nt: usize,
        a: &[AtomicU32],
        out: &mut [f32; 16],
    ) {
        let t = tables(bits_twice);
        let tile_bytes = 16 * bits_twice;
        let k = consts();
        let mut acc = [_mm256_setzero_ps(); 8];
        for kt in 0..k_tiles {
            let at = (kt * n_tiles + nt) * tile_bytes;
            let tile = &trellis[at..at + tile_bytes];
            let av = &a[kt * 32..kt * 32 + 32];
            let acts = [
                load_activations(av[..8].try_into().unwrap()),
                load_activations(av[8..16].try_into().unwrap()),
                load_activations(av[16..24].try_into().unwrap()),
                load_activations(av[24..].try_into().unwrap()),
            ];
            for (c, acc) in acc.iter_mut().enumerate() {
                for (sub, act) in acts.iter().enumerate() {
                    let sums = decode_sums(tile, &t.groups[4 * c + sub], k);
                    *acc = accumulate(*acc, sums, *act);
                }
            }
        }
        for (c, acc) in acc.into_iter().enumerate() {
            out[c] = sum_lanes(_mm256_castps256_ps128(acc));
            out[c + 8] = sum_lanes(_mm256_extractf128_ps::<1>(acc));
        }
    }
}
