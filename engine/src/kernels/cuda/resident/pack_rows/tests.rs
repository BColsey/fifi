//! Tests of the pack kernel of a K-row verify window (unit test 10 of the K-row design): the
//! `[K][ld]` rows come out byte for byte, for the two layouts the step uses (z of a DeltaNet layer
//! beside the beta/alpha columns, q/k/v of an attention layer), with NaN payloads, signed zeros and
//! denormals among the data; the columns no segment covers are not written; every refusal is a
//! `BadRequest` before any native call. A host without a device skips them.

use super::{PackRowsCall, PackSegment};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

const SENTINEL: u32 = 0x5a5a_a5a5;
/// A DeltaNet layer's projection row: z (4096) at 8192, beta (32) at 12288, alpha (32) at 12320.
const GDN_LD: usize = 12_352;
const Z_OFF: usize = 8_192;
const Z_N: usize = 4_096;
/// An attention layer's projection row: q (16 heads x 512, query and gate), k and v (512 each).
const ATTN_LD: usize = 9_216;
const Q_N: usize = 8_192;
const KV_N: usize = 512;

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        Self { ctx, stream }
    }

    fn upload_words(&self, words: &[u32]) -> DeviceBuffer {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        testkit::upload_bytes(&self.ctx, &bytes)
    }

    fn words(&self, buffer: &DeviceBuffer, count: usize) -> Vec<u32> {
        testkit::read_bytes(&self.ctx, buffer, count * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }
}

/// Words that a float instruction could change: NaNs with payloads (quiet and signalling),
/// infinities, negative zero, denormals, the extremes; the rest random bits.
fn awkward_words(rng: &mut Rng, count: usize) -> Vec<u32> {
    const SPECIAL: [u32; 10] = [
        0x7fc0_dead,
        0xffff_ffff,
        0x7f80_0001,
        0x7f80_0000,
        0xff80_0000,
        0x8000_0000,
        0x0000_0001,
        0x807f_ffff,
        0x7f7f_ffff,
        0x0000_0000,
    ];
    (0..count)
        .map(|i| {
            if i % 5 == 0 {
                SPECIAL[(i / 5) % SPECIAL.len()]
            } else {
                rng.next_u64() as u32
            }
        })
        .collect()
}

/// The `[m][ld]` destination the pack must produce: sentinel words, with each segment's `[m][n]`
/// source copied to its columns.
fn expected(m: usize, ld: usize, segments: &[(Vec<u32>, usize, usize)]) -> Vec<u32> {
    let mut dst = vec![SENTINEL; (m - 1) * ld + segments.iter().map(|s| s.2 + s.1).max().unwrap()];
    for (src, n, off) in segments {
        for row in 0..m {
            dst[row * ld + off..row * ld + off + n].copy_from_slice(&src[row * n..(row + 1) * n]);
        }
    }
    dst
}

/// Runs one pack and compares the whole destination with `expected`, sentinels included.
fn check(rig: &Rig, rng: &mut Rng, m: usize, ld: usize, layout: &[(usize, usize)], skew: usize) {
    let sources: Vec<(Vec<u32>, usize, usize)> = layout
        .iter()
        .map(|&(n, off)| (awkward_words(rng, m * n), n, off))
        .collect();
    let want = expected(m, ld, &sources);
    let buffers: Vec<DeviceBuffer> = sources.iter().map(|s| rig.upload_words(&s.0)).collect();
    // The destination starts `skew` words into its buffer, to run unaligned-to-16 addresses.
    let mut initial = vec![SENTINEL; skew];
    initial.extend(vec![SENTINEL; want.len()]);
    let dst = rig.upload_words(&initial);
    let segments: Vec<PackSegment<'_>> = sources
        .iter()
        .zip(&buffers)
        .map(|((src, n, off), buffer)| PackSegment {
            src: buffer.range(0, src.len() * 4).unwrap(),
            n: *n,
            dst_off: *off,
        })
        .collect();
    rig.ctx
        .pack_rows(
            &rig.stream,
            PackRowsCall {
                m,
                ld,
                dst: dst.range(skew * 4, want.len() * 4).unwrap(),
                segments,
            },
        )
        .expect("pack_rows");
    let got = rig.words(&dst, skew + want.len());
    assert!(
        got[..skew].iter().all(|w| *w == SENTINEL),
        "words before the destination changed"
    );
    assert_eq!(
        &got[skew..],
        &want[..],
        "m {m} ld {ld} layout {layout:?} skew {skew}"
    );
}

/// Unit test 10: the bytes of the `[K][ld]` rows, for K = 1 to 8 and 16.
#[test]
fn the_pack_writes_the_rows_byte_for_byte() {
    if !testkit::device_present("pack_rows bytes") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4b52_0010);
    for m in [1usize, 2, 3, 4, 5, 6, 7, 8, 16] {
        // z of a DeltaNet layer; the beta/alpha columns (12288..12352) stay sentinels.
        check(&rig, &mut rng, m, GDN_LD, &[(Z_N, Z_OFF)], 0);
        // q, k and v of an attention layer in one launch.
        check(
            &rig,
            &mut rng,
            m,
            ATTN_LD,
            &[(Q_N, 0), (KV_N, Q_N), (KV_N, Q_N + KV_N)],
            0,
        );
    }
    // Segments listed in another order than their columns, an odd width, a destination that is
    // not 16-byte aligned, and a stride equal to the span (a tight copy).
    check(
        &rig,
        &mut rng,
        4,
        ATTN_LD,
        &[(KV_N, Q_N + KV_N), (Q_N, 0), (KV_N, Q_N)],
        1,
    );
    check(&rig, &mut rng, 3, 1_000, &[(333, 100), (7, 700)], 3);
    check(&rig, &mut rng, 2, 512, &[(512, 0)], 0);
}

fn refusal(result: Result<(), EngError>, needle: &str, case: &str) {
    match result {
        Err(EngError::BadRequest(message)) if message.contains(needle) => {}
        other => panic!("{case}: expected BadRequest containing {needle:?}, got {other:?}"),
    }
}

#[test]
fn the_wrapper_refuses_bad_arguments_before_any_native_call() {
    if !testkit::device_present("pack_rows refusals") {
        return;
    }
    let rig = Rig::new();
    let src = rig.upload_words(&vec![1; 16 * 64]);
    let dst = rig.upload_words(&vec![SENTINEL; 16 * 256]);
    let call = |m: usize, ld: usize, segs: &[(usize, usize)]| PackRowsCall {
        m,
        ld,
        dst: dst.range(0, 16 * 256 * 4).unwrap(),
        segments: segs
            .iter()
            .map(|&(n, off)| PackSegment {
                src: src.range(0, 16 * 64 * 4).unwrap(),
                n,
                dst_off: off,
            })
            .collect(),
    };
    let run = |c: PackRowsCall<'_>| rig.ctx.pack_rows(&rig.stream, c);
    refusal(run(call(0, 256, &[(64, 0)])), "1 to 16 rows", "m = 0");
    refusal(run(call(17, 256, &[(64, 0)])), "1 to 16 rows", "m = 17");
    refusal(run(call(2, 256, &[])), "1 to 3 segments", "no segment");
    refusal(
        run(call(2, 256, &[(8, 0), (8, 8), (8, 16), (8, 24)])),
        "1 to 3 segments",
        "four segments",
    );
    refusal(
        run(call(2, 256, &[(0, 0)])),
        "1 to 65536 columns",
        "empty segment",
    );
    refusal(run(call(2, 100, &[(64, 40)])), "past ld", "segment past ld");
    refusal(
        run(call(2, 256, &[(64, 0), (64, 32)])),
        "overlap in columns",
        "overlapping columns",
    );
    // A source that overlaps the destination.
    let aliased = PackRowsCall {
        m: 2,
        ld: 64,
        dst: src.range(0, 2 * 64 * 4).unwrap(),
        segments: vec![PackSegment {
            src: src.range(0, 2 * 32 * 4).unwrap(),
            n: 32,
            dst_off: 0,
        }],
    };
    refusal(run(aliased), "overlap", "source over destination");
    // A source that is too short for m rows of n columns.
    let short = PackRowsCall {
        m: 4,
        ld: 256,
        dst: dst.range(0, 4 * 256 * 4).unwrap(),
        segments: vec![PackSegment {
            src: src.range(0, 3 * 64 * 4).unwrap(),
            n: 64,
            dst_off: 0,
        }],
    };
    refusal(run(short), "src", "short source");
    // A destination at a 2-byte offset is not 4-byte aligned.
    let odd = PackRowsCall {
        m: 1,
        ld: 64,
        dst: dst.range(2, 64 * 4).unwrap(),
        segments: vec![PackSegment {
            src: src.range(0, 64 * 4).unwrap(),
            n: 64,
            dst_off: 0,
        }],
    };
    refusal(run(odd), "dst", "misaligned destination");
}
