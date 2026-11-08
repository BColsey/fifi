//! The single-row gather and wide paths against the legacy single-call kernel. They run the same
//! decode, the same fused multiply-add per lane and the same reduction order, so every byte the
//! call writes (the rotated input, both partial banks, the output and the sticky flag) must be
//! equal. A host without a device skips these tests. The timing test is ignored: run it from a
//! release test binary with `--ignored --nocapture`.

use super::linear_tests::{fixture, upload_scales};
use super::{Exl3IndexedCall, Exl3LinearCall, IndexedOptions};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, testkit};
use crate::weights::exl3::Codebook;

#[derive(Clone, Copy, Debug)]
enum Path {
    /// `exl3_linear_mode`: the shuffle single-call kernel.
    Legacy,
    /// `exl3_grouped(.., gather = true)` with the call as its only group.
    Gather,
    /// `exl3_grouped_wide` with the call as its only group, `cols` tiles per block.
    Wide(u32),
    /// `exl3_grouped_wide` with one tile per block and the context opted into the compile-time
    /// rate 10 and rate 6 body of book 2, which every other group falls back from.
    WideOdd,
    /// [`Path::WideOdd`] with the DP4A decode of book 2 (the DP4A option on the context).
    WideOddDp4a,
}

struct Rig {
    trellis: DeviceBuffer,
    trellis_offset: usize,
    suh: DeviceBuffer,
    svh: DeviceBuffer,
    bias: Option<DeviceBuffer>,
    x: DeviceBuffer,
    xh: DeviceBuffer,
    z: DeviceBuffer,
    out: DeviceBuffer,
    flag: DeviceBuffer,
    shape: (usize, usize, usize),
    bits_twice: u8,
    book: Codebook,
    fused: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct Written {
    xh: Vec<u8>,
    z: Vec<u8>,
    out: Vec<u8>,
    flag: Vec<u8>,
}

impl Rig {
    fn new(ctx: &CudaContext, case: &Case, x: &[f32]) -> Self {
        Self::with_trellis_offset(ctx, case, x, 0)
    }

    /// The same rig with the trellis `offset` bytes into its buffer (2 is the only alignment the
    /// host admits below the usual 16), so the kernels' unaligned word loads run.
    fn with_trellis_offset(ctx: &CudaContext, case: &Case, x: &[f32], offset: usize) -> Self {
        let Case {
            book,
            bits_twice,
            shape: (k, n),
            rows,
            bias,
            fused,
        } = *case;
        let matrix = fixture(false, book, bits_twice, (k, n));
        let bias = bias.then(|| {
            let values: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect();
            testkit::upload_f32(ctx, &values)
        });
        Self {
            trellis: {
                let mut bytes = vec![0xA5; offset];
                bytes.extend_from_slice(matrix.trellis().host_bytes().unwrap());
                testkit::upload_bytes(ctx, &bytes)
            },
            trellis_offset: offset,
            suh: upload_scales(ctx, matrix.suh()),
            svh: upload_scales(ctx, matrix.svh()),
            bias,
            x: testkit::upload_f32(ctx, x),
            xh: testkit::upload_filled(ctx, 0xff, rows * k * 4),
            z: testkit::upload_filled(ctx, 0xff, rows * n * 4),
            out: testkit::upload_filled(ctx, 0xff, rows * n * 4),
            flag: testkit::upload_bytes(ctx, &0u32.to_le_bytes()),
            shape: (rows, k, n),
            bits_twice,
            book,
            fused,
        }
    }

    fn call(&self) -> Exl3LinearCall<'_> {
        let (rows, k, n) = self.shape;
        Exl3LinearCall {
            trellis: self
                .trellis
                .range(
                    self.trellis_offset,
                    self.trellis.bytes() - self.trellis_offset,
                )
                .unwrap(),
            suh: self.suh.range(0, k * 4).unwrap(),
            svh: self.svh.range(0, n * 4).unwrap(),
            bias: self.bias.as_ref().map(|b| b.range(0, n * 4).unwrap()),
            x: self.x.range(0, rows * k * 4).unwrap(),
            xh: self.xh.range(0, rows * k * 4).unwrap(),
            z: self.z.range(0, rows * n * 4).unwrap(),
            out: self.out.range(0, rows * n * 4).unwrap(),
            flag: self.flag.range(0, 4).unwrap(),
            rows,
            split_k: 2,
            k,
            n,
            bits_twice: self.bits_twice,
            book: self.book,
            fused: self.fused,
        }
    }

    fn run(&self, ctx: &CudaContext, stream: &DeviceStream, path: Path) {
        let call = self.call();
        match path {
            Path::Legacy => ctx.exl3_linear_mode(stream, call, false),
            Path::Gather => ctx.exl3_grouped(stream, std::slice::from_ref(&call), true),
            Path::Wide(cols) => ctx.exl3_grouped_wide(stream, std::slice::from_ref(&call), cols),
            Path::WideOdd | Path::WideOddDp4a => {
                ctx.set_odd_rate_kernel(true);
                if matches!(path, Path::WideOddDp4a) {
                    ctx.set_indexed_options(Some(IndexedOptions::DP4A));
                }
                let done = ctx.exl3_grouped_wide(stream, std::slice::from_ref(&call), 1);
                ctx.set_indexed_options(None);
                ctx.set_odd_rate_kernel(false);
                // The native launch reports the kernel it enqueued: the compile-time body only for
                // book 2 at rate 10 or 6 with one row, the unchanged kernels otherwise.
                let (rows, ..) = self.shape;
                let qualifies =
                    self.book == Codebook::Mul1 && matches!(self.bits_twice, 10 | 6) && rows == 1;
                let want = match (qualifies, matches!(path, Path::WideOddDp4a)) {
                    (false, _) => 2,
                    (true, false) => 3,
                    (true, true) => 4,
                };
                if done.is_ok() {
                    assert_eq!(
                        CudaContext::test_last_odd_variant(),
                        want,
                        "{path:?}: {:?} rate {} rows {rows}",
                        self.book,
                        self.bits_twice
                    );
                }
                done
            }
        }
        .unwrap();
    }

    fn written(&self, ctx: &CudaContext, stream: &DeviceStream) -> Written {
        stream.synchronize().unwrap();
        let (rows, k, n) = self.shape;
        Written {
            xh: testkit::read_bytes(ctx, &self.xh, rows * k * 4),
            z: testkit::read_bytes(ctx, &self.z, rows * n * 4),
            out: testkit::read_bytes(ctx, &self.out, rows * n * 4),
            flag: testkit::read_bytes(ctx, &self.flag, 4),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    book: Codebook,
    bits_twice: u8,
    shape: (usize, usize),
    rows: usize,
    bias: bool,
    fused: bool,
}

fn rates(book: Codebook) -> Vec<u8> {
    (2u8..=16)
        .filter(|bits| bits % 2 == 0 || (book == Codebook::Mul1 && matches!(bits, 3 | 5 | 7)))
        .collect()
}

fn input(rng: &mut testkit::Rng, count: usize, poison: bool) -> Vec<f32> {
    let mut x: Vec<f32> = (0..count)
        .map(|_| (rng.unit() as f32 - 0.5) * 8.0)
        .collect();
    if poison {
        x[count / 2] = f32::INFINITY;
    }
    x
}

#[test]
fn single_row_gather_and_wide_are_bit_identical_to_the_legacy_kernel() {
    if !testkit::device_present("EXL3 single-row gather vs legacy") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0001);
    let mut checked = 0;
    for book in [Codebook::ThreeInst, Codebook::Mcg, Codebook::Mul1] {
        for bits_twice in rates(book) {
            for (shape, rows) in [
                ((256, 128), 1),
                ((512, 256), 8),
                ((256, 384), 3),
                ((4096, 128), 2),
            ] {
                let case = Case {
                    book,
                    bits_twice,
                    shape,
                    rows,
                    bias: bits_twice % 4 == 0,
                    fused: bits_twice % 6 == 0,
                };
                for path in [
                    Path::Gather,
                    Path::Wide(1),
                    Path::WideOdd,
                    Path::WideOddDp4a,
                    Path::Wide(2),
                    Path::Wide(4),
                    Path::Wide(8),
                ] {
                    compare(&ctx, &stream, &case, path, &mut rng, false);
                    checked += 1;
                }
            }
        }
    }
    // Every row count 1..=8, with a nonfinite input: the flag and the zeroed value agree too.
    for rows in 1..=8 {
        let case = Case {
            book: Codebook::ThreeInst,
            bits_twice: 8,
            shape: (2048, 512),
            rows,
            bias: true,
            fused: rows % 2 == 0,
        };
        for path in [
            Path::Gather,
            Path::Wide(1),
            Path::WideOdd,
            Path::WideOddDp4a,
            Path::Wide(4),
        ] {
            compare(&ctx, &stream, &case, path, &mut rng, rows == 5);
            checked += 1;
        }
    }
    // Warps with an uneven or zero share of the k tiles in the one-tile body: k = 128 leaves warps
    // 4 to 7 no tile in either split, k = 384 gives warps 0 to 3 two tiles and the others one,
    // k = 640 three and two.
    for (book, bits_twice) in [
        (Codebook::Mcg, 8u8),
        (Codebook::Mcg, 12),
        (Codebook::ThreeInst, 8),
        (Codebook::Mul1, 10),
        (Codebook::Mul1, 6),
    ] {
        for k in [128usize, 384, 640] {
            let case = Case {
                book,
                bits_twice,
                shape: (k, 256),
                rows: 1,
                bias: false,
                fused: false,
            };
            for path in [
                Path::Gather,
                Path::Wide(1),
                Path::WideOdd,
                Path::WideOddDp4a,
            ] {
                compare(&ctx, &stream, &case, path, &mut rng, false);
                checked += 1;
            }
        }
    }
    // The real head: 6.0 bpw, 2048 -> 248320, one row.
    let head = Case {
        book: Codebook::ThreeInst,
        bits_twice: 12,
        shape: (2048, 248_320),
        rows: 1,
        bias: false,
        fused: false,
    };
    for path in [Path::Gather, Path::Wide(1), Path::Wide(2)] {
        compare(&ctx, &stream, &head, path, &mut rng, false);
        checked += 1;
    }
    println!("single-row gather and wide: {checked} cases bit identical");
}

fn compare(
    ctx: &CudaContext,
    stream: &DeviceStream,
    case: &Case,
    path: Path,
    rng: &mut testkit::Rng,
    poison: bool,
) {
    compare_at(ctx, stream, case, path, (rng, poison), 0);
}

fn compare_at(
    ctx: &CudaContext,
    stream: &DeviceStream,
    case: &Case,
    path: Path,
    (rng, poison): (&mut testkit::Rng, bool),
    offset: usize,
) {
    let (k, _) = case.shape;
    let x = input(rng, case.rows * k, poison);
    let legacy = Rig::with_trellis_offset(ctx, case, &x, offset);
    legacy.run(ctx, stream, Path::Legacy);
    let gather = Rig::with_trellis_offset(ctx, case, &x, offset);
    gather.run(ctx, stream, path);
    let (a, b) = (legacy.written(ctx, stream), gather.written(ctx, stream));
    assert_eq!(
        a, b,
        "{case:?} {path:?} poison {poison} trellis offset {offset}"
    );
    assert_eq!(a.flag != [0; 4], poison, "{case:?}");
}

/// A trellis that starts 2 bytes into its allocation (the host admits that alignment) is read
/// through the byte-assembled word loads: forcing the 32-bit path would fault or read wrong words.
#[test]
fn an_unaligned_trellis_gives_the_bytes_of_the_shuffle_kernel() {
    if !testkit::device_present("EXL3 unaligned trellis") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0005);
    for (book, bits_twice, rows) in [
        (Codebook::Mcg, 8u8, 1usize),
        (Codebook::ThreeInst, 8, 1),
        (Codebook::Mcg, 12, 1),
        (Codebook::Mul1, 5, 1),
        (Codebook::Mul1, 10, 1),
        (Codebook::Mul1, 6, 1),
        // Several rows: the rows kernel reads the same unaligned words.
        (Codebook::Mcg, 8, 3),
        (Codebook::ThreeInst, 8, 8),
        (Codebook::Mcg, 12, 2),
    ] {
        let case = Case {
            book,
            bits_twice,
            shape: (512, 256),
            rows,
            bias: false,
            fused: false,
        };
        for path in [Path::Gather, Path::Wide(1), Path::WideOdd, Path::Wide(4)] {
            compare_at(&ctx, &stream, &case, path, (&mut rng, false), 2);
        }
    }
}

/// Several groups in one launch, as the MoE block issues them: each group has its own weights
/// (rate and codebook), input and rows 1; the wide kernel must write the bytes the gather kernel
/// writes.
#[test]
fn grouped_wide_matches_grouped_gather_across_mixed_groups() {
    if !testkit::device_present("EXL3 grouped wide vs gather") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0003);
    let (k, n) = (512usize, 256usize);
    let kinds = [
        (Codebook::ThreeInst, 8u8),
        (Codebook::ThreeInst, 8),
        (Codebook::ThreeInst, 12),
        (Codebook::Mcg, 8),
        (Codebook::Mul1, 5),
        (Codebook::ThreeInst, 6),
    ];
    let groups = kinds.len();
    let rigs: Vec<Rig> = kinds
        .iter()
        .map(|&(book, bits_twice)| {
            let case = Case {
                book,
                bits_twice,
                shape: (k, n),
                rows: 1,
                bias: false,
                fused: false,
            };
            Rig::new(&ctx, &case, &input(&mut rng, k, false))
        })
        .collect();
    let run = |wide: Option<u32>| {
        let xh = testkit::upload_filled(&ctx, 0xff, groups * k * 4);
        let z = testkit::upload_filled(&ctx, 0xff, groups * n * 4);
        let out = testkit::upload_filled(&ctx, 0xff, groups * n * 4);
        let flag = testkit::upload_bytes(&ctx, &0u32.to_le_bytes());
        let calls: Vec<Exl3LinearCall<'_>> = rigs
            .iter()
            .enumerate()
            .map(|(g, rig)| Exl3LinearCall {
                xh: xh.range(g * k * 4, k * 4).unwrap(),
                z: z.range(g * n * 4, n * 4).unwrap(),
                out: out.range(g * n * 4, n * 4).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                ..rig.call()
            })
            .collect();
        match wide {
            None => ctx.exl3_grouped(&stream, &calls, true),
            Some(cols) => ctx.exl3_grouped_wide(&stream, &calls, cols),
        }
        .unwrap();
        stream.synchronize().unwrap();
        (
            testkit::read_bytes(&ctx, &xh, groups * k * 4),
            testkit::read_bytes(&ctx, &z, groups * n * 4),
            testkit::read_bytes(&ctx, &out, groups * n * 4),
            testkit::read_bytes(&ctx, &flag, 4),
        )
    };
    let expected = run(None);
    for cols in [1, 2, 4, 8] {
        assert_eq!(run(Some(cols)), expected, "wide {cols}");
    }
}

/// Groups of book 2 at rates 10 and 6 in one launch take the compile-time body when the context
/// opts in, and write the bytes of the gather launch; one group of another book or rate sends
/// the same launch to the unchanged wide kernel, which writes them too.
#[test]
fn the_odd_rate_body_matches_the_gather_launch_in_mixed_groups() {
    if !testkit::device_present("EXL3 odd-rate body vs gather") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0007);
    let (k, n) = (640usize, 384usize);
    for kinds in [
        vec![
            (Codebook::Mul1, 10u8),
            (Codebook::Mul1, 6),
            (Codebook::Mul1, 10),
        ],
        vec![
            (Codebook::Mul1, 10),
            (Codebook::Mcg, 8),
            (Codebook::Mul1, 6),
        ],
        vec![
            (Codebook::Mul1, 10),
            (Codebook::Mul1, 8),
            (Codebook::Mul1, 6),
        ],
    ] {
        let groups = kinds.len();
        let rigs: Vec<Rig> = kinds
            .iter()
            .map(|&(book, bits_twice)| {
                let case = Case {
                    book,
                    bits_twice,
                    shape: (k, n),
                    rows: 1,
                    bias: false,
                    fused: false,
                };
                Rig::new(&ctx, &case, &input(&mut rng, k, false))
            })
            .collect();
        let run = |mode: u8| {
            let xh = testkit::upload_filled(&ctx, 0xff, groups * k * 4);
            let z = testkit::upload_filled(&ctx, 0xff, groups * n * 4);
            let out = testkit::upload_filled(&ctx, 0xff, groups * n * 4);
            let flag = testkit::upload_bytes(&ctx, &0u32.to_le_bytes());
            let calls: Vec<Exl3LinearCall<'_>> = rigs
                .iter()
                .enumerate()
                .map(|(g, rig)| Exl3LinearCall {
                    xh: xh.range(g * k * 4, k * 4).unwrap(),
                    z: z.range(g * n * 4, n * 4).unwrap(),
                    out: out.range(g * n * 4, n * 4).unwrap(),
                    flag: flag.range(0, 4).unwrap(),
                    ..rig.call()
                })
                .collect();
            if mode > 0 {
                ctx.set_odd_rate_kernel(true);
                if mode == 2 {
                    ctx.set_indexed_options(Some(IndexedOptions::DP4A));
                }
                ctx.exl3_grouped_wide(&stream, &calls, 1).unwrap();
                ctx.set_indexed_options(None);
                ctx.set_odd_rate_kernel(false);
                // Only a launch whose every group is book 2 at rate 10 or 6 runs the new body.
                let all = kinds
                    .iter()
                    .all(|&(book, bits)| book == Codebook::Mul1 && matches!(bits, 10 | 6));
                let want = if all { 2 + u32::from(mode) } else { 2 };
                assert_eq!(
                    CudaContext::test_last_odd_variant(),
                    want,
                    "{kinds:?} mode {mode}"
                );
            } else {
                ctx.exl3_grouped(&stream, &calls, true).unwrap();
            }
            stream.synchronize().unwrap();
            (
                testkit::read_bytes(&ctx, &xh, groups * k * 4),
                testkit::read_bytes(&ctx, &z, groups * n * 4),
                testkit::read_bytes(&ctx, &out, groups * n * 4),
                testkit::read_bytes(&ctx, &flag, 4),
            )
        };
        let expected = run(0);
        assert_eq!(run(1), expected, "{kinds:?} odd");
        assert_eq!(run(2), expected, "{kinds:?} odd with DP4A");
    }
}

/// Groups of 1 to 8 rows with different books and rates in one launch: the one-tile wide launch
/// (rows kernel when a group has several rows) writes the bytes of the gather launch, which runs
/// one block per row.
#[test]
fn the_rows_kernel_matches_the_gather_launch_across_mixed_row_counts() {
    if !testkit::device_present("EXL3 rows kernel vs gather") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0005);
    let (k, n) = (512usize, 256usize);
    let kinds = [
        (Codebook::ThreeInst, 8u8, 3usize),
        (Codebook::Mcg, 8, 1),
        (Codebook::ThreeInst, 12, 8),
        (Codebook::Mcg, 8, 2),
        (Codebook::Mul1, 5, 5),
        (Codebook::ThreeInst, 6, 4),
        (Codebook::Mcg, 8, 7),
    ];
    let total: usize = kinds.iter().map(|k| k.2).sum();
    let rigs: Vec<Rig> = kinds
        .iter()
        .map(|&(book, bits_twice, rows)| {
            let case = Case {
                book,
                bits_twice,
                shape: (k, n),
                rows,
                bias: bits_twice % 4 == 0,
                fused: false,
            };
            // The input of the 5-row group carries an infinity: the flag and the zeroed value agree.
            Rig::new(&ctx, &case, &input(&mut rng, rows * k, rows == 5))
        })
        .collect();
    let run = |wide: bool, subset: &[usize]| {
        let rows_of: Vec<usize> = subset.iter().map(|&g| kinds[g].2).collect();
        let rows: usize = rows_of.iter().sum();
        let xh = testkit::upload_filled(&ctx, 0xff, rows * k * 4);
        let z = testkit::upload_filled(&ctx, 0xff, rows * n * 4);
        let out = testkit::upload_filled(&ctx, 0xff, rows * n * 4);
        let flag = testkit::upload_bytes(&ctx, &0u32.to_le_bytes());
        let mut first = 0;
        let calls: Vec<Exl3LinearCall<'_>> = subset
            .iter()
            .map(|&g| {
                let r = kinds[g].2;
                let call = Exl3LinearCall {
                    xh: xh.range(first * k * 4, r * k * 4).unwrap(),
                    z: z.range(first * n * 4, r * n * 4).unwrap(),
                    out: out.range(first * n * 4, r * n * 4).unwrap(),
                    flag: flag.range(0, 4).unwrap(),
                    ..rigs[g].call()
                };
                first += r;
                call
            })
            .collect();
        if wide {
            ctx.exl3_grouped_wide(&stream, &calls, 1)
        } else {
            ctx.exl3_grouped(&stream, &calls, true)
        }
        .unwrap();
        stream.synchronize().unwrap();
        (
            testkit::read_bytes(&ctx, &xh, rows * k * 4),
            testkit::read_bytes(&ctx, &z, rows * n * 4),
            testkit::read_bytes(&ctx, &out, rows * n * 4),
            testkit::read_bytes(&ctx, &flag, 4),
        )
    };
    assert!(total <= 48);
    let all: Vec<usize> = (0..kinds.len()).collect();
    assert_eq!(run(true, &all), run(false, &all), "all groups");
    // Row-count bounds of the kernel template (2, 4 and 8) and a single group.
    for subset in [
        vec![1usize, 3],
        vec![0, 1, 3],
        vec![4],
        vec![2],
        vec![6, 0],
        vec![5],
        vec![5, 1],
    ] {
        assert_eq!(run(true, &subset), run(false, &subset), "{subset:?}");
    }
}

/// Words of a rotating pattern that are valid trellis states of any rate.
fn random_bytes(rng: &mut testkit::Rng, count: usize) -> Vec<u8> {
    (0..count).map(|_| rng.next_u64() as u8).collect()
}

/// The id-indexed launch of one decode token's picks against the wide launch of the same members
/// given as ordinary calls: the bytes written must be equal; an id outside the bank skips its pair
/// and sets the flag without touching the other pairs.
#[test]
fn indexed_launch_matches_the_wide_launch_of_the_chosen_members() {
    if !testkit::device_present("EXL3 indexed vs wide") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0004);
    let (k, n, members) = (512usize, 384usize, 6usize);
    // The last two kinds start the bank 2 bytes into its buffer, the alignment the host admits
    // below 4 bytes, so the kernels' byte-assembled word loads run in the indexed launch too.
    for (book, bits_twice, offset) in [
        (Codebook::Mcg, 8u8, 0usize),
        (Codebook::ThreeInst, 8, 0),
        (Codebook::Mcg, 12, 0),
        (Codebook::Mul1, 6, 0),
        // Rates the model does not use take the odd-rate branch of the one-tile body.
        (Codebook::Mul1, 5, 0),
        (Codebook::Mul1, 8, 0),
        (Codebook::Mcg, 8, 2),
        (Codebook::ThreeInst, 12, 2),
    ] {
        let trellis_stride = k * n / 256 * usize::from(bits_twice) * 16;
        let scale_floats = k + n;
        let trellis = testkit::upload_bytes(&ctx, &{
            let mut bytes = vec![0xA5; offset];
            bytes.extend(random_bytes(&mut rng, members * trellis_stride));
            bytes
        });
        let scale_values: Vec<f32> = (0..members * scale_floats)
            .map(|i| 0.4 + (i % 11) as f32 / 16.0)
            .collect();
        let scales = testkit::upload_f32(&ctx, &scale_values);
        let picks: [i32; 8] = [3, 0, 5, 3, 2, 2, 1, 4];
        let pairs = picks.len();
        let x_values = input(&mut rng, pairs * k, false);
        let x = testkit::upload_f32(&ctx, &x_values);
        let ids_of = |ids: &[i32]| {
            testkit::upload_bytes(
                &ctx,
                &ids.iter()
                    .flat_map(|id| id.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        // The same members laid out the way the Flint arena does: one stride per member holding
        // [trellis | suh | svh], so a single byte offset per member addresses all three.
        let arena = (offset == 0).then(|| {
            let host = testkit::read_bytes(&ctx, &trellis, members * trellis_stride);
            let mut bytes = Vec::new();
            for m in 0..members {
                bytes.extend_from_slice(&host[m * trellis_stride..(m + 1) * trellis_stride]);
                for f in 0..scale_floats {
                    bytes.extend_from_slice(&scale_values[m * scale_floats + f].to_le_bytes());
                }
            }
            testkit::upload_bytes(&ctx, &bytes)
        });
        let member_stride = trellis_stride + scale_floats * 4;
        let table_of = |absent: Option<usize>| {
            let words: Vec<u8> = (0..members)
                .flat_map(|m| {
                    let word = if Some(m) == absent {
                        u32::MAX
                    } else {
                        (m * member_stride) as u32
                    };
                    word.to_le_bytes()
                })
                .collect();
            testkit::upload_bytes(&ctx, &words)
        };
        type Written = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);
        let run_with = |ids: &[i32],
                        indexed: Option<u32>,
                        wide: u32,
                        table: Option<&DeviceBuffer>|
         -> Written {
            let xh = testkit::upload_filled(&ctx, 0xff, pairs * k * 4);
            let z = testkit::upload_filled(&ctx, 0xff, pairs * n * 4);
            let out = testkit::upload_filled(&ctx, 0xff, pairs * n * 4);
            let flag = testkit::upload_bytes(&ctx, &0u32.to_le_bytes());
            let id_buffer = ids_of(ids);
            let spans = |p: usize| {
                (
                    xh.range(p * k * 4, k * 4).unwrap(),
                    z.range(p * n * 4, n * 4).unwrap(),
                    out.range(p * n * 4, n * 4).unwrap(),
                )
            };
            let call = |p: usize, trellis_range, suh, svh| {
                let (xh, z, out) = spans(p);
                Exl3LinearCall {
                    trellis: trellis_range,
                    suh,
                    svh,
                    bias: None,
                    x: x.range(p * k * 4, k * 4).unwrap(),
                    xh,
                    z,
                    out,
                    flag: flag.range(0, 4).unwrap(),
                    rows: 1,
                    split_k: 2,
                    k,
                    n,
                    bits_twice,
                    book,
                    fused: false,
                }
            };
            if indexed.is_some() {
                let calls: Vec<Exl3IndexedCall<'_>> = (0..pairs)
                    .map(|p| Exl3IndexedCall {
                        call: match table {
                            Some(_) => {
                                let arena = arena.as_ref().unwrap();
                                call(
                                    p,
                                    arena.range(0, arena.bytes()).unwrap(),
                                    arena
                                        .range(trellis_stride, arena.bytes() - trellis_stride)
                                        .unwrap(),
                                    arena
                                        .range(
                                            trellis_stride + k * 4,
                                            arena.bytes() - trellis_stride - k * 4,
                                        )
                                        .unwrap(),
                                )
                            }
                            None => call(
                                p,
                                trellis.range(offset, members * trellis_stride).unwrap(),
                                scales.range(0, scales.bytes()).unwrap(),
                                scales.range(k * 4, scales.bytes() - k * 4).unwrap(),
                            ),
                        },
                        ids: id_buffer.range(0, id_buffer.bytes()).unwrap(),
                        slot: p as u32,
                        members: members as u32,
                        trellis_stride,
                        scale_stride: scale_floats,
                        offsets: table.map(|t| t.range(0, t.bytes()).unwrap()),
                    })
                    .collect();
                ctx.exl3_grouped_indexed(&stream, &calls, wide).unwrap();
            } else {
                let calls: Vec<Exl3LinearCall<'_>> = (0..pairs)
                    .map(|p| {
                        let member = ids[p] as usize;
                        call(
                            p,
                            trellis
                                .range(offset + member * trellis_stride, trellis_stride)
                                .unwrap(),
                            scales.range(member * scale_floats * 4, k * 4).unwrap(),
                            scales
                                .range((member * scale_floats + k) * 4, n * 4)
                                .unwrap(),
                        )
                    })
                    .collect();
                ctx.exl3_grouped_wide(&stream, &calls, wide).unwrap();
            }
            stream.synchronize().unwrap();
            (
                testkit::read_bytes(&ctx, &xh, pairs * k * 4),
                testkit::read_bytes(&ctx, &z, pairs * n * 4),
                testkit::read_bytes(&ctx, &out, pairs * n * 4),
                testkit::read_bytes(&ctx, &flag, 4),
            )
        };
        let run = |ids: &[i32], indexed: Option<u32>, wide: u32| run_with(ids, indexed, wide, None);
        let mut first_cols: Option<Written> = None;
        for cols in [1u32, 2, 4] {
            let want = run(&picks, None, cols);
            let got = run(&picks, Some(cols), cols);
            assert_eq!(got, want, "{book:?} rate {bits_twice} cols {cols}");
            assert_eq!(got.3, [0; 4]);
            // One, two and four tiles per block write the same bytes.
            match &first_cols {
                None => first_cols = Some(got.clone()),
                Some(one_tile) => {
                    assert_eq!(
                        &got, one_tile,
                        "{book:?} rate {bits_twice} cols {cols} vs 1"
                    )
                }
            }
            if arena.is_some() {
                // The offset table addresses the same members: identical bytes in every span.
                let full = table_of(None);
                let by_table = run_with(&picks, Some(cols), cols, Some(&full));
                assert_eq!(
                    by_table, want,
                    "{book:?} rate {bits_twice} cols {cols} table"
                );
            }
        }
        if arena.is_some() {
            // An absent member skips its pairs and raises the flag; the others match.
            let reference = run(&picks, None, 2);
            let gapped = table_of(Some(3));
            let got = run_with(&picks, Some(2), 2, Some(&gapped));
            assert_eq!(
                got.3,
                1u32.to_le_bytes(),
                "{book:?} rate {bits_twice}: absent flag"
            );
            for p in (0..pairs).filter(|&p| picks[p] != 3) {
                assert_eq!(
                    got.2[p * n * 4..(p + 1) * n * 4],
                    reference.2[p * n * 4..(p + 1) * n * 4],
                    "{book:?} rate {bits_twice}: table pair {p}"
                );
            }
            for p in (0..pairs).filter(|&p| picks[p] == 3) {
                assert!(
                    got.2[p * n * 4..(p + 1) * n * 4].iter().all(|&b| b == 0xff),
                    "{book:?} rate {bits_twice}: absent pair {p} was written"
                );
            }
        }
        // Ids outside the bank: those pairs are skipped and the flag is raised; the others match.
        let mut bad = picks;
        bad[1] = members as i32;
        bad[6] = -1;
        let reference = run(&picks, None, 2);
        let got = run(&bad, Some(2), 2);
        assert_eq!(
            got.3,
            1u32.to_le_bytes(),
            "{book:?} rate {bits_twice}: the flag"
        );
        for p in [0usize, 2, 3, 4, 5, 7] {
            assert_eq!(
                got.2[p * n * 4..(p + 1) * n * 4],
                reference.2[p * n * 4..(p + 1) * n * 4],
                "{book:?} rate {bits_twice}: pair {p}"
            );
        }
        // A skipped pair leaves every span it owns at the fill byte: nothing was written for it.
        for p in [1usize, 6] {
            assert!(
                got.0[p * k * 4..(p + 1) * k * 4].iter().all(|&b| b == 0xff)
                    && got.1[p * n * 4..(p + 1) * n * 4].iter().all(|&b| b == 0xff)
                    && got.2[p * n * 4..(p + 1) * n * 4].iter().all(|&b| b == 0xff),
                "{book:?} rate {bits_twice}: pair {p} was written"
            );
        }
    }
}

/// Microseconds per call of `path` on a weight pool larger than L2, as the decode step sees it.
fn time_path(ctx: &CudaContext, stream: &DeviceStream, case: &Case, path: Path) -> f64 {
    let mut rng = testkit::Rng::new(0x5347_0002);
    let (k, n) = case.shape;
    let x = input(&mut rng, case.rows * k, false);
    let bytes = k * n / 256 * usize::from(case.bits_twice) * 16;
    let copies = (96usize << 20).div_ceil(bytes).clamp(2, 48);
    let rigs: Vec<Rig> = (0..copies).map(|_| Rig::new(ctx, case, &x)).collect();
    let (start, end) = (ctx.event_timing().unwrap(), ctx.event_timing().unwrap());
    let rounds = 6;
    for rig in &rigs {
        rig.run(ctx, stream, path);
    }
    stream.record_timing(&start).unwrap();
    for _ in 0..rounds {
        for rig in &rigs {
            rig.run(ctx, stream, path);
        }
    }
    stream.record_timing(&end).unwrap();
    stream.synchronize().unwrap();
    f64::from(start.elapsed_ms_since(&end).unwrap()) * 1000.0 / (rounds * copies) as f64
}

#[test]
#[ignore = "timing: run from a release test binary with --ignored --nocapture"]
fn single_row_paths_timing_by_decode_shape() {
    if !testkit::device_present("EXL3 single-row timing") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    println!("shape k x n | legacy us | gather us | wide1 | wide2 | wide4 | wide8");
    for (k, n) in [
        (2048, 8192),
        (4096, 2048),
        (2048, 4096),
        (2048, 512),
        (2048, 248_320),
    ] {
        let case = Case {
            book: Codebook::ThreeInst,
            bits_twice: 8,
            shape: (k, n),
            rows: 1,
            bias: false,
            fused: false,
        };
        let legacy = time_path(&ctx, &stream, &case, Path::Legacy);
        let gather = time_path(&ctx, &stream, &case, Path::Gather);
        let pf: Vec<f64> = [1, 2, 4, 8]
            .into_iter()
            .map(|cols| time_path(&ctx, &stream, &case, Path::Wide(cols)))
            .collect();
        println!(
            "{k} x {n} | {legacy:.1} | {gather:.1} | {:.1} | {:.1} | {:.1} | {:.1}",
            pf[0], pf[1], pf[2], pf[3]
        );
    }
    // The real head: rate 12 (6.0 bpw), the bytes the server reads per token.
    let case = Case {
        book: Codebook::Mcg,
        bits_twice: 12,
        shape: (2048, 248_320),
        rows: 1,
        bias: false,
        fused: false,
    };
    let times: Vec<f64> = [1u32, 2, 4]
        .into_iter()
        .map(|cols| time_path(&ctx, &stream, &case, Path::Wide(cols)))
        .collect();
    println!(
        "rate12 head 2048 x 248320 | wide1 {:.1} | wide2 {:.1} | wide4 {:.1}",
        times[0], times[1], times[2]
    );
}

#[test]
#[ignore = "timing: run from a release test binary with --ignored --nocapture"]
fn multi_row_paths_timing_by_decode_shape() {
    if !testkit::device_present("EXL3 multi-row timing") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    println!(
        "book mcg rate 8; shape k x n | rows | one-row wide1 x rows us | rows kernel us | gather per-row-block us | wide2 per-row-block us"
    );
    for (k, n) in [(2048, 8192), (4096, 2048), (2048, 4096), (2048, 512)] {
        let one = Case {
            book: Codebook::Mcg,
            bits_twice: 8,
            shape: (k, n),
            rows: 1,
            bias: false,
            fused: false,
        };
        let solo = time_path(&ctx, &stream, &one, Path::Wide(1));
        for rows in [1usize, 2, 4, 8] {
            let case = Case { rows, ..one };
            let rows_kernel = time_path(&ctx, &stream, &case, Path::Wide(1));
            let gather = time_path(&ctx, &stream, &case, Path::Gather);
            let wide2 = time_path(&ctx, &stream, &case, Path::Wide(2));
            println!(
                "{k} x {n} | {rows} | {:.1} | {rows_kernel:.1} | {gather:.1} | {wide2:.1}",
                solo * rows as f64
            );
        }
    }
}

/// Microseconds per id-indexed launch (input rotate, projection, output rotate) of `pairs` one-row
/// pairs of a `k` x `n` mcg rate-8 bank of `members` experts at `cols` tiles per block, with a new
/// pick set each call so the weights come from memory as in the decode step.
fn time_indexed(
    ctx: &CudaContext,
    stream: &DeviceStream,
    shape: (usize, usize),
    pairs: usize,
    members: usize,
    cols: u32,
) -> f64 {
    let (k, n) = shape;
    let mut rng = testkit::Rng::new(0x5347_0006);
    let bits_twice = 8u8;
    let trellis_stride = k * n / 256 * usize::from(bits_twice) * 16;
    let scale_floats = k + n;
    let trellis = testkit::upload_bytes(ctx, &random_bytes(&mut rng, members * trellis_stride));
    let scales = testkit::upload_f32(ctx, &vec![0.5f32; members * scale_floats]);
    let x = testkit::upload_f32(ctx, &input(&mut rng, pairs * k, false));
    let xh = testkit::upload_filled(ctx, 0, pairs * k * 4);
    let z = testkit::upload_filled(ctx, 0, pairs * n * 4);
    let out = testkit::upload_filled(ctx, 0, pairs * n * 4);
    let flag = testkit::upload_bytes(ctx, &0u32.to_le_bytes());
    let sets = 32usize;
    // `sets` pick sets of eight distinct members, repeated across the pairs.
    let ids: Vec<i32> = (0..sets * 16)
        .map(|i| ((i / 16) * 7 + (i % 8) * 13 + (i / 8) % 3) as i32 % members as i32)
        .collect();
    let id_buffer = testkit::upload_bytes(
        ctx,
        &ids.iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let launch = |set: usize| {
        let calls: Vec<Exl3IndexedCall<'_>> = (0..pairs)
            .map(|p| Exl3IndexedCall {
                call: Exl3LinearCall {
                    trellis: trellis.range(0, members * trellis_stride).unwrap(),
                    suh: scales.range(0, scales.bytes()).unwrap(),
                    svh: scales.range(k * 4, scales.bytes() - k * 4).unwrap(),
                    bias: None,
                    x: x.range(p * k * 4, k * 4).unwrap(),
                    xh: xh.range(p * k * 4, k * 4).unwrap(),
                    z: z.range(p * n * 4, n * 4).unwrap(),
                    out: out.range(p * n * 4, n * 4).unwrap(),
                    flag: flag.range(0, 4).unwrap(),
                    rows: 1,
                    split_k: 2,
                    k,
                    n,
                    bits_twice,
                    book: Codebook::Mcg,
                    fused: false,
                },
                ids: id_buffer.range(0, id_buffer.bytes()).unwrap(),
                slot: (set * 16 + p % 8) as u32,
                members: members as u32,
                trellis_stride,
                scale_stride: scale_floats,
                offsets: None,
            })
            .collect();
        ctx.exl3_grouped_indexed(stream, &calls, cols).unwrap();
    };
    let (start, end) = (ctx.event_timing().unwrap(), ctx.event_timing().unwrap());
    for set in 0..sets {
        launch(set);
    }
    let rounds = 8;
    stream.record_timing(&start).unwrap();
    for _ in 0..rounds {
        for set in 0..sets {
            launch(set);
        }
    }
    stream.record_timing(&end).unwrap();
    stream.synchronize().unwrap();
    f64::from(start.elapsed_ms_since(&end).unwrap()) * 1000.0 / (rounds * sets) as f64
}

#[test]
#[ignore = "timing: run from a release test binary with --ignored --nocapture"]
fn indexed_expert_paths_timing_by_projection() {
    if !testkit::device_present("EXL3 indexed timing") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    println!(
        "book mcg rate 8, 64-member bank; projection | k x n | pairs | cols 1 | cols 2 | cols 4 | cols 8 (us per launch chain)"
    );
    for (name, shape, pairs) in [("gate+up", (2048, 512), 16), ("down", (512, 2048), 8)] {
        let times: Vec<f64> = [1u32, 2, 4, 8]
            .into_iter()
            .map(|cols| time_indexed(&ctx, &stream, shape, pairs, 64, cols))
            .collect();
        println!(
            "{name} | {} x {} | {pairs} | {:.1} | {:.1} | {:.1} | {:.1}",
            shape.0, shape.1, times[0], times[1], times[2], times[3]
        );
    }
}
