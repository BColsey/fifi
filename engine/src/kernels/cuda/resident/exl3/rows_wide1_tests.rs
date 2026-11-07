//! The K-row verifier's rows kernel against the production single-row launch, on the real
//! Qwen3.6 shapes.
//!
//! One group of K rows (K = 2 to 8) through `exl3_grouped_step` runs `grouped_project_rows<R>`
//! (R = 2 for K = 2, 4 for K = 3 and 4, 8 for K = 5 to 8, so K = 3 runs R = 4 with one masked row
//! and K = 5 to 7 run R = 8 with masked rows). The served single-row step runs the same
//! projection as K launches of `grouped_project_small_wide<1>` (`wide1`) on one-row groups. The
//! two kernels are different code with one source-level reduction order, so equal bytes are not
//! automatic. Each case here writes every byte the call writes (the rotated input `xh`, the
//! partial bank `z`, the output `out` and the sticky flag word) from the K-row launch and from the
//! K one-row launches on row t of the same input, and requires them equal. The shapes are the
//! dense projections of the model (2048x8192, 2048x4096, 4096x2048, 2048x512, 512x2048) and the
//! head (2048x248320 at rate 12, 6.0 bits per weight, about 381 MB of trellis built once), the
//! rates are 4, 6 and 8 bits per weight and the codebooks the three the loader reads.
//!
//! Equal bytes cannot tell which kernel ran, so every launch phase also reads the test-only
//! launch counters: the K-row launch must count exactly one launch of the rows kernel of its
//! row count, and the K one-row launches exactly K launches of `wide1`, nothing else.
//!
//! Mutants this kills: a rows kernel that sums rows in reverse tile order for R >= 4, which
//! changes only the low bits and shows only on real data and real shapes, where the tile count
//! per column is large enough for the order to matter), a wrong row-count to kernel dispatch
//! (K = 3 on the R = 2 kernel, K = 5 to 7 on the R = 4 kernel), a K-row call that runs K one-row
//! launches (the counters), and a rows kernel whose flag or zeroed value for a nonfinite row
//! differs from the one-row kernel's.
//!
//! The device tests need the served kernel selection: the context registers `wide1` as the
//! Qwen3.6 start does, and `FIFI_EXL3_SINGLE_GATHER` must be unset (an explicit value wins over
//! the registered kernel, and the tests refuse to run against another kernel). A host without a
//! device skips them. The table test needs no device.

use super::linear_tests::{fixture, upload_scales};
use super::{Exl3LinearCall, SERVED_SINGLE_ROW, SingleRow};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, testkit};
use crate::kernels::row_launches::{RowLaunches, RowsKernel, rows_kernel};
use crate::weights::exl3::Codebook;

/// Weights of one projection: synthetic random trellis words and scales of the real shape.
struct Weights {
    trellis: DeviceBuffer,
    suh: DeviceBuffer,
    svh: DeviceBuffer,
    k: usize,
    n: usize,
    bits_twice: u8,
    book: Codebook,
}

impl Weights {
    fn new(ctx: &CudaContext, book: Codebook, bits_twice: u8, (k, n): (usize, usize)) -> Self {
        let matrix = fixture(false, book, bits_twice, (k, n));
        Self {
            trellis: testkit::upload_bytes(ctx, matrix.trellis().host_bytes().unwrap()),
            suh: upload_scales(ctx, matrix.suh()),
            svh: upload_scales(ctx, matrix.svh()),
            k,
            n,
            bits_twice,
            book,
        }
    }

    /// The call of `rows` rows starting at row `first` of the buffers: rows `first..first + rows`
    /// of the input, the rotated input, the partial bank and the output, and the shared flag.
    fn call<'a>(
        &'a self,
        x: &'a DeviceBuffer,
        out: &'a Outputs,
        first: usize,
        rows: usize,
    ) -> Exl3LinearCall<'a> {
        let (k, n) = (self.k, self.n);
        Exl3LinearCall {
            trellis: self.trellis.range(0, self.trellis.bytes()).unwrap(),
            suh: self.suh.range(0, k * 4).unwrap(),
            svh: self.svh.range(0, n * 4).unwrap(),
            bias: None,
            x: x.range(first * k * 4, rows * k * 4).unwrap(),
            xh: out.xh.range(first * k * 4, rows * k * 4).unwrap(),
            z: out.z.range(first * n * 4, rows * n * 4).unwrap(),
            out: out.out.range(first * n * 4, rows * n * 4).unwrap(),
            flag: out.flag.range(0, 4).unwrap(),
            rows,
            split_k: 2,
            k,
            n,
            bits_twice: self.bits_twice,
            book: self.book,
            fused: false,
        }
    }

    fn label(&self, rows: usize, poison: bool) -> String {
        format!(
            "{:?} bits_twice {} {}x{} K {rows} poison {poison}",
            self.book, self.bits_twice, self.k, self.n
        )
    }
}

/// The spans one launch writes, filled with 0xff so an unwritten byte shows.
struct Outputs {
    xh: DeviceBuffer,
    z: DeviceBuffer,
    out: DeviceBuffer,
    flag: DeviceBuffer,
}

impl Outputs {
    fn new(ctx: &CudaContext, (k, n): (usize, usize), rows: usize) -> Self {
        Self {
            xh: testkit::upload_filled(ctx, 0xff, rows * k * 4),
            z: testkit::upload_filled(ctx, 0xff, rows * n * 4),
            out: testkit::upload_filled(ctx, 0xff, rows * n * 4),
            flag: testkit::upload_bytes(ctx, &0u32.to_le_bytes()),
        }
    }

    fn written(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        (k, n): (usize, usize),
        rows: usize,
    ) -> Written {
        stream.synchronize().unwrap();
        Written {
            xh: testkit::read_bytes(ctx, &self.xh, rows * k * 4),
            z: testkit::read_bytes(ctx, &self.z, rows * n * 4),
            out: testkit::read_bytes(ctx, &self.out, rows * n * 4),
            flag: testkit::read_bytes(ctx, &self.flag, 4),
        }
    }
}

/// Every byte a call writes. Compared with [`first_difference`], not `assert_eq`, so a failure on
/// the head does not print 8 MB of bytes.
struct Written {
    xh: Vec<u8>,
    z: Vec<u8>,
    out: Vec<u8>,
    flag: Vec<u8>,
}

/// Where two writes first differ, or `None` when they are equal; the pair is the byte count of
/// one row of `xh` and of one row of `z` and `out`, so the report names the row.
fn first_difference(a: &Written, b: &Written, (xh_row, n_row): (usize, usize)) -> Option<String> {
    for (name, left, right, row) in [
        ("xh", &a.xh, &b.xh, xh_row),
        ("z", &a.z, &b.z, n_row),
        ("out", &a.out, &b.out, n_row),
        ("flag", &a.flag, &b.flag, 4),
    ] {
        if left.len() != right.len() {
            return Some(format!(
                "{name}: {} bytes against {}",
                left.len(),
                right.len()
            ));
        }
        if let Some(at) = left.iter().zip(right).position(|(l, r)| l != r) {
            let count = left.iter().zip(right).filter(|(l, r)| l != r).count();
            return Some(format!(
                "{name}: {count} bytes differ, first at byte {at} (row {}, byte {} of the row): \
                 {:#04x} against {:#04x}",
                at / row,
                at % row,
                left[at],
                right[at]
            ));
        }
    }
    None
}

/// The same inputs as `single_gather_tests::input`: uniform in (-4, 4).
fn input(rng: &mut testkit::Rng, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| (rng.unit() as f32 - 0.5) * 8.0)
        .collect()
}

/// A context that runs the served single-row kernel, or a panic that says why it cannot.
fn served_context() -> (CudaContext, DeviceStream) {
    let ctx = CudaContext::new().expect("context");
    ctx.set_single_row_default(SERVED_SINGLE_ROW);
    assert_eq!(
        ctx.single_row_setting().expect("single-row setting"),
        SingleRow::Wide(1),
        "FIFI_EXL3_SINGLE_GATHER must be unset (or wide1): the single-row side of this test is wide1"
    );
    let stream = ctx.stream().expect("stream");
    (ctx, stream)
}

/// The counters after `before` plus `launches` projection launches of `kernel`, nothing else.
fn counted(before: RowLaunches, kernel: RowsKernel, launches: u64) -> RowLaunches {
    let mut expected = before;
    for _ in 0..launches {
        expected.record_projection(kernel);
    }
    expected
}

/// One launch of one group of `rows` rows through the step's entry, with the kernel counters
/// checked: exactly one launch of the rows kernel of this row count.
fn run_rows_launch(
    (ctx, stream): (&CudaContext, &DeviceStream),
    weights: &Weights,
    x: &DeviceBuffer,
    rows: usize,
    label: &str,
) -> Written {
    let shape = (weights.k, weights.n);
    let outputs = Outputs::new(ctx, shape, rows);
    let kernel = rows_kernel(rows).expect("a row count of 1 to 8 has a kernel");
    assert_ne!(
        kernel,
        RowsKernel::SmallWide1,
        "{label}: K >= 2 must not map to wide1"
    );
    let before = ctx.row_launch_counts();
    ctx.exl3_grouped_step(stream, &[weights.call(x, &outputs, 0, rows)])
        .unwrap();
    assert_eq!(
        ctx.row_launch_counts(),
        counted(before, kernel, 1),
        "{label}: the K-row launch must be one launch of {kernel:?} and nothing else"
    );
    outputs.written(ctx, stream, shape, rows)
}

/// `rows` launches of the step's entry, each one one-row group on row t, with the kernel counters
/// checked: exactly `rows` launches of `wide1`.
fn run_single_row_launches(
    (ctx, stream): (&CudaContext, &DeviceStream),
    weights: &Weights,
    x: &DeviceBuffer,
    rows: usize,
    label: &str,
) -> Written {
    let shape = (weights.k, weights.n);
    let outputs = Outputs::new(ctx, shape, rows);
    let before = ctx.row_launch_counts();
    for t in 0..rows {
        ctx.exl3_grouped_step(stream, &[weights.call(x, &outputs, t, 1)])
            .unwrap();
    }
    assert_eq!(
        ctx.row_launch_counts(),
        counted(before, RowsKernel::SmallWide1, rows as u64),
        "{label}: {rows} one-row launches must be {rows} launches of wide1 and nothing else"
    );
    outputs.written(ctx, stream, shape, rows)
}

/// The K-row launch against the K one-row launches for one input, byte for byte.
fn compare_launches(
    pair: (&CudaContext, &DeviceStream),
    weights: &Weights,
    x_values: &[f32],
    rows: usize,
    poison: bool,
) {
    let label = weights.label(rows, poison);
    let x = testkit::upload_f32(pair.0, x_values);
    let rows_launch = run_rows_launch(pair, weights, &x, rows, &label);
    let single = run_single_row_launches(pair, weights, &x, rows, &label);
    let row_bytes = (weights.k * 4, weights.n * 4);
    if let Some(difference) = first_difference(&rows_launch, &single, row_bytes) {
        panic!("{label}: the rows kernel differs from wide1: {difference}");
    }
    // The launches wrote something: the fill byte is gone from the output and the rotated input.
    assert!(
        rows_launch.out.iter().any(|&b| b != 0xff) && rows_launch.xh.iter().any(|&b| b != 0xff),
        "{label}: nothing was written"
    );
    // The sticky word is raised exactly when a row carried an infinity.
    assert_eq!(rows_launch.flag != [0; 4], poison, "{label}: the flag word");
}

/// Both inputs for one weight set and row count: finite data, then the same shape of data with an
/// infinity in row 1 so the flag and the zeroed row agree between the kernels.
fn compare_case(
    pair: (&CudaContext, &DeviceStream),
    weights: &Weights,
    rows: usize,
    rng: &mut testkit::Rng,
) {
    let mut x = input(rng, rows * weights.k);
    compare_launches(pair, weights, &x, rows, false);
    x[weights.k + 37] = f32::INFINITY;
    compare_launches(pair, weights, &x, rows, true);
}

/// Dense projections of the model at K = 2..=8 rows, bitwise equal to K launches of `wide1`:
/// 2048x8192 at every K, rate and codebook the loader reads; the other four mid-size shapes at
/// K = 2, 4 and 8 (the sizes of the three rows kernels) at rates 4, 6 and 8 bits per weight.
#[test]
fn the_rows_kernel_matches_the_production_wide1_launch_on_the_real_dense_shapes() {
    if !testkit::device_present("EXL3 rows kernel vs wide1, dense shapes") {
        return;
    }
    let (ctx, stream) = served_context();
    let pair = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5347_0101);
    let mut cases = 0;
    // The qkv / attention-q shape: every K, so K = 3 (R = 4 with a masked row) and K = 5 to 7
    // (R = 8 with masked rows) run, at the three rates and all three codebooks. Mul1 at an odd
    // rate (5) takes the odd-rate branch of the one-tile body.
    for (book, bits_twice) in [
        (Codebook::ThreeInst, 8u8),
        (Codebook::ThreeInst, 12),
        (Codebook::ThreeInst, 16),
        (Codebook::Mcg, 8),
        (Codebook::Mcg, 12),
        (Codebook::Mul1, 8),
        (Codebook::Mul1, 5),
    ] {
        let weights = Weights::new(&ctx, book, bits_twice, (2048, 8192));
        for rows in 2..=8 {
            compare_case(pair, &weights, rows, &mut rng);
            cases += 2;
        }
    }
    // The other dense shapes: z (2048x4096), out and attention o (4096x2048), shared gate/up and
    // attention k/v (2048x512), shared down (512x2048).
    for shape in [(2048, 4096), (4096, 2048), (2048, 512), (512, 2048)] {
        for (book, bits_twice) in [
            (Codebook::ThreeInst, 8u8),
            (Codebook::ThreeInst, 12),
            (Codebook::ThreeInst, 16),
            (Codebook::Mcg, 8),
            (Codebook::Mul1, 5),
        ] {
            let weights = Weights::new(&ctx, book, bits_twice, shape);
            for rows in [2usize, 4, 8] {
                compare_case(pair, &weights, rows, &mut rng);
                cases += 2;
            }
        }
    }
    println!("rows kernel vs wide1, dense shapes: {cases} launch pairs bit identical");
}

/// The LM head (2048x248320, rate 12 = 6.0 bits per weight, about 381 MB of trellis built once
/// and reused for every K) at K = 2..=8, bitwise equal to K launches of `wide1`. It is the
/// largest matrix, so the largest tile count per column: the shape on which a reversed tile
/// order (the reversed-order mutant) changes the low bits most reliably.
#[test]
fn the_rows_kernel_matches_the_production_wide1_launch_on_the_head() {
    if !testkit::device_present("EXL3 rows kernel vs wide1, head") {
        return;
    }
    let (ctx, stream) = served_context();
    let pair = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5347_0102);
    let head = Weights::new(&ctx, Codebook::ThreeInst, 12, (2048, 248_320));
    for rows in 2..=8 {
        compare_case(pair, &head, rows, &mut rng);
    }
    println!("rows kernel vs wide1, head: 7 row counts, finite and nonfinite input, bit identical");
}

/// The host table of the launch counters equals the native dispatch for every row count the
/// launch takes, and both refuse the counts outside 1..=8. Pure host code: no device is
/// needed, so it also runs in the CPU-only native gate.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn the_host_rows_kernel_table_equals_the_native_dispatch() {
    use crate::kernels::cuda::native::resident::exl3::launch_observer;
    use crate::kernels::row_launches::rows_kernel_from_native;

    for rows in 1..=8usize {
        let native = rows_kernel_from_native(launch_observer::rows_kernel(rows as u32));
        assert!(native.is_some(), "native refuses {rows} rows");
        assert_eq!(rows_kernel(rows), native, "row count {rows}");
    }
    for rows in [0usize, 9, 16, 48] {
        assert_eq!(rows_kernel(rows), None, "host table, {rows} rows");
        assert_eq!(
            rows_kernel_from_native(launch_observer::rows_kernel(rows as u32)),
            None,
            "native, {rows} rows"
        );
    }
}
