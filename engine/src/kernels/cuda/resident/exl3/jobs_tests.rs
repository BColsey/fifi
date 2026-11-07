//! The job-table projection against the rows entry of the single matrix: for a bank of
//! random members and tables that mix one-row jobs, a full 256-row job and several heights in
//! between, every job's rows must equal, bit for bit, the same rows through
//! `exl3_linear_midrows` on that member alone (the rows are placed in a 64-row call when they are
//! fewer, with zero rows after them, and cut into calls of 144 rows when they are more: the
//! tensor-core kernel computes each row independently). Jobs of up to 256 rows exercise the three
//! CTA heights (64, 128 and 256 rows) that the tallest job of a call selects.
//! Then the row stride and column offset of the output, the skipped job of an out-of-bounds table
//! entry, and the refusals of the wrapper.

use super::{Exl3Job, Exl3JobsCall, Exl3LinearCall};
use crate::EngError;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, testkit};
use crate::weights::exl3::Codebook;

const K: usize = 256;
const N: usize = 384;
const MEMBERS: usize = 5;
/// Column offset of the strided cases (their row stride is 40 values wider than the call).
const COL: usize = 8;
const CANARY: u8 = 0xa5;

struct Bank {
    k: usize,
    n: usize,
    trellis: DeviceBuffer,
    /// `[suh | svh]` per member.
    scales: DeviceBuffer,
    bits_twice: u8,
    book: Codebook,
}

fn bank(ctx: &CudaContext, bits_twice: u8, book: Codebook, seed: u64) -> Bank {
    bank_of(ctx, (K, N), (bits_twice, book), seed)
}

fn bank_of(
    ctx: &CudaContext,
    (k, n): (usize, usize),
    (bits_twice, book): (u8, Codebook),
    seed: u64,
) -> Bank {
    let mut rng = testkit::Rng::new(seed);
    let words = usize::from(bits_twice) * 8;
    let trellis: Vec<u8> = (0..MEMBERS * k * n / 256 * words)
        .flat_map(|_| (rng.next_u64() as u16).to_le_bytes())
        .collect();
    let scales: Vec<f32> = (0..MEMBERS * (k + n))
        .map(|_| 0.5 + rng.unit() as f32)
        .collect();
    Bank {
        k,
        n,
        trellis: testkit::upload_bytes(ctx, &trellis),
        scales: testkit::upload_f32(ctx, &scales),
        bits_twice,
        book,
    }
}

fn packed_elements(bits_twice: u8) -> usize {
    packed_elements_of(K, N, bits_twice)
}

fn packed_elements_of(k: usize, n: usize, bits_twice: u8) -> usize {
    k * n / 256 * usize::from(bits_twice) * 8
}

fn rows_of(rng: &mut testkit::Rng, rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|_| (rng.unit() as f32 - 0.5) * 4.0)
        .collect()
}

/// The table of a call: member, height and the first row of each job, rows back to back.
fn table(shape: &[(u32, u32)]) -> (Vec<Exl3Job>, usize) {
    let mut row0 = 0;
    let jobs = shape
        .iter()
        .map(|&(member, rows)| {
            let job = Exl3Job { member, row0, rows };
            row0 += rows;
            job
        })
        .collect();
    (jobs, row0 as usize)
}

struct Run {
    out: Vec<f32>,
    flag: u32,
    /// Whether the call took the register-decode kernel.
    registers: bool,
}

/// The kernel the device tests expect on their device (compute capability 8 or more): the
/// register-decode kernel unless `FIFI_EXL3_JOBS_REG=0` rolls it back.
fn registers_expected() -> bool {
    std::env::var("FIFI_EXL3_JOBS_REG").map_or(true, |value| !value.starts_with('0'))
}

/// One job-table call; `out` is the whole destination, canary-filled before the launch.
fn run_jobs(
    rig: (&CudaContext, &DeviceStream),
    bank: &Bank,
    jobs: &[Exl3Job],
    x: &[f32],
    out_shape: (usize, usize),
) -> Result<Run, EngError> {
    run_jobs_at(rig, bank, jobs, x, (out_shape, 0))
}

/// [`run_jobs`] with the rotated input rows starting `xh_offset` bytes into their buffer.
fn run_jobs_at(
    (ctx, stream): (&CudaContext, &DeviceStream),
    bank: &Bank,
    jobs: &[Exl3Job],
    x: &[f32],
    (out_shape, xh_offset): ((usize, usize), usize),
) -> Result<Run, EngError> {
    let (k, n) = (bank.k, bank.n);
    let (out_ld, out_col) = out_shape;
    let total_rows = x.len() / k;
    let max_rows = jobs.iter().map(|job| job.rows as usize).max().unwrap();
    let x_buf = testkit::upload_f32(ctx, x);
    let xh = testkit::upload_filled(ctx, 0, total_rows * k * 4 + 16);
    let z = testkit::upload_filled(ctx, 0, total_rows * n * 4);
    let out_bytes = total_rows * out_ld * 4;
    let out = testkit::upload_filled(ctx, CANARY, out_bytes);
    let flag = testkit::upload_filled(ctx, 0, 4);
    let table = Exl3Job::table(jobs);
    let table_buf = testkit::upload_bytes(ctx, &table);
    let scales = bank.scales.range(0, bank.scales.bytes())?;
    let call = Exl3JobsCall {
        trellis: bank.trellis.range(0, bank.trellis.bytes())?,
        suh: scales,
        svh: bank.scales.range(k * 4, bank.scales.bytes() - k * 4)?,
        trellis_stride: packed_elements_of(k, n, bank.bits_twice),
        scale_stride: k + n,
        members: MEMBERS,
        x: x_buf.range(0, x.len() * 4)?,
        xh: xh.range(xh_offset, total_rows * k * 4)?,
        z: z.range(0, total_rows * n * 4)?,
        out: out.range(0, out_bytes)?,
        flag: flag.range(0, 4)?,
        jobs: table_buf.range(0, table.len())?,
        jobs_count: jobs.len(),
        max_rows,
        total_rows,
        out_ld,
        out_col,
        k,
        n,
        bits_twice: bank.bits_twice,
        book: bank.book,
        in_rows: None,
        out_rows: None,
        x_rows: total_rows,
        out_total_rows: total_rows,
    };
    let registers = ctx.exl3_jobs_use_registers(&call)?;
    ctx.exl3_jobs(stream, call)?;
    stream.synchronize()?;
    let bytes = testkit::read_bytes(ctx, &out, out_bytes);
    let flag = testkit::read_bytes(ctx, &flag, 4);
    Ok(Run {
        out: testkit::floats(&bytes),
        flag: u32::from_le_bytes(flag.try_into().unwrap()),
        registers,
    })
}

fn single(rig: (&CudaContext, &DeviceStream), bank: &Bank, member: usize, x: &[f32]) -> Vec<f32> {
    x.chunks(144 * bank.k)
        .flat_map(|rows| single_call(rig, bank, member, rows))
        .collect()
}

fn single_call(
    (ctx, stream): (&CudaContext, &DeviceStream),
    bank: &Bank,
    member: usize,
    x: &[f32],
) -> Vec<f32> {
    let (k, n) = (bank.k, bank.n);
    let rows = x.len() / k;
    let padded = rows.max(64);
    let mut input = x.to_vec();
    input.resize(padded * k, 0.0);
    let x_buf = testkit::upload_f32(ctx, &input);
    let xh = testkit::upload_filled(ctx, 0, padded * k * 4);
    let z = testkit::upload_filled(ctx, 0, padded * n * 4);
    let out = testkit::upload_filled(ctx, 0, padded * n * 4);
    let flag = testkit::upload_filled(ctx, 0, 4);
    let trellis_bytes = packed_elements_of(k, n, bank.bits_twice) * 2;
    let scale_bytes = (k + n) * 4;
    ctx.exl3_linear_midrows(
        stream,
        Exl3LinearCall {
            trellis: bank
                .trellis
                .range(member * trellis_bytes, trellis_bytes)
                .unwrap(),
            suh: bank.scales.range(member * scale_bytes, k * 4).unwrap(),
            svh: bank
                .scales
                .range(member * scale_bytes + k * 4, n * 4)
                .unwrap(),
            bias: None,
            x: x_buf.range(0, padded * k * 4).unwrap(),
            xh: xh.range(0, padded * k * 4).unwrap(),
            z: z.range(0, padded * n * 4).unwrap(),
            out: out.range(0, padded * n * 4).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            rows: padded,
            split_k: 2,
            k,
            n,
            bits_twice: bank.bits_twice,
            book: bank.book,
            fused: false,
        },
    )
    .expect("single member");
    stream.synchronize().expect("drain");
    testkit::floats(&testkit::read_bytes(ctx, &out, padded * n * 4))[..rows * n].to_vec()
}

fn canary() -> u32 {
    u32::from_le_bytes([CANARY; 4])
}

#[test]
fn every_job_equals_its_member_through_the_rows_entry_and_nothing_else_moves() {
    if !testkit::device_present("EXL3 job table") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x4a4f_4253);
    // The rates and codebooks of the models (the 4.00 bpw pack is mostly 4 and 6 bits twice, MUL1)
    // on the shape of the small bank, then the rates of the pack on shapes with more stages of K
    // and fewer or more column groups.
    let small = (K, N);
    for ((k, n), (bits_twice, book)) in [
        (small, (8, Codebook::Mul1)),
        (small, (6, Codebook::Mul1)),
        (small, (5, Codebook::Mul1)),
        (small, (4, Codebook::Mcg)),
        (small, (10, Codebook::ThreeInst)),
        ((2048, 512), (8, Codebook::Mul1)),
        ((2048, 512), (4, Codebook::Mcg)),
        ((512, 128), (8, Codebook::Mul1)),
        ((1024, 256), (6, Codebook::ThreeInst)),
        // The projections of the Flint pack (4 bpw, MUL1): attention, shared expert, dense layers.
        ((2048, 4096), (8, Codebook::Mul1)),
        ((4096, 2048), (8, Codebook::Mul1)),
        ((2048, 6144), (8, Codebook::Mul1)),
        ((512, 2048), (8, Codebook::Mul1)),
        ((2048, 1024), (8, Codebook::Mul1)),
    ] {
        let bank = bank_of(
            &ctx,
            (k, n),
            (bits_twice, book),
            0x4e45_5757 + u64::from(bits_twice),
        );
        // One row, a part-block, a full row block, two row blocks, a member twice; then tables
        // whose tallest job picks the 128-row and the 256-row CTA.
        let shapes: [&[(u32, u32)]; 4] = [
            &[
                (3, 1),
                (0, 70),
                (4, 144),
                (2, 9),
                (1, 64),
                (4, 33),
                (3, 100),
            ],
            &[(3, 1), (0, 128), (4, 100), (2, 65), (1, 64)],
            &[(3, 1), (0, 256), (4, 200), (2, 129), (1, 130), (0, 12)],
            // Only short jobs: the call that takes the register-decode kernel (rows in 1..=64,
            // none a multiple of 16 but one, the tallest exactly one CTA).
            &[
                (3, 1),
                (0, 64),
                (4, 33),
                (2, 9),
                (1, 17),
                (3, 49),
                (0, 16),
                (4, 2),
            ],
        ];
        for shape in shapes {
            let (jobs, total) = table(shape);
            let x = rows_of(&mut rng, total, k);
            for (ld, col) in [(n, 0), (n + 40, COL)] {
                let run = run_jobs(rig, &bank, &jobs, &x, (ld, col)).expect("jobs");
                assert_eq!(run.flag, 0, "{bits_twice}/{book:?} {k}x{n}: flag");
                assert_eq!(
                    run.registers,
                    registers_expected(),
                    "{bits_twice}/{book:?} {k}x{n}: the kernel of the call"
                );
                for job in &jobs {
                    let (row0, rows) = (job.row0 as usize, job.rows as usize);
                    let want = single(
                        rig,
                        &bank,
                        job.member as usize,
                        &x[row0 * k..(row0 + rows) * k],
                    );
                    for r in 0..rows {
                        for c in 0..n {
                            let got = run.out[(row0 + r) * ld + col + c];
                            assert_eq!(
                                got.to_bits(),
                                want[r * n + c].to_bits(),
                                "{bits_twice}/{book:?} {k}x{n} ld {ld}: member {} row {r} column {c}",
                                job.member
                            );
                        }
                    }
                }
                for row in 0..total {
                    for c in (0..col).chain(col + n..ld) {
                        assert_eq!(run.out[row * ld + c].to_bits(), canary(), "gap {row}/{c}");
                    }
                }
            }
        }
    }
}

/// Row lists of a call: the rows of `x` the table rows read and the rows of `out` they land in.
struct Lists<'a> {
    input: Option<&'a [u32]>,
    output: Option<&'a [u32]>,
    /// Rows of `x` and of `out`.
    x_rows: usize,
    out_rows: usize,
    /// Bytes in front of each uploaded list; its range starts after them (0 is aligned).
    shift: usize,
}

/// The range of an uploaded list after its `shift` bytes of padding.
fn list_range(buf: &DeviceBuffer, shift: usize) -> Result<DeviceRange<'_>, EngError> {
    buf.range(shift, buf.bytes() - shift)
}

/// One job-table call with row lists; `x` holds `lists.x_rows` rows, the destination
/// `lists.out_rows` rows (canary-filled).
fn run_listed(
    (ctx, stream): (&CudaContext, &DeviceStream),
    bank: &Bank,
    jobs: &[Exl3Job],
    x: &[f32],
    lists: &Lists<'_>,
) -> Result<Run, EngError> {
    let (k, n) = (bank.k, bank.n);
    let total_rows = jobs.iter().map(|job| job.rows as usize).sum::<usize>();
    let max_rows = jobs.iter().map(|job| job.rows as usize).max().unwrap();
    let x_buf = testkit::upload_f32(ctx, x);
    let xh = testkit::upload_filled(ctx, 0, total_rows * k * 4 + 16);
    let z = testkit::upload_filled(ctx, 0, total_rows * n * 4);
    let out_bytes = lists.out_rows * n * 4;
    let out = testkit::upload_filled(ctx, CANARY, out_bytes);
    let flag = testkit::upload_filled(ctx, 0, 4);
    let table = Exl3Job::table(jobs);
    let table_buf = testkit::upload_bytes(ctx, &table);
    let upload = |list: &[u32]| {
        let mut bytes = vec![0u8; lists.shift];
        bytes.extend(list.iter().flat_map(|v| v.to_le_bytes()));
        testkit::upload_bytes(ctx, &bytes)
    };
    // A list's range is the whole of what was uploaded, so a short list reaches the checks.
    let listed = |buf| list_range(buf, lists.shift);
    let in_buf = lists.input.map(upload);
    let out_list_buf = lists.output.map(upload);
    let scales = bank.scales.range(0, bank.scales.bytes())?;
    let call = Exl3JobsCall {
        trellis: bank.trellis.range(0, bank.trellis.bytes())?,
        suh: scales,
        svh: bank.scales.range(k * 4, bank.scales.bytes() - k * 4)?,
        trellis_stride: packed_elements_of(k, n, bank.bits_twice),
        scale_stride: k + n,
        members: MEMBERS,
        x: x_buf.range(0, x.len() * 4)?,
        xh: xh.range(0, total_rows * k * 4)?,
        z: z.range(0, total_rows * n * 4)?,
        out: out.range(0, out_bytes)?,
        flag: flag.range(0, 4)?,
        jobs: table_buf.range(0, table.len())?,
        jobs_count: jobs.len(),
        max_rows,
        total_rows,
        out_ld: n,
        out_col: 0,
        k,
        n,
        bits_twice: bank.bits_twice,
        book: bank.book,
        in_rows: in_buf.as_ref().map(listed).transpose()?,
        out_rows: out_list_buf.as_ref().map(listed).transpose()?,
        x_rows: lists.x_rows,
        out_total_rows: lists.out_rows,
    };
    let registers = ctx.exl3_jobs_use_registers(&call)?;
    ctx.exl3_jobs(stream, call)?;
    stream.synchronize()?;
    let bytes = testkit::read_bytes(ctx, &out, out_bytes);
    let flag = testkit::read_bytes(ctx, &flag, 4);
    Ok(Run {
        out: testkit::floats(&bytes),
        flag: u32::from_le_bytes(flag.try_into().unwrap()),
        registers,
    })
}

/// A pseudo-random permutation of `0..len` (Fisher-Yates over the test generator).
fn permutation(rng: &mut testkit::Rng, len: usize) -> Vec<u32> {
    let mut list: Vec<u32> = (0..len as u32).collect();
    for i in (1..len).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        list.swap(i, j);
    }
    list
}

/// One job-table call through the explicit fused-output entry with the request `fused_out`
/// (both legs of the byte-identity test go through it, so its false arm is exercised too);
/// `z` is canary-filled before the launch and returned raw, so the caller can see whether the
/// fused epilogue left it untouched. `listed_out` optionally supplies the output row list.
fn run_jobs_output_mode(
    (ctx, stream): (&CudaContext, &DeviceStream),
    bank: &Bank,
    jobs: &[Exl3Job],
    x: &[f32],
    (out_ld, out_col): (usize, usize),
    listed_out: Option<&[u32]>,
    fused_out: bool,
) -> Result<(Vec<u8>, u32, bool, Vec<u8>), EngError> {
    let (k, n) = (bank.k, bank.n);
    let total_rows = x.len() / k;
    let max_rows = jobs.iter().map(|job| job.rows as usize).max().unwrap();
    let x_buf = testkit::upload_f32(ctx, x);
    let xh = testkit::upload_filled(ctx, 0, total_rows * k * 4);
    let z = testkit::upload_filled(ctx, CANARY, total_rows * n * 4);
    let out_bytes = total_rows * out_ld * 4;
    let out = testkit::upload_filled(ctx, CANARY, out_bytes);
    let flag = testkit::upload_filled(ctx, 0, 4);
    let table = Exl3Job::table(jobs);
    let table_buf = testkit::upload_bytes(ctx, &table);
    let out_rows_buf = listed_out.map(|list| {
        let bytes: Vec<u8> = list.iter().flat_map(|&r| r.to_le_bytes()).collect();
        testkit::upload_bytes(ctx, &bytes)
    });
    let scales = bank.scales.range(0, bank.scales.bytes())?;
    let call = Exl3JobsCall {
        trellis: bank.trellis.range(0, bank.trellis.bytes())?,
        suh: scales,
        svh: bank.scales.range(k * 4, bank.scales.bytes() - k * 4)?,
        trellis_stride: packed_elements_of(k, n, bank.bits_twice),
        scale_stride: k + n,
        members: MEMBERS,
        x: x_buf.range(0, x.len() * 4)?,
        xh: xh.range(0, total_rows * k * 4)?,
        z: z.range(0, total_rows * n * 4)?,
        out: out.range(0, out_bytes)?,
        flag: flag.range(0, 4)?,
        jobs: table_buf.range(0, table.len())?,
        jobs_count: jobs.len(),
        max_rows,
        total_rows,
        out_ld,
        out_col,
        k,
        n,
        bits_twice: bank.bits_twice,
        book: bank.book,
        in_rows: None,
        out_rows: out_rows_buf
            .as_ref()
            .map(|buf| buf.range(0, total_rows * 4))
            .transpose()?,
        x_rows: total_rows,
        out_total_rows: total_rows,
    };
    let registers = ctx.exl3_jobs_use_registers(&call)?;
    ctx.exl3_jobs_fused_out(stream, call, fused_out)?;
    stream.synchronize()?;
    let out_read = testkit::read_bytes(ctx, &out, out_bytes);
    let flag_read = testkit::read_bytes(ctx, &flag, 4);
    let z_read = testkit::read_bytes(ctx, &z, total_rows * n * 4);
    Ok((
        out_read,
        u32::from_le_bytes(flag_read.try_into().unwrap()),
        registers,
        z_read,
    ))
}

/// The opt-in fused output-rotation epilogue (`FIFI_EXL3_JOBS_FUSED_OUT`, driven through the
/// explicit entry so no env flip races the other tests of the process) must publish the same
/// bits as the separate rotation kernel, keep the flag, and leave the z projection buffer
/// untouched. The mixed heights cover one-row jobs, part blocks, odd tops and several CTAs, so
/// both 32-row staging slabs and the absent-row side of a butterfly pair run; the listed-output
/// rounds cover the fused scatter and its out-of-range refusal.
#[test]
fn fused_output_rotation_matches_the_separate_kernel_and_leaves_z_untouched() {
    if !testkit::device_present("EXL3 fused output rotation") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x4655_5345);
    // The rollback has no fused path to compare; the plain entry keeps its own coverage.
    if !registers_expected() {
        return;
    }
    for (bits_twice, book) in [(8, Codebook::Mul1), (4, Codebook::Mcg)] {
        let bank = bank_of(
            &ctx,
            (K, N),
            (bits_twice, book),
            0x4e45_5757 + u64::from(bits_twice),
        );
        let shape: &[(u32, u32)] = &[(3, 1), (0, 70), (4, 144), (2, 9), (1, 64), (4, 33)];
        let (jobs, total) = table(shape);
        let x = rows_of(&mut rng, total, bank.k);
        let reversed: Vec<u32> = (0..total as u32).rev().collect();
        let mut over_range = reversed.clone();
        over_range[2] = total as u32; // one past the rows of out: refused, nothing written
        for (ld, col) in [(bank.n, 0), (bank.n + 40, COL)] {
            for (name, listed) in [("straight", None), ("reversed", Some(&reversed[..]))] {
                let (plain_out, plain_flag, registers, plain_z) =
                    run_jobs_output_mode(rig, &bank, &jobs, &x, (ld, col), listed, false)
                        .expect("jobs");
                assert!(
                    registers,
                    "the fused epilogue needs the register-decode kernel"
                );
                assert_eq!(
                    plain_flag, 0,
                    "{bits_twice}/{book:?}: flag of the plain call"
                );
                assert_ne!(
                    plain_z,
                    vec![CANARY; plain_z.len()],
                    "{bits_twice}/{book:?}: the plain call must write z"
                );
                let (fused_out, fused_flag, _, fused_z) =
                    run_jobs_output_mode(rig, &bank, &jobs, &x, (ld, col), listed, true)
                        .expect("fused jobs");
                assert_eq!(
                    fused_out, plain_out,
                    "{bits_twice}/{book:?} ld={ld} col={col} {name}: fused bits differ"
                );
                assert_eq!(
                    fused_flag, plain_flag,
                    "{bits_twice}/{book:?} {name}: flag differs"
                );
                assert_eq!(
                    fused_z,
                    vec![CANARY; fused_z.len()],
                    "{bits_twice}/{book:?} {name}: the fused call must not write z"
                );
            }
            // The out-of-range scatter: both paths flag it, drop the same row, keep its canary.
            let (plain_out, plain_flag, _, _) =
                run_jobs_output_mode(rig, &bank, &jobs, &x, (ld, col), Some(&over_range), false)
                    .expect("jobs");
            let (fused_out, fused_flag, _, _) =
                run_jobs_output_mode(rig, &bank, &jobs, &x, (ld, col), Some(&over_range), true)
                    .expect("fused jobs");
            assert_eq!(
                plain_flag & 1,
                1,
                "{bits_twice}/{book:?}: the plain call must flag"
            );
            assert_eq!(
                fused_flag, plain_flag,
                "{bits_twice}/{book:?}: refusal flag differs"
            );
            assert_eq!(
                fused_out, plain_out,
                "{bits_twice}/{book:?}: refused rows differ"
            );
            let kept = (0..total)
                .filter(|&r| {
                    fused_out[r * ld * 4..(r + 1) * ld * 4]
                        .iter()
                        .all(|b| *b == CANARY)
                })
                .count();
            assert_eq!(
                kept, 1,
                "k={k} {bits_twice}/{book:?}: exactly the refused row keeps canary"
            );
        }
    }
}

#[test]
fn row_lists_equal_the_gather_before_and_the_scatter_after_bit_for_bit() {
    if !testkit::device_present("EXL3 job table row lists") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5045_524d);
    for (bits_twice, book) in [(8, Codebook::Mul1), (4, Codebook::Mcg)] {
        let bank = bank(&ctx, bits_twice, book, 0x5045_4e4b + u64::from(bits_twice));
        // The expert-sorted table of a routed chunk: a member with no rows (1), one with a
        // single row (4), a tall one and a partial final job; `tokens` rows of x are each used by
        // `top_k` slots, so a row of x appears several times in the list (a token routed to the
        // same expert from two slots reads the same row twice).
        let tokens = 37usize;
        let (jobs, total) = table(&[(0, 41), (2, 1), (3, 70), (0, 3), (4, 13)]);
        let x = rows_of(&mut rng, tokens, K);
        let input: Vec<u32> = (0..total)
            .map(|r| {
                if r % 11 == 5 {
                    7
                } else {
                    (rng.next_u64() % tokens as u64) as u32
                }
            })
            .collect();
        let output = permutation(&mut rng, total);
        // The gather path: the rows gathered into the table order, no lists.
        let gathered: Vec<f32> = input
            .iter()
            .flat_map(|&row| x[row as usize * K..(row as usize + 1) * K].to_vec())
            .collect();
        let plain = run_jobs(rig, &bank, &jobs, &gathered, (N, 0)).expect("plain call");
        assert_eq!(plain.flag, 0);
        // Input list only: the output stays in table order.
        let listed = run_listed(
            rig,
            &bank,
            &jobs,
            &x,
            &Lists {
                input: Some(&input),
                output: None,
                x_rows: tokens,
                out_rows: total,
                shift: 0,
            },
        )
        .expect("input list");
        assert_eq!(listed.flag, 0);
        assert!(
            listed
                .out
                .iter()
                .zip(&plain.out)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{bits_twice}/{book:?}: the input list reads the rows the gather would have copied"
        );
        // Both lists: row r of the table lands in row output[r]; the scatter is the inverse
        // permutation of the gather back.
        let both = run_listed(
            rig,
            &bank,
            &jobs,
            &x,
            &Lists {
                input: Some(&input),
                output: Some(&output),
                x_rows: tokens,
                out_rows: total,
                shift: 0,
            },
        )
        .expect("both lists");
        assert_eq!(both.flag, 0);
        let mut inverse = vec![0u32; total];
        for (row, &target) in output.iter().enumerate() {
            inverse[target as usize] = row as u32;
        }
        for (target, &source) in inverse.iter().enumerate() {
            let source = source as usize;
            for c in 0..N {
                assert_eq!(
                    both.out[target * N + c].to_bits(),
                    plain.out[source * N + c].to_bits(),
                    "{bits_twice}/{book:?}: out row {target} is table row {source}, column {c}"
                );
            }
        }
        // An output buffer wider than the table keeps its fill where no row lands.
        let wide = total + 9;
        let targets: Vec<u32> = permutation(&mut rng, wide)[..total].to_vec();
        let spread = run_listed(
            rig,
            &bank,
            &jobs,
            &x,
            &Lists {
                input: Some(&input),
                output: Some(&targets),
                x_rows: tokens,
                out_rows: wide,
                shift: 0,
            },
        )
        .expect("wide output");
        assert_eq!(spread.flag, 0);
        let mut landed = vec![false; wide];
        for (row, &target) in targets.iter().enumerate() {
            landed[target as usize] = true;
            for c in 0..N {
                assert_eq!(
                    spread.out[target as usize * N + c].to_bits(),
                    plain.out[row * N + c].to_bits()
                );
            }
        }
        for (target, hit) in landed.iter().enumerate() {
            if !hit {
                assert!(
                    spread.out[target * N..(target + 1) * N]
                        .iter()
                        .all(|v| v.to_bits() == canary())
                );
            }
        }
    }
}

#[test]
fn a_row_list_entry_out_of_range_is_dropped_and_flags_the_call() {
    if !testkit::device_present("EXL3 job table row list bounds") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x4f4f_5252);
    let bank = bank(&ctx, 8, Codebook::Mul1, 0x4f4f_4e4b);
    let (jobs, total) = table(&[(1, 5), (2, 4)]);
    let x = rows_of(&mut rng, 6, K);
    let mut input: Vec<u32> = (0..total as u32).map(|r| r % 6).collect();
    let output: Vec<u32> = (0..total as u32).rev().collect();
    input[3] = 6; // one past the rows of x
    let bad_in = run_listed(
        rig,
        &bank,
        &jobs,
        &x,
        &Lists {
            input: Some(&input),
            output: Some(&output),
            x_rows: 6,
            out_rows: total,
            shift: 0,
        },
    )
    .expect("call");
    assert_eq!(bad_in.flag & 1, 1, "an input entry past x flags the call");
    input[3] = 3;
    let mut bad_out = output.clone();
    bad_out[2] = total as u32; // one past the rows of out
    let run = run_listed(
        rig,
        &bank,
        &jobs,
        &x,
        &Lists {
            input: Some(&input),
            output: Some(&bad_out),
            x_rows: 6,
            out_rows: total,
            shift: 0,
        },
    )
    .expect("call");
    assert_eq!(run.flag & 1, 1, "an output entry past out flags the call");
    // The dropped row wrote nothing: with a 9-row table and one entry gone, one row keeps its fill.
    let kept = (0..total)
        .filter(|&r| {
            run.out[r * N..(r + 1) * N]
                .iter()
                .all(|v| v.to_bits() == canary())
        })
        .count();
    assert_eq!(
        kept, 1,
        "exactly the row nothing landed in keeps the canary"
    );
}

#[test]
fn a_short_or_misaligned_row_list_is_refused_before_any_launch() {
    if !testkit::device_present("EXL3 job table row list shape") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5348_4f52);
    let bank = bank(&ctx, 8, Codebook::Mul1, 0x5348_4e4b);
    let (jobs, total) = table(&[(1, 5), (2, 4)]);
    let x = rows_of(&mut rng, total, K);
    let list: Vec<u32> = (0..total as u32).collect();
    fn input_only(list: &[u32], total: usize, shift: usize) -> Lists<'_> {
        Lists {
            input: Some(list),
            output: None,
            x_rows: total,
            out_rows: total,
            shift,
        }
    }
    fn output_only(list: &[u32], total: usize, shift: usize) -> Lists<'_> {
        Lists {
            input: None,
            output: Some(list),
            x_rows: total,
            out_rows: total,
            shift,
        }
    }
    let whole =
        run_listed(rig, &bank, &jobs, &x, &input_only(&list, total, 0)).expect("a whole list runs");
    assert_eq!(whole.flag, 0);
    // One entry short: the last table row would read past the list.
    let short = run_listed(
        rig,
        &bank,
        &jobs,
        &x,
        &input_only(&list[..total - 1], total, 0),
    );
    assert!(short.is_err(), "a list one entry short is refused");
    // A list that starts 2 bytes into its allocation is not 4-byte aligned.
    let skewed = run_listed(rig, &bank, &jobs, &x, &input_only(&list, total, 2));
    assert!(skewed.is_err(), "a misaligned list is refused");
    // The same two refusals for the output list.
    let short = run_listed(
        rig,
        &bank,
        &jobs,
        &x,
        &output_only(&list[..total - 1], total, 0),
    );
    assert!(short.is_err(), "an output list one entry short is refused");
    let skewed = run_listed(rig, &bank, &jobs, &x, &output_only(&list, total, 2));
    assert!(skewed.is_err(), "a misaligned output list is refused");
}

#[test]
fn input_rows_that_are_not_16_byte_aligned_run_the_shared_memory_kernel_and_agree() {
    if !testkit::device_present("EXL3 job table alignment") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x414c_4947);
    let bank = bank(&ctx, 8, Codebook::Mul1, 0x414c_4e4b);
    let (jobs, total) = table(&[(3, 1), (0, 64), (4, 33), (2, 9), (1, 17)]);
    let x = rows_of(&mut rng, total, K);
    // A start 4 bytes into the buffer is aligned for the entry but not for cp.async.
    let run = run_jobs_at(rig, &bank, &jobs, &x, ((N, 0), 4)).expect("unaligned call");
    assert_eq!(run.flag, 0, "flag");
    assert!(
        !run.registers,
        "an unaligned call must not take the register-decode kernel"
    );
    for job in &jobs {
        let (row0, rows) = (job.row0 as usize, job.rows as usize);
        let want = single(
            rig,
            &bank,
            job.member as usize,
            &x[row0 * K..(row0 + rows) * K],
        );
        for r in 0..rows {
            for c in 0..N {
                assert_eq!(
                    run.out[(row0 + r) * N + c].to_bits(),
                    want[r * N + c].to_bits(),
                    "member {} row {r} column {c}",
                    job.member
                );
            }
        }
    }
}

#[test]
fn a_job_outside_its_bounds_is_skipped_and_flags_the_call() {
    if !testkit::device_present("EXL3 job table bounds") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x4241_4453);
    let bank = bank(&ctx, 8, Codebook::Mul1, 0x4241_4e4b);
    let (jobs, total) = table(&[(1, 64), (2, 64), (0, 64)]);
    let x = rows_of(&mut rng, total, K);
    // A member past the bank, a negative first row (as u32 above i32::MAX), rows past the
    // buffers, and a job taller than max_rows: each is skipped by the device and sets the flag.
    for (name, bad) in [
        (
            "member",
            Exl3Job {
                member: MEMBERS as u32,
                ..jobs[1]
            },
        ),
        (
            "row0",
            Exl3Job {
                row0: u32::MAX,
                ..jobs[1]
            },
        ),
        (
            "past the rows",
            Exl3Job {
                row0: total as u32 - 32,
                ..jobs[1]
            },
        ),
    ] {
        let mut table = jobs.clone();
        table[1] = bad;
        let run = run_jobs(rig, &bank, &table, &x, (N, 0)).expect("jobs");
        assert_eq!(run.flag & 1, 1, "{name}: the flag");
        for index in [0usize, 2] {
            let job = jobs[index];
            let (row0, rows) = (job.row0 as usize, job.rows as usize);
            let want = single(
                rig,
                &bank,
                job.member as usize,
                &x[row0 * K..(row0 + rows) * K],
            );
            let got = &run.out[row0 * N..(row0 + rows) * N];
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(g, w)| g.to_bits() == w.to_bits()),
                "{name}: job {index}"
            );
        }
        // The skipped job wrote nothing of its own rows (rows 64..128 hold the canary), unless
        // the bad entry still points at them.
        if name == "member" {
            assert!(
                run.out[64 * N..128 * N]
                    .iter()
                    .all(|v| v.to_bits() == canary()),
                "{name}: the skipped job's rows"
            );
        }
    }
    let tall = [
        Exl3Job {
            member: 0,
            row0: 0,
            rows: 64,
        },
        Exl3Job {
            member: 1,
            row0: 64,
            rows: 64,
        },
    ];
    let (jobs_buf, rows) = (Exl3Job::table(&tall), 128usize);
    let x = rows_of(&mut rng, rows, K);
    let x_buf = testkit::upload_f32(&ctx, &x);
    let xh = testkit::upload_filled(&ctx, 0, rows * K * 4);
    let z = testkit::upload_filled(&ctx, 0, rows * N * 4);
    let out = testkit::upload_filled(&ctx, CANARY, rows * N * 4);
    let flag = testkit::upload_filled(&ctx, 0, 4);
    let table_buf = testkit::upload_bytes(&ctx, &jobs_buf);
    ctx.exl3_jobs(
        &stream,
        Exl3JobsCall {
            trellis: bank.trellis.range(0, bank.trellis.bytes()).unwrap(),
            suh: bank.scales.range(0, bank.scales.bytes()).unwrap(),
            svh: bank
                .scales
                .range(K * 4, bank.scales.bytes() - K * 4)
                .unwrap(),
            trellis_stride: packed_elements(8),
            scale_stride: K + N,
            members: MEMBERS,
            x: x_buf.range(0, rows * K * 4).unwrap(),
            xh: xh.range(0, rows * K * 4).unwrap(),
            z: z.range(0, rows * N * 4).unwrap(),
            out: out.range(0, rows * N * 4).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            jobs: table_buf.range(0, jobs_buf.len()).unwrap(),
            jobs_count: 2,
            max_rows: 32,
            total_rows: rows,
            out_ld: N,
            out_col: 0,
            k: K,
            n: N,
            bits_twice: 8,
            book: Codebook::Mul1,
            in_rows: None,
            out_rows: None,
            x_rows: rows,
            out_total_rows: rows,
        },
    )
    .expect("launch");
    stream.synchronize().expect("drain");
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 1u32.to_le_bytes());
    assert!(
        testkit::read_bytes(&ctx, &out, rows * N * 4)
            .iter()
            .all(|b| *b == CANARY)
    );
}

#[test]
fn the_wrapper_refuses_bad_strides_short_ranges_and_overlaps_before_any_launch() {
    if !testkit::device_present("EXL3 job table refusals") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let bank = bank(&ctx, 8, Codebook::Mul1, 7);
    let (jobs, total) = table(&[(0, 16), (1, 16)]);
    let table_bytes = Exl3Job::table(&jobs);
    let x = testkit::upload_f32(&ctx, &vec![0.5; total * K]);
    let xh = testkit::upload_filled(&ctx, 0, total * K * 4);
    let z = testkit::upload_filled(&ctx, 0, total * N * 4);
    let out = testkit::upload_filled(&ctx, 0, total * N * 4);
    let flag = testkit::upload_filled(&ctx, 0, 4);
    let table_buf = testkit::upload_bytes(&ctx, &table_bytes);
    let base = || Exl3JobsCall {
        trellis: bank.trellis.range(0, bank.trellis.bytes()).unwrap(),
        suh: bank.scales.range(0, bank.scales.bytes()).unwrap(),
        svh: bank
            .scales
            .range(K * 4, bank.scales.bytes() - K * 4)
            .unwrap(),
        trellis_stride: packed_elements(8),
        scale_stride: K + N,
        members: MEMBERS,
        x: x.range(0, total * K * 4).unwrap(),
        xh: xh.range(0, total * K * 4).unwrap(),
        z: z.range(0, total * N * 4).unwrap(),
        out: out.range(0, total * N * 4).unwrap(),
        flag: flag.range(0, 4).unwrap(),
        jobs: table_buf.range(0, table_bytes.len()).unwrap(),
        jobs_count: 2,
        max_rows: 16,
        total_rows: total,
        out_ld: N,
        out_col: 0,
        k: K,
        n: N,
        bits_twice: 8,
        book: Codebook::Mul1,
        in_rows: None,
        out_rows: None,
        x_rows: total,
        out_total_rows: total,
    };
    assert!(ctx.exl3_jobs(&stream, base()).is_ok());
    type Edit = fn(&mut Exl3JobsCall<'_>);
    let edits: [(&str, Edit); 9] = [
        ("trellis stride", |c| c.trellis_stride -= 1),
        ("scale stride", |c| c.scale_stride = K),
        ("members past the trellis", |c| c.members = MEMBERS + 1),
        ("no jobs", |c| c.jobs_count = 0),
        ("too many jobs", |c| c.jobs_count = 4097),
        ("rows over 256", |c| c.max_rows = 257),
        ("short table", |c| c.jobs_count = 3),
        ("window past ld", |c| c.out_col = 8),
        ("bad geometry", |c| c.k = 100),
    ];
    for (name, edit) in edits {
        let mut call = base();
        edit(&mut call);
        assert!(
            matches!(ctx.exl3_jobs(&stream, call), Err(EngError::BadRequest(_))),
            "{name} must be refused"
        );
    }
    // The rotated rows are F16: one word short of rows * k * 2 bytes is refused, exactly that is
    // accepted.
    let mut short = base();
    short.xh = xh.range(0, total * K * 2 - 4).unwrap();
    assert!(matches!(
        ctx.exl3_jobs(&stream, short),
        Err(EngError::BadRequest(_))
    ));
    let mut exact = base();
    exact.xh = xh.range(0, total * K * 2).unwrap();
    ctx.exl3_jobs(&stream, exact)
        .expect("exactly rows * k * 2 bytes of rotated input is accepted");
    // The output inside the input rows.
    let both = testkit::upload_filled(&ctx, 0, total * N.max(K) * 4);
    let mut aliased = base();
    aliased.x = both.range(0, total * K * 4).unwrap();
    aliased.out = both.range(0, total * N * 4).unwrap();
    assert!(matches!(
        ctx.exl3_jobs(&stream, aliased),
        Err(EngError::BadRequest(_))
    ));
    stream.synchronize().expect("drain");
}
