//! The warp-wide input rotation of the job-table projection against the one-thread-per-value
//! kernel it replaces: the rotated rows `xh` must be byte for byte the same, on every cell of a
//! grid of K, row heights, row lists, refused and unordered tables, and the awkward values (zeros,
//! non-finite inputs, subnormal and overflowing scales, products whose fused evaluation differs).
//! The whole pipeline (rotate, project, output rotate) must give the same output bytes with the
//! switch off and on. A host reference of the arithmetic keeps the comparison honest, and the
//! mutation tests (feature `cuda-exl3-rotate-mutants`) show that each named mistake turns the
//! comparison red.

use super::{
    Bank, CANARY, Codebook, CudaContext, DeviceStream, EngError, Exl3Job, Exl3JobsCall, MEMBERS, N,
    bank_of, packed_elements_of, rows_of, table, testkit,
};

mod pair;

/// The scale of the butterfly, `rsqrtf(128.0f)` of the kernels, as the host reference needs it.
/// This is the immediate nvcc folds the expression to (checked in the SASS of both kernels), so
/// the host reference is coupled to that compiler behavior: a toolchain that emits a hardware
/// reciprocal square root instead would move the reference by an ulp.
const ROTATE_SCALE: f32 = f32::from_bits(0x3db5_04f3);

/// How the reference evaluates the products of the first butterfly step: as separate roundings
/// (what the kernels do), or fused with the sum that follows (what a compiler may do to an
/// unpinned product).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fuse {
    Never,
    FirstProduct,
    SecondProduct,
}

/// The F16 bits and the non-finite flag of one row, by the arithmetic of the kernels in f32.
fn reference_row(x: &[f32], suh: &[f32], fuse: Fuse) -> (Vec<u16>, bool) {
    let mut flagged = false;
    let mut bits = Vec::with_capacity(x.len());
    for (xs, ss) in x.chunks(128).zip(suh.chunks(128)) {
        let mut v = [0.0f32; 128];
        for ((slot, x), s) in v.iter_mut().zip(xs).zip(ss) {
            *slot = x * s;
            if !slot.is_finite() {
                flagged = true;
                *slot = 0.0;
            }
        }
        let mut step = 1;
        if fuse != Fuse::Never {
            let p = v;
            for q in (0..128).step_by(4) {
                let (a, b, c, d) = match fuse {
                    Fuse::FirstProduct => (
                        xs[q].mul_add(ss[q], p[q + 1]),
                        xs[q].mul_add(ss[q], -p[q + 1]),
                        xs[q + 2].mul_add(ss[q + 2], p[q + 3]),
                        xs[q + 2].mul_add(ss[q + 2], -p[q + 3]),
                    ),
                    _ => (
                        xs[q + 1].mul_add(ss[q + 1], p[q]),
                        (-xs[q + 1]).mul_add(ss[q + 1], p[q]),
                        xs[q + 3].mul_add(ss[q + 3], p[q + 2]),
                        (-xs[q + 3]).mul_add(ss[q + 3], p[q + 2]),
                    ),
                };
                v[q..q + 4].copy_from_slice(&[a, b, c, d]);
            }
            step = 2;
        }
        network_from(&mut v, step);
        for value in v {
            let scaled = value * ROTATE_SCALE;
            if scaled.is_finite() {
                bits.push(testkit::f16_bits(scaled));
            } else {
                flagged = true;
                bits.push(0);
            }
        }
    }
    (bits, flagged)
}

/// The butterfly steps `first_step, 2 * first_step, .. 64` over one group, as the kernels run them:
/// the element with the step's bit set computes `other - own`, the other `own + other`.
fn network_from(v: &mut [f32; 128], first_step: usize) {
    let mut step = first_step;
    while step < 128 {
        let before = *v;
        for e in 0..128 {
            v[e] = if e & step != 0 {
                before[e ^ step] - before[e]
            } else {
                before[e] + before[e ^ step]
            };
        }
        step *= 2;
    }
}

/// The warp-wide layout on the host: lane `l` holds elements `4l..4l+4`; the steps at distance 1
/// and 2 run inside a lane, the steps at distance 4 to 64 are exchanges with the partner lane
/// (`l` xor `step / 4`), where the lane whose bit is set adds the negation of its own value to
/// the partner's (a subtraction).
fn warp_layout(group: &[f32; 128]) -> [f32; 128] {
    let mut lanes: Vec<[f32; 4]> = group.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
    for lane in &mut lanes {
        let [a, b, c, d] = *lane;
        *lane = [a + b, a - b, c + d, c - d];
        let [a, b, c, d] = *lane;
        *lane = [a + c, b + d, a - c, b - d];
    }
    for bit in 0..5 {
        let before = lanes.clone();
        for (l, lane) in lanes.iter_mut().enumerate() {
            for i in 0..4 {
                let own = before[l][i];
                let other = before[l ^ (1 << bit)][i];
                let signed = if l & (1 << bit) != 0 { -own } else { own };
                lane[i] = signed + other;
            }
        }
    }
    let mut out = [0.0f32; 128];
    for (l, lane) in lanes.iter().enumerate() {
        out[4 * l..4 * l + 4].copy_from_slice(lane);
    }
    out
}

/// A pair `(x0, x1)` whose products with `(s0, s1)` nearly cancel, so that fusing either product
/// with the sum changes the F16 result of a row that holds only this pair.
fn trap_pair(rng: &mut testkit::Rng, s0: f32, s1: f32) -> (f32, f32) {
    for _ in 0..200_000 {
        let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
        let x0 = sign * (2048.0 + rng.unit() as f32 * 4096.0);
        let nominal = -(x0 * s0) / s1;
        let x1 = f32::from_bits((nominal.to_bits() as i64 + rng.below(9) as i64 - 4) as u32);
        let mut x = [0.0f32; 128];
        let mut suh = [1.0f32; 128];
        (x[0], x[1], suh[0], suh[1]) = (x0, x1, s0, s1);
        let plain = reference_row(&x, &suh, Fuse::Never).0;
        if [Fuse::FirstProduct, Fuse::SecondProduct]
            .into_iter()
            .all(|fuse| reference_row(&x, &suh, fuse).0 != plain)
        {
            return (x0, x1);
        }
    }
    panic!("no trap pair for {s0} and {s1}");
}

/// What the scales of member `member` of the awkward bank are: subnormal, mixed extremes, powers
/// of two, ordinary values that the trap rows are built against, or ordinary values.
fn suh_of(member: usize, k: usize, rng: &mut testkit::Rng) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
            match member {
                1 => sign * (1.0e-41 + rng.unit() as f32 * 9.0e-39),
                2 => match i % 4 {
                    0 => 1.0e-20,
                    1 => 1.0e20,
                    2 => 0.0,
                    _ => sign * 3.0,
                },
                3 => sign * 2.0f32.powi((rng.below(9) as i32) - 4),
                _ => 0.5 + rng.unit() as f32,
            }
        })
        .collect()
}

/// A host copy of the scales that a [`Bank`] holds, `[suh | svh]` per member.
struct Scales {
    values: Vec<f32>,
    k: usize,
}

impl Scales {
    fn suh(&self, member: usize) -> &[f32] {
        let at = member * (self.k + N);
        &self.values[at..at + self.k]
    }
}

/// The scales of a bank, `[suh | svh]` per member, on the host.
fn host_scales(k: usize, awkward: bool, seed: u64) -> Scales {
    let mut rng = testkit::Rng::new(seed);
    let mut values = Vec::new();
    for member in 0..MEMBERS {
        values.extend(if awkward {
            suh_of(member, k, &mut rng)
        } else {
            suh_of(0, k, &mut rng)
        });
        values.extend((0..N).map(|_| 0.5 + rng.unit() as f32));
    }
    Scales { values, k }
}

fn scaled_bank(ctx: &CudaContext, k: usize, awkward: bool, seed: u64) -> (Bank, Scales) {
    let scales = host_scales(k, awkward, seed);
    let bank = bank_of(ctx, (k, N), (8, Codebook::Mul1), seed);
    let bank = Bank {
        scales: testkit::upload_f32(ctx, &scales.values),
        ..bank
    };
    (bank, scales)
}

/// One job-table call to run: its table, the rows of `x`, and an optional row list.
struct Case {
    name: String,
    awkward: bool,
    /// Members the call declares: the bank's, or 1 for a single matrix (strides ignored).
    members: usize,
    jobs: Vec<Exl3Job>,
    total_rows: usize,
    max_rows: usize,
    x: Vec<f32>,
    in_rows: Option<Vec<u32>>,
}

fn job(member: u32, row0: u32, rows: u32) -> Exl3Job {
    Exl3Job { member, row0, rows }
}

/// The rows of the awkward-values case, one job of `ROWS_PER_JOB` rows for each member.
const ROWS_PER_JOB: usize = 12;

fn awkward_rows(rng: &mut testkit::Rng, scales: &Scales) -> Vec<f32> {
    let k = scales.k;
    let (trap_a, trap_b) = {
        let suh = scales.suh(4);
        (
            trap_pair(rng, suh[0], suh[1]),
            trap_pair(rng, suh[2], suh[3]),
        )
    };
    let mut rows = Vec::new();
    for _ in 0..MEMBERS {
        for kind in 0..ROWS_PER_JOB {
            let mut row: Vec<f32> = (0..k).map(|_| rng.normal() as f32 * 2.0).collect();
            match kind {
                1 => row.fill(0.0),
                2 => row[5] = f32::NAN,
                3 => {
                    row[3] = f32::INFINITY;
                    row[k - 51] = f32::NEG_INFINITY;
                }
                4 => row.fill(3.0e37),
                5 => row
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, v)| *v = if i % 2 == 0 { 1.0e30 } else { -1.0e30 }),
                6 => row.iter_mut().for_each(|v| *v *= 1.0e-30),
                7 => row.iter_mut().for_each(|v| *v = *v * 1.0e-42 + 1.0e-43),
                8 => row
                    .iter_mut()
                    .for_each(|v| *v = 2.0f32.powi(rng.below(40) as i32 - 20)),
                9..=11 => {
                    row.fill(0.0);
                    if kind != 10 {
                        (row[0], row[1]) = trap_a;
                    }
                    if kind != 9 {
                        (row[2], row[3]) = trap_b;
                    }
                }
                _ => {}
            }
            rows.extend(row);
        }
    }
    rows
}

fn suite(k: usize, scales: &Scales, rng: &mut testkit::Rng) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut plain = |name: &str, shape: &[(u32, u32)], rng: &mut testkit::Rng| {
        let (jobs, total_rows) = table(shape);
        cases.push(Case {
            name: name.into(),
            awkward: false,
            members: MEMBERS,
            max_rows: shape.iter().map(|&(_, r)| r as usize).max().unwrap(),
            x: rows_of(rng, total_rows, k),
            in_rows: None,
            jobs,
            total_rows,
        });
    };
    plain(
        "mixed heights",
        &[(3, 1), (0, 70), (4, 144), (2, 9), (1, 64), (4, 33)],
        rng,
    );
    plain("single row", &[(2, 1)], rng);
    plain("rows off the tile", &[(0, 5), (1, 3), (2, 7), (3, 1)], rng);

    // The expert-sorted table of a routed chunk, reading 37 token rows through a list with
    // repeated entries and one entry past the rows of x.
    let (jobs, total_rows) = table(&[(0, 41), (2, 1), (3, 70), (0, 3), (4, 13)]);
    let tokens = 37usize;
    let list = (0..total_rows as u32)
        .map(|r| match r {
            _ if r % 23 == 7 => tokens as u32 + 3,
            _ if r % 11 == 5 => 7,
            _ => (r * 7 + 3) % tokens as u32,
        })
        .collect();
    cases.push(Case {
        name: "gather list".into(),
        awkward: false,
        members: MEMBERS,
        max_rows: 70,
        x: rows_of(rng, tokens, k),
        in_rows: Some(list),
        jobs,
        total_rows,
    });
    // The same rows read back to front through a permutation: no repeats, nothing refused.
    let (jobs, total_rows) = table(&[(1, 33), (4, 31)]);
    cases.push(Case {
        name: "gather permutation".into(),
        awkward: false,
        members: MEMBERS,
        max_rows: 33,
        x: rows_of(rng, total_rows, k),
        in_rows: Some((0..total_rows as u32).rev().collect()),
        jobs,
        total_rows,
    });
    // Rows 15 to 39 belong to no job: they keep their bytes.
    cases.push(Case {
        name: "rows no job holds".into(),
        awkward: false,
        members: MEMBERS,
        max_rows: 10,
        x: rows_of(rng, 40, k),
        in_rows: None,
        jobs: vec![job(1, 0, 10), job(3, 10, 5)],
        total_rows: 40,
    });
    // Jobs listed out of row order.
    cases.push(Case {
        name: "unordered table".into(),
        awkward: false,
        members: MEMBERS,
        max_rows: 18,
        x: rows_of(rng, 40, k),
        in_rows: None,
        jobs: vec![job(2, 30, 10), job(0, 0, 12), job(1, 12, 18)],
        total_rows: 40,
    });
    // A member outside the bank, a negative first row, a job taller than the grid was sized
    // for and a job past the table are skipped and flag the call; the others run.
    cases.push(Case {
        name: "refused jobs".into(),
        awkward: false,
        members: MEMBERS,
        max_rows: 64,
        x: rows_of(rng, 100, k),
        in_rows: None,
        jobs: vec![
            job(0, 0, 10),
            job(9, 10, 5),
            job(1, 0xffff_fff0, 3),
            job(2, 15, 70),
            job(3, 300, 3),
            job(4, 90, 6),
        ],
        total_rows: 100,
    });
    // A sorted table of many short jobs: the owner of each row is found by bisection.
    let mut shape = Vec::new();
    for index in 0..120u32 {
        shape.push((index % MEMBERS as u32, 1 + rng.below(5) as u32));
    }
    let (jobs, total_rows) = table(&shape);
    cases.push(Case {
        name: "sorted table of many jobs".into(),
        awkward: false,
        members: MEMBERS,
        max_rows: 5,
        x: rows_of(rng, total_rows, k),
        in_rows: None,
        jobs,
        total_rows,
    });
    // A single matrix (members = 1): the strides are ignored and every job reads member 0.
    let (jobs, total_rows) = table(&[(0, 33), (0, 1), (0, 40)]);
    cases.push(Case {
        name: "single member bank".into(),
        awkward: false,
        members: 1,
        max_rows: 40,
        x: rows_of(rng, total_rows, k),
        in_rows: None,
        jobs,
        total_rows,
    });
    // More table entries than the grid has threads (one block of 128 for one row): one valid job
    // and 299 entries that are refused (a member outside the bank, a negative first row, no rows),
    // which the threads of the grid must all flag.
    let mut jobs = vec![job(0, 0, 1)];
    jobs.extend((1..300u32).map(|n| match n % 3 {
        0 => job(MEMBERS as u32 + n, 0, 1),
        1 => job(1, 0x8000_0000 + n, 1),
        _ => job(2, 0, 0),
    }));
    cases.push(Case {
        name: "more entries than threads".into(),
        awkward: false,
        members: MEMBERS,
        max_rows: 1,
        x: rows_of(rng, 1, k),
        in_rows: None,
        jobs,
        total_rows: 1,
    });
    let awkward = awkward_rows(rng, scales);
    cases.push(Case {
        name: "awkward values".into(),
        awkward: true,
        members: MEMBERS,
        max_rows: ROWS_PER_JOB,
        x: awkward.clone(),
        in_rows: None,
        jobs: (0..MEMBERS as u32)
            .map(|m| job(m, m * ROWS_PER_JOB as u32, ROWS_PER_JOB as u32))
            .collect(),
        total_rows: MEMBERS * ROWS_PER_JOB,
    });
    // The same rows in calls of their own, so that one kind of flag cannot hide behind another:
    // rows whose products stay finite and whose sums overflow (flagged at the scaled value only;
    // the members with ordinary scales), and NaN or infinite inputs (flagged at the product only).
    // The extreme but finite kinds (zeros, tiny, subnormal, the cancelling trap rows) get a call of
    // their own too: no flag may be raised for them. The call runs the projection and the output
    // rotation as well, which flag an F16 infinity in xh, so the members with huge scales and the
    // powers-of-two kind (rotated values far above 65,504) stay out.
    for (name, kinds, members) in [
        (
            "awkward overflow rows only",
            &[4usize][..],
            &[0usize, 4][..],
        ),
        (
            "awkward non-finite rows only",
            &[2, 3][..],
            &[0, 1, 2, 3, 4][..],
        ),
        (
            "awkward finite rows only",
            &[0, 1, 6, 7, 9, 10, 11][..],
            &[0, 1, 4][..],
        ),
    ] {
        let mut x = Vec::new();
        let mut jobs = Vec::new();
        for (slot, &member) in members.iter().enumerate() {
            jobs.push(job(
                member as u32,
                (slot * kinds.len()) as u32,
                kinds.len() as u32,
            ));
            for &kind in kinds {
                let row = member * ROWS_PER_JOB + kind;
                x.extend_from_slice(&awkward[row * k..(row + 1) * k]);
            }
        }
        cases.push(Case {
            name: name.into(),
            awkward: true,
            members: MEMBERS,
            max_rows: kinds.len(),
            x,
            in_rows: None,
            jobs,
            total_rows: members.len() * kinds.len(),
        });
    }
    cases
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path {
    /// The one-thread-per-value kernel.
    PerValue,
    /// The warp-wide kernel, as requested.
    Warp,
    /// The entry that reads the environment (`FIFI_EXL3_JOBS_WARP_ROTATE`,
    /// `FIFI_EXL3_JOBS_FUSED_OUT`).
    Env,
    /// A deliberately wrong variant of the warp-wide kernel, alone.
    #[cfg_attr(not(feature = "cuda-exl3-rotate-mutants"), allow(dead_code))]
    Mutant(i32),
}

struct Rotated {
    xh: Vec<u8>,
    out: Vec<u8>,
    flag: u32,
    /// Whether the call meets the alignment rules of the warp-wide kernel.
    usable: bool,
}

/// One call of `case` on `path`; `shift` moves the start of x and of xh by that many bytes into
/// their buffers. The unwritten bytes of xh and out are canary-filled.
fn rotate(
    (ctx, stream): (&CudaContext, &DeviceStream),
    bank: &Bank,
    case: &Case,
    (path, fused_out, shift): (Path, bool, (usize, usize)),
) -> Result<Rotated, EngError> {
    let (k, n) = (bank.k, bank.n);
    let mut x_bytes = vec![0u8; shift.0];
    x_bytes.extend(case.x.iter().flat_map(|value| value.to_le_bytes()));
    let x_buf = testkit::upload_bytes(ctx, &x_bytes);
    let xh_len = case.total_rows * k * 2;
    let xh = testkit::upload_filled(ctx, CANARY, xh_len + shift.1);
    let z = testkit::upload_filled(ctx, 0, case.total_rows * n * 4);
    let out_len = case.total_rows * n * 4;
    let out = testkit::upload_filled(ctx, CANARY, out_len);
    let flag = testkit::upload_filled(ctx, 0, 4);
    let jobs_bytes = Exl3Job::table(&case.jobs);
    let jobs = testkit::upload_bytes(ctx, &jobs_bytes);
    let list = case.in_rows.as_ref().map(|rows| {
        let bytes: Vec<u8> = rows.iter().flat_map(|row| row.to_le_bytes()).collect();
        testkit::upload_bytes(ctx, &bytes)
    });
    let call = Exl3JobsCall {
        trellis: bank.trellis.range(0, bank.trellis.bytes())?,
        suh: bank.scales.range(0, bank.scales.bytes())?,
        svh: bank.scales.range(k * 4, bank.scales.bytes() - k * 4)?,
        trellis_stride: packed_elements_of(k, n, bank.bits_twice),
        scale_stride: k + n,
        members: case.members,
        x: x_buf.range(shift.0, case.x.len() * 4)?,
        xh: xh.range(shift.1, xh_len)?,
        z: z.range(0, case.total_rows * n * 4)?,
        out: out.range(0, out_len)?,
        flag: flag.range(0, 4)?,
        jobs: jobs.range(0, jobs_bytes.len())?,
        jobs_count: case.jobs.len(),
        max_rows: case.max_rows,
        total_rows: case.total_rows,
        out_ld: n,
        out_col: 0,
        k,
        n,
        bits_twice: bank.bits_twice,
        book: bank.book,
        in_rows: list
            .as_ref()
            .map(|buffer| buffer.range(0, case.total_rows * 4))
            .transpose()?,
        out_rows: None,
        x_rows: case.x.len() / k,
        out_total_rows: case.total_rows,
    };
    let usable = ctx.exl3_jobs_wide_input_usable(&call, true)?;
    match path {
        Path::PerValue => ctx.exl3_jobs_modes(stream, call, (fused_out, false))?,
        Path::Warp => ctx.exl3_jobs_modes(stream, call, (fused_out, true))?,
        Path::Env => ctx.exl3_jobs(stream, call)?,
        #[cfg(feature = "cuda-exl3-rotate-mutants")]
        Path::Mutant(mutant) => ctx.exl3_jobs_rotate_mutant(stream, call, mutant)?,
        #[cfg(not(feature = "cuda-exl3-rotate-mutants"))]
        Path::Mutant(_) => unreachable!("mutants are built only with their feature"),
    }
    stream.synchronize()?;
    Ok(Rotated {
        xh: testkit::read_bytes(ctx, &xh, xh_len + shift.1),
        out: testkit::read_bytes(ctx, &out, out_len),
        flag: u32::from_le_bytes(testkit::read_bytes(ctx, &flag, 4).try_into().unwrap()),
        usable,
    })
}

fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or((a.len() != b.len()).then_some(a.len().min(b.len())))
}

/// What differs between two runs of a case, or `None` when xh, the flag and (with `with_out`) the
/// output bytes are the same.
fn difference(old: &Rotated, new: &Rotated, with_out: bool) -> Option<String> {
    if let Some(at) = first_difference(&old.xh, &new.xh) {
        return Some(format!("xh differs at byte {at}"));
    }
    if old.flag != new.flag {
        return Some(format!("flag {} against {}", old.flag, new.flag));
    }
    if with_out && let Some(at) = first_difference(&old.out, &new.out) {
        return Some(format!("out differs at byte {at}"));
    }
    None
}

fn device() -> Option<(CudaContext, DeviceStream)> {
    if !testkit::device_present("EXL3 warp input rotation") {
        return None;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    Some((ctx, stream))
}

/// The limits `jobs_views` enforces on a call, checked on the host against a case: the table and
/// the row counts, the sizes of the row buffers, the row list and the member count.
fn check_case_bounds(case: &Case, k: usize) {
    let name = &case.name;
    assert!(k >= 128 && k.is_multiple_of(128), "{name}: K {k}");
    assert!(
        (1..=4096).contains(&case.jobs.len()),
        "{name}: {} jobs",
        case.jobs.len()
    );
    assert!(
        (1..=256).contains(&case.max_rows),
        "{name}: max_rows {}",
        case.max_rows
    );
    assert!(
        (1..=65536).contains(&case.total_rows),
        "{name}: total_rows {}",
        case.total_rows
    );
    assert!(
        case.total_rows * k <= 1 << 32 && case.total_rows * N <= 1 << 32,
        "{name}"
    );
    assert!((1..=MEMBERS).contains(&case.members), "{name}: members");
    assert!(case.x.len().is_multiple_of(k), "{name}: x is whole rows");
    let x_rows = case.x.len() / k;
    assert!((1..=65536).contains(&x_rows), "{name}: x rows {x_rows}");
    match &case.in_rows {
        None => assert_eq!(
            x_rows, case.total_rows,
            "{name}: no list, x holds the table rows"
        ),
        Some(list) => assert_eq!(list.len(), case.total_rows, "{name}: one entry a table row"),
    }
    assert!(
        case.jobs
            .iter()
            .any(|job| (job.row0 as i32) >= 0 && job.rows >= 1),
        "{name}: at least one job can run"
    );
}

/// The largest rotated value of `case` in F16, by the host reference; infinite when any value
/// overflows F16 (the projection of such an xh flags the call, which the rotation does not).
fn largest_rotated(case: &Case, scales: &Scales) -> f32 {
    let k = scales.k;
    let mut largest = 0.0f32;
    for job in &case.jobs {
        for row in job.row0 as usize..(job.row0 + job.rows) as usize {
            let x = &case.x[row * k..(row + 1) * k];
            let (bits, _) = reference_row(x, scales.suh(job.member as usize), Fuse::Never);
            for bit in bits {
                largest = largest.max(testkit::f16_to_f32(bit).abs());
            }
        }
    }
    largest
}

#[test]
fn the_suites_build_on_the_host_and_every_case_meets_the_call_bounds() {
    // Host only: debug arithmetic on, no device. A panic while a case is built, or a case that
    // jobs_views would refuse, fails here before any lease.
    let mut rng = testkit::Rng::new(0x5741_525f);
    for k in GRID {
        let mut total = 0;
        for awkward in [false, true] {
            let scales = host_scales(k, awkward, 0x5741_5260 + k as u64);
            let mut cases = suite(k, &scales, &mut rng);
            total += cases.len();
            cases.retain(|case| case.awkward == awkward);
            assert_eq!(cases.len(), if awkward { 4 } else { 11 }, "k={k}");
            for case in &cases {
                check_case_bounds(case, k);
                // The finite case runs the whole call: its rotated values must stay inside F16, or
                // the projection of the infinity would flag it. The weights of a decoded tile are
                // below 4 in magnitude, so a bound of 65,504 on xh keeps every partial sum finite
                // in F32 for K up to 4,096 (65,504 * 4 * 4,096 is about 1e9).
                if case.name == "awkward finite rows only" {
                    let largest = largest_rotated(case, &scales);
                    assert!(largest < 65_504.0, "k={k}: largest rotated value {largest}");
                    assert!(largest > 0.0, "k={k}: the case must rotate something");
                }
            }
        }
        // `suite` builds every case each time; the ordinary bank keeps 11 of them, the extreme
        // one 4.
        assert_eq!(total, 2 * CASES_PER_K, "k={k}");
    }
}

/// K values of the grid: the three served shapes and the odd group counts (1, 2, 3 groups a row)
/// that leave the last warp of the grid with fewer than four groups.
const GRID: [usize; 6] = [128, 256, 384, 512, 2048, 4096];

/// Cases of one value of K: 11 on the ordinary bank, 4 on the awkward one.
const CASES_PER_K: usize = 15;

fn banks_and_cases(
    ctx: &CudaContext,
    k: usize,
    rng: &mut testkit::Rng,
) -> Vec<(Bank, Scales, Vec<Case>)> {
    [false, true]
        .into_iter()
        .map(|awkward| {
            let (bank, scales) = scaled_bank(ctx, k, awkward, 0x5741_5250 + k as u64);
            let mut cases = suite(k, &scales, rng);
            cases.retain(|case| case.awkward == awkward);
            (bank, scales, cases)
        })
        .collect()
}

#[test]
fn warp_rotation_is_byte_identical_to_the_per_value_kernel_over_the_grid() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5741_5250);
    let mut compared = 0;
    for k in GRID {
        for (bank, _, cases) in banks_and_cases(&ctx, k, &mut rng) {
            for case in &cases {
                let old = rotate(rig, &bank, case, (Path::PerValue, false, (0, 0))).expect("old");
                let new = rotate(rig, &bank, case, (Path::Warp, false, (0, 0))).expect("new");
                assert!(
                    new.usable,
                    "k={k} {}: the warp kernel must be taken",
                    case.name
                );
                assert_eq!(difference(&old, &new, false), None, "k={k} {}", case.name);
                compared += 1;
            }
        }
    }
    // 6 values of K, 11 plain cases on the ordinary bank and 4 on the awkward bank.
    assert_eq!(compared, 6 * CASES_PER_K);
}

#[test]
fn the_cases_exercise_what_they_name() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5741_5251);
    for (bank, _, cases) in banks_and_cases(&ctx, 512, &mut rng) {
        for case in &cases {
            let run = rotate(rig, &bank, case, (Path::Warp, false, (0, 0))).expect("run");
            let rows_written = |row: usize| {
                run.xh[row * 512 * 2..(row + 1) * 512 * 2]
                    .iter()
                    .any(|byte| *byte != CANARY)
            };
            match case.name.as_str() {
                "rows no job holds" => {
                    assert_eq!(run.flag, 0);
                    assert!((0..15).all(rows_written));
                    assert!((15..40).all(|row| !rows_written(row)));
                }
                "refused jobs" => {
                    assert_eq!(run.flag & 1, 1, "refused jobs flag the call");
                    assert!((0..10).all(rows_written));
                    assert!((10..90).all(|row| !rows_written(row)));
                    assert!((90..96).all(rows_written));
                }
                "gather list" => assert_eq!(run.flag & 1, 1, "an entry past x flags the call"),
                "more entries than threads" => {
                    assert_eq!(run.flag & 1, 1, "refused entries flag the call");
                    assert!(rows_written(0), "the one valid job runs");
                }
                "awkward overflow rows only" | "awkward non-finite rows only" => {
                    assert_eq!(run.flag & 1, 1, "{}", case.name);
                    assert!((0..case.total_rows).all(rows_written));
                }
                "awkward values" => {
                    assert_eq!(run.flag & 1, 1, "non-finite values flag the call");
                    assert!((0..case.total_rows).all(rows_written));
                }
                _ => {
                    assert_eq!(run.flag, 0, "{}", case.name);
                    assert!((0..case.total_rows).all(rows_written), "{}", case.name);
                }
            }
        }
    }
}

#[test]
fn the_per_value_kernel_matches_the_host_reference() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5741_5252);
    for k in [256, 2048] {
        let (bank, scales) = scaled_bank(&ctx, k, true, 0x5741_5253 + k as u64);
        let cases = suite(k, &scales, &mut rng);
        let case = cases
            .iter()
            .find(|case| case.awkward)
            .expect("awkward case");
        let run = rotate(rig, &bank, case, (Path::PerValue, false, (0, 0))).expect("run");
        for (row, x) in case.x.chunks(k).enumerate() {
            let member = row / ROWS_PER_JOB;
            let (want, _) = reference_row(x, scales.suh(member), Fuse::Never);
            let got: Vec<u16> = run.xh[row * k * 2..(row + 1) * k * 2]
                .chunks(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            assert_eq!(got, want, "k={k} row {row} (member {member})");
        }
    }
}

#[test]
fn the_trap_rows_separate_fused_from_separate_products() {
    // Host only: the awkward case must hold rows whose F16 result depends on whether the
    // products are fused with the first sums, in either direction.
    let mut rng = testkit::Rng::new(0x5741_5254);
    let k = 128;
    let mut values = Vec::new();
    for member in 0..MEMBERS {
        values.extend(suh_of(member, k, &mut rng));
        values.extend(std::iter::repeat_n(1.0f32, N));
    }
    let scales = Scales { values, k };
    let rows = awkward_rows(&mut rng, &scales);
    let suh = scales.suh(4);
    for kind in 9..=11 {
        let row = &rows[(4 * ROWS_PER_JOB + kind) * k..(4 * ROWS_PER_JOB + kind + 1) * k];
        let plain = reference_row(row, suh, Fuse::Never).0;
        for fuse in [Fuse::FirstProduct, Fuse::SecondProduct] {
            let fused = reference_row(row, suh, fuse).0;
            // Row kind 10 holds only the second pair and kind 9 only the first: each shows the
            // fusion of the products that the mutant and a compiler can pick.
            assert_ne!(fused, plain, "row kind {kind} under {fuse:?}");
        }
    }
}

#[test]
fn the_warp_layout_reproduces_the_butterfly_network_bit_for_bit() {
    // Host only: the register steps, the lane exchanges and the negate-and-add of the upper partner
    // give the bits of the element-wise network, including on cancelling and extreme values.
    let mut rng = testkit::Rng::new(0x5741_5259);
    for round in 0..2000 {
        let mut group = [0.0f32; 128];
        for (i, value) in group.iter_mut().enumerate() {
            *value = match (round % 4, i % 7) {
                (1, 0) => 0.0,
                (1, 1) => -0.0,
                (2, _) => (rng.normal() as f32) * 2.0f32.powi(rng.below(60) as i32 - 30),
                (3, _) => (rng.below(9) as f32) - 4.0,
                _ => rng.normal() as f32,
            };
        }
        let mut want = group;
        network_from(&mut want, 1);
        let got = warp_layout(&group);
        let same = want
            .iter()
            .zip(&got)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "round {round}");
    }
}

#[test]
fn the_host_reference_rotates_an_impulse_to_a_flat_row_of_the_scale() {
    // Row of zeros with one value: every output is +-value * scale, with the Hadamard sign.
    let mut x = [0.0f32; 128];
    x[0] = 1024.0;
    let (bits, flagged) = reference_row(&x, &[1.0; 128], Fuse::Never);
    assert!(!flagged);
    let expected = testkit::f16_bits(1024.0 * ROTATE_SCALE);
    assert!(bits.iter().all(|b| *b == expected));
    let mut x = [0.0f32; 128];
    x[1] = 1024.0;
    let (bits, _) = reference_row(&x, &[1.0; 128], Fuse::Never);
    assert_eq!(bits[0], expected);
    assert_eq!(bits[1], expected | 0x8000);
    // Non-finite input is zeroed and flagged.
    x[7] = f32::NAN;
    assert!(reference_row(&x, &[1.0; 128], Fuse::Never).1);
}

/// Whether the host reference flags any row of `case` (a non-finite product, or a non-finite
/// scaled value); the rows are read straight, the cases that use it carry no row list.
fn expected_flag(case: &Case, scales: &Scales) -> bool {
    let k = scales.k;
    case.jobs.iter().any(|job| {
        (job.row0 as usize..(job.row0 + job.rows) as usize).any(|row| {
            let x = &case.x[row * k..(row + 1) * k];
            reference_row(x, scales.suh(job.member as usize), Fuse::Never).1
        })
    })
}

#[test]
fn the_device_flag_follows_the_host_reference_on_every_awkward_case() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5741_525a);
    for k in [256, 2048] {
        let (bank, scales) = scaled_bank(&ctx, k, true, 0x5741_525b + k as u64);
        let mut cases = suite(k, &scales, &mut rng);
        cases.retain(|case| case.awkward);
        assert_eq!(cases.len(), 4);
        for case in &cases {
            let want = expected_flag(case, &scales);
            // Only the finite rows stay clear of the flag; the device must agree both ways.
            assert_eq!(
                want,
                case.name != "awkward finite rows only",
                "k={k} {}: the host reference",
                case.name
            );
            for path in [Path::PerValue, Path::Warp] {
                let run = rotate(rig, &bank, case, (path, false, (0, 0))).expect("run");
                assert_eq!(
                    run.flag & 1,
                    u32::from(want),
                    "k={k} {} {path:?}",
                    case.name
                );
            }
        }
    }
}

/// The variable that the child tests set, and the name of this process's role.
const CHILD_MARK: &str = "FIFI_ROTATE_TEST_CHILD";

/// In the parent process: runs the test `name` again in a child process of the test executable,
/// once for each environment of `runs`, and checks that each child passed on a device. Returns
/// true in a child, where the caller runs the body of the test. The switches are read once per
/// process, so each value needs a process of its own.
fn run_in_children(name: &str, runs: &[&[(&str, &str)]]) -> bool {
    if std::env::var_os(CHILD_MARK).is_some() {
        return true;
    }
    if !testkit::device_present("EXL3 rotation switches") {
        return false;
    }
    let path = format!(
        "{}::{name}",
        module_path!().split_once("::").expect("crate path").1
    );
    for env in runs {
        let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", &path, "--test-threads=1", "--nocapture"])
            .env_remove("FIFI_EXL3_JOBS_WARP_ROTATE")
            .env_remove("FIFI_EXL3_JOBS_PAIR_ROTATE")
            .env(CHILD_MARK, "1")
            .envs(env.iter().copied())
            .output()
            .expect("the child process");
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.status.success()
                && text.contains("test result: ok. 1 passed")
                && !text.contains("SKIPPED"),
            "child with {env:?} failed:\n{text}"
        );
    }
    false
}

const WARP_VAR: &str = "FIFI_EXL3_JOBS_WARP_ROTATE";
const PAIR_VAR: &str = "FIFI_EXL3_JOBS_PAIR_ROTATE";

fn runs_of<'a>(var: &'a str, values: &[&'a str]) -> Vec<Vec<(&'a str, &'a str)>> {
    values.iter().map(|value| vec![(var, *value)]).collect()
}

#[test]
fn a_bad_warp_switch_is_refused_by_the_session_check_and_by_the_launch() {
    let runs = runs_of(WARP_VAR, &["2", "on", "", "01"]);
    let refs: Vec<&[(&str, &str)]> = runs.iter().map(Vec::as_slice).collect();
    if !run_in_children(
        "a_bad_warp_switch_is_refused_by_the_session_check_and_by_the_launch",
        &refs,
    ) {
        return;
    }
    let (ctx, stream) = device().expect("a device");
    let refusal = ctx.check_single_gather().expect_err("the session check");
    assert!(format!("{refusal:?}").contains(WARP_VAR), "{refusal:?}");
    let (bank, scales) = scaled_bank(&ctx, 256, false, 0x5741_525c);
    let cases = suite(256, &scales, &mut testkit::Rng::new(0x5741_525d));
    let launch = rotate(
        (&ctx, &stream),
        &bank,
        &cases[0],
        (Path::Env, false, (0, 0)),
    );
    let refusal = launch.err().expect("the launch is refused");
    assert!(format!("{refusal:?}").contains(WARP_VAR), "{refusal:?}");
}

#[test]
fn a_bad_pair_switch_is_refused_by_the_session_check_and_the_enabled_query() {
    let runs = runs_of(PAIR_VAR, &["2", "on", "", "01"]);
    let refs: Vec<&[(&str, &str)]> = runs.iter().map(Vec::as_slice).collect();
    if !run_in_children(
        "a_bad_pair_switch_is_refused_by_the_session_check_and_the_enabled_query",
        &refs,
    ) {
        return;
    }
    let (ctx, _) = device().expect("a device");
    let refusal = ctx.check_single_gather().expect_err("the session check");
    assert!(format!("{refusal:?}").contains(PAIR_VAR), "{refusal:?}");
    let refusal = ctx.exl3_jobs_pair_enabled().expect_err("the enabled query");
    assert!(format!("{refusal:?}").contains(PAIR_VAR), "{refusal:?}");
}

#[test]
fn good_switch_values_pass_the_check_name_themselves_on_the_load_line_and_reach_the_launch() {
    let runs: Vec<Vec<(&str, &str)>> = vec![
        vec![(WARP_VAR, "1")],
        vec![(PAIR_VAR, "1")],
        vec![(WARP_VAR, "1"), (PAIR_VAR, "1")],
        vec![(WARP_VAR, "0"), (PAIR_VAR, "0")],
    ];
    let refs: Vec<&[(&str, &str)]> = runs.iter().map(Vec::as_slice).collect();
    if !run_in_children(
        "good_switch_values_pass_the_check_name_themselves_on_the_load_line_and_reach_the_launch",
        &refs,
    ) {
        return;
    }
    let (ctx, stream) = device().expect("a device");
    ctx.check_single_gather().expect("the session check");
    let on = |var: &str| std::env::var(var).is_ok_and(|value| value == "1");
    let line = ctx.single_gather_suffix();
    assert_eq!(
        line.contains(" exl3_jobs_warp_rotate=on"),
        on(WARP_VAR),
        "{line}"
    );
    assert_eq!(
        line.contains(" exl3_jobs_pair_rotate=on"),
        on(PAIR_VAR),
        "{line}"
    );
    assert_eq!(ctx.exl3_jobs_pair_enabled().expect("pair"), on(PAIR_VAR));
    // The entry that reads the environment gives the bytes of the one-thread-per-value kernel.
    let (bank, scales) = scaled_bank(&ctx, 512, false, 0x5741_525e);
    let cases = suite(512, &scales, &mut testkit::Rng::new(0x5741_525f));
    let rig = (&ctx, &stream);
    for case in &cases[..4] {
        let old = rotate(rig, &bank, case, (Path::PerValue, false, (0, 0))).expect("old");
        let env = rotate(rig, &bank, case, (Path::Env, false, (0, 0))).expect("env");
        assert_eq!(difference(&old, &env, true), None, "{}", case.name);
    }
}

#[test]
fn calls_the_warp_kernel_cannot_take_run_the_per_value_kernel_with_the_same_bytes() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5741_5255);
    let k = 512;
    let (bank, scales) = scaled_bank(&ctx, k, false, 0x5741_5256);
    let cases = suite(k, &scales, &mut rng);
    let case = &cases[0];
    // x 4 bytes in, xh 4 bytes in: not taken. x 16 and xh 8 bytes in: taken.
    for (shift, taken) in [
        ((4, 0), false),
        ((0, 4), false),
        ((16, 8), true),
        ((32, 16), true),
    ] {
        let old = rotate(rig, &bank, case, (Path::PerValue, false, shift)).expect("old");
        let new = rotate(rig, &bank, case, (Path::Warp, false, shift)).expect("new");
        assert_eq!(new.usable, taken, "shift {shift:?}");
        assert_eq!(difference(&old, &new, false), None, "shift {shift:?}");
    }
}

#[test]
fn the_whole_projection_is_byte_identical_with_the_warp_rotation_on_and_off() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5741_5257);
    let mut compared = 0;
    for k in [256, 512, 2048] {
        for (bank, _, cases) in banks_and_cases(&ctx, k, &mut rng) {
            for case in &cases {
                for fused_out in [false, true] {
                    let old =
                        rotate(rig, &bank, case, (Path::PerValue, fused_out, (0, 0))).expect("old");
                    let new =
                        rotate(rig, &bank, case, (Path::Warp, fused_out, (0, 0))).expect("new");
                    assert_eq!(
                        difference(&old, &new, true),
                        None,
                        "k={k} {} fused_out={fused_out}",
                        case.name
                    );
                    assert!(
                        old.out.iter().any(|byte| *byte != CANARY),
                        "k={k} {}: the projection must write",
                        case.name
                    );
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, 3 * CASES_PER_K * 2);
}

/// Whether the mutant `mutant` gives bytes different from the per-value kernel on some case of the
/// grid; the mutation tests below require it.
#[cfg(feature = "cuda-exl3-rotate-mutants")]
fn mutant_is_caught(mutant: i32) -> Option<bool> {
    let (ctx, stream) = device()?;
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5741_5258);
    for k in [256, 512, 2048] {
        for (bank, _, cases) in banks_and_cases(&ctx, k, &mut rng) {
            for case in &cases {
                let old = rotate(rig, &bank, case, (Path::PerValue, false, (0, 0))).expect("old");
                let bad = rotate(rig, &bank, case, (Path::Mutant(mutant), false, (0, 0)))
                    .expect("mutant");
                if difference(&old, &bad, false).is_some() {
                    return Some(true);
                }
            }
        }
    }
    Some(false)
}

#[cfg(feature = "cuda-exl3-rotate-mutants")]
#[test]
fn mutant_reversed_butterfly_order_is_caught() {
    if let Some(caught) = mutant_is_caught(1) {
        assert!(caught);
    }
}

#[cfg(feature = "cuda-exl3-rotate-mutants")]
#[test]
fn mutant_dropped_suh_multiply_is_caught() {
    if let Some(caught) = mutant_is_caught(2) {
        assert!(caught);
    }
}

#[cfg(feature = "cuda-exl3-rotate-mutants")]
#[test]
fn mutant_fused_multiply_add_is_caught() {
    if let Some(caught) = mutant_is_caught(3) {
        assert!(caught);
    }
}

#[cfg(feature = "cuda-exl3-rotate-mutants")]
#[test]
fn mutant_wrong_gather_row_is_caught() {
    if let Some(caught) = mutant_is_caught(4) {
        assert!(caught);
    }
}

#[cfg(feature = "cuda-exl3-rotate-mutants")]
#[test]
fn mutant_dropped_product_flag_is_caught() {
    if let Some(caught) = mutant_is_caught(6) {
        assert!(caught);
    }
}

#[cfg(feature = "cuda-exl3-rotate-mutants")]
#[test]
fn mutant_dropped_scale_flag_is_caught() {
    if let Some(caught) = mutant_is_caught(7) {
        assert!(caught);
    }
}

#[cfg(feature = "cuda-exl3-rotate-mutants")]
#[test]
fn mutant_off_by_one_tail_is_caught() {
    if let Some(caught) = mutant_is_caught(5) {
        assert!(caught);
    }
}

/// The caught mutants must be caught by the cases they are aimed at, not by an unrelated failure
/// of the harness: the unmutated warp kernel passes the very same cases.
#[cfg(feature = "cuda-exl3-rotate-mutants")]
#[test]
fn the_unmutated_warp_kernel_passes_the_cases_the_mutants_fail() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5741_5258);
    for k in [256, 512, 2048] {
        for (bank, _, cases) in banks_and_cases(&ctx, k, &mut rng) {
            for case in &cases {
                let old = rotate(rig, &bank, case, (Path::PerValue, false, (0, 0))).expect("old");
                let new = rotate(rig, &bank, case, (Path::Warp, false, (0, 0))).expect("new");
                assert_eq!(difference(&old, &new, false), None, "k={k} {}", case.name);
            }
        }
    }
}
