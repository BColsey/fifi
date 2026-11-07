//! The combined gate and up input rotation against two separate calls: with the pair switch on, the
//! two projections of the routed experts (own scales, own rotated-row buffer, one table and one
//! set of input rows) must give the same `xh`, the same output and the same flag as the two calls
//! one after the other on the one-thread-per-value kernel. The mutation tests (feature
//! `cuda-exl3-rotate-mutants`) show that a swapped output buffer and the first call's scales
//! applied to the second output each turn the comparison red.

use super::{
    Bank, CANARY, Case, CudaContext, DeviceStream, EngError, Exl3Job, Exl3JobsCall, GRID, MEMBERS,
    device, first_difference, packed_elements_of, scaled_bank, suite, testkit,
};

/// What one run of the two calls left behind.
struct PairRun {
    xh: [Vec<u8>; 2],
    out: [Vec<u8>; 2],
    flag: u32,
    /// Whether the pair rotation accepts the two calls.
    usable: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum How {
    /// Two calls, one after the other, on the one-thread-per-value kernel.
    Separate,
    /// Two calls, one after the other, each on the warp-wide kernel.
    SeparateWarp,
    /// The pair entry with the switch on.
    Pair,
    /// The pair entry with the switch off.
    PairOff,
    /// The pair rotation alone, by a wrong variant.
    #[cfg_attr(not(feature = "cuda-exl3-rotate-mutants"), allow(dead_code))]
    Mutant(i32),
}

/// The two calls of `case` (own bank, own buffers, one table, one input, one flag) run the way
/// `how` says. `shifts` moves the start of x and of the two xh buffers into their allocations.
fn run_pair(
    (ctx, stream): (&CudaContext, &DeviceStream),
    banks: [&Bank; 2],
    case: &Case,
    (how, shifts): (How, [usize; 3]),
) -> Result<PairRun, EngError> {
    let (k, n) = (banks[0].k, banks[0].n);
    let mut x_bytes = vec![0u8; shifts[0]];
    x_bytes.extend(case.x.iter().flat_map(|value| value.to_le_bytes()));
    let x_buf = testkit::upload_bytes(ctx, &x_bytes);
    let xh_len = case.total_rows * k * 2;
    let xh = [
        testkit::upload_filled(ctx, CANARY, xh_len + shifts[1]),
        testkit::upload_filled(ctx, CANARY, xh_len + shifts[2]),
    ];
    let out_len = case.total_rows * n * 4;
    let z = [
        testkit::upload_filled(ctx, 0, out_len),
        testkit::upload_filled(ctx, 0, out_len),
    ];
    let out = [
        testkit::upload_filled(ctx, CANARY, out_len),
        testkit::upload_filled(ctx, CANARY, out_len),
    ];
    let flag = testkit::upload_filled(ctx, 0, 4);
    let jobs_bytes = Exl3Job::table(&case.jobs);
    let jobs = testkit::upload_bytes(ctx, &jobs_bytes);
    let list = case.in_rows.as_ref().map(|rows| {
        let bytes: Vec<u8> = rows.iter().flat_map(|row| row.to_le_bytes()).collect();
        testkit::upload_bytes(ctx, &bytes)
    });
    let call = |at: usize| -> Result<Exl3JobsCall<'_>, EngError> {
        let bank = banks[at];
        Ok(Exl3JobsCall {
            trellis: bank.trellis.range(0, bank.trellis.bytes())?,
            suh: bank.scales.range(0, bank.scales.bytes())?,
            svh: bank.scales.range(k * 4, bank.scales.bytes() - k * 4)?,
            trellis_stride: packed_elements_of(k, n, bank.bits_twice),
            scale_stride: k + n,
            members: case.members,
            x: x_buf.range(shifts[0], case.x.len() * 4)?,
            xh: xh[at].range(shifts[1 + at], xh_len)?,
            z: z[at].range(0, out_len)?,
            out: out[at].range(0, out_len)?,
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
        })
    };
    let usable = ctx.pair_usable(&call(0)?, &call(1)?)?;
    match how {
        How::Separate => {
            for at in 0..2 {
                ctx.exl3_jobs_modes(stream, call(at)?, (false, false))?;
            }
        }
        How::SeparateWarp => {
            for at in 0..2 {
                ctx.exl3_jobs_modes(stream, call(at)?, (false, true))?;
            }
        }
        How::Pair => {
            let taken =
                ctx.exl3_jobs_pair_mode(stream, (call(0)?, call(1)?), (true, Some(false)))?;
            assert_eq!(
                taken, usable,
                "the pair rotation runs exactly when the calls allow it"
            );
        }
        How::PairOff => {
            let taken =
                ctx.exl3_jobs_pair_mode(stream, (call(0)?, call(1)?), (false, Some(false)))?;
            assert!(!taken, "the pair rotation must not run with the switch off");
        }
        #[cfg(feature = "cuda-exl3-rotate-mutants")]
        How::Mutant(mutant) => ctx.exl3_jobs_pair_mutant(stream, (call(0)?, call(1)?), mutant)?,
        #[cfg(not(feature = "cuda-exl3-rotate-mutants"))]
        How::Mutant(_) => unreachable!("mutants are built only with their feature"),
    }
    stream.synchronize()?;
    let read = |buffer, len| testkit::read_bytes(ctx, buffer, len);
    Ok(PairRun {
        xh: [
            read(&xh[0], xh_len + shifts[1]),
            read(&xh[1], xh_len + shifts[2]),
        ],
        out: [read(&out[0], out_len), read(&out[1], out_len)],
        flag: u32::from_le_bytes(read(&flag, 4).try_into().unwrap()),
        usable,
    })
}

/// What differs between two runs, or `None`.
fn difference(old: &PairRun, new: &PairRun, with_out: bool) -> Option<String> {
    for at in 0..2 {
        if let Some(byte) = first_difference(&old.xh[at], &new.xh[at]) {
            return Some(format!("xh of call {at} differs at byte {byte}"));
        }
        if with_out && let Some(byte) = first_difference(&old.out[at], &new.out[at]) {
            return Some(format!("out of call {at} differs at byte {byte}"));
        }
    }
    (old.flag != new.flag).then(|| format!("flag {} against {}", old.flag, new.flag))
}

/// Two banks of one shape with their own scales (`awkward` selects the extreme ones), and the
/// cases of the suite built on the first.
fn setup(ctx: &CudaContext, k: usize, awkward: bool) -> (Bank, Bank, Vec<Case>) {
    let (first, scales) = scaled_bank(ctx, k, awkward, 0x5041_4952 + k as u64);
    let (second, _) = scaled_bank(ctx, k, awkward, 0x5041_4953 + k as u64);
    let mut rng = testkit::Rng::new(0x5041_4954 + k as u64);
    let mut cases = suite(k, &scales, &mut rng);
    cases.retain(|case| case.awkward == awkward);
    (first, second, cases)
}

/// A second bank whose scales hold an infinity, so that only the second rotation of a pair sees a
/// non-finite product (the first bank is ordinary), with the first case of the suite.
fn poisoned_setup(ctx: &CudaContext, k: usize) -> (Bank, Bank, Vec<Case>) {
    let (first, scales) = scaled_bank(ctx, k, false, 0x5041_4952 + k as u64);
    let (second, mut poisoned) = scaled_bank(ctx, k, false, 0x5041_4953 + k as u64);
    poisoned.values[0] = f32::INFINITY;
    let second = Bank {
        scales: testkit::upload_f32(ctx, &poisoned.values),
        ..second
    };
    let mut rng = testkit::Rng::new(0x5041_4955 + k as u64);
    let mut cases = suite(k, &scales, &mut rng);
    cases.truncate(1);
    (first, second, cases)
}

/// The pairs of banks the tests run: ordinary, extreme-valued, and one with a poisoned second bank.
fn all_setups(ctx: &CudaContext, k: usize) -> Vec<(Bank, Bank, Vec<Case>)> {
    vec![
        setup(ctx, k, false),
        setup(ctx, k, true),
        poisoned_setup(ctx, k),
    ]
}

#[test]
fn the_pair_rotation_equals_two_separate_calls_over_the_grid() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let rig = (&ctx, &stream);
    let mut compared = 0;
    for k in GRID {
        for (a, b, cases) in all_setups(&ctx, k) {
            for case in &cases {
                let old = run_pair(rig, [&a, &b], case, (How::Separate, [0; 3])).expect("old");
                let new = run_pair(rig, [&a, &b], case, (How::Pair, [0; 3])).expect("pair");
                assert!(
                    new.usable,
                    "k={k} {}: the pair rotation must be taken",
                    case.name
                );
                assert_eq!(difference(&old, &new, true), None, "k={k} {}", case.name);
                // The two outputs are not the same bytes: each call used its own scales.
                if case.name == "mixed heights" {
                    assert_ne!(new.xh[0], new.xh[1], "k={k}: the two banks differ");
                }
                compared += 1;
            }
        }
    }
    // 6 values of K, 14 cases of the two ordinary setups and 1 of the poisoned one.
    assert_eq!(compared, 6 * (super::CASES_PER_K + 1));
}

#[test]
fn calls_the_pair_cannot_take_run_one_after_the_other_with_the_same_bytes() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let rig = (&ctx, &stream);
    let (a, b, cases) = setup(&ctx, 512, false);
    let case = &cases[0];
    let old = run_pair(rig, [&a, &b], case, (How::Separate, [0; 3])).expect("old");
    // The switch off, and each buffer off its vector boundary in turn: x, then either xh.
    let off = run_pair(rig, [&a, &b], case, (How::PairOff, [0; 3])).expect("off");
    assert_eq!(difference(&old, &off, true), None, "switch off");
    for shifts in [[4, 0, 0], [0, 4, 0], [0, 0, 12], [16, 8, 24]] {
        let taken = shifts == [16, 8, 24];
        let old = run_pair(rig, [&a, &b], case, (How::Separate, shifts)).expect("old");
        let new = run_pair(rig, [&a, &b], case, (How::Pair, shifts)).expect("pair");
        assert_eq!(new.usable, taken, "shifts {shifts:?}");
        assert_eq!(difference(&old, &new, true), None, "shifts {shifts:?}");
    }
}

#[test]
fn the_pair_refuses_calls_that_do_not_read_the_same_rows() {
    let Some((ctx, _)) = device() else {
        return;
    };
    let (a, b, cases) = setup(&ctx, 256, false);
    let case = &cases[0];
    let k = a.k;
    let x = testkit::upload_f32(&ctx, &case.x);
    let other_x = testkit::upload_f32(&ctx, &case.x);
    let jobs_bytes = Exl3Job::table(&case.jobs);
    let jobs = testkit::upload_bytes(&ctx, &jobs_bytes);
    let xh = [
        testkit::upload_filled(&ctx, CANARY, case.total_rows * k * 2),
        testkit::upload_filled(&ctx, CANARY, case.total_rows * k * 2),
    ];
    let z = testkit::upload_filled(&ctx, 0, case.total_rows * a.n * 4);
    let out = testkit::upload_filled(&ctx, CANARY, case.total_rows * a.n * 4);
    let flag = testkit::upload_filled(&ctx, 0, 4);
    let banks = [&a, &b];
    let inputs = [&x, &other_x];
    let call = |bank_at: usize, x_at: usize, xh_at: usize, rows: usize| {
        let (bank, x) = (banks[bank_at], inputs[x_at]);
        Ok::<_, EngError>(Exl3JobsCall {
            trellis: bank.trellis.range(0, bank.trellis.bytes())?,
            suh: bank.scales.range(0, bank.scales.bytes())?,
            svh: bank.scales.range(k * 4, bank.scales.bytes() - k * 4)?,
            trellis_stride: packed_elements_of(k, bank.n, bank.bits_twice),
            scale_stride: k + bank.n,
            members: MEMBERS,
            x: x.range(0, case.x.len() * 4)?,
            xh: xh[xh_at].range(0, case.total_rows * k * 2)?,
            z: z.range(0, case.total_rows * bank.n * 4)?,
            out: out.range(0, case.total_rows * bank.n * 4)?,
            flag: flag.range(0, 4)?,
            jobs: jobs.range(0, jobs_bytes.len())?,
            jobs_count: case.jobs.len(),
            max_rows: case.max_rows,
            total_rows: rows,
            out_ld: bank.n,
            out_col: 0,
            k,
            n: bank.n,
            bits_twice: bank.bits_twice,
            book: bank.book,
            in_rows: None,
            out_rows: None,
            x_rows: rows,
            out_total_rows: rows,
        })
    };
    let rows = case.total_rows;
    let usable = |first, second| ctx.pair_usable(&first, &second).expect("checks");
    assert!(usable(
        call(0, 0, 0, rows).unwrap(),
        call(1, 0, 1, rows).unwrap()
    ));
    // Another input buffer, the same xh buffer twice, another row count: refused.
    assert!(!usable(
        call(0, 0, 0, rows).unwrap(),
        call(1, 1, 1, rows).unwrap()
    ));
    assert!(!usable(
        call(0, 0, 0, rows).unwrap(),
        call(1, 0, 0, rows).unwrap()
    ));
    assert!(!usable(
        call(0, 0, 0, rows).unwrap(),
        call(1, 0, 1, rows - 1).unwrap()
    ));
}

/// Whether the mutant `mutant` of the pair rotation gives bytes different from two separate calls
/// on some case; `None` without a device.
#[cfg(feature = "cuda-exl3-rotate-mutants")]
fn pair_mutant_is_caught(mutant: i32) -> Option<bool> {
    let (ctx, stream) = device()?;
    let rig = (&ctx, &stream);
    for k in [256, 512, 2048] {
        for (a, b, cases) in all_setups(&ctx, k) {
            for case in &cases {
                let old = run_pair(rig, [&a, &b], case, (How::Separate, [0; 3])).expect("old");
                let bad =
                    run_pair(rig, [&a, &b], case, (How::Mutant(mutant), [0; 3])).expect("mutant");
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
fn mutant_swapped_pair_outputs_is_caught() {
    if let Some(caught) = pair_mutant_is_caught(1) {
        assert!(caught);
    }
}

#[cfg(feature = "cuda-exl3-rotate-mutants")]
#[test]
fn mutant_dropped_pair_b_flag_is_caught() {
    if let Some(caught) = pair_mutant_is_caught(3) {
        assert!(caught);
    }
}

#[cfg(feature = "cuda-exl3-rotate-mutants")]
#[test]
fn mutant_second_output_by_the_first_scales_is_caught() {
    if let Some(caught) = pair_mutant_is_caught(2) {
        assert!(caught);
    }
}

/// The routed gate and up shape of a 4K prefill chunk: 2,048 tokens, eight picks each, about 40
/// pairs per job, hidden size 2,048.
fn moe_shape_case(rng: &mut testkit::Rng) -> Case {
    let (tokens, pairs, k) = (2048usize, 16384usize, 2048usize);
    let mut jobs = Vec::new();
    let mut row0 = 0usize;
    while row0 < pairs {
        let rows = (1 + rng.below(79) as usize).min(pairs - row0);
        jobs.push(Exl3Job {
            member: (jobs.len() % MEMBERS) as u32,
            row0: row0 as u32,
            rows: rows as u32,
        });
        row0 += rows;
    }
    Case {
        name: "4K gate/up shape".into(),
        awkward: false,
        members: MEMBERS,
        max_rows: jobs.iter().map(|job| job.rows as usize).max().unwrap(),
        x: super::rows_of(rng, tokens, k),
        in_rows: Some(
            (0..pairs)
                .map(|_| rng.below(tokens as u64) as u32)
                .collect(),
        ),
        jobs,
        total_rows: pairs,
    }
}

#[test]
fn the_gate_up_shape_case_meets_the_call_bounds() {
    // Host only: the case of the timing driver, built with debug arithmetic and checked against the
    // limits of the call.
    let case = moe_shape_case(&mut testkit::Rng::new(0x5041_4958));
    super::check_case_bounds(&case, 2048);
    assert!(case.jobs.len() > 300, "about 400 jobs of up to 80 rows");
    let rows: usize = case.jobs.iter().map(|job| job.rows as usize).sum();
    assert_eq!(rows, case.total_rows, "the jobs cover the table");
}

/// The launches of the 4K gate/up shape in the three ways, five times each, for a profiler run
/// (`nsys profile --trace=cuda`, then the kernel durations of jobs_input_rotate,
/// jobs_input_rotate_wide and jobs_input_rotate_pair). It also checks the three ways give the same
/// bytes. Ignored: it is a measurement driver, run only in a timing window.
#[test]
#[ignore = "timing driver for a profiler run in a device window"]
fn gate_up_rotation_timing_cells() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let rig = (&ctx, &stream);
    let mut rng = testkit::Rng::new(0x5041_4955);
    let (a, _) = scaled_bank(&ctx, 2048, false, 0x5041_4956);
    let (b, _) = scaled_bank(&ctx, 2048, false, 0x5041_4957);
    let case = moe_shape_case(&mut rng);
    let mut first: Option<PairRun> = None;
    for how in [How::Separate, How::SeparateWarp, How::Pair] {
        for round in 0..5 {
            let run = run_pair(rig, [&a, &b], &case, (how, [0; 3])).expect("run");
            match &first {
                None => first = Some(run),
                Some(base) => {
                    assert_eq!(difference(base, &run, true), None, "{how:?} round {round}")
                }
            }
        }
        println!("TIMING_CELLS {how:?} done");
    }
}
