//! Tests of the prefill conv kernel `conv_silu_rows_k` against the vendored `conv_silu_k`: for every
//! shape the two write the same output bytes and the same final window, whatever the rows per
//! thread, the number of rows, the channel count, the row stride and the way a prompt is cut into
//! pieces that carry the window. The host tests of the switch run anywhere; the device tests skip
//! on a host without a device.

use std::ffi::OsStr;

use super::{
    ConvCall, ConvShape, PREFILL_DEFAULT_ROWS, PREFILL_MIN_ROWS, parse_prefill_setting,
    prefill_rows,
};
use crate::EngError;
use crate::kernels::cuda::gdn_mutants::quick;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

const SENTINEL: u8 = 0xa7;
/// Floats of sentinel after the data of `out`: a kernel that writes past its rows changes them.
const TAIL_FLOATS: usize = 512;

/// Qwen3.6: 8,192 channels (q 2,048, k 2,048, v 4,096) in a projection row of 12,320 floats.
const QWEN36: (usize, usize) = (8192, 12_320);
/// Flash-Next: 10,240 channels (q 2,048, k 2,048, v 6,144) in a projection row of 16,480 floats.
const FLASH_NEXT: (usize, usize) = (10_240, 16_480);

fn os(text: &str) -> Option<&OsStr> {
    Some(OsStr::new(text))
}

#[test]
fn the_switch_value_is_strict() {
    assert_eq!(parse_prefill_setting(None), None);
    assert_eq!(parse_prefill_setting(os("1")), Some(PREFILL_DEFAULT_ROWS));
    assert_eq!(parse_prefill_setting(os(" 1 ")), Some(PREFILL_DEFAULT_ROWS));
    for rows in [16usize, 32, 64] {
        assert_eq!(parse_prefill_setting(os(&rows.to_string())), Some(rows));
    }
    for other in [
        "", "0", "2", "8", "128", "on", "true", "yes", "1.0", "-1", "32x",
    ] {
        assert_eq!(parse_prefill_setting(os(other)), None, "{other:?}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let not_utf8 = OsStr::from_bytes(&[0x31, 0xff]);
        assert_eq!(parse_prefill_setting(Some(not_utf8)), None);
    }
}

#[test]
fn only_a_long_launch_under_a_set_switch_takes_the_prefill_kernel() {
    assert_eq!(PREFILL_MIN_ROWS, 64);
    assert_eq!(prefill_rows(None, 4096), None, "switch unset");
    assert_eq!(prefill_rows(Some(32), 1), None, "decode");
    assert_eq!(prefill_rows(Some(32), 8), None, "K-row window");
    assert_eq!(prefill_rows(Some(32), PREFILL_MIN_ROWS - 1), None);
    assert_eq!(prefill_rows(Some(32), PREFILL_MIN_ROWS), Some(32));
    assert_eq!(prefill_rows(Some(16), 1024), Some(16));
    assert_eq!(prefill_rows(Some(64), 4099), Some(64));
}

/// How the pieces test cuts a prompt (rows of each piece) and the prompt's total; the host test below checks that every cut
/// adds up, so a wrong cut cannot panic inside a device test (a panic with the device context alive latches the CUDA backend).
const CONV_CUTS: ([&[usize]; 4], usize) = (
    [
        &[2048, 1024, 1024, 3],
        &[2048, 2048, 3],
        &[1, 2, 4091, 5],
        &[1024, 1024, 1024, 1024, 3],
    ],
    4099,
);
const CONV_CUTS_QUICK: ([&[usize]; 4], usize) = (
    [
        &[128, 100, 70, 2],
        &[150, 150],
        &[1, 2, 290, 7],
        &[64, 64, 64, 64, 44],
    ],
    300,
);

#[test]
fn every_conv_piece_cut_adds_up_to_its_prompt() {
    for (cuts, total) in [CONV_CUTS, CONV_CUTS_QUICK] {
        for cut in cuts {
            assert_eq!(cut.iter().sum::<usize>(), total, "{cut:?}");
        }
    }
}

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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Original,
    Rows(usize),
}

struct Inputs {
    channels: usize,
    ld: usize,
    m: usize,
    w: Vec<f32>,
    state0: Vec<f32>,
    /// `[m][ld]`; the columns past `channels` are NaN and must never be read.
    proj: Vec<f32>,
}

/// Values that stress the arithmetic: infinities, signed zeros, subnormals, saturating and
/// overflowing magnitudes for `__expf`.
const SPECIALS: [f32; 12] = [
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.0,
    -0.0,
    1.0e-40,
    -1.0e-40,
    1.0e30,
    -1.0e30,
    100.0,
    -100.0,
    -88.5,
    88.5,
];

fn inputs(channels: usize, ld: usize, m: usize, seed: u64) -> Inputs {
    let mut rng = Rng::new(seed);
    let mut draw =
        |n: usize, sd: f64| -> Vec<f32> { (0..n).map(|_| (rng.normal() * sd) as f32).collect() };
    let w = draw(channels * 4, 0.7);
    let state0 = draw(channels * 3, 1.0);
    let mut proj = vec![f32::NAN; m * ld];
    for t in 0..m {
        let row = draw(channels, 1.5);
        proj[t * ld..t * ld + channels].copy_from_slice(&row);
    }
    let mut pick = Rng::new(seed ^ 0x5eed);
    let mut state0 = state0;
    for _ in 0..(m * channels / 512).clamp(8, 4096) {
        let at = pick.below((m * channels) as u64) as usize;
        let special = SPECIALS[pick.below(SPECIALS.len() as u64) as usize];
        proj[(at / channels) * ld + at % channels] = special;
    }
    for _ in 0..8 {
        let at = pick.below(state0.len() as u64) as usize;
        state0[at] = SPECIALS[pick.below(SPECIALS.len() as u64) as usize];
    }
    // A few special tap weights as well: zeros of both signs, an infinity, a subnormal.
    let mut w = w;
    for special in [0.0f32, -0.0, f32::INFINITY, 1.0e-40] {
        let at = pick.below(w.len() as u64) as usize;
        w[at] = special;
    }
    Inputs {
        channels,
        ld,
        m,
        w,
        state0,
        proj,
    }
}

struct Run {
    out: Vec<u8>,
    state: Vec<u8>,
    proj_after: Vec<u8>,
}

/// Runs `pieces` (rows, kernel) one after the other over the rows of `inp`, carrying the window
/// in one device buffer, and returns the output bytes, the final window and the projection bytes.
fn run_pieces(rig: &Rig, inp: &Inputs, pieces: &[(usize, Kernel)]) -> Run {
    assert_eq!(pieces.iter().map(|p| p.0).sum::<usize>(), inp.m);
    let ctx = &rig.ctx;
    let proj = testkit::upload_f32(ctx, &inp.proj);
    let w = testkit::upload_f32(ctx, &inp.w);
    let state = testkit::upload_f32(ctx, &inp.state0);
    let out_bytes = inp.m * inp.channels * 4;
    let out = testkit::upload_filled(ctx, SENTINEL, out_bytes + TAIL_FLOATS * 4);
    let mut row0 = 0;
    for &(rows, kernel) in pieces {
        let call = piece_call(inp, (&proj, &w, &state, &out), row0, rows);
        let result = match kernel {
            Kernel::Original => ctx.gdn_conv_original(&rig.stream, call),
            Kernel::Rows(r) => ctx.gdn_conv_rows(&rig.stream, call, r),
        };
        result.unwrap_or_else(|e| panic!("{kernel:?} rows {row0}..{}: {e:?}", row0 + rows));
        row0 += rows;
    }
    rig.stream.synchronize().expect("drain");
    let mut out_all = testkit::read_bytes(ctx, &out, out_bytes + TAIL_FLOATS * 4);
    let tail = out_all.split_off(out_bytes);
    assert!(
        tail.iter().all(|b| *b == SENTINEL),
        "{pieces:?}: the output tail changed"
    );
    Run {
        out: out_all,
        state: testkit::read_bytes(ctx, &state, inp.channels * 12),
        proj_after: testkit::read_bytes(ctx, &proj, inp.proj.len() * 4),
    }
}

fn piece_call<'b>(
    inp: &Inputs,
    (proj, w, state, out): (
        &'b DeviceBuffer,
        &'b DeviceBuffer,
        &'b DeviceBuffer,
        &'b DeviceBuffer,
    ),
    row0: usize,
    rows: usize,
) -> ConvCall<'b> {
    let n = inp.channels;
    ConvCall {
        shape: ConvShape {
            m: rows,
            channels: n,
            ld: inp.ld,
        },
        proj: proj
            .range(row0 * inp.ld * 4, ((rows - 1) * inp.ld + n) * 4)
            .unwrap(),
        weights: w.range(0, n * 16).unwrap(),
        state: state.range(0, n * 12).unwrap(),
        out: out.range(row0 * n * 4, rows * n * 4).unwrap(),
    }
}

fn float_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn assert_same(label: &str, got: &Run, want: &Run, inp: &Inputs) {
    assert_eq!(got.state, want.state, "{label}: final window");
    if got.out != want.out {
        let first = got
            .out
            .chunks(4)
            .zip(want.out.chunks(4))
            .position(|(g, w)| g != w)
            .unwrap();
        panic!(
            "{label}: output bytes differ, first at row {} channel {}",
            first / inp.channels,
            first % inp.channels
        );
    }
    let proj = float_bytes(&inp.proj);
    assert_eq!(got.proj_after, proj, "{label}: proj changed");
}

/// Every kernel variant against the original on one input, one piece.
fn check_one_piece(rig: &Rig, inp: &Inputs, label: &str) {
    let want = run_pieces(rig, inp, &[(inp.m, Kernel::Original)]);
    for rows in [16usize, 32, 64] {
        let got = run_pieces(rig, inp, &[(inp.m, Kernel::Rows(rows))]);
        assert_same(&format!("{label} rows/thread {rows}"), &got, &want, inp);
    }
}

#[test]
fn the_prefill_kernel_writes_the_bytes_of_the_original() {
    if !testkit::device_present("conv prefill kernel against the original") {
        return;
    }
    let rig = Rig::new();
    // 1 (decode), 2, 3 (K - 1), 4 (K), 5, a row count off every tile edge, the tile edges and
    // their neighbours, and the chunk sizes of the prefill.
    let rows: &[usize] = if quick() {
        &[1, 2, 3, 17, 70, 200]
    } else {
        &[
            1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 127, 128, 129, 1024, 2048,
        ]
    };
    for (i, &m) in rows.iter().enumerate() {
        let inp = inputs(QWEN36.0, QWEN36.1, m, 0x6c01_0000 + i as u64);
        check_one_piece(&rig, &inp, &format!("qwen36 m {m}"));
        let inp = inputs(FLASH_NEXT.0, FLASH_NEXT.1, m, 0x6c02_0000 + i as u64);
        check_one_piece(&rig, &inp, &format!("flash-next m {m}"));
    }
    if quick() {
        return;
    }
    // 4,096 plus an odd tail.
    let inp = inputs(QWEN36.0, QWEN36.1, 4099, 0x6c03_0001);
    check_one_piece(&rig, &inp, "qwen36 m 4099");
    let inp = inputs(FLASH_NEXT.0, FLASH_NEXT.1, 2051, 0x6c03_0002);
    check_one_piece(&rig, &inp, "flash-next m 2051");
}

#[test]
fn the_prefill_kernel_handles_channel_counts_off_the_block_size() {
    if !testkit::device_present("conv prefill kernel channel counts") {
        return;
    }
    let rig = Rig::new();
    for (i, &(channels, ld)) in [(1usize, 1usize), (4, 4), (255, 255), (257, 300), (384, 400)]
        .iter()
        .enumerate()
    {
        for m in [1usize, 2, 3, 19, 64, 70, 200] {
            let inp = inputs(channels, ld, m, 0x6c04_0000 + (i * 1000 + m) as u64);
            check_one_piece(&rig, &inp, &format!("channels {channels} ld {ld} m {m}"));
        }
    }
}

#[test]
fn pieces_carry_the_window_across_launches_and_across_kernels() {
    if !testkit::device_present("conv prefill kernel piece handoff") {
        return;
    }
    let rig = Rig::new();
    let total = if quick() {
        CONV_CUTS_QUICK.1
    } else {
        CONV_CUTS.1
    };
    let inp = inputs(QWEN36.0, QWEN36.1, total, 0x6c05_0001);
    let want = run_pieces(&rig, &inp, &[(inp.m, Kernel::Original)]);
    let (cuts, _) = if quick() { CONV_CUTS_QUICK } else { CONV_CUTS };
    for cut in cuts {
        let plans: [Vec<(usize, Kernel)>; 4] = [
            cut.iter().map(|&r| (r, Kernel::Original)).collect(),
            cut.iter().map(|&r| (r, Kernel::Rows(32))).collect(),
            cut.iter()
                .enumerate()
                .map(|(i, &r)| {
                    (
                        r,
                        if i % 2 == 0 {
                            Kernel::Rows(16)
                        } else {
                            Kernel::Original
                        },
                    )
                })
                .collect(),
            cut.iter()
                .enumerate()
                .map(|(i, &r)| {
                    (
                        r,
                        if i % 2 == 0 {
                            Kernel::Original
                        } else {
                            Kernel::Rows(64)
                        },
                    )
                })
                .collect(),
        ];
        for plan in plans {
            let got = run_pieces(&rig, &inp, &plan);
            assert_same(&format!("pieces {plan:?}"), &got, &want, &inp);
        }
    }
}

#[test]
fn decode_after_a_prefill_on_the_prefill_kernel_reads_the_window_it_left() {
    if !testkit::device_present("conv decode after prefill kernel") {
        return;
    }
    let rig = Rig::new();
    for (channels, ld) in [QWEN36, FLASH_NEXT] {
        let m = (if quick() { 300 } else { 2048 }) + 7;
        let inp = inputs(channels, ld, m, 0x6c06_0000 + channels as u64);
        let want = run_pieces(
            &rig,
            &inp,
            &[
                (m - 6, Kernel::Original),
                (1, Kernel::Original),
                (1, Kernel::Original),
                (1, Kernel::Original),
                (1, Kernel::Original),
                (1, Kernel::Original),
                (1, Kernel::Original),
            ],
        );
        for rows in [16usize, 32, 64] {
            let mut plan = vec![(m - 6, Kernel::Rows(rows))];
            plan.extend([(1, Kernel::Original); 6]);
            let got = run_pieces(&rig, &inp, &plan);
            assert_same(
                &format!("channels {channels} rows/thread {rows}"),
                &got,
                &want,
                &inp,
            );
        }
    }
}

#[test]
fn the_switch_routes_only_long_launches_to_the_prefill_kernel() {
    if !testkit::device_present("conv prefill routing") {
        return;
    }
    let rig = Rig::new();
    let ctx = &rig.ctx;
    for (setting, m, on_prefill) in [
        (None, 4096usize, false),
        (Some(32), 1, false),
        (Some(32), 8, false),
        (Some(32), PREFILL_MIN_ROWS - 1, false),
        (Some(32), PREFILL_MIN_ROWS, true),
        (Some(16), 70, true),
        (Some(64), 1024, true),
    ] {
        let inp = inputs(384, 400, m, 0x6c09_0000 + m as u64);
        let proj = testkit::upload_f32(ctx, &inp.proj);
        let w = testkit::upload_f32(ctx, &inp.w);
        let state = testkit::upload_f32(ctx, &inp.state0);
        let out = testkit::upload_filled(ctx, SENTINEL, m * 384 * 4);
        let before = super::ROWS_CALLS.with(std::cell::Cell::get);
        let call = piece_call(&inp, (&proj, &w, &state, &out), 0, m);
        ctx.gdn_conv_with(&rig.stream, call, setting)
            .expect("launch");
        rig.stream.synchronize().expect("drain");
        let ran = super::ROWS_CALLS.with(std::cell::Cell::get) - before;
        assert_eq!(ran, usize::from(on_prefill), "setting {setting:?} m {m}");
        let want = run_pieces(&rig, &inp, &[(m, Kernel::Original)]);
        assert_eq!(
            testkit::read_bytes(ctx, &out, m * 384 * 4),
            want.out,
            "setting {setting:?} m {m}: out"
        );
        assert_eq!(testkit::read_bytes(ctx, &state, 384 * 12), want.state);
    }
}

fn refusal_text(result: Result<(), EngError>) -> String {
    match result {
        Err(EngError::BadRequest(message)) => message,
        other => panic!("expected a BadRequest, got {other:?}"),
    }
}

/// The refusal text of the plain wrapper and of the prefill wrapper for the same bad call.
fn refusals<'b>(rig: &Rig, make: impl Fn() -> ConvCall<'b>) -> (String, String) {
    (
        refusal_text(rig.ctx.gdn_conv_original(&rig.stream, make())),
        refusal_text(rig.ctx.gdn_conv_rows(&rig.stream, make(), 32)),
    )
}

#[test]
fn the_prefill_wrapper_refuses_what_the_plain_wrapper_refuses() {
    if !testkit::device_present("conv prefill wrapper refusals") {
        return;
    }
    let rig = Rig::new();
    let ctx = &rig.ctx;
    let inp = inputs(256, 300, 70, 0x6c07_0001);
    let proj = testkit::upload_f32(ctx, &inp.proj);
    let w = testkit::upload_f32(ctx, &inp.w);
    let state = testkit::upload_f32(ctx, &inp.state0);
    let out = testkit::upload_filled(ctx, SENTINEL, 70 * 256 * 4);
    let good = |m: usize| piece_call(&inp, (&proj, &w, &state, &out), 0, m);
    let (plain, rows) = refusals(&rig, || {
        let mut call = good(70);
        call.shape.m = 0;
        call
    });
    assert_eq!(plain, rows, "m = 0");
    let (plain, rows) = refusals(&rig, || {
        let mut call = good(70);
        call.shape.ld = 100;
        call
    });
    assert_eq!(plain, rows, "ld below the channels");
    let (plain, rows) = refusals(&rig, || {
        let mut call = good(70);
        call.out = out.range(0, 4).unwrap();
        call
    });
    assert_eq!(plain, rows, "a short out");
    let (plain, rows) = refusals(&rig, || {
        let mut call = good(70);
        call.state = out.range(0, 256 * 12).unwrap();
        call
    });
    assert_eq!(plain, rows, "state over out");
    assert!(plain.contains("overlap"), "{plain}");
    for bad in [0usize, 8, 15, 33, 128] {
        let message = refusal_text(ctx.gdn_conv_rows(&rig.stream, good(70), bad));
        assert!(message.contains("rows per thread"), "{bad}: {message}");
    }
    let after = run_pieces(&rig, &inp, &[(70, Kernel::Rows(32))]);
    let want = run_pieces(&rig, &inp, &[(70, Kernel::Original)]);
    assert_same("after the refusals", &after, &want, &inp);
}

/// Wall time of one DeltaNet layer pass of 30 conv launches, queued back to back and drained once,
/// for the original kernel and each prefill variant at the prefill shapes: 2,048 rows (the first
/// pieces) and 1,024 rows (the layer chunks past 4,096 rows). The variants take turns in every
/// round so a drift of the clocks does not favour one. Run alone in an exclusive window:
/// `--ignored --exact --nocapture`. Prints `BENCH` lines.
#[test]
#[ignore = "timing: run alone in an exclusive window"]
fn conv_prefill_kernel_timing() {
    if !testkit::device_present("conv prefill kernel timing") {
        return;
    }
    const LAYERS: usize = 30;
    const ROUNDS: usize = 41;
    const WARM: usize = 5;
    let rig = Rig::new();
    let ctx = &rig.ctx;
    let variants = [
        Kernel::Original,
        Kernel::Rows(16),
        Kernel::Rows(32),
        Kernel::Rows(64),
    ];
    for (name, (channels, ld)) in [("qwen36", QWEN36), ("flash-next", FLASH_NEXT)] {
        for m in [2048usize, 1024] {
            let inp = inputs(channels, ld, m, 0x6c08_0000 + m as u64);
            let proj = testkit::upload_f32(ctx, &inp.proj);
            let w = testkit::upload_f32(ctx, &inp.w);
            let states: Vec<DeviceBuffer> = (0..LAYERS)
                .map(|_| testkit::upload_f32(ctx, &inp.state0))
                .collect();
            let out = testkit::upload_filled(ctx, SENTINEL, m * channels * 4);
            let mut times = vec![Vec::new(); variants.len()];
            for round in 0..WARM + ROUNDS {
                for step in 0..variants.len() {
                    let slot = (step + round) % variants.len();
                    let kernel = variants[slot];
                    let started = std::time::Instant::now();
                    for state in &states {
                        let call = piece_call(&inp, (&proj, &w, state, &out), 0, m);
                        match kernel {
                            Kernel::Original => ctx.gdn_conv_original(&rig.stream, call),
                            Kernel::Rows(r) => ctx.gdn_conv_rows(&rig.stream, call, r),
                        }
                        .expect("launch");
                    }
                    rig.stream.synchronize().expect("drain");
                    if round >= WARM {
                        times[slot].push(started.elapsed().as_secs_f64() * 1e6 / LAYERS as f64);
                    }
                }
            }
            let bytes = (m * channels * 8) as f64;
            for (slot, kernel) in variants.iter().enumerate() {
                let mut t = times[slot].clone();
                t.sort_by(f64::total_cmp);
                let p50 = t[t.len() / 2];
                println!(
                    "BENCH shape={name} m={m} kernel={kernel:?} p50_us={p50:.1} min_us={:.1} gbps={:.0}",
                    t[0],
                    bytes / (p50 * 1e-6) / 1e9
                );
            }
        }
    }
}

/// Each deliberately wrong variant of the conv prefill kernel and of its switch code must turn the
/// byte-identity or switch test that guards it red, and the unmutated control must pass the same short
/// shape lists. Needs the `cuda-gdn-mutants` build and a device for the kernel mutants;
/// run with `--test-threads=1`.
#[cfg(feature = "cuda-gdn-mutants")]
mod mutants {
    // Accepted survivors, by construction (no test can see them): a dispatch-mapping swap such as
    // `case 16: launch_rows<32>` instead of `launch_rows<16>` writes the same bytes at a different launch shape.
    // Mutant 19 (a load one row past the tile) leaves the bytes unchanged and is seen by memcheck, not by bytes;
    // the sanitizer legs of tools/sanitizer_run.sh kill it.
    use super::*;
    use crate::kernels::cuda::gdn_mutants::{self as gm, select};

    /// The mutant `id` must make `body` panic.
    fn killed(id: u32, name: &str, body: fn()) {
        let _selected = select(id);
        let outcome = std::panic::catch_unwind(body);
        assert!(
            outcome.is_err(),
            "mutant {name} (id {id}) survived its test"
        );
    }

    #[test]
    fn mutant_control_without_a_mutant_passes_the_short_lists() {
        let _selected = select(0);
        the_switch_value_is_strict();
        only_a_long_launch_under_a_set_switch_takes_the_prefill_kernel();
        the_prefill_kernel_writes_the_bytes_of_the_original();
        the_prefill_kernel_handles_channel_counts_off_the_block_size();
        pieces_carry_the_window_across_launches_and_across_kernels();
        the_switch_routes_only_long_launches_to_the_prefill_kernel();
    }

    #[test]
    fn mutant_conv_plain_expression_is_killed() {
        killed(
            1,
            "plain expression",
            the_prefill_kernel_writes_the_bytes_of_the_original,
        );
    }

    #[test]
    fn mutant_conv_window_words_swapped_is_killed() {
        killed(
            2,
            "window words swapped",
            the_prefill_kernel_writes_the_bytes_of_the_original,
        );
    }

    #[test]
    fn mutant_conv_state_rows_reversed_is_killed() {
        killed(
            3,
            "state rows reversed",
            the_prefill_kernel_writes_the_bytes_of_the_original,
        );
    }

    #[test]
    fn mutant_conv_window_written_by_the_wrong_cta_row_is_killed() {
        killed(
            4,
            "window from the wrong CTA row",
            the_prefill_kernel_writes_the_bytes_of_the_original,
        );
    }

    #[test]
    fn mutant_conv_last_row_not_stored_is_killed() {
        killed(
            5,
            "last row not stored",
            the_prefill_kernel_writes_the_bytes_of_the_original,
        );
    }

    #[test]
    fn mutant_conv_channel_guard_off_by_one_is_killed() {
        killed(
            21,
            "channel guard c > n",
            the_prefill_kernel_handles_channel_counts_off_the_block_size,
        );
    }

    /// For the sanitizer runs: selects the mutant named by `GDN_MUTANT_ID` and runs the byte tests plainly
    /// (no catch_unwind), so `compute-sanitizer --error-exitcode` reports what memcheck sees. A mutant the byte
    /// tests cannot see (19, a load one row past the tile) is killed by the sanitizer's exit code.
    #[test]
    #[ignore = "sanitizer runs only: set GDN_MUTANT_ID"]
    fn mutant_from_the_environment_runs_the_byte_tests() {
        let id: u32 = std::env::var("GDN_MUTANT_ID")
            .expect("GDN_MUTANT_ID")
            .parse()
            .expect("a mutant id");
        let _selected = select(id);
        the_prefill_kernel_writes_the_bytes_of_the_original();
        the_prefill_kernel_handles_channel_counts_off_the_block_size();
    }

    #[test]
    fn mutant_conv_routing_ignored_is_killed() {
        killed(
            gm::ROUTE_IGNORED,
            "routing ignored",
            the_switch_routes_only_long_launches_to_the_prefill_kernel,
        );
    }

    #[test]
    fn mutant_conv_threshold_one_is_killed() {
        killed(
            gm::MIN_ROWS_ONE,
            "threshold one",
            only_a_long_launch_under_a_set_switch_takes_the_prefill_kernel,
        );
    }

    #[test]
    fn mutant_conv_one_selects_sixteen_is_killed() {
        killed(
            gm::ONE_IS_SIXTEEN,
            "1 selects 16",
            the_switch_value_is_strict,
        );
    }

    #[test]
    fn mutant_conv_value_not_trimmed_is_killed() {
        killed(
            gm::CONV_NO_TRIM,
            "value not trimmed",
            the_switch_value_is_strict,
        );
    }

    #[test]
    fn mutant_conv_accepts_eight_is_killed() {
        killed(
            gm::CONV_ACCEPTS_EIGHT,
            "accepts 8",
            the_switch_value_is_strict,
        );
    }
}
