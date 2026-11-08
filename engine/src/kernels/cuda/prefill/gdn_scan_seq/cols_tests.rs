//! Tests of the scan kernel with narrower blocks (`gdn_scan_seq_cols_k`, 8, 16 or 32 value columns
//! per block) against the original `gdn_scan_seq_k`: the state and the gated `o_out` rows are the
//! same bytes for every block width, form, head geometry, row count and way a prompt is cut into
//! pieces. The host test of the switch runs anywhere; the device tests skip on a host without a
//! device.

use std::ffi::OsStr;

use super::{
    GdnForm, GdnScanSeqCall, SCAN_MIN_ROWS, ScanShape, ScanWeights, parse_scan_cols, scan_bytes,
    scan_cols_for,
};
use crate::EngError;
use crate::kernels::cuda::gdn_mutants::quick;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceStream};

const D: usize = 128;
const WIDTHS: [usize; 3] = [8, 16, 32];
/// Bytes of sentinel after the `o_out` rows: a kernel that stores past its rows changes them.
const TAIL_BYTES: usize = 2048;

#[test]
fn the_scan_columns_value_is_strict() {
    assert_eq!(parse_scan_cols(None), None);
    for cols in WIDTHS {
        assert_eq!(
            parse_scan_cols(Some(OsStr::new(&cols.to_string()))),
            Some(cols)
        );
        assert_eq!(
            parse_scan_cols(Some(OsStr::new(&format!(" {cols} ")))),
            Some(cols)
        );
    }
    for other in [
        "", "0", "1", "4", "7", "64", "128", "016", "+16", "16.0", "on", "-8",
    ] {
        assert_eq!(parse_scan_cols(Some(OsStr::new(other))), None, "{other:?}");
    }
}

#[test]
fn only_a_long_scan_under_a_set_switch_takes_the_narrow_block_kernel() {
    assert_eq!(SCAN_MIN_ROWS, 64);
    assert_eq!(scan_cols_for(None, 4096), None, "switch unset");
    assert_eq!(scan_cols_for(Some(16), 1), None, "a single-token step");
    assert_eq!(scan_cols_for(Some(16), SCAN_MIN_ROWS - 1), None);
    assert_eq!(scan_cols_for(Some(16), SCAN_MIN_ROWS), Some(16));
    assert_eq!(scan_cols_for(Some(8), 1024), Some(8));
    assert_eq!(scan_cols_for(Some(32), 2051), Some(32));
}

/// How the pieces tests cut a prompt (rows of each piece) and the prompt's total; the host test below checks that every cut
/// adds up, so a wrong cut cannot panic inside a device test (a panic with the device context alive latches the CUDA backend
/// and fails every test after it).
const SCAN_CUTS: ([&[usize]; 3], usize) = (
    [
        &[1024, 300, 176],
        &[1, 2, 1497],
        &[512, 512, 452, 16, 1, 3, 4],
    ],
    1500,
);
const SCAN_CUTS_QUICK: ([&[usize]; 3], usize) = (
    [&[200, 99, 1], &[1, 2, 297], &[100, 100, 64, 16, 1, 3, 16]],
    300,
);

#[test]
fn every_scan_piece_cut_adds_up_to_its_prompt() {
    for (cuts, total) in [SCAN_CUTS, SCAN_CUTS_QUICK] {
        for cut in cuts {
            assert_eq!(cut.iter().sum::<usize>(), total, "{cut:?}");
        }
    }
}

/// The harness arithmetic of the scan device tests on the host: the input builder at the real head counts, and for every plan
/// the tests use the ranges of each piece (`scan_bytes` of its shape at its row offset) against the buffers the tests upload.
#[test]
fn the_scan_harness_arithmetic_fits_its_buffers() {
    for &(n_k, n_v) in &[(16usize, 32usize), (16, 48), (2, 4), (3, 6)] {
        // the builder itself, at a few rows
        for m in [1usize, 2, 17, 33] {
            let c = case(
                n_k,
                n_v,
                m,
                GdnForm::QWEN36_GGUF,
                0x5c0a_0000 + (n_v * 100 + m) as u64,
            );
            assert_eq!(c.conv.len(), m * c.conv_ld());
            assert_eq!(c.proj.len(), m * c.ld());
            assert_eq!((c.a.len(), c.dt_bias.len(), c.norm_w.len()), (n_v, n_v, D));
            assert_eq!(c.state0.len(), n_v * D * D);
        }
        let mut plans: Vec<(usize, Vec<usize>)> = Vec::new();
        for (cuts, total) in [SCAN_CUTS, SCAN_CUTS_QUICK] {
            plans.extend(cuts.iter().map(|cut| (total, cut.to_vec())));
        }
        for m in [
            1usize, 2, 7, 8, 9, 15, 16, 17, 31, 33, 100, 300, 1024, 1500, 2051,
        ] {
            plans.push((m, vec![m]));
        }
        plans.push((4, vec![4])); // the refusal and bound-violation tests
        for (m, plan) in plans {
            assert_eq!(plan.iter().sum::<usize>(), m, "{plan:?}");
            let mut row0 = 0;
            for &rows in &plan {
                let shape = scan_shape(n_k, n_v, rows, row0, m + 8);
                let need = scan_bytes(shape).expect("sizes");
                let (conv_ld, ld) = (shape.conv_ld, shape.ld);
                assert!(
                    row0 * conv_ld * 4 + need.conv <= m * conv_ld * 4,
                    "conv {n_k} {n_v} {plan:?}"
                );
                assert!(
                    row0 * ld * 4 + need.proj <= m * ld * 4,
                    "proj {n_k} {n_v} {plan:?}"
                );
                assert!(
                    row0 * n_v * D * 4 + need.o_out <= m * n_v * D * 4,
                    "o_out {n_k} {n_v} {plan:?}"
                );
                assert_eq!(need.state, n_v * D * D * 4);
                assert_eq!(
                    (need.a, need.dt_bias, need.norm_w),
                    (n_v * 4, n_v * 4, D * 4)
                );
                row0 += rows;
            }
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

struct Case {
    n_k: usize,
    n_v: usize,
    m: usize,
    form: GdnForm,
    conv: Vec<f32>,
    proj: Vec<f32>,
    a: Vec<f32>,
    dt_bias: Vec<f32>,
    norm_w: Vec<f32>,
    state0: Vec<f32>,
}

impl Case {
    fn conv_ld(&self) -> usize {
        (2 * self.n_k + self.n_v) * D
    }
    fn beta_off(&self) -> usize {
        self.n_v * D
    }
    fn alpha_off(&self) -> usize {
        self.n_v * D + self.n_v
    }
    /// z, beta and alpha, then four floats the kernel must not read.
    fn ld(&self) -> usize {
        self.n_v * D + 2 * self.n_v + 4
    }
    fn shape(&self, m: usize, p0: usize, ctx_tokens: usize) -> ScanShape {
        scan_shape(self.n_k, self.n_v, m, p0, ctx_tokens)
    }
}

/// The scan shape of the device tests, free of the device so that a host test can run its arithmetic.
fn scan_shape(n_k: usize, n_v: usize, m: usize, p0: usize, ctx_tokens: usize) -> ScanShape {
    ScanShape {
        m,
        p0: p0 as i64,
        ctx_tokens,
        n_k_heads: n_k,
        n_v_heads: n_v,
        head_dim: D,
        conv_ld: (2 * n_k + n_v) * D,
        ld: n_v * D + 2 * n_v + 4,
        z_off: 0,
        beta_off: n_v * D,
        alpha_off: n_v * D + n_v,
    }
}

fn case(n_k: usize, n_v: usize, m: usize, form: GdnForm, seed: u64) -> Case {
    let mut rng = Rng::new(seed);
    let mut draw =
        |n: usize, sd: f64| -> Vec<f32> { (0..n).map(|_| (rng.normal() * sd) as f32).collect() };
    let conv_ld = (2 * n_k + n_v) * D;
    let mut conv = draw(m * conv_ld, 1.0);
    // Rows of very different size, and an all-zero row, so the L2 norms see the whole range.
    for t in 0..m {
        let scale = match t % 7 {
            0 => 1.0e3f32,
            3 => 1.0e-3,
            5 => 0.0,
            _ => 1.0,
        };
        conv[t * conv_ld..(t + 1) * conv_ld]
            .iter_mut()
            .for_each(|v| *v *= scale);
    }
    let ld = n_v * D + 2 * n_v + 4;
    let proj: Vec<f32> = draw(m * ld, 1.0);
    // ssm_a as stored is -exp(A_log): negative.
    let a = draw(n_v, 0.5)
        .into_iter()
        .map(|v| -v.abs() - 0.01)
        .collect();
    Case {
        n_k,
        n_v,
        m,
        form,
        conv,
        proj,
        a,
        dt_bias: draw(n_v, 0.5),
        norm_w: draw(D, 1.0),
        state0: draw(n_v * D * D, 0.3),
    }
}

struct Run {
    o_out: Vec<u8>,
    state: Vec<u8>,
    conv_after: Vec<u8>,
    proj_after: Vec<u8>,
}

fn float_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// Runs `pieces` (rows, kernel) in order over the rows of `c`, the state carried in one buffer.
fn run_pieces(rig: &Rig, c: &Case, pieces: &[(usize, Option<usize>)]) -> Run {
    try_pieces(rig, c, pieces).unwrap_or_else(|e| panic!("{pieces:?}: {e:?}"))
}

fn try_pieces(rig: &Rig, c: &Case, pieces: &[(usize, Option<usize>)]) -> Result<Run, EngError> {
    assert_eq!(pieces.iter().map(|p| p.0).sum::<usize>(), c.m);
    let ctx = &rig.ctx;
    let conv = testkit::upload_f32(ctx, &c.conv);
    let proj = testkit::upload_f32(ctx, &c.proj);
    let a = testkit::upload_f32(ctx, &c.a);
    let dt = testkit::upload_f32(ctx, &c.dt_bias);
    let nw = testkit::upload_f32(ctx, &c.norm_w);
    let state = testkit::upload_f32(ctx, &c.state0);
    let out_bytes = c.m * c.n_v * D * 4;
    let o_out = testkit::upload_filled(ctx, 0xa7, out_bytes + TAIL_BYTES);
    let mut row0 = 0;
    for &(rows, cols) in pieces {
        let shape = c.shape(rows, row0, c.m + 8);
        let need = scan_bytes(shape).expect("sizes");
        let call = GdnScanSeqCall {
            shape,
            eps: super::GDN_SCAN_EPS,
            conv: conv.range(row0 * c.conv_ld() * 4, need.conv).unwrap(),
            proj: proj.range(row0 * c.ld() * 4, need.proj).unwrap(),
            weights: ScanWeights {
                a: a.range(0, need.a).unwrap(),
                dt_bias: dt.range(0, need.dt_bias).unwrap(),
                norm_w: nw.range(0, need.norm_w).unwrap(),
            },
            state: state.range(0, need.state).unwrap(),
            o_out: o_out.range(row0 * c.n_v * D * 4, need.o_out).unwrap(),
            status: None,
        };
        ctx.gdn_scan_seq_cols(&rig.stream, call, c.form, cols)?;
        row0 += rows;
    }
    rig.stream.synchronize().expect("drain");
    let mut rows_and_tail = testkit::read_bytes(ctx, &o_out, out_bytes + TAIL_BYTES);
    let tail = rows_and_tail.split_off(out_bytes);
    assert!(
        tail.iter().all(|b| *b == 0xa7),
        "{pieces:?}: the o_out tail changed"
    );
    Ok(Run {
        o_out: rows_and_tail,
        state: testkit::read_bytes(ctx, &state, c.n_v * D * D * 4),
        conv_after: testkit::read_bytes(ctx, &conv, c.conv.len() * 4),
        proj_after: testkit::read_bytes(ctx, &proj, c.proj.len() * 4),
    })
}

fn assert_same(label: &str, got: &Run, want: &Run, c: &Case) {
    assert_eq!(
        got.conv_after,
        float_bytes(&c.conv),
        "{label}: conv changed"
    );
    assert_eq!(
        got.proj_after,
        float_bytes(&c.proj),
        "{label}: proj changed"
    );
    if got.state != want.state {
        let at = got
            .state
            .chunks(4)
            .zip(want.state.chunks(4))
            .position(|(g, w)| g != w)
            .unwrap();
        panic!(
            "{label}: state differs, first at head {} entry {}",
            at / (D * D),
            at % (D * D)
        );
    }
    if got.o_out != want.o_out {
        let at = got
            .o_out
            .chunks(4)
            .zip(want.o_out.chunks(4))
            .position(|(g, w)| g != w)
            .unwrap();
        panic!(
            "{label}: o_out differs, first at row {} head {} column {}",
            at / (c.n_v * D),
            (at / D) % c.n_v,
            at % D
        );
    }
}

fn check_widths(rig: &Rig, c: &Case, label: &str) {
    let want = run_pieces(rig, c, &[(c.m, None)]);
    for cols in WIDTHS {
        let got = run_pieces(rig, c, &[(c.m, Some(cols))]);
        assert_same(&format!("{label} cols {cols}"), &got, &want, c);
    }
}

#[test]
fn the_narrow_block_scan_writes_the_bytes_of_the_original() {
    if !testkit::device_present("scan cols kernel against the original") {
        return;
    }
    let rig = Rig::new();
    let rows: &[usize] = if quick() {
        &[1, 17, 33]
    } else {
        &[1, 2, 7, 8, 9, 15, 16, 17, 31, 33, 100]
    };
    // (name, key heads, value heads, form): the two models' head counts, a tiled and a grouped
    // small geometry, and a geometry whose value heads are not a power of two per key head.
    let geometries = [
        ("qwen36", 16usize, 32usize, GdnForm::QWEN36_GGUF),
        ("flash-next", 16, 48, GdnForm::FLASH_NEXT),
        ("small-tiled", 2, 4, GdnForm::QWEN36_GGUF),
        ("small-grouped", 3, 6, GdnForm::FLASH_NEXT),
    ];
    // The short lists keep both models' forms: the grouped form addresses q/k differently, and the
    // sanitizer legs run these lists.
    let geometries: &[_] = if quick() {
        &geometries[..2]
    } else {
        &geometries
    };
    for (g, &(name, n_k, n_v, form)) in geometries.iter().enumerate() {
        for (i, &m) in rows.iter().enumerate() {
            let c = case(n_k, n_v, m, form, 0x5c01_0000 + (g * 100 + i) as u64);
            check_widths(&rig, &c, &format!("{name} m {m}"));
        }
    }
    if quick() {
        return;
    }
    // The chunk sizes of the prefill.
    for (name, n_k, n_v, form) in [
        ("qwen36", 16usize, 32usize, GdnForm::QWEN36_GGUF),
        ("flash-next", 16, 48, GdnForm::FLASH_NEXT),
    ] {
        let c = case(n_k, n_v, 1024, form, 0x5c02_0001);
        check_widths(&rig, &c, &format!("{name} m 1024"));
    }
    let c = case(16, 32, 2051, GdnForm::QWEN36_GGUF, 0x5c02_0002);
    check_widths(&rig, &c, "qwen36 m 2051");
}

#[test]
fn pieces_carry_the_state_across_launches_and_across_kernels() {
    if !testkit::device_present("scan cols kernel piece handoff") {
        return;
    }
    let rig = Rig::new();
    let forms = [(GdnForm::QWEN36_GGUF, 32usize), (GdnForm::FLASH_NEXT, 48)];
    for &(form, n_v) in &forms {
        let total = if quick() {
            SCAN_CUTS_QUICK.1
        } else {
            SCAN_CUTS.1
        };
        let c = case(16, n_v, total, form, 0x5c03_0000 + n_v as u64);
        let want = run_pieces(&rig, &c, &[(c.m, None)]);
        let (cuts, _) = if quick() { SCAN_CUTS_QUICK } else { SCAN_CUTS };
        for cut in cuts {
            let plans: [Vec<(usize, Option<usize>)>; 4] = [
                cut.iter().map(|&r| (r, None)).collect(),
                cut.iter().map(|&r| (r, Some(16))).collect(),
                cut.iter()
                    .enumerate()
                    .map(|(i, &r)| (r, [None, Some(8), Some(32), Some(16)][i % 4]))
                    .collect(),
                cut.iter()
                    .enumerate()
                    .map(|(i, &r)| (r, if i % 2 == 0 { Some(16) } else { None }))
                    .collect(),
            ];
            for plan in plans {
                let got = run_pieces(&rig, &c, &plan);
                assert_same(&format!("n_v {n_v} pieces {plan:?}"), &got, &want, &c);
            }
        }
    }
}

#[test]
fn the_switch_routes_only_long_scans_to_the_narrow_block_kernel() {
    if !testkit::device_present("scan cols routing") {
        return;
    }
    let rig = Rig::new();
    for (setting, m, on_cols) in [
        (None, 300usize, false),
        (Some(16), 1, false),
        (Some(16), 8, false),
        (Some(16), SCAN_MIN_ROWS - 1, false),
        (Some(16), SCAN_MIN_ROWS, true),
        (Some(8), 70, true),
        (Some(32), 300, true),
    ] {
        let c = case(2, 4, m, GdnForm::QWEN36_GGUF, 0x5c06_0000 + m as u64);
        let before = super::COLS_CALLS.with(std::cell::Cell::get);
        let call_pieces = |setting: Option<usize>| {
            let ctx = &rig.ctx;
            let conv = testkit::upload_f32(ctx, &c.conv);
            let proj = testkit::upload_f32(ctx, &c.proj);
            let a = testkit::upload_f32(ctx, &c.a);
            let dt = testkit::upload_f32(ctx, &c.dt_bias);
            let nw = testkit::upload_f32(ctx, &c.norm_w);
            let state = testkit::upload_f32(ctx, &c.state0);
            let o_out = testkit::upload_filled(ctx, 0xa7, m * c.n_v * D * 4);
            let shape = c.shape(m, 0, m + 8);
            let need = scan_bytes(shape).expect("sizes");
            let call = GdnScanSeqCall {
                shape,
                eps: super::GDN_SCAN_EPS,
                conv: conv.range(0, need.conv).unwrap(),
                proj: proj.range(0, need.proj).unwrap(),
                weights: ScanWeights {
                    a: a.range(0, need.a).unwrap(),
                    dt_bias: dt.range(0, need.dt_bias).unwrap(),
                    norm_w: nw.range(0, need.norm_w).unwrap(),
                },
                state: state.range(0, need.state).unwrap(),
                o_out: o_out.range(0, need.o_out).unwrap(),
                status: None,
            };
            ctx.gdn_scan_seq_with(&rig.stream, call, c.form, setting)
                .expect("launch");
            rig.stream.synchronize().expect("drain");
            (
                testkit::read_bytes(ctx, &o_out, need.o_out),
                testkit::read_bytes(ctx, &state, need.state),
            )
        };
        let got = call_pieces(setting);
        let ran = super::COLS_CALLS.with(std::cell::Cell::get) - before;
        assert_eq!(ran, usize::from(on_cols), "setting {setting:?} m {m}");
        let want = run_pieces(&rig, &c, &[(m, None)]);
        assert_eq!(
            (got.0, got.1),
            (want.o_out, want.state),
            "setting {setting:?} m {m}"
        );
    }
}

#[test]
fn the_narrow_block_wrapper_refuses_a_width_it_does_not_have() {
    if !testkit::device_present("scan cols wrapper refusals") {
        return;
    }
    let rig = Rig::new();
    let c = case(2, 4, 4, GdnForm::QWEN36_GGUF, 0x5c04_0001);
    for bad in [0usize, 1, 4, 24, 64, 128] {
        match try_pieces(&rig, &c, &[(4, Some(bad))]) {
            Err(EngError::BadRequest(message)) => {
                assert!(
                    message.contains("value columns per block"),
                    "{bad}: {message}"
                );
            }
            other => panic!(
                "width {bad}: expected a BadRequest, got {:?}",
                other.map(|_| ())
            ),
        }
    }
    let want = run_pieces(&rig, &c, &[(4, None)]);
    let got = run_pieces(&rig, &c, &[(4, Some(16))]);
    assert_same("after the refusals", &got, &want, &c);
}

/// Wall time of one DeltaNet layer pass of 30 scan launches (scan plus out-gate), queued back to back
/// and drained once, for the original kernel and each block width at the prefill shapes (2,048 and 1,024
/// rows, the Qwen3.6 and Flash-Next head counts). The kernels take turns in every round. Run alone in an
/// exclusive window: `--ignored --exact --nocapture`. Prints `BENCH` lines.
#[test]
#[ignore = "timing: run alone in an exclusive window"]
fn scan_cols_kernel_timing() {
    if !testkit::device_present("scan cols kernel timing") {
        return;
    }
    const LAYERS: usize = 30;
    const ROUNDS: usize = 31;
    const WARM: usize = 4;
    let rig = Rig::new();
    let ctx = &rig.ctx;
    let kernels = [None, Some(8), Some(16), Some(32)];
    for (name, n_v, form) in [
        ("qwen36", 32usize, GdnForm::QWEN36_GGUF),
        ("flash-next", 48, GdnForm::FLASH_NEXT),
    ] {
        for m in [2048usize, 1024] {
            let c = case(16, n_v, m, form, 0x5c05_0000 + m as u64);
            let conv = testkit::upload_f32(ctx, &c.conv);
            let proj = testkit::upload_f32(ctx, &c.proj);
            let a = testkit::upload_f32(ctx, &c.a);
            let dt = testkit::upload_f32(ctx, &c.dt_bias);
            let nw = testkit::upload_f32(ctx, &c.norm_w);
            let states: Vec<_> = (0..LAYERS)
                .map(|_| testkit::upload_f32(ctx, &c.state0))
                .collect();
            let o_out = testkit::upload_filled(ctx, 0, m * n_v * D * 4);
            let shape = c.shape(m, 0, m + 8);
            let need = scan_bytes(shape).expect("sizes");
            let mut times = vec![Vec::new(); kernels.len()];
            for round in 0..WARM + ROUNDS {
                for step in 0..kernels.len() {
                    let slot = (step + round) % kernels.len();
                    let cols = kernels[slot];
                    let started = std::time::Instant::now();
                    for state in &states {
                        let call = GdnScanSeqCall {
                            shape,
                            eps: super::GDN_SCAN_EPS,
                            conv: conv.range(0, need.conv).unwrap(),
                            proj: proj.range(0, need.proj).unwrap(),
                            weights: ScanWeights {
                                a: a.range(0, need.a).unwrap(),
                                dt_bias: dt.range(0, need.dt_bias).unwrap(),
                                norm_w: nw.range(0, need.norm_w).unwrap(),
                            },
                            state: state.range(0, need.state).unwrap(),
                            o_out: o_out.range(0, need.o_out).unwrap(),
                            status: None,
                        };
                        ctx.gdn_scan_seq_cols(&rig.stream, call, form, cols)
                            .expect("launch");
                    }
                    rig.stream.synchronize().expect("drain");
                    if round >= WARM {
                        times[slot].push(started.elapsed().as_secs_f64() * 1e6 / LAYERS as f64);
                    }
                }
            }
            for (slot, cols) in kernels.iter().enumerate() {
                let mut t = times[slot].clone();
                t.sort_by(f64::total_cmp);
                let kernel = cols.map_or("Original".to_owned(), |w| format!("Cols({w})"));
                println!(
                    "BENCH scan shape={name} m={m} kernel={kernel} p50_us={:.1} min_us={:.1}",
                    t[t.len() / 2],
                    t[0]
                );
            }
        }
    }
}

/// Each deliberately wrong variant of the narrow-block scan kernel and of its switch code must turn the
/// byte-identity or switch test that guards it red, and the unmutated control must pass the same short
/// shape lists. Needs the `cuda-gdn-mutants` build and a device for the kernel mutants;
/// run with `--test-threads=1`.
#[cfg(feature = "cuda-gdn-mutants")]
mod mutants {
    // Accepted survivors, by construction (no test can see them): a dispatch-mapping swap such as
    // `case 16: launch_cols<8>` instead of `launch_cols<4>` writes the same bytes at a different launch shape.
    // Equivalent mutants: removing the clamp `min(t0 + slot, m - 1)` (every use of the row pointer is already
    // under `slot < len`, so it only avoids forming a pointer past the object) and the `slot < len` guard on the
    // q/k staging while that clamp exists (the dead `qk_s` slots are never read). Mutant 20 (the tile barrier) is
    // seen by racecheck, not by bytes; the sanitizer legs of tools/sanitizer_run.sh kill it.
    use super::*;
    use crate::kernels::cuda::gdn_mutants::{self as gm, select};
    use crate::kernels::cuda::native::prefill::gdn_scan_seq::{
        GdnScanSeqRaw, gdn_scan_seq_cols_bad_bounds,
    };

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
        the_scan_columns_value_is_strict();
        only_a_long_scan_under_a_set_switch_takes_the_narrow_block_kernel();
        the_narrow_block_scan_writes_the_bytes_of_the_original();
        pieces_carry_the_state_across_launches_and_across_kernels();
        the_narrow_block_kernel_guards_a_bad_bound_at_every_width();
        the_switch_routes_only_long_scans_to_the_narrow_block_kernel();
    }

    #[test]
    fn mutant_scan_column_offset_is_killed() {
        killed(
            6,
            "column offset",
            the_narrow_block_scan_writes_the_bytes_of_the_original,
        );
    }

    #[test]
    fn mutant_scan_v_offset_is_killed() {
        killed(
            7,
            "v offset",
            the_narrow_block_scan_writes_the_bytes_of_the_original,
        );
    }

    #[test]
    fn mutant_scan_v_staged_for_one_slot_is_killed() {
        killed(
            8,
            "v staged for one slot",
            the_narrow_block_scan_writes_the_bytes_of_the_original,
        );
    }

    #[test]
    fn mutant_scan_grid_too_small_is_killed() {
        killed(
            9,
            "grid too small",
            the_narrow_block_scan_writes_the_bytes_of_the_original,
        );
    }

    #[test]
    fn mutant_scan_sum_order_is_killed() {
        killed(
            10,
            "sum order",
            the_narrow_block_scan_writes_the_bytes_of_the_original,
        );
    }

    #[test]
    fn mutant_scan_state_store_dropped_is_killed() {
        killed(
            11,
            "state store dropped",
            pieces_carry_the_state_across_launches_and_across_kernels,
        );
    }

    #[test]
    fn mutant_scan_threshold_one_is_killed() {
        killed(
            gm::SCAN_MIN_ONE,
            "threshold one",
            only_a_long_scan_under_a_set_switch_takes_the_narrow_block_kernel,
        );
    }

    #[test]
    fn mutant_scan_bound_guard_disabled_is_killed() {
        killed(
            23,
            "bound guard disabled",
            the_narrow_block_kernel_guards_a_bad_bound_at_every_width,
        );
    }

    /// The cols kernel's own bound-violation branch, at every width (the fault-qualification entry
    /// reaches only the original kernel). A refused launch loads no state, NaNs every o_out element and
    /// ORs mask 8 into the status word.
    #[test]
    fn the_narrow_block_kernel_guards_a_bad_bound_at_every_width() {
        if !testkit::device_present("scan cols bound violation") {
            return;
        }
        let rig = Rig::new();
        let ctx = &rig.ctx;
        let c = case(2, 4, 4, GdnForm::QWEN36_GGUF, 0x5c07_0001);
        let shape = c.shape(c.m, 0, 1024);
        let need = scan_bytes(shape).expect("sizes");
        for cols in WIDTHS {
            let conv = testkit::upload_f32(ctx, &c.conv);
            let proj = testkit::upload_f32(ctx, &c.proj);
            let a = testkit::upload_f32(ctx, &c.a);
            let dt = testkit::upload_f32(ctx, &c.dt_bias);
            let nw = testkit::upload_f32(ctx, &c.norm_w);
            let state = testkit::upload_f32(ctx, &c.state0);
            let o_out = testkit::upload_filled(ctx, 0xff, need.o_out);
            let status = testkit::upload_f32(ctx, &[0.0]);
            let io = GdnScanSeqRaw {
                conv: ctx.raw_range(&conv.range(0, need.conv).unwrap()).unwrap(),
                proj: ctx.raw_range(&proj.range(0, need.proj).unwrap()).unwrap(),
                a: ctx.raw_range(&a.range(0, need.a).unwrap()).unwrap(),
                dt_bias: ctx.raw_range(&dt.range(0, need.dt_bias).unwrap()).unwrap(),
                norm_w: ctx.raw_range(&nw.range(0, need.norm_w).unwrap()).unwrap(),
                state: ctx.raw_range(&state.range(0, need.state).unwrap()).unwrap(),
                o_out: ctx.raw_range(&o_out.range(0, need.o_out).unwrap()).unwrap(),
                status: Some(ctx.raw_range(&status.range(0, 4).unwrap()).unwrap()),
                ints: super::super::scan_ints(&shape).expect("ints"),
                eps: super::super::GDN_SCAN_EPS,
                form: 0,
            };
            ctx.launch_on(&rig.stream, |raw| {
                gdn_scan_seq_cols_bad_bounds(raw, io, cols)
            })
            .expect("the guard launch");
            rig.stream.synchronize().expect("drain");
            let out = testkit::floats(&testkit::read_bytes(ctx, &o_out, need.o_out));
            assert!(
                out.iter().all(|v| v.is_nan()),
                "width {cols}: a refused launch must NaN every o_out element"
            );
            assert_eq!(
                testkit::read_bytes(ctx, &state, need.state),
                float_bytes(&c.state0),
                "width {cols}: a refused launch must not touch the state"
            );
            let word = testkit::read_bytes(ctx, &status, 4);
            assert_eq!(
                i32::from_le_bytes([word[0], word[1], word[2], word[3]]),
                crate::kernels::cuda::PREFILL_STATUS_BOUND_VIOLATION,
                "width {cols}: mask 8 of the prefill status word"
            );
        }
    }

    /// For the sanitizer runs: selects the mutant named by `GDN_MUTANT_ID` and runs the byte tests plainly
    /// (no catch_unwind), so `compute-sanitizer --error-exitcode` reports what memcheck, racecheck or
    /// synccheck see. A mutant the byte tests cannot see (19, 20) is killed by the sanitizer's exit code.
    #[test]
    #[ignore = "sanitizer runs only: set GDN_MUTANT_ID"]
    fn mutant_from_the_environment_runs_the_byte_tests() {
        let id: u32 = std::env::var("GDN_MUTANT_ID")
            .expect("GDN_MUTANT_ID")
            .parse()
            .expect("a mutant id");
        let _selected = select(id);
        the_narrow_block_scan_writes_the_bytes_of_the_original();
        pieces_carry_the_state_across_launches_and_across_kernels();
    }

    #[test]
    fn mutant_scan_accepts_four_is_killed() {
        killed(
            gm::SCAN_ACCEPTS_FOUR,
            "accepts 4",
            the_scan_columns_value_is_strict,
        );
    }

    #[test]
    fn mutant_scan_value_not_trimmed_is_killed() {
        killed(
            gm::SCAN_NO_TRIM,
            "value not trimmed",
            the_scan_columns_value_is_strict,
        );
    }
}
