//! Tests of the scan kernel with narrower blocks (`gdn_scan_seq_cols_k`, 8, 16 or 32 value columns
//! per block) against the original `gdn_scan_seq_k`: the state and the gated `o_out` rows are the
//! same bytes for every block width, form, head geometry, row count and way a prompt is cut into
//! pieces. The host test of the switch runs anywhere; the device tests skip on a host without a
//! device.

use std::ffi::OsStr;

use super::{GdnForm, GdnScanSeqCall, ScanShape, ScanWeights, parse_scan_cols, scan_bytes};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceStream};

const D: usize = 128;
const WIDTHS: [usize; 3] = [8, 16, 32];

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
        "", "0", "1", "4", "7", "64", "128", "016", "+16", "16.0", "on", "1", "-8",
    ] {
        assert_eq!(parse_scan_cols(Some(OsStr::new(other))), None, "{other:?}");
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
    let o_out = testkit::upload_filled(ctx, 0xa7, out_bytes);
    let mut row0 = 0;
    for &(rows, cols) in pieces {
        let shape = ScanShape {
            m: rows,
            p0: row0 as i64,
            ctx_tokens: c.m + 8,
            n_k_heads: c.n_k,
            n_v_heads: c.n_v,
            head_dim: D,
            conv_ld: c.conv_ld(),
            ld: c.ld(),
            z_off: 0,
            beta_off: c.beta_off(),
            alpha_off: c.alpha_off(),
        };
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
    Ok(Run {
        o_out: testkit::read_bytes(ctx, &o_out, out_bytes),
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
    let rows = [1usize, 2, 7, 8, 9, 15, 16, 17, 31, 33, 100];
    // (name, key heads, value heads, form): the two models' head counts, a tiled and a grouped
    // small geometry, and a geometry whose value heads are not a power of two per key head.
    let geometries = [
        ("qwen36", 16usize, 32usize, GdnForm::QWEN36_GGUF),
        ("flash-next", 16, 48, GdnForm::FLASH_NEXT),
        ("small-tiled", 2, 4, GdnForm::QWEN36_GGUF),
        ("small-grouped", 3, 6, GdnForm::FLASH_NEXT),
    ];
    for (g, &(name, n_k, n_v, form)) in geometries.iter().enumerate() {
        for (i, &m) in rows.iter().enumerate() {
            let c = case(n_k, n_v, m, form, 0x5c01_0000 + (g * 100 + i) as u64);
            check_widths(&rig, &c, &format!("{name} m {m}"));
        }
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
    for (form, n_v) in [(GdnForm::QWEN36_GGUF, 32usize), (GdnForm::FLASH_NEXT, 48)] {
        let c = case(16, n_v, 1500, form, 0x5c03_0000 + n_v as u64);
        let want = run_pieces(&rig, &c, &[(c.m, None)]);
        let cuts: [&[usize]; 3] = [
            &[1024, 300, 176],
            &[1, 2, 1497],
            &[512, 512, 512, 16, 1, 3, 4],
        ];
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
            let shape = ScanShape {
                m,
                p0: 0,
                ctx_tokens: m + 8,
                n_k_heads: 16,
                n_v_heads: n_v,
                head_dim: D,
                conv_ld: c.conv_ld(),
                ld: c.ld(),
                z_off: 0,
                beta_off: c.beta_off(),
                alpha_off: c.alpha_off(),
            };
            let need = scan_bytes(shape).expect("sizes");
            let mut times = vec![Vec::new(); kernels.len()];
            for round in 0..WARM + ROUNDS {
                for (slot, &cols) in kernels.iter().enumerate() {
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
