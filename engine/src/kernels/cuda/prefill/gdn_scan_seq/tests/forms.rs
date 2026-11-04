//! The form rules of `gdn_scan_seq` (grouped key heads, the `rsqrt(sum + eps)` L2 norm, the
//! sigmoid output gate) against the f64 reference of tests 3 to 11, which models each rule
//! through [`RefOpts`], and the Qwen3.8 Flash-Next geometry (16 key heads, 48 value heads).
//! The form-0 instance is the Qwen3.6 kernel: every test of the parent module runs it.

use super::{
    D, E3, RefOpts, Rig, ScanInputs, ScanLayout, ScanRig, assert_constraints, assert_scan_finite,
    generate_scan, judge_scan_with, le_bytes, median_us, scan_reference, seed_of,
};
use crate::EngError;
use crate::kernels::cuda::DeviceBuffer;
use crate::kernels::cuda::prefill::gdn_scan_seq::{
    GDN_SCAN_EPS, GdnForm, GdnGate, GdnKeyHeads, GdnL2, GdnScanSeqCall, ScanWeights, scan_bytes,
};
use crate::kernels::cuda::resident::gdn_conv::{ConvCall, ConvShape};
use crate::kernels::cuda::resident::gdn_delta::{DeltaCall, DeltaShape, DeltaWeights, delta_bytes};
use crate::kernels::cuda::testkit;

/// Flash-Next: 16 key heads, 48 value heads (`linear_num_key_heads`, `linear_num_value_heads`).
const FN_K: usize = 16;
const FN_V: usize = 48;

/// The form with rule bits `bits` (1 grouped heads, 2 add-eps L2, 4 sigmoid gate; the C bits).
fn form_of(bits: u8) -> GdnForm {
    GdnForm {
        key_heads: if bits & 1 != 0 {
            GdnKeyHeads::Grouped
        } else {
            GdnKeyHeads::Tiled
        },
        l2: if bits & 2 != 0 {
            GdnL2::AddEps
        } else {
            GdnL2::Ggml
        },
        gate: if bits & 4 != 0 {
            GdnGate::Sigmoid
        } else {
            GdnGate::Silu
        },
    }
}

/// The reference variant that models `form`.
fn opts_of(form: GdnForm) -> RefOpts {
    RefOpts {
        mlx_key_map: form.key_heads == GdnKeyHeads::Grouped,
        decode_norm_form: form.l2 == GdnL2::AddEps,
        sigmoid_gate: form.gate == GdnGate::Sigmoid,
        no_q_scale: false,
    }
}

/// The largest gated-row error of `got` against the reference `opts`, each row relative to
/// max(|ref|, 1) as in `judge_scan`. The rows must be finite first.
fn rows_off(inp: &ScanInputs, got: &[f32], opts: RefOpts) -> f64 {
    let want = scan_reference(inp, opts).out;
    let (want_rows, got_rows) = (want.as_chunks::<D>().0, got.as_chunks::<D>().0);
    want_rows
        .iter()
        .zip(got_rows)
        .map(|(w, g)| {
            let scale = w.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            w.iter()
                .zip(g)
                .fold(0.0f64, |m, (a, b)| m.max((a - f64::from(*b)).abs()))
                / scale
        })
        .fold(0.0, f64::max)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn the_form_bits_are_the_header_bits() {
    assert_eq!(form_of(0), GdnForm::QWEN36_GGUF);
    assert_eq!(form_of(7), GdnForm::FLASH_NEXT);
    let forms: Vec<GdnForm> = (0..8).map(form_of).collect();
    for (i, a) in forms.iter().enumerate() {
        for b in &forms[i + 1..] {
            assert_ne!(a, b, "two bit patterns give one form");
        }
    }
}

/// Every one of the 8 forms against the reference that models it (bound E3, the derivation of
/// test 3), and against the reference with each single rule flipped (more than E3 off), so a
/// form whose launch reached the wrong instance, or a rule that does nothing, fails. The data
/// gives every rule a visible effect: 4 key heads over 12 value heads (the tiled and grouped
/// maps differ for every value head but 0 and 11... and 4 and 7), z ~ N(0, 3^2) (silu and
/// sigmoid differ in sign and size), and key head 1's k row is `[1e-5, 0, ...]` on every step
/// (sum 1e-10: the ggml form normalizes it to 1, the add-eps form to about 0.01, test 8's row).
#[test]
fn gdn_scan_seq_every_form_matches_its_reference_and_no_neighbour() -> Result<(), EngError> {
    let (nk, nv, m) = (4, 12, 9);
    let mut inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xf0));
    for t in 0..m {
        let near = &mut inp.conv[t * inp.lay.conv_ld + (nk + 1) * D..][..D];
        near.fill(0.0);
        near[0] = 1e-5;
    }
    assert_constraints(&inp, &[1]);
    if !testkit::device_present("gdn_scan_seq_every_form_matches_its_reference_and_no_neighbour") {
        return Ok(());
    }
    let mut runs = Vec::new();
    {
        let scan = ScanRig::new(Rig::new()?, &inp)?;
        for b in 0..8u8 {
            scan.reset_state(&inp)?;
            scan.poison_out()?;
            scan.run_form(&inp, (m, 0, 0), form_of(b))?;
            let got = scan.finish()?;
            runs.push((b, got.out, got.state));
        }
    }
    for (b, out, state) in &runs {
        let form = form_of(*b);
        let label = format!("forms bits {b}");
        let (rows, st) = judge_scan_with(&inp, (out, state), &label, opts_of(form));
        assert!(rows <= E3, "{label}: rows {rows:.3e} above E3 {E3:.3e}");
        assert!(st <= E3, "{label}: state {st:.3e} above E3 {E3:.3e}");
        for flip in [1u8, 2, 4] {
            let off = rows_off(&inp, out, opts_of(form_of(b ^ flip)));
            println!(
                "REPORTED: {label}: the reference with bit {flip} flipped is off by {off:.3e}"
            );
            assert!(
                off > E3,
                "{label}: the output also matches the reference with bit {flip} flipped"
            );
        }
    }
    Ok(())
}

/// The Flash-Next form at the Flash-Next geometry against the f64 reference at the M edges
/// around the 16-token tile (1, 15, 16, 17, 33) and at 64.
#[test]
fn gdn_scan_seq_flash_next_geometry_m_edges() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_flash_next_geometry_m_edges") {
        return Ok(());
    }
    for &m in &[1usize, 15, 16, 17, 33, 64] {
        let lay = ScanLayout::of(FN_K, FN_V);
        let inp = generate_scan(FN_K, FN_V, m, lay, seed_of(FN_K, FN_V, m, 0xf1));
        assert_constraints(&inp, &[]);
        let got = {
            let scan = ScanRig::new(Rig::new()?, &inp)?;
            scan.run_form(&inp, (m, 0, 0), GdnForm::FLASH_NEXT)?;
            scan.finish()?
        };
        assert_eq!(got.conv, inp.conv, "the scan changed its conv input rows");
        let label = format!("flash-next M {m}");
        let (rows, state) = judge_scan_with(
            &inp,
            (&got.out, &got.state),
            &label,
            opts_of(GdnForm::FLASH_NEXT),
        );
        assert!(rows <= E3, "{label}: rows {rows:.3e} above E3 {E3:.3e}");
        assert!(state <= E3, "{label}: state {state:.3e} above E3 {E3:.3e}");
    }
    Ok(())
}

/// The Flash-Next form over 23 tokens: one chunk, two chunks (17 then 6, across the tile
/// edge) and 23 one-token calls with the state carried in its buffer give the same bits
/// (same expressions in the same order; the state round-trips through f32 either way). This is
/// the decode path: a one-token call of this entry.
#[test]
fn gdn_scan_seq_flash_next_chunks_and_tokens_equal_one_chunk() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_flash_next_chunks_and_tokens_equal_one_chunk") {
        return Ok(());
    }
    let m = 23;
    let inp = generate_scan(
        FN_K,
        FN_V,
        m,
        ScanLayout::of(FN_K, FN_V),
        seed_of(FN_K, FN_V, m, 0xf2),
    );
    assert_constraints(&inp, &[]);
    let form = GdnForm::FLASH_NEXT;
    let (whole, chunks, tokens) = {
        let scan = ScanRig::new(Rig::new()?, &inp)?;
        scan.run_form(&inp, (m, 0, 0), form)?;
        let whole = scan.finish()?;
        scan.reset_state(&inp)?;
        scan.poison_out()?;
        scan.run_form(&inp, (17, 0, 0), form)?;
        scan.run_form(&inp, (6, 17, 17), form)?;
        let chunks = scan.finish()?;
        scan.reset_state(&inp)?;
        scan.poison_out()?;
        for t in 0..m {
            scan.run_form(&inp, (1, t as i64, t), form)?;
        }
        (whole, chunks, scan.finish()?)
    };
    assert_scan_finite("flash-next one chunk", &inp, &whole.out, &whole.state);
    assert_scan_finite("flash-next two chunks", &inp, &chunks.out, &chunks.state);
    assert_scan_finite("flash-next tokens", &inp, &tokens.out, &tokens.state);
    assert_eq!(
        bits(&whole.out),
        bits(&chunks.out),
        "two chunks: rows differ"
    );
    assert_eq!(
        bits(&whole.state),
        bits(&chunks.state),
        "two chunks: state differs"
    );
    assert_eq!(bits(&whole.out), bits(&tokens.out), "tokens: rows differ");
    assert_eq!(
        bits(&whole.state),
        bits(&tokens.state),
        "tokens: state differs"
    );
    Ok(())
}

/// The decode question of the Flash-Next plan, measured: one decode token of one DeltaNet
/// layer through this entry at m = 1 (conv then scan) against the vendored decode delta
/// kernel at m = 1 on the same geometry. Reported, not gated. Each sample queues 36 layers
/// (the Flash-Next DeltaNet layer count) back to back and synchronizes once; the median of 11
/// samples after 2 warm-ups, divided by 36. Wall clock around a drained stream, host wrapper
/// time included (the profile of the test build is printed).
#[test]
fn gdn_scan_seq_flash_next_decode_us() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_flash_next_decode_us") {
        return Ok(());
    }
    const LAYERS: usize = 36;
    let lay = ScanLayout::of(FN_K, FN_V);
    let inp = generate_scan(FN_K, FN_V, 1, lay, seed_of(FN_K, FN_V, 1, 0xf3));
    let rig = Rig::new()?;
    let ctx = &rig.ctx;
    let channels = lay.conv_ld;
    let upload = |values: &[f32]| ctx.upload(&le_bytes(values));
    let (conv, proj) = (upload(&inp.conv)?, upload(&inp.proj)?);
    let (a, dt, nw) = (upload(&inp.a)?, upload(&inp.dt)?, upload(&inp.nw)?);
    let state = upload(&inp.state0)?;
    let w: DeviceBuffer = upload(&vec![0.25f32; channels * 4])?;
    let ring = upload(&vec![0.0f32; 3 * channels])?;
    let need = scan_bytes(inp.shape(1, 0, 1024))?;
    let (conv_out, o_out) = (
        ctx.upload(&vec![0u8; need.conv])?,
        ctx.upload(&vec![0u8; need.o_out])?,
    );
    let conv_shape = ConvShape {
        m: 1,
        channels,
        ld: lay.ld,
    };
    let queue_conv = || {
        ctx.gdn_conv(
            &rig.stream,
            ConvCall {
                shape: conv_shape,
                proj: proj.range(0, channels * 4)?,
                weights: w.range(0, channels * 16)?,
                state: ring.range(0, 3 * channels * 4)?,
                out: conv_out.range(0, need.conv)?,
            },
        )
    };
    let queue_scan = || {
        ctx.gdn_scan_seq_form(
            &rig.stream,
            GdnScanSeqCall {
                shape: inp.shape(1, 0, 1024),
                eps: GDN_SCAN_EPS,
                conv: conv_out.range(0, need.conv)?,
                proj: proj.range(0, need.proj)?,
                weights: ScanWeights {
                    a: a.range(0, need.a)?,
                    dt_bias: dt.range(0, need.dt_bias)?,
                    norm_w: nw.range(0, need.norm_w)?,
                },
                state: state.range(0, need.state)?,
                o_out: o_out.range(0, need.o_out)?,
                status: None,
            },
            GdnForm::FLASH_NEXT,
        )
    };
    let delta_shape = DeltaShape {
        m: 1,
        n_k_heads: FN_K,
        n_v_heads: FN_V,
        conv_ld: lay.conv_ld,
        ld: lay.ld,
        z_off: lay.z_off,
        beta_off: lay.beta_off,
        alpha_off: lay.alpha_off,
    };
    let dneed = delta_bytes(delta_shape)?;
    let dout = ctx.upload(&vec![0u8; dneed.out])?;
    let queue_delta = || {
        ctx.gdn_delta(
            &rig.stream,
            DeltaCall {
                shape: delta_shape,
                eps: GDN_SCAN_EPS,
                conv: conv.range(0, dneed.conv)?,
                proj: proj.range(0, dneed.proj)?,
                weights: DeltaWeights {
                    a: a.range(0, dneed.a)?,
                    dt_bias: dt.range(0, dneed.dt_bias)?,
                    norm_w: nw.range(0, dneed.norm_w)?,
                },
                state: state.range(0, dneed.state)?,
                out: dout.range(0, dneed.out)?,
            },
        )
    };
    let per_layer = |queue: &dyn Fn() -> Result<(), EngError>| -> Result<f64, EngError> {
        let us = median_us(11, 2, || {
            for _ in 0..LAYERS {
                queue()?;
            }
            rig.stream.synchronize()
        })?;
        Ok(us / LAYERS as f64)
    };
    let conv_us = per_layer(&queue_conv)?;
    let scan_us = per_layer(&queue_scan)?;
    let both_us = per_layer(&|| {
        queue_conv()?;
        queue_scan()
    })?;
    let delta_us = per_layer(&queue_delta)?;
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    println!(
        "TIMING flash-next decode ({profile} test build, per DeltaNet layer at m = 1, 16 x 48 heads): conv {conv_us:.1} us, scan entry {scan_us:.1} us, conv + scan {both_us:.1} us; vendored decode delta {delta_us:.1} us; 36 layers conv + scan {:.2} ms per token",
        both_us * LAYERS as f64 / 1e3
    );
    let _ = o_out.bytes();
    Ok(())
}
