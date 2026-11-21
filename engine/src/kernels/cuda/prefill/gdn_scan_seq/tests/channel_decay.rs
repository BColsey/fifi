//! The decay-row form of `gdn_scan_seq`: a decay that is one value per key channel and token,
//! read from rows the caller computed. Its per-channel behaviour is pinned end to end against a
//! host scan by the block_ops delta-scan test; here the form is pinned against the per-head
//! form (a row of one repeated value must give the per-head result) and its refusals.

use super::{
    D, Rig, ScanInputs, ScanLayout, ScanRig, generate_scan, le_bytes, scan_bytes, seed_of,
};
use crate::EngError;
use crate::kernels::cuda::prefill::gdn_scan_seq::{
    GDN_SCAN_EPS, GdnChannelScanCall, GdnForm, GdnGate, GdnKeyHeads, GdnL2, ScanShape,
};
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{DeviceBuffer, DeviceRange};

const FORM: GdnForm = GdnForm {
    key_heads: GdnKeyHeads::Tiled,
    l2: GdnL2::Ggml,
    gate: GdnGate::Sigmoid,
};

/// The per-head gate of the scan kernel as one f32 value per head and token, repeated over the
/// 128 channels of the head: `exp(softplus(alpha + dt_bias) * a)`.
fn repeated_decay(inp: &ScanInputs) -> Vec<f32> {
    let mut rows = Vec::with_capacity(inp.m * inp.nv * D);
    for t in 0..inp.m {
        for h in 0..inp.nv {
            let alpha = inp.proj[t * inp.lay.ld + inp.lay.alpha_off + h] + inp.dt[h];
            let softplus = if alpha > 20.0 {
                alpha
            } else {
                (1.0 + alpha.exp()).ln()
            };
            let decay = (softplus * inp.a[h]).exp();
            rows.extend(std::iter::repeat_n(decay, D));
        }
    }
    rows
}

fn channel_call<'b>(
    scan: &'b ScanRig,
    shape: ScanShape,
    decay: &'b DeviceBuffer,
    decay_bytes: usize,
    norm_eps: f32,
) -> Result<GdnChannelScanCall<'b>, EngError> {
    let need = scan_bytes(shape)?;
    Ok(GdnChannelScanCall {
        shape,
        eps: GDN_SCAN_EPS,
        norm_eps,
        conv: scan.conv.range(0, need.conv)?,
        proj: scan.proj.range(0, need.proj)?,
        decay: decay.range(0, decay_bytes)?,
        norm_w: scan.nw.range(0, need.norm_w)?,
        state: scan.state.range(0, need.state)?,
        o_out: scan.o_out.range(0, need.o_out)?,
        status: None,
    })
}

fn worst_relative(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from((x - y).abs()) / f64::from(x.abs().max(1.0)))
        .fold(0.0, f64::max)
}

/// A decay row that repeats the per-head gate over the channels gives the per-head result
/// (the rows and the final state), over the tile edge and with several tiles.
#[test]
fn a_decay_row_of_one_repeated_value_gives_the_per_head_result() -> Result<(), EngError> {
    if !testkit::device_present("a_decay_row_of_one_repeated_value_gives_the_per_head_result") {
        return Ok(());
    }
    let (nk, nv) = (4usize, 12usize);
    for m in [1usize, 17, 40] {
        let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xc1));
        let scan = ScanRig::new(Rig::new()?, &inp)?;
        scan.reset_state(&inp)?;
        scan.poison_out()?;
        scan.run_form(&inp, (m, 0, 0), FORM)?;
        let per_head = scan.finish()?;
        let decay_values = repeated_decay(&inp);
        let decay = scan.rig.ctx.upload(&le_bytes(&decay_values))?;
        scan.reset_state(&inp)?;
        scan.poison_out()?;
        let shape = inp.shape(m, 0, m.max(1024));
        scan.rig.ctx.gdn_scan_seq_channel_decay(
            &scan.rig.stream,
            channel_call(&scan, shape, &decay, decay_values.len() * 4, GDN_SCAN_EPS)?,
            FORM,
        )?;
        let rows = scan.finish()?;
        let (out, state) = (
            worst_relative(&per_head.out, &rows.out),
            worst_relative(&per_head.state, &rows.state),
        );
        println!("REPORTED: m {m}: rows {out:.3e}, state {state:.3e} against the per-head form");
        assert!(out <= 2e-5, "m {m}: rows differ by {out:.3e}");
        assert!(state <= 2e-5, "m {m}: state differs by {state:.3e}");
        assert_eq!(rows.conv, inp.conv, "the scan changed its conv input rows");
    }
    Ok(())
}

/// The refusals of the decay-row call: a short decay range, a decay range over the state, and an
/// output norm eps outside (0, 1].
#[test]
fn the_decay_row_call_refuses_a_short_range_an_overlap_and_a_bad_norm_eps() -> Result<(), EngError>
{
    if !testkit::device_present(
        "the_decay_row_call_refuses_a_short_range_an_overlap_and_a_bad_norm_eps",
    ) {
        return Ok(());
    }
    let (nk, nv, m) = (4usize, 12usize, 5usize);
    let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xc2));
    let scan = ScanRig::new(Rig::new()?, &inp)?;
    let decay_values = repeated_decay(&inp);
    let decay = scan.rig.ctx.upload(&le_bytes(&decay_values))?;
    let shape = inp.shape(m, 0, 1024);
    let full = decay_values.len() * 4;
    let launch = |call: Result<GdnChannelScanCall<'_>, EngError>| {
        call.and_then(|call| {
            scan.rig
                .ctx
                .gdn_scan_seq_channel_decay(&scan.rig.stream, call, FORM)
        })
    };
    let short: Result<GdnChannelScanCall<'_>, EngError> = (|| {
        let mut call = channel_call(&scan, shape, &decay, full, GDN_SCAN_EPS)?;
        call.decay = decay.range(0, full - 4)?;
        Ok(call)
    })();
    let error = launch(short).expect_err("a decay range one float short");
    assert!(error.describe().contains("decay"), "{}", error.describe());
    let need = scan_bytes(shape)?;
    let over: Result<GdnChannelScanCall<'_>, EngError> = (|| {
        let mut call = channel_call(&scan, shape, &decay, full, GDN_SCAN_EPS)?;
        let state: DeviceRange<'_> = scan.state.range(0, need.state)?;
        call.decay = state;
        Ok(call)
    })();
    // The state buffer is smaller than the decay rows need or equals the state range: either way
    // the call is refused before any launch.
    launch(over).expect_err("a decay range over the state");
    for bad in [0.0f32, -1e-5, 2.0, f32::NAN] {
        let call = channel_call(&scan, shape, &decay, full, bad);
        launch(call).expect_err("an output norm eps outside (0, 1]");
    }
    Ok(())
}
