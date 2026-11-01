//! Token-health arm (`FIFI_FLINT_TOKEN_HEALTH`): unit proofs for the ordered
//! token-end validation, parked-frame behaviour, and the refusal of
//! attention-owned deferred frames at standalone finish. The real-weights
//! armed traversal lives in `real.rs` next to the other full-49 proofs.

use super::*;
use crate::kernels::cuda::device_present;

fn gate_error() -> EngError {
    EngError::ModelOutput("Flint CUDA FFN: late numeric/index health failed; output invalid".into())
}

fn deferred_refusal() -> EngError {
    EngError::BadRequest(
        "Flint CUDA FFN: deferred output requires its attention health observation".into(),
    )
}

fn ordered(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ring: &TokenHealth,
) -> Result<Option<EngError>> {
    let health = Health::for_tests(ctx).expect("test health ring");
    first_ring_fault(ctx, stream, ring, Some(&health)).map(|fault| fault.map(|(error, _)| error))
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn token_health_ring_word_bounds_are_refused() {
    if !device_present("token health ring bounds") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let ring = TokenHealth::new(&ctx).unwrap();
    assert!(ring.word(TOKEN_HEALTH_WORDS).is_err());
    assert!(ring.word(TOKEN_HEALTH_WORDS - 1).is_ok());
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn token_health_gate_passes_a_cleared_ring_and_flags_any_injected_word() {
    if !device_present("token health gate") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let ring = TokenHealth::new(&ctx).unwrap();
    ring.clear(&stream).unwrap();
    assert_eq!(ordered(&ctx, &stream, &ring).unwrap(), None);

    // A fault word in the FIRST layer's FFN slot must surface even though the
    // ring also holds the zero words of every later, clean site.
    ring.inject(&stream, 0, 4).unwrap();
    assert_eq!(
        ordered(&ctx, &stream, &ring).unwrap(),
        Some(gate_error()),
        "first-layer FFN word must surface"
    );

    // Any word: a fault ONLY in the last layer's residual slot surfaces with
    // the same refusal, never a weaker or different class.
    ring.clear(&stream).unwrap();
    ring.inject(&stream, TOKEN_HEALTH_WORDS - 1, 1).unwrap();
    assert_eq!(
        ordered(&ctx, &stream, &ring).unwrap(),
        Some(gate_error()),
        "last-layer residual word must surface"
    );

    // The gate never clears: a repeat read after the fault still refuses, and
    // the token-start clear is what restores a clean ring.
    assert_eq!(ordered(&ctx, &stream, &ring).unwrap(), Some(gate_error()));
    ring.clear(&stream).unwrap();
    assert_eq!(ordered(&ctx, &stream, &ring).unwrap(), None);
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn token_health_reports_every_parked_word_and_a_real_kernel_fault() {
    if !device_present("token health every word") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let ring = TokenHealth::new(&ctx).unwrap();
    let shape = RowShape {
        rows: 1,
        columns: 4,
    };
    // Every ring word, dirtied through a real parked frame's flag range —
    // the word the parked sites target — must surface at the gate.
    for word in 0..TOKEN_HEALTH_WORDS {
        ring.clear(&stream).unwrap();
        let mut parked = Frame::parked(&ctx, &stream, ring.word(word).unwrap()).unwrap();
        let _ = parked.allocate(shape).unwrap();
        parked.test_set_health(4).unwrap();
        drop(parked);
        assert_eq!(
            ordered(&ctx, &stream, &ring).unwrap(),
            Some(gate_error()),
            "parked word {word} must surface"
        );
    }
    assert_eq!(ordered(&ctx, &stream, &ring).unwrap(), Some(gate_error()));

    // A REAL device kernel must flag a parked word: a NaN f32 row through
    // the parked frame's own publish op drives the actual atomicOr path,
    // not a host-side test write.
    ring.clear(&stream).unwrap();
    let nan_row: Vec<u8> = [f32::NAN; 4]
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let raw = ctx.upload(&nan_row).unwrap();
    let mut parked = Frame::parked(&ctx, &stream, ring.word(0).unwrap()).unwrap();
    let input = parked
        .input(raw.range(0, nan_row.len()).unwrap(), shape)
        .unwrap();
    let output = parked.publish(input, shape).unwrap();
    drop(parked);
    let _ = output;
    assert_eq!(
        ordered(&ctx, &stream, &ring).unwrap(),
        Some(gate_error()),
        "the real publish kernel must flag the NaN row into the parked word"
    );
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn token_health_gate_reports_faults_in_execution_order() {
    if !device_present("token health fault order") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let ring = TokenHealth::new(&ctx).unwrap();
    let health = Health::for_tests(&ctx).expect("test health ring");

    // Both rings start clean; every leg re-establishes that before injecting,
    // so each assertion is about exactly one injected fault.
    let clean = |ring: &TokenHealth, health: &Health| {
        ring.clear(&stream).unwrap();
        health.clear(&stream).unwrap();
        assert_eq!(
            first_ring_fault(&ctx, &stream, ring, Some(health))
                .unwrap()
                .map(|(error, _)| error),
            None,
            "gate must be clean before each leg"
        );
    };

    // An FFN fault parked at layer 0 executes before a producer fault parked
    // in attention group 5, so the FFN refusal wins.
    clean(&ring, &health);
    ring.inject(&stream, 0, 4).unwrap();
    health.test_inject(&stream, 5, 7, 1).unwrap();
    assert_eq!(
        first_ring_fault(&ctx, &stream, &ring, Some(&health))
            .unwrap()
            .map(|(error, _)| error),
        Some(gate_error()),
        "layer-0 FFN word executes before group 5's producer fault"
    );

    // Reversed: an attention producer fault at group 0 executes before a
    // residual fault parked at layer 48, so the producer refusal wins.
    clean(&ring, &health);
    health.test_inject(&stream, 0, 7, 1).unwrap();
    ring.inject(&stream, TOKEN_HEALTH_WORDS - 2, 4).unwrap();
    let reported = first_ring_fault(&ctx, &stream, &ring, Some(&health))
        .unwrap()
        .map(|(error, _)| error);
    assert!(
        matches!(
            &reported,
            Some(EngError::Backend {
                operation: "flint_qk_producer",
                ..
            })
        ),
        "group-0 producer fault must be reported first, got {reported:?}"
    );

    // The last layer's group is still validated: a producer fault parked in
    // the final attention group must surface on its own, from rings proven
    // clean immediately before the injection.
    clean(&ring, &health);
    health.test_inject(&stream, LAYERS - 1, 7, 1).unwrap();
    let reported = first_ring_fault(&ctx, &stream, &ring, Some(&health))
        .unwrap()
        .map(|(error, _)| error);
    assert!(
        matches!(
            &reported,
            Some(EngError::Backend {
                operation: "flint_qk_producer",
                ..
            })
        ),
        "the last layer's producer fault must surface, got {reported:?}"
    );
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn token_health_frame_finish_parks_the_word_without_a_status_read() {
    if !device_present("token health frame parking") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let ring = TokenHealth::new(&ctx).unwrap();
    let shape = RowShape {
        rows: 1,
        columns: 4,
    };

    let baseline =
        crate::family::flint::cuda_ffn::device::COMPLETE_READS.with(std::cell::Cell::get);
    let mut frame = Frame::new(&ctx, &stream).unwrap();
    let buffer = frame.allocate(shape).unwrap();
    frame.finish(buffer).unwrap();
    let owned_reads =
        crate::family::flint::cuda_ffn::device::COMPLETE_READS.with(std::cell::Cell::get);
    assert!(owned_reads > baseline, "owned finish must observe health");

    let before = crate::family::flint::cuda_ffn::device::COMPLETE_READS.with(std::cell::Cell::get);
    let mut parked = Frame::parked(&ctx, &stream, ring.word(0).unwrap()).unwrap();
    let buffer = parked.allocate(shape).unwrap();
    parked.finish(buffer).unwrap();
    let parked_reads =
        crate::family::flint::cuda_ffn::device::COMPLETE_READS.with(std::cell::Cell::get);
    assert_eq!(
        parked_reads, before,
        "parked finish must leave the word for the gate, no status read"
    );
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn attention_deferred_frame_finish_keeps_its_refusal() {
    if !device_present("attention deferred refusal") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let ring = TokenHealth::new(&ctx).unwrap();
    let shape = RowShape {
        rows: 1,
        columns: 4,
    };

    // Attention-owned deferred frames still owe their group observation:
    // standalone finish must keep refusing them (health.rs contract), and a
    // token-health word must NOT be accepted as a substitute for that path.
    let mut attention = Frame::deferred(&ctx, &stream, ring.word(0).unwrap()).unwrap();
    let output = attention.allocate(shape).unwrap();
    match attention.finish(output) {
        Err(error) => assert_eq!(error, deferred_refusal()),
        Ok(_) => panic!("attention deferred finish must keep its refusal"),
    }
}

/// Child leg of the env-dispatch matrix: prints the OnceLock-parsed arming of
/// this process, then stops.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn token_health_env_probe() {
    if std::env::var_os("TOKEN_HEALTH_ENV_PROBE").is_none() {
        return;
    }
    let armed = crate::family::flint::settings::token_health_armed() as u8;
    println!("PROBE_TH={armed}");
}

/// Env-to-arming matrix over real processes (the OnceLock is read once, so
/// each leg needs its own): unset and `1`/`true`/`on` arm, `0` and a
/// misspelled or non-UTF-8 value turn it OFF (rollback polarity).
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn token_health_env_dispatch_matrix() {
    use std::process::Command;
    let leg = |value: Option<std::ffi::OsString>| -> u8 {
        let mut cmd = Command::new(std::env::current_exe().expect("exe"));
        cmd.arg("--exact")
            .arg("family::flint::cuda_ffn::device::runtime::tests::token_health::token_health_env_probe")
            .arg("--test-threads=1")
            .arg("--nocapture")
            .env("TOKEN_HEALTH_ENV_PROBE", "1");
        match value {
            Some(v) => {
                cmd.env("FIFI_FLINT_TOKEN_HEALTH", v);
            }
            None => {
                cmd.env_remove("FIFI_FLINT_TOKEN_HEALTH");
            }
        }
        let out = cmd.output().expect("child run");
        assert!(out.status.success(), "child failed: {}", out.status);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        text.lines()
            .find_map(|l| l.find("PROBE_TH=").map(|at| &l[at + "PROBE_TH=".len()..]))
            .expect("probe marker line")
            .parse()
            .expect("armed bit")
    };
    assert_eq!(leg(None), 1, "unset: armed (default on)");
    assert_eq!(leg(Some("1".into())), 1, "\"1\": armed");
    assert_eq!(leg(Some("true".into())), 1, "\"true\": armed");
    assert_eq!(leg(Some("on".into())), 1, "\"on\": armed");
    assert_eq!(leg(Some("0".into())), 0, "\"0\": OFF");
    assert_eq!(leg(Some("yes ".into())), 0, "misspelled: OFF");
    use std::os::unix::ffi::OsStringExt;
    let non_utf8 = std::ffi::OsString::from_vec(vec![0xff, 0xfe]);
    assert_eq!(leg(Some(non_utf8)), 0, "non-UTF-8: OFF");
}

/// The abort provenance rule, pinned without a device: an earlier ring fault
/// (ordinal below the failure threshold) replaces the original refusal; a
/// later or concurrent fault must NOT — the original keeps its message.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn abort_report_keeps_provenance_and_replaces_only_earlier_faults() {
    let earlier = EngError::ModelOutput("earlier parked fault".into());
    let original = EngError::ModelOutput("status set at or before the router".into());
    // Layer 3 failed (threshold 3*3+1 = 10): a word from layer 0 (ordinal 1)
    // precedes it and wins; a word of the failing layer itself (ordinal 10)
    // and a later group (ordinal 12) must keep the original message.
    assert_eq!(
        abort_report(original.clone(), Some((earlier.clone(), 1)), 10),
        earlier
    );
    assert_eq!(
        abort_report(original.clone(), Some((earlier.clone(), 10)), 10),
        original
    );
    assert_eq!(
        abort_report(original.clone(), Some((earlier, 12)), 10),
        original
    );
    assert_eq!(abort_report(original.clone(), None, 10), original);
    // Pre-loop failures (threshold 0) never replace: the ring was never
    // reached, so nothing can precede the refusal.
    let stale = EngError::ModelOutput("stale word".into());
    assert_eq!(
        abort_report(original.clone(), Some((stale, 0)), 0),
        original
    );
}
