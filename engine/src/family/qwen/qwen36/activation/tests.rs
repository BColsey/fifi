use super::*;
use crate::EngError;

pub(super) fn settings() -> ExecutionSettings {
    ExecutionSettings {
        weight_format: "Exl3".into(),
        kv_dtype: "Q8".into(),
        context_tokens: 4096,
        prefill_chunk_tokens: 64,
        prefill_path: "chunk".into(),
        prefill_pieces_sha256: prefill_fingerprint(0, &[2]),
        sync_each: false,
        section_marks: false,
        device_shortlist: true,
        prefix_reused: false,
    }
}

pub(super) fn request(prompt: &[u32]) -> CaptureRequest {
    CaptureRequest::new(
        Fingerprints::new("a".repeat(64), "b".repeat(64), "c".repeat(64)).unwrap(),
        prompt,
        4,
        settings().fingerprint().unwrap(),
    )
    .unwrap()
}

#[test]
fn aligns_only_with_the_pending_last_prompt_token() {
    let req = request(&[7, 9, 11]);
    let good = PromptBoundary {
        history: &[7, 9, 11],
        prompt_len: 3,
        pending: Some(11),
        fed: 2,
    };
    assert!(req.check_pending(good).is_ok());
    for bad in [
        PromptBoundary { fed: 3, ..good },
        PromptBoundary {
            pending: Some(9),
            ..good
        },
        PromptBoundary {
            history: &[7, 8, 11],
            ..good
        },
        PromptBoundary {
            history: &[7, 9, 11, 12],
            ..good
        },
        PromptBoundary {
            prompt_len: 2,
            ..good
        },
        PromptBoundary {
            pending: None,
            ..good
        },
    ] {
        assert!(req.check_pending(bad).is_err());
    }
    // Chunking and prefix import must converge on this same fed/history boundary.
    for pieces in [vec![2], vec![1, 1], vec![0, 2]] {
        let fed = pieces.iter().sum();
        assert!(req.check_pending(PromptBoundary { fed, ..good }).is_ok());
    }
}

#[test]
fn one_prompt_token_is_pending_at_position_zero() {
    let req = request(&[17]);
    assert!(
        req.check_pending(PromptBoundary {
            history: &[17],
            prompt_len: 1,
            pending: Some(17),
            fed: 0,
        })
        .is_ok()
    );
}

#[test]
fn copies_raw_post_block_values_before_arena_reuse_or_normalization() {
    let mut arena = vec![1.0, -2.0, 3.5, -0.0];
    let mut capture = Capture::new(request(&[7]), settings()).unwrap();
    let mut reads = 0;
    for block in 0..20 {
        capture
            .observe(block, 0, || {
                reads += 1;
                Ok(arena.clone())
            })
            .unwrap();
    }
    assert_eq!(reads, 1);
    // The next mixer, arena reset/import and final normalization cannot change the owned row.
    arena.fill(99.0);
    let (_, record) = capture.finish(Ok(())).unwrap();
    assert_eq!(record.values(), &[1.0, -2.0, 3.5, -0.0]);
    assert!(record.values()[3].is_sign_negative());
}

#[test]
fn partial_failure_cancellation_or_missing_boundary_cannot_publish() {
    let mut capture = Capture::new(request(&[7]), settings()).unwrap();
    capture.observe(19, 0, || Ok(vec![1.0; 4])).unwrap();
    assert!(
        capture
            .finish::<()>(Err(EngError::BadRequest("later block failed".into())))
            .is_err()
    );
    let cancelled = Capture::new(request(&[7]), settings()).unwrap();
    assert!(cancelled.finish(Ok(())).is_err());
    let mut failed_copy = Capture::new(request(&[7]), settings()).unwrap();
    assert!(
        failed_copy
            .observe(19, 0, || Err(EngError::BadRequest("copy failed".into())))
            .is_err()
    );
    assert!(failed_copy.finish(Ok(())).is_err());
}

#[test]
fn rejects_wrong_position_duplicate_shape_and_nonfinite_values() {
    let mut capture = Capture::new(request(&[7, 8]), settings()).unwrap();
    assert!(
        capture
            .observe(19, 0, || panic!("must not read the wrong token"))
            .is_err()
    );
    assert!(capture.finish(Ok(())).is_err());
    let mut capture = Capture::new(request(&[7]), settings()).unwrap();
    capture.observe(19, 0, || Ok(vec![1.0; 4])).unwrap();
    assert!(
        capture
            .observe(19, 0, || panic!("must not read twice"))
            .is_err()
    );
    assert!(capture.finish(Ok(())).is_err());
    for row in [vec![1.0; 3], vec![f32::NAN; 4], vec![f32::INFINITY; 4]] {
        let mut capture = Capture::new(request(&[7]), settings()).unwrap();
        assert!(capture.observe(19, 0, || Ok(row)).is_err());
        assert!(capture.finish(Ok(())).is_err());
    }
}

#[test]
fn bounds_requests_and_rejects_bad_fingerprints() {
    assert!(Fingerprints::new("a".repeat(63), "b".repeat(64), "c".repeat(64)).is_err());
    assert!(Fingerprints::new("A".repeat(64), "b".repeat(64), "c".repeat(64)).is_err());
    let ids = Fingerprints::new("a".repeat(64), "b".repeat(64), "c".repeat(64)).unwrap();
    for (prompt, hidden) in [
        (vec![], 4),
        (vec![7], 0),
        (vec![7], MAX_HIDDEN + 1),
        (vec![7; MAX_PROMPT_TOKENS + 1], 4),
    ] {
        assert!(
            CaptureRequest::new(
                ids.clone(),
                &prompt,
                hidden,
                settings().fingerprint().unwrap()
            )
            .is_err()
        );
    }
    assert!(CaptureRequest::new(ids, &[7], 4, "bad".into()).is_err());
    assert_ne!(
        prefill_fingerprint(0, &[2]),
        prefill_fingerprint(0, &[1, 1])
    );
    assert_ne!(prefill_fingerprint(0, &[2]), prefill_fingerprint(1, &[1]));
    let mut different = settings();
    different.sync_each = true;
    assert!(Capture::new(request(&[7]), different).is_err());
}

#[test]
fn deterministic_manifest_has_explicit_provenance_and_bounded_artifact() {
    let mut capture = Capture::new(request(&[7, 9]), settings()).unwrap();
    capture
        .observe(19, 1, || Ok(vec![1.0, 2.0, -0.0, 4.0]))
        .unwrap();
    let (_, record) = capture.finish(Ok(())).unwrap();
    let a = record.to_json().unwrap();
    assert_eq!(a, record.to_json().unwrap());
    assert_eq!(record.fingerprint().unwrap(), sha256(&a));
    assert!(a.len() <= MAX_ARTIFACT_BYTES);
    let json: serde_json::Value = serde_json::from_slice(&a).unwrap();
    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["layer"], 19);
    assert_eq!(json["position"], 1);
    assert_eq!(json["request"]["token_id"], 9);
    assert_eq!(json["representation"], "raw_post_block_residual");
    assert_eq!(json["dtype"], "f32");
    assert_eq!(json["execution_path"], "qwen36_cuda_eager");
    assert_eq!(
        json["request"]["fingerprints"]["model_sha256"],
        "a".repeat(64)
    );
    assert_eq!(json["request"]["prompt_sha256"], prompt_sha256(&[7, 9]));
}

#[test]
fn maximum_width_is_bounded_and_f32_values_roundtrip_exactly() {
    let mut req = request(&[7]);
    req.hidden = MAX_HIDDEN;
    let probes = [
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        -0.0,
        1.000_000_1,
    ];
    let row: Vec<f32> = probes.into_iter().cycle().take(MAX_HIDDEN).collect();
    let mut capture = Capture::new(req, settings()).unwrap();
    capture.observe(19, 0, || Ok(row.clone())).unwrap();
    let (_, record) = capture.finish(Ok(())).unwrap();
    let bytes = record.to_json().unwrap();
    assert!(bytes.len() <= MAX_ARTIFACT_BYTES);
    let decoded: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    for (original, decoded) in row.iter().zip(decoded["values"].as_array().unwrap()) {
        assert_eq!(
            original.to_bits(),
            (decoded.as_f64().unwrap() as f32).to_bits()
        );
    }
}
