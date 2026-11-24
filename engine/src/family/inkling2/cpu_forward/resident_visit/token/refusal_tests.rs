use super::*;

fn warm() -> CpuForward {
    let path = std::env::var_os("FIFI_INKLING_MODEL_DIR").unwrap();
    let mut f = CpuForward::open(std::path::Path::new(&path), 768).unwrap();
    f.enable_cuda(0).unwrap();
    f.token(1, false).unwrap();
    f
}

#[test]
#[ignore = "leased real GGUF; bracketed CUDA events for no-launch trace"]
fn refusals_before_work() {
    let mut f = warm();
    let saved = f.snapshot();
    let Weights::Gguf(w) = &f.weights else {
        panic!("expected GGUF")
    };
    let cuda = w.cuda().unwrap();
    let ctx = cuda.context();
    let probe = ctx.stream().unwrap();
    let events = [
        ctx.event().unwrap(),
        ctx.event().unwrap(),
        ctx.event().unwrap(),
    ];
    let before = ctx.buffer_requests().allocations;
    probe.record(&events[0]).unwrap();
    assert!(
        TokenWorkspace::open_limited(&f, cuda, &f.state, 0)
            .unwrap()
            .is_none(),
        "zero incremental budget"
    );
    probe.record(&events[1]).unwrap();
    let saved_width = f.spec.n_layers_per_attn_res_block;
    f.spec.n_layers_per_attn_res_block = 0;
    assert!(
        TokenWorkspace::open(&f, cuda, &f.state).unwrap().is_none(),
        "bad seal geometry"
    );
    f.spec.n_layers_per_attn_res_block = saved_width;
    probe.record(&events[2]).unwrap();
    probe.synchronize().unwrap();
    assert_eq!(ctx.buffer_requests().allocations, before);
    assert_eq!(f.state, saved.visits);
    assert_eq!(f.position(), saved.position);
    assert!(f.resident_token.is_none());
    eprintln!(
        "refusals: zero workspace allocations; no KV or position change; last three event records bracket memory/geometry declines"
    );
}

#[test]
#[ignore = "leased real GGUF"]
fn partial_enqueue_invalidates_work_without_host_retry() {
    let mut f = warm();
    let saved = f.snapshot();
    assert!(f.admit_resident_token().unwrap());
    let ws = f.resident_token.as_mut().unwrap();
    let before = ws.stream.enqueued();
    // Force the second enqueue to see the cursor just consumed by the first.
    // Test mutates only its private admitted plan, not kernel arguments or pointers.
    ws.slots[1].w.plan.visit.state_slot = ws.slots[0].w.plan.visit.state_slot;
    let err = f.commit_resident_token(2).unwrap_err();
    assert!(
        matches!(&err,EngError::Checkpoint(s) if s.contains("cursor")),
        "{err:?}"
    );
    assert!(f.resident_token.as_ref().unwrap().stream.enqueued() > before);
    assert!(f.failed && f.state.is_empty());
    assert_eq!(f.position(), saved.position);
    assert!(matches!(f.token(2,true),Err(EngError::Checkpoint(s)) if s.contains("invalidated")));
    f.restore(&saved);
    assert!(!f.failed);
    assert_eq!(f.state, saved.visits);
    assert!(!f.admit_resident_token().unwrap());
    eprintln!(
        "partial enqueue: first slot queued; lost second cursor fenced and invalidated sequence; no host retry"
    );
}
