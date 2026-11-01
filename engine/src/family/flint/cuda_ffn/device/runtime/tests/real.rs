use super::*;
use crate::family::flint::exl3_resident::{FlintResidentLimits, load_flint_resident};
use crate::family::project::cuda::Reserve;
use std::path::PathBuf;

fn token(position: usize) -> u32 {
    (630 + 31 * position) as u32
}

/// The resident model with room for every plan's KV and scratch, from the opt-in pack.
pub(super) fn load_real_model(
    ctx: &CudaContext,
    stream: &DeviceStream,
    plans: &[&KvPlan],
) -> FlintExl3Resident {
    let pack = PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("opt-in EXL3 pack"));
    let kv_bytes: usize = plans
        .iter()
        .map(|plan| plan.bytes(4 * 128 * 2 * 2).unwrap())
        .sum();
    let hot_bytes: u64 = plans
        .iter()
        .map(|plan| State::scratch_bytes(plan).unwrap())
        .sum();
    load_flint_resident(
        ctx,
        stream,
        &pack,
        FlintResidentLimits {
            max_staging_bytes: 2 << 30,
            max_weight_bytes: ctx.memory_info().unwrap().free as u64,
            reserve: Reserve {
                kv_cache: kv_bytes as u64,
                work: (512_u64 << 20).checked_add(hot_bytes).unwrap(),
                headroom: 512 << 20,
            },
        },
    )
    .unwrap()
}

fn acceptance_bad_request_is_preserved(
    decoder: &Decoder<'_>,
    state: &mut State,
    stream: &DeviceStream,
) {
    let error = EngError::BadRequest("test acceptance refusal".into());
    let result: Result<()> = decoder.step(state, token(0), |_, _| Err(error.clone()));
    assert_eq!(result, Err(error));
    assert_eq!(state.fed(), 0);
    assert!(state.poisoned);
    decoder.ctx.ensure_usable().unwrap();
    state.reset(stream).unwrap();
}

fn failed_accept_requires_reset(decoder: &Decoder<'_>, state: &mut State, stream: &DeviceStream) {
    assert!(matches!(
        decoder.step(state, VOCAB as u32, |_, _| -> Result<()> {
            panic!("invalid token must refuse before acceptance")
        }),
        Err(EngError::BadRequest(_))
    ));
    assert_eq!(state.fed(), 0);
    assert!(!state.poisoned);
    assert!(
        state
            .kv
            .iter()
            .all(|cache| cache.cursor.committed_prefix() == 0)
    );
    acceptance_bad_request_is_preserved(decoder, state, stream);
    let pristine: Vec<Vec<u32>> = (0..4)
        .map(|position| {
            decoder
                .step(state, token(position), |_, logits| {
                    Ok(logits.iter().map(|value| value.to_bits()).collect())
                })
                .unwrap()
        })
        .collect();
    state.reset(stream).unwrap();
    for position in 0..3 {
        decoder.step(state, token(position), |_, _| Ok(())).unwrap();
    }
    let fed = state.fed();
    let failure = EngError::ModelOutput("injected post-KV acceptance failure".into());
    let mut accepted = false;
    let result: Result<()> = decoder.step(state, token(3), |_, _| {
        accepted = true;
        Err(failure.clone())
    });
    assert!(
        accepted,
        "failure must occur after all 49 layers and the head"
    );
    assert_eq!(result, Err(failure));
    assert_eq!(state.fed(), fed);
    assert!(state.poisoned);
    assert_eq!(state.kv.len(), 49);
    for cache in &mut state.kv {
        assert_eq!(cache.cursor.committed_prefix(), fed);
        assert!(matches!(
            cache.cursor.begin_append(fed, 1),
            Err(EngError::Backend {
                operation: "flint_kv_append",
                ..
            })
        ));
    }
    assert!(matches!(
        decoder.step(state, token(3), |_, _| -> Result<()> {
            panic!("poisoned decoder must refuse before acceptance")
        }),
        Err(EngError::BadRequest(_))
    ));
    assert_eq!(state.fed(), fed);
    decoder.ctx.ensure_usable().unwrap();
    state.reset(stream).unwrap();
    assert_eq!(state.fed(), 0);
    for (position, expected) in pristine.iter().enumerate() {
        decoder
            .step(state, token(position), |_, logits| {
                assert!(
                    logits
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(expected.iter().copied()),
                    "reset replay differs at position {position}"
                );
                Ok(())
            })
            .unwrap();
    }
    println!(
        "real full49 post-KV acceptance failure: fed unchanged, all cursors poisoned, reset replay bitwise"
    );
}

fn armed_traversal_matches_bitwise_and_defers_only_single_row(
    decoder: &Decoder<'_>,
    armed: &mut State,
    plain: &mut State,
) {
    // Single-row steps: armed steps must be bitwise-identical to unarmed
    // steps. The arm removes the per-frame status reads (the ~101+49
    // drains), NOT all completes: exactly three Frame::complete reads remain
    // per armed step (embedding publish, final-norm frame, head projection
    // frame), so the armed count is 3 per step against the unarmed per-frame
    // baseline.
    assert!(
        armed.token_health.is_some(),
        "the armed baseline must be armed"
    );
    assert!(
        plain.token_health.is_none(),
        "the comparison baseline must be explicitly unarmed, not env-derived"
    );
    assert!(
        armed.health.is_some(),
        "this proof requires FIFI_FLINT_LATE_HEALTH=1 (the GPU runner pins it)"
    );
    let mut expected = Vec::new();
    for position in 0..4 {
        let before =
            crate::family::flint::cuda_ffn::device::COMPLETE_READS.with(std::cell::Cell::get);
        let logits: Vec<u32> = decoder
            .step(plain, token(position), |_, logits| {
                Ok(logits.iter().map(|value| value.to_bits()).collect())
            })
            .unwrap();
        let plain_drains = crate::family::flint::cuda_ffn::device::COMPLETE_READS
            .with(std::cell::Cell::get)
            - before;
        assert!(
            plain_drains > 0,
            "unarmed step must observe per-frame health"
        );
        expected.push(logits);
    }
    for (position, expected) in expected.iter().enumerate() {
        let before =
            crate::family::flint::cuda_ffn::device::COMPLETE_READS.with(std::cell::Cell::get);
        decoder
            .step(armed, token(position), |_, logits| {
                assert_eq!(logits.len(), expected.len());
                let mismatch = logits
                    .iter()
                    .zip(expected)
                    .position(|(value, expected)| value.to_bits() != *expected);
                assert!(
                    mismatch.is_none(),
                    "armed decode differs at position {position}, first differing logit {mismatch:?}"
                );
                Ok(())
            })
            .unwrap();
        let armed_drains = crate::family::flint::cuda_ffn::device::COMPLETE_READS
            .with(std::cell::Cell::get)
            - before;
        assert_eq!(
            armed_drains, 3,
            "armed decode parks its health words; only the three retained completes read"
        );
    }

    // Prefill (an 8-row chunk): multi-row calls never arm, so armed and
    // unarmed prefill must drain equally and produce identical logits.
    let chunk: Vec<u32> = (0..8).map(token).collect();
    let before = crate::family::flint::cuda_ffn::device::COMPLETE_READS.with(std::cell::Cell::get);
    let plain_logits = decoder.feed_rows_logits(plain, &chunk).unwrap();
    let plain_drains =
        crate::family::flint::cuda_ffn::device::COMPLETE_READS.with(std::cell::Cell::get) - before;
    let before = crate::family::flint::cuda_ffn::device::COMPLETE_READS.with(std::cell::Cell::get);
    let armed_logits = decoder.feed_rows_logits(armed, &chunk).unwrap();
    let armed_drains =
        crate::family::flint::cuda_ffn::device::COMPLETE_READS.with(std::cell::Cell::get) - before;
    assert_eq!(
        plain_drains, armed_drains,
        "prefill drains must be unchanged"
    );
    assert!(
        plain_drains > 0,
        "prefill must still observe per-frame health"
    );
    let mismatch = armed_logits
        .iter()
        .zip(&plain_logits)
        .position(|(value, expected)| value.to_bits() != expected.to_bits());
    assert!(
        mismatch.is_none(),
        "armed prefill differs at logit {mismatch:?}"
    );

    println!(
        "real full49 token-health: armed single-row bitwise with 3 retained completes, prefill unchanged"
    );
}

fn armed_fault_surfaces_at_the_gate_before_head_and_commits(
    decoder: &Decoder<'_>,
    state: &mut State,
    stream: &DeviceStream,
    pristine: &[Vec<u32>],
) {
    // An attention producer fault, parked in the health ring at the same
    // word the device producers target (group 0, Slot::Producer = word 7),
    // must surface at the token-end gate: BEFORE the head runs (the accept
    // closure must not be reached) and BEFORE the KV appends commit (cursor
    // and fed are unchanged).
    state.test_inject_attention_health(stream, 0, 7, 1).unwrap();
    let fed = state.fed();
    let committed: Vec<_> = state
        .kv
        .iter()
        .map(|cache| cache.cursor.committed_prefix())
        .collect();
    let refusal = decoder.step(state, token(0), |_, _| -> Result<()> {
        panic!("gate must refuse before the head samples")
    });
    assert!(
        matches!(
            &refusal,
            Err(EngError::Backend {
                operation: "flint_qk_producer",
                ..
            })
        ),
        "gate must report the earliest ring fault, got {refusal:?}"
    );
    assert_eq!(state.fed(), fed, "a refused token must not commit");
    assert_eq!(
        state
            .kv
            .iter()
            .map(|cache| cache.cursor.committed_prefix())
            .collect::<Vec<_>>(),
        committed,
        "a refused token must not commit KV appends"
    );

    // The ring fault is sticky across tokens (attention-health faults stay
    // sticky until reset); reset clears both rings and the replay stays
    // bitwise. reset's token-ring clear is proven DIRECTLY — the
    // ring is dirty here, read back, reset, and asserted zero BEFORE any
    // decode (the next decode would clear the ring itself, so only this
    // read makes the reset path non-vacuous; deleting reset's
    // `token.clear` would leave the observed words nonzero and fail here).
    assert!(state.poisoned, "gate refusal poisons the session");
    state.test_dirty_token_word(stream, 5, 4).unwrap();
    assert!(!state.test_token_ring_is_zero(stream).unwrap());
    state.reset(stream).unwrap();
    assert!(
        state.test_token_ring_is_zero(stream).unwrap(),
        "reset must clear the token ring before any decode runs"
    );
    for (position, expected) in pristine.iter().enumerate() {
        decoder
            .step(state, token(position), |_, logits| {
                assert!(
                    logits
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(expected.iter().copied()),
                    "post-reset armed replay differs at position {position}"
                );
                Ok(())
            })
            .unwrap();
    }
    println!("real full49 token-health fault: refused pre-head, no commits, reset replay bitwise");

    // A STALE token-ring word must not poison a healthy decode: reset (which
    // clears both rings) leaves the word dirty on purpose, and the very next
    // token-start clear wipes it — every step stays bitwise-identical. If the
    // token-start clear were removed, the gate would refuse this replay.
    state.reset(stream).unwrap();
    state.test_dirty_token_word(stream, 5, 4).unwrap();
    for (position, expected) in pristine.iter().enumerate() {
        decoder
            .step(state, token(position), |_, logits| {
                assert!(
                    logits
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(expected.iter().copied()),
                    "stale-word armed replay differs at position {position}"
                );
                Ok(())
            })
            .unwrap();
    }
    println!("real full49 stale token word: cleared at token start, replay bitwise");

    // A dirty parked FFN word must be refused by the token-end gate BEFORE
    // the head samples (the accept closure must not run) with the gate's own
    // refusal class — through the public step path. The word is dirtied
    // AFTER the token-start clear, standing in for a producer that faults
    // during this token's body; the stale-word leg above covers the
    // pre-clear case, which must NOT refuse. Word 0 is dense layer 0's FFN
    // word (mlp_only_layers [0, 1]): no router copies it, so nothing trips
    // before the token-end gate reads it.
    state.reset(stream).unwrap();
    crate::family::flint::cuda_ffn::device::route_trace::inject::set_post_clear_words(vec![(0, 4)]);
    let refusal = decoder.step(state, token(0), |_, _| -> Result<()> {
        panic!("the gate must refuse the dirty word before the head samples")
    });
    crate::family::flint::cuda_ffn::device::route_trace::inject::set_post_clear_words(Vec::new());
    assert!(
        matches!(&refusal, Err(EngError::ModelOutput(message)) if
            message.contains("late numeric/index health failed")),
        "dirty FFN word must be refused at the gate, got {refusal:?}"
    );
    assert!(state.poisoned, "the gate refusal poisons the session");
    state.reset(stream).unwrap();
    for (position, expected) in pristine.iter().enumerate() {
        decoder
            .step(state, token(position), |_, logits| {
                assert!(
                    logits
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(expected.iter().copied()),
                    "final armed replay differs at position {position}"
                );
                Ok(())
            })
            .unwrap();
    }
    println!("real full49 dirty FFN word: gate refusal pre-head, reset replay bitwise");
}

fn abort_consult_provenance_and_progress_windows(
    decoder: &Decoder<'_>,
    state: &mut State,
    stream: &DeviceStream,
    pristine: &[Vec<u32>],
) {
    use crate::family::flint::cuda_ffn::device::route_trace::inject;
    // Dense layers (mlp_only_layers [0, 1]) have no router: their parked FFN
    // words are read only by the token gate. MoE layers copy the parked
    // root-frame flag into the ids buffer, so a dirty FFN word on a MoE
    // layer trips the PRODUCTION router-status refusal mid-layer.
    const K_MOE: usize = 7; // ordinal of word 2*K_MOE is 3*7+1 = 22
    const J_DENSE: usize = 0; // ordinal of word 2*J_DENSE is 1; no router reads it
    const K_DENSE: usize = 1; // ordinal of word 2*K_DENSE is 4

    let step_expecting = |state: &mut State, matches: &dyn Fn(&EngError) -> bool| {
        state.reset(stream).unwrap();
        let refusal = decoder
            .step(state, token(0), |_, _| -> Result<()> {
                panic!("an injected abort must surface before the head")
            })
            .unwrap_err();
        assert!(state.poisoned, "the abort must poison the session");
        assert!(
            matches(&refusal),
            "wrong refusal for this provenance window: {refusal:?}"
        );
    };

    // Leg (a) — production router refusal, original KEPT: ONLY word
    // 2*K_MOE dirty. Layer 7's real router-status check refuses mid-layer
    // with its ModelOutput class; the dirty word's ordinal (22) equals the
    // failure point (22), so the consult must keep the original. (If the
    // loop-top progress set were 3*index+2, the threshold would be 23 and
    // the ring fault would wrongly replace the original.)
    inject::set_post_clear_words(vec![(2 * K_MOE, 4)]);
    step_expecting(state, &|error| {
        matches!(error, EngError::ModelOutput(message) if
            message.contains("status set at or before the router"))
    });
    assert_eq!(
        inject::take_aborts_fired(),
        0,
        "the real router refusal needs no injected abort"
    );
    inject::set_post_clear_words(Vec::new());

    // Leg (b) — earlier parked fault REPLACES: word 2*J_DENSE (a dense
    // layer's FFN word, ordinal 1, which no router reads) plus word
    // 2*K_MOE dirty. Layer 7's real router refusal (threshold 22) is
    // replaced by the strictly-earlier ring fault with the gate's class.
    inject::set_post_clear_words(vec![(2 * J_DENSE, 4), (2 * K_MOE, 4)]);
    step_expecting(state, &|error| {
        matches!(error, EngError::ModelOutput(message) if
            message.contains("late numeric/index health failed"))
    });
    assert_eq!(
        inject::take_aborts_fired(),
        0,
        "the replacement is the gate's own fault, not an injected abort"
    );
    inject::set_post_clear_words(Vec::new());

    // Leg (c) — post-FFN window: abort just after DENSE layer 1's FFN
    // completed (no router reads word 2). The FFN word's ordinal (4) is
    // strictly below the threshold (3*1+2 = 5), so it must replace the
    // injected abort; with the post-FFN advance deleted the threshold
    // would stay 4 and the original would survive.
    inject::set_after_ffn_abort(Some(K_DENSE));
    inject::set_post_clear_words(vec![(2 * K_DENSE, 4)]);
    step_expecting(state, &|error| {
        matches!(error, EngError::ModelOutput(message) if
            message.contains("late numeric/index health failed"))
    });
    assert_eq!(
        inject::take_aborts_fired(),
        1,
        "the post-FFN abort must actually be consumed"
    );
    inject::set_after_ffn_abort(None);
    inject::set_post_clear_words(Vec::new());

    // Leg (d) — post-residual window: abort just after MoE layer 7's
    // residual completed, with the residual word 2*K_MOE+1 dirty (ordinal
    // 23; only the gate reads residual words). The threshold (3*7+3 = 24)
    // is strictly above it, so it must replace; with the post-residual
    // advance deleted the threshold would stay 23 and the original would
    // survive.
    inject::set_after_residual_abort(Some(K_MOE));
    inject::set_post_clear_words(vec![(2 * K_MOE + 1, 4)]);
    step_expecting(state, &|error| {
        matches!(error, EngError::ModelOutput(message) if
            message.contains("late numeric/index health failed"))
    });
    assert_eq!(
        inject::take_aborts_fired(),
        1,
        "the post-residual abort must actually be consumed"
    );
    inject::set_after_residual_abort(None);
    inject::set_post_clear_words(Vec::new());

    state.reset(stream).unwrap();
    for (position, expected) in pristine.iter().enumerate() {
        decoder
            .step(state, token(position), |_, logits| {
                assert!(
                    logits
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(expected.iter().copied()),
                    "post-abort-provenance replay differs at position {position}"
                );
                Ok(())
            })
            .unwrap();
    }
    println!(
        "real full49 abort consult: router original kept, earlier word replaces, both progress windows advance"
    );
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 weights; token-health armed traversal, prefill identity, fault surfacing, abort provenance"]
fn real_weights_token_health_arm_bitwise_single_row_and_gate_faults() {
    let pack = PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("opt-in EXL3 pack"));
    let ctx = CudaContext::new().expect("real full49 proof requires CUDA");
    let stream = ctx.stream().unwrap();
    let plan = KvPlan::new(1027, 1, false).unwrap();
    let kv_bytes = plan.bytes(4 * 128 * 2 * 2).unwrap();
    let hot_bytes = State::scratch_bytes(&plan).unwrap() * 2;
    let model = load_flint_resident(
        &ctx,
        &stream,
        &pack,
        FlintResidentLimits {
            max_staging_bytes: 2 << 30,
            max_weight_bytes: ctx.memory_info().unwrap().free as u64,
            reserve: Reserve {
                kv_cache: kv_bytes as u64,
                work: (512_u64 << 20).checked_add(hot_bytes).unwrap(),
                headroom: 512 << 20,
            },
        },
    )
    .unwrap();
    let decoder = Decoder::new(&ctx, &stream, &model).unwrap();
    let mut plain = State::new(&ctx, plan).unwrap();
    // The comparison baseline is explicitly unarmed regardless of the
    // process env, so the drain comparison cannot compare two armed walks.
    plain.token_health = None;
    let mut armed = State::new(&ctx, plan).unwrap();
    armed.test_arm_token_health(&ctx).unwrap();
    armed_traversal_matches_bitwise_and_defers_only_single_row(&decoder, &mut armed, &mut plain);

    armed.reset(&stream).unwrap();
    let pristine: Vec<Vec<u32>> = (0..4)
        .map(|position| {
            decoder
                .step(&mut armed, token(position), |_, logits| {
                    Ok(logits.iter().map(|value| value.to_bits()).collect())
                })
                .unwrap()
        })
        .collect();
    armed_fault_surfaces_at_the_gate_before_head_and_commits(
        &decoder, &mut armed, &stream, &pristine,
    );
    abort_consult_provenance_and_progress_windows(&decoder, &mut armed, &stream, &pristine);
    stream.synchronize().unwrap();
}

fn ring_matches_linear(
    decoder: &Decoder<'_>,
    linear: &mut State,
    ring: &mut State,
    stream: &DeviceStream,
) {
    // Ring wrap boundary for the 131-context plan is cache row 65 on the
    // half-capacity layers, so positions 64/65 and 129/130 straddle the first
    // and second wraps (the second wrap lands on the final slot); the walk
    // must cross both to prove wrap == linear.
    // A post-KV acceptance fault lands on the ring six rows PAST the wrap to
    // prove fault propagation and reset replay across reused slots.
    let mut linear_log: Vec<Vec<u32>> = Vec::new();
    for position in 0..131 {
        if position == 71 {
            ring_fault_across_wrap(decoder, ring, stream, &linear_log);
        }
        let expected: Vec<u32> = decoder
            .step(linear, token(position), |_, logits| {
                Ok(logits.iter().map(|value| value.to_bits()).collect())
            })
            .unwrap();
        linear_log.push(expected.clone());
        decoder
            .step(ring, token(position), |_, logits| {
                assert_eq!(logits.len(), expected.len());
                let mismatch = logits
                    .iter()
                    .zip(&expected)
                    .position(|(value, expected)| value.to_bits() != *expected);
                assert!(
                    mismatch.is_none(),
                    "ring/linear position {position}, first differing logit {mismatch:?}"
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(linear.fed(), position + 1);
        assert_eq!(ring.fed(), position + 1);
        if [64, 65, 129, 130].contains(&position) {
            println!("real full49 ring65/linear bitwise at absolute position {position}");
        }
    }
    assert!(matches!(
        decoder.step(ring, token(131), |_, _| Ok(())),
        Err(EngError::ContextLimit(_))
    ));
    assert_eq!(ring.fed(), 131);
}

fn ring_fault_across_wrap(
    decoder: &Decoder<'_>,
    ring: &mut State,
    stream: &DeviceStream,
    linear_log: &[Vec<u32>],
) {
    assert_eq!(ring.fed(), 71, "fault lands six rows past the 65-row wrap");
    let failure = EngError::ModelOutput("injected post-KV acceptance failure".into());
    let mut accepted = false;
    let result: Result<()> = decoder.step(ring, token(71), |_, _| {
        accepted = true;
        Err(failure.clone())
    });
    assert!(
        accepted,
        "failure must occur after all 49 layers and the head"
    );
    assert_eq!(result, Err(failure));
    assert_eq!(ring.fed(), 71);
    assert!(ring.poisoned);
    for cache in &mut ring.kv {
        assert_eq!(cache.cursor.committed_prefix(), 71);
        assert!(matches!(
            cache.cursor.begin_append(71, 1),
            Err(EngError::Backend {
                operation: "flint_kv_append",
                ..
            })
        ));
    }
    assert!(matches!(
        decoder.step(ring, token(71), |_, _| -> Result<()> {
            panic!("poisoned decoder must refuse before acceptance")
        }),
        Err(EngError::BadRequest(_))
    ));
    decoder.ctx.ensure_usable().unwrap();
    ring.reset(stream).unwrap();
    assert_eq!(ring.fed(), 0);
    for (position, expected) in linear_log.iter().enumerate() {
        decoder
            .step(ring, token(position), |_, logits| {
                assert!(
                    logits
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(expected.iter().copied()),
                    "ring reset replay differs at position {position}"
                );
                Ok(())
            })
            .unwrap();
    }
    println!(
        "real full49 ring65 fault across the wrap: poisoned, reset, replayed {}/71 rows bitwise",
        linear_log.len()
    );
}

/// Decoder-level armed-guard proof: a fault raised by a layer (injected via
/// the cfg-test seam at the exact point a kernel flag would land) rejects
/// with the shared refusal text BEFORE the head samples, poisons the state,
/// leaves the cursor untouched, refuses the next token until reset, and
/// reset does NOT clear the device word — the next step's own per-token
/// clear is what zeroes it, so the replay passes.
#[test]
#[ignore = "requires admitted GPU and full EXL3 weights; kv_device_f16 armed decoder gate"]
fn real_weights_kv_guard_fault_rejects_finish_and_survives_reset() {
    let ctx = CudaContext::new().expect("armed decoder gate requires CUDA");
    let stream = ctx.stream().unwrap();
    let plan = KvPlan::new(1027, 1, false).unwrap();
    let model = load_real_model(&ctx, &stream, &[&plan]);
    let decoder = Decoder::new(&ctx, &stream, &model).unwrap();
    let mut state = State::new(&ctx, plan).unwrap();
    // Inject the guard buffer the armed path would own (the ON env arms the
    // whole process; the test cannot flip the OnceLock, so it takes the
    // same field directly — child-module reach).
    state.kv_guard = Some(ctx.upload(&0u32.to_ne_bytes()).unwrap());
    let step = |state: &mut State, position: usize, fired: &mut bool| -> Result<()> {
        decoder.step(state, token(position), |_, _| {
            *fired = true;
            Ok(())
        })
    };
    let mut accepted = false;
    for position in 0..3 {
        step(&mut state, position, &mut accepted).unwrap();
    }
    assert!(accepted, "clean armed steps must reach the head");
    TEST_KV_FAULT.store(true, std::sync::atomic::Ordering::Relaxed);
    accepted = false;
    let before = state.kv[0].cursor.committed_prefix();
    let rejected = step(&mut state, 3, &mut accepted);
    TEST_KV_FAULT.store(false, std::sync::atomic::Ordering::Relaxed);
    assert!(
        matches!(&rejected, Err(EngError::ModelOutput(message)) if message == KV_F16_REFUSAL),
        "gate must reject with the exact shared text, got {rejected:?}"
    );
    assert!(!accepted, "rejection must prevent the head/finish path");
    assert!(state.poisoned, "rejection poisons the session");
    assert_eq!(
        state.kv[0].cursor.committed_prefix(),
        before,
        "rejected token must not advance the cursor"
    );
    accepted = false;
    assert!(step(&mut state, 4, &mut accepted).is_err());
    assert!(!accepted);
    state.reset(&stream).unwrap();
    let mut word = [0u8; 4];
    state
        .kv_guard
        .as_ref()
        .unwrap()
        .read_at(&stream, 0, &mut word)
        .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        u32::from_ne_bytes(word),
        1,
        "reset must not clear the guard word"
    );
    // Replay from scratch passes: the step path's own per-token clear zeroed
    // the word before the layers ran (this kills a clear-removed mutant).
    accepted = false;
    for position in 0..4 {
        step(&mut state, position, &mut accepted).unwrap();
    }
    assert!(accepted, "replay after reset must reach the head");
    let mut word = [0u8; 4];
    state
        .kv_guard
        .as_ref()
        .unwrap()
        .read_at(&stream, 0, &mut word)
        .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        u32::from_ne_bytes(word),
        0,
        "clean steps leave the word clear"
    );
}

fn step_bits(decoder: &Decoder<'_>, state: &mut State, at: usize) -> Vec<u32> {
    decoder
        .step(state, token(at), |_, logits| {
            Ok(logits.iter().map(|value| value.to_bits()).collect())
        })
        .unwrap()
}

/// The walk of the kv_device_f16 byte compare: 8-row chunks to position 512, four
/// single steps, one chunk across the end of the 520-row sliding ring (two runs per
/// plane), four more steps. Returns the logits of each step.
fn kv_walk(decoder: &Decoder<'_>, state: &mut State) -> Vec<Vec<u32>> {
    let tokens: Vec<u32> = (0..528).map(token).collect();
    for chunk in tokens[..512].chunks(8) {
        decoder.feed_rows(state, chunk).unwrap();
    }
    let mut logits: Vec<_> = (512..516).map(|at| step_bits(decoder, state, at)).collect();
    decoder.feed_rows(state, &tokens[516..524]).unwrap();
    logits.extend((524..528).map(|at| step_bits(decoder, state, at)));
    assert_eq!(state.fed(), 528);
    logits
}

/// Walk-level proof of the kv_device_f16 arm: the same walk through a host-guard
/// state and a device-guard state leaves every K and V plane of all 49 layers
/// byte-identical, poison cells included, and every step's logits bit-identical.
/// The 2,048-B and 16,384-B source-row reads must be present on the host walk and
/// absent on the device walk, so the compare cannot pass with the arm unarmed.
#[test]
#[ignore = "requires admitted GPU and full EXL3 weights; kv_device_f16 walk-level KV byte compare"]
fn real_weights_kv_device_f16_walk_leaves_kv_planes_byte_identical() {
    let ctx = CudaContext::new().expect("walk-level KV compare requires CUDA");
    let stream = ctx.stream().unwrap();
    let plan = KvPlan::new(1027, 8, true).unwrap();
    let model = load_real_model(&ctx, &stream, &[&plan, &plan]);
    let decoder = Decoder::new(&ctx, &stream, &model).unwrap();
    // The guard word is what the step path keys on; the process-wide OnceLock cannot
    // be flipped here, so each state takes the path under test explicitly.
    let mut host = State::new(&ctx, plan).unwrap();
    host.kv_guard = None;
    let mut device = State::new(&ctx, plan).unwrap();
    device.kv_guard = Some(ctx.upload(&0u32.to_ne_bytes()).unwrap());
    for cache in host.kv.iter().chain(&device.kv) {
        for plane in [&cache.keys, &cache.values] {
            plane
                .write_at(&stream, 0, &vec![0x55; plane.bytes()])
                .unwrap();
        }
    }
    let reads = |bytes: usize| {
        ctx.buffer_requests()
            .ranged_d2h
            .get(&bytes)
            .copied()
            .unwrap_or(0)
    };
    let start = (reads(2048), reads(8 * 2048));
    let host_logits = kv_walk(&decoder, &mut host);
    let middle = (reads(2048), reads(8 * 2048));
    let device_logits = kv_walk(&decoder, &mut device);
    let end = (reads(2048), reads(8 * 2048));
    // K and V in each of 49 layers per compute: 8 steps and 65 eight-row chunks.
    assert_eq!(
        (middle.0 - start.0, middle.1 - start.1),
        (2 * 49 * 8, 2 * 49 * 65),
        "host walk must read the K/V source rows"
    );
    assert_eq!(
        (end.0 - middle.0, end.1 - middle.1),
        (0, 0),
        "device walk must not read the K/V source rows"
    );
    for (index, (host, device)) in host_logits.iter().zip(&device_logits).enumerate() {
        assert!(host == device, "step {index} logits differ ON vs OFF");
    }
    let read = |plane: &DeviceBuffer| {
        let mut bytes = vec![0u8; plane.bytes()];
        plane.read_at(&stream, 0, &mut bytes).unwrap();
        bytes
    };
    for (layer, (host, device)) in host.kv.iter().zip(&device.kv).enumerate() {
        for (name, host, device) in [
            ("K", &host.keys, &device.keys),
            ("V", &host.values, &device.values),
        ] {
            let (host, device) = (read(host), read(device));
            assert!(
                host.iter().any(|&byte| byte != 0x55),
                "layer {layer} {name}: the walk wrote nothing"
            );
            let mismatch = host.iter().zip(&device).position(|(a, b)| a != b);
            assert!(
                mismatch.is_none(),
                "layer {layer} {name} plane differs ON vs OFF at byte {mismatch:?}"
            );
        }
    }
    assert_eq!(
        read(device.kv_guard.as_ref().unwrap()),
        0u32.to_ne_bytes(),
        "a clean walk leaves the guard word clear"
    );
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 weights; wrap-boundary walk across both ring wraps, ~6 min"]
fn real_weights_failed_accept_reset_and_ring_boundaries() {
    let pack = PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("opt-in EXL3 pack"));
    let ctx = CudaContext::new().expect("real full49 proof requires CUDA");
    let stream = ctx.stream().unwrap();
    let linear_plan = KvPlan::new_test_window(131, 1, false, 65).unwrap();
    // Test-only 65-row ring window (production WINDOW is 513): non-full layers
    // wrap at row 65, so a 131-position walk crosses both wraps like the
    // production 1027-context walk crossed row 513.
    let ring_plan = KvPlan::new_test_window(131, 1, true, 65).unwrap();
    for layer in 0..49 {
        assert_eq!(
            ring_plan.cache_rows(layer).unwrap(),
            if FULL_LAYERS.contains(&(layer as u64)) {
                131
            } else {
                65
            }
        );
    }
    let kv_bytes =
        linear_plan.bytes(4 * 128 * 2 * 2).unwrap() + ring_plan.bytes(4 * 128 * 2 * 2).unwrap();
    let hot_bytes = State::scratch_bytes(&linear_plan)
        .unwrap()
        .checked_add(State::scratch_bytes(&ring_plan).unwrap())
        .unwrap();
    let model = load_flint_resident(
        &ctx,
        &stream,
        &pack,
        FlintResidentLimits {
            max_staging_bytes: 2 << 30,
            max_weight_bytes: ctx.memory_info().unwrap().free as u64,
            reserve: Reserve {
                kv_cache: kv_bytes as u64,
                work: (512_u64 << 20).checked_add(hot_bytes).unwrap(),
                headroom: 512 << 20,
            },
        },
    )
    .unwrap();
    let decoder = Decoder::new(&ctx, &stream, &model).unwrap();
    let mut linear = State::new(&ctx, linear_plan).unwrap();
    linear.test_arm_token_health(&ctx).unwrap();
    assert!(linear.token_health.is_some(), "linear walk must be armed");
    failed_accept_requires_reset(&decoder, &mut linear, &stream);
    linear.reset(&stream).unwrap();
    let mut ring = State::new(&ctx, ring_plan).unwrap();
    ring.test_arm_token_health(&ctx).unwrap();
    assert!(ring.token_health.is_some(), "ring walk must be armed");
    ring_matches_linear(&decoder, &mut linear, &mut ring, &stream);
    stream.synchronize().unwrap();
    println!(
        "real full49 proof complete: every one of 131 heads compared bitwise across both ring wraps, no CPU-FP64 replay"
    );
}
