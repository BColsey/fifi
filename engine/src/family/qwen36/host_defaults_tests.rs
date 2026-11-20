//! The host-derived MLX defaults, through the settings resolver.

use super::host_plan::HostFacts;
use super::settings::{Backend, Q36Env, Q36Options, Q36Settings};
use crate::EngError;
use std::path::Path;

const GIB: u64 = 1 << 30;
/// The shard bytes of the 4-bit Qwen3.6 35B-A3B pack: 19.0 GiB.
const MODEL_BYTES: u64 = 20_401_094_656;

fn env(memory_gib: u64) -> Q36Env {
    Q36Env {
        host: Some(HostFacts {
            memory_bytes: memory_gib * GIB,
            model_bytes: MODEL_BYTES,
        }),
        ..Q36Env::default()
    }
}

fn resolve(options: Option<&Q36Options>, env: &Q36Env) -> Result<Q36Settings, EngError> {
    Q36Settings::new(Path::new("/a"), Path::new("/a"), options, env, Backend::Mlx)
}

fn refusal(result: Result<Q36Settings, EngError>) -> String {
    match result {
        Err(EngError::BadRequest(text)) => text,
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn a_36_gib_host_with_no_members_reproduces_the_shipped_profile() {
    let got = resolve(None, &env(36)).expect("serves with nothing set");
    assert_eq!(got.cache_limit, 4_294_967_296);
    assert_eq!(got.prefill_budget, Some(32_212_254_720));
    assert_eq!(got.prefill_reserve, 8_286_699_520);
    assert_eq!(got.context_limit, 262_144);
    assert_eq!(got.session_cap, 4 << 30);
    assert!(got.compile);
    assert_eq!(
        got.derived,
        ["cache_limit_bytes", "prefill_budget_bytes", "compile"]
    );
    let line = got.host_note.expect("one startup line");
    assert!(
        line.starts_with(
            "q36: host defaults: unified memory 36.0 GiB, model 19.0 GiB, OS floor 6.0 GiB; "
        ),
        "{line}"
    );
    assert!(
        line.contains("cache_limit_bytes=4294967296 (derived: memory / 9"),
        "{line}"
    );
    assert!(
        line.contains("prefill_budget_bytes=32212254720 (derived"),
        "{line}"
    );
    assert!(
        line.contains("prefill_session_reserve_bytes=8286699520 (derived"),
        "{line}"
    );
    assert!(line.contains("context_limit=262144 (default)"), "{line}");
    assert!(line.contains("compile=true (derived"), "{line}");
    assert!(!line.contains('\n'));
}

#[test]
fn larger_hosts_scale_the_cache_limit_and_the_budget() {
    let sixty_four = resolve(None, &env(64)).expect("64 GiB serves");
    assert_eq!(sixty_four.cache_limit, 7 * GIB);
    let floor = (64 * GIB / 6).div_ceil(1 << 20) << 20;
    assert_eq!(sixty_four.prefill_budget, Some(64 * GIB - floor));
    assert_eq!(
        sixty_four.prefill_reserve,
        10_880 * 262_144 + 65_863_680 + 7 * GIB + GIB
    );
    let big = resolve(None, &env(128)).expect("128 GiB serves");
    assert_eq!(big.cache_limit, 8 * GIB, "the cache limit stops at 8 GiB");
    assert!(big.prefill_budget.is_some_and(|budget| budget > 100 * GIB));
}

#[test]
fn a_host_that_cannot_hold_the_model_and_the_reserve_is_refused_with_the_shortfall() {
    let text = refusal(resolve(None, &env(16)));
    assert!(
        text.starts_with("engine config: this host has 16.0 GiB of unified memory"),
        "{text}"
    );
    assert!(
        text.contains("12.0 GiB after the 4.0 GiB OS floor"),
        "{text}"
    );
    assert!(text.contains("the model needs 19.0 GiB"), "{text}");
    assert!(text.contains("short by 11.7 GiB"), "{text}");
    assert!(text.contains("prefill_budget_bytes yourself"), "{text}");
    // 24 GiB: 20 GiB after the floor against 19.0 + 5.7 needed.
    let text = refusal(resolve(None, &env(24)));
    assert!(text.contains("short by 4.7 GiB"), "{text}");
}

#[test]
fn explicit_members_stay_exact_overrides() {
    let options = Q36Options {
        cache_limit_bytes: Some(1024),
        prefill_budget_bytes: Some(30 * GIB),
        prefill_session_reserve_bytes: Some(7),
        compile: Some(false),
        context_limit: Some(4096),
        ..Q36Options::default()
    };
    // Even on a host too small for the derived plan: nothing was derived to check.
    let got = resolve(Some(&options), &env(16)).expect("explicit values are taken as given");
    assert_eq!(got.cache_limit, 1024);
    assert_eq!(got.prefill_budget, Some(30 * GIB));
    assert_eq!(got.prefill_reserve, 7);
    assert!(!got.compile);
    assert_eq!(got.context_limit, 4096);
    assert!(got.derived.is_empty());
    assert!(got.host_note.is_none(), "nothing derived, no line");
}

#[test]
fn a_member_keeps_its_value_while_the_rest_derive() {
    let options = Q36Options {
        cache_limit_bytes: Some(2 * GIB),
        ..Q36Options::default()
    };
    let got = resolve(Some(&options), &env(36)).expect("serves");
    assert_eq!(got.cache_limit, 2 * GIB);
    assert_eq!(got.prefill_budget, Some(32_212_254_720));
    assert_eq!(got.prefill_reserve, 10_880 * 262_144 + 65_863_680 + 3 * GIB);
    assert_eq!(got.derived, ["prefill_budget_bytes", "compile"]);
    let line = got.host_note.expect("line");
    assert!(
        line.contains("cache_limit_bytes=2147483648 (set)"),
        "{line}"
    );
}

#[test]
fn the_cache_limit_variable_beats_the_host_rule() {
    let mut with_variable = env(36);
    with_variable.cache_limit = Some("2147483648".into());
    let got = resolve(None, &with_variable).expect("serves");
    assert_eq!(got.cache_limit, 2 * GIB);
    assert!(!got.derived.contains(&"cache_limit_bytes"));
    let line = got.host_note.expect("line");
    assert!(
        line.contains("cache_limit_bytes=2147483648 (Q36_CACHE_LIMIT)"),
        "{line}"
    );
}

#[test]
fn a_reserve_member_alone_takes_the_derived_budget() {
    let options = Q36Options {
        prefill_session_reserve_bytes: Some(GIB),
        ..Q36Options::default()
    };
    let got = resolve(Some(&options), &env(36)).expect("serves");
    assert_eq!(got.prefill_reserve, GIB);
    assert_eq!(got.prefill_budget, Some(32_212_254_720));
}

#[test]
fn without_host_facts_the_cache_limit_stays_required() {
    assert_eq!(
        refusal(resolve(None, &Q36Env::default())),
        "engine config: cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set"
    );
}

#[test]
fn the_cuda_arm_derives_nothing() {
    let got = Q36Settings::new(
        Path::new("/a"),
        Path::new("/a"),
        None,
        &env(36),
        Backend::Cuda,
    )
    .expect("CUDA settings");
    assert!(got.derived.is_empty());
    assert!(!got.compile);
}

#[test]
fn the_dense_family_derives_the_budget_but_keeps_compile_off() {
    let options = Q36Options {
        context_limit: Some(32_768),
        ..Q36Options::default()
    };
    let got = Q36Settings::new_for_family(
        Path::new("/a"),
        Path::new("/a"),
        Some(&options),
        &env(128),
        Backend::Mlx,
        crate::family::spec::QWEN3_8_27B_DENSE,
    )
    .expect("dense settings");
    assert!(got.prefill_budget.is_some());
    assert!(!got.compile);
}
