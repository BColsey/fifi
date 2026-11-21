//! The host-derived MLX defaults, through the settings resolver.

use super::host_plan::{ExpertFacts, HostFacts};
use super::settings::{Backend, DiskExperts, Q36Env, Q36Options, Q36Settings};
use crate::EngError;
use std::path::Path;

const GIB: u64 = 1 << 30;
/// The shard bytes of the 4-bit Qwen3.6 35B-A3B pack: 19.0 GiB.
const MODEL_BYTES: u64 = 20_401_094_656;

fn env(memory_gib: u64) -> Q36Env {
    Q36Env {
        host: Some(HostFacts {
            memory_bytes: memory_gib * GIB,
            metal_bytes: None,
            ceiling_bytes: None,
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

/// The shard bytes of a 6-bit dense 27B pack: 21.7 GiB.
const DENSE_6BIT_BYTES: u64 = 23_300_000_000;

fn dense_env(memory_gib: u64, model_bytes: u64) -> Q36Env {
    Q36Env {
        host: Some(HostFacts {
            memory_bytes: memory_gib * GIB,
            metal_bytes: None,
            ceiling_bytes: None,
            model_bytes,
        }),
        ..Q36Env::default()
    }
}

fn dense(options: Option<&Q36Options>, env: &Q36Env) -> Result<Q36Settings, EngError> {
    Q36Settings::new_for_family(
        Path::new("/a"),
        Path::new("/a"),
        options,
        env,
        Backend::Mlx,
        crate::family::spec::QWEN3_8_27B_DENSE,
    )
}

#[test]
fn a_context_that_does_not_fit_the_default_shrinks_to_the_largest_that_fits() {
    let host = dense_env(36, DENSE_6BIT_BYTES);
    let got = dense(None, &host).expect("a smaller context fits");
    let budget = got.prefill_budget.expect("derived budget");
    assert!(got.context_limit < 262_144);
    assert_eq!(got.context_limit % 1024, 0);
    assert!(
        DENSE_6BIT_BYTES + got.prefill_reserve <= budget,
        "the chosen context fits"
    );
    // One step more does not fit.
    let per_position =
        super::settings::cache_growth_per_position(&got.spec, got.kv, got.activation).unwrap();
    assert!(
        DENSE_6BIT_BYTES + got.prefill_reserve + 1024 * per_position > budget,
        "the next step does not fit"
    );
    assert!(got.derived.contains(&"context_limit"));
    let line = got.host_note.expect("one startup line");
    assert!(
        line.contains(&format!(
            "context_limit={} (derived: largest that fits the budget)",
            got.context_limit
        )),
        "{line}"
    );
    assert!(!line.contains('\n'));
    // The session cap follows the chosen context, not the default one.
    assert_eq!(
        got.session_cap,
        per_position * u64::from(got.context_limit)
            + super::settings::delta_state_bytes(&got.spec).unwrap()
    );
}

#[test]
fn a_default_that_fits_is_not_changed() {
    let got = dense(None, &dense_env(128, DENSE_6BIT_BYTES)).expect("serves");
    assert_eq!(got.context_limit, 262_144);
    let line = got.host_note.expect("line");
    assert!(line.contains("context_limit=262144 (default)"), "{line}");
}

#[test]
fn no_context_above_the_floor_is_refused_with_the_shortfall() {
    // 30 GiB budget against 29.5 GiB of weights: the fixed part of the reserve alone is over.
    let text = refusal(dense(None, &dense_env(36, 31_700_000_000)));
    assert!(text.contains("the model needs 29.5 GiB"), "{text}");
    assert!(text.contains("context 262144"), "{text}");
    assert!(text.contains("short by"), "{text}");
}

#[test]
fn a_context_below_the_floor_is_refused() {
    // Room for fewer than 4,096 positions: the model leaves only the fixed part and 2,000 positions.
    let spec = crate::family::spec::QWEN3_8_27B_DENSE;
    let per_position = super::settings::cache_growth_per_position(
        &spec,
        super::settings::KvDtype::Q8,
        super::settings::Activation::Bf16,
    )
    .unwrap();
    let fixed = super::settings::delta_state_bytes(&spec).unwrap() + 4 * GIB + GIB;
    let budget = 30 * GIB;
    let model = budget - fixed - 2_000 * per_position;
    let text = refusal(dense(None, &dense_env(36, model)));
    assert!(text.contains("short by"), "{text}");
    assert!(text.contains("context 262144"), "{text}");
}

#[test]
fn an_explicit_context_that_does_not_fit_keeps_the_refusal() {
    let options = Q36Options {
        context_limit: Some(262_144),
        ..Q36Options::default()
    };
    let text = refusal(dense(Some(&options), &dense_env(36, DENSE_6BIT_BYTES)));
    assert!(text.contains("context 262144"), "{text}");
    assert!(text.contains("short by"), "{text}");
}

#[test]
fn the_moe_family_shares_the_rule() {
    let got = resolve(None, &{
        let mut tight = env(36);
        tight.host.as_mut().unwrap().model_bytes = 23 * GIB;
        tight
    })
    .expect("a smaller context fits");
    assert!(got.context_limit > 150_000 && got.context_limit < 262_144);
    assert!(got.derived.contains(&"context_limit"));
}

/// The routed experts of the 4-bit Qwen3.6 35B-A3B pack: 16.9 GiB, with 0.42 GiB in the largest
/// layer.
const EXPERTS: ExpertFacts = ExpertFacts {
    total_bytes: 18_119_393_280,
    stage_bytes: 452_984_832,
};

fn env_with_experts(memory_gib: u64, disk_experts: DiskExperts) -> Q36Env {
    Q36Env {
        experts: Some(EXPERTS),
        disk_experts,
        ..env(memory_gib)
    }
}

/// What each host class chooses for a start that sets nothing. The 16 GiB and 24 GiB hosts
/// cannot hold the 19.0 GiB model next to its session, so they stream the routed experts from
/// disk at the full context; a 36 GiB host and larger keep everything resident.
#[test]
fn a_host_that_cannot_hold_the_model_streams_its_experts_by_default() {
    for (memory_gib, staged) in [(16, true), (24, true), (36, false), (64, false)] {
        let got = resolve(None, &env_with_experts(memory_gib, DiskExperts::Auto))
            .unwrap_or_else(|error| panic!("{memory_gib} GiB: {error:?}"));
        assert_eq!(
            got.mlx_staged_cap.is_some(),
            staged,
            "{memory_gib} GiB host"
        );
        assert_eq!(got.context_limit, 262_144, "{memory_gib} GiB host");
        let budget = got.prefill_budget.unwrap();
        println!(
            "{memory_gib} GiB host: staged={staged} context={} budget={budget} reserve={}",
            got.context_limit, got.prefill_reserve
        );
        if let Some(cap) = got.mlx_staged_cap {
            assert_eq!(cap, budget);
        }
    }
}

/// `0` keeps the experts resident, so a host that cannot hold them is refused as before, and
/// a model whose headers say nothing about experts keeps its refusal.
#[test]
fn the_override_zero_and_unreadable_experts_keep_the_refusal() {
    let off = resolve(None, &env_with_experts(16, DiskExperts::Off));
    assert!(refusal(off).contains("the model needs 19.0 GiB"));
    let unknown = Q36Env {
        experts: None,
        ..env_with_experts(16, DiskExperts::Auto)
    };
    assert!(refusal(resolve(None, &unknown)).contains("the model needs 19.0 GiB"));
    // `1` behaves like automatic.
    let on = resolve(None, &env_with_experts(16, DiskExperts::On)).unwrap();
    assert!(on.mlx_staged_cap.is_some());
}

/// A host too small even for the staged model is still refused, with the resident numbers.
#[test]
fn a_host_too_small_for_the_staged_model_is_refused() {
    let tiny = resolve(None, &env_with_experts(4, DiskExperts::Auto));
    assert!(refusal(tiny).contains("short by"));
}

fn env_on(memory_gib: u64, metal_gib: Option<u64>, ceiling_mib: Option<u64>) -> Q36Env {
    let mut env = env_with_experts(memory_gib, DiskExperts::Auto);
    let host = env.host.as_mut().unwrap();
    host.metal_bytes = metal_gib.map(|gib| gib * GIB);
    host.ceiling_bytes = ceiling_mib.map(|mib| mib << 20);
    env
}

/// The plan and the load read one budget: the number the settings plan the prefill against is the
/// number `DeviceBudget` enforces when the start loads, on a 16, 24, 36 and 64 GiB host whose
/// Metal recommendation is below the host budget, and the start the plan accepts fits that
/// number at the context it serves.
#[test]
fn the_plan_and_the_load_budget_agree_on_every_host_class() {
    use crate::family::mlx_budget::DeviceBudget;
    for (memory, metal) in [(16, 11), (24, 18), (36, 27), (64, 48)] {
        let env = env_on(memory, Some(metal), None);
        let host = env.host.unwrap();
        let got = resolve(None, &env).unwrap_or_else(|error| panic!("{memory} GiB: {error:?}"));
        let load = DeviceBudget::from_memory(host.memory()).unwrap().bytes();
        assert_eq!(got.prefill_budget, Some(load), "{memory} GiB host");
        assert!(
            load <= metal * GIB,
            "{memory} GiB: the load never passes Metal"
        );
        let resident = if got.mlx_staged_cap.is_some() {
            host.staged(EXPERTS).model_bytes
        } else {
            host.model_bytes
        };
        assert!(
            resident + got.prefill_reserve <= load,
            "{memory} GiB: {resident} + {} > {load}",
            got.prefill_reserve
        );
        println!(
            "{memory} GiB host, Metal {metal} GiB: budget {load} streams={} context={}",
            got.mlx_staged_cap.is_some(),
            got.context_limit
        );
    }
}

/// `FIFI_MLX_WORKING_SET_MIB` lowers the budget the settings plan from, so a start that the load
/// would refuse is not planned as if it had the host budget, and it never raises it.
#[test]
fn the_working_set_setting_bounds_the_plan_as_it_bounds_the_load() {
    let got = resolve(None, &env_on(36, Some(27), Some(12288))).unwrap();
    assert_eq!(got.prefill_budget, Some(12 * GIB));
    assert!(
        got.mlx_staged_cap.is_some(),
        "19 GiB of weights stream under 12 GiB"
    );
    let higher = resolve(None, &env_on(36, Some(27), Some(65536))).unwrap();
    assert_eq!(higher.prefill_budget, Some(27 * GIB));
}
