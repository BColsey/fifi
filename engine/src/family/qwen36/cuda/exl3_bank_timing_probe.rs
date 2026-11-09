//! A banked decode timing probe over the real pack.
//!
//! `FIFI_EXL3_CHECKPOINT` names the pack directory; `FIFI_EXL3_SLOTS`, `FIFI_EXL3_STICKY`
//! and `FIFI_EXL3_TOKENS` override the slot count (69), the sticky pin count (0) and the
//! decode length (96) for policy measurement, and `FIFI_EXL3_TRACE` dumps every decode
//! resolve's picks for the offline admission-policy study. The test loads the banked model (the
//! admission-floor slot count, every expert in the host store), decodes a fixed prompt plus
//! `TOKENS` greedy tokens, and reports the wall time per token beside the shares the test-only
//! probe accumulates: the readback drains (the blocking device-to-host waits of the pick
//! round trips) and the host resolves (the slot mapping and the miss refills). The numbers
//! say whether the follow-up lever is the drain (device-side id mapping) or the refills.
//! The probe changes no production behavior; its only assertion is that the decode finished.

use std::{cell::RefCell, path::Path, time::Instant};

use super::{
    common_tests::{dims, inventory},
    exl3_bank::Exl3SlotResolver,
    exl3_load::{BankedModel, load_exl3_banked_dir},
    exl3_moe::Exl3MoeWeights,
    exl3_step_device_tests::{EOS, HEAD_ROWS, PROMPT, SAMPLER_VOCAB, options},
    layer_table::LayerTable,
    native_device_rig::device_present,
    resources::{StepResources, moe_dims},
    rules::QwenRules,
    step::{CaptureConfig, QwenResidentStep},
};
use crate::family::project::cuda::{CudaProjector, Reserve};

/// The trace path: when set, every decode-phase resolve's picks are dumped (one
/// `layer pick ...` line per call) for the offline admission-policy study.
fn trace_path() -> Option<std::path::PathBuf> {
    std::env::var_os("FIFI_EXL3_TRACE").map(std::path::PathBuf::from)
}

/// Tokens decoded after the prompt: enough for the banks to reach their steady state.
const TOKENS: usize = 96;

#[test]
#[ignore = "requires the pinned EXL3 pack in FIFI_EXL3_CHECKPOINT and a GPU"]
fn banked_decode_timing_probe_reports_the_round_trip_shares() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    assert!(device_present("banked decode probe"), "a GPU is required");
    let projector = CudaProjector::new().expect("projector");
    projector
        .context()
        .set_single_row_default(crate::kernels::cuda::resident::exl3::SingleRow::Wide(1));

    // The serve admission on this card by default; FIFI_EXL3_SLOTS sweeps the curve.
    let slots: usize = std::env::var("FIFI_EXL3_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(69);
    let banked = load_exl3_banked_dir(&projector, Path::new(&dir), slots, Reserve::default())
        .expect("the banked pack");
    let context = projector.context();
    let mut step_options = options();
    let tokens: usize = std::env::var("FIFI_EXL3_TOKENS")
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("FIFI_EXL3_TOKENS is not a number: {v:?}"))
        })
        .unwrap_or(TOKENS);
    step_options.context_tokens = PROMPT.len() + tokens + 1;
    let rules = QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS).expect("rules");
    let resources =
        StepResources::new(context, dims(), &inventory(), &step_options).expect("resources");
    let BankedModel {
        weights,
        store,
        slots,
        experts,
        routed_bytes: _,
        dense_bytes: _,
    } = banked;
    let table = LayerTable::new(&dims(), &inventory()).expect("table");
    let moe = moe_dims(&dims());
    let blocks = (0..table.len())
        .map(|layer| Exl3MoeWeights::resolve_banked(&weights, &table, layer, &moe, slots))
        .collect::<Result<Vec<_>, _>>()
        .expect("banked blocks");
    let sticky: usize = std::env::var("FIFI_EXL3_STICKY")
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("FIFI_EXL3_STICKY is not a number: {v:?}"))
        })
        .unwrap_or(0);
    let bank =
        RefCell::new(Exl3SlotResolver::new(store, &blocks, experts, sticky).expect("resolver"));
    let mut step = QwenResidentStep::new_banked(
        context,
        &resources,
        &weights,
        &step_options,
        rules,
        slots,
        &bank,
    )
    .expect("the banked step");
    step.set_capture_config(CaptureConfig {
        capture: false,
        section_marks: false,
    })
    .expect("capture off");

    // The prompt decodes one token per row (run_token with logits), as the serve path does.
    let began = Instant::now();
    let mut id = 0u32;
    for &token in &PROMPT {
        id = step.run_token(token, true).expect("prompt token");
    }
    let prompt_wall = began.elapsed();
    let ((prompt_drains, prompt_drain_ns), (prompt_resolves, prompt_resolve_ns)) =
        crate::family::qwen36::cuda::exl3_bank::probe::take();
    let ms_prompt = |ns: u64, count: u64| ns as f64 / 1e6 / count.max(1) as f64;
    println!(
        "banked decode probe: prompt {} rows {:.0} ms ({} drains {:.2} ms each, {} resolves \
         {:.2} ms each)",
        PROMPT.len(),
        prompt_wall.as_secs_f64() * 1e3,
        prompt_drains,
        ms_prompt(prompt_drain_ns, prompt_drains),
        prompt_resolves,
        ms_prompt(prompt_resolve_ns, prompt_resolves),
    );
    let began = Instant::now();
    if trace_path().is_some() {
        crate::family::qwen36::cuda::exl3_bank::trace::arm();
    }
    for _ in 0..tokens {
        id = step.run_token(id, true).expect("decoded token");
    }
    let decode_wall = began.elapsed();
    let ((readback_calls, readback_ns), (_resolve_calls, resolve_ns)) =
        crate::family::qwen36::cuda::exl3_bank::probe::take();
    let (hits, misses) = bank.borrow().stats();
    println!(
        "banked decode probe: decode {tokens} tokens {:.1} ms/token total, drains {:.1} \
         ms/token across {} calls, resolves {:.1} ms/token; hits {hits} misses {misses}; \
         last id {id}",
        decode_wall.as_secs_f64() * 1e3 / tokens as f64,
        readback_ns as f64 / 1e6 / tokens as f64,
        readback_calls,
        resolve_ns as f64 / 1e6 / tokens as f64,
    );
    if let Some(path) = &trace_path() {
        let lines =
            crate::family::qwen36::cuda::exl3_bank::trace::dump(path).expect("the trace dump");
        println!(
            "banked decode probe: trace {lines} resolve calls to {}",
            path.display()
        );
    }
    assert!((id as usize) < SAMPLER_VOCAB, "the decode produced a token");
}
