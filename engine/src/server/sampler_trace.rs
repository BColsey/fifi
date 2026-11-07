//! The per-request sampler-trace ring, its wire packet, and the read route.
//!
//! Contract: capture is opt-in twice — the process setting `sample_trace`
//! permits it, and a request arms it with `x-sample-trace: true` before
//! generation begins; nothing is recorded retroactively. Buffers register in
//! the ring at submit with their reply's metadata, so in-progress captures
//! count against the global budget. The ring keeps the last
//! [`MAX_RETAINED_REPLIES`] replies under [`GLOBAL_POSITION_BUDGET`] positions
//! (drop-oldest; a flooded ring may drop a running request's trace, and the
//! route then reports not retained).
//!
//! The packet carries token ids, ranks, and probabilities only — never prompt
//! text, reasoning, or tool arguments. Authorization is the process API key,
//! exactly like every other `/v1` route: one key serves the whole process, so
//! per-user trace isolation is NOT provided — any key holder can read any
//! reply's trace by id. The position budgets bound memory by the serialized
//! worst case: a test-verified byte bound per position times the position
//! counts.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use axum::extract::{Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::error::ServerError;
use super::state::AppState;
use crate::family::sampler::{CaptureBuffer, CaptureHandle};

/// The per-request absolute position budget. 8192 positions at the serialized
/// worst case stay under 3.2 MiB.
pub(crate) const PER_REQUEST_POSITION_LIMIT: usize = 8_192;

/// How many replies the ring retains. The GUI's details view reads one reply's
/// trace right after its stream ends, so the window is small on purpose.
const MAX_RETAINED_REPLIES: usize = 4;

/// The global position budget across the ring, in-progress captures included:
/// [`MAX_RETAINED_REPLIES`] full requests.
const GLOBAL_POSITION_BUDGET: usize = MAX_RETAINED_REPLIES * PER_REQUEST_POSITION_LIMIT;

/// What the packet needs about a reply besides its positions. Registered at
/// submit, when the reply head and the resolved settings are at hand.
#[derive(Clone)]
pub(crate) struct TraceMeta {
    pub(crate) model: String,
    pub(crate) temperature: f64,
    pub(crate) top_p: f64,
    pub(crate) top_k: u32,
}

struct TraceEntry {
    id: String,
    meta: TraceMeta,
    buffer: CaptureHandle,
}

/// The drop-oldest ring of reply id to capture buffer and metadata.
#[derive(Default)]
pub(crate) struct SamplerTraceRing {
    entries: Mutex<VecDeque<TraceEntry>>,
}

impl SamplerTraceRing {
    /// Registers a capture buffer under its reply id, replacing a same-id
    /// registration, then evicts oldest entries until both budgets hold.
    pub(crate) fn register(&self, id: &str, meta: TraceMeta, buffer: CaptureHandle) {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries.retain(|entry| entry.id != id);
        entries.push_back(TraceEntry {
            id: id.to_owned(),
            meta,
            buffer,
        });
        while entries.len() > MAX_RETAINED_REPLIES
            || total_positions(&entries) > GLOBAL_POSITION_BUDGET
        {
            if entries.pop_front().is_none() {
                break;
            }
        }
    }

    /// The capture buffer and metadata of one reply id, when still retained.
    pub(crate) fn get(&self, id: &str) -> Option<(TraceMeta, CaptureHandle)> {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries
            .iter()
            .find(|entry| entry.id == id)
            .map(|entry| (entry.meta.clone(), Arc::clone(&entry.buffer)))
    }
}

fn total_positions(entries: &VecDeque<TraceEntry>) -> usize {
    entries
        .iter()
        .map(|entry| {
            entry
                .buffer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .positions
                .len()
        })
        .sum()
}

#[derive(Deserialize)]
pub(crate) struct TraceQuery {
    id: Option<String>,
}

/// The trace packet of one reply: version 1, ids and probabilities only.
/// Piece alignment (offset-based text mapping) is a later packet revision;
/// this version stays joinable on `token_id`.
fn packet(reply_id: &str, meta: &TraceMeta, buffer: &CaptureBuffer) -> Value {
    let positions: Vec<Value> = buffer
        .positions
        .iter()
        .map(|recorded| {
            json!({
                "token_id": recorded.token,
                "rank": recorded.rank,
                "kept": recorded.kept,
                "greedy": recorded.greedy,
                "selected_prob": recorded.selected_prob,
                "candidates": recorded
                    .candidates
                    .iter()
                    .map(|&(id, prob)| json!({ "token_id": id, "prob": prob }))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    json!({
        "packet": 1,
        "reply_id": reply_id,
        "model": meta.model,
        "sampling": {
            "temperature": meta.temperature,
            "top_p": meta.top_p,
            "top_k": meta.top_k,
        },
        "capture_scope": "decode",
        "truncated": buffer.truncated,
        "captured_positions": buffer.positions.len(),
        "positions": positions,
    })
}

/// `GET /v1/sampler-trace?id=<reply id>` — one reply's packet, no-store.
/// Unknown or evicted ids report not found, so a dropped trace is explicit.
pub(crate) async fn get_sampler_trace(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TraceQuery>,
) -> Result<Response, ServerError> {
    let Some(id) = query.id.filter(|id| !id.is_empty()) else {
        return Err(ServerError::NotFound("sampler-trace needs an id".into()));
    };
    let Some((meta, buffer)) = state.sampler_trace(&id) else {
        return Err(ServerError::NotFound(format!(
            "no sampler trace retained for {id}"
        )));
    };
    let buffer = buffer
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let body = packet(&id, &meta, &buffer);
    Ok((
        StatusCode::OK,
        [(header::CACHE_CONTROL, "no-store")],
        axum::Json(body),
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::sampler::{RecordedSample, SampleObservation, Sampler};

    /// Verified upper bound on one serialized position's bytes. Six candidates
    /// (five shown plus a rank-outside selected) with large ids and
    /// full-precision probabilities measure 372 on serde_json 1.0.151; the
    /// test below fails if the format ever grows past this bound, so the
    /// position budgets of this module stay honest.
    const WORST_CASE_BYTES_PER_POSITION: usize = 384;

    fn filled_handle(first: u32, positions: usize, limit: usize) -> CaptureHandle {
        let mut sampler = Sampler::greedy();
        let handle = sampler.arm_capture(limit);
        for n in 0..positions {
            sampler.record(&SampleObservation {
                selected: first + n as u32,
                selected_rank: 0,
                kept: 1,
                greedy: true,
                candidates: Vec::new(),
            });
        }
        handle
    }

    fn meta() -> TraceMeta {
        TraceMeta {
            model: "test-model".into(),
            temperature: 0.5,
            top_p: 0.9,
            top_k: 40,
        }
    }

    #[test]
    fn ring_evicts_oldest_beyond_the_reply_window() {
        let ring = SamplerTraceRing::default();
        for reply in 0..MAX_RETAINED_REPLIES + 2 {
            ring.register(&format!("r{reply}"), meta(), filled_handle(0, 100, 200));
        }
        assert!(ring.get("r0").is_none(), "older than the reply window");
        assert!(ring.get("r1").is_none());
        assert!(ring.get("r2").is_some(), "inside the reply window");
        assert!(
            ring.get(&format!("r{}", MAX_RETAINED_REPLIES + 1))
                .is_some()
        );
    }

    #[test]
    fn ring_replaces_a_same_id_registration() {
        let ring = SamplerTraceRing::default();
        ring.register("dup", meta(), filled_handle(0, 10, 100));
        ring.register("dup", meta(), filled_handle(100, 5, 100));
        let (_, handle) = ring.get("dup").expect("replaced entry");
        let buffer = handle.lock().expect("buffer lock");
        assert_eq!(buffer.positions.len(), 5);
        assert_eq!(buffer.positions[0].token, 100);
    }

    #[test]
    fn ring_enforces_the_global_position_budget_on_register() {
        let ring = SamplerTraceRing::default();
        // Entries at the full per-request size: the global budget fits
        // MAX_RETAINED_REPLIES of them, so registering one more evicts.
        for reply in 0..MAX_RETAINED_REPLIES {
            ring.register(
                &format!("full{reply}"),
                meta(),
                filled_handle(0, PER_REQUEST_POSITION_LIMIT, PER_REQUEST_POSITION_LIMIT),
            );
        }
        assert!(ring.get("full0").is_some());
        ring.register(
            "overflow",
            meta(),
            filled_handle(0, 1, PER_REQUEST_POSITION_LIMIT),
        );
        assert!(
            ring.get("full0").is_none(),
            "the global budget evicts oldest"
        );
        assert!(ring.get("overflow").is_some());
    }

    #[test]
    fn serialized_worst_case_position_stays_under_the_documented_bound() {
        // Six candidates (five shown plus a rank-outside selected), large ids,
        // full-precision probabilities: the measured serialized size must stay
        // under WORST_CASE_BYTES_PER_POSITION, or the memory budgets are lies.
        let mut sampler = Sampler::greedy();
        let handle = sampler.arm_capture(1);
        sampler.record(&SampleObservation {
            selected: 152_000,
            selected_rank: 9,
            kept: 512,
            greedy: false,
            candidates: vec![
                (151_996, 0.30000000000000004),
                (151_997, 0.20000000000000004),
                (151_998, 0.10000000000000001),
                (151_999, 0.05000000000000001),
                (151_995, 0.020000000000000004),
                (152_000, 0.33000000000000007),
            ],
        });
        let buffer = handle.lock().expect("buffer lock");
        let body = packet("chatcmpl-x", &meta(), &buffer);
        let per_position = serde_json::to_string(&body["positions"][0]).expect("serialize");
        assert!(
            per_position.len() < WORST_CASE_BYTES_PER_POSITION,
            "serialized {} bytes, bound {WORST_CASE_BYTES_PER_POSITION}: {per_position}",
            per_position.len()
        );
    }

    #[test]
    fn packet_reports_truncation_counts_and_metadata() {
        let handle = filled_handle(0, 10, 10);
        {
            let mut buffer = handle.lock().expect("buffer lock");
            buffer.truncated = true;
        }
        let buffer = handle.lock().expect("buffer lock");
        let body = packet("r", &meta(), &buffer);
        assert_eq!(body["packet"], 1);
        assert_eq!(body["truncated"], true);
        assert_eq!(body["captured_positions"], 10);
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["sampling"]["temperature"], 0.5);
        assert_eq!(body["sampling"]["top_k"], 40);
        assert_eq!(body["capture_scope"], "decode");
        assert_eq!(body["positions"].as_array().expect("array").len(), 10);
    }

    #[test]
    fn recorded_sample_round_trips_the_observation_fields() {
        let handle = filled_handle(7, 1, 8);
        let buffer = handle.lock().expect("buffer lock");
        let recorded: &RecordedSample = &buffer.positions[0];
        assert_eq!(recorded.token, 7);
        assert_eq!(recorded.kept, 1);
        assert!(recorded.greedy);
        assert_eq!(recorded.selected_prob, None);
    }

    // ---- Route e2e over the probe engine ---------------------------------------
    //
    // The probe engine's decode drives the REQUEST's sampler (it presents a
    // two-token row), so an armed request's capture holds real positions —
    // the honest end-to-end check of arm, record, ring, and route.

    use std::sync::Arc;

    use axum::http::{StatusCode, header};
    use serde_json::{Value, json};

    use super::super::router::router;
    use super::super::state::AppState;
    use super::super::testkit;

    fn app(sample_trace: bool) -> axum::Router {
        let probes = testkit::probes(std::time::Duration::ZERO);
        let entry = testkit::entry("e1", "model-a", &probes.engine, 100_000);
        let state = AppState::new(testkit::API_KEY, vec![entry]).with_sample_trace(sample_trace);
        router(Arc::new(state))
    }

    #[tokio::test]
    async fn an_armed_request_leaves_a_readable_packet_and_an_unchanged_reply() {
        for trace in [false, true] {
            let app = app(true);
            let body = testkit::post_json(
                "/v1/chat/completions",
                &json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}], "temperature": 0, "max_tokens": 3, "x-sample-trace": trace}),
            );
            let (status, _headers, bytes) = testkit::send(&app, body).await;
            assert_eq!(status, StatusCode::OK);
            let reply = testkit::json_of(&bytes);
            assert_eq!(reply["choices"][0]["finish_reason"], "length");
            if !trace {
                continue;
            }
            // The packet: three greedy positions from the probe's two-token
            // row, no probabilities, this reply's id and model.
            let get = testkit::get(
                &format!("/v1/sampler-trace?id={}", reply["id"].as_str().expect("id")),
                Some(testkit::API_KEY),
            );
            let (status, headers, bytes) = testkit::send(&app, get).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(
                headers
                    .get(header::CACHE_CONTROL)
                    .and_then(|v| v.to_str().ok()),
                Some("no-store")
            );
            let packet = testkit::json_of(&bytes);
            assert_eq!(packet["packet"], 1);
            assert_eq!(packet["reply_id"], reply["id"]);
            assert_eq!(packet["model"], "model-a");
            assert_eq!(packet["sampling"]["temperature"].as_f64(), Some(0.0));
            assert_eq!(packet["truncated"], false);
            assert_eq!(packet["captured_positions"], 3);
            let positions = packet["positions"].as_array().expect("positions");
            assert_eq!(positions.len(), 3);
            for position in positions {
                assert_eq!(position["token_id"], 1, "greedy argmax of [0.0, 1.0]");
                assert_eq!(position["greedy"], true);
                assert_eq!(position["selected_prob"], Value::Null);
                assert_eq!(
                    position["candidates"].as_array().expect("candidates").len(),
                    0
                );
            }
        }
    }

    #[tokio::test]
    async fn capture_is_refused_when_the_process_does_not_permit_it() {
        let app = app(false);
        let body = testkit::post_json(
            "/v1/chat/completions",
            &json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}], "temperature": 0, "max_tokens": 1, "x-sample-trace": true}),
        );
        let (status, _headers, bytes) = testkit::send(&app, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let error = testkit::json_of(&bytes);
        let text = error.to_string();
        assert!(text.contains("sample_trace"), "{text}");
    }

    #[tokio::test]
    async fn an_unarmed_request_registers_nothing_and_unknown_ids_are_not_found() {
        let app = app(true);
        let body = testkit::post_json(
            "/v1/chat/completions",
            &json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}], "temperature": 0, "max_tokens": 1}),
        );
        let (status, _headers, bytes) = testkit::send(&app, body).await;
        assert_eq!(status, StatusCode::OK);
        let reply = testkit::json_of(&bytes);
        let id = reply["id"].as_str().expect("id");
        let unarmed = testkit::get(
            &format!("/v1/sampler-trace?id={id}"),
            Some(testkit::API_KEY),
        );
        let (status, _h, _b) = testkit::send(&app, unarmed).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "no field: no trace");
        let unknown = testkit::get("/v1/sampler-trace?id=chatcmpl-nope", Some(testkit::API_KEY));
        let (status, _h, _b) = testkit::send(&app, unknown).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let bare = testkit::get("/v1/sampler-trace", Some(testkit::API_KEY));
        let (status, _h, _b) = testkit::send(&app, bare).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "missing id is explicit");
    }
}
