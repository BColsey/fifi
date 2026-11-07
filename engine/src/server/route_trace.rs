//! The engine-wide routing-trace packet and its read route.
//!
//! Contract: the stage is opt-in at process level only (`Q36_ROUTE_STAGE=1`, read by the CUDA
//! decode step, which then D2D-stages every plain-MoE layer's router picks and reads them back
//! once per token). There is no per-request arming, unlike the sampler trace: the staging copies
//! run on every decode step of every session when the flag is on, so per-request gating would
//! not change what the device does — the endpoint reports the shared last-16-step ring, and the
//! GUI treats it as a live aggregate view, not a per-reply record. A step's picks are not
//! correlated to a reply id; the reply correlation of the sampler trace is a different packet.
//!
//! The route is a server extension outside the client contract, exactly like
//! `/v1/recent-requests` and `/v1/sampler-trace`, and carries the process API key. When the
//! flag is off the route answers 404 with the enabling hint, never an empty packet, so the GUI
//! cannot mistake "off" for "no activity".

use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use super::error::ServerError;
use crate::family::qwen36::route_capture::LayerPicks;
use crate::family::qwen36::route_stage;

/// The wire packet: version, the ring's step capacity, and the retained steps oldest-first,
/// each with one layer entry per MoE layer in launch order (ids as u16: expert indices of the
/// supported models are far below 65536, and the router-fault `-1` serializes as 65535;
/// weights and the shared gate as f32).
fn packet(steps: &[(u32, Vec<LayerPicks>)]) -> Value {
    json!({
        "packet": 1,
        "ring_capacity": route_stage::RING_STEPS,
        "steps": steps.iter().map(|(step, layers)| json!({
            "step": step,
            "layers": layers.iter().map(|picks| json!({
                "ids": picks.ids.iter().map(|id| *id as u16).collect::<Vec<_>>(),
                "weights": picks.weights,
                "gate": picks.shared_gate,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

/// `GET /v1/route-trace`: the routing stage's ring. 404 with the enabling hint when the stage
/// is off; `no-store` because the ring moves with every decode step.
pub(crate) async fn get_route_trace() -> Result<Response, ServerError> {
    let flag = std::env::var(route_stage::VAR).ok();
    if !route_stage::enabled(flag.as_deref()) {
        return Err(ServerError::NotFound(
            "route stage off: start the engine with Q36_ROUTE_STAGE=1".into(),
        ));
    }
    let body = packet(&route_stage::snapshot());
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

    fn picks(id: i32, weight: f32, gate: f32) -> LayerPicks {
        LayerPicks {
            ids: vec![id, id + 1],
            weights: vec![weight, weight / 2.0],
            shared_gate: gate,
        }
    }

    #[test]
    fn packet_carries_steps_oldest_first_with_u16_ids() {
        let steps = vec![
            (7_u32, vec![picks(3, 0.25, 0.5); 2]),
            (8_u32, vec![picks(500, 0.75, 1.0); 2]),
        ];
        let body = packet(&steps);
        assert_eq!(body["packet"], 1);
        assert_eq!(body["ring_capacity"], route_stage::RING_STEPS);
        let steps = body["steps"].as_array().expect("steps array");
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0]["step"], 7);
        let layers = steps[0]["layers"].as_array().expect("layers array");
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0]["ids"], json!([3, 4]));
        assert_eq!(layers[0]["weights"], json!([0.25, 0.125]));
        assert_eq!(layers[0]["gate"], 0.5);
        assert_eq!(steps[1]["layers"][0]["ids"], json!([500, 501]));
    }

    #[test]
    fn router_fault_ids_serialize_as_u16_max() {
        let steps = vec![(1_u32, vec![picks(-1, 0.0, 0.0)])];
        let body = packet(&steps);
        assert_eq!(body["steps"][0]["layers"][0]["ids"], json!([65535, 0]));
    }

    #[test]
    fn empty_ring_serializes_an_empty_steps_array() {
        let body = packet(&[]);
        assert_eq!(body["steps"], json!([]));
        assert_eq!(body["packet"], 1);
    }

    #[tokio::test]
    async fn the_route_answers_404_with_the_hint_when_the_stage_is_off() {
        // The test binary's environment does not carry the flag (no other test sets it), so the
        // handler takes its off branch; the assertion is skipped if a developer exported it.
        let flag = std::env::var(route_stage::VAR).ok();
        if !route_stage::enabled(flag.as_deref()) {
            let error = get_route_trace().await.expect_err("off stage");
            assert!(
                matches!(error, ServerError::NotFound(ref message) if message.contains("Q36_ROUTE_STAGE"))
            );
        }
    }
}
