//! The routing readback spike (`Q36_ROUTE_CAPTURE=1`): the per-layer MoE router picks of a
//! decode step, read back to the host while serving runs. This answers the D1 question — what
//! a per-step, per-layer routing readback costs — by measuring the UPPER bound: three small
//! synchronous reads (`read_at`) after each layer's MoE block. A host read cannot live inside a
//! captured graph, so a step with the capture on runs EAGER, like the layer trace; the flag is
//! a diagnostic measurement opt-in, not a serving feature, and concurrent decode sessions
//! interleave in the one staging slot.
//!
//! The picks are staged per step: `begin_step` clears the slot and claims a process-wide step
//! index (decode steps only, the same counter discipline as the layer trace), `record` appends
//! one layer's picks in launch order, and `staged` clones the slot for a test or a bench probe.
//! The next `begin_step` overwrites the slot, so `staged` between steps shows the last staged
//! step. When off, a step carries one cached env read (at decoder construction) and one bool
//! check per MoE block.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

/// The env opt-in: `1` arms the capture, any other value (including unset) leaves it off.
pub(in crate::family::qwen36) const VAR: &str = "Q36_ROUTE_CAPTURE";

/// The parse of [`VAR`], separate from the env read so tests can cover it.
pub(in crate::family::qwen36) fn enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// One layer's router picks: the pick ids (i32, `-1` is the router-fault value), the
/// renormalized pick weights, and the shared-expert gate.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::family::qwen36) struct LayerPicks {
    pub(in crate::family::qwen36) ids: Vec<i32>,
    pub(in crate::family::qwen36) weights: Vec<f32>,
    pub(in crate::family::qwen36) shared_gate: f32,
}

/// Process-wide, like the layer-trace counters: every session and warm-up step of the process
/// shares them, so indices line up only between fresh processes.
static NEXT: AtomicU32 = AtomicU32::new(0);

static STAGED: Mutex<Option<(u32, Vec<LayerPicks>)>> = Mutex::new(None);

/// Claims the next step index and clears the staging slot for it.
pub(in crate::family::qwen36) fn begin_step() -> u32 {
    let step = NEXT.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut staged) = STAGED.lock() {
        *staged = Some((step, Vec::new()));
    }
    step
}

/// Appends one layer's picks to the staged step, in launch order. There is no step index to
/// check: `begin_step` and `record` run on the decode thread between two of its steps, so the
/// appends cannot pass a later `begin_step`; concurrent sessions interleave in the one slot
/// (a diagnostic measurement flag, not a serving feature).
pub(in crate::family::qwen36) fn record(picks: LayerPicks) {
    if let Ok(mut staged) = STAGED.lock()
        && let Some((_, layers)) = staged.as_mut()
    {
        layers.push(picks);
    }
}

/// A snapshot of the staged step, for the device test and the bench probe.
pub(in crate::family::qwen36) fn staged() -> Option<(u32, Vec<LayerPicks>)> {
    STAGED.lock().ok().and_then(|guard| guard.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picks(id: i32) -> LayerPicks {
        LayerPicks {
            ids: vec![id; 8],
            weights: vec![0.125; 8],
            shared_gate: 0.5,
        }
    }

    #[test]
    fn only_one_arms_the_capture() {
        assert!(enabled(Some("1")));
        assert!(!enabled(None));
        assert!(!enabled(Some("")));
        assert!(!enabled(Some("0")));
        assert!(!enabled(Some("true")));
    }

    #[test]
    fn begin_clears_and_record_appends_in_launch_order() {
        // One test, in order: the three statics are process-wide, so parallel tests would race.
        let first = begin_step();
        record(picks(3));
        record(picks(4));
        let (step, layers) = staged().expect("the first step is staged");
        assert_eq!(step, first);
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].ids, vec![3; 8]);
        assert_eq!(layers[1].shared_gate, 0.5);

        let second = begin_step();
        assert_eq!(second, first + 1);
        let (step, layers) = staged().expect("begin cleared into the second step");
        assert_eq!(step, second);
        assert!(layers.is_empty());

        record(picks(9));
        let (step, layers) = staged().expect("the record lands in the new slot");
        assert_eq!(step, second);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].ids, vec![9; 8]);
    }
}
