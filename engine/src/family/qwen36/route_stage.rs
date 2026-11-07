//! The per-step routing store of the capture-preserving telemetry stage
//! (`Q36_ROUTE_STAGE=1`): the D2 successor of the eager `route_capture` spike. The CUDA step
//! D2D-copies each MoE layer's router picks (ids, weights, shared gate) into a staging buffer as
//! part of its ordinary enqueued work — launch-only copies, legal inside a graph capture, so a
//! captured step refreshes the stage on every replay — and reads the whole buffer back once per
//! token, after the step's own read-back (the stream is drained by then; the copy rides that
//! sync). The host side keeps the last [`RING_STEPS`] steps in a process-wide ring, newest last,
//! for the server packet of a later slice and for the bench-side sanity log.
//!
//! The ring is process-wide like the layer-trace counters: every session and warm-up step
//! advances [`NEXT`], and concurrent decode sessions interleave (one engine, one decode step in
//! practice). When off, a step carries one env read at construction and `Option` checks.

use super::route_capture::LayerPicks;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

/// The env opt-in: `1` arms the stage, any other value (including unset) leaves it off.
pub(crate) const VAR: &str = "Q36_ROUTE_STAGE";

/// How many steps the ring keeps; older steps fall off the front. One step of the 35B model is
/// 40 layers x 68 bytes of picks, so the cap is a few KiB.
pub(crate) const RING_STEPS: usize = 16;

/// The parse of [`VAR`], separate from the env read so tests can cover it.
pub(crate) fn enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// The stage's build-time facts (MoE layer count, top-k, expert count), set once when the stage
/// is allocated: the server packet reports them so the view's color scale and summaries do not
/// guess them from the ids. First engine wins; one process does not serve two qwen36 shapes.
static FACTS: OnceLock<(usize, usize, usize)> = OnceLock::new();

/// Records the facts alongside stage allocation.
pub(crate) fn set_facts(layers: usize, top_k: usize, experts: usize) {
    let _ = FACTS.set((layers, top_k, experts));
}

/// The recorded facts, `None` while no stage was ever allocated in this process.
pub(crate) fn facts() -> Option<(usize, usize, usize)> {
    FACTS.get().copied()
}

/// Process-wide step counter, advanced once per decode step that stages its picks.
static NEXT: AtomicU32 = AtomicU32::new(0);

static RING: Mutex<VecDeque<(u32, Vec<LayerPicks>)>> = Mutex::new(VecDeque::new());

/// Claims the next step index for a step about to stage its picks.
pub(in crate::family::qwen36) fn begin_step() -> u32 {
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Stores one step's picks; the ring drops the oldest beyond [`RING_STEPS`].
pub(in crate::family::qwen36) fn store(step: u32, picks: Vec<LayerPicks>) {
    if let Ok(mut ring) = RING.lock() {
        ring.push_back((step, picks));
        while ring.len() > RING_STEPS {
            ring.pop_front();
        }
    }
}

/// A copy of the ring, oldest first, for the server's route-trace packet and for tests.
pub(crate) fn snapshot() -> Vec<(u32, Vec<LayerPicks>)> {
    RING.lock()
        .map(|mut ring| ring.make_contiguous().to_vec())
        .unwrap_or_default()
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
    fn only_one_arms_the_stage() {
        assert!(enabled(Some("1")));
        assert!(!enabled(None));
        assert!(!enabled(Some("")));
        assert!(!enabled(Some("0")));
        assert!(!enabled(Some("true")));
    }

    #[test]
    fn the_ring_keeps_the_last_steps_in_order() {
        // One test, in order: the ring is a process-wide static, so parallel tests would race.
        for step in 0u32..(RING_STEPS + 4) as u32 {
            store(step, vec![picks(step as i32); 2]);
        }
        let ring = snapshot();
        assert_eq!(ring.len(), RING_STEPS);
        assert_eq!(ring.first().map(|(step, _)| *step), Some(4));
        assert_eq!(
            ring.last().map(|(step, _)| *step),
            Some((RING_STEPS + 3) as u32)
        );
        assert_eq!(ring[0].1[0].ids, vec![4; 8]);
        assert_eq!(ring.last().unwrap().1.len(), 2);
    }
}
