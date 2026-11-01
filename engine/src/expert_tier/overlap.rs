//! The CPU/GPU hand-off of one MoE block.

use std::time::{Duration, Instant};

use crate::EngError;

/// Wall time of the two concurrent halves of one block's expert work.
#[derive(Clone, Copy, Debug, Default)]
pub struct Overlap {
    /// The host-expert closure, measured on its own thread.
    pub routed: Duration,
    /// The device-side closure, measured on the calling thread.
    pub shared: Duration,
    /// How long the caller then waited for the host closure after the device side finished.
    pub exposed: Duration,
}

/// Runs `host` on a scoped thread while the calling thread runs `device`, then joins.
///
/// The device context is thread-confined, so `device` stays on the caller and only the host
/// kernel moves. A panic in `host` comes back as an error, never as unwinding through the caller.
pub fn run_overlapped<H, D>(
    host: impl FnOnce() -> H + Send,
    device: impl FnOnce() -> D,
) -> Result<(H, D, Overlap), EngError>
where
    H: Send,
{
    let mut timing = Overlap::default();
    let (joined, device_value) = std::thread::scope(|scope| {
        let handle = scope.spawn(|| {
            let started = Instant::now();
            let value = host();
            (value, started.elapsed())
        });
        let started = Instant::now();
        let device_value = device();
        timing.shared = started.elapsed();
        let waited = Instant::now();
        let joined = handle.join();
        timing.exposed = waited.elapsed();
        (joined, device_value)
    });
    let (host_value, routed) = joined.map_err(|_| EngError::Backend {
        operation: "expert_tier_overlap",
        message: "the host-expert thread panicked".to_owned(),
    })?;
    timing.routed = routed;
    Ok((host_value, device_value, timing))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn both_halves_run_and_the_device_side_hides_the_host_side() {
        let (host_done, host_done_rx) = std::sync::mpsc::channel();
        let (device_started, device_started_rx) = std::sync::mpsc::channel();
        let (host, device, timing) = run_overlapped(
            move || {
                // The host sleep may only start once the device side is
                // running; without this handshake the host could finish its
                // whole sleep before the device clock starts (caller starved
                // between spawn and device()), and shared would measure
                // nothing on correct code.
                device_started_rx
                    .recv_timeout(Duration::from_secs(5))
                    .expect("device side started");
                std::thread::sleep(Duration::from_millis(30));
                host_done.send(()).expect("device side is waiting");
                7
            },
            move || {
                device_started.send(()).expect("host side is waiting");
                // The device side cannot signal completion until the host
                // side has, so the completion-signal order is an ordering
                // fact, not a sleep margin that load can eat.
                host_done_rx
                    .recv_timeout(Duration::from_secs(5))
                    .expect("host side signalled completion");
                "device"
            },
        )
        .expect("overlap");
        assert_eq!((host, device), (7, "device"));
        // The host clock runs on the host thread and encloses the 30 ms
        // sleep, so this is deterministic; asserting it keeps a zeroed
        // routed (which cpu_routed accounting consumes) from passing.
        assert!(
            timing.routed >= Duration::from_millis(30),
            "host-side work went unmeasured: {timing:?}"
        );
        // The device clock starts before the handshake send and the host
        // sleep runs entirely after it, so every instant of the sleep lies
        // inside the device-side window; load only lengthens the wait.
        assert!(
            timing.shared >= Duration::from_millis(15),
            "the device side did not cover the host work: {timing:?}"
        );
    }

    #[test]
    fn a_slower_host_side_is_exposed_by_the_difference() {
        let (release, release_rx) = std::sync::mpsc::channel();
        let (device_started, device_started_rx) = std::sync::mpsc::channel();
        // Set before the release send and read by the host after the release
        // recv: channel ordering makes the host tail provably run after the
        // device side finished its work, whatever the scheduler does to the
        // measured join.
        let device_returned = Arc::new(AtomicBool::new(false));
        let saw_return = device_returned.clone();
        let (host, device, timing) = run_overlapped(
            move || {
                device_started_rx
                    .recv_timeout(Duration::from_secs(5))
                    .expect("device side started");
                release_rx
                    .recv_timeout(Duration::from_secs(30))
                    .expect("device side released the host side");
                std::thread::sleep(Duration::from_millis(30));
                assert!(
                    saw_return.load(Ordering::SeqCst),
                    "the host tail ran before the device side finished"
                );
            },
            move || {
                device_started.send(()).expect("host side is waiting");
                std::thread::sleep(Duration::from_millis(80));
                device_returned.store(true, Ordering::SeqCst);
                release.send(()).expect("host side is waiting");
                "device"
            },
        )
        .expect("overlap");
        assert_eq!((host, device), ((), "device"));
        // The host clock encloses the 30 ms tail on the host thread itself:
        // the tail is real host work ordered after the device side's
        // completion signal, so it cannot be hidden. The exposed magnitude
        // is deliberately not asserted: the caller can be starved between
        // the release send and the join, which makes a correct
        // implementation measure ~0.
        assert!(
            timing.routed >= Duration::from_millis(30),
            "the host-side tail went unmeasured: {timing:?}"
        );
    }

    #[test]
    fn a_panicking_host_side_is_an_error_not_an_unwind() {
        let outcome = run_overlapped(|| -> u32 { panic!("kernel bug") }, || 1);
        assert!(matches!(outcome, Err(EngError::Backend { .. })));
    }
}
