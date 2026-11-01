//! A test-only clock for the stages of a layer-major piece. Each mark waits for the stream and
//! adds the time since the previous mark to its stage, so the table says where a piece goes. The
//! waits cost time themselves; the figures rank stages, they do not replace an unmarked run.

use crate::kernels::cuda::DeviceStream;
use std::{cell::RefCell, collections::BTreeMap, time::Instant};

struct Clock {
    last: Instant,
    stages: BTreeMap<&'static str, (f64, u32)>,
}

thread_local! {
    static CLOCK: RefCell<Option<Clock>> = const { RefCell::new(None) };
}

/// Starts a fresh table; marks before this are ignored.
pub(crate) fn arm() {
    CLOCK.with(|clock| {
        *clock.borrow_mut() = Some(Clock {
            last: Instant::now(),
            stages: BTreeMap::new(),
        });
    });
}

/// Charges the time since the previous mark to `stage`, after the stream has caught up.
pub(crate) fn mark(stream: &DeviceStream, stage: &'static str) {
    CLOCK.with(|clock| {
        let mut clock = clock.borrow_mut();
        let Some(clock) = clock.as_mut() else {
            return;
        };
        if stream.synchronize().is_err() {
            return;
        }
        let now = Instant::now();
        let entry = clock.stages.entry(stage).or_insert((0.0, 0));
        entry.0 += now.duration_since(clock.last).as_secs_f64() * 1e3;
        entry.1 += 1;
        clock.last = now;
    });
}

/// The table, stage name with total milliseconds and marks, in descending time.
pub(crate) fn take() -> Vec<(&'static str, f64, u32)> {
    let mut rows: Vec<_> = CLOCK
        .with(|clock| clock.borrow_mut().take())
        .map(|clock| {
            clock
                .stages
                .into_iter()
                .map(|(stage, (ms, marks))| (stage, ms, marks))
                .collect()
        })
        .unwrap_or_default();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    rows
}
