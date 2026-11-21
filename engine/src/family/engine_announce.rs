//! What an engine thread tells its `start` before the weights finish loading.
//!
//! A start that leaves the context (or another figure) to the card cannot know it until the engine
//! thread has read the free memory, yet the server's request bound needs it before the engine is
//! ready. The thread sends the choice once through a one-slot channel and `start` waits for it; a
//! failure before the choice travels the same way, so the start fails with the cause instead of
//! hanging or reporting a closed channel.

use crate::EngError;
use std::sync::mpsc::{Receiver, SyncSender};

/// The choice an engine thread announces, or the failure that ended the start before it.
pub(crate) type Announcement<T> = Result<T, String>;

/// Waits for the announcement. A closed channel means the thread ended without one; `failure`
/// then supplies the cause the engine recorded. `what` names the engine in the refusal.
pub(crate) fn await_announced<T>(
    receiver: &Receiver<Announcement<T>>,
    what: &str,
    failure: impl FnOnce() -> Option<String>,
) -> Result<T, EngError> {
    let failed = |cause: String| {
        EngError::Checkpoint(format!(
            "{what} start failed before the memory plan: {cause}"
        ))
    };
    match receiver.recv() {
        Ok(Ok(choice)) => Ok(choice),
        Ok(Err(cause)) => Err(failed(cause)),
        Err(_) => Err(failed(failure().unwrap_or_else(|| {
            "the engine thread ended without a cause".to_string()
        }))),
    }
}

/// Sends the failure of a build that ended before the announcement. After the announcement the
/// slot is taken (or the receiver has left) and the failure reaches the engine state instead.
pub(crate) fn announce_failure<T, R>(
    built: &Result<R, EngError>,
    announce: Option<&SyncSender<Announcement<T>>>,
) {
    if let (Err(error), Some(announce)) = (built, announce) {
        let _ = announce.try_send(Err(error.describe()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::sync_channel;

    #[test]
    fn the_start_takes_what_the_engine_thread_announced() {
        let (sender, receiver) = sync_channel::<Announcement<u32>>(1);
        sender.send(Ok(75_776)).expect("send");
        let got = await_announced(&receiver, "x", || panic!("no failure is asked for"));
        assert_eq!(got.expect("choice"), 75_776);
    }

    #[test]
    fn a_failure_before_the_choice_is_the_start_failure() {
        let (sender, receiver) = sync_channel::<Announcement<u32>>(1);
        sender
            .send(Err("no CUDA device".to_string()))
            .expect("send");
        let error = await_announced(&receiver, "Bolt", || None)
            .expect_err("failure")
            .describe();
        assert_eq!(
            error,
            "Bolt start failed before the memory plan: no CUDA device"
        );
        // The thread ended and dropped the sender: the engine's recorded cause is used.
        let (sender, receiver) = sync_channel::<Announcement<u32>>(1);
        drop(sender);
        let error = await_announced(&receiver, "Bolt", || Some("unreadable".to_string()))
            .expect_err("failure")
            .describe();
        assert!(
            error.ends_with("before the memory plan: unreadable"),
            "{error}"
        );
        // No cause recorded: the text still says what happened.
        let (sender, receiver) = sync_channel::<Announcement<u32>>(1);
        drop(sender);
        let error = await_announced(&receiver, "Bolt", || None)
            .expect_err("failure")
            .describe();
        assert!(error.contains("ended without a cause"), "{error}");
    }

    #[test]
    fn an_error_is_announced_once_and_a_taken_slot_is_left_alone() {
        let (sender, receiver) = sync_channel::<Announcement<u32>>(1);
        let failed: Result<(), EngError> = Err(EngError::BadRequest("boom".into()));
        announce_failure(&failed, Some(&sender));
        // A second failure finds the slot full and is dropped without blocking.
        announce_failure(&failed, Some(&sender));
        announce_failure::<u32, ()>(&Ok(()), Some(&sender));
        assert!(matches!(receiver.try_recv(), Ok(Err(_))));
        assert!(receiver.try_recv().is_err());
        announce_failure::<u32, ()>(&failed, None);
    }
}
