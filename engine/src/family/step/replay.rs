//! A one-row step that replays as a captured graph.
//!
//! A decode step is a few thousand small launches, and the host cost of queueing them leaves the
//! device idle between kernels. A step whose launches do not depend on the position (every
//! kernel reads the position from device words) is the same sequence every token, so it is
//! recorded once and replayed with one call.
//!
//! [`Replay::run`] takes the body of the step. The first step runs it as it is, which loads
//! the kernels a capture may not load. The second records it and replays the record; every
//! later step replays. A device that refuses the capture runs the body as it is from then on.
//!
//! The body queues launches only: uploads of the step's inputs come before the call. The graph
//! holds the addresses of the buffers the body used, so the owner keeps those buffers for as
//! long as the [`Replay`] and drops the `Replay` first.

use crate::EngError;
use crate::kernels::cuda::{CapturePlan, CudaContext, DeviceStream, GraphExec, is_capture_refusal};

/// Where a step stands: not run yet, run once and ready to record, recorded, or refused.
#[derive(Default)]
enum State {
    #[default]
    Cold,
    Warm,
    Recorded(GraphExec<'static>),
    Refused,
}

/// The record of one step and what the body left behind when it was recorded.
pub(crate) struct Replay<T> {
    state: State,
    left: Option<T>,
}

impl<T> Default for Replay<T> {
    fn default() -> Self {
        Self {
            state: State::Cold,
            left: None,
        }
    }
}

impl<T: Copy> Replay<T> {
    /// Runs one step on `stream` and returns what `body` returns: the host state the step
    /// leaves, which a replay hands back without running the body.
    pub(crate) fn run(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        mut body: impl FnMut() -> Result<T, EngError>,
    ) -> Result<T, EngError> {
        match std::mem::take(&mut self.state) {
            State::Cold => {
                let left = body()?;
                self.state = State::Warm;
                Ok(left)
            }
            State::Warm => {
                let plan = CapturePlan {
                    pins: &[],
                    holds: vec![ctx.workspace_hold()?],
                };
                match ctx.capture(stream, plan, &mut body) {
                    Ok((graph, left)) => {
                        graph.launch(stream)?;
                        self.state = State::Recorded(graph);
                        self.left = Some(left);
                        Ok(left)
                    }
                    Err(refusal) if is_capture_refusal(&refusal) => {
                        self.state = State::Refused;
                        body()
                    }
                    Err(failure) => Err(failure),
                }
            }
            State::Recorded(graph) => {
                let launched = graph.launch(stream);
                self.state = State::Recorded(graph);
                launched?;
                self.left.ok_or_else(|| {
                    EngError::BadRequest("a recorded step lost its host state".to_owned())
                })
            }
            State::Refused => {
                self.state = State::Refused;
                body()
            }
        }
    }

    /// Whether the last step was a replay of the record.
    #[cfg(test)]
    pub(crate) fn recorded(&self) -> bool {
        matches!(self.state, State::Recorded(_))
    }
}
