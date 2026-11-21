//! The owner of a card-resident engine: the step that runs on the card, the resources it reads
//! and the handles of the card it runs on, held together without a reference from one to
//! another.
//!
//! A step is built from clones of the shared handles ([`Handle::Shared`]) and the owner keeps
//! the originals, so no field borrows another and the whole value can move. Dropping the owner
//! is ordered: the stream is quiesced first, so no launch is in flight while anything is freed,
//! then the step (with the buffers it allocated), the resources, the weights, the stream and
//! last the context, which is released when its final holder goes. A family wires this by
//! building its step from [`Handle`]s and handing the owner the same handles; the owner has no
//! family names and no card types in its signature, so the order is tested on the host.

use std::{ops::Deref, rc::Rc};

use crate::EngError;

/// A handle that a step either borrows (a test or a caller that keeps the value alive itself)
/// or shares with its owner.
pub(crate) enum Handle<'a, T> {
    Borrowed(&'a T),
    Shared(Rc<T>),
}

impl<T> Deref for Handle<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        match self {
            Self::Borrowed(value) => value,
            Self::Shared(value) => value,
        }
    }
}

impl<'a, T> Handle<'a, T> {
    /// The borrowed value with its full lifetime, or `None` for a shared handle.
    pub(crate) fn borrowed(&self) -> Option<&'a T> {
        match self {
            Self::Borrowed(value) => Some(value),
            Self::Shared(_) => None,
        }
    }
}

impl<T> Clone for Handle<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Borrowed(value) => Self::Borrowed(value),
            Self::Shared(value) => Self::Shared(Rc::clone(value)),
        }
    }
}

impl<'a, T> From<&'a T> for Handle<'a, T> {
    fn from(value: &'a T) -> Self {
        Self::Borrowed(value)
    }
}

impl<'a, T> From<&'a mut T> for Handle<'a, T> {
    fn from(value: &'a mut T) -> Self {
        Self::Borrowed(value)
    }
}

impl<T> From<Rc<T>> for Handle<'_, T> {
    fn from(value: Rc<T>) -> Self {
        Self::Shared(value)
    }
}

/// A stream that can wait until everything queued on it has finished.
pub(crate) trait Quiesce {
    fn quiesce(&self) -> Result<(), EngError>;
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
impl Quiesce for crate::kernels::cuda::DeviceStream {
    fn quiesce(&self) -> Result<(), EngError> {
        self.synchronize()
    }
}

/// The step `S`, the resources `R` it reads (a model, pinned buffers), the weights `W`, the
/// stream `T` and the context `C`, owned together. The fields drop in declaration order after
/// [`Drop::drop`] has quiesced the stream and dropped the step.
pub(crate) struct EngineOwner<S, R, W, T: Quiesce, C> {
    step: Option<S>,
    resources: R,
    weights: Rc<W>,
    stream: Rc<T>,
    context: Rc<C>,
}

impl<S, R, W, T: Quiesce, C> EngineOwner<S, R, W, T, C> {
    /// The owner of `step`, which was built from clones of `weights`, `stream` and `context`.
    pub(crate) fn new(
        step: S,
        resources: R,
        weights: Rc<W>,
        stream: Rc<T>,
        context: Rc<C>,
    ) -> Self {
        Self {
            step: Some(step),
            resources,
            weights,
            stream,
            context,
        }
    }

    pub(crate) fn step(&self) -> &S {
        self.step.as_ref().expect("the step lives until the drop")
    }

    pub(crate) fn step_mut(&mut self) -> &mut S {
        self.step.as_mut().expect("the step lives until the drop")
    }

    pub(crate) fn resources(&self) -> &R {
        &self.resources
    }

    pub(crate) fn stream(&self) -> &T {
        &self.stream
    }
}

impl<S, R, W, T: Quiesce, C> Drop for EngineOwner<S, R, W, T, C> {
    fn drop(&mut self) {
        // A failed wait leaves nothing to do but free: the context's own teardown drains.
        let _ = self.stream.quiesce();
        drop(self.step.take());
        // `resources`, `weights`, `stream` and `context` drop after this body, in order.
        let _ = (&self.weights, &self.context);
    }
}

/// Device bytes allocated by this process and not yet freed: zero once an engine has stopped.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn live_device_bytes() -> usize {
    crate::kernels::cuda::live_device_bytes()
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use super::{EngineOwner, Handle, Quiesce};
    use crate::EngError;

    type Log = Rc<RefCell<Vec<&'static str>>>;

    struct Named(&'static str, Log);

    impl Drop for Named {
        fn drop(&mut self) {
            self.1.borrow_mut().push(self.0);
        }
    }

    struct FakeStream(Log);

    impl Quiesce for FakeStream {
        fn quiesce(&self) -> Result<(), EngError> {
            self.0.borrow_mut().push("quiesce");
            Ok(())
        }
    }

    impl Drop for FakeStream {
        fn drop(&mut self) {
            self.0.borrow_mut().push("stream");
        }
    }

    #[test]
    fn the_owner_quiesces_then_drops_the_step_the_resources_the_weights_the_stream_and_the_context()
    {
        let log: Log = Rc::default();
        let weights = Rc::new(Named("weights", log.clone()));
        let stream = Rc::new(FakeStream(log.clone()));
        let context = Rc::new(Named("context", log.clone()));
        // The step holds clones of the shared handles, as a family's step does.
        let step = (
            Named("step", log.clone()),
            Handle::from(Rc::clone(&weights)),
            Handle::from(Rc::clone(&stream)),
            Handle::from(Rc::clone(&context)),
        );
        let owner = EngineOwner::new(
            step,
            Named("resources", log.clone()),
            weights,
            stream,
            context,
        );
        drop(owner);
        assert_eq!(
            *log.borrow(),
            [
                "quiesce",
                "step",
                "resources",
                "weights",
                "stream",
                "context"
            ]
        );
    }

    #[test]
    fn a_borrowed_handle_reads_like_the_value_and_a_shared_one_counts_as_a_holder() {
        let value = 7u32;
        let borrowed = Handle::from(&value);
        assert_eq!(*borrowed, 7);
        let shared = Rc::new(9u32);
        let handle = Handle::from(Rc::clone(&shared));
        assert_eq!(Rc::strong_count(&shared), 2);
        let copy = handle.clone();
        assert_eq!((*copy, Rc::strong_count(&shared)), (9, 3));
    }

    #[test]
    fn the_step_is_reachable_until_the_owner_drops() {
        let log: Log = Rc::default();
        let mut owner = EngineOwner::new(
            3u32,
            (),
            Rc::new(()),
            Rc::new(FakeStream(log.clone())),
            Rc::new(()),
        );
        *owner.step_mut() += 1;
        assert_eq!(*owner.step(), 4);
        assert!(log.borrow().is_empty());
    }
}
