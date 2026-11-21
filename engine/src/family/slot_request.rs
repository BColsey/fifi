//! How many expert-bank slots per layer a load asks for, and how a config spells it.

use serde::Deserialize;

/// How many bank slots per layer the banked load holds: an exact operator count, or `Fit` for
/// the largest count the admission arithmetic admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotRequest {
    Fit,
    Exactly(usize),
}

impl<'de> Deserialize<'de> for SlotRequest {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Input {
            Count(u32),
            Name(String),
        }
        match Input::deserialize(deserializer)? {
            Input::Count(count) => usize::try_from(count)
                .map(Self::Exactly)
                .map_err(serde::de::Error::custom),
            Input::Name(name) if name == "auto" => Ok(Self::Fit),
            Input::Name(_) => Err(serde::de::Error::custom(
                "tiered.slots expects an integer or \"auto\"",
            )),
        }
    }
}
