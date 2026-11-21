mod budget;
mod config;
#[cfg(feature = "native-cuda")]
pub(crate) mod executor;
mod gguf;
#[cfg(feature = "native-cuda")]
mod kv_tier;
mod prompt;
#[cfg(feature = "native-cuda")]
pub(crate) mod resident;
pub(crate) mod serve;
#[cfg(test)]
mod tests;

/// The server-family key a config names to select this family. The starter
/// serves the Nemotron Bolt E4B on the native-cuda build and refuses any
/// other model in the family; see `start_gemma4` in the server factory.
pub(crate) const FAMILY: &str = "gemma4";

pub(crate) use config::Gemma4Options;
pub use config::{check_bolt_config, check_humanizer_config};
#[cfg(feature = "native-cuda")]
pub(crate) use gguf::open_bolt_weights;
#[cfg(feature = "native-cuda")]
pub(crate) use gguf::open_humanizer_weights;
pub use gguf::{open_bolt_q8, open_humanizer_q8};
