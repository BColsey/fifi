mod config;
#[cfg(feature = "native-cuda")]
pub(crate) mod executor;
mod gguf;
mod prompt;
#[cfg(feature = "native-cuda")]
pub(crate) mod resident;
#[cfg(test)]
mod tests;

/// The server-family key a config names to select this family. The
/// registry row exists but its starter refuses while the family is
/// staged; see `start_gemma4` in the server factory.
pub(crate) const FAMILY: &str = "gemma4";

pub use config::check_humanizer_config;
pub use gguf::open_humanizer_q8;
#[cfg(feature = "native-cuda")]
pub(crate) use gguf::open_humanizer_weights;
