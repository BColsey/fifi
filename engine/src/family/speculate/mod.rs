//! Speculative decoding, host side: the settings, the draft sources that need only the committed
//! tokens, the acceptance rule and the bookkeeping of a verify window.
//!
//! Nothing here touches a device. A draft is a guess for the tokens that follow the pending
//! token; the target model runs one window of `1 + drafts` rows and draws a token at every row
//! with the job's own sampler. A draft is accepted exactly when it equals that draw, so the
//! committed tokens are the draws a one-row run would make, for greedy and for sampled requests
//! alike (see `accept`). The settings default to `draft_mode = disabled`.

pub(crate) mod accept;
pub(crate) mod calibrator;
pub(crate) mod config;
#[cfg(test)]
pub(crate) mod d1_oracle;
#[cfg(test)]
mod d1_oracle_tests;
pub(crate) mod driver;
pub(crate) mod mtp;
pub(crate) mod repeat;
pub(crate) mod stats;
#[cfg(test)]
mod study;
pub(crate) mod window;
#[cfg(test)]
pub(crate) mod window_cost;
#[cfg(test)]
mod window_cost_tests;
