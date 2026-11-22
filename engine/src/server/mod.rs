//! Contract: `api/openapi.yaml`.
//! The server keeps a fixed dependency set, hand-written wire types, and one error wrapper.

pub(crate) mod anthropic;
mod auth;
#[cfg(test)]
mod cancel_tests;
mod chat;
mod completions;
pub(crate) mod config;
#[cfg(test)]
mod contract;
mod engines;
mod error;
mod fake;
#[cfg(test)]
mod family_registry_tests;
mod fetch_name;
#[cfg(test)]
mod flint_tests;
mod hf;
#[cfg(test)]
mod hf_tests;
mod local_models;
mod messages;
mod model_serve;
mod pieces;
mod reply;
mod reqlog;
mod route_trace;
mod router;
pub(crate) mod run;
mod sampler_trace;
mod sse;
mod state;
pub(crate) mod supervisor;
#[cfg(test)]
mod testkit;
mod tokenize;
mod validate;
mod wire;

pub use config::StartupError;
pub use run::run_cli;
