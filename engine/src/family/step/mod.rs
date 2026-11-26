//! The shared resident step on one CUDA device: what a model's forward pass needs that is not its
//! own math.
//!
//! A step runs a chunk of token rows layer by layer, and decoding is the chunk of one row, so
//! prefill and decode are one walk. [`walk`] holds that walk around a model's layers: the pieces
//! of a feed, the bounds and the head rows. [`rows`] holds the activation rows a projection reads
//! and the projections themselves, chosen by the type of each weight tensor.
//!
//! [`ffn`] holds the gated feed-forward block over those projections, [`experts`] the routed
//! expert projections of a chunk, and [`site`] the residual sites: the plain add and the routed
//! two-stream site.
//!
//! [`delta`] and [`attend`] are the mixers of a visit: the gated delta rule and the gated full
//! attention, each over one row or a chunk.
//!
//! [`replay`] records a one-row step once and replays it, so a decode step is one call.
//!
//! A model supplies its layers (the mixer and the feed-forward of each visit) and its weights by
//! name.

pub(crate) mod attend;
pub(crate) mod delta;
pub(crate) mod experts;
pub(crate) mod ffn;
pub(crate) mod replay;
pub(crate) mod rows;
pub(crate) mod site;
pub(crate) mod walk;
