//! The shared resident step on one CUDA device: what a model's forward pass needs that is not its
//! own math.
//!
//! A step runs a chunk of token rows layer by layer, and decoding is the chunk of one row, so
//! prefill and decode are one walk. [`walk`] holds that walk around a model's layers: the pieces
//! of a feed, the bounds and the head rows. [`rows`] holds the activation rows a projection reads
//! and the projections themselves, chosen by the type of each weight tensor.
//!
//! [`ffn`] holds the gated feed-forward block over those projections, and [`site`] the residual
//! sites: the plain add and the routed two-stream site.
//!
//! A model supplies its layers (the mixer and the feed-forward of each visit) and its weights by
//! name.

pub(crate) mod ffn;
pub(crate) mod rows;
pub(crate) mod site;
pub(crate) mod walk;
