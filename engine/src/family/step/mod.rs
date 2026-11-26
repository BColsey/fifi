//! The shared resident step on one CUDA device: what a model's forward pass needs that is not its
//! own math.
//!
//! A step runs a chunk of token rows layer by layer, and decoding is the chunk of one row, so
//! prefill and decode are one walk. [`walk`] holds that walk around a model's layers: the pieces
//! of a feed, the bounds and the head rows. [`rows`] holds the activation rows a projection reads
//! and the projections themselves, chosen by the type of each weight tensor.
//!
//! A model supplies its layers (the mixer and the feed-forward of each visit) and its weights by
//! name.

pub(crate) mod rows;
pub(crate) mod walk;
