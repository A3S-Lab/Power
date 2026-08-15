//! Compatibility exports for picolm's speculative decode loop.
//!
//! The reusable implementation lives in [`crate::speculative`] so every model
//! backend can consume the same proposal, scheduling, and acceptance primitives.

pub use crate::speculative::{AdaptiveK, SpecMode, DRAFT_K};
