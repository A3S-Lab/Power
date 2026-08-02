//! Validated static tensor graphs for model crates.
//!
//! The executor owns only provider-neutral graph validation and reviewed
//! operators. Model identity, embedded plans, preprocessing, postprocessing,
//! tokenizers, and revision policy stay in the consuming model crate.

mod executor;
mod plan;
mod value;

pub use executor::GraphExecutor;
pub use plan::{GraphIdentity, GraphPlan};

#[cfg(test)]
mod tests;
