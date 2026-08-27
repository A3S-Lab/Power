//! Validated static tensor graphs for model crates.
//!
//! The executor owns only provider-neutral graph validation and reviewed
//! operators. Model identity, embedded plans, preprocessing, postprocessing,
//! tokenizers, and revision policy stay in the consuming model crate.

mod executor;
mod matrix_multiplication;
mod plan;
mod row_softmax_top1;
mod row_top1;
mod transpose_folding;
mod value;

pub use executor::GraphExecutor;
pub use plan::{GraphIdentity, GraphPlan};
pub use row_softmax_top1::{
    row_bias_softmax_top1_last_finite, row_matmul_bias_softmax_top1_last_finite,
    row_softmax_top1_last_finite,
};
pub use row_top1::row_top1_last_finite;

#[cfg(test)]
mod tests;
