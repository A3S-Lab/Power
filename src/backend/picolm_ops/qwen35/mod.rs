//! Qwen3.5 native execution components.
//!
//! The typed layout is built independently from the eventual CPU/CUDA
//! executor so malformed or incompatible artifacts fail before weight access.

mod layout;

pub use layout::Qwen35TensorLayout;

#[cfg(test)]
mod tests;
