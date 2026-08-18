//! Qwen3.5-family native execution components.
//!
//! The typed layout is built independently from the eventual CPU/CUDA
//! executor so malformed or incompatible artifacts fail before weight access.
//! Qwen3.8 models intentionally use the same `qwen35` GGUF architecture.

mod attention;
mod layout;
mod model_state;
mod preprocess;
mod reference;
mod state;

pub use attention::{Qwen35AttentionConfig, Qwen35AttentionState};
pub use layout::Qwen35TensorLayout;
pub use model_state::{Qwen35LayerState, Qwen35ModelState};
pub use preprocess::{
    qwen35_prepare_attention, Qwen35AttentionProjectionInputs, Qwen35AttentionProjectionOutputs,
    Qwen35AttentionProjectionWeights, Qwen35MropeConfig,
};
pub use reference::{qwen35_recurrent_step, Qwen35RecurrentInputs, Qwen35RecurrentWeights};
pub use state::{CausalConv1dState, Qwen35RecurrentConfig, Qwen35RecurrentState};

#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod tests;
