//! picolm inference operations — pure Rust transformer math for layer-streaming.
//!
//! Each submodule is <400 lines and independently testable.
//! Zero C dependencies — fully auditable for TEE supply-chain review.

#[cfg(feature = "picolm")]
pub mod attention;
#[cfg(feature = "picolm")]
pub mod buffers;
#[cfg(feature = "picolm")]
pub mod dequant;
#[cfg(feature = "picolm")]
pub mod ffn;
#[cfg(feature = "picolm")]
pub mod kv_cache;
#[cfg(feature = "picolm")]
pub mod matmul;
#[cfg(feature = "picolm")]
pub mod norm;
#[cfg(feature = "picolm")]
pub mod rope;
#[cfg(feature = "picolm")]
pub mod tensor_cache;
#[cfg(feature = "picolm")]
pub mod tokenizer;
#[cfg(feature = "picolm")]
pub mod vec_dot;
