//! Embedded, library-level model execution.
//!
//! This module is deliberately independent from [`crate::server`] and the
//! OpenAI-compatible backend trait. Constructing an embedded session never
//! binds a socket, starts a listener, downloads a model, or invokes another
//! process.

mod device;
mod limits;
mod receipt;
mod tensor;
mod weights;

#[cfg(feature = "ppocr-v6")]
pub mod ppocr_v6;

pub use device::{DevicePreference, RuntimeDevice, RuntimeDeviceKind};
pub use limits::InferenceLimits;
pub use receipt::{ExecutionReceipt, ModelIdentity, RuntimeIdentity};
pub use tensor::{TensorInput, TensorOutput};
pub use weights::{TensorDescriptor, WeightStore};

pub(crate) const RUNTIME_NAME: &str = "a3s-power-native";
