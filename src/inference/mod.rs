//! Embedded, library-level model execution.
//!
//! This module is deliberately independent from [`crate::server`] and the
//! OpenAI-compatible backend trait. Constructing an embedded session never
//! binds a socket, starts a listener, downloads a model, or invokes another
//! process.

mod device;
pub mod graph;
mod limits;
mod receipt;
mod residency;
mod routing;
mod runtime;
mod telemetry;
mod tensor;
mod weights;

pub use device::{DevicePreference, RuntimeDevice, RuntimeDeviceKind};
pub use limits::InferenceLimits;
pub use receipt::{
    ExecutionDigest, ExecutionReceipt, ExecutionRepresentation, ModelIdentity, RuntimeIdentity,
};
pub use residency::{
    PlacementPreference, PlannedResidencyGroup, PrefetchReport, PrefetchTask, ResidencyApplyReport,
    ResidencyCandidate, ResidencyPlan, ResidencyPolicy, ResidentWeight, WeightHierarchy, WeightKey,
    WeightRequest, WeightTier,
};
pub use routing::{ExpertAssignment, ExpertKey, RoutedExpert, RoutedExpertBatch};
pub use runtime::{EmbeddedRuntime, ExecutionPermit};
pub use telemetry::{PlacementTelemetry, RouteHeat, RoutingHistory, TelemetryMode};
pub use tensor::{TensorInput, TensorOutput};
pub use weights::{TensorDescriptor, WeightFileDescriptor, WeightStore};

pub(crate) const RUNTIME_NAME: &str = "a3s-power-native";
