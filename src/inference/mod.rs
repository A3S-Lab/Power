//! Embedded, library-level model execution.
//!
//! This module is deliberately independent from `crate::server` and the
//! OpenAI-compatible backend trait. Constructing an embedded session never
//! binds a socket, starts a listener, downloads a model, or invokes another
//! process.

mod coupling;
#[cfg(test)]
mod coupling_tests;
mod coupling_types;
mod device;
pub mod graph;
mod hardware;
mod limits;
mod mirror;
mod receipt;
mod residency;
mod routing;
mod runtime;
mod storage_benchmark;
mod telemetry;
mod tensor;
mod weights;

pub use coupling::{
    RouteCouplingEntry, RouteCouplingHistory, RouteCouplingPolicy, RouteHintEvaluation,
    RouteHintTelemetry, RouteLayerGeometry, RoutePrefetchHint, RoutePrefetchHints,
};
pub use device::{DevicePreference, RuntimeDevice, RuntimeDeviceKind};
pub use hardware::{
    HardwareMemorySnapshot, MemoryDiscoverySource, MemoryPoolSnapshot, ResidencyAllocationOrder,
    ResidencyBudgetPlan, ResidencyBudgetPolicy,
};
pub use limits::InferenceLimits;
pub use mirror::{
    WeightMirrorCandidate, WeightMirrorConfidentiality, WeightMirrorPlan,
    WeightMirrorPlanRejection, WeightMirrorPlannedFile, WeightMirrorPolicy,
    WeightMirrorStageReport,
};
pub use receipt::{
    ExecutionDigest, ExecutionReceipt, ExecutionRepresentation, ModelIdentity, RuntimeIdentity,
};
pub use residency::{
    CacheEvictionPolicy, PlacementPreference, PlannedResidencyGroup, PrefetchReport, PrefetchTask,
    ResidencyAdaptation, ResidencyAdaptationPolicy, ResidencyApplyReport, ResidencyCandidate,
    ResidencyPlan, ResidencyPolicy, ResidencyReplacement, ResidentWeight, StagedWeightBatch,
    StagedWeightBatchCompletion, StagedWeightBatchReport, StagedWeightGroup,
    StagedWeightGroupRequest, WeightHierarchy, WeightKey, WeightRequest, WeightTier,
};
pub use routing::{ExpertAssignment, ExpertKey, RoutedExpert, RoutedExpertBatch};
pub use runtime::{EmbeddedRuntime, ExecutionPermit};
pub use storage_benchmark::{
    compare_storage_benchmarks, run_storage_benchmark, StorageBenchmarkComparison,
    StorageBenchmarkConfig, StorageBenchmarkGroup, StorageBenchmarkReport, StorageBenchmarkSample,
    StorageBenchmarkSource, StorageBenchmarkSourceSummary, StorageBenchmarkSystem,
    StorageCachePreparation, StorageCacheState, StorageDistributionSummary,
};
pub use telemetry::{
    PlacementTelemetry, RouteHeat, RoutingHistory, StorageSourceTelemetry, TelemetryMode,
};
pub use tensor::{TensorInput, TensorOutput};
pub use weights::{
    TensorDescriptor, TensorRead, TensorStorageDescriptor, WeightFileDescriptor,
    WeightReadStrategy, WeightSourceConfig, WeightSourceCoverage, WeightSourceDescriptor,
    WeightSourceRole, WeightSourceWeighting, WeightStore, WeightStoreConfig,
};

pub(crate) const RUNTIME_NAME: &str = "a3s-power-native";
