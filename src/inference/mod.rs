//! Embedded, library-level model execution.
//!
//! This module is deliberately independent from `crate::server` and the
//! OpenAI-compatible backend trait. Constructing an embedded session never
//! binds a socket, starts a listener, downloads a model, or invokes another
//! process.

mod accelerator;
#[cfg(test)]
mod accelerator_tests;
mod coupling;
#[cfg(test)]
mod coupling_tests;
mod coupling_types;
mod device;
mod filesystem;
pub mod graph;
mod hardware;
mod limits;
mod mirror;
mod receipt;
mod residency;
mod routing;
mod runtime;
mod sealed_state;
#[cfg(test)]
mod sealed_state_tests;
mod storage_benchmark;
mod telemetry;
mod tensor;
mod tuning;
#[cfg(test)]
mod tuning_tests;
mod tuning_types;
mod weights;

pub use accelerator::{
    AcceleratorBatchResolution, AcceleratorExecutionCompletion, AcceleratorExecutionEvidence,
    AcceleratorExecutionPath, AcceleratorFallback, AcceleratorFallbackMode,
    AcceleratorFallbackReason, AcceleratorFallbackTarget, AcceleratorFusedBatch,
    AcceleratorFusedBatchOutput, AcceleratorFusedBatchSpec, AcceleratorFusedExecution,
    AcceleratorFusedGroup, AcceleratorKernelOutcome, AcceleratorResidencyDeclaration,
    AcceleratorResidencyGroup, AcceleratorSecurityRequirement, ConfidentialGpuBinding,
};
pub use coupling::{
    RouteCouplingEntry, RouteCouplingHistory, RouteCouplingPolicy, RouteHintEvaluation,
    RouteHintTelemetry, RouteLayerGeometry, RoutePrefetchHint, RoutePrefetchHints,
};
pub use device::{DevicePreference, RuntimeDevice, RuntimeDeviceIdentity, RuntimeDeviceKind};
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
pub use sealed_state::{
    OpenedSealedState, RecoveredSealedState, SealedStateBinding, SealedStateEnvelope,
    SealedStateExportScope, SealedStateKey, SealedStateRecoverySource, SealedStateRollbackPolicy,
    SealedStateScope, SealedStateStore, TeeStateExportAuthorization,
};
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
pub use tuning::evaluate_tuning_profile;
pub use tuning_types::{
    TuningCandidateEvidence, TuningCandidateSummary, TuningOrderedEvidence, TuningProfileBinding,
    TuningProfileDecision, TuningProfileEvidence, TuningProfileOutcome, TuningProfilePolicy,
    TuningRoundEvidence, TuningRunEvidence,
};
pub use weights::{
    weight_collection_sha256, LosslessEncodedRecord, LosslessRansNibbleHistogram,
    LosslessRansNibbleTable, TensorDescriptor, TensorRead, TensorStorageDescriptor,
    WeightFileDescriptor, WeightReadStrategy, WeightSourceConfig, WeightSourceCoverage,
    WeightSourceDescriptor, WeightSourceRepresentation, WeightSourceRole, WeightSourceWeighting,
    WeightStore, WeightStoreConfig, LOSSLESS_RANS_FORMAT_METADATA_KEY,
    LOSSLESS_RANS_TABLE_METADATA_KEY,
};

pub(crate) const RUNTIME_NAME: &str = "a3s-power-native";
