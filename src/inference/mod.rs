//! Embedded, library-level model execution.
//!
//! This module is deliberately independent from `crate::server` and the
//! OpenAI-compatible backend trait. Constructing an embedded session never
//! binds a socket, starts a listener, downloads a model, or invokes another
//! process.

mod accelerator;
#[cfg(test)]
mod accelerator_mesh_failure_tests;
#[cfg(test)]
mod accelerator_mesh_tests;
#[cfg(test)]
mod accelerator_tests;
mod coupling;
#[cfg(test)]
mod coupling_tests;
mod coupling_types;
mod device;
mod execution_batch;
#[cfg(test)]
mod execution_batch_failure_tests;
#[cfg(test)]
mod execution_batch_tests;
mod execution_cost;
mod filesystem;
pub mod graph;
mod hardware;
mod hardware_evidence;
#[cfg(test)]
mod hardware_evidence_failure_tests;
#[cfg(test)]
mod hardware_evidence_tests;
mod limits;
mod microbatch;
#[cfg(test)]
mod microbatch_tests;
mod mirror;
mod receipt;
mod release_gate;
#[cfg(test)]
mod release_gate_tests;
mod residency;
mod routing;
mod runtime;
mod sealed_state;
#[cfg(test)]
mod sealed_state_tests;
#[cfg(test)]
mod session_deadline_tests;
mod session_pool;
#[cfg(test)]
mod session_pool_tests;
#[cfg(test)]
mod session_replica_health_tests;
#[cfg(test)]
mod session_replica_tests;
mod shape_profile;
#[cfg(test)]
mod shape_profile_tests;
mod storage_benchmark;
mod telemetry;
mod tensor;
mod tuning;
#[cfg(test)]
mod tuning_tests;
mod tuning_types;
mod weights;

pub use accelerator::{
    AcceleratorBatchResolution, AcceleratorDeviceMesh, AcceleratorDeviceMeshDeclaration,
    AcceleratorExecutionCompletion, AcceleratorExecutionEvidence, AcceleratorExecutionPath,
    AcceleratorFallback, AcceleratorFallbackMode, AcceleratorFallbackReason,
    AcceleratorFallbackTarget, AcceleratorFusedBatch, AcceleratorFusedBatchOutput,
    AcceleratorFusedBatchSpec, AcceleratorFusedExecution, AcceleratorFusedGroup,
    AcceleratorKernelOutcome, AcceleratorMeshDevice, AcceleratorMeshDeviceDeclaration,
    AcceleratorMeshExecution, AcceleratorPeerTransferDeclaration, AcceleratorPeerTransferOutcome,
    AcceleratorPeerTransferSpec, AcceleratorResidencyDeclaration, AcceleratorResidencyGroup,
    AcceleratorSecurityRequirement, ConfidentialGpuBinding,
};
pub use coupling::{
    RouteCouplingEntry, RouteCouplingHistory, RouteCouplingPolicy, RouteHintEvaluation,
    RouteHintTelemetry, RouteLayerGeometry, RoutePrefetchHint, RoutePrefetchHints,
};
pub use device::{DevicePreference, RuntimeDevice, RuntimeDeviceIdentity, RuntimeDeviceKind};
pub use execution_batch::{
    ExecutionBatchBinding, ExecutionBatchLifecycle, ExecutionBatchLifecycleEvidence,
    ExecutionBatchMemberBinding, ExecutionBatchMemberSnapshot, ExecutionBatchMemberSpec,
    ExecutionBatchRow, ExecutionBatchRowDisposition, ExecutionBatchRowOutcome,
    ExecutionBatchRowSpec, ExecutionBatchStep, ExecutionBatchStepEvidence,
};
pub use execution_cost::{
    run_tensor_batch_benchmark, GraphExecutionBoundaryMeasurement, HostAllocationCounter,
    HostAllocationMeasurement, HostAllocationSnapshot, TensorBatchBenchmarkConfig,
    TensorBatchBenchmarkMode, TensorBatchBenchmarkOrder, TensorBatchBenchmarkReport,
    TensorBatchBenchmarkSample, TensorBatchBenchmarkSummary,
};
pub use hardware::{
    HardwareMemorySnapshot, MemoryDiscoverySource, MemoryPoolSnapshot, ResidencyAllocationOrder,
    ResidencyBudgetPlan, ResidencyBudgetPolicy, RuntimeMemoryReservations,
};
pub use hardware_evidence::{HardwareEvidenceBinding, HardwareEvidenceBundle, ModelParityArtifact};
pub use limits::InferenceLimits;
pub use microbatch::{
    MicrobatchCandidate, MicrobatchExecution, MicrobatchLimits, MicrobatchPlan, MicrobatchPolicy,
    PlannedMicrobatch, PlannedMicrobatchSlot,
};
pub use mirror::{
    WeightMirrorCandidate, WeightMirrorConfidentiality, WeightMirrorPlan,
    WeightMirrorPlanRejection, WeightMirrorPlannedFile, WeightMirrorPolicy,
    WeightMirrorStageReport,
};
pub use receipt::{
    ExecutionDigest, ExecutionReceipt, ExecutionRepresentation, MicrobatchExecutionEvidence,
    ModelIdentity, RuntimeIdentity,
};
pub use release_gate::{
    BoundedMemoryEvidence, CancellationContractEvidence, ConfidentialReleaseBinding,
    ExactFallbackEvidence, PeakMemoryEvidence, PeakMemoryMethod, QueueExpiryEvidence,
    ReleaseCapture, ReleaseCaptureSecurity, ReleaseContractEvidence, ReleaseEvidenceBundle,
    ReleaseEvidencePolicy, ReleasePlatform, ReleasePlatformBinding, ReleaseRevisionBinding,
    ReplicaRecoveryEvidence,
};
pub use residency::{
    CacheEvictionPolicy, PlacementPreference, PlannedResidencyGroup, PrefetchReport, PrefetchTask,
    ResidencyAdaptation, ResidencyAdaptationPolicy, ResidencyApplyReport, ResidencyCandidate,
    ResidencyPlan, ResidencyPolicy, ResidencyReplacement, ResidentWeight, StagedWeightBatch,
    StagedWeightBatchCompletion, StagedWeightBatchReport, StagedWeightGroup,
    StagedWeightGroupRequest, WeightHierarchy, WeightKey, WeightRequest, WeightTier,
};
pub use routing::{ExpertAssignment, ExpertKey, RoutedExpert, RoutedExpertBatch};
pub use runtime::{EmbeddedRuntime, ExecutionPermit, ResidentTensorSnapshot};
pub use sealed_state::{
    OpenedSealedState, RecoveredSealedState, SealedStateBinding, SealedStateEnvelope,
    SealedStateExportScope, SealedStateKey, SealedStateRecoverySource, SealedStateRollbackPolicy,
    SealedStateScope, SealedStateStore, TeeStateExportAuthorization,
};
pub use session_pool::{
    ModelSession, ModelSessionBinding, ModelSessionPool, ModelSessionPoolPolicy,
    ModelSessionPoolSnapshot, ModelSessionReplica, ModelSessionSpec,
};
pub use shape_profile::{
    DynamicShapeFallback, ShapeProfile, ShapeProfileBinding, ShapeProfileDeclaration,
    ShapeProfileExecutionEvidence, ShapeProfileExecutionPath, ShapeProfileFallbackReason,
    ShapeProfileRequest, ShapeProfileSelection,
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
pub use weights::{encrypt_seekable_weight_collection, encrypt_seekable_weight_file};
pub use weights::{
    weight_collection_sha256, LosslessEncodedRecord, LosslessRansNibbleHistogram,
    LosslessRansNibbleTable, SeekableEncryptedFile, SeekableEncryptedFileDescriptor,
    SeekableEncryptedFileVerification, SeekableEncryptedWeightCollectionReport,
    SeekableEncryptedWeightFile, SeekableEncryptedWeightManifest, SeekableEncryptedWeightSource,
    SeekableWeightKey, TensorDescriptor, TensorRangeRead, TensorRead, TensorStorageDescriptor,
    WeightFileDescriptor, WeightReadStrategy, WeightSourceConfig, WeightSourceCoverage,
    WeightSourceDescriptor, WeightSourceRepresentation, WeightSourceRole, WeightSourceWeighting,
    WeightStore, WeightStoreConfig, DEFAULT_ENCRYPTED_CHUNK_BYTES, ENCRYPTED_WEIGHT_MANIFEST_FILE,
    ENCRYPTED_WEIGHT_MANIFEST_SCHEMA, LOSSLESS_RANS_FORMAT_METADATA_KEY,
    LOSSLESS_RANS_TABLE_METADATA_KEY,
};

pub(crate) const RUNTIME_NAME: &str = "a3s-power-native";
