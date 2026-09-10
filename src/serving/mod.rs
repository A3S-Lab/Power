//! Model-neutral distributed-serving contracts.
//!
//! These types describe bounded execution capabilities and observations. They
//! do not choose endpoints, create replicas, or define model-owned KV layouts.

mod adapter_contract;
mod backend_owned_phase;
mod backend_phase_execution;
mod backend_phase_state_ownership;
mod bounded_state_transfer;
mod buffered_host_loopback;
mod buffered_host_loopback_phase;
mod direct_device_memory_pull;
mod distributed_operation_evidence;
mod distributed_serving;
mod empty_adapters;
mod execution_profile;
mod llamacpp_phase_execution;
mod llamacpp_phase_state_ownership;
mod observation;
mod phase_executor;
mod state_transfer;
mod state_transfer_service;
#[cfg(feature = "embedded-inference")]
mod transfer_host_buffer;

pub use adapter_contract::{
    AdapterCleanupObligation, AdapterMemoryOwnership, AdapterProvisionState,
    AdapterTransportIntegrity, ProductionAdapterContract,
};
pub use backend_owned_phase::BackendOwnedPhaseExecutor;
pub use backend_phase_execution::{
    bind_backend_phase_execution, BackendPhaseExecution, EmptyBackendPhaseExecution,
    PendingBackendPhaseExecution,
};
pub use backend_phase_state_ownership::{
    bind_backend_phase_state_ownership, BackendPhaseStateOwnership,
    EmptyBackendPhaseStateOwnership, ProfileBoundBackendPhaseStateOwnership,
};
pub use bounded_state_transfer::{BoundedStateTransferService, StateTransferRuntimeSnapshot};
pub use buffered_host_loopback::BufferedHostLoopbackStateTransfer;
pub use buffered_host_loopback_phase::BufferedHostLoopbackPhaseExecutor;
pub use direct_device_memory_pull::{
    DirectDeviceMemoryPullPhaseExecutor, DirectDeviceMemoryPullStateTransfer,
};
pub use distributed_operation_evidence::{
    DistributedOperationEvidence, DistributedOperationKind, DISTRIBUTED_OPERATION_EVIDENCE_SCHEMA,
};
pub use distributed_serving::{
    DecodePhaseRequest, DistributedServingRuntime, PrefillPhaseRequest, PreparedDecodeTransfer,
    PublishedPrefillState,
};
pub use empty_adapters::{
    validate_injected_production_adapters, EmptyServingPhaseExecutor, EmptyStateTransferService,
};
pub use execution_profile::{
    DisaggregatedServingRole, PhaseSessionPoolMode, PhaseWeightCacheMode,
    PrefillDecodeExecutionProfile, ServingCompositionPhaseExecution,
    ServingCompositionPhaseExecutor, ServingCompositionStateOwnership, ServingCompositionTransport,
    ServingDeploymentIdentity, ServingExecutionProfile, ServingPrivacyMode,
};
pub use llamacpp_phase_execution::LlamaCppBackendPhaseExecution;
#[cfg(feature = "llamacpp")]
pub use llamacpp_phase_state_ownership::LlamaCppContextStateApi;
pub use llamacpp_phase_state_ownership::{
    probe_llamacpp_state_transfer_api, FixtureLlamaCppContextStatePort,
    LlamaCppBackendPhaseStateOwnership, LlamaCppContextStatePort, LlamaCppLayoutFacts,
    LlamaCppStateTransferApiProbe, LLAMACPP_STATE_LAYOUT_DOMAIN,
    LLAMACPP_STATE_TRANSFER_API_SYMBOLS,
};
pub use observation::{
    AdmissionObservation, PromptCacheObservation, ServingPhase, TransferHealth, WorkerCapabilities,
    WorkerObservation, WORKER_OBSERVATION_SCHEMA,
};
pub use phase_executor::{
    AbortPhaseExecution, ExecutePhaseExecution, ImportedModelState, PhaseDecision,
    PhaseExecutionHandle, PhaseExecutionOutput, PhaseExecutorCapabilities, PhaseExecutorHealth,
    PhaseRequest, PhaseResponseChunk, PhaseResponseStream, PreparePhaseExecution,
    PreparedDecodePhase, PreparedPhaseExecution, PreparedPrefillPhase, ProducedModelState,
    RecomputeReason, RetryableUnavailableReason, ServingPhaseExecutor, TerminalFailureReason,
};
pub use state_transfer::{
    StateKind, StateTransferBinding, StateTransferCapabilities, StateTransferIntegrity,
    StateTransferProtocol, StateTransferReceipt, StateTransferSource, StateTransferTarget,
    STATE_TRANSFER_RECEIPT_SCHEMA, STATE_TRANSFER_SOURCE_SCHEMA, STATE_TRANSFER_TARGET_SCHEMA,
};
pub use state_transfer_service::{
    AbortStateTransfer, ConsumeStateTransfer, ModelStateHandle, PrepareStateTransfer,
    PublishStateTransfer, StateTransferService,
};
#[cfg(feature = "embedded-inference")]
pub use transfer_host_buffer::{
    open_transfer_host_buffer, seal_transfer_host_buffer, sealed_binding_for_transfer_host_buffer,
    TransferHostBufferIdentity,
};

#[cfg(test)]
mod bounded_state_transfer_tests;
#[cfg(test)]
pub(crate) mod distributed_serving_tests;
#[cfg(test)]
mod execution_profile_tests;
#[cfg(test)]
mod phase_executor_tests;
#[cfg(test)]
mod state_transfer_tests;
