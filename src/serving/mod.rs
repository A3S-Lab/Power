//! Model-neutral distributed-serving contracts.
//!
//! These types describe bounded execution capabilities and observations. They
//! do not choose endpoints, create replicas, or define model-owned KV layouts.

mod bounded_state_transfer;
mod distributed_serving;
mod execution_profile;
mod observation;
mod phase_executor;
mod state_transfer;
mod state_transfer_service;

pub use bounded_state_transfer::{BoundedStateTransferService, StateTransferRuntimeSnapshot};
pub use distributed_serving::{
    DecodePhaseRequest, DistributedServingRuntime, PrefillPhaseRequest, PreparedDecodeTransfer,
    PublishedPrefillState,
};
pub use execution_profile::{
    DisaggregatedServingRole, PrefillDecodeExecutionProfile, ServingExecutionProfile,
    ServingPrivacyMode,
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

#[cfg(test)]
mod bounded_state_transfer_tests;
#[cfg(test)]
mod distributed_serving_tests;
#[cfg(test)]
mod execution_profile_tests;
#[cfg(test)]
mod phase_executor_tests;
#[cfg(test)]
mod state_transfer_tests;
