//! Model-neutral distributed-serving contracts.
//!
//! These types describe bounded execution capabilities and observations. They
//! do not choose endpoints, create replicas, or define model-owned KV layouts.

mod observation;
mod state_transfer;
mod state_transfer_service;

pub use observation::{
    AdmissionObservation, PromptCacheObservation, ServingPhase, TransferHealth, WorkerCapabilities,
    WorkerObservation, WORKER_OBSERVATION_SCHEMA,
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
mod state_transfer_tests;
