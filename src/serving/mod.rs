//! Model-neutral distributed-serving contracts.
//!
//! These types describe bounded execution capabilities and observations. They
//! do not choose endpoints, create replicas, or define model-owned KV layouts.

mod observation;

pub use observation::{
    AdmissionObservation, PromptCacheObservation, ServingPhase, TransferHealth, WorkerCapabilities,
    WorkerObservation, WORKER_OBSERVATION_SCHEMA,
};
