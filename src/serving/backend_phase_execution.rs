//! Product-surface backend phase prepare/execute adapter.
//!
//! Concrete backends (llama.cpp, picolm) implement [`BackendPhaseExecution`]
//! later. Power advances [`PhaseExecutorHealth::Ready`] only when ownership is
//! already [`PhaseExecutorHealth::Eligible`] **and** a non-Empty execution
//! adapter reports [`BackendPhaseExecution::can_produce_ready`]. Default
//! [`EmptyBackendPhaseExecution`] keeps Eligible ownership from becoming Ready.
//!
//! [`PendingBackendPhaseExecution`] is an honest interim composition surface
//! (`phase_execution = pending`): it can unlock Ready health so the execute
//! path is bound, but prepare/execute/abort fail closed until a concrete
//! backend implementor exists. This does **not** invent KV layout or claim
//! model-semantic P/D. Worker advertising remains suppressed by the
//! backend-owned `may_advertise_prefill_decode` gate.

use async_trait::async_trait;

use crate::error::{PowerError, Result};

use super::{
    AbortPhaseExecution, ExecutePhaseExecution, PhaseDecision, PhaseExecutionOutput,
    PhaseExecutorHealth, PreparePhaseExecution, PreparedPhaseExecution, RetryableUnavailableReason,
};

/// Prepare/execute/abort hooks for backend-owned phase work.
///
/// Implementors own tokenization, reservations, and response generation.
/// Power never invents KV semantics here; successful Ready decisions are
/// adapter-owned opaque outcomes only.
#[async_trait]
pub trait BackendPhaseExecution: Send + Sync {
    /// Empty default surface. Never unlocks [`PhaseExecutorHealth::Ready`].
    fn is_empty(&self) -> bool {
        false
    }

    /// Whether Eligible ownership may advance to Ready health.
    ///
    /// Empty returns false. Pending and real backends return true so the
    /// executor can delegate prepare/execute instead of refusing at Eligible.
    fn can_produce_ready(&self) -> bool;

    async fn prepare(
        &self,
        command: PreparePhaseExecution,
    ) -> Result<PhaseDecision<PreparedPhaseExecution>>;

    async fn execute(
        &self,
        command: ExecutePhaseExecution,
    ) -> Result<PhaseDecision<PhaseExecutionOutput>>;

    async fn abort(&self, command: AbortPhaseExecution) -> Result<()>;
}

/// Default Empty execution. Keeps Eligible ownership from becoming Ready.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmptyBackendPhaseExecution;

#[async_trait]
impl BackendPhaseExecution for EmptyBackendPhaseExecution {
    fn is_empty(&self) -> bool {
        true
    }

    fn can_produce_ready(&self) -> bool {
        false
    }

    async fn prepare(
        &self,
        _command: PreparePhaseExecution,
    ) -> Result<PhaseDecision<PreparedPhaseExecution>> {
        PhaseDecision::retryable_unavailable(RetryableUnavailableReason::ExecutorUnavailable, None)
    }

    async fn execute(
        &self,
        _command: ExecutePhaseExecution,
    ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
        PhaseDecision::retryable_unavailable(RetryableUnavailableReason::ExecutorUnavailable, None)
    }

    async fn abort(&self, _command: AbortPhaseExecution) -> Result<()> {
        Err(PowerError::BackendNotAvailable(
            "EmptyBackendPhaseExecution refuses abort until a Ready-capable execute adapter is bound"
                .to_string(),
        ))
    }
}

/// Interim Ready-unlock surface for backend-owned phase composition.
///
/// ACL opt-in `phase_execution = pending` (with `phase_executor = backend-owned`)
/// binds this adapter so Eligible ownership can advance to Ready health.
/// prepare/execute/abort fail closed: this is **not** a llama.cpp / picolm
/// executor and does not claim model-semantic decode success.
#[derive(Debug, Default, Clone, Copy)]
pub struct PendingBackendPhaseExecution;

#[async_trait]
impl BackendPhaseExecution for PendingBackendPhaseExecution {
    fn can_produce_ready(&self) -> bool {
        true
    }

    async fn prepare(
        &self,
        _command: PreparePhaseExecution,
    ) -> Result<PhaseDecision<PreparedPhaseExecution>> {
        Err(PowerError::BackendNotAvailable(
            "PendingBackendPhaseExecution refuses prepare: phase_execution = pending unlocks Ready health only; a concrete backend execute adapter is still required"
                .to_string(),
        ))
    }

    async fn execute(
        &self,
        _command: ExecutePhaseExecution,
    ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
        Err(PowerError::BackendNotAvailable(
            "PendingBackendPhaseExecution refuses execute: phase_execution = pending unlocks Ready health only; a concrete backend execute adapter is still required"
                .to_string(),
        ))
    }

    async fn abort(&self, _command: AbortPhaseExecution) -> Result<()> {
        Err(PowerError::BackendNotAvailable(
            "PendingBackendPhaseExecution refuses abort: phase_execution = pending unlocks Ready health only; a concrete backend execute adapter is still required"
                .to_string(),
        ))
    }
}

/// Combine ownership bind health with an execution adapter.
///
/// - Ownership [`Unavailable`] → [`Unavailable`] (execution cannot skip layout).
/// - Ownership [`Eligible`] + `can_produce_ready` → [`Ready`].
/// - Ownership [`Eligible`] without Ready-capable execution → stays [`Eligible`].
pub fn bind_backend_phase_execution(
    ownership_health: PhaseExecutorHealth,
    execution: &dyn BackendPhaseExecution,
) -> PhaseExecutorHealth {
    match ownership_health {
        PhaseExecutorHealth::Unavailable => PhaseExecutorHealth::Unavailable,
        PhaseExecutorHealth::Eligible if execution.can_produce_ready() => {
            PhaseExecutorHealth::Ready
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_never_unlocks_ready() {
        let empty = EmptyBackendPhaseExecution;
        assert!(empty.is_empty());
        assert!(!empty.can_produce_ready());
        assert_eq!(
            bind_backend_phase_execution(PhaseExecutorHealth::Eligible, &empty),
            PhaseExecutorHealth::Eligible
        );
        assert_eq!(
            bind_backend_phase_execution(PhaseExecutorHealth::Unavailable, &empty),
            PhaseExecutorHealth::Unavailable
        );
    }

    #[test]
    fn pending_unlocks_ready_only_after_eligible() {
        let pending = PendingBackendPhaseExecution;
        assert!(!pending.is_empty());
        assert!(pending.can_produce_ready());
        assert_eq!(
            bind_backend_phase_execution(PhaseExecutorHealth::Eligible, &pending),
            PhaseExecutorHealth::Ready
        );
        assert!(
            bind_backend_phase_execution(PhaseExecutorHealth::Eligible, &pending).accepts_work()
        );
        assert_eq!(
            bind_backend_phase_execution(PhaseExecutorHealth::Unavailable, &pending),
            PhaseExecutorHealth::Unavailable
        );
        assert!(
            !bind_backend_phase_execution(PhaseExecutorHealth::Unavailable, &pending)
                .accepts_work()
        );
    }

    #[tokio::test]
    async fn pending_prepare_execute_abort_fail_closed() {
        let pending = PendingBackendPhaseExecution;
        let err = match pending
            .prepare(PreparePhaseExecution {
                execution_id: uuid::Uuid::new_v4(),
                local_worker_epoch: uuid::Uuid::new_v4(),
                model: "internal/model-v1".to_string(),
                expires_at: chrono::Utc::now() + chrono::Duration::seconds(30),
                request: crate::serving::PhaseRequest::Completion(
                    serde_json::from_value(serde_json::json!({ "prompt": "unused" })).unwrap(),
                ),
            })
            .await
        {
            Ok(_) => panic!("pending prepare must fail closed"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("PendingBackendPhaseExecution"));
        assert!(err.to_string().contains("unlocks Ready health only"));

        let err = match pending
            .abort(AbortPhaseExecution {
                execution_id: uuid::Uuid::new_v4(),
                local_worker_epoch: uuid::Uuid::new_v4(),
                execution: None,
            })
            .await
        {
            Ok(()) => panic!("pending abort must fail closed"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("PendingBackendPhaseExecution"));
    }
}
