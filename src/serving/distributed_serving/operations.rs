use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::{
    ExecutePhaseExecution, PhaseDecision, PhaseExecutionOutput, PreparePhaseExecution,
    PreparedPhaseExecution,
};
use super::lifecycle::{wait_operation, RuntimeOperation, RuntimeOperationGuard};
use super::DistributedServingRuntime;

impl DistributedServingRuntime {
    pub(super) async fn prepare_phase(
        &self,
        command: PreparePhaseExecution,
        cancellation: CancellationToken,
        deadline: Instant,
    ) -> Result<PhaseDecision<PreparedPhaseExecution>> {
        match wait_operation(cancellation, deadline, self.inner.executor.prepare(command)).await {
            RuntimeOperation::Completed(result) => result,
            RuntimeOperation::Cancelled => Err(PowerError::BackendNotAvailable(
                "distributed phase preparation was cancelled".to_string(),
            )),
            RuntimeOperation::TimedOut => Err(PowerError::BackendNotAvailable(
                "distributed phase preparation timed out".to_string(),
            )),
        }
    }

    pub(super) async fn execute_phase(
        &self,
        command: ExecutePhaseExecution,
        cancellation: CancellationToken,
        deadline: Instant,
    ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
        match wait_operation(cancellation, deadline, self.inner.executor.execute(command)).await {
            RuntimeOperation::Completed(result) => result,
            RuntimeOperation::Cancelled => Err(PowerError::BackendNotAvailable(
                "distributed phase execution was cancelled".to_string(),
            )),
            RuntimeOperation::TimedOut => Err(PowerError::BackendNotAvailable(
                "distributed phase execution timed out".to_string(),
            )),
        }
    }
}

pub(super) enum ReadyOrDecision<T, U> {
    Ready(T),
    Decision(PhaseDecision<U>),
}

pub(super) async fn ready_or_cleanup<T, U>(
    decision: PhaseDecision<T>,
    guard: &mut RuntimeOperationGuard,
) -> Result<ReadyOrDecision<T, U>> {
    decision.validate()?;
    match decision {
        PhaseDecision::Ready(value) => Ok(ReadyOrDecision::Ready(value)),
        PhaseDecision::Recompute { reason } => {
            require_cleanup(guard).await?;
            Ok(ReadyOrDecision::Decision(PhaseDecision::Recompute {
                reason,
            }))
        }
        PhaseDecision::RetryableUnavailable {
            reason,
            retry_after_ms,
        } => {
            require_cleanup(guard).await?;
            Ok(ReadyOrDecision::Decision(
                PhaseDecision::RetryableUnavailable {
                    reason,
                    retry_after_ms,
                },
            ))
        }
        PhaseDecision::TerminalFailure { reason } => {
            require_cleanup(guard).await?;
            Ok(ReadyOrDecision::Decision(PhaseDecision::TerminalFailure {
                reason,
            }))
        }
    }
}

async fn require_cleanup(guard: &mut RuntimeOperationGuard) -> Result<()> {
    if guard.cleanup().await {
        Ok(())
    } else {
        Err(PowerError::BackendNotAvailable(
            "distributed phase decision cleanup was not confirmed".to_string(),
        ))
    }
}
