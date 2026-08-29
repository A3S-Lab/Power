use std::sync::Arc;

use a3s_power::backend::types::CompletionResponseChunk;
use a3s_power::error::{PowerError, Result};
use a3s_power::serving::{
    AbortPhaseExecution, ExecutePhaseExecution, ModelStateHandle, PhaseDecision,
    PhaseExecutionHandle, PhaseExecutionOutput, PhaseExecutorCapabilities, PhaseExecutorHealth,
    PhaseResponseChunk, PreparePhaseExecution, PreparedDecodePhase, PreparedPhaseExecution,
    PreparedPrefillPhase, ProducedModelState, ServingPhase, ServingPhaseExecutor,
};
use async_trait::async_trait;

use super::{binding, fixture_state, FixtureStateStore};

pub(super) struct FixturePhaseExecutor {
    pub(super) capabilities: PhaseExecutorCapabilities,
    pub(super) profile_sha256: String,
    pub(super) store: Arc<FixtureStateStore>,
}

#[async_trait]
impl ServingPhaseExecutor for FixturePhaseExecutor {
    fn capabilities(&self) -> PhaseExecutorCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> PhaseExecutorHealth {
        PhaseExecutorHealth::Ready
    }

    async fn prepare(
        &self,
        command: PreparePhaseExecution,
    ) -> Result<PhaseDecision<PreparedPhaseExecution>> {
        let execution = PhaseExecutionHandle::new(format!("phase:{}", command.execution_id))?;
        let prepared = match self.capabilities.phase {
            ServingPhase::Prefill => PreparedPhaseExecution::Prefill(PreparedPrefillPhase::new(
                command.execution_id,
                command.local_worker_epoch,
                self.profile_sha256.clone(),
                execution,
                command.expires_at,
            )?),
            ServingPhase::Decode => PreparedPhaseExecution::Decode(PreparedDecodePhase::new(
                command.execution_id,
                command.local_worker_epoch,
                self.profile_sha256.clone(),
                execution,
                ModelStateHandle::new(format!("destination:{}", command.execution_id))?,
                binding(),
                command.expires_at,
            )?),
            ServingPhase::Aggregated => {
                return Err(PowerError::Config(
                    "the cross-process fixture cannot execute the aggregated phase".to_string(),
                ));
            }
        };
        Ok(PhaseDecision::ready(prepared))
    }

    async fn execute(
        &self,
        command: ExecutePhaseExecution,
    ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
        match command {
            ExecutePhaseExecution::Prefill { prepared } => {
                let source = ModelStateHandle::new(format!("source:{}", prepared.execution_id()))?;
                self.store.insert(&source, fixture_state())?;
                Ok(PhaseDecision::ready(PhaseExecutionOutput::Prefill(
                    ProducedModelState::new(
                        prepared.execution_id(),
                        prepared.local_worker_epoch(),
                        self.profile_sha256.clone(),
                        source,
                        binding(),
                    )?,
                )))
            }
            ExecutePhaseExecution::Decode { prepared, state } => {
                let imported = self.store.take(state.destination())?;
                if imported != fixture_state() {
                    return Err(PowerError::InferenceFailed(
                        "cross-process state did not preserve backend-owned bytes".to_string(),
                    ));
                }
                let stream = futures::stream::once(async {
                    Ok(PhaseResponseChunk::Completion(CompletionResponseChunk {
                        text: "cross-process-token".to_string(),
                        done: true,
                        prompt_tokens: Some(16),
                        done_reason: Some("stop".to_string()),
                        prompt_eval_duration_ns: None,
                        token_id: Some(23),
                    }))
                });
                drop(prepared);
                Ok(PhaseDecision::ready(PhaseExecutionOutput::Decode(
                    Box::pin(stream),
                )))
            }
        }
    }

    async fn abort(&self, command: AbortPhaseExecution) -> Result<()> {
        self.store.remove_execution(command.execution_id)
    }
}
