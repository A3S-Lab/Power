use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{Duration, Utc};
use uuid::Uuid;

use super::super::*;
use crate::backend::types::{CompletionRequest, CompletionResponseChunk};
use crate::error::{PowerError, Result};

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

pub(super) fn profile(role: DisaggregatedServingRole, timeout_ms: u64) -> ServingExecutionProfile {
    ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
        role,
        model: "internal/model-v1".to_string(),
        model_sha256: digest('1'),
        backend: "test-backend".to_string(),
        backend_sha256: digest('2'),
        execution_sha256: digest('3'),
        device_sha256: digest('4'),
        layout_sha256: digest('5'),
        peer_set_sha256: digest('6'),
        generation: 7,
        protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
        state_kind: StateKind::KvCache,
        max_state_bytes: 1024,
        max_inflight_transfers: 2,
        transfer_timeout_ms: timeout_ms,
        cancellation_timeout_ms: timeout_ms.min(10),
        privacy: ServingPrivacyMode::AuthenticatedEncryptedTransport,
        privacy_policy_sha256: digest('7'),
        attestation_policy_sha256: None,
    })
    .unwrap()
}

pub(super) fn binding() -> StateTransferBinding {
    StateTransferBinding {
        model_sha256: digest('1'),
        execution_sha256: digest('3'),
        layout_sha256: digest('5'),
        state_kind: StateKind::KvCache,
        token_count: 16,
        state_bytes: 512,
    }
}

pub(super) fn request() -> PhaseRequest {
    PhaseRequest::Completion(
        serde_json::from_value::<CompletionRequest>(serde_json::json!({
            "prompt": "private prompt",
            "stream": true
        }))
        .unwrap(),
    )
}

#[derive(Default)]
pub(super) struct Calls {
    values: Mutex<Vec<&'static str>>,
    pub(super) phase_aborts: AtomicUsize,
    pub(super) transfer_aborts: AtomicUsize,
}

impl Calls {
    fn push(&self, value: &'static str) {
        self.values.lock().unwrap().push(value);
    }

    pub(super) fn values(&self) -> Vec<&'static str> {
        self.values.lock().unwrap().clone()
    }
}

struct TestTransferDriver {
    capabilities: StateTransferCapabilities,
    calls: Arc<Calls>,
}

#[async_trait]
impl StateTransferService for TestTransferDriver {
    fn capabilities(&self) -> StateTransferCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> TransferHealth {
        TransferHealth::Ready
    }

    async fn prepare_destination(
        &self,
        command: PrepareStateTransfer,
    ) -> Result<StateTransferTarget> {
        self.calls.push("transfer.prepare");
        Ok(StateTransferTarget {
            schema: STATE_TRANSFER_TARGET_SCHEMA.to_string(),
            transfer_id: command.transfer_id,
            destination_worker_epoch: command.local_worker_epoch,
            binding: command.binding,
            protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
            prepared_at: Utc::now(),
            expires_at: command.expires_at,
            ticket: "target-ticket".to_string(),
        })
    }

    async fn publish_source(&self, command: PublishStateTransfer) -> Result<StateTransferSource> {
        self.calls.push("transfer.publish");
        Ok(StateTransferSource {
            schema: STATE_TRANSFER_SOURCE_SCHEMA.to_string(),
            transfer_id: command.target.transfer_id,
            source_worker_epoch: command.local_worker_epoch,
            destination_worker_epoch: command.target.destination_worker_epoch,
            binding: command.target.binding,
            protocol: command.target.protocol,
            published_at: Utc::now(),
            expires_at: command.target.expires_at,
            ticket: "source-ticket".to_string(),
        })
    }

    async fn consume_source(&self, command: ConsumeStateTransfer) -> Result<StateTransferReceipt> {
        self.calls.push("transfer.consume");
        Ok(StateTransferReceipt {
            schema: STATE_TRANSFER_RECEIPT_SCHEMA.to_string(),
            transfer_id: command.source.transfer_id,
            source_worker_epoch: command.source.source_worker_epoch,
            destination_worker_epoch: command.local_worker_epoch,
            binding: command.source.binding,
            protocol: command.source.protocol,
            bytes_transferred: 512,
            integrity: StateTransferIntegrity::TransportVerified,
            completed_at: Utc::now(),
        })
    }

    async fn abort(&self, _command: AbortStateTransfer) -> Result<()> {
        self.calls.push("transfer.abort");
        self.calls.transfer_aborts.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct TestPhaseExecutor {
    capabilities: PhaseExecutorCapabilities,
    profile_sha256: String,
    prepared_expiry_delta: Duration,
    block_prepare: bool,
    retryable_prepare: bool,
    fail_abort: bool,
    calls: Arc<Calls>,
}

#[async_trait]
impl ServingPhaseExecutor for TestPhaseExecutor {
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
        self.calls.push("phase.prepare");
        if self.block_prepare {
            std::future::pending::<()>().await;
        }
        if self.retryable_prepare {
            return PhaseDecision::retryable_unavailable(
                RetryableUnavailableReason::AdmissionPressure,
                Some(5),
            );
        }
        let execution = PhaseExecutionHandle::new("phase-handle")?;
        let expires_at = command.expires_at + self.prepared_expiry_delta;
        let prepared = match self.capabilities.phase {
            ServingPhase::Prefill => PreparedPhaseExecution::Prefill(PreparedPrefillPhase::new(
                command.execution_id,
                command.local_worker_epoch,
                self.profile_sha256.clone(),
                execution,
                expires_at,
            )?),
            ServingPhase::Decode => PreparedPhaseExecution::Decode(PreparedDecodePhase::new(
                command.execution_id,
                command.local_worker_epoch,
                self.profile_sha256.clone(),
                execution,
                ModelStateHandle::new("decode-destination")?,
                binding(),
                expires_at,
            )?),
            ServingPhase::Aggregated => {
                return Err(PowerError::InvalidRequest(
                    "test executor cannot prepare aggregated work".to_string(),
                ));
            }
        };
        Ok(PhaseDecision::ready(prepared))
    }

    async fn execute(
        &self,
        command: ExecutePhaseExecution,
    ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
        self.calls.push("phase.execute");
        match command {
            ExecutePhaseExecution::Prefill { prepared } => Ok(PhaseDecision::ready(
                PhaseExecutionOutput::Prefill(ProducedModelState::new(
                    prepared.execution_id(),
                    prepared.local_worker_epoch(),
                    self.profile_sha256.clone(),
                    ModelStateHandle::new("prefill-source")?,
                    binding(),
                )?),
            )),
            ExecutePhaseExecution::Decode { .. } => {
                let stream = futures::stream::once(async {
                    Ok(PhaseResponseChunk::Completion(CompletionResponseChunk {
                        text: "token".to_string(),
                        done: true,
                        prompt_tokens: Some(16),
                        done_reason: Some("stop".to_string()),
                        prompt_eval_duration_ns: None,
                        token_id: Some(7),
                    }))
                });
                Ok(PhaseDecision::ready(PhaseExecutionOutput::Decode(
                    Box::pin(stream),
                )))
            }
        }
    }

    async fn abort(&self, _command: AbortPhaseExecution) -> Result<()> {
        self.calls.push("phase.abort");
        self.calls.phase_aborts.fetch_add(1, Ordering::SeqCst);
        if self.fail_abort {
            Err(PowerError::BackendNotAvailable(
                "test phase cleanup failed".to_string(),
            ))
        } else {
            Ok(())
        }
    }
}

pub(super) fn runtime(
    profile: &ServingExecutionProfile,
    epoch: Uuid,
    calls: Arc<Calls>,
) -> DistributedServingRuntime {
    runtime_with_expiry_delta(profile, epoch, calls, Duration::zero())
}

pub(super) fn runtime_with_expiry_delta(
    profile: &ServingExecutionProfile,
    epoch: Uuid,
    calls: Arc<Calls>,
    prepared_expiry_delta: Duration,
) -> DistributedServingRuntime {
    runtime_with_behavior(
        profile,
        epoch,
        calls,
        prepared_expiry_delta,
        false,
        false,
        false,
    )
}

pub(super) fn runtime_with_behavior(
    profile: &ServingExecutionProfile,
    epoch: Uuid,
    calls: Arc<Calls>,
    prepared_expiry_delta: Duration,
    block_prepare: bool,
    retryable_prepare: bool,
    fail_abort: bool,
) -> DistributedServingRuntime {
    let capabilities = StateTransferCapabilities {
        execution_profile_sha256: profile.sha256().unwrap(),
        phases: vec![ServingPhase::Prefill, ServingPhase::Decode],
        protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
        max_transfer_bytes: 1024,
        max_inflight_transfers: 2,
    };
    let transfer = BoundedStateTransferService::new(
        profile.clone(),
        epoch,
        Arc::new(TestTransferDriver {
            capabilities,
            calls: calls.clone(),
        }),
    )
    .unwrap();
    DistributedServingRuntime::new(
        profile.clone(),
        Arc::new(transfer),
        Arc::new(TestPhaseExecutor {
            capabilities: PhaseExecutorCapabilities {
                execution_profile_sha256: profile.sha256().unwrap(),
                phase: profile.phase(),
            },
            profile_sha256: profile.sha256().unwrap(),
            prepared_expiry_delta,
            block_prepare,
            retryable_prepare,
            fail_abort,
            calls,
        }),
    )
    .unwrap()
}

pub(super) async fn start_decode(
    runtime: &DistributedServingRuntime,
    epoch: Uuid,
    execution_id: Uuid,
    expires_at: chrono::DateTime<Utc>,
) -> PhaseResponseStream {
    let prepared = runtime
        .prepare_decode(DecodePhaseRequest {
            execution_id,
            model: "internal/model-v1".to_string(),
            request: request(),
            expires_at,
        })
        .await
        .unwrap();
    let target = match prepared {
        PhaseDecision::Ready(PreparedDecodeTransfer { target }) => target,
        _ => panic!("expected a prepared decode target"),
    };
    let source = StateTransferSource {
        schema: STATE_TRANSFER_SOURCE_SCHEMA.to_string(),
        transfer_id: target.transfer_id,
        source_worker_epoch: Uuid::new_v4(),
        destination_worker_epoch: epoch,
        binding: target.binding,
        protocol: target.protocol,
        published_at: Utc::now(),
        expires_at: target.expires_at,
        ticket: "source-ticket".to_string(),
    };
    match runtime.execute_decode(execution_id, source).await.unwrap() {
        PhaseDecision::Ready(stream) => stream,
        _ => panic!("expected a decode stream"),
    }
}

pub(super) async fn wait_for_count(counter: &AtomicUsize, expected: usize) {
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while counter.load(Ordering::SeqCst) != expected {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("cleanup should complete");
}
