//! Product-surface loopback [`ServingPhaseExecutor`] for buffered-host P/D.
//!
//! Owns opaque conformance fixture handles and pairs with
//! [`BufferedHostLoopbackStateTransfer`]. Satisfies
//! [`ProductionAdapterContract::REQUIRED`] so composition can inject both
//! ports without test-only fixtures.
//!
//! This is **not** model-semantic execution, llama.cpp P/D, or high-speed
//! network evidence. Decode never becomes Ready from a transfer receipt alone:
//! the executor must reclaim and verify opaque adapter-owned bytes first.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use uuid::Uuid;

use crate::backend::types::CompletionResponseChunk;
use crate::error::{PowerError, Result};

use super::{
    AbortPhaseExecution, AdapterProvisionState, BufferedHostLoopbackStateTransfer,
    ExecutePhaseExecution, ModelStateHandle, PhaseDecision, PhaseExecutionHandle,
    PhaseExecutionOutput, PhaseExecutorCapabilities, PhaseExecutorHealth, PhaseResponseChunk,
    PreparePhaseExecution, PreparedDecodePhase, PreparedPhaseExecution, PreparedPrefillPhase,
    ProducedModelState, ProductionAdapterContract, RecomputeReason, ServingExecutionProfile,
    ServingPhase, ServingPhaseExecutor, ServingPrivacyMode, StateKind, StateTransferBinding,
    StateTransferProtocol, StateTransferService,
};

/// Fixed opaque payload size for loopback conformance (not KV layout meaning).
const CONFORMANCE_STATE_BYTES: u64 = 64;
const CONFORMANCE_TOKEN_COUNT: u64 = 8;
const CONFORMANCE_TOKEN_TEXT: &str = "loopback-conformance-token";

#[derive(Default)]
struct PreparedLeaseStore {
    /// execution_id → local state handle still owned until publish/consume/abort
    handles: Mutex<HashMap<Uuid, ModelStateHandle>>,
}

impl PreparedLeaseStore {
    fn lock(&self) -> Result<MutexGuard<'_, HashMap<Uuid, ModelStateHandle>>> {
        self.handles.lock().map_err(|_| {
            PowerError::BackendNotAvailable(
                "buffered-host loopback phase lease lock is unavailable".to_string(),
            )
        })
    }

    fn insert(&self, execution_id: Uuid, handle: ModelStateHandle) -> Result<()> {
        let mut leases = self.lock()?;
        if leases.contains_key(&execution_id) {
            return Err(PowerError::InvalidRequest(
                "buffered-host loopback phase already holds a lease for this execution".to_string(),
            ));
        }
        leases.insert(execution_id, handle);
        Ok(())
    }

    fn take(&self, execution_id: Uuid) -> Result<Option<ModelStateHandle>> {
        Ok(self.lock()?.remove(&execution_id))
    }

    fn remove(&self, execution_id: Uuid) -> Result<()> {
        self.lock()?.remove(&execution_id);
        Ok(())
    }
}

/// Injectable buffered-host / loopback [`ServingPhaseExecutor`].
///
/// Construct with [`Self::pair_with`] against the matching
/// [`BufferedHostLoopbackStateTransfer`], or [`Self::paired_for_profile`] to
/// build both ports together.
pub struct BufferedHostLoopbackPhaseExecutor {
    capabilities: PhaseExecutorCapabilities,
    profile_sha256: String,
    binding: StateTransferBinding,
    transfer: Arc<BufferedHostLoopbackStateTransfer>,
    leases: PreparedLeaseStore,
}

impl std::fmt::Debug for BufferedHostLoopbackPhaseExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BufferedHostLoopbackPhaseExecutor")
            .field("capabilities", &self.capabilities)
            .field("binding", &self.binding)
            .finish_non_exhaustive()
    }
}

impl BufferedHostLoopbackPhaseExecutor {
    /// Build both product-surface ports for one buffered-host loopback profile.
    pub fn paired_for_profile(
        profile: &ServingExecutionProfile,
    ) -> Result<(Arc<BufferedHostLoopbackStateTransfer>, Arc<Self>)> {
        let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(profile)?);
        let executor = Arc::new(Self::pair_with(profile, Arc::clone(&transfer))?);
        Ok((transfer, executor))
    }

    /// Bind one phase executor to an already-constructed loopback transfer.
    pub fn pair_with(
        profile: &ServingExecutionProfile,
        transfer: Arc<BufferedHostLoopbackStateTransfer>,
    ) -> Result<Self> {
        let ServingExecutionProfile::PrefillDecode { execution } = profile else {
            return Err(PowerError::Config(
                "buffered-host loopback phase executor requires a prefill-decode serving profile"
                    .to_string(),
            ));
        };
        if !matches!(
            execution.protocol,
            StateTransferProtocol::BufferedHostMemoryPullV1
        ) {
            return Err(PowerError::Config(
                "buffered-host loopback phase executor requires BufferedHostMemoryPullV1"
                    .to_string(),
            ));
        }
        if !matches!(
            execution.privacy,
            ServingPrivacyMode::AuthenticatedEncryptedTransport
        ) {
            return Err(PowerError::Config(
                "buffered-host loopback phase executor requires AuthenticatedEncryptedTransport privacy"
                    .to_string(),
            ));
        }
        if execution.max_state_bytes < CONFORMANCE_STATE_BYTES {
            return Err(PowerError::Config(format!(
                "buffered-host loopback phase executor requires max_state_bytes >= {CONFORMANCE_STATE_BYTES}"
            )));
        }
        let transfer_caps = transfer.capabilities();
        let profile_sha256 = profile.sha256()?;
        if transfer_caps.execution_profile_sha256 != profile_sha256 {
            return Err(PowerError::Config(
                "buffered-host loopback phase executor requires a transfer bound to the same profile digest"
                    .to_string(),
            ));
        }
        let capabilities = PhaseExecutorCapabilities::for_profile(profile)?;
        let binding = StateTransferBinding {
            model_sha256: execution.model_sha256.clone(),
            execution_sha256: execution.execution_sha256.clone(),
            layout_sha256: execution.layout_sha256.clone(),
            state_kind: execution.state_kind,
            token_count: CONFORMANCE_TOKEN_COUNT,
            state_bytes: CONFORMANCE_STATE_BYTES,
        };
        profile.validate_state_binding(&binding)?;
        // Touch state kind only as an opaque label; layout bytes are fixture.
        if !matches!(
            binding.state_kind,
            StateKind::KvCache | StateKind::Recurrent
        ) {
            return Err(PowerError::Config(
                "buffered-host loopback phase executor requires a known opaque state kind"
                    .to_string(),
            ));
        }
        Ok(Self {
            capabilities,
            profile_sha256,
            binding,
            transfer,
            leases: PreparedLeaseStore::default(),
        })
    }

    fn opaque_conformance_state() -> Vec<u8> {
        (0..CONFORMANCE_STATE_BYTES)
            .map(|index| ((index * 17 + 3) % 251) as u8)
            .collect()
    }

    fn reclaim_local_handle(&self, handle: &ModelStateHandle) -> Result<()> {
        // Best-effort reclaim of adapter-owned registration left by this port.
        // Missing handles are fine (already published or never registered).
        let _ = self.transfer.take_owned_state(handle);
        Ok(())
    }
}

#[async_trait]
impl ServingPhaseExecutor for BufferedHostLoopbackPhaseExecutor {
    fn capabilities(&self) -> PhaseExecutorCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> PhaseExecutorHealth {
        PhaseExecutorHealth::Ready
    }

    fn provision(&self) -> AdapterProvisionState {
        AdapterProvisionState::Injected
    }

    fn production_contract(&self) -> ProductionAdapterContract {
        ProductionAdapterContract::REQUIRED
    }

    async fn prepare(
        &self,
        command: PreparePhaseExecution,
    ) -> Result<PhaseDecision<PreparedPhaseExecution>> {
        let execution = PhaseExecutionHandle::new(format!(
            "buffered-host-loopback-phase:{}",
            command.execution_id
        ))?;
        let prepared = match self.capabilities.phase {
            ServingPhase::Prefill => {
                let handle = ModelStateHandle::new(format!("source:{}", command.execution_id))?;
                self.leases.insert(command.execution_id, handle)?;
                PreparedPhaseExecution::Prefill(PreparedPrefillPhase::new(
                    command.execution_id,
                    command.local_worker_epoch,
                    self.profile_sha256.clone(),
                    execution,
                    command.expires_at,
                )?)
            }
            ServingPhase::Decode => {
                let destination =
                    ModelStateHandle::new(format!("destination:{}", command.execution_id))?;
                self.leases
                    .insert(command.execution_id, destination.clone())?;
                PreparedPhaseExecution::Decode(PreparedDecodePhase::new(
                    command.execution_id,
                    command.local_worker_epoch,
                    self.profile_sha256.clone(),
                    execution,
                    destination,
                    self.binding.clone(),
                    command.expires_at,
                )?)
            }
            ServingPhase::Aggregated => {
                return Err(PowerError::Config(
                    "buffered-host loopback phase executor cannot execute the aggregated phase"
                        .to_string(),
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
                let source = self.leases.take(prepared.execution_id())?.ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "buffered-host loopback prefill lease is missing".to_string(),
                    )
                })?;
                let state = Self::opaque_conformance_state();
                self.transfer.register_owned_state(&source, state)?;
                Ok(PhaseDecision::ready(PhaseExecutionOutput::Prefill(
                    ProducedModelState::new(
                        prepared.execution_id(),
                        prepared.local_worker_epoch(),
                        self.profile_sha256.clone(),
                        source,
                        self.binding.clone(),
                    )?,
                )))
            }
            ExecutePhaseExecution::Decode { prepared, state } => {
                // Transport receipt alone is never decode success. Reclaim and
                // verify opaque adapter-owned bytes before any Ready stream.
                let destination = state.destination().clone();
                let imported = match self.transfer.take_owned_state(&destination) {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        let _ = self.leases.remove(prepared.execution_id());
                        return Ok(PhaseDecision::recompute(RecomputeReason::StateMissing));
                    }
                };
                let _ = self.leases.remove(prepared.execution_id());
                if imported.len() as u64 != prepared.binding().state_bytes {
                    return Ok(PhaseDecision::recompute(RecomputeReason::StateIncompatible));
                }
                if imported != Self::opaque_conformance_state() {
                    return Ok(PhaseDecision::recompute(RecomputeReason::StateCorrupt));
                }
                let stream = futures::stream::once(async {
                    Ok(PhaseResponseChunk::Completion(CompletionResponseChunk {
                        text: CONFORMANCE_TOKEN_TEXT.to_string(),
                        done: true,
                        prompt_tokens: Some(CONFORMANCE_TOKEN_COUNT as u32),
                        done_reason: Some("stop".to_string()),
                        prompt_eval_duration_ns: None,
                        token_id: Some(1),
                    }))
                });
                drop(prepared);
                drop(state);
                Ok(PhaseDecision::ready(PhaseExecutionOutput::Decode(
                    Box::pin(stream),
                )))
            }
        }
    }

    async fn abort(&self, command: AbortPhaseExecution) -> Result<()> {
        if let Some(handle) = self.leases.take(command.execution_id)? {
            self.reclaim_local_handle(&handle)?;
        }
        // Prefill execute registers `source:{id}` after releasing the prepare
        // lease; reclaim that handle if publish has not already taken it.
        if let Ok(source) = ModelStateHandle::new(format!("source:{}", command.execution_id)) {
            let _ = self.reclaim_local_handle(&source);
        }
        if let Ok(destination) =
            ModelStateHandle::new(format!("destination:{}", command.execution_id))
        {
            let _ = self.reclaim_local_handle(&destination);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, Utc};
    use futures::StreamExt;

    use super::*;
    use crate::serving::{
        validate_injected_production_adapters, AbortStateTransfer, ConsumeStateTransfer,
        DisaggregatedServingRole, EmptyServingPhaseExecutor, ImportedModelState, PhaseRequest,
        PhaseSessionPoolMode, PhaseWeightCacheMode, PrefillDecodeExecutionProfile,
        PrepareStateTransfer, PublishStateTransfer, StateTransferService,
    };

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn profile(role: DisaggregatedServingRole) -> ServingExecutionProfile {
        ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role,
            model: "fixture".to_string(),
            backend: "buffered-host-loopback".to_string(),
            model_sha256: digest('1'),
            backend_sha256: digest('2'),
            execution_sha256: digest('3'),
            device_sha256: digest('4'),
            layout_sha256: digest('5'),
            peer_set_sha256: digest('6'),
            generation: 7,
            state_kind: StateKind::KvCache,
            protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
            max_state_bytes: 64,
            max_inflight_transfers: 2,
            transfer_timeout_ms: 30_000,
            cancellation_timeout_ms: 5_000,
            privacy: ServingPrivacyMode::AuthenticatedEncryptedTransport,
            privacy_policy_sha256: digest('7'),
            attestation_policy_sha256: None,
            weight_cache: PhaseWeightCacheMode::SharedWeightHierarchy,
            residency_policy_sha256: None,
            session_pool: PhaseSessionPoolMode::SharedSessionPool,
            session_pool_policy_sha256: None,
            transport: None,
            phase_executor: None,
            state_ownership: None,
            phase_execution: None,
        })
        .unwrap()
    }

    fn completion_request() -> PhaseRequest {
        PhaseRequest::Completion(
            serde_json::from_value(serde_json::json!({ "prompt": "unused" })).unwrap(),
        )
    }

    fn imported_proof(
        profile: &ServingExecutionProfile,
        prepared: &PreparedDecodePhase,
        transfer_id: Uuid,
        local_worker_epoch: Uuid,
    ) -> ImportedModelState {
        ImportedModelState::from_parts_for_test(
            transfer_id,
            local_worker_epoch,
            profile.sha256().unwrap(),
            prepared.destination().clone(),
            prepared.binding().clone(),
        )
    }

    #[test]
    fn product_pair_is_injected_with_required_contract() {
        let profile = profile(DisaggregatedServingRole::Decode);
        let (transfer, executor) =
            BufferedHostLoopbackPhaseExecutor::paired_for_profile(&profile).unwrap();
        assert!(transfer.provision().is_injected());
        assert!(executor.provision().is_injected());
        assert_eq!(
            executor.production_contract(),
            ProductionAdapterContract::REQUIRED
        );
        assert_eq!(executor.health(), PhaseExecutorHealth::Ready);
        validate_injected_production_adapters(transfer.as_ref(), executor.as_ref()).unwrap();
    }

    #[test]
    fn refuses_aggregated_and_undersized_profiles() {
        let err = BufferedHostLoopbackPhaseExecutor::paired_for_profile(
            &ServingExecutionProfile::Aggregated {},
        )
        .unwrap_err();
        assert!(err.to_string().contains("prefill-decode"));

        let mut execution = match profile(DisaggregatedServingRole::Prefill) {
            ServingExecutionProfile::PrefillDecode { execution } => *execution,
            ServingExecutionProfile::Aggregated {} => panic!("expected prefill-decode"),
        };
        execution.max_state_bytes = 32;
        let small = ServingExecutionProfile::prefill_decode(execution).unwrap();
        let err = BufferedHostLoopbackPhaseExecutor::paired_for_profile(&small).unwrap_err();
        assert!(err.to_string().contains("max_state_bytes"));
    }

    #[test]
    fn empty_phase_still_fails_with_product_transfer() {
        let profile = profile(DisaggregatedServingRole::Decode);
        let (transfer, _) =
            BufferedHostLoopbackPhaseExecutor::paired_for_profile(&profile).unwrap();
        let empty = EmptyServingPhaseExecutor::for_profile(&profile).unwrap();
        let err = validate_injected_production_adapters(transfer.as_ref(), &empty).unwrap_err();
        assert!(err.to_string().contains("Empty placeholders"));
    }

    #[tokio::test]
    async fn prefill_decode_round_trip_verifies_opaque_bytes_before_ready() {
        let prefill_profile = profile(DisaggregatedServingRole::Prefill);
        let decode_profile = profile(DisaggregatedServingRole::Decode);
        let (prefill_transfer, prefill_executor) =
            BufferedHostLoopbackPhaseExecutor::paired_for_profile(&prefill_profile).unwrap();
        let (decode_transfer, decode_executor) =
            BufferedHostLoopbackPhaseExecutor::paired_for_profile(&decode_profile).unwrap();

        let execution_id = Uuid::new_v4();
        let source_epoch = Uuid::new_v4();
        let destination_epoch = Uuid::new_v4();
        let expires_at = Utc::now() + Duration::seconds(30);

        let prepared = match prefill_executor
            .prepare(PreparePhaseExecution {
                execution_id,
                local_worker_epoch: source_epoch,
                model: "fixture".to_string(),
                request: completion_request(),
                expires_at,
            })
            .await
            .unwrap()
        {
            PhaseDecision::Ready(PreparedPhaseExecution::Prefill(prepared)) => prepared,
            PhaseDecision::Ready(_) => panic!("expected ready prefill reservation"),
            PhaseDecision::Recompute { .. } => panic!("expected ready prefill, got Recompute"),
            PhaseDecision::RetryableUnavailable { .. } => {
                panic!("expected ready prefill, got RetryableUnavailable")
            }
            PhaseDecision::TerminalFailure { .. } => {
                panic!("expected ready prefill, got TerminalFailure")
            }
        };
        let produced = match prefill_executor
            .execute(ExecutePhaseExecution::prefill(prepared))
            .await
            .unwrap()
        {
            PhaseDecision::Ready(PhaseExecutionOutput::Prefill(produced)) => produced,
            PhaseDecision::Ready(_) => panic!("expected ready produced state"),
            PhaseDecision::Recompute { .. } => panic!("expected ready produced, got Recompute"),
            PhaseDecision::RetryableUnavailable { .. } => {
                panic!("expected ready produced, got RetryableUnavailable")
            }
            PhaseDecision::TerminalFailure { .. } => {
                panic!("expected ready produced, got TerminalFailure")
            }
        };

        let decode_prepared = match decode_executor
            .prepare(PreparePhaseExecution {
                execution_id,
                local_worker_epoch: destination_epoch,
                model: "fixture".to_string(),
                request: completion_request(),
                expires_at,
            })
            .await
            .unwrap()
        {
            PhaseDecision::Ready(PreparedPhaseExecution::Decode(prepared)) => prepared,
            PhaseDecision::Ready(_) => panic!("expected ready decode reservation"),
            PhaseDecision::Recompute { .. } => panic!("expected ready decode, got Recompute"),
            PhaseDecision::RetryableUnavailable { .. } => {
                panic!("expected ready decode, got RetryableUnavailable")
            }
            PhaseDecision::TerminalFailure { .. } => {
                panic!("expected ready decode, got TerminalFailure")
            }
        };

        let target = decode_transfer
            .prepare_destination(PrepareStateTransfer {
                transfer_id: execution_id,
                local_worker_epoch: destination_epoch,
                binding: produced.binding().clone(),
                destination: decode_prepared.destination().clone(),
                expires_at,
            })
            .await
            .unwrap();
        let published = prefill_transfer
            .publish_source(PublishStateTransfer {
                local_worker_epoch: source_epoch,
                source: produced.source().clone(),
                target,
            })
            .await
            .unwrap();
        let _receipt = decode_transfer
            .consume_source(ConsumeStateTransfer {
                local_worker_epoch: destination_epoch,
                destination: decode_prepared.destination().clone(),
                source: published,
            })
            .await
            .unwrap();

        // Receipt proved movement only; Ready still requires opaque byte verify.
        let imported = imported_proof(
            &decode_profile,
            &decode_prepared,
            execution_id,
            destination_epoch,
        );
        let decision = decode_executor
            .execute(ExecutePhaseExecution::decode(decode_prepared, imported).unwrap())
            .await
            .unwrap();

        match decision {
            PhaseDecision::Ready(PhaseExecutionOutput::Decode(mut stream)) => {
                let chunk = stream.next().await.unwrap().unwrap();
                match chunk {
                    PhaseResponseChunk::Completion(chunk) => {
                        assert_eq!(chunk.text, CONFORMANCE_TOKEN_TEXT);
                        assert!(chunk.done);
                    }
                    PhaseResponseChunk::Chat(_) => panic!("expected completion chunk"),
                }
            }
            _ => panic!("expected ready decode stream"),
        }
    }

    #[tokio::test]
    async fn decode_execute_recomputes_when_adapter_state_missing() {
        let profile = profile(DisaggregatedServingRole::Decode);
        let (transfer, executor) =
            BufferedHostLoopbackPhaseExecutor::paired_for_profile(&profile).unwrap();
        let execution_id = Uuid::new_v4();
        let epoch = Uuid::new_v4();
        let expires_at = Utc::now() + Duration::seconds(30);
        let prepared = match executor
            .prepare(PreparePhaseExecution {
                execution_id,
                local_worker_epoch: epoch,
                model: "fixture".to_string(),
                request: completion_request(),
                expires_at,
            })
            .await
            .unwrap()
        {
            PhaseDecision::Ready(PreparedPhaseExecution::Decode(prepared)) => prepared,
            _ => panic!("expected ready decode reservation"),
        };

        // Prepare destination reservation but never consume — destination empty.
        transfer
            .prepare_destination(PrepareStateTransfer {
                transfer_id: execution_id,
                local_worker_epoch: epoch,
                binding: prepared.binding().clone(),
                destination: prepared.destination().clone(),
                expires_at,
            })
            .await
            .unwrap();

        let imported = imported_proof(&profile, &prepared, execution_id, epoch);
        let decision = executor
            .execute(ExecutePhaseExecution::decode(prepared, imported).unwrap())
            .await
            .unwrap();
        match decision {
            PhaseDecision::Recompute {
                reason: RecomputeReason::StateMissing,
            } => {}
            _ => panic!("expected StateMissing recompute"),
        }
    }

    #[tokio::test]
    async fn decode_execute_recomputes_on_corrupt_opaque_bytes() {
        let profile = profile(DisaggregatedServingRole::Decode);
        let (transfer, executor) =
            BufferedHostLoopbackPhaseExecutor::paired_for_profile(&profile).unwrap();
        let execution_id = Uuid::new_v4();
        let epoch = Uuid::new_v4();
        let expires_at = Utc::now() + Duration::seconds(30);
        let prepared = match executor
            .prepare(PreparePhaseExecution {
                execution_id,
                local_worker_epoch: epoch,
                model: "fixture".to_string(),
                request: completion_request(),
                expires_at,
            })
            .await
            .unwrap()
        {
            PhaseDecision::Ready(PreparedPhaseExecution::Decode(prepared)) => prepared,
            _ => panic!("expected ready decode reservation"),
        };

        transfer
            .prepare_destination(PrepareStateTransfer {
                transfer_id: execution_id,
                local_worker_epoch: epoch,
                binding: prepared.binding().clone(),
                destination: prepared.destination().clone(),
                expires_at,
            })
            .await
            .unwrap();
        transfer
            .abort(AbortStateTransfer {
                transfer_id: execution_id,
                local_worker_epoch: epoch,
            })
            .await
            .unwrap();
        transfer
            .register_owned_state(
                prepared.destination(),
                vec![0xFF; CONFORMANCE_STATE_BYTES as usize],
            )
            .unwrap();

        let imported = imported_proof(&profile, &prepared, execution_id, epoch);
        let decision = executor
            .execute(ExecutePhaseExecution::decode(prepared, imported).unwrap())
            .await
            .unwrap();
        match decision {
            PhaseDecision::Recompute {
                reason: RecomputeReason::StateCorrupt,
            } => {}
            _ => panic!("expected StateCorrupt recompute"),
        }
    }

    #[tokio::test]
    async fn abort_reclaims_registered_prefill_source() {
        let profile = profile(DisaggregatedServingRole::Prefill);
        let (transfer, executor) =
            BufferedHostLoopbackPhaseExecutor::paired_for_profile(&profile).unwrap();
        let execution_id = Uuid::new_v4();
        let epoch = Uuid::new_v4();
        let expires_at = Utc::now() + Duration::seconds(30);
        let prepared = match executor
            .prepare(PreparePhaseExecution {
                execution_id,
                local_worker_epoch: epoch,
                model: "fixture".to_string(),
                request: completion_request(),
                expires_at,
            })
            .await
            .unwrap()
        {
            PhaseDecision::Ready(PreparedPhaseExecution::Prefill(prepared)) => prepared,
            _ => panic!("expected ready prefill reservation"),
        };
        let produced = match executor
            .execute(ExecutePhaseExecution::prefill(prepared.clone()))
            .await
            .unwrap()
        {
            PhaseDecision::Ready(PhaseExecutionOutput::Prefill(produced)) => produced,
            _ => panic!("expected ready produced state"),
        };
        executor
            .abort(
                AbortPhaseExecution::prepared(execution_id, epoch, prepared.execution().clone())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(transfer.take_owned_state(produced.source()).is_err());
    }
}
