use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use tokio::sync::Semaphore;
use tokio::time::Instant;
use uuid::Uuid;

use crate::error::{PowerError, Result};

use super::{
    BoundedStateTransferService, ConsumeStateTransfer, DisaggregatedServingRole,
    ExecutePhaseExecution, ImportedModelState, PhaseDecision, PhaseExecutionOutput, PhaseRequest,
    PhaseResponseStream, PreparePhaseExecution, PrepareStateTransfer, PreparedPhaseExecution,
    PublishStateTransfer, ServingExecutionProfile, ServingPhase, ServingPhaseExecutor,
    StateTransferService, StateTransferSource, StateTransferTarget, TransferHealth,
};

mod lifecycle;
mod operations;

use lifecycle::{guarded_stream, RuntimeInner, RuntimeOperationGuard};
use operations::{ready_or_cleanup, ReadyOrDecision};

/// Local decode preparation requested by the external request orchestrator.
pub struct DecodePhaseRequest {
    pub execution_id: Uuid,
    pub model: String,
    pub request: PhaseRequest,
    pub expires_at: DateTime<Utc>,
}

impl fmt::Debug for DecodePhaseRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodePhaseRequest")
            .field("execution_id", &self.execution_id)
            .field("model", &self.model)
            .field("request", &self.request)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// Local prefill execution requested against an exact remote decode target.
pub struct PrefillPhaseRequest {
    pub execution_id: Uuid,
    pub model: String,
    pub request: PhaseRequest,
    pub target: StateTransferTarget,
    pub expires_at: DateTime<Utc>,
}

impl fmt::Debug for PrefillPhaseRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrefillPhaseRequest")
            .field("execution_id", &self.execution_id)
            .field("model", &self.model)
            .field("request", &self.request)
            .field("target", &self.target)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// Decode-side target that the orchestrator passes unchanged to prefill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedDecodeTransfer {
    pub target: StateTransferTarget,
}

/// Prefill-side source that the orchestrator passes unchanged to decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedPrefillState {
    pub source: StateTransferSource,
}

/// Process-local application service that composes one phase executor with one
/// bounded state-transfer service.
///
/// It performs no endpoint selection. Gateway supplies one execution ID and
/// threads only the opaque target/source descriptors between workers.
#[derive(Clone)]
pub struct DistributedServingRuntime {
    inner: Arc<RuntimeInner>,
    role: DisaggregatedServingRole,
    transfer_timeout: std::time::Duration,
}

impl fmt::Debug for DistributedServingRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DistributedServingRuntime")
            .field("role", &self.role)
            .field(
                "local_worker_epoch",
                &self.inner.transfer.local_worker_epoch(),
            )
            .field("accepts_work", &self.accepts_work())
            .finish_non_exhaustive()
    }
}

impl DistributedServingRuntime {
    pub fn new(
        profile: ServingExecutionProfile,
        transfer: Arc<BoundedStateTransferService>,
        executor: Arc<dyn ServingPhaseExecutor>,
    ) -> Result<Self> {
        profile.validate()?;
        let ServingExecutionProfile::PrefillDecode { execution } = &profile else {
            return Err(PowerError::Config(
                "aggregated serving cannot create a distributed serving runtime".to_string(),
            ));
        };
        profile.validate_state_transfer_capabilities(&transfer.capabilities())?;
        let executor_capabilities = executor.capabilities();
        profile.validate_phase_executor_capabilities(&executor_capabilities)?;
        if matches!(transfer.health(), TransferHealth::Unsupported) {
            return Err(PowerError::Config(
                "distributed serving runtime requires a supported transfer adapter".to_string(),
            ));
        }
        let maximum = usize::try_from(execution.max_inflight_transfers).map_err(|_| {
            PowerError::Config("distributed phase capacity does not fit this platform".to_string())
        })?;
        Ok(Self {
            inner: Arc::new(RuntimeInner {
                profile: profile.clone(),
                transfer,
                executor,
                executor_capabilities,
                cancellation_timeout: std::time::Duration::from_millis(
                    execution.cancellation_timeout_ms,
                ),
                capacity: Arc::new(Semaphore::new(maximum)),
                leases: Mutex::new(std::collections::HashMap::with_capacity(maximum)),
                tainted: AtomicBool::new(false),
            }),
            role: execution.role,
            transfer_timeout: std::time::Duration::from_millis(execution.transfer_timeout_ms),
        })
    }

    pub fn phase(&self) -> ServingPhase {
        self.role.into()
    }

    /// Immutable profile that binds both injected local services.
    pub fn profile(&self) -> &ServingExecutionProfile {
        &self.inner.profile
    }

    /// Current health of the bounded local transfer path.
    pub fn transfer_health(&self) -> TransferHealth {
        self.inner.transfer.health()
    }

    pub fn accepts_work(&self) -> bool {
        !self.inner.tainted.load(Ordering::Acquire)
            && self.inner.executor.health().accepts_work()
            && matches!(
                self.transfer_health(),
                TransferHealth::Ready | TransferHealth::Degraded
            )
    }

    pub async fn prepare_decode(
        &self,
        request: DecodePhaseRequest,
    ) -> Result<PhaseDecision<PreparedDecodeTransfer>> {
        self.ensure_available(DisaggregatedServingRole::Decode)?;
        let now = Utc::now();
        let command = PreparePhaseExecution {
            execution_id: request.execution_id,
            local_worker_epoch: self.inner.transfer.local_worker_epoch(),
            model: request.model,
            request: request.request,
            expires_at: request.expires_at,
        };
        command.validate_at(now, &self.inner.executor_capabilities, &self.inner.profile)?;
        let deadline = self.deadline(command.expires_at)?;
        let cancellation = self.inner.reserve(command.execution_id, deadline)?;
        let mut guard = RuntimeOperationGuard::new(Arc::clone(&self.inner), command.execution_id);
        let operation = self.prepare_phase(command, cancellation, deadline).await;
        let decision = guard.require(operation).await?;
        let prepared = match ready_or_cleanup(decision, &mut guard).await? {
            ReadyOrDecision::Ready(prepared) => prepared,
            ReadyOrDecision::Decision(decision) => return Ok(decision),
        };
        let PreparedPhaseExecution::Decode(prepared) = prepared else {
            return guard
                .fail(PowerError::InvalidRequest(
                    "decode executor returned a prefill reservation".to_string(),
                ))
                .await;
        };
        let validation = prepared.validate_at(
            Utc::now(),
            &self.inner.executor_capabilities,
            &self.inner.profile,
        );
        guard.require(validation).await?;
        if prepared.execution_id() != request.execution_id
            || prepared.local_worker_epoch() != self.inner.transfer.local_worker_epoch()
            || prepared.expires_at() != request.expires_at
        {
            return guard
                .fail(PowerError::InvalidRequest(
                    "decode reservation does not match its request identity and lifetime"
                        .to_string(),
                ))
                .await;
        }
        let committed = self
            .inner
            .commit_decode_prepared(request.execution_id, prepared.clone());
        guard.require(committed).await?;
        let target = match self
            .inner
            .transfer
            .prepare_destination(PrepareStateTransfer {
                transfer_id: request.execution_id,
                local_worker_epoch: self.inner.transfer.local_worker_epoch(),
                binding: prepared.binding().clone(),
                destination: prepared.destination().clone(),
                expires_at: request.expires_at,
            })
            .await
        {
            Ok(target) => target,
            Err(error) => return guard.fail(error).await,
        };
        let committed = self
            .inner
            .commit_decode_target(request.execution_id, target.clone());
        guard.require(committed).await?;
        guard.disarm();
        Ok(PhaseDecision::ready(PreparedDecodeTransfer { target }))
    }

    pub async fn execute_prefill(
        &self,
        request: PrefillPhaseRequest,
    ) -> Result<PhaseDecision<PublishedPrefillState>> {
        self.ensure_available(DisaggregatedServingRole::Prefill)?;
        let now = Utc::now();
        self.inner
            .profile
            .validate_state_binding(&request.target.binding)?;
        request
            .target
            .validate_at(now, &self.inner.transfer.capabilities())?;
        if request.target.transfer_id != request.execution_id
            || request.target.expires_at != request.expires_at
        {
            return Err(PowerError::InvalidRequest(
                "prefill request does not match the prepared decode target".to_string(),
            ));
        }
        let command = PreparePhaseExecution {
            execution_id: request.execution_id,
            local_worker_epoch: self.inner.transfer.local_worker_epoch(),
            model: request.model,
            request: request.request,
            expires_at: request.expires_at,
        };
        command.validate_at(now, &self.inner.executor_capabilities, &self.inner.profile)?;
        let deadline = self.deadline(command.expires_at)?;
        let cancellation = self.inner.reserve(command.execution_id, deadline)?;
        let mut guard = RuntimeOperationGuard::new(Arc::clone(&self.inner), command.execution_id);
        let operation = self
            .prepare_phase(command, cancellation.clone(), deadline)
            .await;
        let decision = guard.require(operation).await?;
        let prepared = match ready_or_cleanup(decision, &mut guard).await? {
            ReadyOrDecision::Ready(prepared) => prepared,
            ReadyOrDecision::Decision(decision) => return Ok(decision),
        };
        let PreparedPhaseExecution::Prefill(prepared) = prepared else {
            return guard
                .fail(PowerError::InvalidRequest(
                    "prefill executor returned a decode reservation".to_string(),
                ))
                .await;
        };
        let validation = prepared.validate_at(
            Utc::now(),
            &self.inner.executor_capabilities,
            &self.inner.profile,
        );
        guard.require(validation).await?;
        if prepared.execution_id() != request.execution_id
            || prepared.local_worker_epoch() != self.inner.transfer.local_worker_epoch()
            || prepared.expires_at() != request.expires_at
        {
            return guard
                .fail(PowerError::InvalidRequest(
                    "prefill reservation does not match its request identity and lifetime"
                        .to_string(),
                ))
                .await;
        }
        let committed = self
            .inner
            .commit_prefill_prepared(request.execution_id, prepared.clone());
        guard.require(committed).await?;
        let operation = self
            .execute_phase(
                ExecutePhaseExecution::prefill(prepared.clone()),
                cancellation,
                deadline,
            )
            .await;
        let decision = guard.require(operation).await?;
        let output = match ready_or_cleanup(decision, &mut guard).await? {
            ReadyOrDecision::Ready(output) => output,
            ReadyOrDecision::Decision(decision) => return Ok(decision),
        };
        let PhaseExecutionOutput::Prefill(produced) = output else {
            return guard
                .fail(PowerError::InvalidRequest(
                    "prefill executor returned a decode response stream".to_string(),
                ))
                .await;
        };
        let validation = produced.validate_for(&prepared, &self.inner.profile);
        guard.require(validation).await?;
        let source = match self
            .inner
            .transfer
            .publish_source(PublishStateTransfer {
                local_worker_epoch: self.inner.transfer.local_worker_epoch(),
                source: produced.source().clone(),
                target: request.target,
            })
            .await
        {
            Ok(source) => source,
            Err(error) => return guard.fail(error).await,
        };
        let committed = self.inner.commit_prefill_published(request.execution_id);
        guard.require(committed).await?;
        guard.disarm();
        Ok(PhaseDecision::ready(PublishedPrefillState { source }))
    }

    pub async fn execute_decode(
        &self,
        execution_id: Uuid,
        source: StateTransferSource,
    ) -> Result<PhaseDecision<PhaseResponseStream>> {
        self.ensure_available(DisaggregatedServingRole::Decode)?;
        let deadline = self.deadline(source.expires_at)?;
        let (prepared, target, cancellation) = self.inner.begin_decode(execution_id)?;
        let mut guard = RuntimeOperationGuard::new(Arc::clone(&self.inner), execution_id);
        let validation =
            source.validate_for(&target, Utc::now(), &self.inner.transfer.capabilities());
        guard.require(validation).await?;
        let imported = match ImportedModelState::consume_at(
            self.inner.transfer.as_ref(),
            ConsumeStateTransfer {
                local_worker_epoch: self.inner.transfer.local_worker_epoch(),
                destination: prepared.destination().clone(),
                source,
            },
            Utc::now(),
            &self.inner.profile,
        )
        .await
        {
            Ok(imported) => imported,
            Err(error) => return guard.fail(error).await,
        };
        let execute = guard
            .require(ExecutePhaseExecution::decode(prepared, imported))
            .await?;
        let validation = execute.validate_at(
            Utc::now(),
            &self.inner.executor_capabilities,
            &self.inner.transfer.capabilities(),
            &self.inner.profile,
        );
        guard.require(validation).await?;
        let operation = self.execute_phase(execute, cancellation, deadline).await;
        let decision = guard.require(operation).await?;
        let output = match ready_or_cleanup(decision, &mut guard).await? {
            ReadyOrDecision::Ready(output) => output,
            ReadyOrDecision::Decision(decision) => return Ok(decision),
        };
        let PhaseExecutionOutput::Decode(stream) = output else {
            return guard
                .fail(PowerError::InvalidRequest(
                    "decode executor returned produced prefill state".to_string(),
                ))
                .await;
        };
        let committed = self.inner.commit_decode_stream(execution_id);
        let stream_cancellation = guard.require(committed).await?;
        guard.disarm();
        Ok(PhaseDecision::ready(guarded_stream(
            stream,
            Arc::clone(&self.inner),
            execution_id,
            stream_cancellation,
        )))
    }

    pub async fn abort(&self, execution_id: Uuid) -> Result<()> {
        if execution_id.is_nil() {
            return Err(PowerError::InvalidRequest(
                "distributed execution identifier is invalid".to_string(),
            ));
        }
        let Some(lease) = self.inner.take_lease(execution_id)? else {
            return Ok(());
        };
        if RuntimeInner::spawn_cleanup(Arc::clone(&self.inner), execution_id, lease).await {
            Ok(())
        } else {
            Err(PowerError::BackendNotAvailable(
                "distributed execution cleanup was not confirmed".to_string(),
            ))
        }
    }

    fn ensure_available(&self, role: DisaggregatedServingRole) -> Result<()> {
        if self.role != role {
            return Err(PowerError::InvalidRequest(format!(
                "distributed serving operation requires the {role:?} process role"
            )));
        }
        if !self.accepts_work() {
            return Err(PowerError::BackendNotAvailable(
                "distributed serving runtime cannot accept work".to_string(),
            ));
        }
        Ok(())
    }

    fn deadline(&self, expires_at: DateTime<Utc>) -> Result<Instant> {
        let remaining = (expires_at - Utc::now()).to_std().map_err(|_| {
            PowerError::InvalidRequest("distributed execution has expired".to_string())
        })?;
        let remaining = remaining.min(self.transfer_timeout);
        if remaining.is_zero() {
            return Err(PowerError::InvalidRequest(
                "distributed execution has expired".to_string(),
            ));
        }
        Instant::now().checked_add(remaining).ok_or_else(|| {
            PowerError::InvalidRequest(
                "distributed execution deadline is outside the monotonic range".to_string(),
            )
        })
    }
}
