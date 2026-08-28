use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::error::{PowerError, Result};

use super::{
    AbortStateTransfer, ConsumeStateTransfer, DisaggregatedServingRole, PrepareStateTransfer,
    PublishStateTransfer, ServingExecutionProfile, ServingPhase, StateTransferCapabilities,
    StateTransferReceipt, StateTransferService, StateTransferSource, StateTransferTarget,
    TransferHealth,
};

mod lifecycle;

use lifecycle::{
    wait_operation, Inner, LeaseCommand, LeaseState, OperationDropGuard, OperationOutcome,
    TransferLease,
};

/// Content-free counters for one process-local state-transfer lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StateTransferRuntimeSnapshot {
    pub active_transfers: u32,
    pub maximum_inflight_transfers: u32,
    pub prepared_destinations: u64,
    pub published_sources: u64,
    pub completed_consumes: u64,
    pub aborted_transfers: u64,
    pub timeout_expirations: u64,
    pub capacity_rejections: u64,
    pub cleanup_failures: u64,
}

/// Process-local guard around a concrete state-transfer data-path adapter.
///
/// The wrapped adapter still owns memory registration and transport. This
/// guard binds it to one immutable Power process generation and adds the
/// bounded, idempotent lifecycle that every concrete adapter must obey.
#[derive(Clone)]
pub struct BoundedStateTransferService {
    inner: Arc<Inner>,
}

impl fmt::Debug for BoundedStateTransferService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundedStateTransferService")
            .field("local_worker_epoch", &self.inner.local_worker_epoch)
            .field("capabilities", &self.inner.capabilities)
            .field("snapshot", &self.snapshot())
            .finish_non_exhaustive()
    }
}

impl BoundedStateTransferService {
    /// Bind one concrete adapter to an immutable profile and process epoch.
    pub fn new(
        profile: ServingExecutionProfile,
        local_worker_epoch: Uuid,
        delegate: Arc<dyn StateTransferService>,
    ) -> Result<Self> {
        profile.validate()?;
        if local_worker_epoch.is_nil() {
            return Err(PowerError::Config(
                "state-transfer runtime requires a non-nil worker epoch".to_string(),
            ));
        }
        let ServingExecutionProfile::PrefillDecode { execution } = &profile else {
            return Err(PowerError::Config(
                "aggregated serving cannot create a state-transfer runtime".to_string(),
            ));
        };
        let delegate_capabilities = delegate.capabilities();
        profile.validate_state_transfer_capabilities(&delegate_capabilities)?;
        if matches!(delegate.health(), TransferHealth::Unsupported) {
            return Err(PowerError::Config(
                "state-transfer runtime cannot wrap an unsupported adapter".to_string(),
            ));
        }

        let capabilities = StateTransferCapabilities {
            execution_profile_sha256: profile.sha256()?,
            phases: vec![ServingPhase::from(execution.role)],
            protocols: vec![execution.protocol],
            max_transfer_bytes: execution.max_state_bytes,
            max_inflight_transfers: execution.max_inflight_transfers,
        };
        capabilities.validate()?;
        let maximum = usize::try_from(execution.max_inflight_transfers).map_err(|_| {
            PowerError::Config("state-transfer concurrency does not fit this platform".to_string())
        })?;
        Ok(Self {
            inner: Arc::new(Inner {
                profile: profile.clone(),
                delegate,
                local_worker_epoch,
                role: execution.role,
                protocol: execution.protocol,
                transfer_timeout: std::time::Duration::from_millis(execution.transfer_timeout_ms),
                cancellation_timeout: std::time::Duration::from_millis(
                    execution.cancellation_timeout_ms,
                ),
                capabilities,
                capacity: Arc::new(Semaphore::new(maximum)),
                leases: Mutex::new(HashMap::with_capacity(maximum)),
                tainted: AtomicBool::new(false),
                prepared_destinations: AtomicU64::new(0),
                published_sources: AtomicU64::new(0),
                completed_consumes: AtomicU64::new(0),
                aborted_transfers: AtomicU64::new(0),
                timeout_expirations: AtomicU64::new(0),
                capacity_rejections: AtomicU64::new(0),
                cleanup_failures: AtomicU64::new(0),
            }),
        })
    }

    /// Return the only worker epoch accepted by this runtime instance.
    pub fn local_worker_epoch(&self) -> Uuid {
        self.inner.local_worker_epoch
    }

    /// Return content-free bounded lifecycle counters.
    pub fn snapshot(&self) -> StateTransferRuntimeSnapshot {
        let active_transfers = match self.inner.leases.lock() {
            Ok(leases) => u32::try_from(leases.len()).unwrap_or(u32::MAX),
            Err(poisoned) => {
                self.inner.tainted.store(true, Ordering::Release);
                u32::try_from(poisoned.into_inner().len()).unwrap_or(u32::MAX)
            }
        };
        StateTransferRuntimeSnapshot {
            active_transfers,
            maximum_inflight_transfers: self.inner.capabilities.max_inflight_transfers,
            prepared_destinations: self.inner.prepared_destinations.load(Ordering::Relaxed),
            published_sources: self.inner.published_sources.load(Ordering::Relaxed),
            completed_consumes: self.inner.completed_consumes.load(Ordering::Relaxed),
            aborted_transfers: self.inner.aborted_transfers.load(Ordering::Relaxed),
            timeout_expirations: self.inner.timeout_expirations.load(Ordering::Relaxed),
            capacity_rejections: self.inner.capacity_rejections.load(Ordering::Relaxed),
            cleanup_failures: self.inner.cleanup_failures.load(Ordering::Relaxed),
        }
    }

    fn ensure_available(&self) -> Result<()> {
        if self.inner.tainted.load(Ordering::Acquire) {
            return Err(PowerError::BackendNotAvailable(
                "state-transfer runtime cleanup is unconfirmed".to_string(),
            ));
        }
        if !matches!(
            self.inner.delegate.health(),
            TransferHealth::Ready | TransferHealth::Degraded
        ) {
            return Err(PowerError::BackendNotAvailable(
                "state-transfer adapter cannot accept work".to_string(),
            ));
        }
        Ok(())
    }

    fn ensure_phase(&self, expected: DisaggregatedServingRole) -> Result<()> {
        if self.inner.role != expected {
            return Err(PowerError::InvalidRequest(format!(
                "state-transfer operation requires the {expected:?} process role"
            )));
        }
        Ok(())
    }

    fn ensure_local_epoch(&self, epoch: Uuid) -> Result<()> {
        if epoch.is_nil() || epoch != self.inner.local_worker_epoch {
            return Err(PowerError::InvalidRequest(
                "state-transfer command belongs to a different process epoch".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_expiry(&self, now: DateTime<Utc>, expires_at: DateTime<Utc>) -> Result<()> {
        let remaining = (expires_at - now).to_std().map_err(|_| {
            PowerError::InvalidRequest("state-transfer command has expired".to_string())
        })?;
        if remaining.is_zero() || remaining > self.inner.transfer_timeout {
            return Err(PowerError::InvalidRequest(
                "state-transfer expiry exceeds the immutable profile timeout".to_string(),
            ));
        }
        Ok(())
    }

    fn try_permit(&self) -> Result<OwnedSemaphorePermit> {
        match Arc::clone(&self.inner.capacity).try_acquire_owned() {
            Ok(permit) => Ok(permit),
            Err(TryAcquireError::NoPermits) => {
                self.inner
                    .capacity_rejections
                    .fetch_add(1, Ordering::Relaxed);
                Err(PowerError::BackendNotAvailable(
                    "state-transfer capacity is exhausted".to_string(),
                ))
            }
            Err(TryAcquireError::Closed) => Err(PowerError::BackendNotAvailable(
                "state-transfer runtime is closed".to_string(),
            )),
        }
    }

    fn leases(&self) -> Result<MutexGuard<'_, HashMap<Uuid, TransferLease>>> {
        self.inner.leases.lock().map_err(|_| {
            self.inner.tainted.store(true, Ordering::Release);
            PowerError::BackendNotAvailable(
                "state-transfer lifecycle lock is unavailable".to_string(),
            )
        })
    }

    fn operation_deadline(&self, expires_at: DateTime<Utc>) -> Result<Instant> {
        let remaining = (expires_at - Utc::now()).to_std().map_err(|_| {
            PowerError::InvalidRequest("state-transfer command has expired".to_string())
        })?;
        let remaining = remaining.min(self.inner.transfer_timeout);
        if remaining.is_zero() {
            return Err(PowerError::InvalidRequest(
                "state-transfer command has expired".to_string(),
            ));
        }
        Instant::now().checked_add(remaining).ok_or_else(|| {
            PowerError::InvalidRequest(
                "state-transfer deadline is outside the monotonic range".to_string(),
            )
        })
    }

    fn spawn_expiry(&self, transfer_id: Uuid, deadline: Instant, cancellation: CancellationToken) {
        let inner = Arc::clone(&self.inner);
        tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => return,
                _ = tokio::time::sleep_until(deadline) => {}
            }
            let Some(lease) = inner.take_lease(transfer_id) else {
                return;
            };
            inner.timeout_expirations.fetch_add(1, Ordering::Relaxed);
            lease.cancellation.cancel();
            let _ = Inner::spawn_cleanup(inner, transfer_id, lease).await;
        });
    }
}

#[async_trait]
impl StateTransferService for BoundedStateTransferService {
    fn capabilities(&self) -> StateTransferCapabilities {
        self.inner.capabilities.clone()
    }

    fn health(&self) -> TransferHealth {
        if self.inner.tainted.load(Ordering::Acquire) {
            TransferHealth::Unavailable
        } else {
            self.inner.delegate.health()
        }
    }

    async fn prepare_destination(
        &self,
        command: PrepareStateTransfer,
    ) -> Result<StateTransferTarget> {
        self.ensure_available()?;
        self.ensure_phase(DisaggregatedServingRole::Decode)?;
        self.ensure_local_epoch(command.local_worker_epoch)?;
        let started_at = Utc::now();
        self.inner.profile_validate_binding(&command.binding)?;
        command.validate_at(started_at, &self.inner.capabilities)?;
        self.validate_expiry(started_at, command.expires_at)?;
        let deadline = self.operation_deadline(command.expires_at)?;

        if let Some(target) = self.inner.replay_destination(&command)? {
            return Ok(target);
        }
        let permit = self.try_permit()?;
        let cancellation = CancellationToken::new();
        let expiry_cancellation = CancellationToken::new();
        {
            let mut leases = self.leases()?;
            if let Some(target) = Inner::replay_destination_locked(&leases, &command)? {
                return Ok(target);
            }
            leases.insert(
                command.transfer_id,
                TransferLease {
                    command: LeaseCommand::Destination(command.clone()),
                    state: LeaseState::Preparing,
                    cancellation: cancellation.clone(),
                    expiry_cancellation: expiry_cancellation.clone(),
                    _permit: permit,
                },
            );
        }
        self.spawn_expiry(command.transfer_id, deadline, expiry_cancellation);
        let mut guard = OperationDropGuard::new(Arc::clone(&self.inner), command.transfer_id);
        let outcome = wait_operation(
            cancellation,
            deadline,
            self.inner.delegate.prepare_destination(command.clone()),
        )
        .await;
        let target = match outcome {
            OperationOutcome::Completed(Ok(target)) => target,
            OperationOutcome::Completed(Err(error)) => {
                return guard.fail_after_cleanup(error).await;
            }
            OperationOutcome::Cancelled => {
                guard.disarm();
                return Err(PowerError::BackendNotAvailable(
                    "state-transfer destination preparation was cancelled".to_string(),
                ));
            }
            OperationOutcome::TimedOut => {
                self.inner
                    .timeout_expirations
                    .fetch_add(1, Ordering::Relaxed);
                return guard.timeout_after_cleanup().await;
            }
        };
        let verified_at = Utc::now();
        if let Err(error) =
            self.inner
                .validate_prepared_target(&command, &target, started_at, verified_at)
        {
            return guard.fail_after_cleanup(error).await;
        }
        self.inner
            .commit_destination(command.transfer_id, target.clone())?;
        self.inner
            .prepared_destinations
            .fetch_add(1, Ordering::Relaxed);
        guard.disarm();
        Ok(target)
    }

    async fn publish_source(&self, command: PublishStateTransfer) -> Result<StateTransferSource> {
        self.ensure_available()?;
        self.ensure_phase(DisaggregatedServingRole::Prefill)?;
        self.ensure_local_epoch(command.local_worker_epoch)?;
        let started_at = Utc::now();
        self.inner
            .profile_validate_binding(&command.target.binding)?;
        command.validate_at(started_at, &self.inner.capabilities)?;
        self.validate_expiry(started_at, command.target.expires_at)?;
        let deadline = self.operation_deadline(command.target.expires_at)?;

        if let Some(source) = self.inner.replay_source(&command)? {
            return Ok(source);
        }
        let permit = self.try_permit()?;
        let cancellation = CancellationToken::new();
        let expiry_cancellation = CancellationToken::new();
        {
            let mut leases = self.leases()?;
            if let Some(source) = Inner::replay_source_locked(&leases, &command)? {
                return Ok(source);
            }
            leases.insert(
                command.target.transfer_id,
                TransferLease {
                    command: LeaseCommand::Source(command.clone()),
                    state: LeaseState::Preparing,
                    cancellation: cancellation.clone(),
                    expiry_cancellation: expiry_cancellation.clone(),
                    _permit: permit,
                },
            );
        }
        self.spawn_expiry(command.target.transfer_id, deadline, expiry_cancellation);
        let mut guard =
            OperationDropGuard::new(Arc::clone(&self.inner), command.target.transfer_id);
        let outcome = wait_operation(
            cancellation,
            deadline,
            self.inner.delegate.publish_source(command.clone()),
        )
        .await;
        let source = match outcome {
            OperationOutcome::Completed(Ok(source)) => source,
            OperationOutcome::Completed(Err(error)) => {
                return guard.fail_after_cleanup(error).await;
            }
            OperationOutcome::Cancelled => {
                guard.disarm();
                return Err(PowerError::BackendNotAvailable(
                    "state-transfer source publication was cancelled".to_string(),
                ));
            }
            OperationOutcome::TimedOut => {
                self.inner
                    .timeout_expirations
                    .fetch_add(1, Ordering::Relaxed);
                return guard.timeout_after_cleanup().await;
            }
        };
        let verified_at = Utc::now();
        if let Err(error) =
            self.inner
                .validate_published_source(&command, &source, started_at, verified_at)
        {
            return guard.fail_after_cleanup(error).await;
        }
        self.inner
            .commit_source(command.target.transfer_id, source.clone())?;
        self.inner.published_sources.fetch_add(1, Ordering::Relaxed);
        guard.disarm();
        Ok(source)
    }

    async fn consume_source(&self, command: ConsumeStateTransfer) -> Result<StateTransferReceipt> {
        self.ensure_available()?;
        self.ensure_phase(DisaggregatedServingRole::Decode)?;
        self.ensure_local_epoch(command.local_worker_epoch)?;
        let started_at = Utc::now();
        self.inner
            .profile_validate_binding(&command.source.binding)?;
        command.validate_at(started_at, &self.inner.capabilities)?;
        self.validate_expiry(started_at, command.source.expires_at)?;
        let deadline = self.operation_deadline(command.source.expires_at)?;
        let cancellation = self.inner.begin_consume(&command, started_at)?;
        let mut guard =
            OperationDropGuard::new(Arc::clone(&self.inner), command.source.transfer_id);
        let outcome = wait_operation(
            cancellation,
            deadline,
            self.inner.delegate.consume_source(command.clone()),
        )
        .await;
        let receipt = match outcome {
            OperationOutcome::Completed(Ok(receipt)) => receipt,
            OperationOutcome::Completed(Err(error)) => {
                return guard.fail_after_cleanup(error).await;
            }
            OperationOutcome::Cancelled => {
                guard.disarm();
                return Err(PowerError::BackendNotAvailable(
                    "state-transfer consume was cancelled".to_string(),
                ));
            }
            OperationOutcome::TimedOut => {
                self.inner
                    .timeout_expirations
                    .fetch_add(1, Ordering::Relaxed);
                return guard.timeout_after_cleanup().await;
            }
        };
        let verified_at = Utc::now();
        if let Err(error) = self
            .inner
            .validate_receipt(&command, &receipt, started_at, verified_at)
        {
            return guard.fail_after_cleanup(error).await;
        }
        self.inner.finish_consume(command.source.transfer_id)?;
        self.inner
            .completed_consumes
            .fetch_add(1, Ordering::Relaxed);
        guard.disarm();
        Ok(receipt)
    }

    async fn abort(&self, command: AbortStateTransfer) -> Result<()> {
        command.validate()?;
        self.ensure_local_epoch(command.local_worker_epoch)?;
        let Some(lease) = self.inner.take_lease(command.transfer_id) else {
            return Ok(());
        };
        lease.cancellation.cancel();
        if Inner::spawn_cleanup(Arc::clone(&self.inner), command.transfer_id, lease).await {
            Ok(())
        } else {
            Err(PowerError::BackendNotAvailable(
                "state-transfer cleanup was not confirmed".to_string(),
            ))
        }
    }
}
