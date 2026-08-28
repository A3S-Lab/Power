use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use chrono::{DateTime, Utc};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::error::{PowerError, Result};

use super::super::{
    AbortStateTransfer, ConsumeStateTransfer, DisaggregatedServingRole, PrepareStateTransfer,
    PublishStateTransfer, ServingExecutionProfile, StateTransferBinding, StateTransferCapabilities,
    StateTransferProtocol, StateTransferReceipt, StateTransferService, StateTransferSource,
    StateTransferTarget,
};

pub(super) struct Inner {
    pub(super) profile: ServingExecutionProfile,
    pub(super) delegate: Arc<dyn StateTransferService>,
    pub(super) local_worker_epoch: Uuid,
    pub(super) role: DisaggregatedServingRole,
    pub(super) protocol: StateTransferProtocol,
    pub(super) transfer_timeout: std::time::Duration,
    pub(super) cancellation_timeout: std::time::Duration,
    pub(super) capabilities: StateTransferCapabilities,
    pub(super) capacity: Arc<Semaphore>,
    pub(super) leases: Mutex<HashMap<Uuid, TransferLease>>,
    pub(super) tainted: AtomicBool,
    pub(super) prepared_destinations: AtomicU64,
    pub(super) published_sources: AtomicU64,
    pub(super) completed_consumes: AtomicU64,
    pub(super) aborted_transfers: AtomicU64,
    pub(super) timeout_expirations: AtomicU64,
    pub(super) capacity_rejections: AtomicU64,
    pub(super) cleanup_failures: AtomicU64,
}

pub(super) struct TransferLease {
    pub(super) command: LeaseCommand,
    pub(super) state: LeaseState,
    pub(super) cancellation: CancellationToken,
    pub(super) expiry_cancellation: CancellationToken,
    pub(super) _permit: OwnedSemaphorePermit,
}

pub(super) enum LeaseCommand {
    Destination(PrepareStateTransfer),
    Source(PublishStateTransfer),
}

pub(super) enum LeaseState {
    Preparing,
    DestinationReady(StateTransferTarget),
    Consuming,
    SourcePublished(StateTransferSource),
}

pub(super) enum OperationOutcome<T> {
    Completed(Result<T>),
    Cancelled,
    TimedOut,
}

pub(super) async fn wait_operation<T, F>(
    cancellation: CancellationToken,
    deadline: Instant,
    operation: F,
) -> OperationOutcome<T>
where
    F: Future<Output = Result<T>>,
{
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => OperationOutcome::Cancelled,
        _ = tokio::time::sleep_until(deadline) => OperationOutcome::TimedOut,
        result = operation => OperationOutcome::Completed(result),
    }
}

pub(super) struct OperationDropGuard {
    inner: Arc<Inner>,
    transfer_id: Uuid,
    armed: bool,
}

impl OperationDropGuard {
    pub(super) fn new(inner: Arc<Inner>, transfer_id: Uuid) -> Self {
        Self {
            inner,
            transfer_id,
            armed: true,
        }
    }

    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }

    pub(super) async fn fail_after_cleanup<T>(&mut self, error: PowerError) -> Result<T> {
        let confirmed = self.spawn_cleanup().await;
        if confirmed {
            Err(error)
        } else {
            Err(PowerError::BackendNotAvailable(
                "state-transfer operation failed and cleanup was not confirmed".to_string(),
            ))
        }
    }

    pub(super) async fn timeout_after_cleanup<T>(&mut self) -> Result<T> {
        let confirmed = self.spawn_cleanup().await;
        let message = if confirmed {
            "state-transfer operation exceeded the immutable profile timeout"
        } else {
            "state-transfer operation timed out and cleanup was not confirmed"
        };
        Err(PowerError::BackendNotAvailable(message.to_string()))
    }

    async fn spawn_cleanup(&mut self) -> bool {
        self.armed = false;
        let Some(lease) = self.inner.take_lease(self.transfer_id) else {
            return true;
        };
        lease.cancellation.cancel();
        Inner::spawn_cleanup(Arc::clone(&self.inner), self.transfer_id, lease).await
    }
}

impl Drop for OperationDropGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let Some(lease) = self.inner.take_lease(self.transfer_id) else {
            return;
        };
        lease.cancellation.cancel();
        let inner = Arc::clone(&self.inner);
        let transfer_id = self.transfer_id;
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => {
                runtime.spawn(async move {
                    let _ = Inner::spawn_cleanup(inner, transfer_id, lease).await;
                });
            }
            Err(_) => {
                inner.mark_cleanup_failure();
                drop(lease);
            }
        }
    }
}

impl Inner {
    pub(super) fn profile_validate_binding(&self, binding: &StateTransferBinding) -> Result<()> {
        self.profile.validate_state_binding(binding)
    }

    pub(super) fn replay_destination(
        &self,
        command: &PrepareStateTransfer,
    ) -> Result<Option<StateTransferTarget>> {
        let leases = self.leases()?;
        Self::replay_destination_locked(&leases, command)
    }

    pub(super) fn replay_destination_locked(
        leases: &HashMap<Uuid, TransferLease>,
        command: &PrepareStateTransfer,
    ) -> Result<Option<StateTransferTarget>> {
        let Some(lease) = leases.get(&command.transfer_id) else {
            return Ok(None);
        };
        match (&lease.command, &lease.state) {
            (LeaseCommand::Destination(existing), LeaseState::DestinationReady(target))
                if existing == command =>
            {
                Ok(Some(target.clone()))
            }
            (LeaseCommand::Destination(existing), LeaseState::Preparing) if existing == command => {
                Err(PowerError::BackendNotAvailable(
                    "state-transfer destination preparation is already in progress".to_string(),
                ))
            }
            _ => Err(PowerError::InvalidRequest(
                "state-transfer identifier is already bound to another local lease".to_string(),
            )),
        }
    }

    pub(super) fn replay_source(
        &self,
        command: &PublishStateTransfer,
    ) -> Result<Option<StateTransferSource>> {
        let leases = self.leases()?;
        Self::replay_source_locked(&leases, command)
    }

    pub(super) fn replay_source_locked(
        leases: &HashMap<Uuid, TransferLease>,
        command: &PublishStateTransfer,
    ) -> Result<Option<StateTransferSource>> {
        let Some(lease) = leases.get(&command.target.transfer_id) else {
            return Ok(None);
        };
        match (&lease.command, &lease.state) {
            (LeaseCommand::Source(existing), LeaseState::SourcePublished(source))
                if existing == command =>
            {
                Ok(Some(source.clone()))
            }
            (LeaseCommand::Source(existing), LeaseState::Preparing) if existing == command => {
                Err(PowerError::BackendNotAvailable(
                    "state-transfer source publication is already in progress".to_string(),
                ))
            }
            _ => Err(PowerError::InvalidRequest(
                "state-transfer identifier is already bound to another local lease".to_string(),
            )),
        }
    }

    pub(super) fn commit_destination(
        &self,
        transfer_id: Uuid,
        target: StateTransferTarget,
    ) -> Result<()> {
        let mut leases = self.leases()?;
        let lease = leases.get_mut(&transfer_id).ok_or_else(|| {
            PowerError::BackendNotAvailable(
                "state-transfer destination lease expired before commit".to_string(),
            )
        })?;
        if !matches!(lease.state, LeaseState::Preparing) {
            return Err(PowerError::InvalidRequest(
                "state-transfer destination lease has an invalid transition".to_string(),
            ));
        }
        lease.state = LeaseState::DestinationReady(target);
        Ok(())
    }

    pub(super) fn commit_source(
        &self,
        transfer_id: Uuid,
        source: StateTransferSource,
    ) -> Result<()> {
        let mut leases = self.leases()?;
        let lease = leases.get_mut(&transfer_id).ok_or_else(|| {
            PowerError::BackendNotAvailable(
                "state-transfer source lease expired before commit".to_string(),
            )
        })?;
        if !matches!(lease.state, LeaseState::Preparing) {
            return Err(PowerError::InvalidRequest(
                "state-transfer source lease has an invalid transition".to_string(),
            ));
        }
        lease.state = LeaseState::SourcePublished(source);
        Ok(())
    }

    pub(super) fn begin_consume(
        &self,
        command: &ConsumeStateTransfer,
        now: DateTime<Utc>,
    ) -> Result<CancellationToken> {
        let mut leases = self.leases()?;
        let lease = leases.get_mut(&command.source.transfer_id).ok_or_else(|| {
            PowerError::InvalidRequest(
                "state-transfer consume has no prepared local destination".to_string(),
            )
        })?;
        let (prepared, target) = match (&lease.command, &lease.state) {
            (LeaseCommand::Destination(prepared), LeaseState::DestinationReady(target)) => {
                (prepared, target)
            }
            (_, LeaseState::Consuming) => {
                return Err(PowerError::BackendNotAvailable(
                    "state-transfer consume is already in progress".to_string(),
                ));
            }
            _ => {
                return Err(PowerError::InvalidRequest(
                    "state-transfer lease is not a prepared destination".to_string(),
                ));
            }
        };
        command
            .source
            .validate_for(target, now, &self.capabilities)?;
        if command.local_worker_epoch != prepared.local_worker_epoch
            || command.destination != prepared.destination
            || command.source.binding != prepared.binding
        {
            return Err(PowerError::InvalidRequest(
                "state-transfer source does not match its local destination lease".to_string(),
            ));
        }
        lease.state = LeaseState::Consuming;
        Ok(lease.cancellation.clone())
    }

    pub(super) fn finish_consume(&self, transfer_id: Uuid) -> Result<()> {
        let mut leases = self.leases()?;
        let lease = leases.get(&transfer_id).ok_or_else(|| {
            PowerError::BackendNotAvailable(
                "state-transfer destination lease expired before completion".to_string(),
            )
        })?;
        if !matches!(lease.state, LeaseState::Consuming) {
            return Err(PowerError::InvalidRequest(
                "state-transfer destination lease has an invalid completion".to_string(),
            ));
        }
        if let Some(lease) = leases.remove(&transfer_id) {
            lease.expiry_cancellation.cancel();
        }
        Ok(())
    }

    pub(super) fn validate_prepared_target(
        &self,
        command: &PrepareStateTransfer,
        target: &StateTransferTarget,
        started_at: DateTime<Utc>,
        verified_at: DateTime<Utc>,
    ) -> Result<()> {
        target.validate_at(verified_at, &self.capabilities)?;
        if target.transfer_id != command.transfer_id
            || target.destination_worker_epoch != command.local_worker_epoch
            || target.binding != command.binding
            || target.protocol != self.protocol
            || target.prepared_at < started_at
            || target.prepared_at > verified_at
            || target.expires_at > command.expires_at
        {
            return Err(PowerError::InvalidRequest(
                "state-transfer adapter returned a mismatched destination".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn validate_published_source(
        &self,
        command: &PublishStateTransfer,
        source: &StateTransferSource,
        started_at: DateTime<Utc>,
        verified_at: DateTime<Utc>,
    ) -> Result<()> {
        source.validate_for(&command.target, verified_at, &self.capabilities)?;
        if source.source_worker_epoch != command.local_worker_epoch
            || source.published_at < started_at
            || source.published_at > verified_at
        {
            return Err(PowerError::InvalidRequest(
                "state-transfer adapter returned a mismatched source".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn validate_receipt(
        &self,
        command: &ConsumeStateTransfer,
        receipt: &StateTransferReceipt,
        started_at: DateTime<Utc>,
        verified_at: DateTime<Utc>,
    ) -> Result<()> {
        receipt.validate_for(&command.source, &self.capabilities)?;
        if receipt.destination_worker_epoch != self.local_worker_epoch
            || receipt.completed_at < started_at
            || receipt.completed_at > verified_at
        {
            return Err(PowerError::InvalidRequest(
                "state-transfer receipt is outside the local consume attempt".to_string(),
            ));
        }
        Ok(())
    }

    fn leases(&self) -> Result<MutexGuard<'_, HashMap<Uuid, TransferLease>>> {
        self.leases.lock().map_err(|_| {
            self.tainted.store(true, Ordering::Release);
            PowerError::BackendNotAvailable(
                "state-transfer lifecycle lock is unavailable".to_string(),
            )
        })
    }

    pub(super) fn take_lease(&self, transfer_id: Uuid) -> Option<TransferLease> {
        match self.leases.lock() {
            Ok(mut leases) => leases.remove(&transfer_id).inspect(|lease| {
                lease.expiry_cancellation.cancel();
            }),
            Err(_) => {
                self.mark_cleanup_failure();
                None
            }
        }
    }

    pub(super) async fn spawn_cleanup(
        inner: Arc<Self>,
        transfer_id: Uuid,
        lease: TransferLease,
    ) -> bool {
        let command = AbortStateTransfer {
            transfer_id,
            local_worker_epoch: inner.local_worker_epoch,
        };
        let timeout = inner.cancellation_timeout;
        let task_inner = Arc::clone(&inner);
        let cleanup = tokio::spawn(async move {
            let result = tokio::time::timeout(timeout, task_inner.delegate.abort(command)).await;
            drop(lease);
            match result {
                Ok(Ok(())) => {
                    task_inner.aborted_transfers.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Ok(Err(_)) | Err(_) => {
                    task_inner.mark_cleanup_failure();
                    false
                }
            }
        });
        match cleanup.await {
            Ok(confirmed) => confirmed,
            Err(_) => {
                inner.mark_cleanup_failure();
                false
            }
        }
    }

    pub(super) fn mark_cleanup_failure(&self) {
        self.cleanup_failures.fetch_add(1, Ordering::Relaxed);
        self.tainted.store(true, Ordering::Release);
    }
}
