use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};

use futures::Stream;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::error::{PowerError, Result};

use super::super::{
    AbortPhaseExecution, AbortStateTransfer, BoundedStateTransferService,
    PhaseExecutorCapabilities, PhaseResponseChunk, PhaseResponseStream, PreparedDecodePhase,
    PreparedPrefillPhase, ServingExecutionProfile, ServingPhaseExecutor, StateTransferService,
    StateTransferTarget,
};

pub(super) struct RuntimeInner {
    pub(super) profile: ServingExecutionProfile,
    pub(super) transfer: Arc<BoundedStateTransferService>,
    pub(super) executor: Arc<dyn ServingPhaseExecutor>,
    pub(super) executor_capabilities: PhaseExecutorCapabilities,
    pub(super) cancellation_timeout: std::time::Duration,
    pub(super) capacity: Arc<Semaphore>,
    pub(super) leases: Mutex<HashMap<Uuid, LocalExecutionLease>>,
    pub(super) tainted: AtomicBool,
}

pub(super) struct LocalExecutionLease {
    state: LocalExecutionState,
    operation_cancellation: CancellationToken,
    expiry_cancellation: CancellationToken,
    _permit: OwnedSemaphorePermit,
}

enum LocalExecutionState {
    Preparing,
    PrefillPrepared(PreparedPrefillPhase),
    PrefillPublished(PreparedPrefillPhase),
    DecodePhasePrepared(PreparedDecodePhase),
    DecodePrepared {
        prepared: Box<PreparedDecodePhase>,
        target: StateTransferTarget,
    },
    DecodeExecuting(PreparedDecodePhase),
    DecodeStreaming(PreparedDecodePhase),
}

pub(super) enum RuntimeOperation<T> {
    Completed(Result<T>),
    Cancelled,
    TimedOut,
}

pub(super) async fn wait_operation<T, F>(
    cancellation: CancellationToken,
    deadline: Instant,
    operation: F,
) -> RuntimeOperation<T>
where
    F: Future<Output = Result<T>>,
{
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => RuntimeOperation::Cancelled,
        _ = tokio::time::sleep_until(deadline) => RuntimeOperation::TimedOut,
        result = operation => RuntimeOperation::Completed(result),
    }
}

impl RuntimeInner {
    pub(super) fn reserve(
        self: &Arc<Self>,
        execution_id: Uuid,
        deadline: Instant,
    ) -> Result<CancellationToken> {
        let permit = match Arc::clone(&self.capacity).try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::NoPermits) => {
                return Err(PowerError::BackendNotAvailable(
                    "distributed phase capacity is exhausted".to_string(),
                ));
            }
            Err(TryAcquireError::Closed) => {
                return Err(PowerError::BackendNotAvailable(
                    "distributed phase runtime is closed".to_string(),
                ));
            }
        };
        let operation_cancellation = CancellationToken::new();
        let expiry_cancellation = CancellationToken::new();
        {
            let mut leases = self.leases()?;
            if leases.contains_key(&execution_id) {
                return Err(PowerError::InvalidRequest(
                    "distributed execution identifier is already active".to_string(),
                ));
            }
            leases.insert(
                execution_id,
                LocalExecutionLease {
                    state: LocalExecutionState::Preparing,
                    operation_cancellation: operation_cancellation.clone(),
                    expiry_cancellation: expiry_cancellation.clone(),
                    _permit: permit,
                },
            );
        }
        let inner = Arc::clone(self);
        tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = expiry_cancellation.cancelled() => return,
                _ = tokio::time::sleep_until(deadline) => {}
            }
            let Ok(Some(lease)) = inner.take_lease(execution_id) else {
                return;
            };
            lease.operation_cancellation.cancel();
            let _ = Self::spawn_cleanup(inner, execution_id, lease).await;
        });
        Ok(operation_cancellation)
    }

    pub(super) fn commit_prefill_prepared(
        &self,
        execution_id: Uuid,
        prepared: PreparedPrefillPhase,
    ) -> Result<()> {
        let mut leases = self.leases()?;
        let lease = preparing_lease(&mut leases, execution_id)?;
        lease.state = LocalExecutionState::PrefillPrepared(prepared);
        Ok(())
    }

    pub(super) fn commit_prefill_published(&self, execution_id: Uuid) -> Result<()> {
        let mut leases = self.leases()?;
        let lease = leases.get_mut(&execution_id).ok_or_else(missing_lease)?;
        let LocalExecutionState::PrefillPrepared(prepared) = &lease.state else {
            return Err(invalid_transition("publish prefill state"));
        };
        lease.state = LocalExecutionState::PrefillPublished(prepared.clone());
        Ok(())
    }

    pub(super) fn commit_decode_prepared(
        &self,
        execution_id: Uuid,
        prepared: PreparedDecodePhase,
    ) -> Result<()> {
        let mut leases = self.leases()?;
        let lease = preparing_lease(&mut leases, execution_id)?;
        lease.state = LocalExecutionState::DecodePhasePrepared(prepared);
        Ok(())
    }

    pub(super) fn commit_decode_target(
        &self,
        execution_id: Uuid,
        target: StateTransferTarget,
    ) -> Result<()> {
        let mut leases = self.leases()?;
        let lease = leases.get_mut(&execution_id).ok_or_else(missing_lease)?;
        let LocalExecutionState::DecodePhasePrepared(prepared) = &lease.state else {
            return Err(invalid_transition("commit decode transfer target"));
        };
        let prepared = Box::new(prepared.clone());
        lease.state = LocalExecutionState::DecodePrepared { prepared, target };
        Ok(())
    }

    pub(super) fn begin_decode(
        &self,
        execution_id: Uuid,
    ) -> Result<(PreparedDecodePhase, StateTransferTarget, CancellationToken)> {
        let mut leases = self.leases()?;
        let lease = leases.get_mut(&execution_id).ok_or_else(missing_lease)?;
        let LocalExecutionState::DecodePrepared { prepared, target } = &lease.state else {
            return Err(invalid_transition("start decode"));
        };
        let prepared = prepared.as_ref().clone();
        let target = target.clone();
        lease.state = LocalExecutionState::DecodeExecuting(prepared.clone());
        Ok((prepared, target, lease.operation_cancellation.clone()))
    }

    pub(super) fn commit_decode_stream(&self, execution_id: Uuid) -> Result<CancellationToken> {
        let mut leases = self.leases()?;
        let lease = leases.get_mut(&execution_id).ok_or_else(missing_lease)?;
        let LocalExecutionState::DecodeExecuting(prepared) = &lease.state else {
            return Err(invalid_transition("return decode stream"));
        };
        lease.state = LocalExecutionState::DecodeStreaming(prepared.clone());
        lease.expiry_cancellation.cancel();
        Ok(lease.operation_cancellation.clone())
    }

    pub(super) fn take_lease(&self, execution_id: Uuid) -> Result<Option<LocalExecutionLease>> {
        match self.leases.lock() {
            Ok(mut leases) => Ok(leases.remove(&execution_id).inspect(|lease| {
                lease.expiry_cancellation.cancel();
            })),
            Err(_) => {
                self.tainted.store(true, Ordering::Release);
                Err(PowerError::BackendNotAvailable(
                    "distributed phase lifecycle lock is unavailable".to_string(),
                ))
            }
        }
    }

    pub(super) async fn spawn_cleanup(
        inner: Arc<Self>,
        execution_id: Uuid,
        lease: LocalExecutionLease,
    ) -> bool {
        lease.operation_cancellation.cancel();
        lease.expiry_cancellation.cancel();
        let phase_abort =
            match lease.abort_command(execution_id, inner.transfer.local_worker_epoch()) {
                Ok(command) => command,
                Err(_) => {
                    inner.tainted.store(true, Ordering::Release);
                    return false;
                }
            };
        let transfer_abort = AbortStateTransfer {
            transfer_id: execution_id,
            local_worker_epoch: inner.transfer.local_worker_epoch(),
        };
        let task_inner = Arc::clone(&inner);
        let cleanup = tokio::spawn(async move {
            let phase = tokio::time::timeout(
                task_inner.cancellation_timeout,
                task_inner.executor.abort(phase_abort),
            );
            let transfer = task_inner.transfer.abort(transfer_abort);
            let (phase, transfer) = tokio::join!(phase, transfer);
            drop(lease);
            matches!(phase, Ok(Ok(()))) && transfer.is_ok()
        });
        match cleanup.await {
            Ok(true) => true,
            Ok(false) | Err(_) => {
                inner.tainted.store(true, Ordering::Release);
                false
            }
        }
    }

    fn leases(&self) -> Result<MutexGuard<'_, HashMap<Uuid, LocalExecutionLease>>> {
        self.leases.lock().map_err(|_| {
            self.tainted.store(true, Ordering::Release);
            PowerError::BackendNotAvailable(
                "distributed phase lifecycle lock is unavailable".to_string(),
            )
        })
    }
}

impl LocalExecutionLease {
    fn abort_command(&self, execution_id: Uuid, worker_epoch: Uuid) -> Result<AbortPhaseExecution> {
        let prepared = match &self.state {
            LocalExecutionState::Preparing => None,
            LocalExecutionState::PrefillPrepared(prepared)
            | LocalExecutionState::PrefillPublished(prepared) => Some(prepared.execution().clone()),
            LocalExecutionState::DecodePhasePrepared(prepared)
            | LocalExecutionState::DecodeExecuting(prepared)
            | LocalExecutionState::DecodeStreaming(prepared) => Some(prepared.execution().clone()),
            LocalExecutionState::DecodePrepared { prepared, .. } => {
                Some(prepared.execution().clone())
            }
        };
        match prepared {
            Some(handle) => AbortPhaseExecution::prepared(execution_id, worker_epoch, handle),
            None => AbortPhaseExecution::preparing(execution_id, worker_epoch),
        }
    }
}

fn preparing_lease(
    leases: &mut HashMap<Uuid, LocalExecutionLease>,
    execution_id: Uuid,
) -> Result<&mut LocalExecutionLease> {
    let lease = leases.get_mut(&execution_id).ok_or_else(missing_lease)?;
    if !matches!(lease.state, LocalExecutionState::Preparing) {
        return Err(invalid_transition("commit phase preparation"));
    }
    Ok(lease)
}

fn missing_lease() -> PowerError {
    PowerError::BackendNotAvailable("distributed execution lease is unavailable".to_string())
}

fn invalid_transition(operation: &str) -> PowerError {
    PowerError::InvalidRequest(format!(
        "distributed execution cannot {operation} from its current state"
    ))
}

pub(super) struct RuntimeOperationGuard {
    inner: Arc<RuntimeInner>,
    execution_id: Uuid,
    armed: bool,
}

impl RuntimeOperationGuard {
    pub(super) fn new(inner: Arc<RuntimeInner>, execution_id: Uuid) -> Self {
        Self {
            inner,
            execution_id,
            armed: true,
        }
    }

    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }

    pub(super) async fn cleanup(&mut self) -> bool {
        self.armed = false;
        let lease = match self.inner.take_lease(self.execution_id) {
            Ok(Some(lease)) => lease,
            Ok(None) => return true,
            Err(_) => return false,
        };
        RuntimeInner::spawn_cleanup(Arc::clone(&self.inner), self.execution_id, lease).await
    }

    pub(super) async fn fail<T>(&mut self, error: PowerError) -> Result<T> {
        if self.cleanup().await {
            Err(error)
        } else {
            Err(PowerError::BackendNotAvailable(
                "distributed phase failed and cleanup was not confirmed".to_string(),
            ))
        }
    }

    pub(super) async fn require<T>(&mut self, result: Result<T>) -> Result<T> {
        match result {
            Ok(value) => Ok(value),
            Err(error) => self.fail(error).await,
        }
    }
}

impl Drop for RuntimeOperationGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let Ok(Some(lease)) = self.inner.take_lease(self.execution_id) else {
            return;
        };
        let inner = Arc::clone(&self.inner);
        let execution_id = self.execution_id;
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => {
                runtime.spawn(async move {
                    let _ = RuntimeInner::spawn_cleanup(inner, execution_id, lease).await;
                });
            }
            Err(_) => {
                inner.tainted.store(true, Ordering::Release);
                drop(lease);
            }
        }
    }
}

pub(super) fn guarded_stream(
    stream: PhaseResponseStream,
    inner: Arc<RuntimeInner>,
    execution_id: Uuid,
    cancellation: CancellationToken,
) -> PhaseResponseStream {
    Box::pin(RuntimeResponseStream {
        stream,
        cancellation: Box::pin(cancellation.cancelled_owned()),
        cleanup: Some(StreamCleanup {
            inner,
            execution_id,
        }),
        finished: false,
    })
}

struct RuntimeResponseStream {
    stream: PhaseResponseStream,
    cancellation: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
    cleanup: Option<StreamCleanup>,
    finished: bool,
}

impl Stream for RuntimeResponseStream {
    type Item = Result<PhaseResponseChunk>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        if self.cancellation.as_mut().poll(context).is_ready() {
            self.finished = true;
            self.trigger_cleanup();
            return Poll::Ready(Some(Err(PowerError::BackendNotAvailable(
                "distributed decode stream was cancelled".to_string(),
            ))));
        }
        match self.stream.as_mut().poll_next(context) {
            Poll::Ready(None) => {
                self.finished = true;
                self.trigger_cleanup();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(error))) => {
                self.finished = true;
                self.trigger_cleanup();
                Poll::Ready(Some(Err(error)))
            }
            result => result,
        }
    }
}

impl RuntimeResponseStream {
    fn trigger_cleanup(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup.spawn();
        }
    }
}

impl Drop for RuntimeResponseStream {
    fn drop(&mut self) {
        self.trigger_cleanup();
    }
}

struct StreamCleanup {
    inner: Arc<RuntimeInner>,
    execution_id: Uuid,
}

impl StreamCleanup {
    fn spawn(self) {
        let Ok(Some(lease)) = self.inner.take_lease(self.execution_id) else {
            return;
        };
        let inner = Arc::clone(&self.inner);
        let execution_id = self.execution_id;
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => {
                runtime.spawn(async move {
                    let _ = RuntimeInner::spawn_cleanup(inner, execution_id, lease).await;
                });
            }
            Err(_) => {
                inner.tainted.store(true, Ordering::Release);
                drop(lease);
            }
        }
    }
}
