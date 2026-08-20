use std::future::Future;
use std::sync::Arc;

use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::admission::{AdmissionError, AdmissionPermit};
use crate::error::{PowerError, Result};

use super::pool::{lock, ModelSessionPool, SessionAccess, SessionEntry, SessionLoadGuard};
use super::types::{ModelSessionBinding, ModelSessionSpec};
use crate::inference::EmbeddedRuntime;

/// Exclusive lease of one independently initialized session replica.
///
/// The lease is intentionally not cloneable. Dropping it makes the same
/// replica available to another request without exposing a slot identity.
/// Stateful backends can load a synchronization primitive as `T` and mutate
/// that state through the exclusive lease; Power does not interpret the
/// model family, context layout, or state transitions.
pub struct ModelSessionReplica<T> {
    value: Arc<T>,
    lease: ReplicaLease<T>,
}

impl<T> ModelSessionReplica<T> {
    pub fn binding(&self) -> &ModelSessionBinding {
        &self.lease.entry.spec.binding
    }

    pub fn declaration_sha256(&self) -> &str {
        &self.lease.entry.replica_declaration_digest
    }

    pub fn runtime(&self) -> &EmbeddedRuntime {
        &self.lease.entry.runtime
    }

    pub fn value(&self) -> &T {
        &self.value
    }

    /// Retires this replica at the exclusive request boundary.
    ///
    /// The model-owning crate decides when state is no longer reusable. Power
    /// records no reason or model semantics, removes the current generation
    /// before returning the anonymous slot, and initializes a replacement on
    /// the next acquisition.
    pub fn retire(mut self) {
        self.lease.retire_on_drop = true;
    }
}

impl<T> std::fmt::Debug for ModelSessionReplica<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelSessionReplica")
            .field("declaration", &"sha256")
            .field("device", &self.lease.entry.runtime.device().identity())
            .finish_non_exhaustive()
    }
}

struct ReplicaLease<T> {
    pool: ModelSessionPool<T>,
    entry: Arc<SessionEntry<T>>,
    index: Option<usize>,
    retire_on_drop: bool,
    _admission: AdmissionPermit,
}

impl<T> Drop for ReplicaLease<T> {
    fn drop(&mut self) {
        let Some(index) = self.index.take() else {
            return;
        };
        let Some(slot) = self.entry.slots.get(index) else {
            debug_assert!(
                false,
                "model session replica index was out of range during release"
            );
            return;
        };
        let retired = if self.retire_on_drop {
            slot.retire()
        } else {
            None
        };
        if retired.is_some() {
            self.pool.record_replica_retirement();
        }
        {
            let mut available = lock(&self.entry.available_replicas);
            if available.contains(&index) {
                debug_assert!(false, "model session replica was returned more than once");
                return;
            }
            available.push(index);
        }
        // A model-owned destructor must not run under either lifecycle mutex.
        // The slot is already consistent if that destructor unwinds.
        drop(retired);
    }
}

impl<T> ModelSessionPool<T>
where
    T: Send + Sync + 'static,
{
    /// Acquires one exclusive replica, initializing that slot at most once.
    ///
    /// Every replica shares the pool-created runtime and device admission gate.
    /// The loader receives no ordinal, model family switch, or independent
    /// scheduling primitive.
    pub async fn acquire_replica<F, Fut>(
        &self,
        spec: ModelSessionSpec,
        cancellation: &CancellationToken,
        loader: F,
    ) -> Result<ModelSessionReplica<T>>
    where
        F: FnOnce(EmbeddedRuntime, CancellationToken) -> Fut + Send,
        Fut: Future<Output = Result<T>> + Send,
    {
        self.acquire_replica_inner(spec, cancellation, None, loader)
            .await
    }

    /// Acquires one exclusive replica before a monotonic admission deadline.
    ///
    /// The same absolute deadline covers the complete replica wait. It is not
    /// serialized, logged, or converted to wall-clock time.
    pub async fn acquire_replica_until<F, Fut>(
        &self,
        spec: ModelSessionSpec,
        cancellation: &CancellationToken,
        deadline: Instant,
        loader: F,
    ) -> Result<ModelSessionReplica<T>>
    where
        F: FnOnce(EmbeddedRuntime, CancellationToken) -> Fut + Send,
        Fut: Future<Output = Result<T>> + Send,
    {
        self.acquire_replica_inner(spec, cancellation, Some(deadline), loader)
            .await
    }

    async fn acquire_replica_inner<F, Fut>(
        &self,
        spec: ModelSessionSpec,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
        loader: F,
    ) -> Result<ModelSessionReplica<T>>
    where
        F: FnOnce(EmbeddedRuntime, CancellationToken) -> Fut + Send,
        Fut: Future<Output = Result<T>> + Send,
    {
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceCancelled);
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            self.record_expired_replica_request();
            return Err(PowerError::InferenceDeadlineExceeded);
        }
        let (key, entry) = self.entry(spec, SessionAccess::ExclusiveReplica)?;
        let _load_guard = SessionLoadGuard {
            pool: self.clone(),
            key,
            entry: Arc::clone(&entry),
        };
        let admission = match deadline {
            Some(deadline) => {
                entry
                    .replica_admission
                    .acquire_cancellable_until(cancellation, deadline)
                    .await
            }
            None => {
                entry
                    .replica_admission
                    .acquire_cancellable(cancellation)
                    .await
            }
        }
        .map_err(|error| {
            if matches!(error, AdmissionError::DeadlineExceeded) {
                self.record_expired_replica_request();
            }
            map_admission_error(error)
        })?;
        let index = lock(&entry.available_replicas).pop().ok_or_else(|| {
            PowerError::InferenceFailed(
                "replica admission succeeded without an available session slot".to_string(),
            )
        })?;
        let lease = ReplicaLease {
            pool: self.clone(),
            entry: Arc::clone(&entry),
            index: Some(index),
            retire_on_drop: false,
            _admission: admission,
        };
        if index >= entry.slots.len() {
            return Err(PowerError::InferenceFailed(
                "replica free list contained an out-of-range session slot".to_string(),
            ));
        }
        let runtime = entry.runtime.clone();
        let load_cancellation = cancellation.clone();
        let cell = entry.slots[index].cell();
        let initialized = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                return Err(PowerError::InferenceCancelled);
            }
            result = cell.get_or_try_init(|| async move {
                loader(runtime, load_cancellation).await.map(Arc::new)
            }) => result,
        };
        let value = match initialized {
            Ok(value) => Arc::clone(value),
            Err(error) => return Err(error),
        };
        if entry.slots[index].finish_reconstruction(&cell) {
            self.record_replica_reconstruction();
        }
        Ok(ModelSessionReplica { value, lease })
    }
}

fn map_admission_error(error: AdmissionError) -> PowerError {
    match error {
        AdmissionError::QueueFull { maximum } => PowerError::InferenceQueueFull { maximum },
        AdmissionError::Cancelled => PowerError::InferenceCancelled,
        AdmissionError::DeadlineExceeded => PowerError::InferenceDeadlineExceeded,
        AdmissionError::Closed => {
            PowerError::InferenceFailed("replica admission controller closed".to_string())
        }
    }
}
