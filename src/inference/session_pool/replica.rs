use std::future::Future;
use std::sync::Arc;

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
    entry: Arc<SessionEntry<T>>,
    index: Option<usize>,
    _admission: AdmissionPermit,
}

impl<T> Drop for ReplicaLease<T> {
    fn drop(&mut self) {
        let Some(index) = self.index.take() else {
            return;
        };
        let mut available = lock(&self.entry.available_replicas);
        if available.contains(&index) {
            debug_assert!(false, "model session replica was returned more than once");
            return;
        }
        available.push(index);
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
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceCancelled);
        }
        let (key, entry) = self.entry(spec, SessionAccess::ExclusiveReplica)?;
        let _load_guard = SessionLoadGuard {
            pool: self.clone(),
            key,
            entry: Arc::clone(&entry),
        };
        let admission = entry
            .replica_admission
            .acquire_cancellable(cancellation)
            .await
            .map_err(map_admission_error)?;
        let index = lock(&entry.available_replicas).pop().ok_or_else(|| {
            PowerError::InferenceFailed(
                "replica admission succeeded without an available session slot".to_string(),
            )
        })?;
        let lease = ReplicaLease {
            entry: Arc::clone(&entry),
            index: Some(index),
            _admission: admission,
        };
        if index >= entry.values.len() {
            return Err(PowerError::InferenceFailed(
                "replica free list contained an out-of-range session slot".to_string(),
            ));
        }
        let runtime = entry.runtime.clone();
        let load_cancellation = cancellation.clone();
        let initialized = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                return Err(PowerError::InferenceCancelled);
            }
            result = entry.values[index].get_or_try_init(|| async move {
                loader(runtime, load_cancellation).await.map(Arc::new)
            }) => result,
        };
        let value = match initialized {
            Ok(value) => Arc::clone(value),
            Err(error) => return Err(error),
        };
        Ok(ModelSessionReplica { value, lease })
    }
}

fn map_admission_error(error: AdmissionError) -> PowerError {
    match error {
        AdmissionError::QueueFull { maximum } => PowerError::InferenceQueueFull { maximum },
        AdmissionError::Cancelled => PowerError::InferenceCancelled,
        AdmissionError::Closed => {
            PowerError::InferenceFailed("replica admission controller closed".to_string())
        }
    }
}
