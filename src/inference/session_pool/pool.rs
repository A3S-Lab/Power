use std::collections::BTreeMap;
use std::future::Future;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tokio_util::sync::CancellationToken;

use crate::admission::AdmissionController;
use crate::error::{PowerError, Result};

use super::super::{DevicePreference, EmbeddedRuntime, RuntimeDevice};
use super::slot::SessionSlot;
use super::types::{
    replica_declaration_sha256, ModelSessionBinding, ModelSessionPoolPolicy,
    ModelSessionPoolSnapshot, ModelSessionSpec,
};

/// Bounded model-neutral pool of lazily initialized sessions on one device.
pub struct ModelSessionPool<T> {
    inner: Arc<PoolInner<T>>,
}

struct PoolInner<T> {
    device: RuntimeDevice,
    policy: ModelSessionPoolPolicy,
    device_admission: AdmissionController,
    expired_replica_requests: AtomicU64,
    replica_retirements: AtomicU64,
    replica_reconstructions: AtomicU64,
    sessions: Mutex<BTreeMap<String, Arc<SessionEntry<T>>>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SessionAccess {
    Shared,
    ExclusiveReplica,
}

pub(super) struct SessionEntry<T> {
    access: SessionAccess,
    pub(super) spec: ModelSessionSpec,
    shared_declaration_digest: String,
    pub(super) replica_declaration_digest: String,
    reserved_bytes: u64,
    pub(super) runtime: EmbeddedRuntime,
    pub(super) slots: Vec<SessionSlot<T>>,
    pub(super) replica_admission: AdmissionController,
    pub(super) available_replicas: Mutex<Vec<usize>>,
    loading_callers: AtomicUsize,
}

pub(super) struct SessionLoadGuard<T>
where
    T: Send + Sync + 'static,
{
    pub(super) pool: ModelSessionPool<T>,
    pub(super) key: String,
    pub(super) entry: Arc<SessionEntry<T>>,
}

impl<T> Clone for ModelSessionPool<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> ModelSessionPool<T> {
    pub(super) fn record_expired_replica_request(&self) {
        self.inner
            .expired_replica_requests
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn record_replica_retirement(&self) {
        self.inner
            .replica_retirements
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn record_replica_reconstruction(&self) {
        self.inner
            .replica_reconstructions
            .fetch_add(1, Ordering::Relaxed);
    }
}

impl<T> ModelSessionPool<T>
where
    T: Send + Sync + 'static,
{
    pub fn new(preference: DevicePreference, policy: ModelSessionPoolPolicy) -> Result<Self> {
        policy.validate()?;
        let device = RuntimeDevice::resolve(preference)?;
        let device_admission = AdmissionController::new_bounded(
            policy.max_concurrent_device_requests,
            policy.max_queued_device_requests,
        );
        Ok(Self {
            inner: Arc::new(PoolInner {
                device,
                policy,
                device_admission,
                expired_replica_requests: AtomicU64::new(0),
                replica_retirements: AtomicU64::new(0),
                replica_reconstructions: AtomicU64::new(0),
                sessions: Mutex::new(BTreeMap::new()),
            }),
        })
    }

    /// Gets one exact session or initializes it once for all concurrent callers.
    ///
    /// The loader receives the pool-created runtime and a clone of the caller's
    /// cancellation token. It must not create another runtime or device gate.
    pub async fn get_or_load<F, Fut>(
        &self,
        spec: ModelSessionSpec,
        cancellation: &CancellationToken,
        loader: F,
    ) -> Result<ModelSession<T>>
    where
        F: FnOnce(EmbeddedRuntime, CancellationToken) -> Fut + Send,
        Fut: Future<Output = Result<T>> + Send,
    {
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceCancelled);
        }
        if self.inner.policy.max_replicas_per_session != 1 {
            return Err(PowerError::InvalidRequest(
                "get_or_load is only available when max_replicas_per_session is one; use acquire_replica for exclusive replica mode"
                    .to_string(),
            ));
        }
        let (key, entry) = self.entry(spec, SessionAccess::Shared)?;
        let _load_guard = SessionLoadGuard {
            pool: self.clone(),
            key,
            entry: Arc::clone(&entry),
        };
        let runtime = entry.runtime.clone();
        let load_cancellation = cancellation.clone();
        let cell = entry.slots[0].cell();
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
        Ok(ModelSession { entry, value })
    }

    pub fn snapshot(&self) -> ModelSessionPoolSnapshot {
        let sessions = lock(&self.inner.sessions);
        let ready_sessions = sessions
            .values()
            .filter(|entry| entry.slots.iter().any(SessionSlot::is_ready))
            .count();
        let reserved_bytes = sessions.values().fold(0_u64, |total, entry| {
            total.saturating_add(entry.reserved_bytes)
        });
        let ready_replicas = sessions
            .values()
            .map(|entry| entry.slots.iter().filter(|slot| slot.is_ready()).count())
            .fold(0_usize, usize::saturating_add);
        let replica_snapshots = sessions
            .values()
            .map(|entry| entry.replica_admission.snapshot())
            .collect::<Vec<_>>();
        ModelSessionPoolSnapshot {
            device: self.inner.device.identity(),
            maximum_sessions: self.inner.policy.max_sessions,
            maximum_resident_bytes: self.inner.policy.max_resident_bytes,
            registered_sessions: sessions.len(),
            ready_sessions,
            maximum_replicas_per_session: self.inner.policy.max_replicas_per_session,
            reserved_replicas: sessions
                .len()
                .saturating_mul(self.inner.policy.max_replicas_per_session),
            ready_replicas,
            leased_replicas: replica_snapshots
                .iter()
                .map(|snapshot| snapshot.active)
                .fold(0_usize, usize::saturating_add),
            waiting_replica_requests: replica_snapshots
                .iter()
                .map(|snapshot| snapshot.waiting)
                .fold(0_usize, usize::saturating_add),
            expired_replica_requests: self.inner.expired_replica_requests.load(Ordering::Relaxed),
            replicas_pending_reconstruction: sessions
                .values()
                .flat_map(|entry| entry.slots.iter())
                .filter(|slot| slot.reconstruction_pending())
                .count(),
            replica_retirements: self.inner.replica_retirements.load(Ordering::Relaxed),
            replica_reconstructions: self.inner.replica_reconstructions.load(Ordering::Relaxed),
            reserved_bytes,
            device_admission: self.inner.device_admission.snapshot(),
        }
    }

    pub(super) fn entry(
        &self,
        spec: ModelSessionSpec,
        access: SessionAccess,
    ) -> Result<(String, Arc<SessionEntry<T>>)> {
        spec.validate()?;
        let replica_count =
            u64::try_from(self.inner.policy.max_replicas_per_session).map_err(|_| {
                PowerError::InvalidRequest("model session replica count overflowed".to_string())
            })?;
        let entry_reserved_bytes = spec.resident_bytes.checked_mul(replica_count).ok_or(
            PowerError::ModelSessionPoolFull {
                maximum_sessions: self.inner.policy.max_sessions,
                maximum_resident_bytes: self.inner.policy.max_resident_bytes,
            },
        )?;
        let key = spec.binding.key_sha256();
        let mut sessions = lock(&self.inner.sessions);
        if let Some(existing) = sessions.get(&key) {
            if existing.access != access {
                return Err(PowerError::InvalidRequest(
                    "one exact model session cannot mix shared and exclusive replica access"
                        .to_string(),
                ));
            }
            if existing.spec != spec {
                return Err(PowerError::InvalidRequest(
                    "the model session identity is already registered with different resource bounds"
                        .to_string(),
                ));
            }
            existing
                .loading_callers
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |callers| {
                    callers.checked_add(1)
                })
                .map_err(|_| {
                    PowerError::InferenceFailed(
                        "model session loading caller count overflowed".to_string(),
                    )
                })?;
            return Ok((key, Arc::clone(existing)));
        }
        let reserved_bytes = sessions.values().try_fold(0_u64, |total, entry| {
            total.checked_add(entry.reserved_bytes)
        });
        let next_reserved =
            reserved_bytes.and_then(|total| total.checked_add(entry_reserved_bytes));
        if sessions.len() >= self.inner.policy.max_sessions
            || next_reserved.is_none_or(|bytes| bytes > self.inner.policy.max_resident_bytes)
        {
            return Err(PowerError::ModelSessionPoolFull {
                maximum_sessions: self.inner.policy.max_sessions,
                maximum_resident_bytes: self.inner.policy.max_resident_bytes,
            });
        }
        let shared_declaration_digest = spec.declaration_sha256(self.inner.device.identity())?;
        let replica_declaration_digest = replica_declaration_sha256(
            &shared_declaration_digest,
            self.inner.policy.max_replicas_per_session,
            entry_reserved_bytes,
        )?;
        let runtime = EmbeddedRuntime::with_device_admission(
            self.inner.device.clone(),
            spec.limits.clone(),
            self.inner.device_admission.clone(),
        )?;
        let slots = (0..self.inner.policy.max_replicas_per_session)
            .map(|_| SessionSlot::new())
            .collect();
        let available_replicas = (0..self.inner.policy.max_replicas_per_session)
            .rev()
            .collect();
        let max_queued_replicas = spec.limits.max_queued_requests;
        let entry = Arc::new(SessionEntry {
            access,
            spec,
            shared_declaration_digest,
            replica_declaration_digest,
            reserved_bytes: entry_reserved_bytes,
            runtime,
            slots,
            replica_admission: AdmissionController::new_bounded(
                self.inner.policy.max_replicas_per_session,
                max_queued_replicas,
            ),
            available_replicas: Mutex::new(available_replicas),
            loading_callers: AtomicUsize::new(1),
        });
        sessions.insert(key.clone(), Arc::clone(&entry));
        Ok((key, entry))
    }

    fn remove_empty(&self, key: &str, entry: &Arc<SessionEntry<T>>) {
        let mut sessions = lock(&self.inner.sessions);
        let is_current = sessions
            .get(key)
            .is_some_and(|current| Arc::ptr_eq(current, entry));
        if is_current
            && entry.slots.iter().all(|slot| !slot.is_ready())
            && entry
                .slots
                .iter()
                .all(|slot| !slot.reconstruction_pending())
            && entry.loading_callers.load(Ordering::Relaxed) == 0
        {
            sessions.remove(key);
        }
    }
}

impl<T> Drop for SessionLoadGuard<T>
where
    T: Send + Sync + 'static,
{
    fn drop(&mut self) {
        let previous = self.entry.loading_callers.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |callers| callers.checked_sub(1),
        );
        match previous {
            Ok(1) => self.pool.remove_empty(&self.key, &self.entry),
            Ok(_) => {}
            Err(_) => debug_assert!(false, "model session loading caller count underflowed"),
        }
    }
}

impl<T> std::fmt::Debug for ModelSessionPool<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snapshot = {
            let sessions = lock(&self.inner.sessions);
            (sessions.len(), self.inner.policy.max_sessions)
        };
        formatter
            .debug_struct("ModelSessionPool")
            .field("device", &self.inner.device.identity())
            .field("registered_sessions", &snapshot.0)
            .field("maximum_sessions", &snapshot.1)
            .finish_non_exhaustive()
    }
}

/// Shared initialized value and exact Power runtime for one pool entry.
pub struct ModelSession<T> {
    entry: Arc<SessionEntry<T>>,
    value: Arc<T>,
}

impl<T> Clone for ModelSession<T> {
    fn clone(&self) -> Self {
        Self {
            entry: Arc::clone(&self.entry),
            value: Arc::clone(&self.value),
        }
    }
}

impl<T> ModelSession<T> {
    pub fn binding(&self) -> &ModelSessionBinding {
        &self.entry.spec.binding
    }

    pub fn declaration_sha256(&self) -> &str {
        &self.entry.shared_declaration_digest
    }

    pub fn runtime(&self) -> &EmbeddedRuntime {
        &self.entry.runtime
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T> std::fmt::Debug for ModelSession<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelSession")
            .field("declaration", &"sha256")
            .field("device", &self.entry.runtime.device().identity())
            .finish_non_exhaustive()
    }
}

pub(super) fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
