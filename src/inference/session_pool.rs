use std::collections::BTreeMap;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{OnceCell, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::admission::{AdmissionController, AdmissionSnapshot};
use crate::error::{PowerError, Result};

use super::sealed_state::decode_sha256;
use super::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, ModelIdentity, RuntimeDevice,
    RuntimeDeviceIdentity, RuntimeDeviceKind,
};

/// Exact model and model-owned execution identity for one shareable session.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelSessionBinding {
    pub model: ModelIdentity,
    pub execution_sha256: String,
}

impl ModelSessionBinding {
    pub fn new(model: ModelIdentity, execution_sha256: impl Into<String>) -> Self {
        Self {
            model,
            execution_sha256: execution_sha256.into(),
        }
    }

    fn validate(&self, limits: &InferenceLimits) -> Result<()> {
        for (label, value) in [
            ("model session family", self.model.family.as_str()),
            ("model session revision", self.model.revision.as_str()),
        ] {
            if value.is_empty()
                || value.len() > limits.max_graph_name_bytes
                || value.chars().any(char::is_control)
            {
                return Err(PowerError::InvalidRequest(format!(
                    "{label} must be non-empty, control-free, and at most {} bytes",
                    limits.max_graph_name_bytes
                )));
            }
        }
        decode_sha256(&self.model.weights_sha256, "model session weights")?;
        decode_sha256(&self.execution_sha256, "model session execution")?;
        Ok(())
    }

    fn key_sha256(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"a3s-power-model-session-key-v1\0");
        update_text(&mut digest, &self.model.family);
        update_text(&mut digest, &self.model.revision);
        update_text(&mut digest, &self.model.weights_sha256);
        update_text(&mut digest, &self.execution_sha256);
        format!("{:x}", digest.finalize())
    }
}

impl std::fmt::Debug for ModelSessionBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelSessionBinding")
            .field("model", &"bound")
            .field("execution", &"sha256")
            .finish()
    }
}

/// Resource declaration for one pool entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelSessionSpec {
    binding: ModelSessionBinding,
    limits: InferenceLimits,
    resident_bytes: u64,
}

impl ModelSessionSpec {
    pub fn new(
        binding: ModelSessionBinding,
        limits: InferenceLimits,
        resident_bytes: u64,
    ) -> Result<Self> {
        let spec = Self {
            binding,
            limits,
            resident_bytes,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn binding(&self) -> &ModelSessionBinding {
        &self.binding
    }

    pub fn limits(&self) -> &InferenceLimits {
        &self.limits
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Canonical declaration digest, including the exact resolved device and
    /// every resource limit used to construct the shared runtime.
    pub fn declaration_sha256(&self, device: RuntimeDeviceIdentity) -> Result<String> {
        self.validate()?;
        device.validate()?;
        let mut digest = Sha256::new();
        digest.update(b"a3s-power-model-session-declaration-v1\0");
        update_text(&mut digest, &self.binding.key_sha256());
        digest.update(self.resident_bytes.to_le_bytes());
        digest.update([match device.kind {
            RuntimeDeviceKind::Cpu => 0,
            RuntimeDeviceKind::Cuda => 1,
            RuntimeDeviceKind::Metal => 2,
        }]);
        update_optional_usize(&mut digest, device.ordinal)?;
        update_limits(&mut digest, &self.limits)?;
        Ok(format!("{:x}", digest.finalize()))
    }

    fn validate(&self) -> Result<()> {
        self.limits.validate()?;
        self.binding.validate(&self.limits)?;
        if self.resident_bytes == 0 || self.resident_bytes > self.limits.max_model_bytes {
            return Err(PowerError::InvalidRequest(format!(
                "model session resident bytes must be between 1 and {}",
                self.limits.max_model_bytes
            )));
        }
        Ok(())
    }
}

/// Hard bounds shared by every model entry on one resolved device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelSessionPoolPolicy {
    pub max_sessions: usize,
    pub max_resident_bytes: u64,
    pub max_concurrent_device_requests: usize,
    pub max_queued_device_requests: usize,
}

impl ModelSessionPoolPolicy {
    pub fn new(
        max_sessions: usize,
        max_resident_bytes: u64,
        max_concurrent_device_requests: usize,
        max_queued_device_requests: usize,
    ) -> Result<Self> {
        let policy = Self {
            max_sessions,
            max_resident_bytes,
            max_concurrent_device_requests,
            max_queued_device_requests,
        };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<()> {
        if self.max_sessions == 0
            || self.max_resident_bytes == 0
            || self.max_concurrent_device_requests == 0
        {
            return Err(PowerError::Config(
                "model session pool count, resident bytes, and device concurrency must be greater than zero"
                    .to_string(),
            ));
        }
        if self.max_sessions > Semaphore::MAX_PERMITS
            || self.max_concurrent_device_requests > Semaphore::MAX_PERMITS
            || self.max_queued_device_requests > Semaphore::MAX_PERMITS
        {
            return Err(PowerError::Config(format!(
                "model session pool count and admission bounds cannot exceed {}",
                Semaphore::MAX_PERMITS
            )));
        }
        Ok(())
    }
}

/// Aggregate, content-free state for one device-bound session pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSessionPoolSnapshot {
    pub device: RuntimeDeviceIdentity,
    pub maximum_sessions: usize,
    pub maximum_resident_bytes: u64,
    pub registered_sessions: usize,
    pub ready_sessions: usize,
    pub reserved_bytes: u64,
    pub device_admission: AdmissionSnapshot,
}

/// Bounded model-neutral pool of lazily initialized sessions on one device.
pub struct ModelSessionPool<T> {
    inner: Arc<PoolInner<T>>,
}

struct PoolInner<T> {
    device: RuntimeDevice,
    policy: ModelSessionPoolPolicy,
    device_admission: AdmissionController,
    sessions: Mutex<BTreeMap<String, Arc<SessionEntry<T>>>>,
}

struct SessionEntry<T> {
    spec: ModelSessionSpec,
    declaration_sha256: String,
    runtime: EmbeddedRuntime,
    value: OnceCell<Arc<T>>,
    loading_callers: AtomicUsize,
}

struct SessionLoadGuard<T>
where
    T: Send + Sync + 'static,
{
    pool: ModelSessionPool<T>,
    key: String,
    entry: Arc<SessionEntry<T>>,
}

impl<T> Clone for ModelSessionPool<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
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
        let (key, entry) = self.entry(spec)?;
        let _load_guard = SessionLoadGuard {
            pool: self.clone(),
            key,
            entry: Arc::clone(&entry),
        };
        let runtime = entry.runtime.clone();
        let load_cancellation = cancellation.clone();
        let initialized = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                return Err(PowerError::InferenceCancelled);
            }
            result = entry.value.get_or_try_init(|| async move {
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
            .filter(|entry| entry.value.get().is_some())
            .count();
        let reserved_bytes = sessions.values().fold(0_u64, |total, entry| {
            total.saturating_add(entry.spec.resident_bytes)
        });
        ModelSessionPoolSnapshot {
            device: self.inner.device.identity(),
            maximum_sessions: self.inner.policy.max_sessions,
            maximum_resident_bytes: self.inner.policy.max_resident_bytes,
            registered_sessions: sessions.len(),
            ready_sessions,
            reserved_bytes,
            device_admission: self.inner.device_admission.snapshot(),
        }
    }

    fn entry(&self, spec: ModelSessionSpec) -> Result<(String, Arc<SessionEntry<T>>)> {
        spec.validate()?;
        let key = spec.binding.key_sha256();
        let mut sessions = lock(&self.inner.sessions);
        if let Some(existing) = sessions.get(&key) {
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
            total.checked_add(entry.spec.resident_bytes)
        });
        let next_reserved = reserved_bytes.and_then(|total| total.checked_add(spec.resident_bytes));
        if sessions.len() >= self.inner.policy.max_sessions
            || next_reserved.is_none_or(|bytes| bytes > self.inner.policy.max_resident_bytes)
        {
            return Err(PowerError::ModelSessionPoolFull {
                maximum_sessions: self.inner.policy.max_sessions,
                maximum_resident_bytes: self.inner.policy.max_resident_bytes,
            });
        }
        let declaration_sha256 = spec.declaration_sha256(self.inner.device.identity())?;
        let runtime = EmbeddedRuntime::with_device_admission(
            self.inner.device.clone(),
            spec.limits.clone(),
            self.inner.device_admission.clone(),
        )?;
        let entry = Arc::new(SessionEntry {
            spec,
            declaration_sha256,
            runtime,
            value: OnceCell::new(),
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
            && entry.value.get().is_none()
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
        &self.entry.declaration_sha256
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

fn update_text(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value.as_bytes());
}

fn update_optional_usize(digest: &mut Sha256, value: Option<usize>) -> Result<()> {
    digest.update([u8::from(value.is_some())]);
    if let Some(value) = value {
        digest.update(
            u64::try_from(value)
                .map_err(|_| {
                    PowerError::InvalidRequest(
                        "model session device ordinal cannot be represented".to_string(),
                    )
                })?
                .to_le_bytes(),
        );
    }
    Ok(())
}

fn update_limits(digest: &mut Sha256, limits: &InferenceLimits) -> Result<()> {
    for value in [
        limits.max_model_files,
        limits.max_weight_sources,
        limits.max_input_bytes,
        limits.max_tensor_elements,
        limits.max_graph_plan_bytes,
        limits.max_graph_nodes,
        limits.max_graph_initializers,
        limits.max_graph_name_bytes,
        limits.max_context_tokens,
        limits.max_generated_tokens,
        limits.max_concurrent_requests,
        limits.max_queued_requests,
    ] {
        digest.update(
            u64::try_from(value)
                .map_err(|_| {
                    PowerError::InvalidRequest(
                        "model session resource limit cannot be represented".to_string(),
                    )
                })?
                .to_le_bytes(),
        );
    }
    for value in [
        limits.max_model_bytes,
        limits.max_resident_weight_bytes,
        limits.max_state_bytes,
        limits.max_image_pixels,
    ] {
        digest.update(value.to_le_bytes());
    }
    Ok(())
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
