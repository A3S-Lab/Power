//! Colibri-inspired weight placement across storage, host RAM, and device
//! memory without changing model precision or routing semantics.

mod cache;
mod planner;
#[cfg(test)]
mod planner_tests;
mod prefetch;
mod types;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use candle_core::{Device, Tensor};
use tokio_util::sync::CancellationToken;

use crate::admission::AdmissionController;
use crate::error::{PowerError, Result};

use super::routing::{ExpertKey, RoutedExpertBatch};
use super::telemetry::{PlacementTelemetry, RoutingHistory, Telemetry};
use super::{EmbeddedRuntime, ExecutionPermit, RuntimeDeviceKind, TensorDescriptor, WeightStore};
use cache::{CacheInsert, CacheState};
pub use planner::{PlannedResidencyGroup, ResidencyApplyReport, ResidencyCandidate, ResidencyPlan};
pub use types::{
    PlacementPreference, PrefetchReport, PrefetchTask, ResidencyPolicy, ResidentWeight, WeightKey,
    WeightRequest, WeightTier,
};

/// Shared, bounded weight hierarchy for one model session.
#[derive(Clone)]
pub struct WeightHierarchy {
    inner: Arc<HierarchyInner>,
}

struct HierarchyInner {
    store: Arc<WeightStore>,
    runtime: EmbeddedRuntime,
    policy: ResidencyPolicy,
    operations: RwLock<()>,
    cache: Mutex<CacheState>,
    key_locks: Mutex<HashMap<WeightKey, Arc<Mutex<()>>>>,
    prefetch_admission: AdmissionController,
    telemetry: Telemetry,
}

impl WeightHierarchy {
    pub fn new(
        store: Arc<WeightStore>,
        runtime: EmbeddedRuntime,
        policy: ResidencyPolicy,
    ) -> Result<Self> {
        policy.validate()?;
        let resident_bytes = policy
            .host_cache_bytes
            .checked_add(policy.device_cache_bytes)
            .ok_or_else(|| {
                PowerError::Config("weight residency byte budget overflowed".to_string())
            })?;
        if resident_bytes > runtime.limits().max_resident_weight_bytes {
            return Err(PowerError::Config(format!(
                "weight residency budgets total {resident_bytes} bytes, exceeding the {} byte runtime limit",
                runtime.limits().max_resident_weight_bytes
            )));
        }
        let prefetch_admission = AdmissionController::new(Some(policy.max_prefetch_tasks));
        Ok(Self {
            inner: Arc::new(HierarchyInner {
                store,
                runtime,
                telemetry: Telemetry::new(policy.telemetry),
                policy,
                operations: RwLock::new(()),
                cache: Mutex::new(CacheState::new()),
                key_locks: Mutex::new(HashMap::new()),
                prefetch_admission,
            }),
        })
    }

    pub fn store(&self) -> &WeightStore {
        &self.inner.store
    }

    pub fn runtime(&self) -> &EmbeddedRuntime {
        &self.inner.runtime
    }

    pub fn policy(&self) -> &ResidencyPolicy {
        &self.inner.policy
    }

    /// Loads an exact tensor through the configured hierarchy.
    ///
    /// This method performs blocking storage/device work. Async model code
    /// should use `start_prefetch` or call it from `spawn_blocking`.
    pub fn load(
        &self,
        request: &WeightRequest,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<ResidentWeight> {
        let _operation = read(&self.inner.operations);
        self.load_internal(request, permit, cancellation, false)
    }

    pub fn pin(
        &self,
        request: &WeightRequest,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<ResidentWeight> {
        if request.placement == PlacementPreference::Streaming {
            return Err(PowerError::InvalidRequest(
                "a streaming weight request cannot be pinned".to_string(),
            ));
        }
        let _operation = read(&self.inner.operations);
        self.load_internal(request, permit, cancellation, true)
    }

    pub fn unpin(&self, key: &WeightKey, tier: WeightTier) -> bool {
        let _operation = read(&self.inner.operations);
        lock(&self.inner.cache).set_pinned(tier, key, false)
    }

    pub fn clear_unpinned(&self) {
        let _operation = read(&self.inner.operations);
        lock(&self.inner.cache).clear_unpinned(&self.inner.telemetry);
    }

    pub fn record_routes(&self, batch: &RoutedExpertBatch) {
        self.inner.telemetry.routes(batch);
    }

    pub fn telemetry(&self) -> PlacementTelemetry {
        let _operation = read(&self.inner.operations);
        let (host, device) = lock(&self.inner.cache).resident_bytes();
        self.inner.telemetry.snapshot(host, device)
    }

    pub fn routing_history(&self) -> Result<RoutingHistory> {
        self.inner.telemetry.history(self.inner.store.sha256())
    }

    pub fn restore_routing_history(&self, history: &RoutingHistory) -> Result<()> {
        self.inner
            .telemetry
            .restore_history(history, self.inner.store.sha256())
    }

    pub fn hottest_experts(&self, layer: u32, limit: usize) -> Result<Vec<ExpertKey>> {
        let mut entries = self
            .routing_history()?
            .entries
            .into_iter()
            .filter(|entry| entry.key.layer == layer)
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| {
            right
                .selections
                .cmp(&left.selections)
                .then_with(|| left.key.cmp(&right.key))
        });
        Ok(entries
            .into_iter()
            .take(limit)
            .map(|entry| entry.key)
            .collect())
    }

    fn load_internal(
        &self,
        request: &WeightRequest,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        require_resident: bool,
    ) -> Result<ResidentWeight> {
        self.validate_permit(permit)?;
        self.check_cancelled(cancellation)?;
        let descriptor = self.validate_request(request)?.clone();
        let placement = self.resolve_placement(request.placement);
        if let Some(weight) = self.cached(&request.key, placement, require_resident) {
            return Ok(weight);
        }

        let key_lock = {
            let mut locks = lock(&self.inner.key_locks);
            Arc::clone(
                locks
                    .entry(request.key.clone())
                    .or_insert_with(|| Arc::new(Mutex::new(()))),
            )
        };
        let _loading = lock(&key_lock);
        self.check_cancelled(cancellation)?;
        if let Some(weight) = self.cached(&request.key, placement, require_resident) {
            return Ok(weight);
        }

        match placement {
            PlacementPreference::Streaming => {
                let tensor = self
                    .inner
                    .store
                    .load_tensor(&request.key.name, self.inner.runtime.device())?;
                self.verify_tensor(&descriptor, &tensor)?;
                self.inner.telemetry.storage_read(descriptor.bytes);
                Ok(ResidentWeight {
                    tensor,
                    tier: WeightTier::Storage,
                    bytes: descriptor.bytes,
                    cache_hit: false,
                })
            }
            PlacementPreference::Host => {
                let tensor = self.load_host(&descriptor)?;
                self.finish_residency(
                    request,
                    tensor,
                    WeightTier::Host,
                    descriptor.bytes,
                    require_resident,
                    false,
                )
            }
            PlacementPreference::Device => {
                let (host, host_hit) = match self.cached_tensor(&request.key, WeightTier::Host) {
                    Some(tensor) => {
                        self.inner.telemetry.host_hit();
                        (tensor, true)
                    }
                    None => {
                        let tensor = self.load_host(&descriptor)?;
                        lock(&self.inner.cache).insert(
                            WeightTier::Host,
                            CacheInsert {
                                key: request.key.clone(),
                                tensor: tensor.clone(),
                                bytes: descriptor.bytes,
                                pinned: false,
                            },
                            &self.inner.policy,
                            &self.inner.telemetry,
                        );
                        (tensor, false)
                    }
                };
                let tensor = host
                    .to_device(self.inner.runtime.device().tensor_device())
                    .map_err(|error| {
                        PowerError::InferenceFailed(format!(
                            "failed to promote model tensor to the execution device: {error}"
                        ))
                    })?;
                self.verify_tensor(&descriptor, &tensor)?;
                self.inner.telemetry.device_promotion(descriptor.bytes);
                self.finish_residency(
                    request,
                    tensor,
                    WeightTier::Device,
                    descriptor.bytes,
                    require_resident,
                    host_hit,
                )
            }
            PlacementPreference::Fastest => Err(PowerError::InferenceFailed(
                "weight placement was not resolved before loading".to_string(),
            )),
        }
    }

    fn load_host(&self, descriptor: &TensorDescriptor) -> Result<Tensor> {
        let tensor = self.inner.store.load(&descriptor.name, &Device::Cpu)?;
        self.verify_tensor(descriptor, &tensor)?;
        self.inner.telemetry.storage_read(descriptor.bytes);
        Ok(tensor)
    }

    fn finish_residency(
        &self,
        request: &WeightRequest,
        tensor: Tensor,
        tier: WeightTier,
        bytes: u64,
        require_resident: bool,
        cache_hit: bool,
    ) -> Result<ResidentWeight> {
        let cached = lock(&self.inner.cache).insert(
            tier,
            CacheInsert {
                key: request.key.clone(),
                tensor: tensor.clone(),
                bytes,
                pinned: require_resident,
            },
            &self.inner.policy,
            &self.inner.telemetry,
        );
        if require_resident && !cached {
            return Err(PowerError::InferenceFailed(format!(
                "weight '{}' cannot be pinned within the configured {:?} residency bounds",
                request.key.name, tier
            )));
        }
        Ok(ResidentWeight {
            tensor,
            tier: if cached { tier } else { WeightTier::Storage },
            bytes,
            cache_hit,
        })
    }

    fn cached(
        &self,
        key: &WeightKey,
        placement: PlacementPreference,
        pin: bool,
    ) -> Option<ResidentWeight> {
        let tier = match placement {
            PlacementPreference::Host => WeightTier::Host,
            PlacementPreference::Device => WeightTier::Device,
            PlacementPreference::Fastest | PlacementPreference::Streaming => return None,
        };
        let mut cache = lock(&self.inner.cache);
        let lookup = cache.get(tier, key)?;
        if pin {
            cache.set_pinned(tier, key, true);
        }
        match tier {
            WeightTier::Host => self.inner.telemetry.host_hit(),
            WeightTier::Device => self.inner.telemetry.device_hit(),
            WeightTier::Storage => {}
        }
        Some(ResidentWeight {
            tensor: lookup.tensor,
            tier,
            bytes: lookup.bytes,
            cache_hit: true,
        })
    }

    fn cached_tensor(&self, key: &WeightKey, tier: WeightTier) -> Option<Tensor> {
        lock(&self.inner.cache)
            .get(tier, key)
            .map(|value| value.tensor)
    }

    fn validate_request(&self, request: &WeightRequest) -> Result<&TensorDescriptor> {
        if request.key.name.is_empty() || request.key.name.chars().any(char::is_control) {
            return Err(PowerError::InvalidRequest(
                "weight request contains an invalid tensor name".to_string(),
            ));
        }
        self.inner
            .store
            .descriptor(&request.key.name)
            .ok_or_else(|| {
                PowerError::InvalidFormat(format!(
                    "weight store does not contain tensor '{}'",
                    request.key.name
                ))
            })
    }

    fn validate_permit(&self, permit: &ExecutionPermit) -> Result<()> {
        if permit.belongs_to(&self.inner.runtime) {
            Ok(())
        } else {
            Err(PowerError::InvalidRequest(
                "weight operation permit belongs to a different embedded runtime".to_string(),
            ))
        }
    }

    fn check_cancelled(&self, cancellation: &CancellationToken) -> Result<()> {
        if cancellation.is_cancelled() {
            Err(PowerError::InferenceFailed(
                "weight operation was cancelled".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    fn resolve_placement(&self, preference: PlacementPreference) -> PlacementPreference {
        match preference {
            PlacementPreference::Fastest
                if self.inner.runtime.device().kind() != RuntimeDeviceKind::Cpu
                    && self.inner.policy.device_cache_bytes > 0 =>
            {
                PlacementPreference::Device
            }
            PlacementPreference::Fastest if self.inner.policy.host_cache_bytes > 0 => {
                PlacementPreference::Host
            }
            PlacementPreference::Fastest => PlacementPreference::Streaming,
            PlacementPreference::Device
                if self.inner.runtime.device().kind() == RuntimeDeviceKind::Cpu =>
            {
                PlacementPreference::Host
            }
            other => other,
        }
    }

    fn verify_tensor(&self, descriptor: &TensorDescriptor, tensor: &Tensor) -> Result<()> {
        let dtype = format!("{:?}", tensor.dtype()).to_ascii_lowercase();
        if dtype != descriptor.dtype || tensor.dims() != descriptor.shape {
            return Err(PowerError::InvalidFormat(format!(
                "tensor '{}' changed semantics while moving through the weight hierarchy",
                descriptor.name
            )));
        }
        Ok(())
    }
}

impl std::fmt::Debug for WeightHierarchy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (host, device) = lock(&self.inner.cache).resident_bytes();
        formatter
            .debug_struct("WeightHierarchy")
            .field("weights_sha256", &self.inner.store.sha256())
            .field("runtime", &self.inner.runtime)
            .field("policy", &self.inner.policy)
            .field("host_resident_bytes", &host)
            .field("device_resident_bytes", &device)
            .finish_non_exhaustive()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn read<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn write<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests;
