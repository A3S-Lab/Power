use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;

use crate::error::{PowerError, Result};
use crate::inference::TelemetryMode;

use super::super::coupling::RouteCouplingPolicy;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightKey {
    pub layer: u32,
    pub name: String,
}

impl WeightKey {
    pub fn new(layer: u32, name: impl Into<String>) -> Self {
        Self {
            layer,
            name: name.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WeightTier {
    Storage,
    Host,
    Device,
}

/// Bounded cache replacement policy for host and device weight tiers.
///
/// `Lfru` follows Colibri's frequency-first, recency-second policy. One heat
/// observation outweighs the entire recency window, so a single recent access
/// cannot displace a consistently hot weight.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheEvictionPolicy {
    Lru,
    #[default]
    Lfru,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlacementPreference {
    /// Choose the fastest configured tier without changing tensor dtype.
    #[default]
    Fastest,
    /// Materialize for this call but do not retain a cache entry.
    Streaming,
    Host,
    Device,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightRequest {
    pub key: WeightKey,
    pub placement: PlacementPreference,
}

impl WeightRequest {
    pub fn new(key: WeightKey, placement: PlacementPreference) -> Self {
        Self { key, placement }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResidencyPolicy {
    pub host_cache_bytes: u64,
    pub device_cache_bytes: u64,
    pub max_entries_per_layer: usize,
    #[serde(default)]
    pub cache_eviction: CacheEvictionPolicy,
    /// Demand accesses between frequency decay passes for the LFRU policy.
    #[serde(default = "default_cache_heat_decay_interval")]
    pub cache_heat_decay_interval: u64,
    /// Maximum number of layer-ahead prefetch operations in flight.
    pub max_prefetch_tasks: usize,
    /// Maximum blocking weight loads within one prefetch operation.
    pub max_prefetch_workers: usize,
    pub max_prefetch_items: usize,
    pub max_prefetch_bytes: u64,
    /// Maximum canonical bytes concurrently owned by background load workers.
    ///
    /// This is shared by speculative prefetch and exact current-layer staging.
    /// The effective value is capped by `max_prefetch_bytes`.
    #[serde(default = "default_max_background_inflight_bytes")]
    pub max_background_inflight_bytes: u64,
    #[serde(default)]
    pub route_coupling: RouteCouplingPolicy,
    pub telemetry: TelemetryMode,
}

impl Default for ResidencyPolicy {
    fn default() -> Self {
        Self {
            // Caching is opt-in so an embedded model cannot unexpectedly pin a
            // large working set inside a memory-constrained TEE.
            host_cache_bytes: 0,
            device_cache_bytes: 0,
            max_entries_per_layer: 64,
            cache_eviction: CacheEvictionPolicy::Lfru,
            cache_heat_decay_interval: default_cache_heat_decay_interval(),
            max_prefetch_tasks: 1,
            max_prefetch_workers: 4,
            max_prefetch_items: 128,
            max_prefetch_bytes: 1024 * 1024 * 1024,
            max_background_inflight_bytes: default_max_background_inflight_bytes(),
            route_coupling: RouteCouplingPolicy::default(),
            telemetry: TelemetryMode::Disabled,
        }
    }
}

const fn default_cache_heat_decay_interval() -> u64 {
    4_096
}

const fn default_max_background_inflight_bytes() -> u64 {
    1024 * 1024 * 1024
}

impl ResidencyPolicy {
    pub fn validate(&self) -> Result<()> {
        if self.max_entries_per_layer == 0
            || self.max_prefetch_tasks == 0
            || self.max_prefetch_workers == 0
            || self.max_prefetch_items == 0
            || self.max_prefetch_bytes == 0
            || self.max_background_inflight_bytes == 0
            || self.cache_heat_decay_interval == 0
        {
            return Err(PowerError::Config(
                "weight residency cache and prefetch bounds must be greater than zero".to_string(),
            ));
        }
        self.route_coupling.validate()?;
        Ok(())
    }

    pub(super) fn background_inflight_bytes(&self) -> u64 {
        self.max_background_inflight_bytes
            .min(self.max_prefetch_bytes)
    }
}

#[derive(Clone)]
pub struct ResidentWeight {
    pub(super) tensor: Tensor,
    pub(super) tier: WeightTier,
    pub(super) bytes: u64,
    pub(super) cache_hit: bool,
}

impl ResidentWeight {
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }

    pub fn tier(&self) -> WeightTier {
        self.tier
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn cache_hit(&self) -> bool {
        self.cache_hit
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PrefetchReport {
    pub requested: usize,
    pub unique: usize,
    pub cache_hits: usize,
    pub materialized: usize,
    pub bytes: u64,
    #[serde(default)]
    pub peak_inflight_weights: usize,
    #[serde(default)]
    pub peak_inflight_bytes: u64,
}

pub struct PrefetchTask {
    pub(super) handle: Option<JoinHandle<Result<PrefetchReport>>>,
    pub(super) cancellation: tokio_util::sync::CancellationToken,
}

impl PrefetchTask {
    pub async fn wait(mut self) -> Result<PrefetchReport> {
        let handle = self.handle.take().ok_or_else(|| {
            PowerError::InferenceFailed("weight prefetch task has no join handle".to_string())
        })?;
        handle.await.map_err(|error| {
            PowerError::InferenceFailed(format!("weight prefetch task failed: {error}"))
        })?
    }

    pub fn abort(&self) {
        self.cancellation.cancel();
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }
}

impl Drop for PrefetchTask {
    fn drop(&mut self) {
        self.cancellation.cancel();
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }
}
