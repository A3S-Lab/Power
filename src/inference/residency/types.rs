use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;

use crate::error::{PowerError, Result};
use crate::inference::TelemetryMode;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WeightTier {
    Storage,
    Host,
    Device,
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
    pub max_prefetch_items: usize,
    pub max_prefetch_bytes: u64,
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
            max_prefetch_items: 128,
            max_prefetch_bytes: 1024 * 1024 * 1024,
            telemetry: TelemetryMode::Disabled,
        }
    }
}

impl ResidencyPolicy {
    pub fn validate(&self) -> Result<()> {
        if self.max_entries_per_layer == 0
            || self.max_prefetch_items == 0
            || self.max_prefetch_bytes == 0
        {
            return Err(PowerError::Config(
                "weight residency entry and prefetch bounds must be greater than zero".to_string(),
            ));
        }
        Ok(())
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
}

pub struct PrefetchTask {
    pub(super) handle: JoinHandle<Result<PrefetchReport>>,
}

impl PrefetchTask {
    pub async fn wait(self) -> Result<PrefetchReport> {
        self.handle.await.map_err(|error| {
            PowerError::InferenceFailed(format!("weight prefetch task failed: {error}"))
        })?
    }

    pub fn abort(&self) {
        self.handle.abort();
    }
}
