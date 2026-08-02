use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

use super::routing::{ExpertKey, RoutedExpertBatch};

/// Controls inference telemetry that may reveal workload characteristics.
///
/// Telemetry is disabled by default. `Detailed` includes per-expert routing
/// heat, which can correlate with input semantics and must remain inside the
/// TEE unless an explicit privacy policy authorizes export.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TelemetryMode {
    #[default]
    Disabled,
    Aggregate,
    Detailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RouteHeat {
    pub key: ExpertKey,
    pub selections: u64,
}

/// Serializable route history for a model-owned encrypted or sealed store.
///
/// Power never persists this value automatically because route heat is
/// sensitive inference metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RoutingHistory {
    pub schema: String,
    pub weights_sha256: String,
    pub entries: Vec<RouteHeat>,
}

impl RoutingHistory {
    pub const SCHEMA: &'static str = "a3s.power.routing-history.v1";
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlacementTelemetry {
    pub schema: String,
    pub mode: TelemetryMode,
    pub host_cache_hits: u64,
    pub device_cache_hits: u64,
    pub storage_reads: u64,
    pub storage_bytes_read: u64,
    pub device_bytes_promoted: u64,
    pub host_evictions: u64,
    pub device_evictions: u64,
    pub prefetched_weights: u64,
    pub prefetch_cache_hits: u64,
    pub routed_selections: u64,
    pub host_resident_bytes: u64,
    pub device_resident_bytes: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub routing_heat: Vec<RouteHeat>,
}

impl PlacementTelemetry {
    pub const SCHEMA: &'static str = "a3s.power.weight-placement-telemetry.v1";
}

pub(crate) struct Telemetry {
    mode: TelemetryMode,
    host_cache_hits: AtomicU64,
    device_cache_hits: AtomicU64,
    storage_reads: AtomicU64,
    storage_bytes_read: AtomicU64,
    device_bytes_promoted: AtomicU64,
    host_evictions: AtomicU64,
    device_evictions: AtomicU64,
    prefetched_weights: AtomicU64,
    prefetch_cache_hits: AtomicU64,
    routed_selections: AtomicU64,
    routing_heat: Mutex<BTreeMap<ExpertKey, u64>>,
}

impl Telemetry {
    pub(crate) fn new(mode: TelemetryMode) -> Self {
        Self {
            mode,
            host_cache_hits: AtomicU64::new(0),
            device_cache_hits: AtomicU64::new(0),
            storage_reads: AtomicU64::new(0),
            storage_bytes_read: AtomicU64::new(0),
            device_bytes_promoted: AtomicU64::new(0),
            host_evictions: AtomicU64::new(0),
            device_evictions: AtomicU64::new(0),
            prefetched_weights: AtomicU64::new(0),
            prefetch_cache_hits: AtomicU64::new(0),
            routed_selections: AtomicU64::new(0),
            routing_heat: Mutex::new(BTreeMap::new()),
        }
    }

    pub(crate) fn host_hit(&self) {
        self.increment(&self.host_cache_hits, 1);
    }

    pub(crate) fn device_hit(&self) {
        self.increment(&self.device_cache_hits, 1);
    }

    pub(crate) fn storage_read(&self, bytes: u64) {
        self.increment(&self.storage_reads, 1);
        self.increment(&self.storage_bytes_read, bytes);
    }

    pub(crate) fn device_promotion(&self, bytes: u64) {
        self.increment(&self.device_bytes_promoted, bytes);
    }

    pub(crate) fn host_eviction(&self) {
        self.increment(&self.host_evictions, 1);
    }

    pub(crate) fn device_eviction(&self) {
        self.increment(&self.device_evictions, 1);
    }

    pub(crate) fn prefetch(&self, cache_hit: bool) {
        self.increment(&self.prefetched_weights, 1);
        if cache_hit {
            self.increment(&self.prefetch_cache_hits, 1);
        }
    }

    pub(crate) fn routes(&self, batch: &RoutedExpertBatch) {
        if self.mode == TelemetryMode::Disabled {
            return;
        }
        let selections = batch.selections().iter().map(Vec::len).sum::<usize>() as u64;
        self.increment(&self.routed_selections, selections);
        if self.mode != TelemetryMode::Detailed {
            return;
        }
        let mut heat = lock(&self.routing_heat);
        for expert in batch.experts() {
            let key = ExpertKey {
                layer: batch.layer(),
                expert: *expert,
            };
            let count = batch.assignments(*expert).len() as u64;
            let entry = heat.entry(key).or_default();
            *entry = entry.saturating_add(count);
        }
    }

    pub(crate) fn snapshot(
        &self,
        host_resident_bytes: u64,
        device_resident_bytes: u64,
    ) -> PlacementTelemetry {
        PlacementTelemetry {
            schema: PlacementTelemetry::SCHEMA.to_string(),
            mode: self.mode,
            host_cache_hits: self.load(&self.host_cache_hits),
            device_cache_hits: self.load(&self.device_cache_hits),
            storage_reads: self.load(&self.storage_reads),
            storage_bytes_read: self.load(&self.storage_bytes_read),
            device_bytes_promoted: self.load(&self.device_bytes_promoted),
            host_evictions: self.load(&self.host_evictions),
            device_evictions: self.load(&self.device_evictions),
            prefetched_weights: self.load(&self.prefetched_weights),
            prefetch_cache_hits: self.load(&self.prefetch_cache_hits),
            routed_selections: self.load(&self.routed_selections),
            host_resident_bytes,
            device_resident_bytes,
            routing_heat: self.route_heat(),
        }
    }

    pub(crate) fn history(&self, weights_sha256: &str) -> Result<RoutingHistory> {
        if self.mode != TelemetryMode::Detailed {
            return Err(PowerError::PolicyViolation(
                "routing history requires explicitly enabled detailed telemetry".to_string(),
            ));
        }
        Ok(RoutingHistory {
            schema: RoutingHistory::SCHEMA.to_string(),
            weights_sha256: weights_sha256.to_string(),
            entries: self.route_heat(),
        })
    }

    pub(crate) fn restore_history(
        &self,
        history: &RoutingHistory,
        weights_sha256: &str,
    ) -> Result<()> {
        if self.mode != TelemetryMode::Detailed {
            return Err(PowerError::PolicyViolation(
                "restoring routing history requires explicitly enabled detailed telemetry"
                    .to_string(),
            ));
        }
        if history.schema != RoutingHistory::SCHEMA || history.weights_sha256 != weights_sha256 {
            return Err(PowerError::InvalidFormat(
                "routing history schema or model digest does not match this weight store"
                    .to_string(),
            ));
        }
        let mut heat = lock(&self.routing_heat);
        for entry in &history.entries {
            let current = heat.entry(entry.key).or_default();
            *current = current.saturating_add(entry.selections);
        }
        Ok(())
    }

    fn route_heat(&self) -> Vec<RouteHeat> {
        if self.mode != TelemetryMode::Detailed {
            return Vec::new();
        }
        lock(&self.routing_heat)
            .iter()
            .map(|(key, selections)| RouteHeat {
                key: *key,
                selections: *selections,
            })
            .collect()
    }

    fn increment(&self, counter: &AtomicU64, value: u64) {
        if self.mode == TelemetryMode::Disabled {
            return;
        }
        let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_add(value))
        });
    }

    fn load(&self, counter: &AtomicU64) -> u64 {
        if self.mode == TelemetryMode::Disabled {
            0
        } else {
            counter.load(Ordering::Relaxed)
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::RoutedExpert;

    #[test]
    fn telemetry_is_private_by_default() {
        let telemetry = Telemetry::new(TelemetryMode::Disabled);
        let batch = RoutedExpertBatch::new(
            2,
            vec![vec![RoutedExpert {
                expert: 3,
                weight: 1.0,
            }]],
            8,
            1,
        )
        .unwrap();
        telemetry.routes(&batch);
        telemetry.storage_read(128);

        let snapshot = telemetry.snapshot(0, 0);
        assert_eq!(snapshot.routed_selections, 0);
        assert_eq!(snapshot.storage_bytes_read, 0);
        assert!(snapshot.routing_heat.is_empty());
        assert!(telemetry.history("hash").is_err());
    }

    #[test]
    fn detailed_history_is_bound_to_weight_digest() {
        let telemetry = Telemetry::new(TelemetryMode::Detailed);
        let batch = RoutedExpertBatch::new(
            4,
            vec![vec![RoutedExpert {
                expert: 1,
                weight: 1.0,
            }]],
            2,
            1,
        )
        .unwrap();
        telemetry.routes(&batch);
        let history = telemetry.history("weights-a").unwrap();
        assert_eq!(history.entries[0].selections, 1);

        let restored = Telemetry::new(TelemetryMode::Detailed);
        assert!(restored.restore_history(&history, "weights-b").is_err());
        restored.restore_history(&history, "weights-a").unwrap();
        assert_eq!(
            restored.history("weights-a").unwrap().entries[0].selections,
            1
        );
    }
}
