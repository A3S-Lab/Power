use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Versioned, model-neutral worker observation schema.
pub const WORKER_OBSERVATION_SCHEMA: &str = "a3s.power.worker-observation.v1";

/// Execution phases a Power worker can own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ServingPhase {
    /// One worker owns prompt evaluation and token generation.
    Aggregated,
    /// One worker owns prompt evaluation for a disaggregated deployment.
    Prefill,
    /// One worker owns token generation for a disaggregated deployment.
    Decode,
}

/// Stable capabilities of one Power process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerCapabilities {
    /// Phases this process can execute under its active profile.
    pub phases: Vec<ServingPhase>,
    /// Whether at least one registered backend guarantees keyed prefix reuse.
    pub prompt_cache: bool,
    /// Whether authenticated opaque state transfer is available.
    pub state_transfer: bool,
}

/// Content-free admission pressure at one observation instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionObservation {
    /// Configured active execution limit, or `None` when locally unbounded.
    pub active_limit: Option<u64>,
    /// Requests currently holding an execution permit.
    pub active: u64,
    /// Requests currently waiting for an execution permit.
    pub waiting: u64,
}

/// Aggregate prompt-cache pressure without keys, prompts, tokens, or tenants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptCacheObservation {
    /// Whether keyed prompt-prefix reuse is available.
    pub supported: bool,
    /// Reusable contexts currently resident across instrumented backends.
    pub entries: u64,
    /// Current process capacity for loaded models and supporting backends.
    pub capacity: u64,
    /// Saturating cache pressure in basis points (`0..=10000`).
    pub pressure_basis_points: u16,
}

/// Health of the model-neutral opaque state-transfer boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TransferHealth {
    /// This process does not advertise cross-worker state transfer.
    Unsupported,
    /// The configured transfer boundary is ready.
    Ready,
    /// Transfer is available with reduced capacity.
    Degraded,
    /// The configured transfer boundary cannot accept work.
    Unavailable,
}

/// One bounded point-in-time worker observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerObservation {
    pub schema: String,
    /// Random process epoch; changes on every process restart.
    pub worker_epoch: Uuid,
    /// Positive monotonic generation within `worker_epoch`.
    pub observation_generation: u64,
    pub observed_at: DateTime<Utc>,
    /// Exclusive validity bound chosen by the local Power configuration.
    pub expires_at: DateTime<Utc>,
    pub capabilities: WorkerCapabilities,
    /// Phases currently ready to accept work.
    pub ready_phases: Vec<ServingPhase>,
    pub admission: AdmissionObservation,
    pub prompt_cache: PromptCacheObservation,
    pub transfer_health: TransferHealth,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_is_closed_and_contains_no_request_identity() {
        let observed_at = DateTime::parse_from_rfc3339("2026-08-28T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let value = WorkerObservation {
            schema: WORKER_OBSERVATION_SCHEMA.to_string(),
            worker_epoch: Uuid::nil(),
            observation_generation: 1,
            observed_at,
            expires_at: observed_at + chrono::Duration::seconds(15),
            capabilities: WorkerCapabilities {
                phases: vec![ServingPhase::Aggregated],
                prompt_cache: true,
                state_transfer: false,
            },
            ready_phases: vec![ServingPhase::Aggregated],
            admission: AdmissionObservation {
                active_limit: Some(8),
                active: 2,
                waiting: 1,
            },
            prompt_cache: PromptCacheObservation {
                supported: true,
                entries: 2,
                capacity: 8,
                pressure_basis_points: 2_500,
            },
            transfer_health: TransferHealth::Unsupported,
        };

        let json = serde_json::to_string(&value).unwrap();
        assert!(json.contains(WORKER_OBSERVATION_SCHEMA));
        for forbidden in ["prompt_cache_key", "tenant", "token", "kv_bytes"] {
            assert!(!json.contains(forbidden));
        }

        let mut document = serde_json::to_value(&value).unwrap();
        document["unknown"] = serde_json::json!(true);
        assert!(serde_json::from_value::<WorkerObservation>(document).is_err());
    }

    #[test]
    fn contract_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<WorkerObservation>();
        assert_send_sync::<WorkerCapabilities>();
    }
}
