//! Process-local collector for the public worker observation contract.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::serving::{
    AdmissionObservation, PromptCacheObservation, ServingPhase, TransferHealth, WorkerCapabilities,
    WorkerObservation, WORKER_OBSERVATION_SCHEMA,
};

use super::state::AppState;

#[derive(Clone)]
pub(super) struct WorkerObservationSource {
    inner: Arc<WorkerObservationSourceInner>,
}

struct WorkerObservationSourceInner {
    worker_epoch: Uuid,
    observation_generation: AtomicU64,
}

impl WorkerObservationSource {
    pub(super) fn new() -> Self {
        Self {
            inner: Arc::new(WorkerObservationSourceInner {
                worker_epoch: Uuid::new_v4(),
                observation_generation: AtomicU64::new(0),
            }),
        }
    }

    pub(super) fn observe(
        &self,
        state: &AppState,
        observed_at: DateTime<Utc>,
    ) -> WorkerObservation {
        let generation = self
            .inner
            .observation_generation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(1))
            })
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        let supporting_backends =
            u64::try_from(state.backends.prompt_cache_backend_names().len()).unwrap_or(u64::MAX);
        let (entries, capacity) = state.backends.prompt_cache_metrics().into_iter().fold(
            (0_u64, 0_u64),
            |(entries, capacity), (_, snapshot)| {
                (
                    entries.saturating_add(snapshot.entries),
                    capacity.saturating_add(snapshot.capacity),
                )
            },
        );
        let pressure_basis_points = cache_pressure_basis_points(entries, capacity);
        let prompt_cache_supported = supporting_backends > 0;
        let ttl = chrono::Duration::seconds(
            i64::try_from(state.config.worker_observation_ttl_seconds).unwrap_or(i64::MAX),
        );

        WorkerObservation {
            schema: WORKER_OBSERVATION_SCHEMA.to_string(),
            worker_epoch: self.inner.worker_epoch,
            observation_generation: generation,
            observed_at,
            expires_at: observed_at + ttl,
            capabilities: WorkerCapabilities {
                phases: vec![ServingPhase::Aggregated],
                prompt_cache: prompt_cache_supported,
                state_transfer: false,
            },
            ready_phases: vec![ServingPhase::Aggregated],
            admission: AdmissionObservation {
                active_limit: (state.config.max_concurrent_requests > 0)
                    .then_some(state.config.max_concurrent_requests),
                active: state.metrics.running_requests(),
                waiting: state.metrics.waiting_requests(),
            },
            prompt_cache: PromptCacheObservation {
                supported: prompt_cache_supported,
                entries,
                capacity,
                pressure_basis_points,
            },
            transfer_health: TransferHealth::Unsupported,
        }
    }
}

fn cache_pressure_basis_points(entries: u64, capacity: u64) -> u16 {
    if capacity == 0 {
        return if entries == 0 { 0 } else { 10_000 };
    }
    let pressure = u128::from(entries)
        .saturating_mul(10_000)
        .checked_div(u128::from(capacity))
        .unwrap_or(10_000)
        .min(10_000);
    u16::try_from(pressure).unwrap_or(10_000)
}

#[cfg(test)]
mod tests {
    use super::cache_pressure_basis_points;

    #[test]
    fn cache_pressure_is_bounded_and_handles_zero_capacity() {
        assert_eq!(cache_pressure_basis_points(0, 0), 0);
        assert_eq!(cache_pressure_basis_points(1, 0), 10_000);
        assert_eq!(cache_pressure_basis_points(1, 4), 2_500);
        assert_eq!(cache_pressure_basis_points(8, 4), 10_000);
    }
}
