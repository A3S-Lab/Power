use std::collections::{BTreeMap, VecDeque};

use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use super::{
    ExecutionPermit, PlacementPreference, PrefetchReport, PrefetchTask, WeightHierarchy, WeightKey,
    WeightRequest,
};
use crate::error::{PowerError, Result};

use super::load_window::BackgroundLoadWindow;

struct PendingPrefetch {
    request: WeightRequest,
    bytes: u64,
}

struct CompletedPrefetch {
    scheduled_bytes: u64,
    weight_bytes: u64,
    cache_hit: bool,
}

impl WeightHierarchy {
    /// Starts bounded prefetch immediately on Tokio's blocking pool. A model
    /// can start the next layer's task, compute the current layer, then await
    /// the returned handle to overlap I/O and compute.
    pub fn start_prefetch(
        &self,
        requests: Vec<WeightRequest>,
        permit: &ExecutionPermit,
        cancellation: CancellationToken,
    ) -> Result<PrefetchTask> {
        self.validate_permit(permit)?;
        self.check_cancelled(&cancellation)?;
        let requested = requests.len();
        let normalized = self.normalize_prefetch(requests)?;
        let runtime = tokio::runtime::Handle::try_current().map_err(|_| {
            PowerError::BackendNotAvailable(
                "weight prefetch requires an active Tokio runtime".to_string(),
            )
        })?;
        let admission = self.inner.prefetch_admission.try_acquire().ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "weight hierarchy already has {} active prefetch task(s)",
                self.inner.policy.max_prefetch_tasks
            ))
        })?;
        let hierarchy = self.clone();
        let permit = permit.clone();
        let task_cancellation = cancellation.child_token();
        let worker_cancellation = task_cancellation.clone();
        let handle = runtime.spawn(async move {
            hierarchy
                .prefetch(
                    requested,
                    normalized,
                    permit,
                    worker_cancellation,
                    admission,
                )
                .await
        });
        Ok(PrefetchTask {
            handle: Some(handle),
            cancellation: task_cancellation,
        })
    }

    fn normalize_prefetch(&self, requests: Vec<WeightRequest>) -> Result<Vec<PendingPrefetch>> {
        if requests.len() > self.inner.policy.max_prefetch_items {
            return Err(PowerError::InvalidRequest(format!(
                "prefetch requested {} weights, exceeding the {} item limit",
                requests.len(),
                self.inner.policy.max_prefetch_items
            )));
        }
        let mut unique = BTreeMap::<WeightKey, (PlacementPreference, u64)>::new();
        let mut total_bytes = 0_u64;
        let inflight_limit = self.inner.policy.background_inflight_bytes();
        for request in requests {
            let descriptor = self.validate_request(&request)?;
            let bytes = descriptor.bytes;
            if bytes > inflight_limit {
                return Err(PowerError::InvalidRequest(format!(
                    "prefetch weight '{}' requires {bytes} bytes, exceeding the {inflight_limit} byte background in-flight limit",
                    request.key.name
                )));
            }
            let placement = self.resolve_placement(request.placement);
            match unique.entry(request.key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    total_bytes = total_bytes.checked_add(bytes).ok_or_else(|| {
                        PowerError::InvalidRequest("prefetch byte length overflowed".to_string())
                    })?;
                    entry.insert((placement, bytes));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if placement_rank(placement) > placement_rank(entry.get().0) {
                        entry.get_mut().0 = placement;
                    }
                }
            }
        }
        if total_bytes > self.inner.policy.max_prefetch_bytes {
            return Err(PowerError::InvalidRequest(format!(
                "prefetch requires {total_bytes} bytes, exceeding the {} byte limit",
                self.inner.policy.max_prefetch_bytes
            )));
        }
        Ok(unique
            .into_iter()
            .map(|(key, (placement, bytes))| PendingPrefetch {
                request: WeightRequest { key, placement },
                bytes,
            })
            .collect())
    }

    async fn prefetch(
        &self,
        requested: usize,
        requests: Vec<PendingPrefetch>,
        permit: ExecutionPermit,
        cancellation: CancellationToken,
        _admission: crate::admission::AdmissionPermit,
    ) -> Result<PrefetchReport> {
        let mut report = PrefetchReport {
            requested,
            unique: requests.len(),
            cache_hits: 0,
            materialized: 0,
            bytes: 0,
            peak_inflight_weights: 0,
            peak_inflight_bytes: 0,
        };
        if requests.is_empty() {
            self.inner.telemetry.prefetch_batch(&report);
            return Ok(report);
        }

        let mut requests = VecDeque::from(requests);
        let mut workers = JoinSet::new();
        let worker_limit = self.inner.policy.max_prefetch_workers.min(report.unique);
        let mut window =
            BackgroundLoadWindow::new(worker_limit, self.inner.policy.background_inflight_bytes())?;

        while !requests.is_empty() || !window.is_idle() {
            while let Some(request) = window.take_fitting(&mut requests, |request| request.bytes)? {
                self.spawn_prefetch_load(&mut workers, request, &permit, &cancellation);
            }
            if window.is_idle() {
                return Err(PowerError::InferenceFailed(
                    "bounded prefetch queue made no progress".to_string(),
                ));
            }
            let joined = tokio::select! {
                () = cancellation.cancelled() => {
                    workers.abort_all();
                    return Err(PowerError::InferenceFailed(
                        "weight prefetch was cancelled".to_string(),
                    ));
                }
                joined = workers.join_next() => joined,
            };
            let completed = match joined {
                Some(Ok(Ok(result))) => result,
                Some(Ok(Err(error))) => {
                    cancellation.cancel();
                    workers.abort_all();
                    return Err(error);
                }
                Some(Err(error)) => {
                    cancellation.cancel();
                    workers.abort_all();
                    return Err(PowerError::InferenceFailed(format!(
                        "weight prefetch worker failed: {error}"
                    )));
                }
                None => {
                    return Err(PowerError::InferenceFailed(
                        "prefetch worker set ended before the bounded window drained".to_string(),
                    ))
                }
            };
            window.release(completed.scheduled_bytes)?;
            if completed.weight_bytes != completed.scheduled_bytes {
                return Err(PowerError::InferenceFailed(
                    "prefetched weight size changed after bounded admission".to_string(),
                ));
            }
            report.bytes = report.bytes.saturating_add(completed.weight_bytes);
            if completed.cache_hit {
                report.cache_hits += 1;
            } else {
                report.materialized += 1;
            }
            self.inner.telemetry.prefetch(completed.cache_hit);
        }
        report.peak_inflight_weights = window.peak_workers();
        report.peak_inflight_bytes = window.peak_bytes();
        self.inner.telemetry.prefetch_batch(&report);
        Ok(report)
    }

    fn spawn_prefetch_load(
        &self,
        workers: &mut JoinSet<Result<CompletedPrefetch>>,
        pending: PendingPrefetch,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) {
        let hierarchy = self.clone();
        let permit = permit.clone();
        let cancellation = cancellation.clone();
        workers.spawn_blocking(move || {
            let weight = hierarchy.load_prefetch(&pending.request, &permit, &cancellation)?;
            Ok(CompletedPrefetch {
                scheduled_bytes: pending.bytes,
                weight_bytes: weight.bytes(),
                cache_hit: weight.cache_hit(),
            })
        });
    }
}

fn placement_rank(placement: PlacementPreference) -> u8 {
    match placement {
        PlacementPreference::Streaming => 0,
        PlacementPreference::Host => 1,
        PlacementPreference::Device => 2,
        PlacementPreference::Fastest => 3,
    }
}
