use std::collections::BTreeMap;

use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use super::{
    ExecutionPermit, PlacementPreference, PrefetchReport, PrefetchTask, WeightHierarchy, WeightKey,
    WeightRequest,
};
use crate::error::{PowerError, Result};

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

    fn normalize_prefetch(&self, requests: Vec<WeightRequest>) -> Result<Vec<WeightRequest>> {
        if requests.len() > self.inner.policy.max_prefetch_items {
            return Err(PowerError::InvalidRequest(format!(
                "prefetch requested {} weights, exceeding the {} item limit",
                requests.len(),
                self.inner.policy.max_prefetch_items
            )));
        }
        let mut unique = BTreeMap::<WeightKey, PlacementPreference>::new();
        let mut total_bytes = 0_u64;
        for request in requests {
            let descriptor = self.validate_request(&request)?;
            let bytes = descriptor.bytes;
            let placement = self.resolve_placement(request.placement);
            match unique.entry(request.key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    total_bytes = total_bytes.checked_add(bytes).ok_or_else(|| {
                        PowerError::InvalidRequest("prefetch byte length overflowed".to_string())
                    })?;
                    entry.insert(placement);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if placement_rank(placement) > placement_rank(*entry.get()) {
                        entry.insert(placement);
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
            .map(|(key, placement)| WeightRequest { key, placement })
            .collect())
    }

    async fn prefetch(
        &self,
        requested: usize,
        requests: Vec<WeightRequest>,
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
        };

        let mut requests = requests.into_iter();
        let mut workers = JoinSet::new();
        let worker_limit = self.inner.policy.max_prefetch_workers.min(report.unique);
        for _ in 0..worker_limit {
            if let Some(request) = requests.next() {
                self.spawn_prefetch_load(&mut workers, request, &permit, &cancellation);
            }
        }

        while !workers.is_empty() {
            let joined = tokio::select! {
                () = cancellation.cancelled() => {
                    workers.abort_all();
                    return Err(PowerError::InferenceFailed(
                        "weight prefetch was cancelled".to_string(),
                    ));
                }
                joined = workers.join_next() => joined,
            };
            let (bytes, cache_hit) = match joined {
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
                None => break,
            };
            report.bytes = report.bytes.saturating_add(bytes);
            if cache_hit {
                report.cache_hits += 1;
            } else {
                report.materialized += 1;
            }
            self.inner.telemetry.prefetch(cache_hit);

            if let Some(request) = requests.next() {
                self.spawn_prefetch_load(&mut workers, request, &permit, &cancellation);
            }
        }
        Ok(report)
    }

    fn spawn_prefetch_load(
        &self,
        workers: &mut JoinSet<Result<(u64, bool)>>,
        request: WeightRequest,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) {
        let hierarchy = self.clone();
        let permit = permit.clone();
        let cancellation = cancellation.clone();
        workers.spawn_blocking(move || {
            let weight = hierarchy.load_prefetch(&request, &permit, &cancellation)?;
            Ok((weight.bytes(), weight.cache_hit()))
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
