use std::collections::BTreeMap;

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
        let requested = requests.len();
        let normalized = self.normalize_prefetch(requests)?;
        let runtime = tokio::runtime::Handle::try_current().map_err(|_| {
            PowerError::BackendNotAvailable(
                "weight prefetch requires an active Tokio runtime".to_string(),
            )
        })?;
        let hierarchy = self.clone();
        let permit = permit.clone();
        let handle = runtime.spawn_blocking(move || {
            hierarchy.prefetch_blocking(requested, normalized, &permit, &cancellation)
        });
        Ok(PrefetchTask { handle })
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

    fn prefetch_blocking(
        &self,
        requested: usize,
        requests: Vec<WeightRequest>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<PrefetchReport> {
        let mut report = PrefetchReport {
            requested,
            unique: requests.len(),
            cache_hits: 0,
            materialized: 0,
            bytes: 0,
        };
        for request in requests {
            let weight = self.load(&request, permit, cancellation)?;
            report.bytes = report.bytes.saturating_add(weight.bytes());
            if weight.cache_hit() {
                report.cache_hits += 1;
            } else {
                report.materialized += 1;
            }
            self.inner.telemetry.prefetch(weight.cache_hit());
        }
        Ok(report)
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
