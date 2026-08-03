//! Ordered current-layer weight staging for model-owned computation.

use std::collections::{BTreeSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::Notify;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use super::staging_types::{
    elapsed_nanos, StagedBatchState, StagedWeightBatch, StagedWeightBatchCompletion,
    StagedWeightBatchReport, StagedWeightGroupRequest,
};
use super::{
    read, CacheAccess, ExecutionPermit, ResidentWeight, WeightHierarchy, WeightKey, WeightRequest,
};
use crate::admission::AdmissionPermit;
use crate::error::{PowerError, Result};
use crate::inference::TelemetryMode;

use super::load_window::BackgroundLoadWindow;

struct ValidatedStagedBatch {
    groups: Vec<Vec<ValidatedWeight>>,
    requested_weights: usize,
    bytes: u64,
}

struct ValidatedWeight {
    request: WeightRequest,
    bytes: u64,
}

struct PendingWeight {
    group_index: usize,
    weight_index: usize,
    request: WeightRequest,
    bytes: u64,
}

struct CompletedWeight {
    group_index: usize,
    weight_index: usize,
    scheduled_bytes: u64,
    weight: ResidentWeight,
    service_nanos: u64,
}

#[derive(Clone)]
struct StagedBatchSignal {
    state: Arc<Mutex<StagedBatchState>>,
    ready: Arc<Notify>,
}

impl WeightHierarchy {
    /// Starts a bounded current-layer staging batch.
    ///
    /// The complete batch is validated before cache heat or I/O is touched.
    /// Already resident atomic groups are immediately available through
    /// [`StagedWeightBatch::ready_groups`]; only missing weights enter the
    /// existing bounded Tokio blocking path. Staging shares prefetch task,
    /// worker, item, and byte bounds.
    pub fn start_staged_batch(
        &self,
        groups: Vec<StagedWeightGroupRequest>,
        permit: &ExecutionPermit,
        cancellation: CancellationToken,
    ) -> Result<StagedWeightBatch> {
        self.validate_permit(permit)?;
        self.check_cancelled(&cancellation)?;
        let validated = self.validate_staged_batch(groups)?;
        let runtime = tokio::runtime::Handle::try_current().map_err(|_| {
            PowerError::BackendNotAvailable(
                "staged weight batches require an active Tokio runtime".to_string(),
            )
        })?;
        let admission = self.inner.prefetch_admission.try_acquire().ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "weight hierarchy already has {} active background weight task(s)",
                self.inner.policy.max_prefetch_tasks
            ))
        })?;
        self.check_cancelled(&cancellation)?;

        let mut pending = Vec::new();
        let mut resident_weights = 0_usize;
        let mut state = StagedBatchState::with_capacity(validated.groups.len());
        {
            let _operation = read(&self.inner.operations);
            for (group_index, requests) in validated.groups.iter().enumerate() {
                let mut weights = Vec::with_capacity(requests.len());
                for (weight_index, validated_weight) in requests.iter().enumerate() {
                    let request = &validated_weight.request;
                    if let Some(weight) =
                        self.cached(&request.key, request.placement, None, CacheAccess::Demand)
                    {
                        resident_weights = resident_weights.saturating_add(1);
                        weights.push(Some(weight));
                    } else {
                        weights.push(None);
                        pending.push(PendingWeight {
                            group_index,
                            weight_index,
                            request: request.clone(),
                            bytes: validated_weight.bytes,
                        });
                    }
                }
                state.push(weights);
            }
        }
        let ready_groups = state.ready_count();
        let requested_groups = state.len();
        let report = StagedWeightBatchReport {
            requested_groups,
            ready_groups,
            pending_groups: requested_groups.saturating_sub(ready_groups),
            requested_weights: validated.requested_weights,
            resident_weights,
            bytes: validated.bytes,
            ..StagedWeightBatchReport::default()
        };
        let state = Arc::new(Mutex::new(state));
        let ready = Arc::new(Notify::new());
        let signal = StagedBatchSignal {
            state: Arc::clone(&state),
            ready: Arc::clone(&ready),
        };
        let task_cancellation = cancellation.child_token();
        let worker_cancellation = task_cancellation.clone();
        let hierarchy = self.clone();
        let worker_hierarchy = hierarchy.clone();
        let permit = permit.clone();
        let worker_signal = signal.clone();
        let handle = runtime.spawn(async move {
            let result = worker_hierarchy
                .complete_staged_batch(
                    pending,
                    permit,
                    worker_cancellation,
                    worker_signal.clone(),
                    report,
                    admission,
                )
                .await;
            {
                let mut state = lock(&worker_signal.state);
                match &result {
                    Ok(_) => state.finish_success(),
                    Err(error) => state.finish_error(error.to_string()),
                }
            }
            worker_signal.ready.notify_one();
            result
        });
        Ok(StagedWeightBatch::new(
            handle,
            task_cancellation,
            state,
            ready,
            hierarchy,
        ))
    }

    fn validate_staged_batch(
        &self,
        groups: Vec<StagedWeightGroupRequest>,
    ) -> Result<ValidatedStagedBatch> {
        if groups.is_empty() {
            return Err(PowerError::InvalidRequest(
                "staged weight batch must contain at least one group".to_string(),
            ));
        }
        let mut layer = None;
        let mut seen = BTreeSet::<WeightKey>::new();
        let mut requested_weights = 0_usize;
        let mut bytes = 0_u64;
        let inflight_limit = self.inner.policy.background_inflight_bytes();
        let mut validated = Vec::with_capacity(groups.len());
        for group in groups {
            if group.weights.is_empty() {
                return Err(PowerError::InvalidRequest(
                    "staged weight groups must contain at least one weight".to_string(),
                ));
            }
            requested_weights = requested_weights
                .checked_add(group.weights.len())
                .ok_or_else(|| {
                    PowerError::InvalidRequest("staged weight item count overflowed".to_string())
                })?;
            if requested_weights > self.inner.policy.max_prefetch_items {
                return Err(PowerError::InvalidRequest(format!(
                    "staged batch requested {requested_weights} weights, exceeding the {} item limit",
                    self.inner.policy.max_prefetch_items
                )));
            }

            let mut requests = Vec::with_capacity(group.weights.len());
            for request in group.weights {
                let descriptor = self.validate_request(&request)?;
                if descriptor.bytes > inflight_limit {
                    return Err(PowerError::InvalidRequest(format!(
                        "staged weight '{}' requires {} bytes, exceeding the {inflight_limit} byte background in-flight limit",
                        request.key.name, descriptor.bytes
                    )));
                }
                match layer {
                    Some(expected) if expected != request.key.layer => {
                        return Err(PowerError::InvalidRequest(
                            "staged weight batch must describe exactly one current layer"
                                .to_string(),
                        ));
                    }
                    None => layer = Some(request.key.layer),
                    Some(_) => {}
                }
                if !seen.insert(request.key.clone()) {
                    return Err(PowerError::InvalidRequest(
                        "staged weight batch contains a duplicate weight key".to_string(),
                    ));
                }
                bytes = bytes.checked_add(descriptor.bytes).ok_or_else(|| {
                    PowerError::InvalidRequest("staged weight byte length overflowed".to_string())
                })?;
                if bytes > self.inner.policy.max_prefetch_bytes {
                    return Err(PowerError::InvalidRequest(format!(
                        "staged batch requires {bytes} bytes, exceeding the {} byte limit",
                        self.inner.policy.max_prefetch_bytes
                    )));
                }
                requests.push(ValidatedWeight {
                    request: WeightRequest {
                        key: request.key,
                        placement: self.resolve_placement(request.placement),
                    },
                    bytes: descriptor.bytes,
                });
            }
            validated.push(requests);
        }
        Ok(ValidatedStagedBatch {
            groups: validated,
            requested_weights,
            bytes,
        })
    }

    async fn complete_staged_batch(
        &self,
        pending: Vec<PendingWeight>,
        permit: ExecutionPermit,
        cancellation: CancellationToken,
        signal: StagedBatchSignal,
        mut report: StagedWeightBatchReport,
        _admission: AdmissionPermit,
    ) -> Result<StagedWeightBatchCompletion> {
        let background_started = Instant::now();
        let measure_timing = self.inner.policy.telemetry != TelemetryMode::Disabled;
        if pending.is_empty() {
            if measure_timing {
                report.background_elapsed_nanos = elapsed_nanos(background_started);
            }
            let groups = lock(&signal.state).completed_groups()?;
            return Ok(StagedWeightBatchCompletion { groups, report });
        }
        let worker_limit = self.inner.policy.max_prefetch_workers.min(pending.len());
        let mut pending = VecDeque::from(pending);
        let mut workers = JoinSet::new();
        let mut window =
            BackgroundLoadWindow::new(worker_limit, self.inner.policy.background_inflight_bytes())?;

        while !pending.is_empty() || !window.is_idle() {
            while let Some(weight) = window.take_fitting(&mut pending, |weight| weight.bytes)? {
                self.spawn_staged_load(
                    &mut workers,
                    weight,
                    &permit,
                    &cancellation,
                    measure_timing,
                );
            }
            if window.is_idle() {
                return Err(PowerError::InferenceFailed(
                    "bounded staged weight queue made no progress".to_string(),
                ));
            }
            let joined = tokio::select! {
                () = cancellation.cancelled() => {
                    workers.abort_all();
                    return Err(PowerError::InferenceFailed(
                        "staged weight batch was cancelled".to_string(),
                    ));
                }
                joined = workers.join_next() => joined,
            };
            let completed = match joined {
                Some(Ok(Ok(completed))) => completed,
                Some(Ok(Err(error))) => {
                    cancellation.cancel();
                    workers.abort_all();
                    return Err(error);
                }
                Some(Err(error)) => {
                    cancellation.cancel();
                    workers.abort_all();
                    return Err(PowerError::InferenceFailed(format!(
                        "staged weight worker failed: {error}"
                    )));
                }
                None => {
                    return Err(PowerError::InferenceFailed(
                        "staged worker set ended before the bounded window drained".to_string(),
                    ))
                }
            };
            window.release(completed.scheduled_bytes)?;
            if completed.weight.bytes() != completed.scheduled_bytes {
                return Err(PowerError::InferenceFailed(
                    "staged weight size changed after bounded admission".to_string(),
                ));
            }
            if completed.weight.cache_hit() {
                report.load_cache_hits = report.load_cache_hits.saturating_add(1);
            } else {
                report.loaded_weights = report.loaded_weights.saturating_add(1);
            }
            report.cumulative_service_nanos = report
                .cumulative_service_nanos
                .saturating_add(completed.service_nanos);
            {
                lock(&signal.state).insert(
                    completed.group_index,
                    completed.weight_index,
                    completed.weight,
                )?;
            }
            signal.ready.notify_one();
        }
        self.check_cancelled(&cancellation)?;
        report.peak_inflight_weights = window.peak_workers();
        report.peak_inflight_bytes = window.peak_bytes();
        if measure_timing {
            report.background_elapsed_nanos = elapsed_nanos(background_started);
        }
        let groups = lock(&signal.state).completed_groups()?;
        Ok(StagedWeightBatchCompletion { groups, report })
    }

    fn spawn_staged_load(
        &self,
        workers: &mut JoinSet<Result<CompletedWeight>>,
        pending: PendingWeight,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        measure_timing: bool,
    ) {
        let hierarchy = self.clone();
        let permit = permit.clone();
        let cancellation = cancellation.clone();
        workers.spawn_blocking(move || {
            let service_started = Instant::now();
            // Current-layer staging is exact demand. It must not masquerade as
            // speculative prefetch in cache heat or usefulness accounting.
            let weight = hierarchy.load(&pending.request, &permit, &cancellation)?;
            Ok(CompletedWeight {
                group_index: pending.group_index,
                weight_index: pending.weight_index,
                scheduled_bytes: pending.bytes,
                weight,
                service_nanos: if measure_timing {
                    elapsed_nanos(service_started)
                } else {
                    0
                },
            })
        });
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
