use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::{ResidentWeight, WeightHierarchy, WeightRequest};
use crate::error::{PowerError, Result};
use crate::inference::TelemetryMode;

/// One atomic group of weights needed by model-owned current-layer compute.
///
/// The request's position in a batch is its canonical group index. Power does
/// not attach a model-specific expert, route, or operation identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StagedWeightGroupRequest {
    pub weights: Vec<WeightRequest>,
}

impl StagedWeightGroupRequest {
    pub fn new(weights: Vec<WeightRequest>) -> Self {
        Self { weights }
    }
}

/// A completely materialized atomic group.
///
/// Groups may become ready out of order. Model crates must place their output
/// in a slot selected by `canonical_index` and perform any reduction in their
/// architecture's canonical order.
#[derive(Clone)]
pub struct StagedWeightGroup {
    canonical_index: usize,
    weights: Vec<ResidentWeight>,
}

impl StagedWeightGroup {
    pub fn canonical_index(&self) -> usize {
        self.canonical_index
    }

    pub fn weights(&self) -> &[ResidentWeight] {
        &self.weights
    }

    pub fn into_weights(self) -> Vec<ResidentWeight> {
        self.weights
    }
}

impl std::fmt::Debug for StagedWeightGroup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StagedWeightGroup")
            .field("canonical_index", &self.canonical_index)
            .field("weights", &self.weights.len())
            .finish()
    }
}

/// Aggregate evidence for one successful staged batch.
///
/// `ready_groups` and `pending_groups` describe the state when staging starts.
/// Timing is reported only when aggregate or detailed telemetry is enabled.
/// No tensor name, route, expert identity, or tensor value is included.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StagedWeightBatchReport {
    pub requested_groups: usize,
    pub ready_groups: usize,
    pub pending_groups: usize,
    pub requested_weights: usize,
    pub resident_weights: usize,
    pub loaded_weights: usize,
    pub load_cache_hits: usize,
    pub bytes: u64,
    pub cumulative_service_nanos: u64,
    pub background_elapsed_nanos: u64,
    pub foreground_wait_nanos: u64,
}

/// Canonically ordered completion of a staged batch.
pub struct StagedWeightBatchCompletion {
    pub groups: Vec<StagedWeightGroup>,
    pub report: StagedWeightBatchReport,
}

impl std::fmt::Debug for StagedWeightBatchCompletion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StagedWeightBatchCompletion")
            .field("groups", &self.groups)
            .field("report", &self.report)
            .finish()
    }
}

/// Cancellable staged-batch task sharing the hierarchy's prefetch admission.
pub struct StagedWeightBatch {
    handle: Option<JoinHandle<Result<StagedWeightBatchCompletion>>>,
    cancellation: CancellationToken,
    state: Arc<Mutex<StagedBatchState>>,
    hierarchy: WeightHierarchy,
}

impl StagedWeightBatch {
    pub(super) fn new(
        handle: JoinHandle<Result<StagedWeightBatchCompletion>>,
        cancellation: CancellationToken,
        state: Arc<Mutex<StagedBatchState>>,
        hierarchy: WeightHierarchy,
    ) -> Self {
        Self {
            handle: Some(handle),
            cancellation,
            state,
            hierarchy,
        }
    }

    /// Returns newly ready atomic groups sorted by canonical index.
    ///
    /// Each group is returned by this method at most once. The final `wait`
    /// result still contains every group in canonical order.
    pub fn ready_groups(&mut self) -> Vec<StagedWeightGroup> {
        lock(&self.state).take_ready_groups()
    }

    /// Waits for all missing weights and restores the original group order.
    pub async fn wait(mut self) -> Result<StagedWeightBatchCompletion> {
        let handle = self.handle.take().ok_or_else(|| {
            PowerError::InferenceFailed("staged weight batch has no join handle".to_string())
        })?;
        let wait_started = Instant::now();
        let mut completion = handle.await.map_err(|error| {
            PowerError::InferenceFailed(format!("staged weight batch task failed: {error}"))
        })??;
        if self.hierarchy.inner.policy.telemetry != TelemetryMode::Disabled {
            completion.report.foreground_wait_nanos = elapsed_nanos(wait_started);
        }
        self.hierarchy
            .inner
            .telemetry
            .staged_batch(&completion.report);
        Ok(completion)
    }

    /// Cancels pending loads and releases shared background admission.
    pub fn abort(&self) {
        self.cancellation.cancel();
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }
}

impl Drop for StagedWeightBatch {
    fn drop(&mut self) {
        self.cancellation.cancel();
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }
}

impl std::fmt::Debug for StagedWeightBatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = lock(&self.state);
        formatter
            .debug_struct("StagedWeightBatch")
            .field("groups", &state.len())
            .field("complete", &state.is_complete())
            .finish_non_exhaustive()
    }
}

pub(super) struct StagedBatchState {
    groups: Vec<StagedGroupState>,
}

struct StagedGroupState {
    weights: Vec<Option<ResidentWeight>>,
    delivered: bool,
}

impl StagedBatchState {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            groups: Vec::with_capacity(capacity),
        }
    }

    pub(super) fn push(&mut self, weights: Vec<Option<ResidentWeight>>) {
        self.groups.push(StagedGroupState {
            weights,
            delivered: false,
        });
    }

    pub(super) fn len(&self) -> usize {
        self.groups.len()
    }

    pub(super) fn ready_count(&self) -> usize {
        self.groups
            .iter()
            .filter(|group| group.weights.iter().all(Option::is_some))
            .count()
    }

    fn take_ready_groups(&mut self) -> Vec<StagedWeightGroup> {
        let mut ready = Vec::new();
        for (canonical_index, group) in self.groups.iter_mut().enumerate() {
            if group.delivered || group.weights.iter().any(Option::is_none) {
                continue;
            }
            let Some(weights) = group.weights.iter().cloned().collect::<Option<Vec<_>>>() else {
                continue;
            };
            group.delivered = true;
            ready.push(StagedWeightGroup {
                canonical_index,
                weights,
            });
        }
        ready
    }

    pub(super) fn insert(
        &mut self,
        group_index: usize,
        weight_index: usize,
        weight: ResidentWeight,
    ) -> Result<()> {
        let group = self.groups.get_mut(group_index).ok_or_else(|| {
            PowerError::InferenceFailed(
                "staged weight worker returned an unknown group index".to_string(),
            )
        })?;
        let slot = group.weights.get_mut(weight_index).ok_or_else(|| {
            PowerError::InferenceFailed(
                "staged weight worker returned an unknown weight index".to_string(),
            )
        })?;
        if slot.is_some() {
            return Err(PowerError::InferenceFailed(
                "staged weight worker attempted to fill a completed slot".to_string(),
            ));
        }
        *slot = Some(weight);
        Ok(())
    }

    pub(super) fn completed_groups(&self) -> Result<Vec<StagedWeightGroup>> {
        self.groups
            .iter()
            .enumerate()
            .map(|(canonical_index, group)| {
                let weights = group
                    .weights
                    .iter()
                    .cloned()
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        PowerError::InferenceFailed(
                            "staged weight batch completed with an unfilled slot".to_string(),
                        )
                    })?;
                Ok(StagedWeightGroup {
                    canonical_index,
                    weights,
                })
            })
            .collect()
    }

    fn is_complete(&self) -> bool {
        self.groups
            .iter()
            .all(|group| group.weights.iter().all(Option::is_some))
    }
}

pub(super) fn elapsed_nanos(started: Instant) -> u64 {
    duration_nanos(started.elapsed())
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
