use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use crate::error::{PowerError, Result};

pub use super::coupling_types::{
    RouteCouplingEntry, RouteCouplingHistory, RouteCouplingPolicy, RouteHintEvaluation,
    RouteHintTelemetry, RouteLayerGeometry, RoutePrefetchHint, RoutePrefetchHints,
};
use super::{ExpertKey, RoutedExpertBatch, TelemetryMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CouplingKey {
    source: ExpertKey,
    target: ExpertKey,
}

#[derive(Default)]
struct CouplingState {
    layers: BTreeMap<u32, u32>,
    entries: BTreeMap<CouplingKey, u64>,
    evaluations: u64,
    predicted_selections: u64,
    actual_selections: u64,
    matched_selections: u64,
}

pub(super) struct RouteCouplingTracker {
    mode: TelemetryMode,
    weights_sha256: String,
    policy: RouteCouplingPolicy,
    state: Mutex<CouplingState>,
}

impl RouteCouplingTracker {
    pub(super) fn new(
        mode: TelemetryMode,
        weights_sha256: impl Into<String>,
        policy: RouteCouplingPolicy,
    ) -> Self {
        Self {
            mode,
            weights_sha256: weights_sha256.into(),
            policy,
            state: Mutex::new(CouplingState::default()),
        }
    }

    pub(super) fn record_transition(
        &self,
        source: &RoutedExpertBatch,
        target: &RoutedExpertBatch,
    ) -> Result<()> {
        self.require_detailed()?;
        self.validate_transition(source, target)?;

        let mut updates = BTreeMap::<CouplingKey, u64>::new();
        let mut observations = 0_usize;
        for (source_routes, target_routes) in
            source.selections().iter().zip(target.selections().iter())
        {
            let position_observations = source_routes
                .len()
                .checked_mul(target_routes.len())
                .ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "route coupling observation count overflowed".to_string(),
                    )
                })?;
            observations = observations
                .checked_add(position_observations)
                .ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "route coupling observation count overflowed".to_string(),
                    )
                })?;
            if observations > self.policy.max_entries {
                return Err(PowerError::InvalidRequest(format!(
                    "route transition contains {observations} expert pairs, exceeding the {} observation bound",
                    self.policy.max_entries
                )));
            }
            for source_route in source_routes {
                for target_route in target_routes {
                    let key = CouplingKey {
                        source: ExpertKey {
                            layer: source.layer(),
                            expert: source_route.expert,
                        },
                        target: ExpertKey {
                            layer: target.layer(),
                            expert: target_route.expert,
                        },
                    };
                    let count = updates.entry(key).or_default();
                    *count = count.checked_add(1).ok_or_else(|| {
                        PowerError::InvalidRequest(
                            "route coupling observation count overflowed".to_string(),
                        )
                    })?;
                }
            }
        }

        let mut state = lock(&self.state);
        validate_geometry(&state.layers, source.layer(), source.expert_count())?;
        validate_geometry(&state.layers, target.layer(), target.expert_count())?;
        let new_entries = updates
            .keys()
            .filter(|key| !state.entries.contains_key(key))
            .count();
        if state.entries.len().saturating_add(new_entries) > self.policy.max_entries {
            return Err(PowerError::InvalidRequest(format!(
                "route coupling table would exceed the {} entry bound",
                self.policy.max_entries
            )));
        }
        for (key, increment) in &updates {
            state
                .entries
                .get(key)
                .copied()
                .unwrap_or_default()
                .checked_add(*increment)
                .ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "route coupling observation count overflowed".to_string(),
                    )
                })?;
        }

        state.layers.insert(source.layer(), source.expert_count());
        state.layers.insert(target.layer(), target.expert_count());
        for (key, increment) in updates {
            let count = state.entries.entry(key).or_default();
            *count += increment;
        }
        Ok(())
    }

    pub(super) fn hints(
        &self,
        source: &RoutedExpertBatch,
        target_layer: u32,
        hints_per_position: usize,
    ) -> Result<RoutePrefetchHints> {
        self.require_detailed()?;
        self.validate_batch_positions(source)?;
        self.validate_distance(source.layer(), target_layer)?;
        if hints_per_position == 0 || hints_per_position > self.policy.max_hints_per_position {
            return Err(PowerError::InvalidRequest(format!(
                "route coupling requested {hints_per_position} hints per position, outside the 1..={} bound",
                self.policy.max_hints_per_position
            )));
        }

        let state = lock(&self.state);
        validate_geometry(&state.layers, source.layer(), source.expert_count())?;
        let mut selections = Vec::with_capacity(source.selections().len());
        let mut union = BTreeSet::new();
        for source_routes in source.selections() {
            let mut scores = BTreeMap::<u32, u64>::new();
            for source_route in source_routes {
                let source_key = ExpertKey {
                    layer: source.layer(),
                    expert: source_route.expert,
                };
                let start = CouplingKey {
                    source: source_key,
                    target: ExpertKey {
                        layer: target_layer,
                        expert: 0,
                    },
                };
                let end = CouplingKey {
                    source: source_key,
                    target: ExpertKey {
                        layer: target_layer,
                        expert: u32::MAX,
                    },
                };
                for (key, observations) in state.entries.range(start..=end) {
                    let score = scores.entry(key.target.expert).or_default();
                    *score = score.saturating_add(*observations);
                }
            }
            let mut ranked = scores.into_iter().collect::<Vec<_>>();
            ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
            let predicted = ranked
                .into_iter()
                .take(hints_per_position)
                .map(|(expert, score)| {
                    union.insert(expert);
                    RoutePrefetchHint { expert, score }
                })
                .collect();
            selections.push(predicted);
        }

        Ok(RoutePrefetchHints {
            weights_sha256: self.weights_sha256.clone(),
            source_layer: source.layer(),
            target_layer,
            selections,
            union: union.into_iter().collect(),
        })
    }

    pub(super) fn evaluate(
        &self,
        hints: &RoutePrefetchHints,
        actual: &RoutedExpertBatch,
    ) -> Result<RouteHintEvaluation> {
        self.require_detailed()?;
        if hints.weights_sha256 != self.weights_sha256 {
            return Err(PowerError::InvalidFormat(
                "route prefetch hints do not match this weight store".to_string(),
            ));
        }
        if hints.target_layer != actual.layer()
            || hints.selections.len() != actual.selections().len()
        {
            return Err(PowerError::InvalidRequest(
                "route prefetch hints do not match the actual target batch".to_string(),
            ));
        }
        self.validate_batch_positions(actual)?;

        let mut predicted_selections = 0_u64;
        let mut actual_selections = 0_u64;
        let mut matched_selections = 0_u64;
        for (predicted, routed) in hints.selections.iter().zip(actual.selections()) {
            predicted_selections = predicted_selections.saturating_add(predicted.len() as u64);
            actual_selections = actual_selections.saturating_add(routed.len() as u64);
            let predicted = predicted
                .iter()
                .map(|hint| hint.expert)
                .collect::<BTreeSet<_>>();
            matched_selections = matched_selections.saturating_add(
                routed
                    .iter()
                    .filter(|selection| predicted.contains(&selection.expert))
                    .count() as u64,
            );
        }
        let evaluation = RouteHintEvaluation {
            positions: actual.selections().len(),
            predicted_selections,
            actual_selections,
            matched_selections,
        };
        let mut state = lock(&self.state);
        validate_geometry(&state.layers, actual.layer(), actual.expert_count())?;
        state.evaluations = state.evaluations.saturating_add(1);
        state.predicted_selections = state
            .predicted_selections
            .saturating_add(predicted_selections);
        state.actual_selections = state.actual_selections.saturating_add(actual_selections);
        state.matched_selections = state.matched_selections.saturating_add(matched_selections);
        Ok(evaluation)
    }

    pub(super) fn telemetry(&self) -> Result<RouteHintTelemetry> {
        self.require_detailed()?;
        let state = lock(&self.state);
        Ok(RouteHintTelemetry {
            evaluations: state.evaluations,
            predicted_selections: state.predicted_selections,
            actual_selections: state.actual_selections,
            matched_selections: state.matched_selections,
        })
    }

    pub(super) fn history(&self) -> Result<RouteCouplingHistory> {
        self.require_detailed()?;
        let state = lock(&self.state);
        Ok(RouteCouplingHistory {
            schema: RouteCouplingHistory::SCHEMA.to_string(),
            weights_sha256: self.weights_sha256.clone(),
            layers: state
                .layers
                .iter()
                .map(|(layer, expert_count)| RouteLayerGeometry {
                    layer: *layer,
                    expert_count: *expert_count,
                })
                .collect(),
            entries: state
                .entries
                .iter()
                .map(|(key, observations)| RouteCouplingEntry {
                    source: key.source,
                    target: key.target,
                    observations: *observations,
                })
                .collect(),
        })
    }

    pub(super) fn restore(&self, history: &RouteCouplingHistory) -> Result<()> {
        self.require_detailed()?;
        if history.schema != RouteCouplingHistory::SCHEMA
            || history.weights_sha256 != self.weights_sha256
        {
            return Err(PowerError::InvalidFormat(
                "route coupling history schema or model digest does not match this weight store"
                    .to_string(),
            ));
        }
        if history.entries.len() > self.policy.max_entries
            || history.layers.len() > self.policy.max_entries.saturating_mul(2)
        {
            return Err(PowerError::InvalidFormat(
                "route coupling history exceeds the configured bounds".to_string(),
            ));
        }

        let mut restored_layers = BTreeMap::new();
        for layer in &history.layers {
            if layer.expert_count == 0
                || restored_layers
                    .insert(layer.layer, layer.expert_count)
                    .is_some()
            {
                return Err(PowerError::InvalidFormat(
                    "route coupling history contains invalid or duplicate layer geometry"
                        .to_string(),
                ));
            }
        }
        let mut restored_entries = BTreeMap::new();
        let mut referenced_layers = BTreeSet::new();
        for entry in &history.entries {
            self.validate_distance(entry.source.layer, entry.target.layer)
                .map_err(|error| PowerError::InvalidFormat(error.to_string()))?;
            let Some(source_experts) = restored_layers.get(&entry.source.layer) else {
                return Err(PowerError::InvalidFormat(
                    "route coupling history is missing source layer geometry".to_string(),
                ));
            };
            let Some(target_experts) = restored_layers.get(&entry.target.layer) else {
                return Err(PowerError::InvalidFormat(
                    "route coupling history is missing target layer geometry".to_string(),
                ));
            };
            if entry.observations == 0
                || entry.source.expert >= *source_experts
                || entry.target.expert >= *target_experts
            {
                return Err(PowerError::InvalidFormat(
                    "route coupling history contains an invalid expert entry".to_string(),
                ));
            }
            let key = CouplingKey {
                source: entry.source,
                target: entry.target,
            };
            if restored_entries.insert(key, entry.observations).is_some() {
                return Err(PowerError::InvalidFormat(
                    "route coupling history contains a duplicate expert entry".to_string(),
                ));
            }
            referenced_layers.insert(entry.source.layer);
            referenced_layers.insert(entry.target.layer);
        }
        if restored_layers
            .keys()
            .any(|layer| !referenced_layers.contains(layer))
        {
            return Err(PowerError::InvalidFormat(
                "route coupling history contains unreferenced layer geometry".to_string(),
            ));
        }

        let mut state = lock(&self.state);
        for (layer, expert_count) in &restored_layers {
            validate_geometry(&state.layers, *layer, *expert_count)?;
        }
        let new_entries = restored_entries
            .keys()
            .filter(|key| !state.entries.contains_key(key))
            .count();
        if state.entries.len().saturating_add(new_entries) > self.policy.max_entries {
            return Err(PowerError::InvalidFormat(
                "restored route coupling table would exceed the configured entry bound".to_string(),
            ));
        }
        for (key, observations) in &restored_entries {
            state
                .entries
                .get(key)
                .copied()
                .unwrap_or_default()
                .checked_add(*observations)
                .ok_or_else(|| {
                    PowerError::InvalidFormat(
                        "route coupling history observation count overflowed".to_string(),
                    )
                })?;
        }

        state.layers.extend(restored_layers);
        for (key, observations) in restored_entries {
            let count = state.entries.entry(key).or_default();
            *count += observations;
        }
        Ok(())
    }

    fn require_detailed(&self) -> Result<()> {
        if self.mode != TelemetryMode::Detailed {
            return Err(PowerError::PolicyViolation(
                "route coupling requires explicitly enabled detailed telemetry".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_transition(
        &self,
        source: &RoutedExpertBatch,
        target: &RoutedExpertBatch,
    ) -> Result<()> {
        self.validate_batch_positions(source)?;
        self.validate_batch_positions(target)?;
        self.validate_distance(source.layer(), target.layer())?;
        if source.selections().len() != target.selections().len() {
            return Err(PowerError::InvalidRequest(
                "route coupling batches must contain the same positions".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_batch_positions(&self, batch: &RoutedExpertBatch) -> Result<()> {
        if batch.selections().len() > self.policy.max_positions_per_batch {
            return Err(PowerError::InvalidRequest(format!(
                "route batch contains {} positions, exceeding the {} coupling bound",
                batch.selections().len(),
                self.policy.max_positions_per_batch
            )));
        }
        Ok(())
    }

    fn validate_distance(&self, source_layer: u32, target_layer: u32) -> Result<()> {
        let Some(distance) = target_layer.checked_sub(source_layer) else {
            return Err(PowerError::InvalidRequest(
                "route coupling target layer must follow the source layer".to_string(),
            ));
        };
        if distance == 0 || distance > self.policy.max_lookahead_layers {
            return Err(PowerError::InvalidRequest(format!(
                "route coupling lookahead {distance} is outside the 1..={} bound",
                self.policy.max_lookahead_layers
            )));
        }
        Ok(())
    }
}

fn validate_geometry(layers: &BTreeMap<u32, u32>, layer: u32, expert_count: u32) -> Result<()> {
    if layers
        .get(&layer)
        .is_some_and(|existing| *existing != expert_count)
    {
        return Err(PowerError::InvalidFormat(format!(
            "route layer {layer} expert geometry does not match the learned coupling table"
        )));
    }
    Ok(())
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
