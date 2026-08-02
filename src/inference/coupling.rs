use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

use super::{ExpertKey, RoutedExpertBatch, TelemetryMode};

/// Hard bounds for learned, value-preserving cross-layer prefetch hints.
///
/// The table is empty until a model explicitly records route transitions. It
/// never changes router output and is available only with detailed telemetry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RouteCouplingPolicy {
    pub max_lookahead_layers: u32,
    pub max_positions_per_batch: usize,
    pub max_entries: usize,
    pub max_hints_per_position: usize,
}

impl Default for RouteCouplingPolicy {
    fn default() -> Self {
        Self {
            max_lookahead_layers: 2,
            max_positions_per_batch: 4_096,
            // Colibri's two-layer, top-16 coupling table for a 75-layer,
            // 256-expert model contains roughly 614K entries. The bound does
            // not allocate memory eagerly and remains caller-configurable for
            // smaller TEE deployments.
            max_entries: 1_048_576,
            max_hints_per_position: 16,
        }
    }
}

impl RouteCouplingPolicy {
    pub fn validate(&self) -> Result<()> {
        if self.max_lookahead_layers == 0
            || self.max_positions_per_batch == 0
            || self.max_entries == 0
            || self.max_hints_per_position == 0
        {
            return Err(PowerError::Config(
                "route coupling bounds must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

/// Admitted expert geometry for one routed layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RouteLayerGeometry {
    pub layer: u32,
    pub expert_count: u32,
}

/// One exact source-to-target expert co-occurrence count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RouteCouplingEntry {
    pub source: ExpertKey,
    pub target: ExpertKey,
    pub observations: u64,
}

/// Serializable coupling history for a caller-owned encrypted or sealed store.
///
/// Power never persists this value automatically. Expert transitions can
/// correlate with input semantics and must remain inside the applicable trust
/// boundary unless an explicit policy authorizes export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RouteCouplingHistory {
    pub schema: String,
    pub weights_sha256: String,
    pub layers: Vec<RouteLayerGeometry>,
    pub entries: Vec<RouteCouplingEntry>,
}

impl RouteCouplingHistory {
    pub const SCHEMA: &'static str = "a3s.power.route-coupling-history.v1";
}

/// One predicted target expert and its raw learned co-occurrence score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutePrefetchHint {
    pub expert: u32,
    pub score: u64,
}

/// Per-position route hints plus their deterministic batch union.
///
/// Hints are scheduling inputs only. They contain no gate weights and cannot
/// be converted into router selections by Power.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutePrefetchHints {
    weights_sha256: String,
    source_layer: u32,
    target_layer: u32,
    selections: Vec<Vec<RoutePrefetchHint>>,
    union: Vec<u32>,
}

impl RoutePrefetchHints {
    pub fn source_layer(&self) -> u32 {
        self.source_layer
    }

    pub fn target_layer(&self) -> u32 {
        self.target_layer
    }

    pub fn selections(&self) -> &[Vec<RoutePrefetchHint>] {
        &self.selections
    }

    pub fn experts(&self) -> &[u32] {
        &self.union
    }
}

/// Exact recall counters for one hint batch compared with actual router output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RouteHintEvaluation {
    pub positions: usize,
    pub predicted_selections: u64,
    pub actual_selections: u64,
    pub matched_selections: u64,
}

impl RouteHintEvaluation {
    pub fn recall(&self) -> f64 {
        if self.actual_selections == 0 {
            0.0
        } else {
            self.matched_selections as f64 / self.actual_selections as f64
        }
    }
}

/// Aggregate prediction evidence without route identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RouteHintTelemetry {
    pub evaluations: u64,
    pub predicted_selections: u64,
    pub actual_selections: u64,
    pub matched_selections: u64,
}

impl RouteHintTelemetry {
    pub fn recall(&self) -> f64 {
        if self.actual_selections == 0 {
            0.0
        } else {
            self.matched_selections as f64 / self.actual_selections as f64
        }
    }
}

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
        if history.layers.len() > self.policy.max_entries
            || history.entries.len() > self.policy.max_entries
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::RoutedExpert;

    fn batch(layer: u32, routes: &[&[u32]], expert_count: u32) -> RoutedExpertBatch {
        RoutedExpertBatch::new(
            layer,
            routes
                .iter()
                .map(|position| {
                    position
                        .iter()
                        .map(|expert| RoutedExpert {
                            expert: *expert,
                            weight: 1.0 / position.len() as f32,
                        })
                        .collect()
                })
                .collect(),
            expert_count,
            routes.iter().map(|routes| routes.len()).max().unwrap(),
        )
        .unwrap()
    }

    fn tracker(mode: TelemetryMode) -> RouteCouplingTracker {
        RouteCouplingTracker::new(mode, "weights-a", RouteCouplingPolicy::default())
    }

    #[test]
    fn learns_per_position_scores_and_preserves_a_deterministic_union() {
        let tracker = tracker(TelemetryMode::Detailed);
        let source = batch(3, &[&[1, 2], &[1]], 8);
        let target = batch(4, &[&[4, 3], &[3]], 8);
        tracker.record_transition(&source, &target).unwrap();

        let hints = tracker.hints(&source, 4, 2).unwrap();
        assert_eq!(hints.source_layer(), 3);
        assert_eq!(hints.target_layer(), 4);
        assert_eq!(hints.experts(), &[3, 4]);
        assert_eq!(
            hints.selections()[0],
            [
                RoutePrefetchHint {
                    expert: 3,
                    score: 3
                },
                RoutePrefetchHint {
                    expert: 4,
                    score: 2
                }
            ]
        );
        assert_eq!(hints.selections()[1][0].expert, 3);

        let evaluation = tracker.evaluate(&hints, &target).unwrap();
        assert_eq!(evaluation.actual_selections, 3);
        assert_eq!(evaluation.matched_selections, 3);
        assert_eq!(evaluation.recall(), 1.0);
        let telemetry = tracker.telemetry().unwrap();
        assert_eq!(telemetry.evaluations, 1);
        assert_eq!(telemetry.recall(), 1.0);
    }

    #[test]
    fn privacy_modes_below_detailed_reject_coupling_data() {
        let source = batch(0, &[&[0]], 2);
        let target = batch(1, &[&[1]], 2);
        for mode in [TelemetryMode::Disabled, TelemetryMode::Aggregate] {
            let tracker = tracker(mode);
            assert!(matches!(
                tracker.record_transition(&source, &target),
                Err(PowerError::PolicyViolation(_))
            ));
            assert!(matches!(
                tracker.hints(&source, 1, 1),
                Err(PowerError::PolicyViolation(_))
            ));
            assert!(matches!(
                tracker.history(),
                Err(PowerError::PolicyViolation(_))
            ));
        }
    }

    #[test]
    fn history_is_digest_bound_and_restore_is_atomic() {
        let source = batch(0, &[&[0]], 2);
        let target = batch(1, &[&[1]], 2);
        let learned = tracker(TelemetryMode::Detailed);
        learned.record_transition(&source, &target).unwrap();
        let history = learned.history().unwrap();

        let other = RouteCouplingTracker::new(
            TelemetryMode::Detailed,
            "weights-b",
            RouteCouplingPolicy::default(),
        );
        assert!(other.restore(&history).is_err());
        assert!(other.history().unwrap().entries.is_empty());

        let restored = tracker(TelemetryMode::Detailed);
        let mut invalid = history.clone();
        invalid.entries.push(invalid.entries[0]);
        assert!(restored.restore(&invalid).is_err());
        assert!(restored.history().unwrap().entries.is_empty());
        restored.restore(&history).unwrap();
        assert_eq!(restored.history().unwrap().entries, history.entries);
    }

    #[test]
    fn policy_bounds_distance_positions_entries_and_hints() {
        let policy = RouteCouplingPolicy {
            max_lookahead_layers: 1,
            max_positions_per_batch: 1,
            max_entries: 1,
            max_hints_per_position: 1,
        };
        let tracker = RouteCouplingTracker::new(TelemetryMode::Detailed, "weights-a", policy);
        let source = batch(0, &[&[0]], 2);
        let target = batch(1, &[&[1]], 2);
        tracker.record_transition(&source, &target).unwrap();
        assert!(tracker.hints(&source, 1, 2).is_err());
        assert!(tracker.hints(&source, 2, 1).is_err());
        assert!(tracker
            .record_transition(&batch(0, &[&[0], &[0]], 2), &batch(1, &[&[1], &[1]], 2),)
            .is_err());
        assert!(tracker
            .record_transition(&batch(0, &[&[0, 1]], 2), &target)
            .is_err());
    }
}
