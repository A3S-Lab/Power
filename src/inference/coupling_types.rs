use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

use super::ExpertKey;

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

/// Serializable coupling history for a caller-owned [`super::SealedStateEnvelope`].
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
    pub(super) weights_sha256: String,
    pub(super) source_layer: u32,
    pub(super) target_layer: u32,
    pub(super) selections: Vec<Vec<RoutePrefetchHint>>,
    pub(super) union: Vec<u32>,
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
