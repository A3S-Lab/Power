use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

use super::ExecutionDigest;

pub(super) const MAX_ROUNDS_PER_CANDIDATE: usize = 64;

/// Safety gates for model-owned, lossless execution-knob calibration.
///
/// A round always contains both baseline-then-candidate and
/// candidate-then-baseline executions. Requiring at least two rounds prevents
/// a single warm-up or drift event from authorizing a configuration change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TuningProfilePolicy {
    pub minimum_rounds: usize,
    pub minimum_samples_per_run: u64,
    pub minimum_throughput_gain_bps: u32,
    pub maximum_cache_hit_rate_delta_bps: u32,
    pub maximum_p99_latency_regression_bps: u32,
}

impl Default for TuningProfilePolicy {
    fn default() -> Self {
        Self {
            minimum_rounds: 2,
            minimum_samples_per_run: 32,
            minimum_throughput_gain_bps: 300,
            maximum_cache_hit_rate_delta_bps: 50,
            maximum_p99_latency_regression_bps: 2_000,
        }
    }
}

impl TuningProfilePolicy {
    pub fn validate(&self) -> Result<()> {
        if !(2..=MAX_ROUNDS_PER_CANDIDATE).contains(&self.minimum_rounds) {
            return Err(PowerError::Config(format!(
                "tuning profile minimum rounds must be between 2 and {MAX_ROUNDS_PER_CANDIDATE}"
            )));
        }
        if self.minimum_samples_per_run == 0 {
            return Err(PowerError::Config(
                "tuning profile minimum samples must be greater than zero".to_string(),
            ));
        }
        if self.minimum_throughput_gain_bps > 10_000
            || self.maximum_cache_hit_rate_delta_bps > 10_000
            || self.maximum_p99_latency_regression_bps > 10_000
        {
            return Err(PowerError::Config(
                "tuning profile percentage gates must not exceed 10,000 basis points".to_string(),
            ));
        }
        Ok(())
    }
}

/// Opaque identities shared by every run in one calibration submission.
///
/// Model crates define the canonical workload, graph, device, and environment
/// representations. Power accepts only their SHA-256 identities, never their
/// contents, paths, topology, or model-specific configuration values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TuningProfileBinding {
    pub weights_sha256: String,
    pub graph_source_sha256: String,
    pub calibration_workload: ExecutionDigest,
    pub runtime_sha256: String,
    pub device_sha256: String,
    pub environment_sha256: String,
}

/// Aggregate measurements from one model-owned teacher-forced execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TuningRunEvidence {
    pub binding: TuningProfileBinding,
    pub configuration_sha256: String,
    pub output: ExecutionDigest,
    pub sample_count: u64,
    pub completed_units: u64,
    pub elapsed_nanos: u64,
    pub cache_hits: u64,
    pub cache_requests: u64,
    pub p99_latency_nanos: u64,
}

/// Two adjacent runs in their actual execution order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TuningOrderedEvidence {
    pub first: TuningRunEvidence,
    pub second: TuningRunEvidence,
}

/// One complete drift-resistant AB/BA comparison round.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TuningRoundEvidence {
    pub round: u32,
    pub baseline_then_candidate: TuningOrderedEvidence,
    pub candidate_then_baseline: TuningOrderedEvidence,
}

/// Repeated evidence for one opaque candidate configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TuningCandidateEvidence {
    pub configuration_sha256: String,
    pub rounds: Vec<TuningRoundEvidence>,
}

/// Untrusted aggregate evidence submitted by a model crate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TuningProfileEvidence {
    pub schema: String,
    pub binding: TuningProfileBinding,
    pub baseline_configuration_sha256: String,
    pub candidates: Vec<TuningCandidateEvidence>,
}

impl TuningProfileEvidence {
    pub const SCHEMA: &'static str = "a3s.power.lossless-tuning-evidence.v1";
}

/// Privacy-safe aggregate result for one candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TuningCandidateSummary {
    pub configuration_sha256: String,
    pub round_count: usize,
    pub sample_count: u64,
    pub baseline_then_candidate_median_throughput_gain_bps: i64,
    pub candidate_then_baseline_median_throughput_gain_bps: i64,
    pub conservative_throughput_gain_bps: i64,
    pub maximum_cache_hit_rate_delta_bps: u32,
    pub maximum_p99_latency_regression_bps: i64,
    pub eligible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TuningProfileOutcome {
    CandidateSelected,
    BaselineRetainedNoEligibleCandidate,
    BaselineRetainedTie,
}

/// Ephemeral decision that a model crate may place in an authorized sealed
/// store. Power never applies or persists the referenced configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TuningProfileDecision {
    pub schema: String,
    pub binding: TuningProfileBinding,
    pub policy: TuningProfilePolicy,
    pub baseline_configuration_sha256: String,
    pub selected_configuration_sha256: String,
    pub output: ExecutionDigest,
    pub outcome: TuningProfileOutcome,
    pub candidates: Vec<TuningCandidateSummary>,
}

impl TuningProfileDecision {
    pub const SCHEMA: &'static str = "a3s.power.lossless-tuning-profile.v1";
}
