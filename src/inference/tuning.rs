use std::cmp::Ordering;
use std::collections::BTreeSet;

use crate::error::{PowerError, Result};

use super::tuning_types::{
    TuningCandidateEvidence, TuningCandidateSummary, TuningOrderedEvidence, TuningProfileBinding,
    TuningProfileDecision, TuningProfileEvidence, TuningProfileOutcome, TuningProfilePolicy,
    TuningRunEvidence, MAX_ROUNDS_PER_CANDIDATE,
};
use super::ExecutionDigest;

pub(super) const MAX_CANDIDATES: usize = 32;
const BASIS_POINTS: u128 = 10_000;

#[derive(Clone, Copy)]
struct Ratio {
    numerator: u64,
    denominator: u64,
}

struct EvaluatedCandidate {
    summary: TuningCandidateSummary,
    score: Ratio,
}

struct OrderedMetrics {
    throughput: Ratio,
    sample_count: u64,
    cache_delta_bps: u32,
    cache_within_policy: bool,
    p99_regression_bps: i64,
    p99_within_policy: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct MeasurementShape {
    sample_count: u64,
    completed_units: u64,
    cache_requests: u64,
}

#[derive(Default)]
struct ValidationState {
    output: Option<ExecutionDigest>,
    shape: Option<MeasurementShape>,
}

/// Evaluates aggregate calibration evidence without running a model, applying
/// a configuration, changing inference semantics, or writing persistent state.
pub fn evaluate_tuning_profile(
    evidence: &TuningProfileEvidence,
    policy: &TuningProfilePolicy,
) -> Result<TuningProfileDecision> {
    policy.validate()?;
    validate_evidence_header(evidence)?;

    let mut candidate_digests = BTreeSet::new();
    let mut validation = ValidationState::default();
    let mut evaluated = Vec::with_capacity(evidence.candidates.len());
    for candidate in &evidence.candidates {
        validate_sha256(
            &candidate.configuration_sha256,
            "candidate configuration SHA-256",
        )?;
        if candidate.configuration_sha256 == evidence.baseline_configuration_sha256 {
            return Err(PowerError::InvalidFormat(
                "tuning candidate duplicates the baseline configuration".to_string(),
            ));
        }
        if !candidate_digests.insert(candidate.configuration_sha256.as_str()) {
            return Err(PowerError::InvalidFormat(
                "tuning evidence contains a duplicate candidate configuration".to_string(),
            ));
        }
        evaluated.push(evaluate_candidate(
            evidence,
            candidate,
            policy,
            &mut validation,
        )?);
    }
    evaluated.sort_by(|left, right| {
        left.summary
            .configuration_sha256
            .cmp(&right.summary.configuration_sha256)
    });

    let eligible = evaluated
        .iter()
        .filter(|candidate| candidate.summary.eligible)
        .collect::<Vec<_>>();
    let (outcome, selected_configuration_sha256) = if eligible.is_empty() {
        (
            TuningProfileOutcome::BaselineRetainedNoEligibleCandidate,
            evidence.baseline_configuration_sha256.clone(),
        )
    } else {
        let best_score = eligible
            .iter()
            .map(|candidate| candidate.score)
            .max_by(|left, right| compare_ratio(*left, *right))
            .ok_or_else(|| {
                PowerError::InvalidFormat(
                    "tuning evidence lost its eligible candidate during evaluation".to_string(),
                )
            })?;
        let winners = eligible
            .iter()
            .filter(|candidate| compare_ratio(candidate.score, best_score) == Ordering::Equal)
            .collect::<Vec<_>>();
        if winners.len() == 1 {
            (
                TuningProfileOutcome::CandidateSelected,
                winners[0].summary.configuration_sha256.clone(),
            )
        } else {
            (
                TuningProfileOutcome::BaselineRetainedTie,
                evidence.baseline_configuration_sha256.clone(),
            )
        }
    };

    Ok(TuningProfileDecision {
        schema: TuningProfileDecision::SCHEMA.to_string(),
        binding: evidence.binding.clone(),
        policy: policy.clone(),
        baseline_configuration_sha256: evidence.baseline_configuration_sha256.clone(),
        selected_configuration_sha256,
        output: validation.output.ok_or_else(|| {
            PowerError::InvalidFormat("tuning evidence contains no output digest".to_string())
        })?,
        outcome,
        candidates: evaluated
            .into_iter()
            .map(|candidate| candidate.summary)
            .collect(),
    })
}

fn validate_evidence_header(evidence: &TuningProfileEvidence) -> Result<()> {
    if evidence.schema != TuningProfileEvidence::SCHEMA {
        return Err(PowerError::InvalidFormat(
            "tuning evidence schema is unsupported".to_string(),
        ));
    }
    validate_binding(&evidence.binding)?;
    validate_sha256(
        &evidence.baseline_configuration_sha256,
        "baseline configuration SHA-256",
    )?;
    if evidence.candidates.is_empty() || evidence.candidates.len() > MAX_CANDIDATES {
        return Err(PowerError::InvalidFormat(format!(
            "tuning evidence must contain between 1 and {MAX_CANDIDATES} candidates"
        )));
    }
    Ok(())
}

fn evaluate_candidate(
    evidence: &TuningProfileEvidence,
    candidate: &TuningCandidateEvidence,
    policy: &TuningProfilePolicy,
    validation: &mut ValidationState,
) -> Result<EvaluatedCandidate> {
    if candidate.rounds.len() < policy.minimum_rounds
        || candidate.rounds.len() > MAX_ROUNDS_PER_CANDIDATE
    {
        return Err(PowerError::InvalidFormat(format!(
            "tuning candidate must contain between {} and {MAX_ROUNDS_PER_CANDIDATE} complete AB/BA rounds",
            policy.minimum_rounds
        )));
    }

    let mut round_ids = BTreeSet::new();
    let mut forward = Vec::with_capacity(candidate.rounds.len());
    let mut reverse = Vec::with_capacity(candidate.rounds.len());
    let mut sample_count = 0_u64;
    let mut maximum_cache_delta = 0_u32;
    let mut maximum_p99_regression = i64::MIN;
    let mut parity_gates_pass = true;

    for round in &candidate.rounds {
        if !round_ids.insert(round.round) {
            return Err(PowerError::InvalidFormat(
                "tuning candidate contains a duplicate round identifier".to_string(),
            ));
        }
        let forward_metrics = evaluate_ordered(
            &round.baseline_then_candidate,
            &evidence.baseline_configuration_sha256,
            &candidate.configuration_sha256,
            true,
            &evidence.binding,
            policy,
            validation,
        )?;
        let reverse_metrics = evaluate_ordered(
            &round.candidate_then_baseline,
            &candidate.configuration_sha256,
            &evidence.baseline_configuration_sha256,
            false,
            &evidence.binding,
            policy,
            validation,
        )?;
        for metrics in [&forward_metrics, &reverse_metrics] {
            sample_count = sample_count
                .checked_add(metrics.sample_count)
                .ok_or_else(|| {
                    PowerError::InvalidFormat(
                        "tuning evidence aggregate sample count overflowed".to_string(),
                    )
                })?;
            maximum_cache_delta = maximum_cache_delta.max(metrics.cache_delta_bps);
            maximum_p99_regression = maximum_p99_regression.max(metrics.p99_regression_bps);
            parity_gates_pass &= metrics.cache_within_policy && metrics.p99_within_policy;
        }
        forward.push(forward_metrics.throughput);
        reverse.push(reverse_metrics.throughput);
    }

    let forward_median = median_ratio(&mut forward)?;
    let reverse_median = median_ratio(&mut reverse)?;
    let score = if compare_ratio(forward_median, reverse_median) == Ordering::Less {
        forward_median
    } else {
        reverse_median
    };
    let gain_gate = ratio_meets_gain(forward_median, policy.minimum_throughput_gain_bps)
        && ratio_meets_gain(reverse_median, policy.minimum_throughput_gain_bps);

    Ok(EvaluatedCandidate {
        summary: TuningCandidateSummary {
            configuration_sha256: candidate.configuration_sha256.clone(),
            round_count: candidate.rounds.len(),
            sample_count,
            baseline_then_candidate_median_throughput_gain_bps: ratio_gain_bps(forward_median)?,
            candidate_then_baseline_median_throughput_gain_bps: ratio_gain_bps(reverse_median)?,
            conservative_throughput_gain_bps: ratio_gain_bps(score)?,
            maximum_cache_hit_rate_delta_bps: maximum_cache_delta,
            maximum_p99_latency_regression_bps: maximum_p99_regression,
            eligible: gain_gate && parity_gates_pass,
        },
        score,
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_ordered(
    ordered: &TuningOrderedEvidence,
    expected_first: &str,
    expected_second: &str,
    baseline_is_first: bool,
    binding: &TuningProfileBinding,
    policy: &TuningProfilePolicy,
    validation: &mut ValidationState,
) -> Result<OrderedMetrics> {
    validate_run(&ordered.first, expected_first, binding, policy, validation)?;
    validate_run(
        &ordered.second,
        expected_second,
        binding,
        policy,
        validation,
    )?;
    let (baseline, candidate) = if baseline_is_first {
        (&ordered.first, &ordered.second)
    } else {
        (&ordered.second, &ordered.first)
    };
    if baseline.completed_units != candidate.completed_units
        || baseline.sample_count != candidate.sample_count
        || baseline.cache_requests != candidate.cache_requests
    {
        return Err(PowerError::InvalidFormat(
            "tuning comparison runs do not describe the same aggregate workload".to_string(),
        ));
    }

    let cache_delta_bps = cache_delta_bps(baseline, candidate)?;
    let p99_regression_bps = p99_regression_bps(baseline, candidate)?;
    let sample_count = baseline
        .sample_count
        .checked_add(candidate.sample_count)
        .ok_or_else(|| {
            PowerError::InvalidFormat("tuning evidence ordered sample count overflowed".to_string())
        })?;

    Ok(OrderedMetrics {
        throughput: Ratio {
            numerator: baseline.elapsed_nanos,
            denominator: candidate.elapsed_nanos,
        },
        sample_count,
        cache_delta_bps,
        cache_within_policy: cache_delta_within_policy(
            baseline,
            candidate,
            policy.maximum_cache_hit_rate_delta_bps,
        )?,
        p99_regression_bps,
        p99_within_policy: p99_within_policy(
            baseline,
            candidate,
            policy.maximum_p99_latency_regression_bps,
        ),
    })
}

fn validate_run(
    run: &TuningRunEvidence,
    expected_configuration_sha256: &str,
    binding: &TuningProfileBinding,
    policy: &TuningProfilePolicy,
    validation: &mut ValidationState,
) -> Result<()> {
    validate_binding(&run.binding)?;
    validate_sha256(&run.configuration_sha256, "run configuration SHA-256")?;
    validate_execution_digest(&run.output, "tuning output digest")?;
    if run.binding != *binding {
        return Err(PowerError::InvalidFormat(
            "tuning run binding does not match the submitted profile binding".to_string(),
        ));
    }
    if run.configuration_sha256 != expected_configuration_sha256 {
        return Err(PowerError::InvalidFormat(
            "tuning run configuration does not match its declared execution order".to_string(),
        ));
    }
    if run.sample_count < policy.minimum_samples_per_run
        || run.completed_units == 0
        || run.elapsed_nanos == 0
        || run.p99_latency_nanos == 0
        || run.cache_hits > run.cache_requests
    {
        return Err(PowerError::InvalidFormat(
            "tuning run violates its aggregate measurement contract".to_string(),
        ));
    }
    let shape = MeasurementShape {
        sample_count: run.sample_count,
        completed_units: run.completed_units,
        cache_requests: run.cache_requests,
    };
    match validation.shape {
        Some(expected) if expected != shape => {
            return Err(PowerError::InvalidFormat(
                "tuning runs do not share one aggregate calibration shape".to_string(),
            ));
        }
        None => validation.shape = Some(shape),
        Some(_) => {}
    }
    match &validation.output {
        Some(output) if output != &run.output => {
            return Err(PowerError::InvalidFormat(
                "tuning output parity failed across submitted runs".to_string(),
            ));
        }
        None => validation.output = Some(run.output.clone()),
        Some(_) => {}
    }
    Ok(())
}

fn validate_binding(binding: &TuningProfileBinding) -> Result<()> {
    for (value, label) in [
        (&binding.weights_sha256, "tuning weights SHA-256"),
        (&binding.graph_source_sha256, "tuning graph/source SHA-256"),
        (&binding.runtime_sha256, "tuning runtime SHA-256"),
        (&binding.device_sha256, "tuning device SHA-256"),
        (&binding.environment_sha256, "tuning environment SHA-256"),
    ] {
        validate_sha256(value, label)?;
    }
    validate_execution_digest(
        &binding.calibration_workload,
        "tuning calibration workload digest",
    )
}

fn validate_execution_digest(digest: &ExecutionDigest, label: &str) -> Result<()> {
    validate_sha256(&digest.sha256, label)?;
    if digest.byte_length == 0 || digest.item_count == 0 {
        return Err(PowerError::InvalidFormat(format!(
            "{label} must describe non-empty aggregate evidence"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PowerError::InvalidFormat(format!(
            "{label} must contain 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn cache_delta_bps(baseline: &TuningRunEvidence, candidate: &TuningRunEvidence) -> Result<u32> {
    if baseline.cache_requests == 0 {
        return Ok(0);
    }
    let numerator = u128::from(baseline.cache_hits.abs_diff(candidate.cache_hits))
        .checked_mul(BASIS_POINTS)
        .ok_or_else(metric_overflow)?;
    let denominator = u128::from(baseline.cache_requests);
    let rounded_up = numerator
        .checked_add(denominator - 1)
        .ok_or_else(metric_overflow)?
        / denominator;
    u32::try_from(rounded_up).map_err(|_| metric_overflow())
}

fn cache_delta_within_policy(
    baseline: &TuningRunEvidence,
    candidate: &TuningRunEvidence,
    tolerance_bps: u32,
) -> Result<bool> {
    if baseline.cache_requests == 0 {
        return Ok(true);
    }
    let left = u128::from(baseline.cache_hits.abs_diff(candidate.cache_hits))
        .checked_mul(BASIS_POINTS)
        .ok_or_else(metric_overflow)?;
    let right = u128::from(tolerance_bps)
        .checked_mul(u128::from(baseline.cache_requests))
        .ok_or_else(metric_overflow)?;
    Ok(left <= right)
}

fn p99_regression_bps(baseline: &TuningRunEvidence, candidate: &TuningRunEvidence) -> Result<i64> {
    if candidate.p99_latency_nanos >= baseline.p99_latency_nanos {
        positive_change_bps(
            candidate.p99_latency_nanos - baseline.p99_latency_nanos,
            baseline.p99_latency_nanos,
            true,
        )
    } else {
        positive_change_bps(
            baseline.p99_latency_nanos - candidate.p99_latency_nanos,
            baseline.p99_latency_nanos,
            false,
        )
        .and_then(|value| value.checked_neg().ok_or_else(metric_overflow))
    }
}

fn p99_within_policy(
    baseline: &TuningRunEvidence,
    candidate: &TuningRunEvidence,
    maximum_regression_bps: u32,
) -> bool {
    u128::from(candidate.p99_latency_nanos) * BASIS_POINTS
        <= u128::from(baseline.p99_latency_nanos)
            * (BASIS_POINTS + u128::from(maximum_regression_bps))
}

fn ratio_meets_gain(ratio: Ratio, minimum_gain_bps: u32) -> bool {
    u128::from(ratio.numerator) * BASIS_POINTS
        >= u128::from(ratio.denominator) * (BASIS_POINTS + u128::from(minimum_gain_bps))
}

fn ratio_gain_bps(ratio: Ratio) -> Result<i64> {
    if ratio.numerator >= ratio.denominator {
        positive_change_bps(
            ratio.numerator - ratio.denominator,
            ratio.denominator,
            false,
        )
    } else {
        positive_change_bps(
            ratio.denominator - ratio.numerator,
            ratio.denominator,
            false,
        )
        .and_then(|value| value.checked_neg().ok_or_else(metric_overflow))
    }
}

fn positive_change_bps(difference: u64, denominator: u64, round_up: bool) -> Result<i64> {
    let numerator = u128::from(difference)
        .checked_mul(BASIS_POINTS)
        .ok_or_else(metric_overflow)?;
    let value = if round_up && numerator != 0 {
        numerator
            .checked_add(u128::from(denominator) - 1)
            .ok_or_else(metric_overflow)?
            / u128::from(denominator)
    } else {
        numerator / u128::from(denominator)
    };
    i64::try_from(value).map_err(|_| metric_overflow())
}

fn median_ratio(values: &mut [Ratio]) -> Result<Ratio> {
    values.sort_unstable_by(|left, right| compare_ratio(*left, *right));
    values
        .get(values.len().saturating_sub(1) / 2)
        .copied()
        .ok_or_else(|| PowerError::InvalidFormat("tuning ratio set is empty".to_string()))
}

fn compare_ratio(left: Ratio, right: Ratio) -> Ordering {
    (u128::from(left.numerator) * u128::from(right.denominator))
        .cmp(&(u128::from(right.numerator) * u128::from(left.denominator)))
}

fn metric_overflow() -> PowerError {
    PowerError::InvalidFormat("tuning evidence metric arithmetic overflowed".to_string())
}
