use serde_json::json;

use super::{
    evaluate_tuning_profile, ExecutionDigest, TuningCandidateEvidence, TuningCandidateSummary,
    TuningOrderedEvidence, TuningProfileBinding, TuningProfileDecision, TuningProfileEvidence,
    TuningProfileOutcome, TuningProfilePolicy, TuningRoundEvidence, TuningRunEvidence,
};
use crate::error::PowerError;

fn digest(value: char) -> String {
    value.to_string().repeat(64)
}

fn binding() -> TuningProfileBinding {
    TuningProfileBinding {
        weights_sha256: digest('1'),
        graph_source_sha256: digest('2'),
        calibration_workload: ExecutionDigest::token_ids(&[1, 2, 3, 4]),
        runtime_sha256: digest('3'),
        device_sha256: digest('4'),
        environment_sha256: digest('5'),
    }
}

fn run(
    binding: &TuningProfileBinding,
    configuration_sha256: &str,
    elapsed_nanos: u64,
    cache_hits: u64,
    p99_latency_nanos: u64,
) -> TuningRunEvidence {
    TuningRunEvidence {
        binding: binding.clone(),
        configuration_sha256: configuration_sha256.to_string(),
        output: ExecutionDigest::token_ids(&[7, 8, 9]),
        sample_count: 64,
        completed_units: 64,
        elapsed_nanos,
        cache_hits,
        cache_requests: 1_000,
        p99_latency_nanos,
    }
}

fn round(
    binding: &TuningProfileBinding,
    round: u32,
    baseline: &str,
    candidate: &str,
    candidate_elapsed_nanos: u64,
) -> TuningRoundEvidence {
    let baseline_run = || run(binding, baseline, 1_000_000, 950, 1_000);
    let candidate_run = || run(binding, candidate, candidate_elapsed_nanos, 949, 1_100);
    TuningRoundEvidence {
        round,
        baseline_then_candidate: TuningOrderedEvidence {
            first: baseline_run(),
            second: candidate_run(),
        },
        candidate_then_baseline: TuningOrderedEvidence {
            first: candidate_run(),
            second: baseline_run(),
        },
    }
}

fn candidate(
    binding: &TuningProfileBinding,
    baseline: &str,
    configuration_sha256: &str,
    candidate_elapsed_nanos: u64,
) -> TuningCandidateEvidence {
    TuningCandidateEvidence {
        configuration_sha256: configuration_sha256.to_string(),
        rounds: vec![
            round(
                binding,
                1,
                baseline,
                configuration_sha256,
                candidate_elapsed_nanos,
            ),
            round(
                binding,
                2,
                baseline,
                configuration_sha256,
                candidate_elapsed_nanos,
            ),
        ],
    }
}

fn evidence(candidates: Vec<TuningCandidateEvidence>) -> TuningProfileEvidence {
    TuningProfileEvidence {
        schema: TuningProfileEvidence::SCHEMA.to_string(),
        binding: binding(),
        baseline_configuration_sha256: digest('a'),
        candidates,
    }
}

fn good_evidence() -> TuningProfileEvidence {
    let binding = binding();
    let baseline = digest('a');
    evidence(vec![candidate(&binding, &baseline, &digest('b'), 900_000)])
}

#[test]
fn selects_a_lossless_candidate_only_after_repeated_order_reversal() {
    let decision =
        evaluate_tuning_profile(&good_evidence(), &TuningProfilePolicy::default()).unwrap();

    assert_eq!(decision.outcome, TuningProfileOutcome::CandidateSelected);
    assert_eq!(decision.selected_configuration_sha256, digest('b'));
    assert_eq!(decision.candidates.len(), 1);
    assert!(decision.candidates[0].eligible);
    assert!(decision.candidates[0].conservative_throughput_gain_bps >= 1_100);
    assert_eq!(decision.candidates[0].maximum_cache_hit_rate_delta_bps, 10);
    assert_eq!(
        decision.candidates[0].maximum_p99_latency_regression_bps,
        1_000
    );
}

#[test]
fn rejects_mixed_digest_or_environment_bindings() {
    let mut input = good_evidence();
    input.candidates[0].rounds[0]
        .candidate_then_baseline
        .first
        .binding
        .environment_sha256 = digest('6');

    let error = evaluate_tuning_profile(&input, &TuningProfilePolicy::default()).unwrap_err();
    assert!(matches!(error, PowerError::InvalidFormat(_)));
    assert!(error.to_string().contains("binding"));
}

#[test]
fn rejects_any_output_parity_failure() {
    let mut input = good_evidence();
    input.candidates[0].rounds[1]
        .baseline_then_candidate
        .second
        .output = ExecutionDigest::token_ids(&[99]);

    let error = evaluate_tuning_profile(&input, &TuningProfilePolicy::default()).unwrap_err();
    assert!(matches!(error, PowerError::InvalidFormat(_)));
    assert!(error.to_string().contains("output parity"));
}

#[test]
fn rejects_insufficient_rounds_samples_and_wrong_execution_order() {
    let mut insufficient_rounds = good_evidence();
    insufficient_rounds.candidates[0].rounds.pop();
    assert!(
        evaluate_tuning_profile(&insufficient_rounds, &TuningProfilePolicy::default()).is_err()
    );

    let mut insufficient_samples = good_evidence();
    insufficient_samples.candidates[0].rounds[0]
        .baseline_then_candidate
        .first
        .sample_count = 31;
    assert!(
        evaluate_tuning_profile(&insufficient_samples, &TuningProfilePolicy::default()).is_err()
    );

    let mut wrong_order = good_evidence();
    wrong_order.candidates[0].rounds[0]
        .candidate_then_baseline
        .first
        .configuration_sha256 = digest('a');
    assert!(evaluate_tuning_profile(&wrong_order, &TuningProfilePolicy::default()).is_err());
}

#[test]
fn retains_baseline_below_the_minimum_throughput_gain() {
    let binding = binding();
    let baseline = digest('a');
    let input = evidence(vec![candidate(&binding, &baseline, &digest('b'), 980_000)]);
    let decision = evaluate_tuning_profile(&input, &TuningProfilePolicy::default()).unwrap();

    assert_eq!(
        decision.outcome,
        TuningProfileOutcome::BaselineRetainedNoEligibleCandidate
    );
    assert_eq!(decision.selected_configuration_sha256, baseline);
    assert!(!decision.candidates[0].eligible);
}

#[test]
fn retains_baseline_when_the_reverse_order_gain_fails() {
    let mut input = good_evidence();
    for round in &mut input.candidates[0].rounds {
        round.candidate_then_baseline.first.elapsed_nanos = 980_000;
    }

    let decision = evaluate_tuning_profile(&input, &TuningProfilePolicy::default()).unwrap();
    assert_eq!(
        decision.outcome,
        TuningProfileOutcome::BaselineRetainedNoEligibleCandidate
    );
    assert!(decision.candidates[0].baseline_then_candidate_median_throughput_gain_bps >= 1_100);
    assert!(decision.candidates[0].candidate_then_baseline_median_throughput_gain_bps < 300);
}

#[test]
fn retains_baseline_on_p99_or_cache_hit_regressions() {
    let mut p99 = good_evidence();
    for round in &mut p99.candidates[0].rounds {
        round.baseline_then_candidate.second.p99_latency_nanos = 1_201;
        round.candidate_then_baseline.first.p99_latency_nanos = 1_201;
    }
    let p99_decision = evaluate_tuning_profile(&p99, &TuningProfilePolicy::default()).unwrap();
    assert!(!p99_decision.candidates[0].eligible);

    let mut cache = good_evidence();
    for round in &mut cache.candidates[0].rounds {
        round.baseline_then_candidate.second.cache_hits = 944;
        round.candidate_then_baseline.first.cache_hits = 944;
    }
    let cache_decision = evaluate_tuning_profile(&cache, &TuningProfilePolicy::default()).unwrap();
    assert!(!cache_decision.candidates[0].eligible);
}

#[test]
fn reports_p99_improvements_as_negative_regressions() {
    let mut input = good_evidence();
    for round in &mut input.candidates[0].rounds {
        round.baseline_then_candidate.second.p99_latency_nanos = 900;
        round.candidate_then_baseline.first.p99_latency_nanos = 900;
    }

    let decision = evaluate_tuning_profile(&input, &TuningProfilePolicy::default()).unwrap();
    assert_eq!(
        decision.candidates[0].maximum_p99_latency_regression_bps,
        -1_000
    );
}

#[test]
fn accepts_threshold_boundaries_inclusively() {
    let binding = binding();
    let baseline = digest('a');
    let mut input = evidence(vec![candidate(
        &binding,
        &baseline,
        &digest('b'),
        1_000_000,
    )]);
    for round in &mut input.candidates[0].rounds {
        for ordered in [
            &mut round.baseline_then_candidate,
            &mut round.candidate_then_baseline,
        ] {
            let (baseline_run, candidate_run) = if ordered.first.configuration_sha256 == baseline {
                (&mut ordered.first, &mut ordered.second)
            } else {
                (&mut ordered.second, &mut ordered.first)
            };
            baseline_run.elapsed_nanos = 1_030_000;
            candidate_run.elapsed_nanos = 1_000_000;
            baseline_run.cache_hits = 950;
            candidate_run.cache_hits = 945;
            baseline_run.p99_latency_nanos = 1_000;
            candidate_run.p99_latency_nanos = 1_200;
        }
    }

    let decision = evaluate_tuning_profile(&input, &TuningProfilePolicy::default()).unwrap();
    assert_eq!(decision.outcome, TuningProfileOutcome::CandidateSelected);
}

#[test]
fn rejects_malformed_and_duplicate_digest_evidence() {
    let mut malformed = good_evidence();
    malformed.binding.weights_sha256 = digest('A');
    assert!(evaluate_tuning_profile(&malformed, &TuningProfilePolicy::default()).is_err());

    let mut duplicate_candidate = good_evidence();
    duplicate_candidate
        .candidates
        .push(duplicate_candidate.candidates[0].clone());
    assert!(
        evaluate_tuning_profile(&duplicate_candidate, &TuningProfilePolicy::default()).is_err()
    );

    let mut duplicate_round = good_evidence();
    let repeated_round = duplicate_round.candidates[0].rounds[0].clone();
    duplicate_round.candidates[0].rounds.push(repeated_round);
    assert!(evaluate_tuning_profile(&duplicate_round, &TuningProfilePolicy::default()).is_err());
}

#[test]
fn rejects_invalid_measurements_and_arithmetic_overflow() {
    let mut invalid_cache = good_evidence();
    invalid_cache.candidates[0].rounds[0]
        .baseline_then_candidate
        .first
        .cache_hits = 1_001;
    assert!(evaluate_tuning_profile(&invalid_cache, &TuningProfilePolicy::default()).is_err());

    let mut overflow = good_evidence();
    overflow.candidates[0].rounds[0]
        .baseline_then_candidate
        .first
        .sample_count = u64::MAX;
    overflow.candidates[0].rounds[0]
        .baseline_then_candidate
        .second
        .sample_count = u64::MAX;
    let error = evaluate_tuning_profile(&overflow, &TuningProfilePolicy::default()).unwrap_err();
    assert!(matches!(error, PowerError::InvalidFormat(_)));
    assert!(error.to_string().contains("overflow"));
}

#[test]
fn rejects_mixed_calibration_measurement_shapes() {
    let mut input = good_evidence();
    input.candidates[0].rounds[1]
        .candidate_then_baseline
        .first
        .completed_units = 63;

    let error = evaluate_tuning_profile(&input, &TuningProfilePolicy::default()).unwrap_err();
    assert!(matches!(error, PowerError::InvalidFormat(_)));
    assert!(error.to_string().contains("calibration shape"));
}

#[test]
fn winner_is_input_order_independent_and_exact_ties_fail_closed() {
    let binding = binding();
    let baseline = digest('a');
    let slower = candidate(&binding, &baseline, &digest('b'), 900_000);
    let faster = candidate(&binding, &baseline, &digest('c'), 850_000);

    let first = evaluate_tuning_profile(
        &evidence(vec![slower.clone(), faster.clone()]),
        &TuningProfilePolicy::default(),
    )
    .unwrap();
    let second = evaluate_tuning_profile(
        &evidence(vec![faster, slower]),
        &TuningProfilePolicy::default(),
    )
    .unwrap();
    assert_eq!(first.selected_configuration_sha256, digest('c'));
    assert_eq!(first, second);

    let tied = evaluate_tuning_profile(
        &evidence(vec![
            candidate(&binding, &baseline, &digest('b'), 850_000),
            candidate(&binding, &baseline, &digest('c'), 850_000),
        ]),
        &TuningProfilePolicy::default(),
    )
    .unwrap();
    assert_eq!(tied.outcome, TuningProfileOutcome::BaselineRetainedTie);
    assert_eq!(tied.selected_configuration_sha256, baseline);
}

#[test]
fn policy_and_evidence_bounds_fail_closed() {
    let policy = TuningProfilePolicy {
        minimum_rounds: 1,
        ..TuningProfilePolicy::default()
    };
    assert!(matches!(
        evaluate_tuning_profile(&good_evidence(), &policy),
        Err(PowerError::Config(_))
    ));

    let binding = binding();
    let baseline = digest('a');
    let candidates = (0..33)
        .map(|index| candidate(&binding, &baseline, &format!("{index:064x}"), 900_000))
        .collect();
    assert!(
        evaluate_tuning_profile(&evidence(candidates), &TuningProfilePolicy::default()).is_err()
    );
}

#[test]
fn profile_json_and_debug_output_are_aggregate_only() {
    let input = good_evidence();
    let decision = evaluate_tuning_profile(&input, &TuningProfilePolicy::default()).unwrap();
    let json = serde_json::to_string(&decision).unwrap();
    let debug = format!("{decision:?}");

    for secret in [
        "/models/private/checkpoint",
        "teacher-forced prompt text",
        "OMP_NUM_THREADS=24",
    ] {
        assert!(!json.contains(secret));
        assert!(!debug.contains(secret));
    }
    assert!(!json.contains("modelPath"));
    assert!(!json.contains("configurationValues"));

    let mut value = serde_json::to_value(input).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("modelPath".to_string(), json!("/models/private/checkpoint"));
    assert!(serde_json::from_value::<TuningProfileEvidence>(value).is_err());
}

#[test]
fn public_tuning_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<TuningProfilePolicy>();
    assert_send_sync::<TuningProfileBinding>();
    assert_send_sync::<TuningRunEvidence>();
    assert_send_sync::<TuningOrderedEvidence>();
    assert_send_sync::<TuningRoundEvidence>();
    assert_send_sync::<TuningCandidateEvidence>();
    assert_send_sync::<TuningProfileEvidence>();
    assert_send_sync::<TuningCandidateSummary>();
    assert_send_sync::<TuningProfileOutcome>();
    assert_send_sync::<TuningProfileDecision>();
}
