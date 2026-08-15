//! Path-free evidence for end-to-end speculative-decoding benchmarks.
//!
//! The HTTP runner lives in [`client`]. This module owns the stable report,
//! validation, digest, and baseline/candidate comparison contract so reports
//! can be checked without contacting a server or loading model weights.

pub mod client;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::SpeculativeStrategy;
use crate::api::health::InferenceStatus;
use crate::error::{PowerError, Result};

pub const REPORT_SCHEMA: &str = "a3s.power.speculative-benchmark.v1";
pub const COMPARISON_SCHEMA: &str = "a3s.power.speculative-benchmark-comparison.v1";
pub const MAX_BENCHMARK_SAMPLES: usize = 100;
pub const MAX_BENCHMARK_WARMUP_RUNS: u32 = 10;
pub const MAX_COMPLETION_TOKENS: u32 = 4096;

/// Explicit server configuration observed through `GET /health`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeServerConfig {
    pub mode: SpeculativeStrategy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_max: Option<u32>,
    pub draft_min: u32,
    pub draft_p_min: f32,
}

/// Named, path-free hardware identity for controlled comparisons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeBenchmarkSystem {
    pub label: String,
    pub os: String,
    pub architecture: String,
    pub gpu_backend: String,
    pub gpu_name: String,
    pub gpu_vram_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_driver: Option<String>,
}

/// Server, model, configuration, and hardware identity bound to one report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeBenchmarkIdentity {
    pub power_commit: String,
    pub server_version: String,
    pub model: String,
    pub model_sha256: String,
    pub model_bytes: u64,
    pub speculative: SpeculativeServerConfig,
    pub inference: InferenceStatus,
    pub system: SpeculativeBenchmarkSystem,
}

/// Prompt-free workload identity. The actual prompt never enters the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeBenchmarkWorkload {
    pub prompt_sha256: String,
    pub request_sha256: String,
    pub max_tokens: u32,
    pub num_ctx: u32,
    pub seed: i64,
    pub temperature: f32,
    pub top_p: f32,
}

/// One measured Power SSE completion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeBenchmarkSample {
    pub completion_tokens: u32,
    pub completion_token_intervals: u32,
    pub time_to_first_token_ns: u64,
    pub inter_token_duration_ns: u64,
    pub total_duration_ns: u64,
    pub decode_tokens_per_second: f64,
    pub end_to_end_tokens_per_second: f64,
    pub output_sha256: String,
    pub receipt_sha256: String,
}

impl SpeculativeBenchmarkSample {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        completion_tokens: u32,
        completion_token_intervals: u32,
        time_to_first_token_ns: u64,
        inter_token_duration_ns: u64,
        total_duration_ns: u64,
        output_sha256: String,
        receipt_sha256: String,
    ) -> Result<Self> {
        if completion_tokens < 2
            || completion_token_intervals != completion_tokens.saturating_sub(1)
            || inter_token_duration_ns == 0
            || total_duration_ns == 0
        {
            return Err(invalid(
                "a speculative benchmark sample requires at least two tokens, exact token intervals, and non-zero durations",
            ));
        }
        let decode_tokens_per_second = rate(completion_token_intervals, inter_token_duration_ns);
        let end_to_end_tokens_per_second = rate(completion_tokens, total_duration_ns);
        let sample = Self {
            completion_tokens,
            completion_token_intervals,
            time_to_first_token_ns,
            inter_token_duration_ns,
            total_duration_ns,
            decode_tokens_per_second,
            end_to_end_tokens_per_second,
            output_sha256,
            receipt_sha256,
        };
        validate_sample(&sample)?;
        Ok(sample)
    }
}

/// Validated report for one explicit speculative strategy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeBenchmarkReport {
    pub schema: String,
    pub identity: SpeculativeBenchmarkIdentity,
    pub workload: SpeculativeBenchmarkWorkload,
    pub warmup_runs: u32,
    pub samples: Vec<SpeculativeBenchmarkSample>,
    pub output_sha256: String,
    pub median_decode_tokens_per_second: f64,
    pub minimum_decode_tokens_per_second: f64,
    pub min_required_tokens_per_second: f64,
    pub threshold_passed: bool,
}

/// Controlled comparison between an autoregressive baseline and a candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeBenchmarkComparison {
    pub schema: String,
    pub baseline_report_sha256: String,
    pub candidate_report_sha256: String,
    pub baseline_mode: SpeculativeStrategy,
    pub candidate_mode: SpeculativeStrategy,
    pub baseline_median_tokens_per_second: f64,
    pub candidate_median_tokens_per_second: f64,
    pub candidate_min_required_tokens_per_second: f64,
    pub speedup: f64,
    pub output_parity: bool,
    pub candidate_threshold_passed: bool,
    pub passed: bool,
}

pub fn build_report(
    identity: SpeculativeBenchmarkIdentity,
    workload: SpeculativeBenchmarkWorkload,
    warmup_runs: u32,
    samples: Vec<SpeculativeBenchmarkSample>,
    min_required_tokens_per_second: f64,
) -> Result<SpeculativeBenchmarkReport> {
    if warmup_runs > MAX_BENCHMARK_WARMUP_RUNS
        || samples.is_empty()
        || samples.len() > MAX_BENCHMARK_SAMPLES
    {
        return Err(invalid(format!(
            "speculative benchmark requires at most {MAX_BENCHMARK_WARMUP_RUNS} warmups and between 1 and {MAX_BENCHMARK_SAMPLES} samples"
        )));
    }
    let mut rates = samples
        .iter()
        .map(|sample| sample.decode_tokens_per_second)
        .collect::<Vec<_>>();
    rates.sort_by(f64::total_cmp);
    let median_decode_tokens_per_second = median(&rates);
    let minimum_decode_tokens_per_second = rates[0];
    let output_sha256 = samples[0].output_sha256.clone();
    let threshold_passed = median_decode_tokens_per_second >= min_required_tokens_per_second;
    let report = SpeculativeBenchmarkReport {
        schema: REPORT_SCHEMA.to_string(),
        identity,
        workload,
        warmup_runs,
        samples,
        output_sha256,
        median_decode_tokens_per_second,
        minimum_decode_tokens_per_second,
        min_required_tokens_per_second,
        threshold_passed,
    };
    validate_report(&report)?;
    Ok(report)
}

pub fn validate_report(report: &SpeculativeBenchmarkReport) -> Result<()> {
    if report.schema != REPORT_SCHEMA {
        return Err(invalid("unsupported speculative benchmark report schema"));
    }
    validate_identity(&report.identity)?;
    validate_workload(&report.workload)?;
    if report.warmup_runs > MAX_BENCHMARK_WARMUP_RUNS
        || report.samples.is_empty()
        || report.samples.len() > MAX_BENCHMARK_SAMPLES
    {
        return Err(invalid(
            "speculative benchmark warmup or sample count is out of bounds",
        ));
    }
    if !report.min_required_tokens_per_second.is_finite()
        || report.min_required_tokens_per_second < 0.0
    {
        return Err(invalid(
            "speculative benchmark threshold must be finite and non-negative",
        ));
    }
    let mut rates = Vec::with_capacity(report.samples.len());
    for sample in &report.samples {
        validate_sample(sample)?;
        if sample.completion_tokens != report.workload.max_tokens {
            return Err(invalid(
                "a benchmark sample did not reach the fixed output-token limit",
            ));
        }
        if sample.output_sha256 != report.output_sha256 {
            return Err(invalid(
                "benchmark samples did not produce deterministic output",
            ));
        }
        rates.push(sample.decode_tokens_per_second);
    }
    validate_digest("report output", &report.output_sha256)?;
    rates.sort_by(f64::total_cmp);
    let expected_median = median(&rates);
    let expected_minimum = rates[0];
    if !close(report.median_decode_tokens_per_second, expected_median)
        || !close(report.minimum_decode_tokens_per_second, expected_minimum)
    {
        return Err(invalid(
            "speculative benchmark summary does not match its samples",
        ));
    }
    if report.threshold_passed != (expected_median >= report.min_required_tokens_per_second) {
        return Err(invalid(
            "speculative benchmark threshold verdict is inconsistent",
        ));
    }
    Ok(())
}

pub fn compare_reports(
    baseline: &SpeculativeBenchmarkReport,
    candidate: &SpeculativeBenchmarkReport,
) -> Result<SpeculativeBenchmarkComparison> {
    validate_report(baseline)?;
    validate_report(candidate)?;
    if baseline.identity.speculative.mode != SpeculativeStrategy::Off {
        return Err(invalid(
            "the baseline report must use explicit spec_mode=off",
        ));
    }
    if matches!(
        candidate.identity.speculative.mode,
        SpeculativeStrategy::Auto | SpeculativeStrategy::Off
    ) {
        return Err(invalid(
            "the candidate report must use an explicit speculative strategy",
        ));
    }
    if identity_without_strategy(&baseline.identity)
        != identity_without_strategy(&candidate.identity)
        || baseline.workload != candidate.workload
        || baseline.warmup_runs != candidate.warmup_runs
        || baseline.samples.len() != candidate.samples.len()
    {
        return Err(invalid(
            "benchmark reports do not share one revision, model, workload, warmup/sample count, and named hardware environment",
        ));
    }
    let output_parity = baseline.output_sha256 == candidate.output_sha256;
    let speedup =
        candidate.median_decode_tokens_per_second / baseline.median_decode_tokens_per_second;
    let passed = output_parity && candidate.threshold_passed;
    Ok(SpeculativeBenchmarkComparison {
        schema: COMPARISON_SCHEMA.to_string(),
        baseline_report_sha256: report_digest(baseline)?,
        candidate_report_sha256: report_digest(candidate)?,
        baseline_mode: baseline.identity.speculative.mode,
        candidate_mode: candidate.identity.speculative.mode,
        baseline_median_tokens_per_second: baseline.median_decode_tokens_per_second,
        candidate_median_tokens_per_second: candidate.median_decode_tokens_per_second,
        candidate_min_required_tokens_per_second: candidate.min_required_tokens_per_second,
        speedup,
        output_parity,
        candidate_threshold_passed: candidate.threshold_passed,
        passed,
    })
}

pub fn report_digest(report: &SpeculativeBenchmarkReport) -> Result<String> {
    validate_report(report)?;
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(report)?)))
}

fn validate_identity(identity: &SpeculativeBenchmarkIdentity) -> Result<()> {
    if !is_revision(&identity.power_commit) {
        return Err(invalid(
            "power_commit must be a lowercase 40- or 64-character hexadecimal revision",
        ));
    }
    validate_label("server version", &identity.server_version, 64)?;
    validate_label("model", &identity.model, 256)?;
    validate_digest("model", &identity.model_sha256)?;
    if identity.model_bytes == 0 {
        return Err(invalid("model_bytes must be positive"));
    }
    if identity.speculative.mode == SpeculativeStrategy::Auto {
        return Err(invalid(
            "benchmarks require an explicit speculative mode, not auto",
        ));
    }
    if identity
        .speculative
        .draft_max
        .is_some_and(|value| value == 0 || value > 64)
        || identity.speculative.draft_min > 64
        || !identity.speculative.draft_p_min.is_finite()
        || !(0.0..=1.0).contains(&identity.speculative.draft_p_min)
    {
        return Err(invalid("server speculative draft settings are invalid"));
    }
    if identity
        .speculative
        .draft_max
        .is_some_and(|maximum| identity.speculative.draft_min > maximum)
    {
        return Err(invalid(
            "server speculative draft minimum exceeds its maximum",
        ));
    }
    validate_inference_status(&identity.inference)?;
    validate_label("hardware label", &identity.system.label, 256)?;
    validate_label("operating system", &identity.system.os, 64)?;
    validate_label("architecture", &identity.system.architecture, 64)?;
    validate_label("GPU backend", &identity.system.gpu_backend, 64)?;
    validate_label("GPU name", &identity.system.gpu_name, 256)?;
    if let Some(driver) = identity.system.gpu_driver.as_deref() {
        validate_label("GPU driver", driver, 128)?;
    }
    Ok(())
}

fn validate_inference_status(status: &InferenceStatus) -> Result<()> {
    let split_sum = status.tensor_split.iter().copied().sum::<f32>();
    if status.gpu_layers < -1
        || status.main_gpu < 0
        || status.num_thread.is_some_and(|threads| threads == 0)
        || status.num_parallel == 0
        || status.tensor_split.len() > 64
        || status
            .tensor_split
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || (!status.tensor_split.is_empty() && (!split_sum.is_finite() || split_sum <= 0.0))
        || status.suppress_token_metrics
    {
        return Err(invalid(
            "benchmark inference settings are invalid or suppress exact token metrics",
        ));
    }
    Ok(())
}

fn validate_workload(workload: &SpeculativeBenchmarkWorkload) -> Result<()> {
    validate_digest("prompt", &workload.prompt_sha256)?;
    validate_digest("request", &workload.request_sha256)?;
    if !(2..=MAX_COMPLETION_TOKENS).contains(&workload.max_tokens)
        || workload.num_ctx < workload.max_tokens
        || !workload.temperature.is_finite()
        || workload.temperature != 0.0
        || !workload.top_p.is_finite()
        || workload.top_p != 1.0
    {
        return Err(invalid(
            "benchmark workload requires max_tokens 2..=4096, sufficient context, temperature=0, and top_p=1",
        ));
    }
    Ok(())
}

fn validate_sample(sample: &SpeculativeBenchmarkSample) -> Result<()> {
    if sample.completion_tokens < 2
        || sample.completion_token_intervals != sample.completion_tokens.saturating_sub(1)
        || sample.inter_token_duration_ns == 0
        || sample.total_duration_ns == 0
        || sample.time_to_first_token_ns > sample.total_duration_ns
        || sample
            .time_to_first_token_ns
            .checked_add(sample.inter_token_duration_ns)
            .is_none_or(|last_token_ns| last_token_ns > sample.total_duration_ns)
    {
        return Err(invalid("speculative benchmark sample timing is invalid"));
    }
    validate_digest("sample output", &sample.output_sha256)?;
    validate_digest("sample receipt", &sample.receipt_sha256)?;
    let decode = rate(
        sample.completion_token_intervals,
        sample.inter_token_duration_ns,
    );
    let end_to_end = rate(sample.completion_tokens, sample.total_duration_ns);
    if !close(sample.decode_tokens_per_second, decode)
        || !close(sample.end_to_end_tokens_per_second, end_to_end)
    {
        return Err(invalid(
            "speculative benchmark sample rates do not match its timings",
        ));
    }
    Ok(())
}

fn identity_without_strategy(
    identity: &SpeculativeBenchmarkIdentity,
) -> (
    &str,
    &str,
    &str,
    &str,
    u64,
    Option<u32>,
    u32,
    f32,
    &InferenceStatus,
    &SpeculativeBenchmarkSystem,
) {
    (
        &identity.power_commit,
        &identity.server_version,
        &identity.model,
        &identity.model_sha256,
        identity.model_bytes,
        identity.speculative.draft_max,
        identity.speculative.draft_min,
        identity.speculative.draft_p_min,
        &identity.inference,
        &identity.system,
    )
}

fn rate(tokens: u32, duration_ns: u64) -> f64 {
    f64::from(tokens) * 1_000_000_000.0 / duration_ns as f64
}

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn close(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= expected.abs().max(1.0) * 1e-12
}

fn validate_digest(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(format!(
            "{label} SHA-256 must contain exactly 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn is_revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_label(label: &str, value: &str, maximum: usize) -> Result<()> {
    if value.is_empty()
        || value.len() > maximum
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(invalid(format!(
            "{label} must be a bounded non-control string without surrounding whitespace"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> PowerError {
    PowerError::InvalidRequest(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(mode: SpeculativeStrategy) -> SpeculativeBenchmarkIdentity {
        SpeculativeBenchmarkIdentity {
            power_commit: "a".repeat(40),
            server_version: "0.8.0".to_string(),
            model: "qwen3.5-27b-q6-k".to_string(),
            model_sha256: "b".repeat(64),
            model_bytes: 23_000_000_000,
            speculative: SpeculativeServerConfig {
                mode,
                draft_max: Some(3),
                draft_min: 0,
                draft_p_min: 0.0,
            },
            inference: InferenceStatus {
                gpu_layers: -1,
                main_gpu: 0,
                tensor_split: Vec::new(),
                num_thread: Some(16),
                flash_attention: true,
                num_parallel: 1,
                use_mlock: false,
                tee_mode: false,
                suppress_token_metrics: false,
                timing_padding_ms: None,
            },
            system: SpeculativeBenchmarkSystem {
                label: "rtx-4090-cuda".to_string(),
                os: "windows".to_string(),
                architecture: "x86_64".to_string(),
                gpu_backend: "cuda".to_string(),
                gpu_name: "NVIDIA GeForce RTX 4090".to_string(),
                gpu_vram_bytes: 24_000_000_000,
                gpu_driver: Some("610.74".to_string()),
            },
        }
    }

    fn workload() -> SpeculativeBenchmarkWorkload {
        SpeculativeBenchmarkWorkload {
            prompt_sha256: "c".repeat(64),
            request_sha256: "d".repeat(64),
            max_tokens: 4,
            num_ctx: 2048,
            seed: 42,
            temperature: 0.0,
            top_p: 1.0,
        }
    }

    fn sample(duration_ns: u64, output: char) -> SpeculativeBenchmarkSample {
        SpeculativeBenchmarkSample::new(
            4,
            3,
            10,
            duration_ns,
            duration_ns + 100,
            output.to_string().repeat(64),
            "e".repeat(64),
        )
        .unwrap()
    }

    fn report(
        mode: SpeculativeStrategy,
        output: char,
        threshold: f64,
    ) -> SpeculativeBenchmarkReport {
        build_report(
            identity(mode),
            workload(),
            1,
            vec![sample(30_000_000, output), sample(25_000_000, output)],
            threshold,
        )
        .unwrap()
    }

    #[test]
    fn report_recomputes_rates_and_threshold() {
        let report = report(SpeculativeStrategy::Mtp, 'f', 100.0);
        assert_eq!(report.median_decode_tokens_per_second, 110.0);
        assert_eq!(report.minimum_decode_tokens_per_second, 100.0);
        assert!(report.threshold_passed);
        validate_report(&report).unwrap();
    }

    #[test]
    fn comparison_requires_output_parity_for_success() {
        let baseline = report(SpeculativeStrategy::Off, 'f', 0.0);
        let candidate = report(SpeculativeStrategy::Mtp, 'f', 100.0);
        let comparison = compare_reports(&baseline, &candidate).unwrap();
        assert!(comparison.output_parity);
        assert!(comparison.passed);
        assert_eq!(comparison.candidate_min_required_tokens_per_second, 100.0);

        let mismatch = report(SpeculativeStrategy::Mtp, 'a', 100.0);
        let comparison = compare_reports(&baseline, &mismatch).unwrap();
        assert!(!comparison.output_parity);
        assert!(!comparison.passed);
    }

    #[test]
    fn comparison_rejects_identity_drift() {
        let baseline = report(SpeculativeStrategy::Off, 'f', 0.0);
        let mut candidate = report(SpeculativeStrategy::Mtp, 'f', 100.0);
        candidate.identity.model_sha256 = "1".repeat(64);
        assert!(compare_reports(&baseline, &candidate).is_err());

        let mut candidate = report(SpeculativeStrategy::Mtp, 'f', 100.0);
        candidate.identity.inference.num_thread = Some(8);
        assert!(compare_reports(&baseline, &candidate).is_err());

        let mut candidate = report(SpeculativeStrategy::Mtp, 'f', 100.0);
        candidate.identity.speculative.draft_max = Some(4);
        assert!(compare_reports(&baseline, &candidate).is_err());

        let mut candidate = report(SpeculativeStrategy::Mtp, 'f', 100.0);
        candidate.warmup_runs = 2;
        assert!(compare_reports(&baseline, &candidate).is_err());
    }

    #[test]
    fn validation_rejects_tampered_summary() {
        let mut report = report(SpeculativeStrategy::Mtp, 'f', 100.0);
        report.median_decode_tokens_per_second += 1.0;
        assert!(validate_report(&report).is_err());
    }
}
