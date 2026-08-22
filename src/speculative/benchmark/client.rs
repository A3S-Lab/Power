//! Async client for measuring Power's real streaming completion API.

use std::net::IpAddr;
use std::process::Command;
use std::time::{Duration, Instant};

use futures::StreamExt;
use reqwest::{Client, RequestBuilder, Response, StatusCode, Url};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

use super::{
    build_report, is_revision, validate_digest, validate_label, SpeculativeBenchmarkIdentity,
    SpeculativeBenchmarkReport, SpeculativeBenchmarkSample, SpeculativeBenchmarkSystem,
    SpeculativeBenchmarkWorkload, SpeculativeServerConfig, MAX_BENCHMARK_SAMPLES,
    MAX_BENCHMARK_WARMUP_RUNS, MAX_COMPLETION_TOKENS,
};
use crate::api::health::InferenceStatus;
use crate::api::receipt::{completion_receipt, receipt_digest, AttestationReceipt};
use crate::api::types::{CompletionRequest, ModelInfo, StreamingPerformance, Usage};
use crate::error::{PowerError, Result};
use crate::speculative::SpeculativeStrategy;

const MAX_CONTROL_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_SSE_EVENT_BYTES: usize = 2 * 1024 * 1024;
const MAX_PROMPT_BYTES: usize = 1024 * 1024;

/// Inputs that are intentionally absent from the emitted path-free report.
pub struct SpeculativeBenchmarkRunConfig {
    pub base_url: Url,
    pub api_key: Option<Zeroizing<String>>,
    pub model: String,
    pub expected_model_sha256: String,
    pub mode: SpeculativeStrategy,
    pub power_commit: String,
    pub hardware_label: String,
    pub prompt: Zeroizing<String>,
    pub max_tokens: u32,
    pub num_ctx: u32,
    pub num_batch: Option<u32>,
    pub seed: i64,
    pub warmup_runs: u32,
    pub samples: usize,
    pub min_required_tokens_per_second: f64,
    pub min_required_sample_tokens_per_second: Option<f64>,
    pub timeout: Duration,
}

#[derive(Debug, Deserialize)]
struct HealthDocument {
    status: String,
    version: String,
    speculative: HealthSpeculativeConfig,
    inference: InferenceStatus,
}

#[derive(Debug, Deserialize)]
struct HealthSpeculativeConfig {
    mode: String,
    draft_max: Option<u32>,
    #[serde(default = "crate::config::default_spec_mtp_recurrent_snapshots")]
    mtp_recurrent_snapshots: u32,
    #[serde(default = "crate::config::default_spec_mtp_recurrent_chain")]
    mtp_recurrent_chain: bool,
    #[serde(default = "crate::config::default_spec_mtp_adaptive")]
    mtp_adaptive: bool,
    #[serde(default)]
    mtp_fr_vocab_size: Option<u32>,
    draft_min: u32,
    draft_p_min: f32,
}

#[derive(Debug, Deserialize)]
struct StreamEnvelope {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    a3s_performance: Option<StreamingPerformance>,
    #[serde(default)]
    attestation_receipt: Option<AttestationReceipt>,
    #[serde(default)]
    attestation_receipt_sha256: Option<String>,
    #[serde(default)]
    error: Option<StreamError>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamError {
    message: String,
}

#[derive(Serialize)]
struct BenchmarkCompletionRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    stream_options: BenchmarkStreamOptions,
    temperature: f32,
    top_p: f32,
    max_tokens: u32,
    num_ctx: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_batch: Option<u32>,
    seed: i64,
    keep_alive: &'static str,
}

#[derive(Serialize)]
struct BenchmarkStreamOptions {
    include_usage: bool,
}

pub async fn run_benchmark(
    config: &SpeculativeBenchmarkRunConfig,
) -> Result<SpeculativeBenchmarkReport> {
    validate_run_config(config)?;
    let client = Client::builder()
        .timeout(config.timeout)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .map_err(http_error)?;

    let health_url = endpoint(&config.base_url, &["health"])?;
    let health: HealthDocument = get_json(
        &client,
        health_url,
        config.api_key.as_ref().map(|key| key.as_str()),
    )
    .await?;
    if health.status != "ok" {
        return Err(invalid("Power health endpoint did not report status=ok"));
    }
    let observed_mode = SpeculativeStrategy::parse(&health.speculative.mode)
        .ok_or_else(|| invalid("Power health endpoint reported an unknown speculative mode"))?;
    if observed_mode != config.mode {
        return Err(invalid(format!(
            "Power configured spec_mode '{}' but benchmark expected '{}'",
            observed_mode.as_str(),
            config.mode.as_str()
        )));
    }
    if health.inference.suppress_token_metrics {
        return Err(invalid(
            "Power suppress_token_metrics must be false for exact benchmark evidence",
        ));
    }
    let speculative = SpeculativeServerConfig {
        mode: observed_mode,
        draft_max: health.speculative.draft_max,
        mtp_recurrent_snapshots: health.speculative.mtp_recurrent_snapshots,
        mtp_recurrent_chain: health.speculative.mtp_recurrent_chain,
        mtp_adaptive: health.speculative.mtp_adaptive,
        mtp_fr_vocab_size: health.speculative.mtp_fr_vocab_size,
        draft_min: health.speculative.draft_min,
        draft_p_min: health.speculative.draft_p_min,
    };
    validate_speculative_batch(config.num_batch, &speculative)?;

    let model_url = endpoint(&config.base_url, &["v1", "models", &config.model])?;
    let model_info: ModelInfo = get_json(
        &client,
        model_url,
        config.api_key.as_ref().map(|key| key.as_str()),
    )
    .await?;
    if model_info.id != config.model || model_info.format.as_deref() != Some("gguf") {
        return Err(invalid(
            "benchmark target is not the requested registered GGUF model",
        ));
    }
    let registered_sha256 = model_info
        .sha256
        .ok_or_else(|| invalid("registered model does not expose a SHA-256 identity"))?;
    if registered_sha256 != config.expected_model_sha256 {
        return Err(PowerError::IntegrityCheckFailed {
            model: config.model.clone(),
            expected: config.expected_model_sha256.clone(),
            actual: registered_sha256,
        });
    }
    let model_bytes = model_info
        .size_bytes
        .filter(|size| *size > 0)
        .ok_or_else(|| invalid("registered model does not expose a positive artifact size"))?;

    let completion_url = endpoint(&config.base_url, &["v1", "completions"])?;
    let request = BenchmarkCompletionRequest {
        model: &config.model,
        prompt: config.prompt.as_str(),
        stream: true,
        stream_options: BenchmarkStreamOptions {
            include_usage: true,
        },
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: config.max_tokens,
        num_ctx: config.num_ctx,
        num_batch: config.num_batch,
        seed: config.seed,
        keep_alive: "-1",
    };
    let request_bytes = Zeroizing::new(serde_json::to_vec(&request)?);
    let mut receipt_request: CompletionRequest = serde_json::from_slice(request_bytes.as_slice())?;
    let expected_receipt = completion_receipt(&receipt_request);
    receipt_request.prompt.zeroize();
    let expected_receipt = expected_receipt?;
    let workload = SpeculativeBenchmarkWorkload {
        prompt_sha256: hex::encode(Sha256::digest(config.prompt.as_bytes())),
        request_sha256: hex::encode(Sha256::digest(request_bytes.as_slice())),
        max_tokens: config.max_tokens,
        num_ctx: config.num_ctx,
        num_batch: config.num_batch,
        seed: config.seed,
        temperature: 0.0,
        top_p: 1.0,
    };
    drop(request_bytes);

    for _ in 0..config.warmup_runs {
        measure_completion(
            &client,
            completion_url.clone(),
            config.api_key.as_ref().map(|key| key.as_str()),
            &request,
            &expected_receipt,
            config.max_tokens,
        )
        .await?;
    }

    let mut samples = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        samples.push(
            measure_completion(
                &client,
                completion_url.clone(),
                config.api_key.as_ref().map(|key| key.as_str()),
                &request,
                &expected_receipt,
                config.max_tokens,
            )
            .await?,
        );
    }

    let identity = SpeculativeBenchmarkIdentity {
        power_commit: config.power_commit.clone(),
        server_version: health.version,
        model: config.model.clone(),
        model_sha256: config.expected_model_sha256.clone(),
        model_bytes,
        speculative,
        inference: health.inference,
        system: detect_system(&config.hardware_label),
    };
    build_report(
        identity,
        workload,
        config.warmup_runs,
        samples,
        config.min_required_tokens_per_second,
        config.min_required_sample_tokens_per_second,
    )
}

async fn measure_completion(
    client: &Client,
    url: Url,
    api_key: Option<&str>,
    request: &BenchmarkCompletionRequest<'_>,
    expected_receipt: &AttestationReceipt,
    expected_tokens: u32,
) -> Result<SpeculativeBenchmarkSample> {
    let started = Instant::now();
    let response = authorize(client.post(url).json(request), api_key)
        .send()
        .await
        .map_err(http_error)?;
    if !response.status().is_success() {
        return Err(response_failure("completion", response).await?);
    }
    let is_sse = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"));
    if !is_sse {
        return Err(invalid("Power completion response was not an SSE stream"));
    }

    let mut decoder = SseDecoder::default();
    let mut stream = response.bytes_stream();
    let mut output = Sha256::new();
    let mut final_usage = None;
    let mut performance = None;
    let mut receipt = None;
    let mut receipt_sha256 = None;
    let mut finish_reason = None;
    let mut saw_done = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(http_error)?;
        for data in decoder.push(&chunk)? {
            if saw_done {
                return Err(invalid("Power SSE stream contained data after [DONE]"));
            }
            if data == b"[DONE]" {
                saw_done = true;
                continue;
            }
            if performance.is_some() {
                return Err(invalid(
                    "Power SSE stream contained data after its final performance event",
                ));
            }
            let event: StreamEnvelope = serde_json::from_slice(&data)?;
            if let Some(error) = event.error {
                return Err(PowerError::InferenceFailed(format!(
                    "Power SSE error: {}",
                    error.message
                )));
            }
            if event.choices.len() > 1 || event.choices.iter().any(|choice| choice.index != 0) {
                return Err(invalid(
                    "Power SSE benchmark permits at most one choice at index zero",
                ));
            }
            let is_performance_event = event.a3s_performance.is_some();
            if is_performance_event {
                if !event.choices.is_empty()
                    || event.usage.is_none()
                    || event.attestation_receipt.is_none()
                    || event.attestation_receipt_sha256.is_none()
                {
                    return Err(invalid("Power SSE final performance event is incomplete"));
                }
                final_usage = event.usage;
                performance = event.a3s_performance;
                receipt = event.attestation_receipt;
                receipt_sha256 = event.attestation_receipt_sha256;
                continue;
            }
            if event.attestation_receipt.is_some() || event.attestation_receipt_sha256.is_some() {
                return Err(invalid(
                    "Power SSE receipt was not attached to the final performance event",
                ));
            }
            for choice in event.choices {
                output.update(choice.text.as_bytes());
                if choice.finish_reason.is_some() {
                    if finish_reason.is_some() {
                        return Err(invalid(
                            "Power SSE stream contained multiple terminal choices",
                        ));
                    }
                    finish_reason = choice.finish_reason;
                }
            }
        }
    }
    decoder.finish()?;
    if !saw_done {
        return Err(invalid("Power SSE stream ended without [DONE]"));
    }
    if finish_reason.as_deref() != Some("length") {
        return Err(invalid(
            "benchmark completion terminated before the fixed token limit; use a prompt that does not emit EOS",
        ));
    }
    let usage = final_usage.ok_or_else(|| invalid("Power SSE stream omitted exact usage"))?;
    if usage.completion_tokens != expected_tokens {
        return Err(invalid(format!(
            "benchmark expected {expected_tokens} completion tokens but received {}",
            usage.completion_tokens
        )));
    }
    let performance = performance.ok_or_else(|| {
        invalid(
            "Power SSE stream omitted a3s_performance; disable token metric suppression and use a compatible server revision",
        )
    })?;
    if performance.completion_token_intervals != usage.completion_tokens.saturating_sub(1) {
        return Err(invalid(
            "Power SSE timing intervals do not match exact usage",
        ));
    }
    let receipt =
        receipt.ok_or_else(|| invalid("Power SSE stream omitted its inference receipt"))?;
    let receipt_sha256 = receipt_sha256
        .ok_or_else(|| invalid("Power SSE stream omitted its inference receipt digest"))?;
    let actual_receipt_digest = receipt_digest(&receipt)?;
    if actual_receipt_digest != receipt_sha256 {
        return Err(invalid("Power SSE inference receipt digest did not verify"));
    }
    crate::verify::verify_receipt_well_formed(&receipt)?;
    let mut request_receipt = receipt.clone();
    request_receipt.runtime_policy = None;
    request_receipt.effective_prompt = None;
    if &request_receipt != expected_receipt {
        return Err(invalid(
            "Power SSE inference receipt does not match the benchmark request",
        ));
    }

    SpeculativeBenchmarkSample::new(
        usage.completion_tokens,
        performance.completion_token_intervals,
        performance.time_to_first_token_ns,
        performance.inter_token_duration_ns,
        u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        hex::encode(output.finalize()),
        receipt_sha256,
    )
}

async fn get_json<T: DeserializeOwned>(
    client: &Client,
    url: Url,
    api_key: Option<&str>,
) -> Result<T> {
    let response = authorize(client.get(url), api_key)
        .send()
        .await
        .map_err(http_error)?;
    if !response.status().is_success() {
        return Err(response_failure("control", response).await?);
    }
    let bytes = bounded_body(response, MAX_CONTROL_RESPONSE_BYTES).await?;
    Ok(serde_json::from_slice(&bytes)?)
}

async fn bounded_body(response: Response, maximum: usize) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(http_error)?;
        if body.len().saturating_add(chunk.len()) > maximum {
            return Err(invalid(
                "Power control response exceeded the benchmark bound",
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

async fn response_failure(kind: &str, response: Response) -> Result<PowerError> {
    let status = response.status();
    let body = bounded_body(response, MAX_CONTROL_RESPONSE_BYTES).await?;
    Ok(format_response_failure(kind, status, &body))
}

fn format_response_failure(kind: &str, status: StatusCode, body: &[u8]) -> PowerError {
    let detail = String::from_utf8_lossy(body);
    let detail = detail.trim();
    let message = if detail.is_empty() {
        format!("Power {kind} request failed with HTTP {status}")
    } else {
        format!("Power {kind} request failed with HTTP {status}: {detail}")
    };
    invalid(message)
}

fn authorize(builder: RequestBuilder, api_key: Option<&str>) -> RequestBuilder {
    match api_key {
        Some(api_key) => builder.bearer_auth(api_key),
        None => builder,
    }
}

fn endpoint(base: &Url, segments: &[&str]) -> Result<Url> {
    let mut url = base.clone();
    {
        let mut path = url
            .path_segments_mut()
            .map_err(|_| invalid("benchmark base URL cannot be used as a hierarchical URL"))?;
        path.pop_if_empty();
        for segment in segments {
            path.push(segment);
        }
    }
    Ok(url)
}

fn validate_run_config(config: &SpeculativeBenchmarkRunConfig) -> Result<()> {
    validate_label("model", &config.model, 256)?;
    validate_digest("expected model", &config.expected_model_sha256)?;
    if !is_revision(&config.power_commit) {
        return Err(invalid(
            "power_commit must be a lowercase 40- or 64-character hexadecimal revision",
        ));
    }
    validate_label("hardware label", &config.hardware_label, 256)?;
    if config
        .api_key
        .as_ref()
        .is_some_and(|key| key.is_empty() || key.len() > 64 * 1024)
    {
        return Err(invalid("API key must contain between 1 byte and 64 KiB"));
    }
    if config.base_url.query().is_some()
        || config.base_url.fragment().is_some()
        || !config.base_url.username().is_empty()
        || config.base_url.password().is_some()
        || config.base_url.host_str().is_none()
    {
        return Err(invalid(
            "benchmark base URL must not contain credentials, a query, or a fragment",
        ));
    }
    match config.base_url.scheme() {
        "https" => {}
        "http" if is_loopback(&config.base_url) => {}
        _ => {
            return Err(invalid(
                "benchmark transport must use HTTPS, except that HTTP is allowed for loopback hosts",
            ));
        }
    }
    if matches!(config.mode, SpeculativeStrategy::Auto) {
        return Err(invalid("benchmark mode must be explicit, not auto"));
    }
    if config.prompt.is_empty() || config.prompt.len() > MAX_PROMPT_BYTES {
        return Err(invalid(
            "benchmark prompt must contain between 1 byte and 1 MiB",
        ));
    }
    if !(2..=MAX_COMPLETION_TOKENS).contains(&config.max_tokens)
        || config.num_ctx < config.max_tokens
        || config
            .num_batch
            .is_some_and(|batch| batch == 0 || batch > config.num_ctx)
        || config.warmup_runs > MAX_BENCHMARK_WARMUP_RUNS
        || config.samples == 0
        || config.samples > MAX_BENCHMARK_SAMPLES
        || config.timeout.is_zero()
        || !config.min_required_tokens_per_second.is_finite()
        || config.min_required_tokens_per_second < 0.0
        || config
            .min_required_sample_tokens_per_second
            .is_some_and(|threshold| !threshold.is_finite() || threshold < 0.0)
    {
        return Err(invalid(
            "speculative benchmark numeric settings are out of bounds",
        ));
    }
    Ok(())
}

fn validate_speculative_batch(
    num_batch: Option<u32>,
    speculative: &SpeculativeServerConfig,
) -> Result<()> {
    if speculative.mode != SpeculativeStrategy::Mtp {
        return Ok(());
    }
    let minimum_batch = crate::speculative::minimum_mtp_batch(speculative.draft_max.unwrap_or(3));
    if num_batch.is_some_and(|batch| batch < minimum_batch) {
        return Err(invalid(format!(
            "MTP benchmark num_batch must be at least draft_max + 2 ({minimum_batch})"
        )));
    }
    Ok(())
}

fn is_loopback(url: &Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    host.eq_ignore_ascii_case("localhost")
        || host.ends_with(".localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn detect_system(label: &str) -> SpeculativeBenchmarkSystem {
    let gpu = crate::backend::gpu::detect();
    SpeculativeBenchmarkSystem {
        label: label.to_string(),
        os: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
        gpu_backend: gpu.backend.to_string().to_ascii_lowercase(),
        gpu_name: gpu.name,
        gpu_vram_bytes: gpu.vram_bytes,
        gpu_driver: detect_nvidia_driver(),
    }
}

fn detect_nvidia_driver() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let driver = value.lines().next()?.trim();
    (!driver.is_empty()).then(|| driver.to_string())
}

fn http_error(error: reqwest::Error) -> PowerError {
    PowerError::Server(format!("speculative benchmark HTTP failure: {error}"))
}

fn invalid(message: impl Into<String>) -> PowerError {
    PowerError::InvalidRequest(message.into())
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<Vec<u8>>> {
        if self.buffer.len().saturating_add(chunk.len()) > MAX_SSE_EVENT_BYTES {
            return Err(invalid("Power SSE event exceeded the benchmark bound"));
        }
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some((offset, delimiter_len)) = event_boundary(&self.buffer) {
            let frame = self.buffer.drain(..offset).collect::<Vec<_>>();
            self.buffer.drain(..delimiter_len);
            if let Some(data) = event_data(&frame) {
                events.push(data);
            }
        }
        Ok(events)
    }

    fn finish(&self) -> Result<()> {
        if self.buffer.iter().all(u8::is_ascii_whitespace) {
            Ok(())
        } else {
            Err(invalid("Power SSE stream ended with an incomplete event"))
        }
    }
}

fn event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    for index in 0..buffer.len() {
        if buffer.get(index..index + 4) == Some(b"\r\n\r\n") {
            return Some((index, 4));
        }
        if buffer.get(index..index + 2) == Some(b"\n\n") {
            return Some((index, 2));
        }
    }
    None
}

fn event_data(frame: &[u8]) -> Option<Vec<u8>> {
    let mut data = Vec::new();
    for raw_line in frame.split(|byte| *byte == b'\n') {
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        let Some(value) = line.strip_prefix(b"data:") else {
            continue;
        };
        let value = value.strip_prefix(b" ").unwrap_or(value);
        if !data.is_empty() {
            data.push(b'\n');
        }
        data.extend_from_slice(value);
    }
    (!data.is_empty()).then_some(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{header, Response};
    use axum::routing::{get, post};
    use axum::{Json, Router};

    use crate::api::receipt::{completion_receipt, receipt_digest};
    use crate::api::types::CompletionRequest;

    async fn benchmark_health() -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "status": "ok",
            "version": "0.8.0",
            "speculative": {
                "mode": "none",
                "draft_max": 3,
                "mtp_recurrent_snapshots": 5,
                "mtp_recurrent_chain": false,
                "draft_min": 0,
                "draft_p_min": 0.0
            },
            "inference": {
                "gpu_layers": -1,
                "main_gpu": 0,
                "tensor_split": [],
                "num_thread": 8,
                "flash_attention": true,
                "num_parallel": 1,
                "use_mlock": false,
                "tee_mode": false,
                "suppress_token_metrics": false
            }
        }))
    }

    async fn benchmark_model() -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "id": "generic-gguf",
            "object": "model",
            "created": 0,
            "owned_by": "local",
            "format": "gguf",
            "size_bytes": 123,
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        }))
    }

    async fn benchmark_completion(Json(request): Json<CompletionRequest>) -> Response<Body> {
        assert_eq!(request.model, "generic-gguf");
        assert_eq!(request.max_tokens, Some(2));
        assert_eq!(request.temperature, Some(0.0));
        assert_eq!(request.top_p, Some(1.0));
        assert_eq!(request.num_batch, Some(4));
        assert_eq!(request.stream, Some(true));
        let receipt = completion_receipt(&request).unwrap();
        let digest = receipt_digest(&receipt).unwrap();
        let events = [
            serde_json::json!({
                "choices": [{"text": "A", "finish_reason": null}]
            }),
            serde_json::json!({
                "choices": [{"text": "B", "finish_reason": "length"}]
            }),
            serde_json::json!({
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5
                },
                "a3s_performance": {
                    "time_to_first_token_ns": 0,
                    "inter_token_duration_ns": 1,
                    "completion_token_intervals": 1
                },
                "attestation_receipt": receipt,
                "attestation_receipt_sha256": digest
            }),
        ];
        let mut body = events
            .into_iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>();
        body.push_str("data: [DONE]\n\n");
        Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from(body))
            .unwrap()
    }

    async fn spawn_benchmark_server() -> (
        Url,
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<()>,
    ) {
        let app = Router::new()
            .route("/health", get(benchmark_health))
            .route("/v1/models/generic-gguf", get(benchmark_model))
            .route("/v1/completions", post(benchmark_completion));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await
                .unwrap();
        });
        (
            Url::parse(&format!("http://{address}/")).unwrap(),
            shutdown_tx,
            task,
        )
    }

    #[test]
    fn sse_decoder_handles_split_crlf_and_multiple_events() {
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(b"data: {\"one\":1}\r\n\r").unwrap().is_empty());
        let events = decoder
            .push(b"\ndata: {\"two\":2}\n\ndata: [DONE]\n\n")
            .unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], br#"{"one":1}"#);
        assert_eq!(events[1], br#"{"two":2}"#);
        assert_eq!(events[2], b"[DONE]");
        decoder.finish().unwrap();
    }

    #[test]
    fn sse_decoder_joins_multiline_data() {
        let mut decoder = SseDecoder::default();
        let events = decoder
            .push(b"event: message\ndata: first\ndata: second\n\n")
            .unwrap();
        assert_eq!(events, vec![b"first\nsecond".to_vec()]);
    }

    #[test]
    fn benchmark_http_failure_preserves_bounded_server_diagnostic() {
        let error = format_response_failure(
            "completion",
            StatusCode::SERVICE_UNAVAILABLE,
            br#"{"error":{"code":"model_load_failed","message":"device allocation failed"}}"#,
        );
        let message = error.to_string();
        assert!(message.contains("503 Service Unavailable"));
        assert!(message.contains("model_load_failed"));
        assert!(message.contains("device allocation failed"));
    }

    #[test]
    fn benchmark_http_failure_handles_empty_body() {
        let error = format_response_failure("control", StatusCode::BAD_GATEWAY, b"  \r\n");
        assert_eq!(
            error.to_string(),
            "Invalid request: Power control request failed with HTTP 502 Bad Gateway"
        );
    }

    #[test]
    fn benchmark_url_rejects_remote_plaintext_transport() {
        let config = SpeculativeBenchmarkRunConfig {
            base_url: Url::parse("http://example.com/").unwrap(),
            api_key: None,
            model: "model".to_string(),
            expected_model_sha256: "a".repeat(64),
            mode: SpeculativeStrategy::Off,
            power_commit: "b".repeat(40),
            hardware_label: "test".to_string(),
            prompt: Zeroizing::new("prompt".to_string()),
            max_tokens: 16,
            num_ctx: 2048,
            num_batch: Some(4),
            seed: 42,
            warmup_runs: 0,
            samples: 1,
            min_required_tokens_per_second: 0.0,
            min_required_sample_tokens_per_second: None,
            timeout: Duration::from_secs(1),
        };
        assert!(validate_run_config(&config).is_err());
    }

    #[tokio::test]
    async fn benchmark_runs_through_http_sse_and_emits_path_free_evidence() {
        let (base_url, shutdown, task) = spawn_benchmark_server().await;
        let prompt = "private benchmark prompt";
        let config = SpeculativeBenchmarkRunConfig {
            base_url,
            api_key: Some(Zeroizing::new("secret".to_string())),
            model: "generic-gguf".to_string(),
            expected_model_sha256: "a".repeat(64),
            mode: SpeculativeStrategy::Off,
            power_commit: "b".repeat(40),
            hardware_label: "loopback-test".to_string(),
            prompt: Zeroizing::new(prompt.to_string()),
            max_tokens: 2,
            num_ctx: 2048,
            num_batch: Some(4),
            seed: 42,
            warmup_runs: 0,
            samples: 2,
            min_required_tokens_per_second: 100.0,
            min_required_sample_tokens_per_second: Some(100.0),
            timeout: Duration::from_secs(5),
        };

        let report = run_benchmark(&config).await.unwrap();
        let _ = shutdown.send(());
        task.await.unwrap();

        assert_eq!(report.identity.speculative.mode, SpeculativeStrategy::Off);
        assert_eq!(report.identity.speculative.mtp_recurrent_snapshots, 5);
        assert!(!report.identity.speculative.mtp_recurrent_chain);
        assert_eq!(report.workload.num_batch, Some(4));
        assert_eq!(report.samples.len(), 2);
        assert_eq!(report.median_decode_tokens_per_second, 1_000_000_000.0);
        assert!(report.threshold_passed);
        assert!(report.stability.as_ref().unwrap().threshold_passed);
        assert!(report.all_thresholds_passed());
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains(prompt));
        assert!(!json.contains("127.0.0.1"));
        assert!(!json.contains("secret"));
    }

    #[test]
    fn mtp_batch_must_cover_verification_rows_and_recurrent_tail() {
        let speculative = SpeculativeServerConfig {
            mode: SpeculativeStrategy::Mtp,
            draft_max: Some(3),
            mtp_recurrent_snapshots: 7,
            mtp_recurrent_chain: true,
            mtp_adaptive: false,
            mtp_fr_vocab_size: Some(8192),
            draft_min: 0,
            draft_p_min: 0.0,
        };
        assert!(validate_speculative_batch(Some(3), &speculative).is_err());
        assert!(validate_speculative_batch(Some(4), &speculative).is_err());
        validate_speculative_batch(Some(5), &speculative).unwrap();
        validate_speculative_batch(None, &speculative).unwrap();
    }
}
