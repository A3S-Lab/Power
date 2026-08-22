use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::backend::prompt_cache::PromptCacheMetricsSnapshot;
use crate::backend::SpeculativeMetricsSnapshot;
use crate::server::lock::{read_lock, write_lock};
use crate::server::state::AppState;

/// Maximum number of duration samples to keep per vector.
/// Prevents unbounded memory growth on long-running servers.
const MAX_SAMPLES: usize = 10_000;

/// Push to a Vec with a cap, dropping oldest entries when full.
fn capped_push<T>(vec: &mut Vec<T>, item: T) {
    if vec.len() >= MAX_SAMPLES {
        vec.remove(0);
    }
    vec.push(item);
}

/// Key for tracking HTTP request counts by method, path, and status.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct RequestKey {
    method: String,
    path: String,
    status: u16,
}

/// A single recorded request duration.
#[derive(Debug, Clone)]
struct RequestDuration {
    method: String,
    path: String,
    duration_secs: f64,
}

/// A single recorded model load duration.
#[derive(Debug, Clone)]
struct ModelLoadDuration {
    model: String,
    duration_secs: f64,
}

/// Key for tracking inference token counts.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct TokenKey {
    model: String,
    token_type: String,
}

/// A single recorded inference duration entry.
#[derive(Debug, Clone)]
struct InferenceDuration {
    model: String,
    duration_secs: f64,
}

/// A single recorded time-to-first-token entry.
#[derive(Debug, Clone)]
struct TtftEntry {
    model: String,
    ttft_secs: f64,
}

/// Prometheus metrics collector.
pub struct Metrics {
    // --- Existing metrics ---
    /// HTTP request counts by (method, path, status).
    http_requests: RwLock<Vec<(RequestKey, u64)>>,
    /// HTTP request durations.
    http_durations: RwLock<Vec<RequestDuration>>,
    /// Model load durations.
    model_load_durations: RwLock<Vec<ModelLoadDuration>>,
    /// Number of models currently loaded.
    pub models_loaded: AtomicU64,
    /// Inference token counts by (model, type).
    inference_tokens: RwLock<Vec<(TokenKey, u64)>>,

    // --- Phase 6: Inference duration & TTFT ---
    /// Per-model inference duration samples.
    inference_durations: RwLock<Vec<InferenceDuration>>,
    /// Per-model time-to-first-token samples.
    ttft_entries: RwLock<Vec<TtftEntry>>,

    // --- Model lifecycle ---
    /// Total number of model evictions.
    model_evictions: AtomicU64,
    /// Per-model estimated memory usage in bytes.
    model_memory_bytes: RwLock<Vec<(String, u64)>>,

    // --- Phase 6: GPU metrics ---
    /// Per-device GPU memory usage in bytes.
    gpu_memory_bytes: RwLock<Vec<(String, u64)>>,
    /// Per-device GPU utilization (0.0 - 1.0).
    gpu_utilization: RwLock<Vec<(String, f64)>>,

    // --- TEE metrics ---
    /// Total attestation reports generated.
    tee_attestations: AtomicU64,
    /// Total encrypted model decryptions performed.
    tee_model_decryptions: AtomicU64,
    /// Total log redaction calls (privacy provider active).
    tee_redactions: AtomicU64,

    // --- Auth metrics ---
    /// Total authentication failures.
    auth_failures: AtomicU64,

    // --- Request isolation metrics ---
    /// Number of currently active inference requests.
    active_requests: AtomicU64,
    /// Number of requests queued waiting for an admission permit.
    waiting_requests: AtomicU64,
    /// Number of admitted requests currently running (holding a permit).
    running_requests: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            http_requests: RwLock::new(Vec::new()),
            http_durations: RwLock::new(Vec::new()),
            model_load_durations: RwLock::new(Vec::new()),
            models_loaded: AtomicU64::new(0),
            inference_tokens: RwLock::new(Vec::new()),
            inference_durations: RwLock::new(Vec::new()),
            ttft_entries: RwLock::new(Vec::new()),
            model_evictions: AtomicU64::new(0),
            model_memory_bytes: RwLock::new(Vec::new()),
            gpu_memory_bytes: RwLock::new(Vec::new()),
            gpu_utilization: RwLock::new(Vec::new()),
            tee_attestations: AtomicU64::new(0),
            tee_model_decryptions: AtomicU64::new(0),
            tee_redactions: AtomicU64::new(0),
            auth_failures: AtomicU64::new(0),
            active_requests: AtomicU64::new(0),
            waiting_requests: AtomicU64::new(0),
            running_requests: AtomicU64::new(0),
        }
    }

    /// Record an HTTP request.
    pub fn record_request(&self, method: &str, path: &str, status: u16, duration_secs: f64) {
        let key = RequestKey {
            method: method.to_string(),
            path: normalize_path(path),
            status,
        };

        // Increment counter
        {
            let mut requests = write_lock(&self.http_requests);
            if let Some(entry) = requests.iter_mut().find(|(k, _)| *k == key) {
                entry.1 += 1;
            } else {
                requests.push((key, 1));
            }
        }

        // Record duration
        {
            let mut durations = write_lock(&self.http_durations);
            capped_push(
                &mut durations,
                RequestDuration {
                    method: method.to_string(),
                    path: normalize_path(path),
                    duration_secs,
                },
            );
        }
    }

    /// Record a model load duration.
    pub fn record_model_load(&self, model: &str, duration_secs: f64) {
        let mut durations = write_lock(&self.model_load_durations);
        capped_push(
            &mut durations,
            ModelLoadDuration {
                model: model.to_string(),
                duration_secs,
            },
        );
    }

    /// Record inference tokens.
    pub fn record_tokens(&self, model: &str, token_type: &str, count: u64) {
        let key = TokenKey {
            model: model.to_string(),
            token_type: token_type.to_string(),
        };
        let mut tokens = write_lock(&self.inference_tokens);
        if let Some(entry) = tokens.iter_mut().find(|(k, _)| *k == key) {
            entry.1 += count;
        } else {
            tokens.push((key, count));
        }
    }

    /// Record an inference duration for a model.
    pub fn record_inference_duration(&self, model: &str, duration_secs: f64) {
        let mut durations = write_lock(&self.inference_durations);
        capped_push(
            &mut durations,
            InferenceDuration {
                model: model.to_string(),
                duration_secs,
            },
        );
    }

    /// Record time-to-first-token for a model.
    pub fn record_ttft(&self, model: &str, ttft_secs: f64) {
        let mut entries = write_lock(&self.ttft_entries);
        capped_push(
            &mut entries,
            TtftEntry {
                model: model.to_string(),
                ttft_secs,
            },
        );
    }

    /// Increment the model eviction counter.
    pub fn increment_evictions(&self) {
        self.model_evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Set the estimated memory usage for a model.
    pub fn set_model_memory(&self, model: &str, bytes: u64) {
        let mut mem = write_lock(&self.model_memory_bytes);
        if let Some(entry) = mem.iter_mut().find(|(m, _)| m == model) {
            entry.1 = bytes;
        } else {
            mem.push((model.to_string(), bytes));
        }
    }

    /// Remove memory tracking for an unloaded model.
    pub fn remove_model_memory(&self, model: &str) {
        let mut mem = write_lock(&self.model_memory_bytes);
        mem.retain(|(m, _)| m != model);
    }

    /// Set GPU memory usage for a device.
    pub fn set_gpu_memory(&self, device: &str, bytes: u64) {
        let mut mem = write_lock(&self.gpu_memory_bytes);
        if let Some(entry) = mem.iter_mut().find(|(d, _)| d == device) {
            entry.1 = bytes;
        } else {
            mem.push((device.to_string(), bytes));
        }
    }

    /// Set GPU utilization for a device (0.0 - 1.0).
    pub fn set_gpu_utilization(&self, device: &str, utilization: f64) {
        let mut util = write_lock(&self.gpu_utilization);
        if let Some(entry) = util.iter_mut().find(|(d, _)| d == device) {
            entry.1 = utilization;
        } else {
            util.push((device.to_string(), utilization));
        }
    }

    /// Increment the TEE attestation report counter.
    pub fn increment_tee_attestation(&self) {
        self.tee_attestations.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the encrypted model decryption counter.
    pub fn increment_tee_model_decryption(&self) {
        self.tee_model_decryptions.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the privacy redaction call counter.
    pub fn increment_tee_redaction(&self) {
        self.tee_redactions.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the authentication failure counter.
    pub fn increment_auth_failure(&self) {
        self.auth_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the active requests gauge (call at request start).
    pub fn increment_active_requests(&self) {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the active requests gauge (call at request end).
    pub fn decrement_active_requests(&self) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
    }

    /// Return the current number of active inference requests.
    pub fn active_requests(&self) -> u64 {
        self.active_requests.load(Ordering::Relaxed)
    }

    /// Increment the waiting (queued for admission) requests gauge.
    pub fn increment_waiting_requests(&self) {
        self.waiting_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the waiting requests gauge.
    pub fn decrement_waiting_requests(&self) {
        self.waiting_requests.fetch_sub(1, Ordering::Relaxed);
    }

    /// Return the current number of requests queued for admission.
    pub fn waiting_requests(&self) -> u64 {
        self.waiting_requests.load(Ordering::Relaxed)
    }

    /// Increment the running (admitted, in-flight) requests gauge.
    pub fn increment_running_requests(&self) {
        self.running_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the running requests gauge.
    pub fn decrement_running_requests(&self) {
        self.running_requests.fetch_sub(1, Ordering::Relaxed);
    }

    /// Return the current number of admitted, in-flight requests.
    pub fn running_requests(&self) -> u64 {
        self.running_requests.load(Ordering::Relaxed)
    }

    /// Render all metrics in Prometheus text exposition format.
    pub fn render(&self) -> String {
        let mut output = String::new();

        // power_http_requests_total
        output.push_str("# HELP power_http_requests_total Total number of HTTP requests.\n");
        output.push_str("# TYPE power_http_requests_total counter\n");
        {
            let requests = read_lock(&self.http_requests);
            for (key, count) in requests.iter() {
                output.push_str(&format!(
                    "power_http_requests_total{{method=\"{}\",path=\"{}\",status=\"{}\"}} {}\n",
                    key.method, key.path, key.status, count
                ));
            }
        }

        // power_http_request_duration_seconds
        output.push_str(
            "# HELP power_http_request_duration_seconds HTTP request duration in seconds.\n",
        );
        output.push_str("# TYPE power_http_request_duration_seconds summary\n");
        {
            let durations = read_lock(&self.http_durations);
            // Aggregate by method+path: count and sum
            let mut aggregated: Vec<(String, String, u64, f64)> = Vec::new();
            for d in durations.iter() {
                if let Some(entry) = aggregated
                    .iter_mut()
                    .find(|(m, p, _, _)| *m == d.method && *p == d.path)
                {
                    entry.2 += 1;
                    entry.3 += d.duration_secs;
                } else {
                    aggregated.push((d.method.clone(), d.path.clone(), 1, d.duration_secs));
                }
            }
            for (method, path, count, sum) in &aggregated {
                output.push_str(&format!(
                    "power_http_request_duration_seconds_count{{method=\"{}\",path=\"{}\"}} {}\n",
                    method, path, count
                ));
                output.push_str(&format!(
                    "power_http_request_duration_seconds_sum{{method=\"{}\",path=\"{}\"}} {:.6}\n",
                    method, path, sum
                ));
            }
        }

        // power_model_load_duration_seconds
        output
            .push_str("# HELP power_model_load_duration_seconds Model load duration in seconds.\n");
        output.push_str("# TYPE power_model_load_duration_seconds summary\n");
        {
            let durations = read_lock(&self.model_load_durations);
            let mut aggregated: Vec<(String, u64, f64)> = Vec::new();
            for d in durations.iter() {
                if let Some(entry) = aggregated.iter_mut().find(|(m, _, _)| *m == d.model) {
                    entry.1 += 1;
                    entry.2 += d.duration_secs;
                } else {
                    aggregated.push((d.model.clone(), 1, d.duration_secs));
                }
            }
            for (model, count, sum) in &aggregated {
                output.push_str(&format!(
                    "power_model_load_duration_seconds_count{{model=\"{}\"}} {}\n",
                    model, count
                ));
                output.push_str(&format!(
                    "power_model_load_duration_seconds_sum{{model=\"{}\"}} {:.6}\n",
                    model, sum
                ));
            }
        }

        // power_models_loaded
        output.push_str("# HELP power_models_loaded Number of models currently loaded.\n");
        output.push_str("# TYPE power_models_loaded gauge\n");
        output.push_str(&format!(
            "power_models_loaded {}\n",
            self.models_loaded.load(Ordering::Relaxed)
        ));

        // power_inference_tokens_total
        output.push_str("# HELP power_inference_tokens_total Total inference tokens processed.\n");
        output.push_str("# TYPE power_inference_tokens_total counter\n");
        {
            let tokens = read_lock(&self.inference_tokens);
            for (key, count) in tokens.iter() {
                output.push_str(&format!(
                    "power_inference_tokens_total{{model=\"{}\",type=\"{}\"}} {}\n",
                    key.model, key.token_type, count
                ));
            }
        }

        // power_inference_duration_seconds
        output.push_str("# HELP power_inference_duration_seconds Inference duration in seconds.\n");
        output.push_str("# TYPE power_inference_duration_seconds summary\n");
        {
            let durations = read_lock(&self.inference_durations);
            let mut aggregated: Vec<(String, u64, f64)> = Vec::new();
            for d in durations.iter() {
                if let Some(entry) = aggregated.iter_mut().find(|(m, _, _)| *m == d.model) {
                    entry.1 += 1;
                    entry.2 += d.duration_secs;
                } else {
                    aggregated.push((d.model.clone(), 1, d.duration_secs));
                }
            }
            for (model, count, sum) in &aggregated {
                output.push_str(&format!(
                    "power_inference_duration_seconds_count{{model=\"{}\"}} {}\n",
                    model, count
                ));
                output.push_str(&format!(
                    "power_inference_duration_seconds_sum{{model=\"{}\"}} {:.6}\n",
                    model, sum
                ));
            }
        }

        // power_ttft_seconds
        output.push_str("# HELP power_ttft_seconds Time to first token in seconds.\n");
        output.push_str("# TYPE power_ttft_seconds summary\n");
        {
            let entries = read_lock(&self.ttft_entries);
            let mut aggregated: Vec<(String, u64, f64)> = Vec::new();
            for e in entries.iter() {
                if let Some(entry) = aggregated.iter_mut().find(|(m, _, _)| *m == e.model) {
                    entry.1 += 1;
                    entry.2 += e.ttft_secs;
                } else {
                    aggregated.push((e.model.clone(), 1, e.ttft_secs));
                }
            }
            for (model, count, sum) in &aggregated {
                output.push_str(&format!(
                    "power_ttft_seconds_count{{model=\"{}\"}} {}\n",
                    model, count
                ));
                output.push_str(&format!(
                    "power_ttft_seconds_sum{{model=\"{}\"}} {:.6}\n",
                    model, sum
                ));
            }
        }

        // power_model_evictions_total
        output.push_str("# HELP power_model_evictions_total Total number of model evictions.\n");
        output.push_str("# TYPE power_model_evictions_total counter\n");
        output.push_str(&format!(
            "power_model_evictions_total {}\n",
            self.model_evictions.load(Ordering::Relaxed)
        ));

        // power_model_memory_bytes
        output
            .push_str("# HELP power_model_memory_bytes Estimated memory usage per loaded model.\n");
        output.push_str("# TYPE power_model_memory_bytes gauge\n");
        {
            let mem = read_lock(&self.model_memory_bytes);
            for (model, bytes) in mem.iter() {
                output.push_str(&format!(
                    "power_model_memory_bytes{{model=\"{}\"}} {}\n",
                    model, bytes
                ));
            }
        }

        // power_gpu_memory_bytes
        output.push_str("# HELP power_gpu_memory_bytes GPU memory usage in bytes.\n");
        output.push_str("# TYPE power_gpu_memory_bytes gauge\n");
        {
            let mem = read_lock(&self.gpu_memory_bytes);
            for (device, bytes) in mem.iter() {
                output.push_str(&format!(
                    "power_gpu_memory_bytes{{device=\"{}\"}} {}\n",
                    device, bytes
                ));
            }
        }

        // power_gpu_utilization
        output.push_str("# HELP power_gpu_utilization GPU compute utilization (0.0-1.0).\n");
        output.push_str("# TYPE power_gpu_utilization gauge\n");
        {
            let util = read_lock(&self.gpu_utilization);
            for (device, pct) in util.iter() {
                output.push_str(&format!(
                    "power_gpu_utilization{{device=\"{}\"}} {:.6}\n",
                    device, pct
                ));
            }
        }

        // power_tee_attestations_total
        output.push_str(
            "# HELP power_tee_attestations_total Total TEE attestation reports generated.\n",
        );
        output.push_str("# TYPE power_tee_attestations_total counter\n");
        output.push_str(&format!(
            "power_tee_attestations_total {}\n",
            self.tee_attestations.load(Ordering::Relaxed)
        ));

        // power_tee_model_decryptions_total
        output.push_str(
            "# HELP power_tee_model_decryptions_total Total encrypted model decryptions.\n",
        );
        output.push_str("# TYPE power_tee_model_decryptions_total counter\n");
        output.push_str(&format!(
            "power_tee_model_decryptions_total {}\n",
            self.tee_model_decryptions.load(Ordering::Relaxed)
        ));

        // power_tee_redactions_total
        output.push_str("# HELP power_tee_redactions_total Total privacy redaction calls.\n");
        output.push_str("# TYPE power_tee_redactions_total counter\n");
        output.push_str(&format!(
            "power_tee_redactions_total {}\n",
            self.tee_redactions.load(Ordering::Relaxed)
        ));

        // power_auth_failures_total
        output.push_str("# HELP power_auth_failures_total Total authentication failures.\n");
        output.push_str("# TYPE power_auth_failures_total counter\n");
        output.push_str(&format!(
            "power_auth_failures_total {}\n",
            self.auth_failures.load(Ordering::Relaxed)
        ));

        // power_active_requests
        output.push_str(
            "# HELP power_active_requests Number of currently active inference requests.\n",
        );
        output.push_str("# TYPE power_active_requests gauge\n");
        output.push_str(&format!(
            "power_active_requests {}\n",
            self.active_requests.load(Ordering::Relaxed)
        ));

        // power_requests_waiting
        output.push_str(
            "# HELP power_requests_waiting Number of requests queued waiting for an admission permit.\n",
        );
        output.push_str("# TYPE power_requests_waiting gauge\n");
        output.push_str(&format!(
            "power_requests_waiting {}\n",
            self.waiting_requests.load(Ordering::Relaxed)
        ));

        // power_requests_running
        output.push_str(
            "# HELP power_requests_running Number of admitted requests currently running.\n",
        );
        output.push_str("# TYPE power_requests_running gauge\n");
        output.push_str(&format!(
            "power_requests_running {}\n",
            self.running_requests.load(Ordering::Relaxed)
        ));

        output
    }

    /// Render process metrics plus backend-owned prompt-cache counters.
    pub fn render_with_prompt_cache(
        &self,
        prompt_cache: &[(&str, PromptCacheMetricsSnapshot)],
    ) -> String {
        let mut output = self.render();
        append_prompt_cache_metrics(&mut output, prompt_cache);
        output
    }
}

fn append_prompt_cache_metrics(
    output: &mut String,
    snapshots: &[(&str, PromptCacheMetricsSnapshot)],
) {
    let definitions = [
        (
            "power_prompt_cache_requests_total",
            "Total requests carrying prompt_cache_key.",
            "counter",
        ),
        (
            "power_prompt_cache_hits_total",
            "Prompt-cache lookups that reused at least one token.",
            "counter",
        ),
        (
            "power_prompt_cache_misses_total",
            "Prompt-cache lookups that reused no tokens.",
            "counter",
        ),
        (
            "power_prompt_cache_reused_tokens_total",
            "Prompt tokens skipped through verified KV-prefix reuse.",
            "counter",
        ),
        (
            "power_prompt_cache_evaluated_tokens_total",
            "Prompt tokens evaluated for keyed prompt-cache requests.",
            "counter",
        ),
        (
            "power_prompt_cache_evictions_total",
            "Prompt-cache entries evicted by TTL or capacity.",
            "counter",
        ),
        (
            "power_prompt_cache_entries",
            "Resident reusable prompt-cache contexts.",
            "gauge",
        ),
    ];
    for (name, help, metric_type) in definitions {
        output.push_str(&format!("# HELP {name} {help}\n"));
        output.push_str(&format!("# TYPE {name} {metric_type}\n"));
        for (backend, snapshot) in snapshots {
            let value = match name {
                "power_prompt_cache_requests_total" => snapshot.requests,
                "power_prompt_cache_hits_total" => snapshot.hits,
                "power_prompt_cache_misses_total" => snapshot.misses,
                "power_prompt_cache_reused_tokens_total" => snapshot.reused_tokens,
                "power_prompt_cache_evaluated_tokens_total" => snapshot.evaluated_tokens,
                "power_prompt_cache_evictions_total" => snapshot.evictions,
                "power_prompt_cache_entries" => snapshot.entries,
                _ => continue,
            };
            output.push_str(&format!(
                "{name}{{backend=\"{}\"}} {value}\n",
                prometheus_label(backend)
            ));
        }
    }
}

fn append_speculative_metrics(output: &mut String, snapshots: &[SpeculativeMetricsSnapshot]) {
    let definitions = [
        (
            "power_speculative_requests_total",
            "Completed speculative inference requests.",
        ),
        (
            "power_speculative_rounds_total",
            "Speculative verification rounds.",
        ),
        (
            "power_speculative_target_passes_total",
            "Target-model verification passes.",
        ),
        (
            "power_speculative_drafted_tokens_total",
            "Tokens proposed by model-backed drafters.",
        ),
        (
            "power_speculative_accepted_tokens_total",
            "Draft tokens accepted by exact target verification.",
        ),
        (
            "power_speculative_emitted_tokens_total",
            "Output tokens emitted by completed speculative requests.",
        ),
        (
            "power_speculative_decode_seconds_total",
            "Cumulative decode time for completed speculative requests.",
        ),
    ];
    for (name, help) in definitions {
        output.push_str(&format!("# HELP {name} {help}\n"));
        output.push_str(&format!("# TYPE {name} counter\n"));
        for snapshot in snapshots {
            let labels = format!(
                "backend=\"{}\",model=\"{}\",strategy=\"{}\"",
                prometheus_label(&snapshot.backend),
                prometheus_label(&snapshot.model),
                prometheus_label(&snapshot.strategy)
            );
            match name {
                "power_speculative_requests_total" => {
                    output.push_str(&format!("{name}{{{labels}}} {}\n", snapshot.requests));
                }
                "power_speculative_rounds_total" => {
                    output.push_str(&format!("{name}{{{labels}}} {}\n", snapshot.rounds));
                }
                "power_speculative_target_passes_total" => {
                    output.push_str(&format!("{name}{{{labels}}} {}\n", snapshot.target_passes));
                }
                "power_speculative_drafted_tokens_total" => {
                    output.push_str(&format!("{name}{{{labels}}} {}\n", snapshot.drafted_tokens));
                }
                "power_speculative_accepted_tokens_total" => {
                    output.push_str(&format!(
                        "{name}{{{labels}}} {}\n",
                        snapshot.accepted_tokens
                    ));
                }
                "power_speculative_emitted_tokens_total" => {
                    output.push_str(&format!("{name}{{{labels}}} {}\n", snapshot.emitted_tokens));
                }
                "power_speculative_decode_seconds_total" => {
                    output.push_str(&format!(
                        "{name}{{{labels}}} {:.9}\n",
                        snapshot.decode_duration_ns as f64 / 1_000_000_000.0
                    ));
                }
                _ => {}
            }
        }
    }
}

fn prometheus_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Normalize a request path for metric labels (strip IDs, query strings).
fn normalize_path(path: &str) -> String {
    path.split('?').next().unwrap_or(path).to_string()
}

/// GET /metrics - Prometheus metrics endpoint.
pub async fn handler(State(state): State<AppState>) -> impl IntoResponse {
    // Update models_loaded gauge from state
    state
        .metrics
        .models_loaded
        .store(state.loaded_model_count() as u64, Ordering::Relaxed);

    let prompt_cache = state.backends.prompt_cache_metrics();
    let speculative = state.backends.speculative_metrics();
    let mut body = state.metrics.render_with_prompt_cache(&prompt_cache);
    append_speculative_metrics(&mut body, &speculative);
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

/// Axum middleware that records request metrics.
pub async fn middleware(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let start = Instant::now();

    let response = next.run(request).await;

    let duration = start.elapsed().as_secs_f64();
    let status = response.status().as_u16();

    state
        .metrics
        .record_request(&method, &path, status, duration);

    response
}

#[cfg(test)]
mod tests;
