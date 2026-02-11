use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::server::state::AppState;

/// Maximum number of duration/usage samples to keep per vector.
/// Prevents unbounded memory growth on long-running servers.
const MAX_SAMPLES: usize = 10_000;

/// Read from a potentially poisoned RwLock, recovering from poison.
fn read_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| {
        tracing::warn!("Metrics RwLock was poisoned, recovering read guard");
        poisoned.into_inner()
    })
}

/// Write to a potentially poisoned RwLock, recovering from poison.
fn write_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|poisoned| {
        tracing::warn!("Metrics RwLock was poisoned, recovering write guard");
        poisoned.into_inner()
    })
}

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

/// A single inference usage record for cost tracking and the /v1/usage endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub duration_secs: f64,
    pub cost_dollars: f64,
}

/// Prometheus metrics collector with Phase 6 observability support.
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

    // --- Phase 6: Cost tracking ---
    /// Per-call usage records for the /v1/usage endpoint.
    usage_records: RwLock<Vec<UsageRecord>>,
    /// Cumulative cost by model (for Prometheus counter).
    cost_dollars: RwLock<Vec<(String, f64)>>,

    // --- Phase 6: Model lifecycle ---
    /// Total number of model evictions.
    model_evictions: AtomicU64,
    /// Per-model estimated memory usage in bytes.
    model_memory_bytes: RwLock<Vec<(String, u64)>>,

    // --- Phase 6: GPU metrics ---
    /// Per-device GPU memory usage in bytes.
    gpu_memory_bytes: RwLock<Vec<(String, u64)>>,
    /// Per-device GPU utilization (0.0 - 1.0).
    gpu_utilization: RwLock<Vec<(String, f64)>>,
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
            usage_records: RwLock::new(Vec::new()),
            cost_dollars: RwLock::new(Vec::new()),
            model_evictions: AtomicU64::new(0),
            model_memory_bytes: RwLock::new(Vec::new()),
            gpu_memory_bytes: RwLock::new(Vec::new()),
            gpu_utilization: RwLock::new(Vec::new()),
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

    /// Record a complete inference usage record for the /v1/usage endpoint.
    pub fn record_usage(&self, record: UsageRecord) {
        // Accumulate cost by model
        {
            let mut costs = write_lock(&self.cost_dollars);
            if let Some(entry) = costs.iter_mut().find(|(m, _)| *m == record.model) {
                entry.1 += record.cost_dollars;
            } else {
                costs.push((record.model.clone(), record.cost_dollars));
            }
        }
        // Store the full record
        {
            let mut records = write_lock(&self.usage_records);
            capped_push(&mut records, record);
        }
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

    /// Return a snapshot of all usage records, optionally filtered by date range and model.
    pub fn query_usage(
        &self,
        start: Option<chrono::DateTime<chrono::Utc>>,
        end: Option<chrono::DateTime<chrono::Utc>>,
        model: Option<&str>,
    ) -> Vec<UsageRecord> {
        let records = read_lock(&self.usage_records);
        records
            .iter()
            .filter(|r| {
                if let Some(s) = start {
                    if r.timestamp < s {
                        return false;
                    }
                }
                if let Some(e) = end {
                    if r.timestamp > e {
                        return false;
                    }
                }
                if let Some(m) = model {
                    if r.model != m {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect()
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

        // power_cost_dollars
        output.push_str(
            "# HELP power_cost_dollars Estimated cumulative inference cost in dollars.\n",
        );
        output.push_str("# TYPE power_cost_dollars counter\n");
        {
            let costs = read_lock(&self.cost_dollars);
            for (model, total) in costs.iter() {
                output.push_str(&format!(
                    "power_cost_dollars{{model=\"{}\"}} {:.6}\n",
                    model, total
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

        output
    }
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

    let body = state.metrics.render();
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
mod tests {
    use super::*;

    #[test]
    fn test_metrics_new() {
        let metrics = Metrics::new();
        assert_eq!(metrics.models_loaded.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.model_evictions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_record_request_increments() {
        let metrics = Metrics::new();
        metrics.record_request("GET", "/health", 200, 0.001);
        metrics.record_request("GET", "/health", 200, 0.002);
        metrics.record_request("POST", "/api/chat", 200, 0.5);

        let requests = metrics.http_requests.read().unwrap();
        let health_count = requests
            .iter()
            .find(|(k, _)| k.method == "GET" && k.path == "/health" && k.status == 200)
            .map(|(_, c)| *c)
            .unwrap_or(0);
        assert_eq!(health_count, 2);

        let chat_count = requests
            .iter()
            .find(|(k, _)| k.method == "POST" && k.path == "/api/chat")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        assert_eq!(chat_count, 1);
    }

    #[test]
    fn test_record_tokens() {
        let metrics = Metrics::new();
        metrics.record_tokens("llama3", "prompt", 100);
        metrics.record_tokens("llama3", "completion", 50);
        metrics.record_tokens("llama3", "prompt", 200);

        let tokens = metrics.inference_tokens.read().unwrap();
        let prompt_count = tokens
            .iter()
            .find(|(k, _)| k.model == "llama3" && k.token_type == "prompt")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        assert_eq!(prompt_count, 300);
    }

    #[test]
    fn test_render_prometheus_format() {
        let metrics = Metrics::new();
        metrics.record_request("GET", "/health", 200, 0.001);
        metrics.models_loaded.store(2, Ordering::Relaxed);
        metrics.record_tokens("llama3", "prompt", 100);

        let output = metrics.render();

        assert!(output.contains("# HELP power_http_requests_total"));
        assert!(output.contains("# TYPE power_http_requests_total counter"));
        assert!(output.contains(
            "power_http_requests_total{method=\"GET\",path=\"/health\",status=\"200\"} 1"
        ));
        assert!(output.contains("power_models_loaded 2"));
        assert!(
            output.contains("power_inference_tokens_total{model=\"llama3\",type=\"prompt\"} 100")
        );
    }

    #[test]
    fn test_render_empty_metrics() {
        let metrics = Metrics::new();
        let output = metrics.render();

        assert!(output.contains("# HELP power_http_requests_total"));
        assert!(output.contains("power_models_loaded 0"));
        assert!(output.contains("power_model_evictions_total 0"));
    }

    #[test]
    fn test_record_model_load() {
        let metrics = Metrics::new();
        metrics.record_model_load("llama3", 2.5);
        metrics.record_model_load("llama3", 3.0);

        let output = metrics.render();
        assert!(output.contains("power_model_load_duration_seconds_count{model=\"llama3\"} 2"));
        assert!(output.contains("power_model_load_duration_seconds_sum{model=\"llama3\"} 5.5"));
    }

    #[test]
    fn test_normalize_path_strips_query() {
        assert_eq!(normalize_path("/api/chat?stream=true"), "/api/chat");
        assert_eq!(normalize_path("/health"), "/health");
    }

    // --- Phase 6 tests ---

    #[test]
    fn test_record_inference_duration() {
        let metrics = Metrics::new();
        metrics.record_inference_duration("llama3", 1.5);
        metrics.record_inference_duration("llama3", 2.0);
        metrics.record_inference_duration("qwen", 0.5);

        let output = metrics.render();
        assert!(output.contains("power_inference_duration_seconds_count{model=\"llama3\"} 2"));
        assert!(output.contains("power_inference_duration_seconds_sum{model=\"llama3\"} 3.5"));
        assert!(output.contains("power_inference_duration_seconds_count{model=\"qwen\"} 1"));
    }

    #[test]
    fn test_record_ttft() {
        let metrics = Metrics::new();
        metrics.record_ttft("llama3", 0.05);
        metrics.record_ttft("llama3", 0.08);

        let output = metrics.render();
        assert!(output.contains("power_ttft_seconds_count{model=\"llama3\"} 2"));
        assert!(output.contains("power_ttft_seconds_sum{model=\"llama3\"} 0.13"));
    }

    #[test]
    fn test_record_usage_and_cost() {
        let metrics = Metrics::new();
        let record = UsageRecord {
            timestamp: chrono::Utc::now(),
            model: "llama3".to_string(),
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            duration_secs: 1.5,
            cost_dollars: 0.0,
        };
        metrics.record_usage(record);

        let output = metrics.render();
        assert!(output.contains("power_cost_dollars{model=\"llama3\"} 0.000000"));

        let records = metrics.query_usage(None, None, None);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].model, "llama3");
        assert_eq!(records[0].total_tokens, 150);
    }

    #[test]
    fn test_query_usage_with_model_filter() {
        let metrics = Metrics::new();
        let now = chrono::Utc::now();
        metrics.record_usage(UsageRecord {
            timestamp: now,
            model: "llama3".to_string(),
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            duration_secs: 1.0,
            cost_dollars: 0.0,
        });
        metrics.record_usage(UsageRecord {
            timestamp: now,
            model: "qwen".to_string(),
            prompt_tokens: 200,
            completion_tokens: 100,
            total_tokens: 300,
            duration_secs: 2.0,
            cost_dollars: 0.0,
        });

        let llama_records = metrics.query_usage(None, None, Some("llama3"));
        assert_eq!(llama_records.len(), 1);
        assert_eq!(llama_records[0].model, "llama3");

        let all_records = metrics.query_usage(None, None, None);
        assert_eq!(all_records.len(), 2);
    }

    #[test]
    fn test_query_usage_with_date_filter() {
        let metrics = Metrics::new();
        let early = chrono::Utc::now() - chrono::Duration::hours(2);
        let mid = chrono::Utc::now() - chrono::Duration::hours(1);
        let late = chrono::Utc::now();

        metrics.record_usage(UsageRecord {
            timestamp: early,
            model: "llama3".to_string(),
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            duration_secs: 0.5,
            cost_dollars: 0.0,
        });
        metrics.record_usage(UsageRecord {
            timestamp: late,
            model: "llama3".to_string(),
            prompt_tokens: 20,
            completion_tokens: 10,
            total_tokens: 30,
            duration_secs: 1.0,
            cost_dollars: 0.0,
        });

        // Filter: only records after mid
        let recent = metrics.query_usage(Some(mid), None, None);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].total_tokens, 30);
    }

    #[test]
    fn test_increment_evictions() {
        let metrics = Metrics::new();
        assert_eq!(metrics.model_evictions.load(Ordering::Relaxed), 0);

        metrics.increment_evictions();
        metrics.increment_evictions();
        assert_eq!(metrics.model_evictions.load(Ordering::Relaxed), 2);

        let output = metrics.render();
        assert!(output.contains("power_model_evictions_total 2"));
    }

    #[test]
    fn test_set_model_memory() {
        let metrics = Metrics::new();
        metrics.set_model_memory("llama3", 4_000_000_000);
        metrics.set_model_memory("qwen", 2_000_000_000);

        let output = metrics.render();
        assert!(output.contains("power_model_memory_bytes{model=\"llama3\"} 4000000000"));
        assert!(output.contains("power_model_memory_bytes{model=\"qwen\"} 2000000000"));

        // Update existing
        metrics.set_model_memory("llama3", 5_000_000_000);
        let output = metrics.render();
        assert!(output.contains("power_model_memory_bytes{model=\"llama3\"} 5000000000"));
    }

    #[test]
    fn test_remove_model_memory() {
        let metrics = Metrics::new();
        metrics.set_model_memory("llama3", 4_000_000_000);
        metrics.set_model_memory("qwen", 2_000_000_000);

        metrics.remove_model_memory("llama3");
        let output = metrics.render();
        assert!(!output.contains("model=\"llama3\"} 4000000000"));
        assert!(output.contains("power_model_memory_bytes{model=\"qwen\"} 2000000000"));
    }

    #[test]
    fn test_set_gpu_memory() {
        let metrics = Metrics::new();
        metrics.set_gpu_memory("gpu0", 8_000_000_000);

        let output = metrics.render();
        assert!(output.contains("power_gpu_memory_bytes{device=\"gpu0\"} 8000000000"));

        // Update
        metrics.set_gpu_memory("gpu0", 6_000_000_000);
        let output = metrics.render();
        assert!(output.contains("power_gpu_memory_bytes{device=\"gpu0\"} 6000000000"));
    }

    #[test]
    fn test_set_gpu_utilization() {
        let metrics = Metrics::new();
        metrics.set_gpu_utilization("gpu0", 0.75);

        let output = metrics.render();
        assert!(output.contains("power_gpu_utilization{device=\"gpu0\"} 0.750000"));

        // Update
        metrics.set_gpu_utilization("gpu0", 0.5);
        let output = metrics.render();
        assert!(output.contains("power_gpu_utilization{device=\"gpu0\"} 0.500000"));
    }

    #[test]
    fn test_cost_accumulates_across_records() {
        let metrics = Metrics::new();
        let now = chrono::Utc::now();
        metrics.record_usage(UsageRecord {
            timestamp: now,
            model: "llama3".to_string(),
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            duration_secs: 1.0,
            cost_dollars: 0.01,
        });
        metrics.record_usage(UsageRecord {
            timestamp: now,
            model: "llama3".to_string(),
            prompt_tokens: 200,
            completion_tokens: 100,
            total_tokens: 300,
            duration_secs: 2.0,
            cost_dollars: 0.02,
        });

        let output = metrics.render();
        assert!(output.contains("power_cost_dollars{model=\"llama3\"} 0.030000"));
    }

    #[test]
    fn test_render_includes_all_phase6_sections() {
        let metrics = Metrics::new();
        let output = metrics.render();

        // All new metric sections should have HELP/TYPE headers even when empty
        assert!(output.contains("# HELP power_inference_duration_seconds"));
        assert!(output.contains("# TYPE power_inference_duration_seconds summary"));
        assert!(output.contains("# HELP power_ttft_seconds"));
        assert!(output.contains("# TYPE power_ttft_seconds summary"));
        assert!(output.contains("# HELP power_cost_dollars"));
        assert!(output.contains("# TYPE power_cost_dollars counter"));
        assert!(output.contains("# HELP power_model_evictions_total"));
        assert!(output.contains("# TYPE power_model_evictions_total counter"));
        assert!(output.contains("# HELP power_model_memory_bytes"));
        assert!(output.contains("# TYPE power_model_memory_bytes gauge"));
        assert!(output.contains("# HELP power_gpu_memory_bytes"));
        assert!(output.contains("# TYPE power_gpu_memory_bytes gauge"));
        assert!(output.contains("# HELP power_gpu_utilization"));
        assert!(output.contains("# TYPE power_gpu_utilization gauge"));
    }
}
