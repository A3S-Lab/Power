use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::server::state::AppState;

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

/// Prometheus metrics collector.
pub struct Metrics {
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
            let mut requests = self.http_requests.write().unwrap();
            if let Some(entry) = requests.iter_mut().find(|(k, _)| *k == key) {
                entry.1 += 1;
            } else {
                requests.push((key, 1));
            }
        }

        // Record duration
        {
            let mut durations = self.http_durations.write().unwrap();
            durations.push(RequestDuration {
                method: method.to_string(),
                path: normalize_path(path),
                duration_secs,
            });
        }
    }

    /// Record a model load duration.
    pub fn record_model_load(&self, model: &str, duration_secs: f64) {
        let mut durations = self.model_load_durations.write().unwrap();
        durations.push(ModelLoadDuration {
            model: model.to_string(),
            duration_secs,
        });
    }

    /// Record inference tokens.
    pub fn record_tokens(&self, model: &str, token_type: &str, count: u64) {
        let key = TokenKey {
            model: model.to_string(),
            token_type: token_type.to_string(),
        };
        let mut tokens = self.inference_tokens.write().unwrap();
        if let Some(entry) = tokens.iter_mut().find(|(k, _)| *k == key) {
            entry.1 += count;
        } else {
            tokens.push((key, count));
        }
    }

    /// Render all metrics in Prometheus text exposition format.
    pub fn render(&self) -> String {
        let mut output = String::new();

        // power_http_requests_total
        output.push_str("# HELP power_http_requests_total Total number of HTTP requests.\n");
        output.push_str("# TYPE power_http_requests_total counter\n");
        {
            let requests = self.http_requests.read().unwrap();
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
            let durations = self.http_durations.read().unwrap();
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
            let durations = self.model_load_durations.read().unwrap();
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
            let tokens = self.inference_tokens.read().unwrap();
            for (key, count) in tokens.iter() {
                output.push_str(&format!(
                    "power_inference_tokens_total{{model=\"{}\",type=\"{}\"}} {}\n",
                    key.model, key.token_type, count
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
}
