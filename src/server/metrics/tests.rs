use super::*;

#[test]
fn test_metrics_new() {
    let metrics = Metrics::new();
    assert_eq!(metrics.models_loaded.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.model_evictions.load(Ordering::Relaxed), 0);
}

#[test]
fn test_render_prompt_cache_metrics() {
    let metrics = Metrics::new();
    let output = metrics.render_with_prompt_cache(&[(
        "llama.cpp",
        PromptCacheMetricsSnapshot {
            requests: 3,
            hits: 2,
            misses: 1,
            reused_tokens: 1024,
            evaluated_tokens: 128,
            evictions: 1,
            entries: 1,
        },
    )]);
    assert!(output.contains("power_prompt_cache_hits_total{backend=\"llama.cpp\"} 2"));
    assert!(output.contains("power_prompt_cache_reused_tokens_total{backend=\"llama.cpp\"} 1024"));
    assert!(output.contains("power_prompt_cache_entries{backend=\"llama.cpp\"} 1"));
}

#[test]
fn test_render_speculative_metrics() {
    let mut output = String::new();
    append_speculative_metrics(
        &mut output,
        &[SpeculativeMetricsSnapshot {
            backend: "llama.cpp".to_string(),
            model: "qwen\"target".to_string(),
            strategy: "dspark".to_string(),
            requests: 3,
            rounds: 90,
            target_passes: 90,
            drafted_tokens: 360,
            accepted_tokens: 348,
            emitted_tokens: 256,
            decode_duration_ns: 5_000_000_000,
        }],
    );
    assert!(output.contains(
        "power_speculative_accepted_tokens_total{backend=\"llama.cpp\",model=\"qwen\\\"target\",strategy=\"dspark\"} 348"
    ));
    assert!(output.contains("power_speculative_decode_seconds_total"));
    assert!(output.contains("} 5.000000000"));
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
    assert!(output
        .contains("power_http_requests_total{method=\"GET\",path=\"/health\",status=\"200\"} 1"));
    assert!(output.contains("power_models_loaded 2"));
    assert!(output.contains("power_inference_tokens_total{model=\"llama3\",type=\"prompt\"} 100"));
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
fn test_render_includes_all_metric_sections() {
    let metrics = Metrics::new();
    let output = metrics.render();

    // All new metric sections should have HELP/TYPE headers even when empty
    assert!(output.contains("# HELP power_inference_duration_seconds"));
    assert!(output.contains("# TYPE power_inference_duration_seconds summary"));
    assert!(output.contains("# HELP power_ttft_seconds"));
    assert!(output.contains("# TYPE power_ttft_seconds summary"));
    assert!(output.contains("# HELP power_model_evictions_total"));
    assert!(output.contains("# TYPE power_model_evictions_total counter"));
    assert!(output.contains("# HELP power_model_memory_bytes"));
    assert!(output.contains("# TYPE power_model_memory_bytes gauge"));
    assert!(output.contains("# HELP power_gpu_memory_bytes"));
    assert!(output.contains("# TYPE power_gpu_memory_bytes gauge"));
    assert!(output.contains("# HELP power_gpu_utilization"));
    assert!(output.contains("# TYPE power_gpu_utilization gauge"));
}

// --- TEE metrics tests ---

#[test]
fn test_tee_counters_start_at_zero() {
    let metrics = Metrics::new();
    assert_eq!(metrics.tee_attestations.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.tee_model_decryptions.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.tee_redactions.load(Ordering::Relaxed), 0);
}

#[test]
fn test_increment_tee_attestation() {
    let metrics = Metrics::new();
    metrics.increment_tee_attestation();
    metrics.increment_tee_attestation();
    assert_eq!(metrics.tee_attestations.load(Ordering::Relaxed), 2);

    let output = metrics.render();
    assert!(output.contains("power_tee_attestations_total 2"));
}

#[test]
fn test_increment_tee_model_decryption() {
    let metrics = Metrics::new();
    metrics.increment_tee_model_decryption();
    assert_eq!(metrics.tee_model_decryptions.load(Ordering::Relaxed), 1);

    let output = metrics.render();
    assert!(output.contains("power_tee_model_decryptions_total 1"));
}

#[test]
fn test_increment_tee_redaction() {
    let metrics = Metrics::new();
    metrics.increment_tee_redaction();
    metrics.increment_tee_redaction();
    metrics.increment_tee_redaction();
    assert_eq!(metrics.tee_redactions.load(Ordering::Relaxed), 3);

    let output = metrics.render();
    assert!(output.contains("power_tee_redactions_total 3"));
}

#[test]
fn test_render_includes_tee_metric_sections() {
    let metrics = Metrics::new();
    let output = metrics.render();

    assert!(output.contains("# HELP power_tee_attestations_total"));
    assert!(output.contains("# TYPE power_tee_attestations_total counter"));
    assert!(output.contains("power_tee_attestations_total 0"));
    assert!(output.contains("# HELP power_tee_model_decryptions_total"));
    assert!(output.contains("# TYPE power_tee_model_decryptions_total counter"));
    assert!(output.contains("power_tee_model_decryptions_total 0"));
    assert!(output.contains("# HELP power_tee_redactions_total"));
    assert!(output.contains("# TYPE power_tee_redactions_total counter"));
    assert!(output.contains("power_tee_redactions_total 0"));
}

// --- Auth & request isolation metrics tests ---

#[test]
fn test_auth_failures_start_at_zero() {
    let metrics = Metrics::new();
    assert_eq!(metrics.auth_failures.load(Ordering::Relaxed), 0);
}

#[test]
fn test_increment_auth_failure() {
    let metrics = Metrics::new();
    metrics.increment_auth_failure();
    metrics.increment_auth_failure();
    assert_eq!(metrics.auth_failures.load(Ordering::Relaxed), 2);

    let output = metrics.render();
    assert!(output.contains("power_auth_failures_total 2"));
}

#[test]
fn test_active_requests_gauge() {
    let metrics = Metrics::new();
    assert_eq!(metrics.active_requests.load(Ordering::Relaxed), 0);

    metrics.increment_active_requests();
    metrics.increment_active_requests();
    assert_eq!(metrics.active_requests.load(Ordering::Relaxed), 2);

    metrics.decrement_active_requests();
    assert_eq!(metrics.active_requests.load(Ordering::Relaxed), 1);

    let output = metrics.render();
    assert!(output.contains("power_active_requests 1"));
}

#[test]
fn test_render_includes_auth_and_request_metrics() {
    let metrics = Metrics::new();
    let output = metrics.render();

    assert!(output.contains("# HELP power_auth_failures_total"));
    assert!(output.contains("# TYPE power_auth_failures_total counter"));
    assert!(output.contains("power_auth_failures_total 0"));
    assert!(output.contains("# HELP power_active_requests"));
    assert!(output.contains("# TYPE power_active_requests gauge"));
    assert!(output.contains("power_active_requests 0"));
}
