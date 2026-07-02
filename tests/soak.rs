//! Soak / endurance tests for streaming inference and admission control.
//!
//! These run sustained / high-churn workloads and assert *stability*: no panic,
//! every request terminates, the concurrency bound is never violated, and the
//! metrics gauges return to exactly zero (catching permit / gauge / KV leaks).
//!
//! Iteration counts default to CI-safe values; scale them for a real soak with
//! `A3S_SOAK_ITERS=100000 cargo test --features picolm --test soak -- --nocapture`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use a3s_power::server::limiter::ConcurrencyLimiter;
use a3s_power::server::metrics::Metrics;

fn soak_iters(default: usize) -> usize {
    std::env::var("A3S_SOAK_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Admission-control endurance: many short-lived requests over a small permit
/// pool. The concurrency bound must hold on every cycle and both gauges must
/// return to exactly zero — a permit or gauge leak would surface here.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn soak_admission_control_churn() {
    const LIMIT: u64 = 4;
    let total = soak_iters(1000);
    let m = Arc::new(Metrics::new());
    let limiter = Arc::new(ConcurrencyLimiter::new(LIMIT, m.clone()));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let max_seen = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::with_capacity(total);
    for _ in 0..total {
        let limiter = limiter.clone();
        let in_flight = in_flight.clone();
        let max_seen = max_seen.clone();
        handles.push(tokio::spawn(async move {
            let _permit = limiter.acquire().await;
            let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            max_seen.fetch_max(now, Ordering::SeqCst);
            tokio::task::yield_now().await; // brief hold to create real contention
            in_flight.fetch_sub(1, Ordering::SeqCst);
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    assert!(
        max_seen.load(Ordering::SeqCst) <= LIMIT as usize,
        "concurrency bound violated over {total} cycles: {} > {}",
        max_seen.load(Ordering::SeqCst),
        LIMIT
    );
    assert_eq!(
        in_flight.load(Ordering::SeqCst),
        0,
        "in-flight counter leaked"
    );
    assert_eq!(
        m.running_requests(),
        0,
        "running gauge leaked after {total} cycles"
    );
    assert_eq!(
        m.waiting_requests(),
        0,
        "waiting gauge leaked after {total} cycles"
    );
}

// ── picolm streaming endurance (real inference against a tiny synthetic GGUF) ──

#[cfg(feature = "picolm")]
mod gguf_builder;

#[cfg(feature = "picolm")]
mod picolm_soak {
    use super::soak_iters;
    use std::sync::Arc;
    use std::time::Duration;

    use a3s_power::backend::picolm::PicolmBackend;
    use a3s_power::backend::types::{ChatMessage, ChatRequest, MessageContent};
    use a3s_power::backend::Backend;
    use a3s_power::config::PowerConfig;
    use a3s_power::model::manifest::{ModelFormat, ModelManifest};
    use futures::StreamExt;

    fn req(prompt: &str) -> ChatRequest {
        ChatRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text(prompt.to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            }],
            session_id: None,
            temperature: Some(0.7),
            top_p: Some(0.95),
            max_tokens: Some(12),
            stop: None,
            stream: true,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: Some(0), // vary effective output via prompt, not seed
            num_ctx: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            repeat_last_n: None,
            penalize_newline: None,
            num_batch: None,
            num_thread: None,
            num_thread_batch: None,
            flash_attention: None,
            num_gpu: None,
            main_gpu: None,
            use_mmap: None,
            use_mlock: None,
            num_parallel: None,
            images: None,
        }
    }

    /// Run many streaming generations back-to-back. Every one must terminate
    /// (no hang) and produce a single done chunk — over a long run this exercises
    /// the per-request KV cache / working-buffer lifecycle for leaks and torn state.
    #[tokio::test]
    async fn soak_picolm_streaming_endurance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("soak.gguf");
        super::gguf_builder::build_tiny_gguf(
            &path,
            &super::gguf_builder::TinyModelConfig::default(),
        );

        let backend = PicolmBackend::new(Arc::new(PowerConfig::default()));
        let manifest = ModelManifest {
            name: "soak".to_string(),
            format: ModelFormat::Gguf,
            size: std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
            sha256: "sha256:test".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path,
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: vec![],
            family: Some("qwen2".to_string()),
            families: None,
        };
        backend.load(&manifest).await.unwrap();

        let n = soak_iters(60);
        for i in 0..n {
            let mut stream = backend.chat("soak", req("iteration")).await.unwrap();
            let mut done = false;
            let drain = async {
                while let Some(c) = stream.next().await {
                    if c.expect("chunk must not error").done {
                        done = true;
                        break;
                    }
                }
            };
            tokio::time::timeout(Duration::from_secs(10), drain)
                .await
                .unwrap_or_else(|_| panic!("iteration {i} hung"));
            assert!(done, "iteration {i} must terminate with a done chunk");
        }
    }
}
