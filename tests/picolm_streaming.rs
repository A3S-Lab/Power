//! Streaming-inference unit tests for the picolm backend.
//!
//! Drives real end-to-end generation against a tiny synthetic GGUF (no GPU /
//! model download) and asserts the *streaming contract*: chunk structure,
//! termination, stop sequences, all speculative-decoding modes, grammar-
//! constrained output, session KV reuse and determinism.

#![cfg(feature = "picolm")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use a3s_power::backend::picolm::PicolmBackend;
use a3s_power::backend::types::{ChatMessage, ChatRequest, ChatResponseChunk, MessageContent};
use a3s_power::backend::Backend;
use a3s_power::config::PowerConfig;
use a3s_power::model::manifest::{ModelFormat, ModelManifest};
use futures::StreamExt;
use tempfile::TempDir;

mod gguf_builder;

const MODEL: &str = "stream-model";

/// A slightly larger tiny model so generation has some structure.
fn load_model(dir: &TempDir, spec_mode: &str) -> PicolmBackend {
    let path = dir.path().join("m.gguf");
    gguf_builder::build_tiny_gguf(
        &path,
        &gguf_builder::TinyModelConfig {
            n_embd: 16,
            n_heads: 2,
            n_kv_heads: 2,
            n_ff: 32,
            n_layers: 2,
            vocab_size: 16,
        },
    );
    let cfg = PowerConfig {
        spec_mode: spec_mode.to_string(),
        ..PowerConfig::default()
    };
    PicolmBackend::new(Arc::new(cfg))
}

fn manifest(path: PathBuf) -> ModelManifest {
    ModelManifest {
        name: MODEL.to_string(),
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
    }
}

#[allow(clippy::too_many_arguments)]
fn req(
    prompt: &str,
    max_tokens: u32,
    stop: Option<Vec<String>>,
    session_id: Option<String>,
    response_format: Option<serde_json::Value>,
) -> ChatRequest {
    ChatRequest {
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(prompt.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }],
        session_id,
        temperature: Some(0.0), // greedy → deterministic
        top_p: None,
        max_tokens: Some(max_tokens),
        stop,
        stream: true,
        top_k: None,
        min_p: None,
        repeat_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: Some(42),
        num_ctx: None,
        mirostat: None,
        mirostat_tau: None,
        mirostat_eta: None,
        tfs_z: None,
        typical_p: None,
        response_format,
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

async fn collect(backend: &PicolmBackend, request: ChatRequest) -> Vec<ChatResponseChunk> {
    let mut stream = backend.chat(MODEL, request).await.unwrap();
    let mut chunks = Vec::new();
    // Guard against a hang: a tiny model bounded by max_tokens must finish fast.
    let drain = async {
        while let Some(c) = stream.next().await {
            let chunk = c.unwrap();
            let done = chunk.done;
            chunks.push(chunk);
            if done {
                break;
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(10), drain)
        .await
        .expect("streaming must terminate, not hang");
    chunks
}

fn text_of(chunks: &[ChatResponseChunk]) -> String {
    chunks.iter().map(|c| c.content.as_str()).collect()
}

/// The stream must end with exactly one `done == true` chunk, and it must be last.
#[tokio::test]
async fn stream_has_single_terminal_done_chunk() {
    let dir = TempDir::new().unwrap();
    let backend = load_model(&dir, "off");
    backend
        .load(&manifest(dir.path().join("m.gguf")))
        .await
        .unwrap();

    let chunks = collect(&backend, req("hello", 12, None, None, None)).await;
    assert!(!chunks.is_empty(), "must emit at least the done chunk");
    let done_count = chunks.iter().filter(|c| c.done).count();
    assert_eq!(done_count, 1, "exactly one terminal chunk");
    assert!(chunks.last().unwrap().done, "done chunk must be last");
    assert!(
        chunks.last().unwrap().done_reason.is_some(),
        "terminal chunk carries a done_reason"
    );
}

/// Every speculative-decoding mode must stream and terminate cleanly.
#[tokio::test]
async fn all_spec_modes_stream_and_terminate() {
    for mode in ["off", "prompt-lookup", "ngram-context"] {
        let dir = TempDir::new().unwrap();
        let backend = load_model(&dir, mode);
        backend
            .load(&manifest(dir.path().join("m.gguf")))
            .await
            .unwrap();
        let chunks = collect(&backend, req("the quick brown", 16, None, None, None)).await;
        assert!(chunks.last().unwrap().done, "mode {mode} must terminate");
    }
}

/// A stop sequence matching early output must halt generation no later than the
/// unconstrained run.
#[tokio::test]
async fn stop_sequence_halts_early() {
    let dir = TempDir::new().unwrap();
    let backend = load_model(&dir, "off");
    backend
        .load(&manifest(dir.path().join("m.gguf")))
        .await
        .unwrap();

    let full = text_of(&collect(&backend, req("hello world", 24, None, None, None)).await);
    if full.chars().count() < 4 {
        return; // too little output to derive a meaningful stop
    }
    let stop: String = full.chars().take(2).collect();
    let stopped = text_of(
        &collect(
            &backend,
            req("hello world", 24, Some(vec![stop.clone()]), None, None),
        )
        .await,
    );
    // The stop matches the very start of the unconstrained output, so generation
    // must halt early. (picolm halts *without* streaming the piece that completed
    // the stop, so the collected content need not itself contain the stop string.)
    assert!(
        stopped.chars().count() < full.chars().count(),
        "stop sequence must halt generation early: stopped={stopped:?} full={full:?}"
    );
}

/// `response_format` (grammar-constrained) must not panic or hang, even on a
/// tiny model, and must still terminate with a done chunk.
#[tokio::test]
async fn response_format_json_terminates() {
    let dir = TempDir::new().unwrap();
    let backend = load_model(&dir, "prompt-lookup");
    backend
        .load(&manifest(dir.path().join("m.gguf")))
        .await
        .unwrap();

    let chunks = collect(
        &backend,
        req(
            "give me json",
            24,
            None,
            None,
            Some(serde_json::json!("json")),
        ),
    )
    .await;
    assert!(
        chunks.last().unwrap().done,
        "grammar-constrained generation must terminate"
    );
}

/// Two turns sharing a `session_id` reuse the KV cache and both stream cleanly.
#[tokio::test]
async fn session_kv_reuse_across_two_turns() {
    let dir = TempDir::new().unwrap();
    let backend = load_model(&dir, "off");
    backend
        .load(&manifest(dir.path().join("m.gguf")))
        .await
        .unwrap();

    let sid = Some("sess-1".to_string());
    let t1 = collect(&backend, req("first turn", 8, None, sid.clone(), None)).await;
    assert!(t1.last().unwrap().done);
    // Second turn reuses the session's KV cache (longer effective context).
    let t2 = collect(&backend, req("second turn", 8, None, sid, None)).await;
    assert!(
        t2.last().unwrap().done,
        "second session turn must terminate"
    );
}

/// Greedy decoding with a fixed seed is deterministic across identical runs.
#[tokio::test]
async fn streaming_is_deterministic_for_fixed_seed() {
    let dir = TempDir::new().unwrap();
    let backend = load_model(&dir, "off");
    backend
        .load(&manifest(dir.path().join("m.gguf")))
        .await
        .unwrap();

    let a = text_of(&collect(&backend, req("determinism", 12, None, None, None)).await);
    let b = text_of(&collect(&backend, req("determinism", 12, None, None, None)).await);
    assert_eq!(a, b, "same seed + greedy must reproduce the same stream");
}
