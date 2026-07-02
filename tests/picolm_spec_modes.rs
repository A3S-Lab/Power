//! Integration test: speculative-decoding modes are lossless.
//!
//! With greedy decoding (temperature 0) the emitted sequence is the argmax
//! sequence regardless of speculation: accepted drafts, the correction token
//! and the bonus token are all the model's argmax at their position. So the
//! `off`, `prompt-lookup` and `ngram-context` modes MUST produce byte-identical
//! output for the same seed. This exercises the batched layer-streaming verify,
//! lossless acceptance, KV-cache rollback and correction-forward end-to-end
//! against a real forward pass.

#![cfg(feature = "picolm")]

use std::path::PathBuf;
use std::sync::Arc;

use a3s_power::backend::picolm::PicolmBackend;
use a3s_power::backend::types::{ChatMessage, ChatRequest, MessageContent};
use a3s_power::backend::Backend;
use a3s_power::config::PowerConfig;
use a3s_power::model::manifest::{ModelFormat, ModelManifest};
use futures::StreamExt;
use tempfile::TempDir;

mod gguf_builder;

fn fake_manifest(name: &str, path: PathBuf) -> ModelManifest {
    ModelManifest {
        name: name.to_string(),
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

fn greedy_req() -> ChatRequest {
    ChatRequest {
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text("a".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }],
        session_id: None,
        temperature: Some(0.0),
        top_p: None,
        // Long enough that the tiny vocab repeats and drafts get proposed,
        // accepted, and rolled back — exercising the whole spec path.
        max_tokens: Some(32),
        stop: None,
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

async fn run_with_mode(model_path: &std::path::Path, spec_mode: &str) -> String {
    let cfg = PowerConfig {
        spec_mode: spec_mode.to_string(),
        ..PowerConfig::default()
    };
    let backend = PicolmBackend::new(Arc::new(cfg));
    let manifest = fake_manifest("spec-model", model_path.to_path_buf());
    backend.load(&manifest).await.unwrap();

    let mut stream = backend.chat("spec-model", greedy_req()).await.unwrap();
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        let c = chunk.unwrap();
        text.push_str(&c.content);
        if c.done {
            break;
        }
    }
    backend.unload("spec-model").await.unwrap();
    text
}

/// `off` / `prompt-lookup` / `ngram-context` must be byte-identical under greedy
/// decoding — the lossless guarantee of speculative decoding.
#[tokio::test]
async fn test_spec_modes_are_lossless() {
    let dir = TempDir::new().unwrap();
    let model_path = dir.path().join("spec.gguf");
    // Slightly larger than the default tiny model so generation has structure.
    gguf_builder::build_tiny_gguf(
        &model_path,
        &gguf_builder::TinyModelConfig {
            n_embd: 16,
            n_heads: 2,
            n_kv_heads: 2,
            n_ff: 32,
            n_layers: 2,
            vocab_size: 16,
        },
    );

    let baseline = run_with_mode(&model_path, "off").await;
    let prompt_lookup = run_with_mode(&model_path, "prompt-lookup").await;
    let ngram_context = run_with_mode(&model_path, "ngram-context").await;

    assert_eq!(
        baseline, prompt_lookup,
        "prompt-lookup speculation must not change greedy output"
    );
    assert_eq!(
        baseline, ngram_context,
        "ngram-context speculation must not change greedy output"
    );
}
