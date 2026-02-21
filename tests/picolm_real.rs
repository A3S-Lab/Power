//! Real-model integration test for picolm backend.
//!
//! Loads an actual GGUF model file and runs end-to-end inference.
//! Skipped automatically if the model file is not present.
//!
//! To run: cargo test --features picolm --test picolm_real -- --nocapture

use std::sync::Arc;

use a3s_power::backend::picolm::PicolmBackend;
use a3s_power::backend::types::{ChatMessage, ChatRequest, MessageContent};
use a3s_power::backend::Backend;
use a3s_power::config::PowerConfig;
use a3s_power::model::manifest::{ModelFormat, ModelManifest};
use futures::StreamExt;

/// Path to a real GGUF model for testing.
/// The Qwen 2.5 0.5B Q4_K_M model (~469MB) downloaded previously.
const MODEL_PATH: &str = "/tmp/qwen2.5-0.5b-q4_k_m.gguf";

fn model_available() -> bool {
    std::path::Path::new(MODEL_PATH).exists()
}

fn real_manifest() -> ModelManifest {
    ModelManifest {
        name: "qwen2.5:0.5b-q4_k_m".to_string(),
        format: ModelFormat::Gguf,
        size: std::fs::metadata(MODEL_PATH).map(|m| m.len()).unwrap_or(0),
        sha256: "sha256:test".to_string(),
        parameters: None,
        created_at: chrono::Utc::now(),
        path: MODEL_PATH.into(),
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

fn make_chat_request(prompt: &str, max_tokens: u32, temperature: f32) -> ChatRequest {
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
        temperature: Some(temperature),
        top_p: Some(0.9),
        max_tokens: Some(max_tokens),
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

#[cfg(feature = "picolm")]
#[tokio::test]
async fn test_real_model_load_and_metadata() {
    if !model_available() {
        eprintln!("SKIP: model not found at {MODEL_PATH}");
        return;
    }

    let config = Arc::new(PowerConfig::default());
    let backend = PicolmBackend::new(config);
    let manifest = real_manifest();

    let result = backend.load(&manifest).await;
    assert!(result.is_ok(), "Failed to load real model: {:?}", result);

    eprintln!("Model loaded successfully");

    backend.unload("qwen2.5:0.5b-q4_k_m").await.unwrap();
}

#[cfg(feature = "picolm")]
#[tokio::test]
async fn test_real_model_greedy_generation() {
    if !model_available() {
        eprintln!("SKIP: model not found at {MODEL_PATH}");
        return;
    }

    let config = Arc::new(PowerConfig::default());
    let backend = PicolmBackend::new(config);
    backend.load(&real_manifest()).await.unwrap();

    let req = make_chat_request("What is 2+2?", 32, 0.0);
    let mut stream = backend.chat("qwen2.5:0.5b-q4_k_m", req).await.unwrap();

    let mut full_text = String::new();
    let mut chunk_count = 0;
    let start = std::time::Instant::now();

    while let Some(chunk) = stream.next().await {
        let c = chunk.unwrap();
        if !c.content.is_empty() {
            full_text.push_str(&c.content);
            chunk_count += 1;
        }
        if c.done {
            break;
        }
    }

    let elapsed = start.elapsed();
    let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
        chunk_count as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    eprintln!("--- Greedy generation ---");
    eprintln!("Prompt: \"What is 2+2?\"");
    eprintln!(
        "Output ({chunk_count} tokens, {:.1} tok/s, {:.2}s):",
        tokens_per_sec,
        elapsed.as_secs_f64()
    );
    eprintln!("{full_text}");
    eprintln!("---");

    assert!(
        !full_text.is_empty(),
        "Model should produce non-empty output"
    );
    assert!(chunk_count > 0, "Should produce at least one token");

    backend.unload("qwen2.5:0.5b-q4_k_m").await.unwrap();
}

#[cfg(feature = "picolm")]
#[tokio::test]
async fn test_real_model_sampled_generation() {
    if !model_available() {
        eprintln!("SKIP: model not found at {MODEL_PATH}");
        return;
    }

    let config = Arc::new(PowerConfig::default());
    let backend = PicolmBackend::new(config);
    backend.load(&real_manifest()).await.unwrap();

    let req = make_chat_request("Tell me a one-sentence joke.", 64, 0.8);
    let mut stream = backend.chat("qwen2.5:0.5b-q4_k_m", req).await.unwrap();

    let mut full_text = String::new();
    let mut chunk_count = 0;

    while let Some(chunk) = stream.next().await {
        let c = chunk.unwrap();
        if !c.content.is_empty() {
            full_text.push_str(&c.content);
            chunk_count += 1;
        }
        if c.done {
            break;
        }
    }

    eprintln!("--- Sampled generation (temp=0.8) ---");
    eprintln!("Prompt: \"Tell me a one-sentence joke.\"");
    eprintln!("Output ({chunk_count} tokens):");
    eprintln!("{full_text}");
    eprintln!("---");

    assert!(
        !full_text.is_empty(),
        "Model should produce non-empty output"
    );

    backend.unload("qwen2.5:0.5b-q4_k_m").await.unwrap();
}

#[cfg(feature = "picolm")]
#[tokio::test]
async fn test_real_model_long_generation() {
    if !model_available() {
        eprintln!("SKIP: model not found at {MODEL_PATH}");
        return;
    }

    let config = Arc::new(PowerConfig::default());
    let backend = PicolmBackend::new(config);
    backend.load(&real_manifest()).await.unwrap();

    let req = make_chat_request("Count from 1 to 10.", 128, 0.0);
    let mut stream = backend.chat("qwen2.5:0.5b-q4_k_m", req).await.unwrap();

    let mut full_text = String::new();
    let mut chunk_count = 0;
    let mut done_reason = None;
    let start = std::time::Instant::now();

    while let Some(chunk) = stream.next().await {
        let c = chunk.unwrap();
        if !c.content.is_empty() {
            full_text.push_str(&c.content);
            chunk_count += 1;
        }
        if c.done {
            done_reason = c.done_reason.clone();
            break;
        }
    }

    let elapsed = start.elapsed();
    let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
        chunk_count as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    eprintln!("--- Long generation ---");
    eprintln!("Prompt: \"Count from 1 to 10.\"");
    eprintln!(
        "Output ({chunk_count} tokens, {:.1} tok/s, {:.2}s):",
        tokens_per_sec,
        elapsed.as_secs_f64()
    );
    eprintln!("{full_text}");
    eprintln!("Done reason: {:?}", done_reason);
    eprintln!("---");

    assert!(chunk_count > 0, "Should produce at least one token");

    backend.unload("qwen2.5:0.5b-q4_k_m").await.unwrap();
}
