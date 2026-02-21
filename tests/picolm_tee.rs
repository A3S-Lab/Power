//! Integration tests for picolm backend + TEE mode.
//!
//! Tests the full load → infer → unload cycle with the picolm backend,
//! verifying that layer-streaming inference works correctly end-to-end
//! and that TEE mode (log redaction, model integrity) integrates cleanly.

use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

use a3s_power::config::PowerConfig;
use a3s_power::model::manifest::{ModelFormat, ModelManifest};

mod gguf_builder;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Write a minimal but valid GGUF file with real tensors (uses the shared builder).
fn write_fake_gguf(path: &PathBuf) {
    gguf_builder::build_tiny_gguf(path, &gguf_builder::TinyModelConfig::default());
}

fn fake_manifest(name: &str, path: PathBuf) -> ModelManifest {
    ModelManifest {
        name: name.to_string(),
        format: ModelFormat::Gguf,
        size: 1024,
        sha256: format!("sha256:fake{name}"),
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
        family: None,
        families: None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "picolm")]
mod picolm_tests {
    use super::*;
    use a3s_power::backend::picolm::PicolmBackend;
    use a3s_power::backend::types::{ChatMessage, ChatRequest, MessageContent};
    use a3s_power::backend::Backend;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_picolm_load_unload_cycle() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("model.gguf");
        write_fake_gguf(&model_path);

        let config = Arc::new(PowerConfig::default());
        let backend = PicolmBackend::new(config);
        let manifest = fake_manifest("test-model", model_path);

        // Load
        let load_result = backend.load(&manifest).await;
        assert!(load_result.is_ok(), "picolm load failed: {:?}", load_result);

        // Unload
        let unload_result = backend.unload("test-model").await;
        assert!(unload_result.is_ok());
    }

    /// End-to-end test: build a tiny synthetic GGUF, load it, run chat,
    /// verify we get a stream of token chunks ending with done=true.
    #[tokio::test]
    async fn test_picolm_chat_produces_stream() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("tiny.gguf");

        // Build a real (tiny) GGUF with actual tensor data
        let tiny_cfg = gguf_builder::TinyModelConfig::default();
        gguf_builder::build_tiny_gguf(&model_path, &tiny_cfg);

        let config = Arc::new(PowerConfig::default());
        let backend = PicolmBackend::new(config);
        let manifest = fake_manifest("chat-model", model_path);

        backend.load(&manifest).await.unwrap();

        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("a b c".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            }],
            session_id: None,
            temperature: Some(0.0), // greedy
            top_p: None,
            max_tokens: Some(4),
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
        };

        let mut stream = backend.chat("chat-model", req).await.unwrap();

        // Collect all chunks
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            let c = chunk.unwrap();
            let done = c.done;
            chunks.push(c);
            if done {
                break;
            }
        }

        assert!(
            !chunks.is_empty(),
            "Stream should produce at least one chunk"
        );
        assert!(
            chunks.last().unwrap().done,
            "Last chunk should have done=true"
        );

        backend.unload("chat-model").await.unwrap();
    }

    #[tokio::test]
    async fn test_picolm_unload_frees_model() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("model.gguf");
        write_fake_gguf(&model_path);

        let config = Arc::new(PowerConfig::default());
        let backend = PicolmBackend::new(config);
        let manifest = fake_manifest("free-model", model_path);

        backend.load(&manifest).await.unwrap();
        backend.unload("free-model").await.unwrap();

        // After unload, chat should fail with ModelNotFound
        let req = ChatRequest {
            messages: vec![],
            session_id: None,
            temperature: None,
            top_p: None,
            max_tokens: Some(1),
            stop: None,
            stream: false,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
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
        };

        let result = backend.chat("free-model", req).await;
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("free-model"));
    }

    #[tokio::test]
    async fn test_picolm_embed_not_supported() {
        let config = Arc::new(PowerConfig::default());
        let backend = PicolmBackend::new(config);
        let result = backend
            .embed(
                "any",
                a3s_power::backend::types::EmbeddingRequest {
                    input: vec!["test".to_string()],
                },
            )
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("embeddings"));
    }

    #[cfg(feature = "picolm")]
    #[tokio::test]
    async fn test_picolm_registered_in_default_backends() {
        let config = Arc::new(PowerConfig::default());
        let registry = a3s_power::backend::default_backends(config);
        let names = registry.list_names();
        assert!(
            names.contains(&"picolm"),
            "picolm should be registered in default_backends when feature is enabled"
        );
    }

    /// Verify deterministic output: same seed + greedy sampling = same tokens.
    #[tokio::test]
    async fn test_picolm_deterministic_output() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("det.gguf");
        gguf_builder::build_tiny_gguf(&model_path, &gguf_builder::TinyModelConfig::default());

        let config = Arc::new(PowerConfig::default());
        let backend = PicolmBackend::new(config);
        let manifest = fake_manifest("det-model", model_path);
        backend.load(&manifest).await.unwrap();

        let make_req = || ChatRequest {
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
            max_tokens: Some(3),
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
        };

        // Run twice, collect output
        let collect = |mut stream: std::pin::Pin<
            Box<
                dyn futures::Stream<
                        Item = a3s_power::error::Result<
                            a3s_power::backend::types::ChatResponseChunk,
                        >,
                    > + Send,
            >,
        >| async move {
            let mut text = String::new();
            while let Some(chunk) = stream.next().await {
                let c = chunk.unwrap();
                text.push_str(&c.content);
                if c.done {
                    break;
                }
            }
            text
        };

        // First run uses a fresh session, second run uses a new session too
        // (no session_id = no KV cache reuse)
        let s1 = backend.chat("det-model", make_req()).await.unwrap();
        let t1 = collect(s1).await;

        // Unload and reload to reset all state
        backend.unload("det-model").await.unwrap();
        let dir2 = TempDir::new().unwrap();
        let model_path2 = dir2.path().join("det2.gguf");
        gguf_builder::build_tiny_gguf(&model_path2, &gguf_builder::TinyModelConfig::default());
        let manifest2 = fake_manifest("det-model", model_path2);
        backend.load(&manifest2).await.unwrap();

        let s2 = backend.chat("det-model", make_req()).await.unwrap();
        let t2 = collect(s2).await;

        assert_eq!(
            t1, t2,
            "Greedy decoding with same seed should be deterministic"
        );

        backend.unload("det-model").await.unwrap();
    }
}

// ── TEE mode integration ──────────────────────────────────────────────────────

#[cfg(feature = "picolm")]
mod tee_integration {
    use super::*;
    use a3s_power::backend::picolm::PicolmBackend;
    use a3s_power::backend::Backend;

    #[tokio::test]
    async fn test_picolm_with_tee_mode_config() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("model.gguf");
        write_fake_gguf(&model_path);

        // Enable TEE mode in config
        let mut config = PowerConfig::default();
        config.tee_mode = true;
        config.redact_logs = true;

        let backend = PicolmBackend::new(Arc::new(config));
        let manifest = fake_manifest("tee-model", model_path);

        // Load should succeed even with TEE mode enabled
        let result = backend.load(&manifest).await;
        assert!(
            result.is_ok(),
            "picolm load with tee_mode failed: {:?}",
            result
        );

        backend.unload("tee-model").await.unwrap();
    }

    #[tokio::test]
    async fn test_picolm_load_nonexistent_file_fails() {
        let config = Arc::new(PowerConfig::default());
        let backend = PicolmBackend::new(config);
        let manifest = fake_manifest("ghost", std::path::PathBuf::from("/nonexistent/model.gguf"));

        let result = backend.load(&manifest).await;
        assert!(result.is_err(), "Loading nonexistent file should fail");
    }
}
