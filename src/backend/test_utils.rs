use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;

use crate::backend::types::{
    ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk, EmbeddingRequest,
    EmbeddingResponse,
};
use crate::backend::{Backend, BackendRegistry};
use crate::config::PowerConfig;
use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest, ModelParameters};
use crate::model::registry::ModelRegistry;
use crate::server::request_context::RequestContext;
use crate::server::state::AppState;

/// A mock backend for testing handlers without real inference.
pub struct MockBackend {
    load_succeeds: bool,
    /// When true, chat() emits chunks simulating `<think>reasoning</think>answer`.
    emit_thinking: bool,
    /// Counter for cleanup_request calls (for test verification).
    pub cleanup_count: Arc<AtomicU32>,
}

impl MockBackend {
    /// Create a mock backend where all operations succeed.
    pub fn success() -> Self {
        Self {
            load_succeeds: true,
            emit_thinking: false,
            cleanup_count: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Create a mock backend where `load()` returns an error.
    pub fn load_fails() -> Self {
        Self {
            load_succeeds: false,
            emit_thinking: false,
            cleanup_count: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Create a mock backend that emits thinking content in chat responses.
    pub fn with_thinking() -> Self {
        Self {
            load_succeeds: true,
            emit_thinking: true,
            cleanup_count: Arc::new(AtomicU32::new(0)),
        }
    }
}

#[async_trait]
impl Backend for MockBackend {
    fn name(&self) -> &str {
        "mock"
    }

    fn supports(&self, format: &ModelFormat) -> bool {
        matches!(format, ModelFormat::Gguf)
    }

    async fn load(&self, _manifest: &ModelManifest) -> Result<()> {
        if self.load_succeeds {
            Ok(())
        } else {
            Err(PowerError::InferenceFailed("mock load failure".to_string()))
        }
    }

    async fn unload(&self, _model_name: &str) -> Result<()> {
        Ok(())
    }

    async fn chat(
        &self,
        _model_name: &str,
        _request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>> {
        let chunks = if self.emit_thinking {
            vec![
                Ok(ChatResponseChunk {
                    content: "".to_string(),
                    thinking_content: Some("Let me think about this.".to_string()),
                    done: false,
                    prompt_tokens: None,
                    done_reason: None,
                    prompt_eval_duration_ns: None,
                    tool_calls: None,
                }),
                Ok(ChatResponseChunk {
                    content: "The answer is 42.".to_string(),
                    thinking_content: None,
                    done: false,
                    prompt_tokens: None,
                    done_reason: None,
                    prompt_eval_duration_ns: None,
                    tool_calls: None,
                }),
                Ok(ChatResponseChunk {
                    content: "".to_string(),
                    thinking_content: None,
                    done: true,
                    prompt_tokens: Some(5),
                    done_reason: Some("stop".to_string()),
                    prompt_eval_duration_ns: Some(1_000_000),
                    tool_calls: None,
                }),
            ]
        } else {
            vec![
                Ok(ChatResponseChunk {
                    content: "Hello".to_string(),
                    thinking_content: None,
                    done: false,
                    prompt_tokens: None,
                    done_reason: None,
                    prompt_eval_duration_ns: None,
                    tool_calls: None,
                }),
                Ok(ChatResponseChunk {
                    content: "".to_string(),
                    thinking_content: None,
                    done: true,
                    prompt_tokens: Some(5),
                    done_reason: Some("stop".to_string()),
                    prompt_eval_duration_ns: Some(1_000_000),
                    tool_calls: None,
                }),
            ]
        };
        Ok(Box::pin(futures::stream::iter(chunks)))
    }

    async fn complete(
        &self,
        _model_name: &str,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        let chunks = vec![
            Ok(CompletionResponseChunk {
                text: "World".to_string(),
                done: false,
                prompt_tokens: None,
                done_reason: None,
                prompt_eval_duration_ns: None,
                token_id: Some(42),
            }),
            Ok(CompletionResponseChunk {
                text: "".to_string(),
                done: true,
                prompt_tokens: Some(5),
                done_reason: Some("stop".to_string()),
                prompt_eval_duration_ns: Some(1_000_000),
                token_id: None,
            }),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }

    async fn embed(
        &self,
        _model_name: &str,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        let embeddings = request.input.iter().map(|_| vec![0.1, 0.2, 0.3]).collect();
        Ok(EmbeddingResponse { embeddings })
    }

    async fn cleanup_request(&self, _model_name: &str, _ctx: &RequestContext) -> Result<()> {
        self.cleanup_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// Create an `AppState` backed by the given mock backend and an empty registry.
pub fn test_state_with_mock(mock: MockBackend) -> AppState {
    let mut backends = BackendRegistry::new();
    backends.register(Arc::new(mock));
    AppState::new(
        Arc::new(ModelRegistry::new()),
        Arc::new(backends),
        Arc::new(PowerConfig::default()),
    )
}

/// Create a sample `ModelManifest` for testing.
pub fn sample_manifest(name: &str) -> ModelManifest {
    ModelManifest {
        name: name.to_string(),
        format: ModelFormat::Gguf,
        size: 1_000_000,
        sha256: format!("sha256-{name}"),
        parameters: Some(ModelParameters {
            context_length: Some(4096),
            embedding_length: Some(3200),
            parameter_count: Some(3_000_000_000),
            quantization: Some("Q4_K_M".to_string()),
        }),
        created_at: chrono::Utc::now(),
        path: std::path::PathBuf::from(format!("/tmp/blobs/sha256-{name}")),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_backend_success() {
        let mock = MockBackend::success();
        let manifest = sample_manifest("test");
        assert!(mock.load(&manifest).await.is_ok());
        assert!(mock.supports(&ModelFormat::Gguf));
        assert!(!mock.supports(&ModelFormat::SafeTensors));
        assert_eq!(mock.name(), "mock");
    }

    #[tokio::test]
    async fn test_mock_backend_load_failure() {
        let mock = MockBackend::load_fails();
        let manifest = sample_manifest("test");
        let result = mock.load(&manifest).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("mock load failure"));
    }

    #[tokio::test]
    async fn test_mock_backend_stream_output() {
        use futures::StreamExt;

        let mock = MockBackend::success();
        let request = ChatRequest {
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
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
        };
        let mut stream = mock.chat("test", request).await.unwrap();
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.content, "Hello");
        assert!(!first.done);
        let second = stream.next().await.unwrap().unwrap();
        assert!(second.done);
    }

    #[tokio::test]
    async fn test_mock_backend_with_thinking() {
        use futures::StreamExt;

        let mock = MockBackend::with_thinking();
        let request = ChatRequest {
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
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
        };
        let mut stream = mock.chat("test", request).await.unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.content, "");
        assert_eq!(
            first.thinking_content.as_deref(),
            Some("Let me think about this.")
        );
        assert!(!first.done);

        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(second.content, "The answer is 42.");
        assert!(second.thinking_content.is_none());
        assert!(!second.done);

        let third = stream.next().await.unwrap().unwrap();
        assert!(third.done);
        assert_eq!(third.done_reason.as_deref(), Some("stop"));
    }
}
