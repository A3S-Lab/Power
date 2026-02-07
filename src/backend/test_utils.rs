use std::pin::Pin;
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
use crate::server::state::AppState;

/// A mock backend for testing handlers without real inference.
pub struct MockBackend {
    load_succeeds: bool,
}

impl MockBackend {
    /// Create a mock backend where all operations succeed.
    pub fn success() -> Self {
        Self {
            load_succeeds: true,
        }
    }

    /// Create a mock backend where `load()` returns an error.
    pub fn load_fails() -> Self {
        Self {
            load_succeeds: false,
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
        let chunks = vec![
            Ok(ChatResponseChunk {
                content: "Hello".to_string(),
                done: false,
                prompt_tokens: None,
                done_reason: None,
            }),
            Ok(ChatResponseChunk {
                content: "".to_string(),
                done: true,
                prompt_tokens: Some(5),
                done_reason: Some("stop".to_string()),
            }),
        ];
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
            }),
            Ok(CompletionResponseChunk {
                text: "".to_string(),
                done: true,
                prompt_tokens: Some(5),
                done_reason: Some("stop".to_string()),
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
        };
        let mut stream = mock.chat("test", request).await.unwrap();
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.content, "Hello");
        assert!(!first.done);
        let second = stream.next().await.unwrap().unwrap();
        assert!(second.done);
    }
}
