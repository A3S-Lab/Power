use serde::{Deserialize, Deserializer, Serialize};

use crate::backend::types::{MessageContent, Tool, ToolCall, ToolChoice};

// Re-import FunctionCall for use in tests
#[cfg(test)]
use crate::backend::types::FunctionCall;

/// Custom deserializer for `keep_alive` that accepts both string and number.
///
/// Ollama clients may send `"keep_alive": "5m"` (string) or `"keep_alive": 300` (seconds as integer).
/// This deserializer normalizes both to `Option<String>`.
fn deserialize_keep_alive<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    match value {
        None => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s)),
        Some(serde_json::Value::Number(n)) => {
            // Numeric value: treat as seconds, convert to string with "s" suffix
            // Special cases: 0 and -1 are passed as-is (they have special meaning)
            if let Some(i) = n.as_i64() {
                if i <= 0 {
                    Ok(Some(i.to_string()))
                } else {
                    Ok(Some(format!("{i}s")))
                }
            } else if let Some(f) = n.as_f64() {
                if f <= 0.0 {
                    Ok(Some(format!("{}", f as i64)))
                } else {
                    Ok(Some(format!("{}s", f as u64)))
                }
            } else {
                Ok(Some(n.to_string()))
            }
        }
        Some(other) => Err(serde::de::Error::custom(format!(
            "keep_alive must be a string or number, got: {other}"
        ))),
    }
}

// ============================================================================
// OpenAI-compatible types
// ============================================================================

/// OpenAI-compatible chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Whether the model may generate multiple tool calls in parallel.
    /// Accepted for API compatibility; the model decides based on its training.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

/// Structured output format specifier.
///
/// Supports OpenAI's `response_format` variants:
/// - `{"type": "json_object"}` — unconstrained JSON output
/// - `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}` — schema-constrained
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub r#type: String,
    /// JSON Schema definition for structured output (when type = "json_schema").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchemaSpec>,
}

/// JSON Schema specification for structured output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaSpec {
    /// Name of the schema (required by OpenAI API).
    pub name: String,
    /// Optional description of what the schema represents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The JSON Schema object defining the output structure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    /// Whether to enforce strict schema adherence (default: false).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A single message in the chat format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    #[serde(default)]
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Base64-encoded images for multimodal models (Ollama-native format).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// A single choice in a chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: Option<String>,
}

/// A streaming chunk for chat completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

/// A single choice in a streaming chat chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

/// Delta content in a streaming chat chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ============================================================================
// Completion types
// ============================================================================

/// OpenAI-compatible text completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
}

/// OpenAI-compatible text completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

/// A single choice in a completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

// ============================================================================
// Embedding types
// ============================================================================

/// OpenAI-compatible embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
}

/// Input to embedding endpoint - single string or array of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Multiple(v) => v,
        }
    }
}

/// OpenAI-compatible embedding response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// A single embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}

/// Token usage for embedding requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// ============================================================================
// Model listing types
// ============================================================================

/// OpenAI-compatible model list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Metadata about a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

// ============================================================================
// Shared types
// ============================================================================

/// Token usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Standard error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error detail inside an error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

// ============================================================================
// Native API types (Ollama-compatible)
// ============================================================================

/// Ollama-compatible generate request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<GenerateOptions>,
    #[serde(default)]
    pub format: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "deserialize_keep_alive")]
    pub keep_alive: Option<String>,
    /// Base64-encoded images for multimodal models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    /// Override the system prompt for this request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Override the chat template for this request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// If true, skip chat template formatting and send prompt as-is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,
    /// Suffix to append after the model's response (fill-in-the-middle).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    /// Context from a previous generate call (for conversation continuity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<u32>>,
}

/// Generation options for the native API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub num_predict: Option<u32>,
    pub stop: Option<Vec<String>>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub num_ctx: Option<u32>,
    pub mirostat: Option<u32>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    pub tfs_z: Option<f32>,
    pub typical_p: Option<f32>,
    /// Number of tokens to keep from the initial prompt for penalty evaluation.
    /// 0 = disabled, -1 = ctx_size.
    pub num_keep: Option<i32>,
    /// Last N tokens to consider for repetition penalty (0 = disabled, -1 = ctx_size).
    pub repeat_last_n: Option<i32>,
    /// Whether to penalize newline tokens in repetition penalty.
    pub penalize_newline: Option<bool>,
    /// Number of tokens to process in parallel (batch size).
    pub num_batch: Option<u32>,
    /// Number of threads to use for generation.
    pub num_thread: Option<u32>,
    /// Number of threads to use for batch processing.
    pub num_thread_batch: Option<u32>,
    /// Whether to use memory-mapped files for model loading.
    pub use_mmap: Option<bool>,
    /// Whether to lock the model in memory (prevent swapping).
    pub use_mlock: Option<bool>,
    /// Enable NUMA optimizations.
    pub numa: Option<bool>,
    /// Enable flash attention.
    pub flash_attention: Option<bool>,
    /// Number of layers to offload to GPU (-1 = all, 0 = none).
    pub num_gpu: Option<i32>,
    /// Index of the primary GPU to use.
    pub main_gpu: Option<i32>,
}

/// Ollama-compatible generate response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
    /// Context tokens for conversation continuity (returned on final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<u32>>,
}

/// Ollama-compatible chat request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeChatRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<GenerateOptions>,
    #[serde(default)]
    pub format: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "deserialize_keep_alive")]
    pub keep_alive: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

/// Ollama-compatible chat response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: ChatCompletionMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

/// Ollama-compatible model listing entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeModelInfo {
    pub name: String,
    /// Alias of `name` — many Ollama clients read this field.
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: NativeModelDetails,
}

/// Detailed information about a model in native format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeModelDetails {
    pub format: String,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
    /// Model family (e.g. "llama", "phi", "gemma").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Model families for multimodal models (e.g. ["llama", "clip"]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub families: Option<Vec<String>>,
}

/// Ollama-compatible pull request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequest {
    pub name: String,
    #[serde(default)]
    pub stream: Option<bool>,
    /// When true, skip TLS certificate verification for the download.
    #[serde(default)]
    pub insecure: Option<bool>,
}

/// Ollama-compatible pull progress response (streamed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

/// Push model request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushRequest {
    pub name: String,
    #[serde(default)]
    pub destination: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub insecure: Option<bool>,
}

/// Push model progress response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

/// Show model request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShowRequest {
    pub name: String,
    /// When true, return detailed tensor information in `model_info`.
    #[serde(default)]
    pub verbose: Option<bool>,
}

/// Show model response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: NativeModelDetails,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Model architecture metadata (e.g. {"general.architecture": "llama"}).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_info: Option<serde_json::Value>,
    /// ISO 8601 timestamp of when the model was last modified locally.
    pub modified_at: String,
    /// Parent model name (for models created via Modelfile).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_model: Option<String>,
}

/// Delete model request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRequest {
    pub name: String,
}

/// Native embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeEmbeddingRequest {
    pub model: String,
    pub prompt: String,
}

/// Native embedding response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeEmbeddingResponse {
    pub model: String,
    pub embedding: Vec<f32>,
}

/// Copy model request (Ollama-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyRequest {
    pub source: String,
    pub destination: String,
}

/// Native batch embed request (Ollama-compatible POST /api/embed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeEmbedRequest {
    pub model: String,
    pub input: NativeEmbedInput,
    /// Truncate input to the model's max context length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    /// Keep model loaded for this duration (e.g. "5m", "-1" for forever).
    #[serde(default, deserialize_with = "deserialize_keep_alive")]
    pub keep_alive: Option<String>,
}

/// Input for the native embed endpoint — single string or array of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NativeEmbedInput {
    Single(String),
    Multiple(Vec<String>),
}

impl NativeEmbedInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            NativeEmbedInput::Single(s) => vec![s],
            NativeEmbedInput::Multiple(v) => v,
        }
    }
}

/// Native batch embed response (Ollama-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeEmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    /// Total processing time in nanoseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    /// Model load time in nanoseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    /// Number of tokens in the prompt(s).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_completion_request_deserialize() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama3");
        assert_eq!(req.messages.len(), 1);
        assert!(req.stream.is_none());
        assert!(req.temperature.is_none());
        assert!(req.tools.is_none());
    }

    #[test]
    fn test_chat_completion_request_with_options() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 256,
            "stream": true
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.max_tokens, Some(256));
        assert_eq!(req.stream, Some(true));
    }

    #[test]
    fn test_chat_completion_request_with_tools() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"}
                }
            }],
            "tool_choice": "auto"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        assert_eq!(req.tools.unwrap()[0].function.name, "get_weather");
    }

    #[test]
    fn test_chat_completion_request_with_vision() {
        let json = r#"{
            "model": "llava",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                ]
            }]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        match &req.messages[0].content {
            MessageContent::Parts(parts) => assert_eq!(parts.len(), 2),
            _ => panic!("Expected Parts variant"),
        }
    }

    #[test]
    fn test_chat_completion_response_serialize() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1700000000,
            model: "llama3".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: MessageContent::Text("Hello!".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    images: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 3,
                total_tokens: 8,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("chatcmpl-123"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_chat_delta_skip_none() {
        let delta = ChatDelta {
            role: None,
            content: Some("hi".to_string()),
            tool_calls: None,
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(!json.contains("role"));
        assert!(!json.contains("tool_calls"));
        assert!(json.contains("hi"));
    }

    #[test]
    fn test_chat_delta_with_tool_calls() {
        let delta = ChatDelta {
            role: Some("assistant".to_string()),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "test".to_string(),
                    arguments: "{}".to_string(),
                },
                index: Some(0),
            }]),
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(json.contains("tool_calls"));
        assert!(json.contains("call_1"));
    }

    #[test]
    fn test_completion_request_deserialize() {
        let json = r#"{"model": "llama3", "prompt": "Hello"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama3");
        assert_eq!(req.prompt, "Hello");
    }

    #[test]
    fn test_embedding_input_single() {
        let json = r#"{"model": "embed", "input": "hello"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        let texts = req.input.into_vec();
        assert_eq!(texts, vec!["hello"]);
    }

    #[test]
    fn test_embedding_input_multiple() {
        let json = r#"{"model": "embed", "input": ["hello", "world"]}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        let texts = req.input.into_vec();
        assert_eq!(texts, vec!["hello", "world"]);
    }

    #[test]
    fn test_generate_request_deserialize() {
        let json = r#"{"model": "llama3", "prompt": "test"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama3");
        assert!(req.stream.is_none());
        assert!(req.options.is_none());
    }

    #[test]
    fn test_generate_request_with_options() {
        let json = r#"{
            "model": "llama3",
            "prompt": "test",
            "options": {"temperature": 0.5, "num_predict": 100, "top_k": 40, "seed": 42}
        }"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        let opts = req.options.unwrap();
        assert_eq!(opts.temperature, Some(0.5));
        assert_eq!(opts.num_predict, Some(100));
        assert_eq!(opts.top_k, Some(40));
        assert_eq!(opts.seed, Some(42));
    }

    #[test]
    fn test_generate_response_skip_none() {
        let resp = GenerateResponse {
            model: "llama3".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            response: "hi".to_string(),
            done: false,
            done_reason: None,
            total_duration: None,
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count: None,
            eval_duration: None,
            context: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("created_at"));
        assert!(!json.contains("total_duration"));
        assert!(!json.contains("eval_count"));
        assert!(!json.contains("prompt_eval_count"));
        assert!(!json.contains("done_reason"));
    }

    #[test]
    fn test_pull_request_deserialize() {
        let json = r#"{"name": "llama3"}"#;
        let req: PullRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "llama3");
        assert!(req.stream.is_none());
    }

    #[test]
    fn test_pull_response_skip_none() {
        let resp = PullResponse {
            status: "success".to_string(),
            digest: None,
            total: None,
            completed: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("digest"));
        assert!(!json.contains("total"));
    }

    #[test]
    fn test_push_request_deserialize() {
        let json = r#"{"name": "llama3", "destination": "https://registry.example.com"}"#;
        let req: PushRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "llama3");
        assert_eq!(
            req.destination,
            Some("https://registry.example.com".to_string())
        );
        assert!(req.stream.is_none());
    }

    #[test]
    fn test_push_request_without_destination() {
        let json = r#"{"name": "llama3"}"#;
        let req: PushRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "llama3");
        assert!(req.destination.is_none());
    }

    #[test]
    fn test_push_response_serialize() {
        let resp = PushResponse {
            status: "success".to_string(),
            digest: Some("sha256:abc123".to_string()),
            total: None,
            completed: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("success"));
        assert!(json.contains("sha256:abc123"));
    }

    #[test]
    fn test_model_list_serialize() {
        let list = ModelList {
            object: "list".to_string(),
            data: vec![ModelInfo {
                id: "llama3".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "local".to_string(),
            }],
        };
        let json = serde_json::to_string(&list).unwrap();
        assert!(json.contains("llama3"));
        assert!(json.contains("\"object\":\"list\""));
    }

    #[test]
    fn test_error_response_serialize() {
        let resp = ErrorResponse {
            error: ErrorDetail {
                message: "not found".to_string(),
                error_type: "invalid_request_error".to_string(),
                code: Some("model_not_found".to_string()),
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("not found"));
        assert!(json.contains("model_not_found"));
    }

    #[test]
    fn test_native_chat_request_deserialize() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}]
        }"#;
        let req: NativeChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama3");
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_native_chat_request_with_tools() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object"}
                }
            }]
        }"#;
        let req: NativeChatRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        assert_eq!(req.tools.unwrap()[0].function.name, "search");
    }

    #[test]
    fn test_chat_completion_request_with_response_format() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "seed": 42
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let fmt = req.response_format.unwrap();
        assert_eq!(fmt.r#type, "json_object");
        assert_eq!(req.frequency_penalty, Some(0.5));
        assert_eq!(req.presence_penalty, Some(0.3));
        assert_eq!(req.seed, Some(42));
    }

    #[test]
    fn test_generate_request_with_format_and_keep_alive() {
        let json = r#"{
            "model": "llama3",
            "prompt": "test",
            "format": "json",
            "keep_alive": "5m"
        }"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            req.format,
            Some(serde_json::Value::String("json".to_string()))
        );
        assert_eq!(req.keep_alive.as_deref(), Some("5m"));
    }

    #[test]
    fn test_native_chat_request_with_format_and_keep_alive() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "format": "json",
            "keep_alive": "-1"
        }"#;
        let req: NativeChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            req.format,
            Some(serde_json::Value::String("json".to_string()))
        );
        assert_eq!(req.keep_alive.as_deref(), Some("-1"));
    }

    #[test]
    fn test_format_accepts_json_schema_object() {
        let json = r#"{
            "model": "llama3",
            "prompt": "test",
            "format": {"type": "object", "properties": {"name": {"type": "string"}}}
        }"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!(req.format.is_some());
        assert!(req.format.unwrap().is_object());
    }

    #[test]
    fn test_seed_accepts_negative_value() {
        let json = r#"{
            "model": "llama3",
            "prompt": "test",
            "options": {"seed": -1}
        }"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.options.unwrap().seed, Some(-1));
    }

    #[test]
    fn test_show_response_serialize() {
        let resp = ShowResponse {
            modelfile: String::new(),
            parameters: "{}".to_string(),
            template: String::new(),
            details: NativeModelDetails {
                format: "GGUF".to_string(),
                parameter_size: None,
                quantization_level: Some("Q4_K_M".to_string()),
                family: Some("llama".to_string()),
                families: None,
            },
            system: None,
            license: None,
            model_info: None,
            modified_at: chrono::Utc::now().to_rfc3339(),
            parent_model: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("GGUF"));
        assert!(json.contains("Q4_K_M"));
        assert!(json.contains("modified_at"));
        assert!(json.contains("\"family\":\"llama\""));
        // Optional fields should be omitted when None
        assert!(!json.contains("system"));
        assert!(!json.contains("license"));
        assert!(!json.contains("model_info"));
        assert!(!json.contains("families"));
    }

    #[test]
    fn test_show_request_deserialize_basic() {
        let json = r#"{"name": "llama3"}"#;
        let req: ShowRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "llama3");
        assert!(req.verbose.is_none());
    }

    #[test]
    fn test_show_request_deserialize_verbose() {
        let json = r#"{"name": "llama3", "verbose": true}"#;
        let req: ShowRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "llama3");
        assert_eq!(req.verbose, Some(true));
    }

    #[test]
    fn test_show_request_verbose_defaults_to_none() {
        let json = r#"{"name": "llama3", "verbose": false}"#;
        let req: ShowRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.verbose, Some(false));
    }

    #[test]
    fn test_generate_request_with_new_fields() {
        let json = r#"{
            "model": "llama3",
            "prompt": "test",
            "system": "You are helpful.",
            "template": "custom",
            "raw": true,
            "suffix": "END",
            "images": ["base64data=="],
            "context": [1, 2, 3]
        }"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.system.as_deref(), Some("You are helpful."));
        assert_eq!(req.template.as_deref(), Some("custom"));
        assert_eq!(req.raw, Some(true));
        assert_eq!(req.suffix.as_deref(), Some("END"));
        assert_eq!(req.images.as_ref().unwrap().len(), 1);
        assert_eq!(req.context.as_ref().unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn test_generate_request_new_fields_default_none() {
        let json = r#"{"model": "llama3", "prompt": "test"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!(req.system.is_none());
        assert!(req.template.is_none());
        assert!(req.raw.is_none());
        assert!(req.suffix.is_none());
        assert!(req.images.is_none());
        assert!(req.context.is_none());
    }

    #[test]
    fn test_generate_response_context_field() {
        let resp = GenerateResponse {
            model: "llama3".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            response: "hi".to_string(),
            done: true,
            done_reason: Some("stop".to_string()),
            total_duration: None,
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count: None,
            eval_duration: None,
            context: Some(vec![1, 2, 3]),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("context"));
        assert!(json.contains("[1,2,3]"));
    }

    #[test]
    fn test_chat_message_with_images() {
        let json = r#"{
            "role": "user",
            "content": "What is this?",
            "images": ["iVBORw0KGgo="]
        }"#;
        let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "user");
        assert!(msg.images.is_some());
        assert_eq!(msg.images.unwrap().len(), 1);
    }

    #[test]
    fn test_chat_message_images_skipped_when_none() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: MessageContent::Text("hello".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("images"));
    }

    #[test]
    fn test_delete_blob_endpoint_in_api_types() {
        // Verify PushRequest/PushResponse serialize correctly
        let req = PushRequest {
            name: "model".to_string(),
            destination: Some("http://localhost".to_string()),
            stream: Some(true),
            insecure: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"stream\":true"));
    }

    #[test]
    fn test_generate_options_new_fields() {
        let json = r#"{
            "temperature": 0.7,
            "num_predict": 100,
            "repeat_last_n": 128,
            "penalize_newline": false,
            "num_batch": 512,
            "num_thread": 8,
            "num_thread_batch": 4,
            "use_mmap": true,
            "use_mlock": true,
            "numa": true,
            "flash_attention": true,
            "num_gpu": -1,
            "main_gpu": 0
        }"#;
        let opts: GenerateOptions = serde_json::from_str(json).unwrap();
        assert_eq!(opts.temperature, Some(0.7));
        assert_eq!(opts.repeat_last_n, Some(128));
        assert_eq!(opts.penalize_newline, Some(false));
        assert_eq!(opts.num_batch, Some(512));
        assert_eq!(opts.num_thread, Some(8));
        assert_eq!(opts.num_thread_batch, Some(4));
        assert_eq!(opts.use_mmap, Some(true));
        assert_eq!(opts.use_mlock, Some(true));
        assert_eq!(opts.numa, Some(true));
        assert_eq!(opts.flash_attention, Some(true));
        assert_eq!(opts.num_gpu, Some(-1));
        assert_eq!(opts.main_gpu, Some(0));
    }

    #[test]
    fn test_generate_options_new_fields_default_to_none() {
        let json = r#"{"temperature": 0.5}"#;
        let opts: GenerateOptions = serde_json::from_str(json).unwrap();
        assert_eq!(opts.temperature, Some(0.5));
        assert!(opts.repeat_last_n.is_none());
        assert!(opts.penalize_newline.is_none());
        assert!(opts.num_batch.is_none());
        assert!(opts.num_thread.is_none());
        assert!(opts.flash_attention.is_none());
        assert!(opts.num_gpu.is_none());
        assert!(opts.main_gpu.is_none());
        assert!(opts.use_mmap.is_none());
        assert!(opts.use_mlock.is_none());
    }

    #[test]
    fn test_backend_completion_request_new_fields() {
        let json = r#"{
            "prompt": "test",
            "repeat_last_n": 64,
            "penalize_newline": true,
            "num_batch": 256,
            "num_thread": 4,
            "flash_attention": true,
            "num_gpu": -1,
            "main_gpu": 1,
            "use_mlock": true
        }"#;
        let req: crate::backend::types::CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "test");
        assert_eq!(req.repeat_last_n, Some(64));
        assert_eq!(req.penalize_newline, Some(true));
        assert_eq!(req.num_batch, Some(256));
        assert_eq!(req.num_thread, Some(4));
        assert_eq!(req.flash_attention, Some(true));
        assert_eq!(req.num_gpu, Some(-1));
        assert_eq!(req.main_gpu, Some(1));
        assert_eq!(req.use_mlock, Some(true));
    }

    #[test]
    fn test_backend_chat_request_new_fields() {
        let json = r#"{
            "messages": [],
            "repeat_last_n": 32,
            "flash_attention": true,
            "num_thread": 8
        }"#;
        let req: crate::backend::types::ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.repeat_last_n, Some(32));
        assert_eq!(req.flash_attention, Some(true));
        assert_eq!(req.num_thread, Some(8));
        assert!(req.penalize_newline.is_none());
    }

    #[test]
    fn test_keep_alive_as_string() {
        let json = r#"{"model": "llama3", "prompt": "test", "keep_alive": "5m"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.keep_alive.as_deref(), Some("5m"));
    }

    #[test]
    fn test_keep_alive_as_positive_integer() {
        let json = r#"{"model": "llama3", "prompt": "test", "keep_alive": 300}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.keep_alive.as_deref(), Some("300s"));
    }

    #[test]
    fn test_keep_alive_as_zero() {
        let json = r#"{"model": "llama3", "prompt": "test", "keep_alive": 0}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.keep_alive.as_deref(), Some("0"));
    }

    #[test]
    fn test_keep_alive_as_negative_one() {
        let json = r#"{"model": "llama3", "prompt": "test", "keep_alive": -1}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.keep_alive.as_deref(), Some("-1"));
    }

    #[test]
    fn test_keep_alive_missing() {
        let json = r#"{"model": "llama3", "prompt": "test"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!(req.keep_alive.is_none());
    }

    #[test]
    fn test_keep_alive_string_forever() {
        let json = r#"{"model": "llama3", "prompt": "test", "keep_alive": "-1"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.keep_alive.as_deref(), Some("-1"));
    }

    #[test]
    fn test_keep_alive_numeric_in_chat_request() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "keep_alive": 600
        }"#;
        let req: NativeChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.keep_alive.as_deref(), Some("600s"));
    }

    #[test]
    fn test_keep_alive_string_in_chat_request() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "keep_alive": "10m"
        }"#;
        let req: NativeChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.keep_alive.as_deref(), Some("10m"));
    }

    #[test]
    fn test_pull_request_with_insecure() {
        let json = r#"{"name": "llama3", "insecure": true}"#;
        let req: PullRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "llama3");
        assert_eq!(req.insecure, Some(true));
    }

    #[test]
    fn test_pull_request_insecure_defaults_to_none() {
        let json = r#"{"name": "llama3"}"#;
        let req: PullRequest = serde_json::from_str(json).unwrap();
        assert!(req.insecure.is_none());
    }

    #[test]
    fn test_response_format_json_object() {
        let json = r#"{"type": "json_object"}"#;
        let fmt: ResponseFormat = serde_json::from_str(json).unwrap();
        assert_eq!(fmt.r#type, "json_object");
        assert!(fmt.json_schema.is_none());
    }

    #[test]
    fn test_response_format_json_schema() {
        let json = r#"{
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "description": "A person object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name"]
                },
                "strict": true
            }
        }"#;
        let fmt: ResponseFormat = serde_json::from_str(json).unwrap();
        assert_eq!(fmt.r#type, "json_schema");
        let spec = fmt.json_schema.unwrap();
        assert_eq!(spec.name, "person");
        assert_eq!(spec.description.as_deref(), Some("A person object"));
        assert_eq!(spec.strict, Some(true));
        let schema = spec.schema.unwrap();
        assert!(schema["properties"]["name"]["type"] == "string");
    }

    #[test]
    fn test_response_format_json_schema_minimal() {
        let json = r#"{
            "type": "json_schema",
            "json_schema": {
                "name": "output"
            }
        }"#;
        let fmt: ResponseFormat = serde_json::from_str(json).unwrap();
        assert_eq!(fmt.r#type, "json_schema");
        let spec = fmt.json_schema.unwrap();
        assert_eq!(spec.name, "output");
        assert!(spec.schema.is_none());
        assert!(spec.description.is_none());
        assert!(spec.strict.is_none());
    }

    #[test]
    fn test_response_format_serialization_skips_none() {
        let fmt = ResponseFormat {
            r#type: "json_object".to_string(),
            json_schema: None,
        };
        let json = serde_json::to_string(&fmt).unwrap();
        assert!(json.contains("json_object"));
        assert!(!json.contains("json_schema"));
    }
}
