use serde::{Deserialize, Serialize};

use crate::backend::types::{MessageContent, Tool, ToolCall, ToolChoice};

// Re-import FunctionCall for use in tests
#[cfg(test)]
use crate::backend::types::FunctionCall;

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
    /// How long to keep the model loaded after the request (e.g. "5m", "0", "1h").
    /// Overrides the server default for this request only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
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
    /// Reasoning/thinking content from reasoning models (Ollama native wire format).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
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
    /// Reasoning/thinking content from reasoning models (DeepSeek-R1, QwQ).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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
    /// How long to keep the model loaded after the request (e.g. "5m", "0", "1h").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
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
    /// How long to keep the model loaded after the request (e.g. "5m", "0", "1h").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
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
                    thinking: None,
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
            reasoning_content: None,
            tool_calls: None,
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(!json.contains("role"));
        assert!(!json.contains("tool_calls"));
        assert!(!json.contains("reasoning_content"));
        assert!(json.contains("hi"));
    }

    #[test]
    fn test_chat_delta_with_tool_calls() {
        let delta = ChatDelta {
            role: Some("assistant".to_string()),
            content: None,
            reasoning_content: None,
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
            thinking: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("images"));
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

    #[test]
    fn test_chat_delta_with_reasoning_content() {
        let delta = ChatDelta {
            role: None,
            content: None,
            reasoning_content: Some("Let me think...".to_string()),
            tool_calls: None,
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(json.contains("reasoning_content"));
        assert!(json.contains("Let me think..."));
        assert!(!json.contains("\"content\""));
    }

    #[test]
    fn test_chat_delta_reasoning_content_skipped_when_none() {
        let delta = ChatDelta {
            role: None,
            content: Some("answer".to_string()),
            reasoning_content: None,
            tool_calls: None,
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(!json.contains("reasoning_content"));
        assert!(json.contains("answer"));
    }

    #[test]
    fn test_chat_message_thinking_field() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: MessageContent::Text("answer".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
            thinking: Some("reasoning here".to_string()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"thinking\":\"reasoning here\""));
        assert!(json.contains("answer"));
    }

    #[test]
    fn test_chat_message_thinking_skipped_when_none() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: MessageContent::Text("hello".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
            thinking: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("thinking"));
    }
}
