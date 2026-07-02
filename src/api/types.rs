use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::api::receipt::AttestationReceipt;
use crate::backend::types::{
    ContentPart, FunctionChoice, FunctionDefinition, MessageContent, Tool, ToolCall, ToolChoice,
    ToolChoiceSpecific,
};

// Re-import FunctionCall for use in tests
#[cfg(test)]
use crate::backend::types::FunctionCall;

// ============================================================================
// OpenAI-compatible types
// ============================================================================

/// Options controlling streaming behaviour.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StreamOptions {
    /// When true, a final chunk with a `usage` field is emitted before `[DONE]`.
    #[serde(default)]
    pub include_usage: bool,
    /// Unknown streaming options are preserved so request handlers can reject
    /// unsupported protocol choices instead of silently dropping them.
    #[serde(default, flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unsupported: BTreeMap<String, serde_json::Value>,
}

impl StreamOptions {
    /// Return a stable list of unsupported `stream_options` field names.
    pub fn unsupported_fields(&self) -> Vec<&str> {
        self.unsupported.keys().map(String::as_str).collect()
    }

    /// Return a validation message when unsupported fields are present.
    pub fn unsupported_fields_message(&self) -> Option<String> {
        let fields = self.unsupported_fields();
        if fields.is_empty() {
            None
        } else {
            Some(format!(
                "unsupported stream_options field(s): {}; supported field is include_usage",
                fields.join(", ")
            ))
        }
    }
}

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
    /// OpenAI's newer generated-token limit field. When present without
    /// `max_tokens`, Power maps it to the backend's generated-token limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Number of choices to generate. Power currently supports one choice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Whether to return token log probabilities. Power currently rejects
    /// logprob response-shape requests because responses do not carry logprobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of top token log probabilities to return when logprobs are
    /// requested. Power currently rejects logprob response-shape requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    /// Token bias map. Power currently rejects explicit logit bias requests
    /// because backends do not share a verified token-ID bias path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>,
    /// Extended sampling controls accepted by local backends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub penalize_newline: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirostat_eta: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tfs_z: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Streaming-specific options (only meaningful when `stream = true`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// Desired output modalities. Power currently supports text output only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    /// Audio output options. Power currently rejects audio output requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<serde_json::Value>,
    /// Static prediction hints. Power currently rejects prediction hints because
    /// backends do not consume them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction: Option<serde_json::Value>,
    /// Reasoning-effort control. Power currently rejects this control because
    /// backends do not expose a verified reasoning-effort path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Legacy OpenAI function definitions. Power maps these to modern
    /// `tools` when `tools` is not also present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<FunctionDefinition>>,
    /// Legacy OpenAI function-call policy. Power maps this to modern
    /// `tool_choice` when `tool_choice` is not also present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_call: Option<LegacyFunctionChoice>,
    /// Whether the model may generate multiple tool calls in parallel.
    /// Forwarded for remote models; local models reject `false` with tools
    /// because the local backend cannot enforce single tool-call generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// How long to keep the model loaded after the request (e.g. "5m", "0", "1h").
    /// Overrides the server default for this request only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

impl ChatCompletionRequest {
    /// Return true when any chat message carries an image input.
    pub fn has_image_inputs(&self) -> bool {
        self.messages
            .iter()
            .any(ChatCompletionMessage::has_image_inputs)
    }

    /// Return true when any request message carries unsupported thinking input.
    pub fn has_thinking_inputs(&self) -> bool {
        self.messages
            .iter()
            .any(ChatCompletionMessage::has_thinking_input)
    }

    /// Return a validation message for unsupported nested chat message fields.
    pub fn unsupported_message_fields_message(&self) -> Option<String> {
        for (message_index, message) in self.messages.iter().enumerate() {
            if let Some(message) = message.unsupported_fields_message() {
                return Some(format!("messages[{message_index}]: {message}"));
            }

            let MessageContent::Parts(parts) = &message.content else {
                continue;
            };
            for (part_index, part) in parts.iter().enumerate() {
                if let Some(message) = part.unsupported_fields_message() {
                    return Some(format!(
                        "messages[{message_index}].content[{part_index}]: {message}"
                    ));
                }
                if let ContentPart::ImageUrl { image_url, .. } = part {
                    if let Some(message) = image_url.unsupported_fields_message() {
                        return Some(format!(
                            "messages[{message_index}].content[{part_index}].image_url: {message}"
                        ));
                    }
                }
            }
        }
        None
    }

    /// Return true when both generated-token limit aliases are present but disagree.
    pub fn has_conflicting_max_token_limits(&self) -> bool {
        matches!(
            (self.max_tokens, self.max_completion_tokens),
            (Some(max_tokens), Some(max_completion_tokens)) if max_tokens != max_completion_tokens
        )
    }

    /// Effective generated-token limit for backend requests.
    pub fn effective_max_tokens(&self) -> Option<u32> {
        self.max_tokens.or(self.max_completion_tokens)
    }

    /// Return true when streaming-only options are present on a non-streaming request.
    pub fn has_stream_options_without_stream(&self) -> bool {
        self.stream_options.is_some() && !self.stream.unwrap_or(false)
    }

    /// Return true when a request asks for anything other than text output.
    pub fn has_unsupported_modalities(&self) -> bool {
        self.modalities
            .as_ref()
            .is_some_and(|modalities| modalities.len() != 1 || modalities[0] != "text")
    }

    /// Return true when both legacy and modern tool definitions are present.
    pub fn has_conflicting_tool_definitions(&self) -> bool {
        self.tools.is_some() && self.functions.is_some()
    }

    /// Return true when both legacy and modern tool-choice policies are present.
    pub fn has_conflicting_tool_choice(&self) -> bool {
        self.tool_choice.is_some() && self.function_call.is_some()
    }

    /// Effective tool definitions after mapping legacy `functions`.
    pub fn effective_tools(&self) -> Option<Vec<Tool>> {
        self.tools.clone().or_else(|| {
            self.functions.as_ref().map(|functions| {
                functions
                    .iter()
                    .cloned()
                    .map(|function| Tool {
                        tool_type: "function".to_string(),
                        function,
                        unsupported: BTreeMap::new(),
                    })
                    .collect()
            })
        })
    }

    /// Return a validation message for unsupported nested tool definition fields.
    pub fn unsupported_tool_fields_message(&self) -> Option<String> {
        let tools = self.effective_tools()?;
        for tool in tools {
            if let Some(message) = tool.unsupported_fields_message() {
                return Some(message);
            }
            if let Some(message) = tool.function.unsupported_fields_message() {
                return Some(message);
            }
        }
        None
    }

    /// Return a validation message for unsupported nested tool-choice fields.
    pub fn unsupported_tool_choice_fields_message(&self) -> Option<String> {
        if let Some(tool_choice) = &self.tool_choice {
            match tool_choice {
                ToolChoice::String(_) => {}
                ToolChoice::Specific(choice) => {
                    if let Some(message) = choice.unsupported_fields_message() {
                        return Some(message);
                    }
                    if let Some(message) = choice.function.unsupported_fields_message() {
                        return Some(message);
                    }
                }
            }
        }

        self.function_call
            .as_ref()
            .and_then(LegacyFunctionChoice::unsupported_fields_message)
    }

    /// Effective tool-choice policy after mapping legacy `function_call`.
    pub fn effective_tool_choice(&self) -> Option<ToolChoice> {
        self.tool_choice.clone().or_else(|| {
            self.function_call
                .as_ref()
                .map(LegacyFunctionChoice::to_tool_choice)
        })
    }
}

/// Legacy OpenAI function-call policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LegacyFunctionChoice {
    String(String),
    Specific(LegacyFunctionChoiceSpecific),
}

impl LegacyFunctionChoice {
    fn to_tool_choice(&self) -> ToolChoice {
        match self {
            LegacyFunctionChoice::String(value) => ToolChoice::String(value.clone()),
            LegacyFunctionChoice::Specific(choice) => ToolChoice::Specific(ToolChoiceSpecific {
                tool_type: "function".to_string(),
                function: FunctionChoice {
                    name: choice.name.clone(),
                    unsupported: BTreeMap::new(),
                },
                unsupported: BTreeMap::new(),
            }),
        }
    }

    fn unsupported_fields_message(&self) -> Option<String> {
        match self {
            LegacyFunctionChoice::String(_) => None,
            LegacyFunctionChoice::Specific(choice) => choice.unsupported_fields_message(),
        }
    }
}

/// Legacy OpenAI function-call object selecting a function by name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyFunctionChoiceSpecific {
    pub name: String,
    /// Unknown legacy function-call fields are preserved so request validation
    /// can fail closed instead of silently dropping selection policy.
    #[serde(default, flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unsupported: BTreeMap<String, serde_json::Value>,
}

impl LegacyFunctionChoiceSpecific {
    /// Return a stable list of unsupported legacy function-call field names.
    pub fn unsupported_fields(&self) -> Vec<&str> {
        self.unsupported.keys().map(String::as_str).collect()
    }

    /// Return a validation message when unsupported legacy function-call fields exist.
    pub fn unsupported_fields_message(&self) -> Option<String> {
        let fields = self.unsupported_fields();
        if fields.is_empty() {
            None
        } else {
            Some(format!(
                "unsupported function_call field(s): {}; supported field is name",
                fields.join(", ")
            ))
        }
    }
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
    /// Unknown response-format fields are preserved so request handlers can
    /// reject unsupported output policy instead of silently dropping it.
    #[serde(default, flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unsupported: BTreeMap<String, serde_json::Value>,
}

impl ResponseFormat {
    /// Return a stable list of unsupported `response_format` field names.
    pub fn unsupported_fields(&self) -> Vec<&str> {
        self.unsupported.keys().map(String::as_str).collect()
    }

    /// Return a validation message when unsupported fields are present.
    pub fn unsupported_fields_message(&self) -> Option<String> {
        let fields = self.unsupported_fields();
        if fields.is_empty() {
            None
        } else {
            Some(format!(
                "unsupported response_format field(s): {}; supported fields are type and json_schema",
                fields.join(", ")
            ))
        }
    }

    /// Validate that Power can honor this response-format request.
    pub fn validation_error(&self) -> Option<(&'static str, String)> {
        if let Some(message) = self.unsupported_fields_message() {
            return Some(("unsupported_response_format", message));
        }

        match self.r#type.as_str() {
            "text" | "json_object" => {
                if self.json_schema.is_some() {
                    Some((
                        "invalid_response_format",
                        "response_format.json_schema is only valid when response_format.type is json_schema"
                            .to_string(),
                    ))
                } else {
                    None
                }
            }
            "json_schema" => {
                let Some(schema_spec) = self.json_schema.as_ref() else {
                    return Some((
                        "invalid_response_format",
                        "response_format.type json_schema requires response_format.json_schema"
                            .to_string(),
                    ));
                };
                if let Some(message) = schema_spec.unsupported_fields_message() {
                    return Some(("unsupported_response_format", message));
                }
                if schema_spec.schema.is_none() {
                    return Some((
                        "invalid_response_format",
                        "response_format.type json_schema requires response_format.json_schema.schema"
                            .to_string(),
                    ));
                }
                None
            }
            unsupported => Some((
                "unsupported_response_format",
                format!(
                    "unsupported response_format.type '{unsupported}'; supported values are text, json_object, and json_schema"
                ),
            )),
        }
    }
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
    /// Unknown JSON schema spec fields are preserved so request handlers can
    /// reject unsupported structured-output policy instead of silently dropping it.
    #[serde(default, flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unsupported: BTreeMap<String, serde_json::Value>,
}

impl JsonSchemaSpec {
    /// Return a stable list of unsupported `response_format.json_schema` field names.
    pub fn unsupported_fields(&self) -> Vec<&str> {
        self.unsupported.keys().map(String::as_str).collect()
    }

    /// Return a validation message when unsupported fields are present.
    pub fn unsupported_fields_message(&self) -> Option<String> {
        let fields = self.unsupported_fields();
        if fields.is_empty() {
            None
        } else {
            Some(format!(
                "unsupported response_format.json_schema field(s): {}; supported fields are name, description, schema, and strict",
                fields.join(", ")
            ))
        }
    }
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
    /// Unknown message fields are preserved so request handlers can reject
    /// unsupported prompt or output policy instead of silently dropping it.
    #[serde(default, flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unsupported: BTreeMap<String, serde_json::Value>,
}

impl ChatCompletionMessage {
    /// Return true when this message carries OpenAI content-part or Ollama-native images.
    pub fn has_image_inputs(&self) -> bool {
        self.images
            .as_ref()
            .is_some_and(|images| !images.is_empty())
            || matches!(&self.content, MessageContent::Parts(parts)
                if parts.iter().any(|part| matches!(part, ContentPart::ImageUrl { .. })))
    }

    /// Return true when this message carries reasoning/thinking request input.
    pub fn has_thinking_input(&self) -> bool {
        self.thinking.is_some()
    }

    /// Return a stable list of unsupported chat message field names.
    pub fn unsupported_fields(&self) -> Vec<&str> {
        self.unsupported.keys().map(String::as_str).collect()
    }

    /// Return a validation message when unsupported message fields exist.
    pub fn unsupported_fields_message(&self) -> Option<String> {
        let fields = self.unsupported_fields();
        if fields.is_empty() {
            None
        } else {
            Some(format!(
                "unsupported message field(s): {}; supported fields are role, content, name, tool_calls, tool_call_id, images, and thinking",
                fields.join(", ")
            ))
        }
    }
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
    /// Server-side determinism fingerprint (model + sampling config hash).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    /// Request-level receipt covering prompt input and decoding/output policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attestation_receipt: Option<AttestationReceipt>,
    /// SHA-256 digest of `attestation_receipt`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attestation_receipt_sha256: Option<String>,
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
    /// Number of choices to generate. Power currently supports one choice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Number of token log probabilities to return. Power currently rejects
    /// logprob response-shape requests because responses do not carry logprobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,
    /// Whether to echo the prompt in the completion text. Power currently
    /// rejects echo requests because backends return generated text only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    /// Server-side candidates to sample before returning the best completion.
    /// Power currently supports only the default single candidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,
    /// Token bias map. Power currently rejects explicit logit bias requests
    /// because backends do not share a verified token-ID bias path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>,
    /// Extended sampling controls accepted by local backends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub penalize_newline: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirostat_eta: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tfs_z: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Streaming-specific options (only meaningful when `stream = true`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Suffix for fill-in-the-middle text completion. Power currently rejects
    /// explicit suffix requests because backends do not expose a verified
    /// fill-in-the-middle path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    /// How long to keep the model loaded after the request (e.g. "5m", "0", "1h").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

impl CompletionRequest {
    /// Return true when streaming-only options are present on a non-streaming request.
    pub fn has_stream_options_without_stream(&self) -> bool {
        self.stream_options.is_some() && !self.stream.unwrap_or(false)
    }
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
    /// Server-side determinism fingerprint (model + sampling config hash).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    /// Request-level receipt covering prompt input and decoding/output policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attestation_receipt: Option<AttestationReceipt>,
    /// SHA-256 digest of `attestation_receipt`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attestation_receipt_sha256: Option<String>,
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
    /// Output format. Power currently supports "float" (default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    /// Requested embedding dimension override. Power currently returns the
    /// model's native embedding dimension and rejects explicit overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
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
    /// The model this is a fine-tuned version of (null for base models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
    /// The parent model (null if not applicable).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
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
                    "parameters": {"type": "object"},
                    "strict": true
                }
            }],
            "tool_choice": "auto"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        let tool = &req.tools.unwrap()[0];
        assert_eq!(tool.function.name, "get_weather");
        assert_eq!(tool.function.strict, Some(true));
    }

    #[test]
    fn test_chat_completion_request_collects_unsupported_tool_fields() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "cache_control": true,
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                    "x-strict-mode": true
                }
            }]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let tools = req.tools.as_ref().unwrap();

        assert_eq!(tools[0].unsupported_fields(), vec!["cache_control"]);
        assert_eq!(
            tools[0].function.unsupported_fields(),
            vec!["x-strict-mode"]
        );
        assert!(req
            .unsupported_tool_fields_message()
            .unwrap()
            .contains("cache_control"));
    }

    #[test]
    fn test_chat_completion_request_collects_unsupported_tool_choice_fields() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "strict": true
                },
                "cache_control": true
            }
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        let Some(ToolChoice::Specific(choice)) = req.tool_choice.as_ref() else {
            panic!("expected specific tool choice");
        };
        assert_eq!(choice.unsupported_fields(), vec!["cache_control"]);
        assert_eq!(choice.function.unsupported_fields(), vec!["strict"]);
        assert!(req
            .unsupported_tool_choice_fields_message()
            .unwrap()
            .contains("cache_control"));
    }

    #[test]
    fn test_chat_completion_request_collects_unsupported_legacy_function_call_fields() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "function_call": {
                "name": "get_weather",
                "arguments": "{}"
            }
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        let Some(LegacyFunctionChoice::Specific(choice)) = req.function_call.as_ref() else {
            panic!("expected specific function_call");
        };
        assert_eq!(choice.unsupported_fields(), vec!["arguments"]);
        assert!(req
            .unsupported_tool_choice_fields_message()
            .unwrap()
            .contains("function_call field(s): arguments"));
    }

    #[test]
    fn test_chat_completion_request_collects_unsupported_message_fields() {
        let json = r#"{
            "model": "llama3",
            "messages": [{
                "role": "user",
                "content": "hi",
                "metadata": {"source": "test"}
            }]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        assert_eq!(req.messages[0].unsupported_fields(), vec!["metadata"]);
        let message = req.unsupported_message_fields_message().unwrap();
        assert!(message.contains("messages[0]"));
        assert!(message.contains("metadata"));
    }

    #[test]
    fn test_chat_completion_request_collects_unsupported_content_fields() {
        let json = r#"{
            "model": "llama3",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this", "cache_control": true},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/img.jpg",
                            "mime_type": "image/jpeg"
                        }
                    }
                ]
            }]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let MessageContent::Parts(parts) = &req.messages[0].content else {
            panic!("expected content parts");
        };

        assert_eq!(parts[0].unsupported_fields(), vec!["cache_control"]);
        let ContentPart::ImageUrl { image_url, .. } = &parts[1] else {
            panic!("expected image_url part");
        };
        assert_eq!(image_url.unsupported_fields(), vec!["mime_type"]);
        let message = req.unsupported_message_fields_message().unwrap();
        assert!(message.contains("messages[0].content[0]"));
        assert!(message.contains("cache_control"));
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
        assert!(req.has_image_inputs());
        assert!(req.messages[0].has_image_inputs());
        match &req.messages[0].content {
            MessageContent::Parts(parts) => assert_eq!(parts.len(), 2),
            _ => panic!("Expected Parts variant"),
        }
    }

    #[test]
    fn test_chat_completion_request_detects_message_images() {
        let json = r#"{
            "model": "llava",
            "messages": [{
                "role": "user",
                "content": "What is this?",
                "images": ["aGVsbG8="]
            }]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        assert!(req.has_image_inputs());
        assert!(req.messages[0].has_image_inputs());
    }

    #[test]
    fn test_chat_completion_request_without_images_reports_text_only() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        assert!(!req.has_image_inputs());
        assert!(!req.messages[0].has_image_inputs());
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
                    unsupported: Default::default(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 3,
                total_tokens: 8,
            },
            system_fingerprint: None,
            attestation_receipt: None,
            attestation_receipt_sha256: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("chatcmpl-123"));
        assert!(json.contains("Hello!"));
        assert!(!json.contains("attestation_receipt"));
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
    fn test_completion_request_with_response_format() {
        let json = r#"{
            "model": "llama3",
            "prompt": "Hello",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Answer",
                    "schema": { "type": "object" },
                    "strict": true
                }
            }
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        let format = req.response_format.unwrap();
        assert_eq!(format.r#type, "json_schema");
        let schema = format.json_schema.unwrap();
        assert_eq!(schema.name, "Answer");
        assert_eq!(schema.schema.unwrap()["type"], "object");
        assert_eq!(schema.strict, Some(true));
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
                root: None,
                parent: None,
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
    fn test_chat_completion_request_with_extended_sampling_controls() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
            "repeat_last_n": 64,
            "penalize_newline": true,
            "num_ctx": 4096,
            "mirostat": 2,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1,
            "tfs_z": 0.95,
            "typical_p": 0.9
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.top_k, Some(40));
        assert_eq!(req.min_p, Some(0.05));
        assert_eq!(req.repeat_penalty, Some(1.1));
        assert_eq!(req.repeat_last_n, Some(64));
        assert_eq!(req.penalize_newline, Some(true));
        assert_eq!(req.num_ctx, Some(4096));
        assert_eq!(req.mirostat, Some(2));
        assert_eq!(req.mirostat_tau, Some(5.0));
        assert_eq!(req.mirostat_eta, Some(0.1));
        assert_eq!(req.tfs_z, Some(0.95));
        assert_eq!(req.typical_p, Some(0.9));
    }

    #[test]
    fn test_completion_request_with_extended_sampling_controls() {
        let json = r#"{
            "model": "llama3",
            "prompt": "hi",
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
            "repeat_last_n": 64,
            "penalize_newline": false,
            "num_ctx": 2048,
            "mirostat": 1,
            "mirostat_tau": 4.0,
            "mirostat_eta": 0.2,
            "tfs_z": 0.9,
            "typical_p": 0.8
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.top_k, Some(40));
        assert_eq!(req.min_p, Some(0.05));
        assert_eq!(req.repeat_penalty, Some(1.1));
        assert_eq!(req.repeat_last_n, Some(64));
        assert_eq!(req.penalize_newline, Some(false));
        assert_eq!(req.num_ctx, Some(2048));
        assert_eq!(req.mirostat, Some(1));
        assert_eq!(req.mirostat_tau, Some(4.0));
        assert_eq!(req.mirostat_eta, Some(0.2));
        assert_eq!(req.tfs_z, Some(0.9));
        assert_eq!(req.typical_p, Some(0.8));
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
            unsupported: Default::default(),
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
        assert!(fmt.unsupported.is_empty());
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
        assert!(fmt.unsupported.is_empty());
        let spec = fmt.json_schema.unwrap();
        assert_eq!(spec.name, "person");
        assert_eq!(spec.description.as_deref(), Some("A person object"));
        assert_eq!(spec.strict, Some(true));
        assert!(spec.unsupported.is_empty());
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
        assert!(spec.unsupported.is_empty());
    }

    #[test]
    fn test_response_format_serialization_skips_none() {
        let fmt = ResponseFormat {
            r#type: "json_object".to_string(),
            json_schema: None,
            unsupported: BTreeMap::new(),
        };
        let json = serde_json::to_string(&fmt).unwrap();
        assert!(json.contains("json_object"));
        assert!(!json.contains("json_schema"));
    }

    #[test]
    fn test_response_format_preserves_unsupported_fields() {
        let json = r#"{
            "type": "json_object",
            "future_policy": true
        }"#;
        let fmt: ResponseFormat = serde_json::from_str(json).unwrap();

        assert_eq!(fmt.unsupported_fields(), vec!["future_policy"]);
        let (code, message) = fmt.validation_error().unwrap();
        assert_eq!(code, "unsupported_response_format");
        assert!(message.contains("unsupported response_format field(s): future_policy"));
    }

    #[test]
    fn test_json_schema_spec_preserves_unsupported_fields() {
        let json = r#"{
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
                "future_policy": {"mode": "strictest"}
            }
        }"#;
        let fmt: ResponseFormat = serde_json::from_str(json).unwrap();
        let spec = fmt.json_schema.as_ref().unwrap();

        assert_eq!(spec.unsupported_fields(), vec!["future_policy"]);
        let (code, message) = fmt.validation_error().unwrap();
        assert_eq!(code, "unsupported_response_format");
        assert!(message.contains("unsupported response_format.json_schema field(s): future_policy"));
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
            unsupported: Default::default(),
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
            unsupported: Default::default(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("thinking"));
    }

    #[test]
    fn test_stream_options_deserialize_include_usage() {
        let json = r#"{"include_usage": true}"#;
        let opts: StreamOptions = serde_json::from_str(json).unwrap();
        assert!(opts.include_usage);
        assert!(opts.unsupported_fields().is_empty());
    }

    #[test]
    fn test_stream_options_defaults_include_usage_false() {
        let json = r#"{}"#;
        let opts: StreamOptions = serde_json::from_str(json).unwrap();
        assert!(!opts.include_usage);
        assert!(opts.unsupported_fields().is_empty());
    }

    #[test]
    fn test_stream_options_collects_unsupported_fields() {
        let json = r#"{"include_usage": true, "include_obfuscation": true}"#;
        let opts: StreamOptions = serde_json::from_str(json).unwrap();

        assert!(opts.include_usage);
        assert_eq!(opts.unsupported_fields(), vec!["include_obfuscation"]);
        assert!(opts
            .unsupported_fields_message()
            .unwrap()
            .contains("include_obfuscation"));
    }

    #[test]
    fn test_chat_request_with_stream_options() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.stream, Some(true));
        let opts = req.stream_options.unwrap();
        assert!(opts.include_usage);
    }

    #[test]
    fn test_chat_request_detects_stream_options_without_stream() {
        let json = r#"{
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream_options": {"include_usage": true}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        assert!(req.has_stream_options_without_stream());
    }

    #[test]
    fn test_chat_request_stream_options_absent_by_default() {
        let json = r#"{"model": "m", "messages": []}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream_options.is_none());
    }

    #[test]
    fn test_completion_request_with_stream_options() {
        let json = r#"{
            "model": "llama3",
            "prompt": "hello",
            "stream": true,
            "stream_options": {"include_usage": true}
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.stream, Some(true));
        let opts = req.stream_options.unwrap();
        assert!(opts.include_usage);
    }

    #[test]
    fn test_completion_request_detects_stream_options_without_stream() {
        let json = r#"{
            "model": "llama3",
            "prompt": "hello",
            "stream_options": {"include_usage": true}
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();

        assert!(req.has_stream_options_without_stream());
    }

    #[test]
    fn test_completion_request_stream_options_absent_by_default() {
        let json = r#"{"model": "m", "prompt": "hi"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream_options.is_none());
    }
}
