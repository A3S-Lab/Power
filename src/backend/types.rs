use serde::{Deserialize, Serialize};

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Request for chat-based inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default)]
    pub num_ctx: Option<u32>,
    #[serde(default)]
    pub mirostat: Option<u32>,
    #[serde(default)]
    pub mirostat_tau: Option<f32>,
    #[serde(default)]
    pub mirostat_eta: Option<f32>,
    #[serde(default)]
    pub tfs_z: Option<f32>,
    #[serde(default)]
    pub typical_p: Option<f32>,
    #[serde(default)]
    pub response_format: Option<String>,
}

/// A streamed chunk from a chat completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponseChunk {
    pub content: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    /// "stop" when EOS/stop-sequence hit, "length" when max_tokens reached
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
}

/// Request for text completion inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
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
    pub stream: bool,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default)]
    pub num_ctx: Option<u32>,
    #[serde(default)]
    pub mirostat: Option<u32>,
    #[serde(default)]
    pub mirostat_tau: Option<f32>,
    #[serde(default)]
    pub mirostat_eta: Option<f32>,
    #[serde(default)]
    pub tfs_z: Option<f32>,
    #[serde(default)]
    pub typical_p: Option<f32>,
    #[serde(default)]
    pub response_format: Option<String>,
}

/// A streamed chunk from a text completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponseChunk {
    pub text: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    /// "stop" when EOS/stop-sequence hit, "length" when max_tokens reached
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
}

/// Request for embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub input: Vec<String>,
}

/// Response containing generated embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_serialize() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("user"));
        assert!(json.contains("hello"));
    }

    #[test]
    fn test_chat_request_defaults() {
        let json = r#"{"messages": []}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
        assert!(req.max_tokens.is_none());
        assert!(!req.stream);
        assert!(req.top_k.is_none());
        assert!(req.min_p.is_none());
        assert!(req.repeat_penalty.is_none());
        assert!(req.frequency_penalty.is_none());
        assert!(req.presence_penalty.is_none());
        assert!(req.seed.is_none());
        assert!(req.num_ctx.is_none());
        assert!(req.mirostat.is_none());
        assert!(req.response_format.is_none());
    }

    #[test]
    fn test_chat_response_chunk() {
        let chunk = ChatResponseChunk {
            content: "hi".to_string(),
            done: false,
            prompt_tokens: None,
            done_reason: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let parsed: ChatResponseChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.content, "hi");
        assert!(!parsed.done);
        // prompt_tokens and done_reason should not appear in JSON when None
        assert!(!json.contains("prompt_tokens"));
        assert!(!json.contains("done_reason"));
    }

    #[test]
    fn test_completion_request_defaults() {
        let json = r#"{"prompt": "test"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "test");
        assert!(!req.stream);
    }

    #[test]
    fn test_completion_response_chunk() {
        let chunk = CompletionResponseChunk {
            text: "token".to_string(),
            done: true,
            prompt_tokens: Some(10),
            done_reason: Some("stop".to_string()),
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let parsed: CompletionResponseChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.text, "token");
        assert!(parsed.done);
        assert_eq!(parsed.prompt_tokens, Some(10));
        assert_eq!(parsed.done_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn test_embedding_request() {
        let req = EmbeddingRequest {
            input: vec!["hello".to_string(), "world".to_string()],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("hello"));
        assert!(json.contains("world"));
    }

    #[test]
    fn test_embedding_response() {
        let resp = EmbeddingResponse {
            embeddings: vec![vec![0.1, 0.2, 0.3]],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: EmbeddingResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.embeddings.len(), 1);
        assert_eq!(parsed.embeddings[0].len(), 3);
    }
}
