//! Shared OpenAI-compatible requests and owned response streams.
mod stream;
mod tool_policy;
use crate::backend::types::{ChatMessage, ChatRequest};
use crate::error::{PowerError, Result};
use serde_json::{Map, Value};
pub(crate) use stream::chat_response_stream;
pub(crate) use tool_policy::ChatToolPolicy;

/// Validate both status and media type before decoding the common SSE wire.
/// Request errors remain client errors; transient/authenticated upstream
/// unavailability is not reported as an inference crash. Error bodies are not
/// copied into client errors because upstreams may echo prompts or credentials.
pub(crate) fn require_chat_stream(response: reqwest::Response) -> Result<reqwest::Response> {
    let status = response.status();
    if !status.is_success() {
        let message = format!("OpenAI upstream returned {status}");
        return Err(match status.as_u16() {
            400 | 413 | 422 => PowerError::InvalidRequest(message),
            401 | 403 | 503 => PowerError::BackendNotAvailable(message),
            _ => PowerError::InferenceFailed(message),
        });
    }
    let is_sse = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"));
    if !is_sse {
        return Err(PowerError::InferenceFailed(
            "OpenAI upstream did not return text/event-stream".into(),
        ));
    }
    Ok(response)
}

/// The same tool/history wire contract is used by every OpenAI proxy backend.
pub(crate) fn chat_request_body(model: &str, request: &ChatRequest, stream: bool) -> Result<Value> {
    let messages = request
        .messages
        .iter()
        .map(message_json)
        .collect::<Result<Vec<_>>>()?;
    let mut body = serde_json::json!({"model": model, "messages": messages, "stream": stream});
    if stream {
        let mut options = match &request.stream_options {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(options)) => options.clone(),
            _ => {
                return Err(PowerError::InvalidRequest(
                    "stream_options must be an object".into(),
                ))
            }
        };
        // Exact internal observations are collected even when the public caller
        // does not request a usage event; the API owns public usage policy.
        options.insert("include_usage".into(), Value::Bool(true));
        body["stream_options"] = Value::Object(options);
    }
    if let Some(value) = request.temperature {
        body["temperature"] = serde_json::json!(value);
    }
    if let Some(value) = request.top_p {
        body["top_p"] = serde_json::json!(value);
    }
    if let Some(value) = request.max_tokens {
        body["max_tokens"] = serde_json::json!(value);
    }
    if let Some(value) = request.frequency_penalty {
        body["frequency_penalty"] = serde_json::json!(value);
    }
    if let Some(value) = request.presence_penalty {
        body["presence_penalty"] = serde_json::json!(value);
    }
    if let Some(value) = request.seed {
        body["seed"] = serde_json::json!(value);
    }
    if let Some(value) = &request.stop {
        body["stop"] = serde_json::to_value(value).map_err(encode_error)?;
    }
    if let Some(value) = &request.tools {
        if !value.is_empty() {
            body["tools"] = serde_json::to_value(value).map_err(encode_error)?;
        }
    }
    if let Some(value) = &request.tool_choice {
        body["tool_choice"] = serde_json::to_value(value).map_err(encode_error)?;
    }
    if let Some(value) = request.parallel_tool_calls {
        body["parallel_tool_calls"] = Value::Bool(value);
    }
    if let Some(value) = &request.response_format {
        body["response_format"] = value.clone();
    }
    Ok(body)
}
fn message_json(message: &ChatMessage) -> Result<Value> {
    let mut object = Map::new();
    object.insert("role".into(), Value::String(message.role.clone()));
    object.insert(
        "content".into(),
        serde_json::to_value(&message.content).map_err(encode_error)?,
    );
    if let Some(value) = &message.name {
        object.insert("name".into(), Value::String(value.clone()));
    }
    if let Some(value) = &message.tool_calls {
        if !value.is_empty() {
            object.insert(
                "tool_calls".into(),
                serde_json::to_value(value).map_err(encode_error)?,
            );
        }
    }
    if let Some(value) = &message.tool_call_id {
        object.insert("tool_call_id".into(), Value::String(value.clone()));
    }
    Ok(Value::Object(object))
}
fn encode_error(error: serde_json::Error) -> PowerError {
    PowerError::InvalidRequest(format!("OpenAI request could not be encoded: {error}"))
}
#[cfg(test)]
mod tests;
