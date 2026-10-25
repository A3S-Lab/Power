//! Bounded SSE decoding shared by OpenAI-compatible inference backends.

use super::ChatToolPolicy;
use std::pin::Pin;

use futures::{Stream, StreamExt};
use serde_json::Value;

use crate::backend::types::{ChatResponseChunk, FunctionCall, ToolCall};
use crate::error::{PowerError, Result};

const MAX_SSE_BUFFER_BYTES: usize = 8 * 1024 * 1024;

type ByteStream = Pin<Box<dyn Stream<Item = reqwest::Result<Vec<u8>>> + Send>>;

struct ChatStreamState {
    bytes: ByteStream,
    buffer: Vec<u8>,
    finish_reason: Option<String>,
    finished: bool,
    timings: Option<Value>,
    usage: Option<Value>,
    require_exact_usage: bool,
    tool_policy: ChatToolPolicy,
}

/// No spawned producer: dropping this stream drops its upstream HTTP body.
pub(crate) fn chat_response_stream(
    response: reqwest::Response,
    require_exact_usage: bool,
    tool_policy: ChatToolPolicy,
) -> impl Stream<Item = Result<ChatResponseChunk>> + Send {
    let bytes = Box::pin(
        response
            .bytes_stream()
            .map(|part| part.map(|bytes| bytes.to_vec())),
    );
    let state = ChatStreamState {
        bytes,
        buffer: Vec::new(),
        finish_reason: None,
        finished: false,
        timings: None,
        usage: None,
        require_exact_usage,
        tool_policy,
    };
    futures::stream::try_unfold(state, |mut state| async move {
        if state.finished {
            return Ok(None);
        }
        loop {
            let event = next_sse_event(&mut state.bytes, &mut state.buffer)
                .await
                .ok_or_else(|| protocol_error("stream ended before [DONE]"))??;
            let Some(json) = event else {
                if state.finish_reason.is_none() {
                    return Err(protocol_error("[DONE] arrived without a finish reason"));
                }
                if state.require_exact_usage && state.usage.is_none() {
                    return Err(protocol_error("upstream omitted exact chat usage"));
                }
                state.tool_policy.finish(
                    state
                        .finish_reason
                        .as_deref()
                        .ok_or_else(|| protocol_error("missing finish reason"))?,
                )?;
                state.finished = true;
                let chunk = state.chunk(String::new(), None, None, true);
                return Ok(Some((chunk, state)));
            };
            if json.get("error").is_some_and(|error| !error.is_null()) {
                return Err(protocol_error("upstream reported a stream error"));
            }
            if let Some(timings) = json.get("timings").filter(|value| !value.is_null()) {
                validate_timings(timings)?;
                state.timings = Some(timings.clone());
            }
            if let Some(usage) = json.get("usage").filter(|value| !value.is_null()) {
                validate_usage(usage)?;
                state.usage = Some(usage.clone());
            }
            let Some(choices) = json.get("choices") else {
                if json.get("usage").is_some() || json.get("timings").is_some() {
                    continue;
                }
                return Err(protocol_error("event has no choices or observation"));
            };
            let choices = choices
                .as_array()
                .ok_or_else(|| protocol_error("choices is not an array"))?;
            if choices.is_empty() {
                continue;
            }
            if choices.len() != 1 {
                return Err(protocol_error("multiple choices are unsupported"));
            }
            let choice = &choices[0];
            if choice
                .get("index")
                .is_some_and(|index| index.as_u64() != Some(0))
            {
                return Err(protocol_error("unexpected choice index"));
            }
            let delta = choice
                .get("delta")
                .and_then(Value::as_object)
                .ok_or_else(|| protocol_error("choice has no delta object"))?;
            let content = optional_string(delta.get("content"), "content")?.unwrap_or_default();
            let thinking = optional_string(delta.get("reasoning_content"), "reasoning_content")?
                .or(optional_string(delta.get("thinking"), "thinking")?)
                .filter(|value| !value.is_empty());
            let calls = parse_tool_deltas(delta.get("tool_calls"))?;
            if let Some(calls) = &calls {
                state.tool_policy.observe(calls)?;
            }
            let has_payload = !content.is_empty() || thinking.is_some() || calls.is_some();
            if state.finish_reason.is_some() && has_payload {
                return Err(protocol_error("payload arrived after the finish reason"));
            }
            if let Some(reason) = optional_string(choice.get("finish_reason"), "finish_reason")? {
                if !matches!(
                    reason.as_str(),
                    "stop" | "length" | "tool_calls" | "function_call" | "content_filter"
                ) {
                    return Err(protocol_error("unsupported finish reason"));
                }
                if state
                    .finish_reason
                    .as_ref()
                    .is_some_and(|previous| previous != &reason)
                {
                    return Err(protocol_error("invalid or conflicting finish reason"));
                }
                state.finish_reason = Some(reason);
            }
            if has_payload {
                let chunk = state.chunk(content, thinking, calls, false);
                return Ok(Some((chunk, state)));
            }
            // A finish event does not end the stream: authoritative usage and
            // timings may follow it in a separate choices=[] event.
        }
    })
}

impl ChatStreamState {
    fn chunk(
        &self,
        content: String,
        thinking_content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        done: bool,
    ) -> ChatResponseChunk {
        let upstream_timings = if self.timings.is_some() || self.usage.is_some() {
            let mut observation = serde_json::json!({ "source": "upstream-reported" });
            if let Some(timings) = &self.timings {
                observation["timings"] = serde_json::json!(timings);
            }
            if let Some(usage) = &self.usage {
                observation["usage"] = usage.clone();
            }
            Some(observation)
        } else {
            None
        };
        ChatResponseChunk {
            content,
            thinking_content,
            done,
            prompt_tokens: self
                .usage
                .as_ref()
                .and_then(|usage| usage.get("prompt_tokens"))
                .and_then(Value::as_u64)
                .and_then(|count| u32::try_from(count).ok()),
            done_reason: done.then(|| self.finish_reason.clone()).flatten(),
            prompt_eval_duration_ns: self
                .timings
                .as_ref()
                .and_then(|timings| timings.get("prompt_ms"))
                .and_then(Value::as_f64)
                .map(|ms| (ms * 1_000_000.0) as u64),
            tool_calls,
            upstream_timings,
        }
    }
}

fn optional_string(value: Option<&Value>, field: &str) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        _ => Err(protocol_error(&format!("{field} is not a string"))),
    }
}

fn validate_usage(usage: &Value) -> Result<()> {
    let count = |field| {
        usage
            .get(field)
            .and_then(Value::as_u64)
            .filter(|count| *count <= u64::from(u32::MAX))
            .ok_or_else(|| protocol_error(&format!("usage.{field} is not a u32 count")))
    };
    let prompt = count("prompt_tokens")?;
    let completion = count("completion_tokens")?;
    let total = count("total_tokens")?;
    if prompt.checked_add(completion) != Some(total) {
        return Err(protocol_error("usage token counts are inconsistent"));
    }
    Ok(())
}

fn validate_timings(timings: &Value) -> Result<()> {
    if !timings.is_object() {
        return Err(protocol_error("timings is not an object"));
    }
    for field in [
        "prompt_ms",
        "predicted_ms",
        "prompt_per_second",
        "predicted_per_second",
    ] {
        if let Some(value) = timings.get(field).filter(|value| !value.is_null()) {
            if !value
                .as_f64()
                .is_some_and(|number| number.is_finite() && number >= 0.0)
            {
                return Err(protocol_error("invalid timing observation"));
            }
        }
    }
    Ok(())
}

fn parse_tool_deltas(value: Option<&Value>) -> Result<Option<Vec<ToolCall>>> {
    let calls = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(calls)) if calls.is_empty() => return Ok(None),
        Some(Value::Array(calls)) => calls,
        _ => return Err(protocol_error("tool_calls is not an array")),
    };
    let mut deltas = Vec::with_capacity(calls.len());
    for call in calls {
        let index = call
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| u32::try_from(index).ok())
            .ok_or_else(|| protocol_error("tool delta requires an explicit u32 index"))?;
        let function = match call.get("function") {
            None | Some(Value::Null) => None,
            Some(Value::Object(function)) => Some(function),
            _ => return Err(protocol_error("tool function is not an object")),
        };
        let tool_type = optional_string(call.get("type"), "tool type")?.unwrap_or_default();
        if !tool_type.is_empty() && tool_type != "function" {
            return Err(protocol_error("unsupported tool type"));
        }
        deltas.push(ToolCall {
            id: optional_string(call.get("id"), "tool id")?.unwrap_or_default(),
            tool_type,
            index: Some(index),
            function: FunctionCall {
                name: optional_string(
                    function.and_then(|value| value.get("name")),
                    "function name",
                )?
                .unwrap_or_default(),
                arguments: optional_string(
                    function.and_then(|value| value.get("arguments")),
                    "function arguments",
                )?
                .unwrap_or_default(),
            },
        });
    }
    Ok(Some(deltas))
}

/// Read a complete SSE frame, retaining bounded residual bytes between calls.
/// Raw EOF and explicit [DONE] are distinct so callers cannot bless truncation.
pub(super) async fn next_sse_event<S, T>(
    stream: &mut S,
    buffer: &mut Vec<u8>,
) -> Option<Result<Option<Value>>>
where
    S: Stream<Item = reqwest::Result<T>> + Unpin,
    T: AsRef<[u8]>,
{
    loop {
        if let Some(end) = frame_end(buffer) {
            if end > MAX_SSE_BUFFER_BYTES {
                return Some(Err(sse_buffer_limit_error()));
            }
            let frame: Vec<u8> = buffer.drain(..end).collect();
            let frame = match std::str::from_utf8(&frame) {
                Ok(frame) => frame,
                Err(error) => return Some(Err(protocol_error(&format!("invalid UTF-8: {error}")))),
            };
            let data: Vec<&str> = frame
                .lines()
                .filter_map(|line| {
                    line.strip_prefix("data:")
                        .map(|value| value.strip_prefix(' ').unwrap_or(value))
                        .or_else(|| (line == "data").then_some(""))
                })
                .collect();
            if data.is_empty() {
                continue;
            }
            let data = data.join("\n");
            if data.trim() == "[DONE]" {
                return Some(Ok(None));
            }
            if data.trim().is_empty() {
                continue;
            }
            return Some(
                serde_json::from_str(&data)
                    .map(Some)
                    .map_err(|error| protocol_error(&format!("invalid SSE JSON: {error}"))),
            );
        }
        match stream.next().await {
            Some(Ok(bytes)) => {
                let bytes = bytes.as_ref();
                if buffer.len().saturating_add(bytes.len()) > MAX_SSE_BUFFER_BYTES {
                    return Some(Err(sse_buffer_limit_error()));
                }
                buffer.extend_from_slice(bytes);
            }
            Some(Err(error)) => {
                return Some(Err(protocol_error(&format!("stream read failed: {error}"))));
            }
            None if buffer.is_empty() => return None,
            None => {
                return Some(Err(protocol_error(
                    "stream ended with an incomplete SSE frame",
                )))
            }
        }
    }
}

fn frame_end(buffer: &[u8]) -> Option<usize> {
    let mut line_start = 0;
    for (index, byte) in buffer.iter().enumerate() {
        if *byte == b'\n' {
            let line = &buffer[line_start..index];
            if line.is_empty() || line == b"\r" {
                return Some(index + 1);
            }
            line_start = index + 1;
        }
    }
    None
}

fn protocol_error(message: &str) -> PowerError {
    PowerError::InferenceFailed(format!("OpenAI upstream protocol error: {message}"))
}

fn sse_buffer_limit_error() -> PowerError {
    protocol_error(&format!(
        "SSE frame buffer exceeded {MAX_SSE_BUFFER_BYTES} bytes"
    ))
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;
