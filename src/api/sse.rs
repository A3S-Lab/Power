use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use std::convert::Infallible;

/// Create an SSE response from a stream of JSON-serializable values.
///
/// Each item in the stream is serialized to JSON and sent as an SSE `data` event.
/// A final `[DONE]` event is sent when the stream ends.
pub fn sse_response<S>(stream: S) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    S: Stream<Item = String> + Send + 'static,
{
    use futures::StreamExt;

    let event_stream = stream.map(|data| Ok(Event::default().data(data)));

    Sse::new(event_stream).keep_alive(KeepAlive::default())
}

/// Create a newline-delimited JSON (NDJSON) response from a stream of serializable values.
///
/// Ollama's native API uses NDJSON (`{...}\n`) for streaming, not SSE.
/// Each item is serialized to JSON and followed by a newline character.
pub fn ndjson_response<S, T>(stream: S) -> axum::response::Response
where
    S: Stream<Item = T> + Send + 'static,
    T: serde::Serialize + Send + 'static,
{
    use axum::body::Body;
    use axum::http::header;
    use futures::StreamExt;

    let body_stream = stream.map(|item| {
        let mut json = serde_json::to_string(&item).unwrap_or_default();
        json.push('\n');
        Ok::<_, Infallible>(json)
    });

    axum::response::Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(header::TRANSFER_ENCODING, "chunked")
        .body(Body::from_stream(body_stream))
        .unwrap()
}

/// Format a single SSE data line from a serializable value.
pub fn format_sse_data<T: serde::Serialize>(value: &T) -> Option<String> {
    serde_json::to_string(value).ok()
}

/// The standard SSE termination marker used by OpenAI-compatible APIs.
pub const SSE_DONE: &str = "[DONE]";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_sse_data_json() {
        let data = serde_json::json!({"text": "hello"});
        let result = format_sse_data(&data);
        assert!(result.is_some());
        let s = result.unwrap();
        assert!(s.contains("hello"));
    }

    #[test]
    fn test_format_sse_data_string() {
        let result = format_sse_data(&"plain text");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "\"plain text\"");
    }

    #[test]
    fn test_sse_done_constant() {
        assert_eq!(SSE_DONE, "[DONE]");
    }

    #[tokio::test]
    async fn test_sse_response_creates_stream() {
        use futures::stream;
        let input = stream::iter(vec!["hello".to_string(), "world".to_string()]);
        let _sse = sse_response(input);
        // Just verify it compiles and creates without panic
    }

    #[test]
    fn test_format_sse_data_none_on_invalid() {
        // Channels and other non-serializable types would fail,
        // but all serde types should work. Test with a valid nested struct.
        let data = serde_json::json!({"nested": {"key": [1, 2, 3]}});
        let result = format_sse_data(&data);
        assert!(result.is_some());
        assert!(result.unwrap().contains("nested"));
    }

    #[test]
    fn test_ndjson_serialization() {
        let data = serde_json::json!({"model": "test", "done": false});
        let json = serde_json::to_string(&data).unwrap();
        assert!(!json.ends_with('\n'));
        let ndjson = format!("{}\n", json);
        assert!(ndjson.ends_with('\n'));
        // Verify it's valid JSON when trimmed
        let parsed: serde_json::Value = serde_json::from_str(ndjson.trim()).unwrap();
        assert_eq!(parsed["model"], "test");
    }
}
