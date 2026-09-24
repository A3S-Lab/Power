//! Minimal SSE reader for Prism OpenAI-compatible streaming responses.

use futures::Stream;
use futures::StreamExt;

use crate::error::{PowerError, Result};

const MAX_SSE_BUFFER_BYTES: usize = 8 * 1024 * 1024;

/// Pull the next SSE `data:` event from a byte stream.
///
/// Returns `Ok(Some(json))` for a data line, `Ok(None)` for `[DONE]`,
/// `None` when the byte stream ends, or `Err` on transport/decode failure.
pub(super) async fn next_sse_event<S, T>(
    stream: &mut S,
    buf: &mut Vec<u8>,
) -> Option<Result<Option<serde_json::Value>>>
where
    S: Stream<Item = reqwest::Result<T>> + Unpin,
    T: AsRef<[u8]>,
{
    loop {
        while let Some(nl) = buf.iter().position(|byte| *byte == b'\n') {
            if nl >= MAX_SSE_BUFFER_BYTES {
                return Some(Err(sse_buffer_limit_error()));
            }
            let line: Vec<u8> = buf.drain(..=nl).collect();
            let line = match std::str::from_utf8(&line) {
                Ok(line) => line.trim(),
                Err(error) => {
                    return Some(Err(PowerError::InferenceFailed(format!(
                        "prism SSE event line is not valid UTF-8: {error}"
                    ))));
                }
            };
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                return Some(Ok(None));
            }
            if data.is_empty() {
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(data) {
                Ok(value) => return Some(Ok(Some(value))),
                Err(error) => {
                    return Some(Err(PowerError::InferenceFailed(format!(
                        "prism SSE data event decode failed: {error}"
                    ))));
                }
            }
        }
        if buf.len() > MAX_SSE_BUFFER_BYTES {
            return Some(Err(sse_buffer_limit_error()));
        }
        match stream.next().await {
            Some(Ok(bytes)) => buf.extend_from_slice(bytes.as_ref()),
            Some(Err(error)) => {
                return Some(Err(PowerError::InferenceFailed(format!(
                    "prism stream error: {error}"
                ))));
            }
            None if buf.is_empty() => return None,
            None => {
                return Some(Err(PowerError::InferenceFailed(
                    "prism SSE stream ended with an incomplete event line".into(),
                )));
            }
        }
    }
}

fn sse_buffer_limit_error() -> PowerError {
    PowerError::InferenceFailed(format!(
        "prism SSE event buffer exceeded {MAX_SSE_BUFFER_BYTES} bytes"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[tokio::test]
    async fn parses_data_line_and_done() {
        let chunks = vec![
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n".as_slice()),
            Ok(b"data: [DONE]\n\n".as_slice()),
        ];
        let mut stream = Box::pin(stream::iter(chunks));
        let mut buf = Vec::new();
        let first = next_sse_event(&mut stream, &mut buf).await.unwrap().unwrap();
        assert_eq!(
            first
                .as_ref()
                .unwrap()
                .pointer("/choices/0/delta/content")
                .and_then(|v| v.as_str()),
            Some("hi")
        );
        let done = next_sse_event(&mut stream, &mut buf).await.unwrap().unwrap();
        assert!(done.is_none());
    }
}
