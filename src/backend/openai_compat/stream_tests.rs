#![cfg(test)]

use super::*;

fn chunks(bytes: &[u8], width: usize) -> impl Stream<Item = reqwest::Result<Vec<u8>>> + Unpin {
    futures::stream::iter(
        bytes
            .chunks(width)
            .map(|part| Ok(part.to_vec()))
            .collect::<Vec<_>>(),
    )
}

#[tokio::test]
async fn sse_handles_fragmented_utf8_crlf_and_explicit_done() {
    let mut bytes = chunks(
        "data: {\"text\":\"你好\"}\r\n\r\ndata: [DONE]\r\n\r\n".as_bytes(),
        1,
    );
    let mut buffer = Vec::new();
    let event = next_sse_event(&mut bytes, &mut buffer)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(event["text"], "你好");
    assert!(next_sse_event(&mut bytes, &mut buffer)
        .await
        .unwrap()
        .unwrap()
        .is_none());
    assert!(next_sse_event(&mut bytes, &mut buffer).await.is_none());
}

#[tokio::test]
async fn sse_handles_comments_and_multiline_data_frames() {
    let mut bytes = chunks(
        b": keepalive\n\nevent: message\ndata: {\ndata: \"value\": 7\n data: ignored\ndata: }\n\n",
        3,
    );
    let event = next_sse_event(&mut bytes, &mut Vec::new())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(event["value"], 7);
}

#[tokio::test]
async fn sse_rejects_oversized_input_before_growing_the_buffer() {
    let mut bytes = chunks(
        &vec![b'x'; MAX_SSE_BUFFER_BYTES + 1],
        MAX_SSE_BUFFER_BYTES + 1,
    );
    let mut buffer = Vec::new();
    let error = next_sse_event(&mut bytes, &mut buffer)
        .await
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("buffer exceeded"));
    assert!(buffer.len() <= MAX_SSE_BUFFER_BYTES);
}

#[tokio::test]
async fn sse_rejects_incomplete_frames_invalid_utf8_and_bad_json() {
    for input in [
        b"data: {}".as_slice(),
        b"data: \xff\n\n".as_slice(),
        b"data: {bad}\n\n".as_slice(),
    ] {
        let mut bytes = chunks(input, 1);
        assert!(next_sse_event(&mut bytes, &mut Vec::new())
            .await
            .unwrap()
            .is_err());
    }
}

#[tokio::test]
async fn sse_propagates_transport_read_errors() {
    let error = reqwest::Client::new()
        .get("://invalid")
        .build()
        .unwrap_err();
    let mut bytes = futures::stream::iter(vec![Err::<Vec<u8>, _>(error)]);
    let error = next_sse_event(&mut bytes, &mut Vec::new())
        .await
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("stream read failed"));
}

#[test]
fn tool_deltas_preserve_explicit_indexes_and_empty_continuation_metadata() {
    let calls = serde_json::json!([
        {"index": 1, "id": "call_b", "type": "function", "function": {"name": "write", "arguments": "{\"x\":"}},
        {"index": 0, "function": {"arguments": "42}"}}
    ]);
    let parsed = parse_tool_deltas(Some(&calls)).unwrap().unwrap();
    assert_eq!(parsed[0].index, Some(1));
    assert_eq!(parsed[0].function.name, "write");
    assert_eq!(parsed[1].index, Some(0));
    assert_eq!(parsed[1].id, "");
    assert_eq!(parsed[1].tool_type, "");
    assert_eq!(parsed[1].function.name, "");
    assert_eq!(parsed[1].function.arguments, "42}");
}

#[test]
fn malformed_tool_deltas_and_inconsistent_usage_fail_closed() {
    for calls in [
        serde_json::json!([{"id": "missing_index"}]),
        serde_json::json!([{"index": -1}]),
        serde_json::json!([{"index": 0, "function": {"arguments": {}}}]),
        serde_json::json!([{"index": 0, "function": "wrong"}]),
        serde_json::json!([{"index": 0, "type": "nonsense"}]),
    ] {
        assert!(parse_tool_deltas(Some(&calls)).is_err());
    }
    for usage in [
        serde_json::json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 4}),
        serde_json::json!({"prompt_tokens": -1, "completion_tokens": 2, "total_tokens": 1}),
        serde_json::json!({"prompt_tokens": 1, "completion_tokens": 2}),
    ] {
        assert!(validate_usage(&usage).is_err());
    }
}

#[test]
fn exact_usage_and_nonnegative_timing_observations_are_required() {
    for usage in [
        serde_json::json!({"prompt_tokens":4}),
        serde_json::json!({"prompt_tokens":4,"completion_tokens":1.5,"total_tokens":5}),
        serde_json::json!({"prompt_tokens":4,"completion_tokens":17,"total_tokens":20}),
        serde_json::json!({"prompt_tokens":4294967296u64,"completion_tokens":0,"total_tokens":4294967296u64}),
    ] {
        assert!(validate_usage(&usage).is_err());
    }
    let usage =
        serde_json::json!({"prompt_tokens":1000000,"completion_tokens":17,"total_tokens":1000017});
    assert!(validate_usage(&usage).is_ok());
    assert!(validate_timings(&serde_json::json!({"predicted_per_second":85.0})).is_ok());
    assert!(validate_timings(&serde_json::json!({"predicted_per_second":-1})).is_err());
}
