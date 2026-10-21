#![cfg(test)]

use super::{ChatCompletionMessage, ChatCompletionRequest};
use crate::backend::types::{ChatMessage, MessageContent};
use serde_json::{json, Value};

fn tool_call() -> Value {
    json!({
        "id": "call_sum",
        "type": "function",
        "function": {"name": "bench_sum", "arguments": "{\"a\":19,\"b\":23}"}
    })
}

fn request(message: Value) -> Value {
    json!({"model": "test", "messages": [message]})
}

fn assistant(content: Option<Value>) -> Value {
    let mut message = json!({"role": "assistant", "tool_calls": [tool_call()]});
    if let Some(content) = content {
        message["content"] = content;
    }
    message
}

#[test]
fn assistant_completed_tool_history_accepts_null_and_omitted_content() {
    for content in [Some(Value::Null), None] {
        let request: ChatCompletionRequest =
            serde_json::from_value(request(assistant(content))).unwrap();
        let message = &request.messages[0];
        assert!(matches!(&message.content, MessageContent::Text(text) if text.is_empty()));
        let calls = message.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_sum");
        assert_eq!(calls[0].tool_type, "function");
        assert_eq!(calls[0].function.name, "bench_sum");
        assert_eq!(calls[0].function.arguments, "{\"a\":19,\"b\":23}");
    }
}

#[test]
fn null_normalization_is_confined_to_openai_request_messages() {
    assert!(serde_json::from_value::<MessageContent>(Value::Null).is_err());
    assert!(serde_json::from_value::<ChatMessage>(assistant(Some(Value::Null))).is_err());
    assert!(serde_json::from_value::<ChatCompletionMessage>(assistant(Some(Value::Null))).is_err());
}

#[test]
fn null_or_omitted_content_requires_assistant_and_completed_function_calls() {
    for role in ["user", "system", "developer", "tool", "function", "unknown"] {
        for content in [Some(Value::Null), None] {
            let mut message = assistant(content);
            message["role"] = json!(role);
            assert!(
                serde_json::from_value::<ChatCompletionRequest>(request(message)).is_err(),
                "role {role}"
            );
        }
    }
    for calls in [
        None,
        Some(Value::Null),
        Some(json!([])),
        Some(json!({})),
        Some(json!([null])),
    ] {
        for content in [Some(Value::Null), None] {
            let mut message = assistant(content);
            match calls.clone() {
                Some(calls) => message["tool_calls"] = calls,
                None => {
                    message.as_object_mut().unwrap().remove("tool_calls");
                }
            }
            assert!(serde_json::from_value::<ChatCompletionRequest>(request(message)).is_err());
        }
    }
}

#[test]
fn incomplete_or_malformed_tool_calls_do_not_authorize_null_content() {
    let malformed = [
        json!({"type":"function","function":{"name":"bench_sum","arguments":"{}"}}),
        json!({"id":"call_sum","function":{"name":"bench_sum","arguments":"{}"}}),
        json!({"id":"call_sum","type":"function"}),
        json!({"id":"call_sum","type":"function","function":{"arguments":"{}"}}),
        json!({"id":"call_sum","type":"function","function":{"name":"bench_sum"}}),
        json!({"id":7,"type":"function","function":{"name":"bench_sum","arguments":"{}"}}),
        json!({"id":"call_sum","type":"function","function":{"name":"bench_sum","arguments":{}}}),
        json!({"id":"","type":"function","function":{"name":"bench_sum","arguments":"{}"}}),
        json!({"id":" ","type":"function","function":{"name":"bench_sum","arguments":"{}"}}),
        json!({"id":"call_sum","type":"custom","function":{"name":"bench_sum","arguments":"{}"}}),
        json!({"id":"call_sum","type":"function","function":{"name":"","arguments":"{}"}}),
        json!({"id":"call_sum","type":"function","function":{"name":" ","arguments":"{}"}}),
    ];
    for call in malformed {
        for content in [Some(Value::Null), None] {
            let mut message = assistant(content);
            message["tool_calls"] = json!([call.clone()]);
            assert!(serde_json::from_value::<ChatCompletionRequest>(request(message)).is_err());
        }
    }
    let mut message = assistant(Some(Value::Null));
    message["tool_calls"] = json!([tool_call(), {"id":"incomplete"}]);
    assert!(serde_json::from_value::<ChatCompletionRequest>(request(message)).is_err());
}

#[test]
fn normalization_keeps_typed_message_validation_and_unsupported_fields() {
    for (field, malformed) in [
        ("images", json!(7)),
        ("name", json!([])),
        ("tool_call_id", json!({})),
        ("thinking", json!(false)),
    ] {
        let mut message = assistant(Some(Value::Null));
        message[field] = malformed;
        assert!(serde_json::from_value::<ChatCompletionRequest>(request(message)).is_err());
    }
    let mut message = assistant(Some(Value::Null));
    message["unknown_policy"] = json!("must-not-drop");
    let parsed: ChatCompletionRequest = serde_json::from_value(request(message)).unwrap();
    assert_eq!(
        parsed.messages[0].unsupported["unknown_policy"],
        "must-not-drop"
    );
    assert!(parsed.messages[0].unsupported_fields_message().is_some());
    let normal: ChatCompletionRequest = serde_json::from_value(request(
        json!({"role":"user","content":[{"type":"text","text":"hello"}]}),
    ))
    .unwrap();
    assert_eq!(normal.messages[0].content.text(), "hello");
    let empty: ChatCompletionRequest =
        serde_json::from_value(request(json!({"role":"assistant","content":""}))).unwrap();
    assert_eq!(empty.messages[0].content.text(), "");
}

#[test]
fn request_normalization_does_not_hide_duplicate_typed_fields() {
    for message in [
        r#"{"role":"assistant","role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}"#,
        r#"{"role":"assistant","content":null,"content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}"#,
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"a","id":"b","type":"function","function":{"name":"f","arguments":"{}"}}]}"#,
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","name":"g","arguments":"{}"}}]}"#,
    ] {
        let body = format!("{{\"model\":\"test\",\"messages\":[{message}]}}");
        assert!(serde_json::from_str::<ChatCompletionRequest>(&body).is_err());
    }
}

#[tokio::test]
#[serial_test::serial]
async fn tool_history_null_and_omitted_content_forward_over_real_http() {
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::server::router;

    struct HomeGuard(Option<std::ffi::OsString>);
    impl Drop for HomeGuard {
        fn drop(&mut self) {
            if let Some(value) = self.0.take() {
                std::env::set_var("A3S_POWER_HOME", value);
            } else {
                std::env::remove_var("A3S_POWER_HOME");
            }
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let _home = HomeGuard(std::env::var_os("A3S_POWER_HOME"));
    std::env::set_var("A3S_POWER_HOME", directory.path());
    let mock = MockBackend::success();
    let capture = mock.chat_request_capture();
    let state = test_state_with_mock(mock);
    state.registry.register(sample_manifest("test")).unwrap();
    state.mark_loaded("test");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server =
        tokio::spawn(async move { axum::serve(listener, router::build(state)).await.unwrap() });
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap();
    let url = format!("http://{address}/v1/chat/completions");
    for content in [Some(Value::Null), None] {
        let body = json!({"model":"test", "stream":false, "messages":[
            {"role":"user", "content":"Add 19 and 23."},
            assistant(content),
            {"role":"tool", "tool_call_id":"call_sum", "content":"42"}
        ]});
        let response = client.post(&url).json(&body).send().await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let _ = response.bytes().await.unwrap();
        let forwarded = capture.lock().unwrap().clone().unwrap();
        assert_eq!(forwarded.messages.len(), 3);
        assert!(
            matches!(&forwarded.messages[1].content, MessageContent::Text(text) if text.is_empty())
        );
        assert_eq!(
            forwarded.messages[1].tool_calls.as_ref().unwrap()[0].id,
            "call_sum"
        );
        assert_eq!(
            forwarded.messages[1].tool_calls.as_ref().unwrap()[0]
                .function
                .arguments,
            "{\"a\":19,\"b\":23}"
        );
        assert_eq!(
            forwarded.messages[2].tool_call_id.as_deref(),
            Some("call_sum")
        );
        assert_eq!(forwarded.messages[2].content.text(), "42");
    }
    for invalid in [
        json!({"role":"user","content":null}),
        json!({"role":"assistant","content":null,"tool_calls":[]}),
        json!({"role":"assistant","content":null,"tool_calls":[{"id":"incomplete"}]}),
    ] {
        *capture.lock().unwrap() = None;
        let response = client
            .post(&url)
            .json(&request(invalid))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
        assert!(capture.lock().unwrap().is_none());
    }
    server.abort();
    let _ = server.await;
}
