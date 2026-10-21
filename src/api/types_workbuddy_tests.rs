#![cfg(test)]

use super::{ChatCompletionMessage, ChatCompletionRequest};
use serde_json::{json, Value};

const ANNOTATION_FIELDS: [&str; 12] = [
    "agent",
    "conversationRequestId",
    "startsNewUserRequest",
    "messageId",
    "model",
    "requestModelId",
    "requestModelName",
    "stepSeq",
    "traceId",
    "argumentsDisplayText",
    "rawUsage",
    "usage",
];

fn annotations(message: &mut Value) {
    message["agent"] = json!({"name": "synthetic-workbuddy"});
    message["conversationRequestId"] = json!("synthetic-turn-id");
    message["startsNewUserRequest"] = json!(false);
    message["messageId"] = json!("synthetic-message-id");
    message["model"] = json!("history-model-label-not-a-route");
    message["requestModelId"] = json!("synthetic-request-model");
    message["requestModelName"] = json!("synthetic-request-model-name");
    message["stepSeq"] = json!(3);
    message["traceId"] = json!("synthetic-trace");
    message["argumentsDisplayText"] = json!("CLIENT_DISPLAY_OVERRIDE");
    message["rawUsage"] = json!({"input_tokens":999999,"output_tokens":999999});
    message["usage"] = json!({"prompt_tokens":999999,"completion_tokens":999999});
}

fn history() -> Value {
    json!({"model":"test", "messages":[
        {"role":"user", "content":"Add 19 and 23."},
        {"role":"assistant", "content":null, "tool_calls":[
            {"id":"call_sum", "type":"function", "function":{
                "name":"bench_sum", "arguments":"{\"a\":19,\"b\":23}"
            }}
        ]},
        {"role":"tool", "tool_call_id":"call_sum", "content":"42"},
        {"role":"user", "content":[{"type":"text", "text":"Repeat the result."}]}
    ]})
}

#[test]
fn workbuddy_history_preserves_messages_and_tool_calls_without_annotations() {
    let canonical: ChatCompletionRequest = serde_json::from_value(history()).unwrap();
    let mut body = history();
    for message in body["messages"].as_array_mut().unwrap() {
        annotations(message);
    }
    let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
    assert!(request.unsupported_message_fields_message().is_none());
    assert_eq!(request.model, canonical.model);
    assert_eq!(
        serde_json::to_value(request.messages).unwrap(),
        serde_json::to_value(canonical.messages).unwrap()
    );
}

#[test]
fn workbuddy_annotations_are_opaque_request_only_json() {
    for field in ANNOTATION_FIELDS {
        for value in [
            Value::Null,
            json!(false),
            json!(7),
            json!("annotation"),
            json!([{"nested":true}]),
            json!({"content":"not a prompt"}),
        ] {
            let mut message = json!({"role":"user", "content":"hello"});
            message[field] = value;
            let direct: ChatCompletionMessage = serde_json::from_value(message.clone()).unwrap();
            assert_eq!(direct.unsupported_fields(), vec![field]);
            let request: ChatCompletionRequest =
                serde_json::from_value(json!({"model":"test", "messages":[message]})).unwrap();
            assert!(request.unsupported_message_fields_message().is_none());
            assert_eq!(
                serde_json::to_value(&request.messages[0]).unwrap(),
                json!({"role":"user", "content":"hello"})
            );
        }
    }
}

#[test]
fn workbuddy_annotations_do_not_hide_unknown_fields_or_policy() {
    let mut body = history();
    annotations(&mut body["messages"][1]);
    body["messages"][1]["unknown_policy"] = json!({"required":true});
    let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
    assert!(request
        .unsupported_message_fields_message()
        .unwrap()
        .contains("messages[1]: unsupported message field(s): unknown_policy"));
    for field in ANNOTATION_FIELDS {
        let mut body = history();
        body[field] = json!(true);
        if field == "model" {
            // The root model remains the typed routing field, even though a
            // message-level model is opaque WorkBuddy replay metadata.
            assert!(serde_json::from_value::<ChatCompletionRequest>(body).is_err());
            let mut body = history();
            body["model"] = json!("canonical-route");
            let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
            assert_eq!(request.model, "canonical-route");
            assert!(request.unsupported_fields().is_empty());
        } else {
            let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
            assert_eq!(request.unsupported_fields(), vec![field]);
        }
        let mut body = history();
        body["messages"][3]["content"][0][field] = json!(true);
        let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        assert!(request
            .unsupported_message_fields_message()
            .unwrap()
            .contains(field));
    }
}

#[test]
fn workbuddy_annotations_do_not_authorize_invalid_null_content() {
    for message in [
        json!({"role":"user", "content":null}),
        json!({"role":"assistant", "content":null, "tool_calls":[]}),
        json!({"role":"assistant", "content":null, "tool_calls":[{"id":"bad"}]}),
    ] {
        let mut message = message;
        annotations(&mut message);
        assert!(serde_json::from_value::<ChatCompletionRequest>(
            json!({"model":"test", "messages":[message]})
        )
        .is_err());
    }
}

#[test]
fn workbuddy_duplicates_and_duplicate_typed_fields_are_refused() {
    for field in ANNOTATION_FIELDS {
        let body = format!(
            r#"{{"model":"test","messages":[{{"role":"user","content":"hi","{field}":null,"{field}":{{}}}}]}}"#
        );
        assert!(
            serde_json::from_str::<ChatCompletionRequest>(&body).is_err(),
            "{field}"
        );
    }
    for fields in [
        r#""agent":null,"agent":{}"#,
        r#""conversationRequestId":"a","conversationRequestId":"b""#,
        r#""startsNewUserRequest":true,"startsNewUserRequest":false"#,
        r#""agent":null,"role":"assistant""#,
        r#""agent":null,"content":"duplicate""#,
    ] {
        let body =
            format!(r#"{{"model":"test","messages":[{{"role":"user","content":"hi",{fields}}}]}}"#);
        assert!(
            serde_json::from_str::<ChatCompletionRequest>(&body).is_err(),
            "{fields}"
        );
    }
}

#[tokio::test]
#[serial_test::serial]
async fn workbuddy_histories_and_streaming_pass_real_http_without_losing_tool_calls() {
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
    for stream in [false, true] {
        let mut body = history();
        body["stream"] = json!(stream);
        for message in body["messages"].as_array_mut().unwrap() {
            annotations(message);
        }
        let response = client.post(&url).json(&body).send().await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let text = response.text().await.unwrap();
        assert!(!text.contains("CLIENT_DISPLAY_OVERRIDE"));
        assert!(!text.contains("999999"));
        if stream {
            assert!(text.contains("data: [DONE]"));
        } else {
            let response: Value = serde_json::from_str(&text).unwrap();
            assert!(response["usage"]["prompt_tokens"].as_u64().is_some());
            assert!(response["usage"]["completion_tokens"].as_u64().is_some());
        }
        let forwarded = capture.lock().unwrap().clone().unwrap();
        assert_eq!(forwarded.messages.len(), 4);
        assert_eq!(forwarded.messages[0].content.text(), "Add 19 and 23.");
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
        assert_eq!(forwarded.messages[3].content.text(), "Repeat the result.");
    }
    *capture.lock().unwrap() = None;
    let mut body = history();
    annotations(&mut body["messages"][1]);
    body["messages"][1]["unknown_policy"] = json!(true);
    let response = client.post(&url).json(&body).send().await.unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let error: Value = response.json().await.unwrap();
    assert_eq!(error["error"]["code"], "unsupported_message_fields");
    assert!(capture.lock().unwrap().is_none());
    server.abort();
    let _ = server.await;
}
