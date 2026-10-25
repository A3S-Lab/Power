#![cfg(test)]

use super::*;
use serde_json::json;

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "messages": [
            {"role":"system","content":"Use the offered tool."},
            {"role":"user","content":"Read record 17."},
            {"role":"assistant","content":"","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup_record","arguments":"{\"id\":17,\"agent\":\"literal data\"}"}}]},
            {"role":"tool","tool_call_id":"call-1","name":"lookup_record","content":"{\"value\":42}"}
        ],
        "tools":[{"type":"function","function":{"name":"lookup_record","description":"Read one record.","parameters":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":false},"strict":true}}],
        "tool_choice":{"type":"function","function":{"name":"lookup_record"}},
        "parallel_tool_calls":false,
        "max_tokens":256,"temperature":0.0,"top_p":0.9,
        "stream_options":{"include_usage":false}
    })).unwrap()
}

#[test]
fn common_request_preserves_tool_contract_history_and_literal_arguments() {
    let request = request();
    for model in ["prism-target", "deepseek-target"] {
        let body = chat_request_body(model, &request, true).unwrap();
        assert_eq!(body["model"], model);
        assert_eq!(body["stream"], true);
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(
            body["tool_choice"],
            json!({"type":"function","function":{"name":"lookup_record"}})
        );
        assert_eq!(body["tools"][0]["function"]["strict"], true);
        assert_eq!(
            body["tools"][0]["function"]["parameters"],
            json!({"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":false})
        );
        assert_eq!(
            body["messages"],
            json!([
                {"role":"system","content":"Use the offered tool."},
                {"role":"user","content":"Read record 17."},
                {"role":"assistant","content":"","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup_record","arguments":"{\"id\":17,\"agent\":\"literal data\"}"}}]},
                {"role":"tool","tool_call_id":"call-1","name":"lookup_record","content":"{\"value\":42}"}
            ])
        );
        assert_eq!(body["stream_options"]["include_usage"], true);
    }
}

#[test]
fn malformed_stream_options_fail_closed_and_nonstreaming_omits_stream_policy() {
    let mut request = request();
    request.stream_options = Some(json!("invalid"));
    assert!(chat_request_body("target", &request, true).is_err());
    let body = chat_request_body("target", &request, false).unwrap();
    assert_eq!(body["stream"], false);
    assert!(body.get("stream_options").is_none());
}

#[tokio::test]
async fn upstream_validation_preserves_client_errors_and_refuses_fake_streams() {
    use axum::http::{HeaderValue, StatusCode};
    use axum::routing::get;
    let app = axum::Router::new()
        .route(
            "/:status",
            get(
                |axum::extract::Path(code): axum::extract::Path<u16>| async move {
                    (
                        StatusCode::from_u16(code).unwrap(),
                        [(
                            axum::http::header::CONTENT_TYPE,
                            HeaderValue::from_static("Text/Event-Stream; charset=utf-8"),
                        )],
                        "private upstream body must not be copied",
                    )
                },
            ),
        )
        .route(
            "/json",
            get(|| async { axum::Json(serde_json::json!({"choices":[]})) }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let http = reqwest::Client::new();
    for code in [400, 413, 422, 401, 403, 503, 500] {
        let response = http.get(format!("{origin}/{code}")).send().await.unwrap();
        let error = require_chat_stream(response).unwrap_err();
        match code {
            400 | 413 | 422 => assert!(matches!(error, PowerError::InvalidRequest(_))),
            401 | 403 | 503 => assert!(matches!(error, PowerError::BackendNotAvailable(_))),
            _ => assert!(matches!(error, PowerError::InferenceFailed(_))),
        }
        assert!(!error.to_string().contains("private upstream body"));
    }
    let response = http.get(format!("{origin}/200")).send().await.unwrap();
    assert!(require_chat_stream(response).is_ok());
    let response = http.get(format!("{origin}/json")).send().await.unwrap();
    assert!(require_chat_stream(response)
        .unwrap_err()
        .to_string()
        .contains("text/event-stream"));
    server.abort();
}
