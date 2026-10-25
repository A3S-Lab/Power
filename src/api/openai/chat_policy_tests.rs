#![cfg(test)]
use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;
#[tokio::test]
#[serial_test::serial]
async fn explicit_backend_capability_allows_selection_for_local_container_without_weakening_default(
) {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());
    let mock = MockBackend::success().with_openai_tool_selection();
    let captured = mock.chat_request_capture();
    let state = test_state_with_mock(mock);
    state.registry.register(sample_manifest("test")).unwrap();
    state.mark_loaded("test");
    let app = crate::server::router::build(state.clone());
    for choice in [
        json!("auto"),
        json!("none"),
        json!("required"),
        json!({"type":"function","function":{"name":"lookup"}}),
    ] {
        let wire = json!({"model":"test","messages":[{"role":"user","content":"read record"}],
            "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"tool_choice":choice,"parallel_tool_calls":false});
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(wire.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let actual = captured.lock().unwrap().clone().unwrap();
        assert_eq!(serde_json::to_value(actual.tool_choice).unwrap(), choice);
        assert_eq!(actual.parallel_tool_calls, Some(false));
    }
    assert_eq!(state.metrics.active_requests(), 0);
    std::env::remove_var("A3S_POWER_HOME");
}
