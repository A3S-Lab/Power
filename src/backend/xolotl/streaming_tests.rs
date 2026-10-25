#![cfg(test)]

//! Xolotl uses the same OpenAI tool protocol for both public response modes.
use super::*;
use axum::{
    extract::{Json, State},
    response::IntoResponse,
    routing::post,
    Router,
};
use futures::TryStreamExt;
use serde_json::{json, Value};
use std::sync::Mutex;

#[derive(Clone)]
struct Fixture {
    requests: Arc<Mutex<Vec<Value>>>,
    include_usage: bool,
}
async fn upstream(
    State(fixture): State<Fixture>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let is_tool_reply = body["messages"].as_array().unwrap().last().unwrap()["role"] == "tool";
    fixture.requests.lock().unwrap().push(body);
    let mut events = if is_tool_reply {
        vec![
            json!({"choices":[{"index":0,"delta":{"content":"record-proof-42"},"finish_reason":null}]}),
        ]
    } else {
        vec![
            json!({"choices":[{"index":0,"delta":{"reasoning_content":"Inspect record."},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_fixture","type":"function","function":{"name":"lookup_record","arguments":""}}]},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"id\":"}}]},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"17}"}}]},"finish_reason":null}]}),
        ]
    };
    events.push(json!({"choices":[{"index":0,"delta":{},"finish_reason":if is_tool_reply {"stop"} else {"tool_calls"}}]}));
    if fixture.include_usage {
        events.push(json!({"choices":[],"usage":{"prompt_tokens":1000000,"completion_tokens":17,"total_tokens":1000017},"timings":{"prompt_ms":2.0}}));
    }
    let mut wire = events
        .iter()
        .map(|event| format!("data: {event}\r\n\r\n"))
        .collect::<String>();
    wire.push_str("data: [DONE]\r\n\r\n");
    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        wire,
    )
        .into_response()
}
async fn fixture_backend(
    include_usage: bool,
) -> (
    tempfile::TempDir,
    XolotlBackend,
    Arc<Mutex<Vec<Value>>>,
    tokio::task::JoinHandle<()>,
) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/chat/completions", post(upstream))
        .with_state(Fixture {
            requests: Arc::clone(&requests),
            include_usage,
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("DeepSeek-V4.1-Flash-1017-XQ.gguf");
    let binary = root.path().join(if cfg!(windows) {
        "ds41_generate.exe"
    } else {
        "ds41_generate"
    });
    let engram = root.path().join("engram");
    let rank = root.path().join("rank-1017-085.bin");
    std::fs::write(&model, b"fixture").unwrap();
    std::fs::write(&binary, b"fixture").unwrap();
    std::fs::write(&rank, b"fixture").unwrap();
    std::fs::create_dir(&engram).unwrap();
    let mut config = PowerConfig::default();
    config.xolotl.enabled = true;
    config.xolotl.bin = Some(binary.display().to_string());
    config.xolotl.model = Some(model.display().to_string());
    config.xolotl.engram = Some(engram.display().to_string());
    config.xolotl.rank = Some(rank.display().to_string());
    config.xolotl.upstream = Some(origin);
    config.xolotl.gpus = vec![6];
    config.xolotl.cuda_architecture = "8.9".into();
    config.xolotl.layer_split = "none".into();
    let backend = XolotlBackend::new(Arc::new(config)).with_devices(vec![CudaDevice {
        index: 6,
        name: "NVIDIA GeForce RTX 4090 D".into(),
        compute_major: 8,
        compute_minor: 9,
        memory_bytes: 24_564 * 1024 * 1024,
    }]);
    let mut manifest = crate::backend::test_utils::sample_manifest("xq-fixture");
    manifest.path = model;
    backend.load(&manifest).await.unwrap();
    (root, backend, requests, server)
}
fn tool_request(stream: bool) -> ChatRequest {
    serde_json::from_value(json!({
        "messages":[{"role":"user","content":"Read record 17."}],
        "tools":[{"type":"function","function":{"name":"lookup_record","parameters":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":false}}}],
        "tool_choice":{"type":"function","function":{"name":"lookup_record"}},
        "parallel_tool_calls":false,"max_tokens":256,"stream":stream
    })).unwrap()
}

#[tokio::test]
async fn tool_calls_and_tool_result_replay_work_for_both_client_modes() {
    let (_root, backend, requests, server) = fixture_backend(true).await;
    for public_stream in [false, true] {
        let chunks = backend
            .chat("xq-fixture", tool_request(public_stream))
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(
            chunks[0].thinking_content.as_deref(),
            Some("Inspect record.")
        );
        let calls = chunks
            .iter()
            .flat_map(|chunk| chunk.tool_calls.iter().flatten())
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].id, "call_fixture");
        assert_eq!(calls[0].function.name, "lookup_record");
        assert!(calls.iter().all(|call| call.index == Some(0)));
        assert_eq!(
            calls
                .iter()
                .map(|call| call.function.arguments.as_str())
                .collect::<String>(),
            "{\"id\":17}"
        );
        let done = chunks.last().unwrap();
        assert!(done.done);
        assert_eq!(done.done_reason.as_deref(), Some("tool_calls"));
        assert_eq!(done.prompt_tokens, Some(1000000));
        assert_eq!(
            done.upstream_timings.as_ref().unwrap()["usage"]["completion_tokens"],
            17
        );
        let mut follow = tool_request(public_stream);
        follow.tool_choice = Some(crate::backend::types::ToolChoice::String("none".into()));
        let mut history = serde_json::to_value(&follow).unwrap();
        history["messages"] = json!([
            {"role":"user","content":"Read record 17."},
            {"role":"assistant","content":"","tool_calls":[{"id":"call_fixture","type":"function","function":{"name":"lookup_record","arguments":"{\"id\":17}"}}]},
            {"role":"tool","tool_call_id":"call_fixture","name":"lookup_record","content":"record-proof-42"}
        ]);
        let follow = serde_json::from_value(history).unwrap();
        let reply = backend
            .chat("xq-fixture", follow)
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(
            reply
                .iter()
                .map(|chunk| chunk.content.as_str())
                .collect::<String>(),
            "record-proof-42"
        );
        assert_eq!(reply.last().unwrap().done_reason.as_deref(), Some("stop"));
    }
    let posted = requests.lock().unwrap();
    assert_eq!(posted.len(), 4);
    for (index, body) in posted.iter().enumerate() {
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["tools"][0]["function"]["name"], "lookup_record");
        assert_eq!(body["parallel_tool_calls"], false);
        if index % 2 == 1 {
            assert_eq!(body["messages"][2]["tool_call_id"], "call_fixture");
            assert_eq!(body["messages"][2]["content"], "record-proof-42");
            assert_eq!(
                body["messages"][1]["tool_calls"][0]["function"]["arguments"],
                "{\"id\":17}"
            );
        }
    }
    server.abort();
}

#[tokio::test]
async fn successful_tool_finish_without_authoritative_usage_is_not_success() {
    let (_root, backend, _requests, server) = fixture_backend(false).await;
    let result = backend
        .chat("xq-fixture", tool_request(true))
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await;
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("omitted exact chat usage"));
    server.abort();
}

#[tokio::test]
async fn dropping_an_unpolled_chat_does_not_start_upstream_generation() {
    let (_root, backend, requests, server) = fixture_backend(true).await;
    let response = backend
        .chat("xq-fixture", tool_request(true))
        .await
        .unwrap();
    drop(response);
    tokio::task::yield_now().await;
    assert!(requests.lock().unwrap().is_empty());
    server.abort();
}
