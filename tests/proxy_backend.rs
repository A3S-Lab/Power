//! Integration test: ProxyBackend forwards to an upstream OpenAI-compatible
//! server. Spins an in-process mock upstream and drives the proxy against it.

use std::sync::Arc;

use a3s_power::backend::proxy::ProxyBackend;
use a3s_power::backend::types::{ChatMessage, ChatRequest, EmbeddingRequest, MessageContent};
use a3s_power::backend::Backend;
use a3s_power::config::PowerConfig;
use axum::routing::post;
use axum::Router;
use futures::StreamExt;

/// Start a mock upstream server and return its base URL (e.g. http://127.0.0.1:PORT).
async fn start_mock(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn proxy_for(model: &str, upstream: String) -> ProxyBackend {
    let mut config = PowerConfig::default();
    config.proxy_upstreams.insert(model.to_string(), upstream);
    ProxyBackend::new(Arc::new(config))
}

fn chat_req() -> ChatRequest {
    ChatRequest {
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text("hi".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }],
        session_id: None,
        temperature: Some(0.7),
        top_p: None,
        max_tokens: Some(32),
        stop: None,
        stream: true,
        top_k: None,
        min_p: None,
        repeat_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        num_ctx: None,
        mirostat: None,
        mirostat_tau: None,
        mirostat_eta: None,
        tfs_z: None,
        typical_p: None,
        response_format: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        repeat_last_n: None,
        penalize_newline: None,
        num_batch: None,
        num_thread: None,
        num_thread_batch: None,
        flash_attention: None,
        num_gpu: None,
        main_gpu: None,
        use_mmap: None,
        use_mlock: None,
        num_parallel: None,
        images: None,
    }
}

#[tokio::test]
async fn proxy_chat_streams_content_from_upstream() {
    // Mock upstream returns a two-delta SSE stream then [DONE].
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n\
                       data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n\
                       data: [DONE]\n\n";
            ([(axum::http::header::CONTENT_TYPE, "text/event-stream")], sse)
        }),
    );
    let base = start_mock(app).await;
    let backend = proxy_for("mock", base);

    let mut stream = backend.chat("mock", chat_req()).await.unwrap();
    let mut out = String::new();
    let mut saw_done = false;
    while let Some(chunk) = stream.next().await {
        let c = chunk.unwrap();
        out.push_str(&c.content);
        if c.done {
            saw_done = true;
            break;
        }
    }
    assert_eq!(out, "Hello world");
    assert!(saw_done, "proxy must emit a terminal done chunk");
}

#[tokio::test]
async fn proxy_embed_from_upstream() {
    let app = Router::new().route(
        "/v1/embeddings",
        post(|| async {
            axum::Json(serde_json::json!({
                "data": [{ "embedding": [0.1f32, 0.2, 0.3] }]
            }))
        }),
    );
    let base = start_mock(app).await;
    let backend = proxy_for("mock", base);

    let resp = backend
        .embed(
            "mock",
            EmbeddingRequest {
                input: vec!["hi".to_string()],
            },
        )
        .await
        .unwrap();
    assert_eq!(resp.embeddings.len(), 1);
    assert_eq!(resp.embeddings[0].len(), 3);
    assert!((resp.embeddings[0][0] - 0.1).abs() < 1e-6);
}

#[tokio::test]
async fn proxy_unknown_model_errors() {
    let backend = proxy_for("configured", "http://127.0.0.1:1".to_string());
    // A model with no configured upstream must be a clean error, not a panic.
    let result = backend.chat("not-configured", chat_req()).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn proxy_upstream_5xx_surfaces_error() {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async { axum::http::StatusCode::INTERNAL_SERVER_ERROR }),
    );
    let base = start_mock(app).await;
    let backend = proxy_for("mock", base);
    let result = backend.chat("mock", chat_req()).await;
    assert!(result.is_err(), "upstream 5xx should surface as an error");
}
