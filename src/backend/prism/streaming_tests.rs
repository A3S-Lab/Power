//! Loopback transport proofs: no model, GPU, downloads, or external service.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Response, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

use super::*;
use crate::backend::types::ChatRequest;

struct BodyProbe {
    receiver: mpsc::Receiver<String>,
    dropped: Option<oneshot::Sender<()>>,
}

impl Stream for BodyProbe {
    type Item = std::result::Result<String, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx).map(|part| part.map(Ok))
    }
}

impl Drop for BodyProbe {
    fn drop(&mut self) {
        if let Some(dropped) = self.dropped.take() {
            let _ = dropped.send(());
        }
    }
}

#[derive(Clone)]
struct FixtureState {
    body: Arc<StdMutex<Option<BodyProbe>>>,
    request: Arc<StdMutex<Option<oneshot::Sender<Value>>>>,
}

struct Fixture {
    base: String,
    body: mpsc::Sender<String>,
    request: oneshot::Receiver<Value>,
    dropped: oneshot::Receiver<()>,
    server: tokio::task::JoinHandle<()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn chat_handler(
    State(state): State<FixtureState>,
    Json(request): Json<Value>,
) -> Response<Body> {
    if let Some(sender) = state.request.lock().unwrap().take() {
        let _ = sender.send(request);
    }
    let body = state.body.lock().unwrap().take().unwrap();
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(body))
        .unwrap()
}

async fn fixture() -> Fixture {
    let (body, receiver) = mpsc::channel(4);
    let (request_tx, request) = oneshot::channel();
    let (dropped_tx, dropped) = oneshot::channel();
    let state = FixtureState {
        body: Arc::new(StdMutex::new(Some(BodyProbe {
            receiver,
            dropped: Some(dropped_tx),
        }))),
        request: Arc::new(StdMutex::new(Some(request_tx))),
    };
    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/chat/completions", post(chat_handler))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Fixture {
        base,
        body,
        request,
        dropped,
        server,
    }
}

async fn backend(base: &str) -> PrismBackend {
    let config = PowerConfig {
        prism_upstream: Some(base.to_owned()),
        ..PowerConfig::default()
    };
    let backend = PrismBackend::new(Arc::new(config));
    backend
        .loaded
        .write()
        .await
        .insert("prism-test".into(), "fixture".into());
    backend
}

fn request(tools: bool, stream: bool) -> ChatRequest {
    let mut request = json!({
        "messages": [{"role": "user", "content": "Write a file"}],
        "stream": stream
    });
    if tools {
        request["tools"] = json!([{
            "type": "function",
            "function": {"name": "write", "parameters": {"type": "object"}}
        }]);
    }
    serde_json::from_value(request).unwrap()
}

fn event(value: Value) -> String {
    format!("data: {value}\n\n")
}

async fn next_chunk(
    stream: &mut Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>,
) -> ChatResponseChunk {
    tokio::time::timeout(Duration::from_secs(3), stream.next())
        .await
        .expect("stream made progress")
        .expect("expected chunk")
        .unwrap()
}

#[tokio::test]
async fn tool_deltas_arrive_before_finish_and_keep_trailing_authoritative_usage() {
    for client_stream in [true, false] {
        let mut fixture = fixture().await;
        let backend = backend(&fixture.base).await;
        let mut stream = backend
            .chat("prism-test", request(true, client_stream))
            .await
            .unwrap();
        assert!(matches!(
            fixture.request.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        fixture
            .body
            .send(event(json!({"choices": [{
                "index": 0, "delta": {"tool_calls": [{
                    "index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "write", "arguments": "{\"path\":"}
                }]}, "finish_reason": null
            }]})))
            .await
            .unwrap();
        // No finish or usage has been supplied: the tool delta must be observable now.
        let first = next_chunk(&mut stream).await;
        let first_call = &first.tool_calls.as_ref().unwrap()[0];
        assert_eq!(first_call.index, Some(0));
        assert_eq!(first_call.id, "call_1");
        assert_eq!(first_call.function.name, "write");
        assert!(!first.done);
        let upstream_request = (&mut fixture.request).await.unwrap();
        assert_eq!(upstream_request["stream"], true);
        assert_eq!(upstream_request["stream_options"]["include_usage"], true);
        assert_eq!(upstream_request["tools"][0]["function"]["name"], "write");
        fixture
            .body
            .send(event(json!({"choices": [{
                "index": 0, "delta": {"tool_calls": [{
                    "index": 0, "function": {"arguments": "\"a.py\"}"}
                }]}, "finish_reason": "tool_calls"
            }]})))
            .await
            .unwrap();
        fixture.body.send(event(json!({
            "choices": [], "usage": {"prompt_tokens": 19, "completion_tokens": 37, "total_tokens": 56},
            "timings": {"prompt_ms": 12.0, "predicted_n": 37, "predicted_ms": 100.0}
        }))).await.unwrap();
        fixture.body.send("data: [DONE]\n\n".into()).await.unwrap();
        let continuation = next_chunk(&mut stream).await;
        let call = &continuation.tool_calls.unwrap()[0];
        assert_eq!(call.id, "");
        assert_eq!(call.function.name, "");
        assert_eq!(call.function.arguments, "\"a.py\"}");
        let last = next_chunk(&mut stream).await;
        assert!(last.done);
        assert_eq!(last.done_reason.as_deref(), Some("tool_calls"));
        assert_eq!(last.prompt_tokens, Some(19));
        let metadata = last.upstream_timings.unwrap();
        assert_eq!(metadata["source"], "upstream-reported");
        assert_eq!(metadata["usage"]["completion_tokens"], 37);
        assert_eq!(metadata["timings"]["predicted_n"], 37);
        assert!(stream.next().await.is_none());
        assert_eq!(backend.spec_metrics.lock().await.requests, 1);
    }
}

#[tokio::test]
async fn absent_usage_stays_absent_instead_of_counting_text_chunks() {
    let fixture = fixture().await;
    let backend = backend(&fixture.base).await;
    fixture
        .body
        .send(event(json!({"choices": [{
            "delta": {"content": "many tokens in one event"}, "finish_reason": "stop"
        }]})))
        .await
        .unwrap();
    fixture.body.send("data: [DONE]\n\n".into()).await.unwrap();
    let mut stream = backend
        .chat("prism-test", request(false, true))
        .await
        .unwrap();
    assert_eq!(
        next_chunk(&mut stream).await.content,
        "many tokens in one event"
    );
    let final_chunk = next_chunk(&mut stream).await;
    assert!(final_chunk.done);
    assert_eq!(final_chunk.prompt_tokens, None);
    assert_eq!(final_chunk.upstream_timings, None);
}

#[tokio::test]
async fn eof_after_finish_is_an_error_without_a_success_chunk() {
    let mut fixture = fixture().await;
    let backend = backend(&fixture.base).await;
    let mut stream = backend
        .chat("prism-test", request(false, true))
        .await
        .unwrap();
    fixture
        .body
        .send(event(json!({"choices": [{
            "delta": {"content": "partial"}, "finish_reason": "stop"
        }]})))
        .await
        .unwrap();
    assert_eq!(next_chunk(&mut stream).await.content, "partial");
    fixture.body.send("data: ".into()).await.unwrap();
    let (replacement, _) = mpsc::channel(1);
    drop(std::mem::replace(&mut fixture.body, replacement));
    let error = tokio::time::timeout(Duration::from_secs(3), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("incomplete SSE frame"));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn dropping_prefill_wait_closes_the_upstream_body() {
    let mut fixture = fixture().await;
    let backend = backend(&fixture.base).await;
    let mut stream = backend
        .chat("prism-test", request(false, true))
        .await
        .unwrap();
    let mut first = Box::pin(stream.next());
    tokio::select! {
        result = &mut first => panic!("unexpected result before any token: {result:?}"),
        result = &mut fixture.request => { result.unwrap(); }
    }
    drop(first);
    drop(stream);
    tokio::time::timeout(Duration::from_secs(3), &mut fixture.dropped)
        .await
        .expect("upstream observed disconnect")
        .unwrap();
}

#[tokio::test]
async fn terminal_protocol_errors_never_fabricate_a_normal_completion() {
    let finish = event(json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}));
    let late_payload = event(json!({"choices": [{"delta": {"content": "late"}}]}));
    for (frames, expected) in [
        (vec![finish.clone()], "before [DONE]"),
        (vec!["data: [DONE]\n\n".into()], "without a finish reason"),
        (vec![finish, late_payload], "after the finish reason"),
    ] {
        let mut fixture = fixture().await;
        let backend = backend(&fixture.base).await;
        for frame in frames {
            fixture.body.send(frame).await.unwrap();
        }
        let (replacement, _) = mpsc::channel(1);
        drop(std::mem::replace(&mut fixture.body, replacement));
        let mut stream = backend
            .chat("prism-test", request(false, true))
            .await
            .unwrap();
        let error = tokio::time::timeout(Duration::from_secs(3), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        assert!(stream.next().await.is_none());
    }
}

#[tokio::test]
async fn unknown_finish_reason_never_becomes_a_success_chunk() {
    let fixture = fixture().await;
    let backend = backend(&fixture.base).await;
    fixture
        .body
        .send(event(json!({"choices": [{
            "delta": {}, "finish_reason": "nonsense"
        }]})))
        .await
        .unwrap();
    fixture.body.send("data: [DONE]\n\n".into()).await.unwrap();
    let mut stream = backend
        .chat("prism-test", request(false, true))
        .await
        .unwrap();
    let error = tokio::time::timeout(Duration::from_secs(3), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("unsupported finish reason"));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn documented_finish_reasons_preserve_the_upstream_outcome() {
    for reason in [
        "stop",
        "length",
        "tool_calls",
        "function_call",
        "content_filter",
    ] {
        let fixture = fixture().await;
        let backend = backend(&fixture.base).await;
        fixture
            .body
            .send(event(
                json!({"choices": [{"delta": {}, "finish_reason": reason}]}),
            ))
            .await
            .unwrap();
        fixture.body.send("data: [DONE]\n\n".into()).await.unwrap();
        let mut stream = backend
            .chat("prism-test", request(false, true))
            .await
            .unwrap();
        let terminal = next_chunk(&mut stream).await;
        assert!(terminal.done);
        assert_eq!(terminal.done_reason.as_deref(), Some(reason));
        assert!(stream.next().await.is_none());
    }
}
