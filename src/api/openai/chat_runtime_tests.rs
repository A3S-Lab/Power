#![cfg(test)]
use super::*;
use crate::backend::types::FunctionCall;
fn chunk() -> ChatResponseChunk {
    ChatResponseChunk {
        content: String::new(),
        thinking_content: None,
        done: false,
        prompt_tokens: None,
        done_reason: None,
        prompt_eval_duration_ns: None,
        tool_calls: None,
        upstream_timings: None,
    }
}
fn call(index: u32, id: &str, name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        tool_type: if id.is_empty() {
            String::new()
        } else {
            "function".into()
        },
        function: FunctionCall {
            name: name.into(),
            arguments: arguments.into(),
        },
        index: Some(index),
    }
}
#[test]
fn meaningful_excludes_role_metadata_and_tool_identity() {
    let mut value = chunk();
    assert!(!meaningful(&value));
    value.tool_calls = Some(vec![call(0, "call_1", "lookup", "")]);
    assert!(!meaningful(&value));
    value.tool_calls = Some(vec![call(0, "", "", "{")]);
    assert!(meaningful(&value));
    value.tool_calls = None;
    value.thinking_content = Some("reason".into());
    assert!(meaningful(&value));
}
#[test]
fn prism_usage_is_authoritative_u64_and_unknown_is_not_zero() {
    let mut accounting = Accounting::new(true);
    let mut value = chunk();
    value.content = "many tokens in a single chunk".into();
    value.prompt_tokens = Some(999);
    accounting.observe(&value);
    assert!(accounting.usage(false).is_none());
    assert_eq!(accounting.metadata(false).unwrap()["source"], "unavailable");
    value.upstream_timings = Some(
        json!({"source":"upstream-reported","usage":{"prompt_tokens":4_294_967_300_u64,"completion_tokens":37,"total_tokens":4_294_967_337_u64}}),
    );
    accounting.observe(&value);
    assert_eq!(accounting.usage(false).unwrap().completion, 37);
    assert_eq!(accounting.usage(false).unwrap().prompt, 4_294_967_300);
    assert!(
        TokenUsage::parse(&json!({"prompt_tokens":1,"completion_tokens":2,"total_tokens":4}))
            .is_none()
    );
    assert!(TokenUsage::parse(
        &json!({"prompt_tokens":u64::MAX,"completion_tokens":1,"total_tokens":0})
    )
    .is_none());
}
#[test]
fn privacy_rounds_nested_usage_metrics_and_removes_upstream_timing_counts() {
    let mut value = chunk();
    value.upstream_timings = Some(
        json!({"source":"upstream-reported","timings":{"prompt_n":137,"predicted_n":73,"predicted_ms":291},"usage":{"prompt_tokens":137,"completion_tokens":73,"total_tokens":210}}),
    );
    let mut accounting = Accounting::new(true);
    accounting.observe(&value);
    let metadata = accounting.metadata(true).unwrap();
    assert!(metadata.get("timings").is_none());
    assert_eq!(metadata["usage"]["prompt_tokens"], 140);
    assert_eq!(metadata["usage"]["completion_tokens"], 70);
    let metrics = Metrics::new();
    accounting.record(&metrics, "test", true);
    let rendered = metrics.render();
    assert!(!rendered.contains(" 137"));
    assert!(!rendered.contains(" 73"));
}
#[test]
fn interleaved_tool_deltas_reconstruct_once_by_explicit_index() {
    let mut tools = ToolCalls::default();
    tools
        .append(vec![
            call(0, "call_a", "lookup", "{\"a\":"),
            call(1, "call_b", "search", "{"),
        ])
        .unwrap();
    tools
        .append(vec![call(1, "", "", "\"q\":1}"), call(0, "", "", "2}")])
        .unwrap();
    let tools = tools.finish().unwrap().unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.arguments, "{\"a\":2}");
    assert_eq!(tools[1].function.arguments, "{\"q\":1}");
    assert!(tools.iter().all(|call| call.index.is_none()));
    let mut invalid = ToolCalls::default();
    invalid
        .append(vec![call(0, "call_a", "lookup", "")])
        .unwrap();
    assert!(invalid
        .append(vec![call(0, "call_other", "lookup", "")])
        .is_err());
}
async fn resources() -> (Resources, AppState, Arc<std::sync::atomic::AtomicU32>) {
    let mock = crate::backend::test_utils::MockBackend::success();
    let cleanup_count = mock.cleanup_count.clone();
    let state = crate::backend::test_utils::test_state_with_mock(mock);
    let backend = state
        .backends
        .find_for_format(&crate::model::manifest::ModelFormat::Gguf)
        .unwrap();
    state.metrics.increment_active_requests();
    let active = ActiveRequest::new(state.metrics.clone());
    let permit = state.limiter.acquire_http().await.unwrap();
    let resources = Resources::new(
        state.clone(),
        backend,
        "test".into(),
        RequestContext::new(None),
        permit,
        active,
        false,
    );
    (resources, state, cleanup_count)
}
fn presentation(stream: bool, suppress: bool) -> Presentation {
    let request =
        serde_json::from_value(json!({"model":"test","messages":[{"role":"user","content":"hi"}]}))
            .unwrap();
    Presentation {
        id: "chatcmpl-test".into(),
        model: "test".into(),
        started: Instant::now() - Duration::from_millis(200),
        stream,
        include_usage: true,
        suppress,
        receipt: crate::api::receipt::chat_receipt(&request).unwrap(),
        receipt_sha256: "a".repeat(64),
    }
}
#[tokio::test]
async fn json_and_sse_report_same_usage_without_chunk_counting_or_privacy_leak() {
    for streaming in [false, true] {
        let (resources, state, _) = resources().await;
        let mut text = chunk();
        text.content = "multiple upstream tokens".into();
        let mut terminal = chunk();
        terminal.done = true;
        terminal.upstream_timings = Some(
            json!({"source":"upstream-reported","timings":{"prompt_n":137,"predicted_n":73},"usage":{"prompt_tokens":137,"completion_tokens":73,"total_tokens":210}}),
        );
        let stream = Box::pin(futures::stream::iter(vec![Ok(text), Ok(terminal)]));
        let response = respond(stream, resources, presentation(streaming, true), true).await;
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value = if streaming {
            let body = String::from_utf8(bytes.to_vec()).unwrap();
            assert!(body.contains("[DONE]"));
            body.lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .find(|value| value.get("usage").is_some())
                .unwrap()
        } else {
            serde_json::from_slice::<Value>(&bytes).unwrap()
        };
        assert_eq!(value["usage"]["completion_tokens"], 70);
        assert_eq!(value["usage"], value["prism_upstream"]["usage"]);
        assert!(value["prism_upstream"].get("timings").is_none());
        let metrics = state.metrics.render();
        let ttft = metrics
            .lines()
            .find_map(|line| line.strip_prefix("power_ttft_seconds_sum{model=\"test\"} "))
            .unwrap()
            .parse::<f64>()
            .unwrap();
        assert!(
            ttft >= 0.2,
            "TTFT must include work before backend stream creation"
        );
        assert_eq!(state.metrics.active_requests(), 0);
        assert_eq!(state.metrics.running_requests(), 0);
    }
}
#[tokio::test]
async fn missing_prism_usage_is_omitted_on_both_surfaces() {
    for streaming in [false, true] {
        let (resources, _, _) = resources().await;
        let mut text = chunk();
        text.content = "hello".into();
        let response = respond(
            Box::pin(futures::stream::iter(vec![Ok(text)])),
            resources,
            presentation(streaming, false),
            true,
        )
        .await;
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(!text.contains("\"usage\":"));
        assert!(text.contains("\"source\":\"unavailable\""));
    }
}
#[tokio::test]
async fn disconnect_drops_upstream_before_releasing_execution_permit_and_cleans_once() {
    struct PendingSource {
        state: AppState,
        dropped: Arc<std::sync::atomic::AtomicBool>,
    }
    impl Stream for PendingSource {
        type Item = crate::error::Result<ChatResponseChunk>;
        fn poll_next(
            self: Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            std::task::Poll::Pending
        }
    }
    impl Drop for PendingSource {
        fn drop(&mut self) {
            assert_eq!(
                self.state.limiter.snapshot().active,
                1,
                "HTTP ownership must be dropped while execution permit remains held"
            );
            self.dropped
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    let (resources, state, cleanup_count) = resources().await;
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let source = Box::pin(PendingSource {
        state: state.clone(),
        dropped: dropped.clone(),
    });
    let response = respond(source, resources, presentation(true, false), true).await;
    let mut body = response.into_body().into_data_stream();
    assert!(body.next().await.unwrap().is_ok()); // Initial role does not start TTFT.
    let mut pending = Box::pin(body.next());
    assert!(futures::poll!(pending.as_mut()).is_pending());
    drop(pending);
    drop(body);
    assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
    for _ in 0..10 {
        if state.metrics.running_requests() == 0 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(state.limiter.snapshot().active, 0);
    assert_eq!(state.metrics.active_requests(), 0);
    assert_eq!(cleanup_count.load(std::sync::atomic::Ordering::SeqCst), 1);
}
#[tokio::test]
async fn upstream_error_never_emits_success_done_or_success_token_metrics() {
    for streaming in [false, true] {
        let (resources, state, cleanup_count) = resources().await;
        let mut text = chunk();
        text.content = "partial".into();
        let source = Box::pin(futures::stream::iter(vec![
            Ok(text),
            Err(crate::error::PowerError::InferenceFailed(
                "truncated upstream".into(),
            )),
        ]));
        let response = respond(source, resources, presentation(streaming, false), true).await;
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("error"));
        assert!(!body.contains("[DONE]"));
        assert!(!body.contains("\"usage\":"));
        assert_eq!(state.metrics.active_requests(), 0);
        assert_eq!(state.metrics.running_requests(), 0);
        assert_eq!(cleanup_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
#[test]
fn split_tool_names_append_without_duplicating_call_identity() {
    let mut tools = ToolCalls::default();
    tools
        .append(vec![call(0, "call_a", "lookup_", "{")])
        .unwrap();
    tools.append(vec![call(0, "", "weather", "}")]).unwrap();
    let calls = tools.finish().unwrap().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_a");
    assert_eq!(calls[0].function.name, "lookup_weather");
    assert_eq!(calls[0].function.arguments, "{}");
}
#[test]
fn continuation_wire_omits_empty_identity_fields_for_openai_sdks() {
    let value = tool_delta(call(1, "", "", "\"weather\"}"));
    assert_eq!(value["index"], 1);
    assert_eq!(value["function"]["arguments"], "\"weather\"}");
    assert!(value.get("id").is_none());
    assert!(value.get("type").is_none());
    assert!(value["function"].get("name").is_none());
    let first = tool_delta(call(1, "call_a", "lookup_", "{"));
    assert_eq!(first["type"], "function");
    assert_eq!(first["id"], "call_a");
    assert_eq!(first["function"]["name"], "lookup_");
}

#[derive(Default)]
struct CapturedAudit(std::sync::Mutex<Vec<AuditEvent>>);
impl crate::server::audit::AuditLogger for CapturedAudit {
    fn log(&self, event: &AuditEvent) {
        self.0.lock().unwrap().push(event.clone());
    }
}

async fn audited_resources(redact: bool) -> (Resources, Arc<CapturedAudit>) {
    let (mut owned, state, _) = resources().await;
    let audit = Arc::new(CapturedAudit::default());
    let state = state
        .with_privacy(Arc::new(crate::tee::privacy::DefaultPrivacyProvider::new(
            redact,
        )))
        .with_audit(audit.clone());
    owned.state = state.clone();
    owned.cleanup.as_mut().unwrap().state = state;
    (owned, audit)
}

#[tokio::test]
async fn failure_audit_applies_error_and_log_privacy_without_hiding_benign_reasons() {
    let (mut owned, audit) = audited_resources(true).await;
    owned.audit_failure("upstream invalid input; prompt: AUDIT_PROMPT_SECRET".into());
    owned.audit_failure(r#"upstream rejected payload {"content":"AUDIT_JSON_SECRET"}"#.into());
    owned.audit_failure("upstream returned 503 Service Unavailable".into());
    let events = audit.0.lock().unwrap();
    let serialized = serde_json::to_string(&*events).unwrap();
    assert!(!serialized.contains("AUDIT_PROMPT_SECRET"));
    assert!(!serialized.contains("AUDIT_JSON_SECRET"));
    assert!(serialized.contains("[REDACTED]"));
    match &events[2].status {
        crate::server::audit::AuditStatus::Failure { reason } => {
            assert_eq!(reason, "upstream returned 503 Service Unavailable");
        }
        _ => panic!("failure audit changed status"),
    }
    drop(events);
    owned.finish().await;
}

#[tokio::test]
async fn streaming_and_buffered_error_audits_do_not_keep_private_upstream_content() {
    for streaming in [false, true] {
        let (owned, audit) = audited_resources(true).await;
        let source = Box::pin(futures::stream::iter(vec![Err(
            crate::error::PowerError::InferenceFailed(
                "upstream invalid timings; content: AUDIT_STREAM_SECRET".into(),
            ),
        )]));
        let response = respond(source, owned, presentation(streaming, true), true).await;
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(!body.contains("AUDIT_STREAM_SECRET"));
        let events = audit.0.lock().unwrap();
        assert_eq!(events.len(), 1);
        let serialized = serde_json::to_string(&*events).unwrap();
        assert!(!serialized.contains("AUDIT_STREAM_SECRET"));
        assert!(serialized.contains("[REDACTED]"));
        assert!(matches!(
            &events[0].status,
            crate::server::audit::AuditStatus::Failure { .. }
        ));
    }
}

#[tokio::test]
async fn privacy_preserves_success_audit_and_unredacted_host_failure_contract() {
    let (mut owned, audit) = audited_resources(false).await;
    owned.audit_failure("upstream returned 503 Service Unavailable".into());
    match &audit.0.lock().unwrap()[0].status {
        crate::server::audit::AuditStatus::Failure { reason } => {
            assert_eq!(reason, "upstream returned 503 Service Unavailable");
        }
        _ => panic!("failure audit changed status"),
    }
    owned.finish().await;
    for streaming in [false, true] {
        let (owned, audit) = audited_resources(true).await;
        let mut text = chunk();
        text.content = "hello".into();
        let mut terminal = chunk();
        terminal.done = true;
        terminal.upstream_timings = Some(json!({
            "source":"upstream-reported",
            "usage":{"prompt_tokens":11,"completion_tokens":5,"total_tokens":16}
        }));
        let source = Box::pin(futures::stream::iter(vec![Ok(text), Ok(terminal)]));
        let response = respond(source, owned, presentation(streaming, true), true).await;
        axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let events = audit.0.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0].status,
            crate::server::audit::AuditStatus::Success
        ));
        assert_eq!(events[0].action, "chat");
        assert_eq!(events[0].model.as_deref(), Some("test"));
        assert_eq!(events[0].tokens, Some(10));
    }
}
