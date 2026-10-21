//! Chat presentation and request ownership. Prism usage is never inferred from SSE chunks.
use crate::api::receipt::AttestationReceipt;
use crate::backend::types::{ChatResponseChunk, ToolCall};
use crate::backend::Backend;
use crate::server::audit::AuditEvent;
use crate::server::limiter::RequestPermit;
use crate::server::metrics::Metrics;
use crate::server::request_context::RequestContext;
use crate::server::state::AppState;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use zeroize::Zeroize;
type ChatStream = Pin<Box<dyn Stream<Item = crate::error::Result<ChatResponseChunk>> + Send>>;
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(2);
pub(super) fn meaningful(chunk: &ChatResponseChunk) -> bool {
    !chunk.content.is_empty()
        || chunk
            .thinking_content
            .as_ref()
            .is_some_and(|value| !value.is_empty())
        || chunk
            .tool_calls
            .as_ref()
            .is_some_and(|calls| calls.iter().any(|call| !call.function.arguments.is_empty()))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TokenUsage {
    prompt: u64,
    completion: u64,
    total: u64,
}
impl TokenUsage {
    fn parse(value: &Value) -> Option<Self> {
        let result = Self {
            prompt: value.get("prompt_tokens")?.as_u64()?,
            completion: value.get("completion_tokens")?.as_u64()?,
            total: value.get("total_tokens")?.as_u64()?,
        };
        (result.prompt.checked_add(result.completion)? == result.total).then_some(result)
    }
    fn reported(self, suppress: bool) -> Option<Self> {
        if !suppress {
            return Some(self);
        }
        let round =
            |value: u64| -> Option<u64> { u64::try_from(((u128::from(value) + 5) / 10) * 10).ok() };
        let prompt = round(self.prompt)?;
        let completion = round(self.completion)?;
        Some(Self {
            prompt,
            completion,
            total: prompt.checked_add(completion)?,
        })
    }
    fn json(self) -> Value {
        json!({"prompt_tokens":self.prompt,"completion_tokens":self.completion,"total_tokens":self.total})
    }
}
pub(super) struct Accounting {
    prism: bool,
    legacy_prompt: u64,
    legacy_completion: u64,
    usage: Option<TokenUsage>,
    upstream: Option<Value>,
}
impl Accounting {
    pub(super) fn new(prism: bool) -> Self {
        Self {
            prism,
            legacy_prompt: 0,
            legacy_completion: 0,
            usage: None,
            upstream: None,
        }
    }
    pub(super) fn observe(&mut self, chunk: &ChatResponseChunk) {
        if !self.prism {
            if super::chunk_emits_token(chunk) {
                self.legacy_completion = self.legacy_completion.saturating_add(1);
            }
            if let Some(prompt) = chunk.prompt_tokens {
                self.legacy_prompt = u64::from(prompt);
            }
        }
        if let Some(metadata) = &chunk.upstream_timings {
            if let Some(usage) = metadata.get("usage") {
                self.usage = TokenUsage::parse(usage);
            }
            if let Some(fields) = metadata.as_object() {
                let existing = self.upstream.get_or_insert_with(|| json!({}));
                if let Some(existing) = existing.as_object_mut() {
                    for (name, value) in fields {
                        existing.insert(name.clone(), value.clone());
                    }
                }
            }
        }
    }
    fn usage(&self, suppress: bool) -> Option<TokenUsage> {
        let value = if self.prism {
            self.usage
        } else {
            self.legacy_prompt
                .checked_add(self.legacy_completion)
                .map(|total| TokenUsage {
                    prompt: self.legacy_prompt,
                    completion: self.legacy_completion,
                    total,
                })
        };
        value.and_then(|usage| usage.reported(suppress))
    }
    fn metadata(&self, suppress: bool) -> Option<Value> {
        if !self.prism {
            return self.upstream.clone();
        }
        let mut metadata = self.upstream.clone().unwrap_or_else(|| json!({}));
        let fields = metadata.as_object_mut()?;
        fields.remove("usage");
        // Timings expose exact prompt_n/predicted_n and derived rates; omit in privacy mode.
        if suppress {
            fields.remove("timings");
        }
        if let Some(usage) = self.usage(suppress) {
            fields.insert("usage".to_string(), usage.json());
        } else {
            fields.insert("source".to_string(), json!("unavailable"));
        }
        Some(metadata)
    }
    fn record(&self, metrics: &Metrics, model: &str, suppress: bool) {
        if let Some(usage) = self.usage(suppress) {
            metrics.record_tokens(model, "input", usage.prompt);
            metrics.record_tokens(model, "output", usage.completion);
        }
    }
}
fn tool_delta(call: ToolCall) -> Value {
    let mut value = json!({"function":{"arguments":call.function.arguments}});
    if let Some(index) = call.index {
        value["index"] = json!(index);
    }
    if !call.id.is_empty() {
        value["id"] = json!(call.id);
    }
    if !call.tool_type.is_empty() {
        value["type"] = json!(call.tool_type);
    }
    if !call.function.name.is_empty() {
        value["function"]["name"] = json!(call.function.name);
    }
    value
}

#[derive(Default)]
pub(super) struct ToolCalls {
    calls: BTreeMap<u32, ToolCall>,
}
impl ToolCalls {
    pub(super) fn append(&mut self, calls: Vec<ToolCall>) -> Result<(), String> {
        for call in calls {
            if let Some(index) = call.index {
                if !call.id.is_empty()
                    && self.calls.iter().any(|(known_index, existing)| {
                        *known_index != index && existing.id == call.id
                    })
                {
                    return Err("tool-call id is bound to more than one index".to_string());
                }
            }
            let index = call
                .index
                .or_else(|| {
                    self.calls.iter().find_map(|(index, existing)| {
                        (!call.id.is_empty() && existing.id == call.id).then_some(*index)
                    })
                })
                .or_else(|| {
                    (!call.id.is_empty()).then(|| {
                        self.calls
                            .keys()
                            .next_back()
                            .map_or(0, |index| index.saturating_add(1))
                    })
                })
                .ok_or_else(|| "tool-call continuation has neither index nor id".to_string())?;
            if let Some(existing) = self.calls.get_mut(&index) {
                for (old, new) in [
                    (&mut existing.id, &call.id),
                    (&mut existing.tool_type, &call.tool_type),
                ] {
                    if !new.is_empty() {
                        if !old.is_empty() && old != new {
                            return Err("tool-call identity changed during generation".to_string());
                        }
                        if old.is_empty() {
                            old.clone_from(new);
                        }
                    }
                }
                existing.function.name.push_str(&call.function.name);
                existing
                    .function
                    .arguments
                    .push_str(&call.function.arguments);
            } else {
                if self.calls.len() >= 128 {
                    return Err("tool-call count exceeds 128".to_string());
                }
                self.calls.insert(index, call);
            }
        }
        Ok(())
    }
    fn finish(self) -> Result<Option<Vec<ToolCall>>, String> {
        if self.calls.is_empty() {
            return Ok(None);
        }
        let mut calls = self.calls.into_values().collect::<Vec<_>>();
        for call in &mut calls {
            if call.id.is_empty() || call.tool_type != "function" || call.function.name.is_empty() {
                return Err("incomplete generated tool-call identity".to_string());
            }
            call.index = None;
        }
        Ok(Some(calls))
    }
}
pub(super) struct ActiveRequest(Arc<Metrics>);
impl ActiveRequest {
    pub(super) fn new(metrics: Arc<Metrics>) -> Self {
        Self(metrics)
    }
}
impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.0.decrement_active_requests();
    }
}
struct Cleanup {
    state: AppState,
    backend: Arc<dyn Backend>,
    model: String,
    context: RequestContext,
    unload: bool,
    _permit: RequestPermit,
    _active: ActiveRequest,
}
pub(super) struct Resources {
    state: AppState,
    cleanup: Option<Cleanup>,
}
impl Resources {
    pub(super) fn new(
        state: AppState,
        backend: Arc<dyn Backend>,
        model: String,
        context: RequestContext,
        permit: RequestPermit,
        active: ActiveRequest,
        unload: bool,
    ) -> Self {
        Self {
            state: state.clone(),
            cleanup: Some(Cleanup {
                state,
                backend,
                model,
                context,
                unload,
                _permit: permit,
                _active: active,
            }),
        }
    }
    fn state(&self) -> &AppState {
        &self.state
    }
    pub(super) fn audit_failure(&self, message: String) {
        if let (Some(audit), Some(cleanup)) = (&self.state.audit, &self.cleanup) {
            // Error text may echo inference content in either plain prefixes or JSON.
            // Reuse the configured privacy policy while retaining ordinary host reasons.
            let message = self
                .state
                .sanitize_log(&self.state.sanitize_error(&message));
            audit.log(&AuditEvent::failure(
                &cleanup.context.request_id,
                cleanup.context.auth_id.clone(),
                "chat",
                Some(cleanup.model.clone()),
                message,
            ));
        }
    }
    async fn finish(&mut self) {
        if let Some(cleanup) = &self.cleanup {
            bounded_cleanup(cleanup).await;
        }
        self.cleanup.take();
    }
}
async fn bounded_cleanup(cleanup: &Cleanup) {
    let result = tokio::time::timeout(CLEANUP_TIMEOUT, async {
        crate::api::autoload::cleanup_after_request(
            &cleanup.model,
            &cleanup.context,
            &cleanup.backend,
        )
        .await;
        if cleanup.unload {
            crate::api::autoload::unload_after_request(
                &cleanup.state,
                &cleanup.model,
                &cleanup.backend,
            )
            .await;
        }
    })
    .await;
    if result.is_err() {
        tracing::warn!(model = %cleanup.model, "chat resource cleanup deadline elapsed");
    }
}
impl Drop for Resources {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                runtime.spawn(async move {
                    bounded_cleanup(&cleanup).await;
                });
            }
        }
    }
}
/// Explicit drop order: source HTTP future/body, then permit-owning cleanup.
struct OwnedSource {
    stream: Option<ChatStream>,
    resources: Resources,
}
impl Drop for OwnedSource {
    fn drop(&mut self) {
        drop(self.stream.take());
    }
}
pub(super) struct Presentation {
    pub(super) id: String,
    pub(super) model: String,
    pub(super) started: Instant,
    pub(super) stream: bool,
    pub(super) include_usage: bool,
    pub(super) suppress: bool,
    pub(super) receipt: AttestationReceipt,
    pub(super) receipt_sha256: String,
}
pub(super) async fn respond(
    stream: ChatStream,
    resources: Resources,
    presentation: Presentation,
    prism: bool,
) -> Response {
    let owned = OwnedSource {
        stream: Some(stream),
        resources,
    };
    if let Some(padding) = owned.resources.state().timing_padding() {
        tokio::time::sleep(padding).await;
    }
    if presentation.stream {
        streaming(owned, presentation, prism)
    } else {
        buffered(owned, presentation, prism).await
    }
}
fn base(p: &Presentation, created: i64) -> Value {
    json!({"id":p.id,"object":"chat.completion.chunk","created":created,"model":p.model})
}
fn terminal(p: &Presentation, created: i64, accounting: &Accounting) -> Value {
    let mut value = base(p, created);
    value["choices"] = json!([]);
    if p.include_usage {
        if let Some(usage) = accounting.usage(p.suppress) {
            value["usage"] = usage.json();
        }
    }
    if let Some(metadata) = accounting.metadata(p.suppress) {
        value["prism_upstream"] = metadata;
    }
    value["attestation_receipt"] = json!(p.receipt);
    value["attestation_receipt_sha256"] = json!(p.receipt_sha256);
    value
}
struct StreamState {
    owned: OwnedSource,
    presentation: Presentation,
    accounting: Accounting,
    created: i64,
    phase: u8,
    ttft: bool,
    failed: bool,
}
fn streaming(owned: OwnedSource, presentation: Presentation, prism: bool) -> Response {
    let state = StreamState {
        owned,
        presentation,
        accounting: Accounting::new(prism),
        created: chrono::Utc::now().timestamp(),
        phase: 0,
        ttft: false,
        failed: false,
    };
    let output = futures::stream::unfold(state, |mut state| async move {
        let value = match state.phase {
            0 => {
                state.phase = 1;
                let mut value = base(&state.presentation, state.created);
                value["choices"] =
                    json!([{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]);
                value
            }
            1 => match state.owned.stream.as_mut()?.next().await {
                Some(Ok(chunk)) => {
                    if !state.ttft && meaningful(&chunk) {
                        state.owned.resources.state().metrics.record_ttft(
                            &state.presentation.model,
                            state.presentation.started.elapsed().as_secs_f64(),
                        );
                        state.ttft = true;
                    }
                    state.accounting.observe(&chunk);
                    let mut value = base(&state.presentation, state.created);
                    let mut delta = json!({});
                    if !chunk.content.is_empty() {
                        delta["content"] = json!(chunk.content);
                    }
                    if let Some(thinking) = chunk.thinking_content.filter(|text| !text.is_empty()) {
                        delta["reasoning_content"] = json!(thinking);
                    }
                    if let Some(calls) = chunk.tool_calls {
                        delta["tool_calls"] =
                            Value::Array(calls.into_iter().map(tool_delta).collect());
                    }
                    let finish = if chunk.done {
                        Some(chunk.done_reason.unwrap_or_else(|| "stop".to_string()))
                    } else {
                        None
                    };
                    value["choices"] = json!([{"index":0,"delta":delta,"finish_reason":finish}]);
                    value
                }
                Some(Err(error)) => {
                    state.phase = 2;
                    state.failed = true;
                    state.owned.resources.audit_failure(error.to_string());
                    json!({"error":{"type":"server_error","message":state.owned.resources.state().sanitize_error(&error.to_string())}})
                }
                None => {
                    state.phase = 2;
                    terminal(&state.presentation, state.created, &state.accounting)
                }
            },
            2 => {
                drop(state.owned.stream.take());
                let app = state.owned.resources.state();
                app.metrics.record_inference_duration(
                    &state.presentation.model,
                    state.presentation.started.elapsed().as_secs_f64(),
                );
                if !state.failed {
                    state.accounting.record(
                        &app.metrics,
                        &state.presentation.model,
                        state.presentation.suppress,
                    );
                    if let (Some(audit), Some(cleanup)) =
                        (&app.audit, &state.owned.resources.cleanup)
                    {
                        audit.log(&AuditEvent::success(
                            &cleanup.context.request_id,
                            cleanup.context.auth_id.clone(),
                            "chat",
                            Some(state.presentation.model.clone()),
                            Some(cleanup.context.elapsed().as_millis() as u64),
                            state
                                .accounting
                                .usage(state.presentation.suppress)
                                .map(|usage| usage.completion),
                        ));
                    }
                }
                state.owned.resources.finish().await;
                state.phase = 3;
                if state.failed {
                    return None;
                }
                return Some((Ok::<_, Infallible>(Event::default().data("[DONE]")), state));
            }
            _ => return None,
        };
        Some((
            Ok::<_, Infallible>(super::super::sse_json_event(&value)),
            state,
        ))
    });
    Sse::new(output)
        .keep_alive(KeepAlive::default())
        .into_response()
}
async fn buffered(mut owned: OwnedSource, p: Presentation, prism: bool) -> Response {
    let mut accounting = Accounting::new(prism);
    let mut content = String::new();
    let mut thinking = String::new();
    let mut tools = ToolCalls::default();
    let mut finish = "stop".to_string();
    let mut ttft = false;
    loop {
        let result = match owned.stream.as_mut() {
            Some(stream) => stream.next().await,
            None => None,
        };
        let Some(result) = result else {
            break;
        };
        let chunk = match result {
            Ok(chunk) => chunk,
            Err(error) => {
                owned.resources.audit_failure(error.to_string());
                let message = owned.resources.state().sanitize_error(&error.to_string());
                drop(owned.stream.take());
                owned.resources.finish().await;
                return super::super::openai_error("server_error", &message).into_response();
            }
        };
        if !ttft && meaningful(&chunk) {
            owned
                .resources
                .state()
                .metrics
                .record_ttft(&p.model, p.started.elapsed().as_secs_f64());
            ttft = true;
        }
        accounting.observe(&chunk);
        content.push_str(&chunk.content);
        if let Some(text) = chunk.thinking_content {
            thinking.push_str(&text);
        }
        if let Some(reason) = chunk.done_reason {
            finish = reason;
        }
        if let Some(calls) = chunk.tool_calls {
            if let Err(message) = tools.append(calls) {
                return super::super::openai_error("invalid_upstream_tool_calls", &message)
                    .into_response();
            }
        }
    }
    let tools = match tools.finish() {
        Ok(tools) => tools,
        Err(message) => {
            return super::super::openai_error("invalid_upstream_tool_calls", &message)
                .into_response()
        }
    };
    let app = owned.resources.state();
    app.metrics
        .record_inference_duration(&p.model, p.started.elapsed().as_secs_f64());
    accounting.record(&app.metrics, &p.model, p.suppress);
    let mut response = json!({
        "id":p.id,"object":"chat.completion","created":chrono::Utc::now().timestamp(),"model":p.model,
        "choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":finish}],
        "system_fingerprint":"fp_a3s_power","attestation_receipt":p.receipt,"attestation_receipt_sha256":p.receipt_sha256
    });
    if let Some(tools) = tools {
        response["choices"][0]["message"]["tool_calls"] = json!(tools);
    }
    if !thinking.is_empty() {
        response["choices"][0]["message"]["thinking"] = json!(thinking);
    }
    if let Some(usage) = accounting.usage(p.suppress) {
        response["usage"] = usage.json();
    }
    if let Some(metadata) = accounting.metadata(p.suppress) {
        response["prism_upstream"] = metadata;
    }
    if let (Some(audit), Some(cleanup)) = (&app.audit, &owned.resources.cleanup) {
        audit.log(&AuditEvent::success(
            &cleanup.context.request_id,
            cleanup.context.auth_id.clone(),
            "chat",
            Some(p.model.clone()),
            Some(cleanup.context.elapsed().as_millis() as u64),
            accounting.usage(p.suppress).map(|usage| usage.completion),
        ));
    }
    if app.should_redact() {
        content.zeroize();
        thinking.zeroize();
    }
    drop(owned.stream.take());
    owned.resources.finish().await;
    axum::Json(response).into_response()
}
#[cfg(test)]
#[path = "chat_runtime_tests.rs"]
mod tests;
