//! POST /v1/systemone — Jev-compatible typed decision scoring.

use std::collections::BTreeMap;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use super::openai_error;
use crate::api::systemone_types::{
    validate_systemone_request, SystemOneAnswer, SystemOneHttpRequest, SystemOneHttpResponse,
    SystemOneQuestion, SystemOneState, SystemOneUsage,
};
use crate::backend::types::{
    SystemOneAnswerSpec, SystemOneQuestionSpec, SystemOneRequest, SystemOneResponse,
};
use crate::server::audit::AuditEvent;
use crate::server::auth::AuthId;
use crate::server::request_context::RequestContext;
use crate::server::state::AppState;

/// POST /v1/systemone — score typed decisions from option-label logits.
pub async fn handler(
    State(state): State<AppState>,
    auth_id: Option<axum::Extension<AuthId>>,
    Json(request): Json<SystemOneHttpRequest>,
) -> impl IntoResponse {
    if let Err(message) = validate_systemone_request(&request) {
        return openai_error("invalid_systemone_request", &message).into_response();
    }

    let model_name = request.model.clone();
    let ctx = RequestContext::new(auth_id.map(|a| a.0 .0.clone()));
    state.metrics.increment_active_requests();

    let manifest = match state.registry.get(&model_name) {
        Ok(m) => m,
        Err(_) => {
            state.metrics.decrement_active_requests();
            return openai_error(
                "model_not_found",
                &format!("model '{model_name}' not found"),
            )
            .into_response();
        }
    };

    let request_keep_alive = match super::parse_request_keep_alive(request.keep_alive.as_deref()) {
        Ok(keep_alive) => keep_alive,
        Err(message) => {
            state.metrics.decrement_active_requests();
            return openai_error("invalid_keep_alive", &message).into_response();
        }
    };

    // System One requires option-label logits. Prefer a System One-capable
    // backend (llama.cpp) over format-priority chat backends (mistral.rs),
    // which may claim GGUF but cannot score — and may panic on newer arches.
    let backend = match state.backends.find_for_systemone() {
        Ok(b) => b,
        Err(e) => {
            state.metrics.decrement_active_requests();
            return openai_error("backend_unavailable", &state.sanitize_error(&e.to_string()))
                .into_response();
        }
    };

    let load_result = match crate::api::autoload::ensure_loaded_with_keep_alive(
        &state,
        &model_name,
        &manifest,
        &backend,
        request_keep_alive,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            state.metrics.decrement_active_requests();
            return openai_error("model_load_failed", &state.sanitize_error(&e.to_string()))
                .into_response();
        }
    };
    let unload_after_use = load_result.unload_after_use;

    let state_text = match request.state.render() {
        Ok(text) => text,
        Err(message) => {
            state.metrics.decrement_active_requests();
            return openai_error("invalid_systemone_request", &message).into_response();
        }
    };

    let backend_request = SystemOneRequest {
        state: state_text,
        questions: request
            .questions
            .into_iter()
            .map(|(name, question)| (name, to_backend_question(question)))
            .collect(),
    };

    let _permit = state.limiter.acquire().await;

    match backend.systemone(&model_name, backend_request).await {
        Ok(response) => {
            crate::api::autoload::cleanup_after_request(&model_name, &ctx, &backend).await;
            state.metrics.decrement_active_requests();

            if unload_after_use {
                crate::api::autoload::unload_after_request(&state, &model_name, &backend).await;
            }

            if let Some(ref audit) = state.audit {
                audit.log(&AuditEvent::success(
                    &ctx.request_id,
                    ctx.auth_id.clone(),
                    "systemone",
                    Some(model_name.clone()),
                    Some(ctx.elapsed().as_millis() as u64),
                    Some(u64::from(response.input_tokens)),
                ));
            }

            Json(to_http_response(&model_name, response)).into_response()
        }
        Err(e) => {
            state.metrics.decrement_active_requests();
            if let Some(ref audit) = state.audit {
                audit.log(&AuditEvent::failure(
                    &ctx.request_id,
                    ctx.auth_id.clone(),
                    "systemone",
                    Some(model_name.clone()),
                    e.to_string(),
                ));
            }
            let message = state.sanitize_error(&e.to_string());
            let code = if message.contains("does not support System One") {
                "unsupported_systemone"
            } else {
                "inference_failed"
            };
            openai_error(code, &message).into_response()
        }
    }
}

fn to_backend_question(question: SystemOneQuestion) -> SystemOneQuestionSpec {
    match question {
        SystemOneQuestion::Choice {
            instructions,
            criteria,
        } => SystemOneQuestionSpec::Choice {
            instructions,
            criteria,
        },
        SystemOneQuestion::Noul {
            instructions,
            criteria,
        } => SystemOneQuestionSpec::Noul {
            instructions,
            criteria,
        },
        SystemOneQuestion::Score {
            instructions,
            criteria,
        } => SystemOneQuestionSpec::Score {
            instructions,
            criteria,
        },
    }
}

fn to_http_response(model: &str, response: SystemOneResponse) -> SystemOneHttpResponse {
    let answers: BTreeMap<String, SystemOneAnswer> = response
        .answers
        .into_iter()
        .map(|(name, answer)| (name, to_http_answer(answer)))
        .collect();
    SystemOneHttpResponse {
        model: model.to_string(),
        answers,
        usage: SystemOneUsage {
            input_tokens: response.input_tokens,
            output_tokens: 0,
        },
    }
}

fn to_http_answer(answer: SystemOneAnswerSpec) -> SystemOneAnswer {
    match answer {
        SystemOneAnswerSpec::Choice {
            choice,
            confidence,
            probabilities,
        } => SystemOneAnswer::Choice {
            choice,
            confidence,
            probabilities,
        },
        SystemOneAnswerSpec::Noul { noul } => SystemOneAnswer::Noul { noul },
        SystemOneAnswerSpec::Score {
            score,
            confidence,
            legend,
            probabilities,
        } => SystemOneAnswer::Score {
            score,
            confidence,
            legend,
            probabilities,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::server::router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::util::ServiceExt;

    async fn post_systemone(
        state: AppState,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let app = router::build(state);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/systemone")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        (status, json)
    }

    #[tokio::test]
    async fn handler_rejects_empty_choice_criteria() {
        let state = test_state_with_mock(MockBackend::success());
        let (status, json) = post_systemone(
            state,
            serde_json::json!({
                "model": "test-model",
                "state": "hello",
                "questions": {
                    "topic": {
                        "type": "choice",
                        "instructions": "pick",
                        "criteria": {}
                    }
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "invalid_systemone_request");
    }

    #[tokio::test]
    async fn handler_returns_mock_choice_answer() {
        let state = test_state_with_mock(MockBackend::success());
        state
            .registry
            .register(sample_manifest("test-model"))
            .unwrap();
        state.mark_loaded("test-model");

        let (status, json) = post_systemone(
            state,
            serde_json::json!({
                "model": "test-model",
                "state": "Customer charged twice",
                "questions": {
                    "topic": {
                        "type": "choice",
                        "instructions": "What is the issue about?",
                        "criteria": {
                            "billing": "money problems",
                            "bug": "broken product"
                        }
                    }
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{json}");
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["answers"]["topic"]["type"], "choice");
        assert_eq!(json["answers"]["topic"]["choice"], "billing");
        assert!(
            json["answers"]["topic"]["probabilities"]["billing"]
                .as_f64()
                .unwrap()
                > 0.5
        );
        assert_eq!(json["usage"]["output_tokens"], 0);
    }

    #[tokio::test]
    async fn handler_model_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let (status, json) = post_systemone(
            state,
            serde_json::json!({
                "model": "missing",
                "state": "x",
                "questions": {
                    "urgent": { "type": "noul", "instructions": "Escalate?" }
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(json["error"]["code"], "model_not_found");
    }

    #[tokio::test]
    async fn handler_unsupported_systemone_backend() {
        let state = test_state_with_mock(MockBackend::success().without_systemone());
        state
            .registry
            .register(sample_manifest("test-model"))
            .unwrap();
        state.mark_loaded("test-model");

        let (status, json) = post_systemone(
            state,
            serde_json::json!({
                "model": "test-model",
                "state": "x",
                "questions": {
                    "urgent": { "type": "noul", "instructions": "Escalate?" }
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(json["error"]["code"], "backend_unavailable");
    }

    #[test]
    fn http_request_deserializes_jev_shape() {
        let req: SystemOneHttpRequest = serde_json::from_value(serde_json::json!({
            "model": "jev-like",
            "state": {"ticket": "charged twice"},
            "questions": {
                "topic": {
                    "type": "choice",
                    "instructions": "What?",
                    "criteria": {"billing": "money", "bug": "broken"}
                },
                "urgent": { "type": "noul", "instructions": "Escalate?" },
                "severity": {
                    "type": "score",
                    "instructions": "How bad?",
                    "criteria": ["routine", "urgent", "critical"]
                }
            }
        }))
        .unwrap();
        assert!(matches!(req.state, SystemOneState::Object(_)));
        assert_eq!(req.questions.len(), 3);
    }
}
