use super::distributed_serving::{
    AbortDistributedExecutionResponse, DecodePrepareRequest, DistributedDecodeStreamEvent,
    DistributedDecodeStreamFrame, DistributedPhaseDecision, DistributedPhaseResponse,
    DistributedProtocolErrorCode, DistributedProtocolErrorResponse, DistributedResponseChunk,
    PhaseRequestPayload, PreparedDecodeResult, PublishedPrefillResult,
    DISTRIBUTED_PROMPT_CACHE_KEY_PREFIX, DISTRIBUTED_SERVING_SCHEMA,
    DISTRIBUTED_SERVING_STREAM_SCHEMA,
};
use crate::backend::BackendRegistry;
use crate::config::PowerConfig;
use crate::model::registry::ModelRegistry;
use crate::server::auth::ApiKeyAuth;
use crate::server::router;
use crate::server::state::AppState;
use crate::serving::distributed_serving_tests::support::{profile, runtime, wait_for_count, Calls};
use crate::serving::{
    DisaggregatedServingRole, PhaseRequest, ServingExecutionProfile, StateTransferSource,
    STATE_TRANSFER_SOURCE_SCHEMA,
};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use axum::Router;
use chrono::{Duration, Utc};
use serde::de::DeserializeOwned;
use std::sync::Arc;
use tower::ServiceExt;
use uuid::Uuid;

#[test]
fn decode_prepare_contract_is_closed_versioned_and_content_redacted() {
    let document: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/distributed-serving-v1/decode-prepare.json"
    ))
    .unwrap();

    let request: DecodePrepareRequest =
        serde_json::from_value(document.clone()).expect("valid v1 request");
    let debug = format!("{request:?}");
    assert!(!debug.contains("secret prompt"));
    assert!(debug.contains("[REDACTED]"));
    assert_eq!(
        serde_json::to_value(&request).expect("serialize v1 request"),
        document
    );

    let mut unknown = document;
    unknown["unknown"] = serde_json::json!(true);
    assert!(serde_json::from_value::<DecodePrepareRequest>(unknown).is_err());
}

#[test]
fn decode_stream_golden_fixture_is_closed_and_content_redacted_in_debug() {
    let frames = include_str!("../../tests/fixtures/distributed-serving-v1/decode-stream.ndjson")
        .lines()
        .map(|line| serde_json::from_str::<DistributedDecodeStreamFrame>(line).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(frames.len(), 3);
    assert!(matches!(
        frames[0].payload,
        DistributedDecodeStreamEvent::Ready
    ));
    assert!(!format!("{:?}", frames[1]).contains("token"));
    assert!(format!("{:?}", frames[1]).contains("[REDACTED]"));
    assert!(matches!(
        frames[2].payload,
        DistributedDecodeStreamEvent::Completed { sequence: 1 }
    ));

    let mut unknown: serde_json::Value = serde_json::from_str(
        include_str!("../../tests/fixtures/distributed-serving-v1/decode-stream.ndjson")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    unknown["unknown"] = serde_json::json!(true);
    assert!(serde_json::from_value::<DistributedDecodeStreamFrame>(unknown).is_err());
}

fn base_state(config: PowerConfig) -> AppState {
    AppState::new(
        Arc::new(ModelRegistry::new()),
        Arc::new(BackendRegistry::new()),
        Arc::new(config),
    )
}

fn distributed_app(
    role: DisaggregatedServingRole,
    calls: Arc<Calls>,
) -> (Router, ServingExecutionProfile, Uuid) {
    let profile = profile(role, 1_000);
    let config = PowerConfig {
        serving_execution: profile.clone(),
        api_keys: vec!["service-key".to_string()],
        ..PowerConfig::default()
    };
    let state = base_state(config);
    let epoch = state.worker_epoch();
    let phase_runtime = runtime(&profile, epoch, calls);
    let state = state
        .with_distributed_serving(Arc::new(phase_runtime))
        .with_auth(Arc::new(ApiKeyAuth::new(&["service-key".to_string()])));
    (router::build(state), profile, epoch)
}

async fn post_internal(app: &Router, path: &str, document: serde_json::Value) -> Response {
    app.clone()
        .oneshot(
            Request::post(path)
                .header("authorization", "Bearer service-key")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&document).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn parse_json<T: DeserializeOwned>(response: Response) -> T {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn completion_payload() -> serde_json::Value {
    serde_json::json!({
        "endpoint": "completions",
        "body": {
            "model": "internal/model-v1",
            "prompt": "private prompt",
            "stream": true
        }
    })
}

#[tokio::test]
async fn internal_distributed_routes_always_require_service_authentication() {
    let app = router::build(base_state(PowerConfig::default()));
    let response = app
        .oneshot(
            Request::post("/internal/v1/distributed-serving/decode/prepare")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(response
        .headers()
        .get("access-control-allow-origin")
        .is_none());
}

#[tokio::test]
async fn authenticated_distributed_request_fails_closed_without_a_runtime() {
    let profile = ServingExecutionProfile::default();
    let profile_sha256 = profile.sha256().unwrap();
    let state = base_state(PowerConfig::default())
        .with_auth(Arc::new(ApiKeyAuth::new(&["service-key".to_string()])));
    let body = serde_json::json!({
        "schema": DISTRIBUTED_SERVING_SCHEMA,
        "execution_id": "11111111-1111-4111-8111-111111111111",
        "worker_epoch": state.worker_epoch(),
        "execution_profile_sha256": profile_sha256,
        "expires_at": (chrono::Utc::now() + chrono::Duration::seconds(30)),
        "request": {
            "endpoint": "completions",
            "body": {
                "model": "internal/model-v1",
                "prompt": "secret",
                "stream": true
            }
        }
    });
    let app = router::build(state);
    let response = app
        .oneshot(
            Request::post("/internal/v1/distributed-serving/decode/prepare")
                .header("authorization", "Bearer service-key")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
fn presentation_request_maps_openai_chat_without_exposing_content_to_the_domain_contract() {
    let prompt_cache_key = format!("{}{}", DISTRIBUTED_PROMPT_CACHE_KEY_PREFIX, "a".repeat(64));
    let payload: PhaseRequestPayload = serde_json::from_value(serde_json::json!({
        "endpoint": "chat-completions",
        "body": {
            "model": "internal/model-v1",
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 17,
            "stream": true,
            "prompt_cache_key": prompt_cache_key
        }
    }))
    .expect("valid chat payload");

    let (model, request) = payload
        .into_phase_request(&PowerConfig::default())
        .expect("map chat request");
    assert_eq!(model, "internal/model-v1");
    let PhaseRequest::Chat(request) = request else {
        panic!("expected chat request");
    };
    assert_eq!(request.max_tokens, Some(17));
    assert!(request.stream);
    assert_eq!(
        request.session_id.as_deref(),
        Some(prompt_cache_key.as_str())
    );
}

#[test]
fn presentation_request_rejects_unbounded_or_lifecycle_owning_openai_fields() {
    for body in [
        serde_json::json!({
            "model": "internal/model-v1",
            "messages": [{"role": "user", "content": "hello"}],
            "unknown": true
        }),
        serde_json::json!({
            "model": "internal/model-v1",
            "messages": [{"role": "user", "content": "hello"}],
            "keep_alive": "1h"
        }),
        serde_json::json!({
            "model": "internal/model-v1",
            "messages": [{"role": "user", "content": "hello"}],
            "prompt_cache_key": "raw-user-controlled-key"
        }),
    ] {
        let payload = PhaseRequestPayload::ChatCompletions { body };
        assert!(payload.into_phase_request(&PowerConfig::default()).is_err());
    }
}

#[test]
fn distributed_profile_requires_service_authentication_at_startup() {
    let profile = profile(DisaggregatedServingRole::Decode, 1_000);
    let unauthenticated = PowerConfig {
        serving_execution: profile.clone(),
        ..PowerConfig::default()
    };
    let error = unauthenticated
        .validate()
        .expect_err("distributed execution must not start without service keys");
    assert!(error.to_string().contains("api_keys"));

    PowerConfig {
        serving_execution: profile,
        api_keys: vec!["service-key".to_string()],
        ..PowerConfig::default()
    }
    .validate()
    .expect("a distributed profile with service authentication is valid");
}

#[tokio::test]
async fn authenticated_http_contract_executes_decode_prefill_decode_end_to_end() {
    let decode_calls = Arc::new(Calls::default());
    let prefill_calls = Arc::new(Calls::default());
    let (decode_app, decode_profile, decode_epoch) =
        distributed_app(DisaggregatedServingRole::Decode, decode_calls.clone());
    let (prefill_app, prefill_profile, prefill_epoch) =
        distributed_app(DisaggregatedServingRole::Prefill, prefill_calls.clone());
    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::milliseconds(800);

    let response = post_internal(
        &decode_app,
        "/internal/v1/distributed-serving/decode/prepare",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": decode_epoch,
            "execution_profile_sha256": decode_profile.sha256().unwrap(),
            "expires_at": expires_at,
            "request": completion_payload()
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let prepared: DistributedPhaseResponse<PreparedDecodeResult> = parse_json(response).await;
    let target = match prepared.outcome {
        DistributedPhaseDecision::Ready { result } => result.target,
        other => panic!("expected ready decode target, got {other:?}"),
    };

    let response = post_internal(
        &prefill_app,
        "/internal/v1/distributed-serving/prefill/execute",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": prefill_epoch,
            "execution_profile_sha256": prefill_profile.sha256().unwrap(),
            "expires_at": expires_at,
            "request": completion_payload(),
            "target": target
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let published: DistributedPhaseResponse<PublishedPrefillResult> = parse_json(response).await;
    let source = match published.outcome {
        DistributedPhaseDecision::Ready { result } => result.source,
        other => panic!("expected ready prefill source, got {other:?}"),
    };

    let response = post_internal(
        &decode_app,
        "/internal/v1/distributed-serving/decode/execute",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": decode_epoch,
            "execution_profile_sha256": decode_profile.sha256().unwrap(),
            "source": source
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "application/x-ndjson");
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let frames = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<DistributedDecodeStreamFrame>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(frames.len(), 3);
    assert!(frames.iter().all(|frame| {
        frame.schema == DISTRIBUTED_SERVING_STREAM_SCHEMA
            && frame.execution_id == execution_id
            && frame.worker_epoch == decode_epoch
    }));
    assert!(matches!(
        frames[0].payload,
        DistributedDecodeStreamEvent::Ready
    ));
    match &frames[1].payload {
        DistributedDecodeStreamEvent::Chunk {
            sequence: 0,
            response: DistributedResponseChunk::Completions(chunk),
        } => assert_eq!(chunk.text, "token"),
        other => panic!("expected completion chunk, got {other:?}"),
    }
    assert!(matches!(
        frames[2].payload,
        DistributedDecodeStreamEvent::Completed { sequence: 1 }
    ));
    wait_for_count(&decode_calls.phase_aborts, 1).await;
    assert_eq!(
        decode_calls.values()[..4],
        [
            "phase.prepare",
            "transfer.prepare",
            "transfer.consume",
            "phase.execute"
        ]
    );
    assert_eq!(
        prefill_calls.values()[..3],
        ["phase.prepare", "phase.execute", "transfer.publish"]
    );
}

#[tokio::test]
async fn internal_contract_rejects_stale_worker_and_profile_before_execution() {
    let calls = Arc::new(Calls::default());
    let (app, profile, epoch) = distributed_app(DisaggregatedServingRole::Decode, calls.clone());
    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::milliseconds(800);

    let stale = post_internal(
        &app,
        "/internal/v1/distributed-serving/decode/prepare",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": Uuid::new_v4(),
            "execution_profile_sha256": profile.sha256().unwrap(),
            "expires_at": expires_at,
            "request": completion_payload()
        }),
    )
    .await;
    assert_eq!(stale.status(), StatusCode::CONFLICT);
    let stale: DistributedProtocolErrorResponse = parse_json(stale).await;
    assert_eq!(stale.code, DistributedProtocolErrorCode::StaleWorker);

    let mismatch = post_internal(
        &app,
        "/internal/v1/distributed-serving/decode/prepare",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": epoch,
            "execution_profile_sha256": "9".repeat(64),
            "expires_at": expires_at,
            "request": completion_payload()
        }),
    )
    .await;
    assert_eq!(mismatch.status(), StatusCode::CONFLICT);
    let mismatch: DistributedProtocolErrorResponse = parse_json(mismatch).await;
    assert_eq!(mismatch.code, DistributedProtocolErrorCode::ProfileMismatch);
    assert!(calls.values().is_empty());
}

#[tokio::test]
async fn internal_abort_is_authenticated_bound_and_idempotent() {
    let calls = Arc::new(Calls::default());
    let (app, profile, epoch) = distributed_app(DisaggregatedServingRole::Decode, calls.clone());
    let execution_id = Uuid::new_v4();
    let digest = profile.sha256().unwrap();
    let response = post_internal(
        &app,
        "/internal/v1/distributed-serving/decode/prepare",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": epoch,
            "execution_profile_sha256": digest,
            "expires_at": Utc::now() + Duration::milliseconds(800),
            "request": completion_payload()
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    for _ in 0..2 {
        let response = post_internal(
            &app,
            "/internal/v1/distributed-serving/abort",
            serde_json::json!({
                "schema": DISTRIBUTED_SERVING_SCHEMA,
                "execution_id": execution_id,
                "worker_epoch": epoch,
                "execution_profile_sha256": digest
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let response: AbortDistributedExecutionResponse = parse_json(response).await;
        assert!(response.accepted);
    }
    assert_eq!(
        calls.phase_aborts.load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    assert_eq!(
        calls
            .transfer_aborts
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
}

#[tokio::test]
async fn dropping_the_internal_decode_body_reclaims_runtime_ownership() {
    let calls = Arc::new(Calls::default());
    let (app, profile, epoch) = distributed_app(DisaggregatedServingRole::Decode, calls.clone());
    let execution_id = Uuid::new_v4();
    let digest = profile.sha256().unwrap();
    let expires_at = Utc::now() + Duration::milliseconds(800);
    let prepared = post_internal(
        &app,
        "/internal/v1/distributed-serving/decode/prepare",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": epoch,
            "execution_profile_sha256": digest,
            "expires_at": expires_at,
            "request": completion_payload()
        }),
    )
    .await;
    let prepared: DistributedPhaseResponse<PreparedDecodeResult> = parse_json(prepared).await;
    let target = match prepared.outcome {
        DistributedPhaseDecision::Ready { result } => result.target,
        other => panic!("expected ready decode target, got {other:?}"),
    };
    let source = StateTransferSource {
        schema: STATE_TRANSFER_SOURCE_SCHEMA.to_string(),
        transfer_id: target.transfer_id,
        source_worker_epoch: Uuid::new_v4(),
        destination_worker_epoch: target.destination_worker_epoch,
        binding: target.binding,
        protocol: target.protocol,
        published_at: Utc::now(),
        expires_at: target.expires_at,
        ticket: "source-ticket".to_string(),
    };

    let response = post_internal(
        &app,
        "/internal/v1/distributed-serving/decode/execute",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": epoch,
            "execution_profile_sha256": digest,
            "source": source
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    drop(response);

    wait_for_count(&calls.phase_aborts, 1).await;
}
