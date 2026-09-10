//! First-principles authenticated HTTP typed-outcome evidence for the product
//! buffered-host + `BackendOwnedPhaseExecutor` + llamacpp ownership/execution
//! pair.
//!
//! Fixture ports only (no GGUF). Proves:
//! - Prefill execute Ready after ownership capture over HTTP
//! - Decode execute Ready NDJSON only after consume + restore +
//!   `ControlledLlamaCppDecodeTokenPort` (not from transfer alone)
//! - Decode without a decode-token port stays non-NDJSON JSON fail-closed
//!
//! Does **not** flip `may_advertise_prefill_decode` / worker `ready_phases`
//! (BackendOwned stays honest-suppressed). Runtime is `execution_admissible`
//! so phase HTTP can run without claiming HSN.

use super::distributed_serving::{
    DistributedDecodeStreamEvent, DistributedDecodeStreamFrame, DistributedPhaseDecision,
    DistributedPhaseResponse, DistributedProtocolErrorCode, DistributedProtocolErrorResponse,
    DistributedResponseChunk, PreparedDecodeResult, PublishedPrefillResult,
    DISTRIBUTED_SERVING_SCHEMA, DISTRIBUTED_SERVING_STREAM_SCHEMA,
};
use crate::backend::BackendRegistry;
use crate::config::PowerConfig;
use crate::model::registry::ModelRegistry;
use crate::server::auth::ApiKeyAuth;
use crate::server::router;
use crate::server::state::AppState;
use crate::serving::{
    BackendOwnedPhaseExecutor, BoundedStateTransferService, BufferedHostLoopbackStateTransfer,
    ControlledLlamaCppDecodeTokenPort, DisaggregatedServingRole, DistributedServingRuntime,
    LlamaCppBackendPhaseExecution, LlamaCppBackendPhaseStateOwnership, PhaseSessionPoolMode,
    PhaseWeightCacheMode, PrefillDecodeExecutionProfile, ServingCompositionPhaseExecution,
    ServingCompositionPhaseExecutor, ServingCompositionStateOwnership, ServingCompositionTransport,
    ServingExecutionProfile, ServingPrivacyMode, SharedFixtureLlamaCppContextStatePort,
    StateTransferProtocol,
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

const SERVICE_KEY: &str = "service-key";
const MODEL: &str = "internal/model-v1";
const ADAPTER_TOKEN: &str = "http-adapter-token";
const ADAPTER_TOKEN_ID: u32 = 11;

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn fixture_snapshot() -> Vec<u8> {
    (0..32).map(|i| ((i * 13 + 7) % 251) as u8).collect()
}

fn llamacpp_profile(role: DisaggregatedServingRole) -> ServingExecutionProfile {
    ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
        role,
        model: MODEL.to_string(),
        model_sha256: digest('1'),
        backend: "llamacpp".to_string(),
        backend_sha256: digest('2'),
        execution_sha256: digest('3'),
        device_sha256: digest('4'),
        layout_sha256: digest('5'),
        peer_set_sha256: digest('6'),
        generation: 7,
        protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
        state_kind: crate::serving::StateKind::KvCache,
        max_state_bytes: 1024,
        max_inflight_transfers: 2,
        transfer_timeout_ms: 30_000,
        cancellation_timeout_ms: 5_000,
        privacy: ServingPrivacyMode::AuthenticatedEncryptedTransport,
        privacy_policy_sha256: digest('7'),
        attestation_policy_sha256: None,
        weight_cache: PhaseWeightCacheMode::SharedWeightHierarchy,
        residency_policy_sha256: None,
        session_pool: PhaseSessionPoolMode::SharedSessionPool,
        session_pool_policy_sha256: None,
        transport: Some(ServingCompositionTransport::BufferedHostLoopback),
        phase_executor: Some(ServingCompositionPhaseExecutor::BackendOwned),
        state_ownership: Some(ServingCompositionStateOwnership::LlamaCpp),
        phase_execution: Some(ServingCompositionPhaseExecution::LlamaCpp),
    })
    .unwrap()
}

fn completion_payload() -> serde_json::Value {
    serde_json::json!({
        "endpoint": "completions",
        "body": {
            "model": MODEL,
            "prompt": "private typed-outcome prompt",
            "stream": true
        }
    })
}

struct ProductPairApp {
    app: Router,
    profile: ServingExecutionProfile,
    epoch: Uuid,
    runtime: Arc<DistributedServingRuntime>,
    decode_port: SharedFixtureLlamaCppContextStatePort,
}

fn product_pair_app(
    role: DisaggregatedServingRole,
    snapshot: Option<Vec<u8>>,
    decode_tokens: Option<Arc<ControlledLlamaCppDecodeTokenPort>>,
    // Decode prepare must advertise the opaque size prefill will capture so
    // publish/consume bindings match over HTTP (decode prepare runs first).
    expected_state_bytes: Option<u64>,
) -> ProductPairApp {
    let profile = llamacpp_profile(role);
    assert!(
        !profile.may_advertise_prefill_decode(),
        "BackendOwned must not advertise P/D (not HSN; decode may still need a token port)"
    );
    let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&profile).unwrap());
    let ownership = Arc::new(LlamaCppBackendPhaseStateOwnership::for_profile(&profile).unwrap());
    let port = match snapshot {
        Some(bytes) => SharedFixtureLlamaCppContextStatePort::with_snapshot(bytes),
        None => SharedFixtureLlamaCppContextStatePort::empty(),
    };
    let mut execution = LlamaCppBackendPhaseExecution::with_port(&profile, Box::new(port.clone()))
        .unwrap()
        .with_ownership(Arc::clone(&ownership))
        .with_transfer(Arc::clone(&transfer));
    if let Some(state_bytes) = expected_state_bytes {
        execution = execution
            .with_captured_state_bytes(state_bytes)
            .expect("expected_state_bytes fits profile max_state_bytes");
    }
    if let Some(tokens) = decode_tokens {
        execution = execution.with_decode_tokens(tokens);
    }
    let execution = Arc::new(execution);
    let executor = Arc::new(
        BackendOwnedPhaseExecutor::pair_with_ownership_and_execution(
            &profile,
            Arc::clone(&transfer),
            ownership,
            execution,
        )
        .unwrap(),
    );

    let config = PowerConfig {
        serving_execution: profile.clone(),
        api_keys: vec![SERVICE_KEY.to_string()],
        ..PowerConfig::default()
    };
    let state = AppState::new(
        Arc::new(ModelRegistry::new()),
        Arc::new(BackendRegistry::new()),
        Arc::new(config),
    );
    let epoch = state.worker_epoch();
    let bounded =
        Arc::new(BoundedStateTransferService::new(profile.clone(), epoch, transfer).unwrap());
    let runtime =
        Arc::new(DistributedServingRuntime::new(profile.clone(), bounded, executor).unwrap());
    assert!(
        runtime.execution_admissible(),
        "Injected + REQUIRED + Ready must be execution-admissible for typed-outcome HTTP"
    );
    assert!(
        !runtime.accepts_work(),
        "may_advertise stays false → ready_phases must stay empty"
    );
    let state = state
        .with_distributed_serving(Arc::clone(&runtime))
        .with_auth(Arc::new(ApiKeyAuth::new(&[SERVICE_KEY.to_string()])));
    ProductPairApp {
        app: router::build(state),
        profile,
        epoch,
        runtime,
        decode_port: port,
    }
}

async fn post_internal(app: &Router, path: &str, document: serde_json::Value) -> Response {
    app.clone()
        .oneshot(
            Request::post(path)
                .header("authorization", format!("Bearer {SERVICE_KEY}"))
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

#[tokio::test]
async fn authenticated_http_llamacpp_prefill_ready_after_capture_and_decode_ndjson_after_restore_adapter(
) {
    let snapshot = fixture_snapshot();
    let expected_bytes = snapshot.len() as u64;
    let prefill = product_pair_app(
        DisaggregatedServingRole::Prefill,
        Some(snapshot.clone()),
        None,
        None,
    );
    let decode_tokens = Arc::new(ControlledLlamaCppDecodeTokenPort::single_completion(
        ADAPTER_TOKEN,
        ADAPTER_TOKEN_ID,
    ));
    let decode = product_pair_app(
        DisaggregatedServingRole::Decode,
        None,
        Some(decode_tokens),
        Some(expected_bytes),
    );
    assert!(decode.decode_port.snapshot().unwrap().is_empty());

    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::seconds(30);

    let prepared = post_internal(
        &decode.app,
        "/internal/v1/distributed-serving/decode/prepare",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": decode.epoch,
            "execution_profile_sha256": decode.profile.sha256().unwrap(),
            "expires_at": expires_at,
            "request": completion_payload()
        }),
    )
    .await;
    assert_eq!(prepared.status(), StatusCode::OK);
    let prepared: DistributedPhaseResponse<PreparedDecodeResult> = parse_json(prepared).await;
    let target = match prepared.outcome {
        DistributedPhaseDecision::Ready { result } => result.target,
        other => panic!("expected Ready decode prepare, got {other:?}"),
    };

    let published = post_internal(
        &prefill.app,
        "/internal/v1/distributed-serving/prefill/execute",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": prefill.epoch,
            "execution_profile_sha256": prefill.profile.sha256().unwrap(),
            "expires_at": expires_at,
            "request": completion_payload(),
            "target": target
        }),
    )
    .await;
    assert_eq!(published.status(), StatusCode::OK);
    let content_type = published
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    assert!(
        content_type.starts_with("application/json"),
        "prefill Ready must be typed JSON, got {content_type}"
    );
    assert!(!content_type.contains("ndjson"));
    let published: DistributedPhaseResponse<PublishedPrefillResult> = parse_json(published).await;
    let source = match published.outcome {
        DistributedPhaseDecision::Ready { result } => result.source,
        other => panic!("expected Ready prefill after capture, got {other:?}"),
    };
    assert_eq!(source.binding.state_bytes, snapshot.len() as u64);

    let streamed = post_internal(
        &decode.app,
        "/internal/v1/distributed-serving/decode/execute",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": decode.epoch,
            "execution_profile_sha256": decode.profile.sha256().unwrap(),
            "source": source
        }),
    )
    .await;
    assert_eq!(streamed.status(), StatusCode::OK);
    assert_eq!(
        streamed.headers()["content-type"],
        "application/x-ndjson",
        "Ready decode after restore+adapter must open NDJSON"
    );
    let bytes = axum::body::to_bytes(streamed.into_body(), usize::MAX)
        .await
        .unwrap();
    let frames = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<DistributedDecodeStreamFrame>(line).unwrap())
        .collect::<Vec<_>>();
    assert!(frames.len() >= 2);
    assert!(matches!(
        frames[0].payload,
        DistributedDecodeStreamEvent::Ready
    ));
    let chunk = frames.iter().find_map(|frame| match &frame.payload {
        DistributedDecodeStreamEvent::Chunk { response, .. } => Some(response),
        _ => None,
    });
    let Some(DistributedResponseChunk::Completions(chunk)) = chunk else {
        panic!("expected completion chunk from ControlledLlamaCppDecodeTokenPort");
    };
    assert_eq!(chunk.text, ADAPTER_TOKEN);
    assert_eq!(chunk.token_id, Some(ADAPTER_TOKEN_ID));
    assert!(frames.iter().all(|frame| {
        frame.schema == DISTRIBUTED_SERVING_STREAM_SCHEMA
            && frame.execution_id == execution_id
            && frame.worker_epoch == decode.epoch
    }));
    assert_eq!(
        decode.decode_port.snapshot().unwrap(),
        snapshot,
        "HTTP Ready NDJSON must follow consume + set_state_data restore, not transfer alone"
    );
    assert!(
        !decode.runtime.accepts_work(),
        "success must not flip may_advertise / ready_phases"
    );
    assert!(prefill.runtime.execution_admissible());
    assert!(decode.runtime.execution_admissible());
}

#[tokio::test]
async fn authenticated_http_llamacpp_decode_without_token_port_is_json_not_ndjson() {
    let snapshot = fixture_snapshot();
    let expected_bytes = snapshot.len() as u64;
    let prefill = product_pair_app(
        DisaggregatedServingRole::Prefill,
        Some(snapshot.clone()),
        None,
        None,
    );
    // Decode restores via fixture port but has no ControlledLlamaCppDecodeTokenPort.
    let decode = product_pair_app(
        DisaggregatedServingRole::Decode,
        None,
        None,
        Some(expected_bytes),
    );
    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::seconds(30);

    let prepared = post_internal(
        &decode.app,
        "/internal/v1/distributed-serving/decode/prepare",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": decode.epoch,
            "execution_profile_sha256": decode.profile.sha256().unwrap(),
            "expires_at": expires_at,
            "request": completion_payload()
        }),
    )
    .await;
    assert_eq!(prepared.status(), StatusCode::OK);
    let prepared: DistributedPhaseResponse<PreparedDecodeResult> = parse_json(prepared).await;
    let target = match prepared.outcome {
        DistributedPhaseDecision::Ready { result } => result.target,
        other => panic!("expected Ready decode prepare, got {other:?}"),
    };

    let published = post_internal(
        &prefill.app,
        "/internal/v1/distributed-serving/prefill/execute",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": prefill.epoch,
            "execution_profile_sha256": prefill.profile.sha256().unwrap(),
            "expires_at": expires_at,
            "request": completion_payload(),
            "target": target
        }),
    )
    .await;
    assert_eq!(published.status(), StatusCode::OK);
    let published: DistributedPhaseResponse<PublishedPrefillResult> = parse_json(published).await;
    let source = match published.outcome {
        DistributedPhaseDecision::Ready { result } => result.source,
        other => panic!("expected Ready prefill, got {other:?}"),
    };

    let response = post_internal(
        &decode.app,
        "/internal/v1/distributed-serving/decode/execute",
        serde_json::json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": decode.epoch,
            "execution_profile_sha256": decode.profile.sha256().unwrap(),
            "source": source
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    assert!(
        content_type.starts_with("application/json"),
        "non-Ready decode must stay typed JSON, got {content_type}"
    );
    assert!(
        !content_type.contains("ndjson"),
        "transfer+restore without decode-token port must not open NDJSON success"
    );
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_text = String::from_utf8(body.to_vec()).unwrap();
    assert!(!body_text.contains("\"event\":\"ready\""));
    assert!(!body_text.contains(ADAPTER_TOKEN));
    // Protocol error JSON (BackendNotAvailable), not an NDJSON stream.
    let error: DistributedProtocolErrorResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(error.code, DistributedProtocolErrorCode::Unavailable);
    assert_eq!(
        decode.decode_port.snapshot().unwrap(),
        snapshot,
        "fail-closed path must still restore opaque bytes before refusing tokens"
    );
}
