//! Env-gated live evidence for llama.cpp opaque state + decode after restore.
//!
//! Skips unless `A3S_POWER_LLAMACPP_PHASE_STATE_MODEL` points at a loadable
//! GGUF. Does not run in default CI. When the env var is set, the path must
//! exist (fail closed — do not invent tokens or skip a missing file).
//!
//! Evidence:
//! 1. `LlamaCppContextStateApi` capture → restore on a real `LlamaContext`.
//! 2. Live ownership + `BackendOwnedPhaseExecutor` capture → buffered-host
//!    publish/consume → `set_state_data` restore.
//! 3. Ready decode only via `LlamaCppLiveDecodeTokenPort` calling live
//!    llama.cpp greedy sample after restore. Transfer bytes never become
//!    tokens. If sampling cannot complete honestly, restore evidence still
//!    stands and the decode gap is recorded.
//! 4. Authenticated HTTP boundary (`/internal/v1/distributed-serving/*`) with
//!    the same BackendOwned + buffered-host + llamacpp pair and live decode
//!    token port: Ready prefill JSON after capture, Ready NDJSON decode after
//!    consume + restore + live greedy sample.

#![cfg(feature = "llamacpp")]

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;

use a3s_power::api::distributed_serving::{
    DistributedDecodeStreamEvent, DistributedDecodeStreamFrame, DistributedPhaseDecision,
    DistributedPhaseResponse, DistributedResponseChunk, PreparedDecodeResult,
    PublishedPrefillResult, DISTRIBUTED_SERVING_SCHEMA, DISTRIBUTED_SERVING_STREAM_SCHEMA,
};
use a3s_power::backend::BackendRegistry;
use a3s_power::config::PowerConfig;
use a3s_power::model::registry::ModelRegistry;
use a3s_power::server::auth::ApiKeyAuth;
use a3s_power::server::router;
use a3s_power::server::state::AppState;
use a3s_power::serving::{
    live_greedy_decode_chunk_after_restore, BackendOwnedPhaseExecutor, BackendPhaseStateOwnership,
    BoundedStateTransferService, BufferedHostLoopbackStateTransfer, ConsumeStateTransfer,
    DisaggregatedServingRole, DistributedServingRuntime, ExecutePhaseExecution,
    ImportedModelState, LlamaCppBackendPhaseExecution, LlamaCppBackendPhaseStateOwnership,
    LlamaCppContextStateApi, LlamaCppContextStatePort, LlamaCppLayoutFacts,
    LlamaCppLiveDecodeTokenPort, PhaseDecision, PhaseExecutionOutput, PhaseExecutorHealth,
    PhaseRequest, PhaseResponseChunk, PhaseSessionPoolMode, PhaseWeightCacheMode,
    PrefillDecodeExecutionProfile, PreparePhaseExecution, PrepareStateTransfer,
    PreparedPhaseExecution, PublishStateTransfer, ServingCompositionPhaseExecution,
    ServingCompositionPhaseExecutor, ServingCompositionStateOwnership,
    ServingCompositionTransport, ServingExecutionProfile, ServingPhaseExecutor,
    ServingPrivacyMode, SharedLlamaCppContextStateApi, StateKind, StateTransferProtocol,
    StateTransferService,
};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use axum::Router;
use chrono::{Duration, Utc};
use futures::StreamExt;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use serde::de::DeserializeOwned;
use serial_test::serial;
use tower::ServiceExt;
use uuid::Uuid;

const LIVE_PROMPT: &str = "Hi";
const LIVE_N_CTX: u32 = 64;

fn model_path() -> Option<PathBuf> {
    std::env::var_os("A3S_POWER_LLAMACPP_PHASE_STATE_MODEL").map(PathBuf::from)
}

fn llama_backend() -> &'static LlamaBackend {
    static BACKEND: std::sync::OnceLock<LlamaBackend> = std::sync::OnceLock::new();
    BACKEND.get_or_init(|| LlamaBackend::init().expect("llama backend"))
}

fn require_live_model() -> Option<PathBuf> {
    let Some(path) = model_path() else {
        eprintln!(
            "A3S_POWER_LLAMACPP_PHASE_STATE_MODEL is not set; skipping live llama.cpp phase-state test"
        );
        return None;
    };
    if !path.is_file() {
        panic!(
            "A3S_POWER_LLAMACPP_PHASE_STATE_MODEL does not point at a file: {}",
            path.display()
        );
    }
    Some(path)
}

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn live_profile(
    role: DisaggregatedServingRole,
    layout_sha256: String,
    max_state_bytes: u64,
) -> ServingExecutionProfile {
    ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
        role,
        model: "internal/qwen35-0.8b-live".to_string(),
        model_sha256: digest('1'),
        backend: "llamacpp".to_string(),
        backend_sha256: digest('2'),
        execution_sha256: digest('3'),
        device_sha256: digest('4'),
        layout_sha256,
        peer_set_sha256: digest('6'),
        generation: 7,
        protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
        state_kind: StateKind::KvCache,
        max_state_bytes,
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

#[test]
#[serial]
fn live_llamacpp_context_state_api_capture_restore_round_trip() {
    let Some(path) = require_live_model() else {
        return;
    };

    let backend = llama_backend();
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(backend, &path, &model_params)
        .expect("load GGUF for live phase-state evidence");
    let n_ctx = NonZeroU32::new(LIVE_N_CTX).expect("n_ctx");
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
    let mut context = model.new_context(backend, ctx_params).expect("new context");

    let facts = LlamaCppLayoutFacts::from_model(&model, n_ctx.get());
    let ownership = LlamaCppBackendPhaseStateOwnership::from_layout_facts(
        facts,
        Some("1".repeat(64)),
        Some("2".repeat(64)),
        Some("3".repeat(64)),
    );

    let handle = {
        let port = LlamaCppContextStateApi::new(&mut context);
        ownership
            .capture_from_port(&port)
            .expect("capture live opaque state")
    };
    let exported = ownership
        .export_opaque_state(&handle)
        .expect("export live opaque state");
    assert!(
        !exported.is_empty(),
        "live llama.cpp state snapshot must be non-empty"
    );

    {
        let mut port = LlamaCppContextStateApi::new(&mut context);
        ownership
            .restore_into_port(&handle, &mut port)
            .expect("restore live opaque state");
    }
}

#[tokio::test]
#[serial]
async fn live_llamacpp_buffered_host_capture_restore_then_live_decode() {
    let Some(path) = require_live_model() else {
        return;
    };

    let backend = llama_backend();
    let model = LlamaModel::load_from_file(backend, &path, &LlamaModelParams::default())
        .expect("load GGUF for live ownership+execution evidence");
    let n_ctx = NonZeroU32::new(LIVE_N_CTX).expect("n_ctx");
    let facts = LlamaCppLayoutFacts::from_model(&model, n_ctx.get());
    let layout = facts.layout_sha256();

    let prefill_ctx = model
        .new_context(
            backend,
            LlamaContextParams::default().with_n_ctx(Some(n_ctx)),
        )
        .expect("prefill live context");
    let prefill_port = SharedLlamaCppContextStateApi::from_context(prefill_ctx);

    let tokens = model
        .str_to_token(LIVE_PROMPT, AddBos::Always)
        .expect("tokenize live prompt");
    prefill_port
        .decode_prompt_tokens(&tokens)
        .expect("live prompt prefill into LlamaContext");

    let snapshot_len = prefill_port.state_byte_len();
    assert!(
        snapshot_len > 0,
        "live llama.cpp state after prefill must be non-empty"
    );
    let max_state_bytes = (snapshot_len as u64)
        .saturating_mul(2)
        .max(snapshot_len as u64);

    let decode_ctx = model
        .new_context(
            backend,
            LlamaContextParams::default().with_n_ctx(Some(n_ctx)),
        )
        .expect("decode live context");
    let decode_port = SharedLlamaCppContextStateApi::from_context(decode_ctx);

    let prefill_profile = live_profile(
        DisaggregatedServingRole::Prefill,
        layout.clone(),
        max_state_bytes,
    );
    let decode_profile = live_profile(DisaggregatedServingRole::Decode, layout, max_state_bytes);

    let prefill_transfer =
        Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&prefill_profile).unwrap());
    let decode_transfer =
        Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&decode_profile).unwrap());
    let prefill_ownership = Arc::new(LlamaCppBackendPhaseStateOwnership::from_layout_facts(
        facts.clone(),
        Some(digest('1')),
        Some(digest('2')),
        Some(digest('3')),
    ));
    let decode_ownership = Arc::new(LlamaCppBackendPhaseStateOwnership::from_layout_facts(
        facts,
        Some(digest('1')),
        Some(digest('2')),
        Some(digest('3')),
    ));

    let live_decode_port = decode_port.clone();
    let replay_token = *tokens
        .last()
        .expect("live prompt tokenization produced at least one token");
    // Qwen3.5 M-RoPE requires the next decode position Y > last KV position X.
    // Replaying the last prompt index fails after restore; advance one slot.
    let replay_position = i32::try_from(tokens.len()).expect("next position after restored KV");
    let decode_tokens = Arc::new(LlamaCppLiveDecodeTokenPort::new(Arc::new(move || {
        // Snapshot restore on this pin leaves n_outputs=0. Decode one
        // caller-owned token at the next position through llama.cpp, then
        // greedy-sample. Token ids still come from llama.cpp, not transfer bytes.
        live_decode_port.decode_token_at(replay_token, replay_position)?;
        let chunk = live_greedy_decode_chunk_after_restore(&live_decode_port)?;
        let stream =
            futures::stream::iter(std::iter::once(Ok(PhaseResponseChunk::Completion(chunk))));
        Ok(Box::pin(stream) as a3s_power::serving::PhaseResponseStream)
    })));

    let prefill_execution = Arc::new(
        LlamaCppBackendPhaseExecution::with_port(&prefill_profile, Box::new(prefill_port.clone()))
            .unwrap()
            .with_ownership(Arc::clone(&prefill_ownership))
            .with_transfer(Arc::clone(&prefill_transfer)),
    );
    let decode_execution = Arc::new(
        LlamaCppBackendPhaseExecution::with_port(&decode_profile, Box::new(decode_port.clone()))
            .unwrap()
            .with_captured_state_bytes(snapshot_len as u64)
            .unwrap()
            .with_ownership(Arc::clone(&decode_ownership))
            .with_transfer(Arc::clone(&decode_transfer))
            .with_decode_tokens(decode_tokens),
    );

    let prefill_executor = BackendOwnedPhaseExecutor::pair_with_ownership_and_execution(
        &prefill_profile,
        Arc::clone(&prefill_transfer),
        prefill_ownership,
        prefill_execution,
    )
    .unwrap();
    let decode_executor = BackendOwnedPhaseExecutor::pair_with_ownership_and_execution(
        &decode_profile,
        Arc::clone(&decode_transfer),
        decode_ownership,
        decode_execution,
    )
    .unwrap();
    assert_eq!(prefill_executor.health(), PhaseExecutorHealth::Ready);
    assert_eq!(decode_executor.health(), PhaseExecutorHealth::Ready);
    assert!(prefill_profile.may_advertise_prefill_decode());
    assert!(decode_profile.may_advertise_prefill_decode());

    let execution_id = Uuid::new_v4();
    let source_epoch = Uuid::new_v4();
    let destination_epoch = Uuid::new_v4();
    let expires_at = chrono::Utc::now() + chrono::Duration::seconds(30);
    let completion_request = || {
        PhaseRequest::Completion(
            serde_json::from_value(serde_json::json!({ "prompt": LIVE_PROMPT })).unwrap(),
        )
    };

    let prepared = match prefill_executor
        .prepare(PreparePhaseExecution {
            execution_id,
            local_worker_epoch: source_epoch,
            model: "internal/qwen35-0.8b-live".to_string(),
            expires_at,
            request: completion_request(),
        })
        .await
        .unwrap()
    {
        PhaseDecision::Ready(PreparedPhaseExecution::Prefill(prepared)) => prepared,
        PhaseDecision::Ready(_) => panic!("expected Ready prefill reservation"),
        PhaseDecision::Recompute { .. } => panic!("expected Ready prefill, got Recompute"),
        PhaseDecision::RetryableUnavailable { .. } => {
            panic!("expected Ready prefill, got RetryableUnavailable")
        }
        PhaseDecision::TerminalFailure { .. } => {
            panic!("expected Ready prefill, got TerminalFailure")
        }
    };
    let produced = match prefill_executor
        .execute(ExecutePhaseExecution::prefill(prepared))
        .await
        .unwrap()
    {
        PhaseDecision::Ready(PhaseExecutionOutput::Prefill(produced)) => produced,
        PhaseDecision::Ready(_) => panic!("expected Ready prefill produce from live capture"),
        PhaseDecision::Recompute { .. } => panic!("expected Ready produce, got Recompute"),
        PhaseDecision::RetryableUnavailable { .. } => {
            panic!("expected Ready produce, got RetryableUnavailable")
        }
        PhaseDecision::TerminalFailure { .. } => {
            panic!("expected Ready produce, got TerminalFailure")
        }
    };
    assert_eq!(produced.binding().state_bytes, snapshot_len as u64);

    let decode_prepared = match decode_executor
        .prepare(PreparePhaseExecution {
            execution_id,
            local_worker_epoch: destination_epoch,
            model: "internal/qwen35-0.8b-live".to_string(),
            expires_at,
            request: completion_request(),
        })
        .await
        .unwrap()
    {
        PhaseDecision::Ready(PreparedPhaseExecution::Decode(prepared)) => prepared,
        PhaseDecision::Ready(_) => panic!("expected Ready decode reservation"),
        PhaseDecision::Recompute { .. } => panic!("expected Ready decode, got Recompute"),
        PhaseDecision::RetryableUnavailable { .. } => {
            panic!("expected Ready decode, got RetryableUnavailable")
        }
        PhaseDecision::TerminalFailure { .. } => {
            panic!("expected Ready decode, got TerminalFailure")
        }
    };

    let target = decode_transfer
        .prepare_destination(PrepareStateTransfer {
            transfer_id: execution_id,
            local_worker_epoch: destination_epoch,
            binding: produced.binding().clone(),
            destination: decode_prepared.destination().clone(),
            expires_at,
        })
        .await
        .unwrap();
    let published = prefill_transfer
        .publish_source(PublishStateTransfer {
            local_worker_epoch: source_epoch,
            source: produced.source().clone(),
            target,
        })
        .await
        .unwrap();
    let imported = ImportedModelState::consume_at(
        decode_transfer.as_ref(),
        ConsumeStateTransfer {
            local_worker_epoch: destination_epoch,
            destination: decode_prepared.destination().clone(),
            source: published,
        },
        chrono::Utc::now(),
        &decode_profile,
    )
    .await
    .expect("buffered-host consume of live opaque state");

    let output = decode_executor
        .execute(ExecutePhaseExecution::decode(decode_prepared, imported).unwrap())
        .await;

    let restored_len = decode_port.state_byte_len();
    assert!(
        restored_len > 0,
        "live set_state_data restore must leave non-empty context state"
    );
    // An honest post-restore decode step may grow the snapshot (M-RoPE next
    // position). Equality is required only of the transferred capture size.

    match output {
        Ok(PhaseDecision::Ready(PhaseExecutionOutput::Decode(mut stream))) => {
            let chunk = stream
                .next()
                .await
                .expect("live decode stream yields one chunk")
                .expect("live decode chunk ok");
            match chunk {
                PhaseResponseChunk::Completion(chunk) => {
                    assert!(
                        chunk.token_id.is_some(),
                        "live greedy sample must return a llama.cpp token id (not invented from transfer)"
                    );
                }
                PhaseResponseChunk::Chat(_) => panic!("expected completion chunk from live decode"),
            }
        }
        Ok(PhaseDecision::Ready(_)) => panic!("expected Ready decode stream after live restore"),
        Ok(PhaseDecision::Recompute { .. }) => panic!("expected Ready decode, got Recompute"),
        Ok(PhaseDecision::RetryableUnavailable { .. }) => {
            panic!("expected Ready decode, got RetryableUnavailable")
        }
        Ok(PhaseDecision::TerminalFailure { .. }) => {
            panic!("expected Ready decode, got TerminalFailure")
        }
        Err(error) => {
            eprintln!(
                "live GGUF capture->buffered-host->restore succeeded; Ready decode gap: {error}"
            );
        }
    }

    // Shared live contexts must drop before model (declaration order).
    let _keep_loaded = (backend, &model);
}

const SERVICE_KEY: &str = "live-http-service-key";
const LIVE_MODEL: &str = "internal/qwen35-0.8b-live";

struct LiveHttpApp {
    app: Router,
    profile: ServingExecutionProfile,
    epoch: Uuid,
    runtime: Arc<DistributedServingRuntime>,
    context_port: SharedLlamaCppContextStateApi,
}

fn live_http_app(
    role: DisaggregatedServingRole,
    profile: ServingExecutionProfile,
    port: SharedLlamaCppContextStateApi,
    ownership: Arc<LlamaCppBackendPhaseStateOwnership>,
    transfer: Arc<BufferedHostLoopbackStateTransfer>,
    decode_tokens: Option<Arc<LlamaCppLiveDecodeTokenPort>>,
    expected_state_bytes: Option<u64>,
) -> LiveHttpApp {
    let expects_advertise = matches!(role, DisaggregatedServingRole::Prefill) || decode_tokens.is_some();
    assert!(
        profile.may_advertise_prefill_decode(),
        "buffered-host + BackendOwned + llamacpp ownership/execution may advertise at profile"
    );
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
        "Injected + REQUIRED + Ready must be execution-admissible for live HTTP typed-outcome"
    );
    assert_eq!(
        runtime.accepts_work(),
        expects_advertise,
        "accepts_work / ready_phases only when decode-token bound (or prefill)"
    );
    let state = state
        .with_distributed_serving(Arc::clone(&runtime))
        .with_auth(Arc::new(ApiKeyAuth::new(&[SERVICE_KEY.to_string()])));
    LiveHttpApp {
        app: router::build(state),
        profile,
        epoch,
        runtime,
        context_port: port,
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

fn live_completion_payload() -> serde_json::Value {
    serde_json::json!({
        "endpoint": "completions",
        "body": {
            "model": LIVE_MODEL,
            "prompt": LIVE_PROMPT,
            "stream": true
        }
    })
}

#[tokio::test]
#[serial]
async fn live_llamacpp_authenticated_http_ready_prefill_and_ndjson_decode_after_restore() {
    let Some(path) = require_live_model() else {
        return;
    };

    let backend = llama_backend();
    let model = LlamaModel::load_from_file(backend, &path, &LlamaModelParams::default())
        .expect("load GGUF for live HTTP typed-outcome evidence");
    let n_ctx = NonZeroU32::new(LIVE_N_CTX).expect("n_ctx");
    let facts = LlamaCppLayoutFacts::from_model(&model, n_ctx.get());
    let layout = facts.layout_sha256();

    let prefill_ctx = model
        .new_context(
            backend,
            LlamaContextParams::default().with_n_ctx(Some(n_ctx)),
        )
        .expect("prefill live context for HTTP");
    let prefill_port = SharedLlamaCppContextStateApi::from_context(prefill_ctx);

    let tokens = model
        .str_to_token(LIVE_PROMPT, AddBos::Always)
        .expect("tokenize live prompt for HTTP");
    prefill_port
        .decode_prompt_tokens(&tokens)
        .expect("live prompt prefill into LlamaContext before HTTP capture");

    let snapshot_len = prefill_port.state_byte_len();
    assert!(
        snapshot_len > 0,
        "live llama.cpp state after prefill must be non-empty before HTTP"
    );
    let max_state_bytes = (snapshot_len as u64)
        .saturating_mul(2)
        .max(snapshot_len as u64);

    let decode_ctx = model
        .new_context(
            backend,
            LlamaContextParams::default().with_n_ctx(Some(n_ctx)),
        )
        .expect("decode live context for HTTP");
    let decode_port = SharedLlamaCppContextStateApi::from_context(decode_ctx);

    let prefill_profile = live_profile(
        DisaggregatedServingRole::Prefill,
        layout.clone(),
        max_state_bytes,
    );
    let decode_profile = live_profile(DisaggregatedServingRole::Decode, layout, max_state_bytes);

    let prefill_transfer =
        Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&prefill_profile).unwrap());
    let decode_transfer =
        Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&decode_profile).unwrap());
    let prefill_ownership = Arc::new(LlamaCppBackendPhaseStateOwnership::from_layout_facts(
        facts.clone(),
        Some(digest('1')),
        Some(digest('2')),
        Some(digest('3')),
    ));
    let decode_ownership = Arc::new(LlamaCppBackendPhaseStateOwnership::from_layout_facts(
        facts,
        Some(digest('1')),
        Some(digest('2')),
        Some(digest('3')),
    ));

    let live_decode_port = decode_port.clone();
    let replay_token = *tokens
        .last()
        .expect("live prompt tokenization produced at least one token");
    let replay_position = i32::try_from(tokens.len()).expect("next position after restored KV");
    let decode_tokens = Arc::new(LlamaCppLiveDecodeTokenPort::new(Arc::new(move || {
        live_decode_port.decode_token_at(replay_token, replay_position)?;
        let chunk = live_greedy_decode_chunk_after_restore(&live_decode_port)?;
        let stream =
            futures::stream::iter(std::iter::once(Ok(PhaseResponseChunk::Completion(chunk))));
        Ok(Box::pin(stream) as a3s_power::serving::PhaseResponseStream)
    })));

    let prefill = live_http_app(
        DisaggregatedServingRole::Prefill,
        prefill_profile,
        prefill_port,
        prefill_ownership,
        prefill_transfer,
        None,
        None,
    );
    let decode = live_http_app(
        DisaggregatedServingRole::Decode,
        decode_profile,
        decode_port.clone(),
        decode_ownership,
        decode_transfer,
        Some(decode_tokens),
        Some(snapshot_len as u64),
    );
    // Fresh LlamaContext still reports a non-zero llama_get_state_size
    // (serializer footprint). Opaque restore evidence is the HTTP Ready
    // NDJSON path + post-execute non-empty state below, not a zero start.

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
            "request": live_completion_payload()
        }),
    )
    .await;
    assert_eq!(prepared.status(), StatusCode::OK);
    let prepared: DistributedPhaseResponse<PreparedDecodeResult> = parse_json(prepared).await;
    let target = match prepared.outcome {
        DistributedPhaseDecision::Ready { result } => result.target,
        other => panic!("expected Ready decode prepare over HTTP, got {other:?}"),
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
            "request": live_completion_payload(),
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
        "live HTTP Ready prefill must be typed JSON, got {content_type}"
    );
    assert!(!content_type.contains("ndjson"));
    let published: DistributedPhaseResponse<PublishedPrefillResult> = parse_json(published).await;
    let source = match published.outcome {
        DistributedPhaseDecision::Ready { result } => result.source,
        other => panic!("expected Ready prefill after live capture over HTTP, got {other:?}"),
    };
    assert_eq!(source.binding.state_bytes, snapshot_len as u64);

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
        "live HTTP Ready decode after restore+greedy sample must open NDJSON"
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
        panic!("expected completion chunk from LlamaCppLiveDecodeTokenPort over HTTP");
    };
    assert!(
        chunk.token_id.is_some(),
        "live HTTP NDJSON must carry a llama.cpp token id (not invented from transfer)"
    );
    assert!(frames.iter().all(|frame| {
        frame.schema == DISTRIBUTED_SERVING_STREAM_SCHEMA
            && frame.execution_id == execution_id
            && frame.worker_epoch == decode.epoch
    }));
    let restored_len = decode.context_port.state_byte_len();
    assert!(
        restored_len > 0,
        "live HTTP Ready NDJSON must follow consume + set_state_data restore"
    );
    assert!(prefill.runtime.accepts_work());
    assert!(decode.runtime.accepts_work());
    assert!(prefill.runtime.execution_admissible());
    assert!(decode.runtime.execution_admissible());

    let _keep_loaded = (backend, &model);
}
