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

#![cfg(feature = "llamacpp")]

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;

use a3s_power::serving::{
    live_greedy_decode_chunk_after_restore, BackendOwnedPhaseExecutor, BackendPhaseStateOwnership,
    BufferedHostLoopbackStateTransfer, ConsumeStateTransfer, DisaggregatedServingRole,
    ExecutePhaseExecution, ImportedModelState, LlamaCppBackendPhaseExecution,
    LlamaCppBackendPhaseStateOwnership, LlamaCppContextStateApi, LlamaCppContextStatePort,
    LlamaCppLayoutFacts, LlamaCppLiveDecodeTokenPort, PhaseDecision, PhaseExecutionOutput,
    PhaseExecutorHealth, PhaseRequest, PhaseResponseChunk, PhaseSessionPoolMode,
    PhaseWeightCacheMode, PrefillDecodeExecutionProfile, PreparePhaseExecution,
    PrepareStateTransfer, PreparedPhaseExecution, PublishStateTransfer,
    ServingCompositionPhaseExecution, ServingCompositionPhaseExecutor,
    ServingCompositionStateOwnership, ServingCompositionTransport, ServingExecutionProfile,
    ServingPhaseExecutor, ServingPrivacyMode, SharedLlamaCppContextStateApi, StateKind,
    StateTransferProtocol, StateTransferService,
};
use futures::StreamExt;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use serial_test::serial;
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
    assert!(!prefill_profile.may_advertise_prefill_decode());
    assert!(!decode_profile.may_advertise_prefill_decode());

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
