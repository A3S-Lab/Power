//! llama.cpp backend phase prepare/execute surface.
//!
//! Pairs with [`super::LlamaCppBackendPhaseStateOwnership`]. Uses the same
//! pinned context state snapshot contract (`llama_get_state_size` /
//! `llama_copy_state_data` / `llama_set_state_data`, or
//! [`super::FixtureLlamaCppContextStatePort`]) already documented for
//! ownership. Power never invents a second KV layout.
//!
//! ACL opt-in: `phase_execution = "llamacpp"` with `phase_executor =
//! backend-owned`. Binding unlocks Ready health after Eligible ownership.
//!
//! # Honest subset (shipped)
//!
//! - [`BackendPhaseExecution::can_produce_ready`] is true.
//! - `prepare` returns Ready prefill/decode reservations from closed profile
//!   digests (no invented tensors).
//! - Prefill `execute` returns Ready when a [`LlamaCppContextStatePort`] is
//!   bound: captures opaque snapshot bytes via get/copy and optionally
//!   registers them on a paired buffered-host transfer / ownership slots.
//! - Decode `execute` restores via set_state_data, then:
//!   - Without a bound [`LlamaCppDecodeTokenPort`]: fail-closes. Transfer
//!     completion / opaque snapshot bytes alone never yield Ready tokens.
//!   - With a bound decode-token port (production: live session calling
//!     existing llamacpp completion/decode after restore; tests:
//!     [`ControlledLlamaCppDecodeTokenPort`]): returns Ready decode stream.
//!     The decode adapter must not invent tokens from transfer bytes.
//! - Backend-owned composition still suppresses
//!   [`super::ServingExecutionProfile::may_advertise_prefill_decode`] (not
//!   honest to advertise while decode Ready can still require a separate
//!   decode-token port). DistributedServingRuntime may still be
//!   `execution_admissible` for typed-outcome HTTP without listing
//!   `ready_phases`.
//! - Composed with [`super::BackendOwnedPhaseExecutor`] under
//!   `state_ownership = llamacpp` + `phase_execution = llamacpp` +
//!   `transport = buffered-host-loopback`, prefill capture → buffered-host
//!   publish/consume → decode restore is fixture-proven; Ready decode is
//!   proven only when a decode-token adapter is bound. Env-gated live GGUF
//!   evidence (`A3S_POWER_LLAMACPP_PHASE_STATE_MODEL`) now drives capture →
//!   buffered-host publish/consume → restore on a real `LlamaContext`, plus
//!   Ready decode when a live hook materializes logits (this pin omits
//!   them) via llama.cpp decode at the next M-RoPE position and greedy-
//!   samples. Authenticated HTTP typed-outcome evidence is separate (see
//!   API distributed-serving llamacpp HTTP tests). That still does not close
//!   ROADMAP opaque-state / typed-outcome checkboxes.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use uuid::Uuid;

use crate::backend::types::CompletionResponseChunk;
use crate::error::{PowerError, Result};

use super::backend_phase_execution::BackendPhaseExecution;
use super::backend_phase_state_ownership::BackendPhaseStateOwnership;
use super::llamacpp_phase_state_ownership::{
    LlamaCppBackendPhaseStateOwnership, LlamaCppContextStatePort,
};
use super::{
    AbortPhaseExecution, BufferedHostLoopbackStateTransfer, ExecutePhaseExecution,
    ModelStateHandle, PhaseDecision, PhaseExecutionHandle, PhaseExecutionOutput,
    PhaseResponseChunk, PhaseResponseStream, PreparePhaseExecution, PreparedDecodePhase,
    PreparedPhaseExecution, PreparedPrefillPhase, ProducedModelState, ServingExecutionProfile,
    ServingPhase, StateKind, StateTransferBinding,
};

/// Token generation after opaque context restore.
///
/// Distinct from [`LlamaCppContextStatePort`]: get/copy/set move opaque
/// snapshot bytes only. This port owns decode token streams (production:
/// existing llamacpp completion/decode on the restored session; tests: a
/// controlled adapter). Transfer bytes must never become invented tokens.
pub trait LlamaCppDecodeTokenPort: Send + Sync {
    /// Produce a decode response after restore succeeded on the bound context.
    ///
    /// Callers must invoke this only after `set_state_from` applied transferred
    /// opaque bytes. Implementations must not parse or hash those bytes into
    /// token text.
    fn decode_stream_after_restore(&self) -> Result<PhaseResponseStream>;
}

/// Test/production-safe decode adapter with an explicit controlled stream.
///
/// Chunks are supplied by the caller (not derived from opaque transfer bytes).
/// Use this to prove Ready decode after restore without a live GGUF, or to
/// wrap a live session hook that already called llamacpp completion APIs.
#[derive(Clone, Debug)]
pub struct ControlledLlamaCppDecodeTokenPort {
    chunks: Arc<Vec<CompletionResponseChunk>>,
}

impl ControlledLlamaCppDecodeTokenPort {
    /// Bind a fixed completion stream. Empty chunks fail closed at decode time.
    #[must_use]
    pub fn from_chunks(chunks: Vec<CompletionResponseChunk>) -> Self {
        Self {
            chunks: Arc::new(chunks),
        }
    }

    /// Single done chunk with explicit token text (never from transfer bytes).
    #[must_use]
    pub fn single_completion(text: impl Into<String>, token_id: u32) -> Self {
        Self::from_chunks(vec![CompletionResponseChunk {
            text: text.into(),
            done: true,
            prompt_tokens: Some(1),
            done_reason: Some("stop".to_string()),
            prompt_eval_duration_ns: None,
            token_id: Some(token_id),
        }])
    }
}

impl LlamaCppDecodeTokenPort for ControlledLlamaCppDecodeTokenPort {
    fn decode_stream_after_restore(&self) -> Result<PhaseResponseStream> {
        if self.chunks.is_empty() {
            return Err(PowerError::BackendNotAvailable(
                "ControlledLlamaCppDecodeTokenPort refuses empty decode stream (bind real completion chunks or a live llamacpp session hook)"
                    .to_string(),
            ));
        }
        let chunks = (*self.chunks).clone();
        let stream = futures::stream::iter(
            chunks
                .into_iter()
                .map(|chunk| Ok(PhaseResponseChunk::Completion(chunk))),
        );
        Ok(Box::pin(stream))
    }
}

/// Live-session decode hook under the `llamacpp` feature.
///
/// Production binds a closure that runs existing llamacpp completion/decode
/// against the same `LlamaContext` previously restored via
/// [`super::LlamaCppContextStateApi`]. The hook must not invent tokens from
/// opaque snapshot bytes.
#[cfg(feature = "llamacpp")]
pub type LlamaCppLiveDecodeHook = Arc<dyn Fn() -> Result<PhaseResponseStream> + Send + Sync>;

/// Feature-gated adapter that defers token generation to a live session hook.
#[cfg(feature = "llamacpp")]
pub struct LlamaCppLiveDecodeTokenPort {
    hook: LlamaCppLiveDecodeHook,
}

#[cfg(feature = "llamacpp")]
impl LlamaCppLiveDecodeTokenPort {
    #[must_use]
    pub fn new(hook: LlamaCppLiveDecodeHook) -> Self {
        Self { hook }
    }
}

#[cfg(feature = "llamacpp")]
impl LlamaCppDecodeTokenPort for LlamaCppLiveDecodeTokenPort {
    fn decode_stream_after_restore(&self) -> Result<PhaseResponseStream> {
        (self.hook)()
    }
}

/// One greedy token from a restored live context via llama.cpp sampling.
///
/// Call only after `set_state_from` applied transferred opaque bytes. The
/// token id comes from `LlamaSampler::greedy` on the restored logits — never
/// from snapshot bytes. Piece text is omitted so this hook does not need a
/// shared `LlamaModel` across the decode-token port.
#[cfg(feature = "llamacpp")]
pub fn live_greedy_decode_chunk_after_restore(
    port: &super::SharedLlamaCppContextStateApi,
) -> Result<CompletionResponseChunk> {
    port.with_context_mut(|ctx| {
        let mut sampler = llama_cpp_2::sampling::LlamaSampler::chain(
            vec![llama_cpp_2::sampling::LlamaSampler::greedy()],
            false,
        );
        let token = sampler.sample(ctx, -1);
        Ok(CompletionResponseChunk {
            text: String::new(),
            done: true,
            prompt_tokens: Some(1),
            done_reason: Some("stop".to_string()),
            prompt_eval_duration_ns: None,
            token_id: Some(token.0 as u32),
        })
    })
}

/// Default reserved token_count for prepare bindings when no live session
/// token census is available. Not a claim about prompt length.
const RESERVED_TOKEN_COUNT: u64 = 1;

#[derive(Default)]
struct PreparedLeaseStore {
    handles: Mutex<HashMap<Uuid, ModelStateHandle>>,
}

impl PreparedLeaseStore {
    fn lock(&self) -> Result<MutexGuard<'_, HashMap<Uuid, ModelStateHandle>>> {
        self.handles.lock().map_err(|_| {
            PowerError::BackendNotAvailable(
                "LlamaCppBackendPhaseExecution lease lock is unavailable".to_string(),
            )
        })
    }

    fn insert(&self, execution_id: Uuid, handle: ModelStateHandle) -> Result<()> {
        let mut leases = self.lock()?;
        if leases.contains_key(&execution_id) {
            return Err(PowerError::InvalidRequest(
                "LlamaCppBackendPhaseExecution already holds a lease for this execution"
                    .to_string(),
            ));
        }
        leases.insert(execution_id, handle);
        Ok(())
    }

    fn take(&self, execution_id: Uuid) -> Result<Option<ModelStateHandle>> {
        Ok(self.lock()?.remove(&execution_id))
    }

    fn remove(&self, execution_id: Uuid) -> Result<()> {
        self.lock()?.remove(&execution_id);
        Ok(())
    }
}

/// Real llama.cpp prepare/execute adapter for backend-owned phase composition.
///
/// Construct via [`Self::for_profile`] (ACL) or [`Self::with_port`] when a
/// fixture / live context port is available for Ready prefill execute.
pub struct LlamaCppBackendPhaseExecution {
    phase: ServingPhase,
    profile_sha256: String,
    binding_template: StateTransferBinding,
    max_state_bytes: u64,
    port: Mutex<Option<Box<dyn LlamaCppContextStatePort>>>,
    ownership: Option<Arc<LlamaCppBackendPhaseStateOwnership>>,
    transfer: Option<Arc<BufferedHostLoopbackStateTransfer>>,
    decode_tokens: Option<Arc<dyn LlamaCppDecodeTokenPort>>,
    leases: PreparedLeaseStore,
}

impl fmt::Debug for LlamaCppBackendPhaseExecution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlamaCppBackendPhaseExecution")
            .field("phase", &self.phase)
            .field("profile_sha256", &self.profile_sha256)
            .field("binding_template", &self.binding_template)
            .field(
                "has_port",
                &self.port.lock().map(|g| g.is_some()).unwrap_or(false),
            )
            .field("has_ownership", &self.ownership.is_some())
            .field("has_transfer", &self.transfer.is_some())
            .field("has_decode_tokens", &self.decode_tokens.is_some())
            .finish()
    }
}

impl LlamaCppBackendPhaseExecution {
    /// ACL/composition constructor: Ready-capable prepare; execute needs a
    /// later [`Self::bind_port`] (or [`Self::with_port`]) for Ready prefill.
    pub fn for_profile(profile: &ServingExecutionProfile) -> Result<Self> {
        Self::build(profile, None, None, None)
    }

    /// Bind a fixture or live context state port for Ready prefill capture.
    pub fn with_port(
        profile: &ServingExecutionProfile,
        port: Box<dyn LlamaCppContextStatePort>,
    ) -> Result<Self> {
        Self::build(profile, Some(port), None, None)
    }

    /// Pair with ownership after ACL construction.
    #[must_use]
    pub fn with_ownership(mut self, ownership: Arc<LlamaCppBackendPhaseStateOwnership>) -> Self {
        self.ownership = Some(ownership);
        self
    }

    /// Optional transfer registration for opaque prefill publish bytes.
    #[must_use]
    pub fn with_transfer(mut self, transfer: Arc<BufferedHostLoopbackStateTransfer>) -> Self {
        self.transfer = Some(transfer);
        self
    }

    /// Bind a decode-token adapter used only after opaque restore succeeds.
    ///
    /// Production: wrap existing llamacpp completion/decode on the restored
    /// context ([`LlamaCppLiveDecodeTokenPort`] under `llamacpp`). Tests:
    /// [`ControlledLlamaCppDecodeTokenPort`] with explicit chunks — never
    /// derive token text from transferred snapshot bytes.
    #[must_use]
    pub fn with_decode_tokens(mut self, decode_tokens: Arc<dyn LlamaCppDecodeTokenPort>) -> Self {
        self.decode_tokens = Some(decode_tokens);
        self
    }

    /// Install or replace the context state port (fixture or live session).
    pub fn bind_port(&self, port: Box<dyn LlamaCppContextStatePort>) -> Result<()> {
        *self.port.lock().map_err(|_| {
            PowerError::BackendNotAvailable(
                "LlamaCppBackendPhaseExecution port lock poisoned".to_string(),
            )
        })? = Some(port);
        Ok(())
    }

    /// Install or replace the decode-token adapter after construction.
    pub fn bind_decode_tokens(&mut self, decode_tokens: Arc<dyn LlamaCppDecodeTokenPort>) {
        self.decode_tokens = Some(decode_tokens);
    }

    /// Pin the expected opaque snapshot size (decode reservations / live capture).
    ///
    /// Decode prepare must bind the incoming transfer size, not a virgin
    /// context's `get_state_size`, which can differ after prefill.
    pub fn with_captured_state_bytes(mut self, state_bytes: u64) -> Result<Self> {
        self.binding_template = self.binding_for_captured(state_bytes)?;
        Ok(self)
    }

    fn build(
        profile: &ServingExecutionProfile,
        port: Option<Box<dyn LlamaCppContextStatePort>>,
        ownership: Option<Arc<LlamaCppBackendPhaseStateOwnership>>,
        transfer: Option<Arc<BufferedHostLoopbackStateTransfer>>,
    ) -> Result<Self> {
        let ServingExecutionProfile::PrefillDecode { execution } = profile else {
            return Err(PowerError::Config(
                "LlamaCppBackendPhaseExecution requires a prefill-decode serving profile"
                    .to_string(),
            ));
        };
        let state_bytes = port
            .as_ref()
            .map(|p| p.state_byte_len() as u64)
            .filter(|n| *n > 0)
            .unwrap_or(1);
        if state_bytes > execution.max_state_bytes {
            return Err(PowerError::Config(format!(
                "LlamaCppBackendPhaseExecution state snapshot ({state_bytes}) exceeds max_state_bytes ({})",
                execution.max_state_bytes
            )));
        }
        let binding_template = StateTransferBinding {
            model_sha256: execution.model_sha256.clone(),
            execution_sha256: execution.execution_sha256.clone(),
            layout_sha256: execution.layout_sha256.clone(),
            state_kind: execution.state_kind,
            token_count: RESERVED_TOKEN_COUNT,
            state_bytes,
        };
        profile.validate_state_binding(&binding_template)?;
        if !matches!(
            binding_template.state_kind,
            StateKind::KvCache | StateKind::Recurrent
        ) {
            return Err(PowerError::Config(
                "LlamaCppBackendPhaseExecution requires a known opaque state kind".to_string(),
            ));
        }
        Ok(Self {
            phase: ServingPhase::from(execution.role),
            profile_sha256: profile.sha256()?,
            binding_template,
            max_state_bytes: execution.max_state_bytes,
            port: Mutex::new(port),
            ownership,
            transfer,
            decode_tokens: None,
            leases: PreparedLeaseStore::default(),
        })
    }

    fn lock_port(&self) -> Result<MutexGuard<'_, Option<Box<dyn LlamaCppContextStatePort>>>> {
        self.port.lock().map_err(|_| {
            PowerError::BackendNotAvailable(
                "LlamaCppBackendPhaseExecution port lock poisoned".to_string(),
            )
        })
    }

    fn binding_for_captured(&self, state_bytes: u64) -> Result<StateTransferBinding> {
        if state_bytes == 0 || state_bytes > self.max_state_bytes {
            return Err(PowerError::InvalidRequest(format!(
                "LlamaCppBackendPhaseExecution captured state_bytes {state_bytes} outside 1..={}",
                self.max_state_bytes
            )));
        }
        Ok(StateTransferBinding {
            state_bytes,
            ..self.binding_template.clone()
        })
    }

    fn capture_opaque_prefill_state(
        &self,
    ) -> Result<(ModelStateHandle, Vec<u8>, StateTransferBinding)> {
        let mut guard = self.lock_port()?;
        let port = guard.as_mut().ok_or_else(|| {
            PowerError::BackendNotAvailable(
                "LlamaCppBackendPhaseExecution refuses prefill execute: no LlamaCppContextStatePort bound (fixture or live session); prepare Ready only"
                    .to_string(),
            )
        })?;
        let (handle, bytes) = if let Some(ownership) = &self.ownership {
            let handle = ownership.capture_from_port(port.as_ref())?;
            let bytes = ownership.export_opaque_state(&handle)?;
            (handle, bytes)
        } else {
            let size = port.state_byte_len();
            if size == 0 {
                return Err(PowerError::BackendNotAvailable(
                    "LlamaCppBackendPhaseExecution refuses prefill execute: llama.cpp reported zero-sized state"
                        .to_string(),
                ));
            }
            let mut bytes = vec![0u8; size];
            let written = port.copy_state_into(&mut bytes)?;
            if written != size {
                return Err(PowerError::InferenceFailed(format!(
                    "llama.cpp state export size mismatch: expected {size}, got {written}"
                )));
            }
            bytes.truncate(written);
            let handle = ModelStateHandle::new(format!("llamacpp-prefill:{}", Uuid::new_v4()))?;
            (handle, bytes)
        };
        let binding = self.binding_for_captured(bytes.len() as u64)?;
        Ok((handle, bytes, binding))
    }

    fn decode_tokens_unowned() -> PowerError {
        PowerError::BackendNotAvailable(
            "LlamaCppBackendPhaseExecution refuses decode execute: opaque state restore may use llama_set_state_data, but decode token generation requires a bound LlamaCppDecodeTokenPort (live llamacpp completion/decode after restore, or ControlledLlamaCppDecodeTokenPort); transfer completion alone never yields Ready decode"
                .to_string(),
        )
    }

    fn restore_opaque_decode_state(&self, bytes: &[u8]) -> Result<()> {
        let mut guard = self.lock_port()?;
        let port = guard.as_mut().ok_or_else(|| {
            PowerError::BackendNotAvailable(
                "LlamaCppBackendPhaseExecution refuses decode restore: no LlamaCppContextStatePort bound"
                    .to_string(),
            )
        })?;
        let applied = port.set_state_from(bytes)?;
        if applied != bytes.len() {
            return Err(PowerError::InferenceFailed(format!(
                "llama.cpp state import size mismatch: expected {}, got {applied}",
                bytes.len()
            )));
        }
        Ok(())
    }
}

#[async_trait]
impl BackendPhaseExecution for LlamaCppBackendPhaseExecution {
    fn can_produce_ready(&self) -> bool {
        true
    }

    async fn prepare(
        &self,
        command: PreparePhaseExecution,
    ) -> Result<PhaseDecision<PreparedPhaseExecution>> {
        let execution =
            PhaseExecutionHandle::new(format!("llamacpp-phase:{}", command.execution_id))?;
        let prepared = match self.phase {
            ServingPhase::Prefill => {
                let handle = ModelStateHandle::new(format!("source:{}", command.execution_id))?;
                self.leases.insert(command.execution_id, handle)?;
                PreparedPhaseExecution::Prefill(PreparedPrefillPhase::new(
                    command.execution_id,
                    command.local_worker_epoch,
                    self.profile_sha256.clone(),
                    execution,
                    command.expires_at,
                )?)
            }
            ServingPhase::Decode => {
                let destination =
                    ModelStateHandle::new(format!("destination:{}", command.execution_id))?;
                self.leases
                    .insert(command.execution_id, destination.clone())?;
                let binding = self.binding_template.clone();
                PreparedPhaseExecution::Decode(PreparedDecodePhase::new(
                    command.execution_id,
                    command.local_worker_epoch,
                    self.profile_sha256.clone(),
                    execution,
                    destination,
                    binding,
                    command.expires_at,
                )?)
            }
            ServingPhase::Aggregated => {
                return Err(PowerError::Config(
                    "LlamaCppBackendPhaseExecution cannot execute the aggregated phase".to_string(),
                ));
            }
        };
        Ok(PhaseDecision::ready(prepared))
    }

    async fn execute(
        &self,
        command: ExecutePhaseExecution,
    ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
        match command {
            ExecutePhaseExecution::Prefill { prepared } => {
                let _lease = self.leases.take(prepared.execution_id())?;
                let (source, bytes, binding) = self.capture_opaque_prefill_state()?;
                if let Some(transfer) = &self.transfer {
                    transfer.register_owned_state(&source, bytes)?;
                }
                Ok(PhaseDecision::ready(PhaseExecutionOutput::Prefill(
                    ProducedModelState::new(
                        prepared.execution_id(),
                        prepared.local_worker_epoch(),
                        self.profile_sha256.clone(),
                        source,
                        binding,
                    )?,
                )))
            }
            ExecutePhaseExecution::Decode { prepared, state } => {
                let _ = self.leases.remove(prepared.execution_id());
                // Transfer receipt alone is never decode success. Restore
                // opaque bytes through set_state_data, then either Ready via a
                // bound decode-token adapter or fail closed.
                let opaque = if let Some(transfer) = &self.transfer {
                    transfer.take_owned_state(state.destination()).ok()
                } else {
                    None
                };
                let Some(bytes) = opaque else {
                    drop(prepared);
                    drop(state);
                    return Err(Self::decode_tokens_unowned());
                };
                self.restore_opaque_decode_state(&bytes)?;
                let Some(decode_tokens) = &self.decode_tokens else {
                    drop(prepared);
                    drop(state);
                    return Err(Self::decode_tokens_unowned());
                };
                let stream = decode_tokens.decode_stream_after_restore()?;
                drop(prepared);
                drop(state);
                Ok(PhaseDecision::ready(PhaseExecutionOutput::Decode(stream)))
            }
        }
    }

    async fn abort(&self, command: AbortPhaseExecution) -> Result<()> {
        if let Some(handle) = self.leases.take(command.execution_id)? {
            if let Some(transfer) = &self.transfer {
                let _ = transfer.take_owned_state(&handle);
            }
        }
        if let Ok(source) = ModelStateHandle::new(format!("source:{}", command.execution_id)) {
            if let Some(transfer) = &self.transfer {
                let _ = transfer.take_owned_state(&source);
            }
        }
        if let Ok(destination) =
            ModelStateHandle::new(format!("destination:{}", command.execution_id))
        {
            if let Some(transfer) = &self.transfer {
                let _ = transfer.take_owned_state(&destination);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::{
        bind_backend_phase_execution, bind_backend_phase_state_ownership,
        BackendOwnedPhaseExecutor, BackendPhaseStateOwnership, BoundedStateTransferService,
        ConsumeStateTransfer, DisaggregatedServingRole, DistributedServingRuntime,
        EmptyBackendPhaseStateOwnership, FixtureLlamaCppContextStatePort, ImportedModelState,
        PhaseExecutorHealth, PhaseRequest, PhaseSessionPoolMode, PhaseWeightCacheMode,
        PrefillDecodeExecutionProfile, PrepareStateTransfer, PublishStateTransfer,
        ServingCompositionPhaseExecution, ServingCompositionPhaseExecutor,
        ServingCompositionStateOwnership, ServingCompositionTransport, ServingPhaseExecutor,
        ServingPrivacyMode, SharedFixtureLlamaCppContextStatePort, StateTransferProtocol,
        StateTransferService,
    };

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn fixture_snapshot() -> Vec<u8> {
        (0..32).map(|i| ((i * 13 + 7) % 251) as u8).collect()
    }

    fn decode_profile(
        state_ownership: Option<ServingCompositionStateOwnership>,
        phase_execution: Option<ServingCompositionPhaseExecution>,
    ) -> ServingExecutionProfile {
        ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role: DisaggregatedServingRole::Decode,
            model: "internal/model-v1".to_string(),
            model_sha256: digest('1'),
            backend: "llamacpp".to_string(),
            backend_sha256: digest('2'),
            execution_sha256: digest('3'),
            device_sha256: digest('4'),
            layout_sha256: digest('5'),
            peer_set_sha256: digest('6'),
            generation: 7,
            protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
            state_kind: StateKind::KvCache,
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
            state_ownership,
            phase_execution,
        })
        .unwrap()
    }

    fn prefill_profile() -> ServingExecutionProfile {
        let mut profile = decode_profile(
            Some(ServingCompositionStateOwnership::LlamaCpp),
            Some(ServingCompositionPhaseExecution::LlamaCpp),
        );
        if let ServingExecutionProfile::PrefillDecode { execution } = &mut profile {
            execution.role = DisaggregatedServingRole::Prefill;
        }
        profile.validate().unwrap();
        profile
    }

    #[test]
    fn without_eligible_ownership_stays_unavailable() {
        let profile = decode_profile(None, Some(ServingCompositionPhaseExecution::LlamaCpp));
        let execution = LlamaCppBackendPhaseExecution::for_profile(&profile).unwrap();
        assert!(execution.can_produce_ready());
        let ownership_health =
            bind_backend_phase_state_ownership(&EmptyBackendPhaseStateOwnership, &profile).unwrap();
        assert_eq!(ownership_health, PhaseExecutorHealth::Unavailable);
        assert_eq!(
            bind_backend_phase_execution(ownership_health, &execution),
            PhaseExecutorHealth::Unavailable
        );
        assert!(!bind_backend_phase_execution(ownership_health, &execution).accepts_work());
    }

    #[tokio::test]
    async fn with_ownership_and_fixture_port_prepare_and_prefill_execute_ready() {
        let profile = prefill_profile();
        let ownership =
            Arc::new(LlamaCppBackendPhaseStateOwnership::for_profile(&profile).unwrap());
        assert_eq!(
            bind_backend_phase_state_ownership(ownership.as_ref(), &profile).unwrap(),
            PhaseExecutorHealth::Eligible
        );
        let port = Box::new(FixtureLlamaCppContextStatePort::with_snapshot(
            fixture_snapshot(),
        ));
        let execution = LlamaCppBackendPhaseExecution::with_port(&profile, port)
            .unwrap()
            .with_ownership(Arc::clone(&ownership));
        assert_eq!(
            bind_backend_phase_execution(PhaseExecutorHealth::Eligible, &execution),
            PhaseExecutorHealth::Ready
        );

        let decision = execution
            .prepare(PreparePhaseExecution {
                execution_id: Uuid::new_v4(),
                local_worker_epoch: Uuid::new_v4(),
                model: "internal/model-v1".to_string(),
                expires_at: chrono::Utc::now() + chrono::Duration::seconds(30),
                request: PhaseRequest::Completion(
                    serde_json::from_value(serde_json::json!({ "prompt": "hi" })).unwrap(),
                ),
            })
            .await
            .unwrap();
        let prepared = match decision {
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

        let execute = execution
            .execute(ExecutePhaseExecution::prefill(prepared))
            .await
            .unwrap();
        match execute {
            PhaseDecision::Ready(PhaseExecutionOutput::Prefill(produced)) => {
                assert_eq!(
                    produced.binding().state_bytes,
                    fixture_snapshot().len() as u64
                );
                let exported = ownership.export_opaque_state(produced.source()).unwrap();
                assert_eq!(exported, fixture_snapshot());
            }
            PhaseDecision::Ready(_) => panic!("expected Ready prefill produce"),
            PhaseDecision::Recompute { .. } => panic!("expected Ready produce, got Recompute"),
            PhaseDecision::RetryableUnavailable { .. } => {
                panic!("expected Ready produce, got RetryableUnavailable")
            }
            PhaseDecision::TerminalFailure { .. } => {
                panic!("expected Ready produce, got TerminalFailure")
            }
        }
    }

    #[tokio::test]
    async fn decode_execute_never_ready_from_transfer_completion_alone() {
        let profile = decode_profile(
            Some(ServingCompositionStateOwnership::LlamaCpp),
            Some(ServingCompositionPhaseExecution::LlamaCpp),
        );
        let ownership =
            Arc::new(LlamaCppBackendPhaseStateOwnership::for_profile(&profile).unwrap());
        let snapshot = fixture_snapshot();
        let port = Box::new(FixtureLlamaCppContextStatePort::with_snapshot(
            snapshot.clone(),
        ));
        let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&profile).unwrap());
        let execution = LlamaCppBackendPhaseExecution::with_port(&profile, port)
            .unwrap()
            .with_ownership(ownership)
            .with_transfer(Arc::clone(&transfer));

        let local_worker_epoch = Uuid::new_v4();
        let decision = execution
            .prepare(PreparePhaseExecution {
                execution_id: Uuid::new_v4(),
                local_worker_epoch,
                model: "internal/model-v1".to_string(),
                expires_at: chrono::Utc::now() + chrono::Duration::seconds(30),
                request: PhaseRequest::Completion(
                    serde_json::from_value(serde_json::json!({ "prompt": "hi" })).unwrap(),
                ),
            })
            .await
            .unwrap();
        let prepared = match decision {
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

        let destination = prepared.destination().clone();
        transfer
            .register_owned_state(&destination, snapshot)
            .unwrap();
        let imported = ImportedModelState::from_parts_for_test(
            Uuid::new_v4(),
            local_worker_epoch,
            profile.sha256().unwrap(),
            destination,
            prepared.binding().clone(),
        );
        let err = match execution
            .execute(ExecutePhaseExecution::decode(prepared, imported).unwrap())
            .await
        {
            Ok(_) => panic!("transfer alone must never Ready decode"),
            Err(error) => error,
        };
        assert!(err
            .to_string()
            .contains("requires a bound LlamaCppDecodeTokenPort"));
        assert!(err
            .to_string()
            .contains("transfer completion alone never yields Ready decode"));
    }

    #[tokio::test]
    async fn decode_execute_ready_after_restore_with_controlled_decode_adapter() {
        let profile = decode_profile(
            Some(ServingCompositionStateOwnership::LlamaCpp),
            Some(ServingCompositionPhaseExecution::LlamaCpp),
        );
        let ownership =
            Arc::new(LlamaCppBackendPhaseStateOwnership::for_profile(&profile).unwrap());
        let snapshot = fixture_snapshot();
        let port = Box::new(FixtureLlamaCppContextStatePort::with_snapshot(
            snapshot.clone(),
        ));
        let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&profile).unwrap());
        // Controlled adapter supplies explicit tokens — not derived from snapshot.
        let decode_tokens = Arc::new(ControlledLlamaCppDecodeTokenPort::single_completion(
            "adapter-owned-token",
            42,
        ));
        let execution = LlamaCppBackendPhaseExecution::with_port(&profile, port)
            .unwrap()
            .with_ownership(ownership)
            .with_transfer(Arc::clone(&transfer))
            .with_decode_tokens(decode_tokens);

        let local_worker_epoch = Uuid::new_v4();
        let decision = execution
            .prepare(PreparePhaseExecution {
                execution_id: Uuid::new_v4(),
                local_worker_epoch,
                model: "internal/model-v1".to_string(),
                expires_at: chrono::Utc::now() + chrono::Duration::seconds(30),
                request: PhaseRequest::Completion(
                    serde_json::from_value(serde_json::json!({ "prompt": "hi" })).unwrap(),
                ),
            })
            .await
            .unwrap();
        let prepared = match decision {
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

        let destination = prepared.destination().clone();
        transfer
            .register_owned_state(&destination, snapshot)
            .unwrap();
        let imported = ImportedModelState::from_parts_for_test(
            Uuid::new_v4(),
            local_worker_epoch,
            profile.sha256().unwrap(),
            destination,
            prepared.binding().clone(),
        );
        let output = execution
            .execute(ExecutePhaseExecution::decode(prepared, imported).unwrap())
            .await
            .unwrap();
        match output {
            PhaseDecision::Ready(PhaseExecutionOutput::Decode(mut stream)) => {
                use futures::StreamExt;
                let chunk = stream.next().await.expect("one chunk").unwrap();
                match chunk {
                    PhaseResponseChunk::Completion(chunk) => {
                        assert_eq!(chunk.text, "adapter-owned-token");
                        assert_eq!(chunk.token_id, Some(42));
                        assert!(chunk.done);
                        assert_ne!(
                            chunk.text.as_bytes(),
                            &fixture_snapshot()[..],
                            "decode adapter must not invent tokens from transfer bytes"
                        );
                    }
                    PhaseResponseChunk::Chat(_) => panic!("expected completion chunk"),
                }
                assert!(stream.next().await.is_none());
            }
            PhaseDecision::Ready(_) => panic!("expected Ready decode stream"),
            PhaseDecision::Recompute { .. } => panic!("expected Ready decode, got Recompute"),
            PhaseDecision::RetryableUnavailable { .. } => {
                panic!("expected Ready decode, got RetryableUnavailable")
            }
            PhaseDecision::TerminalFailure { .. } => {
                panic!("expected Ready decode, got TerminalFailure")
            }
        }
    }

    #[tokio::test]
    async fn decode_adapter_alone_without_restored_opaque_never_ready() {
        let profile = decode_profile(
            Some(ServingCompositionStateOwnership::LlamaCpp),
            Some(ServingCompositionPhaseExecution::LlamaCpp),
        );
        let ownership =
            Arc::new(LlamaCppBackendPhaseStateOwnership::for_profile(&profile).unwrap());
        let port = Box::new(FixtureLlamaCppContextStatePort::with_snapshot(
            fixture_snapshot(),
        ));
        let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&profile).unwrap());
        let decode_tokens = Arc::new(ControlledLlamaCppDecodeTokenPort::single_completion(
            "must-not-emit",
            1,
        ));
        let execution = LlamaCppBackendPhaseExecution::with_port(&profile, port)
            .unwrap()
            .with_ownership(ownership)
            .with_transfer(transfer)
            .with_decode_tokens(decode_tokens);

        let local_worker_epoch = Uuid::new_v4();
        let prepared = match execution
            .prepare(PreparePhaseExecution {
                execution_id: Uuid::new_v4(),
                local_worker_epoch,
                model: "internal/model-v1".to_string(),
                expires_at: chrono::Utc::now() + chrono::Duration::seconds(30),
                request: PhaseRequest::Completion(
                    serde_json::from_value(serde_json::json!({ "prompt": "hi" })).unwrap(),
                ),
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
        // No register_owned_state — decode adapter must not Ready without restore.
        let imported = ImportedModelState::from_parts_for_test(
            Uuid::new_v4(),
            local_worker_epoch,
            profile.sha256().unwrap(),
            prepared.destination().clone(),
            prepared.binding().clone(),
        );
        let err = match execution
            .execute(ExecutePhaseExecution::decode(prepared, imported).unwrap())
            .await
        {
            Ok(_) => panic!("decode adapter without restored opaque must not Ready"),
            Err(error) => error,
        };
        assert!(err
            .to_string()
            .contains("transfer completion alone never yields Ready decode"));
    }

    #[tokio::test]
    async fn backend_owned_llamacpp_buffered_host_captures_publishes_restores_then_fail_closes() {
        // First-principles composition evidence for opaque-state / typed-outcome
        // progress: BackendOwnedPhaseExecutor + llamacpp ownership/execution +
        // buffered-host product pair. Fixture port only — live GGUF still
        // required before ROADMAP checkboxes close. Without a decode-token
        // adapter, decode stays fail-closed after restore.
        let snapshot = fixture_snapshot();
        let prefill_profile = prefill_profile();
        let decode_profile = decode_profile(
            Some(ServingCompositionStateOwnership::LlamaCpp),
            Some(ServingCompositionPhaseExecution::LlamaCpp),
        );

        let prefill_transfer =
            Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&prefill_profile).unwrap());
        let decode_transfer =
            Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&decode_profile).unwrap());

        let prefill_ownership =
            Arc::new(LlamaCppBackendPhaseStateOwnership::for_profile(&prefill_profile).unwrap());
        let decode_ownership =
            Arc::new(LlamaCppBackendPhaseStateOwnership::for_profile(&decode_profile).unwrap());

        let prefill_port = SharedFixtureLlamaCppContextStatePort::with_snapshot(snapshot.clone());
        let decode_port = SharedFixtureLlamaCppContextStatePort::empty();
        assert!(decode_port.snapshot().unwrap().is_empty());

        let prefill_execution = Arc::new(
            LlamaCppBackendPhaseExecution::with_port(
                &prefill_profile,
                Box::new(prefill_port.clone()),
            )
            .unwrap()
            .with_ownership(Arc::clone(&prefill_ownership))
            .with_transfer(Arc::clone(&prefill_transfer)),
        );
        let decode_execution = Arc::new(
            LlamaCppBackendPhaseExecution::with_port(
                &decode_profile,
                Box::new(decode_port.clone()),
            )
            .unwrap()
            .with_ownership(Arc::clone(&decode_ownership))
            .with_transfer(Arc::clone(&decode_transfer)),
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
                serde_json::from_value(serde_json::json!({ "prompt": "hi" })).unwrap(),
            )
        };

        let prepared = match prefill_executor
            .prepare(PreparePhaseExecution {
                execution_id,
                local_worker_epoch: source_epoch,
                model: "internal/model-v1".to_string(),
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
            PhaseDecision::Ready(_) => {
                panic!("expected Ready prefill produce via ownership capture")
            }
            PhaseDecision::Recompute { .. } => panic!("expected Ready produce, got Recompute"),
            PhaseDecision::RetryableUnavailable { .. } => {
                panic!("expected Ready produce, got RetryableUnavailable")
            }
            PhaseDecision::TerminalFailure { .. } => {
                panic!("expected Ready produce, got TerminalFailure")
            }
        };
        assert_eq!(produced.binding().state_bytes, snapshot.len() as u64);

        let decode_prepared = match decode_executor
            .prepare(PreparePhaseExecution {
                execution_id,
                local_worker_epoch: destination_epoch,
                model: "internal/model-v1".to_string(),
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
        let _receipt = decode_transfer
            .consume_source(ConsumeStateTransfer {
                local_worker_epoch: destination_epoch,
                destination: decode_prepared.destination().clone(),
                source: published,
            })
            .await
            .unwrap();

        let imported = ImportedModelState::from_parts_for_test(
            execution_id,
            destination_epoch,
            decode_profile.sha256().unwrap(),
            decode_prepared.destination().clone(),
            decode_prepared.binding().clone(),
        );
        let err = match decode_executor
            .execute(ExecutePhaseExecution::decode(decode_prepared, imported).unwrap())
            .await
        {
            Ok(_) => panic!("decode must fail-closed after restore"),
            Err(error) => error,
        };
        assert!(
            err.to_string()
                .contains("requires a bound LlamaCppDecodeTokenPort"),
            "{err}"
        );
        assert_eq!(
            decode_port.snapshot().unwrap(),
            snapshot,
            "decode execute must restore opaque bytes via set_state_data before fail-closed tokens"
        );
    }

    #[tokio::test]
    async fn backend_owned_llamacpp_buffered_host_restores_then_ready_decode_with_adapter() {
        // Same composition as fail-closed restore evidence, plus a controlled
        // decode-token adapter (not transfer-byte invention). Live GGUF still
        // required before ROADMAP opaque-state / typed-outcome close.
        let snapshot = fixture_snapshot();
        let prefill_profile = prefill_profile();
        let decode_profile = decode_profile(
            Some(ServingCompositionStateOwnership::LlamaCpp),
            Some(ServingCompositionPhaseExecution::LlamaCpp),
        );

        let prefill_transfer =
            Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&prefill_profile).unwrap());
        let decode_transfer =
            Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&decode_profile).unwrap());

        let prefill_ownership =
            Arc::new(LlamaCppBackendPhaseStateOwnership::for_profile(&prefill_profile).unwrap());
        let decode_ownership =
            Arc::new(LlamaCppBackendPhaseStateOwnership::for_profile(&decode_profile).unwrap());

        let prefill_port = SharedFixtureLlamaCppContextStatePort::with_snapshot(snapshot.clone());
        let decode_port = SharedFixtureLlamaCppContextStatePort::empty();
        let decode_tokens = Arc::new(ControlledLlamaCppDecodeTokenPort::single_completion(
            "composed-adapter-token",
            7,
        ));

        let prefill_execution = Arc::new(
            LlamaCppBackendPhaseExecution::with_port(
                &prefill_profile,
                Box::new(prefill_port.clone()),
            )
            .unwrap()
            .with_ownership(Arc::clone(&prefill_ownership))
            .with_transfer(Arc::clone(&prefill_transfer)),
        );
        let decode_execution = Arc::new(
            LlamaCppBackendPhaseExecution::with_port(
                &decode_profile,
                Box::new(decode_port.clone()),
            )
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

        let execution_id = Uuid::new_v4();
        let source_epoch = Uuid::new_v4();
        let destination_epoch = Uuid::new_v4();
        let expires_at = chrono::Utc::now() + chrono::Duration::seconds(30);
        let completion_request = || {
            PhaseRequest::Completion(
                serde_json::from_value(serde_json::json!({ "prompt": "hi" })).unwrap(),
            )
        };

        let prepared = match prefill_executor
            .prepare(PreparePhaseExecution {
                execution_id,
                local_worker_epoch: source_epoch,
                model: "internal/model-v1".to_string(),
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
            PhaseDecision::Ready(_) => panic!("expected Ready prefill produce"),
            PhaseDecision::Recompute { .. } => panic!("expected Ready produce, got Recompute"),
            PhaseDecision::RetryableUnavailable { .. } => {
                panic!("expected Ready produce, got RetryableUnavailable")
            }
            PhaseDecision::TerminalFailure { .. } => {
                panic!("expected Ready produce, got TerminalFailure")
            }
        };

        let decode_prepared = match decode_executor
            .prepare(PreparePhaseExecution {
                execution_id,
                local_worker_epoch: destination_epoch,
                model: "internal/model-v1".to_string(),
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
        let _receipt = decode_transfer
            .consume_source(ConsumeStateTransfer {
                local_worker_epoch: destination_epoch,
                destination: decode_prepared.destination().clone(),
                source: published,
            })
            .await
            .unwrap();

        let imported = ImportedModelState::from_parts_for_test(
            execution_id,
            destination_epoch,
            decode_profile.sha256().unwrap(),
            decode_prepared.destination().clone(),
            decode_prepared.binding().clone(),
        );
        let output = decode_executor
            .execute(ExecutePhaseExecution::decode(decode_prepared, imported).unwrap())
            .await
            .unwrap();
        match output {
            PhaseDecision::Ready(PhaseExecutionOutput::Decode(mut stream)) => {
                use futures::StreamExt;
                let chunk = stream.next().await.expect("one chunk").unwrap();
                match chunk {
                    PhaseResponseChunk::Completion(chunk) => {
                        assert_eq!(chunk.text, "composed-adapter-token");
                        assert_eq!(chunk.token_id, Some(7));
                    }
                    PhaseResponseChunk::Chat(_) => panic!("expected completion chunk"),
                }
            }
            _ => panic!("expected Ready decode after restore+adapter"),
        }
        assert_eq!(
            decode_port.snapshot().unwrap(),
            snapshot,
            "Ready decode still requires set_state_data restore first"
        );
        assert!(!decode_profile.may_advertise_prefill_decode());
    }

    #[test]
    fn paired_for_profile_wires_shared_llamacpp_ownership_and_buffered_host_transfer() {
        let profile = decode_profile(
            Some(ServingCompositionStateOwnership::LlamaCpp),
            Some(ServingCompositionPhaseExecution::LlamaCpp),
        );
        let (transfer, executor) = BackendOwnedPhaseExecutor::paired_for_profile(&profile).unwrap();
        assert_eq!(transfer.health(), crate::serving::TransferHealth::Ready);
        assert_eq!(executor.health(), PhaseExecutorHealth::Ready);
        assert!(!executor.ownership().is_empty());
        assert!(executor.execution().can_produce_ready());
        assert!(!profile.may_advertise_prefill_decode());
        // ACL composition wires ownership+transfer into llamacpp execution;
        // Ready prefill still needs a bound context port (fixture or live).
    }

    #[test]
    fn acl_llamacpp_phase_execution_never_advertises_without_accepts_work() {
        let profile = decode_profile(
            Some(ServingCompositionStateOwnership::LlamaCpp),
            Some(ServingCompositionPhaseExecution::LlamaCpp),
        );
        assert!(!profile.may_advertise_prefill_decode());
        let (transfer, executor) = BackendOwnedPhaseExecutor::paired_for_profile(&profile).unwrap();
        assert_eq!(executor.health(), PhaseExecutorHealth::Ready);
        assert!(executor.health().accepts_work());
        assert!(executor.execution().can_produce_ready());
        let bounded = Arc::new(
            BoundedStateTransferService::new(profile.clone(), Uuid::new_v4(), transfer).unwrap(),
        );
        let runtime = DistributedServingRuntime::new(profile, bounded, executor).unwrap();
        // Injected + REQUIRED + Ready is execution-admissible for typed-outcome
        // HTTP, but may_advertise stays false so worker ready_phases do not
        // list P/D (not HSN; not honest advertise without a bound decode path).
        assert!(runtime.execution_admissible());
        assert!(!runtime.accepts_work());
    }

    #[tokio::test]
    async fn eligible_without_ready_execution_still_refuses_ready_prepare() {
        let profile = decode_profile(Some(ServingCompositionStateOwnership::LlamaCpp), None);
        let (transfer, executor) = BackendOwnedPhaseExecutor::paired_for_profile(&profile).unwrap();
        assert_eq!(transfer.health(), crate::serving::TransferHealth::Ready);
        assert_eq!(executor.health(), PhaseExecutorHealth::Eligible);
        assert!(!executor.health().accepts_work());
        let decision = executor
            .prepare(PreparePhaseExecution {
                execution_id: Uuid::new_v4(),
                local_worker_epoch: Uuid::new_v4(),
                model: "internal/model-v1".to_string(),
                expires_at: chrono::Utc::now() + chrono::Duration::seconds(30),
                request: PhaseRequest::Completion(
                    serde_json::from_value(serde_json::json!({ "prompt": "hi" })).unwrap(),
                ),
            })
            .await
            .unwrap();
        match decision {
            PhaseDecision::RetryableUnavailable { .. } => {}
            PhaseDecision::Ready(_) => panic!("Eligible without execution must refuse Ready"),
            PhaseDecision::Recompute { .. } => panic!("expected RetryableUnavailable"),
            PhaseDecision::TerminalFailure { .. } => panic!("expected RetryableUnavailable"),
        }
    }
}
