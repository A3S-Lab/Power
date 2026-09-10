//! llama.cpp backend phase state-ownership surface.
//!
//! The pinned `llama-cpp-rs` revision exposes context state snapshot APIs
//! (`llama_get_state_size` / `llama_copy_state_data` / `llama_set_state_data`).
//! This module binds layout identity from model/session facts and moves opaque
//! snapshot bytes through those APIs (or a test fixture that mirrors the same
//! size/copy/set contract). Power never parses KV tensors or invents a second
//! state format.
//!
//! ACL opt-in: `state_ownership = "llamacpp"` with `phase_executor =
//! backend-owned`. Binding advances health to Eligible after digest match.
//! Eligible still refuses Ready execute unless a Ready-capable
//! [`super::BackendPhaseExecution`] is bound, and never advertises P/D.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::{PowerError, Result};

use super::{BackendPhaseStateOwnership, ModelStateHandle, ServingExecutionProfile};

/// Domain-separated layout identity prefix for llama.cpp state ownership.
pub const LLAMACPP_STATE_LAYOUT_DOMAIN: &[u8] = b"a3s.power.llamacpp.state-layout.v1\0";

/// C API symbols this ownership surface requires from the pinned llama.cpp.
///
/// Present in `llama-cpp-rs` rev `dfd12e4d334846367e4284a2a7763fe92c1bf676`
/// (`llama-cpp-2` context session bindings).
pub const LLAMACPP_STATE_TRANSFER_API_SYMBOLS: &[&str] = &[
    "llama_get_state_size",
    "llama_copy_state_data",
    "llama_set_state_data",
];

/// Fail-closed capability probe for the llama.cpp state-transfer API surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaCppStateTransferApiProbe {
    /// Pinned `llama-cpp-rs` git revision Power depends on.
    pub pin_revision: &'static str,
    /// Whether this build links the `llamacpp` feature (sys crate present).
    pub linked_with_llamacpp_feature: bool,
    /// Symbols required for opaque snapshot import/export.
    pub required_symbols: &'static [&'static str],
}

/// Probe documenting that the pin exposes context state snapshot APIs.
///
/// This does **not** claim P/D Ready or close ROADMAP opaque-state exit
/// criteria by itself; it records the API surface concrete adapters must use.
#[must_use]
pub fn probe_llamacpp_state_transfer_api() -> LlamaCppStateTransferApiProbe {
    LlamaCppStateTransferApiProbe {
        pin_revision: "dfd12e4d334846367e4284a2a7763fe92c1bf676",
        linked_with_llamacpp_feature: cfg!(feature = "llamacpp"),
        required_symbols: LLAMACPP_STATE_TRANSFER_API_SYMBOLS,
    }
}

/// Model/session facts that determine llama.cpp KV / state layout identity.
///
/// Digests are computed only from these facts plus the domain prefix. Power
/// does not invent tensor layouts beyond what the backend reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaCppLayoutFacts {
    pub n_embd: u32,
    pub n_layer: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_ctx_train: u32,
    /// Effective context size for the session that owns the state snapshot.
    pub n_ctx: u32,
    pub n_vocab: i32,
    pub is_recurrent: bool,
    pub is_hybrid: bool,
}

impl LlamaCppLayoutFacts {
    /// Domain-separated SHA-256 of these layout facts.
    #[must_use]
    pub fn layout_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(LLAMACPP_STATE_LAYOUT_DOMAIN);
        hasher.update(self.n_embd.to_le_bytes());
        hasher.update(self.n_layer.to_le_bytes());
        hasher.update(self.n_head.to_le_bytes());
        hasher.update(self.n_head_kv.to_le_bytes());
        hasher.update(self.n_ctx_train.to_le_bytes());
        hasher.update(self.n_ctx.to_le_bytes());
        hasher.update(self.n_vocab.to_le_bytes());
        hasher.update([u8::from(self.is_recurrent)]);
        hasher.update([u8::from(self.is_hybrid)]);
        hex::encode(hasher.finalize())
    }
}

/// Thin port over llama.cpp context state snapshot get/set.
///
/// Production wraps `LlamaContext::{get_state_size,copy_state_data,set_state_data}`.
/// Tests may install a fixture that mirrors the same size/copy/set contract
/// without loading a GGUF.
pub trait LlamaCppContextStatePort: Send + Sync {
    fn state_byte_len(&self) -> usize;
    fn copy_state_into(&self, dest: &mut [u8]) -> Result<usize>;
    fn set_state_from(&mut self, src: &[u8]) -> Result<usize>;
}

/// Fixture port for first-principles tests (no GGUF / no native llama link).
///
/// Bytes are treated as opaque snapshots. Size mismatches fail closed the same
/// way a real `llama_set_state_data` shape mismatch would.
#[derive(Debug, Default)]
pub struct FixtureLlamaCppContextStatePort {
    snapshot: Vec<u8>,
}

impl FixtureLlamaCppContextStatePort {
    #[must_use]
    pub fn with_snapshot(snapshot: Vec<u8>) -> Self {
        Self { snapshot }
    }

    #[must_use]
    pub fn snapshot(&self) -> &[u8] {
        &self.snapshot
    }
}

impl LlamaCppContextStatePort for FixtureLlamaCppContextStatePort {
    fn state_byte_len(&self) -> usize {
        self.snapshot.len()
    }

    fn copy_state_into(&self, dest: &mut [u8]) -> Result<usize> {
        if dest.len() < self.snapshot.len() {
            return Err(PowerError::InvalidRequest(format!(
                "llama.cpp fixture state export buffer too small: need {}, got {}",
                self.snapshot.len(),
                dest.len()
            )));
        }
        dest[..self.snapshot.len()].copy_from_slice(&self.snapshot);
        Ok(self.snapshot.len())
    }

    fn set_state_from(&mut self, src: &[u8]) -> Result<usize> {
        if src.len() != self.snapshot.len() && !self.snapshot.is_empty() {
            return Err(PowerError::InvalidRequest(format!(
                "llama.cpp fixture state import size mismatch: expected {}, got {}",
                self.snapshot.len(),
                src.len()
            )));
        }
        self.snapshot = src.to_vec();
        Ok(self.snapshot.len())
    }
}

/// Real llama.cpp ownership adapter for backend-owned phase composition.
///
/// Layout identity comes from [`LlamaCppLayoutFacts`] (or matching profile
/// digests when constructed via [`Self::for_profile`]). Opaque import/export
/// store adapter-owned snapshot bytes; live capture/restore call a
/// [`LlamaCppContextStatePort`] that wraps the pinned session APIs.
pub struct LlamaCppBackendPhaseStateOwnership {
    layout_sha256: String,
    model_sha256: Option<String>,
    backend_sha256: Option<String>,
    execution_sha256: Option<String>,
    layout_facts: Option<LlamaCppLayoutFacts>,
    slots: Mutex<HashMap<String, Vec<u8>>>,
}

impl std::fmt::Debug for LlamaCppBackendPhaseStateOwnership {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LlamaCppBackendPhaseStateOwnership")
            .field("layout_sha256", &self.layout_sha256)
            .field("model_sha256", &self.model_sha256)
            .field("backend_sha256", &self.backend_sha256)
            .field("execution_sha256", &self.execution_sha256)
            .field("has_layout_facts", &self.layout_facts.is_some())
            .field("slot_count", &self.slots.lock().map(|g| g.len()).unwrap_or(0))
            .finish()
    }
}

impl LlamaCppBackendPhaseStateOwnership {
    /// Bind ownership from a prefill-decode profile's closed digests.
    ///
    /// Prefer [`Self::from_layout_facts`] when a loaded model/session is
    /// available so `layout_sha256` is computed from real backend facts.
    pub fn for_profile(profile: &ServingExecutionProfile) -> Result<Self> {
        let ServingExecutionProfile::PrefillDecode { execution } = profile else {
            return Err(PowerError::Config(
                "LlamaCppBackendPhaseStateOwnership requires a prefill-decode serving profile"
                    .to_string(),
            ));
        };
        Ok(Self {
            layout_sha256: execution.layout_sha256.clone(),
            model_sha256: Some(execution.model_sha256.clone()),
            backend_sha256: Some(execution.backend_sha256.clone()),
            execution_sha256: Some(execution.execution_sha256.clone()),
            layout_facts: None,
            slots: Mutex::new(HashMap::new()),
        })
    }

    /// Bind ownership from concrete llama.cpp layout facts plus related digests.
    pub fn from_layout_facts(
        facts: LlamaCppLayoutFacts,
        model_sha256: Option<String>,
        backend_sha256: Option<String>,
        execution_sha256: Option<String>,
    ) -> Self {
        Self {
            layout_sha256: facts.layout_sha256(),
            model_sha256,
            backend_sha256,
            execution_sha256,
            layout_facts: Some(facts),
            slots: Mutex::new(HashMap::new()),
        }
    }

    /// Optional layout facts when constructed via [`Self::from_layout_facts`].
    #[must_use]
    pub fn layout_facts(&self) -> Option<&LlamaCppLayoutFacts> {
        self.layout_facts.as_ref()
    }

    fn lock_slots(&self) -> Result<MutexGuard<'_, HashMap<String, Vec<u8>>>> {
        self.slots.lock().map_err(|_| {
            PowerError::InferenceFailed(
                "LlamaCppBackendPhaseStateOwnership slot map lock poisoned".to_string(),
            )
        })
    }

    /// Capture opaque state from a live context port via get/copy semantics.
    pub fn capture_from_port(
        &self,
        port: &dyn LlamaCppContextStatePort,
    ) -> Result<ModelStateHandle> {
        let size = port.state_byte_len();
        if size == 0 {
            return Err(PowerError::BackendNotAvailable(
                "LlamaCppBackendPhaseStateOwnership refuses capture: llama.cpp reported zero-sized state"
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
        let handle = ModelStateHandle::new(format!("llamacpp-state:{}", Uuid::new_v4()))?;
        self.lock_slots()?.insert(handle.as_str().to_string(), bytes);
        Ok(handle)
    }

    /// Restore opaque handle bytes into a live context port via set semantics.
    pub fn restore_into_port(
        &self,
        handle: &ModelStateHandle,
        port: &mut dyn LlamaCppContextStatePort,
    ) -> Result<()> {
        let bytes = self
            .lock_slots()?
            .get(handle.as_str())
            .cloned()
            .ok_or_else(|| {
                PowerError::InvalidRequest(format!(
                    "LlamaCppBackendPhaseStateOwnership has no opaque state for handle {}",
                    handle.as_str()
                ))
            })?;
        let applied = port.set_state_from(&bytes)?;
        if applied != bytes.len() {
            return Err(PowerError::InferenceFailed(format!(
                "llama.cpp state import size mismatch: expected {}, got {applied}",
                bytes.len()
            )));
        }
        Ok(())
    }
}

impl BackendPhaseStateOwnership for LlamaCppBackendPhaseStateOwnership {
    fn state_layout_sha256(&self) -> Option<&str> {
        Some(&self.layout_sha256)
    }

    fn model_sha256(&self) -> Option<&str> {
        self.model_sha256.as_deref()
    }

    fn backend_sha256(&self) -> Option<&str> {
        self.backend_sha256.as_deref()
    }

    fn execution_sha256(&self) -> Option<&str> {
        self.execution_sha256.as_deref()
    }

    fn import_opaque_state(&self, opaque: &[u8]) -> Result<ModelStateHandle> {
        if opaque.is_empty() {
            return Err(PowerError::InvalidRequest(
                "LlamaCppBackendPhaseStateOwnership refuses empty opaque import (llama.cpp state snapshots are non-empty)"
                    .to_string(),
            ));
        }
        let handle = ModelStateHandle::new(format!("llamacpp-imported:{}", Uuid::new_v4()))?;
        self.lock_slots()?
            .insert(handle.as_str().to_string(), opaque.to_vec());
        Ok(handle)
    }

    fn export_opaque_state(&self, handle: &ModelStateHandle) -> Result<Vec<u8>> {
        self.lock_slots()?
            .get(handle.as_str())
            .cloned()
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "LlamaCppBackendPhaseStateOwnership refuses opaque export: unknown handle (capture_from_port or import_opaque_state first)"
                        .to_string(),
                )
            })
    }
}

/// Wrap a mutable llama.cpp context as a [`LlamaCppContextStatePort`].
///
/// Only available when the `llamacpp` feature links `llama-cpp-2`.
#[cfg(feature = "llamacpp")]
pub struct LlamaCppContextStateApi<'a, 'b> {
    context: &'a mut llama_cpp_2::context::LlamaContext<'b>,
}

#[cfg(feature = "llamacpp")]
impl<'a, 'b> LlamaCppContextStateApi<'a, 'b> {
    #[must_use]
    pub fn new(context: &'a mut llama_cpp_2::context::LlamaContext<'b>) -> Self {
        Self { context }
    }
}

#[cfg(feature = "llamacpp")]
impl LlamaCppContextStatePort for LlamaCppContextStateApi<'_, '_> {
    fn state_byte_len(&self) -> usize {
        self.context.get_state_size()
    }

    fn copy_state_into(&self, dest: &mut [u8]) -> Result<usize> {
        let needed = self.context.get_state_size();
        if dest.len() < needed {
            return Err(PowerError::InvalidRequest(format!(
                "llama.cpp state export buffer too small: need {needed}, got {}",
                dest.len()
            )));
        }
        // SAFETY: dest has at least `needed` bytes; llama_copy_state_data writes
        // at most get_state_size() bytes into the provided buffer.
        let written = unsafe { self.context.copy_state_data(dest.as_mut_ptr()) };
        if written == 0 || written > dest.len() {
            return Err(PowerError::InferenceFailed(format!(
                "llama.cpp llama_copy_state_data failed or returned invalid length {written}"
            )));
        }
        Ok(written)
    }

    fn set_state_from(&mut self, src: &[u8]) -> Result<usize> {
        if src.is_empty() {
            return Err(PowerError::InvalidRequest(
                "llama.cpp llama_set_state_data refuses empty snapshot".to_string(),
            ));
        }
        // SAFETY: src is a contiguous opaque snapshot previously produced by
        // llama_copy_state_data (or an equivalent size-matched fixture in tests).
        let read = unsafe { self.context.set_state_data(src) };
        if read == 0 || read > src.len() {
            return Err(PowerError::InferenceFailed(format!(
                "llama.cpp llama_set_state_data failed or returned invalid length {read}"
            )));
        }
        Ok(read)
    }
}

#[cfg(feature = "llamacpp")]
impl LlamaCppLayoutFacts {
    /// Collect layout facts from a loaded `LlamaModel` and effective `n_ctx`.
    pub fn from_model(model: &llama_cpp_2::model::LlamaModel, n_ctx: u32) -> Self {
        Self {
            n_embd: u32::try_from(model.n_embd()).unwrap_or(0),
            n_layer: model.n_layer(),
            n_head: model.n_head(),
            n_head_kv: model.n_head_kv(),
            n_ctx_train: model.n_ctx_train(),
            n_ctx,
            n_vocab: model.n_vocab(),
            is_recurrent: model.is_recurrent(),
            is_hybrid: model.is_hybrid(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::{
        bind_backend_phase_state_ownership, DisaggregatedServingRole, PhaseExecutorHealth,
        PhaseSessionPoolMode, PhaseWeightCacheMode, PrefillDecodeExecutionProfile,
        ServingCompositionPhaseExecutor, ServingCompositionTransport, ServingPrivacyMode,
        StateKind, StateTransferProtocol,
    };

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn sample_facts() -> LlamaCppLayoutFacts {
        LlamaCppLayoutFacts {
            n_embd: 64,
            n_layer: 2,
            n_head: 4,
            n_head_kv: 2,
            n_ctx_train: 128,
            n_ctx: 64,
            n_vocab: 32,
            is_recurrent: false,
            is_hybrid: false,
        }
    }

    fn profile_with_layout(layout: String) -> ServingExecutionProfile {
        ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role: DisaggregatedServingRole::Decode,
            model: "internal/model-v1".to_string(),
            model_sha256: digest('1'),
            backend: "llamacpp".to_string(),
            backend_sha256: digest('2'),
            execution_sha256: digest('3'),
            device_sha256: digest('4'),
            layout_sha256: layout,
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
            state_ownership: Some(crate::serving::ServingCompositionStateOwnership::LlamaCpp),
            phase_execution: None,
        })
        .unwrap()
    }

    #[test]
    fn layout_facts_digest_is_stable_and_domain_separated() {
        let facts = sample_facts();
        let first = facts.layout_sha256();
        let second = facts.layout_sha256();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);

        let mut changed = facts.clone();
        changed.n_ctx = 65;
        assert_ne!(facts.layout_sha256(), changed.layout_sha256());
    }

    #[test]
    fn probe_documents_required_state_transfer_symbols() {
        let probe = probe_llamacpp_state_transfer_api();
        assert_eq!(
            probe.pin_revision,
            "dfd12e4d334846367e4284a2a7763fe92c1bf676"
        );
        assert!(probe
            .required_symbols
            .contains(&"llama_get_state_size"));
        assert!(probe
            .required_symbols
            .contains(&"llama_copy_state_data"));
        assert!(probe
            .required_symbols
            .contains(&"llama_set_state_data"));
        assert_eq!(
            probe.linked_with_llamacpp_feature,
            cfg!(feature = "llamacpp")
        );
    }

    #[test]
    fn from_layout_facts_binds_eligible_against_matching_profile() {
        let facts = sample_facts();
        let layout = facts.layout_sha256();
        let profile = profile_with_layout(layout.clone());
        let ownership = LlamaCppBackendPhaseStateOwnership::from_layout_facts(
            facts,
            Some(digest('1')),
            Some(digest('2')),
            Some(digest('3')),
        );
        assert_eq!(ownership.state_layout_sha256(), Some(layout.as_str()));
        let health = bind_backend_phase_state_ownership(&ownership, &profile).unwrap();
        assert_eq!(health, PhaseExecutorHealth::Eligible);
        assert!(!health.accepts_work());
    }

    #[test]
    fn fixture_port_capture_restore_round_trips_opaque_bytes() {
        let facts = sample_facts();
        let ownership = LlamaCppBackendPhaseStateOwnership::from_layout_facts(
            facts,
            None,
            None,
            None,
        );
        let port = FixtureLlamaCppContextStatePort::with_snapshot(b"llama-state-fixture".to_vec());
        let handle = ownership.capture_from_port(&port).unwrap();
        let exported = ownership.export_opaque_state(&handle).unwrap();
        assert_eq!(exported, b"llama-state-fixture");

        let mut dest = FixtureLlamaCppContextStatePort::with_snapshot(vec![0u8; exported.len()]);
        ownership.restore_into_port(&handle, &mut dest).unwrap();
        assert_eq!(dest.snapshot(), b"llama-state-fixture");

        // Wire-boundary import then restore.
        let imported = ownership.import_opaque_state(&exported).unwrap();
        let mut dest2 = FixtureLlamaCppContextStatePort::with_snapshot(vec![0u8; exported.len()]);
        ownership.restore_into_port(&imported, &mut dest2).unwrap();
        assert_eq!(dest2.snapshot(), exported);
    }

    #[test]
    fn empty_opaque_import_and_unknown_export_fail_closed() {
        let ownership =
            LlamaCppBackendPhaseStateOwnership::from_layout_facts(sample_facts(), None, None, None);
        let err = ownership.import_opaque_state(&[]).unwrap_err();
        assert!(err.to_string().contains("empty opaque import"));

        let handle = ModelStateHandle::new("missing").unwrap();
        let err = ownership.export_opaque_state(&handle).unwrap_err();
        assert!(err.to_string().contains("unknown handle"));
    }

    #[test]
    fn zero_sized_capture_fails_closed() {
        let ownership =
            LlamaCppBackendPhaseStateOwnership::from_layout_facts(sample_facts(), None, None, None);
        let port = FixtureLlamaCppContextStatePort::with_snapshot(Vec::new());
        let err = ownership.capture_from_port(&port).unwrap_err();
        assert!(err.to_string().contains("zero-sized state"));
    }

    #[test]
    fn for_profile_rejects_non_prefill_decode() {
        let err =
            LlamaCppBackendPhaseStateOwnership::for_profile(&ServingExecutionProfile::default())
                .unwrap_err();
        assert!(err.to_string().contains("prefill-decode"));
    }
}
