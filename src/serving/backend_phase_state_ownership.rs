//! Opaque backend phase state-ownership surface.
//!
//! Concrete backends (llama.cpp, picolm) implement [`BackendPhaseStateOwnership`]
//! later. Power only validates opaque layout identity and related bindings
//! against the immutable serving profile. This module does **not** invent KV
//! tensor layout or model-semantic import/export.
//!
//! [`ProfileBoundBackendPhaseStateOwnership`] is an honest interim product
//! surface: it mirrors closed profile digests (including the closed backend
//! artifact digest) so [`super::BackendOwnedPhaseExecutor`] can become
//! Eligible without claiming a real backend or KV layout.

use crate::error::{PowerError, Result};

use super::{ModelStateHandle, PhaseExecutorHealth, ServingExecutionProfile};

/// Opaque ownership + import/export hooks for backend-owned phase execution.
///
/// Implementors own layout identity and byte movement. Power never inspects KV
/// payloads and does not treat successful import/export as Ready decode.
pub trait BackendPhaseStateOwnership: Send + Sync {
    /// Empty default surface. Never becomes [`PhaseExecutorHealth::Eligible`].
    fn is_empty(&self) -> bool {
        false
    }

    /// Opaque model-owned state layout digest.
    ///
    /// Must equal the immutable profile `layout_sha256` when non-Empty.
    /// Empty returns [`None`].
    fn state_layout_sha256(&self) -> Option<&str>;

    /// Optional related binding; when present must match profile `model_sha256`.
    fn model_sha256(&self) -> Option<&str> {
        None
    }

    /// Optional related binding; when present must match profile `backend_sha256`.
    fn backend_sha256(&self) -> Option<&str> {
        None
    }

    /// Optional related binding; when present must match profile `execution_sha256`.
    fn execution_sha256(&self) -> Option<&str> {
        None
    }

    /// Opaque import: adapter-owned bytes → local handle. No Power-side KV parse.
    fn import_opaque_state(&self, opaque: &[u8]) -> Result<ModelStateHandle>;

    /// Opaque export: local handle → adapter-owned bytes. No Power-side KV parse.
    fn export_opaque_state(&self, handle: &ModelStateHandle) -> Result<Vec<u8>>;
}

/// Default Empty ownership. Keeps [`super::BackendOwnedPhaseExecutor`] Unavailable.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmptyBackendPhaseStateOwnership;

impl BackendPhaseStateOwnership for EmptyBackendPhaseStateOwnership {
    fn is_empty(&self) -> bool {
        true
    }

    fn state_layout_sha256(&self) -> Option<&str> {
        None
    }

    fn import_opaque_state(&self, _opaque: &[u8]) -> Result<ModelStateHandle> {
        Err(PowerError::BackendNotAvailable(
            "EmptyBackendPhaseStateOwnership refuses opaque import until a real backend ownership adapter is bound"
                .to_string(),
        ))
    }

    fn export_opaque_state(&self, _handle: &ModelStateHandle) -> Result<Vec<u8>> {
        Err(PowerError::BackendNotAvailable(
            "EmptyBackendPhaseStateOwnership refuses opaque export until a real backend ownership adapter is bound"
                .to_string(),
        ))
    }
}

/// Profile-digest ownership surface for backend-owned phase composition.
///
/// Binds exact `layout_sha256`, `model_sha256`, `backend_sha256` (closed backend
/// artifact digest), and `execution_sha256` from the immutable serving profile.
/// This is **not** a real llama.cpp / picolm ownership adapter: it does not own
/// KV tensors, does not invent layout semantics, and opaque import/export fail
/// closed until a concrete backend implementor exists.
///
/// Binding this surface to [`super::BackendOwnedPhaseExecutor`] advances health
/// to [`PhaseExecutorHealth::Eligible`] after fail-closed digest validation.
/// Eligible still refuses Ready prepare/execute and never advertises
/// model-semantic P/D.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileBoundBackendPhaseStateOwnership {
    layout_sha256: String,
    model_sha256: String,
    /// Closed backend artifact digest from the immutable profile.
    backend_sha256: String,
    execution_sha256: String,
}

impl ProfileBoundBackendPhaseStateOwnership {
    /// Mirror digests from a prefill-decode serving profile.
    ///
    /// Fails closed when the profile is not prefill-decode. Digests are copied
    /// verbatim; this constructor does not validate SHA-256 shape (bind-time
    /// validation remains in [`bind_backend_phase_state_ownership`]).
    pub fn for_profile(profile: &ServingExecutionProfile) -> Result<Self> {
        let ServingExecutionProfile::PrefillDecode { execution } = profile else {
            return Err(PowerError::Config(
                "ProfileBoundBackendPhaseStateOwnership requires a prefill-decode serving profile"
                    .to_string(),
            ));
        };
        Ok(Self {
            layout_sha256: execution.layout_sha256.clone(),
            model_sha256: execution.model_sha256.clone(),
            backend_sha256: execution.backend_sha256.clone(),
            execution_sha256: execution.execution_sha256.clone(),
        })
    }

    /// Closed backend artifact digest mirrored from the profile.
    pub fn closed_backend_artifact_sha256(&self) -> &str {
        &self.backend_sha256
    }
}

impl BackendPhaseStateOwnership for ProfileBoundBackendPhaseStateOwnership {
    fn state_layout_sha256(&self) -> Option<&str> {
        Some(&self.layout_sha256)
    }

    fn model_sha256(&self) -> Option<&str> {
        Some(&self.model_sha256)
    }

    fn backend_sha256(&self) -> Option<&str> {
        Some(&self.backend_sha256)
    }

    fn execution_sha256(&self) -> Option<&str> {
        Some(&self.execution_sha256)
    }

    fn import_opaque_state(&self, _opaque: &[u8]) -> Result<ModelStateHandle> {
        Err(PowerError::BackendNotAvailable(
            "ProfileBoundBackendPhaseStateOwnership refuses opaque import: profile digest binding is not a real backend KV ownership adapter"
                .to_string(),
        ))
    }

    fn export_opaque_state(&self, _handle: &ModelStateHandle) -> Result<Vec<u8>> {
        Err(PowerError::BackendNotAvailable(
            "ProfileBoundBackendPhaseStateOwnership refuses opaque export: profile digest binding is not a real backend KV ownership adapter"
                .to_string(),
        ))
    }
}

/// Validate ownership against the immutable profile.
///
/// - Empty → [`PhaseExecutorHealth::Unavailable`] (no error).
/// - Non-Empty with matching `state_layout_sha256` (+ related bindings) →
///   [`PhaseExecutorHealth::Eligible`].
/// - Mismatched layout or related digests → fail closed (`Config` error).
///
/// Eligible still does **not** mean Ready execute exists.
pub fn bind_backend_phase_state_ownership(
    ownership: &dyn BackendPhaseStateOwnership,
    profile: &ServingExecutionProfile,
) -> Result<PhaseExecutorHealth> {
    if ownership.is_empty() {
        return Ok(PhaseExecutorHealth::Unavailable);
    }

    let ServingExecutionProfile::PrefillDecode { execution } = profile else {
        return Err(PowerError::Config(
            "backend phase state ownership requires a prefill-decode serving profile".to_string(),
        ));
    };

    let Some(layout) = ownership.state_layout_sha256() else {
        return Err(PowerError::Config(
            "backend phase state ownership must declare state_layout_sha256".to_string(),
        ));
    };
    validate_sha256(layout, "backend phase state_layout")?;
    if layout != execution.layout_sha256 {
        return Err(PowerError::Config(
            "backend phase state ownership state_layout_sha256 does not match the immutable serving profile layout_sha256"
                .to_string(),
        ));
    }

    if let Some(model) = ownership.model_sha256() {
        validate_sha256(model, "backend phase ownership model")?;
        if model != execution.model_sha256 {
            return Err(PowerError::Config(
                "backend phase state ownership model_sha256 does not match the immutable serving profile"
                    .to_string(),
            ));
        }
    }
    if let Some(backend) = ownership.backend_sha256() {
        validate_sha256(backend, "backend phase ownership backend")?;
        if backend != execution.backend_sha256 {
            return Err(PowerError::Config(
                "backend phase state ownership backend_sha256 does not match the immutable serving profile"
                    .to_string(),
            ));
        }
    }
    if let Some(execution_digest) = ownership.execution_sha256() {
        validate_sha256(execution_digest, "backend phase ownership execution")?;
        if execution_digest != execution.execution_sha256 {
            return Err(PowerError::Config(
                "backend phase state ownership execution_sha256 does not match the immutable serving profile"
                    .to_string(),
            ));
        }
    }

    Ok(PhaseExecutorHealth::Eligible)
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PowerError::Config(format!(
            "{label} SHA-256 must contain exactly 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::{
        DisaggregatedServingRole, PhaseSessionPoolMode, PhaseWeightCacheMode,
        PrefillDecodeExecutionProfile, ServingCompositionPhaseExecutor,
        ServingCompositionTransport, ServingPrivacyMode, StateKind, StateTransferProtocol,
    };

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn profile() -> ServingExecutionProfile {
        ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role: DisaggregatedServingRole::Decode,
            model: "internal/model-v1".to_string(),
            model_sha256: digest('1'),
            backend: "backend-owned".to_string(),
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
            state_ownership: None,
        })
        .unwrap()
    }

    struct StubOwnership {
        layout: String,
        model: Option<String>,
        backend: Option<String>,
        execution: Option<String>,
    }

    impl BackendPhaseStateOwnership for StubOwnership {
        fn state_layout_sha256(&self) -> Option<&str> {
            Some(&self.layout)
        }

        fn model_sha256(&self) -> Option<&str> {
            self.model.as_deref()
        }

        fn backend_sha256(&self) -> Option<&str> {
            self.backend.as_deref()
        }

        fn execution_sha256(&self) -> Option<&str> {
            self.execution.as_deref()
        }

        fn import_opaque_state(&self, _opaque: &[u8]) -> Result<ModelStateHandle> {
            ModelStateHandle::new("stub-imported")
        }

        fn export_opaque_state(&self, _handle: &ModelStateHandle) -> Result<Vec<u8>> {
            Ok(b"stub".to_vec())
        }
    }

    #[test]
    fn empty_ownership_stays_unavailable() {
        let health =
            bind_backend_phase_state_ownership(&EmptyBackendPhaseStateOwnership, &profile())
                .unwrap();
        assert_eq!(health, PhaseExecutorHealth::Unavailable);
    }

    #[test]
    fn mismatched_layout_fails_closed() {
        let ownership = StubOwnership {
            layout: digest('9'),
            model: None,
            backend: None,
            execution: None,
        };
        let err = bind_backend_phase_state_ownership(&ownership, &profile()).unwrap_err();
        assert!(err.to_string().contains("state_layout_sha256"));
        assert!(err.to_string().contains("does not match"));
    }

    #[test]
    fn matching_layout_becomes_eligible() {
        let ownership = StubOwnership {
            layout: digest('5'),
            model: Some(digest('1')),
            backend: Some(digest('2')),
            execution: Some(digest('3')),
        };
        let health = bind_backend_phase_state_ownership(&ownership, &profile()).unwrap();
        assert_eq!(health, PhaseExecutorHealth::Eligible);
        assert!(!health.accepts_work());
    }

    #[test]
    fn mismatched_related_binding_fails_closed() {
        let ownership = StubOwnership {
            layout: digest('5'),
            model: Some(digest('9')),
            backend: None,
            execution: None,
        };
        let err = bind_backend_phase_state_ownership(&ownership, &profile()).unwrap_err();
        assert!(err.to_string().contains("model_sha256"));
    }

    #[test]
    fn profile_bound_mirrors_closed_digests_and_becomes_eligible() {
        let profile = profile();
        let ownership = ProfileBoundBackendPhaseStateOwnership::for_profile(&profile).unwrap();
        assert_eq!(ownership.state_layout_sha256(), Some(digest('5').as_str()));
        assert_eq!(ownership.model_sha256(), Some(digest('1').as_str()));
        assert_eq!(
            ownership.closed_backend_artifact_sha256(),
            digest('2').as_str()
        );
        assert_eq!(ownership.backend_sha256(), Some(digest('2').as_str()));
        assert_eq!(ownership.execution_sha256(), Some(digest('3').as_str()));
        assert!(!ownership.is_empty());

        let health = bind_backend_phase_state_ownership(&ownership, &profile).unwrap();
        assert_eq!(health, PhaseExecutorHealth::Eligible);
        assert!(!health.accepts_work());
    }

    #[test]
    fn profile_bound_opaque_hooks_fail_closed_without_kv_semantics() {
        let ownership = ProfileBoundBackendPhaseStateOwnership::for_profile(&profile()).unwrap();
        let import_err = ownership.import_opaque_state(b"bytes").unwrap_err();
        assert!(import_err.to_string().contains("refuses opaque import"));
        assert!(import_err.to_string().contains("not a real backend"));

        let handle = ModelStateHandle::new("local-slot").unwrap();
        let export_err = ownership.export_opaque_state(&handle).unwrap_err();
        assert!(export_err.to_string().contains("refuses opaque export"));
        assert!(export_err.to_string().contains("not a real backend"));
    }

    #[test]
    fn profile_bound_rejects_non_prefill_decode_profile() {
        let err = ProfileBoundBackendPhaseStateOwnership::for_profile(
            &ServingExecutionProfile::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("prefill-decode"));
    }
}
