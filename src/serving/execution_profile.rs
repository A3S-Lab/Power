use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};

use super::state_transfer::{MAX_INFLIGHT_TRANSFERS, MAX_TRANSFER_BYTES};
use super::{
    PhaseExecutorCapabilities, ServingPhase, StateKind, StateTransferBinding,
    StateTransferCapabilities, StateTransferProtocol,
};

const MAX_MODEL_NAME_BYTES: usize = 256;
const MAX_BACKEND_NAME_BYTES: usize = 128;
const MAX_EXACT_ACL_INTEGER: u64 = (1_u64 << 53) - 1;
const MAX_TRANSFER_TIMEOUT_MS: u64 = 300_000;

/// One execution role owned by a worker in a disaggregated deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DisaggregatedServingRole {
    Prefill,
    Decode,
}

impl From<DisaggregatedServingRole> for ServingPhase {
    fn from(role: DisaggregatedServingRole) -> Self {
        match role {
            DisaggregatedServingRole::Prefill => Self::Prefill,
            DisaggregatedServingRole::Decode => Self::Decode,
        }
    }
}

/// Privacy boundary that a concrete state-transfer adapter must enforce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ServingPrivacyMode {
    /// The data path authenticates both peers and encrypts state in transit.
    AuthenticatedEncryptedTransport,
    /// An attested peer set and its private fabric form the reviewed boundary.
    AttestedPrivateFabric,
}

/// Explicit composition selector for product-surface P/D adapters.
///
/// Absent means the composition root must inject both ports through
/// [`crate::server::PowerServerBuilder`]. Setting a value is an honest opt-in
/// that installs a concrete product pair at startup; it is not a silent
/// inference from `protocol` alone and does not claim high-speed network
/// readiness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ServingCompositionTransport {
    /// Install `BufferedHostLoopbackStateTransfer` +
    /// `BufferedHostLoopbackPhaseExecutor` for the immutable profile
    /// (unless `phase_executor = backend-owned` overrides the phase port).
    BufferedHostLoopback,
    /// Install `DirectDeviceMemoryPullStateTransfer` +
    /// `DirectDeviceMemoryPullPhaseExecutor` for `DirectDeviceMemoryPullV1`.
    ///
    /// The pair is a named product port: Injected + required contract, but
    /// Unavailable until a real high-speed adapter exists. It never claims
    /// HSN evidence and must not advertise P/D readiness.
    DirectDeviceMemoryPull,
}

impl ServingCompositionTransport {
    /// Whether a healthy injection of this transport may list prefill/decode
    /// in worker `ready_phases`.
    ///
    /// Buffered-host loopback may advertise when adapters accept work.
    /// DirectDeviceMemoryPull never advertises: no in-tree HSN adapter exists.
    pub fn may_advertise_prefill_decode(self) -> bool {
        match self {
            Self::BufferedHostLoopback => true,
            Self::DirectDeviceMemoryPull => false,
        }
    }
}

impl fmt::Display for ServingCompositionTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferedHostLoopback => formatter.write_str("buffered-host-loopback"),
            Self::DirectDeviceMemoryPull => formatter.write_str("direct-device-memory-pull"),
        }
    }
}

/// Optional honest composition phase-executor product port.
///
/// Absent keeps the transport's default phase companion (Ready loopback
/// conformance or Unavailable DirectDeviceMemoryPull). `backend-owned`
/// installs [`crate::serving::BackendOwnedPhaseExecutor`]: Injected + required
/// contract, Unavailable under Empty ownership until a non-Empty
/// [`crate::serving::BackendPhaseStateOwnership`] binds matching
/// `state_layout_sha256` (Eligible). Ready execute remains blocked. It pairs
/// only with buffered-host loopback transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ServingCompositionPhaseExecutor {
    /// Install `BackendOwnedPhaseExecutor` with buffered-host loopback transfer.
    BackendOwned,
}

impl ServingCompositionPhaseExecutor {
    /// Backend-owned product port never advertises P/D readiness.
    pub fn may_advertise_prefill_decode(self) -> bool {
        match self {
            Self::BackendOwned => false,
        }
    }
}

impl fmt::Display for ServingCompositionPhaseExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendOwned => formatter.write_str("backend-owned"),
        }
    }
}

/// Optional honest composition state-ownership surface for backend-owned phase.
///
/// Absent (or omitted) keeps default Empty ownership → Unavailable.
/// `profile-bound` requires `phase_executor = backend-owned` and installs
/// [`crate::serving::ProfileBoundBackendPhaseStateOwnership`] so the executor
/// can become Eligible after fail-closed digest validation. `llamacpp`
/// requires `phase_executor = backend-owned` and installs
/// [`crate::serving::LlamaCppBackendPhaseStateOwnership`] (layout identity +
/// opaque snapshot import/export via the pinned llama.cpp state APIs or a
/// fixture port). Eligible still refuses Ready execute unless a Ready-capable
/// [`crate::serving::BackendPhaseExecution`] is also bound, and never
/// advertises P/D from ownership alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ServingCompositionStateOwnership {
    /// Mirror closed profile digests into interim Eligible ownership (not a
    /// real llama.cpp / picolm KV adapter).
    ProfileBound,
    /// Real llama.cpp ownership adapter (opaque `llama_*_state_*` snapshots).
    LlamaCpp,
}

impl fmt::Display for ServingCompositionStateOwnership {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProfileBound => formatter.write_str("profile-bound"),
            Self::LlamaCpp => formatter.write_str("llamacpp"),
        }
    }
}

/// Optional honest composition phase-execution surface for backend-owned phase.
///
/// Absent keeps default Empty execution → Eligible ownership still refuses
/// Ready. `pending` requires `phase_executor = backend-owned` and installs
/// [`crate::serving::PendingBackendPhaseExecution`] so Eligible ownership can
/// advance to Ready health. Pending unlocks the Ready gate only; prepare /
/// execute / abort fail closed until a concrete backend implementor exists.
/// Worker `ready_phases` stay suppressed via backend-owned
/// `may_advertise_prefill_decode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ServingCompositionPhaseExecution {
    /// Bind interim Ready-capable execution (not a real llama.cpp / picolm
    /// prepare/execute adapter).
    Pending,
}

impl fmt::Display for ServingCompositionPhaseExecution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => formatter.write_str("pending"),
        }
    }
}

/// How a prefill/decode worker obtains model weights.
///
/// Only the shared process weight hierarchy / residency path is accepted.
/// Unknown serde variants (for example a private phase-local cache) fail closed
/// so P/D composition cannot install a second weight cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PhaseWeightCacheMode {
    /// Reuse Power's process-shared weight hierarchy / residency path.
    #[default]
    SharedWeightHierarchy,
}

impl PhaseWeightCacheMode {
    pub(crate) fn is_shared_weight_hierarchy(&self) -> bool {
        matches!(self, Self::SharedWeightHierarchy)
    }
}

/// Unknown serde variants (for example a private phase-local replica pool) fail
/// closed so P/D composition cannot mint a second session pool.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PhaseSessionPoolMode {
    /// Reuse Power's process-shared `ModelSessionPool` / replica path.
    #[default]
    SharedSessionPool,
}

impl PhaseSessionPoolMode {
    pub(crate) fn is_shared_session_pool(&self) -> bool {
        matches!(self, Self::SharedSessionPool)
    }
}

/// Cloud-certified deployment identity shared by compatible prefill/decode peers.
///
/// Prefill and decode roles differ, so full profile digests differ across peers.
/// Compatible workers must still agree on deployment generation and peer set
/// before opaque state may cross the process boundary. Process epoch alone is
/// insufficient: a restarted peer under a newer generation must fail closed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServingDeploymentIdentity {
    pub generation: u64,
    pub peer_set_sha256: String,
}

impl ServingDeploymentIdentity {
    pub fn validate(&self) -> Result<()> {
        if self.generation == 0 || self.generation > MAX_EXACT_ACL_INTEGER {
            return Err(PowerError::InvalidRequest(format!(
                "serving deployment generation must be within 1..={MAX_EXACT_ACL_INTEGER}"
            )));
        }
        validate_sha256(&self.peer_set_sha256, "serving deployment peer set")?;
        Ok(())
    }
}

/// Static facts required by a prefill or decode process generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillDecodeExecutionProfile {
    pub role: DisaggregatedServingRole,
    pub model: String,
    pub model_sha256: String,
    pub backend: String,
    pub backend_sha256: String,
    pub execution_sha256: String,
    pub device_sha256: String,
    pub layout_sha256: String,
    pub peer_set_sha256: String,
    pub generation: u64,
    pub protocol: StateTransferProtocol,
    pub state_kind: StateKind,
    pub max_state_bytes: u64,
    pub max_inflight_transfers: u32,
    pub transfer_timeout_ms: u64,
    pub cancellation_timeout_ms: u64,
    pub privacy: ServingPrivacyMode,
    pub privacy_policy_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attestation_policy_sha256: Option<String>,
    /// Closed weight-cache binding. Defaults to the shared hierarchy.
    #[serde(
        default,
        skip_serializing_if = "PhaseWeightCacheMode::is_shared_weight_hierarchy"
    )]
    pub weight_cache: PhaseWeightCacheMode,
    /// Optional digest of the process residency policy pinned into this
    /// profile. When set, an installed weight hierarchy must report the same
    /// digest. Absent means shared-hierarchy mode without a pinned policy
    /// document (fixtures / adapters not yet bound).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residency_policy_sha256: Option<String>,
    /// Closed session-replica pool binding. Defaults to the shared pool.
    #[serde(
        default,
        skip_serializing_if = "PhaseSessionPoolMode::is_shared_session_pool"
    )]
    pub session_pool: PhaseSessionPoolMode,
    /// Optional digest of the process `ModelSessionPoolPolicy` pinned into this
    /// profile. When set, an installed session pool must report the same
    /// digest. Absent means shared-pool mode without a pinned policy document
    /// (fixtures / adapters not yet bound).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_pool_policy_sha256: Option<String>,
    /// Honest composition opt-in for a product-surface adapter pair.
    ///
    /// Absent keeps fail-closed external injection. `buffered-host-loopback`
    /// installs the product loopback pair only when protocol/privacy match.
    /// `direct-device-memory-pull` installs the Unavailable HSN product port
    /// when protocol is `DirectDeviceMemoryPullV1`. Protocol alone never
    /// auto-wires and neither opt-in claims HSN evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<ServingCompositionTransport>,
    /// Honest composition opt-in for a product-surface phase executor.
    ///
    /// Absent keeps the transport's default phase companion.
    /// `backend-owned` requires `transport = buffered-host-loopback` and
    /// installs `BackendOwnedPhaseExecutor` (Empty → Unavailable; matching
    /// ownership bind → Eligible; Ready execute still blocked) instead of the
    /// Ready loopback conformance executor. Never claims model-semantic P/D.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_executor: Option<ServingCompositionPhaseExecutor>,
    /// Honest composition opt-in for backend-owned phase state ownership.
    ///
    /// Absent keeps Empty ownership (Unavailable). `profile-bound` requires
    /// `phase_executor = backend-owned` and binds
    /// `ProfileBoundBackendPhaseStateOwnership` → Eligible without Ready unless
    /// a Ready-capable `phase_execution` is also bound. `llamacpp` requires
    /// `phase_executor = backend-owned` and binds
    /// `LlamaCppBackendPhaseStateOwnership` (layout facts / opaque snapshots
    /// via pinned llama.cpp state APIs). Digest mismatch with the immutable
    /// profile fails closed at bind time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_ownership: Option<ServingCompositionStateOwnership>,
    /// Honest composition opt-in for backend-owned phase prepare/execute.
    ///
    /// Absent keeps Empty execution (Eligible still refuses Ready). `pending`
    /// requires `phase_executor = backend-owned` and binds
    /// `PendingBackendPhaseExecution` so Eligible ownership can advance to
    /// Ready health. Pending prepare/execute fail closed; this is not
    /// model-semantic P/D.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_execution: Option<ServingCompositionPhaseExecution>,
}

/// Immutable execution profile for one Power process generation.
///
/// The default aggregated variant preserves ordinary local inference. The
/// disaggregated variant binds every static fact that must agree before a
/// phase executor or state-transfer adapter can be used. Request-specific
/// token counts, state sizes, worker epochs, and deadlines remain bound by the
/// state-transfer command and descriptor types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "profile", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ServingExecutionProfile {
    /// Ordinary local inference. Extra ACL attributes (for example `transport`)
    /// fail closed so aggregated never silently ignores a P/D opt-in.
    Aggregated {},
    PrefillDecode {
        #[serde(flatten)]
        execution: Box<PrefillDecodeExecutionProfile>,
    },
}

impl Default for ServingExecutionProfile {
    fn default() -> Self {
        Self::Aggregated {}
    }
}

impl ServingExecutionProfile {
    pub fn prefill_decode(execution: PrefillDecodeExecutionProfile) -> Result<Self> {
        let profile = Self::PrefillDecode {
            execution: Box::new(execution),
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> Result<()> {
        let Self::PrefillDecode { execution } = self else {
            return Ok(());
        };
        let PrefillDecodeExecutionProfile {
            model,
            model_sha256,
            backend,
            backend_sha256,
            execution_sha256,
            device_sha256,
            layout_sha256,
            peer_set_sha256,
            generation,
            max_state_bytes,
            max_inflight_transfers,
            transfer_timeout_ms,
            cancellation_timeout_ms,
            privacy,
            privacy_policy_sha256,
            attestation_policy_sha256,
            residency_policy_sha256,
            session_pool_policy_sha256,
            transport,
            phase_executor,
            state_ownership,
            phase_execution,
            protocol,
            ..
        } = execution.as_ref();

        validate_identifier(model, MAX_MODEL_NAME_BYTES, "serving model")?;
        validate_identifier(backend, MAX_BACKEND_NAME_BYTES, "serving backend")?;
        for (value, label) in [
            (model_sha256, "serving model"),
            (backend_sha256, "serving backend"),
            (execution_sha256, "serving execution"),
            (device_sha256, "serving device"),
            (layout_sha256, "serving state layout"),
            (peer_set_sha256, "serving peer set"),
            (privacy_policy_sha256, "serving privacy policy"),
        ] {
            validate_sha256(value, label)?;
        }
        if let Some(policy) = attestation_policy_sha256 {
            validate_sha256(policy, "serving attestation policy")?;
        }
        if let Some(policy) = residency_policy_sha256 {
            validate_sha256(policy, "serving residency policy")?;
        }
        if let Some(policy) = session_pool_policy_sha256 {
            validate_sha256(policy, "serving session pool policy")?;
        }
        if matches!(privacy, ServingPrivacyMode::AttestedPrivateFabric)
            && attestation_policy_sha256.is_none()
        {
            return Err(PowerError::Config(
                "attested-private-fabric serving requires attestation_policy_sha256".to_string(),
            ));
        }
        match transport {
            Some(ServingCompositionTransport::BufferedHostLoopback)
                if !matches!(protocol, StateTransferProtocol::BufferedHostMemoryPullV1) =>
            {
                return Err(PowerError::Config(
                    "serving_execution.transport = buffered-host-loopback requires protocol = buffered-host-memory-pull-v1"
                        .to_string(),
                ));
            }
            Some(ServingCompositionTransport::BufferedHostLoopback)
                if !matches!(privacy, ServingPrivacyMode::AuthenticatedEncryptedTransport) =>
            {
                return Err(PowerError::Config(
                    "serving_execution.transport = buffered-host-loopback requires privacy = authenticated-encrypted-transport"
                        .to_string(),
                ));
            }
            Some(ServingCompositionTransport::BufferedHostLoopback) => {}
            Some(ServingCompositionTransport::DirectDeviceMemoryPull)
                if !matches!(protocol, StateTransferProtocol::DirectDeviceMemoryPullV1) =>
            {
                return Err(PowerError::Config(
                    "serving_execution.transport = direct-device-memory-pull requires protocol = direct-device-memory-pull-v1"
                        .to_string(),
                ));
            }
            Some(ServingCompositionTransport::DirectDeviceMemoryPull) => {}
            None => {}
        }
        match phase_executor {
            Some(ServingCompositionPhaseExecutor::BackendOwned)
                if !matches!(
                    transport,
                    Some(ServingCompositionTransport::BufferedHostLoopback)
                ) =>
            {
                return Err(PowerError::Config(
                    "serving_execution.phase_executor = backend-owned requires transport = buffered-host-loopback"
                        .to_string(),
                ));
            }
            Some(ServingCompositionPhaseExecutor::BackendOwned) | None => {}
        }
        match state_ownership {
            Some(ServingCompositionStateOwnership::ProfileBound)
                if !matches!(
                    phase_executor,
                    Some(ServingCompositionPhaseExecutor::BackendOwned)
                ) =>
            {
                return Err(PowerError::Config(
                    "serving_execution.state_ownership = profile-bound requires phase_executor = backend-owned"
                        .to_string(),
                ));
            }
            Some(ServingCompositionStateOwnership::LlamaCpp)
                if !matches!(
                    phase_executor,
                    Some(ServingCompositionPhaseExecutor::BackendOwned)
                ) =>
            {
                return Err(PowerError::Config(
                    "serving_execution.state_ownership = llamacpp requires phase_executor = backend-owned"
                        .to_string(),
                ));
            }
            Some(
                ServingCompositionStateOwnership::ProfileBound
                | ServingCompositionStateOwnership::LlamaCpp,
            )
            | None => {}
        }
        match phase_execution {
            Some(ServingCompositionPhaseExecution::Pending)
                if !matches!(
                    phase_executor,
                    Some(ServingCompositionPhaseExecutor::BackendOwned)
                ) =>
            {
                return Err(PowerError::Config(
                    "serving_execution.phase_execution = pending requires phase_executor = backend-owned"
                        .to_string(),
                ));
            }
            Some(ServingCompositionPhaseExecution::Pending) | None => {}
        }
        if *generation == 0 || *generation > MAX_EXACT_ACL_INTEGER {
            return Err(PowerError::Config(format!(
                "serving generation must be within 1..={MAX_EXACT_ACL_INTEGER}"
            )));
        }
        if *max_state_bytes == 0 || *max_state_bytes > MAX_TRANSFER_BYTES {
            return Err(PowerError::Config(format!(
                "serving max_state_bytes must be within 1..={MAX_TRANSFER_BYTES}"
            )));
        }
        if *max_inflight_transfers == 0 || *max_inflight_transfers > MAX_INFLIGHT_TRANSFERS {
            return Err(PowerError::Config(format!(
                "serving max_inflight_transfers must be within 1..={MAX_INFLIGHT_TRANSFERS}"
            )));
        }
        if *transfer_timeout_ms == 0 || *transfer_timeout_ms > MAX_TRANSFER_TIMEOUT_MS {
            return Err(PowerError::Config(format!(
                "serving transfer_timeout_ms must be within 1..={MAX_TRANSFER_TIMEOUT_MS}"
            )));
        }
        if *cancellation_timeout_ms == 0 || cancellation_timeout_ms > transfer_timeout_ms {
            return Err(PowerError::Config(
                "serving cancellation_timeout_ms must be greater than zero and no greater than transfer_timeout_ms"
                    .to_string(),
            ));
        }
        Ok(())
    }

    pub fn is_aggregated(&self) -> bool {
        matches!(self, Self::Aggregated {})
    }

    /// Explicit product-surface composition transport, when opted in by ACL.
    pub fn composition_transport(&self) -> Option<ServingCompositionTransport> {
        match self {
            Self::Aggregated {} => None,
            Self::PrefillDecode { execution } => execution.transport,
        }
    }

    /// Explicit product-surface composition phase executor, when opted in by ACL.
    pub fn composition_phase_executor(&self) -> Option<ServingCompositionPhaseExecutor> {
        match self {
            Self::Aggregated {} => None,
            Self::PrefillDecode { execution } => execution.phase_executor,
        }
    }

    /// Explicit product-surface composition state ownership, when opted in by ACL.
    pub fn composition_state_ownership(&self) -> Option<ServingCompositionStateOwnership> {
        match self {
            Self::Aggregated {} => None,
            Self::PrefillDecode { execution } => execution.state_ownership,
        }
    }

    /// Explicit product-surface composition phase execution, when opted in by ACL.
    pub fn composition_phase_execution(&self) -> Option<ServingCompositionPhaseExecution> {
        match self {
            Self::Aggregated {} => None,
            Self::PrefillDecode { execution } => execution.phase_execution,
        }
    }

    /// Whether worker observation may list this profile's P/D phase as ready.
    ///
    /// Builder-injected adapters (no composition transport) remain gated only
    /// by provision, contract, and health. Product DirectDeviceMemoryPull and
    /// `phase_executor = backend-owned` never advertise until a real adapter
    /// exists. Eligible-only ownership (`state_ownership = profile-bound` or
    /// `llamacpp`) and pending Ready-unlock (`phase_execution = pending`) never
    /// advertise: backend-owned keeps `may_advertise_prefill_decode` false so
    /// worker `ready_phases` stay empty until a real model-semantic execute
    /// adapter exists.
    pub fn may_advertise_prefill_decode(&self) -> bool {
        if let Some(phase_executor) = self.composition_phase_executor() {
            if !phase_executor.may_advertise_prefill_decode() {
                return false;
            }
        }
        match self.composition_transport() {
            None => true,
            Some(transport) => transport.may_advertise_prefill_decode(),
        }
    }

    pub fn phase(&self) -> ServingPhase {
        match self {
            Self::Aggregated {} => ServingPhase::Aggregated,
            Self::PrefillDecode { execution } => execution.role.into(),
        }
    }

    /// Stable identity used to bind injected adapters to this exact profile.
    pub fn sha256(&self) -> Result<String> {
        self.validate()?;
        let document = serde_json::to_vec(self)?;
        let mut digest = Sha256::new();
        digest.update(b"a3s.power.serving-execution-profile.v1\0");
        digest.update(document);
        Ok(hex::encode(digest.finalize()))
    }

    /// Deployment generation and peer set pinned by this process profile.
    pub fn deployment_identity(&self) -> Result<ServingDeploymentIdentity> {
        self.validate()?;
        let Self::PrefillDecode { execution } = self else {
            return Err(PowerError::Config(
                "aggregated serving does not expose a deployment identity".to_string(),
            ));
        };
        let identity = ServingDeploymentIdentity {
            generation: execution.generation,
            peer_set_sha256: execution.peer_set_sha256.clone(),
        };
        identity.validate()?;
        Ok(identity)
    }

    /// Reject peer descriptors from a different Cloud deployment generation or
    /// peer set. Roles may differ; generation and peer set must not.
    pub fn validate_deployment_identity(&self, identity: &ServingDeploymentIdentity) -> Result<()> {
        identity.validate()?;
        let expected = self.deployment_identity()?;
        if identity != &expected {
            return Err(PowerError::InvalidRequest(
                "state-transfer deployment identity does not match this process generation"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Validate one request-specific state identity against the static profile.
    pub fn validate_state_binding(&self, binding: &StateTransferBinding) -> Result<()> {
        self.validate()?;
        binding.validate()?;
        let Self::PrefillDecode { execution } = self else {
            return Err(PowerError::Config(
                "aggregated serving does not accept distributed state bindings".to_string(),
            ));
        };
        let PrefillDecodeExecutionProfile {
            model_sha256,
            execution_sha256,
            layout_sha256,
            state_kind,
            max_state_bytes,
            ..
        } = execution.as_ref();
        if binding.model_sha256 != *model_sha256
            || binding.execution_sha256 != *execution_sha256
            || binding.layout_sha256 != *layout_sha256
            || binding.state_kind != *state_kind
            || binding.state_bytes > *max_state_bytes
        {
            return Err(PowerError::InvalidRequest(
                "state-transfer binding does not match the immutable serving profile".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate a process-local transfer adapter before server startup.
    pub fn validate_state_transfer_capabilities(
        &self,
        capabilities: &StateTransferCapabilities,
    ) -> Result<()> {
        self.validate()?;
        capabilities.validate()?;
        let Self::PrefillDecode { execution } = self else {
            return Err(PowerError::Config(
                "aggregated serving cannot install a state-transfer adapter".to_string(),
            ));
        };
        let PrefillDecodeExecutionProfile {
            role,
            protocol,
            max_state_bytes,
            max_inflight_transfers,
            ..
        } = execution.as_ref();
        if capabilities.execution_profile_sha256 != self.sha256()?
            || !capabilities.supports_phase((*role).into())
            || !capabilities.supports_protocol(*protocol)
            || capabilities.max_transfer_bytes < *max_state_bytes
            || capabilities.max_inflight_transfers < *max_inflight_transfers
        {
            return Err(PowerError::Config(
                "state-transfer adapter does not satisfy the immutable serving profile".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate a backend-owned phase executor before server startup.
    pub fn validate_phase_executor_capabilities(
        &self,
        capabilities: &PhaseExecutorCapabilities,
    ) -> Result<()> {
        self.validate()?;
        capabilities.validate()?;
        let Self::PrefillDecode { execution } = self else {
            return Err(PowerError::Config(
                "aggregated serving cannot install a distributed phase executor".to_string(),
            ));
        };
        if capabilities.execution_profile_sha256 != self.sha256()?
            || capabilities.phase != ServingPhase::from(execution.role)
            || capabilities.weight_cache != execution.weight_cache
            || capabilities.residency_policy_sha256 != execution.residency_policy_sha256
            || capabilities.session_pool != execution.session_pool
            || capabilities.session_pool_policy_sha256 != execution.session_pool_policy_sha256
        {
            return Err(PowerError::Config(
                "phase executor does not satisfy the immutable serving profile".to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_identifier(value: &str, maximum_bytes: usize, label: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > maximum_bytes
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(PowerError::Config(format!(
            "{label} must be non-empty, trimmed, control-free, and at most {maximum_bytes} bytes"
        )));
    }
    Ok(())
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
