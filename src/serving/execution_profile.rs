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
}

/// Immutable execution profile for one Power process generation.
///
/// The default aggregated variant preserves ordinary local inference. The
/// disaggregated variant binds every static fact that must agree before a
/// phase executor or state-transfer adapter can be used. Request-specific
/// token counts, state sizes, worker epochs, and deadlines remain bound by the
/// state-transfer command and descriptor types.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "profile", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ServingExecutionProfile {
    #[default]
    Aggregated,
    PrefillDecode {
        #[serde(flatten)]
        execution: Box<PrefillDecodeExecutionProfile>,
    },
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
        if matches!(privacy, ServingPrivacyMode::AttestedPrivateFabric)
            && attestation_policy_sha256.is_none()
        {
            return Err(PowerError::Config(
                "attested-private-fabric serving requires attestation_policy_sha256".to_string(),
            ));
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
        matches!(self, Self::Aggregated)
    }

    pub fn phase(&self) -> ServingPhase {
        match self {
            Self::Aggregated => ServingPhase::Aggregated,
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
