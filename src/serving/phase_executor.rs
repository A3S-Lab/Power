use std::fmt;

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::backend::types::{ChatRequest, CompletionRequest};
use crate::error::{PowerError, Result};

use super::{ServingExecutionProfile, ServingPhase};

mod abort;
mod lifecycle;

pub use abort::AbortPhaseExecution;
pub use lifecycle::{
    ExecutePhaseExecution, ImportedModelState, PhaseExecutionOutput, PhaseResponseChunk,
    PhaseResponseStream, PreparedDecodePhase, PreparedPhaseExecution, PreparedPrefillPhase,
    ProducedModelState,
};

const MAX_LOCAL_HANDLE_BYTES: usize = 512;
const MAX_RETRY_AFTER_MS: u64 = 300_000;
const MAX_MODEL_NAME_BYTES: usize = 256;

/// Static identity of one injected backend-owned phase executor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseExecutorCapabilities {
    pub execution_profile_sha256: String,
    pub phase: ServingPhase,
}

impl PhaseExecutorCapabilities {
    pub fn validate(&self) -> Result<()> {
        validate_sha256(
            &self.execution_profile_sha256,
            "phase-executor execution profile",
        )?;
        if matches!(self.phase, ServingPhase::Aggregated) {
            return Err(PowerError::Config(
                "a distributed phase executor must own exactly prefill or decode".to_string(),
            ));
        }
        Ok(())
    }
}

/// Current ability of an installed phase executor to accept new work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PhaseExecutorHealth {
    Ready,
    Degraded,
    Unavailable,
}

impl PhaseExecutorHealth {
    pub fn accepts_work(self) -> bool {
        matches!(self, Self::Ready | Self::Degraded)
    }
}

/// Process-local backend execution handle.
///
/// It is deliberately not serializable and its debug representation never
/// reveals the adapter-owned value.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PhaseExecutionHandle(String);

impl PhaseExecutionHandle {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_opaque(&value, MAX_LOCAL_HANDLE_BYTES, "phase-execution handle")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for PhaseExecutionHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PhaseExecutionHandle([REDACTED])")
    }
}

/// Existing Power backend request passed only inside the local process.
pub enum PhaseRequest {
    Chat(ChatRequest),
    Completion(CompletionRequest),
}

impl fmt::Debug for PhaseRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chat(_) => formatter.write_str("PhaseRequest::Chat([REDACTED])"),
            Self::Completion(_) => formatter.write_str("PhaseRequest::Completion([REDACTED])"),
        }
    }
}

/// Request to validate, tokenize, and reserve one backend-owned phase without
/// generating response bytes.
pub struct PreparePhaseExecution {
    pub execution_id: Uuid,
    pub local_worker_epoch: Uuid,
    pub model: String,
    pub request: PhaseRequest,
    pub expires_at: DateTime<Utc>,
}

impl PreparePhaseExecution {
    pub fn validate_at(
        &self,
        now: DateTime<Utc>,
        capabilities: &PhaseExecutorCapabilities,
        profile: &ServingExecutionProfile,
    ) -> Result<()> {
        profile.validate_phase_executor_capabilities(capabilities)?;
        validate_command_identity(self.execution_id, self.local_worker_epoch)?;
        validate_opaque(&self.model, MAX_MODEL_NAME_BYTES, "phase-execution model")?;
        let configured_model = configured_model(profile)?;
        if self.model != configured_model {
            return Err(PowerError::InvalidRequest(
                "phase-execution model does not match the immutable serving profile".to_string(),
            ));
        }
        validate_expiry(
            now,
            self.expires_at,
            configured_transfer_timeout_ms(profile)?,
        )
    }
}

impl fmt::Debug for PreparePhaseExecution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparePhaseExecution")
            .field("execution_id", &self.execution_id)
            .field("local_worker_epoch", &self.local_worker_epoch)
            .field("model", &self.model)
            .field("request", &self.request)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// Why a decode attempt must repeat prefill instead of trusting transferred
/// state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecomputeReason {
    StateMissing,
    StateStale,
    StateCorrupt,
    StateIncompatible,
}

/// Transient reason for selecting another compatible worker or retrying later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RetryableUnavailableReason {
    AdmissionPressure,
    ExecutorUnavailable,
    TransferUnavailable,
    PeerUnavailable,
    ResourcePressure,
}

/// Closed terminal classification produced before response generation begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TerminalFailureReason {
    InvalidRequest,
    UnsupportedRequest,
    ModelMismatch,
    PolicyViolation,
    ExecutionFailed,
}

/// Pre-response decision from a backend-owned phase operation.
pub enum PhaseDecision<T> {
    Ready(T),
    Recompute {
        reason: RecomputeReason,
    },
    RetryableUnavailable {
        reason: RetryableUnavailableReason,
        retry_after_ms: Option<u64>,
    },
    TerminalFailure {
        reason: TerminalFailureReason,
    },
}

impl<T> PhaseDecision<T> {
    pub fn ready(value: T) -> Self {
        Self::Ready(value)
    }

    pub fn recompute(reason: RecomputeReason) -> Self {
        Self::Recompute { reason }
    }

    pub fn retryable_unavailable(
        reason: RetryableUnavailableReason,
        retry_after_ms: Option<u64>,
    ) -> Result<Self> {
        let decision = Self::RetryableUnavailable {
            reason,
            retry_after_ms,
        };
        decision.validate()?;
        Ok(decision)
    }

    pub fn terminal_failure(reason: TerminalFailureReason) -> Self {
        Self::TerminalFailure { reason }
    }

    pub fn validate(&self) -> Result<()> {
        if let Self::RetryableUnavailable {
            retry_after_ms: Some(delay),
            ..
        } = self
        {
            if *delay == 0 || *delay > MAX_RETRY_AFTER_MS {
                return Err(PowerError::InvalidRequest(format!(
                    "phase retry_after_ms must be within 1..={MAX_RETRY_AFTER_MS}"
                )));
            }
        }
        Ok(())
    }
}

/// Backend-owned execution port for exactly one immutable prefill or decode
/// process role.
///
/// Implementations own tokenization, state layout, phase arithmetic, request
/// reservations, response generation, and cleanup. Power validates the
/// lifecycle and moves only opaque state handles through the separate transfer
/// port.
#[async_trait]
pub trait ServingPhaseExecutor: Send + Sync {
    fn capabilities(&self) -> PhaseExecutorCapabilities;
    fn health(&self) -> PhaseExecutorHealth;

    async fn prepare(
        &self,
        command: PreparePhaseExecution,
    ) -> Result<PhaseDecision<PreparedPhaseExecution>>;

    async fn execute(
        &self,
        command: ExecutePhaseExecution,
    ) -> Result<PhaseDecision<PhaseExecutionOutput>>;

    async fn abort(&self, command: AbortPhaseExecution) -> Result<()>;
}

fn configured_model(profile: &ServingExecutionProfile) -> Result<&str> {
    let ServingExecutionProfile::PrefillDecode { execution } = profile else {
        return Err(PowerError::Config(
            "aggregated serving does not accept phase-execution commands".to_string(),
        ));
    };
    Ok(&execution.model)
}

fn configured_transfer_timeout_ms(profile: &ServingExecutionProfile) -> Result<u64> {
    profile.validate()?;
    let ServingExecutionProfile::PrefillDecode { execution } = profile else {
        return Err(PowerError::Config(
            "aggregated serving does not accept phase-execution commands".to_string(),
        ));
    };
    Ok(execution.transfer_timeout_ms)
}

fn configured_cancellation_timeout_ms(profile: &ServingExecutionProfile) -> Result<u64> {
    profile.validate()?;
    let ServingExecutionProfile::PrefillDecode { execution } = profile else {
        return Err(PowerError::Config(
            "aggregated serving does not accept phase-execution commands".to_string(),
        ));
    };
    Ok(execution.cancellation_timeout_ms)
}

fn validate_command_identity(execution_id: Uuid, worker_epoch: Uuid) -> Result<()> {
    if execution_id.is_nil() || worker_epoch.is_nil() {
        return Err(PowerError::InvalidRequest(
            "phase-execution command identity is invalid".to_string(),
        ));
    }
    Ok(())
}

fn validate_expiry(
    now: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    maximum_lifetime_ms: u64,
) -> Result<()> {
    let maximum_lifetime_ms = i64::try_from(maximum_lifetime_ms).map_err(|_| {
        PowerError::Config("phase-execution timeout exceeds the supported range".to_string())
    })?;
    let maximum_expiry = now
        .checked_add_signed(Duration::milliseconds(maximum_lifetime_ms))
        .ok_or_else(|| {
            PowerError::InvalidRequest(
                "phase-execution expiry is outside the representable range".to_string(),
            )
        })?;
    if expires_at <= now || expires_at > maximum_expiry {
        return Err(PowerError::InvalidRequest(format!(
            "phase-execution expiry must be within the configured {maximum_lifetime_ms} milliseconds"
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

fn validate_opaque(value: &str, maximum_bytes: usize, label: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > maximum_bytes
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(PowerError::InvalidRequest(format!(
            "{label} must be non-empty, trimmed, control-free, and at most {maximum_bytes} bytes"
        )));
    }
    Ok(())
}
