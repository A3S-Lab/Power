use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::backend::types::{ChatResponseChunk, CompletionResponseChunk};
use crate::serving::{
    RecomputeReason, RetryableUnavailableReason, StateTransferSource, StateTransferTarget,
    TerminalFailureReason,
};

/// Versioned machine-to-machine request-flow protocol exposed by Power.
pub const DISTRIBUTED_SERVING_SCHEMA: &str = "a3s.power.distributed-serving.v1";

/// Versioned newline-delimited decode stream produced by Power.
pub const DISTRIBUTED_SERVING_STREAM_SCHEMA: &str = "a3s.power.distributed-serving-stream.v1";

/// Domain separator for Gateway-scoped prompt-cache identities. Raw
/// user-controlled keys are never accepted by the distributed boundary.
pub const DISTRIBUTED_PROMPT_CACHE_KEY_PREFIX: &str = "a3s-gw-pcache-v1:";

/// OpenAI request profile carried unchanged from Gateway into Power's
/// presentation anti-corruption layer.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "endpoint", rename_all = "kebab-case", deny_unknown_fields)]
pub enum PhaseRequestPayload {
    ChatCompletions { body: serde_json::Value },
    Completions { body: serde_json::Value },
}

impl fmt::Debug for PhaseRequestPayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatCompletions { .. } => {
                formatter.write_str("PhaseRequestPayload::ChatCompletions([REDACTED])")
            }
            Self::Completions { .. } => {
                formatter.write_str("PhaseRequestPayload::Completions([REDACTED])")
            }
        }
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodePrepareRequest {
    pub schema: String,
    pub execution_id: Uuid,
    pub worker_epoch: Uuid,
    pub execution_profile_sha256: String,
    pub expires_at: DateTime<Utc>,
    pub request: PhaseRequestPayload,
}

impl fmt::Debug for DecodePrepareRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodePrepareRequest")
            .field("schema", &self.schema)
            .field("execution_id", &self.execution_id)
            .field("worker_epoch", &self.worker_epoch)
            .field("execution_profile_sha256", &self.execution_profile_sha256)
            .field("expires_at", &self.expires_at)
            .field("request", &self.request)
            .finish()
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillExecuteRequest {
    pub schema: String,
    pub execution_id: Uuid,
    pub worker_epoch: Uuid,
    pub execution_profile_sha256: String,
    pub expires_at: DateTime<Utc>,
    pub request: PhaseRequestPayload,
    pub target: StateTransferTarget,
}

impl fmt::Debug for PrefillExecuteRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrefillExecuteRequest")
            .field("schema", &self.schema)
            .field("execution_id", &self.execution_id)
            .field("worker_epoch", &self.worker_epoch)
            .field("execution_profile_sha256", &self.execution_profile_sha256)
            .field("expires_at", &self.expires_at)
            .field("request", &self.request)
            .field("target", &self.target)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeExecuteRequest {
    pub schema: String,
    pub execution_id: Uuid,
    pub worker_epoch: Uuid,
    pub execution_profile_sha256: String,
    pub source: StateTransferSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbortDistributedExecutionRequest {
    pub schema: String,
    pub execution_id: Uuid,
    pub worker_epoch: Uuid,
    pub execution_profile_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedDecodeResult {
    pub target: StateTransferTarget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishedPrefillResult {
    pub source: StateTransferSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "kebab-case", deny_unknown_fields)]
pub enum DistributedPhaseDecision<T> {
    Ready {
        result: T,
    },
    Recompute {
        reason: RecomputeReason,
    },
    RetryableUnavailable {
        reason: RetryableUnavailableReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_ms: Option<u64>,
    },
    TerminalFailure {
        reason: TerminalFailureReason,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedPhaseResponse<T> {
    pub schema: String,
    pub execution_id: Uuid,
    pub worker_epoch: Uuid,
    pub execution_profile_sha256: String,
    pub outcome: DistributedPhaseDecision<T>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbortDistributedExecutionResponse {
    pub schema: String,
    pub execution_id: Uuid,
    pub worker_epoch: Uuid,
    pub execution_profile_sha256: String,
    pub accepted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DistributedProtocolErrorCode {
    UnsupportedSchema,
    InvalidRequest,
    StaleWorker,
    ProfileMismatch,
    Unavailable,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedProtocolErrorResponse {
    pub schema: String,
    pub code: DistributedProtocolErrorCode,
    pub message: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "endpoint", content = "chunk", rename_all = "kebab-case")]
pub enum DistributedResponseChunk {
    ChatCompletions(ChatResponseChunk),
    Completions(CompletionResponseChunk),
}

impl fmt::Debug for DistributedResponseChunk {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatCompletions(_) => {
                formatter.write_str("DistributedResponseChunk::ChatCompletions([REDACTED])")
            }
            Self::Completions(_) => {
                formatter.write_str("DistributedResponseChunk::Completions([REDACTED])")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "kebab-case", deny_unknown_fields)]
pub enum DistributedDecodeStreamEvent {
    Ready,
    Chunk {
        sequence: u64,
        response: DistributedResponseChunk,
    },
    Failed {
        sequence: u64,
        reason: TerminalFailureReason,
    },
    Completed {
        sequence: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedDecodeStreamFrame {
    pub schema: String,
    pub execution_id: Uuid,
    pub worker_epoch: Uuid,
    pub execution_profile_sha256: String,
    pub payload: DistributedDecodeStreamEvent,
}
