use std::fmt;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{PowerError, Result};

use super::ServingPhase;

pub const STATE_TRANSFER_TARGET_SCHEMA: &str = "a3s.power.state-transfer-target.v1";
pub const STATE_TRANSFER_SOURCE_SCHEMA: &str = "a3s.power.state-transfer-source.v1";
pub const STATE_TRANSFER_RECEIPT_SCHEMA: &str = "a3s.power.state-transfer-receipt.v1";

const SHA256_HEX_BYTES: usize = 64;
const MAX_OPAQUE_TICKET_BYTES: usize = 16 * 1024;
pub(super) const MAX_TRANSFER_BYTES: u64 = 16 * 1024 * 1024 * 1024 * 1024;
pub(super) const MAX_INFLIGHT_TRANSFERS: u32 = 65_536;
const MAX_TRANSFER_LIFETIME_SECONDS: i64 = 300;

/// Model-owned mutable state kind. Power does not inspect its layout or bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StateKind {
    KvCache,
    Recurrent,
}

/// Transport semantics guaranteed by a typed state-transfer adapter.
///
/// The enum describes the data path without making Power depend on a concrete
/// library such as NIXL. Adapter-specific connection metadata stays inside the
/// bounded opaque ticket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StateTransferProtocol {
    DirectDeviceMemoryPullV1,
    BufferedHostMemoryPullV1,
}

/// Exact compatibility identity for one transferred model state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateTransferBinding {
    pub model_sha256: String,
    pub execution_sha256: String,
    pub layout_sha256: String,
    pub state_kind: StateKind,
    pub token_count: u64,
    pub state_bytes: u64,
}

impl StateTransferBinding {
    pub fn validate(&self) -> Result<()> {
        validate_sha256(&self.model_sha256, "state-transfer model")?;
        validate_sha256(&self.execution_sha256, "state-transfer execution")?;
        validate_sha256(&self.layout_sha256, "state-transfer layout")?;
        if self.token_count == 0 {
            return Err(PowerError::InvalidRequest(
                "state-transfer token count must be greater than zero".to_string(),
            ));
        }
        if self.state_bytes == 0 || self.state_bytes > MAX_TRANSFER_BYTES {
            return Err(PowerError::InvalidRequest(format!(
                "state-transfer size must be within 1..={MAX_TRANSFER_BYTES} bytes"
            )));
        }
        Ok(())
    }
}

/// Static, process-level limits and phase support of one injected adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateTransferCapabilities {
    /// Digest of the exact immutable serving profile used to compose this adapter.
    pub execution_profile_sha256: String,
    pub phases: Vec<ServingPhase>,
    pub protocols: Vec<StateTransferProtocol>,
    pub max_transfer_bytes: u64,
    pub max_inflight_transfers: u32,
}

impl StateTransferCapabilities {
    pub fn validate(&self) -> Result<()> {
        validate_sha256(
            &self.execution_profile_sha256,
            "state-transfer execution profile",
        )?;
        if self.phases.is_empty()
            || self
                .phases
                .iter()
                .any(|phase| matches!(phase, ServingPhase::Aggregated))
            || !strictly_sorted(&self.phases)
        {
            return Err(PowerError::Config(
                "state-transfer phases must be a sorted, unique subset of prefill and decode"
                    .to_string(),
            ));
        }
        if self.protocols.is_empty() || !strictly_sorted(&self.protocols) {
            return Err(PowerError::Config(
                "state-transfer protocols must be sorted and unique".to_string(),
            ));
        }
        if self.max_transfer_bytes == 0 || self.max_transfer_bytes > MAX_TRANSFER_BYTES {
            return Err(PowerError::Config(format!(
                "state-transfer byte limit must be within 1..={MAX_TRANSFER_BYTES}"
            )));
        }
        if self.max_inflight_transfers == 0 || self.max_inflight_transfers > MAX_INFLIGHT_TRANSFERS
        {
            return Err(PowerError::Config(format!(
                "state-transfer concurrency must be within 1..={MAX_INFLIGHT_TRANSFERS}"
            )));
        }
        Ok(())
    }

    pub fn supports_phase(&self, phase: ServingPhase) -> bool {
        self.phases.binary_search(&phase).is_ok()
    }

    pub fn supports_protocol(&self, protocol: StateTransferProtocol) -> bool {
        self.protocols.binary_search(&protocol).is_ok()
    }
}

/// Decode-side destination preparation returned to the request orchestrator.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateTransferTarget {
    pub schema: String,
    pub transfer_id: Uuid,
    pub destination_worker_epoch: Uuid,
    pub binding: StateTransferBinding,
    pub protocol: StateTransferProtocol,
    pub prepared_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub ticket: String,
}

impl fmt::Debug for StateTransferTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StateTransferTarget")
            .field("schema", &self.schema)
            .field("transfer_id", &self.transfer_id)
            .field("destination_worker_epoch", &self.destination_worker_epoch)
            .field("binding", &self.binding)
            .field("protocol", &self.protocol)
            .field("prepared_at", &self.prepared_at)
            .field("expires_at", &self.expires_at)
            .field("ticket", &"[REDACTED]")
            .finish()
    }
}

impl StateTransferTarget {
    pub fn validate_at(
        &self,
        now: DateTime<Utc>,
        capabilities: &StateTransferCapabilities,
    ) -> Result<()> {
        validate_descriptor(
            DescriptorValidation {
                schema: &self.schema,
                expected_schema: STATE_TRANSFER_TARGET_SCHEMA,
                transfer_id: self.transfer_id,
                worker_epoch: self.destination_worker_epoch,
                binding: &self.binding,
                protocol: self.protocol,
                issued_at: self.prepared_at,
                expires_at: self.expires_at,
                ticket: &self.ticket,
            },
            now,
            capabilities,
        )
    }
}

/// Prefill-side source publication consumed by the selected decode worker.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateTransferSource {
    pub schema: String,
    pub transfer_id: Uuid,
    pub source_worker_epoch: Uuid,
    pub destination_worker_epoch: Uuid,
    pub binding: StateTransferBinding,
    pub protocol: StateTransferProtocol,
    pub published_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub ticket: String,
}

impl fmt::Debug for StateTransferSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StateTransferSource")
            .field("schema", &self.schema)
            .field("transfer_id", &self.transfer_id)
            .field("source_worker_epoch", &self.source_worker_epoch)
            .field("destination_worker_epoch", &self.destination_worker_epoch)
            .field("binding", &self.binding)
            .field("protocol", &self.protocol)
            .field("published_at", &self.published_at)
            .field("expires_at", &self.expires_at)
            .field("ticket", &"[REDACTED]")
            .finish()
    }
}

impl StateTransferSource {
    pub fn validate_at(
        &self,
        now: DateTime<Utc>,
        capabilities: &StateTransferCapabilities,
    ) -> Result<()> {
        validate_descriptor(
            DescriptorValidation {
                schema: &self.schema,
                expected_schema: STATE_TRANSFER_SOURCE_SCHEMA,
                transfer_id: self.transfer_id,
                worker_epoch: self.source_worker_epoch,
                binding: &self.binding,
                protocol: self.protocol,
                issued_at: self.published_at,
                expires_at: self.expires_at,
                ticket: &self.ticket,
            },
            now,
            capabilities,
        )
    }

    pub fn validate_for(
        &self,
        target: &StateTransferTarget,
        now: DateTime<Utc>,
        capabilities: &StateTransferCapabilities,
    ) -> Result<()> {
        target.validate_at(now, capabilities)?;
        self.validate_at(now, capabilities)?;
        if self.transfer_id != target.transfer_id
            || self.destination_worker_epoch != target.destination_worker_epoch
            || self.binding != target.binding
            || self.protocol != target.protocol
            || self.published_at < target.prepared_at
            || self.expires_at > target.expires_at
        {
            return Err(PowerError::InvalidRequest(
                "state-transfer source does not match the prepared destination".to_string(),
            ));
        }
        Ok(())
    }
}

/// How an adapter verified completion without exposing model-state bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum StateTransferIntegrity {
    TransportVerified,
    Sha256 { digest: String },
}

impl StateTransferIntegrity {
    fn validate(&self) -> Result<()> {
        match self {
            Self::TransportVerified => Ok(()),
            Self::Sha256 { digest } => validate_sha256(digest, "transferred state"),
        }
    }
}

/// Terminal evidence for one bounded transfer attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateTransferReceipt {
    pub schema: String,
    pub transfer_id: Uuid,
    pub source_worker_epoch: Uuid,
    pub destination_worker_epoch: Uuid,
    pub binding: StateTransferBinding,
    pub protocol: StateTransferProtocol,
    pub bytes_transferred: u64,
    pub integrity: StateTransferIntegrity,
    pub completed_at: DateTime<Utc>,
}

impl StateTransferReceipt {
    pub fn validate_for(
        &self,
        source: &StateTransferSource,
        capabilities: &StateTransferCapabilities,
    ) -> Result<()> {
        source.validate_at(self.completed_at, capabilities)?;
        if self.schema != STATE_TRANSFER_RECEIPT_SCHEMA
            || self.transfer_id.is_nil()
            || self.source_worker_epoch.is_nil()
            || self.destination_worker_epoch.is_nil()
        {
            return Err(PowerError::InvalidRequest(
                "state-transfer receipt identity or schema is invalid".to_string(),
            ));
        }
        self.binding.validate()?;
        self.integrity.validate()?;
        if self.transfer_id != source.transfer_id
            || self.source_worker_epoch != source.source_worker_epoch
            || self.destination_worker_epoch != source.destination_worker_epoch
            || self.binding != source.binding
            || self.protocol != source.protocol
            || self.bytes_transferred != source.binding.state_bytes
        {
            return Err(PowerError::InvalidRequest(
                "state-transfer receipt does not prove the published source".to_string(),
            ));
        }
        Ok(())
    }
}

struct DescriptorValidation<'a> {
    schema: &'a str,
    expected_schema: &'a str,
    transfer_id: Uuid,
    worker_epoch: Uuid,
    binding: &'a StateTransferBinding,
    protocol: StateTransferProtocol,
    issued_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    ticket: &'a str,
}

fn validate_descriptor(
    descriptor: DescriptorValidation<'_>,
    now: DateTime<Utc>,
    capabilities: &StateTransferCapabilities,
) -> Result<()> {
    let DescriptorValidation {
        schema,
        expected_schema,
        transfer_id,
        worker_epoch,
        binding,
        protocol,
        issued_at,
        expires_at,
        ticket,
    } = descriptor;
    capabilities.validate()?;
    if schema != expected_schema || transfer_id.is_nil() || worker_epoch.is_nil() {
        return Err(PowerError::InvalidRequest(
            "state-transfer descriptor identity or schema is invalid".to_string(),
        ));
    }
    binding.validate()?;
    if binding.state_bytes > capabilities.max_transfer_bytes {
        return Err(PowerError::InvalidRequest(
            "state-transfer descriptor exceeds the adapter byte limit".to_string(),
        ));
    }
    if !capabilities.supports_protocol(protocol) {
        return Err(PowerError::InvalidRequest(
            "state-transfer protocol is not supported by this adapter".to_string(),
        ));
    }
    validate_opaque(ticket, MAX_OPAQUE_TICKET_BYTES, "state-transfer ticket")?;
    let maximum_expiry = issued_at
        .checked_add_signed(Duration::seconds(MAX_TRANSFER_LIFETIME_SECONDS))
        .ok_or_else(|| {
            PowerError::InvalidRequest(
                "state-transfer expiry is outside the representable range".to_string(),
            )
        })?;
    if issued_at > now || expires_at <= now || expires_at > maximum_expiry {
        return Err(PowerError::InvalidRequest(format!(
            "state-transfer expiry must be within {MAX_TRANSFER_LIFETIME_SECONDS} seconds"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != SHA256_HEX_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PowerError::InvalidRequest(format!(
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

fn strictly_sorted<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}
