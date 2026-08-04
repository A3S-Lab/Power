use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::sealed_state::decode_sha256;
use super::super::{ExecutionDigest, ExecutionPermit, InferenceLimits};

/// Digest-only model and scheduler identity for one execution-batch lifecycle.
///
/// Power does not interpret the state layout or scheduler implementation. The
/// digests prevent members from different models or lifecycle semantics from
/// sharing one transcript accidentally.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExecutionBatchBinding {
    pub weights_sha256: String,
    pub state_layout_sha256: String,
    pub scheduler_sha256: String,
}

impl ExecutionBatchBinding {
    pub fn new(
        weights_sha256: impl Into<String>,
        state_layout_sha256: impl Into<String>,
        scheduler_sha256: impl Into<String>,
    ) -> Result<Self> {
        let binding = Self {
            weights_sha256: weights_sha256.into(),
            state_layout_sha256: state_layout_sha256.into(),
            scheduler_sha256: scheduler_sha256.into(),
        };
        binding.validate()?;
        Ok(binding)
    }

    pub fn validate(&self) -> Result<()> {
        decode_sha256(&self.weights_sha256, "execution batch weights")?;
        decode_sha256(&self.state_layout_sha256, "execution batch state layout")?;
        decode_sha256(&self.scheduler_sha256, "execution batch scheduler")?;
        Ok(())
    }
}

impl std::fmt::Debug for ExecutionBatchBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionBatchBinding")
            .field("weights", &"sha256")
            .field("state_layout", &"sha256")
            .field("scheduler", &"sha256")
            .finish()
    }
}

/// Opaque request and model-owned state identities for one admitted member.
///
/// These values are hashes, not anonymization. Callers should use independent,
/// high-entropy identifiers whenever the digest may leave a TEE boundary.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExecutionBatchMemberBinding {
    member_id_sha256: String,
    state_id_sha256: String,
}

impl ExecutionBatchMemberBinding {
    pub fn new(
        member_id_sha256: impl Into<String>,
        state_id_sha256: impl Into<String>,
    ) -> Result<Self> {
        let binding = Self {
            member_id_sha256: member_id_sha256.into(),
            state_id_sha256: state_id_sha256.into(),
        };
        binding.validate()?;
        Ok(binding)
    }

    pub fn for_identifiers(
        member_identifier: &[u8],
        state_identifier: &[u8],
        limits: &InferenceLimits,
    ) -> Result<Self> {
        validate_identifier(member_identifier, limits, "execution batch member")?;
        validate_identifier(state_identifier, limits, "execution batch state")?;
        Self::new(
            identifier_sha256(b"a3s-power-execution-batch-member-v1\0", member_identifier),
            identifier_sha256(b"a3s-power-execution-batch-state-v1\0", state_identifier),
        )
    }

    pub fn member_id_sha256(&self) -> &str {
        &self.member_id_sha256
    }

    pub fn state_id_sha256(&self) -> &str {
        &self.state_id_sha256
    }

    pub(super) fn validate(&self) -> Result<()> {
        decode_sha256(&self.member_id_sha256, "execution batch member")?;
        decode_sha256(&self.state_id_sha256, "execution batch state")?;
        Ok(())
    }
}

impl std::fmt::Debug for ExecutionBatchMemberBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionBatchMemberBinding")
            .field("member", &"redacted-sha256")
            .field("state", &"redacted-sha256")
            .finish()
    }
}

/// Initial bounded lifecycle state supplied by the model-owned scheduler.
pub struct ExecutionBatchMemberSpec {
    pub(super) binding: ExecutionBatchMemberBinding,
    pub(super) position: usize,
    pub(super) generated_items: usize,
    pub(super) max_generated_items: usize,
    pub(super) state_bytes: u64,
}

impl ExecutionBatchMemberSpec {
    pub fn new(
        binding: ExecutionBatchMemberBinding,
        position: usize,
        generated_items: usize,
        max_generated_items: usize,
        state_bytes: u64,
    ) -> Self {
        Self {
            binding,
            position,
            generated_items,
            max_generated_items,
            state_bytes,
        }
    }
}

impl std::fmt::Debug for ExecutionBatchMemberSpec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionBatchMemberSpec")
            .field("binding", &self.binding)
            .finish_non_exhaustive()
    }
}

/// Read-only, digest-bound state visible at a scheduler boundary.
#[derive(Clone, PartialEq, Eq)]
pub struct ExecutionBatchMemberSnapshot {
    member_id_sha256: String,
    position: usize,
    generated_items: usize,
    max_generated_items: usize,
    state_bytes: u64,
}

impl ExecutionBatchMemberSnapshot {
    pub(super) fn new(
        member_id_sha256: String,
        position: usize,
        generated_items: usize,
        max_generated_items: usize,
        state_bytes: u64,
    ) -> Self {
        Self {
            member_id_sha256,
            position,
            generated_items,
            max_generated_items,
            state_bytes,
        }
    }

    pub fn member_id_sha256(&self) -> &str {
        &self.member_id_sha256
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn generated_items(&self) -> usize {
        self.generated_items
    }

    pub fn max_generated_items(&self) -> usize {
        self.max_generated_items
    }

    pub fn state_bytes(&self) -> u64 {
        self.state_bytes
    }
}

impl std::fmt::Debug for ExecutionBatchMemberSnapshot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionBatchMemberSnapshot")
            .field("member", &"redacted-sha256")
            .finish_non_exhaustive()
    }
}

/// One model-owned ragged row proposed for the next fair lifecycle step.
pub struct ExecutionBatchRowSpec {
    pub(super) member_id_sha256: String,
    pub(super) position: usize,
    pub(super) shape: Vec<usize>,
    pub(super) input: ExecutionDigest,
}

impl ExecutionBatchRowSpec {
    pub fn new(
        member_id_sha256: impl Into<String>,
        position: usize,
        shape: Vec<usize>,
        input: ExecutionDigest,
    ) -> Self {
        Self {
            member_id_sha256: member_id_sha256.into(),
            position,
            shape,
            input,
        }
    }
}

impl std::fmt::Debug for ExecutionBatchRowSpec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionBatchRowSpec")
            .field("member", &"redacted-sha256")
            .finish_non_exhaustive()
    }
}

/// Canonically ordered row guard returned for model-owned arithmetic.
pub struct ExecutionBatchRow {
    pub(super) canonical_index: usize,
    pub(super) member_id_sha256: String,
    pub(super) state_id_sha256: String,
    pub(super) position: usize,
    pub(super) shape: Vec<usize>,
    pub(super) input: ExecutionDigest,
    pub(super) permit: ExecutionPermit,
    pub(super) cancellation: CancellationToken,
}

impl ExecutionBatchRow {
    pub fn canonical_index(&self) -> usize {
        self.canonical_index
    }

    pub fn member_id_sha256(&self) -> &str {
        &self.member_id_sha256
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn input(&self) -> &ExecutionDigest {
        &self.input
    }

    pub fn permit(&self) -> &ExecutionPermit {
        &self.permit
    }

    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }
}

impl std::fmt::Debug for ExecutionBatchRow {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionBatchRow")
            .field("canonical_index", &self.canonical_index)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBatchRowDisposition {
    Continue,
    Complete,
}

/// Model-owned row result committed atomically at the lifecycle boundary.
pub struct ExecutionBatchRowOutcome {
    pub(super) member_id_sha256: String,
    pub(super) next_position: usize,
    pub(super) next_generated_items: usize,
    pub(super) next_state_bytes: u64,
    pub(super) disposition: ExecutionBatchRowDisposition,
    pub output: ExecutionDigest,
}

impl ExecutionBatchRowOutcome {
    pub fn continuing(
        member_id_sha256: impl Into<String>,
        next_position: usize,
        next_generated_items: usize,
        next_state_bytes: u64,
        output: ExecutionDigest,
    ) -> Self {
        Self::new(
            member_id_sha256,
            next_position,
            next_generated_items,
            next_state_bytes,
            ExecutionBatchRowDisposition::Continue,
            output,
        )
    }

    pub fn completed(
        member_id_sha256: impl Into<String>,
        next_position: usize,
        next_generated_items: usize,
        next_state_bytes: u64,
        output: ExecutionDigest,
    ) -> Self {
        Self::new(
            member_id_sha256,
            next_position,
            next_generated_items,
            next_state_bytes,
            ExecutionBatchRowDisposition::Complete,
            output,
        )
    }

    fn new(
        member_id_sha256: impl Into<String>,
        next_position: usize,
        next_generated_items: usize,
        next_state_bytes: u64,
        disposition: ExecutionBatchRowDisposition,
        output: ExecutionDigest,
    ) -> Self {
        Self {
            member_id_sha256: member_id_sha256.into(),
            next_position,
            next_generated_items,
            next_state_bytes,
            disposition,
            output,
        }
    }
}

impl std::fmt::Debug for ExecutionBatchRowOutcome {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionBatchRowOutcome")
            .field("member", &"redacted-sha256")
            .field("disposition", &self.disposition)
            .finish_non_exhaustive()
    }
}

/// Privacy-safe evidence for one committed fair step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExecutionBatchStepEvidence {
    pub schema: String,
    pub declaration_sha256: String,
    pub step: u64,
    pub row_count: usize,
    pub continued_members: usize,
    pub completed_members: usize,
    pub cancelled_members: usize,
    pub active_members_after: usize,
    pub state_bytes_after: u64,
    pub input_sha256: String,
    pub output_sha256: String,
}

impl ExecutionBatchStepEvidence {
    pub const SCHEMA: &'static str = "a3s.power.execution-batch-step.v1";
}

/// Aggregate, digest-only evidence for one completed lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExecutionBatchLifecycleEvidence {
    pub schema: String,
    pub declaration_sha256: String,
    pub transcript_sha256: String,
    pub admitted_members: u64,
    pub completed_members: u64,
    pub cancelled_members: u64,
    pub committed_steps: u64,
    pub processed_rows: u64,
    pub max_active_members: usize,
    pub peak_state_bytes: u64,
}

impl ExecutionBatchLifecycleEvidence {
    pub const SCHEMA: &'static str = "a3s.power.execution-batch-lifecycle.v1";
}

pub(super) fn validate_execution_digest(
    digest: &ExecutionDigest,
    limits: &InferenceLimits,
    label: &str,
) -> Result<()> {
    decode_sha256(&digest.sha256, label)?;
    if digest.byte_length == 0 || digest.item_count == 0 {
        return Err(PowerError::InvalidRequest(format!(
            "{label} must describe at least one byte and item"
        )));
    }
    if digest.byte_length > limits.max_input_bytes {
        return Err(PowerError::InvalidRequest(format!(
            "{label} contains {} bytes, exceeding the {} byte input limit",
            digest.byte_length, limits.max_input_bytes
        )));
    }
    if digest.item_count > limits.max_tensor_elements {
        return Err(PowerError::InvalidRequest(format!(
            "{label} contains {} items, exceeding the {} item limit",
            digest.item_count, limits.max_tensor_elements
        )));
    }
    Ok(())
}

fn validate_identifier(value: &[u8], limits: &InferenceLimits, label: &str) -> Result<()> {
    if value.is_empty() || value.len() > limits.max_graph_name_bytes {
        return Err(PowerError::InvalidRequest(format!(
            "{label} identifier must contain between 1 and {} bytes",
            limits.max_graph_name_bytes
        )));
    }
    Ok(())
}

fn identifier_sha256(domain: &[u8], value: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
    format!("{:x}", hasher.finalize())
}
