//! Digest-only evidence for distributed serving operations.
//!
//! Transfer completion proofs remain [`super::StateTransferReceipt`]. This
//! module hashes that content-free proof into a domain-separated digest so
//! callers can bind P/D lifecycle evidence without inventing a second
//! microbatch receipt-v4 shape or embedding KV/ticket bytes.

use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::{PowerError, Result};

use super::{
    StateKind, StateTransferIntegrity, StateTransferProtocol, StateTransferReceipt,
    STATE_TRANSFER_RECEIPT_SCHEMA,
};

/// Schema for optional digest-only distributed-operation evidence.
pub const DISTRIBUTED_OPERATION_EVIDENCE_SCHEMA: &str = "a3s.power.distributed-operation.v1";

const DISTRIBUTED_OPERATION_DOMAIN: &[u8] = b"a3s.power.distributed-operation.v1\0";

/// Closed set of distributed operations that may emit digest-only evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DistributedOperationKind {
    /// Content-free proof that one transfer consume completed.
    TransferConsume,
}

/// Optional digest-only evidence for one distributed serving operation.
///
/// This is deliberately not an embedded microbatch ExecutionReceipt (v4) and
/// not `a3s.power.microbatch-execution.v1`. It reuses SHA-256 domain separation
/// over the existing transfer receipt identity and never carries prompts,
/// tokens, KV, or tickets.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DistributedOperationEvidence {
    pub schema: String,
    pub operation: DistributedOperationKind,
    pub sha256: String,
}

impl DistributedOperationEvidence {
    /// Bind digest-only evidence from a validated transfer consume receipt.
    pub fn from_transfer_receipt(receipt: &StateTransferReceipt) -> Result<Self> {
        if receipt.schema != STATE_TRANSFER_RECEIPT_SCHEMA {
            return Err(PowerError::InvalidRequest(
                "distributed-operation evidence requires the state-transfer receipt schema"
                    .to_string(),
            ));
        }
        receipt.binding.validate()?;
        match &receipt.integrity {
            StateTransferIntegrity::TransportVerified => {}
            StateTransferIntegrity::Sha256 { digest } => {
                validate_sha256_hex(digest, "transferred state")?;
            }
        }
        if receipt.transfer_id.is_nil()
            || receipt.source_worker_epoch.is_nil()
            || receipt.destination_worker_epoch.is_nil()
        {
            return Err(PowerError::InvalidRequest(
                "distributed-operation evidence requires non-nil transfer and worker epochs"
                    .to_string(),
            ));
        }
        if receipt.bytes_transferred != receipt.binding.state_bytes {
            return Err(PowerError::InvalidRequest(
                "distributed-operation evidence requires bytes_transferred to match binding state_bytes"
                    .to_string(),
            ));
        }

        let digest = hash_transfer_consume(receipt);
        Ok(Self {
            schema: DISTRIBUTED_OPERATION_EVIDENCE_SCHEMA.to_string(),
            operation: DistributedOperationKind::TransferConsume,
            sha256: hex::encode(digest),
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != DISTRIBUTED_OPERATION_EVIDENCE_SCHEMA {
            return Err(PowerError::InvalidRequest(
                "distributed-operation evidence schema is invalid".to_string(),
            ));
        }
        validate_sha256_hex(&self.sha256, "distributed-operation evidence")?;
        Ok(())
    }
}

fn hash_transfer_consume(receipt: &StateTransferReceipt) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(DISTRIBUTED_OPERATION_DOMAIN);
    hasher.update(STATE_TRANSFER_RECEIPT_SCHEMA.as_bytes());
    hasher.update(b"\0transfer-consume\0");
    update_uuid(&mut hasher, receipt.transfer_id);
    update_uuid(&mut hasher, receipt.source_worker_epoch);
    update_uuid(&mut hasher, receipt.destination_worker_epoch);
    hasher.update(receipt.deployment.generation.to_le_bytes());
    hasher.update(receipt.deployment.peer_set_sha256.as_bytes());
    hasher.update(b"\0");
    hasher.update(receipt.binding.model_sha256.as_bytes());
    hasher.update(b"\0");
    hasher.update(receipt.binding.execution_sha256.as_bytes());
    hasher.update(b"\0");
    hasher.update(receipt.binding.layout_sha256.as_bytes());
    hasher.update(b"\0");
    hasher.update(match receipt.binding.state_kind {
        StateKind::KvCache => b"kv-cache".as_slice(),
        StateKind::Recurrent => b"recurrent".as_slice(),
    });
    hasher.update(receipt.binding.token_count.to_le_bytes());
    hasher.update(receipt.binding.state_bytes.to_le_bytes());
    hasher.update(match receipt.protocol {
        StateTransferProtocol::DirectDeviceMemoryPullV1 => {
            b"direct-device-memory-pull-v1".as_slice()
        }
        StateTransferProtocol::BufferedHostMemoryPullV1 => {
            b"buffered-host-memory-pull-v1".as_slice()
        }
    });
    hasher.update(receipt.bytes_transferred.to_le_bytes());
    match &receipt.integrity {
        StateTransferIntegrity::TransportVerified => {
            hasher.update(b"transport-verified");
        }
        StateTransferIntegrity::Sha256 { digest } => {
            hasher.update(b"sha256\0");
            hasher.update(digest.as_bytes());
        }
    }
    // Wall-clock is part of the terminal receipt identity, not model state.
    hasher.update(receipt.completed_at.timestamp().to_le_bytes());
    hasher.update(receipt.completed_at.timestamp_subsec_nanos().to_le_bytes());
    hasher.finalize().into()
}

fn update_uuid(hasher: &mut Sha256, value: Uuid) {
    hasher.update(value.as_bytes());
}

fn validate_sha256_hex(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    {
        return Err(PowerError::InvalidRequest(format!(
            "{label} SHA-256 must contain exactly 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, Utc};
    use uuid::Uuid;

    use super::*;
    use crate::serving::StateTransferBinding;

    /// Stable strings from embedded receipt-v4; kept literal so serving tests
    /// do not require the `embedded-inference` feature to reject overfitting.
    const MICROBATCH_RECEIPT_V4_SCHEMA: &str = "a3s.power.embedded-execution-receipt.v4";
    const MICROBATCH_EVIDENCE_SCHEMA: &str = "a3s.power.microbatch-execution.v1";

    fn receipt() -> StateTransferReceipt {
        StateTransferReceipt {
            schema: STATE_TRANSFER_RECEIPT_SCHEMA.to_string(),
            transfer_id: Uuid::from_u128(1),
            source_worker_epoch: Uuid::from_u128(2),
            destination_worker_epoch: Uuid::from_u128(3),
            deployment: crate::serving::ServingDeploymentIdentity {
                generation: 7,
                peer_set_sha256: "6".repeat(64),
            },
            binding: StateTransferBinding {
                model_sha256: "1".repeat(64),
                execution_sha256: "3".repeat(64),
                layout_sha256: "5".repeat(64),
                state_kind: StateKind::KvCache,
                token_count: 16,
                state_bytes: 512,
            },
            protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
            bytes_transferred: 512,
            integrity: StateTransferIntegrity::TransportVerified,
            completed_at: Utc::now(),
        }
    }

    #[test]
    fn transfer_consume_evidence_is_digest_only_and_domain_separated() {
        let evidence = DistributedOperationEvidence::from_transfer_receipt(&receipt()).unwrap();
        evidence.validate().unwrap();
        assert_eq!(evidence.schema, DISTRIBUTED_OPERATION_EVIDENCE_SCHEMA);
        assert_eq!(
            evidence.operation,
            DistributedOperationKind::TransferConsume
        );
        assert_eq!(evidence.sha256.len(), 64);

        let encoded = serde_json::to_string(&evidence).unwrap();
        for forbidden in [
            "prompt",
            "tenant",
            "ticket",
            "payload",
            MICROBATCH_RECEIPT_V4_SCHEMA,
            MICROBATCH_EVIDENCE_SCHEMA,
        ] {
            assert!(
                !encoded.contains(forbidden),
                "evidence must stay digest-only; found {forbidden}"
            );
        }
    }

    #[test]
    fn evidence_digest_changes_when_binding_identity_changes() {
        let baseline = DistributedOperationEvidence::from_transfer_receipt(&receipt())
            .unwrap()
            .sha256;
        let mut mutated = receipt();
        mutated.binding.layout_sha256 = "9".repeat(64);
        let changed = DistributedOperationEvidence::from_transfer_receipt(&mutated)
            .unwrap()
            .sha256;
        assert_ne!(baseline, changed);
    }

    #[test]
    fn evidence_digest_changes_when_deployment_generation_changes() {
        let baseline = DistributedOperationEvidence::from_transfer_receipt(&receipt())
            .unwrap()
            .sha256;
        let mut mutated = receipt();
        mutated.deployment.generation = 8;
        let changed = DistributedOperationEvidence::from_transfer_receipt(&mutated)
            .unwrap()
            .sha256;
        assert_ne!(baseline, changed);
    }

    #[test]
    fn evidence_digest_is_stable_for_identical_receipts() {
        let mut first = receipt();
        first.completed_at = Utc::now() - Duration::seconds(10);
        let second = first.clone();
        assert_eq!(
            DistributedOperationEvidence::from_transfer_receipt(&first)
                .unwrap()
                .sha256,
            DistributedOperationEvidence::from_transfer_receipt(&second)
                .unwrap()
                .sha256
        );
    }

    #[test]
    fn wrong_receipt_schema_or_byte_mismatch_fails_closed() {
        let mut wrong_schema = receipt();
        wrong_schema.schema = MICROBATCH_RECEIPT_V4_SCHEMA.to_string();
        assert!(DistributedOperationEvidence::from_transfer_receipt(&wrong_schema).is_err());

        let mut short = receipt();
        short.bytes_transferred = 511;
        assert!(DistributedOperationEvidence::from_transfer_receipt(&short).is_err());
    }

    #[test]
    fn distributed_operation_schema_is_not_microbatch_receipt_v4() {
        assert_ne!(
            DISTRIBUTED_OPERATION_EVIDENCE_SCHEMA,
            MICROBATCH_RECEIPT_V4_SCHEMA
        );
        assert_ne!(
            DISTRIBUTED_OPERATION_EVIDENCE_SCHEMA,
            MICROBATCH_EVIDENCE_SCHEMA
        );
        assert_ne!(STATE_TRANSFER_RECEIPT_SCHEMA, MICROBATCH_RECEIPT_V4_SCHEMA);
        assert_ne!(STATE_TRANSFER_RECEIPT_SCHEMA, MICROBATCH_EVIDENCE_SCHEMA);
    }
}
