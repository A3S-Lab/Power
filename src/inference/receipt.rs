use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{AcceleratorExecutionEvidence, RuntimeDevice, RUNTIME_NAME};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelIdentity {
    pub family: String,
    pub revision: String,
    pub weights_sha256: String,
}

impl ModelIdentity {
    pub fn new(
        family: impl Into<String>,
        revision: impl Into<String>,
        weights_sha256: impl Into<String>,
    ) -> Self {
        Self {
            family: family.into(),
            revision: revision.into(),
            weights_sha256: weights_sha256.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeIdentity {
    pub name: String,
    pub version: String,
    pub device: String,
}

impl RuntimeIdentity {
    pub(crate) fn current(device: &RuntimeDevice) -> Self {
        Self {
            name: RUNTIME_NAME.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            device: device.name().to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExecutionReceipt {
    pub schema: String,
    pub model: ModelIdentity,
    pub runtime: RuntimeIdentity,
    pub input: ExecutionDigest,
    pub output: ExecutionDigest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accelerator: Option<AcceleratorExecutionEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub microbatch: Option<MicrobatchExecutionEvidence>,
}

impl ExecutionReceipt {
    pub const SCHEMA: &'static str = "a3s.power.embedded-execution-receipt.v1";
    pub const ACCELERATOR_SCHEMA: &'static str = "a3s.power.embedded-execution-receipt.v2";
    pub const ACCELERATOR_MESH_SCHEMA: &'static str = "a3s.power.embedded-execution-receipt.v3";
    pub const MICROBATCH_SCHEMA: &'static str = "a3s.power.embedded-execution-receipt.v4";
}

/// Digest-only scheduling evidence for one admitted microbatch execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MicrobatchExecutionEvidence {
    pub schema: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_declaration_sha256: Option<String>,
    pub plan_sha256: String,
    pub batch_index: usize,
    pub batch_count: usize,
    pub slot_count: usize,
    pub model_admission_queued: bool,
    pub device_admission_queued: bool,
}

impl MicrobatchExecutionEvidence {
    pub const SCHEMA: &'static str = "a3s.power.microbatch-execution.v1";

    pub fn validate(&self) -> crate::error::Result<()> {
        if self.schema != Self::SCHEMA
            || self.batch_count == 0
            || self.batch_index >= self.batch_count
            || self.slot_count == 0
        {
            return Err(crate::error::PowerError::InvalidRequest(
                "microbatch execution evidence shape is invalid".to_string(),
            ));
        }
        super::sealed_state::decode_sha256(&self.plan_sha256, "microbatch execution plan")?;
        if let Some(session) = &self.session_declaration_sha256 {
            super::sealed_state::decode_sha256(session, "microbatch execution session")?;
        }
        Ok(())
    }
}

/// Canonical representation covered by one side of an execution receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionRepresentation {
    F32Tensor,
    ImageRequest,
    TokenIds,
    Utf8Text,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExecutionDigest {
    pub representation: ExecutionRepresentation,
    pub sha256: String,
    pub byte_length: usize,
    pub item_count: usize,
}

impl ExecutionDigest {
    pub fn f32_tensor(shape: &[usize], values: &[f32]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"a3s-power-f32-tensor-v1\0");
        hasher.update((shape.len() as u64).to_le_bytes());
        for dimension in shape {
            hasher.update((*dimension as u64).to_le_bytes());
        }
        for value in values {
            hasher.update(value.to_bits().to_le_bytes());
        }
        Self {
            representation: ExecutionRepresentation::F32Tensor,
            sha256: format!("{:x}", hasher.finalize()),
            byte_length: values.len().saturating_mul(std::mem::size_of::<f32>()),
            item_count: values.len(),
        }
    }

    pub fn token_ids(values: &[u32]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"a3s-power-token-ids-v1\0");
        for value in values {
            hasher.update(value.to_le_bytes());
        }
        Self {
            representation: ExecutionRepresentation::TokenIds,
            sha256: format!("{:x}", hasher.finalize()),
            byte_length: values.len().saturating_mul(std::mem::size_of::<u32>()),
            item_count: values.len(),
        }
    }

    pub fn utf8_text(value: &str) -> Self {
        Self::bytes(
            ExecutionRepresentation::Utf8Text,
            value.as_bytes(),
            value.chars().count(),
        )
    }

    pub fn image_request(bytes: &[u8], image_count: usize) -> Self {
        Self::bytes(ExecutionRepresentation::ImageRequest, bytes, image_count)
    }

    fn bytes(representation: ExecutionRepresentation, bytes: &[u8], item_count: usize) -> Self {
        let domain = match representation {
            ExecutionRepresentation::ImageRequest => b"a3s-power-image-request-v1\0".as_slice(),
            ExecutionRepresentation::Utf8Text => b"a3s-power-utf8-text-v1\0".as_slice(),
            ExecutionRepresentation::F32Tensor | ExecutionRepresentation::TokenIds => {
                b"a3s-power-bytes-v1\0".as_slice()
            }
        };
        let mut hasher = Sha256::new();
        hasher.update(domain);
        hasher.update((item_count as u64).to_le_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
        Self {
            representation,
            sha256: format!("{:x}", hasher.finalize()),
            byte_length: bytes.len(),
            item_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_digest_binds_shape_and_values() {
        assert_ne!(
            ExecutionDigest::f32_tensor(&[1, 2], &[1.0, 2.0]),
            ExecutionDigest::f32_tensor(&[2, 1], &[1.0, 2.0])
        );
    }

    #[test]
    fn token_digest_has_typed_domain_separator() {
        let tokens = ExecutionDigest::token_ids(&[1, 2]);
        let tensor = ExecutionDigest::f32_tensor(&[2], &[f32::from_bits(1), f32::from_bits(2)]);
        assert_ne!(tokens.sha256, tensor.sha256);
    }

    #[test]
    fn image_digest_binds_the_image_count() {
        assert_ne!(
            ExecutionDigest::image_request(b"same bytes", 1).sha256,
            ExecutionDigest::image_request(b"same bytes", 2).sha256,
        );
    }

    #[test]
    fn older_receipts_default_to_no_microbatch_evidence() {
        let encoded = serde_json::json!({
            "schema": ExecutionReceipt::SCHEMA,
            "model": {
                "family": "test-model",
                "revision": "revision-1",
                "weightsSha256": "a".repeat(64),
            },
            "runtime": {
                "name": "a3s-power",
                "version": "0.1.0",
                "device": "cpu",
            },
            "input": ExecutionDigest::token_ids(&[1]),
            "output": ExecutionDigest::token_ids(&[2]),
        });
        let receipt: ExecutionReceipt = serde_json::from_value(encoded).unwrap();
        assert!(receipt.microbatch.is_none());
    }
}
