use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    AcceleratorExecutionEvidence, RuntimeDevice, ShapeProfileExecutionEvidence, RUNTIME_NAME,
};

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape_profile: Option<ShapeProfileExecutionEvidence>,
}

impl ExecutionReceipt {
    pub const SCHEMA: &'static str = "a3s.power.embedded-execution-receipt.v1";
    pub const ACCELERATOR_SCHEMA: &'static str = "a3s.power.embedded-execution-receipt.v2";
    pub const ACCELERATOR_MESH_SCHEMA: &'static str = "a3s.power.embedded-execution-receipt.v3";
    pub const MICROBATCH_SCHEMA: &'static str = "a3s.power.embedded-execution-receipt.v4";
    pub const SHAPE_PROFILE_SCHEMA: &'static str = "a3s.power.embedded-execution-receipt.v5";
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
        update_f32_values(&mut hasher, values);
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
        update_u32_values(&mut hasher, values);
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

#[cfg(target_endian = "little")]
fn update_f32_values(hasher: &mut Sha256, values: &[f32]) {
    hasher.update(bytemuck::cast_slice(values));
}

#[cfg(target_endian = "big")]
fn update_f32_values(hasher: &mut Sha256, values: &[f32]) {
    update_canonical_u32_words(hasher, values.iter().map(|value| value.to_bits()));
}

#[cfg(target_endian = "little")]
fn update_u32_values(hasher: &mut Sha256, values: &[u32]) {
    hasher.update(bytemuck::cast_slice(values));
}

#[cfg(target_endian = "big")]
fn update_u32_values(hasher: &mut Sha256, values: &[u32]) {
    update_canonical_u32_words(hasher, values.iter().copied());
}

#[cfg(target_endian = "big")]
fn update_canonical_u32_words(hasher: &mut Sha256, values: impl Iterator<Item = u32>) {
    const WORDS_PER_CHUNK: usize = 4_096;
    let mut encoded = [0_u8; WORDS_PER_CHUNK * std::mem::size_of::<u32>()];
    let mut encoded_len = 0;

    for value in values {
        encoded[encoded_len..encoded_len + 4].copy_from_slice(&value.to_le_bytes());
        encoded_len += 4;
        if encoded_len == encoded.len() {
            hasher.update(encoded);
            encoded_len = 0;
        }
    }
    hasher.update(&encoded[..encoded_len]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_f32_tensor_digest(shape: &[usize], values: &[f32]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"a3s-power-f32-tensor-v1\0");
        hasher.update((shape.len() as u64).to_le_bytes());
        for dimension in shape {
            hasher.update((*dimension as u64).to_le_bytes());
        }
        for value in values {
            hasher.update(value.to_bits().to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    fn scalar_token_digest(values: &[u32]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"a3s-power-token-ids-v1\0");
        for value in values {
            hasher.update(value.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    #[test]
    fn tensor_digest_binds_shape_and_values() {
        assert_ne!(
            ExecutionDigest::f32_tensor(&[1, 2], &[1.0, 2.0]),
            ExecutionDigest::f32_tensor(&[2, 1], &[1.0, 2.0])
        );
    }

    #[test]
    fn tensor_digest_preserves_the_scalar_v1_encoding() {
        let values = [
            0.0,
            -0.0,
            1.25,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_0001),
        ];
        let digest = ExecutionDigest::f32_tensor(&[2, 3], &values);
        assert_eq!(digest.sha256, scalar_f32_tensor_digest(&[2, 3], &values));
    }

    #[test]
    fn token_digest_has_typed_domain_separator() {
        let tokens = ExecutionDigest::token_ids(&[1, 2]);
        let tensor = ExecutionDigest::f32_tensor(&[2], &[f32::from_bits(1), f32::from_bits(2)]);
        assert_ne!(tokens.sha256, tensor.sha256);
    }

    #[test]
    fn token_digest_preserves_the_scalar_v1_encoding() {
        let values = [0, 1, u32::MAX, 0x0102_0304];
        let digest = ExecutionDigest::token_ids(&values);
        assert_eq!(digest.sha256, scalar_token_digest(&values));
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
        assert!(receipt.shape_profile.is_none());
    }
}
