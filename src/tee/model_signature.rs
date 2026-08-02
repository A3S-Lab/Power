//! Model-neutral Ed25519 verification for signed weight digests.

use std::path::{Path, PathBuf};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};

use crate::error::{PowerError, Result};

/// Verifies an Ed25519 signature over a known SHA-256 model digest.
///
/// The signature file must be at `<signature_anchor_path>.sig` and contain
/// exactly 64 raw signature bytes. The signed message is the 32 decoded bytes
/// of `model_hash_hex`, not its hexadecimal text representation.
pub fn verify_model_signature_hash(
    model_name: &str,
    model_hash_hex: &str,
    signature_anchor_path: &Path,
    public_key_hex: &str,
) -> Result<()> {
    let key_bytes =
        hex::decode(public_key_hex).map_err(|error| PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!("invalid public key hex: {error}"),
        })?;
    if key_bytes.len() != 32 {
        return Err(PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!("public key must be 32 bytes, got {}", key_bytes.len()),
        });
    }
    let key_array: [u8; 32] =
        key_bytes
            .try_into()
            .map_err(|_| PowerError::SignatureVerificationFailed {
                model: model_name.to_string(),
                reason: "public key length changed after validation".to_string(),
            })?;
    let verifying_key = VerifyingKey::from_bytes(&key_array).map_err(|error| {
        PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!("invalid public key: {error}"),
        }
    })?;

    let signature_path = signature_path(signature_anchor_path);
    let signature_bytes = std::fs::read(&signature_path).map_err(|error| {
        PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!(
                "signature file not found at {}: {error}",
                signature_path.display()
            ),
        }
    })?;
    if signature_bytes.len() != 64 {
        return Err(PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!("signature must be 64 bytes, got {}", signature_bytes.len()),
        });
    }
    let signature_array: [u8; 64] =
        signature_bytes
            .try_into()
            .map_err(|_| PowerError::SignatureVerificationFailed {
                model: model_name.to_string(),
                reason: "signature length changed after validation".to_string(),
            })?;
    let signature = Signature::from_bytes(&signature_array);

    let model_hash_bytes =
        hex::decode(model_hash_hex).map_err(|error| PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!("failed to decode model hash: {error}"),
        })?;
    if model_hash_bytes.len() != 32 {
        return Err(PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!(
                "model hash must be 32 bytes, got {}",
                model_hash_bytes.len()
            ),
        });
    }

    verifying_key
        .verify(&model_hash_bytes, &signature)
        .map_err(|error| PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!("signature invalid: {error}"),
        })?;

    tracing::info!(model = %model_name, "Model signature verified");
    Ok(())
}

pub(super) fn signature_path(model_path: &Path) -> PathBuf {
    let mut path = model_path.as_os_str().to_owned();
    path.push(".sig");
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::{Signer, SigningKey};
    use rand::rngs::OsRng;
    use sha2::{Digest, Sha256};

    use super::*;

    #[test]
    fn verifies_a_signature_over_digest_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let anchor = directory.path().join("model.safetensors");
        let digest = Sha256::digest(b"reviewed weights");
        let signing_key = SigningKey::generate(&mut OsRng);
        let signature = signing_key.sign(&digest);
        std::fs::write(signature_path(&anchor), signature.to_bytes()).unwrap();

        verify_model_signature_hash(
            "reviewed-model",
            &hex::encode(digest),
            &anchor,
            &hex::encode(signing_key.verifying_key().to_bytes()),
        )
        .unwrap();
    }

    #[test]
    fn rejects_wrong_digest_and_malformed_key_or_signature() {
        let directory = tempfile::tempdir().unwrap();
        let anchor = directory.path().join("model.safetensors");
        let signing_key = SigningKey::generate(&mut OsRng);
        let digest = Sha256::digest(b"reviewed weights");
        std::fs::write(
            signature_path(&anchor),
            signing_key.sign(&digest).to_bytes(),
        )
        .unwrap();
        let public_key = hex::encode(signing_key.verifying_key().to_bytes());

        assert!(verify_model_signature_hash(
            "reviewed-model",
            &hex::encode(Sha256::digest(b"tampered weights")),
            &anchor,
            &public_key,
        )
        .is_err());
        assert!(
            verify_model_signature_hash("reviewed-model", &hex::encode(digest), &anchor, "00",)
                .is_err()
        );
        std::fs::write(signature_path(&anchor), [0_u8; 63]).unwrap();
        assert!(verify_model_signature_hash(
            "reviewed-model",
            &hex::encode(digest),
            &anchor,
            &public_key,
        )
        .is_err());
    }
}
