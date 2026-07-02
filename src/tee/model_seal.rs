//! Model integrity verification for TEE environments.
//!
//! Verifies model artifact SHA-256 hashes against expected values from config,
//! ensuring models have not been tampered with before loading into the TEE.
//!
//! Also supports Ed25519 signature verification for supply-chain security:
//! model publishers sign the SHA-256 hash of the model file, and the
//! signature is stored alongside the model as `<model_path>.sig`.

use std::collections::HashMap;
use std::path::Path;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};

use crate::error::{PowerError, Result};
use crate::model::registry::ModelRegistry;
use crate::model::storage;
use crate::tee::key_provider::KeyProvider;

/// Verify a single model artifact's SHA-256 against an expected hash.
pub fn verify_model_integrity(model_path: &Path, expected_hash: &str) -> Result<bool> {
    let actual = storage::compute_sha256_path(model_path)?;
    Ok(actual == expected_hash)
}

/// Verify all registered models against the expected hashes from config.
///
/// Returns `Ok(verified_count)` if all models with expected hashes pass.
/// Returns `Err` on the first model that fails verification.
/// Models not listed in `expected_hashes` are skipped.
pub fn verify_all_models(
    registry: &ModelRegistry,
    expected_hashes: &HashMap<String, String>,
) -> Result<usize> {
    if expected_hashes.is_empty() {
        return Ok(0);
    }

    let mut verified = 0;
    for (name, expected) in expected_hashes {
        let manifest = registry
            .get(name)
            .map_err(|_| PowerError::IntegrityCheckFailed {
                model: name.clone(),
                expected: expected.clone(),
                actual: "<not found in registry>".to_string(),
            })?;

        let actual = storage::compute_sha256_path(&manifest.path)?;
        if actual != *expected {
            return Err(PowerError::IntegrityCheckFailed {
                model: name.clone(),
                expected: expected.clone(),
                actual,
            });
        }
        verified += 1;
        tracing::info!(model = %name, "Model integrity verified");
    }

    Ok(verified)
}

/// Verify all registered models, using plaintext hashes for encrypted artifacts.
pub async fn verify_all_models_with_key_provider(
    registry: &ModelRegistry,
    expected_hashes: &HashMap<String, String>,
    key_provider: Option<&dyn KeyProvider>,
) -> Result<usize> {
    if expected_hashes.is_empty() {
        return Ok(0);
    }

    let mut verified = 0;
    for (name, expected) in expected_hashes {
        let manifest = registry
            .get(name)
            .map_err(|_| PowerError::IntegrityCheckFailed {
                model: name.clone(),
                expected: expected.clone(),
                actual: "<not found in registry>".to_string(),
            })?;

        let actual = if manifest
            .path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("enc"))
        {
            let provider = key_provider.ok_or_else(|| {
                PowerError::Config(format!(
                    "Model '{name}' is encrypted (.enc) but no key provider is configured for plaintext integrity verification"
                ))
            })?;
            let key = provider.get_key().await.map_err(|e| {
                PowerError::Config(format!(
                    "Key provider failed while verifying encrypted model '{name}': {e}"
                ))
            })?;
            crate::tee::encrypted_model::compute_plaintext_sha256(&manifest.path, &key)?
        } else {
            storage::compute_sha256_path(&manifest.path)?
        };

        if actual != *expected {
            return Err(PowerError::IntegrityCheckFailed {
                model: name.clone(),
                expected: expected.clone(),
                actual,
            });
        }
        verified += 1;
        tracing::info!(model = %name, "Model integrity verified");
    }

    Ok(verified)
}

/// Verify a model file's Ed25519 signature.
///
/// The signature file must be at `<model_path>.sig` and contain exactly
/// 64 bytes (raw Ed25519 signature over the SHA-256 hash of the model file).
///
/// `public_key_hex` is the hex-encoded 32-byte Ed25519 public key of the
/// model publisher.
pub fn verify_model_signature(model_path: &Path, public_key_hex: &str) -> Result<()> {
    let model_name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");

    // Compute SHA-256 of the model file (the signed message)
    let model_hash_hex = storage::compute_sha256_file(model_path)?;
    verify_model_signature_hash(model_name, &model_hash_hex, model_path, public_key_hex)
}

/// Verify an Ed25519 signature over a known model SHA-256 hash.
///
/// The signature file must be at `<signature_anchor_path>.sig` and contain
/// exactly 64 bytes. This is used for encrypted artifacts where the signed
/// message is the plaintext model hash, but the distributable file is
/// `<model>.enc` and the signature naturally lives beside it as `<model>.enc.sig`.
pub fn verify_model_signature_hash(
    model_name: &str,
    model_hash_hex: &str,
    signature_anchor_path: &Path,
    public_key_hex: &str,
) -> Result<()> {
    // Parse the public key
    let key_bytes =
        hex::decode(public_key_hex).map_err(|e| PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!("invalid public key hex: {}", e),
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
    let verifying_key = VerifyingKey::from_bytes(&key_array).map_err(|e| {
        PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!("invalid public key: {}", e),
        }
    })?;

    // Read the signature file: <signature_anchor_path>.sig (appended, not replacing extension)
    let sig_path = signature_path(signature_anchor_path);
    let sig_bytes =
        std::fs::read(&sig_path).map_err(|e| PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!("signature file not found at {}: {}", sig_path.display(), e),
        })?;
    if sig_bytes.len() != 64 {
        return Err(PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!("signature must be 64 bytes, got {}", sig_bytes.len()),
        });
    }
    let sig_array: [u8; 64] =
        sig_bytes
            .try_into()
            .map_err(|_| PowerError::SignatureVerificationFailed {
                model: model_name.to_string(),
                reason: "signature length changed after validation".to_string(),
            })?;
    let signature = Signature::from_bytes(&sig_array);

    let model_hash_bytes =
        hex::decode(model_hash_hex).map_err(|e| PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!("failed to decode model hash: {}", e),
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

    // Verify the signature
    verifying_key
        .verify(&model_hash_bytes, &signature)
        .map_err(|e| PowerError::SignatureVerificationFailed {
            model: model_name.to_string(),
            reason: format!("signature invalid: {}", e),
        })?;

    tracing::info!(model = %model_name, "Model signature verified");
    Ok(())
}

/// Verify an Ed25519 signature over an in-memory model buffer.
pub fn verify_model_signature_bytes(
    model_name: &str,
    model_bytes: &[u8],
    signature_anchor_path: &Path,
    public_key_hex: &str,
) -> Result<()> {
    let model_hash_hex = storage::compute_sha256(model_bytes);
    verify_model_signature_hash(
        model_name,
        &model_hash_hex,
        signature_anchor_path,
        public_key_hex,
    )
}

fn signature_path(model_path: &Path) -> std::path::PathBuf {
    let mut p = model_path.as_os_str().to_owned();
    p.push(".sig");
    std::path::PathBuf::from(p)
}

/// Verify signatures for all registered models.
///
/// Returns `Ok(verified_count)` if all models pass signature verification.
/// Models are only checked if `public_key_hex` is provided.
pub fn verify_all_signatures(registry: &ModelRegistry, public_key_hex: &str) -> Result<usize> {
    let mut verified = 0;
    for manifest in registry.list()? {
        verify_model_signature(&manifest.path, public_key_hex)?;
        verified += 1;
    }
    Ok(verified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::manifest::{ModelFormat, ModelManifest, ModelParameters};
    use crate::model::storage;

    fn make_manifest(dir: &Path, name: &str, data: &[u8]) -> ModelManifest {
        std::env::set_var("A3S_POWER_HOME", dir);
        let (blob_path, sha256) = storage::store_blob(data).unwrap();
        ModelManifest {
            name: name.to_string(),
            format: ModelFormat::Gguf,
            size: data.len() as u64,
            sha256,
            parameters: Some(ModelParameters {
                context_length: Some(4096),
                embedding_length: None,
                parameter_count: None,
                quantization: None,
            }),
            created_at: chrono::Utc::now(),
            path: blob_path,
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_verify_model_integrity_pass() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = make_manifest(dir.path(), "test", b"model-data");
        let result = verify_model_integrity(&manifest.path, &manifest.sha256).unwrap();
        assert!(result);
    }

    #[test]
    #[serial_test::serial]
    fn test_verify_model_integrity_fail() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = make_manifest(dir.path(), "test", b"model-data");
        let result = verify_model_integrity(&manifest.path, "wrong-hash").unwrap();
        assert!(!result);
    }

    #[test]
    fn test_verify_model_integrity_file_not_found() {
        let result = verify_model_integrity(Path::new("/nonexistent/model.gguf"), "abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_all_models_empty_hashes() {
        let registry = ModelRegistry::new();
        let hashes = HashMap::new();
        let count = verify_all_models(&registry, &hashes).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    #[serial_test::serial]
    fn test_verify_all_models_pass() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = make_manifest(dir.path(), "test-model", b"model-data");
        let expected_hash = manifest.sha256.clone();

        let registry = ModelRegistry::new();
        registry.register(manifest).unwrap();

        let mut hashes = HashMap::new();
        hashes.insert("test-model".to_string(), expected_hash);

        let count = verify_all_models(&registry, &hashes).unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_verify_all_models_with_key_provider_uses_encrypted_plaintext_hash() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        let plaintext = b"encrypted model plaintext";
        std::fs::write(&plain_path, plaintext).unwrap();
        let key = [0x42; 32];
        let enc_path = crate::tee::encrypted_model::encrypt_model_file(&plain_path, &key).unwrap();
        let key_path = dir.path().join("model.key");
        std::fs::write(&key_path, hex::encode(key)).unwrap();
        let manifest = ModelManifest {
            name: "test-model".to_string(),
            format: ModelFormat::Gguf,
            size: std::fs::metadata(&enc_path).unwrap().len(),
            sha256: storage::compute_sha256_file(&enc_path).unwrap(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: enc_path,
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
        };
        let registry = ModelRegistry::new();
        registry.register_transient(manifest).unwrap();
        let hashes =
            HashMap::from([("test-model".to_string(), storage::compute_sha256(plaintext))]);
        let provider = crate::tee::key_provider::StaticKeyProvider::new(
            crate::tee::encrypted_model::KeySource::File(key_path),
        );

        let count = verify_all_models_with_key_provider(&registry, &hashes, Some(&provider)).await;

        assert_eq!(count.unwrap(), 1);
    }

    #[test]
    #[serial_test::serial]
    fn test_verify_all_models_fail_wrong_hash() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = make_manifest(dir.path(), "test-model", b"model-data");

        let registry = ModelRegistry::new();
        registry.register(manifest).unwrap();

        let mut hashes = HashMap::new();
        hashes.insert("test-model".to_string(), "wrong-hash".to_string());

        let result = verify_all_models(&registry, &hashes);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Integrity check failed"));
    }

    #[test]
    fn test_verify_all_models_fail_model_not_found() {
        let registry = ModelRegistry::new();
        let mut hashes = HashMap::new();
        hashes.insert("nonexistent".to_string(), "sha256:abc".to_string());

        let result = verify_all_models(&registry, &hashes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    // --- Ed25519 signature tests ---

    fn generate_test_keypair() -> (ed25519_dalek::SigningKey, String) {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;
        let signing_key = SigningKey::generate(&mut OsRng);
        let public_key_hex = hex::encode(signing_key.verifying_key().to_bytes());
        (signing_key, public_key_hex)
    }

    fn sign_model_file_for_test(model_path: &Path, signing_key: &ed25519_dalek::SigningKey) {
        let hash_hex = storage::compute_sha256_file(model_path).unwrap();
        sign_model_hash_for_test(model_path, &hash_hex, signing_key);
    }

    fn sign_model_hash_for_test(
        signature_anchor_path: &Path,
        hash_hex: &str,
        signing_key: &ed25519_dalek::SigningKey,
    ) {
        use ed25519_dalek::Signer;
        let hash_bytes = hex::decode(&hash_hex).unwrap();
        let signature = signing_key.sign(&hash_bytes);
        let sig_path = signature_path(signature_anchor_path);
        std::fs::write(&sig_path, signature.to_bytes()).unwrap();
    }

    #[test]
    fn test_verify_model_signature_valid() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"fake-model-data").unwrap();

        let (signing_key, public_key_hex) = generate_test_keypair();
        sign_model_file_for_test(&model_path, &signing_key);

        let result = verify_model_signature(&model_path, &public_key_hex);
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
    }

    #[test]
    fn test_verify_model_signature_hash_valid() {
        let dir = tempfile::tempdir().unwrap();
        let anchor_path = dir.path().join("model.gguf.enc");
        std::fs::write(&anchor_path, b"ciphertext").unwrap();
        let plaintext_hash = storage::compute_sha256(b"plaintext model bytes");

        let (signing_key, public_key_hex) = generate_test_keypair();
        sign_model_hash_for_test(&anchor_path, &plaintext_hash, &signing_key);

        let result = verify_model_signature_hash(
            "encrypted-model",
            &plaintext_hash,
            &anchor_path,
            &public_key_hex,
        );
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
    }

    #[test]
    fn test_verify_model_signature_bytes_valid() {
        let dir = tempfile::tempdir().unwrap();
        let anchor_path = dir.path().join("model.gguf.enc");
        std::fs::write(&anchor_path, b"ciphertext").unwrap();
        let plaintext = b"plaintext model bytes";
        let plaintext_hash = storage::compute_sha256(plaintext);

        let (signing_key, public_key_hex) = generate_test_keypair();
        sign_model_hash_for_test(&anchor_path, &plaintext_hash, &signing_key);

        let result = verify_model_signature_bytes(
            "encrypted-model",
            plaintext,
            &anchor_path,
            &public_key_hex,
        );
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
    }

    #[test]
    fn test_verify_model_signature_wrong_key() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"fake-model-data").unwrap();

        let (signing_key, _) = generate_test_keypair();
        sign_model_file_for_test(&model_path, &signing_key);

        // Use a different key for verification
        let (_, wrong_public_key_hex) = generate_test_keypair();
        let result = verify_model_signature(&model_path, &wrong_public_key_hex);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("signature invalid"));
    }

    #[test]
    fn test_verify_model_signature_missing_sig_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"fake-model-data").unwrap();

        let (_, public_key_hex) = generate_test_keypair();
        // No .sig file written
        let result = verify_model_signature(&model_path, &public_key_hex);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("signature file not found"));
    }

    #[test]
    fn test_verify_model_signature_invalid_public_key_hex() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"fake-model-data").unwrap();

        let result = verify_model_signature(&model_path, "not-valid-hex!");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid public key hex"));
    }

    #[test]
    fn test_verify_model_signature_wrong_key_length() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"fake-model-data").unwrap();

        // 16 bytes instead of 32
        let result = verify_model_signature(&model_path, "deadbeefdeadbeefdeadbeefdeadbeef");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("32 bytes"));
    }

    #[test]
    fn test_verify_model_signature_wrong_signature_length() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"fake-model-data").unwrap();

        let (_, public_key_hex) = generate_test_keypair();
        let mut sig_os = model_path.as_os_str().to_owned();
        sig_os.push(".sig");
        let sig_path = std::path::PathBuf::from(sig_os);
        std::fs::write(&sig_path, [0u8; 32]).unwrap();

        let result = verify_model_signature(&model_path, &public_key_hex);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("64 bytes"));
    }

    #[test]
    fn test_verify_model_signature_tampered_model() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"original-model-data").unwrap();

        let (signing_key, public_key_hex) = generate_test_keypair();
        sign_model_file_for_test(&model_path, &signing_key);

        // Tamper with the model file after signing
        std::fs::write(&model_path, b"tampered-model-data").unwrap();

        let result = verify_model_signature(&model_path, &public_key_hex);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("signature invalid"));
    }
}
