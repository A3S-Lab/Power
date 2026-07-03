use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{PowerError, Result};
use crate::tee::encrypted_model::load_key;

/// Trait for model decryption key providers.
///
/// This is an extension point for HSM integration, key rotation, and
/// audit logging. The default implementation (`StaticKeyProvider`) wraps
/// the existing `KeySource` (file or env var).
///
/// Future implementations can integrate with:
/// - AWS KMS (`AwsKmsKeyProvider`)
/// - HashiCorp Vault (`VaultKeyProvider`)
/// - Azure Key Vault (`AzureKeyVaultProvider`)
#[async_trait]
pub trait KeyProvider: Send + Sync {
    /// Fetch the current decryption key.
    async fn get_key(&self) -> Result<[u8; 32]>;

    /// Rotate to a new key.
    ///
    /// Default: returns an error (rotation not supported).
    /// Override in providers that support key rotation.
    async fn rotate_key(&self) -> Result<[u8; 32]> {
        Err(PowerError::Config(
            "Key rotation not supported by this provider".to_string(),
        ))
    }

    /// Provider name for audit logging and diagnostics.
    fn provider_name(&self) -> &str;
}

/// A static key provider that loads a key once from a `KeySource`.
///
/// Wraps the existing `KeySource` (file path or env var) for backward
/// compatibility. The key is loaded on first call and cached.
pub struct StaticKeyProvider {
    source: crate::tee::encrypted_model::KeySource,
    cached: tokio::sync::OnceCell<[u8; 32]>,
}

impl StaticKeyProvider {
    /// Create a new static key provider from a `KeySource`.
    pub fn new(source: crate::tee::encrypted_model::KeySource) -> Self {
        Self {
            source,
            cached: tokio::sync::OnceCell::new(),
        }
    }
}

#[async_trait]
impl KeyProvider for StaticKeyProvider {
    async fn get_key(&self) -> Result<[u8; 32]> {
        self.cached
            .get_or_try_init(|| async { load_key(&self.source) })
            .await
            .copied()
    }

    fn provider_name(&self) -> &str {
        "static"
    }
}

/// A rotating key provider that supports multiple keys in sequence.
///
/// Holds a list of `KeySource` values. The current key is determined by
/// an atomic index. Calling `rotate_key()` advances to the next key,
/// enabling zero-downtime key rotation: deploy new key, rotate, remove old.
#[derive(Debug)]
pub struct RotatingKeyProvider {
    sources: Vec<crate::tee::encrypted_model::KeySource>,
    current: AtomicUsize,
}

impl RotatingKeyProvider {
    /// Create a new rotating key provider from a list of key sources.
    ///
    /// The first source is used initially. `rotate_key()` advances to the next.
    pub fn new(sources: Vec<crate::tee::encrypted_model::KeySource>) -> Result<Self> {
        if sources.is_empty() {
            return Err(PowerError::Config(
                "RotatingKeyProvider requires at least one key source".to_string(),
            ));
        }
        Ok(Self {
            sources,
            current: AtomicUsize::new(0),
        })
    }

    /// Return the index of the currently active key.
    pub fn current_index(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    /// Return the total number of key sources.
    pub fn key_count(&self) -> usize {
        self.sources.len()
    }
}

#[async_trait]
impl KeyProvider for RotatingKeyProvider {
    async fn get_key(&self) -> Result<[u8; 32]> {
        let idx = self.current.load(Ordering::Relaxed);
        load_key(&self.sources[idx])
    }

    async fn rotate_key(&self) -> Result<[u8; 32]> {
        let current = self.current.load(Ordering::Relaxed);
        let next = (current + 1) % self.sources.len();
        let key = load_key(&self.sources[next])?;
        self.current.store(next, Ordering::Relaxed);
        tracing::info!(
            from = current,
            to = next,
            total = self.sources.len(),
            "Key rotated"
        );
        Ok(key)
    }

    fn provider_name(&self) -> &str {
        "rotating"
    }
}

/// Build a `KeyProvider` from config.
///
/// Returns `None` if no key source is configured.
pub fn from_config(config: &crate::config::PowerConfig) -> Result<Option<Arc<dyn KeyProvider>>> {
    match config.key_provider.as_str() {
        "rotating" => {
            let provider = RotatingKeyProvider::new(config.key_rotation_sources.clone())?;
            tracing::info!(
                keys = config.key_rotation_sources.len(),
                "Rotating key provider initialized"
            );
            Ok(Some(Arc::new(provider)))
        }
        "static" => {
            let Some(ref source) = config.model_key_source else {
                return Ok(None);
            };
            let provider = StaticKeyProvider::new(source.clone());
            tracing::info!("Static key provider initialized");
            Ok(Some(Arc::new(provider)))
        }
        other => Err(PowerError::Config(format!(
            "unsupported key_provider '{other}'; expected one of: static, rotating"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tee::encrypted_model::KeySource;

    fn test_key_file(dir: &tempfile::TempDir) -> std::path::PathBuf {
        let key_path = dir.path().join("test.key");
        // load_key expects 64 hex chars (32 bytes encoded as hex)
        let hex: String = [0x42u8; 32].iter().map(|b| format!("{b:02x}")).collect();
        std::fs::write(&key_path, hex).unwrap();
        key_path
    }

    fn write_hex_key(path: &std::path::Path, byte: u8) {
        let hex: String = [byte; 32].iter().map(|b| format!("{b:02x}")).collect();
        std::fs::write(path, hex).unwrap();
    }

    #[tokio::test]
    async fn test_static_provider_loads_key_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = test_key_file(&dir);
        let provider = StaticKeyProvider::new(KeySource::File(key_path));
        let key = provider.get_key().await.unwrap();
        assert_eq!(key, [0x42u8; 32]);
    }

    #[tokio::test]
    async fn test_static_provider_caches_key() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = test_key_file(&dir);
        let provider = StaticKeyProvider::new(KeySource::File(key_path.clone()));

        let key1 = provider.get_key().await.unwrap();
        // Overwrite the file — cached key should still be returned
        std::fs::write(&key_path, [0xFFu8; 32]).unwrap();
        let key2 = provider.get_key().await.unwrap();
        assert_eq!(key1, key2);
        assert_eq!(key1, [0x42u8; 32]);
    }

    #[tokio::test]
    async fn test_static_provider_rotation_not_supported() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = test_key_file(&dir);
        let provider = StaticKeyProvider::new(KeySource::File(key_path));
        let result = provider.rotate_key().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not supported"));
    }

    #[test]
    fn test_static_provider_name() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = test_key_file(&dir);
        let provider = StaticKeyProvider::new(KeySource::File(key_path));
        assert_eq!(provider.provider_name(), "static");
    }

    #[tokio::test]
    async fn test_rotating_provider_loads_first_key() {
        let dir = tempfile::tempdir().unwrap();
        let key1_path = dir.path().join("key1.key");
        let key2_path = dir.path().join("key2.key");
        write_hex_key(&key1_path, 0x11);
        write_hex_key(&key2_path, 0x22);

        let provider =
            RotatingKeyProvider::new(vec![KeySource::File(key1_path), KeySource::File(key2_path)])
                .unwrap();

        let key = provider.get_key().await.unwrap();
        assert_eq!(key, [0x11u8; 32]);
        assert_eq!(provider.current_index(), 0);
    }

    #[tokio::test]
    async fn test_rotating_provider_rotates_to_next_key() {
        let dir = tempfile::tempdir().unwrap();
        let key1_path = dir.path().join("key1.key");
        let key2_path = dir.path().join("key2.key");
        write_hex_key(&key1_path, 0x11);
        write_hex_key(&key2_path, 0x22);

        let provider =
            RotatingKeyProvider::new(vec![KeySource::File(key1_path), KeySource::File(key2_path)])
                .unwrap();

        let rotated_key = provider.rotate_key().await.unwrap();
        assert_eq!(rotated_key, [0x22u8; 32]);
        assert_eq!(provider.current_index(), 1);

        let current_key = provider.get_key().await.unwrap();
        assert_eq!(current_key, [0x22u8; 32]);
    }

    #[tokio::test]
    async fn test_rotating_provider_wraps_around() {
        let dir = tempfile::tempdir().unwrap();
        let key1_path = dir.path().join("key1.key");
        let key2_path = dir.path().join("key2.key");
        write_hex_key(&key1_path, 0x11);
        write_hex_key(&key2_path, 0x22);

        let provider =
            RotatingKeyProvider::new(vec![KeySource::File(key1_path), KeySource::File(key2_path)])
                .unwrap();

        provider.rotate_key().await.unwrap(); // → key2
        provider.rotate_key().await.unwrap(); // → key1 (wraps)
        assert_eq!(provider.current_index(), 0);
    }

    #[test]
    fn test_rotating_provider_empty_sources_fails() {
        let result = RotatingKeyProvider::new(vec![]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least one"));
    }

    #[test]
    fn test_rotating_provider_name() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = test_key_file(&dir);
        let provider = RotatingKeyProvider::new(vec![KeySource::File(key_path)]).unwrap();
        assert_eq!(provider.provider_name(), "rotating");
    }

    #[test]
    fn test_rotating_provider_key_count() {
        let dir = tempfile::tempdir().unwrap();
        let key1_path = dir.path().join("key1.key");
        let key2_path = dir.path().join("key2.key");
        write_hex_key(&key1_path, 0x11);
        write_hex_key(&key2_path, 0x22);

        let provider =
            RotatingKeyProvider::new(vec![KeySource::File(key1_path), KeySource::File(key2_path)])
                .unwrap();
        assert_eq!(provider.key_count(), 2);
    }

    #[tokio::test]
    async fn test_from_config_static_provider_uses_model_key_source() {
        let dir = tempfile::tempdir().unwrap();
        let static_key = dir.path().join("static.key");
        let rotating_key = dir.path().join("rotating.key");
        write_hex_key(&static_key, 0x11);
        write_hex_key(&rotating_key, 0x22);
        let config = crate::config::PowerConfig {
            key_provider: "static".to_string(),
            model_key_source: Some(KeySource::File(static_key)),
            key_rotation_sources: vec![KeySource::File(rotating_key)],
            ..Default::default()
        };

        let provider = from_config(&config).unwrap().unwrap();

        assert_eq!(provider.provider_name(), "static");
        assert_eq!(provider.get_key().await.unwrap(), [0x11; 32]);
    }

    #[tokio::test]
    async fn test_from_config_rotating_provider_uses_rotation_sources() {
        let dir = tempfile::tempdir().unwrap();
        let static_key = dir.path().join("static.key");
        let rotating_key = dir.path().join("rotating.key");
        write_hex_key(&static_key, 0x11);
        write_hex_key(&rotating_key, 0x22);
        let config = crate::config::PowerConfig {
            key_provider: "rotating".to_string(),
            model_key_source: Some(KeySource::File(static_key)),
            key_rotation_sources: vec![KeySource::File(rotating_key)],
            ..Default::default()
        };

        let provider = from_config(&config).unwrap().unwrap();

        assert_eq!(provider.provider_name(), "rotating");
        assert_eq!(provider.get_key().await.unwrap(), [0x22; 32]);
    }

    #[test]
    fn test_from_config_static_without_key_source_returns_none() {
        let config = crate::config::PowerConfig {
            key_provider: "static".to_string(),
            model_key_source: None,
            ..Default::default()
        };

        assert!(from_config(&config).unwrap().is_none());
    }

    #[test]
    fn test_from_config_rotating_without_sources_fails_closed() {
        let config = crate::config::PowerConfig {
            key_provider: "rotating".to_string(),
            key_rotation_sources: vec![],
            ..Default::default()
        };

        let err = match from_config(&config) {
            Ok(_) => panic!("rotating key provider without sources should fail closed"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("at least one key source"));
    }

    #[test]
    fn test_from_config_unknown_provider_fails_closed() {
        let config = crate::config::PowerConfig {
            key_provider: "vault-ish".to_string(),
            ..Default::default()
        };

        let err = match from_config(&config) {
            Ok(_) => panic!("unknown key provider should fail closed"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("key_provider"));
    }
}
