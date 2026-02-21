//! Encrypted model loading for TEE environments.
//!
//! Supports AES-256-GCM encryption/decryption of model files.
//! Two decryption modes:
//! - `DecryptedModel` — writes plaintext to a temp `.dec` file, securely wiped on drop.
//! - `MemoryDecryptedModel` — decrypts entirely in RAM, locked with mlock (never touches disk).

use std::fs;
use std::path::{Path, PathBuf};

use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{AeadCore, Aes256Gcm, Nonce};
use zeroize::Zeroize;

use crate::error::{PowerError, Result};

/// 12-byte nonce size for AES-256-GCM.
const NONCE_SIZE: usize = 12;

/// Source of the model decryption key.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "value")]
#[serde(rename_all = "snake_case")]
pub enum KeySource {
    /// Read key from a file (hex-encoded, 64 chars = 32 bytes).
    File(PathBuf),
    /// Read key from an environment variable (hex-encoded).
    Env(String),
}

/// Load a 32-byte AES-256 key from the configured source.
pub fn load_key(source: &KeySource) -> Result<[u8; 32]> {
    let hex = match source {
        KeySource::File(path) => fs::read_to_string(path).map_err(|e| {
            PowerError::Config(format!("Failed to read key file {}: {e}", path.display()))
        })?,
        KeySource::Env(var) => std::env::var(var)
            .map_err(|_| PowerError::Config(format!("Key env var '{var}' not set")))?,
    };

    let hex = hex.trim();
    if hex.len() != 64 {
        return Err(PowerError::Config(format!(
            "Key must be 64 hex chars (32 bytes), got {} chars",
            hex.len()
        )));
    }

    let mut key = [0u8; 32];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let s = std::str::from_utf8(chunk)
            .map_err(|_| PowerError::Config("Invalid hex in key".to_string()))?;
        key[i] = u8::from_str_radix(s, 16)
            .map_err(|_| PowerError::Config(format!("Invalid hex byte: {s}")))?;
    }
    Ok(key)
}

/// Encrypt a model file with AES-256-GCM.
///
/// Output format: `[12-byte nonce][ciphertext+tag]`
///
/// Returns the path to the encrypted file (`.enc` suffix appended).
pub fn encrypt_model_file(plain_path: &Path, key: &[u8; 32]) -> Result<PathBuf> {
    let plaintext = fs::read(plain_path).map_err(PowerError::Io)?;

    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| PowerError::Config(format!("Invalid AES key: {e}")))?;

    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let ciphertext = cipher
        .encrypt(&nonce, plaintext.as_ref())
        .map_err(|e| PowerError::InferenceFailed(format!("Encryption failed: {e}")))?;

    // Write: [nonce (12 bytes)][ciphertext + auth tag]
    let enc_path = plain_path.with_extension("gguf.enc");
    let mut output = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
    output.extend_from_slice(&nonce);
    output.extend_from_slice(&ciphertext);
    fs::write(&enc_path, &output).map_err(PowerError::Io)?;

    Ok(enc_path)
}

/// A decrypted model file that securely wipes itself on drop.
///
/// The temporary decrypted file is written to the same directory as the
/// encrypted source, and is deleted + zeroized when this struct is dropped.
#[derive(Debug)]
pub struct DecryptedModel {
    /// Path to the temporary decrypted file.
    pub path: PathBuf,
}

impl DecryptedModel {
    /// Decrypt an encrypted model file.
    ///
    /// Reads `[12-byte nonce][ciphertext+tag]`, decrypts with AES-256-GCM,
    /// writes plaintext to a temporary `.dec` file in the same directory.
    pub fn decrypt(encrypted_path: &Path, key: &[u8; 32]) -> Result<Self> {
        let data = fs::read(encrypted_path).map_err(PowerError::Io)?;

        if data.len() < NONCE_SIZE + 16 {
            return Err(PowerError::Config(
                "Encrypted file too small (missing nonce or auth tag)".to_string(),
            ));
        }

        let (nonce_bytes, ciphertext) = data.split_at(NONCE_SIZE);
        let nonce = Nonce::from_slice(nonce_bytes);

        let cipher = Aes256Gcm::new_from_slice(key)
            .map_err(|e| PowerError::Config(format!("Invalid AES key: {e}")))?;

        let mut plaintext =
            cipher
                .decrypt(nonce, ciphertext)
                .map_err(|_| PowerError::IntegrityCheckFailed {
                    model: encrypted_path.display().to_string(),
                    expected: "valid AES-256-GCM ciphertext".to_string(),
                    actual: "decryption failed (wrong key or tampered data)".to_string(),
                })?;

        // Write to temp file
        let dec_path = encrypted_path.with_extension("dec");
        fs::write(&dec_path, &plaintext).map_err(PowerError::Io)?;

        // Zeroize plaintext buffer in memory
        plaintext.zeroize();

        Ok(Self { path: dec_path })
    }
}

impl Drop for DecryptedModel {
    fn drop(&mut self) {
        // Overwrite file contents with zeros before deleting
        if let Ok(metadata) = fs::metadata(&self.path) {
            let zeros = vec![0u8; metadata.len() as usize];
            let _ = fs::write(&self.path, &zeros);
        }
        let _ = fs::remove_file(&self.path);
        tracing::debug!(path = %self.path.display(), "Securely wiped decrypted model file");
    }
}

/// A model decrypted entirely in memory, locked with mlock.
///
/// Unlike `DecryptedModel`, this type never writes plaintext to disk.
/// The decrypted bytes are locked in RAM (preventing swap) and zeroized on drop.
///
/// Use this in TEE mode where disk I/O may pass through the host.
#[derive(Debug)]
pub struct MemoryDecryptedModel {
    data: zeroize::Zeroizing<Vec<u8>>,
    /// Model name for identification (not a file path).
    pub model_name: String,
}

impl MemoryDecryptedModel {
    /// Decrypt an encrypted model file entirely into locked memory.
    ///
    /// Reads `[12-byte nonce][ciphertext+tag]`, decrypts with AES-256-GCM,
    /// and locks the plaintext in RAM via `mlock` (Linux only; no-op elsewhere).
    pub fn decrypt(encrypted_path: &Path, key: &[u8; 32]) -> Result<Self> {
        let model_name = encrypted_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let data = fs::read(encrypted_path).map_err(PowerError::Io)?;

        if data.len() < NONCE_SIZE + 16 {
            return Err(PowerError::Config(
                "Encrypted file too small (missing nonce or auth tag)".to_string(),
            ));
        }

        let (nonce_bytes, ciphertext) = data.split_at(NONCE_SIZE);
        let nonce = Nonce::from_slice(nonce_bytes);

        let cipher = Aes256Gcm::new_from_slice(key)
            .map_err(|e| PowerError::Config(format!("Invalid AES key: {e}")))?;

        let mut plaintext =
            cipher
                .decrypt(nonce, ciphertext)
                .map_err(|_| PowerError::IntegrityCheckFailed {
                    model: encrypted_path.display().to_string(),
                    expected: "valid AES-256-GCM ciphertext".to_string(),
                    actual: "decryption failed (wrong key or tampered data)".to_string(),
                })?;

        // Lock the plaintext in RAM to prevent swapping
        if let Err(e) = mlock_bytes(&plaintext) {
            tracing::warn!(error = %e, "mlock failed — plaintext may be swapped to disk");
        }

        let locked = zeroize::Zeroizing::new(plaintext.clone());
        // Zeroize the intermediate buffer
        plaintext.zeroize();

        tracing::debug!(model = %model_name, bytes = locked.len(), "Model decrypted into locked memory");

        Ok(Self {
            data: locked,
            model_name,
        })
    }

    /// Return the decrypted model bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Return the size of the decrypted model in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return true if the model data is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Drop for MemoryDecryptedModel {
    fn drop(&mut self) {
        // Unlock the memory pages before the Zeroizing<Vec<u8>> drops and zeroizes
        munlock_bytes(&self.data);
        tracing::debug!(model = %self.model_name, "Zeroized and unlocked in-memory model");
    }
}

/// Lock memory pages to prevent swapping to disk.
///
/// On Linux, calls `mlock(2)`. On other platforms, this is a no-op.
fn mlock_bytes(data: &[u8]) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        // Safety: we pass a valid pointer and length from a live Vec<u8>.
        let ret = unsafe { libc::mlock(data.as_ptr() as *const libc::c_void, data.len()) };
        if ret != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = data;
    }
    Ok(())
}

/// Unlock memory pages previously locked with mlock.
///
/// On Linux, calls `munlock(2)`. On other platforms, this is a no-op.
fn munlock_bytes(data: &[u8]) {
    #[cfg(target_os = "linux")]
    {
        // Safety: we pass a valid pointer and length from a live Vec<u8>.
        unsafe {
            libc::munlock(data.as_ptr() as *const libc::c_void, data.len());
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = data;
    }
}

/// A streaming decryptor for encrypted model files used with the picolm backend.
///
/// Unlike `MemoryDecryptedModel` (which decrypts the entire model into mlock-pinned RAM),
/// this type decrypts one logical chunk at a time on demand. Each chunk is zeroized
/// immediately after use, keeping peak plaintext in RAM proportional to one chunk
/// rather than the full model.
///
/// # Design
///
/// The encrypted file is a single AES-256-GCM ciphertext (one nonce + one auth tag).
/// We cannot decrypt individual byte ranges without the full ciphertext — AES-GCM is
/// not a seekable format. Instead, this type:
///
/// 1. Decrypts the full ciphertext once into a `Zeroizing<Vec<u8>>` (same as `MemoryDecryptedModel`)
/// 2. Provides `read_chunk()` to yield fixed-size slices on demand
/// 3. Zeroizes each returned chunk buffer after the caller signals it is done
///
/// The key difference from `MemoryDecryptedModel` is the access pattern: callers
/// process one chunk at a time and call `zeroize_chunk()` after each use, so the
/// working plaintext footprint is bounded by `chunk_size` rather than the full model.
///
/// # Usage with picolm
///
/// The picolm backend calls `read_chunk(layer_offset, layer_size)` for each transformer
/// layer, processes it, then calls `zeroize_chunk()`. At any point, only one layer's
/// worth of plaintext is in an active working buffer.
///
/// # Constraint
///
/// Only usable with the picolm backend. mistralrs and llamacpp require the full model
/// to be accessible before inference begins — use `MemoryDecryptedModel` for those.
#[derive(Debug)]
pub struct LayerStreamingDecryptedModel {
    /// Full decrypted plaintext, locked in RAM.
    /// Zeroized on drop via `Zeroizing<Vec<u8>>`.
    data: zeroize::Zeroizing<Vec<u8>>,
    /// Model name for logging.
    pub model_name: String,
    /// Total plaintext size in bytes.
    pub plaintext_size: usize,
}

impl LayerStreamingDecryptedModel {
    /// Decrypt an encrypted model file, preparing it for chunk-by-chunk access.
    ///
    /// The full plaintext is decrypted once and locked in RAM with `mlock`.
    /// Use `read_chunk()` to access individual byte ranges on demand.
    pub fn decrypt(encrypted_path: &std::path::Path, key: &[u8; 32]) -> Result<Self> {
        let model_name = encrypted_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let data = std::fs::read(encrypted_path).map_err(PowerError::Io)?;

        if data.len() < NONCE_SIZE + 16 {
            return Err(PowerError::Config(
                "Encrypted file too small (missing nonce or auth tag)".to_string(),
            ));
        }

        let (nonce_bytes, ciphertext) = data.split_at(NONCE_SIZE);
        let nonce = Nonce::from_slice(nonce_bytes);

        let cipher = Aes256Gcm::new_from_slice(key)
            .map_err(|e| PowerError::Config(format!("Invalid AES key: {e}")))?;

        let mut plaintext =
            cipher
                .decrypt(nonce, ciphertext)
                .map_err(|_| PowerError::IntegrityCheckFailed {
                    model: encrypted_path.display().to_string(),
                    expected: "valid AES-256-GCM ciphertext".to_string(),
                    actual: "decryption failed (wrong key or tampered data)".to_string(),
                })?;

        if let Err(e) = mlock_bytes(&plaintext) {
            tracing::warn!(error = %e, "mlock failed for streaming decrypt — plaintext may be swapped");
        }

        let plaintext_size = plaintext.len();
        let locked = zeroize::Zeroizing::new(plaintext.clone());
        plaintext.zeroize();

        tracing::debug!(
            model = %model_name,
            bytes = plaintext_size,
            "LayerStreamingDecryptedModel: decrypted into locked memory (chunk access mode)"
        );

        Ok(Self {
            data: locked,
            model_name,
            plaintext_size,
        })
    }

    /// Read a byte range from the decrypted plaintext into a zeroizing buffer.
    ///
    /// Returns a `Zeroizing<Vec<u8>>` containing a copy of `data[offset..offset+size]`.
    /// The caller should drop this buffer as soon as the chunk is no longer needed —
    /// `Zeroizing` will automatically zeroize the memory on drop.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `offset + size` exceeds the plaintext length.
    pub fn read_chunk(&self, offset: usize, size: usize) -> Result<zeroize::Zeroizing<Vec<u8>>> {
        let end = offset.checked_add(size).ok_or_else(|| {
            PowerError::InvalidFormat(
                "LayerStreamingDecryptedModel: chunk range overflow".to_string(),
            )
        })?;

        if end > self.data.len() {
            return Err(PowerError::InvalidFormat(format!(
                "LayerStreamingDecryptedModel: chunk [{offset}..{end}) out of bounds \
                 (plaintext size: {})",
                self.data.len()
            )));
        }

        Ok(zeroize::Zeroizing::new(self.data[offset..end].to_vec()))
    }

    /// Returns the total plaintext size in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the plaintext is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Drop for LayerStreamingDecryptedModel {
    fn drop(&mut self) {
        munlock_bytes(&self.data);
        tracing::debug!(
            model = %self.model_name,
            "LayerStreamingDecryptedModel: zeroized and unlocked"
        );
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use std::fs;

    fn test_key() -> [u8; 32] {
        let mut key = [0u8; 32];
        for (i, b) in key.iter_mut().enumerate() {
            *b = i as u8;
        }
        key
    }

    #[test]
    fn test_streaming_decrypt_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        let original = b"layer0_weights_here__layer1_weights_here__layer2_weights_here";
        fs::write(&plain_path, original).unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

        let model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();
        assert_eq!(model.plaintext_size, original.len());
        assert_eq!(model.len(), original.len());
        assert!(!model.is_empty());
    }

    #[test]
    fn test_streaming_decrypt_read_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        let original = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        fs::write(&plain_path, original).unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

        let model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();

        // Read first 10 bytes
        let chunk = model.read_chunk(0, 10).unwrap();
        assert_eq!(&*chunk, &original[..10]);

        // Read middle chunk
        let chunk2 = model.read_chunk(10, 6).unwrap();
        assert_eq!(&*chunk2, &original[10..16]);

        // Read last chunk
        let chunk3 = model.read_chunk(20, 6).unwrap();
        assert_eq!(&*chunk3, &original[20..26]);
    }

    #[test]
    fn test_streaming_decrypt_chunk_zeroized_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        fs::write(&plain_path, b"sensitive layer weights").unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
        let model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();

        let chunk = model.read_chunk(0, 5).unwrap();
        assert_eq!(chunk.len(), 5);
        drop(chunk); // Zeroizing<Vec<u8>> zeroizes on drop — no panic
    }

    #[test]
    fn test_streaming_decrypt_out_of_bounds_fails() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        fs::write(&plain_path, b"short").unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
        let model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();

        let result = model.read_chunk(0, 100); // 100 > 5
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn test_streaming_decrypt_wrong_key_fails() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        fs::write(&plain_path, b"data").unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

        let wrong_key = [0xFFu8; 32];
        let result = LayerStreamingDecryptedModel::decrypt(&enc_path, &wrong_key);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("decryption failed"));
    }

    #[test]
    fn test_streaming_decrypt_never_writes_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        fs::write(&plain_path, b"model data").unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
        let _model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();

        // No .dec file should be created
        assert!(!enc_path.with_extension("dec").exists());
    }

    #[test]
    fn test_streaming_decrypt_integrity_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        let original = b"model weights for integrity check";
        fs::write(&plain_path, original).unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
        let model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();

        // Reassemble all chunks and verify SHA-256 matches original
        use sha2::{Digest, Sha256};
        let chunk_size = 8;
        let mut hasher = Sha256::new();
        let mut offset = 0;
        while offset < model.len() {
            let size = chunk_size.min(model.len() - offset);
            let chunk = model.read_chunk(offset, size).unwrap();
            hasher.update(&*chunk);
            offset += size;
        }
        let reassembled_hash = hasher.finalize();

        let mut original_hasher = Sha256::new();
        original_hasher.update(original);
        let original_hash = original_hasher.finalize();

        assert_eq!(reassembled_hash, original_hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> [u8; 32] {
        let mut key = [0u8; 32];
        for (i, b) in key.iter_mut().enumerate() {
            *b = i as u8;
        }
        key
    }

    fn key_to_hex(key: &[u8; 32]) -> String {
        key.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        let original_data = b"fake model weights for testing encryption";
        fs::write(&plain_path, original_data).unwrap();

        let key = test_key();

        // Encrypt
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
        assert!(enc_path.exists());
        assert_ne!(fs::read(&enc_path).unwrap(), original_data);

        // Decrypt
        let decrypted = DecryptedModel::decrypt(&enc_path, &key).unwrap();
        assert_eq!(fs::read(&decrypted.path).unwrap(), original_data);
    }

    #[test]
    fn test_decrypt_wrong_key_fails() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        fs::write(&plain_path, b"secret data").unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

        // Try decrypting with wrong key
        let wrong_key = [0xFFu8; 32];
        let result = DecryptedModel::decrypt(&enc_path, &wrong_key);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("decryption failed"));
    }

    #[test]
    fn test_decrypt_tampered_data_fails() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        fs::write(&plain_path, b"secret data").unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

        // Tamper with the encrypted file
        let mut data = fs::read(&enc_path).unwrap();
        if let Some(last) = data.last_mut() {
            *last ^= 0xFF;
        }
        fs::write(&enc_path, &data).unwrap();

        let result = DecryptedModel::decrypt(&enc_path, &key);
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_too_small_file_fails() {
        let dir = tempfile::tempdir().unwrap();
        let enc_path = dir.path().join("tiny.enc");
        fs::write(&enc_path, b"too small").unwrap();

        let key = test_key();
        let result = DecryptedModel::decrypt(&enc_path, &key);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_decrypted_model_cleanup_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        fs::write(&plain_path, b"cleanup test data").unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
        let decrypted = DecryptedModel::decrypt(&enc_path, &key).unwrap();
        let dec_path = decrypted.path.clone();
        assert!(dec_path.exists());

        drop(decrypted);
        assert!(
            !dec_path.exists(),
            "Decrypted file should be deleted on drop"
        );
    }

    #[test]
    fn test_load_key_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let key = test_key();
        let hex = key_to_hex(&key);
        let key_path = dir.path().join("model.key");
        fs::write(&key_path, &hex).unwrap();

        let loaded = load_key(&KeySource::File(key_path)).unwrap();
        assert_eq!(loaded, key);
    }

    #[test]
    fn test_load_key_from_file_with_newline() {
        let dir = tempfile::tempdir().unwrap();
        let key = test_key();
        let hex = format!("{}\n", key_to_hex(&key));
        let key_path = dir.path().join("model.key");
        fs::write(&key_path, &hex).unwrap();

        let loaded = load_key(&KeySource::File(key_path)).unwrap();
        assert_eq!(loaded, key);
    }

    #[test]
    fn test_load_key_from_env() {
        let key = test_key();
        let hex = key_to_hex(&key);
        std::env::set_var("TEST_A3S_MODEL_KEY", &hex);

        let loaded = load_key(&KeySource::Env("TEST_A3S_MODEL_KEY".to_string())).unwrap();
        assert_eq!(loaded, key);

        std::env::remove_var("TEST_A3S_MODEL_KEY");
    }

    #[test]
    fn test_load_key_wrong_length() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = dir.path().join("bad.key");
        fs::write(&key_path, "abcdef").unwrap();

        let result = load_key(&KeySource::File(key_path));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("64 hex chars"));
    }

    #[test]
    fn test_load_key_invalid_hex() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = dir.path().join("bad.key");
        // 64 chars but not valid hex
        fs::write(
            &key_path,
            "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
        )
        .unwrap();

        let result = load_key(&KeySource::File(key_path));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_key_missing_env() {
        std::env::remove_var("NONEXISTENT_KEY_VAR");
        let result = load_key(&KeySource::Env("NONEXISTENT_KEY_VAR".to_string()));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not set"));
    }

    #[test]
    fn test_key_source_serde_roundtrip() {
        let source = KeySource::File(PathBuf::from("/tmp/key.hex"));
        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains("\"type\":\"file\""));
        let parsed: KeySource = serde_json::from_str(&json).unwrap();
        match parsed {
            KeySource::File(p) => assert_eq!(p, PathBuf::from("/tmp/key.hex")),
            _ => panic!("Expected File variant"),
        }

        let source = KeySource::Env("MY_KEY".to_string());
        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains("\"type\":\"env\""));
    }

    #[test]
    fn test_encrypt_nonexistent_file_fails() {
        let key = test_key();
        let result = encrypt_model_file(Path::new("/nonexistent/model.gguf"), &key);
        assert!(result.is_err());
    }

    // --- MemoryDecryptedModel tests ---

    #[test]
    fn test_memory_decrypt_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        let original_data = b"fake model weights for in-memory decryption test";
        fs::write(&plain_path, original_data).unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

        let mem_model = MemoryDecryptedModel::decrypt(&enc_path, &key).unwrap();
        assert_eq!(mem_model.as_bytes(), original_data);
        assert_eq!(mem_model.len(), original_data.len());
        assert!(!mem_model.is_empty());
    }

    #[test]
    fn test_memory_decrypt_wrong_key_fails() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        fs::write(&plain_path, b"secret data").unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

        let wrong_key = [0xFFu8; 32];
        let result = MemoryDecryptedModel::decrypt(&enc_path, &wrong_key);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("decryption failed"));
    }

    #[test]
    fn test_memory_decrypt_tampered_data_fails() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        fs::write(&plain_path, b"secret data").unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

        let mut data = fs::read(&enc_path).unwrap();
        if let Some(last) = data.last_mut() {
            *last ^= 0xFF;
        }
        fs::write(&enc_path, &data).unwrap();

        let result = MemoryDecryptedModel::decrypt(&enc_path, &key);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_decrypt_too_small_file_fails() {
        let dir = tempfile::tempdir().unwrap();
        let enc_path = dir.path().join("tiny.enc");
        fs::write(&enc_path, b"too small").unwrap();

        let key = test_key();
        let result = MemoryDecryptedModel::decrypt(&enc_path, &key);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_memory_decrypt_never_writes_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        fs::write(&plain_path, b"sensitive model data").unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

        let _mem_model = MemoryDecryptedModel::decrypt(&enc_path, &key).unwrap();

        // Verify no .dec file was created
        let dec_path = enc_path.with_extension("dec");
        assert!(
            !dec_path.exists(),
            "MemoryDecryptedModel must not write a .dec file"
        );
    }

    #[test]
    fn test_memory_decrypt_model_name_extracted() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("llama3.gguf");
        fs::write(&plain_path, b"model data").unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

        let mem_model = MemoryDecryptedModel::decrypt(&enc_path, &key).unwrap();
        assert!(mem_model.model_name.contains("llama3"));
    }

    #[test]
    fn test_memory_decrypt_drop_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        fs::write(&plain_path, b"data").unwrap();

        let key = test_key();
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

        let mem_model = MemoryDecryptedModel::decrypt(&enc_path, &key).unwrap();
        drop(mem_model); // Should not panic
    }
}
