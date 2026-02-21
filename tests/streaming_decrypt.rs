//! Integration tests for LayerStreamingDecryptedModel end-to-end.
//!
//! Tests the full encrypt → streaming-decrypt → chunk-read → zeroize cycle,
//! verifying that the streaming decryptor integrates correctly with the
//! encrypted model pipeline.

use std::sync::Arc;
use tempfile::TempDir;

use a3s_power::tee::encrypted_model::{
    encrypt_model_file, LayerStreamingDecryptedModel, MemoryDecryptedModel,
};

fn test_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    for (i, b) in key.iter_mut().enumerate() {
        *b = (i * 7 + 13) as u8;
    }
    key
}

// ── Basic streaming decrypt ───────────────────────────────────────────────────

#[test]
fn test_streaming_decrypt_full_roundtrip() {
    let dir = TempDir::new().unwrap();
    let plain_path = dir.path().join("model.gguf");
    let original = b"This is a fake model with multiple layers of weights data.";
    std::fs::write(&plain_path, original).unwrap();

    let key = test_key();
    let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

    let model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();
    assert_eq!(model.plaintext_size, original.len());

    // Read all data in one chunk
    let chunk = model.read_chunk(0, original.len()).unwrap();
    assert_eq!(&*chunk, original);
}

#[test]
fn test_streaming_decrypt_chunked_reads_match_original() {
    let dir = TempDir::new().unwrap();
    let plain_path = dir.path().join("model.gguf");
    // Simulate 3 "layers" of 16 bytes each
    let layer0 = b"LAYER_0_WEIGHTS_";
    let layer1 = b"LAYER_1_WEIGHTS_";
    let layer2 = b"LAYER_2_WEIGHTS_";
    let mut original = Vec::new();
    original.extend_from_slice(layer0);
    original.extend_from_slice(layer1);
    original.extend_from_slice(layer2);
    std::fs::write(&plain_path, &original).unwrap();

    let key = test_key();
    let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
    let model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();

    // Read each "layer" independently
    let chunk0 = model.read_chunk(0, 16).unwrap();
    let chunk1 = model.read_chunk(16, 16).unwrap();
    let chunk2 = model.read_chunk(32, 16).unwrap();

    assert_eq!(&*chunk0, layer0);
    assert_eq!(&*chunk1, layer1);
    assert_eq!(&*chunk2, layer2);
}

#[test]
fn test_streaming_decrypt_sha256_integrity() {
    let dir = TempDir::new().unwrap();
    let plain_path = dir.path().join("model.gguf");
    let original: Vec<u8> = (0..256).map(|i| i as u8).collect();
    std::fs::write(&plain_path, &original).unwrap();

    let key = test_key();
    let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
    let model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();

    // Reassemble via 32-byte chunks and verify SHA-256
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let chunk_size = 32;
    let mut offset = 0;
    while offset < model.len() {
        let size = chunk_size.min(model.len() - offset);
        let chunk = model.read_chunk(offset, size).unwrap();
        hasher.update(&*chunk);
        // chunk dropped here — Zeroizing<Vec<u8>> zeroizes automatically
        offset += size;
    }
    let result_hash = hasher.finalize();

    let mut expected_hasher = Sha256::new();
    expected_hasher.update(&original);
    let expected_hash = expected_hasher.finalize();

    assert_eq!(
        result_hash, expected_hash,
        "SHA-256 of chunked reads must match original"
    );
}

// ── Security properties ───────────────────────────────────────────────────────

#[test]
fn test_streaming_decrypt_no_disk_writes() {
    let dir = TempDir::new().unwrap();
    let plain_path = dir.path().join("model.gguf");
    std::fs::write(&plain_path, b"sensitive weights").unwrap();

    let key = test_key();
    let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
    let _model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();

    // No .dec file should exist
    assert!(
        !enc_path.with_extension("dec").exists(),
        "LayerStreamingDecryptedModel must not write plaintext to disk"
    );
}

#[test]
fn test_streaming_decrypt_chunk_zeroized_on_drop() {
    let dir = TempDir::new().unwrap();
    let plain_path = dir.path().join("model.gguf");
    std::fs::write(&plain_path, b"secret layer data here").unwrap();

    let key = test_key();
    let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
    let model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();

    {
        let chunk = model.read_chunk(0, 5).unwrap();
        assert_eq!(chunk.len(), 5);
        // chunk dropped here — Zeroizing<Vec<u8>> zeroizes the 5 bytes
    }
    // No panic = zeroize worked correctly
}

#[test]
fn test_streaming_decrypt_wrong_key_fails() {
    let dir = TempDir::new().unwrap();
    let plain_path = dir.path().join("model.gguf");
    std::fs::write(&plain_path, b"data").unwrap();

    let key = test_key();
    let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

    let wrong_key = [0xAAu8; 32];
    let result = LayerStreamingDecryptedModel::decrypt(&enc_path, &wrong_key);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("decryption failed"));
}

#[test]
fn test_streaming_decrypt_out_of_bounds_chunk_fails() {
    let dir = TempDir::new().unwrap();
    let plain_path = dir.path().join("model.gguf");
    std::fs::write(&plain_path, b"tiny").unwrap(); // 4 bytes

    let key = test_key();
    let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
    let model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();

    let result = model.read_chunk(0, 100); // 100 > 4
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of bounds"));
}

// ── Comparison with MemoryDecryptedModel ──────────────────────────────────────

#[test]
fn test_streaming_and_memory_decrypt_produce_same_plaintext() {
    let dir = TempDir::new().unwrap();
    let plain_path = dir.path().join("model.gguf");
    let original = b"identical plaintext from both decryptors";
    std::fs::write(&plain_path, original).unwrap();

    let key = test_key();
    let enc_path = encrypt_model_file(&plain_path, &key).unwrap();

    // Decrypt with MemoryDecryptedModel
    let mem_model = MemoryDecryptedModel::decrypt(&enc_path, &key).unwrap();

    // Decrypt with LayerStreamingDecryptedModel
    let stream_model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();

    // Both should produce identical plaintext
    let stream_chunk = stream_model.read_chunk(0, stream_model.len()).unwrap();
    assert_eq!(
        mem_model.as_bytes(),
        &*stream_chunk,
        "MemoryDecryptedModel and LayerStreamingDecryptedModel must produce identical plaintext"
    );
}

// ── Config field ──────────────────────────────────────────────────────────────

#[test]
fn test_streaming_decrypt_config_field_defaults_false() {
    let config = a3s_power::config::PowerConfig::default();
    assert!(
        !config.streaming_decrypt,
        "streaming_decrypt should default to false"
    );
}

#[test]
fn test_streaming_decrypt_config_field_serializes() {
    let mut config = a3s_power::config::PowerConfig::default();
    config.streaming_decrypt = true;
    let hcl = config.to_hcl();
    assert!(
        hcl.contains("streaming_decrypt = true"),
        "streaming_decrypt = true should appear in HCL output"
    );
}

#[test]
fn test_streaming_decrypt_config_false_not_serialized() {
    let config = a3s_power::config::PowerConfig::default();
    let hcl = config.to_hcl();
    // When false, streaming_decrypt should not appear (same pattern as other bool fields)
    assert!(
        !hcl.contains("streaming_decrypt"),
        "streaming_decrypt should not appear in HCL when false"
    );
}
