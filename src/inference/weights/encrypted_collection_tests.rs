use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use candle_core::Device;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tokio_util::sync::CancellationToken;

use crate::inference::{
    DevicePreference, EmbeddedRuntime, PlacementPreference, ResidencyPolicy, WeightHierarchy,
    WeightKey, WeightRequest,
};

use super::*;

const CHUNK_BYTES: u32 = 4096;

fn key(seed: u8) -> SeekableWeightKey {
    SeekableWeightKey::new([seed; 32])
}

fn write_fixture(root: &std::path::Path) {
    for (file, name, seed, bytes) in [
        ("a.safetensors", "layer.0.weight", 3_u8, 12_000_usize),
        ("b.safetensors", "layer.1.weight", 7_u8, 9_000_usize),
    ] {
        let values = (0..bytes)
            .map(|index| seed.wrapping_add((index % 239) as u8))
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::U8, vec![bytes], values.as_slice()).unwrap();
        serialize_to_file([(name, view)], None, &root.join(file)).unwrap();
    }
}

#[test]
fn encrypted_collection_preserves_logical_identity_and_tensor_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("plain");
    let encrypted = directory.path().join("encrypted");
    std::fs::create_dir(&source).unwrap();
    write_fixture(&source);
    let limits = InferenceLimits::default();
    let canonical = WeightStore::open(&source, &limits).unwrap();

    let report =
        encrypt_seekable_weight_collection(&source, &encrypted, &key(17), CHUNK_BYTES, &limits)
            .unwrap();
    assert_eq!(report.plaintext_sha256, canonical.sha256());
    assert_eq!(report.files, 2);
    assert_eq!(report.peak_plaintext_chunk_bytes, u64::from(CHUNK_BYTES));
    assert!(!encrypted.join("a.safetensors").exists());
    assert!(encrypted.join("a.safetensors.a3se").is_file());

    let source =
        SeekableEncryptedWeightSource::new(&encrypted, report.manifest_sha256.clone(), key(17))
            .unwrap();
    let opened = WeightStore::open_seekable_encrypted(source, &limits).unwrap();
    assert_eq!(opened.sha256(), canonical.sha256());
    assert_eq!(opened.files(), canonical.files());
    assert_eq!(opened.inventory().len(), 2);
    assert!(matches!(
        opened.sources()[0].representation,
        WeightSourceRepresentation::SeekableAes256GcmV1 { .. }
    ));
    for name in ["layer.0.weight", "layer.1.weight"] {
        assert_eq!(
            opened.read_tensor_bytes(name).unwrap().bytes(),
            canonical.read_tensor_bytes(name).unwrap().bytes()
        );
        assert_eq!(
            opened
                .load(name, &Device::Cpu)
                .unwrap()
                .to_vec1::<u8>()
                .unwrap(),
            canonical
                .load(name, &Device::Cpu)
                .unwrap()
                .to_vec1::<u8>()
                .unwrap()
        );
    }
    let encrypted_range = opened
        .read_tensor_range("layer.0.weight", 3_500, 5_000)
        .unwrap();
    let canonical_range = canonical
        .read_tensor_range("layer.0.weight", 3_500, 5_000)
        .unwrap();
    assert_eq!(encrypted_range.bytes(), canonical_range.bytes());
    assert_eq!(encrypted_range.tensor_offset(), 3_500);
    assert!(matches!(
        encrypted_range.representation(),
        WeightSourceRepresentation::SeekableAes256GcmV1 { .. }
    ));

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert!(opened
        .read_tensor_bytes_with_cancellation("layer.0.weight", &cancellation)
        .is_err());
    assert!(opened
        .read_tensor_range_with_cancellation("layer.0.weight", 0, 1, &cancellation)
        .is_err());
    let cancelled_source =
        SeekableEncryptedWeightSource::new(&encrypted, report.manifest_sha256, key(17)).unwrap();
    assert!(WeightStore::open_seekable_encrypted_with_cancellation(
        cancelled_source,
        &limits,
        &cancellation,
    )
    .is_err());

    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap();
    let hierarchy = WeightHierarchy::new(
        Arc::new(opened),
        runtime.clone(),
        ResidencyPolicy {
            host_cache_bytes: 21_000,
            ..ResidencyPolicy::default()
        },
    )
    .unwrap();
    let active = CancellationToken::new();
    let permit = runtime.begin(&active).unwrap();
    let resident = hierarchy
        .load(
            &WeightRequest::new(
                WeightKey::new(0, "layer.0.weight"),
                PlacementPreference::Host,
            ),
            &permit,
            &active,
        )
        .unwrap();
    assert_eq!(resident.tensor().dims(), [12_000]);
}

#[test]
fn encrypted_collection_rejects_wrong_trust_anchor_key_and_ciphertext() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("plain");
    let encrypted = directory.path().join("encrypted");
    std::fs::create_dir(&source).unwrap();
    write_fixture(&source);
    let limits = InferenceLimits::default();
    let report =
        encrypt_seekable_weight_collection(&source, &encrypted, &key(23), CHUNK_BYTES, &limits)
            .unwrap();

    let wrong_manifest =
        SeekableEncryptedWeightSource::new(&encrypted, "0".repeat(64), key(23)).unwrap();
    assert!(WeightStore::open_seekable_encrypted(wrong_manifest, &limits).is_err());
    let wrong_key =
        SeekableEncryptedWeightSource::new(&encrypted, report.manifest_sha256.clone(), key(24))
            .unwrap();
    assert!(WeightStore::open_seekable_encrypted(wrong_key, &limits).is_err());

    let artifact = encrypted.join("a.safetensors.a3se");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&artifact)
        .unwrap();
    file.seek(SeekFrom::Start(64 + 101)).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::Start(64 + 101)).unwrap();
    file.write_all(&[byte[0] ^ 1]).unwrap();
    file.sync_all().unwrap();
    let tampered =
        SeekableEncryptedWeightSource::new(&encrypted, report.manifest_sha256, key(23)).unwrap();
    assert!(WeightStore::open_seekable_encrypted(tampered, &limits).is_err());
}

#[test]
fn encrypted_collection_rejects_unmanifested_artifacts() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("plain");
    let encrypted = directory.path().join("encrypted");
    std::fs::create_dir(&source).unwrap();
    write_fixture(&source);
    let limits = InferenceLimits::default();
    let report =
        encrypt_seekable_weight_collection(&source, &encrypted, &key(29), CHUNK_BYTES, &limits)
            .unwrap();
    std::fs::write(encrypted.join("unexpected.bin"), [1_u8]).unwrap();
    let source =
        SeekableEncryptedWeightSource::new(&encrypted, report.manifest_sha256, key(29)).unwrap();
    assert!(WeightStore::open_seekable_encrypted(source, &limits).is_err());
}
