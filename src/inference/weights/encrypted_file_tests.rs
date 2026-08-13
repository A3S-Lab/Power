use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use super::{encrypt_seekable_weight_file, SeekableEncryptedFile, SeekableWeightKey};

const CHUNK_BYTES: u32 = 4096;

fn key(seed: u8) -> SeekableWeightKey {
    SeekableWeightKey::new([seed; 32])
}

fn plaintext() -> Vec<u8> {
    (0..10_000_u32)
        .map(|index| (index.wrapping_mul(31) % 251) as u8)
        .collect()
}

#[test]
fn seekable_encrypted_file_roundtrips_cross_chunk_ranges() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("weights.safetensors");
    let encrypted = directory.path().join("weights.safetensors.a3se");
    let expected = plaintext();
    std::fs::write(&source, &expected).unwrap();

    let report = encrypt_seekable_weight_file(&source, &encrypted, &key(7), CHUNK_BYTES).unwrap();
    assert_eq!(report.descriptor.plaintext_bytes, expected.len() as u64);
    assert_eq!(report.descriptor.chunks, 3);
    assert_eq!(report.peak_plaintext_chunk_bytes, u64::from(CHUNK_BYTES));
    assert_eq!(
        report.descriptor.plaintext_sha256,
        format!("{:x}", Sha256::digest(&expected))
    );

    let reader = SeekableEncryptedFile::open(&encrypted, &key(7)).unwrap();
    let verification = reader.verify(&CancellationToken::new()).unwrap();
    assert_eq!(verification, report);
    assert_eq!(
        reader
            .read_range(3_000, 6_000, &CancellationToken::new())
            .unwrap()
            .as_slice(),
        &expected[3_000..9_000]
    );
    assert!(reader
        .read_range(expected.len() as u64, 0, &CancellationToken::new())
        .unwrap()
        .is_empty());
}

#[test]
fn seekable_encrypted_file_fails_closed_on_wrong_key_tampering_and_bounds() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("weights.safetensors");
    let encrypted = directory.path().join("weights.safetensors.a3se");
    let expected = plaintext();
    std::fs::write(&source, &expected).unwrap();
    encrypt_seekable_weight_file(&source, &encrypted, &key(11), CHUNK_BYTES).unwrap();

    let wrong_key = SeekableEncryptedFile::open(&encrypted, &key(12)).unwrap();
    assert!(wrong_key
        .read_range(0, 1, &CancellationToken::new())
        .is_err());

    let reader = SeekableEncryptedFile::open(&encrypted, &key(11)).unwrap();
    assert!(reader
        .read_range(expected.len() as u64, 1, &CancellationToken::new())
        .is_err());
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert!(reader.read_range(0, 1, &cancelled).is_err());
    drop(reader);

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&encrypted)
        .unwrap();
    file.seek(SeekFrom::Start(64 + 73)).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::Start(64 + 73)).unwrap();
    file.write_all(&[byte[0] ^ 1]).unwrap();
    file.sync_all().unwrap();
    let tampered = SeekableEncryptedFile::open(&encrypted, &key(11)).unwrap();
    assert!(tampered.verify(&CancellationToken::new()).is_err());
}

#[test]
fn seekable_weight_keys_are_redacted_and_reader_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SeekableEncryptedFile>();
    assert_send_sync::<SeekableWeightKey>();
    assert_eq!(format!("{:?}", key(9)), "SeekableWeightKey([REDACTED])");
}
