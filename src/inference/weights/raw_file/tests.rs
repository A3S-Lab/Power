use std::io::{Seek, SeekFrom, Write};

use super::*;

fn fixture(raw: &[u8]) -> (tempfile::TempDir, WeightFileDescriptor) {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("weights.bin"), raw).unwrap();
    let descriptor = WeightFileDescriptor {
        relative_path: "weights.bin".into(),
        bytes: raw.len() as u64,
        sha256: format!("{:x}", Sha256::digest(raw)),
    };
    (root, descriptor)
}

fn open(root: &Path, descriptor: WeightFileDescriptor, budget: u64) -> VerifiedRawWeightFile {
    VerifiedRawWeightFile::open(
        root,
        descriptor,
        &InferenceLimits::default(),
        budget,
        &CancellationToken::new(),
    )
    .unwrap()
}

#[test]
fn unaligned_ranges_cross_digest_chunks_without_changing_bytes() {
    let raw: Vec<_> = (0..RANGE_READ_CHUNK_BYTES + 113)
        .map(|index| (index % 251) as u8)
        .collect();
    let (root, descriptor) = fixture(&raw);
    let file = open(root.path(), descriptor.clone(), 4096);
    for (offset, length) in [(0, 19), (1, 4096), (RANGE_READ_CHUNK_BYTES - 7, 120)] {
        let read = file
            .read_range(offset as u64, length as u64, &CancellationToken::new())
            .unwrap();
        assert_eq!(read.bytes(), &raw[offset..offset + length]);
        assert_eq!(read.descriptor(), &descriptor);
        assert_eq!(read.offset(), offset as u64);
    }
    assert_eq!(file.chunk_hashes.len(), 2);
}

#[test]
fn nested_portable_paths_are_resolved_under_the_explicit_root() {
    let (root, mut descriptor) = fixture(b"weights");
    std::fs::create_dir(root.path().join("nested")).unwrap();
    std::fs::rename(
        root.path().join("weights.bin"),
        root.path().join("nested/weights.bin"),
    )
    .unwrap();
    descriptor.relative_path = "nested/weights.bin".into();
    let file = open(root.path(), descriptor, 7);
    assert_eq!(
        file.read_range(0, 7, &CancellationToken::new())
            .unwrap()
            .bytes(),
        b"weights"
    );
}

#[test]
fn bad_descriptors_fail_before_path_resolution_or_allocation() {
    let (_, descriptor) = fixture(b"weights");
    for path in [
        "",
        ".",
        "..",
        "../weights.bin",
        "/weights.bin",
        "a/./b",
        "a//b",
        "a/",
        "a\\b",
        "C:/a",
        "a:b",
        "a\0b",
    ] {
        let mut bad = descriptor.clone();
        bad.relative_path = path.into();
        assert!(validate_descriptor(&bad).is_err(), "{path:?}");
    }
    for digest in ["".into(), "a".repeat(63), "A".repeat(64), "g".repeat(64)] {
        let mut bad = descriptor.clone();
        bad.sha256 = digest;
        assert!(validate_descriptor(&bad).is_err());
    }
    let mut empty = descriptor;
    empty.bytes = 0;
    assert!(validate_descriptor(&empty).is_err());
}

#[test]
fn full_file_hash_size_and_regular_file_kind_are_required() {
    let (root, descriptor) = fixture(b"weights");
    let mut wrong_hash = descriptor.clone();
    wrong_hash.sha256 = "0".repeat(64);
    let result = VerifiedRawWeightFile::open(
        root.path(),
        wrong_hash,
        &InferenceLimits::default(),
        7,
        &CancellationToken::new(),
    );
    assert!(matches!(
        result,
        Err(PowerError::IntegrityCheckFailed { .. })
    ));
    let mut wrong_size = descriptor.clone();
    wrong_size.bytes += 1;
    assert!(VerifiedRawWeightFile::open(
        root.path(),
        wrong_size,
        &InferenceLimits::default(),
        7,
        &CancellationToken::new()
    )
    .is_err());
    std::fs::remove_file(root.path().join("weights.bin")).unwrap();
    std::fs::create_dir(root.path().join("weights.bin")).unwrap();
    assert!(VerifiedRawWeightFile::open(
        root.path(),
        descriptor,
        &InferenceLimits::default(),
        7,
        &CancellationToken::new()
    )
    .is_err());
}

#[test]
fn model_index_scratch_and_output_share_explicit_limits() {
    let (root, descriptor) = fixture(b"weights");
    let limits = InferenceLimits {
        max_model_bytes: 6,
        ..InferenceLimits::default()
    };
    assert!(VerifiedRawWeightFile::open(
        root.path(),
        descriptor.clone(),
        &limits,
        7,
        &CancellationToken::new()
    )
    .is_err());
    // One 32-byte digest + 7-byte scratch + 7-byte output requires 46 bytes.
    let limits = InferenceLimits {
        max_state_bytes: 45,
        ..InferenceLimits::default()
    };
    assert!(VerifiedRawWeightFile::open(
        root.path(),
        descriptor.clone(),
        &limits,
        7,
        &CancellationToken::new()
    )
    .is_err());
    let limits = InferenceLimits {
        max_state_bytes: 46,
        ..InferenceLimits::default()
    };
    assert!(VerifiedRawWeightFile::open(
        root.path(),
        descriptor.clone(),
        &limits,
        7,
        &CancellationToken::new()
    )
    .is_ok());
    for budget in [0, u64::MAX] {
        assert!(VerifiedRawWeightFile::open(
            root.path(),
            descriptor.clone(),
            &InferenceLimits::default(),
            budget,
            &CancellationToken::new()
        )
        .is_err());
    }
}

#[test]
fn empty_overflowing_and_over_budget_ranges_are_refused() {
    let (root, descriptor) = fixture(b"weights");
    let file = open(root.path(), descriptor, 4);
    for (offset, bytes) in [(0, 0), (0, 5), (6, 2), (7, 1), (u64::MAX, 2)] {
        assert!(file
            .read_range(offset, bytes, &CancellationToken::new())
            .is_err());
    }
}

#[test]
fn cancellation_applies_to_full_verification_and_range_reads() {
    let (root, descriptor) = fixture(b"weights");
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert!(matches!(
        VerifiedRawWeightFile::open(
            root.path(),
            descriptor.clone(),
            &InferenceLimits::default(),
            7,
            &cancellation
        ),
        Err(PowerError::InferenceCancelled)
    ));
    let file = open(root.path(), descriptor, 7);
    assert!(matches!(
        file.read_range(0, 1, &cancellation),
        Err(PowerError::InferenceCancelled)
    ));
}

#[test]
fn same_size_in_place_changes_are_authenticated_per_touched_chunk() {
    let raw = vec![7; RANGE_READ_CHUNK_BYTES + 113];
    let (root, descriptor) = fixture(&raw);
    let file = open(root.path(), descriptor, 128);
    let mut writer = std::fs::OpenOptions::new()
        .write(true)
        .open(root.path().join("weights.bin"))
        .unwrap();
    writer
        .seek(SeekFrom::Start(RANGE_READ_CHUNK_BYTES as u64 + 80))
        .unwrap();
    writer.write_all(&[9]).unwrap();
    writer.flush().unwrap();
    assert!(file
        .read_range(RANGE_READ_CHUNK_BYTES as u64, 1, &CancellationToken::new())
        .is_err());
    assert_eq!(
        file.read_range(0, 1, &CancellationToken::new())
            .unwrap()
            .bytes(),
        &[7]
    );
}

#[test]
fn truncation_or_growth_never_returns_a_range() {
    for bytes in [6, 8] {
        let (root, descriptor) = fixture(b"weights");
        let file = open(root.path(), descriptor, 7);
        let writer = std::fs::OpenOptions::new()
            .write(true)
            .open(root.path().join("weights.bin"))
            .unwrap();
        writer.set_len(bytes).unwrap();
        assert!(file.read_range(0, 1, &CancellationToken::new()).is_err());
    }
}

#[test]
fn replacing_a_path_cannot_redirect_the_verified_handle() {
    let (root, descriptor) = fixture(b"weights");
    let file = open(root.path(), descriptor, 7);
    std::fs::rename(root.path().join("weights.bin"), root.path().join("old.bin")).unwrap();
    std::fs::write(root.path().join("weights.bin"), b"changed").unwrap();
    assert_eq!(
        file.read_range(0, 7, &CancellationToken::new())
            .unwrap()
            .bytes(),
        b"weights"
    );
}

#[test]
fn debug_output_does_not_expose_payload_or_paths() {
    let (root, descriptor) = fixture(b"sensitive-payload");
    let file = open(root.path(), descriptor, 17);
    let read = file.read_range(0, 17, &CancellationToken::new()).unwrap();
    let debug = format!("{file:?} {read:?}");
    assert!(!debug.contains("sensitive"));
    assert!(!debug.contains("weights.bin"));
    assert!(!debug.contains(&root.path().display().to_string()));
}

#[cfg(unix)]
#[test]
fn leaf_and_directory_symlinks_are_rejected() {
    use std::os::unix::fs::symlink;

    let (root, descriptor) = fixture(b"weights");
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("weights.bin"), b"weights").unwrap();
    std::fs::remove_file(root.path().join("weights.bin")).unwrap();
    symlink(
        outside.path().join("weights.bin"),
        root.path().join("weights.bin"),
    )
    .unwrap();
    assert!(VerifiedRawWeightFile::open(
        root.path(),
        descriptor.clone(),
        &InferenceLimits::default(),
        7,
        &CancellationToken::new()
    )
    .is_err());
    symlink(outside.path(), root.path().join("nested")).unwrap();
    let mut nested = descriptor;
    nested.relative_path = "nested/weights.bin".into();
    assert!(VerifiedRawWeightFile::open(
        root.path(),
        nested,
        &InferenceLimits::default(),
        7,
        &CancellationToken::new()
    )
    .is_err());
}
