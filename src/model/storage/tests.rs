use super::*;
use serial_test::serial;

#[test]
fn test_compute_sha256() {
    let hash = compute_sha256(b"hello world");
    assert_eq!(
        hash,
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    );
}

#[test]
#[serial]
fn test_store_blob() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let data = b"test model data";
    let (path, hash) = store_blob(data).unwrap();

    assert!(path.exists());
    assert!(path.to_string_lossy().contains(&format!("sha256-{hash}")));

    let stored = std::fs::read(&path).unwrap();
    assert_eq!(stored, data);

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
#[serial]
fn test_store_blob_deduplication() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let data = b"identical data";
    let (path1, hash1) = store_blob(data).unwrap();
    let (path2, hash2) = store_blob(data).unwrap();

    assert_eq!(path1, path2);
    assert_eq!(hash1, hash2);

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
#[serial]
fn test_verify_blob() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let data = b"verify me";
    let (path, hash) = store_blob(data).unwrap();

    assert!(verify_blob(&path, &hash).unwrap());
    assert!(!verify_blob(&path, "wrong-hash").unwrap());

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
fn test_delete_blob() {
    let dir = tempfile::tempdir().unwrap();

    // Write a blob file directly to avoid env var races through store_blob
    let blob_path = dir.path().join("blob-to-delete");
    std::fs::write(&blob_path, b"to be deleted").unwrap();
    assert!(blob_path.exists());

    let manifest = crate::model::manifest::ModelManifest {
        name: "test".to_string(),
        format: crate::model::manifest::ModelFormat::Gguf,
        size: 13,
        sha256: "test".to_string(),
        parameters: None,
        created_at: chrono::Utc::now(),
        path: blob_path.clone(),
        system_prompt: None,
        template_override: None,
        default_parameters: None,
        modelfile_content: None,
        license: None,
        adapter_path: None,
        adapter_artifact: None,
        external_draft: None,
        projector_path: None,
        projector_artifact: None,
        messages: vec![],
        family: None,
        families: None,
    };

    delete_blob(&manifest).unwrap();
    assert!(!blob_path.exists());
}

#[test]
#[serial]
fn test_delete_blob_nonexistent_path() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    // Manifest pointing to a nonexistent file — should succeed (no-op)
    let manifest = crate::model::manifest::ModelManifest {
        name: "ghost".to_string(),
        format: crate::model::manifest::ModelFormat::Gguf,
        size: 0,
        sha256: "none".to_string(),
        parameters: None,
        created_at: chrono::Utc::now(),
        path: std::path::PathBuf::from("/tmp/nonexistent-blob-file"),
        system_prompt: None,
        template_override: None,
        default_parameters: None,
        modelfile_content: None,
        license: None,
        adapter_path: None,
        adapter_artifact: None,
        external_draft: None,
        projector_path: None,
        projector_artifact: None,
        messages: vec![],
        family: None,
        families: None,
    };

    // Should not error — file doesn't exist, so nothing to delete
    delete_blob(&manifest).unwrap();

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
fn test_compute_sha256_empty() {
    let hash = compute_sha256(b"");
    assert_eq!(
        hash,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn test_compute_sha256_directory_is_order_stable() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();

    std::fs::create_dir(first.path().join("nested")).unwrap();
    std::fs::write(first.path().join("config.json"), br#"{"model":"a"}"#).unwrap();
    std::fs::write(
        first.path().join("nested").join("weights.safetensors"),
        b"weights",
    )
    .unwrap();

    std::fs::create_dir(second.path().join("nested")).unwrap();
    std::fs::write(
        second.path().join("nested").join("weights.safetensors"),
        b"weights",
    )
    .unwrap();
    std::fs::write(second.path().join("config.json"), br#"{"model":"a"}"#).unwrap();

    assert_eq!(
        compute_sha256_directory(first.path()).unwrap(),
        compute_sha256_directory(second.path()).unwrap()
    );
}

#[test]
fn test_compute_sha256_directory_changes_when_file_changes() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("model.safetensors");
    std::fs::write(&model_path, b"weights-v1").unwrap();
    let first = compute_sha256_directory(dir.path()).unwrap();

    std::fs::write(&model_path, b"weights-v2").unwrap();
    let second = compute_sha256_directory(dir.path()).unwrap();

    assert_ne!(first, second);
}

#[test]
#[serial]
fn test_verify_blob_nonexistent_file() {
    let result = verify_blob(std::path::Path::new("/tmp/nonexistent-verify-test"), "abc");
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_prune_unused_blobs_removes_orphans() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    // Store two blobs
    let (path_a, _) = store_blob(b"model-a-data").unwrap();
    let (_, _) = store_blob(b"orphan-data").unwrap();

    // Only reference path_a in manifests
    let manifest = crate::model::manifest::ModelManifest {
        name: "model-a".to_string(),
        format: crate::model::manifest::ModelFormat::Gguf,
        size: 12,
        sha256: "test".to_string(),
        parameters: None,
        created_at: chrono::Utc::now(),
        path: path_a.clone(),
        system_prompt: None,
        template_override: None,
        default_parameters: None,
        modelfile_content: None,
        license: None,
        adapter_path: None,
        adapter_artifact: None,
        external_draft: None,
        projector_path: None,
        projector_artifact: None,
        messages: vec![],
        family: None,
        families: None,
    };

    let (removed, freed) = prune_unused_blobs(&[manifest]).unwrap();
    assert_eq!(removed, 1);
    assert!(freed > 0);
    // Referenced blob should still exist
    assert!(path_a.exists());

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
#[serial]
fn test_prune_unused_blobs_no_orphans() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let (path_a, _) = store_blob(b"data-a").unwrap();

    let manifest = crate::model::manifest::ModelManifest {
        name: "a".to_string(),
        format: crate::model::manifest::ModelFormat::Gguf,
        size: 6,
        sha256: "test".to_string(),
        parameters: None,
        created_at: chrono::Utc::now(),
        path: path_a,
        system_prompt: None,
        template_override: None,
        default_parameters: None,
        modelfile_content: None,
        license: None,
        adapter_path: None,
        adapter_artifact: None,
        external_draft: None,
        projector_path: None,
        projector_artifact: None,
        messages: vec![],
        family: None,
        families: None,
    };

    let (removed, freed) = prune_unused_blobs(&[manifest]).unwrap();
    assert_eq!(removed, 0);
    assert_eq!(freed, 0);

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
#[serial]
fn test_prune_unused_blobs_preserves_external_draft_artifact() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let (target_path, target_sha256) = store_blob(b"target-model").unwrap();
    let (draft_path, draft_sha256) = store_blob(b"external-draft").unwrap();
    let draft_size = std::fs::metadata(&draft_path).unwrap().len();
    let manifest = crate::model::manifest::ModelManifest {
        name: "target".to_string(),
        format: crate::model::manifest::ModelFormat::Gguf,
        size: std::fs::metadata(&target_path).unwrap().len(),
        sha256: target_sha256.clone(),
        parameters: None,
        created_at: chrono::Utc::now(),
        path: target_path,
        system_prompt: None,
        template_override: None,
        default_parameters: None,
        modelfile_content: None,
        license: None,
        adapter_path: None,
        adapter_artifact: None,
        external_draft: Some(crate::model::manifest::ExternalDraftArtifact {
            kind: crate::model::manifest::ExternalDraftKind::Dspark,
            path: draft_path.clone(),
            size: draft_size,
            sha256: draft_sha256,
            target_sha256,
            source: None,
            revision: None,
            license: None,
        }),
        projector_path: None,
        projector_artifact: None,
        messages: vec![],
        family: None,
        families: None,
    };

    let (removed, freed) = prune_unused_blobs(&[manifest]).unwrap();
    assert_eq!((removed, freed), (0, 0));
    assert!(draft_path.exists());

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
#[serial]
fn test_prune_unused_blobs_empty_dir() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    // Create blobs dir but leave it empty
    std::fs::create_dir_all(dirs::blobs_dir()).unwrap();

    let (removed, freed) = prune_unused_blobs(&[]).unwrap();
    assert_eq!(removed, 0);
    assert_eq!(freed, 0);

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
fn test_prune_unused_blobs_nonexistent_dir() {
    // When blobs dir doesn't exist, should return (0, 0)
    let _dir = tempfile::tempdir().unwrap();
    // prune_unused_blobs checks dirs::blobs_dir() which may or may not exist;
    // the function handles missing dirs gracefully by returning (0, 0).
    let result = prune_unused_blobs(&[]);
    assert!(result.is_ok());
}

#[test]
fn test_blob_file_size_reports_metadata_errors() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing-blob");

    let err = blob_file_size(&missing).unwrap_err();

    assert!(
        err.to_string().contains("Failed to inspect blob"),
        "error: {err}"
    );
    assert!(err.to_string().contains("missing-blob"), "error: {err}");
}

#[test]
fn test_compute_sha256_file() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("test.bin");
    std::fs::write(&file_path, b"hello world").unwrap();

    let hash = compute_sha256_file(&file_path).unwrap();
    let expected = compute_sha256(b"hello world");
    assert_eq!(hash, expected);
}

#[test]
fn test_compute_sha256_file_nonexistent() {
    let result = compute_sha256_file(std::path::Path::new("/nonexistent/file.bin"));
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_store_blob_from_path() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let source_dir = tempfile::tempdir().unwrap();
    let source_path = source_dir.path().join("model.gguf");
    std::fs::write(&source_path, b"fake gguf data").unwrap();

    let (blob_path, _hash) = store_blob_from_path(&source_path).unwrap();
    assert!(blob_path.exists());

    // Verify content matches
    let stored = std::fs::read(&blob_path).unwrap();
    assert_eq!(stored, b"fake gguf data");

    // Verify blob name contains sha256
    let filename = blob_path.file_name().unwrap().to_str().unwrap();
    assert!(filename.starts_with("sha256-"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
#[serial]
fn test_store_blob_from_path_dedup() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let source_dir = tempfile::tempdir().unwrap();
    let source_path = source_dir.path().join("model.gguf");
    std::fs::write(&source_path, b"same content").unwrap();

    let (path1, _) = store_blob_from_path(&source_path).unwrap();
    let (path2, _) = store_blob_from_path(&source_path).unwrap();
    assert_eq!(path1, path2);

    std::env::remove_var("A3S_POWER_HOME");
}

// ========================================================================
// store_blob_from_temp integration tests
// ========================================================================

#[test]
#[serial]
fn test_store_blob_from_temp_moves_file() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let source_dir = tempfile::tempdir().unwrap();
    let source_path = source_dir.path().join("partial-abc123");
    std::fs::write(&source_path, b"large model data").unwrap();
    assert!(source_path.exists());

    let (blob_path, hash) = store_blob_from_temp(&source_path).unwrap();

    // Blob should exist with correct content
    assert!(blob_path.exists());
    let stored = std::fs::read(&blob_path).unwrap();
    assert_eq!(stored, b"large model data");

    // Hash should be valid hex
    assert!(!hash.is_empty());
    let filename = blob_path.file_name().unwrap().to_str().unwrap();
    assert_eq!(filename, format!("sha256-{hash}"));

    // Source temp file should be gone (renamed or deleted)
    assert!(!source_path.exists());

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
#[serial]
fn test_store_blob_from_temp_dedup_cleans_source() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    // First: store via normal path to create the blob
    let source_dir = tempfile::tempdir().unwrap();
    let source1 = source_dir.path().join("original.bin");
    std::fs::write(&source1, b"dedup content").unwrap();
    let (blob_path, _) = store_blob_from_path(&source1).unwrap();
    assert!(blob_path.exists());

    // Second: store_blob_from_temp with same content — blob already exists
    let source2 = source_dir.path().join("partial-duplicate");
    std::fs::write(&source2, b"dedup content").unwrap();
    let (blob_path2, hash2) = store_blob_from_temp(&source2).unwrap();

    // Should return same blob path
    assert_eq!(blob_path, blob_path2);
    assert!(!hash2.is_empty());

    // Temp source should be cleaned up even though blob already existed
    assert!(!source2.exists());

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
#[serial]
fn test_store_blob_from_temp_same_dir_uses_rename() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    // Create blobs dir and put the temp file directly in it (same filesystem)
    let blobs_dir = dirs::blobs_dir();
    std::fs::create_dir_all(&blobs_dir).unwrap();
    let source_path = blobs_dir.join("partial-inplace");
    std::fs::write(&source_path, b"rename me").unwrap();

    let (blob_path, hash) = store_blob_from_temp(&source_path).unwrap();

    assert!(blob_path.exists());
    assert!(!source_path.exists()); // renamed away
    let stored = std::fs::read(&blob_path).unwrap();
    assert_eq!(stored, b"rename me");
    assert!(blob_path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("sha256-"));
    assert!(!hash.is_empty());

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
#[serial]
fn test_store_blob_from_path_preserves_source() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let source_dir = tempfile::tempdir().unwrap();
    let source_path = source_dir.path().join("user-model.gguf");
    std::fs::write(&source_path, b"user data").unwrap();

    let (blob_path, hash) = store_blob_from_path(&source_path).unwrap();

    // Blob should exist
    assert!(blob_path.exists());
    assert!(!hash.is_empty());

    // Source file should still exist (not moved/deleted)
    assert!(source_path.exists());
    let original = std::fs::read(&source_path).unwrap();
    assert_eq!(original, b"user data");

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
#[serial]
fn test_store_blob_from_path_returns_correct_hash() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let source_dir = tempfile::tempdir().unwrap();
    let source_path = source_dir.path().join("hashtest.bin");
    std::fs::write(&source_path, b"hash me").unwrap();

    let (blob_path, hash) = store_blob_from_path(&source_path).unwrap();

    // Hash from store_blob_from_path should match compute_sha256
    let expected_hash = compute_sha256(b"hash me");
    assert_eq!(hash, expected_hash);
    assert_eq!(
        blob_path.file_name().unwrap().to_str().unwrap(),
        format!("sha256-{expected_hash}")
    );

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
#[serial]
fn test_store_blob_from_temp_nonexistent_source_errors() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let result = store_blob_from_temp(std::path::Path::new("/nonexistent/partial-xyz"));
    assert!(result.is_err());

    std::env::remove_var("A3S_POWER_HOME");
}
