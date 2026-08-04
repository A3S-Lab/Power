use std::path::Path;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

use super::*;

fn model(root: &Path) {
    let first = [1_u8; 16];
    let second = [2_u8; 32];
    serialize_to_file(
        [
            (
                "layer.0.weight",
                TensorView::new(Dtype::F32, vec![4], &first).unwrap(),
            ),
            (
                "layer.1.weight",
                TensorView::new(Dtype::F32, vec![8], &second).unwrap(),
            ),
        ],
        None,
        &root.join("model.safetensors"),
    )
    .unwrap();
}

fn config(root: &Path, strategy: WeightReadStrategy) -> StorageBenchmarkConfig {
    StorageBenchmarkConfig {
        weights: WeightStoreConfig::new(root).with_primary_read_strategy(strategy),
        power_commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
        filesystem_class: "test-filesystem".to_string(),
        device_class: "test-device".to_string(),
        cpu_model: "test-cpu".to_string(),
        ram_bytes: 1024,
        cache_state: StorageCacheState::Warm,
        cache_preparation: StorageCachePreparation::WarmSequence,
        concurrency: 2,
        samples: 2,
        max_tensors: 16,
    }
}

#[test]
fn mmap_and_positional_reports_have_exact_output_parity_without_paths() {
    let root = tempfile::tempdir().unwrap();
    model(root.path());
    let mmap = run_storage_benchmark(
        &config(root.path(), WeightReadStrategy::Mmap),
        &InferenceLimits::default(),
    )
    .unwrap();
    let positional = run_storage_benchmark(
        &config(root.path(), WeightReadStrategy::PositionalBuffered),
        &InferenceLimits::default(),
    )
    .unwrap();

    assert_eq!(mmap.sequence_sha256, positional.sequence_sha256);
    assert_eq!(mmap.output_sha256, positional.output_sha256);
    assert_eq!(mmap.total_requested_bytes, mmap.total_read_bytes);
    assert_eq!(
        positional.total_requested_bytes,
        positional.total_read_bytes
    );
    let report = serde_json::to_string(&positional).unwrap();
    assert!(!report.contains(root.path().to_string_lossy().as_ref()));
    assert!(!report.contains("layer.0.weight"));
}

#[test]
fn sharded_primary_reports_root_count_without_exposing_paths() {
    let primary = tempfile::tempdir().unwrap();
    let shard = tempfile::tempdir().unwrap();
    let first = [1_u8; 16];
    let second = [2_u8; 32];
    serialize_to_file(
        [(
            "layer.0.weight",
            TensorView::new(Dtype::F32, vec![4], &first).unwrap(),
        )],
        None,
        &primary.path().join("first.safetensors"),
    )
    .unwrap();
    serialize_to_file(
        [(
            "layer.1.weight",
            TensorView::new(Dtype::F32, vec![8], &second).unwrap(),
        )],
        None,
        &shard.path().join("second.safetensors"),
    )
    .unwrap();
    let mut benchmark = config(primary.path(), WeightReadStrategy::PositionalBuffered);
    benchmark.weights = benchmark.weights.with_primary_shard_root(shard.path());

    let report = run_storage_benchmark(&benchmark, &InferenceLimits::default()).unwrap();

    assert_eq!(report.sources[0].root_count, 2);
    assert_eq!(report.sources[0].verified_files, 2);
    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains("\"rootCount\":2"));
    assert!(!json.contains(primary.path().to_string_lossy().as_ref()));
    assert!(!json.contains(shard.path().to_string_lossy().as_ref()));
}

#[test]
fn cold_labels_require_one_runner_verified_sample() {
    let root = tempfile::tempdir().unwrap();
    model(root.path());
    let mut benchmark = config(root.path(), WeightReadStrategy::Mmap);
    benchmark.cache_state = StorageCacheState::Cold;
    benchmark.cache_preparation = StorageCachePreparation::WarmSequence;
    assert!(benchmark.validate().is_err());
    benchmark.cache_preparation = StorageCachePreparation::LinuxFadviseDontNeed;
    benchmark.samples = 2;
    assert!(benchmark.validate().is_err());
    benchmark.samples = 1;
    if cfg!(target_os = "linux") {
        assert!(benchmark.validate().is_ok());
    } else {
        assert!(matches!(
            benchmark.validate(),
            Err(PowerError::BackendNotAvailable(_))
        ));
    }
}

#[cfg(target_os = "linux")]
#[test]
fn linux_cold_run_discards_and_verifies_requested_pages_after_integrity_open() {
    let root = tempfile::tempdir().unwrap();
    model(root.path());
    let mut benchmark = config(root.path(), WeightReadStrategy::PositionalBuffered);
    benchmark.cache_state = StorageCacheState::Cold;
    benchmark.cache_preparation = StorageCachePreparation::LinuxFadviseDontNeed;
    benchmark.samples = 1;

    let report = run_storage_benchmark(&benchmark, &InferenceLimits::default()).unwrap();
    assert!(report.cache_state_verified);
    assert_eq!(
        report.cache_preparation,
        StorageCachePreparation::LinuxFadviseDontNeed
    );
    assert_eq!(report.samples.len(), 1);
}

#[test]
fn debug_output_redacts_model_roots() {
    let root = tempfile::tempdir().unwrap();
    let benchmark = config(root.path(), WeightReadStrategy::Mmap);
    assert!(!format!("{benchmark:?}").contains(root.path().to_string_lossy().as_ref()));
}
