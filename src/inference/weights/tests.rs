use std::path::{Path, PathBuf};

use candle_core::Device;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tokio_util::sync::CancellationToken;

use super::*;

fn write_weights(root: &Path, byte: u8) {
    write_weight_file(root, "model.safetensors", "layer.0.weight", 0, 128, byte);
}

fn write_weight_file(
    root: &Path,
    file_name: &str,
    tensor_prefix: &str,
    first: usize,
    count: usize,
    byte: u8,
) {
    let values = [byte; 4];
    let tensors = (first..first.saturating_add(count))
        .map(|index| {
            (
                format!("{tensor_prefix}.{index}"),
                TensorView::new(Dtype::F32, vec![1], values.as_slice()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(tensors, None, &root.join(file_name)).unwrap();
}

fn write_large_weight(root: &Path, file_name: &str, tensor_name: &str, bytes: usize) {
    assert_eq!(bytes % 4, 0);
    let values = (0..bytes)
        .map(|index| u8::try_from(index % 251).unwrap())
        .collect::<Vec<_>>();
    let tensor = TensorView::new(Dtype::F32, vec![bytes / 4], values.as_slice()).unwrap();
    serialize_to_file([(tensor_name, tensor)], None, &root.join(file_name)).unwrap();
}

#[test]
fn positional_index_and_exact_bytes_match_the_mmap_default() {
    let root = tempfile::tempdir().unwrap();
    write_weight_file(root.path(), "model.safetensors", "layer.weight", 0, 8, 7);
    let mmap = WeightStore::open(root.path(), &InferenceLimits::default()).unwrap();
    let positional = WeightStore::open_config(
        &WeightStoreConfig::new(root.path())
            .with_primary_read_strategy(WeightReadStrategy::PositionalBuffered),
        &InferenceLimits::default(),
    )
    .unwrap();
    let name = "layer.weight.3";

    assert!(mmap.tensors.is_some());
    assert!(mmap.readers.is_empty());
    assert!(positional.tensors.is_none());
    assert_eq!(positional.readers.len(), 1);

    let storage = positional.storage_descriptor(name).unwrap();
    assert_eq!(storage.file_index, 0);
    assert!(storage.absolute_offset >= 8);
    assert_eq!(storage.bytes, 4);
    assert_eq!(storage.dtype, "f32");
    assert_eq!(storage.shape, [1]);

    let mmap_read = mmap.read_tensor_bytes(name).unwrap();
    let positional_read = positional.read_tensor_bytes(name).unwrap();
    assert_eq!(mmap_read.strategy(), WeightReadStrategy::Mmap);
    assert_eq!(
        positional_read.strategy(),
        WeightReadStrategy::PositionalBuffered
    );
    assert_eq!(positional_read.storage(), &storage);
    assert_eq!(mmap_read.bytes(), positional_read.bytes());

    let mmap_tensor = mmap.load(name, &Device::Cpu).unwrap();
    let positional_tensor = positional.load(name, &Device::Cpu).unwrap();
    assert_eq!(
        mmap_tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        positional_tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    );
}

#[test]
fn verified_tensor_ranges_match_for_mmap_and_positional_sources() {
    let root = tempfile::tempdir().unwrap();
    write_large_weight(root.path(), "model.safetensors", "range.weight", 64);
    let mmap = WeightStore::open(root.path(), &InferenceLimits::default()).unwrap();
    let positional = WeightStore::open_config(
        &WeightStoreConfig::new(root.path())
            .with_primary_read_strategy(WeightReadStrategy::PositionalBuffered),
        &InferenceLimits::default(),
    )
    .unwrap();

    let expected = mmap.read_tensor_bytes("range.weight").unwrap();
    let mmap_range = mmap.read_tensor_range("range.weight", 7, 19).unwrap();
    let positional_range = positional.read_tensor_range("range.weight", 7, 19).unwrap();

    assert_eq!(mmap_range.bytes(), &expected.bytes()[7..26]);
    assert_eq!(positional_range.bytes(), mmap_range.bytes());
    assert_eq!(mmap_range.tensor_offset(), 7);
    assert_eq!(mmap_range.storage().bytes, 64);
    assert_eq!(mmap_range.strategy(), WeightReadStrategy::Mmap);
    assert_eq!(
        positional_range.strategy(),
        WeightReadStrategy::PositionalBuffered
    );
    assert_eq!(mmap_range.source_index(), 0);
    assert!(!mmap_range.fell_back());
}

#[test]
fn verified_tensor_ranges_reject_empty_overflowing_and_cancelled_reads() {
    let root = tempfile::tempdir().unwrap();
    write_large_weight(root.path(), "model.safetensors", "range.weight", 64);
    let store = WeightStore::open(root.path(), &InferenceLimits::default()).unwrap();

    assert!(matches!(
        store.read_tensor_range("range.weight", 0, 0),
        Err(PowerError::InvalidRequest(_))
    ));
    assert!(matches!(
        store.read_tensor_range("range.weight", 63, 2),
        Err(PowerError::InvalidRequest(_))
    ));
    assert!(matches!(
        store.read_tensor_range("range.weight", u64::MAX, 2),
        Err(PowerError::InvalidRequest(_))
    ));

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert!(matches!(
        store.read_tensor_range_with_cancellation("range.weight", 0, 1, &cancelled),
        Err(PowerError::InferenceFailed(_))
    ));
}

#[test]
fn positional_reads_are_chunked_cancellable_and_fail_on_truncation() {
    let root = tempfile::tempdir().unwrap();
    let tensor_bytes = range_io::RANGE_READ_CHUNK_BYTES * 2 + 4096;
    write_large_weight(
        root.path(),
        "large.safetensors",
        "large.weight",
        tensor_bytes,
    );
    let store = WeightStore::open_config(
        &WeightStoreConfig::new(root.path())
            .with_primary_read_strategy(WeightReadStrategy::PositionalBuffered),
        &InferenceLimits::default(),
    )
    .unwrap();

    assert_eq!(
        store
            .read_tensor_bytes("large.weight")
            .unwrap()
            .bytes()
            .len(),
        tensor_bytes
    );
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert!(matches!(
        store.read_tensor_bytes_with_cancellation("large.weight", &cancelled),
        Err(PowerError::InferenceFailed(_))
    ));

    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(root.path().join("large.safetensors"))
        .unwrap();
    file.set_len(store.files()[0].bytes - 1).unwrap();
    assert!(matches!(
        store.read_tensor_bytes("large.weight"),
        Err(PowerError::InvalidFormat(_))
    ));
}

#[test]
fn positional_replica_failure_uses_the_existing_primary_fallback() {
    let primary = tempfile::tempdir().unwrap();
    let replica = tempfile::tempdir().unwrap();
    write_weights(primary.path(), 9);
    std::fs::copy(
        primary.path().join("model.safetensors"),
        replica.path().join("model.safetensors"),
    )
    .unwrap();
    let source = |root: &Path| {
        WeightSourceConfig::new(root)
            .with_read_weight(127)
            .with_read_strategy(WeightReadStrategy::PositionalBuffered)
    };
    let config = WeightStoreConfig::new(primary.path())
        .with_primary_read_strategy(WeightReadStrategy::PositionalBuffered)
        .with_primary_read_weight(1)
        .with_replica(source(replica.path()));
    let store = WeightStore::open_config(&config, &InferenceLimits::default()).unwrap();
    let selected = store
        .inventory()
        .map(|descriptor| descriptor.name.as_str())
        .find(|name| store.select_source(name) == 1)
        .unwrap()
        .to_string();

    let replica_file = std::fs::OpenOptions::new()
        .write(true)
        .open(replica.path().join("model.safetensors"))
        .unwrap();
    replica_file.set_len(8).unwrap();
    let read = store.read_tensor_bytes(&selected).unwrap();
    assert_eq!(read.source_index(), 0);
    assert!(read.fell_back());
    assert_eq!(read.strategy(), WeightReadStrategy::PositionalBuffered);
    assert_eq!(read.bytes(), [9_u8; 4]);

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert!(matches!(
        store.read_tensor_bytes_with_cancellation(&selected, &cancelled),
        Err(PowerError::InferenceFailed(_))
    ));
}

#[test]
fn direct_reads_are_exact_or_explicitly_unsupported() {
    let root = tempfile::tempdir().unwrap();
    write_large_weight(root.path(), "model.safetensors", "direct.weight", 16 * 1024);
    let expected = WeightStore::open(root.path(), &InferenceLimits::default())
        .unwrap()
        .read_tensor_bytes("direct.weight")
        .unwrap()
        .bytes()
        .to_vec();
    let direct = WeightStore::open_config(
        &WeightStoreConfig::new(root.path())
            .with_primary_read_strategy(WeightReadStrategy::PositionalDirect),
        &InferenceLimits::default(),
    );
    match direct {
        Ok(store) => match store.read_tensor_bytes("direct.weight") {
            Ok(read) => {
                assert_eq!(read.strategy(), WeightReadStrategy::PositionalDirect);
                assert_eq!(read.bytes(), expected);
            }
            Err(PowerError::BackendNotAvailable(_)) => {}
            Err(error) => {
                panic!("direct read failed without an explicit unsupported result: {error}")
            }
        },
        Err(PowerError::BackendNotAvailable(_)) => {}
        Err(error) => panic!("direct open failed without an explicit unsupported result: {error}"),
    }
}

#[test]
fn cache_bypass_reads_are_exact_or_explicitly_unsupported() {
    let root = tempfile::tempdir().unwrap();
    write_large_weight(
        root.path(),
        "model.safetensors",
        "cache-bypass.weight",
        16 * 1024,
    );
    let expected = WeightStore::open(root.path(), &InferenceLimits::default())
        .unwrap()
        .read_tensor_bytes("cache-bypass.weight")
        .unwrap()
        .bytes()
        .to_vec();
    let cache_bypass = WeightStore::open_config(
        &WeightStoreConfig::new(root.path())
            .with_primary_read_strategy(WeightReadStrategy::PositionalCacheBypass),
        &InferenceLimits::default(),
    );
    match cache_bypass {
        Ok(store) => {
            let read = store.read_tensor_bytes("cache-bypass.weight").unwrap();
            assert_eq!(read.strategy(), WeightReadStrategy::PositionalCacheBypass);
            assert_eq!(read.bytes(), expected);
        }
        Err(PowerError::BackendNotAvailable(message)) => {
            #[cfg(target_os = "macos")]
            panic!("cache-bypass is expected to be available on macOS: {message}");
            #[cfg(not(target_os = "macos"))]
            assert!(!message.is_empty());
        }
        Err(error) => {
            panic!("cache-bypass open failed without an explicit unsupported result: {error}")
        }
    }
}

#[test]
fn weighted_replicas_are_exact_and_deterministic() {
    let primary = tempfile::tempdir().unwrap();
    let replica = tempfile::tempdir().unwrap();
    write_weights(primary.path(), 1);
    std::fs::copy(
        primary.path().join("model.safetensors"),
        replica.path().join("model.safetensors"),
    )
    .unwrap();
    let config = WeightStoreConfig::new(primary.path())
        .with_primary_read_weight(1)
        .with_replica(WeightSourceConfig::new(replica.path()).with_read_weight(3));

    let store = WeightStore::open_config(&config, &InferenceLimits::default()).unwrap();
    assert_eq!(store.sources().len(), 2);
    assert_eq!(store.sources()[1].role, WeightSourceRole::Replica);
    assert_eq!(store.sources()[1].coverage, WeightSourceCoverage::Complete);
    assert_eq!(store.sources()[0].read_weight, 1);
    assert_eq!(store.sources()[1].read_weight, 3);
    assert_eq!(
        store.sources()[1].source_weighting,
        WeightSourceWeighting::Configured
    );
    let selected = store
        .inventory()
        .map(|descriptor| store.select_source(&descriptor.name))
        .collect::<Vec<_>>();
    assert!(selected.contains(&0));
    assert!(selected.contains(&1));
    for descriptor in store.inventory() {
        let first = store.select_source(&descriptor.name);
        let second = store.select_source(&descriptor.name);
        assert_eq!(first, second);
        let loaded = store.load_tracked(&descriptor.name, &Device::Cpu).unwrap();
        assert_eq!(loaded.source_index, first);
        assert!(!loaded.fell_back);
    }
}

#[test]
fn sharded_primary_matches_one_root_identity_and_loads_every_tensor() {
    let combined = tempfile::tempdir().unwrap();
    let primary = tempfile::tempdir().unwrap();
    let shard = tempfile::tempdir().unwrap();
    write_weight_file(
        combined.path(),
        "first.safetensors",
        "layer.0.weight",
        0,
        8,
        1,
    );
    write_weight_file(
        combined.path(),
        "second.safetensors",
        "layer.1.weight",
        0,
        8,
        2,
    );
    std::fs::copy(
        combined.path().join("first.safetensors"),
        primary.path().join("first.safetensors"),
    )
    .unwrap();
    std::fs::copy(
        combined.path().join("second.safetensors"),
        shard.path().join("second.safetensors"),
    )
    .unwrap();

    let expected = WeightStore::open(combined.path(), &InferenceLimits::default()).unwrap();
    let split = WeightStore::open_config(
        &WeightStoreConfig::new(primary.path()).with_primary_shard_root(shard.path()),
        &InferenceLimits::default(),
    )
    .unwrap();
    let primary_root = std::fs::canonicalize(primary.path()).unwrap();
    let shard_root = std::fs::canonicalize(shard.path()).unwrap();

    assert_eq!(split.sha256(), expected.sha256());
    assert_eq!(split.files(), expected.files());
    assert_eq!(
        split.roots(),
        [primary_root.as_path(), shard_root.as_path()]
    );
    assert_eq!(split.sources()[0].shard_roots, [shard_root.as_path()]);
    assert!(split.paths()[0].starts_with(&primary_root));
    assert!(split.paths()[1].starts_with(&shard_root));
    for name in ["layer.0.weight.3", "layer.1.weight.5"] {
        let expected = expected
            .load(name, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = split
            .load(name, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn sharded_collections_reject_ambiguous_or_overlapping_roots() {
    let primary = tempfile::tempdir().unwrap();
    let duplicate_name = tempfile::tempdir().unwrap();
    let empty = tempfile::tempdir().unwrap();
    write_weight_file(
        primary.path(),
        "model.safetensors",
        "layer.0.weight",
        0,
        1,
        1,
    );
    write_weight_file(
        duplicate_name.path(),
        "model.safetensors",
        "layer.1.weight",
        0,
        1,
        2,
    );
    let duplicate_file =
        WeightStoreConfig::new(primary.path()).with_primary_shard_root(duplicate_name.path());
    assert!(WeightStore::open_config(&duplicate_file, &InferenceLimits::default()).is_err());

    let nested = primary.path().join("nested");
    std::fs::create_dir(&nested).unwrap();
    write_weight_file(&nested, "nested.safetensors", "layer.2.weight", 0, 1, 3);
    let overlapping = WeightStoreConfig::new(primary.path()).with_primary_shard_root(&nested);
    assert!(WeightStore::open_config(&overlapping, &InferenceLimits::default()).is_err());

    let repeated = WeightStoreConfig::new(primary.path()).with_primary_shard_root(primary.path());
    assert!(WeightStore::open_config(&repeated, &InferenceLimits::default()).is_err());

    let empty_root = WeightStoreConfig::new(primary.path()).with_primary_shard_root(empty.path());
    assert!(WeightStore::open_config(&empty_root, &InferenceLimits::default()).is_err());
}

#[test]
fn sharded_complete_replicas_reuse_the_existing_source_router() {
    let primary = tempfile::tempdir().unwrap();
    let primary_shard = tempfile::tempdir().unwrap();
    let replica = tempfile::tempdir().unwrap();
    let replica_shard = tempfile::tempdir().unwrap();
    write_weight_file(
        primary.path(),
        "first.safetensors",
        "layer.0.weight",
        0,
        64,
        1,
    );
    write_weight_file(
        primary_shard.path(),
        "second.safetensors",
        "layer.1.weight",
        0,
        64,
        2,
    );
    std::fs::copy(
        primary.path().join("first.safetensors"),
        replica.path().join("first.safetensors"),
    )
    .unwrap();
    std::fs::copy(
        primary_shard.path().join("second.safetensors"),
        replica_shard.path().join("second.safetensors"),
    )
    .unwrap();

    let config = WeightStoreConfig::new(primary.path())
        .with_primary_shard_root(primary_shard.path())
        .with_primary_read_weight(1)
        .with_replica(
            WeightSourceConfig::new(replica.path())
                .with_shard_root(replica_shard.path())
                .with_read_weight(31),
        );
    let store = WeightStore::open_config(&config, &InferenceLimits::default()).unwrap();
    let replica_shard_root = std::fs::canonicalize(replica_shard.path()).unwrap();

    assert_eq!(store.sources().len(), 2);
    assert_eq!(store.sources()[0].verified_files, 2);
    assert_eq!(store.sources()[1].verified_files, 2);
    assert_eq!(
        store.sources()[1].shard_roots,
        [replica_shard_root.as_path()]
    );
    let routed = store
        .inventory()
        .map(|descriptor| store.select_source(&descriptor.name))
        .collect::<Vec<_>>();
    assert!(routed.contains(&0));
    assert!(routed.contains(&1));
}

#[test]
fn replica_digest_mismatch_fails_closed() {
    let primary = tempfile::tempdir().unwrap();
    let replica = tempfile::tempdir().unwrap();
    write_weights(primary.path(), 1);
    write_weights(replica.path(), 2);
    let config = WeightStoreConfig::new(primary.path())
        .with_replica(WeightSourceConfig::new(replica.path()));

    assert!(matches!(
        WeightStore::open_config(&config, &InferenceLimits::default()),
        Err(PowerError::IntegrityCheckFailed { .. })
    ));
}

#[test]
fn partial_replica_serves_only_its_verified_tensor_subset() {
    let primary = tempfile::tempdir().unwrap();
    let replica = tempfile::tempdir().unwrap();
    write_weight_file(primary.path(), "hot.safetensors", "layer.0.hot", 0, 128, 1);
    write_weight_file(
        primary.path(),
        "cold.safetensors",
        "layer.1.cold",
        0,
        128,
        2,
    );
    std::fs::copy(
        primary.path().join("hot.safetensors"),
        replica.path().join("hot.safetensors"),
    )
    .unwrap();
    let config = WeightStoreConfig::new(primary.path())
        .with_primary_read_strategy(WeightReadStrategy::PositionalBuffered)
        .with_primary_read_weight(1)
        .with_partial_replica(
            WeightSourceConfig::new(replica.path())
                .with_read_weight(31)
                .with_read_strategy(WeightReadStrategy::PositionalBuffered),
        );

    let store = WeightStore::open_config(&config, &InferenceLimits::default()).unwrap();
    let descriptor = &store.sources()[1];
    assert_eq!(descriptor.coverage, WeightSourceCoverage::Partial);
    assert_eq!(
        descriptor.read_strategy,
        WeightReadStrategy::PositionalBuffered
    );
    assert_eq!(descriptor.verified_files, 1);
    assert_eq!(descriptor.verified_tensors, 128);
    assert!(descriptor.verified_bytes > 0);

    let mirrored_name = (0..128)
        .map(|index| format!("layer.0.hot.{index}"))
        .find(|name| store.select_source(name) == 1)
        .expect("at least one covered tensor must route to the weighted replica");
    let mirrored = store.load_tracked(&mirrored_name, &Device::Cpu).unwrap();
    assert_eq!(mirrored.source_index, 1);
    assert!(!mirrored.fell_back);

    for index in 0..128 {
        let name = format!("layer.1.cold.{index}");
        assert_eq!(store.select_source(&name), 0);
        let loaded = store.load_tracked(&name, &Device::Cpu).unwrap();
        assert_eq!(loaded.source_index, 0);
        assert!(!loaded.fell_back);
    }
}

#[test]
fn partial_replica_files_are_verified_against_the_primary() {
    let primary = tempfile::tempdir().unwrap();
    let replica = tempfile::tempdir().unwrap();
    write_weight_file(primary.path(), "hot.safetensors", "layer.0.hot", 0, 8, 1);
    write_weight_file(replica.path(), "hot.safetensors", "layer.0.hot", 0, 8, 2);
    let config = WeightStoreConfig::new(primary.path()).with_replica(
        WeightSourceConfig::new(replica.path()).with_coverage(WeightSourceCoverage::Partial),
    );

    assert!(matches!(
        WeightStore::open_config(&config, &InferenceLimits::default()),
        Err(PowerError::IntegrityCheckFailed { .. })
    ));
}

#[test]
fn complete_replica_still_requires_the_complete_collection() {
    let primary = tempfile::tempdir().unwrap();
    let replica = tempfile::tempdir().unwrap();
    write_weight_file(primary.path(), "one.safetensors", "layer.0.weight", 0, 8, 1);
    write_weight_file(primary.path(), "two.safetensors", "layer.1.weight", 0, 8, 2);
    std::fs::copy(
        primary.path().join("one.safetensors"),
        replica.path().join("one.safetensors"),
    )
    .unwrap();
    let config = WeightStoreConfig::new(primary.path())
        .with_replica(WeightSourceConfig::new(replica.path()));

    assert!(matches!(
        WeightStore::open_config(&config, &InferenceLimits::default()),
        Err(PowerError::IntegrityCheckFailed { .. })
    ));
}

#[test]
fn validation_throughput_can_weight_sources_without_an_extra_probe() {
    let primary = tempfile::tempdir().unwrap();
    let replica = tempfile::tempdir().unwrap();
    write_weights(primary.path(), 1);
    std::fs::copy(
        primary.path().join("model.safetensors"),
        replica.path().join("model.safetensors"),
    )
    .unwrap();
    let config = WeightStoreConfig::new(primary.path())
        .with_primary_read_weight(7)
        .with_replica(WeightSourceConfig::new(replica.path()).with_read_weight(11))
        .with_source_weighting(WeightSourceWeighting::ValidationThroughput);

    let store = WeightStore::open_config(&config, &InferenceLimits::default()).unwrap();
    for source in store.sources() {
        assert_eq!(
            source.source_weighting,
            WeightSourceWeighting::ValidationThroughput
        );
        assert!(source.validation_bytes_per_second > 0);
        assert!((1..=1_024).contains(&source.read_weight));
    }
    assert_eq!(store.sources()[0].configured_read_weight, 7);
    assert_eq!(store.sources()[1].configured_read_weight, 11);
}

#[test]
fn throughput_weights_are_bounded_and_ratio_preserving() {
    assert_eq!(normalized_throughput_weights(&[]), Vec::<u32>::new());
    assert_eq!(normalized_throughput_weights(&[0, 0]), [1, 1]);
    assert_eq!(normalized_throughput_weights(&[9, 3, 0]), [1_024, 341, 1]);
}

#[test]
fn new_source_options_have_backward_compatible_serde_defaults() {
    let config: WeightStoreConfig = serde_json::from_value(serde_json::json!({
        "primary": {"root": "/models/primary", "readWeight": 9},
        "replicas": [{"root": "/models/replica", "readWeight": 3}]
    }))
    .unwrap();

    assert_eq!(config.source_weighting, WeightSourceWeighting::Configured);
    assert!(config.primary.shard_roots.is_empty());
    assert_eq!(config.primary.coverage, WeightSourceCoverage::Complete);
    assert_eq!(config.primary.read_strategy, WeightReadStrategy::Mmap);
    assert_eq!(
        config.primary.representation,
        WeightSourceRepresentation::CanonicalSafeTensors
    );
    assert_eq!(config.replicas[0].coverage, WeightSourceCoverage::Complete);
    assert_eq!(config.replicas[0].read_strategy, WeightReadStrategy::Mmap);
    assert!(config.replicas[0].shard_roots.is_empty());
    assert_eq!(
        config.replicas[0].representation,
        WeightSourceRepresentation::CanonicalSafeTensors
    );
    let serialized = serde_json::to_value(&config).unwrap();
    assert!(serialized["primary"].get("shardRoots").is_none());
    assert!(serialized["replicas"][0].get("shardRoots").is_none());
}

#[test]
fn source_shard_roots_have_an_explicit_serde_shape() {
    let config: WeightStoreConfig = serde_json::from_value(serde_json::json!({
        "primary": {
            "root": "/models/primary-a",
            "shardRoots": ["/models/primary-b"],
            "readWeight": 1
        },
        "replicas": []
    }))
    .unwrap();

    assert_eq!(
        config.primary.shard_roots,
        [PathBuf::from("/models/primary-b")]
    );
}

#[test]
fn duplicate_or_zero_weight_sources_are_rejected() {
    let primary = tempfile::tempdir().unwrap();
    write_weights(primary.path(), 1);
    let duplicate = WeightStoreConfig::new(primary.path())
        .with_replica(WeightSourceConfig::new(primary.path()));
    assert!(WeightStore::open_config(&duplicate, &InferenceLimits::default()).is_err());

    let zero = WeightStoreConfig::new(primary.path()).with_primary_read_weight(0);
    assert!(WeightStore::open_config(&zero, &InferenceLimits::default()).is_err());

    let partial_primary = WeightStoreConfig {
        primary: WeightSourceConfig::new(primary.path())
            .with_coverage(WeightSourceCoverage::Partial),
        replicas: Vec::new(),
        source_weighting: WeightSourceWeighting::Configured,
    };
    assert!(WeightStore::open_config(&partial_primary, &InferenceLimits::default()).is_err());

    let limited = InferenceLimits {
        max_weight_sources: 1,
        ..InferenceLimits::default()
    };
    assert!(WeightStore::open_config(&duplicate, &limited).is_err());
}
