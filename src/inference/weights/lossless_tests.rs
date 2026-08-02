use std::collections::HashMap;
use std::path::Path;

use candle_core::Device;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tokio_util::sync::CancellationToken;

use crate::inference::{
    run_storage_benchmark, StorageBenchmarkConfig, StorageCachePreparation, StorageCacheState,
};

use super::lossless::{
    weight_collection_sha256, LosslessRansNibbleHistogram, LosslessRansNibbleTable,
    LOSSLESS_RANS_FORMAT_METADATA_KEY,
};
use super::*;

const SCRATCH_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone)]
struct FixtureTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl FixtureTensor {
    fn f32(name: impl Into<String>, byte: u8, bytes: usize) -> Self {
        assert_eq!(bytes % 4, 0);
        Self {
            name: name.into(),
            dtype: Dtype::F32,
            shape: vec![bytes / 4],
            bytes: vec![byte; bytes],
        }
    }
}

fn write_primary(root: &Path, tensors: &[FixtureTensor]) {
    let views = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.name.as_str(),
                TensorView::new(tensor.dtype, tensor.shape.clone(), tensor.bytes.as_slice())
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &root.join("model.safetensors")).unwrap();
}

fn table_for(tensors: &[FixtureTensor]) -> LosslessRansNibbleTable {
    let mut histogram = LosslessRansNibbleHistogram::default();
    for tensor in tensors {
        histogram.observe(&tensor.bytes).unwrap();
    }
    histogram.build().unwrap()
}

fn write_lossless(root: &Path, tensors: &[FixtureTensor]) -> LosslessRansNibbleTable {
    let table = table_for(tensors);
    write_lossless_with_metadata(root, tensors, table.safetensors_metadata().unwrap());
    table
}

fn write_lossless_with_metadata(
    root: &Path,
    tensors: &[FixtureTensor],
    metadata: HashMap<String, String>,
) {
    let table = table_for(tensors);
    let encoded = tensors
        .iter()
        .map(|tensor| table.encode_record(&tensor.bytes, SCRATCH_BYTES).unwrap())
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .zip(&encoded)
        .map(|(tensor, record)| {
            (
                tensor.name.as_str(),
                TensorView::new(Dtype::U8, vec![record.len()], record.as_slice()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, Some(metadata), &root.join("compressed.safetensors")).unwrap();
}

fn representation(root: &Path, limits: &InferenceLimits) -> WeightSourceRepresentation {
    WeightSourceRepresentation::LosslessRansNibble256V1 {
        artifact_sha256: weight_collection_sha256(root, limits).unwrap(),
    }
}

fn config(
    primary: &Path,
    compressed: &Path,
    coverage: WeightSourceCoverage,
    limits: &InferenceLimits,
) -> WeightStoreConfig {
    WeightStoreConfig::new(primary)
        .with_primary_read_strategy(WeightReadStrategy::PositionalBuffered)
        .with_primary_read_weight(1)
        .with_replica(
            WeightSourceConfig::new(compressed)
                .with_read_weight(u32::MAX)
                .with_coverage(coverage)
                .with_read_strategy(WeightReadStrategy::PositionalBuffered)
                .with_representation(representation(compressed, limits)),
        )
}

#[test]
fn codec_round_trip_is_deterministic_and_byte_exact() {
    let bytes = vec![0_u8; 64 * 1024];
    let table = LosslessRansNibbleHistogram::from_bytes(&bytes)
        .unwrap()
        .build()
        .unwrap();
    let first = table.encode_record(&bytes, SCRATCH_BYTES).unwrap();
    let second = table.encode_record(&bytes, SCRATCH_BYTES).unwrap();

    assert_eq!(first.as_slice(), second.as_slice());
    assert!(first.len() < bytes.len());
    let decoded = table
        .decode_record(first.as_slice(), bytes.len() as u64, SCRATCH_BYTES)
        .unwrap();
    assert_eq!(decoded.as_slice(), bytes);
}

#[test]
fn codec_round_trip_uses_one_shared_multi_symbol_table() {
    let bytes = (0..64 * 1024)
        .map(|index| if index % 8 == 0 { 0x11 } else { 0x00 })
        .collect::<Vec<_>>();
    let table = LosslessRansNibbleHistogram::from_bytes(&bytes)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(
        table
            .frequencies()
            .iter()
            .filter(|value| **value > 0)
            .count(),
        2
    );

    let record = table.encode_record(&bytes, SCRATCH_BYTES).unwrap();
    assert!(record.len() < bytes.len());
    let decoded = table
        .decode_record(&record, bytes.len() as u64, SCRATCH_BYTES)
        .unwrap();
    assert_eq!(decoded.as_slice(), bytes);
}

#[test]
fn codec_round_trip_covers_stream_boundaries_and_full_alphabet() {
    for length in [1_usize, 2, 127, 128, 129, 255, 256, 257, 4_097, 65_535] {
        let mut state = 0x9e37_79b9_u32;
        let bytes = (0..length)
            .map(|index| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8 ^ index as u8
            })
            .collect::<Vec<_>>();
        let table = LosslessRansNibbleHistogram::from_bytes(&bytes)
            .unwrap()
            .build()
            .unwrap();
        let record = table.encode_record(&bytes, SCRATCH_BYTES).unwrap();
        let decoded = table
            .decode_record(&record, bytes.len() as u64, SCRATCH_BYTES)
            .unwrap();
        assert_eq!(decoded.as_slice(), bytes, "failed at byte length {length}");
    }
}

#[test]
fn codec_refuses_malformed_framing_padding_and_states() {
    let bytes = vec![0_u8; 16 * 1024];
    let table = LosslessRansNibbleHistogram::from_bytes(&bytes)
        .unwrap()
        .build()
        .unwrap();
    let record = table.encode_record(&bytes, SCRATCH_BYTES).unwrap();

    assert!(table
        .decode_record(&record[..32], bytes.len() as u64, SCRATCH_BYTES)
        .is_err());

    let mut bad_offset = record.to_vec();
    bad_offset[16] = 1;
    assert!(table
        .decode_record(&bad_offset, bytes.len() as u64, SCRATCH_BYTES)
        .is_err());

    let mut bad_header_padding = record.to_vec();
    bad_header_padding[1_044] = 1;
    assert!(table
        .decode_record(&bad_header_padding, bytes.len() as u64, SCRATCH_BYTES)
        .is_err());

    let mut bad_state = record.to_vec();
    bad_state[1_056..1_060].fill(0);
    assert!(table
        .decode_record(&bad_state, bytes.len() as u64, SCRATCH_BYTES)
        .is_err());

    let mut trailing = record.to_vec();
    trailing.push(0);
    assert!(table
        .decode_record(&trailing, bytes.len() as u64, SCRATCH_BYTES)
        .is_err());
}

#[test]
fn codec_checks_expected_size_scratch_and_cancellation_before_decode() {
    let bytes = vec![0_u8; 16 * 1024];
    let table = LosslessRansNibbleHistogram::from_bytes(&bytes)
        .unwrap()
        .build()
        .unwrap();
    let record = table.encode_record(&bytes, SCRATCH_BYTES).unwrap();

    assert!(table
        .decode_record(&record, bytes.len() as u64 + 1, SCRATCH_BYTES)
        .is_err());
    assert!(table
        .decode_record(
            &record,
            bytes.len() as u64,
            record.len() as u64 + bytes.len() as u64 - 1,
        )
        .is_err());

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert!(matches!(
        table.decode_record_with_cancellation(
            &record,
            bytes.len() as u64,
            SCRATCH_BYTES,
            &cancelled,
        ),
        Err(PowerError::InferenceFailed(_))
    ));
}

#[test]
fn codec_rejects_amplification_claim_before_output_allocation() {
    let bytes = vec![0_u8; 16 * 1024];
    let table = LosslessRansNibbleHistogram::from_bytes(&bytes)
        .unwrap()
        .build()
        .unwrap();
    let record = table.encode_record(&bytes, SCRATCH_BYTES).unwrap();
    let mut bomb = record.to_vec();
    let payload_offset = 16 + 256 * 4;
    let payload_bytes =
        u32::from_le_bytes(bomb[payload_offset..payload_offset + 4].try_into().unwrap()) as u64;
    let amplification_bound = payload_bytes * 8 * (1_u64 << 15);
    let claimed_bytes = amplification_bound / 2 + 1;
    bomb[..8].copy_from_slice(&(claimed_bytes * 2).to_le_bytes());
    bomb[8..16].copy_from_slice(&claimed_bytes.to_le_bytes());

    let error = table
        .decode_record(&bomb, claimed_bytes, claimed_bytes + bomb.len() as u64)
        .unwrap_err();
    assert!(format!("{error}").contains("amplification"));
}

#[test]
fn verified_lossless_replica_reuses_source_routing_and_preserves_tensor_bytes() {
    let primary = tempfile::tempdir().unwrap();
    let compressed = tempfile::tempdir().unwrap();
    let tensors = (0..32)
        .map(|index| FixtureTensor::f32(format!("expert.{index}.weight"), 0, 16 * 1024))
        .collect::<Vec<_>>();
    write_primary(primary.path(), &tensors);
    write_lossless(compressed.path(), &tensors);
    let limits = InferenceLimits::default();
    let store = WeightStore::open_config(
        &config(
            primary.path(),
            compressed.path(),
            WeightSourceCoverage::Complete,
            &limits,
        ),
        &limits,
    )
    .unwrap();
    let selected = tensors
        .iter()
        .find(|tensor| store.select_source(&tensor.name) == 1)
        .unwrap();

    assert_eq!(store.sources().len(), 2);
    assert!(matches!(
        store.sources()[1].representation,
        WeightSourceRepresentation::LosslessRansNibble256V1 { .. }
    ));
    assert!(store.sources()[1].verified_bytes < store.sources()[0].verified_bytes);
    let read = store.read_tensor_bytes(&selected.name).unwrap();
    assert_eq!(read.source_index(), 1);
    assert!(!read.fell_back());
    assert_eq!(read.bytes(), selected.bytes);
    assert!(matches!(
        read.representation(),
        WeightSourceRepresentation::LosslessRansNibble256V1 { .. }
    ));

    let tensor = store.load_tracked(&selected.name, &Device::Cpu).unwrap();
    assert_eq!(tensor.source_index, 1);
    assert_eq!(
        tensor
            .tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        vec![0.0_f32; 4_096]
    );
}

#[test]
fn lossless_replica_reuses_the_existing_mmap_materialization_path() {
    let primary = tempfile::tempdir().unwrap();
    let compressed = tempfile::tempdir().unwrap();
    let tensors = (0..16)
        .map(|index| FixtureTensor::f32(format!("expert.{index}.weight"), 0, 16 * 1024))
        .collect::<Vec<_>>();
    write_primary(primary.path(), &tensors);
    write_lossless(compressed.path(), &tensors);
    let limits = InferenceLimits::default();
    let mut weights = config(
        primary.path(),
        compressed.path(),
        WeightSourceCoverage::Complete,
        &limits,
    );
    weights.primary.read_strategy = WeightReadStrategy::Mmap;
    weights.replicas[0].read_strategy = WeightReadStrategy::Mmap;
    let store = WeightStore::open_config(&weights, &limits).unwrap();
    let selected = tensors
        .iter()
        .find(|tensor| store.select_source(&tensor.name) == 1)
        .unwrap();

    let read = store.read_tensor_bytes(&selected.name).unwrap();
    assert_eq!(read.strategy(), WeightReadStrategy::Mmap);
    assert_eq!(read.source_index(), 1);
    assert_eq!(read.bytes(), selected.bytes);
    assert_eq!(read.storage().dtype, "u8");
    assert!(read.storage().bytes < selected.bytes.len() as u64);
}

#[cfg(target_os = "linux")]
#[test]
fn cache_preparation_uses_compressed_physical_record_ranges() {
    let primary = tempfile::tempdir().unwrap();
    let compressed = tempfile::tempdir().unwrap();
    let tensors = vec![FixtureTensor::f32("expert.weight", 0, 16 * 1024)];
    write_primary(primary.path(), &tensors);
    write_lossless(compressed.path(), &tensors);
    let limits = InferenceLimits::default();
    let store = WeightStore::open_config(
        &config(
            primary.path(),
            compressed.path(),
            WeightSourceCoverage::Complete,
            &limits,
        ),
        &limits,
    )
    .unwrap();

    let ranges = store
        .verified_cache_ranges(&["expert.weight".to_string()])
        .unwrap();
    assert_eq!(ranges.len(), 2);
    assert!(ranges.iter().any(|range| range.bytes == 16 * 1024));
    assert!(ranges.iter().any(|range| range.bytes < 16 * 1024));
}

#[test]
fn artifact_digest_is_verified_before_container_metadata() {
    let primary = tempfile::tempdir().unwrap();
    let compressed = tempfile::tempdir().unwrap();
    let tensors = vec![FixtureTensor::f32("expert.weight", 0, 16 * 1024)];
    write_primary(primary.path(), &tensors);
    write_lossless_with_metadata(compressed.path(), &tensors, HashMap::new());
    let limits = InferenceLimits::default();
    let bad = WeightSourceRepresentation::LosslessRansNibble256V1 {
        artifact_sha256: "0".repeat(64),
    };
    let config = WeightStoreConfig::new(primary.path()).with_replica(
        WeightSourceConfig::new(compressed.path())
            .with_representation(bad)
            .with_coverage(WeightSourceCoverage::Complete),
    );

    assert!(matches!(
        WeightStore::open_config(&config, &limits),
        Err(PowerError::IntegrityCheckFailed { .. })
    ));
}

#[test]
fn mandatory_stamp_and_table_fail_closed() {
    let primary = tempfile::tempdir().unwrap();
    let compressed = tempfile::tempdir().unwrap();
    let tensors = vec![FixtureTensor::f32("expert.weight", 0, 16 * 1024)];
    write_primary(primary.path(), &tensors);
    write_lossless_with_metadata(compressed.path(), &tensors, HashMap::new());
    let limits = InferenceLimits::default();

    assert!(matches!(
        WeightStore::open_config(
            &config(
                primary.path(),
                compressed.path(),
                WeightSourceCoverage::Complete,
                &limits,
            ),
            &limits,
        ),
        Err(PowerError::InvalidFormat(_))
    ));

    let metadata = HashMap::from([(
        LOSSLESS_RANS_FORMAT_METADATA_KEY.to_string(),
        LosslessRansNibbleTable::FORMAT.to_string(),
    )]);
    write_lossless_with_metadata(compressed.path(), &tensors, metadata);
    assert!(matches!(
        WeightStore::open_config(
            &config(
                primary.path(),
                compressed.path(),
                WeightSourceCoverage::Complete,
                &limits,
            ),
            &limits,
        ),
        Err(PowerError::InvalidFormat(_))
    ));
}

#[test]
fn malformed_table_identity_and_frequency_sum_fail_closed() {
    let primary = tempfile::tempdir().unwrap();
    let compressed = tempfile::tempdir().unwrap();
    let tensors = vec![FixtureTensor::f32("expert.weight", 0, 16 * 1024)];
    write_primary(primary.path(), &tensors);
    let limits = InferenceLimits::default();

    let unsupported = HashMap::from([
        (
            LOSSLESS_RANS_FORMAT_METADATA_KEY.to_string(),
            LosslessRansNibbleTable::FORMAT.to_string(),
        ),
        (
            super::lossless::LOSSLESS_RANS_TABLE_METADATA_KEY.to_string(),
            serde_json::json!({
                "schema": "a3s.power.rans-nibble-256-table.v0",
                "streams": 256,
                "scaleBits": 14,
                "frequencies": [16384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            })
            .to_string(),
        ),
    ]);
    write_lossless_with_metadata(compressed.path(), &tensors, unsupported);
    assert!(WeightStore::open_config(
        &config(
            primary.path(),
            compressed.path(),
            WeightSourceCoverage::Complete,
            &limits,
        ),
        &limits,
    )
    .is_err());

    assert!(LosslessRansNibbleTable::from_frequencies([1_u32; 16]).is_err());
}

#[test]
fn source_checks_encoded_plus_decoded_scratch_before_record_reads() {
    let primary = tempfile::tempdir().unwrap();
    let compressed = tempfile::tempdir().unwrap();
    let tensors = vec![FixtureTensor::f32("expert.weight", 0, 16 * 1024)];
    write_primary(primary.path(), &tensors);
    write_lossless(compressed.path(), &tensors);
    let limits = InferenceLimits {
        max_state_bytes: 16 * 1024,
        ..InferenceLimits::default()
    };

    let error = WeightStore::open_config(
        &config(
            primary.path(),
            compressed.path(),
            WeightSourceCoverage::Complete,
            &limits,
        ),
        &limits,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("encoded-plus-decoded"));
}

#[test]
fn decoded_byte_mismatch_and_incompressible_records_are_rejected() {
    let primary = tempfile::tempdir().unwrap();
    let compressed = tempfile::tempdir().unwrap();
    let canonical = vec![FixtureTensor::f32("expert.weight", 0, 16 * 1024)];
    let changed = vec![FixtureTensor::f32("expert.weight", 1, 16 * 1024)];
    write_primary(primary.path(), &canonical);
    write_lossless(compressed.path(), &changed);
    let limits = InferenceLimits::default();
    assert!(matches!(
        WeightStore::open_config(
            &config(
                primary.path(),
                compressed.path(),
                WeightSourceCoverage::Complete,
                &limits,
            ),
            &limits,
        ),
        Err(PowerError::IntegrityCheckFailed { .. })
    ));

    let incompressible = tempfile::tempdir().unwrap();
    let bytes = (0..16 * 1024)
        .map(|index| u8::try_from(index % 256).unwrap())
        .collect::<Vec<_>>();
    let randomish = vec![FixtureTensor {
        name: "expert.weight".to_string(),
        dtype: Dtype::F32,
        shape: vec![bytes.len() / 4],
        bytes,
    }];
    write_primary(primary.path(), &randomish);
    write_lossless(incompressible.path(), &randomish);
    assert!(matches!(
        WeightStore::open_config(
            &config(
                primary.path(),
                incompressible.path(),
                WeightSourceCoverage::Complete,
                &limits,
            ),
            &limits,
        ),
        Err(PowerError::InvalidFormat(_))
    ));
}

#[test]
fn partial_representation_covers_only_admitted_tensors() {
    let primary = tempfile::tempdir().unwrap();
    let compressed = tempfile::tempdir().unwrap();
    let tensors = vec![
        FixtureTensor::f32("expert.hot.weight", 0, 16 * 1024),
        FixtureTensor::f32("expert.cold.weight", 1, 16 * 1024),
    ];
    write_primary(primary.path(), &tensors);
    write_lossless(compressed.path(), &tensors[..1]);
    let limits = InferenceLimits::default();
    let partial = WeightStore::open_config(
        &config(
            primary.path(),
            compressed.path(),
            WeightSourceCoverage::Partial,
            &limits,
        ),
        &limits,
    )
    .unwrap();
    assert!(partial.replicas[0].contains("expert.hot.weight"));
    assert!(!partial.replicas[0].contains("expert.cold.weight"));
    assert_eq!(partial.select_source("expert.cold.weight"), 0);

    assert!(WeightStore::open_config(
        &config(
            primary.path(),
            compressed.path(),
            WeightSourceCoverage::Complete,
            &limits,
        ),
        &limits,
    )
    .is_err());
}

#[test]
fn corrupted_representation_uses_existing_primary_fallback() {
    let primary = tempfile::tempdir().unwrap();
    let compressed = tempfile::tempdir().unwrap();
    let tensors = (0..16)
        .map(|index| FixtureTensor::f32(format!("expert.{index}.weight"), 0, 16 * 1024))
        .collect::<Vec<_>>();
    write_primary(primary.path(), &tensors);
    write_lossless(compressed.path(), &tensors);
    let limits = InferenceLimits::default();
    let store = WeightStore::open_config(
        &config(
            primary.path(),
            compressed.path(),
            WeightSourceCoverage::Complete,
            &limits,
        ),
        &limits,
    )
    .unwrap();
    let selected = tensors
        .iter()
        .find(|tensor| store.select_source(&tensor.name) == 1)
        .unwrap();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(compressed.path().join("compressed.safetensors"))
        .unwrap();
    file.set_len(8).unwrap();

    let read = store.read_tensor_bytes(&selected.name).unwrap();
    assert_eq!(read.source_index(), 0);
    assert!(read.fell_back());
    assert_eq!(read.bytes(), selected.bytes);
}

#[test]
fn existing_storage_benchmark_records_representation_and_decoded_parity() {
    let primary = tempfile::tempdir().unwrap();
    let compressed = tempfile::tempdir().unwrap();
    let tensors = (0..8)
        .map(|index| FixtureTensor::f32(format!("expert.{index}.weight"), 0, 16 * 1024))
        .collect::<Vec<_>>();
    write_primary(primary.path(), &tensors);
    write_lossless(compressed.path(), &tensors);
    let limits = InferenceLimits::default();
    let report = run_storage_benchmark(
        &StorageBenchmarkConfig {
            weights: config(
                primary.path(),
                compressed.path(),
                WeightSourceCoverage::Complete,
                &limits,
            ),
            power_commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
            filesystem_class: "test-filesystem".to_string(),
            device_class: "test-device".to_string(),
            cpu_model: "test-cpu".to_string(),
            ram_bytes: 1024,
            cache_state: StorageCacheState::Warm,
            cache_preparation: StorageCachePreparation::WarmSequence,
            concurrency: 2,
            samples: 1,
            max_tensors: tensors.len(),
        },
        &limits,
    )
    .unwrap();

    assert_eq!(report.sources.len(), 2);
    assert!(matches!(
        report.sources[1].representation,
        WeightSourceRepresentation::LosslessRansNibble256V1 { .. }
    ));
    assert_eq!(report.total_requested_bytes, report.total_read_bytes);
    assert_eq!(report.output_sha256.len(), 64);
}

#[test]
fn representation_types_are_send_sync_and_debug_output_is_byte_free() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<LosslessRansNibbleHistogram>();
    assert_send_sync::<LosslessRansNibbleTable>();
    assert_send_sync::<super::lossless::LosslessEncodedRecord>();
    assert_send_sync::<WeightSourceRepresentation>();

    let bytes = vec![0xab; 16 * 1024];
    let table = LosslessRansNibbleHistogram::from_bytes(&bytes)
        .unwrap()
        .build()
        .unwrap();
    let record = table.encode_record(&bytes, SCRATCH_BYTES).unwrap();
    assert!(!format!("{table:?}").contains("171, 171"));
    let debug = format!("{record:?}");
    assert!(debug.contains("bytes"));
    assert!(!debug.contains('['));
    assert!(!debug.contains("171, 171"));
}

#[test]
fn representation_serde_identity_is_explicit_and_stable() {
    let representation = WeightSourceRepresentation::LosslessRansNibble256V1 {
        artifact_sha256: "a".repeat(64),
    };
    let value = serde_json::to_value(&representation).unwrap();
    assert_eq!(
        value,
        serde_json::json!({
            "kind": "lossless-rans-nibble-256-v1",
            "artifactSha256": "a".repeat(64),
        })
    );
    assert_eq!(
        serde_json::from_value::<WeightSourceRepresentation>(value).unwrap(),
        representation
    );
}
