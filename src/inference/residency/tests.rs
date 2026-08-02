use std::sync::Arc;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

use super::*;
use crate::inference::{DevicePreference, InferenceLimits, TelemetryMode};

fn weight_store(
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: &[u8],
) -> (tempfile::TempDir, Arc<WeightStore>) {
    let directory = tempfile::tempdir().unwrap();
    let view = TensorView::new(dtype, shape, bytes).unwrap();
    serialize_to_file(
        vec![("layer.0.expert.0", view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let store = WeightStore::open(directory.path(), &InferenceLimits::default()).unwrap();
    (directory, Arc::new(store))
}

fn new_runtime() -> EmbeddedRuntime {
    EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default()).unwrap()
}

#[test]
fn hierarchy_preserves_dtype_and_uses_layer_cache() {
    let (_directory, store) = weight_store(Dtype::BF16, vec![2], &[0, 0, 0, 0]);
    let runtime = new_runtime();
    let hierarchy = WeightHierarchy::new(
        store,
        runtime.clone(),
        ResidencyPolicy {
            host_cache_bytes: 4,
            telemetry: TelemetryMode::Aggregate,
            ..ResidencyPolicy::default()
        },
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let request = WeightRequest::new(
        WeightKey::new(0, "layer.0.expert.0"),
        PlacementPreference::Host,
    );

    let first = hierarchy.load(&request, &permit, &cancellation).unwrap();
    let second = hierarchy.load(&request, &permit, &cancellation).unwrap();

    assert_eq!(first.tensor().dtype(), candle_core::DType::BF16);
    assert_eq!(first.tier(), WeightTier::Host);
    assert!(!first.cache_hit());
    assert!(second.cache_hit());
    let telemetry = hierarchy.telemetry();
    assert_eq!(telemetry.storage_reads, 1);
    assert_eq!(telemetry.host_cache_hits, 1);
    assert_eq!(telemetry.host_resident_bytes, 4);
}

#[tokio::test]
async fn prefetch_unions_duplicate_requests_and_reuses_residency() {
    let (_directory, store) = weight_store(Dtype::F32, vec![2], &[0; 8]);
    let runtime = new_runtime();
    let hierarchy = WeightHierarchy::new(
        store,
        runtime.clone(),
        ResidencyPolicy {
            host_cache_bytes: 8,
            telemetry: TelemetryMode::Aggregate,
            ..ResidencyPolicy::default()
        },
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let request = WeightRequest::new(
        WeightKey::new(0, "layer.0.expert.0"),
        PlacementPreference::Host,
    );

    let first = hierarchy
        .start_prefetch(
            vec![request.clone(), request.clone()],
            &permit,
            cancellation.clone(),
        )
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert_eq!(first.requested, 2);
    assert_eq!(first.unique, 1);
    assert_eq!(first.materialized, 1);

    let second = hierarchy
        .start_prefetch(vec![request], &permit, cancellation)
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert_eq!(second.cache_hits, 1);
    assert_eq!(hierarchy.telemetry().storage_reads, 1);
}

#[test]
fn hierarchy_rejects_a_permit_from_another_runtime() {
    let (_directory, store) = weight_store(Dtype::F32, vec![1], &[0; 4]);
    let runtime = new_runtime();
    let other = new_runtime();
    let hierarchy = WeightHierarchy::new(store, runtime, ResidencyPolicy::default()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = other.begin(&cancellation).unwrap();
    let request = WeightRequest::new(
        WeightKey::new(0, "layer.0.expert.0"),
        PlacementPreference::Streaming,
    );

    assert!(hierarchy.load(&request, &permit, &cancellation).is_err());
}
