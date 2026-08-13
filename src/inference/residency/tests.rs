use std::sync::Arc;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

use super::*;
use crate::inference::{
    DevicePreference, InferenceLimits, RoutedExpert, TelemetryMode, WeightReadStrategy,
    WeightStoreConfig,
};

fn weight_store(
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: &[u8],
) -> (tempfile::TempDir, Arc<WeightStore>) {
    weight_store_with_strategy(dtype, shape, bytes, WeightReadStrategy::Mmap)
}

fn weight_store_with_strategy(
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: &[u8],
    read_strategy: WeightReadStrategy,
) -> (tempfile::TempDir, Arc<WeightStore>) {
    let directory = tempfile::tempdir().unwrap();
    let view = TensorView::new(dtype, shape, bytes).unwrap();
    serialize_to_file(
        vec![("layer.0.expert.0", view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let store = WeightStore::open_config(
        &WeightStoreConfig::new(directory.path()).with_primary_read_strategy(read_strategy),
        &InferenceLimits::default(),
    )
    .unwrap();
    (directory, Arc::new(store))
}

fn new_runtime() -> EmbeddedRuntime {
    EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default()).unwrap()
}

#[test]
fn hierarchy_accounts_for_model_owned_fixed_weights() {
    let (_directory, store) = weight_store(Dtype::F32, vec![2], &[0; 8]);
    let limits = InferenceLimits {
        max_resident_weight_bytes: 12,
        ..InferenceLimits::default()
    };
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap();
    let policy = ResidencyPolicy {
        host_cache_bytes: 4,
        ..ResidencyPolicy::default()
    };

    let hierarchy = WeightHierarchy::new_with_fixed_weight_bytes(
        Arc::clone(&store),
        runtime.clone(),
        policy.clone(),
        8,
    )
    .unwrap();
    assert_eq!(hierarchy.fixed_weight_bytes(), 8);

    let error =
        WeightHierarchy::new_with_fixed_weight_bytes(store, runtime, policy, 9).unwrap_err();
    assert!(error.to_string().contains("total 13 bytes"));
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
    assert_eq!(telemetry.storage_sources.len(), 1);
    assert_eq!(telemetry.storage_sources[0].source_index, 0);
    assert_eq!(telemetry.storage_sources[0].bytes_read, 4);
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
    assert_eq!(first.peak_inflight_weights, 1);
    assert_eq!(first.peak_inflight_bytes, 8);

    let demand = hierarchy.load(&request, &permit, &cancellation).unwrap();
    assert!(demand.cache_hit());
    let telemetry = hierarchy.telemetry();
    assert_eq!(telemetry.prefetch_useful_weights, 1);
    assert_eq!(telemetry.prefetch_useful_bytes, 8);

    let second = hierarchy
        .start_prefetch(vec![request], &permit, cancellation)
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert_eq!(second.cache_hits, 1);
    assert_eq!(second.peak_inflight_weights, 1);
    assert_eq!(second.peak_inflight_bytes, 8);
    let telemetry = hierarchy.telemetry();
    assert_eq!(telemetry.storage_reads, 1);
    assert_eq!(telemetry.prefetch_batches, 2);
    assert_eq!(telemetry.prefetch_peak_inflight_weights, 1);
    assert_eq!(telemetry.prefetch_peak_inflight_bytes, 8);
}

#[tokio::test]
async fn positional_demand_and_prefetch_serialize_one_materialization() {
    let (_directory, store) = weight_store_with_strategy(
        Dtype::F32,
        vec![2],
        &[0; 8],
        WeightReadStrategy::PositionalBuffered,
    );
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

    let key_lock = Arc::new(Mutex::new(()));
    lock(&hierarchy.inner.key_locks).insert(request.key.clone(), Arc::clone(&key_lock));
    let held = lock(&key_lock);
    let prefetch = hierarchy
        .start_prefetch(vec![request.clone()], &permit, cancellation.clone())
        .unwrap();
    let demand_hierarchy = hierarchy.clone();
    let demand_request = request.clone();
    let demand_permit = permit.clone();
    let demand_cancellation = cancellation.clone();
    let demand = tokio::task::spawn_blocking(move || {
        demand_hierarchy.load(&demand_request, &demand_permit, &demand_cancellation)
    });
    drop(held);

    let (prefetch, demand) = tokio::join!(prefetch.wait(), demand);
    let prefetch = prefetch.unwrap();
    let demand = demand.unwrap().unwrap();
    assert_eq!(prefetch.cache_hits + usize::from(demand.cache_hit()), 1);
    assert_eq!(hierarchy.telemetry().storage_reads, 1);
}

#[tokio::test]
async fn prefetch_tasks_are_bounded_and_release_capacity() {
    let (_directory, store) = weight_store(Dtype::F32, vec![2], &[0; 8]);
    let runtime = new_runtime();
    let hierarchy = WeightHierarchy::new(
        store,
        runtime.clone(),
        ResidencyPolicy {
            host_cache_bytes: 8,
            max_prefetch_tasks: 1,
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

    let key_lock = Arc::new(Mutex::new(()));
    lock(&hierarchy.inner.key_locks).insert(request.key.clone(), Arc::clone(&key_lock));
    let loading = lock(&key_lock);
    let first = hierarchy
        .start_prefetch(vec![request.clone()], &permit, cancellation.clone())
        .unwrap();
    assert!(hierarchy
        .start_prefetch(vec![request.clone()], &permit, cancellation.clone())
        .is_err());
    drop(loading);
    first.wait().await.unwrap();

    hierarchy
        .start_prefetch(vec![request], &permit, cancellation)
        .unwrap()
        .wait()
        .await
        .unwrap();
}

#[test]
fn residency_policy_rejects_unbounded_prefetch_controls() {
    assert!(ResidencyPolicy {
        max_prefetch_tasks: 0,
        ..ResidencyPolicy::default()
    }
    .validate()
    .is_err());
    assert!(ResidencyPolicy {
        max_prefetch_workers: 0,
        ..ResidencyPolicy::default()
    }
    .validate()
    .is_err());
    assert!(ResidencyPolicy {
        max_background_inflight_bytes: 0,
        ..ResidencyPolicy::default()
    }
    .validate()
    .is_err());
    assert!(ResidencyPolicy {
        route_coupling: super::super::coupling::RouteCouplingPolicy {
            max_entries: 0,
            ..super::super::coupling::RouteCouplingPolicy::default()
        },
        ..ResidencyPolicy::default()
    }
    .validate()
    .is_err());
}

#[test]
fn route_coupling_policy_has_a_backward_compatible_serde_default() {
    let mut serialized = serde_json::to_value(ResidencyPolicy::default()).unwrap();
    serialized.as_object_mut().unwrap().remove("routeCoupling");
    let restored: ResidencyPolicy = serde_json::from_value(serialized).unwrap();
    assert_eq!(
        restored.route_coupling,
        super::super::coupling::RouteCouplingPolicy::default()
    );
}

#[test]
fn background_byte_window_has_a_backward_compatible_serde_default() {
    let mut serialized = serde_json::to_value(ResidencyPolicy::default()).unwrap();
    serialized
        .as_object_mut()
        .unwrap()
        .remove("maxBackgroundInflightBytes");
    let restored: ResidencyPolicy = serde_json::from_value(serialized).unwrap();
    assert_eq!(
        restored.max_background_inflight_bytes,
        ResidencyPolicy::default().max_background_inflight_bytes
    );
}

#[test]
fn hierarchy_exposes_digest_bound_value_preserving_route_hints() {
    let (_directory, store) = weight_store(Dtype::F32, vec![1], &[0; 4]);
    let hierarchy = WeightHierarchy::new(
        store,
        new_runtime(),
        ResidencyPolicy {
            telemetry: TelemetryMode::Detailed,
            ..ResidencyPolicy::default()
        },
    )
    .unwrap();
    let source = RoutedExpertBatch::new(
        0,
        vec![vec![RoutedExpert {
            expert: 1,
            weight: 1.0,
        }]],
        4,
        1,
    )
    .unwrap();
    let target = RoutedExpertBatch::new(
        1,
        vec![vec![RoutedExpert {
            expert: 3,
            weight: 1.0,
        }]],
        4,
        1,
    )
    .unwrap();

    hierarchy.record_route_transition(&source, &target).unwrap();
    let hints = hierarchy.route_prefetch_hints(&source, 1, 1).unwrap();
    assert_eq!(hints.experts(), &[3]);
    assert_eq!(target.selections()[0][0].expert, 3);
    let evaluation = hierarchy
        .evaluate_route_prefetch_hints(&hints, &target)
        .unwrap();
    assert_eq!(evaluation.recall(), 1.0);

    let history = hierarchy.route_coupling_history().unwrap();
    assert_eq!(history.weights_sha256, hierarchy.store().sha256());
    assert_eq!(history.entries.len(), 1);
    assert_eq!(hierarchy.route_hint_telemetry().unwrap().evaluations, 1);
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
