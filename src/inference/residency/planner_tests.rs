use std::sync::Arc;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::inference::{DevicePreference, InferenceLimits};

fn hierarchy(
    host_bytes: u64,
    max_entries_per_layer: usize,
) -> (tempfile::TempDir, WeightHierarchy, EmbeddedRuntime) {
    let directory = tempfile::tempdir().unwrap();
    let bytes = [[0_u8; 4], [1_u8; 4], [2_u8; 4], [3_u8; 4]];
    let names = ["hot", "warm", "cold", "paired"];
    let views = names
        .iter()
        .zip(bytes.iter())
        .map(|(name, data)| {
            (
                *name,
                TensorView::new(Dtype::F32, vec![1], data.as_slice()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
    let limits = InferenceLimits::default();
    let store = Arc::new(WeightStore::open(directory.path(), &limits).unwrap());
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap();
    let hierarchy = WeightHierarchy::new(
        store,
        runtime.clone(),
        ResidencyPolicy {
            host_cache_bytes: host_bytes,
            max_entries_per_layer,
            ..ResidencyPolicy::default()
        },
    )
    .unwrap();
    (directory, hierarchy, runtime)
}

fn candidate(id: &str, heat: u64, names: &[&str]) -> ResidencyCandidate {
    ResidencyCandidate::new(
        id,
        heat,
        names.iter().map(|name| WeightKey::new(0, *name)).collect(),
    )
}

#[test]
fn hottest_groups_fill_host_budget_deterministically_on_cpu() {
    let (_directory, hierarchy, _runtime) = hierarchy(8, 2);
    let first = hierarchy
        .plan_residency(&[
            candidate("cold", 1, &["cold"]),
            candidate("hot", 100, &["hot"]),
            candidate("warm", 10, &["warm"]),
        ])
        .unwrap();
    let second = hierarchy
        .plan_residency(&[
            candidate("warm", 10, &["warm"]),
            candidate("hot", 100, &["hot"]),
            candidate("cold", 1, &["cold"]),
        ])
        .unwrap();

    assert_eq!(first, second);
    assert_eq!(first.host_planned_bytes, 8);
    assert_eq!(first.device_planned_bytes, 0);
    assert_eq!(first.groups[0].id, "hot");
    assert_eq!(first.groups[0].tier, WeightTier::Host);
    assert_eq!(first.groups[1].id, "warm");
    assert_eq!(first.groups[1].tier, WeightTier::Host);
    assert_eq!(first.groups[2].tier, WeightTier::Storage);
}

#[test]
fn atomic_group_is_not_partially_pinned() {
    let (_directory, hierarchy, _runtime) = hierarchy(4, 2);
    let plan = hierarchy
        .plan_residency(&[
            candidate("large-hot", 100, &["hot", "paired"]),
            candidate("small-cold", 1, &["cold"]),
        ])
        .unwrap();

    assert_eq!(plan.groups[0].tier, WeightTier::Storage);
    assert_eq!(plan.groups[0].bytes, 8);
    assert_eq!(plan.groups[1].tier, WeightTier::Host);
}

#[test]
fn plan_application_pins_the_selected_hot_set() {
    let (_directory, hierarchy, runtime) = hierarchy(8, 2);
    let plan = hierarchy
        .plan_residency(&[
            candidate("hot", 100, &["hot"]),
            candidate("cold", 1, &["cold"]),
        ])
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let report = hierarchy
        .apply_residency_plan(&plan, &permit, &cancellation)
        .unwrap();

    assert_eq!(report.groups_pinned, 2);
    assert_eq!(report.weights_pinned, 2);
    assert_eq!(report.host_bytes, 8);
    let hot = hierarchy
        .load(
            &WeightRequest::new(WeightKey::new(0, "hot"), PlacementPreference::Host),
            &permit,
            &cancellation,
        )
        .unwrap();
    assert!(hot.cache_hit());
}

#[test]
fn plan_is_bound_to_exact_weights_and_rejects_duplicate_tensors() {
    let (_directory, hierarchy, runtime) = hierarchy(8, 2);
    assert!(hierarchy
        .plan_residency(&[
            candidate("first", 2, &["hot"]),
            candidate("second", 1, &["hot"]),
        ])
        .is_err());

    let mut plan = hierarchy
        .plan_residency(&[candidate("hot", 2, &["hot"])])
        .unwrap();
    plan.weights_sha256 = "0".repeat(64);
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    assert!(hierarchy
        .apply_residency_plan(&plan, &permit, &cancellation)
        .is_err());
    assert_eq!(hierarchy.telemetry().host_resident_bytes, 0);
}

#[test]
fn failed_application_restores_prior_cache_and_pin_state() {
    let (_directory, hierarchy, runtime) = hierarchy(8, 3);
    let plan = hierarchy
        .plan_residency(&[
            candidate("hot", 100, &["hot"]),
            candidate("warm", 10, &["warm"]),
        ])
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();

    hierarchy
        .pin(
            &WeightRequest::new(WeightKey::new(0, "cold"), PlacementPreference::Host),
            &permit,
            &cancellation,
        )
        .unwrap();
    assert!(hierarchy
        .apply_residency_plan(&plan, &permit, &cancellation)
        .is_err());

    let telemetry = hierarchy.telemetry();
    assert_eq!(telemetry.host_resident_bytes, 4);
    assert!(hierarchy
        .load(
            &WeightRequest::new(WeightKey::new(0, "cold"), PlacementPreference::Host),
            &permit,
            &cancellation,
        )
        .unwrap()
        .cache_hit());
    assert!(!hierarchy
        .load(
            &WeightRequest::new(WeightKey::new(0, "hot"), PlacementPreference::Host),
            &permit,
            &cancellation,
        )
        .unwrap()
        .cache_hit());
}
