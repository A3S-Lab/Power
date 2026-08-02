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
    assert_eq!(report.groups_released, 0);
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
fn plan_application_replaces_only_plan_owned_hot_set() {
    let (_directory, hierarchy, runtime) = hierarchy(8, 2);
    let first = hierarchy
        .plan_residency(&[
            candidate("hot", 100, &["hot"]),
            candidate("warm", 10, &["warm"]),
            candidate("cold", 1, &["cold"]),
            candidate("paired", 0, &["paired"]),
        ])
        .unwrap();
    let second = hierarchy
        .plan_residency(&[
            candidate("hot", 1, &["hot"]),
            candidate("warm", 0, &["warm"]),
            candidate("cold", 100, &["cold"]),
            candidate("paired", 10, &["paired"]),
        ])
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();

    hierarchy
        .apply_residency_plan(&first, &permit, &cancellation)
        .unwrap();
    let report = hierarchy
        .apply_residency_plan(&second, &permit, &cancellation)
        .unwrap();

    assert_eq!(report.groups_released, 2);
    assert_eq!(report.weights_released, 2);
    assert_eq!(hierarchy.active_residency_plan(), Some(second));
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

#[test]
fn clearing_plan_preserves_explicit_pins() {
    let (_directory, hierarchy, runtime) = hierarchy(8, 2);
    let plan = hierarchy
        .plan_residency(&[candidate("hot", 100, &["hot"])])
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();

    hierarchy
        .pin(
            &WeightRequest::new(WeightKey::new(0, "hot"), PlacementPreference::Host),
            &permit,
            &cancellation,
        )
        .unwrap();
    hierarchy
        .apply_residency_plan(&plan, &permit, &cancellation)
        .unwrap();
    assert_eq!(hierarchy.clear_residency_plan(), Some(plan));
    hierarchy.clear_unpinned();

    assert_eq!(hierarchy.telemetry().host_resident_bytes, 4);
    assert!(hierarchy
        .load(
            &WeightRequest::new(WeightKey::new(0, "hot"), PlacementPreference::Host),
            &permit,
            &cancellation,
        )
        .unwrap()
        .cache_hit());
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

#[test]
fn live_adaptation_uses_colibri_hysteresis_exactly() {
    let (_directory, hierarchy, runtime) = hierarchy(4, 1);
    let initial = hierarchy
        .plan_residency(&[
            candidate("incumbent", 100, &["hot"]),
            candidate("challenger", 1, &["cold"]),
        ])
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    hierarchy
        .apply_residency_plan(&initial, &permit, &cancellation)
        .unwrap();

    let at_threshold = hierarchy
        .plan_residency_adaptation(
            &[
                candidate("incumbent", 100, &["hot"]),
                candidate("challenger", 129, &["cold"]),
            ],
            &ResidencyAdaptationPolicy::default(),
        )
        .unwrap();
    assert!(at_threshold.is_noop());

    let above_threshold = hierarchy
        .plan_residency_adaptation(
            &[
                candidate("challenger", 130, &["cold"]),
                candidate("incumbent", 100, &["hot"]),
            ],
            &ResidencyAdaptationPolicy::default(),
        )
        .unwrap();
    assert_eq!(above_threshold.replacements().len(), 1);
    assert_eq!(above_threshold.replacements()[0].demoted_id, "incumbent");
    assert_eq!(above_threshold.replacements()[0].promoted_id, "challenger");
    assert_eq!(above_threshold.replacements()[0].heat_gain, 30);
    assert_eq!(
        above_threshold
            .plan()
            .groups
            .iter()
            .find(|group| group.id == "challenger")
            .unwrap()
            .tier,
        WeightTier::Host
    );
}

#[test]
fn live_adaptation_hysteresis_does_not_overflow() {
    let (_directory, hierarchy, runtime) = hierarchy(4, 1);
    let initial = hierarchy
        .plan_residency(&[
            candidate("incumbent", 100, &["hot"]),
            candidate("challenger", 1, &["cold"]),
        ])
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    hierarchy
        .apply_residency_plan(&initial, &permit, &cancellation)
        .unwrap();

    let adaptation = hierarchy
        .plan_residency_adaptation(
            &[
                candidate("incumbent", u64::MAX - u64::MAX / 10, &["hot"]),
                candidate("challenger", u64::MAX, &["cold"]),
            ],
            &ResidencyAdaptationPolicy::default(),
        )
        .unwrap();

    assert!(adaptation.is_noop());
}

#[test]
fn live_adaptation_is_deterministic_and_bounded() {
    let (_directory, hierarchy, runtime) = hierarchy(8, 2);
    let initial = hierarchy
        .plan_residency(&[
            candidate("first", 100, &["hot"]),
            candidate("second", 90, &["warm"]),
            candidate("third", 2, &["cold"]),
            candidate("fourth", 1, &["paired"]),
        ])
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    hierarchy
        .apply_residency_plan(&initial, &permit, &cancellation)
        .unwrap();
    let policy = ResidencyAdaptationPolicy {
        max_replacements: 1,
        ..ResidencyAdaptationPolicy::default()
    };

    let first = hierarchy
        .plan_residency_adaptation(
            &[
                candidate("first", 1, &["hot"]),
                candidate("second", 2, &["warm"]),
                candidate("third", 200, &["cold"]),
                candidate("fourth", 300, &["paired"]),
            ],
            &policy,
        )
        .unwrap();
    let second = hierarchy
        .plan_residency_adaptation(
            &[
                candidate("fourth", 300, &["paired"]),
                candidate("third", 200, &["cold"]),
                candidate("second", 2, &["warm"]),
                candidate("first", 1, &["hot"]),
            ],
            &policy,
        )
        .unwrap();

    assert_eq!(first, second);
    assert_eq!(first.replacements().len(), 1);
    assert_eq!(first.replacements()[0].demoted_id, "first");
    assert_eq!(first.replacements()[0].promoted_id, "fourth");
}

#[test]
fn live_adaptation_does_not_swap_incompatible_atomic_groups() {
    let (_directory, hierarchy, runtime) = hierarchy(8, 2);
    let initial = hierarchy
        .plan_residency(&[
            candidate("large", 100, &["hot", "paired"]),
            candidate("small", 1, &["cold"]),
        ])
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    hierarchy
        .apply_residency_plan(&initial, &permit, &cancellation)
        .unwrap();

    let adaptation = hierarchy
        .plan_residency_adaptation(
            &[
                candidate("large", 1, &["hot", "paired"]),
                candidate("small", 1_000, &["cold"]),
            ],
            &ResidencyAdaptationPolicy::default(),
        )
        .unwrap();

    assert!(adaptation.is_noop());
    assert_eq!(adaptation.plan(), &initial);
}

#[test]
fn live_adaptation_preserves_per_layer_entry_footprints() {
    let (_directory, hierarchy, runtime) = hierarchy(4, 1);
    let initial = hierarchy
        .plan_residency(&[
            candidate("layer-zero", 100, &["hot"]),
            ResidencyCandidate::new("layer-one", 1, vec![WeightKey::new(1, "cold")]),
        ])
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    hierarchy
        .apply_residency_plan(&initial, &permit, &cancellation)
        .unwrap();

    let adaptation = hierarchy
        .plan_residency_adaptation(
            &[
                candidate("layer-zero", 1, &["hot"]),
                ResidencyCandidate::new("layer-one", 1_000, vec![WeightKey::new(1, "cold")]),
            ],
            &ResidencyAdaptationPolicy::default(),
        )
        .unwrap();

    assert!(adaptation.is_noop());
}

#[test]
fn live_adaptation_application_preserves_manual_pins() {
    let (_directory, hierarchy, runtime) = hierarchy(8, 1);
    let initial = hierarchy
        .plan_residency(&[
            candidate("incumbent", 100, &["hot"]),
            candidate("challenger", 1, &["cold"]),
        ])
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    hierarchy
        .apply_residency_plan(&initial, &permit, &cancellation)
        .unwrap();
    hierarchy
        .pin(
            &WeightRequest::new(WeightKey::new(1, "paired"), PlacementPreference::Host),
            &permit,
            &cancellation,
        )
        .unwrap();
    let adaptation = hierarchy
        .plan_residency_adaptation(
            &[
                candidate("incumbent", 1, &["hot"]),
                candidate("challenger", 100, &["cold"]),
            ],
            &ResidencyAdaptationPolicy::default(),
        )
        .unwrap();

    hierarchy
        .apply_residency_adaptation(&adaptation, &permit, &cancellation)
        .unwrap();
    hierarchy.clear_residency_plan();
    hierarchy.clear_unpinned();

    assert_eq!(hierarchy.telemetry().host_resident_bytes, 4);
    assert!(hierarchy
        .load(
            &WeightRequest::new(WeightKey::new(1, "paired"), PlacementPreference::Host),
            &permit,
            &cancellation,
        )
        .unwrap()
        .cache_hit());
}

#[test]
fn stale_live_adaptation_fails_closed() {
    let (_directory, hierarchy, runtime) = hierarchy(4, 1);
    let initial = hierarchy
        .plan_residency(&[
            candidate("first", 100, &["hot"]),
            candidate("second", 1, &["cold"]),
        ])
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    hierarchy
        .apply_residency_plan(&initial, &permit, &cancellation)
        .unwrap();
    let adaptation = hierarchy
        .plan_residency_adaptation(
            &[
                candidate("first", 1, &["hot"]),
                candidate("second", 100, &["cold"]),
            ],
            &ResidencyAdaptationPolicy::default(),
        )
        .unwrap();
    let intervening = hierarchy
        .plan_residency(&[
            candidate("first", 1, &["hot"]),
            candidate("second", 100, &["cold"]),
        ])
        .unwrap();
    hierarchy
        .apply_residency_plan(&intervening, &permit, &cancellation)
        .unwrap();

    assert!(hierarchy
        .apply_residency_adaptation(&adaptation, &permit, &cancellation)
        .is_err());
    assert_eq!(hierarchy.active_residency_plan(), Some(intervening));
}

#[test]
fn invalid_live_adaptation_policy_is_rejected() {
    let policy = ResidencyAdaptationPolicy {
        max_replacements: 0,
        ..ResidencyAdaptationPolicy::default()
    };
    assert!(policy.validate().is_err());

    let policy = ResidencyAdaptationPolicy {
        hysteresis_basis_points: 10_001,
        ..ResidencyAdaptationPolicy::default()
    };
    assert!(policy.validate().is_err());
}

#[test]
fn public_live_adaptation_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<ResidencyAdaptationPolicy>();
    assert_send_sync::<ResidencyAdaptation>();
    assert_send_sync::<ResidencyReplacement>();
}
