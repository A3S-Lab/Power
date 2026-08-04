use std::path::Path;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

use super::*;
use crate::inference::{
    InferenceLimits, WeightSourceConfig, WeightSourceCoverage, WeightStoreConfig,
};

fn write_weight_file(root: &Path, file_name: &str, tensor_name: &str, byte: u8) {
    let values = [byte; 16];
    let view = TensorView::new(Dtype::U8, vec![values.len()], values.as_slice()).unwrap();
    serialize_to_file([(tensor_name, view)], None, &root.join(file_name)).unwrap();
}

fn open_three_file_store() -> (tempfile::TempDir, WeightStore) {
    let primary = tempfile::tempdir().unwrap();
    write_weight_file(primary.path(), "a.safetensors", "layer.a", 1);
    write_weight_file(primary.path(), "b.safetensors", "layer.b", 2);
    write_weight_file(primary.path(), "c.safetensors", "layer.c", 3);
    let store = WeightStore::open(primary.path(), &InferenceLimits::default()).unwrap();
    (primary, store)
}

fn approved_policy(max_bytes: u64, reserve_bytes: u64) -> WeightMirrorPolicy {
    WeightMirrorPolicy::new(max_bytes, reserve_bytes)
        .unwrap()
        .with_confidentiality(WeightMirrorConfidentiality::CallerManagedPlaintext)
}

#[test]
fn planner_selects_usage_ranked_files_deterministically_within_budget() {
    let (_primary, store) = open_three_file_store();
    let by_name = store
        .files()
        .iter()
        .map(|file| (file.relative_path.as_str(), file.bytes))
        .collect::<std::collections::BTreeMap<_, _>>();
    let budget = by_name["a.safetensors"] + by_name["b.safetensors"];
    let destination_parent = tempfile::tempdir().unwrap();
    let destination = destination_parent.path().join("mirror");
    let candidates = [
        WeightMirrorCandidate::new("c.safetensors", 10),
        WeightMirrorCandidate::new("a.safetensors", 30),
        WeightMirrorCandidate::new("b.safetensors", 20),
    ];
    let policy = approved_policy(budget, 0);

    let first = store
        .plan_partial_mirror(&destination, &candidates, &policy)
        .unwrap();
    let second = store
        .plan_partial_mirror(&destination, &candidates, &policy)
        .unwrap();

    assert!(first.admitted);
    assert_eq!(first.files, second.files);
    assert_eq!(first.selected_bytes, second.selected_bytes);
    assert_eq!(first.rejection, second.rejection);
    assert_eq!(first.selected_bytes, budget);
    assert_eq!(first.copy_bytes, budget);
    assert_eq!(
        first
            .files
            .iter()
            .map(|file| file.relative_path.as_str())
            .collect::<Vec<_>>(),
        ["a.safetensors", "b.safetensors"]
    );
}

#[test]
fn staging_requires_explicit_plaintext_authority_and_opens_as_exact_partial_replica() {
    let (primary, store) = open_three_file_store();
    let destination_parent = tempfile::tempdir().unwrap();
    let destination = destination_parent.path().join("mirror");
    let file = &store.files()[0];
    let candidates = [WeightMirrorCandidate::new(&file.relative_path, 1)];
    let denied = WeightMirrorPolicy::new(file.bytes, 0).unwrap();

    let denied_plan = store
        .plan_partial_mirror(&destination, &candidates, &denied)
        .unwrap();
    assert_eq!(
        denied_plan.rejection,
        Some(WeightMirrorPlanRejection::PlaintextStagingDenied)
    );
    assert!(store
        .stage_partial_mirror_blocking(
            &destination,
            &candidates,
            &denied,
            &CancellationToken::new(),
        )
        .is_err());
    assert!(!destination.exists());

    let approved = approved_policy(file.bytes, 0);
    let report = store
        .stage_partial_mirror_blocking(
            &destination,
            &candidates,
            &approved,
            &CancellationToken::new(),
        )
        .unwrap();
    assert_eq!(report.copied_files, 1);
    assert_eq!(report.copied_bytes, file.bytes);
    assert_eq!(report.reused_files, 0);
    assert!(destination.join(&file.relative_path).is_file());

    let reopened = WeightStore::open_config(
        &WeightStoreConfig::new(primary.path()).with_replica(
            WeightSourceConfig::new(&destination).with_coverage(WeightSourceCoverage::Partial),
        ),
        &InferenceLimits::default(),
    )
    .unwrap();
    assert_eq!(reopened.sources().len(), 2);
    assert_eq!(
        reopened.sources()[1].coverage,
        WeightSourceCoverage::Partial
    );
    assert_eq!(reopened.sources()[1].verified_files, 1);
    assert_eq!(std::fs::read_dir(&destination).unwrap().count(), 1);
}

#[test]
fn exact_files_are_reused_and_conflicting_files_are_never_overwritten() {
    let (_primary, store) = open_three_file_store();
    let destination_parent = tempfile::tempdir().unwrap();
    let destination = destination_parent.path().join("mirror");
    let file = &store.files()[0];
    let candidates = [WeightMirrorCandidate::new(&file.relative_path, 1)];
    let policy = approved_policy(file.bytes, 0);
    let cancellation = CancellationToken::new();

    store
        .stage_partial_mirror_blocking(&destination, &candidates, &policy, &cancellation)
        .unwrap();
    let reused = store
        .stage_partial_mirror_blocking(&destination, &candidates, &policy, &cancellation)
        .unwrap();
    assert_eq!(reused.copied_files, 0);
    assert_eq!(reused.reused_files, 1);
    assert_eq!(reused.reused_bytes, file.bytes);

    let target = destination.join(&file.relative_path);
    std::fs::write(&target, b"corrupt").unwrap();
    let before = std::fs::read(&target).unwrap();
    let conflict = store
        .plan_partial_mirror(&destination, &candidates, &policy)
        .unwrap();
    assert_eq!(
        conflict.rejection,
        Some(WeightMirrorPlanRejection::DestinationConflict)
    );
    assert!(store
        .stage_partial_mirror_blocking(&destination, &candidates, &policy, &cancellation)
        .is_err());
    assert_eq!(std::fs::read(&target).unwrap(), before);
}

#[test]
fn cancellation_and_space_reserve_fail_before_publication() {
    let (_primary, store) = open_three_file_store();
    let destination_parent = tempfile::tempdir().unwrap();
    let destination = destination_parent.path().join("mirror");
    let file = &store.files()[0];
    let candidates = [WeightMirrorCandidate::new(&file.relative_path, 1)];
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    assert!(store
        .stage_partial_mirror_blocking(
            &destination,
            &candidates,
            &approved_policy(file.bytes, 0),
            &cancellation,
        )
        .is_err());
    assert!(!destination.exists());

    let plan = store
        .plan_partial_mirror(
            &destination,
            &candidates,
            &approved_policy(file.bytes, u64::MAX),
        )
        .unwrap();
    assert_eq!(
        plan.rejection,
        Some(WeightMirrorPlanRejection::InsufficientSpace)
    );
}

#[test]
fn invalid_candidate_sets_fail_closed() {
    let (_primary, store) = open_three_file_store();
    let destination_parent = tempfile::tempdir().unwrap();
    let destination = destination_parent.path().join("mirror");
    let policy = approved_policy(store.bytes(), 0);

    for candidates in [
        vec![WeightMirrorCandidate::new("missing.safetensors", 1)],
        vec![WeightMirrorCandidate::new("../a.safetensors", 1)],
        vec![WeightMirrorCandidate::new("a.safetensors", 0)],
        vec![
            WeightMirrorCandidate::new("a.safetensors", 1),
            WeightMirrorCandidate::new("a.safetensors", 2),
        ],
    ] {
        assert!(store
            .plan_partial_mirror(&destination, &candidates, &policy)
            .is_err());
    }
}

#[test]
fn source_mutation_is_detected_before_any_file_is_published() {
    let (_primary, store) = open_three_file_store();
    let destination_parent = tempfile::tempdir().unwrap();
    let destination = destination_parent.path().join("mirror");
    let file = &store.files()[0];
    let candidates = [WeightMirrorCandidate::new(&file.relative_path, 1)];
    let source = store.root().join(&file.relative_path);
    let mut changed = std::fs::read(&source).unwrap();
    let last = changed.last_mut().unwrap();
    *last ^= 0xff;
    std::fs::write(&source, changed).unwrap();

    assert!(matches!(
        store.stage_partial_mirror_blocking(
            &destination,
            &candidates,
            &approved_policy(file.bytes, 0),
            &CancellationToken::new(),
        ),
        Err(PowerError::IntegrityCheckFailed { .. })
    ));
    assert!(destination.is_dir());
    assert_eq!(std::fs::read_dir(&destination).unwrap().count(), 0);
}

#[test]
fn nested_shards_stage_without_escaping_and_unselected_shards_conflict() {
    let primary = tempfile::tempdir().unwrap();
    std::fs::create_dir(primary.path().join("nested")).unwrap();
    write_weight_file(primary.path(), "nested/hot.safetensors", "layer.hot", 1);
    write_weight_file(primary.path(), "cold.safetensors", "layer.cold", 2);
    let store = WeightStore::open(primary.path(), &InferenceLimits::default()).unwrap();
    let hot = store
        .files()
        .iter()
        .find(|file| file.relative_path.contains("hot.safetensors"))
        .unwrap();
    let cold = store
        .files()
        .iter()
        .find(|file| file.relative_path == "cold.safetensors")
        .unwrap();
    let destination_parent = tempfile::tempdir().unwrap();
    let destination = destination_parent.path().join("mirror");
    let candidates = [WeightMirrorCandidate::new(&hot.relative_path, 1)];
    let policy = approved_policy(hot.bytes, 0);

    store
        .stage_partial_mirror_blocking(
            &destination,
            &candidates,
            &policy,
            &CancellationToken::new(),
        )
        .unwrap();
    assert!(destination.join(&hot.relative_path).is_file());

    std::fs::copy(
        store.root().join(&cold.relative_path),
        destination.join(&cold.relative_path),
    )
    .unwrap();
    let plan = store
        .plan_partial_mirror(&destination, &candidates, &policy)
        .unwrap();
    assert_eq!(
        plan.rejection,
        Some(WeightMirrorPlanRejection::DestinationConflict)
    );
    assert_eq!(
        plan.conflicts.as_slice(),
        std::slice::from_ref(&cold.relative_path)
    );
}

#[test]
fn mirror_staging_resolves_files_from_every_primary_shard_root() {
    let primary = tempfile::tempdir().unwrap();
    let shard = tempfile::tempdir().unwrap();
    write_weight_file(primary.path(), "first.safetensors", "layer.first", 1);
    write_weight_file(shard.path(), "second.safetensors", "layer.second", 2);
    let store = WeightStore::open_config(
        &WeightStoreConfig::new(primary.path()).with_primary_shard_root(shard.path()),
        &InferenceLimits::default(),
    )
    .unwrap();
    let selected = store
        .files()
        .iter()
        .find(|file| file.relative_path == "second.safetensors")
        .unwrap();
    let destination_parent = tempfile::tempdir().unwrap();
    let destination = destination_parent.path().join("mirror");

    store
        .stage_partial_mirror_blocking(
            &destination,
            &[WeightMirrorCandidate::new(&selected.relative_path, 1)],
            &approved_policy(selected.bytes, 0),
            &CancellationToken::new(),
        )
        .unwrap();

    assert_eq!(
        std::fs::read(destination.join("second.safetensors")).unwrap(),
        std::fs::read(shard.path().join("second.safetensors")).unwrap()
    );
    assert!(store
        .plan_partial_mirror(
            shard.path().join("nested-mirror"),
            &[WeightMirrorCandidate::new(&selected.relative_path, 1)],
            &approved_policy(selected.bytes, 0),
        )
        .is_err());
}

#[test]
fn serialized_policy_defaults_to_denying_plaintext_staging() {
    let policy: WeightMirrorPolicy = serde_json::from_value(serde_json::json!({
        "maxBytes": 1024,
        "reserveBytes": 0
    }))
    .unwrap();

    assert_eq!(
        policy.confidentiality,
        WeightMirrorConfidentiality::DenyPlaintext
    );
}

#[test]
fn mirror_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<WeightMirrorCandidate>();
    assert_send_sync::<WeightMirrorPolicy>();
    assert_send_sync::<WeightMirrorPlan>();
    assert_send_sync::<WeightMirrorStageReport>();
}
