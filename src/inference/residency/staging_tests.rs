use std::sync::{Arc, Mutex};
use std::time::Duration;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

use super::*;
use crate::inference::{
    DevicePreference, InferenceLimits, TelemetryMode, WeightReadStrategy, WeightStoreConfig,
};

const WEIGHT_A: &str = "layer.0.expert.0.a";
const WEIGHT_B: &str = "layer.0.expert.0.b";
const WEIGHT_C: &str = "layer.0.expert.1.a";

struct BlockingLock {
    release: Option<std::sync::mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl BlockingLock {
    fn acquire(mutex: Arc<Mutex<()>>) -> Self {
        let (ready_sender, ready_receiver) = std::sync::mpsc::sync_channel(1);
        let (release_sender, release_receiver) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let _held = lock(&mutex);
            let _ = ready_sender.send(());
            let _ = release_receiver.recv();
        });
        ready_receiver.recv().unwrap();
        Self {
            release: Some(release_sender),
            thread: Some(thread),
        }
    }

    fn release(mut self) {
        self.unblock();
    }

    fn unblock(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for BlockingLock {
    fn drop(&mut self) {
        self.unblock();
    }
}

fn staged_weight_store() -> (tempfile::TempDir, Arc<WeightStore>) {
    let directory = tempfile::tempdir().unwrap();
    let payloads = [
        1.0_f32.to_le_bytes(),
        2.0_f32.to_le_bytes(),
        3.0_f32.to_le_bytes(),
    ];
    let views = [WEIGHT_A, WEIGHT_B, WEIGHT_C]
        .into_iter()
        .zip(payloads.iter())
        .map(|(name, bytes)| {
            (
                name,
                TensorView::new(Dtype::F32, vec![1], bytes.as_slice()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
    let store = WeightStore::open_config(
        &WeightStoreConfig::new(directory.path())
            .with_primary_read_strategy(WeightReadStrategy::PositionalBuffered),
        &InferenceLimits::default(),
    )
    .unwrap();
    (directory, Arc::new(store))
}

fn new_runtime() -> EmbeddedRuntime {
    EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default()).unwrap()
}

fn request(layer: u32, name: &str) -> WeightRequest {
    WeightRequest::new(WeightKey::new(layer, name), PlacementPreference::Host)
}

fn group(layer: u32, names: &[&str]) -> StagedWeightGroupRequest {
    StagedWeightGroupRequest::new(names.iter().map(|name| request(layer, name)).collect())
}

fn value(weight: &ResidentWeight) -> f32 {
    weight.tensor().to_vec1::<f32>().unwrap()[0]
}

fn hierarchy(
    store: Arc<WeightStore>,
    runtime: &EmbeddedRuntime,
    telemetry: TelemetryMode,
) -> WeightHierarchy {
    WeightHierarchy::new(
        store,
        runtime.clone(),
        ResidencyPolicy {
            host_cache_bytes: 64,
            max_prefetch_tasks: 2,
            telemetry,
            ..ResidencyPolicy::default()
        },
    )
    .unwrap()
}

#[tokio::test]
async fn packed_u8_weight_record_survives_positional_staging_exactly() {
    const PACKED_EXPERT: &str = "layer.0.expert.7.packed";
    let directory = tempfile::tempdir().unwrap();
    // One model-owned record can place gate, up, down, and scale payloads next
    // to each other while Power treats the bytes as one opaque atomic weight.
    let packed = [
        0x47, 0x41, 0x54, 0x45, // GATE
        0x55, 0x50, 0x00, 0x01, // UP
        0x44, 0x4f, 0x57, 0x4e, // DOWN
        0x10, 0x20, 0x30, 0x40, // scales
    ];
    let view = TensorView::new(Dtype::U8, vec![packed.len()], packed.as_slice()).unwrap();
    serialize_to_file(
        [(PACKED_EXPERT, view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let store = Arc::new(
        WeightStore::open_config(
            &WeightStoreConfig::new(directory.path())
                .with_primary_read_strategy(WeightReadStrategy::PositionalBuffered),
            &InferenceLimits::default(),
        )
        .unwrap(),
    );
    let descriptor = store.descriptor(PACKED_EXPERT).unwrap();
    assert_eq!(descriptor.dtype, "u8");
    assert_eq!(descriptor.shape, [packed.len()]);
    assert_eq!(descriptor.bytes, packed.len() as u64);

    let runtime = new_runtime();
    let hierarchy = hierarchy(store, &runtime, TelemetryMode::Aggregate);
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let completion = hierarchy
        .start_staged_batch(vec![group(0, &[PACKED_EXPERT])], &permit, cancellation)
        .unwrap()
        .wait()
        .await
        .unwrap();

    assert_eq!(completion.groups.len(), 1);
    assert_eq!(completion.groups[0].weights().len(), 1);
    assert_eq!(
        completion.groups[0].weights()[0]
            .tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
        packed
    );
    assert_eq!(completion.report.loaded_weights, 1);
    assert_eq!(completion.report.bytes, packed.len() as u64);
}

#[tokio::test]
async fn staged_batch_exposes_resident_groups_before_background_loads() {
    let (_directory, store) = staged_weight_store();
    let runtime = new_runtime();
    let hierarchy = hierarchy(store, &runtime, TelemetryMode::Aggregate);
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    hierarchy
        .load(&request(0, WEIGHT_A), &permit, &cancellation)
        .unwrap();

    let key_lock = Arc::new(Mutex::new(()));
    lock(&hierarchy.inner.key_locks).insert(WeightKey::new(0, WEIGHT_B), Arc::clone(&key_lock));
    let held = lock(&key_lock);
    let mut batch = hierarchy
        .start_staged_batch(
            vec![group(0, &[WEIGHT_A]), group(0, &[WEIGHT_B])],
            &permit,
            cancellation.clone(),
        )
        .unwrap();

    let ready = batch.ready_groups();
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].canonical_index(), 0);
    assert_eq!(value(&ready[0].weights()[0]), 1.0);
    assert!(batch.ready_groups().is_empty());

    drop(held);
    let completion = batch.wait().await.unwrap();
    assert_eq!(
        completion
            .groups
            .iter()
            .map(StagedWeightGroup::canonical_index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(completion.report.ready_groups, 1);
    assert_eq!(completion.report.pending_groups, 1);
    assert_eq!(completion.report.resident_weights, 1);
    assert_eq!(completion.report.loaded_weights, 1);
    assert_eq!(completion.report.peak_inflight_weights, 1);
    assert_eq!(completion.report.peak_inflight_bytes, 4);
}

#[tokio::test]
async fn staged_group_is_atomic_and_only_loads_missing_weights() {
    let (_directory, store) = staged_weight_store();
    let runtime = new_runtime();
    let hierarchy = hierarchy(store, &runtime, TelemetryMode::Aggregate);
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    hierarchy
        .load(&request(0, WEIGHT_A), &permit, &cancellation)
        .unwrap();

    let key_lock = Arc::new(Mutex::new(()));
    lock(&hierarchy.inner.key_locks).insert(WeightKey::new(0, WEIGHT_B), Arc::clone(&key_lock));
    let held = lock(&key_lock);
    let mut batch = hierarchy
        .start_staged_batch(
            vec![group(0, &[WEIGHT_A, WEIGHT_B])],
            &permit,
            cancellation.clone(),
        )
        .unwrap();

    assert!(batch.ready_groups().is_empty());
    drop(held);
    let completion = batch.wait().await.unwrap();
    assert_eq!(completion.groups.len(), 1);
    assert_eq!(
        completion.groups[0]
            .weights()
            .iter()
            .map(value)
            .collect::<Vec<_>>(),
        vec![1.0, 2.0]
    );
    assert_eq!(completion.report.requested_weights, 2);
    assert_eq!(completion.report.resident_weights, 1);
    assert_eq!(completion.report.loaded_weights, 1);
    assert_eq!(hierarchy.telemetry().storage_reads, 2);
}

#[tokio::test]
async fn staged_completion_restores_canonical_order_after_out_of_order_readiness() {
    let (_directory, store) = staged_weight_store();
    let runtime = new_runtime();
    let hierarchy = hierarchy(store, &runtime, TelemetryMode::Aggregate);
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();

    let first_lock = Arc::new(Mutex::new(()));
    let second_lock = Arc::new(Mutex::new(()));
    lock(&hierarchy.inner.key_locks).insert(WeightKey::new(0, WEIGHT_A), Arc::clone(&first_lock));
    lock(&hierarchy.inner.key_locks).insert(WeightKey::new(0, WEIGHT_C), Arc::clone(&second_lock));
    let first_held = BlockingLock::acquire(first_lock);
    let second_held = BlockingLock::acquire(second_lock);
    let mut batch = hierarchy
        .start_staged_batch(
            vec![group(0, &[WEIGHT_A]), group(0, &[WEIGHT_C])],
            &permit,
            cancellation.clone(),
        )
        .unwrap();

    let release_second = async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        second_held.release();
    };
    let ((), out_of_order) = tokio::join!(
        release_second,
        tokio::time::timeout(Duration::from_secs(1), batch.next_ready_group())
    );
    let out_of_order = out_of_order.unwrap().unwrap().unwrap();
    assert_eq!(out_of_order.canonical_index(), 1);
    let release_first = async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        first_held.release();
    };
    let ((), first) = tokio::join!(
        release_first,
        tokio::time::timeout(Duration::from_secs(1), batch.next_ready_group())
    );
    let first = first.unwrap().unwrap().unwrap();
    assert_eq!(first.canonical_index(), 0);
    assert!(
        tokio::time::timeout(Duration::from_secs(1), batch.next_ready_group())
            .await
            .unwrap()
            .unwrap()
            .is_none()
    );

    let completion = batch.wait().await.unwrap();
    assert_eq!(
        completion
            .groups
            .iter()
            .map(StagedWeightGroup::canonical_index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(value(&completion.groups[0].weights()[0]), 1.0);
    assert_eq!(value(&completion.groups[1].weights()[0]), 3.0);
    assert!(completion.report.event_wait_nanos > 0);
    assert_eq!(
        hierarchy.telemetry().staged_event_wait_nanos,
        completion.report.event_wait_nanos
    );
}

#[tokio::test]
async fn staged_background_window_is_bounded_by_canonical_bytes() {
    let (_directory, store) = staged_weight_store();
    let runtime = new_runtime();
    let hierarchy = WeightHierarchy::new(
        store,
        runtime.clone(),
        ResidencyPolicy {
            host_cache_bytes: 64,
            max_prefetch_workers: 2,
            max_background_inflight_bytes: 4,
            telemetry: TelemetryMode::Aggregate,
            ..ResidencyPolicy::default()
        },
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();

    let completion = hierarchy
        .start_staged_batch(
            vec![group(0, &[WEIGHT_A]), group(0, &[WEIGHT_C])],
            &permit,
            cancellation,
        )
        .unwrap()
        .wait()
        .await
        .unwrap();

    assert_eq!(completion.report.requested_weights, 2);
    assert_eq!(completion.report.peak_inflight_weights, 1);
    assert_eq!(completion.report.peak_inflight_bytes, 4);
    let telemetry = hierarchy.telemetry();
    assert_eq!(telemetry.staged_peak_inflight_weights, 1);
    assert_eq!(telemetry.staged_peak_inflight_bytes, 4);
}

#[test]
fn staged_batch_rejects_the_complete_invalid_batch_before_cache_or_telemetry() {
    let invalid_cases = vec![
        Vec::new(),
        vec![StagedWeightGroupRequest::new(Vec::new())],
        vec![group(0, &[WEIGHT_A]), group(0, &[WEIGHT_A])],
        vec![group(0, &[WEIGHT_A]), group(1, &[WEIGHT_B])],
        vec![group(0, &[""])],
        vec![group(0, &["missing.weight"])],
    ];

    for groups in invalid_cases {
        let (_directory, store) = staged_weight_store();
        let runtime = new_runtime();
        let hierarchy = hierarchy(store, &runtime, TelemetryMode::Aggregate);
        let cancellation = CancellationToken::new();
        let permit = runtime.begin(&cancellation).unwrap();
        let before = hierarchy.telemetry();
        assert!(hierarchy
            .start_staged_batch(groups, &permit, cancellation.clone())
            .is_err());
        assert_eq!(hierarchy.telemetry(), before);
    }

    let (_directory, store) = staged_weight_store();
    let runtime = new_runtime();
    let hierarchy = hierarchy(store, &runtime, TelemetryMode::Aggregate);
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    hierarchy
        .load(&request(0, WEIGHT_A), &permit, &cancellation)
        .unwrap();
    let before = hierarchy.telemetry();
    assert!(hierarchy
        .start_staged_batch(
            vec![group(0, &[WEIGHT_A]), group(0, &["missing.weight"])],
            &permit,
            cancellation,
        )
        .is_err());
    assert_eq!(hierarchy.telemetry(), before);

    for policy in [
        ResidencyPolicy {
            host_cache_bytes: 64,
            max_prefetch_items: 1,
            telemetry: TelemetryMode::Aggregate,
            ..ResidencyPolicy::default()
        },
        ResidencyPolicy {
            host_cache_bytes: 64,
            max_prefetch_bytes: 3,
            telemetry: TelemetryMode::Aggregate,
            ..ResidencyPolicy::default()
        },
    ] {
        let (_directory, store) = staged_weight_store();
        let runtime = new_runtime();
        let hierarchy = WeightHierarchy::new(store, runtime.clone(), policy).unwrap();
        let cancellation = CancellationToken::new();
        let permit = runtime.begin(&cancellation).unwrap();
        let before = hierarchy.telemetry();
        assert!(hierarchy
            .start_staged_batch(vec![group(0, &[WEIGHT_A, WEIGHT_B])], &permit, cancellation,)
            .is_err());
        assert_eq!(hierarchy.telemetry(), before);
    }
}

#[tokio::test]
async fn cancelling_a_running_staged_batch_stops_pending_work() {
    let (_directory, store) = staged_weight_store();
    let runtime = new_runtime();
    let hierarchy = hierarchy(store, &runtime, TelemetryMode::Aggregate);
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let key_lock = Arc::new(Mutex::new(()));
    lock(&hierarchy.inner.key_locks).insert(WeightKey::new(0, WEIGHT_A), Arc::clone(&key_lock));
    let held = BlockingLock::acquire(key_lock);
    let batch = hierarchy
        .start_staged_batch(vec![group(0, &[WEIGHT_A])], &permit, cancellation.clone())
        .unwrap();

    cancellation.cancel();
    held.release();
    assert!(batch.wait().await.is_err());
    assert_eq!(hierarchy.telemetry().staged_batches, 0);
}

#[test]
fn staged_batch_rejects_foreign_permits_and_pre_cancelled_work() {
    let (_directory, store) = staged_weight_store();
    let runtime = new_runtime();
    let foreign_runtime = new_runtime();
    let hierarchy = hierarchy(store, &runtime, TelemetryMode::Aggregate);
    let cancellation = CancellationToken::new();
    let foreign_permit = foreign_runtime.begin(&cancellation).unwrap();
    assert!(hierarchy
        .start_staged_batch(
            vec![group(0, &[WEIGHT_A])],
            &foreign_permit,
            cancellation.clone(),
        )
        .is_err());

    let permit = runtime.begin(&cancellation).unwrap();
    cancellation.cancel();
    assert!(hierarchy
        .start_staged_batch(vec![group(0, &[WEIGHT_A])], &permit, cancellation)
        .is_err());
    assert_eq!(hierarchy.telemetry().staged_batches, 0);
}

#[tokio::test]
async fn dropping_staged_batch_cancels_work_and_releases_shared_admission() {
    let (_directory, store) = staged_weight_store();
    let runtime = new_runtime();
    let hierarchy = WeightHierarchy::new(
        store,
        runtime.clone(),
        ResidencyPolicy {
            host_cache_bytes: 64,
            max_prefetch_tasks: 1,
            ..ResidencyPolicy::default()
        },
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let key_lock = Arc::new(Mutex::new(()));
    lock(&hierarchy.inner.key_locks).insert(WeightKey::new(0, WEIGHT_A), Arc::clone(&key_lock));
    let held = BlockingLock::acquire(key_lock);

    let batch = hierarchy
        .start_staged_batch(vec![group(0, &[WEIGHT_A])], &permit, cancellation.clone())
        .unwrap();
    assert!(hierarchy
        .start_prefetch(vec![request(0, WEIGHT_B)], &permit, cancellation.clone(),)
        .is_err());
    drop(batch);

    let next = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if let Ok(task) =
                hierarchy.start_prefetch(vec![request(0, WEIGHT_B)], &permit, cancellation.clone())
            {
                break task;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    next.abort();
    drop(next);
    drop(held);
}

#[tokio::test]
async fn demand_prefetch_and_staging_share_per_key_materialization() {
    let (_directory, store) = staged_weight_store();
    let runtime = new_runtime();
    let hierarchy = hierarchy(store, &runtime, TelemetryMode::Aggregate);
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let key_lock = Arc::new(Mutex::new(()));
    lock(&hierarchy.inner.key_locks).insert(WeightKey::new(0, WEIGHT_A), Arc::clone(&key_lock));
    let held = lock(&key_lock);

    let staged = hierarchy
        .start_staged_batch(vec![group(0, &[WEIGHT_A])], &permit, cancellation.clone())
        .unwrap();
    let prefetched = hierarchy
        .start_prefetch(vec![request(0, WEIGHT_A)], &permit, cancellation.clone())
        .unwrap();
    let demand_hierarchy = hierarchy.clone();
    let demand_permit = permit.clone();
    let demand_cancellation = cancellation.clone();
    let demand = tokio::task::spawn_blocking(move || {
        demand_hierarchy.load(&request(0, WEIGHT_A), &demand_permit, &demand_cancellation)
    });
    drop(held);

    let (staged, prefetched, demand) = tokio::join!(staged.wait(), prefetched.wait(), demand);
    let staged = staged.unwrap();
    let prefetched = prefetched.unwrap();
    let demand = demand.unwrap().unwrap();
    assert_eq!(hierarchy.telemetry().storage_reads, 1);
    assert_eq!(
        staged.report.load_cache_hits + prefetched.cache_hits + usize::from(demand.cache_hit()),
        2
    );
}

#[tokio::test]
async fn staged_report_separates_service_background_and_foreground_wait() {
    let (_directory, store) = staged_weight_store();
    let runtime = new_runtime();
    let hierarchy = hierarchy(store, &runtime, TelemetryMode::Aggregate);
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let key_lock = Arc::new(Mutex::new(()));
    lock(&hierarchy.inner.key_locks).insert(WeightKey::new(0, WEIGHT_A), Arc::clone(&key_lock));
    let held = BlockingLock::acquire(key_lock);
    let batch = hierarchy
        .start_staged_batch(vec![group(0, &[WEIGHT_A])], &permit, cancellation.clone())
        .unwrap();

    let release = async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        held.release();
    };
    let ((), completion) = tokio::join!(release, batch.wait());
    let completion = completion.unwrap();
    assert!(completion.report.cumulative_service_nanos > 0);
    assert!(completion.report.background_elapsed_nanos > 0);
    assert!(completion.report.foreground_wait_nanos > 0);
    let telemetry = hierarchy.telemetry();
    assert_eq!(telemetry.staged_batches, 1);
    assert_eq!(telemetry.staged_groups, 1);
    assert_eq!(telemetry.staged_loaded_weights, 1);
    assert_eq!(
        telemetry.staged_service_nanos,
        completion.report.cumulative_service_nanos
    );
    assert_eq!(
        telemetry.staged_foreground_wait_nanos,
        completion.report.foreground_wait_nanos
    );
}

#[tokio::test]
async fn disabled_telemetry_keeps_staged_timing_private() {
    let (_directory, store) = staged_weight_store();
    let runtime = new_runtime();
    let hierarchy = hierarchy(store, &runtime, TelemetryMode::Disabled);
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let completion = hierarchy
        .start_staged_batch(vec![group(0, &[WEIGHT_A])], &permit, cancellation)
        .unwrap()
        .wait()
        .await
        .unwrap();

    assert_eq!(completion.report.cumulative_service_nanos, 0);
    assert_eq!(completion.report.background_elapsed_nanos, 0);
    assert_eq!(completion.report.event_wait_nanos, 0);
    assert_eq!(completion.report.foreground_wait_nanos, 0);
    let telemetry = hierarchy.telemetry();
    assert_eq!(telemetry.staged_batches, 0);
    assert_eq!(telemetry.staged_groups, 0);
    assert_eq!(telemetry.staged_loaded_weights, 0);
    assert_eq!(telemetry.staged_service_nanos, 0);
    assert_eq!(telemetry.staged_background_elapsed_nanos, 0);
    assert_eq!(telemetry.staged_event_wait_nanos, 0);
    assert_eq!(telemetry.staged_foreground_wait_nanos, 0);
    assert_eq!(telemetry.staged_peak_inflight_weights, 0);
    assert_eq!(telemetry.staged_peak_inflight_bytes, 0);
}

#[test]
fn staged_public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<StagedWeightGroupRequest>();
    assert_send_sync::<StagedWeightGroup>();
    assert_send_sync::<StagedWeightBatch>();
    assert_send_sync::<StagedWeightBatchCompletion>();
    assert_send_sync::<StagedWeightBatchReport>();
}
