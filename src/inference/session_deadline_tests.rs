use std::sync::Arc;

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::error::PowerError;

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn spec(character: char, resident_bytes: u64, maximum_waiters: usize) -> ModelSessionSpec {
    ModelSessionSpec::new(
        ModelSessionBinding::new(
            ModelIdentity::new("opaque-model", "revision-1", digest(character)),
            digest('e'),
        ),
        InferenceLimits {
            max_queued_requests: maximum_waiters,
            ..InferenceLimits::default()
        },
        resident_bytes,
    )
    .unwrap()
}

fn policy(maximum_sessions: usize, maximum_resident_bytes: u64) -> ModelSessionPoolPolicy {
    ModelSessionPoolPolicy::new(maximum_sessions, maximum_resident_bytes, 1, 1).unwrap()
}

async fn wait_for<F>(condition: F)
where
    F: Fn() -> bool,
{
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !condition() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("deadline state did not converge");
}

#[tokio::test]
async fn waiting_admission_uses_one_monotonic_deadline() {
    let runtime = EmbeddedRuntime::new(
        DevicePreference::Cpu,
        InferenceLimits {
            max_concurrent_requests: 1,
            max_queued_requests: 1,
            ..InferenceLimits::default()
        },
    )
    .unwrap();
    let active = runtime.begin(&CancellationToken::new()).unwrap();
    let result = runtime
        .begin_wait_until(
            &CancellationToken::new(),
            tokio::time::Instant::now() + std::time::Duration::from_millis(20),
        )
        .await;

    assert!(matches!(result, Err(PowerError::InferenceDeadlineExceeded)));
    let snapshot = runtime.admission_snapshot();
    assert_eq!(snapshot.active, 1);
    assert_eq!(snapshot.waiting, 0);
    assert_eq!(snapshot.deadline_expirations, 1);
    drop(active);
}

#[tokio::test]
async fn one_deadline_covers_model_and_shared_device_admission() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(2, 64)).unwrap();
    let first = pool
        .get_or_load(
            spec('a', 32, 1),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(1_u32) },
        )
        .await
        .unwrap();
    let second = pool
        .get_or_load(
            spec('b', 32, 1),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(2_u32) },
        )
        .await
        .unwrap();
    let active_device = first.runtime().begin(&CancellationToken::new()).unwrap();
    let waiting = tokio::spawn({
        let runtime = second.runtime().clone();
        async move {
            runtime
                .begin_wait_until(
                    &CancellationToken::new(),
                    tokio::time::Instant::now() + std::time::Duration::from_millis(500),
                )
                .await
        }
    });
    wait_for(|| pool.snapshot().device_admission.waiting == 1).await;
    assert_eq!(second.runtime().admission_snapshot().active, 1);

    assert!(matches!(
        waiting.await.unwrap(),
        Err(PowerError::InferenceDeadlineExceeded)
    ));
    assert_eq!(second.runtime().admission_snapshot().active, 0);
    let device = pool.snapshot().device_admission;
    assert_eq!(device.active, 1);
    assert_eq!(device.waiting, 0);
    assert_eq!(device.deadline_expirations, 1);

    drop(active_device);
    assert_eq!(pool.snapshot().device_admission.active, 0);
}

#[tokio::test]
async fn replica_deadline_expiry_is_aggregate_and_releases_the_lease_queue() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let active = pool
        .acquire_replica(
            spec('a', 32, 1),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(7_u32) },
        )
        .await
        .unwrap();
    let waiting = tokio::spawn({
        let pool = pool.clone();
        async move {
            pool.acquire_replica_until(
                spec('a', 32, 1),
                &CancellationToken::new(),
                tokio::time::Instant::now() + std::time::Duration::from_millis(500),
                |_runtime, _cancellation| async { Ok(8_u32) },
            )
            .await
        }
    });
    wait_for(|| pool.snapshot().waiting_replica_requests == 1).await;

    assert!(matches!(
        waiting.await.unwrap(),
        Err(PowerError::InferenceDeadlineExceeded)
    ));
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.leased_replicas, 1);
    assert_eq!(snapshot.waiting_replica_requests, 0);
    assert_eq!(snapshot.expired_replica_requests, 1);

    drop(active);
    let reused = pool
        .acquire_replica(
            spec('a', 32, 1),
            &CancellationToken::new(),
            |_runtime, _cancellation| async {
                Err::<u32, _>(PowerError::InferenceFailed(
                    "deadline expiry must not discard ready state".to_string(),
                ))
            },
        )
        .await
        .unwrap();
    assert_eq!(*reused.value(), 7);
}

#[tokio::test]
async fn replica_expiry_evidence_survives_removal_of_an_empty_entry() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let loading_cancellation = CancellationToken::new();
    let started = Arc::new(Notify::new());
    let loading = tokio::spawn({
        let pool = pool.clone();
        let loading_cancellation = loading_cancellation.clone();
        let started = Arc::clone(&started);
        async move {
            pool.acquire_replica(
                spec('a', 32, 1),
                &loading_cancellation,
                move |_runtime, _cancellation| async move {
                    started.notify_one();
                    std::future::pending::<crate::error::Result<u32>>().await
                },
            )
            .await
        }
    });
    started.notified().await;

    assert!(matches!(
        pool.acquire_replica_until(
            spec('a', 32, 1),
            &CancellationToken::new(),
            tokio::time::Instant::now() + std::time::Duration::from_millis(20),
            |_runtime, _cancellation| async { Ok(8_u32) },
        )
        .await,
        Err(PowerError::InferenceDeadlineExceeded)
    ));
    loading_cancellation.cancel();
    assert!(matches!(
        loading.await.unwrap(),
        Err(PowerError::InferenceCancelled)
    ));
    wait_for(|| pool.snapshot().registered_sessions == 0).await;
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.expired_replica_requests, 1);
    assert_eq!(snapshot.reserved_bytes, 0);
}
