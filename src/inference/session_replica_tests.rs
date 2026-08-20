use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::error::PowerError;

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn spec(resident_bytes: u64) -> ModelSessionSpec {
    ModelSessionSpec::new(
        ModelSessionBinding::new(
            ModelIdentity::new("test-model", "revision-1", digest('a')),
            digest('e'),
        ),
        InferenceLimits::default(),
        resident_bytes,
    )
    .unwrap()
}

fn family_spec(family: &str, identity: char) -> ModelSessionSpec {
    ModelSessionSpec::new(
        ModelSessionBinding::new(
            ModelIdentity::new(family, "opaque-revision", digest(identity)),
            digest('e'),
        ),
        InferenceLimits::default(),
        16,
    )
    .unwrap()
}

fn queued_spec(resident_bytes: u64, maximum_waiters: usize) -> ModelSessionSpec {
    let limits = InferenceLimits {
        max_queued_requests: maximum_waiters,
        ..InferenceLimits::default()
    };
    ModelSessionSpec::new(
        ModelSessionBinding::new(
            ModelIdentity::new("test-model", "revision-1", digest('a')),
            digest('e'),
        ),
        limits,
        resident_bytes,
    )
    .unwrap()
}

fn policy(max_replicas: usize, max_resident_bytes: u64) -> ModelSessionPoolPolicy {
    ModelSessionPoolPolicy::new(1, max_resident_bytes, 1, 1)
        .unwrap()
        .with_max_replicas_per_session(max_replicas)
        .unwrap()
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
    .expect("replica pool state did not converge");
}

#[tokio::test]
async fn replicas_are_lazy_exclusive_independent_and_share_one_device_gate() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(2, 64)).unwrap();
    let loads = Arc::new(AtomicUsize::new(0));
    let first = pool
        .acquire_replica(spec(32), &CancellationToken::new(), {
            let loads = Arc::clone(&loads);
            move |_runtime, _cancellation| async move {
                Ok(Mutex::new(loads.fetch_add(1, Ordering::Relaxed) + 1))
            }
        })
        .await
        .unwrap();
    *first.value().lock().unwrap() = 11;
    let second = pool
        .acquire_replica(spec(32), &CancellationToken::new(), {
            let loads = Arc::clone(&loads);
            move |_runtime, _cancellation| async move {
                Ok(Mutex::new(loads.fetch_add(1, Ordering::Relaxed) + 1))
            }
        })
        .await
        .unwrap();
    *second.value().lock().unwrap() = 22;

    assert!(!std::ptr::eq(first.value(), second.value()));
    assert_eq!(first.declaration_sha256(), second.declaration_sha256());
    assert_eq!(loads.load(Ordering::Relaxed), 2);
    let device = first.runtime().begin(&CancellationToken::new()).unwrap();
    assert!(second.runtime().begin(&CancellationToken::new()).is_err());
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.maximum_replicas_per_session, 2);
    assert_eq!(snapshot.reserved_replicas, 2);
    assert_eq!(snapshot.ready_replicas, 2);
    assert_eq!(snapshot.leased_replicas, 2);
    assert_eq!(snapshot.waiting_replica_requests, 0);
    assert_eq!(snapshot.reserved_bytes, 64);
    assert!(!format!("{first:?}").contains("test-model"));
    assert!(!format!("{first:?}").contains(&digest('a')));
    drop(device);
    drop(first);
    drop(second);

    let reused = pool
        .acquire_replica(
            spec(32),
            &CancellationToken::new(),
            |_runtime, _cancellation| async {
                Err::<Mutex<usize>, _>(PowerError::InferenceFailed(
                    "an initialized replica must be reused".to_string(),
                ))
            },
        )
        .await
        .unwrap();
    assert!(matches!(*reused.value().lock().unwrap(), 11 | 22));
    assert_eq!(loads.load(Ordering::Relaxed), 2);
    assert_eq!(pool.snapshot().leased_replicas, 1);
    drop(reused);
    assert_eq!(pool.snapshot().leased_replicas, 0);
}

#[tokio::test]
async fn model_families_are_opaque_session_identity_not_runtime_dispatch() {
    let policy = ModelSessionPoolPolicy::new(3, 48, 1, 1)
        .unwrap()
        .with_max_replicas_per_session(1)
        .unwrap();
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy).unwrap();
    let cancellation = CancellationToken::new();

    let language = pool
        .acquire_replica(
            family_spec("language.decoder", 'a'),
            &cancellation,
            |_runtime, _cancellation| async { Ok(Mutex::new("language")) },
        )
        .await
        .unwrap();
    let vision = pool
        .acquire_replica(
            family_spec("vision.encoder", 'b'),
            &cancellation,
            |_runtime, _cancellation| async { Ok(Mutex::new("vision")) },
        )
        .await
        .unwrap();
    let embedding = pool
        .acquire_replica(
            family_spec("embedding.encoder", 'c'),
            &cancellation,
            |_runtime, _cancellation| async { Ok(Mutex::new("embedding")) },
        )
        .await
        .unwrap();

    assert_eq!(language.binding().model.family, "language.decoder");
    assert_eq!(vision.binding().model.family, "vision.encoder");
    assert_eq!(embedding.binding().model.family, "embedding.encoder");
    assert_ne!(language.declaration_sha256(), vision.declaration_sha256());
    assert_ne!(vision.declaration_sha256(), embedding.declaration_sha256());
    assert_eq!(pool.snapshot().registered_sessions, 3);
}

#[tokio::test]
async fn worst_case_replica_residency_is_reserved_before_loading() {
    let loads = Arc::new(AtomicUsize::new(0));
    let too_small = ModelSessionPool::new(DevicePreference::Cpu, policy(2, 63)).unwrap();
    let result = too_small
        .acquire_replica(spec(32), &CancellationToken::new(), {
            let loads = Arc::clone(&loads);
            move |_runtime, _cancellation| async move {
                loads.fetch_add(1, Ordering::Relaxed);
                Ok(1_u32)
            }
        })
        .await;
    assert!(matches!(
        result,
        Err(PowerError::ModelSessionPoolFull {
            maximum_sessions: 1,
            maximum_resident_bytes: 63,
        })
    ));
    assert_eq!(loads.load(Ordering::Relaxed), 0);
    assert_eq!(too_small.snapshot().registered_sessions, 0);

    let exact = ModelSessionPool::new(DevicePreference::Cpu, policy(2, 64)).unwrap();
    let replica = exact
        .acquire_replica(
            spec(32),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(1_u32) },
        )
        .await
        .unwrap();
    let snapshot = exact.snapshot();
    assert_eq!(snapshot.registered_sessions, 1);
    assert_eq!(snapshot.reserved_replicas, 2);
    assert_eq!(snapshot.ready_replicas, 1);
    assert_eq!(snapshot.reserved_bytes, 64);
    let replicated_declaration = replica.declaration_sha256().to_string();
    drop(replica);

    let single = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 64)).unwrap();
    let single_replica = single
        .acquire_replica(
            spec(32),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(1_u32) },
        )
        .await
        .unwrap();
    assert_ne!(single_replica.declaration_sha256(), replicated_declaration);
}

#[tokio::test]
async fn cancelled_replica_wait_releases_queue_and_reuses_the_ready_value() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let active = pool
        .acquire_replica(
            spec(32),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(7_u32) },
        )
        .await
        .unwrap();
    let cancellation = CancellationToken::new();
    let waiting = tokio::spawn({
        let pool = pool.clone();
        let cancellation = cancellation.clone();
        async move {
            pool.acquire_replica(spec(32), &cancellation, |_runtime, _cancellation| async {
                Ok(9_u32)
            })
            .await
        }
    });
    wait_for(|| pool.snapshot().waiting_replica_requests == 1).await;
    cancellation.cancel();
    assert!(matches!(
        waiting.await.unwrap(),
        Err(PowerError::InferenceCancelled)
    ));
    wait_for(|| pool.snapshot().waiting_replica_requests == 0).await;
    assert_eq!(pool.snapshot().leased_replicas, 1);
    drop(active);

    let reused = pool
        .acquire_replica(
            spec(32),
            &CancellationToken::new(),
            |_runtime, _cancellation| async {
                Err::<u32, _>(PowerError::InferenceFailed(
                    "cancelled wait must not discard the ready replica".to_string(),
                ))
            },
        )
        .await
        .unwrap();
    assert_eq!(*reused.value(), 7);
}

#[tokio::test]
async fn replica_waiting_queue_rejects_work_beyond_its_declared_bound() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let active = pool
        .acquire_replica(
            queued_spec(32, 1),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(7_u32) },
        )
        .await
        .unwrap();
    let waiting_cancellation = CancellationToken::new();
    let waiting = tokio::spawn({
        let pool = pool.clone();
        let waiting_cancellation = waiting_cancellation.clone();
        async move {
            pool.acquire_replica(
                queued_spec(32, 1),
                &waiting_cancellation,
                |_runtime, _cancellation| async { Ok(8_u32) },
            )
            .await
        }
    });
    wait_for(|| pool.snapshot().waiting_replica_requests == 1).await;

    assert!(matches!(
        pool.acquire_replica(
            queued_spec(32, 1),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(9_u32) },
        )
        .await,
        Err(PowerError::InferenceQueueFull { maximum: 1 })
    ));
    assert_eq!(pool.snapshot().waiting_replica_requests, 1);

    waiting_cancellation.cancel();
    assert!(matches!(
        waiting.await.unwrap(),
        Err(PowerError::InferenceCancelled)
    ));
    drop(active);
}

#[tokio::test]
async fn cancelled_or_dropped_initialization_releases_replica_budget() {
    let cancelled_pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let cancellation = CancellationToken::new();
    let started = Arc::new(Notify::new());
    let loading = tokio::spawn({
        let pool = cancelled_pool.clone();
        let cancellation = cancellation.clone();
        let started = Arc::clone(&started);
        async move {
            pool.acquire_replica(
                spec(32),
                &cancellation,
                move |_runtime, _cancellation| async move {
                    started.notify_one();
                    std::future::pending::<crate::error::Result<u32>>().await
                },
            )
            .await
        }
    });
    started.notified().await;
    assert_eq!(cancelled_pool.snapshot().reserved_bytes, 32);
    cancellation.cancel();
    assert!(matches!(
        loading.await.unwrap(),
        Err(PowerError::InferenceCancelled)
    ));
    wait_for(|| cancelled_pool.snapshot().registered_sessions == 0).await;
    assert_eq!(cancelled_pool.snapshot().reserved_bytes, 0);

    let dropped_pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let started = Arc::new(Notify::new());
    let loading = tokio::spawn({
        let pool = dropped_pool.clone();
        let started = Arc::clone(&started);
        async move {
            pool.acquire_replica(
                spec(32),
                &CancellationToken::new(),
                move |_runtime, _cancellation| async move {
                    started.notify_one();
                    std::future::pending::<crate::error::Result<u32>>().await
                },
            )
            .await
        }
    });
    started.notified().await;
    loading.abort();
    let _ = loading.await;
    wait_for(|| dropped_pool.snapshot().registered_sessions == 0).await;
    assert_eq!(dropped_pool.snapshot().reserved_bytes, 0);

    let recovered = dropped_pool
        .acquire_replica(
            spec(32),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(9_u32) },
        )
        .await
        .unwrap();
    assert_eq!(*recovered.value(), 9);
}

#[tokio::test]
async fn legacy_shared_session_api_is_rejected_when_replica_mode_is_enabled() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(2, 64)).unwrap();
    assert!(matches!(
        pool.get_or_load(
            spec(32),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(1_u32) },
        )
        .await,
        Err(PowerError::InvalidRequest(_))
    ));
    assert_eq!(pool.snapshot().registered_sessions, 0);
}

#[tokio::test]
async fn one_exact_session_cannot_mix_shared_and_exclusive_access() {
    let exclusive_pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let exclusive = exclusive_pool
        .acquire_replica(
            spec(32),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(Mutex::new(1_u32)) },
        )
        .await
        .unwrap();
    assert!(matches!(
        exclusive_pool
            .get_or_load(
                spec(32),
                &CancellationToken::new(),
                |_runtime, _cancellation| async { Ok(Mutex::new(2_u32)) },
            )
            .await,
        Err(PowerError::InvalidRequest(_))
    ));
    drop(exclusive);

    let shared_pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let shared = shared_pool
        .get_or_load(
            spec(32),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(Mutex::new(1_u32)) },
        )
        .await
        .unwrap();
    assert!(matches!(
        shared_pool
            .acquire_replica(
                spec(32),
                &CancellationToken::new(),
                |_runtime, _cancellation| async { Ok(Mutex::new(2_u32)) },
            )
            .await,
        Err(PowerError::InvalidRequest(_))
    ));
    drop(shared);
}

#[test]
fn replica_policy_defaults_are_bounded_and_backward_compatible() {
    let single = ModelSessionPoolPolicy::new(1, 32, 1, 1).unwrap();
    assert_eq!(single.max_replicas_per_session, 1);
    assert!(single.clone().with_max_replicas_per_session(0).is_err());
    assert!(single.clone().with_max_replicas_per_session(257).is_err());

    let mut legacy = serde_json::to_value(&single).unwrap();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("maxReplicasPerSession");
    let restored: ModelSessionPoolPolicy = serde_json::from_value(legacy).unwrap();
    assert_eq!(restored.max_replicas_per_session, 1);

    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ModelSessionReplica<Mutex<u32>>>();
}
