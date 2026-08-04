use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::error::PowerError;

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn binding(character: char) -> ModelSessionBinding {
    ModelSessionBinding::new(
        ModelIdentity::new("test-model", "revision-1", digest(character)),
        digest('e'),
    )
}

fn spec(character: char, resident_bytes: u64) -> ModelSessionSpec {
    ModelSessionSpec::new(
        binding(character),
        InferenceLimits::default(),
        resident_bytes,
    )
    .unwrap()
}

fn policy(max_sessions: usize, max_resident_bytes: u64) -> ModelSessionPoolPolicy {
    ModelSessionPoolPolicy::new(max_sessions, max_resident_bytes, 1, 1).unwrap()
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
    .expect("session pool state did not converge");
}

#[tokio::test]
async fn exact_session_is_initialized_once_and_reused() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(2, 64)).unwrap();
    let loads = Arc::new(AtomicUsize::new(0));
    let first = pool
        .get_or_load(spec('a', 32), &CancellationToken::new(), {
            let loads = Arc::clone(&loads);
            move |_runtime, _cancellation| async move {
                loads.fetch_add(1, Ordering::Relaxed);
                Ok(7_u32)
            }
        })
        .await
        .unwrap();
    let second = pool
        .get_or_load(spec('a', 32), &CancellationToken::new(), {
            let loads = Arc::clone(&loads);
            move |_runtime, _cancellation| async move {
                loads.fetch_add(1, Ordering::Relaxed);
                Ok(9_u32)
            }
        })
        .await
        .unwrap();

    assert_eq!(*first.value(), 7);
    assert_eq!(*second.value(), 7);
    assert_eq!(loads.load(Ordering::Relaxed), 1);
    assert_eq!(first.declaration_sha256(), second.declaration_sha256());
    assert!(std::ptr::eq(first.value(), second.value()));
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.registered_sessions, 1);
    assert_eq!(snapshot.ready_sessions, 1);
    assert_eq!(snapshot.reserved_bytes, 32);
}

#[tokio::test]
async fn concurrent_callers_share_one_initialization() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(2, 64)).unwrap();
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let loads = Arc::new(AtomicUsize::new(0));
    let first = tokio::spawn({
        let pool = pool.clone();
        let started = Arc::clone(&started);
        let release = Arc::clone(&release);
        let loads = Arc::clone(&loads);
        async move {
            pool.get_or_load(
                spec('a', 32),
                &CancellationToken::new(),
                move |_runtime, _cancellation| async move {
                    loads.fetch_add(1, Ordering::Relaxed);
                    started.notify_one();
                    release.notified().await;
                    Ok(11_u32)
                },
            )
            .await
        }
    });
    started.notified().await;
    let second = tokio::spawn({
        let pool = pool.clone();
        let loads = Arc::clone(&loads);
        async move {
            pool.get_or_load(
                spec('a', 32),
                &CancellationToken::new(),
                move |_runtime, _cancellation| async move {
                    loads.fetch_add(1, Ordering::Relaxed);
                    Ok(13_u32)
                },
            )
            .await
        }
    });
    tokio::task::yield_now().await;
    release.notify_waiters();

    assert_eq!(*first.await.unwrap().unwrap().value(), 11);
    assert_eq!(*second.await.unwrap().unwrap().value(), 11);
    assert_eq!(loads.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn count_bytes_and_exact_resource_identity_are_bounded() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    pool.get_or_load(
        spec('a', 32),
        &CancellationToken::new(),
        |_runtime, _cancellation| async { Ok(1_u32) },
    )
    .await
    .unwrap();

    let overflow = pool
        .get_or_load(
            spec('b', 1),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(2_u32) },
        )
        .await;
    assert!(matches!(
        overflow,
        Err(PowerError::ModelSessionPoolFull {
            maximum_sessions: 1,
            maximum_resident_bytes: 32,
        })
    ));

    let conflicting = ModelSessionSpec::new(
        binding('a'),
        InferenceLimits {
            max_context_tokens: 16,
            ..InferenceLimits::default()
        },
        32,
    )
    .unwrap();
    assert!(pool
        .get_or_load(
            conflicting,
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(3_u32) },
        )
        .await
        .is_err());
}

#[tokio::test]
async fn cancelled_initialization_releases_an_unshared_pool_slot() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let cancellation = CancellationToken::new();
    let started = Arc::new(Notify::new());
    let loading = tokio::spawn({
        let pool = pool.clone();
        let cancellation = cancellation.clone();
        let started = Arc::clone(&started);
        async move {
            pool.get_or_load(
                spec('a', 32),
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
    cancellation.cancel();
    assert!(matches!(
        loading.await.unwrap(),
        Err(PowerError::InferenceCancelled)
    ));
    assert_eq!(pool.snapshot().registered_sessions, 0);

    pool.get_or_load(
        spec('b', 32),
        &CancellationToken::new(),
        |_runtime, _cancellation| async { Ok(2_u32) },
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn dropped_initialization_future_releases_an_unshared_pool_slot() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let started = Arc::new(Notify::new());
    let loading = tokio::spawn({
        let pool = pool.clone();
        let started = Arc::clone(&started);
        async move {
            pool.get_or_load(
                spec('a', 32),
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
    wait_for(|| pool.snapshot().registered_sessions == 0).await;

    pool.get_or_load(
        spec('b', 32),
        &CancellationToken::new(),
        |_runtime, _cancellation| async { Ok(2_u32) },
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn distinct_model_sessions_share_the_device_gate() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(2, 64)).unwrap();
    let first = pool
        .get_or_load(
            spec('a', 32),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(1_u32) },
        )
        .await
        .unwrap();
    let second = pool
        .get_or_load(
            spec('b', 32),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(2_u32) },
        )
        .await
        .unwrap();
    let active = first.runtime().begin(&CancellationToken::new()).unwrap();
    assert!(second.runtime().begin(&CancellationToken::new()).is_err());

    let cancellation = CancellationToken::new();
    let waiting = tokio::spawn({
        let runtime = second.runtime().clone();
        let cancellation = cancellation.clone();
        async move { runtime.begin_wait(&cancellation).await }
    });
    wait_for(|| pool.snapshot().device_admission.waiting == 1).await;
    assert_eq!(second.runtime().admission_snapshot().active, 1);
    cancellation.cancel();
    assert!(waiting.await.unwrap().is_err());
    assert_eq!(second.runtime().admission_snapshot().active, 0);
    assert_eq!(pool.snapshot().device_admission.waiting, 0);
    drop(active);
    assert_eq!(pool.snapshot().device_admission.active, 0);
}

#[test]
fn declarations_are_device_and_limit_bound_without_debug_identity_leaks() {
    let spec = spec('a', 32);
    let cpu = spec
        .declaration_sha256(RuntimeDeviceIdentity {
            kind: RuntimeDeviceKind::Cpu,
            ordinal: None,
        })
        .unwrap();
    let cuda = spec
        .declaration_sha256(RuntimeDeviceIdentity {
            kind: RuntimeDeviceKind::Cuda,
            ordinal: Some(0),
        })
        .unwrap();
    assert_ne!(cpu, cuda);
    assert!(!format!("{:?}", spec.binding()).contains("test-model"));
}

#[test]
fn session_pool_public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ModelSessionPool<u32>>();
    assert_send_sync::<ModelSession<u32>>();
    assert_send_sync::<ModelSessionPoolPolicy>();
    assert_send_sync::<ModelSessionSpec>();
}
