use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::error::PowerError;

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn spec(family: &str, identity: char) -> ModelSessionSpec {
    ModelSessionSpec::new(
        ModelSessionBinding::new(
            ModelIdentity::new(family, "opaque-revision", digest(identity)),
            digest('e'),
        ),
        InferenceLimits::default(),
        32,
    )
    .unwrap()
}

fn policy(maximum_replicas: usize, maximum_resident_bytes: u64) -> ModelSessionPoolPolicy {
    ModelSessionPoolPolicy::new(3, maximum_resident_bytes, 1, 1)
        .unwrap()
        .with_max_replicas_per_session(maximum_replicas)
        .unwrap()
}

async fn wait_for_waiting(pool: &ModelSessionPool<u32>, expected: usize) {
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while pool.snapshot().waiting_replica_requests != expected {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("replica waiting count did not converge");
}

#[tokio::test]
async fn retired_replica_is_reconstructed_lazily_at_the_next_safe_boundary() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let loads = Arc::new(AtomicUsize::new(0));
    let first = pool
        .acquire_replica(
            spec("opaque.stateful", 'a'),
            &CancellationToken::new(),
            {
                let loads = Arc::clone(&loads);
                move |_runtime, _cancellation| async move {
                    Ok(loads.fetch_add(1, Ordering::Relaxed) + 1)
                }
            },
        )
        .await
        .unwrap();
    let declaration = first.declaration_sha256().to_string();
    assert_eq!(*first.value(), 1);

    first.retire();
    let retired = pool.snapshot();
    assert_eq!(retired.ready_replicas, 0);
    assert_eq!(retired.leased_replicas, 0);
    assert_eq!(retired.replicas_pending_reconstruction, 1);
    assert_eq!(retired.replica_retirements, 1);
    assert_eq!(retired.replica_reconstructions, 0);
    assert_eq!(retired.reserved_bytes, 32);

    let rebuilt = pool
        .acquire_replica(
            spec("opaque.stateful", 'a'),
            &CancellationToken::new(),
            {
                let loads = Arc::clone(&loads);
                move |_runtime, _cancellation| async move {
                    Ok(loads.fetch_add(1, Ordering::Relaxed) + 1)
                }
            },
        )
        .await
        .unwrap();
    assert_eq!(*rebuilt.value(), 2);
    assert_eq!(rebuilt.declaration_sha256(), declaration);
    let reconstructed = pool.snapshot();
    assert_eq!(reconstructed.ready_replicas, 1);
    assert_eq!(reconstructed.replicas_pending_reconstruction, 0);
    assert_eq!(reconstructed.replica_retirements, 1);
    assert_eq!(reconstructed.replica_reconstructions, 1);
    drop(rebuilt);

    let reused = pool
        .acquire_replica(
            spec("opaque.stateful", 'a'),
            &CancellationToken::new(),
            |_runtime, _cancellation| async {
                Err::<usize, _>(PowerError::InferenceFailed(
                    "a reconstructed replica must be reused".to_string(),
                ))
            },
        )
        .await
        .unwrap();
    assert_eq!(*reused.value(), 2);
    assert_eq!(loads.load(Ordering::Relaxed), 2);
}

#[tokio::test]
async fn failed_reconstruction_preserves_the_retired_slot_and_healthy_peer() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(2, 64)).unwrap();
    let retired = pool
        .acquire_replica(
            spec("vision.encoder", 'b'),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(11_u32) },
        )
        .await
        .unwrap();
    let healthy = pool
        .acquire_replica(
            spec("vision.encoder", 'b'),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(22_u32) },
        )
        .await
        .unwrap();
    retired.retire();

    assert!(matches!(
        pool.acquire_replica(
            spec("vision.encoder", 'b'),
            &CancellationToken::new(),
            |_runtime, _cancellation| async {
                Err::<u32, _>(PowerError::InferenceFailed(
                    "replacement initialization failed".to_string(),
                ))
            },
        )
        .await,
        Err(PowerError::InferenceFailed(_))
    ));
    let failed = pool.snapshot();
    assert_eq!(failed.registered_sessions, 1);
    assert_eq!(failed.ready_replicas, 1);
    assert_eq!(failed.leased_replicas, 1);
    assert_eq!(failed.replicas_pending_reconstruction, 1);
    assert_eq!(failed.replica_retirements, 1);
    assert_eq!(failed.replica_reconstructions, 0);
    assert_eq!(*healthy.value(), 22);

    let reconstructed = pool
        .acquire_replica(
            spec("vision.encoder", 'b'),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(33_u32) },
        )
        .await
        .unwrap();
    assert_eq!(*reconstructed.value(), 33);
    assert_eq!(*healthy.value(), 22);
    let recovered = pool.snapshot();
    assert_eq!(recovered.ready_replicas, 2);
    assert_eq!(recovered.leased_replicas, 2);
    assert_eq!(recovered.replicas_pending_reconstruction, 0);
    assert_eq!(recovered.replica_retirements, 1);
    assert_eq!(recovered.replica_reconstructions, 1);
}

#[tokio::test]
async fn cancelled_reconstruction_keeps_the_reserved_generation_retryable() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let replica = pool
        .acquire_replica(
            spec("multimodal.context", 'd'),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(41_u32) },
        )
        .await
        .unwrap();
    replica.retire();

    let cancellation = CancellationToken::new();
    let started = Arc::new(Notify::new());
    let loading = tokio::spawn({
        let pool = pool.clone();
        let cancellation = cancellation.clone();
        let started = Arc::clone(&started);
        async move {
            pool.acquire_replica(
                spec("multimodal.context", 'd'),
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

    let cancelled = pool.snapshot();
    assert_eq!(cancelled.registered_sessions, 1);
    assert_eq!(cancelled.ready_replicas, 0);
    assert_eq!(cancelled.leased_replicas, 0);
    assert_eq!(cancelled.replicas_pending_reconstruction, 1);
    assert_eq!(cancelled.replica_retirements, 1);
    assert_eq!(cancelled.replica_reconstructions, 0);
    assert_eq!(cancelled.reserved_bytes, 32);

    let recovered = pool
        .acquire_replica(
            spec("multimodal.context", 'd'),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(43_u32) },
        )
        .await
        .unwrap();
    assert_eq!(*recovered.value(), 43);
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.replicas_pending_reconstruction, 0);
    assert_eq!(snapshot.replica_retirements, 1);
    assert_eq!(snapshot.replica_reconstructions, 1);
}

#[tokio::test]
async fn queued_request_cannot_observe_state_retired_by_the_prior_lease() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 32)).unwrap();
    let active = pool
        .acquire_replica(
            spec("ocr.graph", 'f'),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(51_u32) },
        )
        .await
        .unwrap();
    let waiting = tokio::spawn({
        let pool = pool.clone();
        async move {
            pool.acquire_replica(
                spec("ocr.graph", 'f'),
                &CancellationToken::new(),
                |_runtime, _cancellation| async { Ok(53_u32) },
            )
            .await
        }
    });
    wait_for_waiting(&pool, 1).await;

    active.retire();
    let replacement = waiting.await.unwrap().unwrap();
    assert_eq!(*replacement.value(), 53);
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.replica_retirements, 1);
    assert_eq!(snapshot.replica_reconstructions, 1);
    assert_eq!(snapshot.replicas_pending_reconstruction, 0);
}

#[tokio::test]
async fn health_retirement_treats_model_families_as_opaque_identity() {
    let pool = ModelSessionPool::new(DevicePreference::Cpu, policy(1, 96)).unwrap();

    for (family, identity) in [
        ("language.decoder", 'a'),
        ("vision.encoder", 'b'),
        ("embedding.encoder", 'c'),
    ] {
        let replica = pool
            .acquire_replica(
                spec(family, identity),
                &CancellationToken::new(),
                |_runtime, _cancellation| async { Ok(1_u8) },
            )
            .await
            .unwrap();
        replica.retire();
        let replacement = pool
            .acquire_replica(
                spec(family, identity),
                &CancellationToken::new(),
                |_runtime, _cancellation| async { Ok(2_u8) },
            )
            .await
            .unwrap();
        assert_eq!(*replacement.value(), 2);
    }

    let snapshot = pool.snapshot();
    assert_eq!(snapshot.registered_sessions, 3);
    assert_eq!(snapshot.replica_retirements, 3);
    assert_eq!(snapshot.replica_reconstructions, 3);
    assert_eq!(snapshot.replicas_pending_reconstruction, 0);
    let debug = format!("{snapshot:?}");
    for family in ["language.decoder", "vision.encoder", "embedding.encoder"] {
        assert!(!debug.contains(family));
    }
}
