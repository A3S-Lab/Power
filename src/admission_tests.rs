use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::admission::{AdmissionController, AdmissionError};

async fn wait_for_waiting(controller: &AdmissionController, expected: usize) {
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if controller.snapshot().waiting == expected {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("admission waiting count did not converge");
}

#[tokio::test]
async fn clones_share_capacity() {
    let controller = AdmissionController::new(Some(1));
    let clone = controller.clone();
    let permit = controller.try_acquire().unwrap();
    assert!(clone.try_acquire().is_none());
    drop(permit);
    assert!(clone.try_acquire().is_some());
}

#[tokio::test]
async fn waiting_acquire_is_released_by_drop() {
    let controller = AdmissionController::new(Some(1));
    let permit = controller.try_acquire().unwrap();
    let waiter = tokio::spawn({
        let controller = controller.clone();
        async move { controller.acquire().await }
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert!(!waiter.is_finished());
    drop(permit);
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), waiter)
            .await
            .is_ok()
    );
}

#[test]
fn unbounded_controller_always_admits() {
    let controller = AdmissionController::new(None);
    let permits = (0..1_000)
        .map(|_| controller.try_acquire().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(permits.len(), 1_000);
}

#[test]
fn excessive_capacity_is_safely_clamped() {
    let controller = AdmissionController::new(Some(usize::MAX));
    assert_eq!(controller.maximum(), Some(Semaphore::MAX_PERMITS));
}

#[tokio::test]
async fn bounded_queue_rejects_overflow_and_reports_counts() {
    let controller = AdmissionController::new_bounded(1, 1);
    let active = controller.try_acquire().unwrap();
    let queued_cancellation = CancellationToken::new();
    let queued = tokio::spawn({
        let controller = controller.clone();
        let cancellation = queued_cancellation.clone();
        async move { controller.acquire_cancellable(&cancellation).await }
    });
    wait_for_waiting(&controller, 1).await;

    let overflow = controller
        .acquire_cancellable(&CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(overflow, AdmissionError::QueueFull { maximum: 1 });
    assert_eq!(controller.snapshot().queue_rejections, 1);

    drop(active);
    let admitted = queued.await.unwrap().unwrap();
    assert!(admitted.was_queued());
    let snapshot = controller.snapshot();
    assert_eq!(snapshot.active, 1);
    assert_eq!(snapshot.waiting, 0);
    assert_eq!(snapshot.peak_waiting, 1);
    assert_eq!(snapshot.admitted, 2);
}

#[tokio::test]
async fn cancellation_and_future_drop_release_bounded_queue_slots() {
    let controller = AdmissionController::new_bounded(1, 1);
    let _active = controller.try_acquire().unwrap();

    let cancellation = CancellationToken::new();
    let cancelled_waiter = tokio::spawn({
        let controller = controller.clone();
        let cancellation = cancellation.clone();
        async move { controller.acquire_cancellable(&cancellation).await }
    });
    wait_for_waiting(&controller, 1).await;
    cancellation.cancel();
    assert_eq!(
        cancelled_waiter.await.unwrap().unwrap_err(),
        AdmissionError::Cancelled
    );
    wait_for_waiting(&controller, 0).await;
    assert_eq!(controller.snapshot().cancelled_waiters, 1);

    let dropped_waiter = tokio::spawn({
        let controller = controller.clone();
        async move {
            controller
                .acquire_cancellable(&CancellationToken::new())
                .await
        }
    });
    wait_for_waiting(&controller, 1).await;
    dropped_waiter.abort();
    let _ = dropped_waiter.await;
    wait_for_waiting(&controller, 0).await;

    let replacement = tokio::spawn({
        let controller = controller.clone();
        async move {
            controller
                .acquire_cancellable(&CancellationToken::new())
                .await
        }
    });
    wait_for_waiting(&controller, 1).await;
    replacement.abort();
    let _ = replacement.await;
    wait_for_waiting(&controller, 0).await;
}

#[tokio::test]
async fn monotonic_deadlines_expire_waiters_and_release_queue_capacity() {
    let controller = AdmissionController::new_bounded(1, 1);
    let active = controller.try_acquire().unwrap();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(20);

    let expired = controller
        .acquire_cancellable_until(&CancellationToken::new(), deadline)
        .await
        .unwrap_err();
    assert_eq!(expired, AdmissionError::DeadlineExceeded);
    let snapshot = controller.snapshot();
    assert_eq!(snapshot.waiting, 0);
    assert_eq!(snapshot.active, 1);
    assert_eq!(snapshot.deadline_expirations, 1);

    drop(active);
    assert!(controller.try_acquire().is_some());
}

#[tokio::test]
async fn an_expired_deadline_never_admits_even_when_capacity_is_ready() {
    let controller = AdmissionController::new_bounded(1, 1);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert_eq!(
        controller
            .acquire_cancellable_until(&cancellation, tokio::time::Instant::now())
            .await
            .unwrap_err(),
        AdmissionError::Cancelled
    );
    assert_eq!(controller.snapshot().deadline_expirations, 0);

    assert_eq!(
        controller
            .acquire_cancellable_until(&CancellationToken::new(), tokio::time::Instant::now())
            .await
            .unwrap_err(),
        AdmissionError::DeadlineExceeded
    );
    let snapshot = controller.snapshot();
    assert_eq!(snapshot.active, 0);
    assert_eq!(snapshot.admitted, 0);
    assert_eq!(snapshot.deadline_expirations, 1);
}
