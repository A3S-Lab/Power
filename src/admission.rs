//! Shared inference admission control.
//!
//! Both the HTTP server and embedded runtimes use this controller so one
//! concurrency primitive defines request capacity. Server callers may wait for
//! capacity, while latency-sensitive embedded callers may fail fast.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio_util::sync::CancellationToken;

/// Reason a cancellation-aware admission request was not accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AdmissionError {
    #[error("admission waiting queue is full at {maximum} request(s)")]
    QueueFull { maximum: usize },
    #[error("admission was cancelled while waiting")]
    Cancelled,
    #[error("admission controller was closed")]
    Closed,
}

/// Content-free operational counters for one shared admission controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionSnapshot {
    pub active_limit: Option<usize>,
    pub waiting_limit: Option<usize>,
    pub active: usize,
    pub waiting: usize,
    pub peak_active: usize,
    pub peak_waiting: usize,
    pub admitted: u64,
    pub queue_rejections: u64,
    pub cancelled_waiters: u64,
}

/// Cloneable admission controller with an optional concurrency bound.
#[derive(Debug, Clone)]
pub struct AdmissionController {
    inner: Arc<AdmissionInner>,
}

#[derive(Debug)]
struct AdmissionInner {
    semaphore: Option<Arc<Semaphore>>,
    maximum: Option<usize>,
    queue_slots: Option<Arc<Semaphore>>,
    waiting_limit: Option<usize>,
    active: AtomicUsize,
    waiting: AtomicUsize,
    peak_active: AtomicUsize,
    peak_waiting: AtomicUsize,
    admitted: AtomicU64,
    queue_rejections: AtomicU64,
    cancelled_waiters: AtomicU64,
}

/// RAII permit returned for an admitted request.
#[derive(Debug)]
pub struct AdmissionPermit {
    _permit: Option<OwnedSemaphorePermit>,
    inner: Arc<AdmissionInner>,
    was_queued: bool,
}

impl AdmissionPermit {
    pub fn was_queued(&self) -> bool {
        self.was_queued
    }
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        let previous = self.inner.active.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(previous > 0, "admission active count underflowed");
    }
}

struct WaitingGuard {
    inner: Arc<AdmissionInner>,
    _queue_slot: Option<OwnedSemaphorePermit>,
}

impl Drop for WaitingGuard {
    fn drop(&mut self) {
        let previous = self.inner.waiting.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(previous > 0, "admission waiting count underflowed");
    }
}

impl AdmissionController {
    /// Creates a controller. `None` means unbounded admission.
    ///
    /// The legacy waiting path is intentionally unbounded. Embedded runtimes
    /// should use [`Self::new_bounded`] and [`Self::acquire_cancellable`].
    pub fn new(maximum: Option<usize>) -> Self {
        let maximum = maximum.map(|value| value.min(Semaphore::MAX_PERMITS));
        Self {
            inner: Arc::new(AdmissionInner {
                semaphore: maximum.map(|value| Arc::new(Semaphore::new(value))),
                maximum,
                queue_slots: None,
                waiting_limit: None,
                active: AtomicUsize::new(0),
                waiting: AtomicUsize::new(0),
                peak_active: AtomicUsize::new(0),
                peak_waiting: AtomicUsize::new(0),
                admitted: AtomicU64::new(0),
                queue_rejections: AtomicU64::new(0),
                cancelled_waiters: AtomicU64::new(0),
            }),
        }
    }

    /// Creates a concurrency-bound controller with a finite waiting queue.
    ///
    /// A zero waiting limit preserves fail-fast behavior when active capacity
    /// is exhausted.
    pub fn new_bounded(maximum: usize, maximum_waiting: usize) -> Self {
        let maximum = maximum.min(Semaphore::MAX_PERMITS);
        let maximum_waiting = maximum_waiting.min(Semaphore::MAX_PERMITS);
        Self {
            inner: Arc::new(AdmissionInner {
                semaphore: Some(Arc::new(Semaphore::new(maximum))),
                maximum: Some(maximum),
                queue_slots: Some(Arc::new(Semaphore::new(maximum_waiting))),
                waiting_limit: Some(maximum_waiting),
                active: AtomicUsize::new(0),
                waiting: AtomicUsize::new(0),
                peak_active: AtomicUsize::new(0),
                peak_waiting: AtomicUsize::new(0),
                admitted: AtomicU64::new(0),
                queue_rejections: AtomicU64::new(0),
                cancelled_waiters: AtomicU64::new(0),
            }),
        }
    }

    pub fn maximum(&self) -> Option<usize> {
        self.inner.maximum
    }

    pub fn maximum_waiting(&self) -> Option<usize> {
        self.inner.waiting_limit
    }

    pub fn snapshot(&self) -> AdmissionSnapshot {
        AdmissionSnapshot {
            active_limit: self.inner.maximum,
            waiting_limit: self.inner.waiting_limit,
            active: self.inner.active.load(Ordering::Relaxed),
            waiting: self.inner.waiting.load(Ordering::Relaxed),
            peak_active: self.inner.peak_active.load(Ordering::Relaxed),
            peak_waiting: self.inner.peak_waiting.load(Ordering::Relaxed),
            admitted: self.inner.admitted.load(Ordering::Relaxed),
            queue_rejections: self.inner.queue_rejections.load(Ordering::Relaxed),
            cancelled_waiters: self.inner.cancelled_waiters.load(Ordering::Relaxed),
        }
    }

    /// Attempts immediate admission and returns `None` at capacity.
    pub fn try_acquire(&self) -> Option<AdmissionPermit> {
        let Some(semaphore) = &self.inner.semaphore else {
            return Some(self.admitted_permit(None, false));
        };
        match Arc::clone(semaphore).try_acquire_owned() {
            Ok(permit) => Some(self.admitted_permit(Some(permit), false)),
            Err(TryAcquireError::NoPermits) => None,
            Err(TryAcquireError::Closed) => {
                // The semaphore is private and this type exposes no close
                // operation, so a closed controller is an internal invariant
                // violation rather than a recoverable capacity condition.
                debug_assert!(false, "private admission semaphore was closed");
                None
            }
        }
    }

    /// Waits until request capacity is available.
    pub async fn acquire(&self) -> AdmissionPermit {
        if let Some(permit) = self.try_acquire() {
            return permit;
        }
        let Some(semaphore) = &self.inner.semaphore else {
            return self.admitted_permit(None, false);
        };
        let waiting = self.register_waiter(None);
        match Arc::clone(semaphore).acquire_owned().await {
            Ok(permit) => {
                drop(waiting);
                self.admitted_permit(Some(permit), true)
            }
            Err(_) => {
                // The semaphore is private and cannot be closed through this
                // API. Avoid a production panic if that invariant changes.
                debug_assert!(false, "private admission semaphore was closed");
                drop(waiting);
                self.admitted_permit(None, true)
            }
        }
    }

    /// Waits through the configured finite queue and observes cancellation.
    pub async fn acquire_cancellable(
        &self,
        cancellation: &CancellationToken,
    ) -> std::result::Result<AdmissionPermit, AdmissionError> {
        if cancellation.is_cancelled() {
            return Err(AdmissionError::Cancelled);
        }
        if let Some(permit) = self.try_acquire() {
            return Ok(permit);
        }
        let queue_slot = match &self.inner.queue_slots {
            Some(slots) => match Arc::clone(slots).try_acquire_owned() {
                Ok(slot) => Some(slot),
                Err(TryAcquireError::NoPermits) => {
                    self.inner.queue_rejections.fetch_add(1, Ordering::Relaxed);
                    return Err(AdmissionError::QueueFull {
                        maximum: self.inner.waiting_limit.unwrap_or(0),
                    });
                }
                Err(TryAcquireError::Closed) => return Err(AdmissionError::Closed),
            },
            None => None,
        };
        let waiting = self.register_waiter(queue_slot);
        let semaphore = self
            .inner
            .semaphore
            .as_ref()
            .ok_or(AdmissionError::Closed)?;
        let permit = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                self.inner.cancelled_waiters.fetch_add(1, Ordering::Relaxed);
                return Err(AdmissionError::Cancelled);
            }
            result = Arc::clone(semaphore).acquire_owned() => {
                result.map_err(|_| AdmissionError::Closed)?
            }
        };
        if cancellation.is_cancelled() {
            drop(permit);
            self.inner.cancelled_waiters.fetch_add(1, Ordering::Relaxed);
            return Err(AdmissionError::Cancelled);
        }
        drop(waiting);
        Ok(self.admitted_permit(Some(permit), true))
    }

    fn register_waiter(&self, queue_slot: Option<OwnedSemaphorePermit>) -> WaitingGuard {
        let waiting = self.inner.waiting.fetch_add(1, Ordering::Relaxed) + 1;
        self.inner
            .peak_waiting
            .fetch_max(waiting, Ordering::Relaxed);
        WaitingGuard {
            inner: Arc::clone(&self.inner),
            _queue_slot: queue_slot,
        }
    }

    fn admitted_permit(
        &self,
        permit: Option<OwnedSemaphorePermit>,
        was_queued: bool,
    ) -> AdmissionPermit {
        let active = self.inner.active.fetch_add(1, Ordering::Relaxed) + 1;
        self.inner.peak_active.fetch_max(active, Ordering::Relaxed);
        self.inner.admitted.fetch_add(1, Ordering::Relaxed);
        AdmissionPermit {
            _permit: permit,
            inner: Arc::clone(&self.inner),
            was_queued,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

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
}
