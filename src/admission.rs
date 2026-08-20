//! Shared inference admission control.
//!
//! Both the HTTP server and embedded runtimes use this controller so one
//! concurrency primitive defines request capacity. Server callers may wait for
//! capacity, while latency-sensitive embedded callers may fail fast.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

/// Reason a cancellation-aware admission request was not accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AdmissionError {
    #[error("admission waiting queue is full at {maximum} request(s)")]
    QueueFull { maximum: usize },
    #[error("admission was cancelled while waiting")]
    Cancelled,
    #[error("admission deadline was exceeded")]
    DeadlineExceeded,
    #[error("admission controller was closed")]
    Closed,
}

/// Content-free operational counters for one shared admission controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
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
    pub deadline_expirations: u64,
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
    deadline_expirations: AtomicU64,
}

/// RAII permit returned for an admitted request.
#[derive(Debug)]
pub struct AdmissionPermit {
    _permit: Option<OwnedSemaphorePermit>,
    inner: Arc<AdmissionInner>,
    was_queued: bool,
}

enum ImmediatePermit {
    Unbounded,
    Bounded(OwnedSemaphorePermit),
}

impl ImmediatePermit {
    fn into_owned(self) -> Option<OwnedSemaphorePermit> {
        match self {
            Self::Unbounded => None,
            Self::Bounded(permit) => Some(permit),
        }
    }
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
                deadline_expirations: AtomicU64::new(0),
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
                deadline_expirations: AtomicU64::new(0),
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
            deadline_expirations: self.inner.deadline_expirations.load(Ordering::Relaxed),
        }
    }

    /// Attempts immediate admission and returns `None` at capacity.
    pub fn try_acquire(&self) -> Option<AdmissionPermit> {
        match self.try_acquire_immediate() {
            Ok(Some(permit)) => Some(self.admitted_permit(permit.into_owned(), false)),
            Ok(None) => None,
            Err(AdmissionError::Closed) => {
                debug_assert!(false, "private admission semaphore was closed");
                None
            }
            Err(_) => {
                debug_assert!(false, "immediate admission returned a waiting-only error");
                None
            }
        }
    }

    fn try_acquire_immediate(
        &self,
    ) -> std::result::Result<Option<ImmediatePermit>, AdmissionError> {
        let Some(semaphore) = &self.inner.semaphore else {
            return Ok(Some(ImmediatePermit::Unbounded));
        };
        match Arc::clone(semaphore).try_acquire_owned() {
            Ok(permit) => Ok(Some(ImmediatePermit::Bounded(permit))),
            Err(TryAcquireError::NoPermits) => Ok(None),
            Err(TryAcquireError::Closed) => Err(AdmissionError::Closed),
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
        self.acquire_cancellable_inner(cancellation, None).await
    }

    /// Waits through the finite queue until one monotonic deadline.
    ///
    /// The deadline is checked before immediate admission, while queued, and
    /// again after the semaphore wakes. Cancellation has deterministic
    /// precedence when both signals are already observable.
    pub async fn acquire_cancellable_until(
        &self,
        cancellation: &CancellationToken,
        deadline: Instant,
    ) -> std::result::Result<AdmissionPermit, AdmissionError> {
        self.acquire_cancellable_inner(cancellation, Some(deadline))
            .await
    }

    async fn acquire_cancellable_inner(
        &self,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> std::result::Result<AdmissionPermit, AdmissionError> {
        if cancellation.is_cancelled() {
            return Err(AdmissionError::Cancelled);
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(self.deadline_exceeded());
        }
        if let Some(permit) = self.try_acquire_immediate()? {
            if cancellation.is_cancelled() {
                return Err(AdmissionError::Cancelled);
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(self.deadline_exceeded());
            }
            return Ok(self.admitted_permit(permit.into_owned(), false));
        }
        if cancellation.is_cancelled() {
            return Err(AdmissionError::Cancelled);
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(self.deadline_exceeded());
        }
        let queue_slot = match &self.inner.queue_slots {
            Some(slots) => match Arc::clone(slots).try_acquire_owned() {
                Ok(slot) => Some(slot),
                Err(TryAcquireError::NoPermits) => {
                    if cancellation.is_cancelled() {
                        return Err(AdmissionError::Cancelled);
                    }
                    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                        return Err(self.deadline_exceeded());
                    }
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
        let permit = match deadline {
            Some(deadline) => tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    self.inner.cancelled_waiters.fetch_add(1, Ordering::Relaxed);
                    return Err(AdmissionError::Cancelled);
                }
                _ = tokio::time::sleep_until(deadline) => {
                    return Err(self.deadline_exceeded());
                }
                result = Arc::clone(semaphore).acquire_owned() => {
                    result.map_err(|_| AdmissionError::Closed)?
                }
            },
            None => tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    self.inner.cancelled_waiters.fetch_add(1, Ordering::Relaxed);
                    return Err(AdmissionError::Cancelled);
                }
                result = Arc::clone(semaphore).acquire_owned() => {
                    result.map_err(|_| AdmissionError::Closed)?
                }
            },
        };
        if cancellation.is_cancelled() {
            drop(permit);
            self.inner.cancelled_waiters.fetch_add(1, Ordering::Relaxed);
            return Err(AdmissionError::Cancelled);
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            drop(permit);
            return Err(self.deadline_exceeded());
        }
        drop(waiting);
        Ok(self.admitted_permit(Some(permit), true))
    }

    fn deadline_exceeded(&self) -> AdmissionError {
        self.inner
            .deadline_expirations
            .fetch_add(1, Ordering::Relaxed);
        AdmissionError::DeadlineExceeded
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
