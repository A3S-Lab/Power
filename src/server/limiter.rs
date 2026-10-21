//! Request admission control.
//!
//! Bounds the number of inference requests running concurrently, the way
//! vLLM's scheduler caps `max_num_seqs`. Without a bound, a burst of
//! concurrent requests each pins a KV cache and working set — on a
//! memory-constrained box (especially a TEE EPC) that is a fast path to OOM.
//!
//! A request acquires a [`RequestPermit`] before inference and holds it until
//! the response (including the streamed body) completes. When the configured
//! legacy library constructor receives 0 it remains unbounded. HTTP state uses
//! a positive execution limit, finite queue and deadline through `new_bounded`.

use std::sync::Arc;

use crate::admission::{AdmissionController, AdmissionError, AdmissionPermit};
use crate::server::metrics::{HttpAdmissionOutcome, Metrics};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Admits inference requests up to a fixed concurrency limit.
pub struct ConcurrencyLimiter {
    admission: AdmissionController,
    metrics: Arc<Metrics>,
    queue_timeout: Option<Duration>,
}

/// Held for the lifetime of an admitted request. Dropping it (on response
/// completion, or early client disconnect) returns the permit to the pool and
/// decrements the running-requests gauge.
pub struct RequestPermit {
    _permit: AdmissionPermit,
    metrics: Arc<Metrics>,
}

impl Drop for RequestPermit {
    fn drop(&mut self) {
        self.metrics.decrement_running_requests();
    }
}

/// Decrements the waiting gauge on drop. Held only while a request is queued, so
/// a request cancelled *while waiting* (e.g. the client disconnects before a
/// permit frees) still releases its waiting count instead of leaking the gauge.
struct WaitGuard(Arc<Metrics>);

impl Drop for WaitGuard {
    fn drop(&mut self) {
        self.0.decrement_waiting_requests();
    }
}

struct AdmissionWaitGuard {
    metrics: Arc<Metrics>,
    started: tokio::time::Instant,
    outcome: HttpAdmissionOutcome,
}
impl Drop for AdmissionWaitGuard {
    fn drop(&mut self) {
        self.metrics
            .record_http_admission_wait(self.outcome, self.started.elapsed().as_secs_f64());
    }
}

impl ConcurrencyLimiter {
    /// `max_concurrent == 0` means unbounded.
    pub fn new(max_concurrent: u64, metrics: Arc<Metrics>) -> Self {
        let maximum = if max_concurrent == 0 {
            None
        } else {
            Some(usize::try_from(max_concurrent).unwrap_or(usize::MAX))
        };
        Self {
            admission: AdmissionController::new(maximum),
            metrics,
            queue_timeout: None,
        }
    }

    /// HTTP policy: finite execution capacity, finite waiting capacity and deadline.
    /// The legacy constructor and `acquire` remain available to existing library callers.
    pub fn new_bounded(
        maximum: usize,
        maximum_waiting: usize,
        queue_timeout: Duration,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            admission: AdmissionController::new_bounded(maximum, maximum_waiting),
            metrics,
            queue_timeout: Some(queue_timeout),
        }
    }

    pub fn snapshot(&self) -> crate::admission::AdmissionSnapshot {
        self.admission.snapshot()
    }

    /// HTTP callers must use this fallible path; dropping it releases queue capacity.
    pub async fn acquire_http(&self) -> Result<RequestPermit, AdmissionError> {
        let Some(timeout) = self.queue_timeout else {
            return Ok(self.acquire().await);
        };
        let mut observation = AdmissionWaitGuard {
            metrics: self.metrics.clone(),
            started: tokio::time::Instant::now(),
            outcome: HttpAdmissionOutcome::Cancelled,
        };
        let permit = if let Some(permit) = self.admission.try_acquire() {
            permit
        } else {
            self.metrics.increment_waiting_requests();
            let _waiting = WaitGuard(self.metrics.clone());
            let deadline = tokio::time::Instant::now() + timeout;
            match self
                .admission
                .acquire_cancellable_until(&CancellationToken::new(), deadline)
                .await
            {
                Ok(permit) => permit,
                Err(error) => {
                    observation.outcome = match error {
                        AdmissionError::QueueFull { .. } => HttpAdmissionOutcome::QueueFull,
                        AdmissionError::DeadlineExceeded => HttpAdmissionOutcome::Timeout,
                        AdmissionError::Cancelled => HttpAdmissionOutcome::Cancelled,
                        AdmissionError::Closed => HttpAdmissionOutcome::Unavailable,
                    };
                    return Err(error);
                }
            }
        };
        observation.outcome = HttpAdmissionOutcome::Admitted;
        self.metrics.increment_running_requests();
        Ok(RequestPermit {
            _permit: permit,
            metrics: self.metrics.clone(),
        })
    }

    /// Acquire an admission permit, awaiting if the server is at capacity.
    /// Returns immediately when unbounded. Time spent waiting is reflected in
    /// the `power_requests_waiting` gauge.
    pub async fn acquire(&self) -> RequestPermit {
        let permit = match self.admission.try_acquire() {
            Some(permit) => permit,
            None => {
                self.metrics.increment_waiting_requests();
                let _wait = WaitGuard(self.metrics.clone());
                self.admission.acquire().await
            }
        };
        self.metrics.increment_running_requests();
        RequestPermit {
            _permit: permit,
            metrics: self.metrics.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn metrics() -> Arc<Metrics> {
        Arc::new(Metrics::new())
    }

    #[tokio::test]
    async fn unlimited_never_blocks() {
        let limiter = ConcurrencyLimiter::new(0, metrics());
        // Acquire many without releasing — must not block.
        let mut permits = Vec::new();
        for _ in 0..1000 {
            permits.push(limiter.acquire().await);
        }
        assert_eq!(permits.len(), 1000);
    }

    #[tokio::test]
    async fn bounds_concurrency_to_limit() {
        const LIMIT: u64 = 3;
        const TASKS: usize = 12;
        let m = metrics();
        let limiter = Arc::new(ConcurrencyLimiter::new(LIMIT, m.clone()));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..TASKS {
            let limiter = limiter.clone();
            let in_flight = in_flight.clone();
            let max_seen = max_seen.clone();
            handles.push(tokio::spawn(async move {
                let _permit = limiter.acquire().await;
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                max_seen.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        assert!(
            max_seen.load(Ordering::SeqCst) <= LIMIT as usize,
            "concurrent holders {} exceeded limit {}",
            max_seen.load(Ordering::SeqCst),
            LIMIT
        );
        // Some contention should have occurred with 12 tasks over 3 permits.
        assert!(max_seen.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn permit_release_admits_next() {
        let limiter = ConcurrencyLimiter::new(1, metrics());
        let p1 = limiter.acquire().await;
        // Second acquire must not resolve while p1 is held.
        let blocked = tokio::time::timeout(Duration::from_millis(50), limiter.acquire()).await;
        assert!(blocked.is_err(), "second acquire should block at limit 1");
        drop(p1);
        // After releasing, it should admit promptly.
        let admitted = tokio::time::timeout(Duration::from_millis(50), limiter.acquire()).await;
        assert!(admitted.is_ok(), "permit should be available after release");
    }

    #[tokio::test]
    async fn running_gauge_tracks_inflight() {
        let m = metrics();
        let limiter = ConcurrencyLimiter::new(0, m.clone()); // unbounded still counts running
        assert_eq!(m.running_requests(), 0);
        let p1 = limiter.acquire().await;
        let p2 = limiter.acquire().await;
        assert_eq!(m.running_requests(), 2);
        drop(p1);
        assert_eq!(m.running_requests(), 1);
        drop(p2);
        assert_eq!(
            m.running_requests(),
            0,
            "gauge returns to zero when permits drop"
        );
    }

    #[tokio::test]
    async fn waiting_gauge_tracks_queue() {
        let m = metrics();
        let limiter = Arc::new(ConcurrencyLimiter::new(1, m.clone()));
        let _held = limiter.acquire().await; // occupy the only permit
        assert_eq!(m.waiting_requests(), 0);

        let limiter2 = limiter.clone();
        let waiter = tokio::spawn(async move { limiter2.acquire().await });
        // Give the waiter time to register as waiting.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(m.waiting_requests(), 1, "queued request should be counted");

        drop(_held);
        let _admitted = waiter.await.unwrap();
        assert_eq!(
            m.waiting_requests(),
            0,
            "gauge returns to zero after admission"
        );
    }

    #[tokio::test]
    async fn cancelled_while_queued_does_not_leak_waiting_gauge() {
        let m = metrics();
        let limiter = Arc::new(ConcurrencyLimiter::new(1, m.clone()));
        let _held = limiter.acquire().await; // occupy the only permit

        let limiter2 = limiter.clone();
        let waiter = tokio::spawn(async move { limiter2.acquire().await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(m.waiting_requests(), 1, "request should be queued");

        // Client disconnects: the queued acquire future is dropped mid-await.
        waiter.abort();
        let _ = waiter.await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(
            m.waiting_requests(),
            0,
            "cancelled-while-queued request must not leak the waiting gauge"
        );
    }
}

#[cfg(test)]
#[path = "limiter_http_tests.rs"]
mod http_tests;

/// Stable OpenAI-shaped refusal for finite HTTP admission.
pub(crate) fn admission_response(error: AdmissionError) -> axum::response::Response {
    use axum::response::IntoResponse;
    let code = match error {
        AdmissionError::QueueFull { .. } => "admission_queue_full",
        AdmissionError::DeadlineExceeded => "admission_queue_timeout",
        AdmissionError::Cancelled => "admission_cancelled",
        AdmissionError::Closed => "admission_unavailable",
    };
    (
        axum::http::StatusCode::TOO_MANY_REQUESTS,
        axum::Json(serde_json::json!({
            "error": { "type": "rate_limit_error", "code": code, "message": error.to_string() }
        })),
    )
        .into_response()
}
