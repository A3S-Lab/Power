//! Shared inference admission control.
//!
//! Both the HTTP server and embedded runtimes use this controller so one
//! concurrency primitive defines request capacity. Server callers may wait for
//! capacity, while latency-sensitive embedded callers may fail fast.

use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

/// Cloneable admission controller with an optional concurrency bound.
#[derive(Debug, Clone)]
pub struct AdmissionController {
    semaphore: Option<Arc<Semaphore>>,
    maximum: Option<usize>,
}

/// RAII permit returned for an admitted request.
#[derive(Debug)]
pub struct AdmissionPermit {
    _permit: Option<OwnedSemaphorePermit>,
}

impl AdmissionController {
    /// Creates a controller. `None` means unbounded admission.
    pub fn new(maximum: Option<usize>) -> Self {
        let maximum = maximum.map(|value| value.min(Semaphore::MAX_PERMITS));
        Self {
            semaphore: maximum.map(|value| Arc::new(Semaphore::new(value))),
            maximum,
        }
    }

    pub fn maximum(&self) -> Option<usize> {
        self.maximum
    }

    /// Attempts immediate admission and returns `None` at capacity.
    pub fn try_acquire(&self) -> Option<AdmissionPermit> {
        let Some(semaphore) = &self.semaphore else {
            return Some(AdmissionPermit { _permit: None });
        };
        match Arc::clone(semaphore).try_acquire_owned() {
            Ok(permit) => Some(AdmissionPermit {
                _permit: Some(permit),
            }),
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
        let Some(semaphore) = &self.semaphore else {
            return AdmissionPermit { _permit: None };
        };
        match Arc::clone(semaphore).acquire_owned().await {
            Ok(permit) => AdmissionPermit {
                _permit: Some(permit),
            },
            Err(_) => {
                // The semaphore is private and cannot be closed through this
                // API. Avoid a production panic if that invariant changes.
                debug_assert!(false, "private admission semaphore was closed");
                AdmissionPermit { _permit: None }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
