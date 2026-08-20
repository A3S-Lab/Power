use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

/// Aggregate accounting for opaque graph tensors retained on one runtime.
///
/// This snapshot contains no tensor values, shapes, graph identities, or
/// request identities. It is returned only when a caller explicitly requests
/// it; Power does not export it as telemetry automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResidentTensorSnapshot {
    pub maximum_bytes: u64,
    pub active_handles: u64,
    pub resident_bytes: u64,
    pub peak_resident_bytes: u64,
    pub rejected_reservations: u64,
}

#[derive(Clone)]
pub(crate) struct ResidentTensorBudget {
    inner: Arc<BudgetInner>,
}

struct BudgetInner {
    maximum_bytes: u64,
    active_handles: AtomicU64,
    resident_bytes: AtomicU64,
    peak_resident_bytes: AtomicU64,
    rejected_reservations: AtomicU64,
}

impl ResidentTensorBudget {
    pub(crate) fn for_f32_elements(maximum_elements: usize) -> Result<Self> {
        let maximum_elements = u64::try_from(maximum_elements).map_err(|_| {
            PowerError::Config(
                "embedded resident tensor element limit exceeds u64 addressability".to_string(),
            )
        })?;
        let maximum_bytes = maximum_elements
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| {
                PowerError::Config("embedded resident tensor byte limit overflowed".to_string())
            })?;
        Ok(Self {
            inner: Arc::new(BudgetInner {
                maximum_bytes,
                active_handles: AtomicU64::new(0),
                resident_bytes: AtomicU64::new(0),
                peak_resident_bytes: AtomicU64::new(0),
                rejected_reservations: AtomicU64::new(0),
            }),
        })
    }

    pub(crate) fn reserve(&self, bytes: u64) -> Result<ResidentTensorReservation> {
        if bytes == 0 || bytes > self.inner.maximum_bytes {
            self.reject();
            return Err(capacity_error(bytes, self.inner.maximum_bytes));
        }
        let mut current = self.inner.resident_bytes.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(bytes) else {
                self.reject();
                return Err(capacity_error(bytes, self.inner.maximum_bytes));
            };
            if next > self.inner.maximum_bytes {
                self.reject();
                return Err(capacity_error(bytes, self.inner.maximum_bytes));
            }
            match self.inner.resident_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    saturating_increment(&self.inner.active_handles);
                    self.inner
                        .peak_resident_bytes
                        .fetch_max(next, Ordering::Relaxed);
                    return Ok(ResidentTensorReservation {
                        inner: Arc::clone(&self.inner),
                        bytes,
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }

    pub(crate) fn snapshot(&self) -> ResidentTensorSnapshot {
        ResidentTensorSnapshot {
            maximum_bytes: self.inner.maximum_bytes,
            active_handles: self.inner.active_handles.load(Ordering::Acquire),
            resident_bytes: self.inner.resident_bytes.load(Ordering::Acquire),
            peak_resident_bytes: self.inner.peak_resident_bytes.load(Ordering::Relaxed),
            rejected_reservations: self.inner.rejected_reservations.load(Ordering::Relaxed),
        }
    }

    fn reject(&self) {
        saturating_increment(&self.inner.rejected_reservations);
    }
}

impl std::fmt::Debug for ResidentTensorBudget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResidentTensorBudget")
            .field("maximum_bytes", &self.inner.maximum_bytes)
            .finish_non_exhaustive()
    }
}

pub(crate) struct ResidentTensorReservation {
    inner: Arc<BudgetInner>,
    bytes: u64,
}

impl ResidentTensorReservation {
    pub(crate) fn resize(&mut self, bytes: u64) -> Result<()> {
        if bytes == 0 || bytes > self.inner.maximum_bytes {
            saturating_increment(&self.inner.rejected_reservations);
            return Err(capacity_error(bytes, self.inner.maximum_bytes));
        }
        let mut current = self.inner.resident_bytes.load(Ordering::Acquire);
        loop {
            let retained = current.checked_sub(self.bytes).ok_or_else(|| {
                PowerError::InferenceFailed(
                    "resident tensor accounting moved below its live reservation".to_string(),
                )
            })?;
            let Some(next) = retained.checked_add(bytes) else {
                saturating_increment(&self.inner.rejected_reservations);
                return Err(capacity_error(bytes, self.inner.maximum_bytes));
            };
            if next > self.inner.maximum_bytes {
                saturating_increment(&self.inner.rejected_reservations);
                return Err(capacity_error(bytes, self.inner.maximum_bytes));
            }
            match self.inner.resident_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.bytes = bytes;
                    self.inner
                        .peak_resident_bytes
                        .fetch_max(next, Ordering::Relaxed);
                    return Ok(());
                }
                Err(observed) => current = observed,
            }
        }
    }
}

impl Drop for ResidentTensorReservation {
    fn drop(&mut self) {
        let bytes = self.bytes;
        let _ = self.inner.resident_bytes.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |current| Some(current.saturating_sub(bytes)),
        );
        let _ = self.inner.active_handles.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |current| Some(current.saturating_sub(1)),
        );
    }
}

fn saturating_increment(counter: &AtomicU64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(1))
    });
}

fn capacity_error(requested_bytes: u64, maximum_bytes: u64) -> PowerError {
    PowerError::InferenceFailed(format!(
        "resident tensor requires {requested_bytes} bytes beyond the shared {maximum_bytes}-byte runtime budget"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_resize_atomically_and_release_exact_bytes() {
        let budget = ResidentTensorBudget::for_f32_elements(4).unwrap();
        let first = budget.reserve(8).unwrap();
        let mut second = budget.reserve(8).unwrap();

        assert!(second.resize(16).is_err());
        assert_eq!(budget.snapshot().resident_bytes, 16);
        drop(first);
        second.resize(16).unwrap();
        let snapshot = budget.snapshot();
        assert_eq!(snapshot.active_handles, 1);
        assert_eq!(snapshot.resident_bytes, 16);
        assert_eq!(snapshot.peak_resident_bytes, 16);
        assert_eq!(snapshot.rejected_reservations, 1);

        drop(second);
        assert_eq!(budget.snapshot().resident_bytes, 0);
        assert_eq!(budget.snapshot().active_handles, 0);
    }

    #[test]
    fn budget_and_reservations_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ResidentTensorBudget>();
        assert_send_sync::<ResidentTensorReservation>();
        assert_send_sync::<ResidentTensorSnapshot>();
    }
}
