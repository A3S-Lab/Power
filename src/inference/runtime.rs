use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::admission::{AdmissionController, AdmissionPermit};
use crate::error::{PowerError, Result};

use super::{
    DevicePreference, ExecutionDigest, ExecutionReceipt, InferenceLimits, ModelIdentity,
    RuntimeDevice, RuntimeIdentity,
};

/// Shared execution context for every embedded model implementation.
///
/// A runtime owns the resolved tensor device, hard resource bounds, request
/// admission, cancellation checks, and receipt construction. Model modules
/// provide network-specific control flow only.
#[derive(Clone)]
pub struct EmbeddedRuntime {
    inner: Arc<RuntimeInner>,
}

struct RuntimeInner {
    device: RuntimeDevice,
    limits: InferenceLimits,
    admission: AdmissionController,
}

impl EmbeddedRuntime {
    pub fn new(preference: DevicePreference, limits: InferenceLimits) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            inner: Arc::new(RuntimeInner {
                device: RuntimeDevice::resolve(preference)?,
                admission: AdmissionController::new(Some(limits.max_concurrent_requests)),
                limits,
            }),
        })
    }

    pub fn device(&self) -> &RuntimeDevice {
        &self.inner.device
    }

    pub fn limits(&self) -> &InferenceLimits {
        &self.inner.limits
    }

    /// Acquires one shared embedded-inference admission permit.
    ///
    /// A model-level operation should hold this permit across all of its
    /// component graph calls so nested models do not create independent
    /// concurrency controls.
    pub fn begin(&self, cancellation: &CancellationToken) -> Result<ExecutionPermit> {
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "embedded inference was cancelled".to_string(),
            ));
        }
        let admission = self.inner.admission.try_acquire().ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "embedded runtime already has {} active request(s)",
                self.inner.limits.max_concurrent_requests
            ))
        })?;
        Ok(ExecutionPermit {
            inner: Arc::new(PermitInner {
                runtime: Arc::clone(&self.inner),
                _admission: admission,
            }),
        })
    }

    pub fn receipt(
        &self,
        model: ModelIdentity,
        input: ExecutionDigest,
        output: ExecutionDigest,
    ) -> ExecutionReceipt {
        ExecutionReceipt {
            schema: ExecutionReceipt::SCHEMA.to_string(),
            model,
            runtime: RuntimeIdentity::current(self.device()),
            input,
            output,
        }
    }
}

impl std::fmt::Debug for EmbeddedRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EmbeddedRuntime")
            .field("device", &self.inner.device)
            .field("limits", &self.inner.limits)
            .field("admission", &self.inner.admission)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct ExecutionPermit {
    inner: Arc<PermitInner>,
}

struct PermitInner {
    runtime: Arc<RuntimeInner>,
    _admission: AdmissionPermit,
}

impl ExecutionPermit {
    pub(crate) fn belongs_to(&self, runtime: &EmbeddedRuntime) -> bool {
        Arc::ptr_eq(&self.inner.runtime, &runtime.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_is_shared_by_runtime_clones() {
        let runtime = EmbeddedRuntime::new(
            DevicePreference::Cpu,
            InferenceLimits {
                max_concurrent_requests: 1,
                ..InferenceLimits::default()
            },
        )
        .unwrap();
        let clone = runtime.clone();
        let cancellation = CancellationToken::new();
        let permit = runtime.begin(&cancellation).unwrap();
        assert!(clone.begin(&cancellation).is_err());
        drop(permit);
        assert!(clone.begin(&cancellation).is_ok());
    }

    #[test]
    fn cancelled_request_is_not_admitted() {
        let runtime =
            EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default()).unwrap();
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        assert!(runtime.begin(&cancellation).is_err());
    }

    #[test]
    fn runtime_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EmbeddedRuntime>();
    }
}
