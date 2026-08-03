use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::admission::{AdmissionController, AdmissionPermit};
use crate::error::{PowerError, Result};

#[cfg(test)]
use super::RuntimeDeviceKind;
use super::{
    AcceleratorExecutionEvidence, DevicePreference, ExecutionDigest, ExecutionReceipt,
    HardwareMemorySnapshot, InferenceLimits, ModelIdentity, ResidencyBudgetPlan,
    ResidencyBudgetPolicy, RuntimeDevice, RuntimeIdentity,
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
        Self::with_device(RuntimeDevice::resolve(preference)?, limits)
    }

    fn with_device(device: RuntimeDevice, limits: InferenceLimits) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            inner: Arc::new(RuntimeInner {
                device,
                admission: AdmissionController::new(Some(limits.max_concurrent_requests)),
                limits,
            }),
        })
    }

    #[cfg(test)]
    pub(crate) fn new_test_accelerator(
        kind: RuntimeDeviceKind,
        ordinal: usize,
        limits: InferenceLimits,
    ) -> Result<Self> {
        Self::with_device(RuntimeDevice::test_accelerator(kind, ordinal)?, limits)
    }

    pub fn device(&self) -> &RuntimeDevice {
        &self.inner.device
    }

    pub fn limits(&self) -> &InferenceLimits {
        &self.inner.limits
    }

    /// Discovers memory for the resolved device without spawning a process.
    pub fn memory_snapshot(&self) -> Result<HardwareMemorySnapshot> {
        self.inner.device.memory_snapshot()
    }

    /// Produces an opt-in, device-bound cache budget capped by this runtime's
    /// resident-weight limit.
    pub fn plan_residency_budget(
        &self,
        policy: &ResidencyBudgetPolicy,
    ) -> Result<ResidencyBudgetPlan> {
        policy.plan(&self.memory_snapshot()?, &self.inner.limits)
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
            accelerator: None,
        }
    }

    /// Constructs a receipt that commits to the actual accelerator or exact
    /// fallback path selected by a declaration-bound execution.
    pub fn receipt_with_accelerator(
        &self,
        model: ModelIdentity,
        input: ExecutionDigest,
        output: ExecutionDigest,
        accelerator: AcceleratorExecutionEvidence,
    ) -> Result<ExecutionReceipt> {
        accelerator.validate()?;
        if accelerator.weights_sha256 != model.weights_sha256
            || accelerator.runtime_device != self.device().identity()
            || accelerator.input_sha256 != input.sha256
            || accelerator.output_sha256 != output.sha256
        {
            return Err(PowerError::InvalidRequest(
                "accelerator execution evidence does not match the receipt model, runtime, input, or output"
                    .to_string(),
            ));
        }
        Ok(ExecutionReceipt {
            schema: ExecutionReceipt::ACCELERATOR_SCHEMA.to_string(),
            model,
            runtime: RuntimeIdentity::current(self.device()),
            input,
            output,
            accelerator: Some(accelerator),
        })
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

    #[test]
    fn automatic_budget_is_bound_to_runtime_device_and_limit() {
        let limits = InferenceLimits {
            max_resident_weight_bytes: 1_024,
            ..InferenceLimits::default()
        };
        let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap();
        let policy = ResidencyBudgetPolicy::new(10_000, 0).unwrap();

        let plan = runtime.plan_residency_budget(&policy).unwrap();

        assert_eq!(plan.runtime_device, runtime.device().name());
        assert!(plan.total_cache_bytes <= 1_024);
        assert_eq!(plan.device_cache_bytes, 0);
    }
}
