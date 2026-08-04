use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::admission::{AdmissionController, AdmissionError, AdmissionPermit, AdmissionSnapshot};
use crate::error::{PowerError, Result};

#[cfg(test)]
use super::RuntimeDeviceKind;
use super::{
    AcceleratorExecutionEvidence, DevicePreference, ExecutionDigest, ExecutionReceipt,
    HardwareMemorySnapshot, InferenceLimits, ModelIdentity, ResidencyBudgetPlan,
    ResidencyBudgetPolicy, ResidencyPolicy, RuntimeDevice, RuntimeIdentity,
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
    device_admission: Option<AdmissionController>,
}

impl EmbeddedRuntime {
    pub fn new(preference: DevicePreference, limits: InferenceLimits) -> Result<Self> {
        Self::with_device(RuntimeDevice::resolve(preference)?, limits)
    }

    fn with_device(device: RuntimeDevice, limits: InferenceLimits) -> Result<Self> {
        Self::with_optional_device_admission(device, limits, None)
    }

    pub(super) fn with_device_admission(
        device: RuntimeDevice,
        limits: InferenceLimits,
        device_admission: AdmissionController,
    ) -> Result<Self> {
        Self::with_optional_device_admission(device, limits, Some(device_admission))
    }

    fn with_optional_device_admission(
        device: RuntimeDevice,
        limits: InferenceLimits,
        device_admission: Option<AdmissionController>,
    ) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            inner: Arc::new(RuntimeInner {
                device,
                admission: AdmissionController::new_bounded(
                    limits.max_concurrent_requests,
                    limits.max_queued_requests,
                ),
                device_admission,
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

    /// Returns content-free counters for this runtime's shared execution gate.
    pub fn admission_snapshot(&self) -> AdmissionSnapshot {
        self.inner.admission.snapshot()
    }

    /// Returns the shared physical-device gate when this runtime came from a
    /// model session pool.
    pub fn device_admission_snapshot(&self) -> Option<AdmissionSnapshot> {
        self.inner
            .device_admission
            .as_ref()
            .map(AdmissionController::snapshot)
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

    /// Revalidates current memory pressure and applies only the planned cache
    /// bytes to a caller-owned residency policy.
    pub fn apply_residency_budget(
        &self,
        plan: &ResidencyBudgetPlan,
        base: &ResidencyPolicy,
    ) -> Result<ResidencyPolicy> {
        plan.apply_to_revalidated(base, &self.memory_snapshot()?)
    }

    /// Acquires one shared embedded-inference admission permit.
    ///
    /// A model-level operation should hold this permit across all of its
    /// component graph calls so nested models do not create independent
    /// concurrency controls.
    pub fn begin(&self, cancellation: &CancellationToken) -> Result<ExecutionPermit> {
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceCancelled);
        }
        let admission = self.inner.admission.try_acquire().ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "embedded runtime already has {} active request(s)",
                self.inner.limits.max_concurrent_requests
            ))
        })?;
        let device_admission = match &self.inner.device_admission {
            Some(controller) => Some(controller.try_acquire().ok_or_else(|| {
                PowerError::InferenceFailed(format!(
                    "embedded device already has {} active request(s)",
                    controller.maximum().unwrap_or(0)
                ))
            })?),
            None => None,
        };
        Ok(self.execution_permit(admission, device_admission))
    }

    /// Waits for execution capacity through the runtime's bounded queue.
    ///
    /// Cancellation before or during the wait releases the queue slot. The
    /// returned permit is identical to a fail-fast [`Self::begin`] permit and
    /// must be held across every component graph in the logical request.
    pub async fn begin_wait(&self, cancellation: &CancellationToken) -> Result<ExecutionPermit> {
        let admission = self
            .inner
            .admission
            .acquire_cancellable(cancellation)
            .await
            .map_err(|error| match error {
                AdmissionError::QueueFull { maximum } => PowerError::InferenceQueueFull { maximum },
                AdmissionError::Cancelled => PowerError::InferenceCancelled,
                AdmissionError::Closed => PowerError::InferenceFailed(
                    "embedded runtime admission controller was closed".to_string(),
                ),
            })?;
        let device_admission = match &self.inner.device_admission {
            Some(controller) => Some(controller.acquire_cancellable(cancellation).await.map_err(
                |error| match error {
                    AdmissionError::QueueFull { maximum } => {
                        PowerError::InferenceQueueFull { maximum }
                    }
                    AdmissionError::Cancelled => PowerError::InferenceCancelled,
                    AdmissionError::Closed => PowerError::InferenceFailed(
                        "embedded device admission controller was closed".to_string(),
                    ),
                },
            )?),
            None => None,
        };
        Ok(self.execution_permit(admission, device_admission))
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
            microbatch: None,
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
        let schema = if accelerator.device_mesh_sha256.is_some() {
            ExecutionReceipt::ACCELERATOR_MESH_SCHEMA
        } else {
            ExecutionReceipt::ACCELERATOR_SCHEMA
        };
        Ok(ExecutionReceipt {
            schema: schema.to_string(),
            model,
            runtime: RuntimeIdentity::current(self.device()),
            input,
            output,
            accelerator: Some(accelerator),
            microbatch: None,
        })
    }

    fn execution_permit(
        &self,
        admission: AdmissionPermit,
        device_admission: Option<AdmissionPermit>,
    ) -> ExecutionPermit {
        ExecutionPermit {
            inner: Arc::new(PermitInner {
                runtime: Arc::clone(&self.inner),
                _admission: admission,
                _device_admission: device_admission,
            }),
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
            .field("device_admission", &self.inner.device_admission)
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
    _device_admission: Option<AdmissionPermit>,
}

impl ExecutionPermit {
    pub(crate) fn belongs_to(&self, runtime: &EmbeddedRuntime) -> bool {
        Arc::ptr_eq(&self.inner.runtime, &runtime.inner)
    }

    pub(crate) fn same_admission(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub(crate) fn model_admission_was_queued(&self) -> bool {
        self.inner._admission.was_queued()
    }

    pub(crate) fn device_admission_was_queued(&self) -> bool {
        self.inner
            ._device_admission
            .as_ref()
            .is_some_and(AdmissionPermit::was_queued)
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

    #[tokio::test]
    async fn waiting_admission_is_bounded_and_cancellation_safe() {
        let runtime = EmbeddedRuntime::new(
            DevicePreference::Cpu,
            InferenceLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 1,
                ..InferenceLimits::default()
            },
        )
        .unwrap();
        let active = runtime.begin(&CancellationToken::new()).unwrap();
        let cancellation = CancellationToken::new();
        let waiter = tokio::spawn({
            let runtime = runtime.clone();
            let cancellation = cancellation.clone();
            async move { runtime.begin_wait(&cancellation).await }
        });
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while runtime.admission_snapshot().waiting != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let overflow = runtime.begin_wait(&CancellationToken::new()).await;
        assert!(matches!(
            overflow,
            Err(PowerError::InferenceQueueFull { maximum: 1 })
        ));

        cancellation.cancel();
        assert!(waiter.await.unwrap().is_err());
        assert_eq!(runtime.admission_snapshot().waiting, 0);
        assert_eq!(runtime.admission_snapshot().active, 1);
        drop(active);
        assert_eq!(runtime.admission_snapshot().active, 0);
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
        let projected = runtime
            .apply_residency_budget(&plan, &ResidencyPolicy::default())
            .unwrap();

        assert_eq!(plan.runtime_device, runtime.device().name());
        assert!(plan.total_cache_bytes <= 1_024);
        assert_eq!(plan.device_cache_bytes, 0);
        assert_eq!(projected.host_cache_bytes, plan.host_cache_bytes);
        assert_eq!(projected.device_cache_bytes, plan.device_cache_bytes);
    }
}
