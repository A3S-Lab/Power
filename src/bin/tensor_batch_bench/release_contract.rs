use std::time::Duration;

use a3s_power::error::{PowerError, Result};
use a3s_power::inference::graph::GraphExecutor;
use a3s_power::inference::{
    BoundedMemoryEvidence, CancellationContractEvidence, DevicePreference, DynamicShapeFallback,
    EmbeddedRuntime, ExactFallbackEvidence, ExecutionBatchBinding, ExecutionBatchMemberBinding,
    ExecutionBatchMemberSpec, ExecutionBatchRowSpec, ExecutionDigest, InferenceLimits,
    MemoryDiscoverySource, ModelIdentity, ModelSessionBinding, ModelSessionPool,
    ModelSessionPoolPolicy, ModelSessionSpec, PeakMemoryEvidence, PeakMemoryMethod,
    QueueExpiryEvidence, ReleaseContractEvidence, ReplicaRecoveryEvidence, RuntimeDeviceIdentity,
    RuntimeDeviceKind, RuntimeMemoryReservations, ShapeProfile, ShapeProfileBinding,
    ShapeProfileDeclaration, ShapeProfileRequest, TensorBatchBenchmarkReport, TensorInput,
};
use sha2::{Digest, Sha256};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::allocator::ProcessAllocationCounter;
use super::Arguments;

const DEVICE_SAMPLE_INTERVAL_NANOS: u64 = 1_000_000;
const DEVICE_SAMPLE_INTERVAL: Duration = Duration::from_nanos(DEVICE_SAMPLE_INTERVAL_NANOS);
const QUEUE_EXPIRY_INTERVAL: Duration = Duration::from_millis(20);

/// Caller-owned graph contract projected into model-neutral release evidence.
///
/// Shape classes and implementation identities stay opaque. The collector
/// executes the reviewed graph and compares it with the caller's independent
/// typed reference digest without interpreting architecture or tensor meaning.
pub(super) struct ReleaseContractWorkload<'a> {
    pub(super) graph: &'a GraphExecutor,
    pub(super) runtime: &'a EmbeddedRuntime,
    pub(super) inputs: &'a [TensorInput],
    pub(super) profile_implementation_sha256: String,
    pub(super) profile_shape_class_sha256: String,
    pub(super) fallback_implementation_sha256: String,
    pub(super) fallback_request_class_sha256: String,
    pub(super) reference_output: ExecutionDigest,
}

pub(super) fn parse_reservations(arguments: &mut Arguments) -> Result<RuntimeMemoryReservations> {
    let reservations = RuntimeMemoryReservations::default()
        .with_host_fixed_bytes(arguments.required_number("--host-fixed-bytes")?)
        .with_host_scratch_bytes(arguments.required_number("--host-scratch-bytes")?)
        .with_device_fixed_bytes(arguments.required_number("--device-fixed-bytes")?)
        .with_device_scratch_bytes(arguments.required_number("--device-scratch-bytes")?);
    reservations.validate()?;
    Ok(reservations)
}

pub(super) fn validate_reservations(
    reservations: &RuntimeMemoryReservations,
    device: RuntimeDeviceIdentity,
) -> Result<()> {
    let host = checked_sum(
        reservations.host_fixed_bytes,
        reservations.host_scratch_bytes,
        "host memory reservation",
    )?;
    let accelerator = checked_sum(
        reservations.device_fixed_bytes,
        reservations.device_scratch_bytes,
        "device memory reservation",
    )?;
    if host == 0 {
        return Err(PowerError::InvalidRequest(
            "release capture requires a positive host memory reservation".to_string(),
        ));
    }
    match device.kind {
        RuntimeDeviceKind::Cpu if accelerator != 0 => Err(PowerError::InvalidRequest(
            "CPU release capture requires zero device memory reservations".to_string(),
        )),
        RuntimeDeviceKind::Cuda | RuntimeDeviceKind::Metal if accelerator == 0 => {
            Err(PowerError::InvalidRequest(
                "accelerator release capture requires a positive device memory reservation"
                    .to_string(),
            ))
        }
        _ => Ok(()),
    }
}

pub(super) async fn collect_contracts(
    workload: &ReleaseContractWorkload<'_>,
    tensor_batch: &TensorBatchBenchmarkReport,
    shape_binding: &ShapeProfileBinding,
    device_memory: &mut Option<DevicePoolObservation>,
) -> Result<ReleaseContractEvidence> {
    if workload.inputs.is_empty() {
        return Err(PowerError::InvalidRequest(
            "release contract workload requires at least one input".to_string(),
        ));
    }
    if workload.runtime.device().identity() != shape_binding.runtime_device {
        return Err(PowerError::InferenceFailed(
            "release contract runtime changed after shape binding".to_string(),
        ));
    }
    if tensor_batch.binding.weights_sha256 != shape_binding.weights_sha256 {
        return Err(PowerError::InferenceFailed(
            "release benchmark and shape binding use different weights".to_string(),
        ));
    }
    if let Some(observation) = device_memory.as_mut() {
        observation.sample(workload.runtime).await?;
    }
    Ok(ReleaseContractEvidence {
        peak_memory: collect_peak_memory(workload, device_memory).await?,
        cancellation: collect_cancellation(workload, shape_binding)?,
        queue_expiry: collect_queue_expiry(workload.runtime).await?,
        replica_recovery: collect_replica_recovery(shape_binding).await?,
        exact_fallback: collect_exact_fallback(workload, shape_binding)?,
    })
}

async fn collect_peak_memory(
    workload: &ReleaseContractWorkload<'_>,
    device_memory: &mut Option<DevicePoolObservation>,
) -> Result<PeakMemoryEvidence> {
    let baseline = ProcessAllocationCounter::begin_live_observation();
    let cancellation = CancellationToken::new();
    let permit = workload.runtime.begin(&cancellation)?;
    let resident =
        workload
            .graph
            .run_to_resident(workload.inputs[0].clone(), &permit, &cancellation)?;
    let live_resident = workload.runtime.resident_tensor_snapshot();
    if live_resident.active_handles == 0 || live_resident.resident_bytes == 0 {
        return Err(PowerError::InferenceFailed(
            "release memory probe did not retain a graph tensor".to_string(),
        ));
    }
    if let Some(observation) = device_memory.as_mut() {
        observation.sample(workload.runtime).await?;
    }
    drop(resident);
    drop(permit);
    drop(cancellation);
    let released = workload.runtime.resident_tensor_snapshot();
    if released.active_handles != 0 || released.resident_bytes != 0 {
        return Err(PowerError::InferenceFailed(
            "release memory probe retained graph tensor state".to_string(),
        ));
    }
    if let Some(observation) = device_memory.as_mut() {
        observation.sample(workload.runtime).await?;
    }
    let host = ProcessAllocationCounter::finish_live_observation(baseline);
    let host = BoundedMemoryEvidence::host_allocator(
        host.baseline_bytes,
        host.peak_bytes,
        host.final_bytes,
    )?;
    let device = device_memory
        .as_ref()
        .map(DevicePoolObservation::evidence)
        .transpose()?;
    Ok(PeakMemoryEvidence { host, device })
}

fn collect_cancellation(
    workload: &ReleaseContractWorkload<'_>,
    shape_binding: &ShapeProfileBinding,
) -> Result<CancellationContractEvidence> {
    let host_bound = checked_sum(
        shape_binding.runtime_reservations.host_fixed_bytes,
        shape_binding.runtime_reservations.host_scratch_bytes,
        "cancellation state reservation",
    )?;
    let batch = workload
        .runtime
        .execution_batch(ExecutionBatchBinding::new(
            &shape_binding.weights_sha256,
            domain_sha256(b"a3s-power-release-contract-state-layout-v1\0"),
            domain_sha256(b"a3s-power-release-contract-scheduler-v1\0"),
        )?)?;
    let cancellation = CancellationToken::new();
    let member = ExecutionBatchMemberBinding::for_identifiers(
        b"release-contract-member",
        b"release-contract-state",
        workload.runtime.limits(),
    )?;
    batch.admit(
        ExecutionBatchMemberSpec::new(member.clone(), 0, 0, 1, host_bound.clamp(1, 4_096)),
        workload.runtime.begin(&cancellation)?,
        cancellation,
    )?;
    let input = &workload.inputs[0];
    let step = batch.begin_step(vec![ExecutionBatchRowSpec::new(
        member.member_id_sha256(),
        0,
        input.shape.clone(),
        ExecutionDigest::f32_tensor(&input.shape, &input.values),
    )])?;
    let resident = {
        let row = step.rows().first().ok_or_else(|| {
            PowerError::InferenceFailed("release cancellation step has no row".to_string())
        })?;
        workload
            .graph
            .run_to_resident(input.clone(), row.permit(), row.cancellation())?
    };
    batch.cancel(member.member_id_sha256())?;
    drop(resident);
    step.commit(Vec::new())?;
    let lifecycle = batch.finish()?;
    Ok(CancellationContractEvidence {
        lifecycle,
        admission_after: workload.runtime.admission_snapshot(),
        resident_after: workload.runtime.resident_tensor_snapshot(),
    })
}

async fn collect_queue_expiry(runtime: &EmbeddedRuntime) -> Result<QueueExpiryEvidence> {
    let before = runtime.admission_snapshot();
    let holder_cancellation = CancellationToken::new();
    let holder = runtime.begin(&holder_cancellation)?;
    let waiting_cancellation = CancellationToken::new();
    let result = runtime
        .begin_wait_until(
            &waiting_cancellation,
            Instant::now() + QUEUE_EXPIRY_INTERVAL,
        )
        .await;
    match result {
        Err(PowerError::InferenceDeadlineExceeded) => {}
        Err(error) => return Err(error),
        Ok(_) => {
            return Err(PowerError::InferenceFailed(
                "release queue probe was admitted before its occupied deadline".to_string(),
            ))
        }
    }
    drop(holder);
    tokio::task::yield_now().await;
    Ok(QueueExpiryEvidence {
        before,
        after: runtime.admission_snapshot(),
    })
}

async fn collect_replica_recovery(
    shape_binding: &ShapeProfileBinding,
) -> Result<ReplicaRecoveryEvidence> {
    let pool = ModelSessionPool::new(
        preference(shape_binding.runtime_device)?,
        ModelSessionPoolPolicy::new(1, 32, 1, 1)?.with_max_replicas_per_session(1)?,
    )?;
    let spec = ModelSessionSpec::new(
        ModelSessionBinding::new(
            ModelIdentity::new(
                "opaque-release-contract",
                "contract-v1",
                &shape_binding.weights_sha256,
            ),
            &shape_binding.graph_sha256,
        ),
        InferenceLimits::default(),
        32,
    )?;
    let first = pool
        .acquire_replica(
            spec.clone(),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(1_u64) },
        )
        .await?;
    drop(first);
    let before = pool.snapshot();
    let retired = pool
        .acquire_replica(
            spec.clone(),
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(99_u64) },
        )
        .await?;
    retired.retire();
    let retired = pool.snapshot();
    let recovered = pool
        .acquire_replica(
            spec,
            &CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(2_u64) },
        )
        .await?;
    drop(recovered);
    Ok(ReplicaRecoveryEvidence {
        before,
        retired,
        recovered: pool.snapshot(),
    })
}

fn collect_exact_fallback(
    workload: &ReleaseContractWorkload<'_>,
    shape_binding: &ShapeProfileBinding,
) -> Result<ExactFallbackEvidence> {
    let tensor_elements = workload.inputs.iter().try_fold(0_usize, |total, input| {
        total.checked_add(input.values.len()).ok_or_else(|| {
            PowerError::InvalidRequest(
                "release profile tensor element count overflowed".to_string(),
            )
        })
    })?;
    let declaration = ShapeProfileDeclaration::new(
        shape_binding.clone(),
        vec![ShapeProfile::new(
            &workload.profile_implementation_sha256,
            &workload.profile_shape_class_sha256,
            workload.inputs.len(),
            tensor_elements,
            0,
            0,
        )?],
        DynamicShapeFallback::allow(&workload.fallback_implementation_sha256)?,
    )?;
    let input = &workload.inputs[0];
    let input_digest = ExecutionDigest::f32_tensor(&input.shape, &input.values);
    let request = ShapeProfileRequest::new(
        &input_digest.sha256,
        &workload.fallback_request_class_sha256,
        input.shape[0],
        input.values.len(),
    )?;
    let selection = declaration.select(shape_binding, &request)?;
    let cancellation = CancellationToken::new();
    let permit = workload.runtime.begin(&cancellation)?;
    let fallback = workload.graph.run(input.clone(), &permit, &cancellation)?;
    let fallback_output = ExecutionDigest::f32_tensor(&fallback.shape, &fallback.values);
    Ok(ExactFallbackEvidence {
        selection: selection.evidence().clone(),
        reference_output: workload.reference_output.clone(),
        fallback_output,
    })
}

pub(super) struct DevicePoolObservation {
    runtime_device: String,
    total_bytes: u64,
    source: MemoryDiscoverySource,
    unified_with_host: bool,
    baseline_used_bytes: u64,
    peak_used_bytes: u64,
    final_used_bytes: u64,
    sample_count: u64,
}

impl DevicePoolObservation {
    pub(super) fn begin(runtime: &EmbeddedRuntime) -> Result<Option<Self>> {
        let snapshot = runtime.memory_snapshot()?;
        let Some(device) = snapshot.device else {
            return Ok(None);
        };
        let used = device.total_bytes.saturating_sub(device.available_bytes);
        Ok(Some(Self {
            runtime_device: snapshot.runtime_device,
            total_bytes: device.total_bytes,
            source: device.source,
            unified_with_host: device.unified_with_host,
            baseline_used_bytes: used,
            peak_used_bytes: used,
            final_used_bytes: used,
            sample_count: 1,
        }))
    }

    async fn sample(&mut self, runtime: &EmbeddedRuntime) -> Result<()> {
        tokio::time::sleep(DEVICE_SAMPLE_INTERVAL).await;
        let snapshot = runtime.memory_snapshot()?;
        let device = snapshot.device.ok_or_else(|| {
            PowerError::InferenceFailed(
                "accelerator release memory pool disappeared during sampling".to_string(),
            )
        })?;
        if snapshot.runtime_device != self.runtime_device
            || device.total_bytes != self.total_bytes
            || device.source != self.source
            || device.unified_with_host != self.unified_with_host
        {
            return Err(PowerError::InferenceFailed(
                "accelerator release memory pool identity changed during sampling".to_string(),
            ));
        }
        let used = device.total_bytes.saturating_sub(device.available_bytes);
        self.peak_used_bytes = self.peak_used_bytes.max(used);
        self.final_used_bytes = used;
        self.sample_count = self.sample_count.saturating_add(1);
        Ok(())
    }

    fn evidence(&self) -> Result<BoundedMemoryEvidence> {
        BoundedMemoryEvidence::sampled(
            PeakMemoryMethod::DevicePoolAvailability {
                sample_interval_nanos: DEVICE_SAMPLE_INTERVAL_NANOS,
            },
            self.baseline_used_bytes,
            self.peak_used_bytes,
            self.final_used_bytes,
            self.sample_count,
        )
    }
}

fn preference(device: RuntimeDeviceIdentity) -> Result<DevicePreference> {
    match (device.kind, device.ordinal) {
        (RuntimeDeviceKind::Cpu, None) => Ok(DevicePreference::Cpu),
        (RuntimeDeviceKind::Cuda, Some(ordinal)) => Ok(DevicePreference::Cuda { ordinal }),
        (RuntimeDeviceKind::Metal, Some(ordinal)) => Ok(DevicePreference::Metal { ordinal }),
        _ => Err(PowerError::InferenceFailed(
            "release capture contains an invalid typed runtime device".to_string(),
        )),
    }
}

pub(super) fn domain_sha256(domain: &[u8]) -> String {
    format!("{:x}", Sha256::digest(domain))
}

fn checked_sum(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| PowerError::InvalidRequest(format!("{label} overflowed")))
}
