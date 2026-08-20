use crate::admission::AdmissionSnapshot;
use crate::error::{PowerError, Result};

use super::super::{
    ExecutionBatchLifecycleEvidence, ExecutionDigest, ModelSessionPoolSnapshot,
    ResidentTensorSnapshot, RuntimeDeviceKind, RuntimeMemoryReservations, ShapeProfileBinding,
    ShapeProfileExecutionPath,
};
use super::types::{
    BoundedMemoryEvidence, ExactFallbackEvidence, PeakMemoryMethod, QueueExpiryEvidence,
    ReleaseContractEvidence, ReplicaRecoveryEvidence,
};
use super::validation::{invalid, validate_memory_observation, validate_sha256};

pub(super) fn validate_contracts(
    contracts: &ReleaseContractEvidence,
    shape: &ShapeProfileBinding,
) -> Result<()> {
    validate_peak_memory(&contracts.peak_memory, shape)?;
    validate_cancellation(
        &contracts.cancellation.lifecycle,
        &contracts.cancellation.admission_after,
        &contracts.cancellation.resident_after,
        &shape.runtime_reservations,
    )?;
    validate_queue_expiry(&contracts.queue_expiry)?;
    validate_replica_recovery(&contracts.replica_recovery, shape)?;
    validate_exact_fallback(&contracts.exact_fallback, shape)
}

fn validate_peak_memory(
    evidence: &super::PeakMemoryEvidence,
    shape: &ShapeProfileBinding,
) -> Result<()> {
    validate_memory_observation(&evidence.host, "host")?;
    if matches!(
        evidence.host.method,
        PeakMemoryMethod::DevicePoolAvailability { .. }
    ) {
        return invalid("host peak-memory evidence uses a device-only measurement method");
    }
    let reservations = &shape.runtime_reservations;
    let host_limit = checked_sum(
        reservations.host_fixed_bytes,
        reservations.host_scratch_bytes,
        "host runtime memory bound",
    )?;
    validate_memory_bound(
        &evidence.host,
        host_limit,
        reservations.host_fixed_bytes,
        "host",
    )?;

    let device_limit = checked_sum(
        reservations.device_fixed_bytes,
        reservations.device_scratch_bytes,
        "device runtime memory bound",
    )?;
    match (shape.runtime_device.kind, &evidence.device) {
        (RuntimeDeviceKind::Cpu, None) if device_limit == 0 => Ok(()),
        (RuntimeDeviceKind::Cpu, _) => {
            invalid("CPU release evidence must not declare device memory use or reservations")
        }
        (RuntimeDeviceKind::Cuda | RuntimeDeviceKind::Metal, Some(device)) => {
            validate_memory_observation(device, "device")?;
            if !matches!(
                device.method,
                PeakMemoryMethod::DevicePoolAvailability { .. }
            ) {
                return invalid(
                    "accelerator peak-memory evidence must use sampled device-pool availability",
                );
            }
            validate_memory_bound(
                device,
                device_limit,
                reservations.device_fixed_bytes,
                "device",
            )
        }
        (RuntimeDeviceKind::Cuda | RuntimeDeviceKind::Metal, None) => {
            invalid("accelerator release evidence is missing device peak-memory evidence")
        }
    }
}

fn validate_memory_bound(
    evidence: &BoundedMemoryEvidence,
    total_bound: u64,
    fixed_bound: u64,
    label: &str,
) -> Result<()> {
    if total_bound == 0 || evidence.additional_peak_bytes() == 0 {
        return invalid(format!(
            "{label} peak-memory evidence must contain a positive measured peak and bound"
        ));
    }
    if evidence.additional_peak_bytes() > total_bound {
        return invalid(format!(
            "{label} measured peak exceeds the declared runtime memory bound"
        ));
    }
    let retained = evidence
        .final_used_bytes
        .saturating_sub(evidence.baseline_used_bytes);
    if retained > fixed_bound {
        return invalid(format!(
            "{label} final memory use retains more than the declared fixed bound"
        ));
    }
    Ok(())
}

fn validate_cancellation(
    lifecycle: &ExecutionBatchLifecycleEvidence,
    admission_after: &AdmissionSnapshot,
    resident_after: &ResidentTensorSnapshot,
    reservations: &RuntimeMemoryReservations,
) -> Result<()> {
    if lifecycle.schema != ExecutionBatchLifecycleEvidence::SCHEMA {
        return invalid("cancellation lifecycle has an unsupported schema");
    }
    validate_sha256(
        &lifecycle.declaration_sha256,
        "cancellation lifecycle declaration",
    )?;
    validate_sha256(
        &lifecycle.transcript_sha256,
        "cancellation lifecycle transcript",
    )?;
    let finished = lifecycle
        .completed_members
        .checked_add(lifecycle.cancelled_members)
        .ok_or_else(|| {
            PowerError::InvalidFormat("cancellation lifecycle member count overflowed".to_string())
        })?;
    let host_bound = checked_sum(
        reservations.host_fixed_bytes,
        reservations.host_scratch_bytes,
        "cancellation host state bound",
    )?;
    if lifecycle.admitted_members == 0
        || lifecycle.cancelled_members == 0
        || lifecycle.admitted_members != finished
        || lifecycle.committed_steps == 0
        || lifecycle.processed_rows == 0
        || lifecycle.max_active_members == 0
        || lifecycle.peak_state_bytes == 0
        || lifecycle.peak_state_bytes > host_bound
    {
        return invalid("cancellation lifecycle does not prove bounded active-work cancellation");
    }
    validate_admission_snapshot(admission_after, "post-cancellation admission")?;
    if admission_after.active != 0 || admission_after.waiting != 0 {
        return invalid("cancellation evidence leaves active or waiting admission work");
    }
    validate_resident_snapshot(resident_after)?;
    if resident_after.active_handles != 0 || resident_after.resident_bytes != 0 {
        return invalid("cancellation evidence leaves resident tensor handles or bytes");
    }
    Ok(())
}

fn validate_queue_expiry(evidence: &QueueExpiryEvidence) -> Result<()> {
    validate_admission_snapshot(&evidence.before, "queue-expiry before")?;
    validate_admission_snapshot(&evidence.after, "queue-expiry after")?;
    if evidence.before.active_limit != evidence.after.active_limit
        || evidence.before.waiting_limit != evidence.after.waiting_limit
        || evidence.before.active != 0
        || evidence.before.waiting != 0
        || evidence.after.active != 0
        || evidence.after.waiting != 0
        || evidence.before.active_limit.is_none_or(|limit| limit == 0)
        || evidence.before.waiting_limit.is_none_or(|limit| limit == 0)
    {
        return invalid("queue-expiry evidence does not use one quiescent bounded controller");
    }
    validate_admission_counters_monotonic(&evidence.before, &evidence.after)?;
    if evidence.after.deadline_expirations <= evidence.before.deadline_expirations {
        return invalid("queue-expiry evidence contains no observed deadline expiration");
    }
    Ok(())
}

fn validate_replica_recovery(
    evidence: &ReplicaRecoveryEvidence,
    shape: &ShapeProfileBinding,
) -> Result<()> {
    for (label, snapshot) in [
        ("replica before", &evidence.before),
        ("replica retired", &evidence.retired),
        ("replica recovered", &evidence.recovered),
    ] {
        validate_pool_snapshot(snapshot, label)?;
        if snapshot.device != shape.runtime_device {
            return invalid("replica-recovery evidence uses a different runtime device");
        }
    }
    if !same_pool(&evidence.before, &evidence.retired)
        || !same_pool(&evidence.before, &evidence.recovered)
    {
        return invalid("replica-recovery evidence does not describe one stable pool");
    }
    if evidence.before.registered_sessions == 0
        || evidence.before.ready_sessions == 0
        || evidence.before.ready_replicas == 0
        || evidence.before.leased_replicas != 0
        || evidence.before.waiting_replica_requests != 0
        || evidence.before.replicas_pending_reconstruction != 0
    {
        return invalid("replica-recovery baseline is not healthy and quiescent");
    }
    if evidence.retired.replica_retirements <= evidence.before.replica_retirements
        || evidence.retired.ready_replicas >= evidence.before.ready_replicas
        || evidence.retired.replicas_pending_reconstruction == 0
        || evidence.retired.leased_replicas != 0
        || evidence.retired.waiting_replica_requests != 0
    {
        return invalid("replica-recovery evidence contains no observable retirement state");
    }
    if evidence.recovered.replica_retirements < evidence.retired.replica_retirements
        || evidence.recovered.replica_reconstructions <= evidence.before.replica_reconstructions
        || evidence.recovered.ready_replicas < evidence.before.ready_replicas
        || evidence.recovered.replicas_pending_reconstruction != 0
        || evidence.recovered.leased_replicas != 0
        || evidence.recovered.waiting_replica_requests != 0
        || evidence.recovered.reserved_bytes != evidence.before.reserved_bytes
    {
        return invalid("replica-recovery evidence does not restore the quiescent replica set");
    }
    Ok(())
}

fn validate_exact_fallback(
    evidence: &ExactFallbackEvidence,
    shape: &ShapeProfileBinding,
) -> Result<()> {
    evidence.selection.validate()?;
    if !matches!(
        evidence.selection.path,
        ShapeProfileExecutionPath::DynamicFallback { .. }
    ) {
        return invalid("release fallback evidence did not select an explicit dynamic fallback");
    }
    validate_execution_digest(&evidence.reference_output, "fallback reference output")?;
    validate_execution_digest(&evidence.fallback_output, "fallback tested output")?;
    if evidence.reference_output != evidence.fallback_output {
        return invalid("explicit fallback output does not have exact typed parity");
    }
    if evidence.selection.binding_sha256 != shape.binding_sha256()?
        || evidence.selection.weights_sha256 != shape.weights_sha256
        || evidence.selection.runtime_device != shape.runtime_device
    {
        return invalid("explicit fallback evidence does not match the release shape binding");
    }
    Ok(())
}

fn validate_execution_digest(digest: &ExecutionDigest, label: &str) -> Result<()> {
    validate_sha256(&digest.sha256, label)?;
    if digest.byte_length == 0 || digest.item_count == 0 {
        return invalid(format!("{label} must describe at least one byte and item"));
    }
    Ok(())
}

fn validate_admission_snapshot(snapshot: &AdmissionSnapshot, label: &str) -> Result<()> {
    if snapshot
        .active_limit
        .is_some_and(|limit| limit == 0 || snapshot.active > limit || snapshot.peak_active > limit)
        || snapshot
            .waiting_limit
            .is_some_and(|limit| snapshot.waiting > limit || snapshot.peak_waiting > limit)
        || snapshot.active > snapshot.peak_active
        || snapshot.waiting > snapshot.peak_waiting
    {
        return invalid(format!("{label} counters exceed their declared bounds"));
    }
    Ok(())
}

fn validate_admission_counters_monotonic(
    before: &AdmissionSnapshot,
    after: &AdmissionSnapshot,
) -> Result<()> {
    if after.peak_active < before.peak_active
        || after.peak_waiting < before.peak_waiting
        || after.admitted < before.admitted
        || after.queue_rejections < before.queue_rejections
        || after.cancelled_waiters < before.cancelled_waiters
        || after.deadline_expirations < before.deadline_expirations
    {
        return invalid("admission evidence counters moved backwards");
    }
    Ok(())
}

fn validate_resident_snapshot(snapshot: &ResidentTensorSnapshot) -> Result<()> {
    if snapshot.maximum_bytes == 0
        || snapshot.resident_bytes > snapshot.maximum_bytes
        || snapshot.peak_resident_bytes > snapshot.maximum_bytes
        || snapshot.resident_bytes > snapshot.peak_resident_bytes
    {
        return invalid("resident tensor evidence exceeds its declared byte bound");
    }
    Ok(())
}

fn validate_pool_snapshot(snapshot: &ModelSessionPoolSnapshot, label: &str) -> Result<()> {
    snapshot.device.validate()?;
    validate_admission_snapshot(&snapshot.device_admission, label)?;
    let replica_capacity = snapshot
        .registered_sessions
        .checked_mul(snapshot.maximum_replicas_per_session)
        .ok_or_else(|| PowerError::InvalidFormat(format!("{label} replica capacity overflowed")))?;
    if snapshot.maximum_sessions == 0
        || snapshot.maximum_resident_bytes == 0
        || snapshot.maximum_replicas_per_session == 0
        || snapshot.registered_sessions > snapshot.maximum_sessions
        || snapshot.ready_sessions > snapshot.registered_sessions
        || snapshot.reserved_replicas > replica_capacity
        || snapshot.ready_replicas > snapshot.reserved_replicas
        || snapshot.leased_replicas > snapshot.ready_replicas
        || snapshot.replicas_pending_reconstruction > snapshot.reserved_replicas
        || snapshot.reserved_bytes > snapshot.maximum_resident_bytes
    {
        return invalid(format!(
            "{label} counters exceed their declared pool bounds"
        ));
    }
    Ok(())
}

fn same_pool(left: &ModelSessionPoolSnapshot, right: &ModelSessionPoolSnapshot) -> bool {
    left.device == right.device
        && left.maximum_sessions == right.maximum_sessions
        && left.maximum_resident_bytes == right.maximum_resident_bytes
        && left.registered_sessions == right.registered_sessions
        && left.maximum_replicas_per_session == right.maximum_replicas_per_session
        && left.reserved_replicas == right.reserved_replicas
}

fn checked_sum(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| PowerError::InvalidFormat(format!("{label} overflowed")))
}
