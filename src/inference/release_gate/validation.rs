use std::collections::BTreeSet;

use crate::error::{PowerError, Result};

use super::super::RuntimeDeviceKind;
use super::types::{
    BoundedMemoryEvidence, PeakMemoryMethod, ReleaseCapture, ReleaseCaptureSecurity,
    ReleaseEvidenceBundle, ReleaseEvidencePolicy, ReleasePlatform, ReleaseRevisionBinding,
};

const MAX_LABEL_BYTES: usize = 512;
const MAX_MEMORY_SAMPLES: u64 = 100_000_000;
const MAX_SAMPLE_INTERVAL_NANOS: u64 = 1_000_000_000;
const MAX_RELEASE_CAPTURES: usize = 4;

pub(super) fn validate_revision_binding(binding: &ReleaseRevisionBinding) -> Result<()> {
    validate_label(&binding.power_version, "release Power version")?;
    validate_revision(&binding.power_commit, "release Power commit")?;
    for (value, label) in [
        (&binding.weights_sha256, "release weights"),
        (&binding.graph_source_sha256, "release graph source"),
        (
            &binding.graph_declaration_sha256,
            "release graph declaration",
        ),
        (
            &binding.shape_profile_declaration_sha256,
            "release shape-profile declaration",
        ),
        (&binding.tee_policy_sha256, "release TEE policy"),
    ] {
        validate_sha256(value, label)?;
    }
    Ok(())
}

pub(super) fn validate_policy(policy: &ReleaseEvidencePolicy) -> Result<()> {
    if policy.schema != ReleaseEvidencePolicy::SCHEMA {
        return invalid("release evidence policy has an unsupported schema");
    }
    validate_revision_binding(&policy.revision)?;
    if policy.required_platforms.is_empty()
        || policy.required_platforms.len() > MAX_RELEASE_CAPTURES
    {
        return invalid("release evidence policy must require between one and four platforms");
    }
    if !policy
        .required_platforms
        .windows(2)
        .all(|pair| pair[0] < pair[1])
    {
        return invalid("release evidence policy platforms must be unique and canonically ordered");
    }
    Ok(())
}

pub(super) fn validate_memory_observation(
    evidence: &BoundedMemoryEvidence,
    label: &str,
) -> Result<()> {
    if evidence.sample_count == 0 || evidence.sample_count > MAX_MEMORY_SAMPLES {
        return invalid(format!(
            "{label} peak-memory evidence has an invalid sample count"
        ));
    }
    if evidence.peak_used_bytes < evidence.baseline_used_bytes
        || evidence.peak_used_bytes < evidence.final_used_bytes
    {
        return invalid(format!(
            "{label} peak-memory evidence does not contain a valid used-byte peak"
        ));
    }
    match evidence.method {
        PeakMemoryMethod::HostAllocator => {
            if evidence.sample_count != 1 {
                return invalid(
                    "allocator peak-memory evidence must use one continuous counter observation",
                );
            }
        }
        PeakMemoryMethod::ProcessResidentSet {
            sample_interval_nanos,
        }
        | PeakMemoryMethod::DevicePoolAvailability {
            sample_interval_nanos,
        } => {
            if sample_interval_nanos == 0
                || sample_interval_nanos > MAX_SAMPLE_INTERVAL_NANOS
                || evidence.sample_count < 2
            {
                return invalid(format!(
                    "{label} sampled peak-memory evidence has an invalid interval or sample count"
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_security(security: &ReleaseCaptureSecurity) -> Result<()> {
    if let ReleaseCaptureSecurity::ConfidentialGpu { binding } = security {
        validate_sha256(
            &binding.verified_claims_sha256,
            "verified confidential GPU claims",
        )?;
        validate_sha256(
            &binding.accelerator_declaration_sha256,
            "confidential GPU accelerator declaration",
        )?;
        validate_sha256(&binding.weights_sha256, "confidential GPU weights")?;
        validate_sha256(
            &binding.execution_policy_sha256,
            "confidential GPU execution policy",
        )?;
        if let Some(device_mesh_sha256) = &binding.device_mesh_sha256 {
            validate_sha256(device_mesh_sha256, "confidential GPU device mesh")?;
        }
        binding.runtime_device.validate()?;
        if binding.runtime_device.kind != RuntimeDeviceKind::Cuda {
            return invalid("verified confidential-GPU binding must use a typed CUDA device");
        }
    }
    Ok(())
}

pub(super) fn capture_platform(capture: &ReleaseCapture) -> Result<ReleasePlatform> {
    validate_security(&capture.security)?;
    match (
        capture.tensor_batch.binding.runtime_device.kind,
        &capture.security,
    ) {
        (RuntimeDeviceKind::Cpu, ReleaseCaptureSecurity::Local) => Ok(ReleasePlatform::Cpu),
        (RuntimeDeviceKind::Cuda, ReleaseCaptureSecurity::Local) => Ok(ReleasePlatform::Cuda),
        (RuntimeDeviceKind::Metal, ReleaseCaptureSecurity::Local) => Ok(ReleasePlatform::Metal),
        (RuntimeDeviceKind::Cuda, ReleaseCaptureSecurity::ConfidentialGpu { .. }) => {
            Ok(ReleasePlatform::ConfidentialGpu)
        }
        (_, ReleaseCaptureSecurity::ConfidentialGpu { .. }) => {
            invalid("confidential-GPU release evidence must execute on a typed CUDA device")
        }
    }
}

pub(super) fn validate_capture_structure(capture: &ReleaseCapture) -> Result<()> {
    if capture.schema != ReleaseCapture::SCHEMA {
        return invalid("release capture has an unsupported schema");
    }
    capture.tensor_batch.verify()?;
    capture.shape_binding.binding_sha256()?;
    validate_security(&capture.security)?;
    capture_platform(capture)?;

    let batch_binding = &capture.tensor_batch.binding;
    if batch_binding.weights_sha256 != capture.shape_binding.weights_sha256
        || batch_binding.runtime_device != capture.shape_binding.runtime_device
    {
        return invalid(
            "release capture tensor and shape evidence do not share weights and runtime device",
        );
    }
    if let ReleaseCaptureSecurity::ConfidentialGpu { binding } = &capture.security {
        if binding.weights_sha256 != capture.shape_binding.weights_sha256
            || binding.runtime_device != capture.shape_binding.runtime_device
        {
            return invalid(
                "confidential-GPU binding does not share the capture's weights and runtime device",
            );
        }
    }
    if !capture.tensor_batch.exact_output_parity {
        return invalid("release capture does not prove exact scalar/batch output parity");
    }
    super::contract_validation::validate_contracts(&capture.contracts, &capture.shape_binding)?;
    Ok(())
}

pub(super) fn verify_capture(capture: &ReleaseCapture) -> Result<()> {
    validate_capture_structure(capture)?;
    validate_sha256(&capture.sha256, "release capture")?;
    if capture.sha256 != super::digest::capture_sha256(capture)? {
        return invalid("release capture digest does not match its canonical contents");
    }
    Ok(())
}

pub(super) fn validate_bundle_structure(bundle: &ReleaseEvidenceBundle) -> Result<()> {
    if bundle.schema != ReleaseEvidenceBundle::SCHEMA {
        return invalid("release evidence bundle has an unsupported schema");
    }
    validate_policy(&bundle.policy)?;
    if bundle.captures.len() != bundle.policy.required_platforms.len()
        || bundle.captures.is_empty()
        || bundle.captures.len() > MAX_RELEASE_CAPTURES
    {
        return invalid("release evidence coverage does not exactly match its policy");
    }

    let mut platforms = Vec::with_capacity(bundle.captures.len());
    let mut report_digests = BTreeSet::new();
    for capture in &bundle.captures {
        verify_capture(capture)?;
        validate_capture_against_policy(capture, &bundle.policy)?;
        platforms.push(capture_platform(capture)?);
        if !report_digests.insert(&capture.tensor_batch.sha256) {
            return invalid(
                "release evidence must not reuse one tensor benchmark as two platform captures",
            );
        }
    }
    if platforms != bundle.policy.required_platforms {
        return invalid(
            "release evidence platforms are missing, duplicated, undeclared, or not canonically ordered",
        );
    }
    Ok(())
}

pub(super) fn verify_bundle(bundle: &ReleaseEvidenceBundle) -> Result<()> {
    validate_bundle_structure(bundle)?;
    validate_sha256(&bundle.sha256, "release evidence bundle")?;
    if bundle.sha256 != super::digest::bundle_sha256(bundle)? {
        return invalid("release evidence bundle digest does not match its canonical contents");
    }
    Ok(())
}

pub(super) fn verify_pinned_bundle(
    bundle: &ReleaseEvidenceBundle,
    expected_sha256: &str,
) -> Result<()> {
    verify_bundle(bundle)?;
    validate_sha256(expected_sha256, "pinned release evidence bundle")?;
    if bundle.sha256 != expected_sha256 {
        return Err(PowerError::PolicyViolation(
            "release evidence bundle does not match the caller-owned pinned digest".to_string(),
        ));
    }
    Ok(())
}

fn validate_capture_against_policy(
    capture: &ReleaseCapture,
    policy: &ReleaseEvidencePolicy,
) -> Result<()> {
    let expected = &policy.revision;
    let batch = &capture.tensor_batch.binding;
    let shape = &capture.shape_binding;
    let fallback = &capture.contracts.exact_fallback.selection;
    if batch.power_version != expected.power_version
        || batch.power_commit != expected.power_commit
        || batch.weights_sha256 != expected.weights_sha256
        || batch.graph_source_sha256 != expected.graph_source_sha256
        || shape.weights_sha256 != expected.weights_sha256
        || shape.graph_sha256 != expected.graph_declaration_sha256
        || shape.tee_policy_sha256 != expected.tee_policy_sha256
        || fallback.declaration_sha256 != expected.shape_profile_declaration_sha256
    {
        return invalid(
            "release capture does not match the policy's immutable revision and workload binding",
        );
    }
    Ok(())
}

pub(super) fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid(format!(
            "{label} must contain 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn validate_revision(value: &str, label: &str) -> Result<()> {
    if !matches!(value.len(), 40 | 64)
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid(format!(
            "{label} must contain 40 or 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn validate_label(value: &str, label: &str) -> Result<()> {
    if value.trim() != value
        || value.is_empty()
        || value.len() > MAX_LABEL_BYTES
        || value.chars().any(char::is_control)
    {
        return invalid(format!(
            "{label} must be a bounded non-control string without surrounding whitespace"
        ));
    }
    Ok(())
}

pub(super) fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(PowerError::InvalidFormat(message.into()))
}
