use std::collections::BTreeSet;

use crate::error::{PowerError, Result};
use crate::tee::attestation::TeeType;

use super::super::RuntimeDeviceKind;
use super::types::{
    BoundedMemoryEvidence, PeakMemoryMethod, ReleaseCapture, ReleaseCaptureSecurity,
    ReleaseEvidenceBundle, ReleaseEvidencePolicy, ReleasePlatform, ReleasePlatformBinding,
    ReleaseRevisionBinding,
};

const MAX_LABEL_BYTES: usize = 512;
const MAX_MEMORY_SAMPLES: u64 = 100_000_000;
const MAX_SAMPLE_INTERVAL_NANOS: u64 = 1_000_000_000;
const MAX_RELEASE_CAPTURES: usize = 4;
const STRICT_V1_PLATFORMS: [ReleasePlatform; 4] = [
    ReleasePlatform::Cpu,
    ReleasePlatform::Cuda,
    ReleasePlatform::Metal,
    ReleasePlatform::ConfidentialGpu,
];

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
    ] {
        validate_sha256(value, label)?;
    }
    Ok(())
}

pub(super) fn validate_platform_binding(binding: &ReleasePlatformBinding) -> Result<()> {
    validate_sha256(
        &binding.shape_profile_declaration_sha256,
        "release platform shape-profile declaration",
    )?;
    validate_sha256(&binding.tee_policy_sha256, "release platform TEE policy")
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
    for binding in &policy.required_platforms {
        validate_platform_binding(binding)?;
    }
    if !policy
        .required_platforms
        .windows(2)
        .all(|pair| pair[0].platform < pair[1].platform)
    {
        return invalid("release evidence policy platforms must be unique and canonically ordered");
    }
    Ok(())
}

pub(super) fn validate_strict_v1_policy(policy: &ReleaseEvidencePolicy) -> Result<()> {
    validate_policy(policy)?;
    let actual = policy
        .required_platforms
        .iter()
        .map(|binding| binding.platform)
        .collect::<Vec<_>>();
    if actual != STRICT_V1_PLATFORMS {
        return Err(PowerError::PolicyViolation(
            "strict v1 release evidence requires CPU, CUDA, Metal, and confidential-GPU platform bindings"
                .to_string(),
        ));
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
        if !matches!(binding.tee_type, TeeType::SevSnp | TeeType::Tdx) {
            return invalid("verified confidential-GPU binding requires a hardware TEE");
        }
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
            || binding.execution_policy_sha256 != capture.shape_binding.tee_policy_sha256
        {
            return invalid(
                "confidential-GPU binding does not share the capture's weights, runtime device, and TEE policy",
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
    let required_platforms = bundle
        .policy
        .required_platforms
        .iter()
        .map(|binding| binding.platform)
        .collect::<Vec<_>>();
    if platforms != required_platforms {
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

pub(super) fn verify_strict_v1_release(
    bundle: &ReleaseEvidenceBundle,
    expected_sha256: &str,
    expected_power_version: &str,
    expected_power_commit: &str,
) -> Result<()> {
    verify_pinned_bundle(bundle, expected_sha256)?;
    validate_strict_v1_policy(&bundle.policy)?;
    validate_release_identity(
        &bundle.policy.revision,
        expected_power_version,
        expected_power_commit,
    )?;

    let binding = bundle
        .captures
        .iter()
        .find_map(|capture| match &capture.security {
            ReleaseCaptureSecurity::ConfidentialGpu { binding } => Some(binding),
            ReleaseCaptureSecurity::Local => None,
        })
        .ok_or_else(|| {
            PowerError::PolicyViolation(
                "strict v1 release evidence is missing its confidential-GPU capture".to_string(),
            )
        })?;
    validate_strict_v1_confidential_tee(binding.tee_type)
}

fn validate_release_identity(
    revision: &ReleaseRevisionBinding,
    expected_power_version: &str,
    expected_power_commit: &str,
) -> Result<()> {
    validate_label(expected_power_version, "expected release Power version")?;
    validate_revision(expected_power_commit, "expected release Power commit")?;
    if revision.power_version != expected_power_version
        || revision.power_commit != expected_power_commit
    {
        return Err(PowerError::PolicyViolation(
            "release evidence bundle does not match the expected Power version and source revision"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_strict_v1_confidential_tee(tee_type: TeeType) -> Result<()> {
    if tee_type != TeeType::SevSnp {
        return Err(PowerError::PolicyViolation(
            "strict v1 confidential-GPU evidence requires AMD SEV-SNP; Intel TDX is unsupported until reviewed DCAP Quote/QVL verification exists"
                .to_string(),
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
    {
        return invalid(
            "release capture does not match the policy's immutable revision and workload binding",
        );
    }
    let platform = capture_platform(capture)?;
    let platform_binding = policy
        .required_platforms
        .iter()
        .find(|binding| binding.platform == platform)
        .ok_or_else(|| {
            PowerError::InvalidFormat(
                "release capture platform is not declared by its policy".to_string(),
            )
        })?;
    if fallback.declaration_sha256 != platform_binding.shape_profile_declaration_sha256
        || shape.tee_policy_sha256 != platform_binding.tee_policy_sha256
    {
        return invalid(
            "release capture does not match its platform-specific shape-profile and TEE binding",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_v1_confidential_gpu_supports_only_sev_snp() {
        validate_strict_v1_confidential_tee(TeeType::SevSnp).unwrap();
        for tee_type in [TeeType::Tdx, TeeType::Simulated, TeeType::None] {
            let error = validate_strict_v1_confidential_tee(tee_type).unwrap_err();
            assert!(error.to_string().contains("requires AMD SEV-SNP"));
        }
    }

    #[test]
    fn strict_release_identity_requires_the_exact_version_and_revision() {
        let revision = ReleaseRevisionBinding::new(
            "1.0.0",
            "a".repeat(40),
            "b".repeat(64),
            "c".repeat(64),
            "d".repeat(64),
        )
        .unwrap();
        validate_release_identity(&revision, "1.0.0", &"a".repeat(40)).unwrap();

        let version_error =
            validate_release_identity(&revision, "1.0.1", &"a".repeat(40)).unwrap_err();
        assert!(version_error
            .to_string()
            .contains("version and source revision"));
        let revision_error =
            validate_release_identity(&revision, "1.0.0", &"e".repeat(40)).unwrap_err();
        assert!(revision_error
            .to_string()
            .contains("version and source revision"));
    }
}
