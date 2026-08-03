use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};

use super::super::{RuntimeDeviceIdentity, StorageBenchmarkSystem};
use super::types::{HardwareEvidenceBinding, HardwareEvidenceBundle};

pub(super) const MAX_CANONICAL_BUNDLE_BYTES: usize = 32 * 1024 * 1024;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeDigest<'a> {
    schema: &'static str,
    name: &'static str,
    version: &'a str,
    commit: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DeviceDigest<'a> {
    schema: &'static str,
    runtime_device: RuntimeDeviceIdentity,
    environment_sha256: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EnvironmentDigest<'a> {
    schema: &'static str,
    system: &'a StorageBenchmarkSystem,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BundlePayload<'a> {
    schema: &'a str,
    binding: &'a HardwareEvidenceBinding,
    storage_reports: &'a [super::super::StorageBenchmarkReport],
    storage_comparison: &'a super::super::StorageBenchmarkComparison,
    tuning_evidence: &'a super::super::TuningProfileEvidence,
    tuning_policy: &'a super::super::TuningProfilePolicy,
    tuning_decision: &'a super::super::TuningProfileDecision,
    parity_artifacts: &'a [super::types::ModelParityArtifact],
}

pub(super) fn runtime_sha256(version: &str, commit: &str) -> Result<String> {
    canonical_sha256(
        b"a3s-power-hardware-evidence-runtime-v1\0",
        &RuntimeDigest {
            schema: "a3s.power.hardware-evidence-runtime.v1",
            name: super::super::RUNTIME_NAME,
            version,
            commit,
        },
    )
}

pub(super) fn environment_sha256(system: &StorageBenchmarkSystem) -> Result<String> {
    canonical_sha256(
        b"a3s-power-hardware-evidence-environment-v1\0",
        &EnvironmentDigest {
            schema: "a3s.power.hardware-evidence-environment.v1",
            system,
        },
    )
}

pub(super) fn device_sha256(
    runtime_device: RuntimeDeviceIdentity,
    environment_sha256: &str,
) -> Result<String> {
    canonical_sha256(
        b"a3s-power-hardware-evidence-device-v1\0",
        &DeviceDigest {
            schema: "a3s.power.hardware-evidence-device.v1",
            runtime_device,
            environment_sha256,
        },
    )
}

pub(super) fn report_sort_key(report: &super::super::StorageBenchmarkReport) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(report)?)
}

pub(super) fn bundle_sha256(bundle: &HardwareEvidenceBundle) -> Result<String> {
    let payload = BundlePayload {
        schema: &bundle.schema,
        binding: &bundle.binding,
        storage_reports: &bundle.storage_reports,
        storage_comparison: &bundle.storage_comparison,
        tuning_evidence: &bundle.tuning_evidence,
        tuning_policy: &bundle.tuning_policy,
        tuning_decision: &bundle.tuning_decision,
        parity_artifacts: &bundle.parity_artifacts,
    };
    let bytes = serde_json::to_vec(&payload)?;
    if bytes.len() > MAX_CANONICAL_BUNDLE_BYTES {
        return Err(PowerError::InvalidRequest(format!(
            "hardware evidence bundle contains {} canonical bytes, exceeding the {} byte limit",
            bytes.len(),
            MAX_CANONICAL_BUNDLE_BYTES
        )));
    }
    Ok(domain_sha256(
        b"a3s-power-hardware-evidence-bundle-v1\0",
        &bytes,
    ))
}

fn canonical_sha256<T: Serialize>(domain: &[u8], value: &T) -> Result<String> {
    Ok(domain_sha256(domain, &serde_json::to_vec(value)?))
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}
