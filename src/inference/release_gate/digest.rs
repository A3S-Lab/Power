use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error::Result;

use super::types::{ReleaseCapture, ReleaseEvidenceBundle, ReleaseEvidencePolicy};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CaptureDigestView<'a> {
    schema: &'a str,
    security: &'a super::ReleaseCaptureSecurity,
    shape_binding: &'a super::super::ShapeProfileBinding,
    tensor_batch: &'a super::super::TensorBatchBenchmarkReport,
    contracts: &'a super::ReleaseContractEvidence,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BundleDigestView<'a> {
    schema: &'a str,
    policy: &'a ReleaseEvidencePolicy,
    captures: &'a [ReleaseCapture],
}

pub(super) fn policy_sha256(policy: &ReleaseEvidencePolicy) -> Result<String> {
    canonical_sha256(b"a3s-power-release-evidence-policy-v1\0", policy)
}

pub(super) fn capture_sha256(capture: &ReleaseCapture) -> Result<String> {
    canonical_sha256(
        b"a3s-power-release-capture-v1\0",
        &CaptureDigestView {
            schema: &capture.schema,
            security: &capture.security,
            shape_binding: &capture.shape_binding,
            tensor_batch: &capture.tensor_batch,
            contracts: &capture.contracts,
        },
    )
}

pub(super) fn bundle_sha256(bundle: &ReleaseEvidenceBundle) -> Result<String> {
    canonical_sha256(
        b"a3s-power-release-evidence-bundle-v1\0",
        &BundleDigestView {
            schema: &bundle.schema,
            policy: &bundle.policy,
            captures: &bundle.captures,
        },
    )
}

fn canonical_sha256<T: Serialize>(domain: &[u8], value: &T) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(serde_json::to_vec(value)?);
    Ok(format!("{:x}", digest.finalize()))
}
