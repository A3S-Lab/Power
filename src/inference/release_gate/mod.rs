//! Model-, format-, and backend-neutral production release evidence.
//!
//! Architecture integrations own the weights, graph declarations, and exact
//! implementations behind the digests. This module validates only generic
//! execution properties and never dispatches on a model family.

mod contract_validation;
mod digest;
mod types;
mod validation;

pub use types::{
    BoundedMemoryEvidence, CancellationContractEvidence, ConfidentialReleaseBinding,
    ExactFallbackEvidence, PeakMemoryEvidence, PeakMemoryMethod, QueueExpiryEvidence,
    ReleaseCapture, ReleaseCaptureSecurity, ReleaseContractEvidence, ReleaseEvidenceBundle,
    ReleaseEvidencePolicy, ReleasePlatform, ReleasePlatformBinding, ReleaseRevisionBinding,
    ReplicaRecoveryEvidence,
};

use crate::error::Result;

pub(super) fn build_bundle(
    policy: ReleaseEvidencePolicy,
    captures: Vec<ReleaseCapture>,
) -> Result<ReleaseEvidenceBundle> {
    validation::validate_policy(&policy)?;
    let mut captures_with_platform = Vec::with_capacity(captures.len());
    for capture in captures {
        capture.verify()?;
        captures_with_platform.push((capture.platform()?, capture));
    }
    captures_with_platform.sort_by_key(|(platform, _)| *platform);
    let captures = captures_with_platform
        .into_iter()
        .map(|(_, capture)| capture)
        .collect();

    let mut bundle = ReleaseEvidenceBundle {
        schema: ReleaseEvidenceBundle::SCHEMA.to_string(),
        policy,
        captures,
        sha256: String::new(),
    };
    validation::validate_bundle_structure(&bundle)?;
    bundle.sha256 = digest::bundle_sha256(&bundle)?;
    bundle.verify()?;
    Ok(bundle)
}
