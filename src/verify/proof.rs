use crate::error::{PowerError, Result};
use crate::tee::attestation::{AttestationReport, TeeType};

use super::{verify_report_with_policy, VerificationPolicy, VerifyOptions, VerifyResult};

struct VerificationSeal;

/// Opaque proof that one exact report passed Power's strict hardware profile.
///
/// The proof borrows the verified report, cannot be deserialized, and cannot be
/// constructed outside the verifier module. Keeping the report reference in
/// the type prevents a successful result for one report from authorizing a
/// different report.
#[must_use = "a strict attestation proof has no effect until consumed by a protected operation"]
pub struct VerifiedHardwareAttestation<'report> {
    report: &'report AttestationReport,
    _seal: VerificationSeal,
}

impl<'report> VerifiedHardwareAttestation<'report> {
    fn issue(report: &'report AttestationReport) -> Self {
        Self {
            report,
            _seal: VerificationSeal,
        }
    }

    #[cfg(any(feature = "embedded-inference", test))]
    pub(crate) fn report(&self) -> &'report AttestationReport {
        self.report
    }

    pub fn tee_type(&self) -> TeeType {
        self.report.tee_type
    }
}

impl std::fmt::Debug for VerifiedHardwareAttestation<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VerifiedHardwareAttestation")
            .field("tee_type", &self.report.tee_type)
            .finish_non_exhaustive()
    }
}

/// Opaque proof that one exact report passed the full confidential-GPU profile.
///
/// This is stronger than [`VerifiedHardwareAttestation`]: it also proves nonce
/// freshness, canonical claims, NVIDIA evidence/verdict and identity pins, and
/// the pinned GPU execution policy required by the profile.
#[must_use = "a confidential-GPU proof has no effect until consumed by a protected operation"]
pub struct VerifiedConfidentialGpuAttestation<'report> {
    hardware: VerifiedHardwareAttestation<'report>,
}

impl<'report> VerifiedConfidentialGpuAttestation<'report> {
    fn issue(report: &'report AttestationReport) -> Self {
        Self {
            hardware: VerifiedHardwareAttestation::issue(report),
        }
    }

    pub fn as_hardware(&self) -> &VerifiedHardwareAttestation<'report> {
        &self.hardware
    }

    #[cfg(any(feature = "embedded-inference", test))]
    pub(crate) fn report(&self) -> &'report AttestationReport {
        self.hardware.report()
    }
}

impl std::fmt::Debug for VerifiedConfidentialGpuAttestation<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VerifiedConfidentialGpuAttestation")
            .field("tee_type", &self.hardware.tee_type())
            .finish_non_exhaustive()
    }
}

/// Verify one report under the fixed production profile and return an opaque
/// proof tied to that report's lifetime.
///
/// Protected APIs consume this proof instead of trusting a raw report or a
/// caller-supplied "verified" label.
pub fn verify_report_strict_with_proof<'report>(
    report: &'report AttestationReport,
    opts: &VerifyOptions<'_>,
) -> Result<(VerifyResult, VerifiedHardwareAttestation<'report>)> {
    let result = verify_report_with_policy(report, opts, VerificationPolicy::strict())?;
    if !result.hardware_verified
        || !result.measurement_verified
        || !matches!(result.tee_type, TeeType::SevSnp | TeeType::Tdx)
    {
        return Err(PowerError::AttestationVerificationFailed(
            "strict verification completed without the hardware and measurement guarantees required to issue a proof"
                .to_string(),
        ));
    }
    Ok((result, VerifiedHardwareAttestation::issue(report)))
}

/// Verify one report under the fixed confidential-GPU production profile and
/// return an opaque proof tied to that exact report.
pub fn verify_confidential_gpu_attestation<'report>(
    report: &'report AttestationReport,
    opts: &VerifyOptions<'_>,
) -> Result<(VerifyResult, VerifiedConfidentialGpuAttestation<'report>)> {
    let result = verify_report_with_policy(report, opts, VerificationPolicy::gpu_confidential())?;
    if !result.hardware_verified
        || !result.measurement_verified
        || !result.nonce_verified
        || !result.claims_verified
        || !result.gpu_evidence_verified
        || !result.gpu_device_claims_verified
        || !result.runtime_policy_verified
        || !matches!(result.tee_type, TeeType::SevSnp | TeeType::Tdx)
    {
        return Err(PowerError::AttestationVerificationFailed(
            "confidential-GPU verification completed without every guarantee required to issue a proof"
                .to_string(),
        ));
    }
    Ok((result, VerifiedConfidentialGpuAttestation::issue(report)))
}

#[cfg(test)]
mod tests {
    use super::{VerifiedConfidentialGpuAttestation, VerifiedHardwareAttestation};

    #[test]
    fn proof_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<VerifiedHardwareAttestation<'static>>();
        assert_send_sync::<VerifiedConfidentialGpuAttestation<'static>>();
    }
}
