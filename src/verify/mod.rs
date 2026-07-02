//! Client-side attestation report verification.
//!
//! Verifies that an [`AttestationReport`] from `GET /v1/attestation` is
//! cryptographically bound to the expected nonce and model hash, and that
//! the platform measurement matches a known-good value.
//!
//! # What this module verifies (offline, no network required)
//!
//! 1. **Claims binding** — when `AttestationClaimsV2` is present,
//!    `report_data` must equal `sha256(canonical_claims_v2)` padded to 64 bytes.
//!
//! 2. **Nonce binding** — v2 reports verify the nonce from claims; legacy reports
//!    verify that `report_data[0..32]` matches the client-supplied nonce.
//!    Prevents replay attacks: a report generated for a different request cannot
//!    be substituted.
//!
//! 3. **Model hash binding** — v2 reports verify the model digest from claims;
//!    legacy reports verify that `report_data[32..64]` matches the expected
//!    SHA-256 of the model file.
//!
//! 4. **Measurement check** — the platform launch digest matches a known-good
//!    value pinned by the operator. Proves the TEE firmware/OS has not been
//!    tampered with.
//!
//! 5. **GPU evidence binding** — v2 reports can bind NVIDIA GPU
//!    confidential-computing evidence and NRAS verdict digests into CPU TEE
//!    `report_data`. Strict GPU evidence policy also requires an expected nonce
//!    and rejects GPU evidence nonce mismatches when the claim exposes one.
//!    Verifiers can additionally pin the evidence provider, byte formats, and
//!    evidence count. The production GPU confidential profile requires a
//!    top-level GPU evidence nonce claim and an expected NVIDIA NRAS verdict
//!    digest specifically.
//!
//! 6. **Runtime policy binding** — v2 reports can bind applied prompt
//!    construction, applied default decoding policy, and GPU execution/offload
//!    policy digests into CPU TEE `report_data`.
//!
//! 7. **Request receipt digest checks** — inference responses can include an
//!    `attestation_receipt` and `attestation_receipt_sha256`; helpers in this
//!    module verify that the digest matches the receipt bytes, that receipt
//!    request-derived fields match the original request, and that receipt
//!    runtime-policy claims match the attested runtime policy.
//!
//! # What this module does NOT verify
//!
//! Hardware signature verification (AMD KDS / Intel PCS certificate chain)
//! requires network access and platform-specific tooling. Implement the
//! [`HardwareVerifier`] trait to plug in your own verifier. Production callers
//! should use [`verify_report_strict`] or [`verify_report_with_policy`] with a
//! strict policy and an operator-pinned launch measurement.
//!
//! # Example
//!
//! ```rust,no_run
//! use a3s_power::verify::{VerifyOptions, verify_report};
//! use a3s_power::tee::attestation::AttestationReport;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let report: AttestationReport = serde_json::from_str("...")?;
//! let nonce = hex::decode("deadbeef")?;
//! let model_hash = hex::decode("abcd1234")?;
//!
//! let opts = VerifyOptions {
//!     nonce: Some(nonce),
//!     expected_model_hash: Some(model_hash),
//!     expected_measurement: None,
//!     expected_gpu_evidence_digest: None,
//!     expected_gpu_verdict_digest: None,
//!     expected_gpu_evidence: None,
//!     expected_gpu_devices: None,
//!     expected_chat_template_digest: None,
//!     expected_decoding_parameters_digest: None,
//!     expected_gpu_execution_digest: None,
//!     hardware_verifier: None,
//! };
//!
//! let result = verify_report(&report, &opts)?;
//! assert!(!result.hardware_verified); // Explicit offline/permissive verification.
//! # Ok(())
//! # }
//! ```

use crate::api::receipt::{
    chat_receipt, completion_receipt, receipt_decoding_parameters_digest, receipt_digest,
    AttestationReceipt, ReceiptRequestType,
};
use crate::api::types::{ChatCompletionRequest, CompletionRequest};
use crate::error::{PowerError, Result};
use crate::tee::attestation::{
    build_claims_report_data, AttestationClaimsV2, AttestationReport, GpuDeviceClaim,
    GpuDeviceValidationClaim, RuntimePolicyClaim, TeeType,
};

pub mod hw_verify;

#[cfg(feature = "hw-verify")]
pub use hw_verify::{SevSnpVerifier, TdxVerifier};

// ============================================================================
// Public types
// ============================================================================

/// Options controlling which checks are performed during verification.
pub struct VerifyOptions<'a> {
    /// Client-supplied nonce (raw bytes). For v2 reports, verifies
    /// `claims.nonce`; for legacy reports, verifies `report_data[0..32]`.
    pub nonce: Option<Vec<u8>>,

    /// Expected SHA-256 hash of the model file (raw bytes, 32 bytes). For v2
    /// reports, verifies `claims.model.digest`; for legacy reports, verifies
    /// `report_data[32..64]`.
    pub expected_model_hash: Option<Vec<u8>>,

    /// Expected platform measurement (raw bytes). When set, verifies that
    /// `report.measurement` matches this value exactly.
    pub expected_measurement: Option<Vec<u8>>,

    /// Expected SHA-256 digest of the NVIDIA GPU CC evidence bytes.
    pub expected_gpu_evidence_digest: Option<Vec<u8>>,

    /// Expected SHA-256 digest of the NVIDIA NRAS verdict bytes.
    pub expected_gpu_verdict_digest: Option<Vec<u8>>,

    /// Expected NVIDIA GPU evidence metadata pins.
    pub expected_gpu_evidence: Option<ExpectedGpuEvidence>,

    /// Expected NVIDIA GPU topology, claims-version, and identity/version pins.
    pub expected_gpu_devices: Option<ExpectedGpuDevices>,

    /// Expected SHA-256 digest of the effective chat template string.
    pub expected_chat_template_digest: Option<Vec<u8>>,

    /// Expected SHA-256 digest of canonical applied default decoding parameters.
    pub expected_decoding_parameters_digest: Option<Vec<u8>>,

    /// Expected SHA-256 digest of canonical GPU execution parameters.
    pub expected_gpu_execution_digest: Option<Vec<u8>>,

    /// Optional hardware signature verifier. When `None`, hardware signature
    /// verification is skipped (offline mode).
    pub hardware_verifier: Option<&'a dyn HardwareVerifier>,
}

/// Expected NVIDIA GPU evidence provider/format policy.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpectedGpuEvidence {
    /// Expected GPU evidence provider label, for example `nvidia-nras`.
    pub provider: Option<String>,
    /// Expected raw evidence byte format label.
    pub evidence_format: Option<String>,
    /// Expected raw verdict byte format label.
    pub verdict_format: Option<String>,
    /// Expected number of GPU evidence entries.
    pub evidence_count: Option<u32>,
}

impl ExpectedGpuEvidence {
    pub fn is_empty(&self) -> bool {
        self.provider
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .is_empty()
            && self
                .evidence_format
                .as_deref()
                .map(str::trim)
                .unwrap_or("")
                .is_empty()
            && self
                .verdict_format
                .as_deref()
                .map(str::trim)
                .unwrap_or("")
                .is_empty()
            && self.evidence_count.is_none()
    }

    fn has_gpu_confidential_pins(&self) -> bool {
        nonempty_expected_string(self.provider.as_deref()).is_some()
            && nonempty_expected_string(self.evidence_format.as_deref()).is_some()
            && nonempty_expected_string(self.verdict_format.as_deref()).is_some()
            && self
                .evidence_count
                .is_some_and(|evidence_count| evidence_count > 0)
    }
}

/// Expected NVIDIA GPU topology, claims-version, and identity/version policy.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpectedGpuDevices {
    /// Expected number of NVIDIA GPU device claims.
    pub gpu_count: Option<u32>,
    /// Expected number of NVIDIA NVSwitch device claims.
    pub nvswitch_count: Option<u32>,
    /// Exact expected set of NVIDIA GPU UEIDs. When non-empty, every attested
    /// GPU must have a UEID, every expected UEID must be present, and no extra
    /// GPU UEID is allowed.
    pub gpu_ueids: Vec<String>,
    /// Allowed NVIDIA GPU OEM IDs. When non-empty, every attested GPU must
    /// report one of these values.
    pub oemids: Vec<String>,
    /// Allowed NVIDIA GPU claims schema versions, for example `3.0`.
    pub claims_versions: Vec<String>,
    /// Allowed NVIDIA GPU hardware model strings. When non-empty, every
    /// attested GPU must report one of these values.
    pub hwmodels: Vec<String>,
    /// Allowed NVIDIA GPU driver versions. When non-empty, every attested GPU
    /// must report one of these values.
    pub driver_versions: Vec<String>,
    /// Allowed NVIDIA GPU firmware/VBIOS versions. When non-empty, every
    /// attested GPU must report one of these values.
    pub firmware_versions: Vec<String>,
    /// Exact expected set of NVIDIA NVSwitch UEIDs. When non-empty, every
    /// attested NVSwitch must have a UEID, every expected UEID must be present,
    /// and no extra NVSwitch UEID is allowed.
    pub nvswitch_ueids: Vec<String>,
    /// Allowed NVIDIA NVSwitch OEM IDs. When non-empty, every attested
    /// NVSwitch must report one of these values.
    pub nvswitch_oemids: Vec<String>,
    /// Allowed NVIDIA NVSwitch claims schema versions, for example `3.0`.
    pub nvswitch_claims_versions: Vec<String>,
    /// Allowed NVIDIA NVSwitch hardware model strings. When non-empty, every
    /// attested NVSwitch must report one of these values.
    pub nvswitch_hwmodels: Vec<String>,
    /// Allowed NVIDIA NVSwitch firmware/BIOS versions. When non-empty, every
    /// attested NVSwitch must report one of these values.
    pub nvswitch_firmware_versions: Vec<String>,
}

impl ExpectedGpuDevices {
    pub fn is_empty(&self) -> bool {
        self.gpu_count.is_none()
            && self.nvswitch_count.is_none()
            && self.gpu_ueids.is_empty()
            && self.oemids.is_empty()
            && self.claims_versions.is_empty()
            && self.hwmodels.is_empty()
            && self.driver_versions.is_empty()
            && self.firmware_versions.is_empty()
            && self.nvswitch_ueids.is_empty()
            && self.nvswitch_oemids.is_empty()
            && self.nvswitch_claims_versions.is_empty()
            && self.nvswitch_hwmodels.is_empty()
            && self.nvswitch_firmware_versions.is_empty()
    }

    fn has_gpu_confidential_pins(&self) -> bool {
        let has_identity_version_pin = !self.hwmodels.is_empty()
            || !self.driver_versions.is_empty()
            || !self.firmware_versions.is_empty();
        let has_gpu_pins = !self.claims_versions.is_empty()
            && (!self.gpu_ueids.is_empty()
                || self
                    .gpu_count
                    .is_some_and(|gpu_count| gpu_count > 0 && has_identity_version_pin));

        let nvswitch_count = self.nvswitch_count.unwrap_or_default();
        let has_nvswitch_identity_version_pin =
            !self.nvswitch_hwmodels.is_empty() || !self.nvswitch_firmware_versions.is_empty();
        let has_required_nvswitch_pins = nvswitch_count == 0
            || (!self.nvswitch_claims_versions.is_empty()
                && (!self.nvswitch_ueids.is_empty() || has_nvswitch_identity_version_pin));

        has_gpu_pins && has_required_nvswitch_pins
    }
}

impl ExpectedGpuDevices {
    fn missing_gpu_confidential_pins_message() -> &'static str {
        "GPU confidential verification requires --gpu-claims-version plus an exact GPU UEID set, or --gpu-claims-version plus a nonzero expected GPU count and at least one GPU hwmodel/driver/firmware version pin; when --nvswitch-count is nonzero it also requires --nvswitch-claims-version plus an exact NVSwitch UEID set, or --nvswitch-claims-version plus at least one NVSwitch hwmodel/firmware version pin"
    }
}

/// Fail-open or fail-closed verification policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationPolicy {
    /// Require a hardware signature verifier to be configured and pass.
    pub require_hardware_signature: bool,
    /// Reject `tee_type = "simulated"`.
    pub reject_simulated: bool,
    /// Require `VerifyOptions::nonce` to be present and verified.
    pub require_nonce: bool,
    /// Require the expected nonce to be exactly 32 bytes.
    pub require_32_byte_nonce: bool,
    /// Require `VerifyOptions::expected_model_hash` to be present and verified.
    pub require_model_hash: bool,
    /// Require `VerifyOptions::expected_measurement` to be present and verified.
    pub require_measurement: bool,
    /// Require a v2 claim set and verify its binding to CPU TEE `report_data`.
    pub require_claims: bool,
    /// Require a GPU evidence claim and an expected evidence or verdict digest.
    pub require_gpu_evidence: bool,
    /// Require the top-level GPU evidence claim to carry the verifier nonce.
    pub require_gpu_evidence_nonce: bool,
    /// Require an expected NVIDIA NRAS verdict digest specifically.
    pub require_gpu_verdict_digest: bool,
    /// Require verifier-pinned GPU evidence provider/format/count policy.
    pub require_gpu_evidence_metadata_pins: bool,
    /// Require structured NVIDIA device identity/freshness claims.
    pub require_gpu_device_claims: bool,
    /// Require verifier-pinned NVIDIA GPU topology, claims-version, and identity/version policy.
    pub require_gpu_device_identity_pins: bool,
    /// Require a runtime policy claim and an expected runtime digest.
    pub require_runtime_policy: bool,
    /// Require a pinned GPU execution/offload digest in the runtime policy.
    pub require_gpu_execution_policy: bool,
}

impl VerificationPolicy {
    /// Permissive/offline verification. Optional checks are performed only when
    /// their inputs are present.
    pub const fn permissive() -> Self {
        Self {
            require_hardware_signature: false,
            reject_simulated: false,
            require_nonce: false,
            require_32_byte_nonce: false,
            require_model_hash: false,
            require_measurement: false,
            require_claims: false,
            require_gpu_evidence: false,
            require_gpu_evidence_nonce: false,
            require_gpu_verdict_digest: false,
            require_gpu_evidence_metadata_pins: false,
            require_gpu_device_claims: false,
            require_gpu_device_identity_pins: false,
            require_runtime_policy: false,
            require_gpu_execution_policy: false,
        }
    }

    /// Production baseline: hardware signature verification and launch
    /// measurement pinning are mandatory, and simulated reports are rejected.
    /// Callers can additionally require nonce, model-hash, v2 claims, GPU
    /// evidence, and runtime policy checks with the builder methods below.
    pub const fn strict() -> Self {
        Self {
            require_hardware_signature: true,
            reject_simulated: true,
            require_nonce: false,
            require_32_byte_nonce: false,
            require_model_hash: false,
            require_measurement: true,
            require_claims: false,
            require_gpu_evidence: false,
            require_gpu_evidence_nonce: false,
            require_gpu_verdict_digest: false,
            require_gpu_evidence_metadata_pins: false,
            require_gpu_device_claims: false,
            require_gpu_device_identity_pins: false,
            require_runtime_policy: false,
            require_gpu_execution_policy: false,
        }
    }

    /// Production GPU confidential-computing profile.
    ///
    /// This starts from [`Self::strict`] and additionally requires v2 claims,
    /// nonce freshness, NVIDIA GPU evidence, an expected NVIDIA NRAS verdict
    /// digest, structured NVIDIA device claims, runtime policy claims, and a
    /// pinned GPU execution/offload digest.
    pub const fn gpu_confidential() -> Self {
        Self::strict().require_gpu_confidential()
    }

    pub const fn require_nonce(mut self) -> Self {
        self.require_nonce = true;
        self
    }

    pub const fn require_model_hash(mut self) -> Self {
        self.require_model_hash = true;
        self
    }

    pub const fn require_measurement(mut self) -> Self {
        self.require_measurement = true;
        self
    }

    pub const fn require_claims(mut self) -> Self {
        self.require_claims = true;
        self
    }

    pub const fn require_gpu_evidence(mut self) -> Self {
        self.require_gpu_evidence = true;
        self.require_claims = true;
        self
    }

    pub const fn require_gpu_verdict_digest(mut self) -> Self {
        self.require_gpu_verdict_digest = true;
        self.require_gpu_evidence = true;
        self.require_claims = true;
        self
    }

    pub const fn require_gpu_evidence_metadata_pins(mut self) -> Self {
        self.require_gpu_evidence_metadata_pins = true;
        self.require_gpu_evidence = true;
        self.require_claims = true;
        self
    }

    pub const fn require_gpu_device_claims(mut self) -> Self {
        self.require_gpu_device_claims = true;
        self.require_gpu_evidence = true;
        self.require_claims = true;
        self
    }

    pub const fn require_gpu_device_identity_pins(mut self) -> Self {
        self.require_gpu_device_identity_pins = true;
        self.require_gpu_device_claims = true;
        self.require_gpu_evidence = true;
        self.require_claims = true;
        self
    }

    pub const fn require_runtime_policy(mut self) -> Self {
        self.require_runtime_policy = true;
        self.require_claims = true;
        self
    }

    pub const fn require_gpu_execution_policy(mut self) -> Self {
        self.require_gpu_execution_policy = true;
        self.require_runtime_policy = true;
        self.require_claims = true;
        self
    }

    pub const fn require_gpu_confidential(mut self) -> Self {
        self.require_nonce = true;
        self.require_32_byte_nonce = true;
        self.require_claims = true;
        self.require_gpu_evidence = true;
        self.require_gpu_evidence_nonce = true;
        self.require_gpu_verdict_digest = true;
        self.require_gpu_evidence_metadata_pins = true;
        self.require_gpu_device_claims = true;
        self.require_gpu_device_identity_pins = true;
        self.require_runtime_policy = true;
        self.require_gpu_execution_policy = true;
        self
    }
}

/// Extension point for hardware attestation signature verification.
///
/// Implement this trait to verify the raw TEE report against the
/// AMD KDS or Intel PCS certificate chain.
pub trait HardwareVerifier: Send + Sync {
    /// Verify the raw attestation report bytes.
    ///
    /// Returns `Ok(())` if the signature is valid, `Err` otherwise.
    fn verify_hardware_signature(&self, report: &AttestationReport) -> Result<()>;
}

/// Result of a successful attestation verification.
#[derive(Debug)]
pub struct VerifyResult {
    /// TEE type reported by the hardware.
    pub tee_type: TeeType,
    /// Whether nonce binding was checked and passed.
    pub nonce_verified: bool,
    /// Whether model hash binding was checked and passed.
    pub model_hash_verified: bool,
    /// Whether measurement was checked and passed.
    pub measurement_verified: bool,
    /// Whether hardware signature was checked and passed.
    pub hardware_verified: bool,
    /// Whether v2 claims were checked against CPU TEE `report_data`.
    pub claims_verified: bool,
    /// Whether a GPU evidence/verdict digest was checked and passed.
    pub gpu_evidence_verified: bool,
    /// Whether structured NVIDIA device identity/freshness claims were checked.
    pub gpu_device_claims_verified: bool,
    /// Whether a runtime policy digest was checked and passed.
    pub runtime_policy_verified: bool,
}

/// Expected request-level receipt fields that verifier policy can pin.
#[derive(Debug, Clone, Default)]
pub struct ExpectedReceipt {
    /// Expected model name in the receipt.
    pub model: Option<String>,
    /// Expected request type in the receipt.
    pub request_type: Option<ReceiptRequestType>,
    /// Expected SHA-256 digest of the receipt's prompt-bearing input.
    pub input_digest: Option<Vec<u8>>,
    /// Expected SHA-256 digest of the canonical receipt decoding parameter map.
    pub decoding_parameters_digest: Option<Vec<u8>>,
    /// Expected SHA-256 digest of the canonical stream-options JSON value.
    pub stream_options_digest: Option<Vec<u8>>,
    /// Expected SHA-256 digest of the canonical stop-token JSON value.
    pub stop_tokens_digest: Option<Vec<u8>>,
    /// Expected SHA-256 digest of the canonical response-format JSON value.
    pub response_format_digest: Option<Vec<u8>>,
    /// Expected SHA-256 digest of the canonical tools JSON value.
    pub tools_digest: Option<Vec<u8>>,
    /// Expected SHA-256 digest of the canonical tool-choice JSON value.
    pub tool_choice_digest: Option<Vec<u8>>,
    /// Expected SHA-256 digest of the backend effective prompt representation.
    pub effective_prompt_digest: Option<Vec<u8>>,
    /// Require the receipt to omit `effective_prompt`.
    ///
    /// Use this for opaque multimodal paths where the backend cannot expose the
    /// exact post-template prompt representation and must not overclaim one.
    pub effective_prompt_absent: bool,
    /// Expected backend label for the optional effective prompt digest.
    pub effective_prompt_backend: Option<String>,
    /// Expected semantic kind for the optional effective prompt digest.
    pub effective_prompt_kind: Option<String>,
}

impl ExpectedReceipt {
    /// Return true when no receipt policy pin is configured.
    pub fn is_empty(&self) -> bool {
        self.model.is_none()
            && self.request_type.is_none()
            && self.input_digest.is_none()
            && self.decoding_parameters_digest.is_none()
            && self.stream_options_digest.is_none()
            && self.stop_tokens_digest.is_none()
            && self.response_format_digest.is_none()
            && self.tools_digest.is_none()
            && self.tool_choice_digest.is_none()
            && self.effective_prompt_digest.is_none()
            && !self.effective_prompt_absent
            && self.effective_prompt_backend.is_none()
            && self.effective_prompt_kind.is_none()
    }
}

// ============================================================================
// Core verification logic
// ============================================================================

/// Verify an attestation report against the provided options.
///
/// Returns [`VerifyResult`] on success, or the first [`PowerError`] encountered.
pub fn verify_report(report: &AttestationReport, opts: &VerifyOptions<'_>) -> Result<VerifyResult> {
    verify_report_with_policy(report, opts, VerificationPolicy::permissive())
}

/// Verify an attestation report under a caller-supplied policy.
///
/// Use [`VerificationPolicy::strict`] for production verifier paths. The legacy
/// [`verify_report`] helper intentionally remains permissive for explicit
/// offline/development use.
pub fn verify_report_with_policy(
    report: &AttestationReport,
    opts: &VerifyOptions<'_>,
    policy: VerificationPolicy,
) -> Result<VerifyResult> {
    validate_required_inputs(report, opts, policy)?;

    let mut result = VerifyResult {
        tee_type: report.tee_type,
        nonce_verified: false,
        model_hash_verified: false,
        measurement_verified: false,
        hardware_verified: false,
        claims_verified: false,
        gpu_evidence_verified: false,
        gpu_device_claims_verified: false,
        runtime_policy_verified: false,
    };

    if report.claims.is_some() {
        verify_claims_binding(report)?;
        result.claims_verified = true;
    }

    // 1. Nonce binding check
    if let Some(ref nonce) = opts.nonce {
        if let Some(ref claims) = report.claims {
            verify_claims_nonce_binding(claims, nonce)?;
        } else {
            verify_nonce_binding(&report.report_data, nonce)?;
        }
        result.nonce_verified = true;
    }

    // 2. Model hash binding check
    if let Some(ref model_hash) = opts.expected_model_hash {
        if let Some(ref claims) = report.claims {
            verify_claims_model_hash_binding(claims, model_hash)?;
        } else {
            verify_model_hash_binding(&report.report_data, model_hash)?;
        }
        result.model_hash_verified = true;
    }

    // 3. Measurement check
    if let Some(ref expected) = opts.expected_measurement {
        verify_measurement(&report.measurement, expected)?;
        result.measurement_verified = true;
    }

    // 4. GPU evidence/verdict digest and metadata checks
    let expected_gpu_evidence = opts
        .expected_gpu_evidence
        .as_ref()
        .filter(|expected| !expected.is_empty());
    if opts.expected_gpu_evidence_digest.is_some()
        || opts.expected_gpu_verdict_digest.is_some()
        || expected_gpu_evidence.is_some()
    {
        let claims = report.claims.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "GPU evidence verification requires v2 attestation claims".to_string(),
            )
        })?;
        verify_claims_gpu_evidence_binding(
            claims,
            opts.expected_gpu_evidence_digest.as_deref(),
            opts.expected_gpu_verdict_digest.as_deref(),
        )?;
        if let Some(expected) = expected_gpu_evidence {
            verify_claims_expected_gpu_evidence(claims, expected)?;
        }
        result.gpu_evidence_verified = true;
    }

    // 5. Structured NVIDIA device identity/freshness checks
    let expected_gpu_devices = opts
        .expected_gpu_devices
        .as_ref()
        .filter(|expected| !expected.is_empty());
    if policy.require_gpu_device_claims || expected_gpu_devices.is_some() {
        let claims = report.claims.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "GPU device-claim verification requires v2 attestation claims".to_string(),
            )
        })?;
        verify_claims_gpu_device_claims(claims)?;
        if let Some(expected) = expected_gpu_devices {
            verify_claims_expected_gpu_devices(claims, expected)?;
        }
        result.gpu_device_claims_verified = true;
    }

    // 6. Runtime prompt/decoding policy digest check
    if opts.expected_chat_template_digest.is_some()
        || opts.expected_decoding_parameters_digest.is_some()
        || opts.expected_gpu_execution_digest.is_some()
    {
        let claims = report.claims.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "runtime policy verification requires v2 attestation claims".to_string(),
            )
        })?;
        verify_claims_runtime_policy_binding(
            claims,
            opts.expected_chat_template_digest.as_deref(),
            opts.expected_decoding_parameters_digest.as_deref(),
            opts.expected_gpu_execution_digest.as_deref(),
        )?;
        result.runtime_policy_verified = true;
    }

    // 7. Hardware signature verification (optional)
    if let Some(verifier) = opts.hardware_verifier {
        verifier.verify_hardware_signature(report)?;
        result.hardware_verified = true;
    }

    Ok(result)
}

/// Verify an attestation report with production fail-closed defaults.
///
/// This requires hardware signature verification and rejects simulated reports.
/// It does not invent deployment-specific expected nonce/model/measurement
/// values; set those in [`VerifyOptions`] and use [`verify_report_with_policy`]
/// with the corresponding `require_*` policy methods when they are mandatory.
pub fn verify_report_strict(
    report: &AttestationReport,
    opts: &VerifyOptions<'_>,
) -> Result<VerifyResult> {
    verify_report_with_policy(report, opts, VerificationPolicy::strict())
}

fn validate_required_inputs(
    report: &AttestationReport,
    opts: &VerifyOptions<'_>,
    policy: VerificationPolicy,
) -> Result<()> {
    if policy.reject_simulated && report.tee_type == TeeType::Simulated {
        return Err(PowerError::AttestationVerificationFailed(
            "simulated TEE reports are rejected by strict verification policy".to_string(),
        ));
    }

    if policy.require_hardware_signature && opts.hardware_verifier.is_none() {
        return Err(PowerError::AttestationVerificationFailed(
            "hardware signature verification is required by strict verification policy".to_string(),
        ));
    }

    if policy.require_nonce && opts.nonce.is_none() {
        return Err(PowerError::AttestationVerificationFailed(
            "nonce verification is required by verification policy".to_string(),
        ));
    }
    if policy.require_32_byte_nonce {
        let nonce = opts.nonce.as_deref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "32-byte nonce verification is required by verification policy".to_string(),
            )
        })?;
        if nonce.len() != 32 {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "GPU confidential verification requires a 32-byte nonce, got {} bytes",
                nonce.len()
            )));
        }
    }

    if policy.require_model_hash && opts.expected_model_hash.is_none() {
        return Err(PowerError::AttestationVerificationFailed(
            "model hash verification is required by verification policy".to_string(),
        ));
    }

    if policy.require_measurement && opts.expected_measurement.is_none() {
        return Err(PowerError::AttestationVerificationFailed(
            "measurement verification is required by verification policy".to_string(),
        ));
    }

    if policy.require_claims && report.claims.is_none() {
        return Err(PowerError::AttestationVerificationFailed(
            "v2 attestation claims are required by verification policy".to_string(),
        ));
    }

    if policy.require_gpu_evidence {
        let claims = report.claims.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "GPU evidence verification requires v2 attestation claims".to_string(),
            )
        })?;
        if claims.gpu.is_none() {
            return Err(PowerError::AttestationVerificationFailed(
                "GPU evidence claim is required by verification policy".to_string(),
            ));
        }
        if claims.nonce.is_none() {
            return Err(PowerError::AttestationVerificationFailed(
                "GPU evidence verification requires a nonce claim for freshness".to_string(),
            ));
        }
        if opts.nonce.is_none() {
            return Err(PowerError::AttestationVerificationFailed(
                "GPU evidence verification requires an expected nonce".to_string(),
            ));
        }
        if policy.require_gpu_evidence_nonce
            && claims
                .gpu
                .as_ref()
                .and_then(|gpu| gpu.nonce.as_ref())
                .is_none()
        {
            return Err(PowerError::AttestationVerificationFailed(
                "GPU confidential verification requires a GPU evidence nonce claim".to_string(),
            ));
        }
        if opts.expected_gpu_evidence_digest.is_none() && opts.expected_gpu_verdict_digest.is_none()
        {
            return Err(PowerError::AttestationVerificationFailed(
                "GPU evidence verification requires an expected evidence or verdict digest"
                    .to_string(),
            ));
        }
    }

    if policy.require_gpu_verdict_digest && opts.expected_gpu_verdict_digest.is_none() {
        return Err(PowerError::AttestationVerificationFailed(
            "GPU confidential verification requires an expected NVIDIA NRAS verdict digest"
                .to_string(),
        ));
    }

    if policy.require_gpu_evidence_metadata_pins {
        let expected = opts
            .expected_gpu_evidence
            .as_ref()
            .filter(|expected| !expected.is_empty())
            .ok_or_else(|| {
                PowerError::AttestationVerificationFailed(
                    "GPU confidential verification requires expected GPU evidence provider/format/count pins"
                        .to_string(),
                )
            })?;
        if !expected.has_gpu_confidential_pins() {
            return Err(PowerError::AttestationVerificationFailed(
                "GPU confidential verification requires --gpu-provider, --gpu-evidence-format, --gpu-verdict-format, and a nonzero --gpu-evidence-count"
                    .to_string(),
            ));
        }
    }

    if policy.require_gpu_device_claims {
        let claims = report.claims.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "GPU device-claim verification requires v2 attestation claims".to_string(),
            )
        })?;
        let gpu = claims.gpu.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "GPU device-claim verification requires a GPU evidence claim".to_string(),
            )
        })?;
        if gpu.devices.is_empty() {
            return Err(PowerError::AttestationVerificationFailed(
                "GPU device-claim verification requires structured NVIDIA device claims"
                    .to_string(),
            ));
        }
    }

    if policy.require_gpu_device_identity_pins {
        let expected = opts
            .expected_gpu_devices
            .as_ref()
            .filter(|expected| !expected.is_empty())
            .ok_or_else(|| {
                PowerError::AttestationVerificationFailed(
                    "GPU confidential verification requires expected GPU identity/version pins"
                        .to_string(),
                )
            })?;
        if !expected.has_gpu_confidential_pins() {
            return Err(PowerError::AttestationVerificationFailed(
                ExpectedGpuDevices::missing_gpu_confidential_pins_message().to_string(),
            ));
        }
    }

    if policy.require_gpu_execution_policy && opts.expected_gpu_execution_digest.is_none() {
        return Err(PowerError::AttestationVerificationFailed(
            "GPU confidential verification requires an expected GPU execution digest".to_string(),
        ));
    }

    if policy.require_runtime_policy {
        let claims = report.claims.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "runtime policy verification requires v2 attestation claims".to_string(),
            )
        })?;
        if claims.runtime.is_none() {
            return Err(PowerError::AttestationVerificationFailed(
                "runtime policy claim is required by verification policy".to_string(),
            ));
        }
        if opts.expected_chat_template_digest.is_none()
            && opts.expected_decoding_parameters_digest.is_none()
            && opts.expected_gpu_execution_digest.is_none()
        {
            return Err(PowerError::AttestationVerificationFailed(
                "runtime policy verification requires an expected chat template, decoding parameter, or GPU execution digest"
                    .to_string(),
            ));
        }
    }

    Ok(())
}

/// Verify that CPU TEE `report_data` binds the included v2 claims.
pub fn verify_claims_binding(report: &AttestationReport) -> Result<()> {
    let claims = report.claims.as_ref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed(
            "attestation report does not include v2 claims".to_string(),
        )
    })?;

    if claims.tee_type != report.tee_type {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "claims tee_type mismatch: claims.tee_type = {}, report.tee_type = {}",
            claims.tee_type, report.tee_type,
        )));
    }

    verify_claims_well_formed(claims)?;

    let expected_report_data = build_claims_report_data(claims)?;
    if !constant_time_eq(&report.report_data, &expected_report_data) {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "claims binding mismatch: report_data = {}, expected {}",
            hex::encode(&report.report_data),
            hex::encode(&expected_report_data),
        )));
    }

    Ok(())
}

fn verify_claims_well_formed(claims: &AttestationClaimsV2) -> Result<()> {
    if claims.schema != AttestationClaimsV2::SCHEMA {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "claims schema mismatch: claims.schema = {}, expected {}",
            claims.schema,
            AttestationClaimsV2::SCHEMA,
        )));
    }

    if let Some(model) = claims.model.as_ref() {
        if model.name.trim().is_empty() {
            return Err(PowerError::AttestationVerificationFailed(
                "claims.model.name must not be empty".to_string(),
            ));
        }
        require_sha256_digest_bytes("claims.model.digest", &model.digest)?;
        require_optional_sha256_digest_bytes(
            "claims.model.plaintext_digest",
            model.plaintext_digest.as_deref(),
        )?;
        require_optional_sha256_digest_bytes(
            "claims.model.ciphertext_digest",
            model.ciphertext_digest.as_deref(),
        )?;
    }

    if let Some(gpu) = claims.gpu.as_ref() {
        if gpu.provider.trim().is_empty() {
            return Err(PowerError::AttestationVerificationFailed(
                "claims.gpu.provider must not be empty".to_string(),
            ));
        }
        require_sha256_digest_bytes("claims.gpu.evidence_digest", &gpu.evidence_digest)?;
        require_optional_sha256_digest_bytes(
            "claims.gpu.verdict_digest",
            gpu.verdict_digest.as_deref(),
        )?;
    }

    if let Some(runtime) = claims.runtime.as_ref() {
        verify_runtime_policy_well_formed("claims.runtime", runtime)?;
    }

    Ok(())
}

/// Verify that v2 claims bind the expected nonce.
pub fn verify_claims_nonce_binding(claims: &AttestationClaimsV2, nonce: &[u8]) -> Result<()> {
    let claim_nonce = claims.nonce.as_deref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed("v2 claims do not include a nonce".to_string())
    })?;

    if !constant_time_eq(claim_nonce, nonce) {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "Claims nonce mismatch: claims.nonce = {}, expected {}",
            hex::encode(claim_nonce),
            hex::encode(nonce),
        )));
    }

    Ok(())
}

/// Verify that v2 claims bind the expected model digest.
pub fn verify_claims_model_hash_binding(
    claims: &AttestationClaimsV2,
    model_hash: &[u8],
) -> Result<()> {
    require_sha256_digest_bytes("expected model hash", model_hash)?;

    let model = claims.model.as_ref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed(
            "v2 claims do not include a model digest".to_string(),
        )
    })?;

    if !constant_time_eq(&model.digest, model_hash) {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "Claims model hash mismatch: claims.model.digest = {}, expected {}",
            hex::encode(&model.digest),
            hex::encode(model_hash),
        )));
    }

    Ok(())
}

/// Verify that v2 claims bind the expected NVIDIA GPU CC evidence/verdict digests.
pub fn verify_claims_gpu_evidence_binding(
    claims: &AttestationClaimsV2,
    expected_evidence_digest: Option<&[u8]>,
    expected_verdict_digest: Option<&[u8]>,
) -> Result<()> {
    let gpu = claims.gpu.as_ref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed(
            "v2 claims do not include a GPU evidence claim".to_string(),
        )
    })?;

    if let Some(gpu_nonce) = gpu.nonce.as_deref() {
        let claim_nonce = claims.nonce.as_deref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "GPU evidence claim includes a nonce but v2 claims do not include a CPU nonce"
                    .to_string(),
            )
        })?;
        if !constant_time_eq(gpu_nonce, claim_nonce) {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "GPU evidence nonce mismatch: claims.gpu.nonce = {}, claims.nonce = {}",
                hex::encode(gpu_nonce),
                hex::encode(claim_nonce),
            )));
        }
    }

    if let Some(expected) = expected_evidence_digest {
        if !constant_time_eq(&gpu.evidence_digest, expected) {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "GPU evidence digest mismatch: claims.gpu.evidence_digest = {}, expected {}",
                hex::encode(&gpu.evidence_digest),
                hex::encode(expected),
            )));
        }
    }

    if let Some(expected) = expected_verdict_digest {
        let verdict = gpu.verdict_digest.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "v2 GPU evidence claim does not include a verdict digest".to_string(),
            )
        })?;
        if !constant_time_eq(verdict, expected) {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "GPU verdict digest mismatch: claims.gpu.verdict_digest = {}, expected {}",
                hex::encode(verdict),
                hex::encode(expected),
            )));
        }
    }

    Ok(())
}

/// Verify NVIDIA GPU evidence provider/format pins against the GPU claim.
pub fn verify_claims_expected_gpu_evidence(
    claims: &AttestationClaimsV2,
    expected: &ExpectedGpuEvidence,
) -> Result<()> {
    if expected.is_empty() {
        return Ok(());
    }

    let gpu = claims.gpu.as_ref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed(
            "v2 claims do not include a GPU evidence claim".to_string(),
        )
    })?;

    if let Some(provider) = nonempty_expected_string(expected.provider.as_deref()) {
        if gpu.provider != provider {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "GPU evidence provider mismatch: claims.gpu.provider = {:?}, expected {:?}",
                gpu.provider, provider
            )));
        }
    }

    verify_optional_gpu_string_claim(
        "evidence_format",
        gpu.evidence_format.as_deref(),
        expected.evidence_format.as_deref(),
    )?;
    verify_optional_gpu_string_claim(
        "verdict_format",
        gpu.verdict_format.as_deref(),
        expected.verdict_format.as_deref(),
    )?;

    if let Some(expected_count) = expected.evidence_count {
        let actual = gpu.evidence_count.ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "GPU evidence claim does not include evidence_count required by policy".to_string(),
            )
        })?;
        if actual != expected_count {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "GPU evidence_count mismatch: claims.gpu.evidence_count = {}, expected {}",
                actual, expected_count
            )));
        }
    }

    Ok(())
}

fn nonempty_expected_string(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn verify_optional_gpu_string_claim(
    field: &str,
    actual: Option<&str>,
    expected: Option<&str>,
) -> Result<()> {
    let Some(expected) = nonempty_expected_string(expected) else {
        return Ok(());
    };
    let actual = actual.ok_or_else(|| {
        PowerError::AttestationVerificationFailed(format!(
            "GPU evidence claim does not include {field} required by policy"
        ))
    })?;
    if actual != expected {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "GPU {field} mismatch: claims.gpu.{field} = {:?}, expected {:?}",
            actual, expected
        )));
    }

    Ok(())
}

/// Verify structured NVIDIA device identity/freshness claims.
///
/// This check validates the normalized fields Power extracts from NVIDIA
/// NVAT/NRAS verdict claims. It is intended to be used together with
/// `verify_claims_gpu_evidence_binding`, because the raw verdict digest is the
/// cryptographic link back to NVIDIA evidence while these fields make policy
/// decisions explicit.
pub fn verify_claims_gpu_device_claims(claims: &AttestationClaimsV2) -> Result<()> {
    let gpu = claims.gpu.as_ref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed(
            "v2 claims do not include a GPU evidence claim".to_string(),
        )
    })?;

    if gpu.devices.is_empty() {
        return Err(PowerError::AttestationVerificationFailed(
            "GPU evidence claim does not include structured NVIDIA device claims".to_string(),
        ));
    }

    let claim_nonce = claims.nonce.as_deref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed(
            "GPU device-claim verification requires a CPU nonce claim".to_string(),
        )
    })?;

    let mut gpu_count = 0usize;
    for device in &gpu.devices {
        if !matches!(device.device_type.as_str(), "gpu" | "nvswitch") {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "GPU device claim {} has unsupported device_type {:?}",
                device.index, device.device_type
            )));
        }
        if device.device_type == "gpu" {
            gpu_count += 1;
        }

        let device_nonce = device.attestation_nonce.as_deref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(format!(
                "GPU device claim {} does not include an attestation nonce",
                device.index
            ))
        })?;
        if !constant_time_eq(device_nonce, claim_nonce) {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "GPU device nonce mismatch: device {} nonce = {}, claims.nonce = {}",
                device.index,
                hex::encode(device_nonce),
                hex::encode(claim_nonce),
            )));
        }

        if device.measurements_result.as_deref() != Some("success") {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "GPU device claim {} measres must be \"success\", got {:?}",
                device.index,
                device.measurements_result.as_deref()
            )));
        }

        verify_gpu_device_security_state(device)?;
        verify_gpu_device_validation(device.index, &device.device_type, &device.validation)?;
    }

    if gpu_count == 0 {
        return Err(PowerError::AttestationVerificationFailed(
            "GPU evidence claim does not include any NVIDIA GPU device claims".to_string(),
        ));
    }

    Ok(())
}

/// Verify NVIDIA GPU topology, claims-version, and identity/version pins.
pub fn verify_claims_expected_gpu_devices(
    claims: &AttestationClaimsV2,
    expected: &ExpectedGpuDevices,
) -> Result<()> {
    if expected.is_empty() {
        return Ok(());
    }

    verify_claims_gpu_device_claims(claims)?;

    let gpu = claims.gpu.as_ref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed(
            "v2 claims do not include a GPU evidence claim".to_string(),
        )
    })?;
    let devices: Vec<&GpuDeviceClaim> = gpu
        .devices
        .iter()
        .filter(|device| device.device_type == "gpu")
        .collect();
    let nvswitch_devices: Vec<&GpuDeviceClaim> = gpu
        .devices
        .iter()
        .filter(|device| device.device_type == "nvswitch")
        .collect();

    verify_expected_device_count("GPU", devices.len(), expected.gpu_count)?;
    verify_expected_device_count("NVSwitch", nvswitch_devices.len(), expected.nvswitch_count)?;

    require_allowed_device_string_claim(
        "GPU",
        &devices,
        &expected.claims_versions,
        "claims_version",
        |device| device.claims_version.as_deref(),
    )?;

    require_exact_device_ueid_set("GPU", &devices, &expected.gpu_ueids)?;

    require_allowed_device_string_claim(
        "GPU",
        &devices,
        &expected.hwmodels,
        "hwmodel",
        |device| device.hwmodel.as_deref(),
    )?;
    require_allowed_device_string_claim("GPU", &devices, &expected.oemids, "oemid", |device| {
        device.oemid.as_deref()
    })?;
    require_allowed_device_string_claim(
        "GPU",
        &devices,
        &expected.driver_versions,
        "driver_version",
        |device| device.driver_version.as_deref(),
    )?;
    require_allowed_device_string_claim(
        "GPU",
        &devices,
        &expected.firmware_versions,
        "firmware_version",
        |device| device.firmware_version.as_deref(),
    )?;

    require_allowed_device_string_claim(
        "NVSwitch",
        &nvswitch_devices,
        &expected.nvswitch_claims_versions,
        "claims_version",
        |device| device.claims_version.as_deref(),
    )?;
    require_exact_device_ueid_set("NVSwitch", &nvswitch_devices, &expected.nvswitch_ueids)?;
    require_allowed_device_string_claim(
        "NVSwitch",
        &nvswitch_devices,
        &expected.nvswitch_hwmodels,
        "hwmodel",
        |device| device.hwmodel.as_deref(),
    )?;
    require_allowed_device_string_claim(
        "NVSwitch",
        &nvswitch_devices,
        &expected.nvswitch_oemids,
        "oemid",
        |device| device.oemid.as_deref(),
    )?;
    require_allowed_device_string_claim(
        "NVSwitch",
        &nvswitch_devices,
        &expected.nvswitch_firmware_versions,
        "firmware_version",
        |device| device.firmware_version.as_deref(),
    )?;

    Ok(())
}

fn verify_expected_device_count(label: &str, actual: usize, expected: Option<u32>) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let actual = u32::try_from(actual).map_err(|_| {
        PowerError::AttestationVerificationFailed(format!(
            "attested {label} device count exceeds verifier range"
        ))
    })?;
    if actual != expected {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "{label} device count mismatch: actual = {}, expected {}",
            actual, expected
        )));
    }
    Ok(())
}

fn sorted_unique_strings(values: Vec<String>) -> Vec<String> {
    let mut values: Vec<String> = values
        .into_iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect();
    values.sort();
    values.dedup();
    values
}

fn require_exact_device_ueid_set(
    label: &str,
    devices: &[&GpuDeviceClaim],
    expected_ueids: &[String],
) -> Result<()> {
    if expected_ueids.is_empty() {
        return Ok(());
    }

    let mut actual_ueids = Vec::with_capacity(devices.len());
    for device in devices {
        let ueid = device.ueid.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(format!(
                "{label} device claim {} does not include a UEID required by policy",
                device.index
            ))
        })?;
        actual_ueids.push(ueid.clone());
    }

    let actual = sorted_unique_strings(actual_ueids);
    let expected = sorted_unique_strings(expected_ueids.to_vec());
    if actual != expected {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "{label} UEID set mismatch: actual = {:?}, expected {:?}",
            actual, expected
        )));
    }

    Ok(())
}

fn require_allowed_device_string_claim(
    label: &str,
    devices: &[&GpuDeviceClaim],
    allowed: &[String],
    field: &str,
    actual: impl Fn(&GpuDeviceClaim) -> Option<&str>,
) -> Result<()> {
    let allowed = sorted_unique_strings(allowed.to_vec());
    if allowed.is_empty() {
        return Ok(());
    }

    for device in devices {
        let value = actual(device).ok_or_else(|| {
            PowerError::AttestationVerificationFailed(format!(
                "{label} device claim {} does not include {field} required by policy",
                device.index
            ))
        })?;
        if !allowed.iter().any(|expected| expected == value) {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "{label} device claim {} {field} mismatch: actual = {:?}, allowed {:?}",
                device.index, value, allowed
            )));
        }
    }

    Ok(())
}

fn verify_gpu_device_security_state(device: &GpuDeviceClaim) -> Result<()> {
    match device.secure_boot {
        Some(true) => {}
        Some(false) => {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "GPU device claim {} {} secure_boot is false",
                device.index, device.device_type
            )));
        }
        None => {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "GPU device claim {} {} secure_boot is missing",
                device.index, device.device_type
            )));
        }
    }

    match device.debug_status.as_deref() {
        Some(value) if value.eq_ignore_ascii_case("disabled") => Ok(()),
        Some(value) => Err(PowerError::AttestationVerificationFailed(format!(
            "GPU device claim {} {} debug_status must be \"disabled\", got {:?}",
            device.index, device.device_type, value
        ))),
        None => Err(PowerError::AttestationVerificationFailed(format!(
            "GPU device claim {} {} debug_status is missing",
            device.index, device.device_type
        ))),
    }
}

fn verify_gpu_device_validation(
    index: u32,
    device_type: &str,
    validation: &GpuDeviceValidationClaim,
) -> Result<()> {
    require_validation_true(index, device_type, "arch_check", validation.arch_check)?;
    require_validation_true(
        index,
        device_type,
        "attestation_report_cert_chain_fwid_match",
        validation.attestation_report_cert_chain_fwid_match,
    )?;
    require_validation_true(
        index,
        device_type,
        "attestation_report_parsed",
        validation.attestation_report_parsed,
    )?;
    require_validation_true(
        index,
        device_type,
        "attestation_report_nonce_match",
        validation.attestation_report_nonce_match,
    )?;
    require_validation_true(
        index,
        device_type,
        "attestation_report_signature_verified",
        validation.attestation_report_signature_verified,
    )?;
    require_validation_true(
        index,
        device_type,
        "firmware_rim_fetched",
        validation.firmware_rim_fetched,
    )?;
    require_validation_true(
        index,
        device_type,
        "firmware_rim_schema_validated",
        validation.firmware_rim_schema_validated,
    )?;
    require_validation_true(
        index,
        device_type,
        "firmware_rim_signature_verified",
        validation.firmware_rim_signature_verified,
    )?;
    require_validation_true(
        index,
        device_type,
        "firmware_rim_version_match",
        validation.firmware_rim_version_match,
    )?;
    require_validation_true(
        index,
        device_type,
        "firmware_rim_measurements_available",
        validation.firmware_rim_measurements_available,
    )?;

    if device_type == "gpu" {
        require_validation_true(
            index,
            device_type,
            "driver_rim_fetched",
            validation.driver_rim_fetched,
        )?;
        require_validation_true(
            index,
            device_type,
            "driver_rim_schema_validated",
            validation.driver_rim_schema_validated,
        )?;
        require_validation_true(
            index,
            device_type,
            "driver_rim_signature_verified",
            validation.driver_rim_signature_verified,
        )?;
        require_validation_true(
            index,
            device_type,
            "driver_rim_version_match",
            validation.driver_rim_version_match,
        )?;
        require_validation_true(
            index,
            device_type,
            "driver_rim_measurements_available",
            validation.driver_rim_measurements_available,
        )?;
        require_validation_true(
            index,
            device_type,
            "firmware_index_no_conflict",
            validation.firmware_index_no_conflict,
        )?;
    }

    Ok(())
}

fn require_validation_true(
    index: u32,
    device_type: &str,
    field: &str,
    value: Option<bool>,
) -> Result<()> {
    match value {
        Some(true) => Ok(()),
        Some(false) => Err(PowerError::AttestationVerificationFailed(format!(
            "GPU device claim {index} {device_type} validation {field} is false"
        ))),
        None => Err(PowerError::AttestationVerificationFailed(format!(
            "GPU device claim {index} {device_type} validation {field} is missing"
        ))),
    }
}

/// Verify that v2 claims bind the expected runtime policy digests.
pub fn verify_claims_runtime_policy_binding(
    claims: &AttestationClaimsV2,
    expected_chat_template_digest: Option<&[u8]>,
    expected_decoding_parameters_digest: Option<&[u8]>,
    expected_gpu_execution_digest: Option<&[u8]>,
) -> Result<()> {
    let runtime = claims.runtime.as_ref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed(
            "v2 claims do not include a runtime policy claim".to_string(),
        )
    })?;

    if let Some(expected) = expected_chat_template_digest {
        let prompt = runtime.prompt.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "v2 runtime policy claim does not include prompt policy".to_string(),
            )
        })?;
        let actual = prompt.chat_template_sha256.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "v2 prompt policy claim does not include a chat template digest".to_string(),
            )
        })?;
        if !constant_time_eq(actual, expected) {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "Chat template digest mismatch: claims.runtime.prompt.chat_template_sha256 = {}, expected {}",
                hex::encode(actual),
                hex::encode(expected),
            )));
        }
    }

    if let Some(expected) = expected_decoding_parameters_digest {
        let decoding = runtime.decoding.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "v2 runtime policy claim does not include decoding policy".to_string(),
            )
        })?;
        if !constant_time_eq(&decoding.parameters_sha256, expected) {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "Decoding parameters digest mismatch: claims.runtime.decoding.parameters_sha256 = {}, expected {}",
                hex::encode(&decoding.parameters_sha256),
                hex::encode(expected),
            )));
        }
    }

    if let Some(expected) = expected_gpu_execution_digest {
        let execution = runtime.execution.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "v2 runtime policy claim does not include execution policy".to_string(),
            )
        })?;
        if !constant_time_eq(&execution.gpu_sha256, expected) {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "GPU execution digest mismatch: claims.runtime.execution.gpu_sha256 = {}, expected {}",
                hex::encode(&execution.gpu_sha256),
                hex::encode(expected),
            )));
        }
    }

    Ok(())
}

/// Verify that `report_data[0..32]` matches the client nonce.
///
/// The nonce is zero-padded to 32 bytes if shorter. This matches the
/// layout produced by `build_report_data()` on the server side.
pub fn verify_nonce_binding(report_data: &[u8], nonce: &[u8]) -> Result<()> {
    if report_data.len() < 32 {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "report_data too short: {} bytes (need at least 32)",
            report_data.len()
        )));
    }

    let nonce_len = nonce.len().min(32);
    let report_nonce = &report_data[..nonce_len];
    let provided_nonce = &nonce[..nonce_len];

    // Constant-time comparison to prevent timing side-channels
    if !constant_time_eq(report_nonce, provided_nonce) {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "Nonce mismatch: report_data[0..{}] = {}, expected {}",
            nonce_len,
            hex::encode(report_nonce),
            hex::encode(provided_nonce),
        )));
    }

    // Verify the zero-padding after the nonce is actually zero
    if nonce_len < 32 {
        let padding = &report_data[nonce_len..32];
        if padding.iter().any(|&b| b != 0) {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "Nonce padding is not zero: report_data[{}..32] = {}",
                nonce_len,
                hex::encode(padding),
            )));
        }
    }

    Ok(())
}

/// Verify that `report_data[32..64]` matches the expected model SHA-256 hash.
///
/// The expected model hash must be a full 32-byte SHA-256 digest.
pub fn verify_model_hash_binding(report_data: &[u8], model_hash: &[u8]) -> Result<()> {
    require_sha256_digest_bytes("expected model hash", model_hash)?;

    if report_data.len() < 64 {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "report_data too short for model hash: {} bytes (need 64)",
            report_data.len()
        )));
    }

    let report_hash = &report_data[32..64];

    if !constant_time_eq(report_hash, model_hash) {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "Model hash mismatch: report_data[32..64] = {}, expected {}",
            hex::encode(report_hash),
            hex::encode(model_hash),
        )));
    }

    Ok(())
}

/// Verify that the platform measurement matches the expected value.
pub fn verify_measurement(measurement: &[u8], expected: &[u8]) -> Result<()> {
    if measurement.len() != 48 {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "report measurement must be a 48-byte launch measurement, got {} bytes",
            measurement.len(),
        )));
    }
    if expected.len() != 48 {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "expected measurement must be a 48-byte launch measurement, got {} bytes",
            expected.len(),
        )));
    }

    if !constant_time_eq(measurement, expected) {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "Measurement mismatch: got {}, expected {}",
            hex::encode(measurement),
            hex::encode(expected),
        )));
    }
    Ok(())
}

/// Verify that an inference receipt matches the expected 32-byte SHA-256 digest.
pub fn verify_receipt_digest(receipt: &AttestationReceipt, expected_digest: &[u8]) -> Result<()> {
    if expected_digest.len() != 32 {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "receipt digest must be 32 bytes, got {}",
            expected_digest.len()
        )));
    }

    let actual_hex = receipt_digest(receipt)?;
    let actual = hex::decode(&actual_hex).map_err(|e| {
        PowerError::AttestationVerificationFailed(format!(
            "failed to decode computed receipt digest: {e}"
        ))
    })?;

    if !constant_time_eq(&actual, expected_digest) {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "Receipt digest mismatch: receipt digest = {}, expected {}",
            actual_hex,
            hex::encode(expected_digest),
        )));
    }

    Ok(())
}

/// Verify that an inference receipt matches a hex SHA-256 digest string.
pub fn verify_receipt_digest_hex(
    receipt: &AttestationReceipt,
    expected_digest_hex: &str,
) -> Result<()> {
    let digest_hex = expected_digest_hex
        .strip_prefix("sha256:")
        .unwrap_or(expected_digest_hex);
    if digest_hex.len() != 64 {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "receipt digest must be 64 hex characters, got {}",
            digest_hex.len()
        )));
    }
    if !digest_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(PowerError::AttestationVerificationFailed(
            "receipt digest contains non-hex characters".to_string(),
        ));
    }

    let expected = hex::decode(digest_hex).map_err(|e| {
        PowerError::AttestationVerificationFailed(format!(
            "failed to decode expected receipt digest: {e}"
        ))
    })?;
    verify_receipt_digest(receipt, &expected)
}

/// Verify that a receipt has the expected schema and digest-shaped fields.
///
/// This is intentionally separate from `verify_receipt_digest()`: a verifier may
/// receive a structurally malformed receipt even when it has not pinned the
/// whole canonical receipt hash.
pub fn verify_receipt_well_formed(receipt: &AttestationReceipt) -> Result<()> {
    if receipt.schema != AttestationReceipt::SCHEMA {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "receipt schema mismatch: receipt.schema = {}, expected {}",
            receipt.schema,
            AttestationReceipt::SCHEMA,
        )));
    }

    if receipt.model.trim().is_empty() {
        return Err(PowerError::AttestationVerificationFailed(
            "receipt model must not be empty".to_string(),
        ));
    }

    let expected_input_kind = match receipt.request_type {
        ReceiptRequestType::ChatCompletion => "chat.messages",
        ReceiptRequestType::TextCompletion => "text.prompt",
    };
    if receipt.input.kind != expected_input_kind {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "receipt input kind mismatch: receipt.input.kind = {}, expected {} for {:?}",
            receipt.input.kind, expected_input_kind, receipt.request_type,
        )));
    }

    require_sha256_hex("receipt.input.sha256", &receipt.input.sha256)?;
    require_optional_sha256_hex(
        "receipt.decoding.stream_options_sha256",
        receipt.decoding.stream_options_sha256.as_deref(),
    )?;
    require_optional_sha256_hex(
        "receipt.decoding.stop_tokens_sha256",
        receipt.decoding.stop_tokens_sha256.as_deref(),
    )?;
    require_optional_sha256_hex(
        "receipt.decoding.response_format_sha256",
        receipt.decoding.response_format_sha256.as_deref(),
    )?;
    require_optional_sha256_hex(
        "receipt.decoding.tools_sha256",
        receipt.decoding.tools_sha256.as_deref(),
    )?;
    require_optional_sha256_hex(
        "receipt.decoding.tool_choice_sha256",
        receipt.decoding.tool_choice_sha256.as_deref(),
    )?;

    if let Some(effective_prompt) = receipt.effective_prompt.as_ref() {
        if effective_prompt.backend.trim().is_empty() {
            return Err(PowerError::AttestationVerificationFailed(
                "receipt.effective_prompt.backend must not be empty".to_string(),
            ));
        }
        if effective_prompt.kind.trim().is_empty() {
            return Err(PowerError::AttestationVerificationFailed(
                "receipt.effective_prompt.kind must not be empty".to_string(),
            ));
        }
        require_sha256_hex("receipt.effective_prompt.sha256", &effective_prompt.sha256)?;
    }

    if let Some(runtime_policy) = receipt.runtime_policy.as_ref() {
        verify_runtime_policy_well_formed("receipt.runtime_policy", runtime_policy)?;
    }

    Ok(())
}

fn verify_runtime_policy_well_formed(
    field_prefix: &str,
    runtime_policy: &RuntimePolicyClaim,
) -> Result<()> {
    if let Some(prompt) = runtime_policy.prompt.as_ref() {
        require_optional_sha256_digest_bytes(
            &format!("{field_prefix}.prompt.chat_template_sha256"),
            prompt.chat_template_sha256.as_deref(),
        )?;
        require_optional_sha256_digest_bytes(
            &format!("{field_prefix}.prompt.system_prompt_sha256"),
            prompt.system_prompt_sha256.as_deref(),
        )?;
        require_optional_sha256_digest_bytes(
            &format!("{field_prefix}.prompt.messages_sha256"),
            prompt.messages_sha256.as_deref(),
        )?;
    }

    if let Some(decoding) = runtime_policy.decoding.as_ref() {
        require_sha256_digest_bytes(
            &format!("{field_prefix}.decoding.parameters_sha256"),
            &decoding.parameters_sha256,
        )?;
    }

    if let Some(execution) = runtime_policy.execution.as_ref() {
        require_sha256_digest_bytes(
            &format!("{field_prefix}.execution.gpu_sha256"),
            &execution.gpu_sha256,
        )?;
    }

    Ok(())
}

/// Verify receipt fields against verifier-pinned request policy.
pub fn verify_receipt_policy(
    receipt: &AttestationReceipt,
    expected: &ExpectedReceipt,
) -> Result<()> {
    verify_receipt_well_formed(receipt)?;

    if let Some(expected_model) = expected.model.as_deref() {
        if receipt.model != expected_model {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "receipt model mismatch: receipt.model = {}, expected {}",
                receipt.model, expected_model,
            )));
        }
    }

    if let Some(expected_request_type) = expected.request_type {
        if receipt.request_type != expected_request_type {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "receipt request type mismatch: receipt.request_type = {:?}, expected {:?}",
                receipt.request_type, expected_request_type,
            )));
        }
    }

    if let Some(expected_input_digest) = expected.input_digest.as_deref() {
        verify_receipt_field_digest_hex(
            "receipt.input.sha256",
            &receipt.input.sha256,
            expected_input_digest,
        )?;
    }

    if let Some(expected_parameters_digest) = expected.decoding_parameters_digest.as_deref() {
        let actual = receipt_decoding_parameters_digest(receipt).map_err(|e| {
            PowerError::AttestationVerificationFailed(format!(
                "failed to compute receipt decoding parameter digest: {e}"
            ))
        })?;
        verify_receipt_field_digest_hex(
            "receipt.decoding.parameters_sha256",
            &actual,
            expected_parameters_digest,
        )?;
    }

    verify_optional_receipt_field_digest_hex(
        "receipt.decoding.stream_options_sha256",
        receipt.decoding.stream_options_sha256.as_deref(),
        expected.stream_options_digest.as_deref(),
    )?;
    verify_optional_receipt_field_digest_hex(
        "receipt.decoding.stop_tokens_sha256",
        receipt.decoding.stop_tokens_sha256.as_deref(),
        expected.stop_tokens_digest.as_deref(),
    )?;
    verify_optional_receipt_field_digest_hex(
        "receipt.decoding.response_format_sha256",
        receipt.decoding.response_format_sha256.as_deref(),
        expected.response_format_digest.as_deref(),
    )?;
    verify_optional_receipt_field_digest_hex(
        "receipt.decoding.tools_sha256",
        receipt.decoding.tools_sha256.as_deref(),
        expected.tools_digest.as_deref(),
    )?;
    verify_optional_receipt_field_digest_hex(
        "receipt.decoding.tool_choice_sha256",
        receipt.decoding.tool_choice_sha256.as_deref(),
        expected.tool_choice_digest.as_deref(),
    )?;

    if expected.effective_prompt_absent
        && (expected.effective_prompt_digest.is_some()
            || expected.effective_prompt_backend.is_some()
            || expected.effective_prompt_kind.is_some())
    {
        return Err(PowerError::AttestationVerificationFailed(
            "receipt effective prompt policy cannot require absence while also pinning digest, backend, or kind"
                .to_string(),
        ));
    }

    if expected.effective_prompt_absent && receipt.effective_prompt.is_some() {
        return Err(PowerError::AttestationVerificationFailed(
            "receipt effective prompt digest is present but verifier policy requires it to be absent"
                .to_string(),
        ));
    }

    if let Some(expected_digest) = expected.effective_prompt_digest.as_deref() {
        verify_receipt_effective_prompt_digest(receipt, expected_digest)?;
    }

    if expected.effective_prompt_backend.is_some() || expected.effective_prompt_kind.is_some() {
        let effective_prompt = receipt.effective_prompt.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "receipt does not include an effective prompt digest".to_string(),
            )
        })?;
        if let Some(expected_backend) = expected.effective_prompt_backend.as_deref() {
            if effective_prompt.backend != expected_backend {
                return Err(PowerError::AttestationVerificationFailed(format!(
                    "receipt effective prompt backend mismatch: receipt.effective_prompt.backend = {}, expected {}",
                    effective_prompt.backend, expected_backend,
                )));
            }
        }
        if let Some(expected_kind) = expected.effective_prompt_kind.as_deref() {
            if effective_prompt.kind != expected_kind {
                return Err(PowerError::AttestationVerificationFailed(format!(
                    "receipt effective prompt kind mismatch: receipt.effective_prompt.kind = {}, expected {}",
                    effective_prompt.kind, expected_kind,
                )));
            }
        }
    }

    Ok(())
}

/// Verify that a chat-completion receipt matches the original request.
///
/// This checks every request-derived receipt field: request type, model,
/// prompt-bearing input digest, decoding parameters, streaming options, stop
/// tokens, response format, tools, and tool choice. Backend-specific
/// `runtime_policy` and `effective_prompt` claims are intentionally left to
/// [`verify_receipt_against_attestation()`] and the effective-prompt policy
/// helpers.
pub fn verify_receipt_matches_chat_request(
    receipt: &AttestationReceipt,
    request: &ChatCompletionRequest,
) -> Result<()> {
    let expected = chat_receipt(request)?;
    verify_receipt_matches_request_receipt(receipt, &expected)
}

/// Verify that a text-completion receipt matches the original request.
///
/// This checks every request-derived receipt field: request type, model,
/// prompt digest, decoding parameters, streaming options, and stop tokens.
/// Backend-specific `runtime_policy` claims are intentionally left to
/// [`verify_receipt_against_attestation()`].
pub fn verify_receipt_matches_completion_request(
    receipt: &AttestationReceipt,
    request: &CompletionRequest,
) -> Result<()> {
    let expected = completion_receipt(request)?;
    verify_receipt_matches_request_receipt(receipt, &expected)
}

fn verify_receipt_matches_request_receipt(
    receipt: &AttestationReceipt,
    expected: &AttestationReceipt,
) -> Result<()> {
    verify_receipt_well_formed(receipt)?;

    if receipt.request_type != expected.request_type {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "receipt request type mismatch: receipt.request_type = {:?}, expected {:?}",
            receipt.request_type, expected.request_type,
        )));
    }

    if receipt.model != expected.model {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "receipt model mismatch: receipt.model = {}, expected {}",
            receipt.model, expected.model,
        )));
    }

    let expected_input_digest =
        decode_receipt_sha256_hex("expected receipt.input.sha256", &expected.input.sha256)?;
    verify_receipt_field_digest_hex(
        "receipt.input.sha256",
        &receipt.input.sha256,
        &expected_input_digest,
    )?;

    let expected_parameters_digest = receipt_decoding_parameters_digest(expected).map_err(|e| {
        PowerError::AttestationVerificationFailed(format!(
            "failed to compute expected receipt decoding parameter digest: {e}"
        ))
    })?;
    let expected_parameters_digest = decode_receipt_sha256_hex(
        "expected receipt.decoding.parameters_sha256",
        &expected_parameters_digest,
    )?;
    verify_receipt_field_digest_hex(
        "receipt.decoding.parameters_sha256",
        &receipt_decoding_parameters_digest(receipt).map_err(|e| {
            PowerError::AttestationVerificationFailed(format!(
                "failed to compute receipt decoding parameter digest: {e}"
            ))
        })?,
        &expected_parameters_digest,
    )?;

    verify_optional_receipt_field_matches(
        "receipt.decoding.stream_options_sha256",
        receipt.decoding.stream_options_sha256.as_deref(),
        expected.decoding.stream_options_sha256.as_deref(),
    )?;
    verify_optional_receipt_field_matches(
        "receipt.decoding.stop_tokens_sha256",
        receipt.decoding.stop_tokens_sha256.as_deref(),
        expected.decoding.stop_tokens_sha256.as_deref(),
    )?;
    verify_optional_receipt_field_matches(
        "receipt.decoding.response_format_sha256",
        receipt.decoding.response_format_sha256.as_deref(),
        expected.decoding.response_format_sha256.as_deref(),
    )?;
    verify_optional_receipt_field_matches(
        "receipt.decoding.tools_sha256",
        receipt.decoding.tools_sha256.as_deref(),
        expected.decoding.tools_sha256.as_deref(),
    )?;
    verify_optional_receipt_field_matches(
        "receipt.decoding.tool_choice_sha256",
        receipt.decoding.tool_choice_sha256.as_deref(),
        expected.decoding.tool_choice_sha256.as_deref(),
    )?;

    Ok(())
}

fn verify_optional_receipt_field_matches(
    field: &str,
    actual_hex: Option<&str>,
    expected_hex: Option<&str>,
) -> Result<()> {
    match (actual_hex, expected_hex) {
        (None, None) => Ok(()),
        (Some(actual), Some(expected)) => {
            let expected = decode_receipt_sha256_hex(&format!("expected {field}"), expected)?;
            verify_receipt_field_digest_hex(field, actual, &expected)
        }
        (None, Some(expected)) => Err(PowerError::AttestationVerificationFailed(format!(
            "{field} is absent but the original request requires {expected}"
        ))),
        (Some(actual), None) => Err(PowerError::AttestationVerificationFailed(format!(
            "{field} is present ({actual}) but the original request does not set that policy"
        ))),
    }
}

fn decode_receipt_sha256_hex(field: &str, value: &str) -> Result<Vec<u8>> {
    require_sha256_hex(field, value)?;
    hex::decode(value).map_err(|e| {
        PowerError::AttestationVerificationFailed(format!("failed to decode {field}: {e}"))
    })
}

fn verify_optional_receipt_field_digest_hex(
    field: &str,
    actual_hex: Option<&str>,
    expected_digest: Option<&[u8]>,
) -> Result<()> {
    let Some(expected_digest) = expected_digest else {
        return Ok(());
    };
    let actual_hex = actual_hex.ok_or_else(|| {
        PowerError::AttestationVerificationFailed(format!(
            "{field} is absent but verifier policy pinned it"
        ))
    })?;
    verify_receipt_field_digest_hex(field, actual_hex, expected_digest)
}

fn verify_receipt_field_digest_hex(
    field: &str,
    actual_hex: &str,
    expected_digest: &[u8],
) -> Result<()> {
    if expected_digest.len() != 32 {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "{field} expected digest must be 32 bytes, got {}",
            expected_digest.len(),
        )));
    }
    let actual = hex::decode(actual_hex).map_err(|e| {
        PowerError::AttestationVerificationFailed(format!("failed to decode {field}: {e}"))
    })?;
    if !constant_time_eq(&actual, expected_digest) {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "{field} mismatch: {field} = {}, expected {}",
            actual_hex,
            hex::encode(expected_digest),
        )));
    }
    Ok(())
}

fn require_optional_sha256_hex(field: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        require_sha256_hex(field, value)?;
    }
    Ok(())
}

fn require_optional_sha256_digest_bytes(field: &str, digest: Option<&[u8]>) -> Result<()> {
    if let Some(digest) = digest {
        require_sha256_digest_bytes(field, digest)?;
    }
    Ok(())
}

fn require_sha256_digest_bytes(field: &str, digest: &[u8]) -> Result<()> {
    if digest.len() != 32 {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "{field} must be a 32-byte SHA-256 digest, got {} bytes",
            digest.len(),
        )));
    }
    Ok(())
}

fn require_sha256_hex(field: &str, value: &str) -> Result<()> {
    let hex = value.trim();
    if hex.len() != 64 {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "{field} must be a 64-character SHA-256 hex digest, got {} characters",
            hex.len(),
        )));
    }
    hex::decode(hex).map_err(|e| {
        PowerError::AttestationVerificationFailed(format!("{field} is not valid SHA-256 hex: {e}"))
    })?;
    Ok(())
}

/// Verify that a receipt's effective prompt digest matches a 32-byte SHA-256 digest.
pub fn verify_receipt_effective_prompt_digest(
    receipt: &AttestationReceipt,
    expected_digest: &[u8],
) -> Result<()> {
    if expected_digest.len() != 32 {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "effective prompt digest must be 32 bytes, got {}",
            expected_digest.len()
        )));
    }

    let actual_hex = receipt
        .effective_prompt
        .as_ref()
        .ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "receipt does not include an effective prompt digest".to_string(),
            )
        })?
        .sha256
        .as_str();

    if actual_hex.len() != 64 || !actual_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(PowerError::AttestationVerificationFailed(
            "receipt effective prompt digest is not a valid SHA-256 hex string".to_string(),
        ));
    }

    let actual = hex::decode(actual_hex).map_err(|e| {
        PowerError::AttestationVerificationFailed(format!(
            "failed to decode receipt effective prompt digest: {e}"
        ))
    })?;

    if !constant_time_eq(&actual, expected_digest) {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "Effective prompt digest mismatch: receipt digest = {}, expected {}",
            actual_hex,
            hex::encode(expected_digest),
        )));
    }

    Ok(())
}

/// Verify that a receipt's effective prompt digest matches a hex SHA-256 digest string.
pub fn verify_receipt_effective_prompt_digest_hex(
    receipt: &AttestationReceipt,
    expected_digest_hex: &str,
) -> Result<()> {
    let digest_hex = expected_digest_hex
        .strip_prefix("sha256:")
        .unwrap_or(expected_digest_hex);
    if digest_hex.len() != 64 {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "effective prompt digest must be 64 hex characters, got {}",
            digest_hex.len()
        )));
    }
    if !digest_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(PowerError::AttestationVerificationFailed(
            "effective prompt digest contains non-hex characters".to_string(),
        ));
    }

    let expected = hex::decode(digest_hex).map_err(|e| {
        PowerError::AttestationVerificationFailed(format!(
            "failed to decode expected effective prompt digest: {e}"
        ))
    })?;
    verify_receipt_effective_prompt_digest(receipt, &expected)
}

/// Verify that a response receipt is bound to the same attested runtime policy.
///
/// This helper combines the checks verifiers normally need after receiving an
/// inference response:
///
/// - the attestation report's v2 claims are bound into CPU TEE `report_data`;
/// - the receipt uses the expected receipt schema;
/// - when the attestation names a model, the receipt names the same model;
/// - the receipt's runtime policy exactly matches the attested runtime policy;
/// - optional receipt and effective-prompt digest pins match.
pub fn verify_receipt_against_attestation(
    report: &AttestationReport,
    receipt: &AttestationReceipt,
    expected_receipt_digest: Option<&[u8]>,
    expected_effective_prompt_digest: Option<&[u8]>,
) -> Result<()> {
    verify_claims_binding(report)?;
    verify_receipt_well_formed(receipt)?;

    let claims = report.claims.as_ref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed(
            "attestation report does not include v2 claims".to_string(),
        )
    })?;

    if let Some(model) = claims.model.as_ref() {
        if model.name != receipt.model {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "receipt model mismatch: receipt.model = {}, claims.model.name = {}",
                receipt.model, model.name,
            )));
        }
    }

    let attested_runtime = claims.runtime.as_ref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed(
            "attestation report does not include a runtime policy claim".to_string(),
        )
    })?;
    let receipt_runtime = receipt.runtime_policy.as_ref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed(
            "receipt does not include a runtime policy claim".to_string(),
        )
    })?;

    if receipt_runtime != attested_runtime {
        return Err(PowerError::AttestationVerificationFailed(
            "Receipt runtime policy mismatch".to_string(),
        ));
    }

    if let Some(expected) = expected_receipt_digest {
        verify_receipt_digest(receipt, expected)?;
    }

    if let Some(expected) = expected_effective_prompt_digest {
        verify_receipt_effective_prompt_digest(receipt, expected)?;
    }

    Ok(())
}

/// Constant-time byte slice comparison to prevent timing side-channels.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::receipt::{
        chat_receipt, chat_receipt_with_runtime_policy_and_effective_prompt, completion_receipt,
        receipt_digest, AttestationReceipt, ReceiptDecodingPolicy, ReceiptInputDigest,
        ReceiptRequestType,
    };
    use crate::api::types::{
        ChatCompletionMessage, ChatCompletionRequest, CompletionRequest, StreamOptions,
    };
    use crate::backend::types::{ContentPart, EffectivePromptDigest, ImageUrl, MessageContent};
    use crate::tee::attestation::{
        build_claims_report_data, AttestationClaimsV2, AttestationReport, DecodingPolicyClaim,
        ExecutionPolicyClaim, GpuDeviceClaim, GpuDeviceValidationClaim, GpuEvidenceClaim,
        ModelDigestClaim, ModelDigestKind, PromptPolicyClaim, RuntimePolicyClaim, TeeType,
    };
    use std::collections::BTreeMap;

    fn make_report(report_data: Vec<u8>, measurement: Vec<u8>) -> AttestationReport {
        AttestationReport {
            version: "1.0".to_string(),
            tee_type: TeeType::Simulated,
            report_data,
            measurement,
            raw_report: None,
            timestamp: chrono::Utc::now(),
            nonce: None,
            claims: None,
        }
    }

    fn build_report_data(nonce: Option<&[u8]>, model_hash: Option<&[u8]>) -> Vec<u8> {
        let mut data = vec![0u8; 64];
        if let Some(n) = nonce {
            let len = n.len().min(32);
            data[..len].copy_from_slice(&n[..len]);
        }
        if let Some(h) = model_hash {
            let len = h.len().min(32);
            data[32..32 + len].copy_from_slice(&h[..len]);
        }
        data
    }

    fn gpu_nonce() -> [u8; 32] {
        [0x11; 32]
    }

    fn make_claims_report(
        nonce: Option<&[u8]>,
        model_hash: Vec<u8>,
    ) -> (AttestationReport, AttestationClaimsV2) {
        let claims = AttestationClaimsV2::new(TeeType::SevSnp)
            .with_nonce(nonce)
            .with_model(ModelDigestClaim {
                name: "test-model".to_string(),
                kind: ModelDigestKind::PlaintextWeightsSha256,
                digest: model_hash,
                plaintext_digest: None,
                ciphertext_digest: None,
            });
        let report_data = build_claims_report_data(&claims).unwrap();
        let mut report = make_report(report_data, vec![0x03; 48]);
        report.tee_type = TeeType::SevSnp;
        report.nonce = claims.nonce.clone();
        report.claims = Some(claims.clone());
        (report, claims)
    }

    fn make_gpu_claims_report(
        evidence_digest: Vec<u8>,
        verdict_digest: Option<Vec<u8>>,
    ) -> (AttestationReport, AttestationClaimsV2) {
        let nonce = gpu_nonce();
        let mut gpu = GpuEvidenceClaim::new("nvidia-nras", evidence_digest)
            .with_evidence_format("nvidia-nvattest-evidence-json")
            .with_evidence_count(1)
            .with_nonce(&nonce);
        if let Some(verdict_digest) = verdict_digest {
            gpu = gpu
                .with_verdict_format("nvidia-nvattest-attestation-json")
                .with_verdict_digest(verdict_digest);
        }
        let claims = AttestationClaimsV2::new(TeeType::SevSnp)
            .with_nonce(Some(&nonce))
            .with_gpu(gpu);
        let report_data = build_claims_report_data(&claims).unwrap();
        let mut report = make_report(report_data, vec![0x03; 48]);
        report.tee_type = TeeType::SevSnp;
        report.nonce = claims.nonce.clone();
        report.claims = Some(claims.clone());
        (report, claims)
    }

    fn make_gpu_claims_report_with_device_claims(
        evidence_digest: Vec<u8>,
        verdict_digest: Option<Vec<u8>>,
    ) -> (AttestationReport, AttestationClaimsV2) {
        let nonce = gpu_nonce();
        let mut gpu = GpuEvidenceClaim::new("nvidia-nras", evidence_digest)
            .with_evidence_format("nvidia-nvattest-evidence-json")
            .with_evidence_count(1)
            .with_nonce(&nonce)
            .with_devices(vec![gpu_device_claim(&nonce)]);
        if let Some(verdict_digest) = verdict_digest {
            gpu = gpu
                .with_verdict_format("nvidia-nvattest-attestation-json")
                .with_verdict_digest(verdict_digest);
        }
        let claims = AttestationClaimsV2::new(TeeType::SevSnp)
            .with_nonce(Some(&nonce))
            .with_gpu(gpu);
        let report_data = build_claims_report_data(&claims).unwrap();
        let mut report = make_report(report_data, vec![0x03; 48]);
        report.tee_type = TeeType::SevSnp;
        report.nonce = claims.nonce.clone();
        report.claims = Some(claims.clone());
        (report, claims)
    }

    fn expected_gpu_evidence_pins() -> ExpectedGpuEvidence {
        ExpectedGpuEvidence {
            provider: Some("nvidia-nras".to_string()),
            evidence_format: Some("nvidia-nvattest-evidence-json".to_string()),
            verdict_format: Some("nvidia-nvattest-attestation-json".to_string()),
            evidence_count: Some(1),
        }
    }

    fn expected_gpu_device_pins() -> ExpectedGpuDevices {
        ExpectedGpuDevices {
            gpu_count: Some(1),
            nvswitch_count: Some(0),
            gpu_ueids: vec!["655333107904478077882826344426270545524203067314".to_string()],
            oemids: vec!["5703".to_string()],
            claims_versions: vec!["3.0".to_string()],
            hwmodels: vec!["GH100 A01 GSP BROM".to_string()],
            driver_versions: vec!["590.12".to_string()],
            firmware_versions: vec!["96.00.A5.00.01".to_string()],
            nvswitch_ueids: Vec::new(),
            nvswitch_oemids: Vec::new(),
            nvswitch_claims_versions: Vec::new(),
            nvswitch_hwmodels: Vec::new(),
            nvswitch_firmware_versions: Vec::new(),
        }
    }

    fn gpu_device_claim(nonce: &[u8]) -> GpuDeviceClaim {
        GpuDeviceClaim {
            index: 0,
            device_type: "gpu".to_string(),
            attestation_nonce: Some(nonce.to_vec()),
            hwmodel: Some("GH100 A01 GSP BROM".to_string()),
            ueid: Some("655333107904478077882826344426270545524203067314".to_string()),
            oemid: Some("5703".to_string()),
            claims_version: Some("3.0".to_string()),
            driver_version: Some("590.12".to_string()),
            firmware_version: Some("96.00.A5.00.01".to_string()),
            measurements_result: Some("success".to_string()),
            secure_boot: Some(true),
            debug_status: Some("disabled".to_string()),
            validation: GpuDeviceValidationClaim {
                arch_check: Some(true),
                attestation_report_cert_chain_fwid_match: Some(true),
                attestation_report_parsed: Some(true),
                attestation_report_nonce_match: Some(true),
                attestation_report_signature_verified: Some(true),
                driver_rim_fetched: Some(true),
                driver_rim_schema_validated: Some(true),
                driver_rim_signature_verified: Some(true),
                driver_rim_version_match: Some(true),
                driver_rim_measurements_available: Some(true),
                firmware_rim_fetched: Some(true),
                firmware_rim_schema_validated: Some(true),
                firmware_rim_signature_verified: Some(true),
                firmware_rim_version_match: Some(true),
                firmware_rim_measurements_available: Some(true),
                firmware_index_no_conflict: Some(true),
            },
        }
    }

    fn nvswitch_device_claim(index: u32, nonce: &[u8]) -> GpuDeviceClaim {
        GpuDeviceClaim {
            index,
            device_type: "nvswitch".to_string(),
            attestation_nonce: Some(nonce.to_vec()),
            hwmodel: Some("NVSwitch B01".to_string()),
            ueid: Some("nvswitch-ueid-0".to_string()),
            oemid: Some("5703".to_string()),
            claims_version: Some("3.0".to_string()),
            driver_version: None,
            firmware_version: Some("1.2.3".to_string()),
            measurements_result: Some("success".to_string()),
            secure_boot: Some(true),
            debug_status: Some("disabled".to_string()),
            validation: GpuDeviceValidationClaim {
                arch_check: Some(true),
                attestation_report_cert_chain_fwid_match: Some(true),
                attestation_report_parsed: Some(true),
                attestation_report_nonce_match: Some(true),
                attestation_report_signature_verified: Some(true),
                driver_rim_fetched: None,
                driver_rim_schema_validated: None,
                driver_rim_signature_verified: None,
                driver_rim_version_match: None,
                driver_rim_measurements_available: None,
                firmware_rim_fetched: Some(true),
                firmware_rim_schema_validated: Some(true),
                firmware_rim_signature_verified: Some(true),
                firmware_rim_version_match: Some(true),
                firmware_rim_measurements_available: Some(true),
                firmware_index_no_conflict: None,
            },
        }
    }

    fn make_runtime_claims_report(
        chat_template_digest: Vec<u8>,
        decoding_parameters_digest: Vec<u8>,
    ) -> (AttestationReport, AttestationClaimsV2) {
        let runtime = RuntimePolicyClaim::new()
            .with_prompt(PromptPolicyClaim {
                chat_template_source: Some("manifest.template_override".to_string()),
                chat_template_sha256: Some(chat_template_digest),
                system_prompt_sha256: None,
                messages_sha256: None,
            })
            .with_decoding(DecodingPolicyClaim {
                parameters_sha256: decoding_parameters_digest,
            });
        let claims = AttestationClaimsV2::new(TeeType::SevSnp).with_runtime(runtime);
        let report_data = build_claims_report_data(&claims).unwrap();
        let mut report = make_report(report_data, vec![0x03; 48]);
        report.tee_type = TeeType::SevSnp;
        report.claims = Some(claims.clone());
        (report, claims)
    }

    fn make_runtime_execution_claims_report(
        gpu_execution_digest: Vec<u8>,
    ) -> (AttestationReport, AttestationClaimsV2) {
        let runtime = RuntimePolicyClaim::new().with_execution(ExecutionPolicyClaim {
            gpu_sha256: gpu_execution_digest,
        });
        let claims = AttestationClaimsV2::new(TeeType::SevSnp).with_runtime(runtime);
        let report_data = build_claims_report_data(&claims).unwrap();
        let mut report = make_report(report_data, vec![0x03; 48]);
        report.tee_type = TeeType::SevSnp;
        report.claims = Some(claims.clone());
        (report, claims)
    }

    fn make_model_runtime_claims_report() -> (AttestationReport, AttestationClaimsV2) {
        let runtime = RuntimePolicyClaim::new()
            .with_prompt(PromptPolicyClaim {
                chat_template_source: Some("manifest.template_override".to_string()),
                chat_template_sha256: Some(vec![0x33; 32]),
                system_prompt_sha256: None,
                messages_sha256: None,
            })
            .with_decoding(DecodingPolicyClaim {
                parameters_sha256: vec![0x44; 32],
            });
        let claims = AttestationClaimsV2::new(TeeType::SevSnp)
            .with_model(ModelDigestClaim {
                name: "test-model".to_string(),
                kind: ModelDigestKind::PlaintextWeightsSha256,
                digest: vec![0x02; 32],
                plaintext_digest: None,
                ciphertext_digest: None,
            })
            .with_runtime(runtime);
        let report_data = build_claims_report_data(&claims).unwrap();
        let mut report = make_report(report_data, vec![0x03; 48]);
        report.tee_type = TeeType::SevSnp;
        report.claims = Some(claims.clone());
        (report, claims)
    }

    fn make_receipt() -> AttestationReceipt {
        AttestationReceipt {
            schema: AttestationReceipt::SCHEMA.to_string(),
            request_type: ReceiptRequestType::ChatCompletion,
            model: "test-model".to_string(),
            input: ReceiptInputDigest {
                kind: "chat.messages".to_string(),
                sha256: "00".repeat(32),
            },
            runtime_policy: None,
            effective_prompt: None,
            decoding: ReceiptDecodingPolicy {
                parameters: BTreeMap::from([("temperature".to_string(), serde_json::json!(0.2))]),
                stream_options_sha256: None,
                stop_tokens_sha256: None,
                response_format_sha256: None,
                tools_sha256: None,
                tool_choice_sha256: None,
            },
        }
    }

    fn make_receipt_with_effective_prompt() -> AttestationReceipt {
        let mut receipt = make_receipt();
        receipt.effective_prompt = Some(EffectivePromptDigest::chat_rendered_prompt(
            "test-backend",
            "rendered prompt",
        ));
        receipt
    }

    fn make_receipt_with_runtime_policy(runtime: RuntimePolicyClaim) -> AttestationReceipt {
        let mut receipt = make_receipt();
        receipt.runtime_policy = Some(runtime);
        receipt
    }

    fn chat_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hello".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
                thinking: None,
            }],
            temperature: Some(0.2),
            top_p: Some(0.9),
            max_tokens: Some(128),
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            penalize_newline: None,
            num_ctx: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            stop: Some(vec!["</s>".to_string()]),
            stream: Some(false),
            stream_options: None,
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.0),
            seed: Some(7),
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            keep_alive: None,
        }
    }

    fn completion_request() -> CompletionRequest {
        CompletionRequest {
            model: "test-model".to_string(),
            prompt: "hello".to_string(),
            temperature: Some(0.2),
            top_p: Some(0.9),
            max_tokens: Some(128),
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            penalize_newline: None,
            num_ctx: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            stop: Some(vec!["</s>".to_string()]),
            stream: Some(false),
            stream_options: None,
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.0),
            seed: Some(7),
            keep_alive: None,
        }
    }

    // --- verify_nonce_binding ---

    #[test]
    fn test_nonce_binding_passes_when_matching() {
        let nonce = vec![0x01u8; 16];
        let report_data = build_report_data(Some(&nonce), None);
        assert!(verify_nonce_binding(&report_data, &nonce).is_ok());
    }

    #[test]
    fn test_nonce_binding_fails_when_mismatch() {
        let nonce = vec![0x01u8; 16];
        let wrong_nonce = vec![0x02u8; 16];
        let report_data = build_report_data(Some(&nonce), None);
        let err = verify_nonce_binding(&report_data, &wrong_nonce).unwrap_err();
        assert!(err.to_string().contains("Nonce mismatch"));
    }

    #[test]
    fn test_nonce_binding_fails_when_report_data_too_short() {
        let nonce = vec![0x01u8; 16];
        let short_data = vec![0u8; 10];
        let err = verify_nonce_binding(&short_data, &nonce).unwrap_err();
        assert!(err.to_string().contains("too short"));
    }

    #[test]
    fn test_nonce_binding_full_32_bytes() {
        let nonce = vec![0xABu8; 32];
        let report_data = build_report_data(Some(&nonce), None);
        assert!(verify_nonce_binding(&report_data, &nonce).is_ok());
    }

    #[test]
    fn test_nonce_binding_fails_nonzero_padding() {
        let nonce = vec![0x01u8; 8];
        let mut report_data = build_report_data(Some(&nonce), None);
        // Corrupt the padding bytes
        report_data[8] = 0xFF;
        let err = verify_nonce_binding(&report_data, &nonce).unwrap_err();
        assert!(err.to_string().contains("padding is not zero"));
    }

    // --- verify_model_hash_binding ---

    #[test]
    fn test_model_hash_binding_passes_when_matching() {
        let model_hash = vec![0xBBu8; 32];
        let report_data = build_report_data(None, Some(&model_hash));
        assert!(verify_model_hash_binding(&report_data, &model_hash).is_ok());
    }

    #[test]
    fn test_model_hash_binding_rejects_short_expected_hash() {
        let model_hash = vec![0xBBu8; 32];
        let short_hash = vec![0xBBu8; 31];
        let report_data = build_report_data(None, Some(&model_hash));

        let err = verify_model_hash_binding(&report_data, &short_hash).unwrap_err();

        assert!(err
            .to_string()
            .contains("expected model hash must be a 32-byte SHA-256 digest"));
    }

    #[test]
    fn test_model_hash_binding_fails_when_mismatch() {
        let model_hash = vec![0xBBu8; 32];
        let wrong_hash = vec![0xCCu8; 32];
        let report_data = build_report_data(None, Some(&model_hash));
        let err = verify_model_hash_binding(&report_data, &wrong_hash).unwrap_err();
        assert!(err.to_string().contains("Model hash mismatch"));
    }

    #[test]
    fn test_model_hash_binding_fails_when_report_data_too_short() {
        let model_hash = vec![0xBBu8; 32];
        let short_data = vec![0u8; 40]; // < 64
        let err = verify_model_hash_binding(&short_data, &model_hash).unwrap_err();
        assert!(err.to_string().contains("too short for model hash"));
    }

    // --- verify_measurement ---

    #[test]
    fn test_measurement_passes_when_matching() {
        let m = vec![0xCCu8; 48];
        assert!(verify_measurement(&m, &m).is_ok());
    }

    #[test]
    fn test_measurement_rejects_short_expected_measurement() {
        let measurement = vec![0xCCu8; 48];
        let expected = vec![0xCCu8; 47];

        let err = verify_measurement(&measurement, &expected).unwrap_err();

        assert!(err
            .to_string()
            .contains("expected measurement must be a 48-byte launch measurement"));
    }

    #[test]
    fn test_measurement_rejects_short_report_measurement() {
        let measurement = vec![0xCCu8; 47];
        let expected = vec![0xCCu8; 48];

        let err = verify_measurement(&measurement, &expected).unwrap_err();

        assert!(err
            .to_string()
            .contains("report measurement must be a 48-byte launch measurement"));
    }

    #[test]
    fn test_measurement_fails_when_mismatch() {
        let m = vec![0xCCu8; 48];
        let wrong = vec![0xDDu8; 48];
        let err = verify_measurement(&m, &wrong).unwrap_err();
        assert!(err.to_string().contains("Measurement mismatch"));
    }

    // --- verify_receipt_digest ---

    #[test]
    fn test_verify_receipt_digest_hex_passes_when_matching() {
        let receipt = make_receipt();
        let digest = receipt_digest(&receipt).unwrap();

        verify_receipt_digest_hex(&receipt, &digest).unwrap();
        verify_receipt_digest_hex(&receipt, &format!("sha256:{digest}")).unwrap();
    }

    #[test]
    fn test_verify_receipt_digest_fails_when_mismatch() {
        let receipt = make_receipt();
        let wrong = vec![0x99; 32];

        let err = verify_receipt_digest(&receipt, &wrong).unwrap_err();

        assert!(err.to_string().contains("Receipt digest mismatch"));
    }

    #[test]
    fn test_verify_receipt_digest_hex_rejects_invalid_hex() {
        let receipt = make_receipt();

        let err = verify_receipt_digest_hex(&receipt, "not-hex").unwrap_err();

        assert!(err.to_string().contains("64 hex characters"));
    }

    #[test]
    fn test_verify_receipt_well_formed_accepts_generated_receipt() {
        let receipt = make_receipt_with_effective_prompt();

        verify_receipt_well_formed(&receipt).unwrap();
    }

    #[test]
    fn test_verify_receipt_well_formed_rejects_bad_input_digest() {
        let mut receipt = make_receipt();
        receipt.input.sha256 = "not-hex".to_string();

        let err = verify_receipt_well_formed(&receipt).unwrap_err();

        assert!(err.to_string().contains("receipt.input.sha256"));
    }

    #[test]
    fn test_verify_receipt_well_formed_rejects_request_type_kind_mismatch() {
        let mut receipt = make_receipt();
        receipt.request_type = ReceiptRequestType::TextCompletion;

        let err = verify_receipt_well_formed(&receipt).unwrap_err();

        assert!(err.to_string().contains("receipt input kind mismatch"));
    }

    #[test]
    fn test_verify_receipt_well_formed_rejects_bad_optional_digest() {
        let mut receipt = make_receipt();
        receipt.decoding.stop_tokens_sha256 = Some("not-hex".to_string());

        let err = verify_receipt_well_formed(&receipt).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt.decoding.stop_tokens_sha256"));
    }

    #[test]
    fn test_verify_receipt_well_formed_rejects_bad_stream_options_digest() {
        let mut receipt = make_receipt();
        receipt.decoding.stream_options_sha256 = Some("not-hex".to_string());

        let err = verify_receipt_well_formed(&receipt).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt.decoding.stream_options_sha256"));
    }

    #[test]
    fn test_verify_receipt_well_formed_rejects_bad_effective_prompt_digest() {
        let mut receipt = make_receipt_with_effective_prompt();
        receipt.effective_prompt.as_mut().unwrap().sha256 = "not-hex".to_string();

        let err = verify_receipt_well_formed(&receipt).unwrap_err();

        assert!(err.to_string().contains("receipt.effective_prompt.sha256"));
    }

    #[test]
    fn test_verify_receipt_well_formed_rejects_short_runtime_decoding_digest() {
        let receipt = make_receipt_with_runtime_policy(RuntimePolicyClaim::new().with_decoding(
            DecodingPolicyClaim {
                parameters_sha256: vec![0x44; 31],
            },
        ));

        let err = verify_receipt_well_formed(&receipt).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt.runtime_policy.decoding.parameters_sha256"));
    }

    #[test]
    fn test_verify_receipt_well_formed_rejects_short_runtime_prompt_digest() {
        let receipt = make_receipt_with_runtime_policy(RuntimePolicyClaim::new().with_prompt(
            PromptPolicyClaim {
                chat_template_source: Some("manifest.template_override".to_string()),
                chat_template_sha256: Some(vec![0x33; 31]),
                system_prompt_sha256: None,
                messages_sha256: None,
            },
        ));

        let err = verify_receipt_well_formed(&receipt).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt.runtime_policy.prompt.chat_template_sha256"));
    }

    #[test]
    fn test_verify_receipt_well_formed_rejects_short_runtime_execution_digest() {
        let receipt = make_receipt_with_runtime_policy(RuntimePolicyClaim::new().with_execution(
            ExecutionPolicyClaim {
                gpu_sha256: vec![0x55; 31],
            },
        ));

        let err = verify_receipt_well_formed(&receipt).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt.runtime_policy.execution.gpu_sha256"));
    }

    #[test]
    fn test_verify_receipt_policy_passes_for_pinned_fields() {
        let mut receipt = make_receipt_with_effective_prompt();
        receipt.decoding.stream_options_sha256 = Some("55".repeat(32));
        receipt.decoding.stop_tokens_sha256 = Some("11".repeat(32));
        let decoding_parameters_digest =
            hex::decode(receipt_decoding_parameters_digest(&receipt).unwrap()).unwrap();
        let expected = ExpectedReceipt {
            model: Some("test-model".to_string()),
            request_type: Some(ReceiptRequestType::ChatCompletion),
            input_digest: Some(vec![0x00; 32]),
            decoding_parameters_digest: Some(decoding_parameters_digest),
            stream_options_digest: Some(vec![0x55; 32]),
            stop_tokens_digest: Some(vec![0x11; 32]),
            response_format_digest: None,
            tools_digest: None,
            tool_choice_digest: None,
            effective_prompt_digest: Some(
                hex::decode(receipt.effective_prompt.as_ref().unwrap().sha256.as_str()).unwrap(),
            ),
            effective_prompt_absent: false,
            effective_prompt_backend: Some("test-backend".to_string()),
            effective_prompt_kind: Some("chat.rendered-prompt".to_string()),
        };

        verify_receipt_policy(&receipt, &expected).unwrap();
    }

    #[test]
    fn test_verify_receipt_policy_fails_on_input_digest_mismatch() {
        let receipt = make_receipt();
        let expected = ExpectedReceipt {
            input_digest: Some(vec![0x99; 32]),
            ..Default::default()
        };

        let err = verify_receipt_policy(&receipt, &expected).unwrap_err();

        assert!(err.to_string().contains("receipt.input.sha256 mismatch"));
    }

    #[test]
    fn test_verify_receipt_policy_fails_on_decoding_parameters_mismatch() {
        let receipt = make_receipt();
        let expected = ExpectedReceipt {
            decoding_parameters_digest: Some(vec![0x99; 32]),
            ..Default::default()
        };

        let err = verify_receipt_policy(&receipt, &expected).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt.decoding.parameters_sha256 mismatch"));
    }

    #[test]
    fn test_verify_receipt_policy_fails_on_missing_stop_tokens_digest() {
        let receipt = make_receipt();
        let expected = ExpectedReceipt {
            stop_tokens_digest: Some(vec![0x11; 32]),
            ..Default::default()
        };

        let err = verify_receipt_policy(&receipt, &expected).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt.decoding.stop_tokens_sha256 is absent"));
    }

    #[test]
    fn test_verify_receipt_policy_fails_on_stream_options_digest_mismatch() {
        let mut receipt = make_receipt();
        receipt.decoding.stream_options_sha256 = Some("55".repeat(32));
        let expected = ExpectedReceipt {
            stream_options_digest: Some(vec![0x44; 32]),
            ..Default::default()
        };

        let err = verify_receipt_policy(&receipt, &expected).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt.decoding.stream_options_sha256 mismatch"));
    }

    #[test]
    fn test_verify_receipt_policy_fails_on_missing_stream_options_digest() {
        let receipt = make_receipt();
        let expected = ExpectedReceipt {
            stream_options_digest: Some(vec![0x55; 32]),
            ..Default::default()
        };

        let err = verify_receipt_policy(&receipt, &expected).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt.decoding.stream_options_sha256 is absent"));
    }

    #[test]
    fn test_verify_receipt_policy_fails_on_tools_digest_mismatch() {
        let mut receipt = make_receipt();
        receipt.decoding.tools_sha256 = Some("33".repeat(32));
        let expected = ExpectedReceipt {
            tools_digest: Some(vec![0x44; 32]),
            ..Default::default()
        };

        let err = verify_receipt_policy(&receipt, &expected).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt.decoding.tools_sha256 mismatch"));
    }

    #[test]
    fn test_verify_receipt_policy_fails_on_request_type_mismatch() {
        let receipt = make_receipt();
        let expected = ExpectedReceipt {
            request_type: Some(ReceiptRequestType::TextCompletion),
            ..Default::default()
        };

        let err = verify_receipt_policy(&receipt, &expected).unwrap_err();

        assert!(err.to_string().contains("receipt request type mismatch"));
    }

    #[test]
    fn test_verify_receipt_policy_fails_on_effective_prompt_kind_mismatch() {
        let receipt = make_receipt_with_effective_prompt();
        let expected = ExpectedReceipt {
            effective_prompt_kind: Some("chat.prompt-token-ids".to_string()),
            ..Default::default()
        };

        let err = verify_receipt_policy(&receipt, &expected).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt effective prompt kind mismatch"));
    }

    #[test]
    fn test_verify_receipt_policy_fails_on_effective_prompt_digest_mismatch() {
        let receipt = make_receipt_with_effective_prompt();
        let expected = ExpectedReceipt {
            effective_prompt_digest: Some(vec![0x99; 32]),
            ..Default::default()
        };

        let err = verify_receipt_policy(&receipt, &expected).unwrap_err();

        assert!(err.to_string().contains("Effective prompt digest mismatch"));
    }

    #[test]
    fn test_verify_receipt_policy_fails_on_missing_effective_prompt_digest() {
        let receipt = make_receipt();
        let expected = ExpectedReceipt {
            effective_prompt_digest: Some(vec![0x11; 32]),
            ..Default::default()
        };

        let err = verify_receipt_policy(&receipt, &expected).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt does not include an effective prompt digest"));
    }

    #[test]
    fn test_verify_receipt_policy_passes_when_effective_prompt_absent_required() {
        let receipt = make_receipt();
        let expected = ExpectedReceipt {
            effective_prompt_absent: true,
            ..Default::default()
        };

        verify_receipt_policy(&receipt, &expected).unwrap();
    }

    #[test]
    fn test_verify_receipt_policy_fails_when_effective_prompt_present_but_absent_required() {
        let receipt = make_receipt_with_effective_prompt();
        let expected = ExpectedReceipt {
            effective_prompt_absent: true,
            ..Default::default()
        };

        let err = verify_receipt_policy(&receipt, &expected).unwrap_err();

        assert!(err.to_string().contains("requires it to be absent"));
    }

    #[test]
    fn test_verify_receipt_policy_rejects_conflicting_effective_prompt_policy() {
        let receipt = make_receipt();
        let expected = ExpectedReceipt {
            effective_prompt_absent: true,
            effective_prompt_digest: Some(vec![0x11; 32]),
            effective_prompt_kind: Some("chat.rendered-prompt".to_string()),
            ..Default::default()
        };

        let err = verify_receipt_policy(&receipt, &expected).unwrap_err();

        assert!(err.to_string().contains("cannot require absence"));
    }

    #[test]
    fn test_verify_receipt_matches_chat_request_allows_runtime_and_effective_prompt_claims() {
        let request = chat_request();
        let runtime = RuntimePolicyClaim::new().with_decoding(DecodingPolicyClaim {
            parameters_sha256: vec![0x44; 32],
        });
        let receipt = chat_receipt_with_runtime_policy_and_effective_prompt(
            &request,
            Some(runtime),
            Some(EffectivePromptDigest::chat_rendered_prompt(
                "test-backend",
                "rendered prompt",
            )),
        )
        .unwrap();

        verify_receipt_matches_chat_request(&receipt, &request).unwrap();
    }

    #[test]
    fn test_verify_receipt_matches_chat_request_catches_multimodal_input_mismatch() {
        let mut request = chat_request();
        request.messages[0].content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "describe".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,aW1hZ2UtYQ==".to_string(),
                    detail: Some("low".to_string()),
                },
            },
        ]);
        let receipt = chat_receipt(&request).unwrap();

        let mut changed = request.clone();
        changed.messages[0].content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "describe".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,aW1hZ2UtYg==".to_string(),
                    detail: Some("low".to_string()),
                },
            },
        ]);

        let err = verify_receipt_matches_chat_request(&receipt, &changed).unwrap_err();

        assert!(err.to_string().contains("receipt.input.sha256 mismatch"));
    }

    #[test]
    fn test_verify_receipt_matches_chat_request_rejects_unrequested_output_policy_digest() {
        let request = chat_request();
        let mut receipt = chat_receipt(&request).unwrap();
        receipt.decoding.response_format_sha256 = Some("22".repeat(32));

        let err = verify_receipt_matches_chat_request(&receipt, &request).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt.decoding.response_format_sha256 is present"));
    }

    #[test]
    fn test_verify_receipt_matches_completion_request_catches_missing_stream_options() {
        let mut request = completion_request();
        request.stream = Some(true);
        request.stream_options = Some(StreamOptions {
            include_usage: true,
        });
        let mut receipt = completion_receipt(&request).unwrap();
        receipt.decoding.stream_options_sha256 = None;

        let err = verify_receipt_matches_completion_request(&receipt, &request).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt.decoding.stream_options_sha256 is absent"));
    }

    #[test]
    fn test_verify_receipt_effective_prompt_digest_hex_passes_when_matching() {
        let receipt = make_receipt_with_effective_prompt();
        let digest = receipt.effective_prompt.as_ref().unwrap().sha256.clone();

        verify_receipt_effective_prompt_digest_hex(&receipt, &digest).unwrap();
        verify_receipt_effective_prompt_digest_hex(&receipt, &format!("sha256:{digest}")).unwrap();
    }

    #[test]
    fn test_verify_receipt_effective_prompt_digest_fails_when_missing() {
        let receipt = make_receipt();

        let err = verify_receipt_effective_prompt_digest(&receipt, &[0u8; 32]).unwrap_err();

        assert!(err
            .to_string()
            .contains("does not include an effective prompt digest"));
    }

    #[test]
    fn test_verify_receipt_effective_prompt_digest_fails_when_mismatch() {
        let receipt = make_receipt_with_effective_prompt();
        let wrong = [0xAA; 32];

        let err = verify_receipt_effective_prompt_digest(&receipt, &wrong).unwrap_err();

        assert!(err.to_string().contains("Effective prompt digest mismatch"));
    }

    #[test]
    fn test_verify_receipt_against_attestation_passes_when_runtime_matches() {
        let (report, claims) = make_runtime_claims_report(vec![0x33; 32], vec![0x44; 32]);
        let receipt = make_receipt_with_runtime_policy(claims.runtime.as_ref().unwrap().clone());

        verify_receipt_against_attestation(&report, &receipt, None, None).unwrap();
    }

    #[test]
    fn test_verify_receipt_against_attestation_checks_receipt_and_prompt_digest() {
        let (report, claims) = make_runtime_claims_report(vec![0x33; 32], vec![0x44; 32]);
        let mut receipt = make_receipt_with_effective_prompt();
        receipt.runtime_policy = Some(claims.runtime.as_ref().unwrap().clone());
        let receipt_digest = hex::decode(receipt_digest(&receipt).unwrap()).unwrap();
        let prompt_digest =
            hex::decode(&receipt.effective_prompt.as_ref().unwrap().sha256).unwrap();

        verify_receipt_against_attestation(
            &report,
            &receipt,
            Some(&receipt_digest),
            Some(&prompt_digest),
        )
        .unwrap();
    }

    #[test]
    fn test_verify_receipt_against_attestation_fails_on_runtime_mismatch() {
        let (report, _) = make_runtime_claims_report(vec![0x33; 32], vec![0x44; 32]);
        let mismatched_runtime = RuntimePolicyClaim::new().with_decoding(DecodingPolicyClaim {
            parameters_sha256: vec![0x99; 32],
        });
        let receipt = make_receipt_with_runtime_policy(mismatched_runtime);

        let err = verify_receipt_against_attestation(&report, &receipt, None, None).unwrap_err();

        assert!(err.to_string().contains("Receipt runtime policy mismatch"));
    }

    #[test]
    fn test_verify_receipt_against_attestation_fails_when_report_runtime_missing() {
        let (report, _) = make_claims_report(Some(b"nonce"), vec![0x02; 32]);
        let receipt = make_receipt_with_runtime_policy(RuntimePolicyClaim::new());

        let err = verify_receipt_against_attestation(&report, &receipt, None, None).unwrap_err();

        assert!(err
            .to_string()
            .contains("does not include a runtime policy claim"));
    }

    #[test]
    fn test_verify_receipt_against_attestation_fails_when_receipt_runtime_missing() {
        let (report, _) = make_runtime_claims_report(vec![0x33; 32], vec![0x44; 32]);
        let receipt = make_receipt();

        let err = verify_receipt_against_attestation(&report, &receipt, None, None).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt does not include a runtime policy claim"));
    }

    #[test]
    fn test_verify_receipt_against_attestation_fails_on_model_mismatch() {
        let (report, claims) = make_model_runtime_claims_report();
        let mut receipt =
            make_receipt_with_runtime_policy(claims.runtime.as_ref().unwrap().clone());
        receipt.model = "wrong-model".to_string();

        let err = verify_receipt_against_attestation(&report, &receipt, None, None).unwrap_err();

        assert!(err.to_string().contains("receipt model mismatch"));
    }

    #[test]
    fn test_verify_receipt_against_attestation_rejects_malformed_receipt() {
        let (report, claims) = make_runtime_claims_report(vec![0x33; 32], vec![0x44; 32]);
        let mut receipt =
            make_receipt_with_runtime_policy(claims.runtime.as_ref().unwrap().clone());
        receipt.input.sha256 = "not-hex".to_string();

        let err = verify_receipt_against_attestation(&report, &receipt, None, None).unwrap_err();

        assert!(err.to_string().contains("receipt.input.sha256"));
    }

    // --- verify_report (integration) ---

    #[test]
    fn test_verify_report_all_checks_pass() {
        let nonce = vec![0x01u8; 16];
        let model_hash = vec![0x02u8; 32];
        let measurement = vec![0x03u8; 48];
        let report_data = build_report_data(Some(&nonce), Some(&model_hash));
        let report = make_report(report_data, measurement.clone());

        let opts = VerifyOptions {
            nonce: Some(nonce),
            expected_model_hash: Some(model_hash),
            expected_measurement: Some(measurement),
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let result = verify_report(&report, &opts).unwrap();
        assert!(result.nonce_verified);
        assert!(result.model_hash_verified);
        assert!(result.measurement_verified);
        assert!(!result.hardware_verified);
        assert_eq!(result.tee_type, TeeType::Simulated);
    }

    #[test]
    fn test_verify_report_no_checks_passes() {
        let report = make_report(vec![0u8; 64], vec![0u8; 48]);
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };
        let result = verify_report(&report, &opts).unwrap();
        assert!(!result.nonce_verified);
        assert!(!result.model_hash_verified);
        assert!(!result.measurement_verified);
        assert!(!result.hardware_verified);
    }

    #[test]
    fn test_verify_report_nonce_mismatch_fails() {
        let nonce = vec![0x01u8; 16];
        let wrong_nonce = vec![0x99u8; 16];
        let report_data = build_report_data(Some(&nonce), None);
        let report = make_report(report_data, vec![0u8; 48]);

        let opts = VerifyOptions {
            nonce: Some(wrong_nonce),
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };
        assert!(verify_report(&report, &opts).is_err());
    }

    #[test]
    fn test_verify_report_with_hardware_verifier() {
        struct AlwaysOk;
        impl HardwareVerifier for AlwaysOk {
            fn verify_hardware_signature(&self, _report: &AttestationReport) -> Result<()> {
                Ok(())
            }
        }

        let report = make_report(vec![0u8; 64], vec![0u8; 48]);
        let verifier = AlwaysOk;
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: Some(&verifier),
        };
        let result = verify_report(&report, &opts).unwrap();
        assert!(result.hardware_verified);
    }

    #[test]
    fn test_verify_report_hardware_verifier_failure_propagates() {
        struct AlwaysFail;
        impl HardwareVerifier for AlwaysFail {
            fn verify_hardware_signature(&self, _report: &AttestationReport) -> Result<()> {
                Err(PowerError::AttestationVerificationFailed(
                    "signature invalid".to_string(),
                ))
            }
        }

        let report = make_report(vec![0u8; 64], vec![0u8; 48]);
        let verifier = AlwaysFail;
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: Some(&verifier),
        };
        let err = verify_report(&report, &opts).unwrap_err();
        assert!(err.to_string().contains("signature invalid"));
    }

    #[test]
    fn test_verify_report_strict_requires_hardware_verifier() {
        let mut report = make_report(vec![0u8; 64], vec![0u8; 48]);
        report.tee_type = TeeType::SevSnp;
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let err = verify_report_strict(&report, &opts).unwrap_err();
        assert!(err
            .to_string()
            .contains("hardware signature verification is required"));
    }

    #[test]
    fn test_verify_report_strict_rejects_simulated_even_with_verifier() {
        struct AlwaysOk;
        impl HardwareVerifier for AlwaysOk {
            fn verify_hardware_signature(&self, _report: &AttestationReport) -> Result<()> {
                Ok(())
            }
        }

        let report = make_report(vec![0u8; 64], vec![0u8; 48]);
        let verifier = AlwaysOk;
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: Some(&verifier),
        };

        let err = verify_report_strict(&report, &opts).unwrap_err();
        assert!(err
            .to_string()
            .contains("simulated TEE reports are rejected"));
    }

    #[test]
    fn test_verify_report_strict_requires_measurement_pin() {
        struct AlwaysOk;
        impl HardwareVerifier for AlwaysOk {
            fn verify_hardware_signature(&self, _report: &AttestationReport) -> Result<()> {
                Ok(())
            }
        }

        let mut report = make_report(vec![0u8; 64], vec![0u8; 48]);
        report.tee_type = TeeType::SevSnp;
        let verifier = AlwaysOk;
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: Some(&verifier),
        };

        let err = verify_report_strict(&report, &opts).unwrap_err();
        assert!(err
            .to_string()
            .contains("measurement verification is required"));
    }

    #[test]
    fn test_verify_report_strict_passes_for_hardware_type_with_verifier() {
        struct AlwaysOk;
        impl HardwareVerifier for AlwaysOk {
            fn verify_hardware_signature(&self, _report: &AttestationReport) -> Result<()> {
                Ok(())
            }
        }

        let mut report = make_report(vec![0u8; 64], vec![0u8; 48]);
        report.tee_type = TeeType::SevSnp;
        let measurement = report.measurement.clone();
        let verifier = AlwaysOk;
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: Some(measurement),
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: Some(&verifier),
        };

        let result = verify_report_strict(&report, &opts).unwrap();
        assert!(result.hardware_verified);
        assert!(result.measurement_verified);
        assert_eq!(result.tee_type, TeeType::SevSnp);
    }

    #[test]
    fn test_gpu_confidential_policy_enables_required_checks() {
        let policy = VerificationPolicy::gpu_confidential();

        assert!(policy.require_hardware_signature);
        assert!(policy.require_measurement);
        assert!(policy.reject_simulated);
        assert!(policy.require_nonce);
        assert!(policy.require_32_byte_nonce);
        assert!(policy.require_claims);
        assert!(policy.require_gpu_evidence);
        assert!(policy.require_gpu_evidence_nonce);
        assert!(policy.require_gpu_verdict_digest);
        assert!(policy.require_gpu_evidence_metadata_pins);
        assert!(policy.require_gpu_device_claims);
        assert!(policy.require_gpu_device_identity_pins);
        assert!(policy.require_runtime_policy);
        assert!(policy.require_gpu_execution_policy);
    }

    #[test]
    fn test_gpu_confidential_policy_requires_32_byte_nonce() {
        let (report, _) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let opts = VerifyOptions {
            nonce: Some(b"nonce".to_vec()),
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: Some(vec![0x22; 32]),
            expected_gpu_evidence: Some(expected_gpu_evidence_pins()),
            expected_gpu_devices: Some(expected_gpu_device_pins()),
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: Some(vec![0x55; 32]),
            hardware_verifier: None,
        };

        let err = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_gpu_confidential(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("requires a 32-byte nonce"));
    }

    #[test]
    fn test_gpu_confidential_policy_requires_gpu_evidence_nonce_claim() {
        let (mut report, mut claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        claims.gpu.as_mut().unwrap().nonce = None;
        report.report_data = build_claims_report_data(&claims).unwrap();
        report.claims = Some(claims);

        let opts = VerifyOptions {
            nonce: Some(gpu_nonce().to_vec()),
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: Some(vec![0x22; 32]),
            expected_gpu_evidence: Some(expected_gpu_evidence_pins()),
            expected_gpu_devices: Some(expected_gpu_device_pins()),
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: Some(vec![0x55; 32]),
            hardware_verifier: None,
        };

        let err = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_gpu_confidential(),
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("requires a GPU evidence nonce claim"));
    }

    #[test]
    fn test_gpu_confidential_policy_requires_gpu_evidence_metadata_pins() {
        let (report, _) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let opts = VerifyOptions {
            nonce: Some(gpu_nonce().to_vec()),
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: Some(vec![0x22; 32]),
            expected_gpu_evidence: None,
            expected_gpu_devices: Some(ExpectedGpuDevices {
                hwmodels: vec!["GH100 A01 GSP BROM".to_string()],
                ..Default::default()
            }),
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: Some(vec![0x55; 32]),
            hardware_verifier: None,
        };

        let err = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_gpu_confidential(),
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("requires expected GPU evidence provider/format/count pins"));
    }

    #[test]
    fn test_gpu_confidential_policy_requires_nras_verdict_digest() {
        let (report, _) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let opts = VerifyOptions {
            nonce: Some(gpu_nonce().to_vec()),
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: Some(vec![0x11; 32]),
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: Some(expected_gpu_evidence_pins()),
            expected_gpu_devices: Some(expected_gpu_device_pins()),
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: Some(vec![0x55; 32]),
            hardware_verifier: None,
        };

        let err = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_gpu_confidential(),
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("requires an expected NVIDIA NRAS verdict digest"));
    }

    #[test]
    fn test_gpu_confidential_policy_requires_gpu_execution_digest() {
        let (report, _) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let opts = VerifyOptions {
            nonce: Some(gpu_nonce().to_vec()),
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: Some(vec![0x22; 32]),
            expected_gpu_evidence: Some(expected_gpu_evidence_pins()),
            expected_gpu_devices: Some(expected_gpu_device_pins()),
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let err = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_gpu_confidential(),
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("requires an expected GPU execution digest"));
    }

    #[test]
    fn test_gpu_confidential_policy_requires_gpu_identity_pins() {
        let (report, _) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let opts = VerifyOptions {
            nonce: Some(gpu_nonce().to_vec()),
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: Some(vec![0x22; 32]),
            expected_gpu_evidence: Some(expected_gpu_evidence_pins()),
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: Some(vec![0x55; 32]),
            hardware_verifier: None,
        };

        let err = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_gpu_confidential(),
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("requires expected GPU identity/version pins"));
    }

    #[test]
    fn test_verify_report_with_policy_requires_declared_model_hash_check() {
        let mut report = make_report(vec![0u8; 64], vec![0u8; 48]);
        report.tee_type = TeeType::Tdx;
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let err = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_model_hash(),
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("model hash verification is required"));
    }

    #[test]
    fn test_verify_claims_binding_passes_when_report_data_matches_claims() {
        let (report, _) = make_claims_report(Some(b"nonce"), vec![0x02; 32]);
        verify_claims_binding(&report).unwrap();
    }

    #[test]
    fn test_verify_claims_binding_fails_when_report_data_tampered() {
        let (mut report, _) = make_claims_report(Some(b"nonce"), vec![0x02; 32]);
        report.report_data[0] ^= 0xFF;

        let err = verify_claims_binding(&report).unwrap_err();
        assert!(err.to_string().contains("claims binding mismatch"));
    }

    #[test]
    fn test_verify_claims_binding_fails_when_tee_type_mismatches() {
        let (mut report, _) = make_claims_report(Some(b"nonce"), vec![0x02; 32]);
        report.tee_type = TeeType::Tdx;

        let err = verify_claims_binding(&report).unwrap_err();
        assert!(err.to_string().contains("claims tee_type mismatch"));
    }

    #[test]
    fn test_verify_claims_binding_rejects_wrong_schema() {
        let (mut report, mut claims) = make_claims_report(Some(b"nonce"), vec![0x02; 32]);
        claims.schema = "a3s.power.attestation.v1".to_string();
        report.report_data = build_claims_report_data(&claims).unwrap();
        report.claims = Some(claims);

        let err = verify_claims_binding(&report).unwrap_err();

        assert!(err.to_string().contains("claims schema mismatch"));
    }

    #[test]
    fn test_verify_claims_binding_rejects_short_model_digest() {
        let (mut report, mut claims) = make_claims_report(Some(b"nonce"), vec![0x02; 32]);
        claims.model.as_mut().unwrap().digest = vec![0x02; 31];
        report.report_data = build_claims_report_data(&claims).unwrap();
        report.claims = Some(claims);

        let err = verify_claims_binding(&report).unwrap_err();

        assert!(err
            .to_string()
            .contains("claims.model.digest must be a 32-byte SHA-256 digest"));
    }

    #[test]
    fn test_verify_claims_binding_rejects_short_gpu_evidence_digest() {
        let (report, _) = make_gpu_claims_report(vec![0x11; 31], Some(vec![0x22; 32]));

        let err = verify_claims_binding(&report).unwrap_err();

        assert!(err
            .to_string()
            .contains("claims.gpu.evidence_digest must be a 32-byte SHA-256 digest"));
    }

    #[test]
    fn test_verify_claims_binding_rejects_short_runtime_decoding_digest() {
        let (report, _) = make_runtime_claims_report(vec![0x33; 32], vec![0x44; 31]);

        let err = verify_claims_binding(&report).unwrap_err();

        assert!(err
            .to_string()
            .contains("claims.runtime.decoding.parameters_sha256"));
    }

    #[test]
    fn test_verify_report_uses_v2_claims_for_nonce_and_model_hash() {
        let nonce = b"nonce";
        let model_hash = vec![0x02; 32];
        let (report, _) = make_claims_report(Some(nonce), model_hash.clone());
        let opts = VerifyOptions {
            nonce: Some(nonce.to_vec()),
            expected_model_hash: Some(model_hash),
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let result = verify_report(&report, &opts).unwrap();
        assert!(result.claims_verified);
        assert!(result.nonce_verified);
        assert!(result.model_hash_verified);
    }

    #[test]
    fn test_verify_report_with_policy_requires_v2_claims() {
        let report = make_report(vec![0u8; 64], vec![0u8; 48]);
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let err = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_claims(),
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("v2 attestation claims are required"));
    }

    #[test]
    fn test_verify_claims_gpu_evidence_binding_passes_for_evidence_and_verdict() {
        let evidence_digest = vec![0x11; 32];
        let verdict_digest = vec![0x22; 32];
        let (_, claims) =
            make_gpu_claims_report(evidence_digest.clone(), Some(verdict_digest.clone()));

        verify_claims_gpu_evidence_binding(&claims, Some(&evidence_digest), Some(&verdict_digest))
            .unwrap();
    }

    #[test]
    fn test_verify_claims_gpu_evidence_binding_fails_on_mismatch() {
        let (_, claims) = make_gpu_claims_report(vec![0x11; 32], Some(vec![0x22; 32]));

        let err = verify_claims_gpu_evidence_binding(&claims, Some(&[0x99; 32]), None).unwrap_err();
        assert!(err.to_string().contains("GPU evidence digest mismatch"));
    }

    #[test]
    fn test_verify_claims_gpu_evidence_binding_fails_on_nonce_mismatch() {
        let (_, mut claims) = make_gpu_claims_report(vec![0x11; 32], Some(vec![0x22; 32]));
        claims.gpu.as_mut().unwrap().nonce = Some(vec![0x99]);

        let err = verify_claims_gpu_evidence_binding(&claims, Some(&[0x11; 32]), None).unwrap_err();

        assert!(err.to_string().contains("GPU evidence nonce mismatch"));
    }

    #[test]
    fn test_verify_claims_expected_gpu_evidence_passes_for_metadata_pins() {
        let (_, claims) = make_gpu_claims_report(vec![0x11; 32], Some(vec![0x22; 32]));
        let expected = ExpectedGpuEvidence {
            provider: Some("nvidia-nras".to_string()),
            evidence_format: Some("nvidia-nvattest-evidence-json".to_string()),
            verdict_format: Some("nvidia-nvattest-attestation-json".to_string()),
            evidence_count: Some(1),
        };

        verify_claims_expected_gpu_evidence(&claims, &expected).unwrap();
    }

    #[test]
    fn test_verify_claims_expected_gpu_evidence_fails_on_provider_mismatch() {
        let (_, claims) = make_gpu_claims_report(vec![0x11; 32], Some(vec![0x22; 32]));
        let expected = ExpectedGpuEvidence {
            provider: Some("other-provider".to_string()),
            ..Default::default()
        };

        let err = verify_claims_expected_gpu_evidence(&claims, &expected).unwrap_err();

        assert!(err.to_string().contains("GPU evidence provider mismatch"));
    }

    #[test]
    fn test_verify_claims_expected_gpu_evidence_fails_on_count_mismatch() {
        let (_, claims) = make_gpu_claims_report(vec![0x11; 32], Some(vec![0x22; 32]));
        let expected = ExpectedGpuEvidence {
            evidence_count: Some(2),
            ..Default::default()
        };

        let err = verify_claims_expected_gpu_evidence(&claims, &expected).unwrap_err();

        assert!(err.to_string().contains("GPU evidence_count mismatch"));
    }

    #[test]
    fn test_verify_report_checks_gpu_evidence_metadata() {
        let (report, _) = make_gpu_claims_report(vec![0x11; 32], Some(vec![0x22; 32]));
        let opts = VerifyOptions {
            nonce: Some(gpu_nonce().to_vec()),
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: Some(expected_gpu_evidence_pins()),
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let result =
            verify_report_with_policy(&report, &opts, VerificationPolicy::permissive()).unwrap();

        assert!(result.claims_verified);
        assert!(result.gpu_evidence_verified);
    }

    #[test]
    fn test_verify_claims_gpu_device_claims_passes_for_structured_nvidia_claims() {
        let (_, claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));

        verify_claims_gpu_device_claims(&claims).unwrap();
    }

    #[test]
    fn test_verify_claims_gpu_device_claims_fails_when_missing() {
        let (_, claims) = make_gpu_claims_report(vec![0x11; 32], Some(vec![0x22; 32]));

        let err = verify_claims_gpu_device_claims(&claims).unwrap_err();

        assert!(err
            .to_string()
            .contains("does not include structured NVIDIA device claims"));
    }

    #[test]
    fn test_verify_claims_gpu_device_claims_fails_on_device_nonce_mismatch() {
        let (_, mut claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        claims
            .gpu
            .as_mut()
            .unwrap()
            .devices
            .first_mut()
            .unwrap()
            .attestation_nonce = Some(vec![0x99]);

        let err = verify_claims_gpu_device_claims(&claims).unwrap_err();

        assert!(err.to_string().contains("GPU device nonce mismatch"));
    }

    #[test]
    fn test_verify_claims_gpu_device_claims_fails_on_failed_validation() {
        let (_, mut claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        claims
            .gpu
            .as_mut()
            .unwrap()
            .devices
            .first_mut()
            .unwrap()
            .validation
            .driver_rim_version_match = Some(false);

        let err = verify_claims_gpu_device_claims(&claims).unwrap_err();

        assert!(err
            .to_string()
            .contains("driver_rim_version_match is false"));
    }

    #[test]
    fn test_verify_claims_gpu_device_claims_fails_on_driver_rim_schema_not_validated() {
        let (_, mut claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        claims
            .gpu
            .as_mut()
            .unwrap()
            .devices
            .first_mut()
            .unwrap()
            .validation
            .driver_rim_schema_validated = Some(false);

        let err = verify_claims_gpu_device_claims(&claims).unwrap_err();

        assert!(err
            .to_string()
            .contains("driver_rim_schema_validated is false"));
    }

    #[test]
    fn test_verify_claims_gpu_device_claims_fails_on_nvswitch_rim_schema_not_validated() {
        let (_, mut claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let mut nvswitch = nvswitch_device_claim(1, &gpu_nonce());
        nvswitch.validation.firmware_rim_schema_validated = Some(false);
        claims.gpu.as_mut().unwrap().devices.push(nvswitch);

        let err = verify_claims_gpu_device_claims(&claims).unwrap_err();

        assert!(err
            .to_string()
            .contains("firmware_rim_schema_validated is false"));
    }

    #[test]
    fn test_verify_claims_gpu_device_claims_fails_when_secure_boot_disabled() {
        let (_, mut claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        claims
            .gpu
            .as_mut()
            .unwrap()
            .devices
            .first_mut()
            .unwrap()
            .secure_boot = Some(false);

        let err = verify_claims_gpu_device_claims(&claims).unwrap_err();

        assert!(err.to_string().contains("secure_boot is false"));
    }

    #[test]
    fn test_verify_claims_gpu_device_claims_fails_when_debug_enabled() {
        let (_, mut claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        claims
            .gpu
            .as_mut()
            .unwrap()
            .devices
            .first_mut()
            .unwrap()
            .debug_status = Some("enabled".to_string());

        let err = verify_claims_gpu_device_claims(&claims).unwrap_err();

        assert!(err
            .to_string()
            .contains("debug_status must be \"disabled\""));
    }

    #[test]
    fn test_verify_claims_gpu_device_claims_fails_when_nvswitch_secure_boot_disabled() {
        let (_, mut claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let mut nvswitch = nvswitch_device_claim(1, &gpu_nonce());
        nvswitch.secure_boot = Some(false);
        claims.gpu.as_mut().unwrap().devices.push(nvswitch);

        let err = verify_claims_gpu_device_claims(&claims).unwrap_err();

        assert!(err.to_string().contains("nvswitch secure_boot is false"));
    }

    #[test]
    fn test_verify_claims_gpu_device_claims_fails_when_nvswitch_debug_enabled() {
        let (_, mut claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let mut nvswitch = nvswitch_device_claim(1, &gpu_nonce());
        nvswitch.debug_status = Some("enabled".to_string());
        claims.gpu.as_mut().unwrap().devices.push(nvswitch);

        let err = verify_claims_gpu_device_claims(&claims).unwrap_err();

        assert!(err
            .to_string()
            .contains("nvswitch debug_status must be \"disabled\""));
    }

    #[test]
    fn test_verify_claims_expected_gpu_devices_passes_for_exact_identity_pins() {
        let (_, claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let expected = expected_gpu_device_pins();

        verify_claims_expected_gpu_devices(&claims, &expected).unwrap();
    }

    #[test]
    fn test_verify_claims_expected_gpu_devices_fails_on_gpu_count_mismatch() {
        let (_, claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let expected = ExpectedGpuDevices {
            gpu_count: Some(2),
            ..Default::default()
        };

        let err = verify_claims_expected_gpu_devices(&claims, &expected).unwrap_err();

        assert!(err.to_string().contains("GPU device count mismatch"));
    }

    #[test]
    fn test_verify_claims_expected_gpu_devices_passes_for_nvswitch_pins() {
        let (_, mut claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        claims
            .gpu
            .as_mut()
            .unwrap()
            .devices
            .push(nvswitch_device_claim(1, &gpu_nonce()));
        let expected = ExpectedGpuDevices {
            nvswitch_count: Some(1),
            nvswitch_ueids: vec!["nvswitch-ueid-0".to_string()],
            nvswitch_oemids: vec!["5703".to_string()],
            nvswitch_claims_versions: vec!["3.0".to_string()],
            nvswitch_hwmodels: vec!["NVSwitch B01".to_string()],
            nvswitch_firmware_versions: vec!["1.2.3".to_string()],
            ..Default::default()
        };

        verify_claims_expected_gpu_devices(&claims, &expected).unwrap();
    }

    #[test]
    fn test_verify_claims_expected_gpu_devices_fails_on_nvswitch_ueid_mismatch() {
        let (_, mut claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        claims
            .gpu
            .as_mut()
            .unwrap()
            .devices
            .push(nvswitch_device_claim(1, &gpu_nonce()));
        let expected = ExpectedGpuDevices {
            nvswitch_ueids: vec!["unexpected-switch".to_string()],
            ..Default::default()
        };

        let err = verify_claims_expected_gpu_devices(&claims, &expected).unwrap_err();

        assert!(err.to_string().contains("NVSwitch UEID set mismatch"));
    }

    #[test]
    fn test_verify_claims_expected_gpu_devices_fails_on_claims_version_mismatch() {
        let (_, claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let expected = ExpectedGpuDevices {
            claims_versions: vec!["2.0".to_string()],
            ..Default::default()
        };

        let err = verify_claims_expected_gpu_devices(&claims, &expected).unwrap_err();

        assert!(err.to_string().contains("claims_version mismatch"));
    }

    #[test]
    fn test_verify_claims_expected_gpu_devices_fails_on_ueid_mismatch() {
        let (_, claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let expected = ExpectedGpuDevices {
            gpu_ueids: vec!["unexpected-ueid".to_string()],
            ..Default::default()
        };

        let err = verify_claims_expected_gpu_devices(&claims, &expected).unwrap_err();

        assert!(err.to_string().contains("GPU UEID set mismatch"));
    }

    #[test]
    fn test_verify_claims_expected_gpu_devices_fails_on_oemid_mismatch() {
        let (_, claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let expected = ExpectedGpuDevices {
            oemids: vec!["9999".to_string()],
            ..Default::default()
        };

        let err = verify_claims_expected_gpu_devices(&claims, &expected).unwrap_err();

        assert!(err.to_string().contains("oemid mismatch"));
    }

    #[test]
    fn test_verify_claims_expected_gpu_devices_fails_on_driver_version_mismatch() {
        let (_, claims) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(vec![0x22; 32]));
        let expected = ExpectedGpuDevices {
            driver_versions: vec!["999.99".to_string()],
            ..Default::default()
        };

        let err = verify_claims_expected_gpu_devices(&claims, &expected).unwrap_err();

        assert!(err.to_string().contains("driver_version mismatch"));
    }

    #[test]
    fn test_verify_report_with_policy_requires_gpu_evidence_claim() {
        let (report, _) = make_claims_report(Some(b"nonce"), vec![0x02; 32]);
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: Some(vec![0x11; 32]),
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let err = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_gpu_evidence(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("GPU evidence claim is required"));
    }

    #[test]
    fn test_verify_report_checks_gpu_verdict_digest() {
        let verdict_digest = vec![0x22; 32];
        let (report, _) = make_gpu_claims_report(vec![0x11; 32], Some(verdict_digest.clone()));
        let opts = VerifyOptions {
            nonce: Some(gpu_nonce().to_vec()),
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: Some(verdict_digest),
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let result = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_gpu_evidence(),
        )
        .unwrap();
        assert!(result.claims_verified);
        assert!(result.gpu_evidence_verified);
    }

    #[test]
    fn test_verify_report_with_policy_checks_gpu_device_claims() {
        let verdict_digest = vec![0x22; 32];
        let (report, _) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(verdict_digest.clone()));
        let opts = VerifyOptions {
            nonce: Some(gpu_nonce().to_vec()),
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: Some(verdict_digest),
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let result = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_gpu_device_claims(),
        )
        .unwrap();

        assert!(result.gpu_evidence_verified);
        assert!(result.gpu_device_claims_verified);
    }

    #[test]
    fn test_verify_report_with_policy_checks_expected_gpu_device_pins() {
        let verdict_digest = vec![0x22; 32];
        let (report, _) =
            make_gpu_claims_report_with_device_claims(vec![0x11; 32], Some(verdict_digest.clone()));
        let opts = VerifyOptions {
            nonce: Some(gpu_nonce().to_vec()),
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: Some(verdict_digest),
            expected_gpu_evidence: None,
            expected_gpu_devices: Some(expected_gpu_device_pins()),
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let result =
            verify_report_with_policy(&report, &opts, VerificationPolicy::permissive()).unwrap();

        assert!(result.gpu_device_claims_verified);
    }

    #[test]
    fn test_verify_report_with_policy_requires_expected_nonce_for_gpu_evidence() {
        let verdict_digest = vec![0x22; 32];
        let (report, _) = make_gpu_claims_report(vec![0x11; 32], Some(verdict_digest.clone()));
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: Some(verdict_digest),
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: None,
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let err = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_gpu_evidence(),
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("GPU evidence verification requires an expected nonce"));
    }

    #[test]
    fn test_verify_claims_runtime_policy_binding_passes() {
        let chat_template_digest = vec![0x33; 32];
        let decoding_digest = vec![0x44; 32];
        let (_, claims) =
            make_runtime_claims_report(chat_template_digest.clone(), decoding_digest.clone());

        verify_claims_runtime_policy_binding(
            &claims,
            Some(&chat_template_digest),
            Some(&decoding_digest),
            None,
        )
        .unwrap();
    }

    #[test]
    fn test_verify_claims_runtime_policy_binding_fails_on_mismatch() {
        let (_, claims) = make_runtime_claims_report(vec![0x33; 32], vec![0x44; 32]);

        let err = verify_claims_runtime_policy_binding(&claims, Some(&[0x99; 32]), None, None)
            .unwrap_err();
        assert!(err.to_string().contains("Chat template digest mismatch"));
    }

    #[test]
    fn test_verify_claims_runtime_policy_binding_checks_gpu_execution() {
        let gpu_execution_digest = vec![0x55; 32];
        let (_, claims) = make_runtime_execution_claims_report(gpu_execution_digest.clone());

        verify_claims_runtime_policy_binding(&claims, None, None, Some(&gpu_execution_digest))
            .unwrap();
    }

    #[test]
    fn test_verify_claims_runtime_policy_binding_fails_on_gpu_execution_mismatch() {
        let (_, claims) = make_runtime_execution_claims_report(vec![0x55; 32]);

        let err = verify_claims_runtime_policy_binding(&claims, None, None, Some(&[0x99; 32]))
            .unwrap_err();

        assert!(err.to_string().contains("GPU execution digest mismatch"));
    }

    #[test]
    fn test_verify_report_with_policy_requires_runtime_policy_claim() {
        let (report, _) = make_claims_report(Some(b"nonce"), vec![0x02; 32]);
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: Some(vec![0x33; 32]),
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let err = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_runtime_policy(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("runtime policy claim is required"));
    }

    #[test]
    fn test_verify_report_checks_runtime_policy_digest() {
        let chat_template_digest = vec![0x33; 32];
        let (report, _) = make_runtime_claims_report(chat_template_digest.clone(), vec![0x44; 32]);
        let opts = VerifyOptions {
            nonce: None,
            expected_model_hash: None,
            expected_measurement: None,
            expected_gpu_evidence_digest: None,
            expected_gpu_verdict_digest: None,
            expected_gpu_evidence: None,
            expected_gpu_devices: None,
            expected_chat_template_digest: Some(chat_template_digest),
            expected_decoding_parameters_digest: None,
            expected_gpu_execution_digest: None,
            hardware_verifier: None,
        };

        let result = verify_report_with_policy(
            &report,
            &opts,
            VerificationPolicy::permissive().require_runtime_policy(),
        )
        .unwrap();
        assert!(result.claims_verified);
        assert!(result.runtime_policy_verified);
    }

    // --- constant_time_eq ---

    #[test]
    fn test_constant_time_eq_equal() {
        assert!(constant_time_eq(b"hello", b"hello"));
    }

    #[test]
    fn test_constant_time_eq_different() {
        assert!(!constant_time_eq(b"hello", b"world"));
    }

    #[test]
    fn test_constant_time_eq_different_lengths() {
        assert!(!constant_time_eq(b"hello", b"hell"));
    }

    #[test]
    fn test_constant_time_eq_empty() {
        assert!(constant_time_eq(b"", b""));
    }
}
