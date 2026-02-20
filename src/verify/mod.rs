//! Client-side attestation report verification.
//!
//! Verifies that an [`AttestationReport`] from `GET /v1/attestation` is
//! cryptographically bound to the expected nonce and model hash, and that
//! the platform measurement matches a known-good value.
//!
//! # What this module verifies (offline, no network required)
//!
//! 1. **Nonce binding** — `report_data[0..32]` matches the client-supplied nonce.
//!    Prevents replay attacks: a report generated for a different request cannot
//!    be substituted.
//!
//! 2. **Model hash binding** — `report_data[32..64]` matches the expected
//!    SHA-256 of the model file. Proves the attested server is serving the
//!    specific model the client expects.
//!
//! 3. **Measurement check** — the platform launch digest matches a known-good
//!    value pinned by the operator. Proves the TEE firmware/OS has not been
//!    tampered with.
//!
//! # What this module does NOT verify
//!
//! Hardware signature verification (AMD KDS / Intel PCS certificate chain)
//! requires network access and platform-specific tooling. Implement the
//! [`HardwareVerifier`] trait to plug in your own verifier.
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
//!     hardware_verifier: None,
//! };
//!
//! verify_report(&report, &opts)?;
//! println!("Attestation verified");
//! # Ok(())
//! # }
//! ```

use crate::error::{PowerError, Result};
use crate::tee::attestation::{AttestationReport, TeeType};

// ============================================================================
// Public types
// ============================================================================

/// Options controlling which checks are performed during verification.
pub struct VerifyOptions<'a> {
    /// Client-supplied nonce (raw bytes). When set, verifies that
    /// `report_data[0..32]` matches this value.
    pub nonce: Option<Vec<u8>>,

    /// Expected SHA-256 hash of the model file (raw bytes, 32 bytes).
    /// When set, verifies that `report_data[32..64]` matches this value.
    pub expected_model_hash: Option<Vec<u8>>,

    /// Expected platform measurement (raw bytes). When set, verifies that
    /// `report.measurement` matches this value exactly.
    pub expected_measurement: Option<Vec<u8>>,

    /// Optional hardware signature verifier. When `None`, hardware signature
    /// verification is skipped (offline mode).
    pub hardware_verifier: Option<&'a dyn HardwareVerifier>,
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
}

// ============================================================================
// Core verification logic
// ============================================================================

/// Verify an attestation report against the provided options.
///
/// Returns [`VerifyResult`] on success, or the first [`PowerError`] encountered.
pub fn verify_report(report: &AttestationReport, opts: &VerifyOptions<'_>) -> Result<VerifyResult> {
    let mut result = VerifyResult {
        tee_type: report.tee_type,
        nonce_verified: false,
        model_hash_verified: false,
        measurement_verified: false,
        hardware_verified: false,
    };

    // 1. Nonce binding check
    if let Some(ref nonce) = opts.nonce {
        verify_nonce_binding(&report.report_data, nonce)?;
        result.nonce_verified = true;
    }

    // 2. Model hash binding check
    if let Some(ref model_hash) = opts.expected_model_hash {
        verify_model_hash_binding(&report.report_data, model_hash)?;
        result.model_hash_verified = true;
    }

    // 3. Measurement check
    if let Some(ref expected) = opts.expected_measurement {
        verify_measurement(&report.measurement, expected)?;
        result.measurement_verified = true;
    }

    // 4. Hardware signature verification (optional)
    if let Some(verifier) = opts.hardware_verifier {
        verifier.verify_hardware_signature(report)?;
        result.hardware_verified = true;
    }

    Ok(result)
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
/// The model hash is zero-padded to 32 bytes if shorter. This matches the
/// layout produced by `build_report_data()` on the server side.
pub fn verify_model_hash_binding(report_data: &[u8], model_hash: &[u8]) -> Result<()> {
    if report_data.len() < 64 {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "report_data too short for model hash: {} bytes (need 64)",
            report_data.len()
        )));
    }

    let hash_len = model_hash.len().min(32);
    let report_hash = &report_data[32..32 + hash_len];
    let provided_hash = &model_hash[..hash_len];

    if !constant_time_eq(report_hash, provided_hash) {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "Model hash mismatch: report_data[32..{}] = {}, expected {}",
            32 + hash_len,
            hex::encode(report_hash),
            hex::encode(provided_hash),
        )));
    }

    Ok(())
}

/// Verify that the platform measurement matches the expected value.
pub fn verify_measurement(measurement: &[u8], expected: &[u8]) -> Result<()> {
    if !constant_time_eq(measurement, expected) {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "Measurement mismatch: got {}, expected {}",
            hex::encode(measurement),
            hex::encode(expected),
        )));
    }
    Ok(())
}

/// Constant-time byte slice comparison to prevent timing side-channels.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tee::attestation::{AttestationReport, TeeType};

    fn make_report(report_data: Vec<u8>, measurement: Vec<u8>) -> AttestationReport {
        AttestationReport {
            version: "1.0".to_string(),
            tee_type: TeeType::Simulated,
            report_data,
            measurement,
            raw_report: None,
            timestamp: chrono::Utc::now(),
            nonce: None,
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
    fn test_measurement_fails_when_mismatch() {
        let m = vec![0xCCu8; 48];
        let wrong = vec![0xDDu8; 48];
        let err = verify_measurement(&m, &wrong).unwrap_err();
        assert!(err.to_string().contains("Measurement mismatch"));
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
            hardware_verifier: Some(&verifier),
        };
        let err = verify_report(&report, &opts).unwrap_err();
        assert!(err.to_string().contains("signature invalid"));
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
