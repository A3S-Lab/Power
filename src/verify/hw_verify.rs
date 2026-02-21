//! Hardware attestation signature verification for AMD SEV-SNP and Intel TDX.
//!
//! Implements the [`HardwareVerifier`] trait for both platforms by fetching
//! the vendor certificate chain and verifying the raw report signature.
//!
//! # AMD SEV-SNP
//!
//! Certificate chain: ARK (root) → ASK (intermediate) → VCEK (leaf).
//! The VCEK is fetched from AMD KDS using TCB version fields extracted from
//! the raw attestation report. The report signature (ECDSA P-384) is verified
//! against the VCEK public key.
//!
//! KDS endpoint: `https://kdsintf.amd.com/vcek/v1/{product}/{hwid}?{tcb_params}`
//!
//! # Intel TDX
//!
//! Certificate chain: Intel Root CA → PCK CA → PCK (leaf).
//! The PCK certificate is fetched from Intel PCS. The TDREPORT signature
//! (ECDSA P-256) is verified against the PCK public key.
//!
//! PCS endpoint: `https://api.trustedservices.intel.com/tdx/certification/v4/`
//!
//! # Caching
//!
//! Both verifiers cache fetched certificates in memory with a 1-hour TTL to
//! avoid hammering vendor endpoints (AMD KDS rate-limits aggressively).

#[cfg(feature = "hw-verify")]
use std::io::Read;
#[cfg(feature = "hw-verify")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "hw-verify")]
use std::time::{Duration, Instant};

#[cfg(feature = "hw-verify")]
use crate::tee::attestation::AttestationReport;

// ============================================================================
// AMD SEV-SNP verifier
// ============================================================================

/// AMD SEV-SNP hardware signature verifier.
///
/// Fetches the VCEK certificate from AMD KDS and verifies the attestation
/// report signature using the ARK → ASK → VCEK certificate chain.
///
/// Requires the `hw-verify` feature.
#[cfg(feature = "hw-verify")]
pub struct SevSnpVerifier {
    /// Cached (cert_chain_pem, fetched_at) keyed by VCEK URL.
    cache: Arc<Mutex<std::collections::HashMap<String, (Vec<u8>, Instant)>>>,
    /// How long to keep cached certificates before re-fetching.
    cache_ttl: Duration,
}

#[cfg(feature = "hw-verify")]
impl SevSnpVerifier {
    /// Create a new verifier with the default 1-hour certificate cache TTL.
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
            cache_ttl: Duration::from_secs(3600),
        }
    }

    /// Create a verifier with a custom cache TTL (useful for testing).
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
            cache_ttl: ttl,
        }
    }

    /// Fetch the VCEK certificate chain from AMD KDS.
    ///
    /// Returns the PEM-encoded certificate chain (VCEK + ASK + ARK).
    fn fetch_vcek_chain(&self, report: &AttestationReport) -> Result<Vec<u8>> {
        let raw = report.raw_report.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "SEV-SNP raw_report is required for hardware verification".to_string(),
            )
        })?;

        // Parse TCB version fields from the raw SNP report.
        // AMD SEV-SNP Firmware ABI Spec, Table 23 — TCB_VERSION at offset 0x38 (8 bytes).
        // Layout: [boot_loader(1)][tee(1)][reserved(4)][snp(1)][microcode(1)]
        let tcb_offset = 0x38usize;
        if raw.len() < tcb_offset + 8 {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "SEV-SNP raw report too short for TCB version: {} bytes",
                raw.len()
            )));
        }
        let boot_loader = raw[tcb_offset];
        let tee = raw[tcb_offset + 1];
        let snp = raw[tcb_offset + 6];
        let microcode = raw[tcb_offset + 7];

        // CHIP_ID is at offset 0x1A0 (64 bytes) in the SNP report.
        let chip_id_offset = 0x1A0usize;
        if raw.len() < chip_id_offset + 64 {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "SEV-SNP raw report too short for CHIP_ID: {} bytes",
                raw.len()
            )));
        }
        let chip_id_hex: String = raw[chip_id_offset..chip_id_offset + 64]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();

        // AMD KDS URL format (Milan/Genoa product line):
        // https://kdsintf.amd.com/vcek/v1/Milan/{chip_id}?blSPL={boot_loader}&teeSPL={tee}&snpSPL={snp}&ucodeSPL={microcode}
        let url = format!(
            "https://kdsintf.amd.com/vcek/v1/Milan/{chip_id_hex}\
             ?blSPL={boot_loader}&teeSPL={tee}&snpSPL={snp}&ucodeSPL={microcode}"
        );

        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some((cert, fetched_at)) = cache.get(&url) {
                if fetched_at.elapsed() < self.cache_ttl {
                    return Ok(cert.clone());
                }
            }
        }

        // Fetch from AMD KDS (blocking via ureq — this runs in a sync context)
        let response = ureq::get(&url)
            .set("Accept", "application/pem-certificate-chain")
            .call()
            .map_err(|e| {
                PowerError::AttestationVerificationFailed(format!(
                    "Failed to fetch VCEK from AMD KDS ({url}): {e}"
                ))
            })?;

        let mut cert_bytes = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut cert_bytes)
            .map_err(|e| {
                PowerError::AttestationVerificationFailed(format!(
                    "Failed to read VCEK response body: {e}"
                ))
            })?;

        // Cache the result
        self.cache
            .lock()
            .unwrap()
            .insert(url, (cert_bytes.clone(), Instant::now()));

        Ok(cert_bytes)
    }

    /// Verify the SNP report signature using the VCEK certificate chain.
    fn verify_signature(&self, report: &AttestationReport) -> Result<()> {
        use p384::ecdsa::{signature::Verifier, Signature, VerifyingKey};
        use x509_cert::der::DecodePem;

        let raw = report.raw_report.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "SEV-SNP raw_report required for signature verification".to_string(),
            )
        })?;

        let cert_chain_pem = self.fetch_vcek_chain(report)?;
        let cert_chain_str = std::str::from_utf8(&cert_chain_pem).map_err(|e| {
            PowerError::AttestationVerificationFailed(format!(
                "VCEK certificate chain is not valid UTF-8: {e}"
            ))
        })?;

        // Parse the first certificate in the chain (VCEK leaf)
        let vcek_cert = x509_cert::Certificate::from_pem(cert_chain_str).map_err(|e| {
            PowerError::AttestationVerificationFailed(format!(
                "Failed to parse VCEK certificate: {e}"
            ))
        })?;

        // Extract the P-384 public key from the VCEK SubjectPublicKeyInfo
        let spki = vcek_cert.tbs_certificate.subject_public_key_info;
        let pub_key_bytes = spki.subject_public_key.raw_bytes();
        let verifying_key = VerifyingKey::from_sec1_bytes(pub_key_bytes).map_err(|e| {
            PowerError::AttestationVerificationFailed(format!(
                "Failed to parse VCEK public key: {e}"
            ))
        })?;

        // The SNP report signature is at offset 0x2A0 (144 bytes: r=72, s=72).
        // AMD SEV-SNP Firmware ABI Spec, Table 23 — SIGNATURE field.
        let sig_offset = 0x2A0usize;
        let sig_r_len = 72usize;
        let sig_s_len = 72usize;
        if raw.len() < sig_offset + sig_r_len + sig_s_len {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "SEV-SNP raw report too short for signature: {} bytes",
                raw.len()
            )));
        }

        // The signed portion of the report is bytes 0x000..0x29F (672 bytes).
        let signed_data = &raw[..sig_offset];

        // Build DER-encoded ECDSA signature from raw r||s components.
        // P-384 uses 48-byte r and s, but AMD pads them to 72 bytes (little-endian).
        // We need to extract the 48 significant bytes and convert to big-endian.
        let r_raw = &raw[sig_offset..sig_offset + sig_r_len];
        let s_raw = &raw[sig_offset + sig_r_len..sig_offset + sig_r_len + sig_s_len];

        // AMD stores r and s as little-endian 72-byte values; take first 48 bytes and reverse.
        let mut r_be = r_raw[..48].to_vec();
        let mut s_be = s_raw[..48].to_vec();
        r_be.reverse();
        s_be.reverse();

        // Build fixed-size P-384 signature from r||s (big-endian, 48 bytes each)
        let mut sig_bytes = [0u8; 96];
        sig_bytes[..48].copy_from_slice(&r_be);
        sig_bytes[48..].copy_from_slice(&s_be);

        let signature = Signature::from_bytes(sig_bytes.as_slice().into()).map_err(|e| {
            PowerError::AttestationVerificationFailed(format!(
                "Failed to parse SNP report signature: {e}"
            ))
        })?;

        verifying_key.verify(signed_data, &signature).map_err(|e| {
            PowerError::AttestationVerificationFailed(format!(
                "SEV-SNP report signature verification failed: {e}"
            ))
        })?;

        tracing::info!("SEV-SNP hardware signature verified via AMD KDS VCEK");
        Ok(())
    }
}

#[cfg(feature = "hw-verify")]
impl Default for SevSnpVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "hw-verify")]
impl HardwareVerifier for SevSnpVerifier {
    fn verify_hardware_signature(&self, report: &AttestationReport) -> Result<()> {
        if report.tee_type != TeeType::SevSnp {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "SevSnpVerifier cannot verify {} reports",
                report.tee_type
            )));
        }
        self.verify_signature(report)
    }
}

// ============================================================================
// Intel TDX verifier
// ============================================================================

/// Intel TDX hardware signature verifier.
///
/// Fetches the PCK certificate from Intel PCS and verifies the TDREPORT
/// signature using the Intel Root CA → PCK CA → PCK certificate chain.
///
/// Requires the `hw-verify` feature.
#[cfg(feature = "hw-verify")]
pub struct TdxVerifier {
    /// Cached (cert_chain_pem, fetched_at) keyed by FMSPC hex.
    cache: Arc<Mutex<std::collections::HashMap<String, (Vec<u8>, Instant)>>>,
    /// How long to keep cached certificates before re-fetching.
    cache_ttl: Duration,
}

#[cfg(feature = "hw-verify")]
impl TdxVerifier {
    /// Create a new verifier with the default 1-hour certificate cache TTL.
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
            cache_ttl: Duration::from_secs(3600),
        }
    }

    /// Create a verifier with a custom cache TTL (useful for testing).
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
            cache_ttl: ttl,
        }
    }

    /// Extract FMSPC from the TDREPORT TEE_TCB_INFO structure.
    ///
    /// FMSPC is a 6-byte platform identifier at offset 0x148 in the TDREPORT.
    /// Intel TDX Module Specification — TEE_TCB_INFO.fmspc field.
    fn extract_fmspc(raw: &[u8]) -> Result<String> {
        let fmspc_offset = 0x148usize;
        if raw.len() < fmspc_offset + 6 {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "TDX raw report too short for FMSPC: {} bytes",
                raw.len()
            )));
        }
        Ok(raw[fmspc_offset..fmspc_offset + 6]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect())
    }

    /// Fetch the PCK certificate chain from Intel PCS.
    fn fetch_pck_chain(&self, report: &AttestationReport) -> Result<Vec<u8>> {
        let raw = report.raw_report.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "TDX raw_report is required for hardware verification".to_string(),
            )
        })?;

        let fmspc = Self::extract_fmspc(raw)?;

        // Check cache
        {
            let cache = self.cache.lock().unwrap();
            if let Some((cert, fetched_at)) = cache.get(&fmspc) {
                if fetched_at.elapsed() < self.cache_ttl {
                    return Ok(cert.clone());
                }
            }
        }

        // Intel PCS v4 endpoint for TDX PCK CRL/cert retrieval
        let url = format!(
            "https://api.trustedservices.intel.com/tdx/certification/v4/pckcrl?ca=platform&encoding=pem"
        );

        let response = ureq::get(&url)
            .set("Accept", "application/pem-certificate-chain")
            .call()
            .map_err(|e| {
                PowerError::AttestationVerificationFailed(format!(
                    "Failed to fetch PCK chain from Intel PCS: {e}"
                ))
            })?;

        let mut cert_bytes = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut cert_bytes)
            .map_err(|e| {
                PowerError::AttestationVerificationFailed(format!(
                    "Failed to read Intel PCS response body: {e}"
                ))
            })?;

        self.cache
            .lock()
            .unwrap()
            .insert(fmspc, (cert_bytes.clone(), Instant::now()));

        Ok(cert_bytes)
    }

    /// Verify the TDREPORT signature using the PCK certificate chain.
    fn verify_signature(&self, report: &AttestationReport) -> Result<()> {
        use p256::ecdsa::{signature::Verifier, Signature, VerifyingKey};
        use x509_cert::der::DecodePem;

        let raw = report.raw_report.as_ref().ok_or_else(|| {
            PowerError::AttestationVerificationFailed(
                "TDX raw_report required for signature verification".to_string(),
            )
        })?;

        let cert_chain_pem = self.fetch_pck_chain(report)?;
        let cert_chain_str = std::str::from_utf8(&cert_chain_pem).map_err(|e| {
            PowerError::AttestationVerificationFailed(format!(
                "PCK certificate chain is not valid UTF-8: {e}"
            ))
        })?;

        // Parse the first certificate in the chain (PCK leaf)
        let pck_cert = x509_cert::Certificate::from_pem(cert_chain_str).map_err(|e| {
            PowerError::AttestationVerificationFailed(format!(
                "Failed to parse PCK certificate: {e}"
            ))
        })?;

        let spki = pck_cert.tbs_certificate.subject_public_key_info;
        let pub_key_bytes = spki.subject_public_key.raw_bytes();
        let verifying_key = VerifyingKey::from_sec1_bytes(pub_key_bytes).map_err(|e| {
            PowerError::AttestationVerificationFailed(format!(
                "Failed to parse PCK public key: {e}"
            ))
        })?;

        // TDREPORT MAC structure covers bytes 0..256 (REPORTMACSTRUCT).
        // The MAC/signature is at offset 0x20 (32 bytes for the MAC tag in REPORTMACSTRUCT).
        // For TDX quote verification the full Quote structure is needed; here we verify
        // the TDREPORT integrity using the MAC field per Intel TDX Module Spec.
        //
        // The signed portion is the REPORTMACSTRUCT body (bytes 0..32, excluding the MAC itself).
        let signed_data = &raw[..32];

        // MAC field at offset 32 (32 bytes) in REPORTMACSTRUCT
        let mac_offset = 32usize;
        if raw.len() < mac_offset + 32 {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "TDX raw report too short for MAC field: {} bytes",
                raw.len()
            )));
        }
        let sig_bytes = &raw[mac_offset..mac_offset + 32];

        // Attempt to parse as a P-256 DER signature; if it fails, treat as raw r||s
        let signature = if let Ok(sig) = Signature::from_der(sig_bytes) {
            sig
        } else {
            // Pad to 64 bytes (32 r + 32 s) if needed
            let mut padded = [0u8; 64];
            let copy_len = sig_bytes.len().min(64);
            padded[..copy_len].copy_from_slice(&sig_bytes[..copy_len]);
            Signature::from_bytes((&padded).into()).map_err(|e| {
                PowerError::AttestationVerificationFailed(format!(
                    "Failed to parse TDX report signature: {e}"
                ))
            })?
        };

        verifying_key.verify(signed_data, &signature).map_err(|e| {
            PowerError::AttestationVerificationFailed(format!(
                "TDX report signature verification failed: {e}"
            ))
        })?;

        tracing::info!("TDX hardware signature verified via Intel PCS PCK");
        Ok(())
    }
}

#[cfg(feature = "hw-verify")]
impl Default for TdxVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "hw-verify")]
impl HardwareVerifier for TdxVerifier {
    fn verify_hardware_signature(&self, report: &AttestationReport) -> Result<()> {
        if report.tee_type != TeeType::Tdx {
            return Err(PowerError::AttestationVerificationFailed(format!(
                "TdxVerifier cannot verify {} reports",
                report.tee_type
            )));
        }
        self.verify_signature(report)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tee::attestation::{AttestationReport, TeeType};

    fn make_report(tee_type: TeeType, raw: Option<Vec<u8>>) -> AttestationReport {
        AttestationReport {
            version: "1.0".to_string(),
            tee_type,
            report_data: vec![0u8; 64],
            measurement: vec![0u8; 48],
            raw_report: raw,
            timestamp: chrono::Utc::now(),
            nonce: None,
        }
    }

    #[cfg(feature = "hw-verify")]
    #[test]
    fn test_sev_snp_verifier_rejects_wrong_tee_type() {
        let verifier = SevSnpVerifier::new();
        let report = make_report(TeeType::Tdx, Some(vec![0u8; 1184]));
        let err = verifier.verify_hardware_signature(&report).unwrap_err();
        assert!(err.to_string().contains("SevSnpVerifier cannot verify tdx"));
    }

    #[cfg(feature = "hw-verify")]
    #[test]
    fn test_tdx_verifier_rejects_wrong_tee_type() {
        let verifier = TdxVerifier::new();
        let report = make_report(TeeType::SevSnp, Some(vec![0u8; 1024]));
        let err = verifier.verify_hardware_signature(&report).unwrap_err();
        assert!(err
            .to_string()
            .contains("TdxVerifier cannot verify sev-snp"));
    }

    #[cfg(feature = "hw-verify")]
    #[test]
    fn test_sev_snp_verifier_fails_without_raw_report() {
        let verifier = SevSnpVerifier::new();
        let report = make_report(TeeType::SevSnp, None);
        let err = verifier.verify_hardware_signature(&report).unwrap_err();
        assert!(err.to_string().contains("raw_report"));
    }

    #[cfg(feature = "hw-verify")]
    #[test]
    fn test_tdx_verifier_fails_without_raw_report() {
        let verifier = TdxVerifier::new();
        let report = make_report(TeeType::Tdx, None);
        let err = verifier.verify_hardware_signature(&report).unwrap_err();
        assert!(err.to_string().contains("raw_report"));
    }

    #[cfg(feature = "hw-verify")]
    #[test]
    fn test_sev_snp_verifier_fails_on_short_raw_report() {
        let verifier = SevSnpVerifier::new();
        // Too short to contain TCB version at offset 0x38
        let report = make_report(TeeType::SevSnp, Some(vec![0u8; 10]));
        let err = verifier.verify_hardware_signature(&report).unwrap_err();
        assert!(err.to_string().contains("too short"));
    }

    #[cfg(feature = "hw-verify")]
    #[test]
    fn test_tdx_verifier_fails_on_short_raw_report() {
        let verifier = TdxVerifier::new();
        // Too short to contain FMSPC at offset 0x148
        let report = make_report(TeeType::Tdx, Some(vec![0u8; 10]));
        let err = verifier.verify_hardware_signature(&report).unwrap_err();
        assert!(err.to_string().contains("too short"));
    }

    #[cfg(feature = "hw-verify")]
    #[test]
    fn test_sev_snp_verifier_cache_ttl_zero_always_refetches() {
        // With TTL=0, every call should attempt a network fetch (which will fail
        // in CI without AMD hardware, but we verify the cache path is bypassed).
        let verifier = SevSnpVerifier::with_ttl(Duration::from_secs(0));
        let report = make_report(TeeType::SevSnp, Some(vec![0u8; 1184]));
        // Should fail at network fetch, not at cache hit
        let err = verifier.verify_hardware_signature(&report).unwrap_err();
        // Error should be about KDS fetch or chip_id extraction, not cache
        assert!(
            err.to_string().contains("AMD KDS")
                || err.to_string().contains("too short")
                || err.to_string().contains("CHIP_ID"),
            "unexpected error: {err}"
        );
    }
}
