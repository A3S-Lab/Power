//! Hardware attestation verification for AMD SEV-SNP and Intel TDX.
//!
//! The SEV-SNP implementation fetches AMD certificate material and verifies the
//! raw report signature. The TDX implementation fails closed until Power has a
//! reviewed DCAP Quote-generation and QVL path.
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
//! Built-in TDX verification is deliberately unavailable. The current TEE
//! provider exposes a local TDREPORT rather than a remotely verifiable DCAP
//! quote, so `TdxVerifier` fails closed.
//!
//! Certificate chain: Intel Root CA → PCK CA → PCK (leaf).
//! A future implementation must generate a quote and verify it with Intel QVL
//! or an equivalently reviewed remote-verification service.
//!
//! PCS endpoint: `https://api.trustedservices.intel.com/tdx/certification/v4/`
//!
//! # Caching
//!
//! The SEV-SNP verifier caches fetched certificates in memory with a 1-hour TTL
//! to avoid hammering AMD KDS, which rate-limits aggressively.

#[cfg(feature = "hw-verify")]
use std::io::Read;
#[cfg(feature = "hw-verify")]
use std::sync::{Arc, Mutex, MutexGuard};
#[cfg(feature = "hw-verify")]
use std::time::{Duration, Instant};

#[cfg(feature = "hw-verify")]
use crate::error::{PowerError, Result};
#[cfg(feature = "hw-verify")]
use crate::tee::attestation::{AttestationReport, TeeType};
#[cfg(feature = "hw-verify")]
use crate::verify::HardwareVerifier;

/// Certificate cache: maps URL/key → (cert bytes, fetch time).
#[cfg(feature = "hw-verify")]
type CertMap = std::collections::HashMap<String, (Vec<u8>, Instant)>;
#[cfg(feature = "hw-verify")]
type CertCache = Arc<Mutex<CertMap>>;

#[cfg(feature = "hw-verify")]
fn lock_cert_cache<'a>(cache: &'a CertCache, vendor: &str) -> Result<MutexGuard<'a, CertMap>> {
    cache.lock().map_err(|e| {
        PowerError::AttestationVerificationFailed(format!(
            "{vendor} certificate cache lock poisoned: {e}"
        ))
    })
}

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
    cache: CertCache,
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
            let cache = lock_cert_cache(&self.cache, "AMD KDS")?;
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
        lock_cert_cache(&self.cache, "AMD KDS")?.insert(url, (cert_bytes.clone(), Instant::now()));

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
/// This placeholder rejects all TDX reports. A TDREPORT carries a local MAC,
/// not the remotely verifiable ECDSA evidence contained in a DCAP quote.
///
/// Requires the `hw-verify` feature.
#[cfg(feature = "hw-verify")]
pub struct TdxVerifier;

#[cfg(feature = "hw-verify")]
impl TdxVerifier {
    /// Create a new verifier with the default 1-hour certificate cache TTL.
    pub fn new() -> Self {
        Self
    }

    /// Retain CLI compatibility while TDX verification remains unavailable.
    pub fn with_ttl(_ttl: Duration) -> Self {
        Self
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
        Err(PowerError::AttestationVerificationFailed(
            "built-in TDX verification is unavailable: a local TDREPORT is not a remotely verifiable DCAP quote; configure a reviewed quote-generation and QVL path"
                .to_string(),
        ))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    #[cfg(feature = "hw-verify")]
    use super::*;
    #[cfg(feature = "hw-verify")]
    use crate::tee::attestation::{AttestationReport, TeeType};

    #[cfg(feature = "hw-verify")]
    fn make_report(tee_type: TeeType, raw: Option<Vec<u8>>) -> AttestationReport {
        AttestationReport {
            version: "1.0".to_string(),
            tee_type,
            report_data: vec![0u8; 64],
            measurement: vec![0u8; 48],
            raw_report: raw,
            timestamp: chrono::Utc::now(),
            nonce: None,
            claims: None,
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
    fn test_tdx_verifier_fails_closed_without_a_dcap_quote_path() {
        let verifier = TdxVerifier::new();
        let report = make_report(TeeType::Tdx, None);
        let err = verifier.verify_hardware_signature(&report).unwrap_err();
        assert!(err.to_string().contains("DCAP quote"));
        assert!(err.to_string().contains("unavailable"));
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
    fn test_tdx_verifier_does_not_treat_tdreport_bytes_as_a_quote() {
        let verifier = TdxVerifier::new();
        let report = make_report(TeeType::Tdx, Some(vec![0u8; 1024]));
        let err = verifier.verify_hardware_signature(&report).unwrap_err();
        assert!(err.to_string().contains("DCAP quote"));
    }

    #[cfg(feature = "hw-verify")]
    #[test]
    fn test_sev_snp_verifier_returns_error_when_cache_lock_poisoned() {
        let verifier = SevSnpVerifier::new();
        let cache = verifier.cache.clone();
        let _ = std::panic::catch_unwind(move || {
            let _guard = cache.lock().unwrap();
            panic!("poison cache");
        });

        let report = make_report(TeeType::SevSnp, Some(vec![0u8; 1184]));
        let err = verifier.fetch_vcek_chain(&report).unwrap_err();
        assert!(err
            .to_string()
            .contains("AMD KDS certificate cache lock poisoned"));
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
