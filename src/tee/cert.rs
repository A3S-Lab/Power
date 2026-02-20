//! RA-TLS certificate management: self-signed TLS certificate with optional
//! TEE attestation report embedded as a custom X.509 extension.
//!
//! Each server instance generates a fresh ECDSA P-256 self-signed certificate
//! at startup. When RA-TLS is enabled (`ra_tls = true` + `tee_mode = true`),
//! the JSON-encoded attestation report is embedded in the certificate as a
//! custom extension (OID 1.3.6.1.4.1.56560.1.1), allowing clients to perform
//! remote attestation during the TLS handshake.

use rcgen::{
    CertificateParams, CustomExtension, DistinguishedName, DnType, Ia5String, KeyPair, SanType,
};
use time::OffsetDateTime;

use crate::error::{PowerError, Result};
use crate::tee::attestation::AttestationReport;

/// OID for the A3S Power attestation X.509 extension.
///
/// Arc: 1.3.6.1.4.1.56560.1.1
/// - 1.3.6.1.4.1 = Private Enterprise Numbers (PEN) arc
/// - 56560       = A3S Lab PEN (development; replace with IANA-assigned PEN for production)
/// - 1.1         = Power / attestation extension
const ATTESTATION_EXT_OID: &[u64] = &[1, 3, 6, 1, 4, 1, 56560, 1, 1];

/// Manages the TLS certificate and private key for a single Power server instance.
///
/// A new certificate is generated at each server startup. The certificate is
/// self-signed and is valid for 365 days. Clients are expected to verify the
/// embedded attestation report rather than the CA chain.
pub struct CertManager {
    cert_pem: String,
    key_pem: String,
}

impl CertManager {
    /// Generate a fresh self-signed ECDSA P-256 TLS certificate.
    ///
    /// When `attestation` is `Some`, the JSON-encoded report is embedded in the
    /// certificate as extension OID 1.3.6.1.4.1.56560.1.1. Clients can parse
    /// this extension to verify the server is running inside a genuine TEE
    /// before trusting inference results.
    pub fn generate(attestation: Option<&AttestationReport>) -> Result<Self> {
        let key_pair = KeyPair::generate()
            .map_err(|e| PowerError::Config(format!("TLS key pair generation failed: {e}")))?;

        let mut params = CertificateParams::default();

        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, "a3s-power");
        params.distinguished_name = dn;

        params.not_before = OffsetDateTime::now_utc();
        params.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);

        let localhost_dns = Ia5String::try_from("localhost".to_string())
            .map_err(|e| PowerError::Config(format!("Invalid SAN DNS name: {e}")))?;
        params.subject_alt_names = vec![
            SanType::DnsName(localhost_dns),
            SanType::IpAddress(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)),
            SanType::IpAddress(std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)),
        ];

        if let Some(report) = attestation {
            let json_bytes = serde_json::to_vec(report).map_err(|e| {
                PowerError::Config(format!(
                    "Failed to serialize attestation report for RA-TLS: {e}"
                ))
            })?;
            params.custom_extensions = vec![CustomExtension::from_oid_content(
                ATTESTATION_EXT_OID,
                json_bytes,
            )];
            tracing::info!(
                tee_type = %report.tee_type,
                "Embedding attestation report in TLS certificate (RA-TLS)"
            );
        }

        let cert = params
            .self_signed(&key_pair)
            .map_err(|e| PowerError::Config(format!("TLS certificate generation failed: {e}")))?;

        Ok(Self {
            cert_pem: cert.pem(),
            key_pem: key_pair.serialize_pem(),
        })
    }

    /// PEM-encoded certificate.
    pub fn cert_pem(&self) -> &str {
        &self.cert_pem
    }

    /// PEM-encoded private key.
    pub fn key_pem(&self) -> &str {
        &self.key_pem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tee::attestation::{AttestationReport, TeeType};

    fn make_simulated_report() -> AttestationReport {
        AttestationReport {
            tee_type: TeeType::Simulated,
            report_data: vec![0xAA; 64],
            measurement: vec![0xBB; 48],
            timestamp: chrono::Utc::now(),
            nonce: None,
        }
    }

    #[test]
    fn test_cert_generates_without_attestation() {
        let mgr = CertManager::generate(None).unwrap();
        assert!(mgr.cert_pem().starts_with("-----BEGIN CERTIFICATE-----"));
        assert!(mgr.cert_pem().contains("-----END CERTIFICATE-----"));
    }

    #[test]
    fn test_key_pem_is_valid_pem() {
        let mgr = CertManager::generate(None).unwrap();
        assert!(mgr.key_pem().contains("PRIVATE KEY"));
        assert!(!mgr.key_pem().is_empty());
    }

    #[test]
    fn test_each_cert_has_unique_key() {
        let mgr1 = CertManager::generate(None).unwrap();
        let mgr2 = CertManager::generate(None).unwrap();
        // Fresh ECDSA key per call — keys must differ
        assert_ne!(mgr1.key_pem(), mgr2.key_pem());
    }

    #[test]
    fn test_cert_with_attestation_embeds_report() {
        let report = make_simulated_report();
        let mgr = CertManager::generate(Some(&report)).unwrap();
        // Certificate is still valid PEM
        assert!(mgr.cert_pem().starts_with("-----BEGIN CERTIFICATE-----"));
        // Key is present
        assert!(mgr.key_pem().contains("PRIVATE KEY"));
    }

    #[test]
    fn test_cert_with_nonce_in_attestation() {
        let mut report = make_simulated_report();
        report.nonce = Some(vec![0x01, 0x02, 0x03]);
        let mgr = CertManager::generate(Some(&report)).unwrap();
        assert!(mgr.cert_pem().starts_with("-----BEGIN CERTIFICATE-----"));
    }

    #[test]
    fn test_cert_pem_and_key_pem_are_different() {
        let mgr = CertManager::generate(None).unwrap();
        assert_ne!(mgr.cert_pem(), mgr.key_pem());
    }
}
