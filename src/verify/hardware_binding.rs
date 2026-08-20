use crate::error::{PowerError, Result};
use crate::tee::attestation::{AttestationReport, TeeType};
use crate::tee::tdx_report;

const SNP_REPORT_DATA_OFFSET: usize = 0x50;
const SNP_MEASUREMENT_OFFSET: usize = 0x90;
const REPORT_DATA_BYTES: usize = 64;
const MEASUREMENT_BYTES: usize = 48;

/// Bind policy-visible fields to the exact raw evidence passed to the hardware
/// verifier. A valid signature over unrelated bytes must never authenticate
/// substituted `report_data` or `measurement` JSON fields.
pub(super) fn verify_signed_fields(report: &AttestationReport) -> Result<()> {
    let raw = report.raw_report.as_deref().ok_or_else(|| {
        PowerError::AttestationVerificationFailed(
            "raw hardware evidence is required to bind signed report fields".to_string(),
        )
    })?;
    match report.tee_type {
        TeeType::SevSnp => {
            verify_raw_field(
                raw,
                SNP_REPORT_DATA_OFFSET,
                REPORT_DATA_BYTES,
                &report.report_data,
                "SEV-SNP report_data",
            )?;
            verify_raw_field(
                raw,
                SNP_MEASUREMENT_OFFSET,
                MEASUREMENT_BYTES,
                &report.measurement,
                "SEV-SNP measurement",
            )
        }
        TeeType::Tdx => {
            let fields = tdx_report::parse_tdreport(raw).ok_or_else(|| {
                PowerError::AttestationVerificationFailed(format!(
                    "unsupported TDX raw evidence length {}; Power requires a typed DCAP quote path before remote TDX verification can be enabled",
                    raw.len()
                ))
            })?;
            verify_field(&report.report_data, fields.report_data, "TDX report_data")?;
            verify_field(&report.measurement, fields.mrtd, "TDX measurement")?;
            Err(PowerError::AttestationVerificationFailed(
                "a local TDX TDREPORT is not a remotely verifiable DCAP quote; TDX hardware verification is unavailable until quote generation and QVL verification are implemented"
                    .to_string(),
            ))
        }
        TeeType::Simulated => Err(PowerError::AttestationVerificationFailed(
            "simulated evidence cannot satisfy hardware verification".to_string(),
        )),
        TeeType::None => Err(PowerError::AttestationVerificationFailed(
            "non-TEE evidence cannot satisfy hardware verification".to_string(),
        )),
    }
}

fn verify_raw_field(
    raw: &[u8],
    offset: usize,
    length: usize,
    exposed: &[u8],
    label: &str,
) -> Result<()> {
    let end = offset.checked_add(length).ok_or_else(|| {
        PowerError::AttestationVerificationFailed(format!("{label} raw-evidence range overflow"))
    })?;
    if raw.len() < end {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "raw evidence is too short for {label}: expected at least {end} bytes, got {}",
            raw.len()
        )));
    }
    verify_field(exposed, &raw[offset..end], label)
}

fn verify_field(exposed: &[u8], authenticated: &[u8], label: &str) -> Result<()> {
    if exposed.len() != authenticated.len() || !super::constant_time_eq(exposed, authenticated) {
        return Err(PowerError::AttestationVerificationFailed(format!(
            "{label} does not match the signed raw evidence"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tee::attestation::TeeType;

    fn report(tee_type: TeeType, raw_len: usize) -> AttestationReport {
        let report_data = vec![0x11; 64];
        let measurement = vec![0x22; 48];
        let mut raw = vec![0; raw_len];
        match tee_type {
            TeeType::SevSnp if raw_len >= SNP_MEASUREMENT_OFFSET + measurement.len() => {
                raw[SNP_REPORT_DATA_OFFSET..SNP_REPORT_DATA_OFFSET + report_data.len()]
                    .copy_from_slice(&report_data);
                raw[SNP_MEASUREMENT_OFFSET..SNP_MEASUREMENT_OFFSET + measurement.len()]
                    .copy_from_slice(&measurement);
            }
            TeeType::Tdx if raw_len == tdx_report::TDREPORT_BYTES => {
                raw[128..128 + report_data.len()].copy_from_slice(&report_data);
                raw[528..528 + measurement.len()].copy_from_slice(&measurement);
            }
            _ => {}
        }
        AttestationReport {
            version: "1.0".to_string(),
            tee_type,
            report_data,
            measurement,
            raw_report: Some(raw),
            timestamp: chrono::Utc::now(),
            nonce: None,
            claims: None,
        }
    }

    #[test]
    fn sev_snp_fields_match_the_signed_report() {
        verify_signed_fields(&report(TeeType::SevSnp, 0x2a0)).unwrap();
    }

    #[test]
    fn sev_snp_rejects_substituted_report_data() {
        let mut report = report(TeeType::SevSnp, 0x2a0);
        report.report_data[0] ^= 1;
        let error = verify_signed_fields(&report).unwrap_err();
        assert!(error.to_string().contains("report_data"));
    }

    #[test]
    fn sev_snp_rejects_substituted_measurement() {
        let mut report = report(TeeType::SevSnp, 0x2a0);
        report.measurement[0] ^= 1;
        let error = verify_signed_fields(&report).unwrap_err();
        assert!(error.to_string().contains("measurement"));
    }

    #[test]
    fn sev_snp_rejects_truncated_raw_evidence() {
        let error = verify_signed_fields(&report(TeeType::SevSnp, 64)).unwrap_err();
        assert!(error.to_string().contains("too short"));
    }

    #[test]
    fn tdx_tdreport_cannot_be_presented_as_a_remote_quote() {
        let error = verify_signed_fields(&report(TeeType::Tdx, 1024)).unwrap_err();
        assert!(error.to_string().contains("TDREPORT"));
        assert!(error.to_string().contains("DCAP quote"));
    }

    #[test]
    fn simulated_evidence_cannot_be_hardware_verified() {
        let error = verify_signed_fields(&report(TeeType::Simulated, 0)).unwrap_err();
        assert!(error.to_string().contains("simulated"));
    }
}
