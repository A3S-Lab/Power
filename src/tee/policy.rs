use crate::error::{PowerError, Result};
use crate::tee::attestation::TeeType;

/// Policy for validating TEE attestation reports.
///
/// This is an extension point — implement this trait to add custom
/// attestation validation logic. The default implementation
/// (`DefaultTeePolicy`) checks allowed TEE types and expected measurements.
pub trait TeePolicy: Send + Sync {
    /// Validate that the detected TEE type is acceptable.
    fn validate_tee_type(&self, tee_type: TeeType) -> Result<()>;

    /// Validate that the measurement matches expected values.
    /// `measurement` is the raw measurement bytes from the attestation report.
    fn validate_measurement(&self, tee_type: TeeType, measurement: &[u8]) -> Result<()>;
}

/// Default TEE policy implementation.
///
/// Validates TEE type against an allowlist and optionally checks
/// measurements against expected values from config.
#[derive(Debug)]
pub struct DefaultTeePolicy {
    /// Allowed TEE type names. Empty = all allowed.
    allowed_types: Vec<String>,
    /// Expected measurements per TEE type (hex-encoded).
    expected_measurements: std::collections::HashMap<String, Vec<u8>>,
}

impl DefaultTeePolicy {
    /// Create a new policy from config values.
    ///
    /// `allowed_types`: list of allowed TEE type names (e.g., ["sev-snp", "tdx"]).
    ///   Empty means all types are allowed.
    /// `expected_measurements`: map of tee_type_name → hex measurement string.
    pub fn new(
        allowed_types: Vec<String>,
        expected_measurements: std::collections::HashMap<String, String>,
    ) -> Result<Self> {
        let mut parsed_measurements = std::collections::HashMap::new();
        for (tee_type, hex_measurement) in expected_measurements {
            let bytes = hex::decode(&hex_measurement).map_err(|e| {
                PowerError::Config(format!(
                    "Invalid hex measurement for TEE type '{}': {}",
                    tee_type, e
                ))
            })?;
            parsed_measurements.insert(tee_type, bytes);
        }
        Ok(Self {
            allowed_types,
            expected_measurements: parsed_measurements,
        })
    }

    /// Create a permissive policy that allows all TEE types and skips measurement checks.
    pub fn permissive() -> Self {
        Self {
            allowed_types: Vec::new(),
            expected_measurements: std::collections::HashMap::new(),
        }
    }

    /// Create a strict policy that only allows hardware TEE types (no simulated).
    pub fn strict() -> Self {
        Self {
            allowed_types: vec!["sev-snp".to_string(), "tdx".to_string()],
            expected_measurements: std::collections::HashMap::new(),
        }
    }
}

impl TeePolicy for DefaultTeePolicy {
    fn validate_tee_type(&self, tee_type: TeeType) -> Result<()> {
        if self.allowed_types.is_empty() {
            return Ok(());
        }
        let type_name = tee_type_name(tee_type);
        if self.allowed_types.iter().any(|t| t == type_name) {
            Ok(())
        } else {
            Err(PowerError::PolicyViolation(format!(
                "TEE type '{}' is not in the allowed list: [{}]",
                type_name,
                self.allowed_types.join(", ")
            )))
        }
    }

    fn validate_measurement(&self, tee_type: TeeType, measurement: &[u8]) -> Result<()> {
        let type_name = tee_type_name(tee_type);
        match self.expected_measurements.get(type_name) {
            None => Ok(()), // No expected measurement configured — skip check
            Some(expected) => {
                if measurement == expected.as_slice() {
                    Ok(())
                } else {
                    Err(PowerError::PolicyViolation(format!(
                        "TEE measurement mismatch for '{}': expected {}, got {}",
                        type_name,
                        hex::encode(expected),
                        hex::encode(measurement)
                    )))
                }
            }
        }
    }
}

/// Convert a TeeType to its canonical string name.
fn tee_type_name(tee_type: TeeType) -> &'static str {
    match tee_type {
        TeeType::SevSnp => "sev-snp",
        TeeType::Tdx => "tdx",
        TeeType::Simulated => "simulated",
        TeeType::None => "none",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_permissive_policy_allows_all() {
        let policy = DefaultTeePolicy::permissive();
        assert!(policy.validate_tee_type(TeeType::SevSnp).is_ok());
        assert!(policy.validate_tee_type(TeeType::Tdx).is_ok());
        assert!(policy.validate_tee_type(TeeType::Simulated).is_ok());
        assert!(policy.validate_tee_type(TeeType::None).is_ok());
    }

    #[test]
    fn test_strict_policy_rejects_simulated() {
        let policy = DefaultTeePolicy::strict();
        assert!(policy.validate_tee_type(TeeType::SevSnp).is_ok());
        assert!(policy.validate_tee_type(TeeType::Tdx).is_ok());
        assert!(policy.validate_tee_type(TeeType::Simulated).is_err());
        assert!(policy.validate_tee_type(TeeType::None).is_err());
    }

    #[test]
    fn test_custom_allowed_types() {
        let policy = DefaultTeePolicy::new(vec!["sev-snp".to_string()], HashMap::new()).unwrap();
        assert!(policy.validate_tee_type(TeeType::SevSnp).is_ok());
        assert!(policy.validate_tee_type(TeeType::Tdx).is_err());
        assert!(policy.validate_tee_type(TeeType::Simulated).is_err());
    }

    #[test]
    fn test_measurement_check_passes_when_matching() {
        let mut measurements = HashMap::new();
        measurements.insert("sev-snp".to_string(), "deadbeef".to_string());
        let policy = DefaultTeePolicy::new(Vec::new(), measurements).unwrap();
        assert!(policy
            .validate_measurement(TeeType::SevSnp, &[0xde, 0xad, 0xbe, 0xef])
            .is_ok());
    }

    #[test]
    fn test_measurement_check_fails_when_mismatch() {
        let mut measurements = HashMap::new();
        measurements.insert("sev-snp".to_string(), "deadbeef".to_string());
        let policy = DefaultTeePolicy::new(Vec::new(), measurements).unwrap();
        let err = policy
            .validate_measurement(TeeType::SevSnp, &[0x00, 0x01, 0x02, 0x03])
            .unwrap_err();
        assert!(err.to_string().contains("measurement mismatch"));
    }

    #[test]
    fn test_measurement_check_skipped_when_not_configured() {
        let policy = DefaultTeePolicy::permissive();
        // Any measurement passes when no expected value is configured
        assert!(policy
            .validate_measurement(TeeType::SevSnp, &[0x00; 48])
            .is_ok());
    }

    #[test]
    fn test_invalid_hex_measurement_returns_error() {
        let mut measurements = HashMap::new();
        measurements.insert("sev-snp".to_string(), "not-valid-hex!".to_string());
        let result = DefaultTeePolicy::new(Vec::new(), measurements);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid hex"));
    }

    #[test]
    fn test_policy_violation_error_message() {
        let policy = DefaultTeePolicy::strict();
        let err = policy.validate_tee_type(TeeType::Simulated).unwrap_err();
        assert!(err.to_string().contains("simulated"));
        assert!(err.to_string().contains("not in the allowed list"));
    }
}
