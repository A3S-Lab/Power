use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

/// Content identity for a local model artifact that participates in inference.
///
/// The path locates bytes on this host. Size and SHA-256 are the portable
/// identity that startup verification and attestation bind.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuxiliaryModelArtifact {
    /// Absolute path to the local artifact.
    pub path: PathBuf,
    /// Exact artifact size in bytes.
    pub size: u64,
    /// Exact SHA-256 digest as 64 hexadecimal characters.
    pub sha256: String,
}

impl AuxiliaryModelArtifact {
    /// Capture a regular local file without trusting caller-supplied identity.
    pub fn capture(path: PathBuf) -> Result<Self> {
        if !path.is_absolute() {
            return Err(PowerError::Config(format!(
                "Auxiliary model artifact path must be absolute: {}",
                path.display()
            )));
        }
        let metadata = std::fs::metadata(&path).map_err(|error| {
            PowerError::Config(format!(
                "Failed to inspect auxiliary model artifact {}: {error}",
                path.display()
            ))
        })?;
        if !metadata.is_file() {
            return Err(PowerError::Config(format!(
                "Auxiliary model artifact is not a regular file: {}",
                path.display()
            )));
        }
        if metadata.len() == 0 {
            return Err(PowerError::Config(format!(
                "Auxiliary model artifact is empty: {}",
                path.display()
            )));
        }

        Ok(Self {
            sha256: super::storage::compute_sha256_file(&path)?,
            path,
            size: metadata.len(),
        })
    }

    /// Validate the portable identity without reading the artifact.
    pub fn validate(&self, label: &str) -> Result<()> {
        if !self.path.is_absolute() {
            return Err(PowerError::Config(format!(
                "{label} path must be absolute: {}",
                self.path.display()
            )));
        }
        if self.size == 0 {
            return Err(PowerError::Config(format!(
                "{label} size must be greater than zero"
            )));
        }
        validate_sha256(&format!("{label} sha256"), &self.sha256).map_err(PowerError::Config)
    }

    /// Re-read and verify the exact regular-file identity.
    pub fn verify(&self, label: &str) -> Result<()> {
        self.validate(label)?;
        verify_regular_file_identity(label, &self.path, self.size, &self.sha256)?;
        Ok(())
    }
}

pub(crate) fn verify_regular_file_identity(
    label: &str,
    path: &Path,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<String> {
    let metadata = std::fs::metadata(path).map_err(|error| {
        PowerError::Config(format!(
            "Failed to inspect {label} {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(PowerError::Config(format!(
            "{label} is not a regular file: {}",
            path.display()
        )));
    }
    if metadata.len() != expected_size {
        return Err(PowerError::Config(format!(
            "{label} size mismatch for {}: expected {}, found {}",
            path.display(),
            expected_size,
            metadata.len()
        )));
    }
    let actual_sha256 = super::storage::compute_sha256_file(path)?;
    if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
        return Err(PowerError::Config(format!(
            "{label} SHA-256 mismatch for {}: expected {}, found {}",
            path.display(),
            expected_sha256.to_ascii_lowercase(),
            actual_sha256
        )));
    }
    Ok(actual_sha256)
}

pub(crate) fn validate_sha256(label: &str, value: &str) -> std::result::Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "{label} must contain exactly 64 hexadecimal characters"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_and_verify_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("projector.gguf");
        std::fs::write(&path, b"projector").unwrap();

        let artifact = AuxiliaryModelArtifact::capture(path).unwrap();

        assert_eq!(artifact.size, 9);
        artifact.verify("Multimodal projector").unwrap();
    }

    #[test]
    fn verification_rejects_replaced_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("adapter.gguf");
        std::fs::write(&path, b"adapter-a").unwrap();
        let artifact = AuxiliaryModelArtifact::capture(path.clone()).unwrap();
        std::fs::write(path, b"adapter-b").unwrap();

        let error = artifact.verify("LoRA adapter").unwrap_err();

        assert!(error.to_string().contains("SHA-256 mismatch"));
    }
}
