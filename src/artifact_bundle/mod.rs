//! Verified, first-use provisioning for small multi-file model bundles.
//!
//! Consumers own the immutable bundle specification. Power owns bounded
//! download, digest admission, cross-process serialization, and offline reuse.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use sha2::{Digest, Sha256};

const MAX_BUNDLE_ARTIFACTS: usize = 64;
const DEFAULT_MAX_TOTAL_BYTES: u64 = 512 * 1024 * 1024;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const RECEIPT_NAME: &str = ".a3s-power-bundle.json";
const LOCK_NAME: &str = ".a3s-power-bundle.lock";

#[derive(Clone)]
enum ArtifactSource {
    Inline(Arc<[u8]>),
    Remote(String),
}

/// One immutable file in a model artifact bundle.
#[derive(Clone)]
pub struct BundleArtifact {
    name: String,
    sha256: String,
    max_bytes: u64,
    source: ArtifactSource,
}

impl fmt::Debug for BundleArtifact {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BundleArtifact")
            .field("name", &self.name)
            .field("sha256", &self.sha256)
            .field("max_bytes", &self.max_bytes)
            .field(
                "source",
                &match &self.source {
                    ArtifactSource::Inline(_) => "inline",
                    ArtifactSource::Remote(_) => "remote",
                },
            )
            .finish()
    }
}

impl BundleArtifact {
    /// Construct an artifact whose trusted bytes are compiled into the caller.
    pub fn inline(
        name: impl Into<String>,
        bytes: impl Into<Arc<[u8]>>,
        sha256: impl Into<String>,
    ) -> Result<Self, ArtifactBundleError> {
        let bytes = bytes.into();
        let max_bytes = u64::try_from(bytes.len()).map_err(|_| {
            ArtifactBundleError::Invalid("inline artifact size does not fit in u64".to_string())
        })?;
        let actual_sha256 = sha256_bytes(&bytes);
        let artifact = Self {
            name: name.into(),
            sha256: sha256.into(),
            max_bytes,
            source: ArtifactSource::Inline(bytes),
        };
        artifact.validate()?;
        if actual_sha256 != artifact.sha256 {
            return Err(ArtifactBundleError::Integrity {
                artifact: artifact.name.clone(),
            });
        }
        Ok(artifact)
    }

    /// Construct an artifact fetched from an HTTPS URL or loopback HTTP test
    /// service. The URL is intentionally omitted from `Debug` and errors.
    pub fn remote(
        name: impl Into<String>,
        url: impl Into<String>,
        sha256: impl Into<String>,
        max_bytes: u64,
    ) -> Result<Self, ArtifactBundleError> {
        let artifact = Self {
            name: name.into(),
            sha256: sha256.into(),
            max_bytes,
            source: ArtifactSource::Remote(url.into()),
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn validate(&self) -> Result<(), ArtifactBundleError> {
        validate_artifact_name(&self.name)?;
        validate_sha256(&self.sha256)?;
        if self.max_bytes == 0 {
            return Err(ArtifactBundleError::Invalid(format!(
                "artifact '{}' must allow at least one byte",
                self.name
            )));
        }
        if let ArtifactSource::Remote(url) = &self.source {
            validate_remote_url(url)?;
        }
        Ok(())
    }
}

/// A revision-locked set of model artifacts.
#[derive(Clone, Debug)]
pub struct ArtifactBundle {
    name: String,
    revision: String,
    artifacts: Vec<BundleArtifact>,
}

impl ArtifactBundle {
    pub fn new(
        name: impl Into<String>,
        revision: impl Into<String>,
        mut artifacts: Vec<BundleArtifact>,
    ) -> Result<Self, ArtifactBundleError> {
        let name = name.into();
        let revision = revision.into();
        validate_label(&name, "bundle name")?;
        validate_label(&revision, "bundle revision")?;
        if artifacts.is_empty() || artifacts.len() > MAX_BUNDLE_ARTIFACTS {
            return Err(ArtifactBundleError::Invalid(format!(
                "bundle must contain between 1 and {MAX_BUNDLE_ARTIFACTS} artifacts"
            )));
        }
        artifacts.sort_by(|left, right| left.name.cmp(&right.name));
        let mut names = BTreeSet::new();
        for artifact in &artifacts {
            artifact.validate()?;
            if !names.insert(artifact.name.clone()) {
                return Err(ArtifactBundleError::Invalid(format!(
                    "bundle contains duplicate artifact '{}'",
                    artifact.name
                )));
            }
        }
        Ok(Self {
            name,
            revision,
            artifacts,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }
}

/// Host policy for one provisioning operation.
#[derive(Clone, Debug)]
pub struct BundleProvisionPolicy {
    destination: PathBuf,
    allow_network: bool,
    request_timeout: Duration,
    max_total_bytes: u64,
}

impl BundleProvisionPolicy {
    pub fn new(destination: impl Into<PathBuf>) -> Self {
        Self {
            destination: destination.into(),
            allow_network: true,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            max_total_bytes: DEFAULT_MAX_TOTAL_BYTES,
        }
    }

    pub fn with_network(mut self, allow_network: bool) -> Self {
        self.allow_network = allow_network;
        self
    }

    pub fn with_request_timeout(mut self, request_timeout: Duration) -> Self {
        self.request_timeout = request_timeout;
        self
    }

    pub fn with_max_total_bytes(mut self, max_total_bytes: u64) -> Self {
        self.max_total_bytes = max_total_bytes;
        self
    }
}

/// Evidence returned after every artifact has passed digest admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvisionedArtifactBundle {
    root: PathBuf,
    installed_artifacts: usize,
    reused_artifacts: usize,
}

impl ProvisionedArtifactBundle {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn installed_artifacts(&self) -> usize {
        self.installed_artifacts
    }

    pub fn reused_artifacts(&self) -> usize {
        self.reused_artifacts
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ArtifactBundleError {
    #[error("invalid artifact bundle: {0}")]
    Invalid(String),
    #[error("artifact '{artifact}' is not installed and network provisioning is disabled")]
    OfflineMissing { artifact: String },
    #[error("artifact '{artifact}' download failed: {reason}")]
    Network {
        artifact: String,
        reason: &'static str,
    },
    #[error("artifact '{artifact}' download returned HTTP status {status}")]
    HttpStatus { artifact: String, status: u16 },
    #[error("artifact '{artifact}' exceeds its {maximum_bytes}-byte admission limit")]
    TooLarge {
        artifact: String,
        maximum_bytes: u64,
    },
    #[error("artifact '{artifact}' failed SHA-256 admission")]
    Integrity { artifact: String },
    #[error("artifact bundle I/O failed while {operation}: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("artifact bundle blocking task failed")]
    BlockingTask,
}

fn validate_label(value: &str, label: &str) -> Result<(), ArtifactBundleError> {
    if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        return Err(ArtifactBundleError::Invalid(format!(
            "{label} must be 1 to 256 printable bytes"
        )));
    }
    Ok(())
}

fn validate_artifact_name(value: &str) -> Result<(), ArtifactBundleError> {
    validate_label(value, "artifact name")?;
    if value.contains(['/', '\\']) {
        return Err(ArtifactBundleError::Invalid(format!(
            "artifact name '{value}' must be one safe relative path component"
        )));
    }
    let path = Path::new(value);
    let mut components = path.components();
    if !matches!(components.next(), Some(std::path::Component::Normal(_)))
        || components.next().is_some()
        || value == RECEIPT_NAME
        || value == LOCK_NAME
    {
        return Err(ArtifactBundleError::Invalid(format!(
            "artifact name '{value}' must be one safe relative path component"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), ArtifactBundleError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ArtifactBundleError::Invalid(
            "artifact SHA-256 must be 64 lowercase hexadecimal characters".to_string(),
        ));
    }
    Ok(())
}

fn validate_remote_url(value: &str) -> Result<(), ArtifactBundleError> {
    let url = reqwest::Url::parse(value)
        .map_err(|_| ArtifactBundleError::Invalid("artifact URL must be absolute".to_string()))?;
    if !remote_url_is_allowed(&url) {
        return Err(ArtifactBundleError::Invalid(
            "artifact URL must use HTTPS (or loopback HTTP) without credentials or fragments"
                .to_string(),
        ));
    }
    Ok(())
}

fn remote_url_is_allowed(url: &reqwest::Url) -> bool {
    let loopback_http = url.scheme() == "http"
        && url.host_str().is_some_and(|host| {
            host.eq_ignore_ascii_case("localhost")
                || host
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|address| address.is_loopback())
        });
    (url.scheme() == "https" || loopback_http)
        && url.host_str().is_some()
        && url.username().is_empty()
        && url.password().is_none()
        && url.fragment().is_none()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

mod provision;
pub use provision::provision_artifact_bundle;

#[cfg(test)]
mod tests;
