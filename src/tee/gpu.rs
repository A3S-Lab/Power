//! NVIDIA GPU confidential-computing evidence providers.
//!
//! Power binds GPU CC evidence into the CPU TEE report by hashing NVIDIA
//! evidence/verdict bytes and placing those digests in `AttestationClaimsV2`.
//! Evidence can be supplied as configured bytes or collected live with
//! NVIDIA's `nvattest` CLI. Power can also send configured GPU evidence
//! directly to the NVIDIA NRAS REST API.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use base64::{
    engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD},
    Engine as _,
};
use rand::RngCore;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;
use uuid::Uuid;

use crate::config::{GpuAttestationConfig, GpuAttestationSource};
use crate::error::{PowerError, Result};
use crate::tee::attestation::{GpuDeviceClaim, GpuDeviceValidationClaim, GpuEvidenceClaim};

const DEFAULT_NRAS_ATTEST_GPU_URL: &str = "https://nras.attestation.nvidia.com/v4/attest/gpu";
const MAX_GPU_EVIDENCE_SOURCE_BYTES: u64 = 64 * 1024 * 1024;

/// Source of GPU CC evidence bytes.
#[derive(Debug, Clone)]
enum EvidenceSource {
    Hex { field: &'static str, value: String },
    Path { field: &'static str, path: PathBuf },
}

impl EvidenceSource {
    fn from_config(
        field_prefix: &'static str,
        hex_value: Option<&String>,
        path: Option<&PathBuf>,
    ) -> Result<Option<Self>> {
        match (hex_value, path) {
            (Some(_), Some(_)) => Err(PowerError::Config(format!(
                "{field_prefix}_hex and {field_prefix}_path are mutually exclusive"
            ))),
            (Some(value), None) => Ok(Some(Self::Hex {
                field: field_prefix,
                value: value.clone(),
            })),
            (None, Some(path)) => Ok(Some(Self::Path {
                field: field_prefix,
                path: path.clone(),
            })),
            (None, None) => Ok(None),
        }
    }

    async fn read(&self) -> Result<Vec<u8>> {
        let bytes = match self {
            Self::Hex { field, value } => decode_hex_bytes(field, value)?,
            Self::Path { field, path } => read_nonempty_file(field, path).await?,
        };

        if bytes.is_empty() {
            return Err(PowerError::Config(format!(
                "{} is empty; GPU confidential-computing evidence must contain bytes",
                self.field_name()
            )));
        }

        Ok(bytes)
    }

    fn field_name(&self) -> &'static str {
        match self {
            Self::Hex { field, .. } | Self::Path { field, .. } => field,
        }
    }
}

/// Source of a GPU evidence claim.
#[async_trait::async_trait]
pub trait GpuEvidenceProvider: Send + Sync {
    /// Build the GPU evidence claim to bind into CPU TEE `report_data`.
    async fn evidence_claim(&self) -> Result<GpuEvidenceClaim> {
        self.evidence_claim_for_nonce(None).await
    }

    /// Build the GPU evidence claim using the same verifier nonce as the CPU
    /// TEE attestation request when the provider can collect live evidence.
    async fn evidence_claim_for_nonce(&self, nonce: Option<&[u8]>) -> Result<GpuEvidenceClaim>;
}

/// Build a GPU evidence provider from config.
pub fn provider_from_config(
    config: &GpuAttestationConfig,
) -> Result<Option<Box<dyn GpuEvidenceProvider>>> {
    match config.source {
        GpuAttestationSource::Configured => ConfiguredGpuEvidenceProvider::from_config(config)
            .map(|provider| provider.map(|p| Box::new(p) as Box<dyn GpuEvidenceProvider>)),
        GpuAttestationSource::NvattestCli => Ok(Some(Box::new(
            NvattestCliGpuEvidenceProvider::from_config(config)?,
        ) as Box<dyn GpuEvidenceProvider>)),
        GpuAttestationSource::NrasRest => Ok(Some(Box::new(
            NrasRestGpuEvidenceProvider::from_config(config)?,
        ) as Box<dyn GpuEvidenceProvider>)),
    }
}

/// GPU evidence provider backed by configured file or hex byte sources.
#[derive(Debug, Clone)]
pub struct ConfiguredGpuEvidenceProvider {
    provider: String,
    evidence: EvidenceSource,
    verdict: Option<EvidenceSource>,
}

impl ConfiguredGpuEvidenceProvider {
    /// Build a provider from `gpu_attestation` config. Returns `Ok(None)` when
    /// no evidence source is configured.
    pub fn from_config(config: &GpuAttestationConfig) -> Result<Option<Self>> {
        let evidence = EvidenceSource::from_config(
            "gpu_attestation.evidence",
            config.evidence_hex.as_ref(),
            config.evidence_path.as_ref(),
        )?;
        let verdict = EvidenceSource::from_config(
            "gpu_attestation.verdict",
            config.verdict_hex.as_ref(),
            config.verdict_path.as_ref(),
        )?;

        let Some(evidence) = evidence else {
            return Ok(None);
        };

        let provider = config.provider.trim();
        if provider.is_empty() {
            return Err(PowerError::Config(
                "gpu_attestation.provider must not be empty".to_string(),
            ));
        }

        Ok(Some(Self {
            provider: provider.to_string(),
            evidence,
            verdict,
        }))
    }
}

#[async_trait::async_trait]
impl GpuEvidenceProvider for ConfiguredGpuEvidenceProvider {
    async fn evidence_claim_for_nonce(&self, nonce: Option<&[u8]>) -> Result<GpuEvidenceClaim> {
        let evidence = self.evidence.read().await?;
        let evidence_digest = sha256_bytes(&evidence);
        let mut claim = GpuEvidenceClaim::new(self.provider.clone(), evidence_digest)
            .with_evidence_format("configured-raw-bytes");

        if let Some(verdict) = &self.verdict {
            let verdict = verdict.read().await?;
            let devices = if let Some(nonce) = nonce.filter(|nonce| !nonce.is_empty()) {
                Some(validate_configured_verdict_json(
                    &verdict,
                    &hex::encode(nonce),
                )?)
            } else {
                None
            };

            claim = claim
                .with_verdict_format("configured-raw-bytes")
                .with_verdict_digest(sha256_bytes(&verdict));
            if let Some(nonce) = nonce.filter(|nonce| !nonce.is_empty()) {
                claim = claim.with_nonce(nonce);
            }
            if let Some(devices) = devices {
                claim = claim.with_devices(devices);
            }
        }

        Ok(claim)
    }
}

/// GPU evidence provider backed by NVIDIA's `nvattest` CLI.
#[derive(Debug, Clone)]
pub struct NvattestCliGpuEvidenceProvider {
    provider: String,
    command: PathBuf,
    verifier: String,
    gpu_evidence_source: String,
    gpu_architecture: Option<String>,
    nras_url: Option<String>,
    rim_url: Option<String>,
    ocsp_url: Option<String>,
    relying_party_policy_path: Option<PathBuf>,
    timeout: Duration,
}

impl NvattestCliGpuEvidenceProvider {
    pub fn from_config(config: &GpuAttestationConfig) -> Result<Self> {
        let provider = config.provider.trim();
        if provider.is_empty() {
            return Err(PowerError::Config(
                "gpu_attestation.provider must not be empty".to_string(),
            ));
        }

        let command = &config.nvattest_path;
        if command.as_os_str().is_empty() {
            return Err(PowerError::Config(
                "gpu_attestation.nvattest_path must not be empty".to_string(),
            ));
        }

        let verifier = config.nvattest_verifier.trim().to_ascii_lowercase();
        if !matches!(verifier.as_str(), "local" | "remote") {
            return Err(PowerError::Config(format!(
                "gpu_attestation.nvattest_verifier must be \"remote\" or \"local\", got {:?}",
                config.nvattest_verifier
            )));
        }

        let gpu_evidence_source = config
            .nvattest_gpu_evidence_source
            .trim()
            .to_ascii_lowercase();
        if !matches!(gpu_evidence_source.as_str(), "nvml" | "corelib") {
            return Err(PowerError::Config(format!(
                "gpu_attestation.nvattest_gpu_evidence_source must be \"nvml\" or \"corelib\", got {:?}",
                config.nvattest_gpu_evidence_source
            )));
        }
        if gpu_evidence_source == "corelib" && config.nvattest_gpu_architecture.is_none() {
            return Err(PowerError::Config(
                "gpu_attestation.nvattest_gpu_architecture is required when nvattest_gpu_evidence_source = \"corelib\"".to_string(),
            ));
        }

        if config.nvattest_timeout_secs == 0 {
            return Err(PowerError::Config(
                "gpu_attestation.nvattest_timeout_secs must be greater than zero".to_string(),
            ));
        }

        Ok(Self {
            provider: provider.to_string(),
            command: command.clone(),
            verifier,
            gpu_evidence_source,
            gpu_architecture: config.nvattest_gpu_architecture.clone(),
            nras_url: config.nras_url.clone(),
            rim_url: config.rim_url.clone(),
            ocsp_url: config.ocsp_url.clone(),
            relying_party_policy_path: config.relying_party_policy_path.clone(),
            timeout: Duration::from_secs(config.nvattest_timeout_secs),
        })
    }

    fn nonce_bytes(nonce: Option<&[u8]>) -> Result<Vec<u8>> {
        match nonce {
            Some(bytes) if bytes.len() == 32 => Ok(bytes.to_vec()),
            Some(bytes) => Err(PowerError::Config(format!(
                "gpu_attestation.source = \"nvattest-cli\" requires a 32-byte nonce for NVIDIA GPU evidence freshness (got {} bytes)",
                bytes.len()
            ))),
            _ => {
                let mut nonce = vec![0u8; 32];
                rand::thread_rng().fill_bytes(&mut nonce);
                Ok(nonce)
            }
        }
    }

    fn collect_args(&self, nonce_hex: &str) -> Vec<String> {
        let mut args = vec![
            "collect-evidence".to_string(),
            "--device".to_string(),
            "gpu".to_string(),
            "--nonce".to_string(),
            nonce_hex.to_string(),
            "--gpu-evidence-source".to_string(),
            self.gpu_evidence_source.clone(),
        ];

        if let Some(architecture) = &self.gpu_architecture {
            args.push("--gpu-architecture".to_string());
            args.push(architecture.clone());
        }

        args.push("--format".to_string());
        args.push("json".to_string());
        args
    }

    fn attest_args(&self, nonce_hex: &str, evidence_file: &Path) -> Vec<String> {
        let mut args = vec![
            "attest".to_string(),
            "--device".to_string(),
            "gpu".to_string(),
            "--verifier".to_string(),
            self.verifier.clone(),
            "--nonce".to_string(),
            nonce_hex.to_string(),
            "--gpu-evidence-source".to_string(),
            "file".to_string(),
            "--gpu-evidence-file".to_string(),
            evidence_file.display().to_string(),
        ];

        if let Some(url) = &self.nras_url {
            args.push("--nras-url".to_string());
            args.push(url.clone());
        }
        if let Some(url) = &self.rim_url {
            args.push("--rim-url".to_string());
            args.push(url.clone());
        }
        if let Some(url) = &self.ocsp_url {
            args.push("--ocsp-url".to_string());
            args.push(url.clone());
        }
        if let Some(path) = &self.relying_party_policy_path {
            args.push("--relying-party-policy".to_string());
            args.push(path.display().to_string());
        }

        args.push("--format".to_string());
        args.push("json".to_string());
        args
    }

    async fn run_nvattest(&self, args: &[String]) -> Result<Vec<u8>> {
        let mut command = Command::new(&self.command);
        command
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = timeout(self.timeout, command.output())
            .await
            .map_err(|_| {
                PowerError::Config(format!(
                    "nvattest command timed out after {}s: {} {}",
                    self.timeout.as_secs(),
                    self.command.display(),
                    args.join(" ")
                ))
            })?
            .map_err(|e| {
                PowerError::Config(format!(
                    "failed to run nvattest command '{}': {e}",
                    self.command.display()
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PowerError::Config(format!(
                "nvattest command failed with status {}: {} {}: {}",
                output.status,
                self.command.display(),
                args.join(" "),
                stderr.trim()
            )));
        }

        if output.stdout.is_empty() {
            return Err(PowerError::Config(format!(
                "nvattest command returned empty stdout: {} {}",
                self.command.display(),
                args.join(" ")
            )));
        }

        Ok(output.stdout)
    }
}

#[async_trait::async_trait]
impl GpuEvidenceProvider for NvattestCliGpuEvidenceProvider {
    async fn evidence_claim_for_nonce(&self, nonce: Option<&[u8]>) -> Result<GpuEvidenceClaim> {
        let nonce = Self::nonce_bytes(nonce)?;
        let nonce_hex = hex::encode(&nonce);

        let evidence = self.run_nvattest(&self.collect_args(&nonce_hex)).await?;
        let evidence_count = validate_nvattest_evidence_json(&evidence, &nonce_hex)?;

        let evidence_file =
            write_private_temp_file("a3s-power-nvattest-gpu-evidence", &evidence).await?;

        let verdict_result = self
            .run_nvattest(&self.attest_args(&nonce_hex, &evidence_file))
            .await;
        let cleanup_result = tokio::fs::remove_file(&evidence_file).await;
        if let Err(e) = cleanup_result {
            tracing::warn!(
                path = %evidence_file.display(),
                error = %e,
                "failed to remove temporary nvattest evidence file"
            );
        }

        let verdict = verdict_result?;
        let devices = validate_nvattest_verdict_json(&verdict, &nonce_hex)?;

        Ok(
            GpuEvidenceClaim::new(self.provider.clone(), sha256_bytes(&evidence))
                .with_evidence_format("nvidia-nvattest-evidence-json")
                .with_evidence_count(evidence_count)
                .with_nonce(&nonce)
                .with_verdict_format("nvidia-nvattest-attestation-json")
                .with_verdict_digest(sha256_bytes(&verdict))
                .with_devices(devices),
        )
    }
}

/// GPU evidence provider backed by the NVIDIA NRAS REST API.
#[derive(Debug, Clone)]
pub struct NrasRestGpuEvidenceProvider {
    provider: String,
    evidence: EvidenceSource,
    endpoint: String,
    architecture: String,
    claims_version: String,
    bearer_token_env: Option<String>,
    timeout: Duration,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct NrasRestGpuAttestationRequest {
    nonce: String,
    arch: String,
    evidence_list: Vec<serde_json::Value>,
    claims_version: String,
}

impl NrasRestGpuEvidenceProvider {
    pub fn from_config(config: &GpuAttestationConfig) -> Result<Self> {
        let provider = config.provider.trim();
        if provider.is_empty() {
            return Err(PowerError::Config(
                "gpu_attestation.provider must not be empty".to_string(),
            ));
        }

        let evidence = EvidenceSource::from_config(
            "gpu_attestation.evidence",
            config.evidence_hex.as_ref(),
            config.evidence_path.as_ref(),
        )?
        .ok_or_else(|| {
            PowerError::Config(
                "gpu_attestation.source = \"nras-rest\" requires evidence_hex or evidence_path"
                    .to_string(),
            )
        })?;

        if config.verdict_configured() {
            return Err(PowerError::Config(
                "gpu_attestation.source = \"nras-rest\" obtains the verdict from NRAS; \
                 verdict_hex/verdict_path must not be configured"
                    .to_string(),
            ));
        }

        let architecture = config
            .nras_gpu_architecture
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                PowerError::Config(
                    "gpu_attestation.nras_gpu_architecture is required when source = \"nras-rest\""
                        .to_string(),
                )
            })?;
        let architecture = architecture.to_ascii_uppercase();
        if !matches!(architecture.as_str(), "HOPPER" | "BLACKWELL") {
            return Err(PowerError::Config(format!(
                "gpu_attestation.nras_gpu_architecture must be \"HOPPER\" or \"BLACKWELL\", got {:?}",
                config.nras_gpu_architecture
            )));
        }

        let claims_version = config.nras_claims_version.trim();
        if !matches!(claims_version, "2.0" | "3.0") {
            return Err(PowerError::Config(format!(
                "gpu_attestation.nras_claims_version must be \"2.0\" or \"3.0\", got {:?}",
                config.nras_claims_version
            )));
        }

        if config.nras_timeout_secs == 0 {
            return Err(PowerError::Config(
                "gpu_attestation.nras_timeout_secs must be greater than zero".to_string(),
            ));
        }

        let endpoint = normalize_nras_rest_endpoint(config.nras_url.as_deref())?;
        let bearer_token_env = normalize_optional_env_name(
            "gpu_attestation.nras_bearer_token_env",
            config.nras_bearer_token_env.as_deref(),
        )?;
        let timeout = Duration::from_secs(config.nras_timeout_secs);
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| PowerError::Config(format!("failed to build NRAS REST client: {e}")))?;

        Ok(Self {
            provider: provider.to_string(),
            evidence,
            endpoint,
            architecture,
            claims_version: claims_version.to_string(),
            bearer_token_env,
            timeout,
            client,
        })
    }

    fn nonce_bytes(nonce: Option<&[u8]>) -> Result<Vec<u8>> {
        match nonce {
            Some(bytes) if bytes.len() == 32 => Ok(bytes.to_vec()),
            Some(bytes) => Err(PowerError::Config(format!(
                "gpu_attestation.source = \"nras-rest\" requires a 32-byte nonce for NRAS \
                 (got {} bytes)",
                bytes.len()
            ))),
            None => {
                let mut nonce = vec![0u8; 32];
                rand::thread_rng().fill_bytes(&mut nonce);
                Ok(nonce)
            }
        }
    }

    fn bearer_token(&self) -> Result<Option<String>> {
        let Some(env_name) = self.bearer_token_env.as_deref() else {
            return Ok(None);
        };
        let token = std::env::var(env_name).map_err(|e| {
            PowerError::Config(format!(
                "failed to read NRAS bearer token from environment variable {env_name}: {e}"
            ))
        })?;
        let token = token.trim();
        if token.is_empty() {
            return Err(PowerError::Config(format!(
                "NRAS bearer token environment variable {env_name} is empty"
            )));
        }
        Ok(Some(token.to_string()))
    }
}

#[async_trait::async_trait]
impl GpuEvidenceProvider for NrasRestGpuEvidenceProvider {
    async fn evidence_claim_for_nonce(&self, nonce: Option<&[u8]>) -> Result<GpuEvidenceClaim> {
        let nonce = Self::nonce_bytes(nonce)?;
        let nonce_hex = hex::encode(&nonce);

        let evidence = self.evidence.read().await?;
        let (evidence_list, evidence_count) =
            nras_rest_evidence_list_from_json(&evidence, &nonce_hex)?;

        let request = NrasRestGpuAttestationRequest {
            nonce: nonce_hex.clone(),
            arch: self.architecture.clone(),
            evidence_list,
            claims_version: self.claims_version.clone(),
        };

        let mut http_request = self.client.post(&self.endpoint).json(&request);
        if let Some(token) = self.bearer_token()? {
            http_request = http_request.bearer_auth(token);
        }

        let response = http_request.send().await.map_err(|e| {
            PowerError::Config(format!(
                "NRAS REST request to {} failed after up to {}s: {e}",
                self.endpoint,
                self.timeout.as_secs()
            ))
        })?;
        let status = response.status();
        let verdict = response.bytes().await.map_err(|e| {
            PowerError::Config(format!("failed to read NRAS REST response body: {e}"))
        })?;
        if !status.is_success() {
            return Err(PowerError::Config(format!(
                "NRAS REST request to {} failed with status {}: {}",
                self.endpoint,
                status,
                String::from_utf8_lossy(&verdict)
                    .chars()
                    .take(1024)
                    .collect::<String>()
            )));
        }
        if verdict.is_empty() {
            return Err(PowerError::Config(
                "NRAS REST response body is empty".to_string(),
            ));
        }

        let devices = validate_nras_rest_verdict_json(&verdict, &nonce_hex)?;

        Ok(
            GpuEvidenceClaim::new(self.provider.clone(), sha256_bytes(&evidence))
                .with_evidence_format("nvidia-nras-rest-evidence-json")
                .with_evidence_count(evidence_count)
                .with_nonce(&nonce)
                .with_verdict_format("nvidia-nras-rest-detached-eat-json")
                .with_verdict_digest(sha256_bytes(&verdict))
                .with_devices(devices),
        )
    }
}

/// Static provider for tests and callers that already have digest claims.
#[derive(Debug, Clone)]
pub struct StaticGpuEvidenceProvider {
    claim: GpuEvidenceClaim,
}

impl StaticGpuEvidenceProvider {
    pub fn new(claim: GpuEvidenceClaim) -> Self {
        Self { claim }
    }
}

#[async_trait::async_trait]
impl GpuEvidenceProvider for StaticGpuEvidenceProvider {
    async fn evidence_claim_for_nonce(&self, _nonce: Option<&[u8]>) -> Result<GpuEvidenceClaim> {
        Ok(self.claim.clone())
    }
}

fn decode_hex_bytes(field: &str, value: &str) -> Result<Vec<u8>> {
    let hex = value.trim().strip_prefix("sha256:").unwrap_or(value.trim());
    if !hex.len().is_multiple_of(2) {
        return Err(PowerError::Config(format!(
            "{field}_hex must be an even-length hex string"
        )));
    }
    validate_evidence_source_size(field, (hex.len() / 2) as u64)?;
    hex::decode(hex).map_err(|e| {
        PowerError::Config(format!(
            "{field}_hex must contain raw hex-encoded bytes: {e}"
        ))
    })
}

async fn read_nonempty_file(field: &str, path: &Path) -> Result<Vec<u8>> {
    let metadata = tokio::fs::metadata(path).await.map_err(|e| {
        PowerError::Config(format!(
            "failed to stat {field}_path '{}': {e}",
            path.display()
        ))
    })?;
    if metadata.len() == 0 {
        return Err(PowerError::Config(format!(
            "{field}_path '{}' is empty",
            path.display()
        )));
    }
    validate_evidence_source_size(field, metadata.len())?;

    let bytes = tokio::fs::read(path).await.map_err(|e| {
        PowerError::Config(format!(
            "failed to read {field}_path '{}': {e}",
            path.display()
        ))
    })?;
    if bytes.is_empty() {
        return Err(PowerError::Config(format!(
            "{field}_path '{}' is empty",
            path.display()
        )));
    }
    Ok(bytes)
}

fn validate_evidence_source_size(field: &str, len: u64) -> Result<()> {
    if len > MAX_GPU_EVIDENCE_SOURCE_BYTES {
        return Err(PowerError::Config(format!(
            "{field} must be at most {} bytes, got {len}",
            MAX_GPU_EVIDENCE_SOURCE_BYTES
        )));
    }
    Ok(())
}

fn normalize_optional_env_name(field: &str, value: Option<&str>) -> Result<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.trim();
    if value.is_empty() {
        return Err(PowerError::Config(format!("{field} must not be empty")));
    }
    if value.contains('=') || value.contains('\0') {
        return Err(PowerError::Config(format!(
            "{field} must not contain '=' or NUL characters"
        )));
    }
    Ok(Some(value.to_string()))
}

fn sha256_bytes(bytes: &[u8]) -> Vec<u8> {
    Sha256::digest(bytes).to_vec()
}

async fn write_private_temp_file(prefix: &str, bytes: &[u8]) -> Result<PathBuf> {
    let path = std::env::temp_dir().join(format!("{prefix}-{}.json", Uuid::new_v4()));
    let mut options = tokio::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);

    let mut file = options.open(&path).await.map_err(|e| {
        PowerError::Config(format!(
            "failed to create private temporary nvattest evidence file '{}': {e}",
            path.display()
        ))
    })?;
    if let Err(e) = file.write_all(bytes).await {
        let _ = tokio::fs::remove_file(&path).await;
        return Err(PowerError::Config(format!(
            "failed to write private temporary nvattest evidence file '{}': {e}",
            path.display()
        )));
    }
    if let Err(e) = file.flush().await {
        let _ = tokio::fs::remove_file(&path).await;
        return Err(PowerError::Config(format!(
            "failed to flush private temporary nvattest evidence file '{}': {e}",
            path.display()
        )));
    }

    Ok(path)
}

fn normalize_nras_rest_endpoint(url: Option<&str>) -> Result<String> {
    let Some(url) = url.map(str::trim).filter(|url| !url.is_empty()) else {
        return Ok(DEFAULT_NRAS_ATTEST_GPU_URL.to_string());
    };

    let mut parsed = reqwest::Url::parse(url)
        .map_err(|e| PowerError::Config(format!("gpu_attestation.nras_url is invalid: {e}")))?;
    if parsed.scheme() != "https" {
        return Err(PowerError::Config(format!(
            "gpu_attestation.nras_url must use https, got {:?}",
            parsed.scheme()
        )));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(PowerError::Config(
            "gpu_attestation.nras_url must not include embedded credentials; use nras_bearer_token_env instead"
                .to_string(),
        ));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(PowerError::Config(
            "gpu_attestation.nras_url must not include query parameters or fragments".to_string(),
        ));
    }

    let trimmed_path = parsed.path().trim_end_matches('/').to_string();
    if trimmed_path.is_empty() {
        parsed.set_path("/v4/attest/gpu");
    } else if trimmed_path.ends_with("/v4/attest/gpu") {
        parsed.set_path(&trimmed_path);
    } else if contains_versioned_path_segment(&trimmed_path) {
        return Err(PowerError::Config(format!(
            "gpu_attestation.nras_url must be a service root/base path or the full /v4/attest/gpu endpoint, got path {:?}",
            parsed.path()
        )));
    } else {
        parsed.set_path(&format!("{trimmed_path}/v4/attest/gpu"));
    }

    Ok(parsed.to_string())
}

fn contains_versioned_path_segment(path: &str) -> bool {
    path.split('/').any(|segment| {
        let Some(rest) = segment.strip_prefix('v') else {
            return false;
        };
        !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit())
    })
}

fn nras_rest_evidence_list_from_json(
    bytes: &[u8],
    nonce_hex: &str,
) -> Result<(Vec<serde_json::Value>, u32)> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| {
        PowerError::Config(format!(
            "gpu_attestation.source = \"nras-rest\" evidence is not valid JSON: {e}"
        ))
    })?;

    let evidence_values =
        if let Some(evidences) = value.get("evidences").and_then(serde_json::Value::as_array) {
            if value.get("result_code").is_some() {
                validate_nvattest_result_code("collect-evidence", &value)?;
            }
            evidences.iter().collect::<Vec<_>>()
        } else if let Some(evidence_list) = value
            .get("evidence_list")
            .and_then(serde_json::Value::as_array)
        {
            evidence_list.iter().collect::<Vec<_>>()
        } else if let Some(array) = value.as_array() {
            array.iter().collect::<Vec<_>>()
        } else if value.is_object() {
            vec![&value]
        } else {
            return Err(PowerError::Config(
                "NRAS REST evidence must be a DeviceEvidence object, an evidence_list array, \
             or an nvattest collect-evidence JSON object"
                    .to_string(),
            ));
        };

    if evidence_values.is_empty() {
        return Err(PowerError::Config(
            "NRAS REST evidence_list is empty".to_string(),
        ));
    }

    let mut evidence_list = Vec::with_capacity(evidence_values.len());
    for (index, evidence) in evidence_values.iter().enumerate() {
        evidence_list.push(normalize_nras_device_evidence(index, evidence, nonce_hex)?);
    }

    let evidence_count = u32::try_from(evidence_list.len()).map_err(|_| {
        PowerError::Config("NRAS REST evidence_list has too many entries".to_string())
    })?;

    Ok((evidence_list, evidence_count))
}

fn normalize_nras_device_evidence(
    index: usize,
    evidence: &serde_json::Value,
    nonce_hex: &str,
) -> Result<serde_json::Value> {
    let object = evidence.as_object().ok_or_else(|| {
        PowerError::Config(format!(
            "NRAS REST evidence_list[{index}] must be a JSON object"
        ))
    })?;

    if let Some(evidence_nonce) = object.get("nonce").and_then(serde_json::Value::as_str) {
        if !evidence_nonce.eq_ignore_ascii_case(nonce_hex) {
            return Err(PowerError::Config(format!(
                "NRAS REST evidence_list[{index}] nonce mismatch: evidence nonce {}, expected {}",
                evidence_nonce, nonce_hex
            )));
        }
    }

    let evidence_b64 = required_base64_string_value(index, evidence, "evidence")?;
    let certificate = required_base64_string_value(index, evidence, "certificate")?;
    let mut normalized = serde_json::Map::new();
    normalized.insert(
        "evidence".to_string(),
        serde_json::Value::String(evidence_b64),
    );
    normalized.insert(
        "certificate".to_string(),
        serde_json::Value::String(certificate),
    );
    if let Some(firmware_version) = object
        .get("firmware_version")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        normalized.insert(
            "firmware_version".to_string(),
            serde_json::Value::String(firmware_version.to_string()),
        );
    }

    Ok(serde_json::Value::Object(normalized))
}

fn required_string_value(index: usize, object: &serde_json::Value, field: &str) -> Result<String> {
    let value = object
        .get(field)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            PowerError::Config(format!(
                "NRAS REST evidence_list[{index}] must include non-empty {field}"
            ))
        })?;
    Ok(value.to_string())
}

fn required_base64_string_value(
    index: usize,
    object: &serde_json::Value,
    field: &str,
) -> Result<String> {
    let value = required_string_value(index, object, field)?;
    validate_base64_string(index, field, &value)?;
    Ok(value)
}

fn validate_base64_string(index: usize, field: &str, value: &str) -> Result<()> {
    let bytes = STANDARD
        .decode(value)
        .or_else(|_| STANDARD_NO_PAD.decode(value))
        .or_else(|_| URL_SAFE.decode(value))
        .or_else(|_| URL_SAFE_NO_PAD.decode(value))
        .map_err(|e| {
            PowerError::Config(format!(
                "NRAS REST evidence_list[{index}] {field} must be non-empty base64 or base64url: {e}"
            ))
        })?;
    if bytes.is_empty() {
        return Err(PowerError::Config(format!(
            "NRAS REST evidence_list[{index}] {field} must decode to non-empty bytes"
        )));
    }
    Ok(())
}

fn validate_nras_rest_verdict_json(bytes: &[u8], nonce_hex: &str) -> Result<Vec<GpuDeviceClaim>> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| PowerError::Config(format!("NRAS REST returned invalid JSON: {e}")))?;

    if let Some(claims) = value.get("claims").and_then(serde_json::Value::as_array) {
        if claims.is_empty() {
            return Err(PowerError::Config(
                "NRAS REST response contains an empty claims array".to_string(),
            ));
        }
        return parse_nvattest_device_claims(claims, nonce_hex);
    }

    let tokens = collect_nras_eat_tokens(&value);
    if tokens.is_empty() {
        return Err(PowerError::Config(
            "NRAS REST response does not contain claims or detached EAT tokens".to_string(),
        ));
    }

    let mut claim_values = Vec::new();
    for token in tokens {
        if let Some(payload) = decode_jwt_payload_json(&token)? {
            collect_nvidia_device_claim_values(&payload, &mut claim_values);
        }
    }

    if claim_values.is_empty() {
        tracing::warn!(
            "NRAS REST response did not expose parseable NVIDIA device claims; \
             binding verdict digest without structured device claims"
        );
        return Ok(Vec::new());
    }

    parse_nvattest_device_claims(&claim_values, nonce_hex)
}

fn validate_configured_verdict_json(bytes: &[u8], nonce_hex: &str) -> Result<Vec<GpuDeviceClaim>> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| {
        PowerError::Config(format!(
            "configured GPU verdict must be NVIDIA nvattest or NRAS JSON when a nonce is supplied: {e}"
        ))
    })?;

    let result = if value.get("result_code").is_some() || value.get("detached_eat").is_some() {
        validate_nvattest_verdict_json(bytes, nonce_hex)
    } else {
        validate_nras_rest_verdict_json(bytes, nonce_hex)
    };

    let devices = result.map_err(|e| {
        PowerError::Config(format!(
            "configured GPU verdict does not bind the supplied nonce: {e}"
        ))
    })?;
    if devices.is_empty() {
        return Err(PowerError::Config(
            "configured GPU verdict does not expose nonce-bound NVIDIA device claims".to_string(),
        ));
    }

    Ok(devices)
}

fn collect_nras_eat_tokens(value: &serde_json::Value) -> Vec<String> {
    let mut tokens = Vec::new();
    collect_nras_eat_tokens_inner(value, &mut tokens);
    tokens
}

fn collect_nras_eat_tokens_inner(value: &serde_json::Value, tokens: &mut Vec<String>) {
    match value {
        serde_json::Value::String(value) => {
            if looks_like_jwt(value) {
                tokens.push(value.clone());
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_nras_eat_tokens_inner(value, tokens);
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values() {
                collect_nras_eat_tokens_inner(value, tokens);
            }
        }
        _ => {}
    }
}

fn looks_like_jwt(value: &str) -> bool {
    value.split('.').count() == 3
}

fn decode_jwt_payload_json(token: &str) -> Result<Option<serde_json::Value>> {
    let mut parts = token.split('.');
    let _header = parts.next();
    let Some(payload) = parts.next() else {
        return Ok(None);
    };
    let Some(_signature) = parts.next() else {
        return Ok(None);
    };
    let Ok(bytes) = URL_SAFE_NO_PAD
        .decode(payload)
        .or_else(|_| URL_SAFE.decode(payload))
    else {
        return Ok(None);
    };
    let value = serde_json::from_slice(&bytes).map_err(|e| {
        PowerError::Config(format!(
            "NRAS REST returned a JWT with invalid JSON payload: {e}"
        ))
    })?;
    Ok(Some(value))
}

fn collect_nvidia_device_claim_values(value: &serde_json::Value, out: &mut Vec<serde_json::Value>) {
    match value {
        serde_json::Value::Object(object) => {
            if object.contains_key("x-nvidia-device-type") {
                out.push(value.clone());
                return;
            }
            if let Some(claims) = object.get("claims").and_then(serde_json::Value::as_array) {
                for claim in claims {
                    collect_nvidia_device_claim_values(claim, out);
                }
            }
            if let Some(submods) = object.get("submods").and_then(serde_json::Value::as_object) {
                for submod in submods.values() {
                    collect_nvidia_device_claim_values(submod, out);
                }
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_nvidia_device_claim_values(value, out);
            }
        }
        _ => {}
    }
}

fn validate_nvattest_evidence_json(bytes: &[u8], nonce_hex: &str) -> Result<u32> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| {
        PowerError::Config(format!(
            "nvattest collect-evidence returned invalid JSON: {e}"
        ))
    })?;
    validate_nvattest_result_code("collect-evidence", &value)?;

    let evidences = value
        .get("evidences")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            PowerError::Config(
                "nvattest collect-evidence JSON does not include an evidences array".to_string(),
            )
        })?;
    if evidences.is_empty() {
        return Err(PowerError::Config(
            "nvattest collect-evidence returned an empty evidences array".to_string(),
        ));
    }

    for evidence in evidences {
        let evidence_nonce = evidence
            .get("nonce")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                PowerError::Config(
                    "nvattest collect-evidence entry does not include a nonce".to_string(),
                )
            })?;
        if !evidence_nonce.eq_ignore_ascii_case(nonce_hex) {
            return Err(PowerError::Config(format!(
                "nvattest collect-evidence nonce mismatch: evidence nonce {}, expected {}",
                evidence_nonce, nonce_hex
            )));
        }
    }

    u32::try_from(evidences.len()).map_err(|_| {
        PowerError::Config(
            "nvattest collect-evidence returned too many evidence entries".to_string(),
        )
    })
}

fn validate_nvattest_verdict_json(bytes: &[u8], nonce_hex: &str) -> Result<Vec<GpuDeviceClaim>> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| PowerError::Config(format!("nvattest attest returned invalid JSON: {e}")))?;
    validate_nvattest_result_code("attest", &value)?;

    let claims = value
        .get("claims")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            PowerError::Config("nvattest attest JSON does not include a claims array".to_string())
        })?;
    if claims.is_empty() {
        return Err(PowerError::Config(
            "nvattest attest returned an empty claims array".to_string(),
        ));
    }

    if value.get("detached_eat").is_none() {
        return Err(PowerError::Config(
            "nvattest attest JSON does not include detached_eat".to_string(),
        ));
    }

    parse_nvattest_device_claims(claims, nonce_hex)
}

fn parse_nvattest_device_claims(
    claims: &[serde_json::Value],
    nonce_hex: &str,
) -> Result<Vec<GpuDeviceClaim>> {
    let mut devices = Vec::with_capacity(claims.len());
    let mut gpu_count = 0usize;

    for (index, claim) in claims.iter().enumerate() {
        let device_type = required_string_claim(index, claim, "x-nvidia-device-type")?;
        if !matches!(device_type.as_str(), "gpu" | "nvswitch") {
            return Err(PowerError::Config(format!(
                "nvattest attest claims[{index}] has unsupported x-nvidia-device-type {device_type:?}"
            )));
        }
        if device_type == "gpu" {
            gpu_count += 1;
        }

        let eat_nonce = required_string_claim(index, claim, "eat_nonce")?;
        if !eat_nonce.eq_ignore_ascii_case(nonce_hex) {
            return Err(PowerError::Config(format!(
                "nvattest attest claims[{index}] eat_nonce mismatch: claim nonce {}, expected {}",
                eat_nonce, nonce_hex
            )));
        }
        let attestation_nonce = hex::decode(&eat_nonce).map_err(|e| {
            PowerError::Config(format!(
                "nvattest attest claims[{index}] eat_nonce is not valid hex: {e}"
            ))
        })?;

        let measurements_result = optional_string_claim(claim, "measres");
        if measurements_result.as_deref() != Some("success") {
            return Err(PowerError::Config(format!(
                "nvattest attest claims[{index}] measres must be \"success\", got {:?}",
                measurements_result.as_deref()
            )));
        }
        let secure_boot = optional_bool_claim(claim, "secboot");
        let debug_status = optional_string_claim(claim, "dbgstat");
        validate_nvattest_device_security_state(
            index,
            &device_type,
            secure_boot,
            debug_status.as_deref(),
        )?;

        let validation = validation_claim_from_nvattest(index, &device_type, claim)?;
        validate_nvattest_device_validation(index, &device_type, &validation)?;

        let (claims_version, driver_version, firmware_version) = match device_type.as_str() {
            "gpu" => (
                optional_string_claim(claim, "x-nvidia-gpu-claims-version"),
                optional_string_claim(claim, "x-nvidia-gpu-driver-version"),
                optional_string_claim(claim, "x-nvidia-gpu-vbios-version"),
            ),
            "nvswitch" => (
                optional_string_claim(claim, "x-nvidia-switch-claims-version"),
                None,
                optional_string_claim(claim, "x-nvidia-switch-bios-version"),
            ),
            _ => unreachable!("unsupported device type checked above"),
        };

        let index_u32 = u32::try_from(index).map_err(|_| {
            PowerError::Config("nvattest attest returned too many device claims".to_string())
        })?;

        devices.push(GpuDeviceClaim {
            index: index_u32,
            device_type,
            attestation_nonce: Some(attestation_nonce),
            hwmodel: optional_string_claim(claim, "hwmodel"),
            ueid: optional_string_claim(claim, "ueid"),
            oemid: optional_string_claim(claim, "oemid"),
            claims_version,
            driver_version,
            firmware_version,
            measurements_result,
            secure_boot,
            debug_status,
            validation,
        });
    }

    if gpu_count == 0 {
        return Err(PowerError::Config(
            "nvattest attest returned no GPU device claims".to_string(),
        ));
    }

    Ok(devices)
}

fn validate_nvattest_device_security_state(
    index: usize,
    device_type: &str,
    secure_boot: Option<bool>,
    debug_status: Option<&str>,
) -> Result<()> {
    match secure_boot {
        Some(true) => {}
        Some(false) => {
            return Err(PowerError::Config(format!(
                "nvattest attest claims[{index}] {device_type} secboot is false"
            )));
        }
        None => {
            return Err(PowerError::Config(format!(
                "nvattest attest claims[{index}] {device_type} secboot is missing"
            )));
        }
    }

    match debug_status {
        Some(value) if value.eq_ignore_ascii_case("disabled") => Ok(()),
        Some(value) => Err(PowerError::Config(format!(
            "nvattest attest claims[{index}] {device_type} dbgstat must be \"disabled\", got {value:?}"
        ))),
        None => Err(PowerError::Config(format!(
            "nvattest attest claims[{index}] {device_type} dbgstat is missing"
        ))),
    }
}

fn validation_claim_from_nvattest(
    index: usize,
    device_type: &str,
    claim: &serde_json::Value,
) -> Result<GpuDeviceValidationClaim> {
    let (prefix, firmware_label) = match device_type {
        "gpu" => ("x-nvidia-gpu", "vbios"),
        "nvswitch" => ("x-nvidia-switch", "bios"),
        _ => {
            return Err(PowerError::Config(format!(
                "nvattest attest claims[{index}] has unsupported device type {device_type:?}"
            )));
        }
    };

    Ok(GpuDeviceValidationClaim {
        arch_check: optional_bool_claim(claim, &format!("{prefix}-arch-check")),
        attestation_report_cert_chain_fwid_match: optional_bool_claim(
            claim,
            &format!("{prefix}-attestation-report-cert-chain-fwid-match"),
        ),
        attestation_report_parsed: optional_bool_claim(
            claim,
            &format!("{prefix}-attestation-report-parsed"),
        ),
        attestation_report_nonce_match: optional_bool_claim(
            claim,
            &format!("{prefix}-attestation-report-nonce-match"),
        ),
        attestation_report_signature_verified: optional_bool_claim(
            claim,
            &format!("{prefix}-attestation-report-signature-verified"),
        ),
        driver_rim_fetched: optional_bool_claim(claim, "x-nvidia-gpu-driver-rim-fetched"),
        driver_rim_schema_validated: optional_bool_claim(
            claim,
            "x-nvidia-gpu-driver-rim-schema-validated",
        ),
        driver_rim_signature_verified: optional_bool_claim(
            claim,
            "x-nvidia-gpu-driver-rim-signature-verified",
        ),
        driver_rim_version_match: optional_bool_claim(
            claim,
            "x-nvidia-gpu-driver-rim-version-match",
        ),
        driver_rim_measurements_available: optional_bool_claim(
            claim,
            "x-nvidia-gpu-driver-rim-measurements-available",
        ),
        firmware_rim_fetched: optional_bool_claim(
            claim,
            &format!("{prefix}-{firmware_label}-rim-fetched"),
        ),
        firmware_rim_schema_validated: optional_bool_claim(
            claim,
            &format!("{prefix}-{firmware_label}-rim-schema-validated"),
        ),
        firmware_rim_signature_verified: optional_bool_claim(
            claim,
            &format!("{prefix}-{firmware_label}-rim-signature-verified"),
        ),
        firmware_rim_version_match: optional_bool_claim(
            claim,
            &format!("{prefix}-{firmware_label}-rim-version-match"),
        ),
        firmware_rim_measurements_available: optional_bool_claim(
            claim,
            &format!("{prefix}-{firmware_label}-rim-measurements-available"),
        ),
        firmware_index_no_conflict: optional_bool_claim(
            claim,
            "x-nvidia-gpu-vbios-index-no-conflict",
        ),
    })
}

fn validate_nvattest_device_validation(
    index: usize,
    device_type: &str,
    validation: &GpuDeviceValidationClaim,
) -> Result<()> {
    require_true_claim(index, device_type, "arch_check", validation.arch_check)?;
    require_true_claim(
        index,
        device_type,
        "attestation_report_cert_chain_fwid_match",
        validation.attestation_report_cert_chain_fwid_match,
    )?;
    require_true_claim(
        index,
        device_type,
        "attestation_report_parsed",
        validation.attestation_report_parsed,
    )?;
    require_true_claim(
        index,
        device_type,
        "attestation_report_nonce_match",
        validation.attestation_report_nonce_match,
    )?;
    require_true_claim(
        index,
        device_type,
        "attestation_report_signature_verified",
        validation.attestation_report_signature_verified,
    )?;
    require_true_claim(
        index,
        device_type,
        "firmware_rim_fetched",
        validation.firmware_rim_fetched,
    )?;
    require_true_claim(
        index,
        device_type,
        "firmware_rim_schema_validated",
        validation.firmware_rim_schema_validated,
    )?;
    require_true_claim(
        index,
        device_type,
        "firmware_rim_signature_verified",
        validation.firmware_rim_signature_verified,
    )?;
    require_true_claim(
        index,
        device_type,
        "firmware_rim_version_match",
        validation.firmware_rim_version_match,
    )?;
    require_true_claim(
        index,
        device_type,
        "firmware_rim_measurements_available",
        validation.firmware_rim_measurements_available,
    )?;

    if device_type == "gpu" {
        require_true_claim(
            index,
            device_type,
            "driver_rim_fetched",
            validation.driver_rim_fetched,
        )?;
        require_true_claim(
            index,
            device_type,
            "driver_rim_schema_validated",
            validation.driver_rim_schema_validated,
        )?;
        require_true_claim(
            index,
            device_type,
            "driver_rim_signature_verified",
            validation.driver_rim_signature_verified,
        )?;
        require_true_claim(
            index,
            device_type,
            "driver_rim_version_match",
            validation.driver_rim_version_match,
        )?;
        require_true_claim(
            index,
            device_type,
            "driver_rim_measurements_available",
            validation.driver_rim_measurements_available,
        )?;
        require_true_claim(
            index,
            device_type,
            "firmware_index_no_conflict",
            validation.firmware_index_no_conflict,
        )?;
    }

    Ok(())
}

fn require_true_claim(
    index: usize,
    device_type: &str,
    field: &str,
    value: Option<bool>,
) -> Result<()> {
    match value {
        Some(true) => Ok(()),
        Some(false) => Err(PowerError::Config(format!(
            "nvattest attest claims[{index}] {device_type} validation {field} is false"
        ))),
        None => Err(PowerError::Config(format!(
            "nvattest attest claims[{index}] {device_type} validation {field} is missing"
        ))),
    }
}

fn required_string_claim(index: usize, claim: &serde_json::Value, field: &str) -> Result<String> {
    optional_string_claim(claim, field).ok_or_else(|| {
        PowerError::Config(format!(
            "nvattest attest claims[{index}] does not include required field {field}"
        ))
    })
}

fn optional_string_claim(claim: &serde_json::Value, field: &str) -> Option<String> {
    match claim.get(field)? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn optional_bool_claim(claim: &serde_json::Value, field: &str) -> Option<bool> {
    match claim.get(field)? {
        serde_json::Value::Bool(value) => Some(*value),
        serde_json::Value::String(value) if value.eq_ignore_ascii_case("true") => Some(true),
        serde_json::Value::String(value) if value.eq_ignore_ascii_case("false") => Some(false),
        _ => None,
    }
}

fn validate_nvattest_result_code(command: &str, value: &serde_json::Value) -> Result<()> {
    let result_code_value = value.get("result_code").ok_or_else(|| {
        PowerError::Config(format!(
            "nvattest {command} JSON does not include result_code"
        ))
    })?;
    let result_code = match result_code_value {
        serde_json::Value::Number(n) => n.as_i64(),
        serde_json::Value::String(s) => s.parse::<i64>().ok(),
        _ => None,
    }
    .ok_or_else(|| {
        PowerError::Config(format!(
            "nvattest {command} JSON result_code must be an integer or integer string"
        ))
    })?;
    if result_code != 0 {
        let message = value
            .get("result_message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("no result_message");
        return Err(PowerError::Config(format!(
            "nvattest {command} failed with result_code {result_code}: {message}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::storage;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::sync::{Arc, Mutex};

    fn nvidia_gpu_claim(nonce: &str) -> serde_json::Value {
        serde_json::json!({
            "dbgstat": "disabled",
            "eat_nonce": nonce,
            "hwmodel": "GH100 A01 GSP BROM",
            "measres": "success",
            "oemid": "5703",
            "secboot": true,
            "ueid": "655333107904478077882826344426270545524203067314",
            "x-nvidia-device-type": "gpu",
            "x-nvidia-gpu-arch-check": true,
            "x-nvidia-gpu-attestation-report-cert-chain-fwid-match": true,
            "x-nvidia-gpu-attestation-report-nonce-match": true,
            "x-nvidia-gpu-attestation-report-parsed": true,
            "x-nvidia-gpu-attestation-report-signature-verified": true,
            "x-nvidia-gpu-claims-version": "3.0",
            "x-nvidia-gpu-driver-rim-fetched": true,
            "x-nvidia-gpu-driver-rim-measurements-available": true,
            "x-nvidia-gpu-driver-rim-schema-validated": true,
            "x-nvidia-gpu-driver-rim-signature-verified": true,
            "x-nvidia-gpu-driver-rim-version-match": true,
            "x-nvidia-gpu-driver-version": "590.12",
            "x-nvidia-gpu-vbios-index-no-conflict": true,
            "x-nvidia-gpu-vbios-rim-fetched": true,
            "x-nvidia-gpu-vbios-rim-measurements-available": true,
            "x-nvidia-gpu-vbios-rim-schema-validated": true,
            "x-nvidia-gpu-vbios-rim-signature-verified": true,
            "x-nvidia-gpu-vbios-rim-version-match": true,
            "x-nvidia-gpu-vbios-version": "96.00.A5.00.01"
        })
    }

    fn nvidia_nvswitch_claim(nonce: &str) -> serde_json::Value {
        serde_json::json!({
            "dbgstat": "disabled",
            "eat_nonce": nonce,
            "hwmodel": "NVSwitch B01",
            "measres": "success",
            "oemid": "5703",
            "secboot": true,
            "ueid": "nvswitch-ueid-0",
            "x-nvidia-device-type": "nvswitch",
            "x-nvidia-switch-arch-check": true,
            "x-nvidia-switch-attestation-report-cert-chain-fwid-match": true,
            "x-nvidia-switch-attestation-report-nonce-match": true,
            "x-nvidia-switch-attestation-report-parsed": true,
            "x-nvidia-switch-attestation-report-signature-verified": true,
            "x-nvidia-switch-bios-rim-fetched": true,
            "x-nvidia-switch-bios-rim-measurements-available": true,
            "x-nvidia-switch-bios-rim-schema-validated": true,
            "x-nvidia-switch-bios-rim-signature-verified": true,
            "x-nvidia-switch-bios-rim-version-match": true,
            "x-nvidia-switch-bios-version": "1.2.3",
            "x-nvidia-switch-claims-version": "3.0"
        })
    }

    fn nvattest_verdict_with_claim(claim: serde_json::Value) -> Vec<u8> {
        nvattest_verdict_with_claims(vec![claim])
    }

    fn nvattest_verdict_with_claims(claims: Vec<serde_json::Value>) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "claims": claims,
            "detached_eat": ["jwt", {}],
            "result_code": 0,
            "result_message": "ok"
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn configured_provider_hashes_hex_evidence_and_verdict() {
        let config = GpuAttestationConfig {
            evidence_hex: Some(hex::encode(b"gpu-evidence")),
            verdict_hex: Some(hex::encode(b"nras-verdict")),
            ..Default::default()
        };

        let provider = ConfiguredGpuEvidenceProvider::from_config(&config)
            .unwrap()
            .unwrap();
        let claim = provider.evidence_claim().await.unwrap();

        assert_eq!(claim.provider, "nvidia-nras");
        assert_eq!(
            claim.evidence_format.as_deref(),
            Some("configured-raw-bytes")
        );
        assert_eq!(
            hex::encode(claim.evidence_digest),
            storage::compute_sha256(b"gpu-evidence")
        );
        assert_eq!(
            claim.verdict_format.as_deref(),
            Some("configured-raw-bytes")
        );
        assert_eq!(
            claim.verdict_digest.map(hex::encode),
            Some(storage::compute_sha256(b"nras-verdict"))
        );
    }

    #[tokio::test]
    async fn configured_provider_with_nonce_extracts_device_claims() {
        let nonce = [0x11u8; 32];
        let nonce_hex = hex::encode(nonce);
        let verdict = serde_json::to_vec(&serde_json::json!({
            "claims": [nvidia_gpu_claim(&nonce_hex)]
        }))
        .unwrap();
        let config = GpuAttestationConfig {
            evidence_hex: Some(hex::encode(b"gpu-evidence")),
            verdict_hex: Some(hex::encode(&verdict)),
            ..Default::default()
        };

        let provider = ConfiguredGpuEvidenceProvider::from_config(&config)
            .unwrap()
            .unwrap();
        let claim = provider
            .evidence_claim_for_nonce(Some(&nonce))
            .await
            .unwrap();

        assert_eq!(claim.nonce, Some(nonce.to_vec()));
        assert_eq!(claim.devices.len(), 1);
        assert_eq!(
            claim.devices[0].hwmodel.as_deref(),
            Some("GH100 A01 GSP BROM")
        );
        assert_eq!(
            claim.devices[0].attestation_nonce.as_deref(),
            Some(&nonce[..])
        );
    }

    #[tokio::test]
    async fn configured_provider_with_nonce_rejects_stale_verdict() {
        let stale_nonce = "010203";
        let request_nonce = [0x04, 0x05, 0x06];
        let verdict = serde_json::to_vec(&serde_json::json!({
            "claims": [nvidia_gpu_claim(stale_nonce)]
        }))
        .unwrap();
        let config = GpuAttestationConfig {
            evidence_hex: Some(hex::encode(b"gpu-evidence")),
            verdict_hex: Some(hex::encode(&verdict)),
            ..Default::default()
        };

        let provider = ConfiguredGpuEvidenceProvider::from_config(&config)
            .unwrap()
            .unwrap();
        let err = provider
            .evidence_claim_for_nonce(Some(&request_nonce))
            .await
            .unwrap_err();

        assert!(err.to_string().contains("does not bind the supplied nonce"));
    }

    #[tokio::test]
    async fn configured_provider_with_nonce_rejects_unstructured_verdict() {
        let request_nonce = [0x04, 0x05, 0x06];
        let verdict = serde_json::to_vec(&serde_json::json!({
            "tokens": ["e30.eyJzdWIiOiJ4In0.sig"]
        }))
        .unwrap();
        let config = GpuAttestationConfig {
            evidence_hex: Some(hex::encode(b"gpu-evidence")),
            verdict_hex: Some(hex::encode(&verdict)),
            ..Default::default()
        };

        let provider = ConfiguredGpuEvidenceProvider::from_config(&config)
            .unwrap()
            .unwrap();
        let err = provider
            .evidence_claim_for_nonce(Some(&request_nonce))
            .await
            .unwrap_err();

        assert!(err
            .to_string()
            .contains("does not expose nonce-bound NVIDIA device claims"));
    }

    #[tokio::test]
    async fn configured_provider_hashes_file_sources() {
        let dir = tempfile::tempdir().unwrap();
        let evidence_path = dir.path().join("gpu.evidence");
        let verdict_path = dir.path().join("nras.verdict");
        std::fs::write(&evidence_path, b"gpu-evidence").unwrap();
        std::fs::write(&verdict_path, b"nras-verdict").unwrap();
        let config = GpuAttestationConfig {
            evidence_path: Some(evidence_path),
            verdict_path: Some(verdict_path),
            ..Default::default()
        };

        let provider = ConfiguredGpuEvidenceProvider::from_config(&config)
            .unwrap()
            .unwrap();
        let claim = provider.evidence_claim().await.unwrap();

        assert_eq!(
            claim.evidence_format.as_deref(),
            Some("configured-raw-bytes")
        );
        assert_eq!(
            hex::encode(claim.evidence_digest),
            storage::compute_sha256(b"gpu-evidence")
        );
        assert_eq!(
            claim.verdict_format.as_deref(),
            Some("configured-raw-bytes")
        );
        assert_eq!(
            claim.verdict_digest.map(hex::encode),
            Some(storage::compute_sha256(b"nras-verdict"))
        );
    }

    #[test]
    fn gpu_evidence_source_size_limit_allows_boundary() {
        validate_evidence_source_size("gpu_attestation.evidence", MAX_GPU_EVIDENCE_SOURCE_BYTES)
            .unwrap();
    }

    #[test]
    fn gpu_evidence_source_size_limit_rejects_oversized_len() {
        let err = validate_evidence_source_size(
            "gpu_attestation.evidence",
            MAX_GPU_EVIDENCE_SOURCE_BYTES + 1,
        )
        .unwrap_err();

        assert!(err.to_string().contains("must be at most"));
    }

    #[tokio::test]
    async fn configured_provider_rejects_oversized_file_source() {
        let dir = tempfile::tempdir().unwrap();
        let evidence_path = dir.path().join("gpu.evidence");
        let file = std::fs::File::create(&evidence_path).unwrap();
        file.set_len(MAX_GPU_EVIDENCE_SOURCE_BYTES + 1).unwrap();

        let config = GpuAttestationConfig {
            evidence_path: Some(evidence_path),
            ..Default::default()
        };
        let provider = ConfiguredGpuEvidenceProvider::from_config(&config)
            .unwrap()
            .unwrap();

        let err = provider.evidence_claim().await.unwrap_err();

        assert!(err
            .to_string()
            .contains("gpu_attestation.evidence must be at most"));
    }

    #[test]
    fn configured_provider_rejects_ambiguous_sources() {
        let config = GpuAttestationConfig {
            evidence_hex: Some("00".to_string()),
            evidence_path: Some(PathBuf::from("/tmp/evidence")),
            ..Default::default()
        };

        let err = ConfiguredGpuEvidenceProvider::from_config(&config).unwrap_err();
        assert!(err.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn configured_provider_returns_none_without_evidence() {
        let provider =
            ConfiguredGpuEvidenceProvider::from_config(&GpuAttestationConfig::default()).unwrap();
        assert!(provider.is_none());
    }

    #[test]
    fn provider_from_config_builds_nvattest_provider() {
        let config = GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            ..Default::default()
        };

        assert!(provider_from_config(&config).unwrap().is_some());
    }

    #[test]
    fn provider_from_config_builds_nras_rest_provider() {
        let config = GpuAttestationConfig {
            source: GpuAttestationSource::NrasRest,
            evidence_hex: Some(hex::encode(
                br#"{"evidence":"ZXZpZGVuY2U","certificate":"Y2VydA"}"#,
            )),
            nras_gpu_architecture: Some("HOPPER".to_string()),
            ..Default::default()
        };

        assert!(provider_from_config(&config).unwrap().is_some());
    }

    #[test]
    fn nras_rest_provider_rejects_missing_evidence() {
        let config = GpuAttestationConfig {
            source: GpuAttestationSource::NrasRest,
            nras_gpu_architecture: Some("HOPPER".to_string()),
            ..Default::default()
        };

        let err = NrasRestGpuEvidenceProvider::from_config(&config).unwrap_err();

        assert!(err.to_string().contains("requires evidence_hex"));
    }

    #[test]
    fn nras_rest_provider_rejects_missing_architecture() {
        let config = GpuAttestationConfig {
            source: GpuAttestationSource::NrasRest,
            evidence_hex: Some(hex::encode(
                br#"{"evidence":"ZXZpZGVuY2U","certificate":"Y2VydA"}"#,
            )),
            ..Default::default()
        };

        let err = NrasRestGpuEvidenceProvider::from_config(&config).unwrap_err();

        assert!(err.to_string().contains("nras_gpu_architecture"));
    }

    #[test]
    fn nras_rest_provider_trims_bearer_token_env_name() {
        let config = GpuAttestationConfig {
            source: GpuAttestationSource::NrasRest,
            evidence_hex: Some(hex::encode(
                br#"{"evidence":"ZXZpZGVuY2U","certificate":"Y2VydA"}"#,
            )),
            nras_gpu_architecture: Some("HOPPER".to_string()),
            nras_bearer_token_env: Some(" NRAS_TOKEN ".to_string()),
            ..Default::default()
        };

        let provider = NrasRestGpuEvidenceProvider::from_config(&config).unwrap();

        assert_eq!(provider.bearer_token_env.as_deref(), Some("NRAS_TOKEN"));
    }

    #[test]
    fn nras_rest_provider_rejects_empty_bearer_token_env_name() {
        let config = GpuAttestationConfig {
            source: GpuAttestationSource::NrasRest,
            evidence_hex: Some(hex::encode(
                br#"{"evidence":"ZXZpZGVuY2U","certificate":"Y2VydA"}"#,
            )),
            nras_gpu_architecture: Some("HOPPER".to_string()),
            nras_bearer_token_env: Some("  ".to_string()),
            ..Default::default()
        };

        let err = NrasRestGpuEvidenceProvider::from_config(&config).unwrap_err();

        assert!(err
            .to_string()
            .contains("gpu_attestation.nras_bearer_token_env must not be empty"));
    }

    #[test]
    fn nras_rest_provider_rejects_invalid_bearer_token_env_name() {
        let config = GpuAttestationConfig {
            source: GpuAttestationSource::NrasRest,
            evidence_hex: Some(hex::encode(
                br#"{"evidence":"ZXZpZGVuY2U","certificate":"Y2VydA"}"#,
            )),
            nras_gpu_architecture: Some("HOPPER".to_string()),
            nras_bearer_token_env: Some("NRAS=TOKEN".to_string()),
            ..Default::default()
        };

        let err = NrasRestGpuEvidenceProvider::from_config(&config).unwrap_err();

        assert!(err.to_string().contains("must not contain '='"));
    }

    #[test]
    fn nras_rest_endpoint_accepts_https_root() {
        let endpoint =
            normalize_nras_rest_endpoint(Some("https://nras.attestation.nvidia.com")).unwrap();

        assert_eq!(
            endpoint,
            "https://nras.attestation.nvidia.com/v4/attest/gpu"
        );
    }

    #[test]
    fn nras_rest_endpoint_accepts_https_base_path() {
        let endpoint = normalize_nras_rest_endpoint(Some("https://proxy.example/nras")).unwrap();

        assert_eq!(endpoint, "https://proxy.example/nras/v4/attest/gpu");
    }

    #[test]
    fn nras_rest_endpoint_accepts_full_attest_gpu_endpoint() {
        let endpoint =
            normalize_nras_rest_endpoint(Some("https://proxy.example/nras/v4/attest/gpu/"))
                .unwrap();

        assert_eq!(endpoint, "https://proxy.example/nras/v4/attest/gpu");
    }

    #[test]
    fn nras_rest_endpoint_rejects_http() {
        let err =
            normalize_nras_rest_endpoint(Some("http://nras.attestation.nvidia.com")).unwrap_err();

        assert!(err.to_string().contains("must use https"));
    }

    #[test]
    fn nras_rest_endpoint_rejects_unsupported_versioned_path() {
        let err =
            normalize_nras_rest_endpoint(Some("https://proxy.example/v3/attest/gpu")).unwrap_err();

        assert!(err
            .to_string()
            .contains("service root/base path or the full /v4/attest/gpu endpoint"));
    }

    #[test]
    fn nras_rest_endpoint_rejects_query_parameters() {
        let err = normalize_nras_rest_endpoint(Some(
            "https://nras.attestation.nvidia.com/v4/attest/gpu?token=secret",
        ))
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("must not include query parameters"));
    }

    #[test]
    fn nras_rest_endpoint_rejects_fragments() {
        let err = normalize_nras_rest_endpoint(Some(
            "https://nras.attestation.nvidia.com/v4/attest/gpu#fragment",
        ))
        .unwrap_err();

        assert!(err.to_string().contains("fragments"));
    }

    #[test]
    fn nras_rest_endpoint_rejects_embedded_credentials() {
        let err = normalize_nras_rest_endpoint(Some("https://token@nras.attestation.nvidia.com"))
            .unwrap_err();

        assert!(err
            .to_string()
            .contains("must not include embedded credentials"));
    }

    #[test]
    fn nras_rest_evidence_list_normalizes_nvattest_output() {
        let nonce = "1111111111111111111111111111111111111111111111111111111111111111";
        let evidence = format!(
            r#"{{
                "evidences": [{{
                    "nonce": "{nonce}",
                    "evidence": "ZXZpZGVuY2U",
                    "certificate": "Y2VydA",
                    "firmware_version": "96.00.81.00.0F",
                    "ignored": "field"
                }}],
                "result_code": 0
            }}"#
        );

        let (evidence_list, count) =
            nras_rest_evidence_list_from_json(evidence.as_bytes(), nonce).unwrap();

        assert_eq!(count, 1);
        assert_eq!(evidence_list[0]["evidence"], "ZXZpZGVuY2U");
        assert_eq!(evidence_list[0]["certificate"], "Y2VydA");
        assert_eq!(evidence_list[0]["firmware_version"], "96.00.81.00.0F");
        assert!(evidence_list[0].get("nonce").is_none());
        assert!(evidence_list[0].get("ignored").is_none());
    }

    #[test]
    fn nras_rest_evidence_list_rejects_nonce_mismatch() {
        let evidence = br#"{"evidences":[{"nonce":"aabb","evidence":"ZXZpZGVuY2U","certificate":"Y2VydA"}],"result_code":0}"#;

        let err = nras_rest_evidence_list_from_json(evidence, "010203").unwrap_err();

        assert!(err.to_string().contains("nonce mismatch"));
    }

    #[test]
    fn nras_rest_evidence_list_rejects_invalid_evidence_base64() {
        let nonce = "1111111111111111111111111111111111111111111111111111111111111111";
        let evidence =
            br#"{"evidences":[{"nonce":"1111111111111111111111111111111111111111111111111111111111111111","evidence":"not base64!","certificate":"Y2VydA"}],"result_code":0}"#;

        let err = nras_rest_evidence_list_from_json(evidence, nonce).unwrap_err();

        assert!(err
            .to_string()
            .contains("evidence must be non-empty base64"));
    }

    #[test]
    fn nras_rest_evidence_list_rejects_invalid_certificate_base64() {
        let nonce = "1111111111111111111111111111111111111111111111111111111111111111";
        let evidence =
            br#"{"evidences":[{"nonce":"1111111111111111111111111111111111111111111111111111111111111111","evidence":"ZXZpZGVuY2U","certificate":"not base64!"}],"result_code":0}"#;

        let err = nras_rest_evidence_list_from_json(evidence, nonce).unwrap_err();

        assert!(err
            .to_string()
            .contains("certificate must be non-empty base64"));
    }

    #[tokio::test]
    async fn nras_rest_provider_posts_evidence_and_binds_verdict() {
        let nonce = [0x11u8; 32];
        let nonce_hex = hex::encode(nonce);
        let evidence_bytes = format!(
            r#"{{
                "evidences": [{{
                    "nonce": "{nonce_hex}",
                    "evidence": "ZXZpZGVuY2U",
                    "certificate": "Y2VydA"
                }}],
                "result_code": 0
            }}"#
        )
        .into_bytes();

        let dir = tempfile::tempdir().unwrap();
        let evidence_path = dir.path().join("gpu-evidence.json");
        std::fs::write(&evidence_path, &evidence_bytes).unwrap();

        let received = Arc::new(Mutex::new(None::<serde_json::Value>));
        let handler_received = received.clone();
        let handler_nonce = nonce_hex.clone();
        let app = axum::Router::new().route(
            "/v4/attest/gpu",
            axum::routing::post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                let handler_received = handler_received.clone();
                let handler_nonce = handler_nonce.clone();
                async move {
                    {
                        let mut received = handler_received.lock().unwrap();
                        *received = Some(body);
                    }
                    axum::Json(serde_json::json!({
                        "claims": [nvidia_gpu_claim(&handler_nonce)]
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let mut provider = NrasRestGpuEvidenceProvider::from_config(&GpuAttestationConfig {
            source: GpuAttestationSource::NrasRest,
            evidence_path: Some(evidence_path),
            nras_url: Some("https://nras.attestation.nvidia.com".to_string()),
            nras_gpu_architecture: Some("hopper".to_string()),
            nras_timeout_secs: 5,
            ..Default::default()
        })
        .unwrap();
        provider.endpoint = format!("http://{addr}/v4/attest/gpu");

        let claim = provider
            .evidence_claim_for_nonce(Some(&nonce))
            .await
            .unwrap();
        server.abort();

        assert_eq!(claim.provider, "nvidia-nras");
        assert_eq!(
            claim.evidence_format.as_deref(),
            Some("nvidia-nras-rest-evidence-json")
        );
        assert_eq!(
            hex::encode(claim.evidence_digest),
            storage::compute_sha256(&evidence_bytes)
        );
        assert_eq!(
            claim.verdict_format.as_deref(),
            Some("nvidia-nras-rest-detached-eat-json")
        );
        assert_eq!(claim.evidence_count, Some(1));
        assert_eq!(claim.nonce, Some(nonce.to_vec()));
        assert_eq!(claim.devices.len(), 1);
        assert_eq!(
            claim.devices[0].hwmodel.as_deref(),
            Some("GH100 A01 GSP BROM")
        );

        let received = received.lock().unwrap().clone().unwrap();
        assert_eq!(received["nonce"], nonce_hex);
        assert_eq!(received["arch"], "HOPPER");
        assert_eq!(received["claims_version"], "3.0");
        assert_eq!(received["evidence_list"][0]["evidence"], "ZXZpZGVuY2U");
        assert_eq!(received["evidence_list"][0]["certificate"], "Y2VydA");
        assert!(received["evidence_list"][0].get("nonce").is_none());
    }

    #[test]
    fn nvattest_provider_rejects_corelib_without_architecture() {
        let config = GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            nvattest_gpu_evidence_source: "corelib".to_string(),
            ..Default::default()
        };

        let err = NvattestCliGpuEvidenceProvider::from_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("nvattest_gpu_architecture is required"));
    }

    #[tokio::test]
    async fn nvattest_provider_rejects_non_32_byte_nonce() {
        let provider = NvattestCliGpuEvidenceProvider::from_config(&GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            ..Default::default()
        })
        .unwrap();

        let err = provider
            .evidence_claim_for_nonce(Some(&[0x01, 0x02, 0x03]))
            .await
            .unwrap_err();

        assert!(err.to_string().contains("32-byte nonce"));
    }

    #[test]
    fn validate_nvattest_evidence_rejects_nonce_mismatch() {
        let evidence = br#"{"evidences":[{"nonce":"aabb"}],"result_code":0,"result_message":"ok"}"#;

        let err = validate_nvattest_evidence_json(evidence, "010203").unwrap_err();

        assert!(err.to_string().contains("nonce mismatch"));
    }

    #[test]
    fn validate_nvattest_verdict_requires_claims() {
        let verdict = br#"{"claims":[],"detached_eat":["jwt",{}],"result_code":0}"#;

        let err = validate_nvattest_verdict_json(verdict, "010203").unwrap_err();

        assert!(err.to_string().contains("empty claims array"));
    }

    #[test]
    fn validate_nvattest_verdict_extracts_device_claims() {
        let verdict = nvattest_verdict_with_claim(nvidia_gpu_claim("010203"));

        let devices = validate_nvattest_verdict_json(&verdict, "010203").unwrap();

        assert_eq!(devices.len(), 1);
        let device = &devices[0];
        assert_eq!(device.device_type, "gpu");
        assert_eq!(device.attestation_nonce.as_deref(), Some(&[1, 2, 3][..]));
        assert_eq!(device.hwmodel.as_deref(), Some("GH100 A01 GSP BROM"));
        assert_eq!(device.driver_version.as_deref(), Some("590.12"));
        assert_eq!(device.firmware_version.as_deref(), Some("96.00.A5.00.01"));
        assert_eq!(device.validation.attestation_report_nonce_match, Some(true));
    }

    #[test]
    fn validate_nvattest_verdict_rejects_secure_boot_disabled() {
        let mut claim = nvidia_gpu_claim("010203");
        claim["secboot"] = serde_json::Value::Bool(false);
        let verdict = nvattest_verdict_with_claim(claim);

        let err = validate_nvattest_verdict_json(&verdict, "010203").unwrap_err();

        assert!(err.to_string().contains("secboot is false"));
    }

    #[test]
    fn validate_nvattest_verdict_rejects_debug_enabled() {
        let mut claim = nvidia_gpu_claim("010203");
        claim["dbgstat"] = serde_json::Value::String("enabled".to_string());
        let verdict = nvattest_verdict_with_claim(claim);

        let err = validate_nvattest_verdict_json(&verdict, "010203").unwrap_err();

        assert!(err.to_string().contains("dbgstat must be \"disabled\""));
    }

    #[test]
    fn validate_nvattest_verdict_rejects_nvswitch_secure_boot_disabled() {
        let mut nvswitch = nvidia_nvswitch_claim("010203");
        nvswitch["secboot"] = serde_json::Value::Bool(false);
        let verdict = nvattest_verdict_with_claims(vec![nvidia_gpu_claim("010203"), nvswitch]);

        let err = validate_nvattest_verdict_json(&verdict, "010203").unwrap_err();

        assert!(err.to_string().contains("nvswitch secboot is false"));
    }

    #[test]
    fn validate_nvattest_verdict_rejects_nvswitch_debug_enabled() {
        let mut nvswitch = nvidia_nvswitch_claim("010203");
        nvswitch["dbgstat"] = serde_json::Value::String("enabled".to_string());
        let verdict = nvattest_verdict_with_claims(vec![nvidia_gpu_claim("010203"), nvswitch]);

        let err = validate_nvattest_verdict_json(&verdict, "010203").unwrap_err();

        assert!(err
            .to_string()
            .contains("nvswitch dbgstat must be \"disabled\""));
    }

    #[test]
    fn validate_nvattest_verdict_rejects_failed_core_validation() {
        let mut claim = nvidia_gpu_claim("010203");
        claim["x-nvidia-gpu-attestation-report-nonce-match"] = serde_json::Value::Bool(false);
        let verdict = nvattest_verdict_with_claim(claim);

        let err = validate_nvattest_verdict_json(&verdict, "010203").unwrap_err();

        assert!(err
            .to_string()
            .contains("attestation_report_nonce_match is false"));
    }

    #[test]
    fn validate_nvattest_verdict_rejects_driver_rim_schema_not_validated() {
        let mut claim = nvidia_gpu_claim("010203");
        claim["x-nvidia-gpu-driver-rim-schema-validated"] = serde_json::Value::Bool(false);
        let verdict = nvattest_verdict_with_claim(claim);

        let err = validate_nvattest_verdict_json(&verdict, "010203").unwrap_err();

        assert!(err
            .to_string()
            .contains("driver_rim_schema_validated is false"));
    }

    #[test]
    fn validate_nvattest_verdict_rejects_nvswitch_bios_rim_schema_not_validated() {
        let mut nvswitch = nvidia_nvswitch_claim("010203");
        nvswitch["x-nvidia-switch-bios-rim-schema-validated"] = serde_json::Value::Bool(false);
        let verdict = nvattest_verdict_with_claims(vec![nvidia_gpu_claim("010203"), nvswitch]);

        let err = validate_nvattest_verdict_json(&verdict, "010203").unwrap_err();

        assert!(err
            .to_string()
            .contains("firmware_rim_schema_validated is false"));
    }

    #[test]
    fn validate_nvattest_result_code_accepts_string_zero() {
        let value = serde_json::json!({
            "result_code": "0",
            "result_message": "ok"
        });

        validate_nvattest_result_code("attest", &value).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn nvattest_cli_provider_collects_and_attests_with_nonce() {
        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("nvattest");
        std::fs::write(
            &script_path,
            r#"#!/bin/sh
cmd="$1"
shift
nonce=""
evidence_file=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --nonce)
      shift
      nonce="$1"
      ;;
    --gpu-evidence-file)
      shift
      evidence_file="$1"
      ;;
  esac
  shift || true
done

case "$cmd" in
  collect-evidence)
    printf '{"evidences":[{"nonce":"%s","evidence":"abc"}],"result_code":0,"result_message":"ok"}' "$nonce"
    ;;
  attest)
    if [ ! -s "$evidence_file" ]; then
      echo "missing evidence file" >&2
      exit 2
    fi
    printf '{"claims":[{"dbgstat":"disabled","eat_nonce":"%s","hwmodel":"GH100 A01 GSP BROM","measres":"success","oemid":"5703","secboot":true,"ueid":"655333107904478077882826344426270545524203067314","x-nvidia-device-type":"gpu","x-nvidia-gpu-arch-check":true,"x-nvidia-gpu-attestation-report-cert-chain-fwid-match":true,"x-nvidia-gpu-attestation-report-nonce-match":true,"x-nvidia-gpu-attestation-report-parsed":true,"x-nvidia-gpu-attestation-report-signature-verified":true,"x-nvidia-gpu-claims-version":"3.0","x-nvidia-gpu-driver-rim-fetched":true,"x-nvidia-gpu-driver-rim-measurements-available":true,"x-nvidia-gpu-driver-rim-schema-validated":true,"x-nvidia-gpu-driver-rim-signature-verified":true,"x-nvidia-gpu-driver-rim-version-match":true,"x-nvidia-gpu-driver-version":"590.12","x-nvidia-gpu-vbios-index-no-conflict":true,"x-nvidia-gpu-vbios-rim-fetched":true,"x-nvidia-gpu-vbios-rim-measurements-available":true,"x-nvidia-gpu-vbios-rim-schema-validated":true,"x-nvidia-gpu-vbios-rim-signature-verified":true,"x-nvidia-gpu-vbios-rim-version-match":true,"x-nvidia-gpu-vbios-version":"96.00.A5.00.01"}],"detached_eat":["jwt",{}],"result_code":0,"result_message":"ok"}' "$nonce"
    ;;
  *)
    echo "unexpected command: $cmd" >&2
    exit 3
    ;;
esac
"#,
        )
        .unwrap();
        let mut perms = std::fs::metadata(&script_path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script_path, perms).unwrap();

        let provider = NvattestCliGpuEvidenceProvider::from_config(&GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            nvattest_path: script_path,
            nras_url: Some("https://nras.attestation.nvidia.com".to_string()),
            ..Default::default()
        })
        .unwrap();

        let nonce = [0x01u8; 32];
        let nonce_hex = hex::encode(nonce);
        let claim = provider
            .evidence_claim_for_nonce(Some(&nonce))
            .await
            .unwrap();

        let expected_evidence = format!(
            r#"{{"evidences":[{{"nonce":"{nonce_hex}","evidence":"abc"}}],"result_code":0,"result_message":"ok"}}"#
        );
        let expected_verdict = format!(
            r#"{{"claims":[{{"dbgstat":"disabled","eat_nonce":"{nonce_hex}","hwmodel":"GH100 A01 GSP BROM","measres":"success","oemid":"5703","secboot":true,"ueid":"655333107904478077882826344426270545524203067314","x-nvidia-device-type":"gpu","x-nvidia-gpu-arch-check":true,"x-nvidia-gpu-attestation-report-cert-chain-fwid-match":true,"x-nvidia-gpu-attestation-report-nonce-match":true,"x-nvidia-gpu-attestation-report-parsed":true,"x-nvidia-gpu-attestation-report-signature-verified":true,"x-nvidia-gpu-claims-version":"3.0","x-nvidia-gpu-driver-rim-fetched":true,"x-nvidia-gpu-driver-rim-measurements-available":true,"x-nvidia-gpu-driver-rim-schema-validated":true,"x-nvidia-gpu-driver-rim-signature-verified":true,"x-nvidia-gpu-driver-rim-version-match":true,"x-nvidia-gpu-driver-version":"590.12","x-nvidia-gpu-vbios-index-no-conflict":true,"x-nvidia-gpu-vbios-rim-fetched":true,"x-nvidia-gpu-vbios-rim-measurements-available":true,"x-nvidia-gpu-vbios-rim-schema-validated":true,"x-nvidia-gpu-vbios-rim-signature-verified":true,"x-nvidia-gpu-vbios-rim-version-match":true,"x-nvidia-gpu-vbios-version":"96.00.A5.00.01"}}],"detached_eat":["jwt",{{}}],"result_code":0,"result_message":"ok"}}"#
        );
        assert_eq!(claim.provider, "nvidia-nras");
        assert_eq!(
            claim.evidence_format.as_deref(),
            Some("nvidia-nvattest-evidence-json")
        );
        assert_eq!(claim.evidence_count, Some(1));
        assert_eq!(claim.nonce, Some(nonce.to_vec()));
        assert_eq!(
            hex::encode(claim.evidence_digest),
            storage::compute_sha256(expected_evidence.as_bytes())
        );
        assert_eq!(
            claim.verdict_format.as_deref(),
            Some("nvidia-nvattest-attestation-json")
        );
        assert_eq!(
            claim.verdict_digest.as_ref().map(hex::encode),
            Some(storage::compute_sha256(expected_verdict.as_bytes()))
        );
        assert_eq!(claim.devices.len(), 1);
        assert_eq!(claim.devices[0].device_type, "gpu");
        assert_eq!(claim.devices[0].attestation_nonce, Some(nonce.to_vec()));
        assert_eq!(
            claim.devices[0].hwmodel.as_deref(),
            Some("GH100 A01 GSP BROM")
        );
        assert_eq!(claim.devices[0].driver_version.as_deref(), Some("590.12"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn write_private_temp_file_uses_owner_only_permissions() {
        let path = write_private_temp_file("a3s-power-test-gpu-evidence", b"evidence")
            .await
            .unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        std::fs::remove_file(&path).unwrap();

        assert_eq!(mode, 0o600);
    }

    #[tokio::test]
    async fn static_provider_returns_claim() {
        let claim = GpuEvidenceClaim::new("nvidia-nras", vec![0x11; 32])
            .with_verdict_digest(vec![0x22; 32]);
        let provider = StaticGpuEvidenceProvider::new(claim.clone());
        assert_eq!(provider.evidence_claim().await.unwrap(), claim);
    }
}
