use std::collections::HashMap;
use std::fmt::Display;
use std::path::PathBuf;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::dirs;
use crate::error::{PowerError, Result};

mod acl;
mod behavior;

/// Attestation policy mode for TEE deployments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TeePolicyMode {
    /// Development mode permits simulated TEE evidence and skipped production checks.
    Development,
    /// Strict mode requires hardware TEE evidence, launch measurement pins, and
    /// pinned local model integrity policy.
    #[default]
    Strict,
    /// Strict mode plus NVIDIA GPU confidential-computing evidence binding.
    GpuConfidential,
}

/// Source used to produce NVIDIA GPU confidential-computing evidence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuAttestationSource {
    /// Consume configured raw evidence/verdict bytes from file or hex fields.
    #[default]
    Configured,
    /// Invoke NVIDIA's `nvattest` CLI to collect evidence and request an NRAS
    /// verdict for each attestation request.
    NvattestCli,
    /// Send configured GPU evidence directly to the NVIDIA NRAS REST API.
    NrasRest,
}

impl Display for GpuAttestationSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Configured => write!(f, "configured"),
            Self::NvattestCli => write!(f, "nvattest-cli"),
            Self::NrasRest => write!(f, "nras-rest"),
        }
    }
}

impl FromStr for GpuAttestationSource {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "configured" | "static" | "file" | "hex" => Ok(Self::Configured),
            "nvattest-cli" | "nvattest" | "nvidia-nvattest-cli" => Ok(Self::NvattestCli),
            "nras-rest" | "nvidia-nras-rest" | "nras" => Ok(Self::NrasRest),
            other => Err(format!(
                "unknown GPU attestation source '{other}', expected configured, nvattest-cli, or nras-rest"
            )),
        }
    }
}

impl Display for TeePolicyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Development => write!(f, "development"),
            Self::Strict => write!(f, "strict"),
            Self::GpuConfidential => write!(f, "gpu-confidential"),
        }
    }
}

impl FromStr for TeePolicyMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "development" | "dev" => Ok(Self::Development),
            "strict" | "production" | "prod" => Ok(Self::Strict),
            "gpu-confidential" | "gpu_confidential" | "gpu-confidential-computing" => {
                Ok(Self::GpuConfidential)
            }
            other => Err(format!(
                "unknown TEE policy mode '{other}', expected development, strict, or gpu-confidential"
            )),
        }
    }
}

/// GPU acceleration settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuConfig {
    /// Number of layers to offload to GPU. 0 = CPU only, -1 = all layers.
    #[serde(default)]
    pub gpu_layers: i32,

    /// Index of the primary GPU to use (default: 0).
    #[serde(default)]
    pub main_gpu: i32,

    /// Proportion of work to distribute across multiple GPUs.
    /// Each value is a float representing the fraction of work for that GPU.
    /// Example: `[0.5, 0.5]` splits evenly across 2 GPUs.
    /// Empty means use a single GPU (default behavior).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tensor_split: Vec<f32>,
}

fn default_gpu_attestation_provider() -> String {
    "nvidia-nras".to_string()
}

fn default_nvattest_path() -> PathBuf {
    PathBuf::from("nvattest")
}

fn default_nvattest_verifier() -> String {
    "remote".to_string()
}

fn default_nvattest_gpu_evidence_source() -> String {
    "nvml".to_string()
}

fn default_nvattest_timeout_secs() -> u64 {
    30
}

fn default_proxy_effective_prompt_digest_path() -> String {
    "/v1/chat/effective-prompt-digest".to_string()
}

fn default_nras_claims_version() -> String {
    "3.0".to_string()
}

fn default_nras_timeout_secs() -> u64 {
    30
}

/// NVIDIA GPU confidential-computing attestation evidence settings.
///
/// Power can consume evidence/verdict bytes produced by an external NVIDIA GPU
/// CC verifier, or invoke NVIDIA's `nvattest` CLI to collect live evidence and
/// request an NRAS verdict. In both modes, Power hashes the raw evidence and
/// verdict bytes and binds those digests into CPU TEE attestation claims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuAttestationConfig {
    /// Evidence source used by `gpu-confidential` mode.
    #[serde(default)]
    pub source: GpuAttestationSource,

    /// Evidence provider label emitted in the attestation claim.
    #[serde(default = "default_gpu_attestation_provider")]
    pub provider: String,

    /// Raw GPU CC evidence bytes, hex-encoded. Mutually exclusive with
    /// `evidence_path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_hex: Option<String>,

    /// Path to raw GPU CC evidence bytes. Mutually exclusive with
    /// `evidence_hex`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_path: Option<PathBuf>,

    /// Raw NVIDIA NRAS verdict bytes, hex-encoded. Mutually exclusive with
    /// `verdict_path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict_hex: Option<String>,

    /// Path to raw NVIDIA NRAS verdict bytes. Mutually exclusive with
    /// `verdict_hex`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict_path: Option<PathBuf>,

    /// Path to NVIDIA's `nvattest` CLI when `source = "nvattest-cli"`.
    #[serde(default = "default_nvattest_path")]
    pub nvattest_path: PathBuf,

    /// `nvattest attest --verifier` value. Production deployments should use
    /// `remote` so NRAS signs/verifies the GPU evidence.
    #[serde(default = "default_nvattest_verifier")]
    pub nvattest_verifier: String,

    /// Live GPU evidence source passed to `nvattest collect-evidence`.
    /// `nvml` is the Confidential Computing path for H100-class GPUs.
    #[serde(default = "default_nvattest_gpu_evidence_source")]
    pub nvattest_gpu_evidence_source: String,

    /// Required by `nvattest` only for `corelib` evidence sources.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nvattest_gpu_architecture: Option<String>,

    /// Optional NRAS service URL for `nvattest attest --nras-url`.
    /// When `source = "nras-rest"`, this may be either the API root or the full
    /// `/v4/attest/gpu` endpoint. Defaults to NVIDIA's public NRAS endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nras_url: Option<String>,

    /// GPU architecture passed to NVIDIA NRAS REST requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nras_gpu_architecture: Option<String>,

    /// NVIDIA NRAS REST claims version. Supported by NRAS v4: `2.0` and `3.0`.
    #[serde(default = "default_nras_claims_version")]
    pub nras_claims_version: String,

    /// Environment variable containing an optional bearer token for NRAS REST.
    /// Must be a portable ASCII name: `[A-Za-z_][A-Za-z0-9_]*`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nras_bearer_token_env: Option<String>,

    /// Maximum time allowed for each NRAS REST request.
    #[serde(default = "default_nras_timeout_secs")]
    pub nras_timeout_secs: u64,

    /// Optional RIM service URL for `nvattest attest --rim-url`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rim_url: Option<String>,

    /// Optional OCSP service URL for `nvattest attest --ocsp-url`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocsp_url: Option<String>,

    /// Optional relying-party policy file for `nvattest attest`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relying_party_policy_path: Option<PathBuf>,

    /// Maximum time allowed for each `nvattest` command.
    #[serde(default = "default_nvattest_timeout_secs")]
    pub nvattest_timeout_secs: u64,
}

impl Default for GpuAttestationConfig {
    fn default() -> Self {
        Self {
            source: GpuAttestationSource::default(),
            provider: default_gpu_attestation_provider(),
            evidence_hex: None,
            evidence_path: None,
            verdict_hex: None,
            verdict_path: None,
            nvattest_path: default_nvattest_path(),
            nvattest_verifier: default_nvattest_verifier(),
            nvattest_gpu_evidence_source: default_nvattest_gpu_evidence_source(),
            nvattest_gpu_architecture: None,
            nras_url: None,
            nras_gpu_architecture: None,
            nras_claims_version: default_nras_claims_version(),
            nras_bearer_token_env: None,
            nras_timeout_secs: default_nras_timeout_secs(),
            rim_url: None,
            ocsp_url: None,
            relying_party_policy_path: None,
            nvattest_timeout_secs: default_nvattest_timeout_secs(),
        }
    }
}

impl GpuAttestationConfig {
    pub fn evidence_configured(&self) -> bool {
        self.evidence_hex.is_some() || self.evidence_path.is_some()
    }

    pub fn verdict_configured(&self) -> bool {
        self.verdict_hex.is_some() || self.verdict_path.is_some()
    }
}

/// User-configurable settings for the Power server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerConfig {
    /// Host address for the HTTP server (default: 127.0.0.1)
    #[serde(default = "default_host")]
    pub host: String,

    /// Port for the HTTP server (default: 11434)
    #[serde(default = "default_port")]
    pub port: u16,

    /// Base directory for model storage
    #[serde(default = "dirs::power_home")]
    pub data_dir: PathBuf,

    /// Maximum number of models to keep loaded in memory
    #[serde(default = "default_max_loaded_models")]
    pub max_loaded_models: usize,

    /// GPU acceleration settings
    #[serde(default)]
    pub gpu: GpuConfig,

    /// NVIDIA GPU confidential-computing evidence binding settings.
    #[serde(default)]
    pub gpu_attestation: GpuAttestationConfig,

    /// Speculative-decoding mode for the picolm backend: "off", "prompt-lookup",
    /// or "ngram-context" (DSpark-like self-speculation). Default: "prompt-lookup".
    #[serde(default = "default_spec_mode")]
    pub spec_mode: String,

    /// Default model keep-alive duration (e.g. "5m", "1h", "0", "-1").
    /// "0" = unload immediately after request, "-1" = never unload.
    /// Default: "5m".
    #[serde(default = "default_keep_alive")]
    pub keep_alive: String,

    /// Lock model weights in memory to prevent swapping (default: false).
    #[serde(default)]
    pub use_mlock: bool,

    /// Number of threads for generation (default: auto-detect).
    #[serde(default)]
    pub num_thread: Option<u32>,

    /// Enable flash attention globally (default: false).
    #[serde(default)]
    pub flash_attention: bool,

    /// Number of parallel request slots (concurrent inference). Default: 1.
    #[serde(default = "default_num_parallel")]
    pub num_parallel: usize,

    /// Enable TEE mode: model integrity verification, log redaction,
    /// memory zeroing after inference (default: false).
    #[serde(default)]
    pub tee_mode: bool,

    /// Attestation policy mode. Defaults to strict so `tee_mode = true` fails
    /// closed unless development mode is explicitly requested.
    #[serde(default)]
    pub tee_policy_mode: TeePolicyMode,

    /// Redact inference content from logs (default: true when tee_mode is enabled).
    #[serde(default)]
    pub redact_logs: bool,

    /// Expected SHA-256 hashes for model integrity verification.
    /// Key: model name, Value: expected SHA-256 hash.
    /// Only checked when tee_mode is enabled.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub model_hashes: HashMap<String, String>,

    /// Source of the AES-256-GCM key for encrypted model loading.
    /// If set, models with `.enc` extension are decrypted at load time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_key_source: Option<crate::tee::encrypted_model::KeySource>,

    /// Port for the TLS (HTTPS) server. When set, a TLS server is started
    /// alongside the plain HTTP server. Requires the `tls` feature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tls_port: Option<u16>,

    /// Additional Subject Alternative Names for the TLS certificate.
    /// Each entry is a DNS name (e.g. "myserver.internal") or IP address (e.g. "10.0.0.1").
    /// "localhost" and 127.0.0.1 are always included.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tls_sans: Vec<String>,

    /// Embed a TEE attestation report in the TLS certificate (RA-TLS).
    /// Requires `tls_port` to be set and `tee_mode` to be enabled.
    #[serde(default)]
    pub ra_tls: bool,

    /// Vsock port for guest-host communication inside a3s-box MicroVMs.
    /// When set, a vsock server is started alongside the plain HTTP server.
    /// Requires the `vsock` feature and Linux with AF_VSOCK kernel support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vsock_port: Option<u32>,

    /// API keys for authentication. When non-empty, all /v1/* endpoints
    /// require a valid `Authorization: Bearer <key>` header.
    /// Keys are SHA-256 hashes of the actual tokens for secure storage.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub api_keys: Vec<String>,

    // --- TEE Policy ---
    /// Allowed TEE types. Default: all types allowed.
    /// Set to ["sev-snp", "tdx"] to reject simulated TEE in production.
    /// Overridden by A3S_POWER_TEE_STRICT=1 (removes "simulated").
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_tee_types: Vec<String>,

    /// Expected 48-byte launch measurements per TEE type (hex-encoded).
    /// Strict and GPU-confidential policy require a pin for the detected TEE
    /// type. Keys use canonical TEE names such as "sev-snp" or "tdx".
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub expected_measurements: HashMap<String, String>,

    // --- Audit Logging ---
    /// Enable structured audit logging. Default: false.
    #[serde(default)]
    pub audit_log: bool,

    /// Path to audit log file. Default: $A3S_POWER_HOME/audit.jsonl.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audit_log_path: Option<std::path::PathBuf>,

    /// Encrypt audit log entries at rest with AES-256-GCM. Default: false.
    ///
    /// When enabled, each log line is encrypted with a fresh nonce and written
    /// as `<nonce_hex>.<base64_ciphertext>`. Requires `audit_key_source` to be set.
    #[serde(default)]
    pub audit_log_encrypt: bool,

    /// Key source for audit log encryption. Required when `audit_log_encrypt` is true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audit_key_source: Option<crate::tee::encrypted_model::KeySource>,

    // --- Model Signing ---
    /// Ed25519 **verifying** (public) key for model signature verification (hex-encoded, 32 bytes).
    /// Despite the field name, this is the public key used to *verify* signatures — not a private
    /// signing key. When set, all models must have a corresponding `.sig` file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_signing_key: Option<String>,

    // --- Key Provider ---
    /// Key provider type. "static" (default) uses model_key_source.
    /// "rotating" uses key_rotation_sources for zero-downtime key rotation.
    #[serde(default = "default_key_provider")]
    pub key_provider: String,

    /// For the rotating key provider: list of key sources in rotation order.
    /// The first source is active initially; rotate_key() advances to the next.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub key_rotation_sources: Vec<crate::tee::encrypted_model::KeySource>,

    // --- In-Memory Decryption ---
    /// Decrypt encrypted models entirely in RAM (mlock) instead of writing a temp file.
    ///
    /// Backends must explicitly support locked in-memory plaintext loading.
    /// Unsupported backends fail closed before load.
    #[serde(default)]
    pub in_memory_decrypt: bool,

    /// Decrypt encrypted models through a chunked plaintext access primitive.
    ///
    /// Backends must explicitly support layer-streaming decrypted plaintext
    /// loading. Unsupported backends fail closed before load.
    #[serde(default)]
    pub streaming_decrypt: bool,

    // --- Token Metrics Side-Channel Mitigation ---
    /// Round token counts in responses to the nearest 10.
    /// Prevents exact token-count side-channel inference. Default: false.
    #[serde(default)]
    pub suppress_token_metrics: bool,

    // --- Rate Limiting ---
    /// Max requests per second for /v1/* endpoints. 0 = unlimited (default).
    #[serde(default)]
    pub rate_limit_rps: u64,

    /// Max concurrent requests for /v1/* endpoints. 0 = unlimited (default).
    #[serde(default)]
    pub max_concurrent_requests: u64,

    /// Remote models proxied to upstream OpenAI-compatible servers (vLLM, TGI,
    /// SGLang, ...). Maps served model name → upstream base URL (e.g.
    /// `"llama-70b" = "http://vllm:8000"`). Power fronts these with its routing,
    /// auth, rate-limit and log-redaction layers. Proxied inference runs on the
    /// upstream (outside any TEE) — no hardware attestation covers its content.
    #[serde(default)]
    pub proxy_upstreams: HashMap<String, String>,

    /// Ask proxy upstreams for the exact rendered chat prompt digest before
    /// inference, using `proxy_effective_prompt_digest_path`.
    #[serde(default)]
    pub proxy_effective_prompt_digest: bool,

    /// Require proxy upstreams to provide an effective prompt digest. When true,
    /// missing/unsupported upstream digest endpoints fail the request closed.
    #[serde(default)]
    pub proxy_effective_prompt_digest_required: bool,

    /// Upstream endpoint path for proxy effective prompt digest requests.
    #[serde(default = "default_proxy_effective_prompt_digest_path")]
    pub proxy_effective_prompt_digest_path: String,

    // --- Timing Side-Channel Mitigation ---
    /// Minimum response time in milliseconds for all inference requests.
    ///
    /// When set, responses are padded to at least this duration by delaying
    /// the first token. A ±20% jitter is applied to prevent statistical
    /// timing attacks. Default: None (disabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing_padding_ms: Option<u64>,
}

fn default_key_provider() -> String {
    "static".to_string()
}

fn default_keep_alive() -> String {
    "5m".to_string()
}

fn default_spec_mode() -> String {
    "prompt-lookup".to_string()
}

fn is_valid_spec_mode(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "off"
            | "none"
            | "false"
            | "prompt-lookup"
            | "prompt_lookup"
            | "lookup"
            | "true"
            | "ngram-context"
            | "ngram_context"
            | "context"
            | "dspark"
    )
}

fn validate_model_signing_key(value: &str) -> std::result::Result<(), String> {
    if value.len() != 64 {
        return Err(format!(
            "model_signing_key must be a 64-character hex-encoded Ed25519 public key (32 bytes), got {} characters",
            value.len()
        ));
    }

    let bytes = hex::decode(value).map_err(|e| {
        format!("model_signing_key must be hex-encoded Ed25519 public key bytes: {e}")
    })?;
    let key_bytes: [u8; 32] = bytes.try_into().map_err(|bytes: Vec<u8>| {
        format!(
            "model_signing_key must decode to 32 bytes, got {} bytes",
            bytes.len()
        )
    })?;
    ed25519_dalek::VerifyingKey::from_bytes(&key_bytes)
        .map_err(|e| format!("model_signing_key is not a valid Ed25519 public key: {e}"))?;
    Ok(())
}

fn default_num_parallel() -> usize {
    1
}

/// Parse a keep-alive duration string into a `std::time::Duration`.
///
/// Supported formats:
/// - `"5m"` → 5 minutes
/// - `"1h"` → 1 hour
/// - `"30s"` → 30 seconds
/// - `"0"` → Duration::ZERO (unload immediately)
/// - `"-1"` → Duration::MAX (never unload)
pub fn parse_keep_alive(s: &str) -> std::result::Result<std::time::Duration, String> {
    parse_keep_alive_duration(s).ok_or_else(|| {
        format!(
            "invalid keep_alive '{}'; expected a duration such as \"5m\", \"1h\", \"30s\", raw seconds, \"0\", or \"-1\"",
            s
        )
    })
}

fn parse_keep_alive_duration(s: &str) -> Option<std::time::Duration> {
    let s = s.trim();
    if s == "0" {
        return Some(std::time::Duration::ZERO);
    }
    if s == "-1" {
        return Some(std::time::Duration::MAX);
    }

    if let Some(num_str) = s.strip_suffix('s') {
        if let Ok(n) = num_str.parse::<u64>() {
            return Some(std::time::Duration::from_secs(n));
        }
    }
    if let Some(num_str) = s.strip_suffix('m') {
        if let Ok(n) = num_str.parse::<u64>() {
            return n.checked_mul(60).map(std::time::Duration::from_secs);
        }
    }
    if let Some(num_str) = s.strip_suffix('h') {
        if let Ok(n) = num_str.parse::<u64>() {
            return n.checked_mul(3600).map(std::time::Duration::from_secs);
        }
    }

    // Fallback: try to parse as raw seconds
    if let Ok(n) = s.parse::<u64>() {
        return Some(std::time::Duration::from_secs(n));
    }

    None
}

fn parse_required_env_override<T>(name: &str, value: &str) -> Result<T>
where
    T: FromStr,
    T::Err: Display,
{
    value.trim().parse::<T>().map_err(|e| {
        PowerError::Config(format!(
            "invalid environment override {name}={value:?}: {e}"
        ))
    })
}

fn parse_required_bool_env_override(name: &str, value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(PowerError::Config(format!(
            "invalid boolean environment override {name}={value:?}; expected true/false, 1/0, yes/no, or on/off"
        ))),
    }
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    11434
}

fn default_max_loaded_models() -> usize {
    1
}

impl Default for PowerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            data_dir: dirs::power_home(),
            max_loaded_models: default_max_loaded_models(),
            gpu: GpuConfig::default(),
            gpu_attestation: GpuAttestationConfig::default(),
            spec_mode: default_spec_mode(),
            keep_alive: default_keep_alive(),
            use_mlock: false,
            num_thread: None,
            flash_attention: false,
            num_parallel: default_num_parallel(),
            tee_mode: false,
            tee_policy_mode: TeePolicyMode::default(),
            redact_logs: false,
            model_hashes: HashMap::new(),
            model_key_source: None,
            tls_port: None,
            tls_sans: Vec::new(),
            ra_tls: false,
            vsock_port: None,
            api_keys: Vec::new(),
            allowed_tee_types: Vec::new(),
            expected_measurements: HashMap::new(),
            audit_log: false,
            audit_log_path: None,
            audit_log_encrypt: false,
            audit_key_source: None,
            model_signing_key: None,
            key_provider: default_key_provider(),
            key_rotation_sources: Vec::new(),
            in_memory_decrypt: false,
            streaming_decrypt: false,
            suppress_token_metrics: false,
            rate_limit_rps: 0,
            max_concurrent_requests: 0,
            proxy_upstreams: HashMap::new(),
            proxy_effective_prompt_digest: false,
            proxy_effective_prompt_digest_required: false,
            proxy_effective_prompt_digest_path: default_proxy_effective_prompt_digest_path(),
            timing_padding_ms: None,
        }
    }
}

#[cfg(test)]
mod tests;
