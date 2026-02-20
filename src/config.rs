use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::dirs;
use crate::error::Result;

/// GPU acceleration settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

/// User-configurable settings for the Power server.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Expected measurements per TEE type (hex-encoded).
    /// When set, attestation reports must match the expected measurement.
    /// Key: tee type (e.g., "sev-snp"), Value: hex measurement string.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub expected_measurements: HashMap<String, String>,

    // --- Audit Logging ---
    /// Enable structured audit logging. Default: false.
    #[serde(default)]
    pub audit_log: bool,

    /// Path to audit log file. Default: $A3S_POWER_HOME/audit.jsonl.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audit_log_path: Option<std::path::PathBuf>,

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
    /// Default: true when tee_mode is enabled. Prevents plaintext from touching disk.
    #[serde(default)]
    pub in_memory_decrypt: bool,

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
}

fn default_key_provider() -> String {
    "static".to_string()
}

fn default_keep_alive() -> String {
    "5m".to_string()
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
pub fn parse_keep_alive(s: &str) -> std::time::Duration {
    let s = s.trim();
    if s == "0" {
        return std::time::Duration::ZERO;
    }
    if s == "-1" {
        return std::time::Duration::MAX;
    }

    if let Some(num_str) = s.strip_suffix('s') {
        if let Ok(n) = num_str.parse::<u64>() {
            return std::time::Duration::from_secs(n);
        }
    }
    if let Some(num_str) = s.strip_suffix('m') {
        if let Ok(n) = num_str.parse::<u64>() {
            return std::time::Duration::from_secs(n * 60);
        }
    }
    if let Some(num_str) = s.strip_suffix('h') {
        if let Ok(n) = num_str.parse::<u64>() {
            return std::time::Duration::from_secs(n * 3600);
        }
    }

    // Fallback: try to parse as raw seconds
    if let Ok(n) = s.parse::<u64>() {
        return std::time::Duration::from_secs(n);
    }

    // Default to 5 minutes if unparsable
    std::time::Duration::from_secs(300)
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
            keep_alive: default_keep_alive(),
            use_mlock: false,
            num_thread: None,
            flash_attention: false,
            num_parallel: default_num_parallel(),
            tee_mode: false,
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
            model_signing_key: None,
            key_provider: default_key_provider(),
            key_rotation_sources: Vec::new(),
            in_memory_decrypt: false,
            suppress_token_metrics: false,
            rate_limit_rps: 0,
            max_concurrent_requests: 0,
        }
    }
}

impl PowerConfig {
    /// Load configuration from the default config file path (HCL format).
    /// Returns default config if the file does not exist.
    ///
    /// After loading from file, applies `A3S_POWER_*` environment variable overrides.
    pub fn load() -> Result<Self> {
        let path = dirs::config_path();
        let mut config = if path.exists() {
            let content = std::fs::read_to_string(&path).map_err(|e| {
                crate::error::PowerError::Config(format!(
                    "Failed to read config file {}: {}",
                    path.display(),
                    e
                ))
            })?;
            hcl::from_str(&content).map_err(|e| {
                crate::error::PowerError::HclDe(format!(
                    "Failed to parse HCL config {}: {}",
                    path.display(),
                    e
                ))
            })?
        } else {
            Self::default()
        };

        config.apply_env_overrides();
        config.validate();
        Ok(config)
    }

    /// Emit warnings for known misconfiguration patterns.
    ///
    /// None of these are fatal — the server will still start — but they indicate
    /// settings that will have no effect or produce unexpected behavior.
    pub fn validate(&self) {
        // Warn if keep_alive is set to something unparseable (will silently fall back to 5m).
        let ka = self.keep_alive.trim();
        let parseable = ka == "0"
            || ka == "-1"
            || ka
                .strip_suffix('s')
                .and_then(|n| n.parse::<u64>().ok())
                .is_some()
            || ka
                .strip_suffix('m')
                .and_then(|n| n.parse::<u64>().ok())
                .is_some()
            || ka
                .strip_suffix('h')
                .and_then(|n| n.parse::<u64>().ok())
                .is_some()
            || ka.parse::<u64>().is_ok();
        if !parseable {
            tracing::warn!(
                keep_alive = %self.keep_alive,
                "keep_alive value is not parseable; defaulting to 5m. \
                 Valid formats: \"5m\", \"1h\", \"30s\", \"0\", \"-1\"."
            );
        }

        // Warn if model_signing_key is set but is not a valid 64-char hex string (Ed25519 pubkey).
        if let Some(ref key) = self.model_signing_key {
            if key.len() != 64 || !key.chars().all(|c| c.is_ascii_hexdigit()) {
                tracing::warn!(
                    "model_signing_key must be a 64-character hex-encoded Ed25519 public key \
                     (32 bytes). The current value has length {} and may fail at runtime.",
                    key.len()
                );
            }
        }

        // Warn if ra_tls is enabled but tls_port is not set (RA-TLS requires a TLS listener).
        if self.ra_tls && self.tls_port.is_none() {
            tracing::warn!(
                "ra_tls = true but tls_port is not set. RA-TLS requires a TLS listener. \
                 Set tls_port to enable the TLS server."
            );
        }

        // Warn if key_provider is "rotating" but key_rotation_sources is empty.
        if self.key_provider == "rotating" && self.key_rotation_sources.is_empty() {
            tracing::warn!(
                "key_provider = \"rotating\" but key_rotation_sources is empty. \
                 No keys are available for model decryption."
            );
        }
    }

    /// Apply `A3S_POWER_*` environment variable overrides.
    fn apply_env_overrides(&mut self) {
        if let Ok(host) = std::env::var("A3S_POWER_HOST") {
            self.host = host;
        }

        if let Ok(port_str) = std::env::var("A3S_POWER_PORT") {
            if let Ok(port) = port_str.parse::<u16>() {
                self.port = port;
            }
        }

        if let Ok(data_dir) = std::env::var("A3S_POWER_DATA_DIR") {
            self.data_dir = PathBuf::from(data_dir);
        }

        if let Ok(max_str) = std::env::var("A3S_POWER_MAX_MODELS") {
            if let Ok(max) = max_str.parse::<usize>() {
                self.max_loaded_models = max;
            }
        }

        if let Ok(keep_alive) = std::env::var("A3S_POWER_KEEP_ALIVE") {
            self.keep_alive = keep_alive;
        }

        if let Ok(gpu_str) = std::env::var("A3S_POWER_GPU_LAYERS") {
            if let Ok(gpu) = gpu_str.parse::<i32>() {
                self.gpu.gpu_layers = gpu;
            }
        }

        if let Ok(tee_str) = std::env::var("A3S_POWER_TEE_MODE") {
            if tee_str == "1" || tee_str.eq_ignore_ascii_case("true") {
                self.tee_mode = true;
            }
        }

        if let Ok(redact_str) = std::env::var("A3S_POWER_REDACT_LOGS") {
            if redact_str == "1" || redact_str.eq_ignore_ascii_case("true") {
                self.redact_logs = true;
            }
        }

        // When TEE mode is enabled, default redact_logs to true unless explicitly disabled
        if self.tee_mode && std::env::var("A3S_POWER_REDACT_LOGS").is_err() && !self.redact_logs {
            self.redact_logs = true;
        }

        if let Ok(tls_port_str) = std::env::var("A3S_POWER_TLS_PORT") {
            if let Ok(port) = tls_port_str.parse::<u16>() {
                self.tls_port = Some(port);
            }
        }

        if let Ok(ra_tls_str) = std::env::var("A3S_POWER_RA_TLS") {
            if ra_tls_str == "1" || ra_tls_str.eq_ignore_ascii_case("true") {
                self.ra_tls = true;
            }
        }

        if let Ok(vsock_str) = std::env::var("A3S_POWER_VSOCK_PORT") {
            if let Ok(port) = vsock_str.parse::<u32>() {
                self.vsock_port = Some(port);
            }
        }

        if let Ok(keys_str) = std::env::var("A3S_POWER_API_KEYS") {
            let keys: Vec<String> = keys_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !keys.is_empty() {
                self.api_keys = keys;
            }
        }

        // A3S_POWER_TEE_STRICT=1 removes "simulated" from allowed TEE types
        if std::env::var("A3S_POWER_TEE_STRICT").as_deref() == Ok("1") {
            if self.allowed_tee_types.is_empty() {
                // Default to all hardware types when strict mode is enabled
                self.allowed_tee_types = vec!["sev-snp".to_string(), "tdx".to_string()];
            } else {
                self.allowed_tee_types.retain(|t| t != "simulated");
            }
        }

        if std::env::var("A3S_POWER_AUDIT_LOG").as_deref() == Ok("1") {
            self.audit_log = true;
        }
    }

    /// Save the current configuration to the default config file path (HCL format).
    pub fn save(&self) -> Result<()> {
        let path = dirs::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = self.to_hcl();
        std::fs::write(&path, content)?;
        Ok(())
    }

    /// Serialize the config to HCL format.
    fn to_hcl(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("host = \"{}\"\n", self.host));
        out.push_str(&format!("port = {}\n", self.port));
        out.push_str(&format!("data_dir = \"{}\"\n", self.data_dir.display()));
        out.push_str(&format!("max_loaded_models = {}\n", self.max_loaded_models));
        out.push_str(&format!("keep_alive = \"{}\"\n", self.keep_alive));
        out.push_str(&format!("use_mlock = {}\n", self.use_mlock));
        if let Some(nt) = self.num_thread {
            out.push_str(&format!("num_thread = {}\n", nt));
        }
        out.push_str(&format!("flash_attention = {}\n", self.flash_attention));
        out.push_str(&format!("num_parallel = {}\n", self.num_parallel));
        out.push_str(&format!("tee_mode = {}\n", self.tee_mode));
        out.push_str(&format!("redact_logs = {}\n", self.redact_logs));

        // GPU block
        out.push_str("\ngpu {\n");
        out.push_str(&format!("  gpu_layers = {}\n", self.gpu.gpu_layers));
        out.push_str(&format!("  main_gpu = {}\n", self.gpu.main_gpu));
        if !self.gpu.tensor_split.is_empty() {
            let splits: Vec<String> = self
                .gpu
                .tensor_split
                .iter()
                .map(|v| v.to_string())
                .collect();
            out.push_str(&format!("  tensor_split = [{}]\n", splits.join(", ")));
        }
        out.push_str("}\n");

        // Model hashes
        if !self.model_hashes.is_empty() {
            out.push_str("\nmodel_hashes = {\n");
            for (name, hash) in &self.model_hashes {
                out.push_str(&format!("  \"{}\" = \"{}\"\n", name, hash));
            }
            out.push_str("}\n");
        }

        // TLS settings
        if let Some(tls_port) = self.tls_port {
            out.push_str(&format!("tls_port = {}\n", tls_port));
        }
        if self.ra_tls {
            out.push_str(&format!("ra_tls = {}\n", self.ra_tls));
        }

        // Vsock transport
        if let Some(vsock_port) = self.vsock_port {
            out.push_str(&format!("vsock_port = {}\n", vsock_port));
        }

        // API keys
        if !self.api_keys.is_empty() {
            let keys: Vec<String> = self.api_keys.iter().map(|k| format!("\"{}\"", k)).collect();
            out.push_str(&format!("api_keys = [{}]\n", keys.join(", ")));
        }

        // TEE policy
        if !self.allowed_tee_types.is_empty() {
            let types: Vec<String> = self
                .allowed_tee_types
                .iter()
                .map(|t| format!("\"{}\"", t))
                .collect();
            out.push_str(&format!("allowed_tee_types = [{}]\n", types.join(", ")));
        }
        if !self.expected_measurements.is_empty() {
            out.push_str("expected_measurements = {\n");
            for (k, v) in &self.expected_measurements {
                out.push_str(&format!("  {} = \"{}\"\n", k, v));
            }
            out.push_str("}\n");
        }

        // Audit logging
        if self.audit_log {
            out.push_str("audit_log = true\n");
        }
        if let Some(ref path) = self.audit_log_path {
            out.push_str(&format!("audit_log_path = \"{}\"\n", path.display()));
        }

        // Model signing
        if let Some(ref key) = self.model_signing_key {
            out.push_str(&format!("model_signing_key = \"{}\"\n", key));
        }

        // Key provider
        if self.key_provider != "static" {
            out.push_str(&format!("key_provider = \"{}\"\n", self.key_provider));
        }

        // Rate limiting
        if self.rate_limit_rps > 0 {
            out.push_str(&format!("rate_limit_rps = {}\n", self.rate_limit_rps));
        }
        if self.max_concurrent_requests > 0 {
            out.push_str(&format!(
                "max_concurrent_requests = {}\n",
                self.max_concurrent_requests
            ));
        }

        // TEE in-memory decryption / token metrics
        if self.in_memory_decrypt {
            out.push_str("in_memory_decrypt = true\n");
        }
        if self.suppress_token_metrics {
            out.push_str("suppress_token_metrics = true\n");
        }

        out
    }

    /// Returns the server bind address string (e.g., "127.0.0.1:11434").
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_default_config() {
        let config = PowerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 11434);
        assert_eq!(config.max_loaded_models, 1);
        assert!(!config.tee_mode);
        assert!(!config.redact_logs);
        assert!(config.model_hashes.is_empty());
    }

    #[test]
    fn test_bind_address() {
        let config = PowerConfig::default();
        assert_eq!(config.bind_address(), "127.0.0.1:11434");
    }

    #[test]
    fn test_config_deserialize_hcl() {
        let hcl_str = r#"
            host = "0.0.0.0"
            port = 8080
            max_loaded_models = 3
        "#;
        let config: PowerConfig = hcl::from_str(hcl_str).unwrap();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_loaded_models, 3);
    }

    #[test]
    fn test_config_serialize_hcl() {
        let config = PowerConfig::default();
        let serialized = config.to_hcl();
        assert!(serialized.contains("host"));
        assert!(serialized.contains("port"));
        assert!(serialized.contains("gpu {"));
    }

    #[test]
    #[serial]
    fn test_config_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let config = PowerConfig {
            host: "0.0.0.0".to_string(),
            port: 9999,
            data_dir: dir.path().to_path_buf(),
            max_loaded_models: 5,
            gpu: GpuConfig::default(),
            keep_alive: "5m".to_string(),
            use_mlock: false,
            num_thread: None,
            flash_attention: false,
            num_parallel: 4,
            tee_mode: true,
            redact_logs: true,
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
            model_signing_key: None,
            key_provider: "static".to_string(),
            key_rotation_sources: Vec::new(),
            in_memory_decrypt: false,
            suppress_token_metrics: false,
            rate_limit_rps: 0,
            max_concurrent_requests: 0,
        };
        config.save().unwrap();

        let loaded = PowerConfig::load().unwrap();
        assert_eq!(loaded.host, "0.0.0.0");
        assert_eq!(loaded.port, 9999);
        assert_eq!(loaded.max_loaded_models, 5);
        assert_eq!(loaded.num_parallel, 4);
        assert!(loaded.tee_mode);
        assert!(loaded.redact_logs);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_gpu_config_defaults() {
        let config = PowerConfig::default();
        assert_eq!(config.gpu.gpu_layers, 0);
        assert_eq!(config.gpu.main_gpu, 0);
    }

    #[test]
    fn test_gpu_config_deserialize_hcl() {
        let hcl_str = r#"
            host = "127.0.0.1"
            port = 11434

            gpu {
                gpu_layers = -1
                main_gpu = 1
            }
        "#;
        let config: PowerConfig = hcl::from_str(hcl_str).unwrap();
        assert_eq!(config.gpu.gpu_layers, -1);
        assert_eq!(config.gpu.main_gpu, 1);
    }

    #[test]
    fn test_gpu_config_missing_uses_defaults() {
        let hcl_str = r#"
            host = "127.0.0.1"
            port = 11434
        "#;
        let config: PowerConfig = hcl::from_str(hcl_str).unwrap();
        assert_eq!(config.gpu.gpu_layers, 0);
        assert_eq!(config.gpu.main_gpu, 0);
    }

    #[test]
    fn test_default_keep_alive() {
        let config = PowerConfig::default();
        assert_eq!(config.keep_alive, "5m");
    }

    #[test]
    fn test_parse_keep_alive_minutes() {
        assert_eq!(parse_keep_alive("5m"), std::time::Duration::from_secs(300));
    }

    #[test]
    fn test_parse_keep_alive_hours() {
        assert_eq!(parse_keep_alive("1h"), std::time::Duration::from_secs(3600));
    }

    #[test]
    fn test_parse_keep_alive_seconds() {
        assert_eq!(parse_keep_alive("30s"), std::time::Duration::from_secs(30));
    }

    #[test]
    fn test_parse_keep_alive_zero() {
        assert_eq!(parse_keep_alive("0"), std::time::Duration::ZERO);
    }

    #[test]
    fn test_parse_keep_alive_never() {
        assert_eq!(parse_keep_alive("-1"), std::time::Duration::MAX);
    }

    #[test]
    fn test_parse_keep_alive_raw_number() {
        assert_eq!(parse_keep_alive("120"), std::time::Duration::from_secs(120));
    }

    #[test]
    fn test_parse_keep_alive_invalid_defaults() {
        assert_eq!(parse_keep_alive("abc"), std::time::Duration::from_secs(300));
    }

    // ---------------------------------------------------------------
    // Environment variable override tests
    // ---------------------------------------------------------------

    #[test]
    #[serial]
    fn test_env_a3s_power_host() {
        std::env::set_var("A3S_POWER_HOST", "0.0.0.0");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.host, "0.0.0.0");
        std::env::remove_var("A3S_POWER_HOST");
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_port() {
        std::env::set_var("A3S_POWER_PORT", "8080");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.port, 8080);
        std::env::remove_var("A3S_POWER_PORT");
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_data_dir() {
        std::env::set_var("A3S_POWER_DATA_DIR", "/tmp/my-models");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.data_dir, PathBuf::from("/tmp/my-models"));
        std::env::remove_var("A3S_POWER_DATA_DIR");
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_max_models() {
        std::env::set_var("A3S_POWER_MAX_MODELS", "4");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.max_loaded_models, 4);
        std::env::remove_var("A3S_POWER_MAX_MODELS");
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_keep_alive() {
        std::env::set_var("A3S_POWER_KEEP_ALIVE", "10m");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.keep_alive, "10m");
        std::env::remove_var("A3S_POWER_KEEP_ALIVE");
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_gpu_layers() {
        std::env::set_var("A3S_POWER_GPU_LAYERS", "-1");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.gpu.gpu_layers, -1);
        std::env::remove_var("A3S_POWER_GPU_LAYERS");
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_tee_mode() {
        std::env::set_var("A3S_POWER_TEE_MODE", "true");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.tee_mode);
        assert!(config.redact_logs); // auto-enabled when tee_mode
        std::env::remove_var("A3S_POWER_TEE_MODE");
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_redact_logs() {
        std::env::set_var("A3S_POWER_REDACT_LOGS", "1");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.redact_logs);
        std::env::remove_var("A3S_POWER_REDACT_LOGS");
    }

    #[test]
    #[serial]
    fn test_env_invalid_values_ignored() {
        std::env::set_var("A3S_POWER_MAX_MODELS", "not-a-number");
        std::env::set_var("A3S_POWER_GPU_LAYERS", "abc");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.max_loaded_models, 1); // unchanged
        assert_eq!(config.gpu.gpu_layers, 0); // unchanged
        std::env::remove_var("A3S_POWER_MAX_MODELS");
        std::env::remove_var("A3S_POWER_GPU_LAYERS");
    }

    #[test]
    fn test_config_new_fields_defaults() {
        let config = PowerConfig::default();
        assert!(!config.use_mlock);
        assert!(config.num_thread.is_none());
        assert!(!config.flash_attention);
        assert_eq!(config.num_parallel, 1);
    }

    #[test]
    fn test_config_tee_fields_from_hcl() {
        let hcl_str = r#"
            tee_mode = true
            redact_logs = true

            model_hashes = {
                "llama3" = "sha256:abc123"
            }
        "#;
        let config: PowerConfig = hcl::from_str(hcl_str).unwrap();
        assert!(config.tee_mode);
        assert!(config.redact_logs);
        assert_eq!(
            config.model_hashes.get("llama3"),
            Some(&"sha256:abc123".to_string())
        );
    }

    #[test]
    fn test_gpu_config_tensor_split_default_empty() {
        let config = GpuConfig::default();
        assert!(config.tensor_split.is_empty());
    }

    #[test]
    fn test_gpu_config_tensor_split_from_hcl() {
        let hcl_str = r#"
            host = "127.0.0.1"
            port = 11434

            gpu {
                gpu_layers = -1
                tensor_split = [0.5, 0.5]
            }
        "#;
        let config: PowerConfig = hcl::from_str(hcl_str).unwrap();
        assert_eq!(config.gpu.tensor_split, vec![0.5, 0.5]);
    }

    #[test]
    fn test_gpu_config_tensor_split_serialization_skips_empty() {
        let config = PowerConfig::default();
        let serialized = config.to_hcl();
        assert!(!serialized.contains("tensor_split"));
    }

    #[test]
    fn test_config_hcl_invalid() {
        let result: std::result::Result<PowerConfig, _> = hcl::from_str("{{{{ invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_tls_port_defaults_to_none() {
        let config = PowerConfig::default();
        assert!(config.tls_port.is_none());
    }

    #[test]
    fn test_ra_tls_defaults_to_false() {
        let config = PowerConfig::default();
        assert!(!config.ra_tls);
    }

    #[test]
    fn test_tls_port_from_hcl() {
        let hcl_str = r#"tls_port = 8443"#;
        let config: PowerConfig = hcl::from_str(hcl_str).unwrap();
        assert_eq!(config.tls_port, Some(8443));
    }

    #[test]
    fn test_ra_tls_from_hcl() {
        let hcl_str = r#"ra_tls = true"#;
        let config: PowerConfig = hcl::from_str(hcl_str).unwrap();
        assert!(config.ra_tls);
    }

    #[test]
    fn test_tls_port_not_serialized_when_none() {
        let config = PowerConfig::default();
        let serialized = config.to_hcl();
        assert!(!serialized.contains("tls_port"));
    }

    #[test]
    fn test_ra_tls_not_serialized_when_false() {
        let config = PowerConfig::default();
        let serialized = config.to_hcl();
        assert!(!serialized.contains("ra_tls"));
    }

    #[test]
    fn test_tls_port_serialized_when_set() {
        let config = PowerConfig {
            tls_port: Some(8443),
            ..Default::default()
        };
        let serialized = config.to_hcl();
        assert!(serialized.contains("tls_port = 8443"));
    }

    #[test]
    fn test_ra_tls_serialized_when_true() {
        let config = PowerConfig {
            ra_tls: true,
            ..Default::default()
        };
        let serialized = config.to_hcl();
        assert!(serialized.contains("ra_tls = true"));
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_tls_port() {
        std::env::set_var("A3S_POWER_TLS_PORT", "8443");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.tls_port, Some(8443));
        std::env::remove_var("A3S_POWER_TLS_PORT");
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_ra_tls() {
        std::env::set_var("A3S_POWER_RA_TLS", "true");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.ra_tls);
        std::env::remove_var("A3S_POWER_RA_TLS");
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_tls_port_invalid_ignored() {
        std::env::set_var("A3S_POWER_TLS_PORT", "not-a-port");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.tls_port.is_none());
        std::env::remove_var("A3S_POWER_TLS_PORT");
    }

    #[test]
    fn test_vsock_port_defaults_to_none() {
        let config = PowerConfig::default();
        assert!(config.vsock_port.is_none());
    }

    #[test]
    fn test_vsock_port_from_hcl() {
        let hcl_str = r#"vsock_port = 11434"#;
        let config: PowerConfig = hcl::from_str(hcl_str).unwrap();
        assert_eq!(config.vsock_port, Some(11434));
    }

    #[test]
    fn test_vsock_port_not_serialized_when_none() {
        let config = PowerConfig::default();
        let serialized = config.to_hcl();
        assert!(!serialized.contains("vsock_port"));
    }

    #[test]
    fn test_vsock_port_serialized_when_set() {
        let config = PowerConfig {
            vsock_port: Some(11434),
            ..Default::default()
        };
        let serialized = config.to_hcl();
        assert!(serialized.contains("vsock_port = 11434"));
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_vsock_port() {
        std::env::set_var("A3S_POWER_VSOCK_PORT", "11434");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.vsock_port, Some(11434));
        std::env::remove_var("A3S_POWER_VSOCK_PORT");
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_vsock_port_invalid_ignored() {
        std::env::set_var("A3S_POWER_VSOCK_PORT", "not-a-port");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.vsock_port.is_none());
        std::env::remove_var("A3S_POWER_VSOCK_PORT");
    }

    #[test]
    #[serial]
    fn test_load_config_hcl_file() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let hcl_path = dir.path().join("config.hcl");
        std::fs::write(
            &hcl_path,
            r#"
                host = "0.0.0.0"
                port = 9090
                max_loaded_models = 2
            "#,
        )
        .unwrap();

        let config = PowerConfig::load().unwrap();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 9090);
        assert_eq!(config.max_loaded_models, 2);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_api_keys_defaults_to_empty() {
        let config = PowerConfig::default();
        assert!(config.api_keys.is_empty());
    }

    #[test]
    fn test_api_keys_from_hcl() {
        let hcl_str = r#"api_keys = ["sha256hash1", "sha256hash2"]"#;
        let config: PowerConfig = hcl::from_str(hcl_str).unwrap();
        assert_eq!(config.api_keys, vec!["sha256hash1", "sha256hash2"]);
    }

    #[test]
    fn test_api_keys_not_serialized_when_empty() {
        let config = PowerConfig::default();
        let serialized = config.to_hcl();
        assert!(!serialized.contains("api_keys"));
    }

    #[test]
    fn test_api_keys_serialized_when_set() {
        let config = PowerConfig {
            api_keys: vec!["key1".to_string(), "key2".to_string()],
            ..Default::default()
        };
        let serialized = config.to_hcl();
        assert!(serialized.contains("api_keys"));
        assert!(serialized.contains("key1"));
        assert!(serialized.contains("key2"));
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_api_keys() {
        std::env::set_var("A3S_POWER_API_KEYS", "key_a,key_b,key_c");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.api_keys, vec!["key_a", "key_b", "key_c"]);
        std::env::remove_var("A3S_POWER_API_KEYS");
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_api_keys_trims_whitespace() {
        std::env::set_var("A3S_POWER_API_KEYS", " key_a , key_b ");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.api_keys, vec!["key_a", "key_b"]);
        std::env::remove_var("A3S_POWER_API_KEYS");
    }

    #[test]
    #[serial]
    fn test_env_a3s_power_api_keys_empty_ignored() {
        std::env::set_var("A3S_POWER_API_KEYS", "");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.api_keys.is_empty());
        std::env::remove_var("A3S_POWER_API_KEYS");
    }

    #[test]
    fn test_allowed_tee_types_defaults_to_empty() {
        let config = PowerConfig::default();
        assert!(config.allowed_tee_types.is_empty());
    }

    #[test]
    fn test_allowed_tee_types_from_hcl() {
        let hcl_str = r#"allowed_tee_types = ["sev-snp", "tdx"]"#;
        let config: PowerConfig = hcl::from_str(hcl_str).unwrap();
        assert_eq!(config.allowed_tee_types, vec!["sev-snp", "tdx"]);
    }

    #[test]
    #[serial]
    fn test_tee_strict_env_removes_simulated() {
        std::env::set_var("A3S_POWER_TEE_STRICT", "1");
        let mut config = PowerConfig {
            allowed_tee_types: vec![
                "sev-snp".to_string(),
                "simulated".to_string(),
                "tdx".to_string(),
            ],
            ..Default::default()
        };
        config.apply_env_overrides();
        assert!(!config.allowed_tee_types.contains(&"simulated".to_string()));
        assert!(config.allowed_tee_types.contains(&"sev-snp".to_string()));
        std::env::remove_var("A3S_POWER_TEE_STRICT");
    }

    #[test]
    #[serial]
    fn test_tee_strict_env_sets_hardware_defaults_when_empty() {
        std::env::set_var("A3S_POWER_TEE_STRICT", "1");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.allowed_tee_types.contains(&"sev-snp".to_string()));
        assert!(config.allowed_tee_types.contains(&"tdx".to_string()));
        assert!(!config.allowed_tee_types.contains(&"simulated".to_string()));
        std::env::remove_var("A3S_POWER_TEE_STRICT");
    }

    #[test]
    fn test_audit_log_defaults_to_false() {
        let config = PowerConfig::default();
        assert!(!config.audit_log);
    }

    #[test]
    #[serial]
    fn test_audit_log_env_override() {
        std::env::set_var("A3S_POWER_AUDIT_LOG", "1");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.audit_log);
        std::env::remove_var("A3S_POWER_AUDIT_LOG");
    }

    #[test]
    fn test_model_signing_key_defaults_to_none() {
        let config = PowerConfig::default();
        assert!(config.model_signing_key.is_none());
    }

    #[test]
    fn test_model_signing_key_from_hcl() {
        let hcl_str = r#"model_signing_key = "aabbccdd""#;
        let config: PowerConfig = hcl::from_str(hcl_str).unwrap();
        assert_eq!(config.model_signing_key.as_deref(), Some("aabbccdd"));
    }

    #[test]
    fn test_to_hcl_includes_policy_fields_when_set() {
        let mut measurements = HashMap::new();
        measurements.insert("sev-snp".to_string(), "deadbeef".to_string());
        let config = PowerConfig {
            allowed_tee_types: vec!["sev-snp".to_string()],
            expected_measurements: measurements,
            audit_log: true,
            model_signing_key: Some("pubkey123".to_string()),
            ..Default::default()
        };
        let hcl = config.to_hcl();
        assert!(hcl.contains("allowed_tee_types"));
        assert!(hcl.contains("sev-snp"));
        assert!(hcl.contains("expected_measurements"));
        assert!(hcl.contains("deadbeef"));
        assert!(hcl.contains("audit_log = true"));
        assert!(hcl.contains("model_signing_key"));
        assert!(hcl.contains("pubkey123"));
    }

    #[test]
    fn test_tls_sans_defaults_to_empty() {
        let config = PowerConfig::default();
        assert!(config.tls_sans.is_empty());
    }

    #[test]
    fn test_tls_sans_from_hcl() {
        let hcl_str = r#"tls_sans = ["myserver.internal", "10.0.0.1"]"#;
        let config: PowerConfig = hcl::from_str(hcl_str).unwrap();
        assert_eq!(config.tls_sans, vec!["myserver.internal", "10.0.0.1"]);
    }

    // --- validate() tests ---

    #[test]
    fn test_validate_default_config_no_warnings() {
        // Default config is valid — validate() should not panic
        let config = PowerConfig::default();
        config.validate(); // must not panic
    }

    #[test]
    fn test_validate_keep_alive_valid_formats() {
        // All valid formats should pass without warnings
        for ka in &["0", "-1", "5m", "1h", "30s", "300"] {
            let config = PowerConfig {
                keep_alive: ka.to_string(),
                ..Default::default()
            };
            config.validate(); // must not panic
        }
    }

    #[test]
    fn test_validate_model_signing_key_valid_hex() {
        let config = PowerConfig {
            model_signing_key: Some("a".repeat(64)),
            ..Default::default()
        };
        config.validate(); // must not panic
    }

    #[test]
    fn test_validate_model_signing_key_wrong_length() {
        // 32-char key (wrong length for Ed25519 — should emit warning but not panic)
        let config = PowerConfig {
            model_signing_key: Some("deadbeef".repeat(4)),
            ..Default::default()
        };
        config.validate(); // must not panic
    }

    #[test]
    fn test_validate_ra_tls_without_tls_port_emits_warning() {
        let config = PowerConfig {
            ra_tls: true,
            tls_port: None,
            ..Default::default()
        };
        config.validate(); // must not panic; warning is emitted via tracing
    }

    #[test]
    fn test_validate_ra_tls_with_tls_port_is_valid() {
        let config = PowerConfig {
            ra_tls: true,
            tls_port: Some(11435),
            ..Default::default()
        };
        config.validate(); // must not panic, no warning
    }

    #[test]
    fn test_validate_rotating_provider_empty_sources() {
        let config = PowerConfig {
            key_provider: "rotating".to_string(),
            key_rotation_sources: vec![],
            ..Default::default()
        };
        config.validate(); // must not panic; warning is emitted via tracing
    }
}
