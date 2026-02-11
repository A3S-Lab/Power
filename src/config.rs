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

/// User-configurable settings for the Power server and CLI.
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

    /// Custom CORS origins (comma-separated). Empty = permissive.
    #[serde(default)]
    pub origins: Vec<String>,

    /// Custom temporary directory for downloads and scratch files.
    #[serde(default)]
    pub tmpdir: Option<PathBuf>,

    /// Disable automatic pruning of unused blobs (default: false).
    #[serde(default)]
    pub noprune: bool,

    /// Spread model layers across all available GPUs (default: false).
    /// When true, distributes layers evenly instead of filling one GPU first.
    #[serde(default)]
    pub sched_spread: bool,
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

    // Try to parse as number + suffix
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
            origins: Vec::new(),
            tmpdir: None,
            noprune: false,
            sched_spread: false,
        }
    }
}

impl PowerConfig {
    /// Load configuration from the default config file path.
    /// Returns default config if the file does not exist.
    ///
    /// After loading from file, applies Ollama-compatible environment variable
    /// overrides: `OLLAMA_HOST`, `OLLAMA_MODELS`, `OLLAMA_KEEP_ALIVE`,
    /// `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_NUM_GPU`.
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
            toml::from_str(&content)?
        } else {
            Self::default()
        };

        config.apply_env_overrides();
        Ok(config)
    }

    /// Apply Ollama-compatible environment variable overrides.
    ///
    /// Supported variables:
    /// - `OLLAMA_HOST` — `"host:port"` or `"host"` (overrides both host and port)
    /// - `OLLAMA_MODELS` — model storage directory
    /// - `OLLAMA_KEEP_ALIVE` — default keep-alive duration (e.g. `"5m"`, `"-1"`)
    /// - `OLLAMA_MAX_LOADED_MODELS` — maximum concurrent loaded models
    /// - `OLLAMA_NUM_GPU` — number of GPU layers to offload (-1 = all)
    /// - `OLLAMA_NUM_PARALLEL` — number of parallel request slots
    /// - `OLLAMA_DEBUG` — enable debug logging (`"1"` or `"true"`)
    /// - `OLLAMA_ORIGINS` — comma-separated CORS origins
    /// - `OLLAMA_FLASH_ATTENTION` — enable flash attention (`"1"` or `"true"`)
    /// - `OLLAMA_TMPDIR` — custom temporary directory
    /// - `OLLAMA_NOPRUNE` — disable automatic blob pruning (`"1"` or `"true"`)
    /// - `OLLAMA_SCHED_SPREAD` — spread layers across GPUs (`"1"` or `"true"`)
    fn apply_env_overrides(&mut self) {
        if let Ok(host_str) = std::env::var("OLLAMA_HOST") {
            // OLLAMA_HOST can be "host:port" or just "host"
            if let Some((host, port_str)) = host_str.rsplit_once(':') {
                if let Ok(port) = port_str.parse::<u16>() {
                    // Strip scheme prefix if present (e.g. "http://0.0.0.0")
                    let host = host
                        .strip_prefix("http://")
                        .or_else(|| host.strip_prefix("https://"))
                        .unwrap_or(host);
                    self.host = host.to_string();
                    self.port = port;
                } else {
                    // No valid port — treat entire string as host
                    let host = host_str
                        .strip_prefix("http://")
                        .or_else(|| host_str.strip_prefix("https://"))
                        .unwrap_or(&host_str);
                    self.host = host.to_string();
                }
            } else {
                let host = host_str
                    .strip_prefix("http://")
                    .or_else(|| host_str.strip_prefix("https://"))
                    .unwrap_or(&host_str);
                self.host = host.to_string();
            }
        }

        if let Ok(models_dir) = std::env::var("OLLAMA_MODELS") {
            self.data_dir = std::path::PathBuf::from(models_dir);
        }

        if let Ok(keep_alive) = std::env::var("OLLAMA_KEEP_ALIVE") {
            self.keep_alive = keep_alive;
        }

        if let Ok(max_str) = std::env::var("OLLAMA_MAX_LOADED_MODELS") {
            if let Ok(max) = max_str.parse::<usize>() {
                self.max_loaded_models = max;
            }
        }

        if let Ok(gpu_str) = std::env::var("OLLAMA_NUM_GPU") {
            if let Ok(gpu) = gpu_str.parse::<i32>() {
                self.gpu.gpu_layers = gpu;
            }
        }

        if let Ok(par_str) = std::env::var("OLLAMA_NUM_PARALLEL") {
            if let Ok(par) = par_str.parse::<usize>() {
                self.num_parallel = par;
            }
        }

        if let Ok(debug_str) = std::env::var("OLLAMA_DEBUG") {
            if debug_str == "1" || debug_str.eq_ignore_ascii_case("true") {
                // Set RUST_LOG to debug if not already set, so tracing picks it up.
                if std::env::var("RUST_LOG").is_err() {
                    std::env::set_var("RUST_LOG", "debug");
                }
            }
        }

        if let Ok(origins_str) = std::env::var("OLLAMA_ORIGINS") {
            self.origins = origins_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        if let Ok(fa_str) = std::env::var("OLLAMA_FLASH_ATTENTION") {
            if fa_str == "1" || fa_str.eq_ignore_ascii_case("true") {
                self.flash_attention = true;
            }
        }

        if let Ok(tmpdir) = std::env::var("OLLAMA_TMPDIR") {
            self.tmpdir = Some(std::path::PathBuf::from(tmpdir));
        }

        if let Ok(noprune_str) = std::env::var("OLLAMA_NOPRUNE") {
            if noprune_str == "1" || noprune_str.eq_ignore_ascii_case("true") {
                self.noprune = true;
            }
        }

        if let Ok(spread_str) = std::env::var("OLLAMA_SCHED_SPREAD") {
            if spread_str == "1" || spread_str.eq_ignore_ascii_case("true") {
                self.sched_spread = true;
            }
        }
    }

    /// Save the current configuration to the default config file path.
    pub fn save(&self) -> Result<()> {
        let path = dirs::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        Ok(())
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
    }

    #[test]
    fn test_bind_address() {
        let config = PowerConfig::default();
        assert_eq!(config.bind_address(), "127.0.0.1:11434");
    }

    #[test]
    fn test_config_deserialize() {
        let toml_str = r#"
            host = "0.0.0.0"
            port = 8080
            max_loaded_models = 3
        "#;
        let config: PowerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_loaded_models, 3);
    }

    #[test]
    fn test_config_serialize() {
        let config = PowerConfig::default();
        let serialized = toml::to_string_pretty(&config).unwrap();
        assert!(serialized.contains("host"));
        assert!(serialized.contains("port"));
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
            origins: vec!["http://localhost:3000".to_string()],
            tmpdir: None,
            noprune: false,
            sched_spread: false,
        };
        config.save().unwrap();

        let loaded = PowerConfig::load().unwrap();
        assert_eq!(loaded.host, "0.0.0.0");
        assert_eq!(loaded.port, 9999);
        assert_eq!(loaded.max_loaded_models, 5);
        assert_eq!(loaded.num_parallel, 4);
        assert_eq!(loaded.origins, vec!["http://localhost:3000"]);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_gpu_config_defaults() {
        let config = PowerConfig::default();
        assert_eq!(config.gpu.gpu_layers, 0);
        assert_eq!(config.gpu.main_gpu, 0);
    }

    #[test]
    fn test_gpu_config_deserialize() {
        let toml_str = r#"
            host = "127.0.0.1"
            port = 11434

            [gpu]
            gpu_layers = -1
            main_gpu = 1
        "#;
        let config: PowerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.gpu.gpu_layers, -1);
        assert_eq!(config.gpu.main_gpu, 1);
    }

    #[test]
    fn test_gpu_config_missing_uses_defaults() {
        let toml_str = r#"
            host = "127.0.0.1"
            port = 11434
        "#;
        let config: PowerConfig = toml::from_str(toml_str).unwrap();
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
        assert_eq!(
            super::parse_keep_alive("5m"),
            std::time::Duration::from_secs(300)
        );
    }

    #[test]
    fn test_parse_keep_alive_hours() {
        assert_eq!(
            super::parse_keep_alive("1h"),
            std::time::Duration::from_secs(3600)
        );
    }

    #[test]
    fn test_parse_keep_alive_seconds() {
        assert_eq!(
            super::parse_keep_alive("30s"),
            std::time::Duration::from_secs(30)
        );
    }

    #[test]
    fn test_parse_keep_alive_zero() {
        assert_eq!(super::parse_keep_alive("0"), std::time::Duration::ZERO);
    }

    #[test]
    fn test_parse_keep_alive_never() {
        assert_eq!(super::parse_keep_alive("-1"), std::time::Duration::MAX);
    }

    #[test]
    fn test_parse_keep_alive_raw_number() {
        assert_eq!(
            super::parse_keep_alive("120"),
            std::time::Duration::from_secs(120)
        );
    }

    #[test]
    fn test_parse_keep_alive_invalid_defaults() {
        assert_eq!(
            super::parse_keep_alive("abc"),
            std::time::Duration::from_secs(300)
        );
    }

    // ---------------------------------------------------------------
    // Environment variable override tests
    // ---------------------------------------------------------------

    #[test]
    #[serial]
    fn test_env_ollama_host_with_port() {
        std::env::set_var("OLLAMA_HOST", "0.0.0.0:8080");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        std::env::remove_var("OLLAMA_HOST");
    }

    #[test]
    #[serial]
    fn test_env_ollama_host_without_port() {
        std::env::set_var("OLLAMA_HOST", "192.168.1.1");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.host, "192.168.1.1");
        assert_eq!(config.port, 11434); // port unchanged
        std::env::remove_var("OLLAMA_HOST");
    }

    #[test]
    #[serial]
    fn test_env_ollama_host_with_scheme() {
        std::env::set_var("OLLAMA_HOST", "http://0.0.0.0:9999");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 9999);
        std::env::remove_var("OLLAMA_HOST");
    }

    #[test]
    #[serial]
    fn test_env_ollama_host_scheme_no_port() {
        std::env::set_var("OLLAMA_HOST", "http://myhost");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.host, "myhost");
        assert_eq!(config.port, 11434);
        std::env::remove_var("OLLAMA_HOST");
    }

    #[test]
    #[serial]
    fn test_env_ollama_models() {
        std::env::set_var("OLLAMA_MODELS", "/tmp/my-models");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.data_dir, std::path::PathBuf::from("/tmp/my-models"));
        std::env::remove_var("OLLAMA_MODELS");
    }

    #[test]
    #[serial]
    fn test_env_ollama_keep_alive() {
        std::env::set_var("OLLAMA_KEEP_ALIVE", "10m");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.keep_alive, "10m");
        std::env::remove_var("OLLAMA_KEEP_ALIVE");
    }

    #[test]
    #[serial]
    fn test_env_ollama_max_loaded_models() {
        std::env::set_var("OLLAMA_MAX_LOADED_MODELS", "4");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.max_loaded_models, 4);
        std::env::remove_var("OLLAMA_MAX_LOADED_MODELS");
    }

    #[test]
    #[serial]
    fn test_env_ollama_num_gpu() {
        std::env::set_var("OLLAMA_NUM_GPU", "-1");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.gpu.gpu_layers, -1);
        std::env::remove_var("OLLAMA_NUM_GPU");
    }

    #[test]
    #[serial]
    fn test_env_ollama_invalid_values_ignored() {
        std::env::set_var("OLLAMA_MAX_LOADED_MODELS", "not-a-number");
        std::env::set_var("OLLAMA_NUM_GPU", "abc");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.max_loaded_models, 1); // unchanged
        assert_eq!(config.gpu.gpu_layers, 0); // unchanged
        std::env::remove_var("OLLAMA_MAX_LOADED_MODELS");
        std::env::remove_var("OLLAMA_NUM_GPU");
    }

    #[test]
    fn test_config_new_fields_defaults() {
        let config = PowerConfig::default();
        assert!(!config.use_mlock);
        assert!(config.num_thread.is_none());
        assert!(!config.flash_attention);
        assert_eq!(config.num_parallel, 1);
        assert!(config.origins.is_empty());
        assert!(config.tmpdir.is_none());
        assert!(!config.noprune);
        assert!(!config.sched_spread);
    }

    #[test]
    #[serial]
    fn test_config_new_fields_from_toml() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let toml = r#"
            use_mlock = true
            flash_attention = true
            num_thread = 8
        "#;
        let config: PowerConfig = toml::from_str(toml).unwrap();
        assert!(config.use_mlock);
        assert!(config.flash_attention);
        assert_eq!(config.num_thread, Some(8));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_env_ollama_num_parallel() {
        std::env::set_var("OLLAMA_NUM_PARALLEL", "8");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.num_parallel, 8);
        std::env::remove_var("OLLAMA_NUM_PARALLEL");
    }

    #[test]
    #[serial]
    fn test_env_ollama_num_parallel_invalid_ignored() {
        std::env::set_var("OLLAMA_NUM_PARALLEL", "abc");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.num_parallel, 1); // unchanged
        std::env::remove_var("OLLAMA_NUM_PARALLEL");
    }

    #[test]
    #[serial]
    fn test_env_ollama_debug_sets_rust_log() {
        std::env::remove_var("RUST_LOG");
        std::env::set_var("OLLAMA_DEBUG", "1");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(std::env::var("RUST_LOG").unwrap(), "debug");
        std::env::remove_var("OLLAMA_DEBUG");
        std::env::remove_var("RUST_LOG");
    }

    #[test]
    #[serial]
    fn test_env_ollama_debug_true_string() {
        std::env::remove_var("RUST_LOG");
        std::env::set_var("OLLAMA_DEBUG", "true");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(std::env::var("RUST_LOG").unwrap(), "debug");
        std::env::remove_var("OLLAMA_DEBUG");
        std::env::remove_var("RUST_LOG");
    }

    #[test]
    #[serial]
    fn test_env_ollama_debug_does_not_override_existing_rust_log() {
        std::env::set_var("RUST_LOG", "trace");
        std::env::set_var("OLLAMA_DEBUG", "1");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(std::env::var("RUST_LOG").unwrap(), "trace"); // unchanged
        std::env::remove_var("OLLAMA_DEBUG");
        std::env::remove_var("RUST_LOG");
    }

    #[test]
    #[serial]
    fn test_env_ollama_origins() {
        std::env::set_var("OLLAMA_ORIGINS", "http://localhost:3000,https://example.com");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(
            config.origins,
            vec!["http://localhost:3000", "https://example.com"]
        );
        std::env::remove_var("OLLAMA_ORIGINS");
    }

    #[test]
    #[serial]
    fn test_env_ollama_origins_trims_whitespace() {
        std::env::set_var("OLLAMA_ORIGINS", " http://a.com , http://b.com ");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.origins, vec!["http://a.com", "http://b.com"]);
        std::env::remove_var("OLLAMA_ORIGINS");
    }

    #[test]
    #[serial]
    fn test_env_ollama_flash_attention() {
        std::env::set_var("OLLAMA_FLASH_ATTENTION", "1");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.flash_attention);
        std::env::remove_var("OLLAMA_FLASH_ATTENTION");
    }

    #[test]
    #[serial]
    fn test_env_ollama_flash_attention_false_ignored() {
        std::env::set_var("OLLAMA_FLASH_ATTENTION", "0");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(!config.flash_attention); // unchanged
        std::env::remove_var("OLLAMA_FLASH_ATTENTION");
    }

    #[test]
    #[serial]
    fn test_env_ollama_tmpdir() {
        std::env::set_var("OLLAMA_TMPDIR", "/tmp/ollama-scratch");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert_eq!(
            config.tmpdir,
            Some(std::path::PathBuf::from("/tmp/ollama-scratch"))
        );
        std::env::remove_var("OLLAMA_TMPDIR");
    }

    #[test]
    #[serial]
    fn test_env_ollama_noprune() {
        std::env::set_var("OLLAMA_NOPRUNE", "1");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.noprune);
        std::env::remove_var("OLLAMA_NOPRUNE");
    }

    #[test]
    #[serial]
    fn test_env_ollama_noprune_true_string() {
        std::env::set_var("OLLAMA_NOPRUNE", "true");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.noprune);
        std::env::remove_var("OLLAMA_NOPRUNE");
    }

    #[test]
    #[serial]
    fn test_env_ollama_noprune_false_ignored() {
        std::env::set_var("OLLAMA_NOPRUNE", "0");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(!config.noprune); // unchanged
        std::env::remove_var("OLLAMA_NOPRUNE");
    }

    #[test]
    #[serial]
    fn test_env_ollama_sched_spread() {
        std::env::set_var("OLLAMA_SCHED_SPREAD", "1");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.sched_spread);
        std::env::remove_var("OLLAMA_SCHED_SPREAD");
    }

    #[test]
    #[serial]
    fn test_env_ollama_sched_spread_true_string() {
        std::env::set_var("OLLAMA_SCHED_SPREAD", "true");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(config.sched_spread);
        std::env::remove_var("OLLAMA_SCHED_SPREAD");
    }

    #[test]
    #[serial]
    fn test_env_ollama_sched_spread_false_ignored() {
        std::env::set_var("OLLAMA_SCHED_SPREAD", "0");
        let mut config = PowerConfig::default();
        config.apply_env_overrides();
        assert!(!config.sched_spread); // unchanged
        std::env::remove_var("OLLAMA_SCHED_SPREAD");
    }

    #[test]
    fn test_gpu_config_tensor_split_default_empty() {
        let config = GpuConfig::default();
        assert!(config.tensor_split.is_empty());
    }

    #[test]
    fn test_gpu_config_tensor_split_from_toml() {
        let toml_str = r#"
            host = "127.0.0.1"
            port = 11434

            [gpu]
            gpu_layers = -1
            tensor_split = [0.5, 0.5]
        "#;
        let config: PowerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.gpu.tensor_split, vec![0.5, 0.5]);
    }

    #[test]
    fn test_gpu_config_tensor_split_serialization_skips_empty() {
        let config = PowerConfig::default();
        let serialized = toml::to_string_pretty(&config).unwrap();
        assert!(!serialized.contains("tensor_split"));
    }
}
