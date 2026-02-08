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
}

/// User-configurable settings for the Power server and CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerConfig {
    /// Host address for the HTTP server (default: 127.0.0.1)
    #[serde(default = "default_host")]
    pub host: String,

    /// Port for the HTTP server (default: 11435)
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
}

fn default_keep_alive() -> String {
    "5m".to_string()
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
    11435
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
        }
    }
}

impl PowerConfig {
    /// Load configuration from the default config file path.
    /// Returns default config if the file does not exist.
    pub fn load() -> Result<Self> {
        let path = dirs::config_path();
        if path.exists() {
            let content = std::fs::read_to_string(&path).map_err(|e| {
                crate::error::PowerError::Config(format!(
                    "Failed to read config file {}: {}",
                    path.display(),
                    e
                ))
            })?;
            let config: PowerConfig = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Self::default())
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

    /// Returns the server bind address string (e.g., "127.0.0.1:11435").
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
        assert_eq!(config.port, 11435);
        assert_eq!(config.max_loaded_models, 1);
    }

    #[test]
    fn test_bind_address() {
        let config = PowerConfig::default();
        assert_eq!(config.bind_address(), "127.0.0.1:11435");
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
        };
        config.save().unwrap();

        let loaded = PowerConfig::load().unwrap();
        assert_eq!(loaded.host, "0.0.0.0");
        assert_eq!(loaded.port, 9999);
        assert_eq!(loaded.max_loaded_models, 5);

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
            port = 11435

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
            port = 11435
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
}
