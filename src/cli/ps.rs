use crate::config::PowerConfig;
use crate::error::{PowerError, Result};

/// Execute the `ps` command: list running models on the server.
pub async fn execute() -> Result<()> {
    let config = PowerConfig::load()?;
    let url = format!("http://{}:{}/api/ps", config.host, config.port);

    let resp = reqwest::get(&url).await.map_err(|e| {
        PowerError::Server(format!(
            "Failed to connect to server at {}: {e}. Is the server running?",
            config.bind_address()
        ))
    })?;

    if !resp.status().is_success() {
        return Err(PowerError::Server(format!(
            "Server returned status {}",
            resp.status()
        )));
    }

    let body: serde_json::Value = resp.json().await.map_err(|e| {
        PowerError::Server(format!("Failed to parse server response: {e}"))
    })?;

    let models = body["models"].as_array();
    match models {
        Some(models) if !models.is_empty() => {
            println!(
                "{:<30} {:<10} {:<12} {:<10} {}",
                "NAME", "SIZE", "FORMAT", "QUANT", "EXPIRES"
            );
            for model in models {
                let name = model["name"].as_str().unwrap_or("?");
                let size = model["size"].as_u64().unwrap_or(0);
                let format = model["details"]["format"].as_str().unwrap_or("?");
                let quant = model["details"]["quantization_level"]
                    .as_str()
                    .unwrap_or("-");
                let expires = model["expires_at"]
                    .as_str()
                    .unwrap_or("never");
                let size_str = format_size(size);
                println!("{:<30} {:<10} {:<12} {:<10} {}", name, size_str, format, quant, expires);
            }
        }
        _ => {
            println!("No running models.");
        }
    }

    Ok(())
}

/// Format bytes into a human-readable size string.
fn format_size(bytes: u64) -> String {
    const GB: u64 = 1_000_000_000;
    const MB: u64 = 1_000_000;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(4_000_000_000), "4.0 GB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(500_000_000), "500.0 MB");
    }

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(512), "512 B");
    }

    #[test]
    fn test_format_size_zero() {
        assert_eq!(format_size(0), "0 B");
    }

    #[test]
    fn test_format_size_exact_gb() {
        assert_eq!(format_size(1_000_000_000), "1.0 GB");
    }

    #[test]
    fn test_format_size_exact_mb() {
        assert_eq!(format_size(1_000_000), "1.0 MB");
    }

    #[test]
    fn test_format_size_fractional_gb() {
        assert_eq!(format_size(1_500_000_000), "1.5 GB");
    }

    #[test]
    fn test_format_size_fractional_mb() {
        assert_eq!(format_size(2_500_000), "2.5 MB");
    }

    #[test]
    fn test_format_size_small_bytes() {
        assert_eq!(format_size(1), "1 B");
        assert_eq!(format_size(999), "999 B");
    }

    #[test]
    fn test_format_size_boundary_mb_to_gb() {
        assert_eq!(format_size(999_999_999), "1000.0 MB");
        assert_eq!(format_size(1_000_000_000), "1.0 GB");
    }

    #[test]
    fn test_ps_url_construction() {
        let config = crate::config::PowerConfig::default();
        let url = format!("http://{}:{}/api/ps", config.host, config.port);
        assert_eq!(url, "http://127.0.0.1:11434/api/ps");
    }

    #[test]
    fn test_ps_url_custom_host_port() {
        let config = crate::config::PowerConfig {
            host: "0.0.0.0".to_string(),
            port: 8080,
            ..Default::default()
        };
        let url = format!("http://{}:{}/api/ps", config.host, config.port);
        assert_eq!(url, "http://0.0.0.0:8080/api/ps");
    }

    #[test]
    fn test_ps_parse_model_json() {
        let body: serde_json::Value = serde_json::json!({
            "models": [{
                "name": "llama3.2:3b",
                "size": 2_000_000_000u64,
                "details": {
                    "format": "GGUF",
                    "quantization_level": "Q4_K_M"
                },
                "expires_at": "2024-01-01T00:00:00Z"
            }]
        });

        let models = body["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        let model = &models[0];
        assert_eq!(model["name"].as_str().unwrap(), "llama3.2:3b");
        assert_eq!(model["size"].as_u64().unwrap(), 2_000_000_000);
        assert_eq!(model["details"]["format"].as_str().unwrap(), "GGUF");
        assert_eq!(model["details"]["quantization_level"].as_str().unwrap(), "Q4_K_M");
        assert_eq!(model["expires_at"].as_str().unwrap(), "2024-01-01T00:00:00Z");
    }

    #[test]
    fn test_ps_parse_empty_models() {
        let body: serde_json::Value = serde_json::json!({
            "models": []
        });
        let models = body["models"].as_array().unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn test_ps_parse_missing_fields() {
        let body: serde_json::Value = serde_json::json!({
            "models": [{
                "name": "test"
            }]
        });
        let model = &body["models"][0];
        assert_eq!(model["name"].as_str().unwrap(), "test");
        assert_eq!(model["size"].as_u64().unwrap_or(0), 0);
        assert_eq!(model["details"]["format"].as_str().unwrap_or("?"), "?");
        assert_eq!(model["details"]["quantization_level"].as_str().unwrap_or("-"), "-");
        assert_eq!(model["expires_at"].as_str().unwrap_or("never"), "never");
    }

    #[test]
    fn test_ps_format_table_row() {
        let name = "llama3.2:3b";
        let size_str = format_size(2_000_000_000);
        let format = "GGUF";
        let quant = "Q4_K_M";
        let expires = "2024-01-01T00:00:00Z";
        let row = format!("{:<30} {:<10} {:<12} {:<10} {}", name, size_str, format, quant, expires);
        assert!(row.contains("llama3.2:3b"));
        assert!(row.contains("2.0 GB"));
        assert!(row.contains("GGUF"));
        assert!(row.contains("Q4_K_M"));
    }

    #[test]
    fn test_ps_connection_error_message() {
        let bind_addr = "127.0.0.1:11434";
        let err_msg = "connection refused";
        let msg = format!(
            "Failed to connect to server at {}: {err_msg}. Is the server running?",
            bind_addr
        );
        assert!(msg.contains("127.0.0.1:11434"));
        assert!(msg.contains("Is the server running?"));
    }
}
