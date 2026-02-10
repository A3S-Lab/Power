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
}
