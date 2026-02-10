use crate::config::PowerConfig;
use crate::error::Result;
use crate::server;

/// Execute the `serve` command: start the HTTP server.
pub async fn execute(host: &str, port: u16) -> Result<()> {
    let mut config = PowerConfig::load()?;

    // Override config with CLI arguments
    config.host = host.to_string();
    config.port = port;

    println!("A3S Power server starting...");
    println!("Listening on http://{}:{}", config.host, config.port);
    println!("Press Ctrl+C to stop");

    server::start(config).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_override_logic() {
        let mut config = PowerConfig::default();
        let host = "0.0.0.0";
        let port = 8080;

        // Simulate what execute() does
        config.host = host.to_string();
        config.port = port;

        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
    }

    #[test]
    fn test_config_override_preserves_other_fields() {
        let mut config = PowerConfig {
            host: "127.0.0.1".to_string(),
            port: 11434,
            max_loaded_models: 5,
            keep_alive: "10m".to_string(),
            ..Default::default()
        };

        // Override only host and port
        config.host = "0.0.0.0".to_string();
        config.port = 9000;

        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 9000);
        assert_eq!(config.max_loaded_models, 5);
        assert_eq!(config.keep_alive, "10m");
    }
}
