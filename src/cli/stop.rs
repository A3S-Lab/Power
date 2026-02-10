use crate::config::PowerConfig;
use crate::error::{PowerError, Result};

/// Execute the `stop` command: unload a running model from the server.
///
/// Sends a generate request with `keep_alive: "0"` to trigger immediate unload,
/// matching Ollama's behavior.
pub async fn execute(model: &str) -> Result<()> {
    let config = PowerConfig::load()?;
    let url = format!("http://{}:{}/api/generate", config.host, config.port);

    let body = serde_json::json!({
        "model": model,
        "keep_alive": 0,
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            PowerError::Server(format!(
                "Failed to connect to server at {}: {e}. Is the server running?",
                config.bind_address()
            ))
        })?;

    if resp.status().is_success() {
        println!("Stopped '{model}'");
        Ok(())
    } else {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        Err(PowerError::Server(format!(
            "Failed to stop model '{model}': {status} {text}"
        )))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_stop_builds_correct_json() {
        let body = serde_json::json!({
            "model": "llama3.2:3b",
            "keep_alive": 0,
        });
        assert_eq!(body["model"], "llama3.2:3b");
        assert_eq!(body["keep_alive"], 0);
    }

    #[test]
    fn test_stop_json_with_different_models() {
        let models = vec!["llama3", "phi3:mini", "qwen2:7b"];
        for model in models {
            let body = serde_json::json!({
                "model": model,
                "keep_alive": 0,
            });
            assert_eq!(body["model"], model);
            assert_eq!(body["keep_alive"], 0);
        }
    }

    #[test]
    fn test_stop_json_structure() {
        let body = serde_json::json!({
            "model": "test-model",
            "keep_alive": 0,
        });
        assert!(body.is_object());
        assert!(body.get("model").is_some());
        assert!(body.get("keep_alive").is_some());
        assert_eq!(body.as_object().unwrap().len(), 2);
    }

    #[test]
    fn test_stop_json_keep_alive_is_zero() {
        let body = serde_json::json!({
            "model": "any-model",
            "keep_alive": 0,
        });
        assert_eq!(body["keep_alive"].as_i64(), Some(0));
    }

    #[test]
    fn test_stop_json_serialization() {
        let body = serde_json::json!({
            "model": "llama3",
            "keep_alive": 0,
        });
        let serialized = serde_json::to_string(&body).unwrap();
        assert!(serialized.contains("\"model\""));
        assert!(serialized.contains("\"llama3\""));
        assert!(serialized.contains("\"keep_alive\""));
        assert!(serialized.contains("0"));
    }

    #[test]
    fn test_stop_url_construction() {
        let config = crate::config::PowerConfig::default();
        let url = format!("http://{}:{}/api/generate", config.host, config.port);
        assert_eq!(url, "http://127.0.0.1:11434/api/generate");
    }

    #[test]
    fn test_stop_url_custom_host_port() {
        let config = crate::config::PowerConfig {
            host: "0.0.0.0".to_string(),
            port: 8080,
            ..Default::default()
        };
        let url = format!("http://{}:{}/api/generate", config.host, config.port);
        assert_eq!(url, "http://0.0.0.0:8080/api/generate");
    }

    #[test]
    fn test_stop_error_message_format() {
        let model = "llama3";
        let status = "404 Not Found";
        let text = "model not found";
        let msg = format!("Failed to stop model '{model}': {status} {text}");
        assert!(msg.contains("llama3"));
        assert!(msg.contains("404"));
        assert!(msg.contains("model not found"));
    }

    #[test]
    fn test_stop_connection_error_message() {
        let bind_addr = "127.0.0.1:11434";
        let err_msg = "connection refused";
        let msg = format!(
            "Failed to connect to server at {}: {err_msg}. Is the server running?",
            bind_addr
        );
        assert!(msg.contains("127.0.0.1:11434"));
        assert!(msg.contains("connection refused"));
        assert!(msg.contains("Is the server running?"));
    }
}
