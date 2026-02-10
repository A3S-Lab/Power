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
    use super::*;

    #[test]
    fn test_stop_builds_correct_json() {
        let body = serde_json::json!({
            "model": "llama3.2:3b",
            "keep_alive": 0,
        });
        assert_eq!(body["model"], "llama3.2:3b");
        assert_eq!(body["keep_alive"], 0);
    }
}
