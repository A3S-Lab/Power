#[derive(Debug, thiserror::Error)]
pub enum PowerError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    #[error("Download failed for {model}: {source}")]
    DownloadFailed {
        model: String,
        source: reqwest::Error,
    },

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Invalid model format: {0}")]
    InvalidFormat(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Server error: {0}")]
    Server(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("TOML deserialization error: {0}")]
    TomlDe(#[from] toml::de::Error),

    #[error("TOML serialization error: {0}")]
    TomlSer(#[from] toml::ser::Error),
}

pub type Result<T> = std::result::Result<T, PowerError>;

impl From<PowerError> for axum::response::Response {
    fn from(err: PowerError) -> Self {
        use axum::http::StatusCode;
        use axum::response::IntoResponse;

        let (status, message) = match &err {
            PowerError::ModelNotFound(_) => (StatusCode::NOT_FOUND, err.to_string()),
            PowerError::BackendNotAvailable(_) => {
                (StatusCode::SERVICE_UNAVAILABLE, err.to_string())
            }
            PowerError::InvalidFormat(_) => (StatusCode::BAD_REQUEST, err.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
        };

        let body = serde_json::json!({ "error": message });
        (status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_not_found_display() {
        let err = PowerError::ModelNotFound("llama3".to_string());
        assert_eq!(err.to_string(), "Model not found: llama3");
    }

    #[test]
    fn test_backend_not_available_display() {
        let err = PowerError::BackendNotAvailable("no GPU".to_string());
        assert_eq!(err.to_string(), "Backend not available: no GPU");
    }

    #[test]
    fn test_inference_failed_display() {
        let err = PowerError::InferenceFailed("context too long".to_string());
        assert_eq!(err.to_string(), "Inference failed: context too long");
    }

    #[test]
    fn test_invalid_format_display() {
        let err = PowerError::InvalidFormat("bin".to_string());
        assert_eq!(err.to_string(), "Invalid model format: bin");
    }

    #[test]
    fn test_server_error_display() {
        let err = PowerError::Server("bind failed".to_string());
        assert_eq!(err.to_string(), "Server error: bind failed");
    }

    #[test]
    fn test_config_error_display() {
        let err = PowerError::Config("invalid toml".to_string());
        assert_eq!(err.to_string(), "Configuration error: invalid toml");
    }

    #[test]
    fn test_io_error_from() {
        let io_err = std::io::Error::other("disk full");
        let err = PowerError::from(io_err);
        assert!(err.to_string().contains("disk full"));
    }

    #[test]
    fn test_serialization_error_from() {
        let json_err = serde_json::from_str::<String>("not json").unwrap_err();
        let err = PowerError::from(json_err);
        assert!(err.to_string().contains("Serialization error"));
    }

    #[test]
    fn test_error_to_response_model_not_found() {
        use axum::http::StatusCode;
        use axum::response::Response;

        let err = PowerError::ModelNotFound("test".to_string());
        let resp: Response = err.into();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_error_to_response_backend_unavailable() {
        use axum::http::StatusCode;
        use axum::response::Response;

        let err = PowerError::BackendNotAvailable("disabled".to_string());
        let resp: Response = err.into();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_error_to_response_invalid_format() {
        use axum::http::StatusCode;
        use axum::response::Response;

        let err = PowerError::InvalidFormat("bad".to_string());
        let resp: Response = err.into();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_error_to_response_internal() {
        use axum::http::StatusCode;
        use axum::response::Response;

        let err = PowerError::Server("crash".to_string());
        let resp: Response = err.into();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
