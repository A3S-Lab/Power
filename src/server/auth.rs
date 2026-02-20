use axum::body::Body;
use axum::extract::State;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use sha2::{Digest, Sha256};

use crate::server::audit::{AuditEvent, AuditLogger};
use crate::server::state::AppState;

/// Trait for authentication providers.
///
/// This is an extension point — implement this trait to add custom
/// authentication (e.g., JWT, OAuth). The default implementation
/// (`ApiKeyAuth`) checks Bearer tokens against SHA-256 hashes.
pub trait AuthProvider: Send + Sync {
    /// Check if the given token is valid.
    fn authenticate(&self, token: &str) -> bool;

    /// Return an identifier for the authenticated key (for audit logging).
    /// Returns `None` if the token is invalid.
    fn identify(&self, token: &str) -> Option<String>;
}

/// API key authentication using SHA-256 hash comparison.
///
/// Keys are stored as hex-encoded SHA-256 hashes. When a Bearer token
/// arrives, we hash it and compare against the stored hashes. This way
/// the raw API keys are never stored in memory or config.
pub struct ApiKeyAuth {
    /// SHA-256 hashes of valid API keys (hex-encoded, lowercase).
    key_hashes: Vec<String>,
}

impl ApiKeyAuth {
    /// Create a new `ApiKeyAuth` from a list of SHA-256 key hashes.
    ///
    /// The hashes should be hex-encoded lowercase strings. If a value
    /// doesn't look like a hex SHA-256 hash (64 chars), it's treated
    /// as a raw key and hashed automatically.
    pub fn new(keys: &[String]) -> Self {
        let key_hashes = keys
            .iter()
            .map(|k| {
                if k.len() == 64 && k.chars().all(|c| c.is_ascii_hexdigit()) {
                    // Already a hex SHA-256 hash
                    k.to_lowercase()
                } else {
                    // Raw key — hash it
                    hash_key(k)
                }
            })
            .collect();
        Self { key_hashes }
    }
}

impl AuthProvider for ApiKeyAuth {
    fn authenticate(&self, token: &str) -> bool {
        let token_hash = hash_key(token);
        self.key_hashes.iter().any(|h| h == &token_hash)
    }

    fn identify(&self, token: &str) -> Option<String> {
        let token_hash = hash_key(token);
        self.key_hashes
            .iter()
            .position(|h| h == &token_hash)
            .map(|i| format!("key-{i}"))
    }
}

/// Compute the SHA-256 hash of a key, returned as lowercase hex.
fn hash_key(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Extract the Bearer token from an Authorization header value.
fn extract_bearer(header_value: &str) -> Option<&str> {
    let trimmed = header_value.trim();
    if let Some(token) = trimmed.strip_prefix("Bearer ") {
        let token = token.trim();
        if token.is_empty() {
            None
        } else {
            Some(token)
        }
    } else {
        None
    }
}

/// Axum middleware that enforces API key authentication.
///
/// When `AppState.auth` is `Some`, requests to protected routes must
/// include a valid `Authorization: Bearer <token>` header. If auth is
/// not configured (`None`), all requests pass through.
pub async fn middleware(
    State(state): State<AppState>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    let auth = match &state.auth {
        Some(auth) => auth,
        None => return next.run(request).await,
    };

    let token = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(extract_bearer);

    match token {
        Some(token) => {
            if auth.authenticate(token) {
                // Store auth_id in request extensions for handlers to access
                if let Some(id) = auth.identify(token) {
                    request.extensions_mut().insert(AuthId(id));
                }
                next.run(request).await
            } else {
                state.metrics.increment_auth_failure();
                if let Some(ref audit) = state.audit {
                    audit.log(&AuditEvent::failure(
                        uuid::Uuid::new_v4().to_string(),
                        None,
                        "auth_failure",
                        None,
                        "invalid API key",
                    ));
                }
                unauthorized_response("Invalid API key")
            }
        }
        None => {
            state.metrics.increment_auth_failure();
            if let Some(ref audit) = state.audit {
                audit.log(&AuditEvent::failure(
                    uuid::Uuid::new_v4().to_string(),
                    None,
                    "auth_failure",
                    None,
                    "missing Authorization header",
                ));
            }
            unauthorized_response(
                "Missing or invalid Authorization header. Expected: Bearer <token>",
            )
        }
    }
}

/// Wrapper type for storing auth identity in request extensions.
#[derive(Debug, Clone)]
pub struct AuthId(pub String);

/// Build a 401 response with OpenAI-compatible error JSON.
fn unauthorized_response(message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": "unauthorized"
        }
    });
    (axum::http::StatusCode::UNAUTHORIZED, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hash(key: &str) -> String {
        hash_key(key)
    }

    #[test]
    fn test_hash_key_deterministic() {
        let h1 = hash_key("my-secret-key");
        let h2 = hash_key("my-secret-key");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn test_hash_key_different_inputs() {
        let h1 = hash_key("key-a");
        let h2 = hash_key("key-b");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_api_key_auth_with_raw_keys() {
        let auth = ApiKeyAuth::new(&["my-secret".to_string()]);
        assert!(auth.authenticate("my-secret"));
        assert!(!auth.authenticate("wrong-key"));
    }

    #[test]
    fn test_api_key_auth_with_hash() {
        let hash = test_hash("my-secret");
        let auth = ApiKeyAuth::new(&[hash]);
        assert!(auth.authenticate("my-secret"));
        assert!(!auth.authenticate("wrong-key"));
    }

    #[test]
    fn test_api_key_auth_multiple_keys() {
        let auth = ApiKeyAuth::new(&["key-a".to_string(), "key-b".to_string()]);
        assert!(auth.authenticate("key-a"));
        assert!(auth.authenticate("key-b"));
        assert!(!auth.authenticate("key-c"));
    }

    #[test]
    fn test_api_key_auth_identify() {
        let auth = ApiKeyAuth::new(&["key-a".to_string(), "key-b".to_string()]);
        assert_eq!(auth.identify("key-a"), Some("key-0".to_string()));
        assert_eq!(auth.identify("key-b"), Some("key-1".to_string()));
        assert_eq!(auth.identify("key-c"), None);
    }

    #[test]
    fn test_extract_bearer_valid() {
        assert_eq!(extract_bearer("Bearer my-token"), Some("my-token"));
        assert_eq!(extract_bearer("Bearer  spaced-token"), Some("spaced-token"));
    }

    #[test]
    fn test_extract_bearer_invalid() {
        assert_eq!(extract_bearer("Basic abc123"), None);
        assert_eq!(extract_bearer("Bearer "), None);
        assert_eq!(extract_bearer(""), None);
        assert_eq!(extract_bearer("BearerNoSpace"), None);
    }

    #[test]
    fn test_unauthorized_response_structure() {
        let resp = unauthorized_response("test error");
        assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
    }
}
