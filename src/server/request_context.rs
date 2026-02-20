use std::time::Instant;

/// Per-request context for isolation and audit tracking.
///
/// Every inference request gets a unique `RequestContext` that carries
/// the request ID, authenticated identity, and timing information.
/// This context is passed to backends for request-scoped cleanup.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Unique request ID (UUID v4).
    pub request_id: String,
    /// Authenticated user/key identifier (if auth enabled).
    pub auth_id: Option<String>,
    /// Timestamp of request arrival.
    pub created_at: Instant,
}

impl RequestContext {
    /// Create a new request context with a generated UUID.
    pub fn new(auth_id: Option<String>) -> Self {
        Self {
            request_id: uuid::Uuid::new_v4().to_string(),
            auth_id,
            created_at: Instant::now(),
        }
    }

    /// Create a request context with a specific request ID.
    /// Used when the client provides an `X-Request-ID` header.
    pub fn with_id(request_id: String, auth_id: Option<String>) -> Self {
        Self {
            request_id,
            auth_id,
            created_at: Instant::now(),
        }
    }

    /// Elapsed time since this request was created.
    pub fn elapsed(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_context_has_unique_id() {
        let ctx1 = RequestContext::new(None);
        let ctx2 = RequestContext::new(None);
        assert_ne!(ctx1.request_id, ctx2.request_id);
    }

    #[test]
    fn test_request_context_carries_auth_id() {
        let ctx = RequestContext::new(Some("user-123".to_string()));
        assert_eq!(ctx.auth_id.as_deref(), Some("user-123"));
    }

    #[test]
    fn test_request_context_no_auth_id() {
        let ctx = RequestContext::new(None);
        assert!(ctx.auth_id.is_none());
    }

    #[test]
    fn test_request_context_with_id() {
        let ctx = RequestContext::with_id("custom-id-42".to_string(), None);
        assert_eq!(ctx.request_id, "custom-id-42");
    }

    #[test]
    fn test_request_context_elapsed() {
        let ctx = RequestContext::new(None);
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(ctx.elapsed().as_millis() >= 4);
    }
}
