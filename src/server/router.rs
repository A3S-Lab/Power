use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use super::state::AppState;
use crate::api;
use crate::server::{auth, metrics};

/// Shared state for the rate limiter middleware.
#[derive(Clone)]
struct RateLimiter {
    /// Max requests per second.
    rps: u64,
    /// Max concurrent requests (0 = unlimited).
    max_concurrent: u64,
    inner: Arc<Mutex<RateLimiterInner>>,
}

struct RateLimiterInner {
    /// Token bucket: tokens available.
    tokens: f64,
    /// Last refill time.
    last_refill: Instant,
    /// Current concurrent request count.
    concurrent: u64,
}

impl RateLimiter {
    fn new(rps: u64, max_concurrent: u64) -> Self {
        Self {
            rps,
            max_concurrent,
            inner: Arc::new(Mutex::new(RateLimiterInner {
                tokens: rps as f64,
                last_refill: Instant::now(),
                concurrent: 0,
            })),
        }
    }

    /// Try to acquire a request slot. Returns false if rate or concurrency limit exceeded.
    ///
    /// Concurrency is checked before consuming a token so that requests rejected
    /// by the concurrency limit do not silently drain the rate-limit bucket.
    fn try_acquire(&self) -> bool {
        let mut inner = self.inner.lock().unwrap();

        // Check concurrency limit first — no token consumed if concurrency is at capacity.
        if self.max_concurrent > 0 && inner.concurrent >= self.max_concurrent {
            return false;
        }

        // Refill and consume from the token bucket.
        if self.rps > 0 {
            let now = Instant::now();
            let elapsed = now.duration_since(inner.last_refill).as_secs_f64();
            inner.tokens = (inner.tokens + elapsed * self.rps as f64).min(self.rps as f64);
            inner.last_refill = now;

            if inner.tokens < 1.0 {
                return false;
            }
            inner.tokens -= 1.0;
        }

        inner.concurrent += 1;
        true
    }

    /// Release a concurrent request slot.
    fn release(&self) {
        if self.max_concurrent > 0 {
            let mut inner = self.inner.lock().unwrap();
            inner.concurrent = inner.concurrent.saturating_sub(1);
        }
    }
}

/// Middleware that enforces rate and concurrency limits.
async fn rate_limit_middleware(
    State(limiter): State<RateLimiter>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if !limiter.try_acquire() {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            axum::Json(serde_json::json!({
                "error": {
                    "message": "Too many requests",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded"
                }
            })),
        )
            .into_response();
    }
    let response = next.run(request).await;
    limiter.release();
    response
}

/// Build the complete axum Router with all API routes.
pub fn build(state: AppState) -> Router {
    let rate_limit_rps = state.config.rate_limit_rps;
    let max_concurrent = state.config.max_concurrent_requests;

    // Apply auth middleware only to /v1/* routes
    let v1_routes = api::openai::routes().layer(middleware::from_fn_with_state(
        state.clone(),
        auth::middleware,
    ));

    let mut router = Router::new()
        .route("/health", get(api::health::handler))
        .route("/metrics", get(metrics::handler))
        .nest("/v1", v1_routes)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .layer(middleware::from_fn_with_state(
            state.clone(),
            metrics::middleware,
        ))
        .layer(middleware::from_fn(request_id_middleware))
        .with_state(state);

    // Apply rate/concurrency limiting as outermost middleware when configured
    if rate_limit_rps > 0 || max_concurrent > 0 {
        let limiter = RateLimiter::new(rate_limit_rps, max_concurrent);
        router = router.layer(middleware::from_fn_with_state(
            limiter,
            rate_limit_middleware,
        ));
    }

    router
}

/// Middleware that ensures every request has an `X-Request-ID`.
///
/// If the client sends an `X-Request-ID` header, it is preserved.
/// Otherwise, a new UUID v4 is generated. The ID is added to the
/// response headers for traceability.
async fn request_id_middleware(request: Request<Body>, next: Next) -> Response {
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let mut response = next.run(request).await;

    if let Ok(value) = axum::http::HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", value);
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendRegistry;
    use crate::config::PowerConfig;
    use crate::model::registry::ModelRegistry;
    use crate::server::auth::ApiKeyAuth;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::util::ServiceExt;

    fn test_state() -> AppState {
        AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        )
    }

    fn test_state_with_auth() -> AppState {
        let auth = ApiKeyAuth::new(&["test-secret-key".to_string()]);
        test_state().with_auth(Arc::new(auth))
    }

    #[tokio::test]
    async fn test_v1_models_returns_ok() {
        let app = build(test_state());
        let req = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_unknown_route_returns_404() {
        let app = build(test_state());
        let req = Request::builder()
            .uri("/nonexistent")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_health_endpoint_returns_ok() {
        let app = build(test_state());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_endpoint_json_body() {
        let app = build(test_state());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
        assert!(json["version"].is_string());
        assert!(json["uptime_seconds"].is_number());
        assert!(json["loaded_models"].is_number());
    }

    #[tokio::test]
    async fn test_metrics_endpoint_returns_ok() {
        let app = build(test_state());
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_returns_prometheus_format() {
        let app = build(test_state());
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("# HELP power_http_requests_total"));
        assert!(text.contains("power_models_loaded"));
    }

    // --- Auth middleware tests ---

    #[tokio::test]
    async fn test_no_auth_configured_allows_all() {
        let app = build(test_state());
        let req = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_valid_bearer_token_passes() {
        let app = build(test_state_with_auth());
        let req = Request::builder()
            .uri("/v1/models")
            .header("authorization", "Bearer test-secret-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_invalid_bearer_token_rejected() {
        let app = build(test_state_with_auth());
        let req = Request::builder()
            .uri("/v1/models")
            .header("authorization", "Bearer wrong-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_missing_auth_header_rejected() {
        let app = build(test_state_with_auth());
        let req = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_health_endpoint_no_auth_required() {
        let app = build(test_state_with_auth());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_no_auth_required() {
        let app = build(test_state_with_auth());
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_failure_returns_openai_error_format() {
        let app = build(test_state_with_auth());
        let req = Request::builder()
            .uri("/v1/models")
            .header("authorization", "Bearer bad-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unauthorized");
        assert!(json["error"]["message"].as_str().is_some());
    }

    // --- Request ID middleware tests ---

    #[tokio::test]
    async fn test_response_has_request_id_header() {
        let app = build(test_state());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert!(resp.headers().contains_key("x-request-id"));
    }

    #[tokio::test]
    async fn test_client_request_id_preserved() {
        let app = build(test_state());
        let req = Request::builder()
            .uri("/health")
            .header("x-request-id", "my-custom-id-123")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.headers()
                .get("x-request-id")
                .unwrap()
                .to_str()
                .unwrap(),
            "my-custom-id-123"
        );
    }

    // --- RateLimiter unit tests ---

    #[test]
    fn test_rate_limiter_allows_within_limit() {
        let limiter = RateLimiter::new(10, 0);
        assert!(limiter.try_acquire());
        limiter.release();
    }

    #[test]
    fn test_rate_limiter_concurrency_limit_does_not_consume_token() {
        // Set max_concurrent=1, rps=2 (so bucket starts with 2 tokens).
        let limiter = RateLimiter::new(2, 1);
        // First acquire: takes a token and increments concurrent.
        assert!(limiter.try_acquire());
        // At max concurrency — should reject WITHOUT consuming another token.
        assert!(!limiter.try_acquire());
        // Token bucket should still have 1 token (not 0).
        {
            let inner = limiter.inner.lock().unwrap();
            assert!(
                inner.tokens >= 1.0,
                "concurrency rejection must not consume a rate-limit token"
            );
        }
        // After release, a new acquire should succeed.
        limiter.release();
        assert!(limiter.try_acquire());
        limiter.release();
    }

    #[test]
    fn test_rate_limiter_token_bucket_exhausted() {
        // 1 RPS, no concurrency limit.
        let limiter = RateLimiter::new(1, 0);
        assert!(limiter.try_acquire()); // uses the 1 token
        limiter.release();
        // Bucket is empty — second acquire should fail.
        assert!(!limiter.try_acquire());
    }

    #[test]
    fn test_rate_limiter_no_limits_always_allows() {
        let limiter = RateLimiter::new(0, 0);
        for _ in 0..100 {
            assert!(limiter.try_acquire());
            limiter.release();
        }
    }
}
