#![cfg(test)]
use super::*;
use std::time::Duration;
#[tokio::test(start_paused = true)]
async fn bounded_fifo_deadline_and_rejection_are_finite() {
    let metrics = Arc::new(Metrics::new());
    let limiter = Arc::new(ConcurrencyLimiter::new_bounded(
        1,
        1,
        Duration::from_millis(50),
        metrics.clone(),
    ));
    let held = limiter.acquire_http().await.unwrap();
    let other = limiter.clone();
    let waiter = tokio::spawn(async move { other.acquire_http().await });
    tokio::task::yield_now().await;
    assert_eq!(limiter.snapshot().waiting, 1);
    assert!(matches!(
        limiter.acquire_http().await,
        Err(AdmissionError::QueueFull { maximum: 1 })
    ));
    tokio::time::advance(Duration::from_millis(51)).await;
    assert!(matches!(
        waiter.await.unwrap(),
        Err(AdmissionError::DeadlineExceeded)
    ));
    assert_eq!(limiter.snapshot().waiting, 0);
    assert_eq!(metrics.waiting_requests(), 0);
    drop(held);
    let next = limiter.acquire_http().await.unwrap();
    assert_eq!(metrics.running_requests(), 1);
    drop(next);
    assert_eq!(metrics.running_requests(), 0);
}
#[tokio::test(start_paused = true)]
async fn queued_cancellation_releases_slot_and_fifo_order() {
    let limiter = Arc::new(ConcurrencyLimiter::new_bounded(
        1,
        2,
        Duration::from_secs(10),
        Arc::new(Metrics::new()),
    ));
    let held = limiter.acquire_http().await.unwrap();
    let first_limiter = limiter.clone();
    let first = tokio::spawn(async move { first_limiter.acquire_http().await });
    tokio::task::yield_now().await;
    let second_limiter = limiter.clone();
    let second = tokio::spawn(async move { second_limiter.acquire_http().await });
    tokio::task::yield_now().await;
    assert_eq!(limiter.snapshot().waiting, 2);
    first.abort();
    let _ = first.await;
    assert_eq!(limiter.snapshot().waiting, 1);
    drop(held);
    let second_permit = second.await.unwrap().unwrap();
    assert_eq!(limiter.snapshot().active, 1);
    drop(second_permit);
    assert_eq!(limiter.snapshot().active, 0);
}
#[tokio::test]
async fn legacy_unbounded_acquire_remains_available() {
    let limiter = ConcurrencyLimiter::new(0, Arc::new(Metrics::new()));
    let first = limiter.acquire().await;
    let second = limiter.acquire_http().await.unwrap();
    assert_eq!(limiter.snapshot().active_limit, None);
    drop((first, second));
}
#[tokio::test(start_paused = true)]
async fn admission_preserves_fifo_without_bypassing_waiters() {
    let limiter = Arc::new(ConcurrencyLimiter::new_bounded(
        1,
        2,
        Duration::from_secs(10),
        Arc::new(Metrics::new()),
    ));
    let held = limiter.acquire_http().await.unwrap();
    let first_limiter = limiter.clone();
    let first = tokio::spawn(async move { first_limiter.acquire_http().await });
    tokio::task::yield_now().await;
    let second_limiter = limiter.clone();
    let second = tokio::spawn(async move { second_limiter.acquire_http().await });
    tokio::task::yield_now().await;
    drop(held);
    let first = first.await.unwrap().unwrap();
    assert!(
        !second.is_finished(),
        "second waiter cannot bypass first holder"
    );
    drop(first);
    drop(second.await.unwrap().unwrap());
    assert_eq!(limiter.snapshot().active, 0);
}
#[tokio::test(start_paused = true)]
#[serial_test::serial]
async fn all_http_inference_endpoints_share_queue_and_timeout_refusals() {
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::util::ServiceExt;
    let directory = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", directory.path());
    for (waiting, expected) in [(0, "admission_queue_full"), (1, "admission_queue_timeout")] {
        let mut state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");
        state.limiter = Arc::new(ConcurrencyLimiter::new_bounded(
            1,
            waiting,
            Duration::from_millis(10),
            state.metrics.clone(),
        ));
        let held = state.limiter.acquire_http().await.unwrap();
        for (endpoint, body) in [
            (
                "chat/completions",
                serde_json::json!({"model":"test","messages":[{"role":"user","content":"hi"}]}),
            ),
            (
                "completions",
                serde_json::json!({"model":"test","prompt":"hi"}),
            ),
            (
                "embeddings",
                serde_json::json!({"model":"test","input":"hi"}),
            ),
            (
                "systemone",
                serde_json::json!({"model":"test","state":"hi","questions":{"topic":{"type":"choice","instructions":"pick","criteria":{"a":"one","b":"two"}}}}),
            ),
        ] {
            let response = crate::server::router::build(state.clone())
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(format!("/v1/{endpoint}"))
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::TOO_MANY_REQUESTS,
                "{endpoint}"
            );
            let bytes = axum::body::to_bytes(response.into_body(), 8192)
                .await
                .unwrap();
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["error"]["code"], expected, "{endpoint}");
            assert_eq!(
                state.metrics.active_requests(),
                0,
                "{endpoint} leaked active gauge"
            );
            assert_eq!(state.limiter.snapshot().waiting, 0);
        }
        drop(held);
        assert_eq!(state.metrics.running_requests(), 0);
    }
    std::env::remove_var("A3S_POWER_HOME");
}
