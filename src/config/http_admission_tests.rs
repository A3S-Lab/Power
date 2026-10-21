#![cfg(test)]
use super::*;
#[test]
fn finite_http_defaults_and_acl_roundtrip() {
    let config = acl::deserialize("host = \"127.0.0.1\"").unwrap();
    assert_eq!(config.max_concurrent_requests, 1);
    assert_eq!(config.max_queued_requests, 8);
    assert_eq!(config.queue_timeout_ms, 1_000);
    config.validate().unwrap();
    let encoded = config.to_acl().unwrap();
    let decoded = acl::deserialize(&encoded).unwrap();
    assert_eq!(decoded.max_queued_requests, 8);
    assert_eq!(decoded.queue_timeout_ms, 1_000);
}
#[test]
fn invalid_http_policy_is_refused() {
    for source in [
        "max_concurrent_requests = 0",
        "max_concurrent_requests = 100001",
        "max_queued_requests = 100001",
        "queue_timeout_ms = 0",
        "queue_timeout_ms = 300001",
    ] {
        let config = acl::deserialize(source).unwrap();
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("HTTP admission"));
    }
    acl::deserialize("max_queued_requests = 0")
        .unwrap()
        .validate()
        .unwrap();
}
