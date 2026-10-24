use super::proxy_chat_usage;
use serde_json::json;

#[test]
fn missing_or_null_usage_is_not_an_exact_count() {
    assert_eq!(proxy_chat_usage(&json!({})).unwrap(), None);
    assert_eq!(proxy_chat_usage(&json!({"usage": null})).unwrap(), None);
}

#[test]
fn exact_integer_usage_preserves_counts_and_ignores_unrelated_fields() {
    let usage = json!({"prompt_tokens": 17, "completion_tokens": 25, "total_tokens": 42});
    assert_eq!(
        proxy_chat_usage(&json!({"usage": usage, "choices": []})).unwrap(),
        Some(usage)
    );
    assert_eq!(
        proxy_chat_usage(&json!({"usage": {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
        }}))
        .unwrap(),
        Some(json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}))
    );
}

#[test]
fn partial_fractional_negative_inconsistent_and_overflowed_counts_refuse() {
    for usage in [
        json!({}),
        json!({"prompt_tokens": 17, "completion_tokens": 25}),
        json!({"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0}),
        json!({"prompt_tokens": 17.5, "completion_tokens": 25, "total_tokens": 42}),
        json!({"prompt_tokens": "17", "completion_tokens": 25, "total_tokens": 42}),
        json!({"prompt_tokens": 17, "completion_tokens": 25, "total_tokens": 41}),
        json!({"prompt_tokens": u64::MAX, "completion_tokens": 1, "total_tokens": 0}),
    ] {
        assert!(proxy_chat_usage(&json!({"usage": usage})).is_err());
    }
}
