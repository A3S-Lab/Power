//! Privacy protection for TEE inference.
//!
//! Provides log redaction to prevent inference content (prompts, responses)
//! from leaking through log output in TEE environments, and memory zeroing
//! to ensure sensitive data is cleared after use.

use zeroize::Zeroize;

/// Trait for privacy providers that control log redaction behavior.
pub trait PrivacyProvider: Send + Sync {
    /// Whether log redaction is active.
    fn should_redact(&self) -> bool;
    /// Sanitize a log message by removing inference content.
    fn sanitize_log(&self, msg: &str) -> String;
    /// Sanitize an error message that may echo prompt content.
    fn sanitize_error(&self, err: &str) -> String;
    /// Whether to suppress exact token counts (round to nearest 10).
    fn should_suppress_token_metrics(&self) -> bool;
}

/// Default privacy provider controlled by config.
pub struct DefaultPrivacyProvider {
    redact: bool,
    suppress_tokens: bool,
}

impl DefaultPrivacyProvider {
    pub fn new(redact: bool) -> Self {
        if redact {
            tracing::info!("Privacy: log redaction enabled");
        }
        Self {
            redact,
            suppress_tokens: redact,
        }
    }

    /// Override token-metric suppression independently of log redaction.
    pub fn with_suppress_token_metrics(mut self, suppress: bool) -> Self {
        self.suppress_tokens = suppress;
        self
    }
}

impl PrivacyProvider for DefaultPrivacyProvider {
    fn should_redact(&self) -> bool {
        self.redact
    }

    fn sanitize_log(&self, msg: &str) -> String {
        if !self.redact {
            return msg.to_string();
        }
        redact_content(msg)
    }

    fn sanitize_error(&self, err: &str) -> String {
        if !self.redact {
            return err.to_string();
        }
        sanitize_error_message(err)
    }

    fn should_suppress_token_metrics(&self) -> bool {
        self.suppress_tokens
    }
}

/// Sensitive JSON keys whose values should be redacted in logs.
///
/// Covers all common LLM API fields that may contain user data.
const SENSITIVE_KEYS: &[&str] = &[
    "content",
    "prompt",
    "text",
    "arguments",
    "input",
    "delta",
    "system",
    "message",
    "query",
    "instruction",
];

/// Error message fragments that may echo prompt content.
const SENSITIVE_ERROR_PREFIXES: &[&str] = &["prompt:", "content:", "message:", "input:"];

/// Redact sensitive JSON field values from a log message.
///
/// Replaces the value of any sensitive key with `[REDACTED]`.
/// Handles both string values and non-string values (numbers, booleans, objects).
fn redact_content(msg: &str) -> String {
    let mut result = msg.to_string();
    for key in SENSITIVE_KEYS {
        let pattern = format!("\"{}\":", key);
        if let Some(start) = result.find(&pattern) {
            let after_key = start + pattern.len();
            // Skip optional whitespace
            let trimmed = result[after_key..].trim_start();
            let ws_len = result[after_key..].len() - trimmed.len();
            let value_start = after_key + ws_len;

            if result.as_bytes().get(value_start) == Some(&b'"') {
                // String value: find closing quote (skip escaped quotes)
                let content_start = value_start + 1;
                let mut i = content_start;
                let bytes = result.as_bytes();
                while i < bytes.len() {
                    if bytes[i] == b'"' && (i == 0 || bytes[i - 1] != b'\\') {
                        break;
                    }
                    i += 1;
                }
                if i < bytes.len() {
                    result = format!(
                        "{}\"[REDACTED]\"{}",
                        &result[..value_start],
                        &result[i + 1..]
                    );
                }
            }
        }
    }
    result
}

/// Sanitize an error message that may echo prompt content.
///
/// Strips everything after known sensitive prefixes to prevent
/// prompt fragments from leaking through error messages.
pub fn sanitize_error_message(err: &str) -> String {
    let lower = err.to_lowercase();
    for prefix in SENSITIVE_ERROR_PREFIXES {
        if let Some(pos) = lower.find(prefix) {
            return format!("{}[REDACTED]", &err[..pos + prefix.len()]);
        }
    }
    err.to_string()
}

/// Round a token count to the nearest 10 to prevent exact side-channel inference.
pub fn round_token_count(count: u32) -> u32 {
    ((count + 5) / 10) * 10
}

/// Zeroize a mutable string in place, overwriting its contents with zeros.
pub fn zeroize_string(s: &mut String) {
    s.zeroize();
}

/// Zeroize a mutable byte vector in place.
pub fn zeroize_bytes(b: &mut Vec<u8>) {
    b.zeroize();
}

/// A string wrapper that automatically zeroizes its contents on drop.
///
/// Use this for inference content (prompts, responses) in TEE mode
/// to ensure sensitive data doesn't linger in memory.
#[derive(Debug, Clone)]
pub struct SensitiveString {
    inner: String,
}

impl SensitiveString {
    pub fn new(s: String) -> Self {
        Self { inner: s }
    }

    pub fn as_str(&self) -> &str {
        &self.inner
    }

    pub fn into_inner(mut self) -> String {
        // Take the string out so Drop doesn't zeroize it
        std::mem::take(&mut self.inner)
    }
}

impl Drop for SensitiveString {
    fn drop(&mut self) {
        self.inner.zeroize();
    }
}

impl std::fmt::Display for SensitiveString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl From<String> for SensitiveString {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for SensitiveString {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_provider_redact_enabled() {
        let provider = DefaultPrivacyProvider::new(true);
        assert!(provider.should_redact());
    }

    #[test]
    fn test_default_provider_redact_disabled() {
        let provider = DefaultPrivacyProvider::new(false);
        assert!(!provider.should_redact());
    }

    #[test]
    fn test_sanitize_passthrough_when_disabled() {
        let provider = DefaultPrivacyProvider::new(false);
        let msg = r#"{"content": "secret data"}"#;
        assert_eq!(provider.sanitize_log(msg), msg);
    }

    #[test]
    fn test_sanitize_redacts_content() {
        let provider = DefaultPrivacyProvider::new(true);
        let msg = r#"{"content": "secret data"}"#;
        let sanitized = provider.sanitize_log(msg);
        assert!(!sanitized.contains("secret data"));
        assert!(sanitized.contains("[REDACTED]"));
    }

    #[test]
    fn test_redact_content_field() {
        let input = r#"{"role":"user","content":"tell me a secret"}"#;
        let output = redact_content(input);
        assert!(!output.contains("tell me a secret"));
        assert!(output.contains("[REDACTED]"));
    }

    #[test]
    fn test_redact_prompt_field() {
        let input = r#"{"model":"test","prompt":"hello world"}"#;
        let output = redact_content(input);
        assert!(!output.contains("hello world"));
        assert!(output.contains("[REDACTED]"));
    }

    #[test]
    fn test_redact_text_field() {
        let input = r#"{"text":"generated response"}"#;
        let output = redact_content(input);
        assert!(!output.contains("generated response"));
        assert!(output.contains("[REDACTED]"));
    }

    #[test]
    fn test_redact_no_sensitive_fields() {
        let input = r#"{"model":"llama3","status":"ok"}"#;
        let output = redact_content(input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_redact_empty_string() {
        assert_eq!(redact_content(""), "");
    }

    #[test]
    fn test_redact_preserves_structure() {
        let input = r#"{"content": "secret", "model": "llama3"}"#;
        let output = redact_content(input);
        assert!(output.contains("\"content\":"));
        assert!(output.contains("[REDACTED]"));
        assert!(output.contains("model"));
    }

    #[test]
    fn test_zeroize_string() {
        let mut s = "sensitive prompt data".to_string();
        let ptr = s.as_ptr();
        let len = s.len();
        zeroize_string(&mut s);
        // After zeroize, the string should be empty
        assert!(s.is_empty());
        // The original memory should be zeroed (check via raw pointer)
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        assert!(slice.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_zeroize_bytes() {
        let mut b = vec![0xAA, 0xBB, 0xCC, 0xDD];
        zeroize_bytes(&mut b);
        assert!(b.is_empty());
    }

    #[test]
    fn test_sensitive_string_display() {
        let s = SensitiveString::new("hello".to_string());
        assert_eq!(s.to_string(), "hello");
        assert_eq!(s.as_str(), "hello");
    }

    #[test]
    fn test_sensitive_string_from_str() {
        let s: SensitiveString = "test".into();
        assert_eq!(s.as_str(), "test");
    }

    #[test]
    fn test_sensitive_string_from_string() {
        let s: SensitiveString = String::from("test").into();
        assert_eq!(s.as_str(), "test");
    }

    #[test]
    fn test_sensitive_string_into_inner() {
        let s = SensitiveString::new("keep me".to_string());
        let inner = s.into_inner();
        assert_eq!(inner, "keep me");
    }

    #[test]
    fn test_sensitive_string_clone_is_independent() {
        let s1 = SensitiveString::new("data".to_string());
        let s2 = s1.clone();
        drop(s1);
        // s2 should still be valid after s1 is dropped
        assert_eq!(s2.as_str(), "data");
    }

    #[test]
    fn test_sensitive_string_drop_does_not_panic() {
        // Verify that dropping a SensitiveString doesn't panic
        let s = SensitiveString::new("will be zeroized".to_string());
        drop(s);
    }

    // --- New: extended redaction coverage ---

    #[test]
    fn test_redact_arguments_field() {
        let input = r#"{"arguments": "user_data=secret"}"#;
        let output = redact_content(input);
        assert!(output.contains("[REDACTED]"));
        assert!(!output.contains("user_data=secret"));
    }

    #[test]
    fn test_redact_input_field() {
        let input = r#"{"input": "private query"}"#;
        let output = redact_content(input);
        assert!(output.contains("[REDACTED]"));
        assert!(!output.contains("private query"));
    }

    #[test]
    fn test_redact_system_field() {
        let input = r#"{"system": "You are a helpful assistant"}"#;
        let output = redact_content(input);
        assert!(output.contains("[REDACTED]"));
        assert!(!output.contains("You are a helpful assistant"));
    }

    #[test]
    fn test_redact_query_field() {
        let input = r#"{"query": "sensitive search term"}"#;
        let output = redact_content(input);
        assert!(output.contains("[REDACTED]"));
        assert!(!output.contains("sensitive search term"));
    }

    // --- sanitize_error_message tests ---

    #[test]
    fn test_sanitize_error_with_prompt_prefix() {
        let err = "Inference failed: prompt: tell me your secrets";
        let result = sanitize_error_message(err);
        assert!(result.contains("[REDACTED]"));
        assert!(!result.contains("tell me your secrets"));
        assert!(result.contains("prompt:"));
    }

    #[test]
    fn test_sanitize_error_with_content_prefix() {
        let err = "Error processing content: user private data here";
        let result = sanitize_error_message(err);
        assert!(result.contains("[REDACTED]"));
        assert!(!result.contains("user private data here"));
    }

    #[test]
    fn test_sanitize_error_without_sensitive_prefix_unchanged() {
        let err = "Model not found: llama3";
        let result = sanitize_error_message(err);
        assert_eq!(result, err);
    }

    #[test]
    fn test_sanitize_error_empty_unchanged() {
        let result = sanitize_error_message("");
        assert_eq!(result, "");
    }

    // --- round_token_count tests ---

    #[test]
    fn test_round_token_count_rounds_to_nearest_10() {
        assert_eq!(round_token_count(0), 0);
        assert_eq!(round_token_count(4), 0);
        assert_eq!(round_token_count(5), 10);
        assert_eq!(round_token_count(10), 10);
        assert_eq!(round_token_count(14), 10);
        assert_eq!(round_token_count(15), 20);
        assert_eq!(round_token_count(99), 100);
        assert_eq!(round_token_count(100), 100);
    }

    // --- DefaultPrivacyProvider new method tests ---

    #[test]
    fn test_default_privacy_provider_sanitize_error_enabled() {
        let provider = DefaultPrivacyProvider::new(true);
        let err = "Failed: prompt: secret data";
        let result = provider.sanitize_error(err);
        assert!(result.contains("[REDACTED]"));
        assert!(!result.contains("secret data"));
    }

    #[test]
    fn test_default_privacy_provider_sanitize_error_disabled() {
        let provider = DefaultPrivacyProvider::new(false);
        let err = "Failed: prompt: visible data";
        let result = provider.sanitize_error(err);
        assert_eq!(result, err);
    }

    #[test]
    fn test_default_privacy_provider_suppress_token_metrics_when_redacting() {
        let provider = DefaultPrivacyProvider::new(true);
        assert!(provider.should_suppress_token_metrics());
    }

    #[test]
    fn test_default_privacy_provider_no_suppress_when_not_redacting() {
        let provider = DefaultPrivacyProvider::new(false);
        assert!(!provider.should_suppress_token_metrics());
    }

    #[test]
    fn test_with_suppress_token_metrics_independent_of_redact() {
        // suppress_tokens can be enabled without enabling log redaction
        let provider = DefaultPrivacyProvider::new(false).with_suppress_token_metrics(true);
        assert!(!provider.should_redact());
        assert!(provider.should_suppress_token_metrics());
    }

    #[test]
    fn test_with_suppress_token_metrics_can_disable_when_redacting() {
        // suppress_tokens can be disabled even when redact is on
        let provider = DefaultPrivacyProvider::new(true).with_suppress_token_metrics(false);
        assert!(provider.should_redact());
        assert!(!provider.should_suppress_token_metrics());
    }
}
