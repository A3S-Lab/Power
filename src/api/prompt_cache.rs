use sha2::{Digest, Sha256};

pub const MAX_PROMPT_CACHE_KEY_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCacheEndpoint {
    Chat,
    Completion,
}

impl PromptCacheEndpoint {
    fn label(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Completion => "completion",
        }
    }
}

/// Validate the caller-facing cache identifier before it reaches logs or a backend.
pub fn validate_prompt_cache_key(key: Option<&str>) -> Result<(), String> {
    let Some(key) = key else {
        return Ok(());
    };
    if key.is_empty()
        || key.len() > MAX_PROMPT_CACHE_KEY_BYTES
        || key.trim() != key
        || key.chars().any(char::is_control)
    {
        return Err(format!(
            "prompt_cache_key must be a non-empty identifier of at most {MAX_PROMPT_CACHE_KEY_BYTES} UTF-8 bytes without surrounding whitespace or control characters"
        ));
    }
    Ok(())
}

/// Derive an opaque backend key scoped to identity, endpoint, and model.
///
/// Length prefixes make the domain separation unambiguous. The caller's raw
/// identifier is never stored in the backend cache or emitted in logs.
pub fn scoped_prompt_cache_key(
    auth_id: Option<&str>,
    endpoint: PromptCacheEndpoint,
    model: &str,
    caller_key: &str,
) -> String {
    let mut digest = Sha256::new();
    hash_part(&mut digest, b"a3s.power.prompt-cache-key.v1");
    match auth_id {
        Some(auth_id) => {
            hash_part(&mut digest, b"authenticated");
            hash_part(&mut digest, auth_id.as_bytes());
        }
        None => hash_part(&mut digest, b"anonymous"),
    }
    hash_part(&mut digest, endpoint.label().as_bytes());
    hash_part(&mut digest, model.as_bytes());
    hash_part(&mut digest, caller_key.as_bytes());
    format!("a3s-pcache-v1:{}", hex::encode(digest.finalize()))
}

fn hash_part(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_be_bytes());
    digest.update(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_key_is_stable_and_does_not_expose_input() {
        let key = scoped_prompt_cache_key(
            Some("key-0"),
            PromptCacheEndpoint::Chat,
            "qwen",
            "private-agent-prefix",
        );
        assert_eq!(
            key,
            scoped_prompt_cache_key(
                Some("key-0"),
                PromptCacheEndpoint::Chat,
                "qwen",
                "private-agent-prefix"
            )
        );
        assert!(key.starts_with("a3s-pcache-v1:"));
        assert!(!key.contains("private-agent-prefix"));
        assert_eq!(key.len(), "a3s-pcache-v1:".len() + 64);
    }

    #[test]
    fn scoped_key_isolated_by_identity_model_and_endpoint() {
        let base =
            scoped_prompt_cache_key(Some("key-0"), PromptCacheEndpoint::Chat, "qwen", "shared");
        assert_ne!(
            base,
            scoped_prompt_cache_key(Some("key-1"), PromptCacheEndpoint::Chat, "qwen", "shared")
        );
        assert_ne!(
            base,
            scoped_prompt_cache_key(Some("key-0"), PromptCacheEndpoint::Chat, "llama", "shared")
        );
        assert_ne!(
            base,
            scoped_prompt_cache_key(
                Some("key-0"),
                PromptCacheEndpoint::Completion,
                "qwen",
                "shared"
            )
        );
    }

    #[test]
    fn validation_rejects_ambiguous_or_unbounded_keys() {
        for key in ["", " leading", "trailing ", "line\nbreak"] {
            assert!(validate_prompt_cache_key(Some(key)).is_err(), "key={key:?}");
        }
        let oversized = "x".repeat(MAX_PROMPT_CACHE_KEY_BYTES + 1);
        assert!(validate_prompt_cache_key(Some(&oversized)).is_err());
        assert!(validate_prompt_cache_key(Some("agent-prefix-v1")).is_ok());
        assert!(validate_prompt_cache_key(None).is_ok());
    }
}
