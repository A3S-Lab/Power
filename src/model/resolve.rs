use crate::error::{PowerError, Result};
use crate::model::manifest::ModelFormat;
use crate::model::ollama_registry::{self, OllamaRegistryModel};

/// A parsed model reference in `name:tag` format.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRef {
    pub name: String,
    pub tag: String,
}

impl ModelRef {
    /// Parse a model reference string. Defaults tag to "latest" if not specified.
    pub fn parse(input: &str) -> Self {
        if let Some((name, tag)) = input.split_once(':') {
            Self {
                name: name.to_string(),
                tag: tag.to_string(),
            }
        } else {
            Self {
                name: input.to_string(),
                tag: "latest".to_string(),
            }
        }
    }
}

/// Where a resolved model came from, carrying source-specific metadata.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// Direct URL or built-in registry — no extra metadata.
    Direct,
    /// Resolved from Ollama registry — carries template, system prompt, params, etc.
    OllamaRegistry(Box<OllamaRegistryModel>),
}

/// A resolved model with its download URL and source metadata.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub name: String,
    pub url: String,
    pub format: ModelFormat,
    pub source: ModelSource,
}

/// Check if the input looks like a direct URL.
pub fn is_url(input: &str) -> bool {
    input.starts_with("http://") || input.starts_with("https://")
}

/// Resolve a model name (or name:tag) to a download URL.
///
/// Resolution strategy:
/// 1. Query Ollama registry (`registry.ollama.ai`) — primary source
/// 2. Check built-in known_models.json registry — offline fallback
/// 3. Query HuggingFace API for GGUF models — last resort
/// 4. Error if not found
pub async fn resolve(model_ref: &str) -> Result<ResolvedModel> {
    let parsed = ModelRef::parse(model_ref);

    // 1. Query Ollama registry (primary)
    if let Some(resolved) = resolve_from_ollama(&parsed).await {
        return Ok(resolved);
    }

    // 2. Check built-in registry (offline fallback)
    if let Some(resolved) = resolve_from_builtin(&parsed) {
        return Ok(resolved);
    }

    // 3. Query HuggingFace API (last resort)
    if let Some(resolved) = resolve_from_huggingface(&parsed).await? {
        return Ok(resolved);
    }

    Err(PowerError::ModelNotFound(format!(
        "Could not resolve model '{}'. Try providing a direct URL instead.",
        model_ref
    )))
}

/// Attempt to resolve a model from the Ollama registry.
///
/// Returns `None` on any network or registry error (non-fatal fallthrough).
async fn resolve_from_ollama(model_ref: &ModelRef) -> Option<ResolvedModel> {
    let registry_model = ollama_registry::fetch_registry_model(&model_ref.name, &model_ref.tag)
        .await
        .ok()?;

    let url = ollama_registry::blob_url(&model_ref.name, &registry_model.model_digest);

    Some(ResolvedModel {
        name: format!("{}:{}", model_ref.name, model_ref.tag),
        url,
        format: ModelFormat::Gguf,
        source: ModelSource::OllamaRegistry(Box::new(registry_model)),
    })
}

/// Built-in model registry embedded at compile time.
fn resolve_from_builtin(model_ref: &ModelRef) -> Option<ResolvedModel> {
    let registry: serde_json::Value =
        serde_json::from_str(include_str!("known_models.json")).ok()?;

    let models = registry.get("models")?.as_array()?;

    for entry in models {
        let name = entry.get("name")?.as_str()?;
        let tag = entry.get("tag")?.as_str()?;
        let url = entry.get("url")?.as_str()?;
        let format_str = entry.get("format")?.as_str()?;

        let matches = name == model_ref.name && (tag == model_ref.tag || model_ref.tag == "latest");

        if matches {
            let format = match format_str {
                "gguf" => ModelFormat::Gguf,
                "safetensors" => ModelFormat::SafeTensors,
                _ => ModelFormat::Gguf,
            };
            return Some(ResolvedModel {
                name: format!("{}:{}", name, tag),
                url: url.to_string(),
                format,
                source: ModelSource::Direct,
            });
        }
    }

    None
}

/// Query HuggingFace API for GGUF models matching the name.
async fn resolve_from_huggingface(model_ref: &ModelRef) -> Result<Option<ResolvedModel>> {
    let search_url = format!(
        "https://huggingface.co/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit=5",
        model_ref.name
    );

    let response = reqwest::get(&search_url)
        .await
        .map_err(|e| PowerError::Config(format!("Failed to query HuggingFace API: {e}")))?;

    if !response.status().is_success() {
        return Ok(None);
    }

    let results: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| PowerError::Config(format!("Failed to parse HuggingFace response: {e}")))?;

    // Find the first result that has GGUF files
    for result in &results {
        let model_id = match result.get("modelId").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => continue,
        };

        // Check siblings for GGUF files
        if let Some(siblings) = result.get("siblings").and_then(|v| v.as_array()) {
            // Look for a Q4_K_M quantization first, then any .gguf file
            let mut gguf_file: Option<&str> = None;
            let mut q4_file: Option<&str> = None;

            for sibling in siblings {
                if let Some(filename) = sibling.get("rfilename").and_then(|v| v.as_str()) {
                    if filename.ends_with(".gguf") {
                        if gguf_file.is_none() {
                            gguf_file = Some(filename);
                        }
                        if filename.contains("Q4_K_M") {
                            q4_file = Some(filename);
                        }
                    }
                }
            }

            let chosen = q4_file.or(gguf_file);
            if let Some(filename) = chosen {
                let url = format!(
                    "https://huggingface.co/{}/resolve/main/{}",
                    model_id, filename
                );
                return Ok(Some(ResolvedModel {
                    name: format!("{}:{}", model_ref.name, model_ref.tag),
                    url,
                    format: ModelFormat::Gguf,
                    source: ModelSource::Direct,
                }));
            }
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_name_with_tag() {
        let r = ModelRef::parse("llama3.2:3b");
        assert_eq!(r.name, "llama3.2");
        assert_eq!(r.tag, "3b");
    }

    #[test]
    fn test_parse_name_without_tag() {
        let r = ModelRef::parse("llama3.2");
        assert_eq!(r.name, "llama3.2");
        assert_eq!(r.tag, "latest");
    }

    #[test]
    fn test_is_url_true() {
        assert!(is_url("https://example.com/model.gguf"));
        assert!(is_url("http://example.com/model.gguf"));
    }

    #[test]
    fn test_is_url_false() {
        assert!(!is_url("llama3.2:3b"));
        assert!(!is_url("model-name"));
    }

    #[test]
    fn test_resolve_known_model() {
        let parsed = ModelRef::parse("llama3.2:3b");
        let resolved = resolve_from_builtin(&parsed);
        assert!(resolved.is_some());
        let resolved = resolved.unwrap();
        assert_eq!(resolved.name, "llama3.2:3b");
        assert!(resolved.url.contains("huggingface.co"));
        assert_eq!(resolved.format, ModelFormat::Gguf);
    }

    #[test]
    fn test_resolve_known_model_latest_tag() {
        // "latest" tag should match the first entry for that name
        let parsed = ModelRef::parse("phi3");
        let resolved = resolve_from_builtin(&parsed);
        assert!(resolved.is_some());
        let resolved = resolved.unwrap();
        assert!(resolved.url.contains("Phi-3"));
    }

    #[test]
    fn test_resolve_unknown_model_returns_none() {
        let parsed = ModelRef::parse("nonexistent-model:v1");
        let resolved = resolve_from_builtin(&parsed);
        assert!(resolved.is_none());
    }

    #[test]
    fn test_model_ref_equality() {
        let a = ModelRef::parse("llama3.2:3b");
        let b = ModelRef {
            name: "llama3.2".to_string(),
            tag: "3b".to_string(),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_parse_name_with_multiple_colons() {
        let r = ModelRef::parse("model:tag:extra");
        assert_eq!(r.name, "model");
        assert_eq!(r.tag, "tag:extra");
    }

    #[test]
    fn test_parse_empty_string() {
        let r = ModelRef::parse("");
        assert_eq!(r.name, "");
        assert_eq!(r.tag, "latest");
    }

    #[test]
    fn test_is_url_edge_cases() {
        assert!(!is_url(""));
        assert!(!is_url("ftp://example.com"));
        assert!(!is_url("https"));
        assert!(is_url("https://x"));
    }

    #[test]
    fn test_resolved_model_fields() {
        let resolved = ResolvedModel {
            name: "test:latest".to_string(),
            url: "https://example.com/model.gguf".to_string(),
            format: ModelFormat::Gguf,
            source: ModelSource::Direct,
        };
        assert_eq!(resolved.name, "test:latest");
        assert_eq!(resolved.format, ModelFormat::Gguf);
    }

    #[test]
    fn test_resolve_builtin_with_specific_tag() {
        // Test that a specific tag that doesn't exist returns None
        let parsed = ModelRef::parse("llama3.2:nonexistent-tag");
        let resolved = resolve_from_builtin(&parsed);
        assert!(resolved.is_none());
    }

    #[test]
    fn test_model_ref_debug() {
        let r = ModelRef::parse("test:v1");
        let debug = format!("{:?}", r);
        assert!(debug.contains("test"));
        assert!(debug.contains("v1"));
    }

    #[test]
    fn test_model_ref_clone() {
        let r = ModelRef::parse("test:v1");
        let cloned = r.clone();
        assert_eq!(r, cloned);
    }

    #[test]
    fn test_model_ref_parse_with_tag() {
        let r = ModelRef::parse("llama3:7b");
        assert_eq!(r.name, "llama3");
        assert_eq!(r.tag, "7b");
    }

    #[test]
    fn test_model_ref_parse_without_tag() {
        let r = ModelRef::parse("llama3");
        assert_eq!(r.name, "llama3");
        assert_eq!(r.tag, "latest");
    }

    #[test]
    fn test_model_ref_parse_with_namespace() {
        let r = ModelRef::parse("library/llama3:7b");
        assert_eq!(r.name, "library/llama3");
        assert_eq!(r.tag, "7b");
    }

    #[test]
    fn test_is_url_http() {
        assert!(is_url("http://example.com/model.gguf"));
        assert!(is_url("https://example.com/model.gguf"));
    }

    #[test]
    fn test_is_url_not_url() {
        assert!(!is_url("llama3:7b"));
        assert!(!is_url("my-model"));
    }

    #[test]
    fn test_resolve_from_builtin_known_model() {
        let r = ModelRef::parse("llama3.2:3b");
        let result = resolve_from_builtin(&r);
        assert!(result.is_some());
        let resolved = result.unwrap();
        assert!(resolved.url.contains("huggingface.co") || resolved.url.contains("hf.co"));
    }

    #[test]
    fn test_resolve_from_builtin_unknown_model() {
        let r = ModelRef::parse("totally-unknown-model:latest");
        let result = resolve_from_builtin(&r);
        assert!(result.is_none());
    }

    #[test]
    fn test_model_ref_full_name() {
        let r = ModelRef {
            name: "llama3".to_string(),
            tag: "7b".to_string(),
        };
        assert_eq!(format!("{}:{}", r.name, r.tag), "llama3:7b");
    }

    #[test]
    fn test_model_ref_parse_colon_in_tag() {
        let r = ModelRef::parse("model:tag:with:colons");
        assert_eq!(r.name, "model");
        assert_eq!(r.tag, "tag:with:colons");
    }

    #[test]
    fn test_model_ref_parse_only_colon() {
        let r = ModelRef::parse(":");
        assert_eq!(r.name, "");
        assert_eq!(r.tag, "");
    }

    #[test]
    fn test_model_ref_parse_trailing_colon() {
        let r = ModelRef::parse("model:");
        assert_eq!(r.name, "model");
        assert_eq!(r.tag, "");
    }

    #[test]
    fn test_model_ref_parse_leading_colon() {
        let r = ModelRef::parse(":tag");
        assert_eq!(r.name, "");
        assert_eq!(r.tag, "tag");
    }

    #[test]
    fn test_is_url_with_port() {
        assert!(is_url("http://localhost:8080/model.gguf"));
        assert!(is_url("https://example.com:443/model.gguf"));
    }

    #[test]
    fn test_is_url_with_path() {
        assert!(is_url("https://example.com/path/to/model.gguf"));
        assert!(is_url("http://example.com/a/b/c/model.gguf"));
    }

    #[test]
    fn test_is_url_with_query() {
        assert!(is_url("https://example.com/model.gguf?download=true"));
    }

    #[test]
    fn test_is_url_with_fragment() {
        assert!(is_url("https://example.com/model.gguf#section"));
    }

    #[test]
    fn test_is_url_case_sensitive() {
        assert!(!is_url("HTTP://example.com/model.gguf"));
        assert!(!is_url("HTTPS://example.com/model.gguf"));
    }

    #[test]
    fn test_resolve_builtin_with_different_tags() {
        let r1 = ModelRef::parse("llama3.2:3b");
        let r2 = ModelRef::parse("llama3.2:1b");
        let resolved1 = resolve_from_builtin(&r1);
        let resolved2 = resolve_from_builtin(&r2);
        // Both should resolve if they exist in known_models.json
        if let (Some(r1), Some(r2)) = (resolved1, resolved2) {
            assert_ne!(r1.url, r2.url);
        }
    }

    #[test]
    fn test_resolve_builtin_case_sensitive() {
        let r1 = ModelRef::parse("llama3.2:3b");
        let r2 = ModelRef::parse("LLAMA3.2:3B");
        let resolved1 = resolve_from_builtin(&r1);
        let resolved2 = resolve_from_builtin(&r2);
        // Names are case-sensitive
        if resolved1.is_some() {
            assert!(resolved2.is_none());
        }
    }

    #[test]
    fn test_resolved_model_clone() {
        let resolved = ResolvedModel {
            name: "test:latest".to_string(),
            url: "https://example.com/model.gguf".to_string(),
            format: ModelFormat::Gguf,
            source: ModelSource::Direct,
        };
        let cloned = resolved.clone();
        assert_eq!(resolved.name, cloned.name);
        assert_eq!(resolved.url, cloned.url);
    }

    #[test]
    fn test_model_ref_with_special_chars() {
        let r = ModelRef::parse("model-name_v1.2:tag-v3.4");
        assert_eq!(r.name, "model-name_v1.2");
        assert_eq!(r.tag, "tag-v3.4");
    }

    #[test]
    fn test_model_ref_with_slash() {
        let r = ModelRef::parse("org/model:tag");
        assert_eq!(r.name, "org/model");
        assert_eq!(r.tag, "tag");
    }

    #[test]
    fn test_model_ref_with_dots() {
        let r = ModelRef::parse("llama3.2.1:3b");
        assert_eq!(r.name, "llama3.2.1");
        assert_eq!(r.tag, "3b");
    }

    #[test]
    fn test_is_url_incomplete() {
        assert!(!is_url("http:/"));
        assert!(!is_url("https:/"));
        assert!(!is_url("http"));
        assert!(!is_url("https"));
    }

    #[test]
    fn test_model_source_debug() {
        let source = ModelSource::Direct;
        let debug = format!("{:?}", source);
        assert!(debug.contains("Direct"));
    }
}
