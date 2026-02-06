use crate::error::{PowerError, Result};
use crate::model::manifest::ModelFormat;

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

/// A resolved model with its download URL.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub name: String,
    pub url: String,
    pub format: ModelFormat,
}

/// Check if the input looks like a direct URL.
pub fn is_url(input: &str) -> bool {
    input.starts_with("http://") || input.starts_with("https://")
}

/// Resolve a model name (or name:tag) to a download URL.
///
/// Resolution strategy:
/// 1. Check built-in known_models.json registry
/// 2. Query HuggingFace API for GGUF models matching the name
/// 3. Error if not found
pub async fn resolve(model_ref: &str) -> Result<ResolvedModel> {
    let parsed = ModelRef::parse(model_ref);

    // 1. Check built-in registry
    if let Some(resolved) = resolve_from_builtin(&parsed) {
        return Ok(resolved);
    }

    // 2. Query HuggingFace API
    if let Some(resolved) = resolve_from_huggingface(&parsed).await? {
        return Ok(resolved);
    }

    Err(PowerError::ModelNotFound(format!(
        "Could not resolve model '{}'. Try providing a direct URL instead.",
        model_ref
    )))
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
}
