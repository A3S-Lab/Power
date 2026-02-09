use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest, ModelParameters};
use crate::model::ollama_registry;
use crate::model::resolve::{self, ModelSource};
use crate::model::storage;

/// Progress callback type for download reporting.
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

/// Download a model from a direct URL or resolve a model name first.
///
/// If `name_or_url` is a URL, downloads directly.
/// Otherwise, resolves the name via the model registry, then downloads.
/// Returns the resulting `ModelManifest` after download and storage.
pub async fn pull_model(
    name_or_url: &str,
    url_override: Option<&str>,
    progress: Option<ProgressCallback>,
) -> Result<ModelManifest> {
    let (name, url, source) = if let Some(url) = url_override {
        (
            name_or_url.to_string(),
            url.to_string(),
            ModelSource::Direct,
        )
    } else if resolve::is_url(name_or_url) {
        let name = extract_name_from_url(name_or_url);
        (name, name_or_url.to_string(), ModelSource::Direct)
    } else {
        let resolved = resolve::resolve(name_or_url).await?;
        (resolved.name, resolved.url, resolved.source)
    };

    tracing::info!(name = %name, url = %url, "Pulling model");

    let response = reqwest::get(&url)
        .await
        .map_err(|e| PowerError::DownloadFailed {
            model: name.to_string(),
            source: e,
        })?;

    let total_size = response.content_length().unwrap_or(0);
    let bytes = download_with_progress(response, total_size, progress).await?;

    let (blob_path, sha256) = storage::store_blob(&bytes)?;
    let format = detect_format(&name, &blob_path);

    let manifest = match source {
        ModelSource::OllamaRegistry(reg) => {
            let reg = *reg; // Unbox to move fields out
            let parameter_count = reg
                .config
                .model_type
                .as_deref()
                .and_then(ollama_registry::parse_parameter_count);

            ModelManifest {
                name,
                format,
                size: bytes.len() as u64,
                sha256,
                parameters: Some(ModelParameters {
                    context_length: None,
                    embedding_length: None,
                    parameter_count,
                    quantization: reg.config.file_type,
                }),
                created_at: chrono::Utc::now(),
                path: blob_path,
                system_prompt: reg.system_prompt,
                template_override: reg.template,
                default_parameters: reg.params,
                modelfile_content: None,
                license: reg.license,
                adapter_path: None,
                messages: vec![],
            }
        }
        ModelSource::Direct => ModelManifest {
            name,
            format,
            size: bytes.len() as u64,
            sha256,
            parameters: None,
            created_at: chrono::Utc::now(),
            path: blob_path,
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            messages: vec![],
        },
    };

    Ok(manifest)
}

async fn download_with_progress(
    response: reqwest::Response,
    total_size: u64,
    progress: Option<ProgressCallback>,
) -> Result<Vec<u8>> {
    use futures::StreamExt;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut data = Vec::with_capacity(total_size as usize);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| PowerError::DownloadFailed {
            model: "unknown".to_string(),
            source: e,
        })?;
        data.extend_from_slice(&chunk);
        downloaded += chunk.len() as u64;

        if let Some(ref cb) = progress {
            cb(downloaded, total_size);
        }
    }

    Ok(data)
}

/// Extract a reasonable model name from a URL.
///
/// Strips the file extension (.gguf, .safetensors) and returns the filename.
pub fn extract_name_from_url(url: &str) -> String {
    url.rsplit('/')
        .next()
        .unwrap_or("unknown")
        .trim_end_matches(".gguf")
        .trim_end_matches(".safetensors")
        .to_string()
}

/// Detect model format from filename extension.
fn detect_format(name: &str, path: &std::path::Path) -> ModelFormat {
    let name_lower = name.to_lowercase();
    let path_str = path.to_string_lossy().to_lowercase();

    if name_lower.contains("gguf") || path_str.ends_with(".gguf") {
        ModelFormat::Gguf
    } else if name_lower.contains("safetensors") || path_str.ends_with(".safetensors") {
        ModelFormat::SafeTensors
    } else {
        // Default to GGUF as it's the most common format for local inference
        ModelFormat::Gguf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_detect_format_gguf() {
        let path = PathBuf::from("/tmp/model.gguf");
        assert_eq!(detect_format("model", &path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_safetensors() {
        let path = PathBuf::from("/tmp/model.safetensors");
        assert_eq!(detect_format("model", &path), ModelFormat::SafeTensors);
    }

    #[test]
    fn test_detect_format_from_name() {
        let path = PathBuf::from("/tmp/sha256-abc123");
        assert_eq!(detect_format("model-gguf", &path), ModelFormat::Gguf);
        assert_eq!(
            detect_format("model-safetensors", &path),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_detect_format_default() {
        let path = PathBuf::from("/tmp/sha256-abc123");
        assert_eq!(detect_format("some-model", &path), ModelFormat::Gguf);
    }

    #[test]
    fn test_extract_name_from_url_gguf() {
        assert_eq!(
            extract_name_from_url("https://example.com/model.gguf"),
            "model"
        );
    }

    #[test]
    fn test_extract_name_from_url_with_path() {
        assert_eq!(
            extract_name_from_url("https://example.com/models/llama3-8b-q4.gguf"),
            "llama3-8b-q4"
        );
    }

    #[test]
    fn test_extract_name_from_url_safetensors() {
        assert_eq!(
            extract_name_from_url("https://example.com/model.safetensors"),
            "model"
        );
    }

    #[test]
    fn test_extract_name_from_url_no_extension() {
        assert_eq!(
            extract_name_from_url("https://example.com/plain-model"),
            "plain-model"
        );
    }

    #[test]
    fn test_extract_name_from_url_empty() {
        // rsplit('/') on empty string returns [""], not None
        assert_eq!(extract_name_from_url(""), "");
    }

    #[test]
    fn test_model_source_direct_creates_basic_manifest() {
        // Verify that ModelSource::Direct produces a manifest with no enrichment
        let manifest = ModelManifest {
            name: "test:latest".to_string(),
            format: ModelFormat::Gguf,
            size: 1024,
            sha256: "abc123".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: PathBuf::from("/tmp/test"),
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            messages: vec![],
        };
        assert!(manifest.parameters.is_none());
        assert!(manifest.system_prompt.is_none());
        assert!(manifest.template_override.is_none());
        assert!(manifest.default_parameters.is_none());
        assert!(manifest.license.is_none());
    }

    #[test]
    fn test_model_source_ollama_enriches_manifest() {
        use crate::model::ollama_registry::{OllamaModelConfig, OllamaRegistryModel};
        use std::collections::HashMap;

        let reg = OllamaRegistryModel {
            model_digest: "sha256:abc".to_string(),
            model_size: 4_000_000_000,
            template: Some("{{ .System }}{{ .Prompt }}".to_string()),
            system_prompt: Some("You are a helpful assistant.".to_string()),
            params: Some(HashMap::from([(
                "temperature".to_string(),
                serde_json::Value::from(0.7),
            )])),
            license: Some("MIT".to_string()),
            config: OllamaModelConfig {
                model_format: Some("gguf".to_string()),
                model_family: Some("llama".to_string()),
                model_type: Some("3.2B".to_string()),
                file_type: Some("Q4_K_M".to_string()),
            },
        };

        let parameter_count = reg
            .config
            .model_type
            .as_deref()
            .and_then(crate::model::ollama_registry::parse_parameter_count);

        let manifest = ModelManifest {
            name: "llama3.2:3b".to_string(),
            format: ModelFormat::Gguf,
            size: 2_000_000_000,
            sha256: "abc123".to_string(),
            parameters: Some(ModelParameters {
                context_length: None,
                embedding_length: None,
                parameter_count,
                quantization: reg.config.file_type.clone(),
            }),
            created_at: chrono::Utc::now(),
            path: PathBuf::from("/tmp/test"),
            system_prompt: reg.system_prompt.clone(),
            template_override: reg.template.clone(),
            default_parameters: reg.params.clone(),
            modelfile_content: None,
            license: reg.license.clone(),
            adapter_path: None,
            messages: vec![],
        };

        // Verify enrichment from registry metadata
        assert_eq!(
            manifest.system_prompt.as_deref(),
            Some("You are a helpful assistant.")
        );
        assert_eq!(
            manifest.template_override.as_deref(),
            Some("{{ .System }}{{ .Prompt }}")
        );
        assert_eq!(manifest.license.as_deref(), Some("MIT"));
        assert!(manifest.default_parameters.is_some());
        let params = manifest.default_parameters.unwrap();
        assert_eq!(params["temperature"], 0.7);

        // Verify parameter extraction
        let mp = manifest.parameters.unwrap();
        assert_eq!(mp.parameter_count, Some(3_200_000_000));
        assert_eq!(mp.quantization.as_deref(), Some("Q4_K_M"));
    }
}
