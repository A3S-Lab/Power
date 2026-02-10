use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest, ModelParameters};
use crate::model::ollama_registry;
use crate::model::resolve::{self, ModelSource};
use crate::model::storage;
use crate::{dirs, model};

/// Progress callback type for download reporting.
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

/// Download a model from a direct URL or resolve a model name first.
///
/// If `name_or_url` is a URL, downloads directly.
/// Otherwise, resolves the name via the model registry, then downloads.
/// Supports resuming interrupted downloads via HTTP Range requests.
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

    let bytes = download_with_resume(&name, &url, progress).await?;

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

            // Download projector blob if present (for vision models)
            let projector_path = if let Some(ref proj_digest) = reg.projector_digest {
                let proj_name = resolve::ModelRef::parse(name_or_url);
                let proj_url = ollama_registry::blob_url(&proj_name.name, proj_digest);
                tracing::info!(digest = %proj_digest, "Downloading multimodal projector");
                let proj_resp = reqwest::get(&proj_url)
                    .await
                    .map_err(|e| PowerError::DownloadFailed {
                        model: name.clone(),
                        source: e,
                    })?;
                if proj_resp.status().is_success() {
                    let proj_bytes = proj_resp.bytes().await.map_err(|e| {
                        PowerError::DownloadFailed {
                            model: name.clone(),
                            source: e,
                        }
                    })?;
                    let (proj_blob_path, _) = storage::store_blob(&proj_bytes)?;
                    tracing::info!(
                        path = %proj_blob_path.display(),
                        "Projector stored"
                    );
                    Some(proj_blob_path.to_string_lossy().to_string())
                } else {
                    tracing::warn!(
                        status = %proj_resp.status(),
                        "Failed to download projector, vision will be unavailable"
                    );
                    None
                }
            } else {
                None
            };

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
                projector_path,
                messages: vec![],
                family: reg.config.model_family.clone(),
                families: None,
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
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
        },
    };

    Ok(manifest)
}

/// Download with HTTP Range-based resume support.
///
/// Uses a `.partial` temp file in the blobs directory to track incomplete downloads.
/// If a partial file exists, sends a `Range: bytes={existing_size}-` header to resume.
/// On completion, reads the full data and cleans up the partial file.
async fn download_with_resume(
    name: &str,
    url: &str,
    progress: Option<ProgressCallback>,
) -> Result<Vec<u8>> {
    use std::io::Write;

    let blob_dir = dirs::blobs_dir();
    std::fs::create_dir_all(&blob_dir)?;

    // Use a deterministic partial filename based on URL hash
    let url_hash = model::storage::compute_sha256(url.as_bytes());
    let partial_path = blob_dir.join(format!("partial-{}", &url_hash[..16]));

    // Check for existing partial download
    let existing_size = if partial_path.exists() {
        std::fs::metadata(&partial_path)
            .map(|m| m.len())
            .unwrap_or(0)
    } else {
        0
    };

    let client = reqwest::Client::new();
    let mut request = client.get(url);

    if existing_size > 0 {
        tracing::info!(
            name = %name,
            existing_bytes = existing_size,
            "Resuming download from partial file"
        );
        request = request.header("Range", format!("bytes={existing_size}-"));
    }

    let response = request.send().await.map_err(|e| PowerError::DownloadFailed {
        model: name.to_string(),
        source: e,
    })?;

    let status = response.status();

    // 206 Partial Content = server supports range, resuming
    // 200 OK = server doesn't support range or fresh download
    // If server returns 200 when we asked for a range, discard partial and start over
    let resume = status == reqwest::StatusCode::PARTIAL_CONTENT && existing_size > 0;

    if !resume && existing_size > 0 {
        tracing::info!("Server does not support range requests, restarting download");
        let _ = std::fs::remove_file(&partial_path);
    }

    if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(PowerError::DownloadFailed {
            model: name.to_string(),
            source: response.error_for_status().unwrap_err(),
        });
    }

    // Determine total size for progress reporting
    let content_length = response.content_length().unwrap_or(0);
    let total_size = if resume {
        existing_size + content_length
    } else {
        content_length
    };

    let starting_offset = if resume { existing_size } else { 0 };

    // Stream to partial file
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(resume)
        .write(true)
        .truncate(!resume)
        .open(&partial_path)
        .map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "Failed to open partial file {}: {e}",
                partial_path.display()
            )))
        })?;

    {
        use futures::StreamExt;
        let mut stream = response.bytes_stream();
        let mut downloaded = starting_offset;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| PowerError::DownloadFailed {
                model: name.to_string(),
                source: e,
            })?;
            file.write_all(&chunk).map_err(|e| {
                PowerError::Io(std::io::Error::other(format!(
                    "Failed to write to partial file: {e}"
                )))
            })?;
            downloaded += chunk.len() as u64;

            if let Some(ref cb) = progress {
                cb(downloaded, total_size);
            }
        }
    }

    // Flush and close the file
    file.flush().map_err(|e| {
        PowerError::Io(std::io::Error::other(format!(
            "Failed to flush partial file: {e}"
        )))
    })?;
    drop(file);

    // Read the complete file into memory
    let bytes = std::fs::read(&partial_path).map_err(|e| {
        PowerError::Io(std::io::Error::other(format!(
            "Failed to read completed download: {e}"
        )))
    })?;

    // Clean up partial file
    let _ = std::fs::remove_file(&partial_path);

    Ok(bytes)
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
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
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
            projector_digest: None,
            projector_size: None,
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
            projector_path: None,
            messages: vec![],
            family: reg.config.model_family.clone(),
            families: None,
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

    #[test]
    fn test_partial_filename_deterministic() {
        let hash1 = crate::model::storage::compute_sha256(b"https://example.com/model.gguf");
        let hash2 = crate::model::storage::compute_sha256(b"https://example.com/model.gguf");
        assert_eq!(hash1, hash2);
        // Partial filename uses first 16 chars of URL hash
        let partial1 = format!("partial-{}", &hash1[..16]);
        let partial2 = format!("partial-{}", &hash2[..16]);
        assert_eq!(partial1, partial2);
    }

    #[test]
    fn test_partial_filename_differs_for_different_urls() {
        let hash1 = crate::model::storage::compute_sha256(b"https://example.com/model-a.gguf");
        let hash2 = crate::model::storage::compute_sha256(b"https://example.com/model-b.gguf");
        assert_ne!(&hash1[..16], &hash2[..16]);
    }

    #[test]
    fn test_extract_name_from_url_with_query_params() {
        // Query params are kept as part of the filename after stripping .gguf
        assert_eq!(
            extract_name_from_url("https://example.com/model.gguf?download=true"),
            "model.gguf?download=true"
        );
    }

    #[test]
    fn test_extract_name_from_url_with_fragment() {
        // Fragment is kept as part of the filename after stripping .gguf
        assert_eq!(
            extract_name_from_url("https://example.com/model.gguf#section"),
            "model.gguf#section"
        );
    }

    #[test]
    fn test_extract_name_from_url_multiple_extensions() {
        assert_eq!(
            extract_name_from_url("https://example.com/model.tar.gguf"),
            "model.tar"
        );
    }

    #[test]
    fn test_extract_name_from_url_no_slash() {
        assert_eq!(extract_name_from_url("model.gguf"), "model");
    }

    #[test]
    fn test_detect_format_case_insensitive() {
        let path = std::path::PathBuf::from("/tmp/MODEL.GGUF");
        assert_eq!(detect_format("test", &path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_name_priority() {
        let path = std::path::PathBuf::from("/tmp/sha256-abc");
        assert_eq!(detect_format("model-GGUF-test", &path), ModelFormat::Gguf);
        assert_eq!(
            detect_format("model-SAFETENSORS-test", &path),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_detect_format_path_priority() {
        let path = std::path::PathBuf::from("/tmp/model.gguf");
        assert_eq!(detect_format("random-name", &path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_both_match() {
        let path = std::path::PathBuf::from("/tmp/model.gguf");
        assert_eq!(detect_format("model-gguf", &path), ModelFormat::Gguf);
    }
}
