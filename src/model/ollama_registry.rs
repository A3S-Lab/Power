use std::collections::HashMap;

use serde::Deserialize;

use crate::error::{PowerError, Result};

/// Base URL for the Ollama registry.
const REGISTRY_BASE: &str = "https://registry.ollama.ai";

/// Accept header required by the Ollama registry for manifest requests.
const MANIFEST_ACCEPT: &str = "application/vnd.docker.distribution.manifest.v2+json";

// OCI media types used by Ollama registry layers.
const MEDIA_MODEL: &str = "application/vnd.ollama.image.model";
const MEDIA_TEMPLATE: &str = "application/vnd.ollama.image.template";
const MEDIA_SYSTEM: &str = "application/vnd.ollama.image.system";
const MEDIA_PARAMS: &str = "application/vnd.ollama.image.params";
const MEDIA_LICENSE: &str = "application/vnd.ollama.image.license";

/// OCI-like manifest returned by the Ollama registry.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OllamaManifest {
    #[allow(dead_code)]
    schema_version: u32,
    #[allow(dead_code)]
    media_type: String,
    config: LayerDescriptor,
    layers: Vec<LayerDescriptor>,
}

/// Descriptor for a single layer or config blob in the manifest.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LayerDescriptor {
    media_type: String,
    digest: String,
    size: u64,
}

/// Parsed config blob metadata from the Ollama registry.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct OllamaModelConfig {
    pub model_format: Option<String>,
    pub model_family: Option<String>,
    pub model_type: Option<String>,
    pub file_type: Option<String>,
}

/// All metadata extracted from the Ollama registry for a model.
///
/// Contains the model weights blob info plus small metadata blobs
/// (template, system prompt, params, license).
#[derive(Debug, Clone)]
pub struct OllamaRegistryModel {
    /// Digest of the model weights blob (sha256:...)
    pub model_digest: String,
    /// Size of the model weights blob in bytes
    pub model_size: u64,
    /// Chat template (Go template string)
    pub template: Option<String>,
    /// System prompt
    pub system_prompt: Option<String>,
    /// Default generation parameters
    pub params: Option<HashMap<String, serde_json::Value>>,
    /// License text (concatenated if multiple)
    pub license: Option<String>,
    /// Config metadata (model family, type, quantization, etc.)
    pub config: OllamaModelConfig,
}

/// Fetch manifest and metadata from the Ollama registry.
///
/// Downloads the OCI manifest and all small metadata blobs (template, system,
/// params, config, license). Does NOT download the model weights — only metadata.
pub async fn fetch_registry_model(name: &str, tag: &str) -> Result<OllamaRegistryModel> {
    let client = reqwest::Client::new();

    // Bare names like "llama3.2" map to "library/llama3.2" in the registry.
    let namespace = if name.contains('/') {
        name.to_string()
    } else {
        format!("library/{name}")
    };

    // 1. Fetch the OCI manifest
    let manifest_url = format!("{REGISTRY_BASE}/v2/{namespace}/manifests/{tag}");
    let resp = client
        .get(&manifest_url)
        .header("Accept", MANIFEST_ACCEPT)
        .send()
        .await
        .map_err(|e| {
            PowerError::Config(format!(
                "Failed to fetch Ollama registry manifest for {name}:{tag}: {e}"
            ))
        })?;

    if !resp.status().is_success() {
        return Err(PowerError::ModelNotFound(format!(
            "Ollama registry returned {} for {name}:{tag}",
            resp.status()
        )));
    }

    let manifest: OllamaManifest = resp.json().await.map_err(|e| {
        PowerError::Config(format!(
            "Failed to parse Ollama manifest for {name}:{tag}: {e}"
        ))
    })?;

    // 2. Find the model weights layer (mediaType = model)
    let model_layer = find_layer(&manifest.layers, MEDIA_MODEL).ok_or_else(|| {
        PowerError::Config(format!(
            "Ollama manifest for {name}:{tag} has no model weights layer"
        ))
    })?;

    // 3. Fetch small metadata blobs in parallel
    let template_layer = find_layer(&manifest.layers, MEDIA_TEMPLATE);
    let system_layer = find_layer(&manifest.layers, MEDIA_SYSTEM);
    let params_layer = find_layer(&manifest.layers, MEDIA_PARAMS);
    let license_layers = find_all_layers(&manifest.layers, MEDIA_LICENSE);

    let (config, template, system_prompt, params, license) = tokio::try_join!(
        fetch_config_blob(&client, &namespace, &manifest.config),
        fetch_optional_text_blob(&client, &namespace, template_layer),
        fetch_optional_text_blob(&client, &namespace, system_layer),
        fetch_optional_text_blob(&client, &namespace, params_layer),
        fetch_license_blobs(&client, &namespace, &license_layers),
    )?;

    let params_map: Option<HashMap<String, serde_json::Value>> =
        params.as_deref().and_then(|s| serde_json::from_str(s).ok());

    Ok(OllamaRegistryModel {
        model_digest: model_layer.digest.clone(),
        model_size: model_layer.size,
        template,
        system_prompt,
        params: params_map,
        license,
        config,
    })
}

/// Build the full blob download URL for streaming the model weights.
pub fn blob_url(name: &str, digest: &str) -> String {
    let namespace = if name.contains('/') {
        name.to_string()
    } else {
        format!("library/{name}")
    };
    format!("{REGISTRY_BASE}/v2/{namespace}/blobs/{digest}")
}

/// Parse a model_type string like "3.2B" into a parameter count.
pub fn parse_parameter_count(model_type: &str) -> Option<u64> {
    let s = model_type.trim().to_uppercase();
    if let Some(num_str) = s.strip_suffix('B') {
        let num: f64 = num_str.parse().ok()?;
        Some((num * 1_000_000_000.0) as u64)
    } else if let Some(num_str) = s.strip_suffix('M') {
        let num: f64 = num_str.parse().ok()?;
        Some((num * 1_000_000.0) as u64)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Find the first layer matching a given media type.
fn find_layer<'a>(layers: &'a [LayerDescriptor], media_type: &str) -> Option<&'a LayerDescriptor> {
    layers.iter().find(|l| l.media_type == media_type)
}

/// Find all layers matching a given media type.
fn find_all_layers<'a>(
    layers: &'a [LayerDescriptor],
    media_type: &str,
) -> Vec<&'a LayerDescriptor> {
    layers
        .iter()
        .filter(|l| l.media_type == media_type)
        .collect()
}

/// Fetch and parse the config blob (JSON with model metadata).
async fn fetch_config_blob(
    client: &reqwest::Client,
    namespace: &str,
    descriptor: &LayerDescriptor,
) -> Result<OllamaModelConfig> {
    let url = format!("{REGISTRY_BASE}/v2/{namespace}/blobs/{}", descriptor.digest);
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| PowerError::Config(format!("Failed to fetch config blob: {e}")))?;

    if !resp.status().is_success() {
        return Ok(OllamaModelConfig::default());
    }

    let config: OllamaModelConfig = resp.json().await.unwrap_or_default();
    Ok(config)
}

/// Fetch an optional small text blob (template, system prompt, or params).
async fn fetch_optional_text_blob(
    client: &reqwest::Client,
    namespace: &str,
    descriptor: Option<&LayerDescriptor>,
) -> Result<Option<String>> {
    let descriptor = match descriptor {
        Some(d) => d,
        None => return Ok(None),
    };

    let url = format!("{REGISTRY_BASE}/v2/{namespace}/blobs/{}", descriptor.digest);
    let resp = client.get(&url).send().await.map_err(|e| {
        PowerError::Config(format!("Failed to fetch blob {}: {e}", descriptor.digest))
    })?;

    if !resp.status().is_success() {
        return Ok(None);
    }

    let text = resp.text().await.map_err(|e| {
        PowerError::Config(format!("Failed to read blob {}: {e}", descriptor.digest))
    })?;

    Ok(Some(text))
}

/// Fetch and concatenate all license blobs (there can be multiple).
async fn fetch_license_blobs(
    client: &reqwest::Client,
    namespace: &str,
    descriptors: &[&LayerDescriptor],
) -> Result<Option<String>> {
    if descriptors.is_empty() {
        return Ok(None);
    }

    let mut parts = Vec::with_capacity(descriptors.len());
    for desc in descriptors {
        let url = format!("{REGISTRY_BASE}/v2/{namespace}/blobs/{}", desc.digest);
        let resp = client.get(&url).send().await.map_err(|e| {
            PowerError::Config(format!("Failed to fetch license blob {}: {e}", desc.digest))
        })?;

        if resp.status().is_success() {
            let text = resp.text().await.map_err(|e| {
                PowerError::Config(format!("Failed to read license blob {}: {e}", desc.digest))
            })?;
            parts.push(text);
        }
    }

    if parts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(parts.join("\n---\n")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ollama_manifest() {
        let json = r#"{
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": "sha256:34bb5ab01051",
                "size": 561
            },
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256:dde5aa3fc5ff",
                    "size": 2019377376
                },
                {
                    "mediaType": "application/vnd.ollama.image.template",
                    "digest": "sha256:966de95ca8a6",
                    "size": 1429
                },
                {
                    "mediaType": "application/vnd.ollama.image.license",
                    "digest": "sha256:fcc5a6bec9da",
                    "size": 7711
                },
                {
                    "mediaType": "application/vnd.ollama.image.params",
                    "digest": "sha256:56bb8bd477a5",
                    "size": 96
                }
            ]
        }"#;

        let manifest: OllamaManifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.schema_version, 2);
        assert_eq!(manifest.layers.len(), 4);
        assert_eq!(manifest.config.digest, "sha256:34bb5ab01051");
    }

    #[test]
    fn test_parse_config_blob() {
        let json = r#"{
            "model_format": "gguf",
            "model_family": "llama",
            "model_type": "3.2B",
            "file_type": "Q4_K_M"
        }"#;

        let config: OllamaModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_format.as_deref(), Some("gguf"));
        assert_eq!(config.model_family.as_deref(), Some("llama"));
        assert_eq!(config.model_type.as_deref(), Some("3.2B"));
        assert_eq!(config.file_type.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn test_parse_config_blob_with_extra_fields() {
        // Real config blobs have extra fields like model_families, architecture, os, rootfs
        let json = r#"{
            "model_format": "gguf",
            "model_family": "llama",
            "model_families": ["llama"],
            "model_type": "3.2B",
            "file_type": "Q4_K_M",
            "architecture": "amd64",
            "os": "linux",
            "rootfs": {"type": "layers", "diff_ids": []}
        }"#;

        let config: OllamaModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_format.as_deref(), Some("gguf"));
        assert_eq!(config.model_type.as_deref(), Some("3.2B"));
    }

    #[test]
    fn test_blob_url_construction() {
        let url = blob_url("llama3.2", "sha256:abc123");
        assert_eq!(
            url,
            "https://registry.ollama.ai/v2/library/llama3.2/blobs/sha256:abc123"
        );
    }

    #[test]
    fn test_blob_url_with_namespace() {
        let url = blob_url("myorg/mymodel", "sha256:def456");
        assert_eq!(
            url,
            "https://registry.ollama.ai/v2/myorg/mymodel/blobs/sha256:def456"
        );
    }

    #[test]
    fn test_namespace_resolution() {
        // Bare name → library/name
        let url = blob_url("llama3.2", "sha256:abc");
        assert!(url.contains("/library/llama3.2/"));

        // Already namespaced → keep as-is
        let url = blob_url("user/model", "sha256:abc");
        assert!(url.contains("/user/model/"));
        assert!(!url.contains("/library/"));
    }

    #[test]
    fn test_extract_model_layer() {
        let layers = vec![
            LayerDescriptor {
                media_type: MEDIA_TEMPLATE.to_string(),
                digest: "sha256:template".to_string(),
                size: 1429,
            },
            LayerDescriptor {
                media_type: MEDIA_MODEL.to_string(),
                digest: "sha256:model".to_string(),
                size: 2_000_000_000,
            },
            LayerDescriptor {
                media_type: MEDIA_LICENSE.to_string(),
                digest: "sha256:license".to_string(),
                size: 7711,
            },
        ];

        let model = find_layer(&layers, MEDIA_MODEL).unwrap();
        assert_eq!(model.digest, "sha256:model");
        assert_eq!(model.size, 2_000_000_000);
    }

    #[test]
    fn test_extract_template_layer() {
        let layers = vec![
            LayerDescriptor {
                media_type: MEDIA_MODEL.to_string(),
                digest: "sha256:model".to_string(),
                size: 2_000_000_000,
            },
            LayerDescriptor {
                media_type: MEDIA_TEMPLATE.to_string(),
                digest: "sha256:tmpl".to_string(),
                size: 1429,
            },
        ];

        let template = find_layer(&layers, MEDIA_TEMPLATE).unwrap();
        assert_eq!(template.digest, "sha256:tmpl");
    }

    #[test]
    fn test_extract_missing_layer_returns_none() {
        let layers = vec![LayerDescriptor {
            media_type: MEDIA_MODEL.to_string(),
            digest: "sha256:model".to_string(),
            size: 100,
        }];

        assert!(find_layer(&layers, MEDIA_TEMPLATE).is_none());
        assert!(find_layer(&layers, MEDIA_SYSTEM).is_none());
        assert!(find_layer(&layers, MEDIA_PARAMS).is_none());
    }

    #[test]
    fn test_find_all_license_layers() {
        let layers = vec![
            LayerDescriptor {
                media_type: MEDIA_LICENSE.to_string(),
                digest: "sha256:lic1".to_string(),
                size: 7711,
            },
            LayerDescriptor {
                media_type: MEDIA_MODEL.to_string(),
                digest: "sha256:model".to_string(),
                size: 2_000_000_000,
            },
            LayerDescriptor {
                media_type: MEDIA_LICENSE.to_string(),
                digest: "sha256:lic2".to_string(),
                size: 6016,
            },
        ];

        let licenses = find_all_layers(&layers, MEDIA_LICENSE);
        assert_eq!(licenses.len(), 2);
        assert_eq!(licenses[0].digest, "sha256:lic1");
        assert_eq!(licenses[1].digest, "sha256:lic2");
    }

    #[test]
    fn test_parse_params_blob() {
        let json = r#"{"stop":["<|start_header_id|>","<|end_header_id|>","<|eot_id|>"]}"#;
        let params: HashMap<String, serde_json::Value> = serde_json::from_str(json).unwrap();
        assert!(params.contains_key("stop"));
        let stop = params["stop"].as_array().unwrap();
        assert_eq!(stop.len(), 3);
    }

    #[test]
    fn test_manifest_no_model_layer_errors() {
        let layers = vec![
            LayerDescriptor {
                media_type: MEDIA_TEMPLATE.to_string(),
                digest: "sha256:tmpl".to_string(),
                size: 100,
            },
            LayerDescriptor {
                media_type: MEDIA_LICENSE.to_string(),
                digest: "sha256:lic".to_string(),
                size: 200,
            },
        ];

        // No model layer → find_layer returns None
        assert!(find_layer(&layers, MEDIA_MODEL).is_none());
    }

    #[test]
    fn test_parse_parameter_count_billions() {
        assert_eq!(parse_parameter_count("3.2B"), Some(3_200_000_000));
        assert_eq!(parse_parameter_count("7B"), Some(7_000_000_000));
        assert_eq!(parse_parameter_count("70B"), Some(70_000_000_000));
        assert_eq!(parse_parameter_count("0.5B"), Some(500_000_000));
    }

    #[test]
    fn test_parse_parameter_count_millions() {
        assert_eq!(parse_parameter_count("125M"), Some(125_000_000));
        assert_eq!(parse_parameter_count("350M"), Some(350_000_000));
    }

    #[test]
    fn test_parse_parameter_count_invalid() {
        assert_eq!(parse_parameter_count("unknown"), None);
        assert_eq!(parse_parameter_count(""), None);
        assert_eq!(parse_parameter_count("3.2"), None);
    }

    #[test]
    fn test_parse_parameter_count_case_insensitive() {
        assert_eq!(parse_parameter_count("3.2b"), Some(3_200_000_000));
        assert_eq!(parse_parameter_count("125m"), Some(125_000_000));
    }

    #[test]
    fn test_ollama_model_config_default() {
        let config = OllamaModelConfig::default();
        assert!(config.model_format.is_none());
        assert!(config.model_family.is_none());
        assert!(config.model_type.is_none());
        assert!(config.file_type.is_none());
    }
}
