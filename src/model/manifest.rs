use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Describes a locally stored model and its metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Model name, e.g. "llama3.2:3b"
    pub name: String,

    /// File format of the model weights
    pub format: ModelFormat,

    /// Total size in bytes
    pub size: u64,

    /// SHA-256 hash of the model file for integrity verification
    pub sha256: String,

    /// Model parameters and metadata
    pub parameters: Option<ModelParameters>,

    /// Timestamp when the model was pulled/created locally
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Path to the model blob on disk (content-addressed)
    pub path: PathBuf,

    /// System prompt from Modelfile (if set)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Chat template override from Modelfile
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template_override: Option<String>,

    /// Default generation parameters from Modelfile
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_parameters: Option<HashMap<String, serde_json::Value>>,

    /// Raw Modelfile content (if model was created via Modelfile)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modelfile_content: Option<String>,

    /// License text from Ollama registry (if available)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// LoRA/QLoRA adapter path (from Modelfile ADAPTER directive)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_path: Option<String>,

    /// Multimodal projector path (for vision models like llava)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projector_path: Option<String>,

    /// Pre-seeded conversation messages (from Modelfile MESSAGE directive)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<ManifestMessage>,

    /// Model family (e.g. "llama", "phi", "gemma") from Ollama registry config
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,

    /// Model families for multimodal models (e.g. ["llama", "clip"])
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub families: Option<Vec<String>>,
}

/// A pre-seeded message stored in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestMessage {
    pub role: String,
    pub content: String,
}

/// Supported model file formats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelFormat::Gguf => write!(f, "GGUF"),
            ModelFormat::SafeTensors => write!(f, "SafeTensors"),
        }
    }
}

/// Optional parameter metadata about a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Maximum context length in tokens
    pub context_length: Option<u32>,

    /// Embedding dimension size
    pub embedding_length: Option<u32>,

    /// Total number of model parameters
    pub parameter_count: Option<u64>,

    /// Quantization level, e.g. "Q4_K_M", "Q8_0"
    pub quantization: Option<String>,
}

impl ModelManifest {
    /// Returns the sanitized filename for the manifest file.
    /// Replaces ':' and '/' with '-' to produce a safe filename.
    pub fn manifest_filename(&self) -> String {
        let safe = self.name.replace([':', '/'], "-");
        format!("{safe}.json")
    }

    /// Returns a human-readable size string (e.g. "4.2 GB").
    pub fn size_display(&self) -> String {
        const GB: u64 = 1_000_000_000;
        const MB: u64 = 1_000_000;
        const KB: u64 = 1_000;

        if self.size >= GB {
            format!("{:.1} GB", self.size as f64 / GB as f64)
        } else if self.size >= MB {
            format!("{:.1} MB", self.size as f64 / MB as f64)
        } else if self.size >= KB {
            format!("{:.1} KB", self.size as f64 / KB as f64)
        } else {
            format!("{} B", self.size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> ModelManifest {
        ModelManifest {
            name: "llama3.2:3b".to_string(),
            format: ModelFormat::Gguf,
            size: 2_000_000_000,
            sha256: "abc123".to_string(),
            parameters: Some(ModelParameters {
                context_length: Some(4096),
                embedding_length: Some(3200),
                parameter_count: Some(3_000_000_000),
                quantization: Some("Q4_K_M".to_string()),
            }),
            created_at: chrono::Utc::now(),
            path: PathBuf::from("/tmp/blobs/sha256-abc123"),
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
        }
    }

    #[test]
    fn test_manifest_filename() {
        let manifest = sample_manifest();
        assert_eq!(manifest.manifest_filename(), "llama3.2-3b.json");
    }

    #[test]
    fn test_manifest_filename_with_slash() {
        let mut manifest = sample_manifest();
        manifest.name = "library/llama3:latest".to_string();
        assert_eq!(manifest.manifest_filename(), "library-llama3-latest.json");
    }

    #[test]
    fn test_size_display_gb() {
        let manifest = sample_manifest();
        assert_eq!(manifest.size_display(), "2.0 GB");
    }

    #[test]
    fn test_size_display_mb() {
        let mut manifest = sample_manifest();
        manifest.size = 500_000_000;
        assert_eq!(manifest.size_display(), "500.0 MB");
    }

    #[test]
    fn test_size_display_kb() {
        let mut manifest = sample_manifest();
        manifest.size = 1_500;
        assert_eq!(manifest.size_display(), "1.5 KB");
    }

    #[test]
    fn test_size_display_bytes() {
        let mut manifest = sample_manifest();
        manifest.size = 512;
        assert_eq!(manifest.size_display(), "512 B");
    }

    #[test]
    fn test_model_format_display() {
        assert_eq!(ModelFormat::Gguf.to_string(), "GGUF");
        assert_eq!(ModelFormat::SafeTensors.to_string(), "SafeTensors");
    }

    #[test]
    fn test_manifest_serialization_roundtrip() {
        let manifest = sample_manifest();
        let json = serde_json::to_string(&manifest).unwrap();
        let deserialized: ModelManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, manifest.name);
        assert_eq!(deserialized.format, manifest.format);
        assert_eq!(deserialized.size, manifest.size);
        assert_eq!(deserialized.sha256, manifest.sha256);
    }

    #[test]
    fn test_manifest_optional_fields_serialization() {
        let mut manifest = sample_manifest();
        manifest.system_prompt = Some("You are helpful.".to_string());
        manifest.template_override = Some("{{ .System }}".to_string());
        manifest.license = Some("MIT".to_string());
        manifest.adapter_path = Some("/tmp/adapter.bin".to_string());
        manifest.projector_path = Some("/tmp/projector.bin".to_string());
        manifest.family = Some("llama".to_string());
        manifest.families = Some(vec!["llama".to_string(), "clip".to_string()]);
        manifest.messages = vec![
            ManifestMessage { role: "user".to_string(), content: "hi".to_string() },
            ManifestMessage { role: "assistant".to_string(), content: "hello".to_string() },
        ];

        let json = serde_json::to_string(&manifest).unwrap();
        let deserialized: ModelManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.system_prompt.as_deref(), Some("You are helpful."));
        assert_eq!(deserialized.template_override.as_deref(), Some("{{ .System }}"));
        assert_eq!(deserialized.license.as_deref(), Some("MIT"));
        assert_eq!(deserialized.adapter_path.as_deref(), Some("/tmp/adapter.bin"));
        assert_eq!(deserialized.projector_path.as_deref(), Some("/tmp/projector.bin"));
        assert_eq!(deserialized.family.as_deref(), Some("llama"));
        assert_eq!(deserialized.families.as_ref().unwrap().len(), 2);
        assert_eq!(deserialized.messages.len(), 2);
        assert_eq!(deserialized.messages[0].role, "user");
    }

    #[test]
    fn test_manifest_default_parameters_serialization() {
        let mut manifest = sample_manifest();
        let mut params = HashMap::new();
        params.insert("temperature".to_string(), serde_json::json!(0.7));
        params.insert("top_p".to_string(), serde_json::json!(0.9));
        manifest.default_parameters = Some(params);

        let json = serde_json::to_string(&manifest).unwrap();
        let deserialized: ModelManifest = serde_json::from_str(&json).unwrap();
        let defaults = deserialized.default_parameters.unwrap();
        assert_eq!(defaults["temperature"], 0.7);
        assert_eq!(defaults["top_p"], 0.9);
    }

    #[test]
    fn test_manifest_skip_serializing_none_fields() {
        let manifest = sample_manifest();
        let json = serde_json::to_string(&manifest).unwrap();
        // None fields with skip_serializing_if should not appear
        assert!(!json.contains("system_prompt"));
        assert!(!json.contains("template_override"));
        assert!(!json.contains("license"));
        assert!(!json.contains("adapter_path"));
        assert!(!json.contains("projector_path"));
    }

    #[test]
    fn test_model_format_serde() {
        let gguf_json = serde_json::to_string(&ModelFormat::Gguf).unwrap();
        assert_eq!(gguf_json, "\"gguf\"");
        let st_json = serde_json::to_string(&ModelFormat::SafeTensors).unwrap();
        assert_eq!(st_json, "\"safetensors\"");

        let gguf: ModelFormat = serde_json::from_str("\"gguf\"").unwrap();
        assert_eq!(gguf, ModelFormat::Gguf);
        let st: ModelFormat = serde_json::from_str("\"safetensors\"").unwrap();
        assert_eq!(st, ModelFormat::SafeTensors);
    }

    #[test]
    fn test_model_parameters_all_none() {
        let params = ModelParameters {
            context_length: None,
            embedding_length: None,
            parameter_count: None,
            quantization: None,
        };
        let json = serde_json::to_string(&params).unwrap();
        let deserialized: ModelParameters = serde_json::from_str(&json).unwrap();
        assert!(deserialized.context_length.is_none());
        assert!(deserialized.embedding_length.is_none());
        assert!(deserialized.parameter_count.is_none());
        assert!(deserialized.quantization.is_none());
    }

    #[test]
    fn test_manifest_filename_no_special_chars() {
        let mut manifest = sample_manifest();
        manifest.name = "simple-model".to_string();
        assert_eq!(manifest.manifest_filename(), "simple-model.json");
    }

    #[test]
    fn test_size_display_zero() {
        let mut manifest = sample_manifest();
        manifest.size = 0;
        assert_eq!(manifest.size_display(), "0 B");
    }
}
