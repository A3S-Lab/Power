//! Shared runtime prompt and execution policy digest builders.
//!
//! These helpers are used by both `/v1/attestation?model=...` and per-request
//! inference receipts so verifiers can compare the same runtime-policy semantics
//! across model-bound attestation and request-bound receipts.

use sha2::{Digest, Sha256};

use crate::config::GpuConfig;
use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest};
use crate::tee::attestation::{ExecutionPolicyClaim, PromptPolicyClaim, RuntimePolicyClaim};

pub(crate) fn runtime_policy_claim_with_gpu_config(
    manifest: &ModelManifest,
    gpu: Option<&GpuConfig>,
) -> Result<Option<RuntimePolicyClaim>> {
    let mut runtime = RuntimePolicyClaim::new();

    if let Some(prompt) = prompt_policy_claim(manifest)? {
        runtime = runtime.with_prompt(prompt);
    }

    if let Some(gpu) = gpu {
        runtime = runtime.with_execution(ExecutionPolicyClaim {
            gpu_sha256: canonical_gpu_execution_digest(gpu)?,
        });
    }

    if runtime.prompt.is_none() && runtime.decoding.is_none() && runtime.execution.is_none() {
        return Ok(None);
    }

    Ok(Some(runtime))
}

fn prompt_policy_claim(
    manifest: &ModelManifest,
) -> crate::error::Result<Option<PromptPolicyClaim>> {
    let mut prompt = PromptPolicyClaim {
        chat_template_source: None,
        chat_template_sha256: None,
        system_prompt_sha256: None,
        messages_sha256: None,
    };

    if let Some((source, template)) = effective_chat_template(manifest) {
        prompt.chat_template_source = Some(source.to_string());
        prompt.chat_template_sha256 = Some(sha256_bytes(template.as_bytes()));
    }

    if prompt.is_empty() {
        return Ok(None);
    }

    Ok(Some(prompt))
}

fn gguf_chat_template(manifest: &ModelManifest) -> Option<String> {
    if manifest.format != ModelFormat::Gguf || !manifest.path.is_file() {
        return None;
    }

    match crate::model::gguf::read_metadata(&manifest.path) {
        Ok(metadata) => match metadata.metadata.get("tokenizer.chat_template") {
            Some(crate::model::gguf::GgufValue::String(template)) => Some(template.clone()),
            _ => None,
        },
        Err(e) => {
            tracing::debug!(
                model = %manifest.name,
                path = %manifest.path.display(),
                error = %e,
                "GGUF chat template metadata unavailable for runtime policy digest"
            );
            None
        }
    }
}

fn effective_chat_template(manifest: &ModelManifest) -> Option<(&'static str, String)> {
    if let Some(template) = &manifest.template_override {
        return Some(("manifest.template_override", template.clone()));
    }

    gguf_chat_template(manifest).map(|template| ("gguf.tokenizer.chat_template", template))
}

/// Return the SHA-256 digest of Power's canonical GPU execution/offload policy.
///
/// This digest is emitted in `claims.runtime.execution.gpu_sha256` and can be
/// pinned by verifiers that require an exact execution/offload configuration.
pub fn canonical_gpu_execution_digest(gpu: &GpuConfig) -> Result<Vec<u8>> {
    let mut canonical = serde_json::Map::new();
    canonical.insert(
        "gpu_layers".to_string(),
        serde_json::Value::Number(gpu.gpu_layers.into()),
    );
    canonical.insert(
        "main_gpu".to_string(),
        serde_json::Value::Number(gpu.main_gpu.into()),
    );
    canonical.insert(
        "tensor_split".to_string(),
        serde_json::Value::Array(
            gpu.tensor_split
                .iter()
                .enumerate()
                .map(|(index, value)| canonical_f32(*value, index))
                .collect::<Result<Vec<_>>>()?,
        ),
    );

    let bytes = serde_json::to_vec(&serde_json::Value::Object(canonical))?;
    Ok(sha256_bytes(&bytes))
}

fn canonical_f32(value: f32, index: usize) -> Result<serde_json::Value> {
    if !value.is_finite() {
        return Err(PowerError::Config(format!(
            "gpu.tensor_split[{index}] must be finite"
        )));
    }
    let number = serde_json::Number::from_f64(value as f64).ok_or_else(|| {
        PowerError::Config(format!(
            "gpu.tensor_split[{index}] cannot be represented as JSON"
        ))
    })?;
    Ok(serde_json::Value::Number(number))
}

fn sha256_bytes(bytes: &[u8]) -> Vec<u8> {
    Sha256::digest(bytes).to_vec()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::*;
    use crate::model::manifest::{ManifestMessage, ModelManifest};

    fn manifest() -> ModelManifest {
        ModelManifest {
            name: "test".to_string(),
            format: ModelFormat::Gguf,
            size: 0,
            sha256: "hash".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: PathBuf::from("/tmp/nonexistent.gguf"),
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: Vec::new(),
            family: None,
            families: None,
        }
    }

    #[test]
    fn runtime_policy_empty_manifest_returns_none() {
        assert!(runtime_policy_claim_with_gpu_config(&manifest(), None)
            .unwrap()
            .is_none());
    }

    #[test]
    fn runtime_policy_hashes_applied_chat_template_only() {
        let mut manifest = manifest();
        manifest.template_override = Some("{{ messages }}".to_string());
        manifest.system_prompt = Some("system".to_string());
        manifest.messages = vec![ManifestMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }];

        let runtime = runtime_policy_claim_with_gpu_config(&manifest, None)
            .unwrap()
            .unwrap();
        let prompt = runtime.prompt.unwrap();
        assert_eq!(
            prompt.chat_template_source.as_deref(),
            Some("manifest.template_override")
        );
        assert!(prompt.chat_template_sha256.is_some());
        assert!(prompt.system_prompt_sha256.is_none());
        assert!(prompt.messages_sha256.is_none());
        assert!(runtime.decoding.is_none());
    }

    #[test]
    fn runtime_policy_does_not_claim_unapplied_manifest_defaults() {
        let mut manifest = manifest();
        manifest.system_prompt = Some("system".to_string());
        manifest.messages = vec![ManifestMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }];
        manifest.default_parameters = Some(HashMap::from([
            ("temperature".to_string(), serde_json::json!(0.2)),
            ("top_p".to_string(), serde_json::json!(0.9)),
        ]));

        assert!(runtime_policy_claim_with_gpu_config(&manifest, None)
            .unwrap()
            .is_none());
    }

    #[test]
    fn runtime_policy_gpu_execution_digest_is_stable() {
        let gpu = GpuConfig {
            gpu_layers: -1,
            main_gpu: 0,
            tensor_split: vec![0.5, 0.5],
        };
        let runtime = runtime_policy_claim_with_gpu_config(&manifest(), Some(&gpu))
            .unwrap()
            .unwrap();

        assert_eq!(
            runtime.execution.as_ref().unwrap().gpu_sha256,
            canonical_gpu_execution_digest(&gpu).unwrap()
        );
    }

    #[test]
    fn runtime_policy_gpu_execution_digest_changes_with_gpu_layers() {
        let a = GpuConfig {
            gpu_layers: -1,
            main_gpu: 0,
            tensor_split: Vec::new(),
        };
        let b = GpuConfig {
            gpu_layers: 0,
            main_gpu: 0,
            tensor_split: Vec::new(),
        };

        assert_ne!(
            canonical_gpu_execution_digest(&a).unwrap(),
            canonical_gpu_execution_digest(&b).unwrap()
        );
    }
}
