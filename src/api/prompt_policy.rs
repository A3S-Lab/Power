//! Shared runtime prompt and execution policy digest builders.
//!
//! These helpers are used by both `/v1/attestation?model=...` and per-request
//! inference receipts so verifiers can compare the same runtime-policy semantics
//! across model-bound attestation and request-bound receipts.

use sha2::{Digest, Sha256};

use crate::config::{GpuConfig, PowerConfig};
use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest};
use crate::speculative::SpeculativeStrategy;
use crate::tee::attestation::{ExecutionPolicyClaim, PromptPolicyClaim, RuntimePolicyClaim};

pub(crate) fn runtime_policy_claim(
    manifest: &ModelManifest,
    config: Option<&PowerConfig>,
) -> Result<Option<RuntimePolicyClaim>> {
    let mut runtime = RuntimePolicyClaim::new();

    if let Some(prompt) = prompt_policy_claim(manifest)? {
        runtime = runtime.with_prompt(prompt);
    }

    if let Some(config) = config.filter(|_| manifest.format != ModelFormat::Remote) {
        runtime = runtime.with_execution(ExecutionPolicyClaim {
            gpu_sha256: canonical_gpu_execution_digest(&config.gpu)?,
            inference_sha256: Some(canonical_inference_execution_digest(config)?),
            auxiliary_artifacts_sha256: canonical_auxiliary_artifacts_digest(manifest)?,
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
    gpu.validate()?;
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
        "cpu_tensors".to_string(),
        serde_json::Value::Array(
            gpu.cpu_tensors
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        ),
    );
    if !gpu.gpu_tensors.is_empty() {
        canonical.insert(
            "gpu_tensors".to_string(),
            serde_json::Value::Array(
                gpu.gpu_tensors
                    .iter()
                    .cloned()
                    .map(serde_json::Value::String)
                    .collect(),
            ),
        );
    }
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

/// Return the SHA-256 digest of Power's canonical server-side inference policy.
///
/// This binds the configured speculative-decoding, prompt-cache, memory-load,
/// threading, Flash Attention, and request-slot settings. Request-specific
/// overrides remain bound by inference receipts. `spec_mode = "auto"` is
/// intentionally distinct from every explicit strategy; callers that need to
/// prove one exact decoder must configure and pin that explicit mode.
pub fn canonical_inference_execution_digest(config: &PowerConfig) -> Result<Vec<u8>> {
    let spec_mode = SpeculativeStrategy::parse(&config.spec_mode).ok_or_else(|| {
        PowerError::Config(format!("unsupported spec_mode '{}'", config.spec_mode))
    })?;
    let draft_p_min = canonical_named_f32(config.spec_draft_p_min, "spec_draft_p_min")?;
    let keep_alive_seconds = crate::config::parse_keep_alive(&config.keep_alive)
        .map_err(PowerError::Config)?
        .as_secs();

    let canonical = serde_json::json!({
        "flash_attention": config.flash_attention,
        "keep_alive_seconds": keep_alive_seconds,
        "max_loaded_models": config.max_loaded_models,
        "num_parallel": config.num_parallel,
        "num_thread": config.num_thread,
        "prompt_cache_max_entries": config.prompt_cache_max_entries,
        "prompt_cache_ttl_seconds": config.prompt_cache_ttl_seconds,
        "spec_draft_max": config.spec_draft_max,
        "spec_draft_min": config.spec_draft_min,
        "spec_draft_p_min": draft_p_min,
        "spec_mode": spec_mode.as_str(),
        "spec_mtp_adaptive": config.spec_mtp_adaptive,
        "spec_mtp_fr_vocab_size": config.spec_mtp_fr_vocab_size,
        "spec_mtp_recurrent_chain": config.spec_mtp_recurrent_chain,
        "spec_mtp_recurrent_snapshots": config.spec_mtp_recurrent_snapshots,
        "use_mlock": config.use_mlock,
        "use_mmap": config.use_mmap,
    });
    let bytes = serde_json::to_vec(&canonical)?;
    let mut hasher = Sha256::new();
    hasher.update(b"a3s.power.inference-execution.v1\0");
    hasher.update(bytes);
    Ok(hasher.finalize().to_vec())
}

/// Return the portable digest of every content-addressed auxiliary inference
/// artifact declared by a model manifest.
///
/// Paths are deliberately excluded because they are host-local locators. The
/// role, decoder contract, byte length, and SHA-256 identity are bound. Legacy
/// path-only adapter/projector manifests return `None`; strict TEE startup
/// rejects those manifests before inference.
pub fn canonical_auxiliary_artifacts_digest(manifest: &ModelManifest) -> Result<Option<Vec<u8>>> {
    manifest.validate_auxiliary_artifact_bindings(false)?;
    if (manifest.adapter_path.is_some() && manifest.adapter_artifact.is_none())
        || (manifest.projector_path.is_some() && manifest.projector_artifact.is_none())
    {
        return Ok(None);
    }

    let mut records = Vec::new();
    if let Some(draft) = &manifest.external_draft {
        draft
            .validate_for_target(&manifest.sha256)
            .map_err(PowerError::Config)?;
        records.push(AuxiliaryArtifactDigestRecord {
            role: "external-draft",
            contract: Some(draft.kind.as_str()),
            size: draft.size,
            sha256: draft.sha256.as_str(),
            target_sha256: Some(draft.target_sha256.as_str()),
        });
    }
    if let Some(adapter) = &manifest.adapter_artifact {
        adapter.validate("LoRA adapter")?;
        records.push(AuxiliaryArtifactDigestRecord {
            role: "lora-adapter",
            contract: None,
            size: adapter.size,
            sha256: adapter.sha256.as_str(),
            target_sha256: None,
        });
    }
    if let Some(projector) = &manifest.projector_artifact {
        projector.validate("Multimodal projector")?;
        records.push(AuxiliaryArtifactDigestRecord {
            role: "multimodal-projector",
            contract: None,
            size: projector.size,
            sha256: projector.sha256.as_str(),
            target_sha256: None,
        });
    }
    if records.is_empty() {
        return Ok(None);
    }

    let mut hasher = Sha256::new();
    hasher.update(b"a3s.power.auxiliary-artifacts.v1\0");
    hasher.update((records.len() as u64).to_le_bytes());
    for record in records {
        update_length_prefixed(&mut hasher, record.role.as_bytes());
        update_length_prefixed(&mut hasher, record.contract.unwrap_or_default().as_bytes());
        hasher.update(record.size.to_le_bytes());
        let sha256 = hex::decode(record.sha256).map_err(|error| {
            PowerError::Config(format!(
                "{} sha256 is not valid hexadecimal: {error}",
                record.role
            ))
        })?;
        update_length_prefixed(&mut hasher, &sha256);
        let target_sha256 = record
            .target_sha256
            .map(hex::decode)
            .transpose()
            .map_err(|error| {
                PowerError::Config(format!(
                    "{} target sha256 is not valid hexadecimal: {error}",
                    record.role
                ))
            })?;
        update_length_prefixed(&mut hasher, target_sha256.as_deref().unwrap_or_default());
    }
    Ok(Some(hasher.finalize().to_vec()))
}

struct AuxiliaryArtifactDigestRecord<'a> {
    role: &'static str,
    contract: Option<&'static str>,
    size: u64,
    sha256: &'a str,
    target_sha256: Option<&'a str>,
}

fn update_length_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn canonical_f32(value: f32, index: usize) -> Result<serde_json::Value> {
    canonical_named_f32(value, &format!("gpu.tensor_split[{index}]"))
}

fn canonical_named_f32(value: f32, field: &str) -> Result<serde_json::Value> {
    if !value.is_finite() {
        return Err(PowerError::Config(format!("{field} must be finite")));
    }
    let number = serde_json::Number::from_f64(value as f64)
        .ok_or_else(|| PowerError::Config(format!("{field} cannot be represented as JSON")))?;
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
            adapter_artifact: None,
            external_draft: None,
            projector_path: None,
            projector_artifact: None,
            messages: Vec::new(),
            family: None,
            families: None,
        }
    }

    #[test]
    fn runtime_policy_empty_manifest_returns_none() {
        assert!(runtime_policy_claim(&manifest(), None).unwrap().is_none());
    }

    #[test]
    fn runtime_policy_does_not_claim_local_execution_for_remote_model() {
        let remote = ModelManifest::remote("upstream-model");

        assert!(runtime_policy_claim(&remote, Some(&PowerConfig::default()))
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

        let runtime = runtime_policy_claim(&manifest, None).unwrap().unwrap();
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

        assert!(runtime_policy_claim(&manifest, None).unwrap().is_none());
    }

    #[test]
    fn runtime_policy_gpu_execution_digest_is_stable() {
        let gpu = GpuConfig {
            gpu_layers: -1,
            main_gpu: 0,
            tensor_split: vec![0.5, 0.5],
            cpu_tensors: Vec::new(),
            gpu_tensors: Vec::new(),
        };
        let config = PowerConfig {
            gpu: gpu.clone(),
            ..PowerConfig::default()
        };
        let runtime = runtime_policy_claim(&manifest(), Some(&config))
            .unwrap()
            .unwrap();

        assert_eq!(
            runtime.execution.as_ref().unwrap().gpu_sha256,
            canonical_gpu_execution_digest(&gpu).unwrap()
        );
        assert_eq!(
            runtime.execution.as_ref().unwrap().inference_sha256,
            Some(canonical_inference_execution_digest(&config).unwrap())
        );
    }

    #[test]
    fn runtime_policy_gpu_execution_digest_changes_with_gpu_layers() {
        let a = GpuConfig {
            gpu_layers: -1,
            main_gpu: 0,
            tensor_split: Vec::new(),
            cpu_tensors: Vec::new(),
            gpu_tensors: Vec::new(),
        };
        let b = GpuConfig {
            gpu_layers: 0,
            main_gpu: 0,
            tensor_split: Vec::new(),
            cpu_tensors: Vec::new(),
            gpu_tensors: Vec::new(),
        };

        assert_ne!(
            canonical_gpu_execution_digest(&a).unwrap(),
            canonical_gpu_execution_digest(&b).unwrap()
        );
    }

    #[test]
    fn runtime_policy_gpu_execution_digest_changes_with_cpu_tensor_placement() {
        let base = GpuConfig {
            gpu_layers: -1,
            ..Default::default()
        };
        let placed = GpuConfig {
            cpu_tensors: vec!["output.weight".to_string()],
            ..base.clone()
        };

        assert_ne!(
            canonical_gpu_execution_digest(&base).unwrap(),
            canonical_gpu_execution_digest(&placed).unwrap()
        );
    }

    #[test]
    fn runtime_policy_gpu_execution_digest_changes_with_gpu_tensor_placement() {
        let base = GpuConfig {
            gpu_layers: -1,
            ..Default::default()
        };
        let placed = GpuConfig {
            gpu_tensors: vec!["token_embd.weight".to_string()],
            ..base.clone()
        };

        assert_ne!(
            canonical_gpu_execution_digest(&base).unwrap(),
            canonical_gpu_execution_digest(&placed).unwrap()
        );
    }

    #[test]
    fn inference_execution_digest_changes_with_optimization_policy() {
        let baseline = PowerConfig::default();
        let tuned = PowerConfig {
            spec_mode: "mtp".to_string(),
            spec_draft_max: Some(7),
            flash_attention: true,
            ..baseline.clone()
        };

        assert_ne!(
            canonical_inference_execution_digest(&baseline).unwrap(),
            canonical_inference_execution_digest(&tuned).unwrap()
        );
    }

    #[test]
    fn inference_execution_digest_normalizes_strategy_aliases_and_keep_alive() {
        let first = PowerConfig {
            spec_mode: "prompt_lookup".to_string(),
            keep_alive: "5m".to_string(),
            ..PowerConfig::default()
        };
        let second = PowerConfig {
            spec_mode: "prompt-lookup".to_string(),
            keep_alive: "300".to_string(),
            ..PowerConfig::default()
        };

        assert_eq!(
            canonical_inference_execution_digest(&first).unwrap(),
            canonical_inference_execution_digest(&second).unwrap()
        );
    }

    #[test]
    fn inference_execution_digest_rejects_invalid_float() {
        let config = PowerConfig {
            spec_draft_p_min: f32::NAN,
            ..PowerConfig::default()
        };

        assert!(canonical_inference_execution_digest(&config)
            .unwrap_err()
            .to_string()
            .contains("spec_draft_p_min must be finite"));
    }

    #[test]
    fn auxiliary_artifact_digest_is_portable_and_content_sensitive() {
        let first_directory = tempfile::tempdir().unwrap();
        let second_directory = tempfile::tempdir().unwrap();
        let mut first = manifest();
        let first_path = first_directory.path().join("projector.gguf");
        first.projector_path = Some(first_path.display().to_string());
        first.projector_artifact = Some(crate::model::artifact::AuxiliaryModelArtifact {
            path: first_path,
            size: 128,
            sha256: "ab".repeat(32),
        });
        let mut relocated = first.clone();
        let relocated_path = second_directory.path().join("projector.gguf");
        relocated.projector_path = Some(relocated_path.display().to_string());
        relocated.projector_artifact.as_mut().unwrap().path = relocated_path;
        let mut replaced = first.clone();
        replaced.projector_artifact.as_mut().unwrap().sha256 = "cd".repeat(32);

        let first_digest = canonical_auxiliary_artifacts_digest(&first)
            .unwrap()
            .unwrap();

        assert_eq!(
            first_digest,
            canonical_auxiliary_artifacts_digest(&relocated)
                .unwrap()
                .unwrap()
        );
        assert_ne!(
            first_digest,
            canonical_auxiliary_artifacts_digest(&replaced)
                .unwrap()
                .unwrap()
        );
    }

    #[test]
    fn auxiliary_artifact_digest_binds_external_draft_contract() {
        let mut dflash = manifest();
        dflash.sha256 = "11".repeat(32);
        dflash.external_draft = Some(crate::model::manifest::ExternalDraftArtifact {
            kind: crate::model::manifest::ExternalDraftKind::Dflash,
            path: PathBuf::from("draft.gguf"),
            size: 256,
            sha256: "22".repeat(32),
            target_sha256: dflash.sha256.clone(),
            source: None,
            revision: None,
            license: None,
        });
        let mut dflash2 = dflash.clone();
        dflash2.external_draft.as_mut().unwrap().kind =
            crate::model::manifest::ExternalDraftKind::Dflash2;

        assert_ne!(
            canonical_auxiliary_artifacts_digest(&dflash).unwrap(),
            canonical_auxiliary_artifacts_digest(&dflash2).unwrap()
        );
    }

    #[test]
    fn legacy_path_only_auxiliary_artifact_has_no_attestable_digest() {
        let mut legacy = manifest();
        legacy.projector_path = Some("projector.gguf".to_string());

        assert!(canonical_auxiliary_artifacts_digest(&legacy)
            .unwrap()
            .is_none());
    }

    #[test]
    fn runtime_policy_includes_auxiliary_artifact_digest() {
        let directory = tempfile::tempdir().unwrap();
        let mut model = manifest();
        model.adapter_artifact = Some(crate::model::artifact::AuxiliaryModelArtifact {
            path: directory.path().join("adapter.gguf"),
            size: 64,
            sha256: "ef".repeat(32),
        });
        let gpu = GpuConfig::default();

        let config = PowerConfig {
            gpu,
            ..PowerConfig::default()
        };
        let runtime = runtime_policy_claim(&model, Some(&config))
            .unwrap()
            .unwrap();

        assert_eq!(
            runtime.execution.unwrap().auxiliary_artifacts_sha256,
            canonical_auxiliary_artifacts_digest(&model).unwrap()
        );
    }
}
