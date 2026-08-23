use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

use super::artifact::{validate_sha256, verify_regular_file_identity};
use super::gguf::{self, GgufMetadata, GgufValue};

/// Decoder contract implemented by an external GGUF draft artifact.
///
/// The kind is part of the artifact identity rather than a runtime hint. A
/// DSpark-trained head must not be started through the DFlash decoder (or vice
/// versa), even when both share the same GGUF architecture name.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ExternalDraftKind {
    Dflash,
    Dflash2,
    Dspark,
}

impl ExternalDraftKind {
    /// Stable configuration spelling for this decoder contract.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dflash => "dflash",
            Self::Dflash2 => "dflash2",
            Self::Dspark => "dspark",
        }
    }
}

/// Content-addressed external speculative-draft model bound to one target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalDraftArtifact {
    /// Decoder contract required by the trained draft head.
    pub kind: ExternalDraftKind,

    /// Path to the GGUF draft artifact.
    pub path: PathBuf,

    /// Exact artifact size in bytes.
    pub size: u64,

    /// SHA-256 digest of the draft GGUF.
    pub sha256: String,

    /// SHA-256 digest of the only target artifact this draft may accelerate.
    pub target_sha256: String,

    /// Source repository or immutable artifact locator, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Immutable upstream source revision, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,

    /// SPDX license identifier supplied by the artifact publisher, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
}

impl ExternalDraftArtifact {
    /// Capture and validate a local draft artifact for a content-addressed target.
    ///
    /// The caller supplies only provenance and the already-computed target
    /// digest. Size and digest for the draft are always measured by Power, so
    /// registration cannot trust client-supplied integrity metadata.
    pub fn capture_for_target(
        kind: ExternalDraftKind,
        path: PathBuf,
        target_sha256: String,
        source: Option<String>,
        revision: Option<String>,
        license: Option<String>,
    ) -> Result<Self> {
        validate_sha256("target model sha256", &target_sha256).map_err(PowerError::Config)?;
        if !path.is_absolute() {
            return Err(PowerError::Config(format!(
                "External draft path must be absolute: {}",
                path.display()
            )));
        }
        let metadata = std::fs::metadata(&path).map_err(|error| {
            PowerError::Config(format!(
                "Failed to inspect external draft {}: {error}",
                path.display()
            ))
        })?;
        if !metadata.is_file() {
            return Err(PowerError::Config(format!(
                "External draft is not a regular file: {}",
                path.display()
            )));
        }
        if metadata.len() == 0 {
            return Err(PowerError::Config(format!(
                "External draft is empty: {}",
                path.display()
            )));
        }

        let sha256 = super::storage::compute_sha256_file(&path)?;
        let gguf = gguf::read_metadata(&path)?;
        validate_gguf_contract(kind, &gguf, &path)?;

        Ok(Self {
            kind,
            path,
            size: metadata.len(),
            sha256,
            target_sha256: target_sha256.to_ascii_lowercase(),
            source,
            revision,
            license,
        })
    }

    /// Validate immutable identity fields before touching the filesystem.
    pub fn validate_for_target(&self, target_sha256: &str) -> std::result::Result<(), String> {
        validate_sha256("external draft sha256", &self.sha256)?;
        validate_sha256("external draft target_sha256", &self.target_sha256)?;
        if self.size == 0 {
            return Err("external draft size must be greater than zero".to_string());
        }
        if !self.target_sha256.eq_ignore_ascii_case(target_sha256) {
            return Err(format!(
                "external draft target_sha256 does not match target model sha256 ({})",
                target_sha256.to_ascii_lowercase()
            ));
        }
        Ok(())
    }

    /// Verify both target and draft identities plus the decoder-specific GGUF contract.
    ///
    /// Pairing only against a manifest string would allow a replaced target file
    /// to bypass the trained-head contract. The target is therefore re-hashed at
    /// load time whenever an external draft is selected.
    pub fn verify_for_target_file(
        &self,
        target_path: &Path,
        target_size: u64,
        target_sha256: &str,
    ) -> Result<VerifiedExternalDraft> {
        self.validate_for_target(target_sha256)
            .map_err(PowerError::Config)?;
        validate_sha256("target model sha256", target_sha256).map_err(PowerError::Config)?;
        if target_size == 0 {
            return Err(PowerError::Config(
                "target model size must be greater than zero when an external draft is selected"
                    .to_string(),
            ));
        }
        let verified_target_sha256 =
            verify_regular_file_identity("Target model", target_path, target_size, target_sha256)?;
        self.verify_for_verified_target_sha256(&verified_target_sha256)
    }

    /// Verify the draft after the caller has already re-hashed the target.
    pub(crate) fn verify_for_verified_target_sha256(
        &self,
        verified_target_sha256: &str,
    ) -> Result<VerifiedExternalDraft> {
        self.validate_for_target(verified_target_sha256)
            .map_err(PowerError::Config)?;
        let actual_sha256 =
            verify_regular_file_identity("External draft", &self.path, self.size, &self.sha256)?;
        let gguf = gguf::read_metadata(&self.path)?;
        validate_gguf_contract(self.kind, &gguf, &self.path)?;

        Ok(VerifiedExternalDraft {
            kind: self.kind,
            path: self.path.clone(),
            size: self.size,
            sha256: actual_sha256,
            target_sha256: verified_target_sha256.to_ascii_lowercase(),
        })
    }
}

/// Immutable external-draft identity after disk and GGUF verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExternalDraft {
    pub kind: ExternalDraftKind,
    pub path: PathBuf,
    pub size: u64,
    pub sha256: String,
    pub target_sha256: String,
}

fn validate_gguf_contract(kind: ExternalDraftKind, gguf: &GgufMetadata, path: &Path) -> Result<()> {
    let architecture = match gguf.metadata.get("general.architecture") {
        Some(GgufValue::String(value)) => value.as_str(),
        _ => "",
    };
    if architecture != "dflash" {
        return Err(PowerError::Config(format!(
            "External {} draft {} has GGUF architecture '{}', expected 'dflash'",
            kind.as_str(),
            path.display(),
            if architecture.is_empty() {
                "missing"
            } else {
                architecture
            }
        )));
    }
    match gguf.metadata.get("dflash.block_size") {
        Some(GgufValue::Uint32(value)) if *value > 1 => {}
        _ => {
            return Err(PowerError::Config(format!(
                "External {} draft {} is missing a valid dflash.block_size",
                kind.as_str(),
                path.display()
            )))
        }
    }
    match gguf.metadata.get("dflash.target_layers") {
        Some(GgufValue::Array(layers))
            if !layers.is_empty()
                && layers.iter().all(|layer| match layer {
                    GgufValue::Uint32(_) => true,
                    GgufValue::Int32(value) => *value >= 0,
                    _ => false,
                }) => {}
        _ => {
            return Err(PowerError::Config(format!(
                "External {} draft {} is missing dflash.target_layers",
                kind.as_str(),
                path.display()
            )))
        }
    }

    let has_tensor = |name: &str| gguf.tensors.iter().any(|tensor| tensor.name == name);
    let has_markov_w1 = has_tensor("markov_w1.weight");
    let has_markov_w2 = has_tensor("markov_w2.weight");
    let has_confidence = has_tensor("conf_proj.weight");
    let has_dflash2_metadata = [
        "dflash.conv_kernel_size",
        "dflash.conv_group_size",
        "dflash.selector_rank",
        "dflash.selector_top_k",
    ]
    .iter()
    .any(|key| gguf.metadata.contains_key(*key));
    let has_dflash2_tensor = gguf.tensors.iter().any(|tensor| {
        tensor.name.starts_with("selector_")
            || tensor.name.contains(".attn_conv_")
            || tensor.name.contains(".ffn_conv_")
    });
    match kind {
        ExternalDraftKind::Dspark if !has_markov_w1 || !has_markov_w2 || !has_confidence => {
            return Err(PowerError::Config(format!(
                "External DSpark draft {} is missing its Markov or confidence head",
                path.display()
            )))
        }
        ExternalDraftKind::Dspark if has_dflash2_metadata || has_dflash2_tensor => {
            return Err(PowerError::Config(format!(
                "External DSpark draft {} contains DFlash2 convolution or selector state",
                path.display()
            )))
        }
        ExternalDraftKind::Dflash | ExternalDraftKind::Dflash2
            if has_markov_w1 || has_markov_w2 || has_confidence =>
        {
            return Err(PowerError::Config(format!(
                "External {} draft {} contains a DSpark Markov head; declare it as kind 'dspark'",
                kind.as_str(),
                path.display()
            )))
        }
        ExternalDraftKind::Dflash if has_dflash2_metadata || has_dflash2_tensor => {
            return Err(PowerError::Config(format!(
                "External DFlash draft {} contains DFlash2 convolution or selector state; declare it as kind 'dflash2'",
                path.display()
            )))
        }
        _ => {}
    }

    if kind == ExternalDraftKind::Dflash2 {
        validate_dflash2_contract(gguf, path)?;
    }

    Ok(())
}

fn validate_dflash2_contract(gguf: &GgufMetadata, path: &Path) -> Result<()> {
    let positive_u32 = |key: &str| match gguf.metadata.get(key) {
        Some(GgufValue::Uint32(value)) if *value > 0 => Some(*value),
        _ => None,
    };
    let block_count = positive_u32("dflash.block_count").ok_or_else(|| {
        PowerError::Config(format!(
            "External DFlash2 draft {} is missing a valid dflash.block_count",
            path.display()
        ))
    })?;
    for key in [
        "dflash.conv_kernel_size",
        "dflash.conv_group_size",
        "dflash.selector_rank",
        "dflash.selector_top_k",
    ] {
        if positive_u32(key).is_none() {
            return Err(PowerError::Config(format!(
                "External DFlash2 draft {} is missing a valid {key}",
                path.display()
            )));
        }
    }

    let has_tensor = |name: &str| gguf.tensors.iter().any(|tensor| tensor.name == name);
    for name in [
        "selector_predecessor.weight",
        "selector_successor.weight",
        "selector_hidden.weight",
    ] {
        if !has_tensor(name) {
            return Err(PowerError::Config(format!(
                "External DFlash2 draft {} is missing selector tensor {name}",
                path.display()
            )));
        }
    }
    for layer in 0..block_count {
        for name in [
            format!("blk.{layer}.attn_conv_base"),
            format!("blk.{layer}.attn_conv_proj.weight"),
            format!("blk.{layer}.ffn_conv_base"),
            format!("blk.{layer}.ffn_conv_proj.weight"),
        ] {
            if !has_tensor(&name) {
                return Err(PowerError::Config(format!(
                    "External DFlash2 draft {} is missing convolution tensor {name}",
                    path.display()
                )));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::super::gguf::GgufTensor;
    use super::*;

    fn artifact() -> ExternalDraftArtifact {
        ExternalDraftArtifact {
            kind: ExternalDraftKind::Dspark,
            path: PathBuf::from("draft.gguf"),
            size: 42,
            sha256: "a".repeat(64),
            target_sha256: "B".repeat(64),
            source: None,
            revision: None,
            license: None,
        }
    }

    fn tensor(name: &str) -> GgufTensor {
        GgufTensor {
            name: name.to_string(),
            dimensions: vec![1],
            tensor_type: 0,
            offset: 0,
        }
    }

    fn gguf_with_tensors(names: &[&str]) -> GgufMetadata {
        GgufMetadata {
            version: 3,
            tensor_count: names.len() as u64,
            metadata: HashMap::from([
                (
                    "general.architecture".to_string(),
                    GgufValue::String("dflash".to_string()),
                ),
                ("dflash.block_size".to_string(), GgufValue::Uint32(16)),
                (
                    "dflash.target_layers".to_string(),
                    GgufValue::Array(vec![GgufValue::Int32(8), GgufValue::Uint32(16)]),
                ),
            ]),
            tensors: names.iter().map(|name| tensor(name)).collect(),
        }
    }

    fn dflash2_gguf() -> GgufMetadata {
        let mut gguf = gguf_with_tensors(&[
            "selector_predecessor.weight",
            "selector_successor.weight",
            "selector_hidden.weight",
            "blk.0.attn_conv_base",
            "blk.0.attn_conv_proj.weight",
            "blk.0.ffn_conv_base",
            "blk.0.ffn_conv_proj.weight",
        ]);
        gguf.metadata
            .insert("dflash.block_count".to_string(), GgufValue::Uint32(1));
        gguf.metadata
            .insert("dflash.conv_kernel_size".to_string(), GgufValue::Uint32(2));
        gguf.metadata
            .insert("dflash.conv_group_size".to_string(), GgufValue::Uint32(128));
        gguf.metadata
            .insert("dflash.selector_rank".to_string(), GgufValue::Uint32(64));
        gguf.metadata
            .insert("dflash.selector_top_k".to_string(), GgufValue::Uint32(16));
        gguf
    }

    #[test]
    fn identity_is_bound_to_target_digest() {
        let artifact = artifact();
        assert!(artifact.validate_for_target(&"b".repeat(64)).is_ok());
        let error = artifact.validate_for_target(&"c".repeat(64)).unwrap_err();
        assert!(error.contains("does not match"));
    }

    #[test]
    fn identity_rejects_malformed_digest_and_empty_file() {
        let mut artifact = artifact();
        artifact.sha256 = "not-a-digest".to_string();
        assert!(artifact
            .validate_for_target(&"b".repeat(64))
            .unwrap_err()
            .contains("64 hexadecimal"));

        artifact.sha256 = "a".repeat(64);
        artifact.size = 0;
        assert!(artifact
            .validate_for_target(&"b".repeat(64))
            .unwrap_err()
            .contains("greater than zero"));
    }

    #[test]
    fn dspark_contract_requires_complete_markov_and_confidence_heads() {
        let valid =
            gguf_with_tensors(&["markov_w1.weight", "markov_w2.weight", "conf_proj.weight"]);
        assert!(
            validate_gguf_contract(ExternalDraftKind::Dspark, &valid, Path::new("draft.gguf"))
                .is_ok()
        );

        for missing in ["markov_w1.weight", "markov_w2.weight", "conf_proj.weight"] {
            let names = ["markov_w1.weight", "markov_w2.weight", "conf_proj.weight"]
                .into_iter()
                .filter(|name| *name != missing)
                .collect::<Vec<_>>();
            let error = validate_gguf_contract(
                ExternalDraftKind::Dspark,
                &gguf_with_tensors(&names),
                Path::new("draft.gguf"),
            )
            .unwrap_err();
            assert!(error.to_string().contains("Markov or confidence head"));
        }
    }

    #[test]
    fn dflash_contract_rejects_every_dspark_only_tensor() {
        assert!(validate_gguf_contract(
            ExternalDraftKind::Dflash,
            &gguf_with_tensors(&[]),
            Path::new("draft.gguf")
        )
        .is_ok());

        for name in ["markov_w1.weight", "markov_w2.weight", "conf_proj.weight"] {
            let error = validate_gguf_contract(
                ExternalDraftKind::Dflash,
                &gguf_with_tensors(&[name]),
                Path::new("draft.gguf"),
            )
            .unwrap_err();
            assert!(error.to_string().contains("declare it as kind 'dspark'"));
        }
    }

    #[test]
    fn dflash2_contract_requires_complete_selector_and_convolution_stack() {
        let valid = dflash2_gguf();
        assert!(validate_gguf_contract(
            ExternalDraftKind::Dflash2,
            &valid,
            Path::new("draft.gguf")
        )
        .is_ok());

        for missing in [
            "selector_predecessor.weight",
            "selector_successor.weight",
            "selector_hidden.weight",
            "blk.0.attn_conv_base",
            "blk.0.attn_conv_proj.weight",
            "blk.0.ffn_conv_base",
            "blk.0.ffn_conv_proj.weight",
        ] {
            let mut incomplete = dflash2_gguf();
            incomplete.tensors.retain(|tensor| tensor.name != missing);
            let error = validate_gguf_contract(
                ExternalDraftKind::Dflash2,
                &incomplete,
                Path::new("draft.gguf"),
            )
            .unwrap_err();
            assert!(error.to_string().contains("DFlash2"));
        }
    }

    #[test]
    fn dflash_and_dflash2_contracts_are_not_interchangeable() {
        let error = validate_gguf_contract(
            ExternalDraftKind::Dflash,
            &dflash2_gguf(),
            Path::new("draft.gguf"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("kind 'dflash2'"));

        let error = validate_gguf_contract(
            ExternalDraftKind::Dflash2,
            &gguf_with_tensors(&[]),
            Path::new("draft.gguf"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("DFlash2"));
    }

    #[test]
    fn contract_rejects_non_integer_target_layers() {
        let mut gguf =
            gguf_with_tensors(&["markov_w1.weight", "markov_w2.weight", "conf_proj.weight"]);
        gguf.metadata.insert(
            "dflash.target_layers".to_string(),
            GgufValue::Array(vec![GgufValue::Float32(8.0)]),
        );

        let error =
            validate_gguf_contract(ExternalDraftKind::Dspark, &gguf, Path::new("draft.gguf"))
                .unwrap_err();
        assert!(error.to_string().contains("dflash.target_layers"));
    }

    #[test]
    fn regular_file_identity_checks_size_and_digest() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("artifact.gguf");
        std::fs::write(&path, b"content-addressed").unwrap();
        let digest = super::super::storage::compute_sha256_file(&path).unwrap();

        assert_eq!(
            verify_regular_file_identity("Artifact", &path, 17, &digest).unwrap(),
            digest
        );
        assert!(verify_regular_file_identity("Artifact", &path, 16, &digest)
            .unwrap_err()
            .to_string()
            .contains("size mismatch"));
        assert!(
            verify_regular_file_identity("Artifact", &path, 17, &"0".repeat(64))
                .unwrap_err()
                .to_string()
                .contains("SHA-256 mismatch")
        );
    }
}
