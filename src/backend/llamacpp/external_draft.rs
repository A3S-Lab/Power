use std::sync::Arc;

use llama_cpp_2::model::LlamaModel;

use crate::error::{PowerError, Result};
use crate::model::external_draft::{
    ExternalDraftArtifact, ExternalDraftKind, VerifiedExternalDraft,
};
use crate::model::manifest::ModelManifest;
use crate::speculative::SpeculativeStrategy;

pub(super) fn dflash2_backend_unavailable() -> PowerError {
    PowerError::Config(
        "spec_mode 'dflash2' is not executable by Power's pinned llama.cpp binding; \
         use the exact-revision standalone benchmark runner until a reviewed binding update lands"
            .to_string(),
    )
}

/// Whether the configured runtime may execute an embedded MTP draft head.
///
/// llama.cpp skips MTP tensors at model-load time unless `load_mtp` is set.
/// `auto` loads them when no verified external draft was selected; loading two
/// draft mechanisms at once wastes device memory and can make both unusable.
pub(super) fn loads_mtp_weights(spec_mode: &str, external_draft_selected: bool) -> bool {
    matches!(
        SpeculativeStrategy::parse(spec_mode),
        Some(SpeculativeStrategy::Mtp)
    ) || (!external_draft_selected
        && matches!(
            SpeculativeStrategy::parse(spec_mode),
            Some(SpeculativeStrategy::Auto)
        ))
}

pub(super) fn selected_external_draft(
    manifest: &ModelManifest,
    spec_mode: &str,
) -> Result<Option<ExternalDraftArtifact>> {
    let requested = SpeculativeStrategy::parse(spec_mode)
        .ok_or_else(|| PowerError::Config(format!("unsupported spec_mode '{spec_mode}'")))?;
    if matches!(requested, SpeculativeStrategy::Dflash2) {
        return Err(dflash2_backend_unavailable());
    }
    let expected_kind = match requested {
        SpeculativeStrategy::Dflash => Some(ExternalDraftKind::Dflash),
        SpeculativeStrategy::Dflash2 => Some(ExternalDraftKind::Dflash2),
        SpeculativeStrategy::Dspark => Some(ExternalDraftKind::Dspark),
        _ => None,
    };
    if let Some(expected_kind) = expected_kind {
        #[cfg(not(feature = "llamacpp-external-draft"))]
        return Err(PowerError::Config(format!(
            "spec_mode '{}' requires the llamacpp-external-draft crate feature and reviewed llama-cpp-rs patch",
            expected_kind.as_str()
        )));

        #[cfg(feature = "llamacpp-external-draft")]
        {
            let artifact = manifest.external_draft.as_ref().ok_or_else(|| {
                PowerError::Config(format!(
                    "spec_mode '{}' requires model manifest.external_draft",
                    expected_kind.as_str()
                ))
            })?;
            if artifact.kind != expected_kind {
                return Err(PowerError::Config(format!(
                    "spec_mode '{}' requires an external_draft of kind '{}', found '{}'",
                    expected_kind.as_str(),
                    expected_kind.as_str(),
                    artifact.kind.as_str()
                )));
            }
            return Ok(Some(artifact.clone()));
        }
    }
    if matches!(requested, SpeculativeStrategy::Auto) {
        if manifest
            .external_draft
            .as_ref()
            .is_some_and(|artifact| artifact.kind == ExternalDraftKind::Dflash2)
        {
            return Err(dflash2_backend_unavailable());
        }
        #[cfg(feature = "llamacpp-external-draft")]
        return Ok(manifest.external_draft.clone());

        #[cfg(not(feature = "llamacpp-external-draft"))]
        return Ok(None);
    }
    Ok(None)
}

pub(super) fn external_draft_strategy(kind: ExternalDraftKind) -> SpeculativeStrategy {
    match kind {
        ExternalDraftKind::Dflash => SpeculativeStrategy::Dflash,
        ExternalDraftKind::Dflash2 => SpeculativeStrategy::Dflash2,
        ExternalDraftKind::Dspark => SpeculativeStrategy::Dspark,
    }
}

pub(super) struct LoadedExternalDraft {
    pub(super) model: Arc<LlamaModel>,
    pub(super) identity: VerifiedExternalDraft,
}
