//! Prism runtime acceleration profiles (backend-owned, not Power `spec_mode`).

use crate::error::{PowerError, Result};

/// Named Prism upstream acceleration profile.
///
/// Distinct from Power [`crate::speculative::SpeculativeStrategy`]: Power
/// `spec_mode = dspark` means the pinned llamacpp adapter; `prism_profile =
/// dspark` means a Prism `llama-server` started with `--spec-type draft-dspark`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrismProfile {
    /// Multi-turn / prompt-cache friendly serving (no Prism speculation).
    #[default]
    Baseline,
    /// Single-slot Prism DSpark speculative decode (requires a target-matched drafter).
    ///
    /// Official `*dspark-dflash*` sidecars exist for Ternary-Bonsai **1** 27B.
    /// Bonsai **2** has no official DSpark pin yet (setup.ps1 / HF tree); keep
    /// fail-closed until one is provisioned.
    Dspark,
    /// Single-slot in-file / grafted MTP (`--spec-type draft-mtp`).
    ///
    /// First-principles Bonsai-2 acceleration while official DSpark is absent:
    /// community MTP grafts (e.g. PTQ1_0-mtp) on Prism binaries that accept the
    /// Hadamard-aware MTP graph.
    Mtp,
    /// Community DFlash2 sidecar (`--spec-type draft-dflash`) for Bonsai-2.
    ///
    /// Requires a target-matched draft GGUF via `prism_drafter`. Stock Prism
    /// may refuse without the companion embedding/runtime patches — fail open
    /// at Power admit if the file exists; upstream load errors remain upstream.
    Dflash,
    /// Long-context KV compression on top of baseline flags.
    Kv4,
}

impl PrismProfile {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "baseline" | "default" => Some(Self::Baseline),
            "dspark" | "draft-dspark" | "prism-dspark" => Some(Self::Dspark),
            "mtp" | "draft-mtp" | "prism-mtp" => Some(Self::Mtp),
            "dflash" | "draft-dflash" | "dflash2" | "prism-dflash" => Some(Self::Dflash),
            "kv4" | "kv-4" => Some(Self::Kv4),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Dspark => "dspark",
            Self::Mtp => "mtp",
            Self::Dflash => "dflash",
            Self::Kv4 => "kv4",
        }
    }

    /// Cross-request prompt-cache reuse is withdrawn for single-slot speculation.
    pub fn allows_cross_request_prompt_cache(self) -> bool {
        !matches!(self, Self::Dspark | Self::Mtp | Self::Dflash)
    }

    /// External `-md` drafter GGUF required.
    pub fn requires_drafter(self) -> bool {
        matches!(self, Self::Dspark | Self::Dflash)
    }

    /// Upstream must report `draft_n` when this profile is selected.
    pub fn expects_speculation_timings(self) -> bool {
        matches!(self, Self::Dspark | Self::Mtp | Self::Dflash)
    }

    pub fn upstream_source_label(self) -> &'static str {
        match self {
            Self::Baseline => "prism-baseline-upstream",
            Self::Dspark => "prism-dspark-upstream",
            Self::Mtp => "prism-mtp-upstream",
            Self::Dflash => "prism-dflash-upstream",
            Self::Kv4 => "prism-kv4-upstream",
        }
    }
}

/// Resolve `prism_profile` from config text (default baseline).
pub fn resolve_profile(configured: Option<&str>) -> Result<PrismProfile> {
    let Some(raw) = configured.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(PrismProfile::Baseline);
    };
    PrismProfile::parse(raw).ok_or_else(|| {
        PowerError::Config(format!(
            "unsupported prism_profile '{raw}' (expected baseline|dspark|mtp|dflash|kv4). \
             Do not set Power spec_mode=dspark|mtp|dflash for Prism packs — that selects \
             the pinned llamacpp adapter, not Prism upstream acceleration."
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aliases_and_defaults() {
        assert_eq!(resolve_profile(None).unwrap(), PrismProfile::Baseline);
        assert_eq!(
            resolve_profile(Some("dspark")).unwrap(),
            PrismProfile::Dspark
        );
        assert_eq!(resolve_profile(Some("mtp")).unwrap(), PrismProfile::Mtp);
        assert_eq!(resolve_profile(Some("dflash")).unwrap(), PrismProfile::Dflash);
        assert_eq!(resolve_profile(Some("kv4")).unwrap(), PrismProfile::Kv4);
        assert!(resolve_profile(Some("eagle")).is_err());
    }

    #[test]
    fn speculation_profiles_withdraw_prompt_cache() {
        assert!(PrismProfile::Baseline.allows_cross_request_prompt_cache());
        assert!(!PrismProfile::Dspark.allows_cross_request_prompt_cache());
        assert!(!PrismProfile::Mtp.allows_cross_request_prompt_cache());
        assert!(!PrismProfile::Dflash.allows_cross_request_prompt_cache());
        assert!(PrismProfile::Dspark.requires_drafter());
        assert!(PrismProfile::Dflash.requires_drafter());
        assert!(!PrismProfile::Mtp.requires_drafter());
        assert!(PrismProfile::Mtp.expects_speculation_timings());
        assert!(PrismProfile::Dflash.expects_speculation_timings());
    }
}
