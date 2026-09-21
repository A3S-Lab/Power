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
    /// Multi-turn / prompt-cache friendly serving (no Prism DSpark).
    #[default]
    Baseline,
    /// Single-slot Prism DSpark speculative decode (requires a target-matched drafter).
    Dspark,
    /// Long-context KV compression on top of baseline flags.
    Kv4,
}

impl PrismProfile {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "baseline" | "default" => Some(Self::Baseline),
            "dspark" | "draft-dspark" | "prism-dspark" => Some(Self::Dspark),
            "kv4" | "kv-4" => Some(Self::Kv4),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Dspark => "dspark",
            Self::Kv4 => "kv4",
        }
    }

    /// Cross-request prompt-cache reuse is withdrawn for DSpark (Prism constraint).
    pub fn allows_cross_request_prompt_cache(self) -> bool {
        !matches!(self, Self::Dspark)
    }

    pub fn requires_drafter(self) -> bool {
        matches!(self, Self::Dspark)
    }
}

/// Resolve `prism_profile` from config text (default baseline).
pub fn resolve_profile(configured: Option<&str>) -> Result<PrismProfile> {
    let Some(raw) = configured.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(PrismProfile::Baseline);
    };
    PrismProfile::parse(raw).ok_or_else(|| {
        PowerError::Config(format!(
            "unsupported prism_profile '{raw}' (expected baseline|dspark|kv4). \
             Do not set Power spec_mode=dspark for Prism packs — that selects \
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
        assert_eq!(resolve_profile(Some("kv4")).unwrap(), PrismProfile::Kv4);
        assert!(resolve_profile(Some("mtp")).is_err());
    }

    #[test]
    fn dspark_withdraws_prompt_cache_claim() {
        assert!(PrismProfile::Baseline.allows_cross_request_prompt_cache());
        assert!(!PrismProfile::Dspark.allows_cross_request_prompt_cache());
    }
}
