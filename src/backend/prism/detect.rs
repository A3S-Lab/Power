//! Detect GGUF packs that require the PrismML llama.cpp runtime.

use std::path::Path;

use crate::model::manifest::{ModelFormat, ModelManifest};

/// Return true when `path` names a Prism-required Bonsai / Ternary pack.
///
/// Detection is filename-based (the contract Bonsai-demo uses). Prism-only
/// quant type IDs are not stable across forks for header probing yet, and
/// lookalike `Q2_0` packs must never be treated as ordinary GGUF.
pub fn is_prism_required_path(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    if !(lower.ends_with(".gguf")) {
        return false;
    }
    // Explicit Prism-only bands (safe + required).
    if name.contains("PTQ1_0") || name.contains("PQ2_0") {
        return true;
    }
    // Bonsai 2 language packs always need Prism (Hadamard activation transform).
    if name.contains("Ternary-Bonsai-2") || name.contains("Bonsai-2-") {
        return true;
    }
    // Published dangerous lookalike for upstreaming work — refuse on stock backends.
    if name.contains("prism-fork-required") {
        return true;
    }
    false
}

/// Manifest-level Prism requirement check.
pub fn is_prism_required_manifest(manifest: &ModelManifest) -> bool {
    if manifest.format != ModelFormat::Gguf {
        return false;
    }
    if is_prism_required_path(&manifest.path) {
        return true;
    }
    // Optional explicit family marker for callers that set it.
    if let Some(family) = manifest.family.as_deref() {
        let f = family.to_ascii_lowercase();
        if f == "bonsai2" || f == "prism" || f.starts_with("ternary-bonsai-2") {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::backend::test_utils::sample_manifest;

    #[test]
    fn path_detection_covers_shipped_bands() {
        assert!(is_prism_required_path(Path::new(
            "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
        )));
        assert!(is_prism_required_path(Path::new(
            "models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
        )));
        assert!(is_prism_required_path(Path::new(
            "Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf"
        )));
        assert!(!is_prism_required_path(Path::new("model-Q4_K_M.gguf")));
        assert!(!is_prism_required_path(Path::new("readme.txt")));
    }

    #[test]
    fn family_marker_selects_prism() {
        let mut manifest = sample_manifest("x");
        manifest.format = ModelFormat::Gguf;
        manifest.path = PathBuf::from("weights.bin.gguf");
        manifest.family = Some("bonsai2".into());
        assert!(is_prism_required_manifest(&manifest));
    }
}
