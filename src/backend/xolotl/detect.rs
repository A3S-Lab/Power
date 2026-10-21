//! Claim the Xolotl XQ GGUF pack, and nothing that llama.cpp can already run.

use std::path::Path;

use crate::model::manifest::{ModelFormat, ModelManifest};

/// Published filename markers for the Xolotl DeepSeek-V4.1-Flash-1017 XQ pack.
///
/// `V41-GSQ` is the older pack name. Unsloth `UD-*` GGUF and official
/// safetensors are different runtimes and must not match.
const XQ_MARKERS: &[&str] = &["V41-XQ", "XQ-1017", "1017-XQ"];

/// Return true when `path` names an Xolotl XQ GGUF.
pub fn is_xolotl_xq_path(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if !name.to_ascii_lowercase().ends_with(".gguf") || is_unsloth_ud_name(name) {
        return false;
    }
    XQ_MARKERS.iter().any(|marker| name.contains(marker))
}

/// Return true when this manifest must be served by the Xolotl backend.
///
/// A `family = "xolotl"` marker claims the file even when the filename differs,
/// so llama.cpp cannot load an operator-marked pack. Load still refuses a
/// Unsloth UD filename.
pub fn is_xolotl_required_manifest(manifest: &ModelManifest) -> bool {
    if manifest.format != ModelFormat::Gguf || is_unsloth_ud_path(&manifest.path) {
        return false;
    }
    if is_xolotl_xq_path(&manifest.path) {
        return true;
    }
    matches!(
        manifest.family.as_deref().map(str::trim),
        Some(family) if family.eq_ignore_ascii_case("xolotl")
    )
}

fn is_unsloth_ud_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(is_unsloth_ud_name)
}

fn is_unsloth_ud_name(name: &str) -> bool {
    name.contains("-UD-") || name.contains("-UD_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::backend::test_utils::sample_manifest;

    #[test]
    fn path_detection_covers_the_1017_xq_pack_only() {
        assert!(is_xolotl_xq_path(Path::new(
            "DeepSeek-V4.1-Flash-1017-XQ-Uncensored.gguf"
        )));
        assert!(is_xolotl_xq_path(Path::new("models/V41-XQ.gguf")));
        assert!(is_xolotl_xq_path(Path::new("v2/XQ-1017.gguf")));
        assert!(!is_xolotl_xq_path(Path::new("V41-GSQ.gguf")));
        assert!(!is_xolotl_xq_path(Path::new(
            "DeepSeek-V4.1-Flash-1017-UD-IQ1_M.gguf"
        )));
        assert!(!is_xolotl_xq_path(Path::new(
            "DeepSeek-V4.1-Flash-1017-Q2.gguf"
        )));
        assert!(!is_xolotl_xq_path(Path::new(
            "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
        )));
        assert!(!is_xolotl_xq_path(Path::new("model.safetensors")));
    }

    #[test]
    fn family_marker_selects_xolotl_and_ud_packs_stay_out() {
        let mut manifest = sample_manifest("xq");
        manifest.format = ModelFormat::Gguf;
        manifest.path = PathBuf::from("weights.gguf");
        manifest.family = Some("xolotl".into());
        assert!(is_xolotl_required_manifest(&manifest));

        manifest.family = Some(" XOLOTL ".into());
        assert!(is_xolotl_required_manifest(&manifest));

        manifest.path = PathBuf::from("DeepSeek-V4.1-Flash-1017-UD-IQ1_S.gguf");
        assert!(!is_xolotl_required_manifest(&manifest));

        manifest.family = None;
        manifest.format = ModelFormat::SafeTensors;
        manifest.path = PathBuf::from("DeepSeek-V4.1-Flash-1017-XQ.safetensors");
        assert!(!is_xolotl_required_manifest(&manifest));
    }
}
