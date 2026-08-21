use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use a3s_power::inference::{
    AcceleratorResidencyDeclaration, AcceleratorSecurityRequirement, ReleaseCapture,
    ReleaseCaptureSecurity, ReleasePlatform,
};
use a3s_power::verify::VerifiedConfidentialGpuAttestation;
use serde::de::DeserializeOwned;
use serde::Serialize;

const MAX_RELEASE_CAPTURE_JSON_BYTES: usize = 32 * 1024 * 1024;
const MAX_ACCELERATOR_DECLARATION_JSON_BYTES: usize = 4 * 1024 * 1024;

/// Inputs that are structurally and cryptographically consistent before the
/// hardware verifier is invoked. The opaque proof is accepted only by
/// `promote`, keeping proof issuance and consumption in one process.
pub(super) struct PreparedReleasePromotion {
    source_capture: ReleaseCapture,
    declaration: AcceleratorResidencyDeclaration,
    output_path: PathBuf,
}

impl PreparedReleasePromotion {
    pub(super) fn load(
        capture_path: impl AsRef<Path>,
        declaration_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
        expected_model_hash: &[u8],
        expected_execution_digest: &[u8],
    ) -> anyhow::Result<Self> {
        let source_capture: ReleaseCapture = read_bounded_json(
            capture_path.as_ref(),
            MAX_RELEASE_CAPTURE_JSON_BYTES,
            "release capture",
        )?;
        source_capture
            .verify()
            .map_err(|error| anyhow::anyhow!("release capture verification failed: {error}"))?;
        if source_capture.security != ReleaseCaptureSecurity::Local {
            anyhow::bail!("only a local release capture can be promoted");
        }
        if source_capture
            .platform()
            .map_err(|error| anyhow::anyhow!("release platform verification failed: {error}"))?
            != ReleasePlatform::Cuda
        {
            anyhow::bail!("confidential release promotion requires a local CUDA capture");
        }

        let declaration: AcceleratorResidencyDeclaration = read_bounded_json(
            declaration_path.as_ref(),
            MAX_ACCELERATOR_DECLARATION_JSON_BYTES,
            "accelerator declaration",
        )?;
        declaration.verify().map_err(|error| {
            anyhow::anyhow!("accelerator declaration verification failed: {error}")
        })?;
        if declaration.security != AcceleratorSecurityRequirement::ConfidentialGpu {
            anyhow::bail!(
                "confidential release promotion requires a confidential-GPU accelerator declaration"
            );
        }
        if declaration.runtime_device != source_capture.shape_binding.runtime_device {
            anyhow::bail!(
                "accelerator declaration runtime device does not match the release capture"
            );
        }
        if declaration.weights_sha256 != source_capture.shape_binding.weights_sha256 {
            anyhow::bail!("accelerator declaration weights do not match the release capture");
        }
        if declaration.execution_policy_sha256 != source_capture.shape_binding.tee_policy_sha256 {
            anyhow::bail!(
                "accelerator declaration execution policy does not match the release capture TEE policy"
            );
        }

        require_pinned_digest(
            expected_model_hash,
            &source_capture.shape_binding.weights_sha256,
            "model hash",
        )?;
        require_pinned_digest(
            expected_execution_digest,
            &declaration.execution_policy_sha256,
            "GPU execution digest",
        )?;

        let output_path = output_path.as_ref();
        if output_path.as_os_str().is_empty() {
            anyhow::bail!("promoted release output path must not be empty");
        }
        if output_path.exists() {
            anyhow::bail!(
                "promoted release output already exists: {}",
                output_path.display()
            );
        }

        Ok(Self {
            source_capture,
            declaration,
            output_path: output_path.to_path_buf(),
        })
    }

    pub(super) fn promote(
        self,
        proof: &VerifiedConfidentialGpuAttestation<'_>,
    ) -> anyhow::Result<String> {
        let promoted = self
            .source_capture
            .promote_confidential_gpu(proof, &self.declaration)
            .map_err(|error| anyhow::anyhow!("release promotion failed: {error}"))?;
        let sha256 = promoted.sha256.clone();
        write_json_create_new(
            &self.output_path,
            &promoted,
            MAX_RELEASE_CAPTURE_JSON_BYTES,
            "promoted release capture",
        )?;
        Ok(sha256)
    }
}

fn require_pinned_digest(actual: &[u8], expected_hex: &str, label: &str) -> anyhow::Result<()> {
    if actual.len() != 32 {
        anyhow::bail!("pinned {label} must contain exactly 32 bytes");
    }
    if hex::encode(actual) != expected_hex {
        anyhow::bail!("pinned {label} does not match the release inputs");
    }
    Ok(())
}

fn read_bounded_json<T: DeserializeOwned>(
    path: &Path,
    max_bytes: usize,
    label: &str,
) -> anyhow::Result<T> {
    let file = File::open(path)
        .map_err(|error| anyhow::anyhow!("failed to open {label} {}: {error}", path.display()))?;
    let metadata = file.metadata().map_err(|error| {
        anyhow::anyhow!("failed to inspect {label} {}: {error}", path.display())
    })?;
    if !metadata.is_file() {
        anyhow::bail!("{label} path is not a regular file: {}", path.display());
    }
    if metadata.len() == 0 {
        anyhow::bail!("{label} file is empty: {}", path.display());
    }
    if metadata.len() > max_bytes as u64 {
        anyhow::bail!(
            "{label} exceeds the {max_bytes}-byte input limit: {}",
            path.display()
        );
    }

    let read_limit = u64::try_from(max_bytes)
        .map_err(|_| anyhow::anyhow!("{label} input limit is unsupported on this platform"))?
        .saturating_add(1);
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| anyhow::anyhow!("failed to read {label} {}: {error}", path.display()))?;
    if bytes.len() > max_bytes {
        anyhow::bail!(
            "{label} grew beyond the {max_bytes}-byte input limit while being read: {}",
            path.display()
        );
    }
    if bytes.iter().all(u8::is_ascii_whitespace) {
        anyhow::bail!("{label} file is empty: {}", path.display());
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| anyhow::anyhow!("failed to parse {label} JSON: {error}"))
}

fn write_json_create_new<T: Serialize>(
    path: &Path,
    value: &T,
    max_bytes: usize,
    label: &str,
) -> anyhow::Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| anyhow::anyhow!("failed to serialize {label}: {error}"))?;
    bytes.push(b'\n');
    if bytes.len() > max_bytes {
        anyhow::bail!("serialized {label} exceeds the {max_bytes}-byte output limit");
    }

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::Builder::new()
        .prefix(".a3s-power-release-")
        .tempfile_in(parent)
        .map_err(|error| {
            anyhow::anyhow!(
                "failed to create temporary {label} beside {}: {error}",
                path.display()
            )
        })?;
    temporary.write_all(&bytes).map_err(|error| {
        anyhow::anyhow!(
            "failed to write temporary {label} for {}: {error}",
            path.display()
        )
    })?;
    temporary.as_file().sync_all().map_err(|error| {
        anyhow::anyhow!(
            "failed to sync temporary {label} for {}: {error}",
            path.display()
        )
    })?;
    temporary.persist_noclobber(path).map_err(|error| {
        anyhow::anyhow!(
            "failed to create new {label} {}: {}",
            path.display(),
            error.error
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;
    use tempfile::tempdir;

    use super::{read_bounded_json, write_json_create_new};

    #[test]
    fn bounded_reader_rejects_empty_and_oversized_inputs() {
        let directory = tempdir().unwrap();
        let empty = directory.path().join("empty.json");
        fs::write(&empty, b" \n\t").unwrap();
        let empty_error =
            read_bounded_json::<serde_json::Value>(&empty, 16, "fixture").unwrap_err();
        assert!(empty_error.to_string().contains("empty"));

        let oversized = directory.path().join("oversized.json");
        fs::write(&oversized, br#"{"value":"too-large"}"#).unwrap();
        let oversized_error =
            read_bounded_json::<serde_json::Value>(&oversized, 8, "fixture").unwrap_err();
        assert!(oversized_error.to_string().contains("input limit"));
    }

    #[test]
    fn create_new_writer_never_replaces_an_existing_artifact() {
        let directory = tempdir().unwrap();
        let output = directory.path().join("capture.json");
        write_json_create_new(&output, &json!({ "generation": 1 }), 1_024, "fixture").unwrap();
        let original = fs::read(&output).unwrap();

        let error = write_json_create_new(&output, &json!({ "generation": 2 }), 1_024, "fixture")
            .unwrap_err();
        assert!(error.to_string().contains("failed to create new"));
        assert_eq!(fs::read(output).unwrap(), original);
    }
}
