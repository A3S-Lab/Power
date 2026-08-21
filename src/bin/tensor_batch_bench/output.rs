use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use a3s_power::error::{PowerError, Result};
use a3s_power::inference::ReleaseEvidenceBundle;

use super::MAX_INPUT_DOCUMENT_BYTES;

pub(super) fn write_json_output(output: &serde_json::Value, path: Option<&Path>) -> Result<()> {
    let mut encoded = serde_json::to_vec_pretty(output)?;
    encoded.push(b'\n');
    let Some(path) = path else {
        std::io::stdout().write_all(&encoded)?;
        return Ok(());
    };
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            PowerError::InvalidRequest(format!(
                "failed to create new benchmark output '{}': {error}",
                path.display()
            ))
        })?;
    if let Err(error) = file.write_all(&encoded).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = std::fs::remove_file(path);
        return Err(error.into());
    }
    Ok(())
}

pub(super) fn write_release_bundle_outputs(
    bundle: &ReleaseEvidenceBundle,
    bundle_path: &Path,
    pin_path: &Path,
) -> Result<()> {
    let bundle_path = resolve_new_output_target(bundle_path)?;
    let pin_path = resolve_new_output_target(pin_path)?;
    if same_output_target(&bundle_path, &pin_path) {
        return Err(PowerError::InvalidRequest(
            "release bundle and digest pin must use different output paths".to_string(),
        ));
    }

    let mut bundle_bytes = serde_json::to_vec_pretty(bundle)?;
    bundle_bytes.push(b'\n');
    let bundle_len = u64::try_from(bundle_bytes.len()).map_err(|_| {
        PowerError::InvalidRequest(
            "release evidence bundle size cannot be represented by the verifier".to_string(),
        )
    })?;
    if bundle_len > MAX_INPUT_DOCUMENT_BYTES {
        return Err(PowerError::InvalidRequest(format!(
            "release evidence bundle contains {} bytes, exceeding the {MAX_INPUT_DOCUMENT_BYTES}-byte verifier limit",
            bundle_bytes.len()
        )));
    }
    let pin_bytes = format!("{}\n", bundle.sha256).into_bytes();

    let mut bundle_file = create_new(&bundle_path, "release evidence bundle")?;
    if let Err(error) = write_and_sync(
        &mut bundle_file,
        &bundle_bytes,
        &bundle_path,
        "release evidence bundle",
    ) {
        drop(bundle_file);
        let _ = std::fs::remove_file(&bundle_path);
        return Err(error);
    }
    drop(bundle_file);

    let mut pin_file = match create_new(&pin_path, "release evidence digest pin") {
        Ok(file) => file,
        Err(error) => {
            let _ = std::fs::remove_file(&bundle_path);
            return Err(error);
        }
    };
    if let Err(error) = write_and_sync(
        &mut pin_file,
        &pin_bytes,
        &pin_path,
        "release evidence digest pin",
    ) {
        drop(pin_file);
        let _ = std::fs::remove_file(&pin_path);
        let _ = std::fs::remove_file(&bundle_path);
        return Err(error);
    }
    Ok(())
}

fn resolve_new_output_target(path: &Path) -> Result<std::path::PathBuf> {
    let file_name = path.file_name().ok_or_else(|| {
        PowerError::InvalidRequest(format!(
            "release output '{}' must name a file",
            path.display()
        ))
    })?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let resolved_parent = std::fs::canonicalize(parent).map_err(|error| {
        PowerError::InvalidRequest(format!(
            "release output parent '{}' is unavailable: {error}",
            parent.display()
        ))
    })?;
    Ok(resolved_parent.join(file_name))
}

#[cfg(windows)]
fn same_output_target(left: &Path, right: &Path) -> bool {
    left.to_string_lossy()
        .eq_ignore_ascii_case(&right.to_string_lossy())
}

#[cfg(not(windows))]
fn same_output_target(left: &Path, right: &Path) -> bool {
    left == right
}

fn create_new(path: &Path, label: &str) -> Result<std::fs::File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            PowerError::InvalidRequest(format!(
                "failed to create new {label} '{}': {error}",
                path.display()
            ))
        })
}

fn write_and_sync(file: &mut std::fs::File, bytes: &[u8], path: &Path, label: &str) -> Result<()> {
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| {
            PowerError::InvalidRequest(format!(
                "failed to synchronize {label} '{}': {error}",
                path.display()
            ))
        })
}

#[cfg(test)]
mod tests {
    use a3s_power::inference::{
        ReleaseCapture, ReleaseEvidenceBundle, ReleaseEvidencePolicy, ReleaseRevisionBinding,
    };

    use super::*;

    fn partial_bundle() -> ReleaseEvidenceBundle {
        let capture: ReleaseCapture = serde_json::from_str(include_str!(
            "../../../docs/benchmarks/release-contract-windows-20260821/cpu.json"
        ))
        .unwrap();
        let revision = ReleaseRevisionBinding::from_capture(&capture).unwrap();
        let policy =
            ReleaseEvidencePolicy::new(revision, vec![capture.platform_binding().unwrap()])
                .unwrap();
        ReleaseEvidenceBundle::build(policy, vec![capture]).unwrap()
    }

    #[test]
    fn release_bundle_outputs_are_create_new_and_rollback_as_a_pair() {
        let directory = tempfile::tempdir().unwrap();
        let bundle_path = directory.path().join("release-evidence.json");
        let pin_path = directory.path().join("release-evidence.sha256");
        let bundle = partial_bundle();

        write_release_bundle_outputs(&bundle, &bundle_path, &pin_path).unwrap();
        let restored: ReleaseEvidenceBundle =
            serde_json::from_slice(&std::fs::read(&bundle_path).unwrap()).unwrap();
        assert_eq!(restored, bundle);
        assert_eq!(
            std::fs::read_to_string(&pin_path).unwrap(),
            format!("{}\n", bundle.sha256)
        );
        assert!(write_release_bundle_outputs(&bundle, &bundle_path, &pin_path).is_err());

        let rollback_bundle = directory.path().join("rollback.json");
        let occupied_pin = directory.path().join("occupied.sha256");
        std::fs::write(&occupied_pin, b"caller-owned").unwrap();
        assert!(write_release_bundle_outputs(&bundle, &rollback_bundle, &occupied_pin).is_err());
        assert!(!rollback_bundle.exists());
        assert_eq!(std::fs::read(&occupied_pin).unwrap(), b"caller-owned");
    }

    #[test]
    fn release_bundle_outputs_reject_one_aliased_target() {
        let directory = tempfile::tempdir().unwrap();
        let direct = directory.path().join("same-output");
        let aliased = directory.path().join(".").join("same-output");
        let error = write_release_bundle_outputs(&partial_bundle(), &direct, &aliased).unwrap_err();
        assert!(error.to_string().contains("different output paths"));
        assert!(!direct.exists());
    }
}
