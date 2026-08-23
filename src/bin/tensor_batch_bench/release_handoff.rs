use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use a3s_power::error::{PowerError, Result};
use a3s_power::inference::ReleasePlatform;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::output::encode_json_output;
use super::{read_bounded_regular, release_bundle::parse_platform, Arguments};

const HANDOFF_SCHEMA: &str = "a3s.power.release-handoff.v1";
const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const MAX_FILES: usize = 1_024;
const MAX_DIRECTORY_ENTRIES: usize = 4_096;
const MAX_DEPTH: usize = 16;
const MAX_PATH_BYTES: usize = 1_024;
const MAX_COMPONENT_BYTES: usize = 255;
const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 128 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleaseHandoffManifest {
    schema: String,
    platform: ReleasePlatform,
    power_version: String,
    power_commit: String,
    files: Vec<ReleaseHandoffFile>,
    total_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleaseHandoffFile {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReleaseHandoffBuild<'a> {
    schema: &'static str,
    built: bool,
    exact_root_inventory: bool,
    strict_v1_bundle_required: bool,
    external_authentication_required: bool,
    manifest_sha256: &'a str,
    platform: ReleasePlatform,
    power_version: &'a str,
    power_commit: &'a str,
    file_count: usize,
    total_bytes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReleaseHandoffVerification<'a> {
    schema: &'static str,
    verified: bool,
    exact_root_inventory: bool,
    strict_v1_bundle_required: bool,
    external_authentication_required: bool,
    manifest_sha256: &'a str,
    platform: ReleasePlatform,
    power_version: &'a str,
    power_commit: &'a str,
    file_count: usize,
    total_bytes: u64,
}

pub(super) struct BuiltReleaseHandoff {
    pub(super) manifest: serde_json::Value,
    pub(super) receipt: serde_json::Value,
}

pub(super) fn build(
    arguments: &mut Arguments,
    manifest_path: &Path,
) -> Result<BuiltReleaseHandoff> {
    let root = arguments.required_path("--root")?;
    let platform = parse_platform(&arguments.required("--platform")?)?;
    let power_version = arguments.required("--power-version")?;
    let power_commit = arguments.required("--power-commit")?;
    validate_source_binding(&power_version, &power_commit)?;
    arguments.ensure_empty()?;

    let root = resolve_root(&root)?;
    ensure_manifest_outside_root(&root, manifest_path, false)?;
    let files = scan_root(&root)?;
    let total_bytes = total_bytes(&files)?;
    let manifest = ReleaseHandoffManifest {
        schema: HANDOFF_SCHEMA.to_string(),
        platform,
        power_version,
        power_commit,
        files,
        total_bytes,
    };
    validate_manifest(&manifest)?;

    let manifest_value = serde_json::to_value(&manifest)?;
    let manifest_sha256 = format!("{:x}", Sha256::digest(encode_json_output(&manifest_value)?));
    let receipt = serde_json::to_value(ReleaseHandoffBuild {
        schema: "a3s.power.release-handoff-build.v1",
        built: true,
        exact_root_inventory: true,
        strict_v1_bundle_required: true,
        external_authentication_required: true,
        manifest_sha256: &manifest_sha256,
        platform,
        power_version: &manifest.power_version,
        power_commit: &manifest.power_commit,
        file_count: manifest.files.len(),
        total_bytes,
    })?;
    Ok(BuiltReleaseHandoff {
        manifest: manifest_value,
        receipt,
    })
}

pub(super) fn verify(arguments: &mut Arguments) -> Result<serde_json::Value> {
    let manifest_path = arguments.required_path("--manifest")?;
    let root = arguments.required_path("--root")?;
    let expected_platform = parse_platform(&arguments.required("--platform")?)?;
    let expected_power_version = arguments.required("--power-version")?;
    let expected_power_commit = arguments.required("--power-commit")?;
    validate_source_binding(&expected_power_version, &expected_power_commit)?;
    arguments.ensure_empty()?;

    let root = resolve_root(&root)?;
    ensure_manifest_outside_root(&root, &manifest_path, true)?;
    let encoded = read_bounded_regular(
        &manifest_path,
        MAX_MANIFEST_BYTES,
        "release handoff manifest",
    )?;
    let manifest: ReleaseHandoffManifest = serde_json::from_slice(&encoded)?;
    validate_manifest(&manifest)?;
    if manifest.platform != expected_platform
        || manifest.power_version != expected_power_version
        || manifest.power_commit != expected_power_commit
    {
        return Err(PowerError::PolicyViolation(
            "release handoff does not match the expected platform, version, and source revision"
                .to_string(),
        ));
    }

    let actual_files = scan_root(&root)?;
    let actual_total_bytes = total_bytes(&actual_files)?;
    if actual_files != manifest.files || actual_total_bytes != manifest.total_bytes {
        return Err(PowerError::PolicyViolation(
            "release handoff root inventory differs from the manifest".to_string(),
        ));
    }

    let manifest_sha256 = format!("{:x}", Sha256::digest(&encoded));
    serde_json::to_value(ReleaseHandoffVerification {
        schema: "a3s.power.release-handoff-verification.v1",
        verified: true,
        exact_root_inventory: true,
        strict_v1_bundle_required: true,
        external_authentication_required: true,
        manifest_sha256: &manifest_sha256,
        platform: manifest.platform,
        power_version: &manifest.power_version,
        power_commit: &manifest.power_commit,
        file_count: manifest.files.len(),
        total_bytes: manifest.total_bytes,
    })
    .map_err(PowerError::from)
}

fn validate_source_binding(power_version: &str, power_commit: &str) -> Result<()> {
    if power_version.is_empty()
        || power_version.len() > 64
        || !power_version
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'+'))
    {
        return Err(PowerError::InvalidRequest(
            "Power version must contain 1..=64 portable semantic-version characters".to_string(),
        ));
    }
    if power_commit.len() != 40
        || !power_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(PowerError::InvalidRequest(
            "Power commit must contain exactly 40 lowercase hexadecimal characters".to_string(),
        ));
    }
    Ok(())
}

fn validate_manifest(manifest: &ReleaseHandoffManifest) -> Result<()> {
    if manifest.schema != HANDOFF_SCHEMA {
        return Err(PowerError::InvalidFormat(
            "release handoff schema is unsupported".to_string(),
        ));
    }
    validate_source_binding(&manifest.power_version, &manifest.power_commit)?;
    if manifest.files.is_empty() || manifest.files.len() > MAX_FILES {
        return Err(PowerError::InvalidFormat(format!(
            "release handoff must contain 1..={MAX_FILES} files"
        )));
    }

    let mut previous: Option<&str> = None;
    let mut case_folded_paths = BTreeSet::new();
    let mut sum = 0_u64;
    for file in &manifest.files {
        validate_portable_path(&file.path)?;
        if previous.is_some_and(|path| path >= file.path.as_str()) {
            return Err(PowerError::InvalidFormat(
                "release handoff file paths must be unique and strictly sorted".to_string(),
            ));
        }
        previous = Some(&file.path);
        if !case_folded_paths.insert(file.path.to_ascii_lowercase()) {
            return Err(PowerError::InvalidFormat(
                "release handoff file paths must remain unique on case-insensitive filesystems"
                    .to_string(),
            ));
        }
        if file.bytes > MAX_FILE_BYTES {
            return Err(PowerError::InvalidFormat(format!(
                "release handoff file '{}' exceeds the {MAX_FILE_BYTES}-byte bound",
                file.path
            )));
        }
        if file.sha256.len() != 64
            || !file
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(PowerError::InvalidFormat(format!(
                "release handoff file '{}' has an invalid SHA-256 digest",
                file.path
            )));
        }
        sum = sum.checked_add(file.bytes).ok_or_else(|| {
            PowerError::InvalidFormat("release handoff total byte count overflowed".to_string())
        })?;
        if sum > MAX_TOTAL_BYTES {
            return Err(PowerError::InvalidFormat(format!(
                "release handoff exceeds the {MAX_TOTAL_BYTES}-byte aggregate bound"
            )));
        }
    }
    if sum != manifest.total_bytes {
        return Err(PowerError::InvalidFormat(
            "release handoff total byte count does not match its file entries".to_string(),
        ));
    }
    Ok(())
}

fn resolve_root(root: &Path) -> Result<PathBuf> {
    let metadata = std::fs::symlink_metadata(root).map_err(|error| {
        PowerError::InvalidRequest(format!(
            "release handoff root '{}' is unavailable: {error}",
            root.display()
        ))
    })?;
    if is_link_like(&metadata) || !metadata.is_dir() {
        return Err(PowerError::InvalidRequest(
            "release handoff root must be a regular non-symlink directory".to_string(),
        ));
    }
    std::fs::canonicalize(root).map_err(PowerError::from)
}

fn scan_root(root: &Path) -> Result<Vec<ReleaseHandoffFile>> {
    let mut files = BTreeMap::new();
    let mut directories = vec![(root.to_path_buf(), Vec::<String>::new(), 0_usize)];
    let mut entry_count = 0_usize;

    while let Some((directory, relative, depth)) = directories.pop() {
        if depth > MAX_DEPTH {
            return Err(PowerError::InvalidRequest(format!(
                "release handoff directory depth exceeds {MAX_DEPTH}"
            )));
        }
        let mut entries = std::fs::read_dir(&directory)?
            .map(|entry| {
                let entry = entry?;
                let name = entry.file_name().into_string().map_err(|_| {
                    PowerError::InvalidRequest(
                        "release handoff paths must contain valid UTF-8".to_string(),
                    )
                })?;
                validate_component(&name)?;
                Ok((name, entry.path()))
            })
            .collect::<Result<Vec<_>>>()?;
        entries.sort_by(|left, right| left.0.cmp(&right.0));

        for (name, path) in entries {
            entry_count = entry_count.checked_add(1).ok_or_else(|| {
                PowerError::InvalidRequest("release handoff entry count overflowed".to_string())
            })?;
            if entry_count > MAX_DIRECTORY_ENTRIES {
                return Err(PowerError::InvalidRequest(format!(
                    "release handoff contains more than {MAX_DIRECTORY_ENTRIES} directory entries"
                )));
            }
            let metadata = std::fs::symlink_metadata(&path)?;
            if is_link_like(&metadata) {
                return Err(PowerError::InvalidRequest(format!(
                    "release handoff symbolic links or reparse points are not allowed: '{}'",
                    path.display()
                )));
            }
            let mut child_relative = relative.clone();
            child_relative.push(name);
            if metadata.is_dir() {
                directories.push((path, child_relative, depth + 1));
                continue;
            }
            if !metadata.is_file() {
                return Err(PowerError::InvalidRequest(format!(
                    "release handoff contains a non-regular artifact: '{}'",
                    path.display()
                )));
            }
            if files.len() >= MAX_FILES {
                return Err(PowerError::InvalidRequest(format!(
                    "release handoff contains more than {MAX_FILES} files"
                )));
            }
            let portable_path = child_relative.join("/");
            validate_portable_path(&portable_path)?;
            let file = hash_file(root, &path, &portable_path, &metadata)?;
            if files.insert(portable_path, file).is_some() {
                return Err(PowerError::InvalidRequest(
                    "release handoff contains duplicate portable paths".to_string(),
                ));
            }
        }
    }

    if files.is_empty() {
        return Err(PowerError::InvalidRequest(
            "release handoff root must contain at least one regular file".to_string(),
        ));
    }
    Ok(files.into_values().collect())
}

fn hash_file(
    root: &Path,
    path: &Path,
    portable_path: &str,
    scanned_metadata: &std::fs::Metadata,
) -> Result<ReleaseHandoffFile> {
    if scanned_metadata.len() > MAX_FILE_BYTES {
        return Err(PowerError::InvalidRequest(format!(
            "release handoff file '{portable_path}' exceeds the {MAX_FILE_BYTES}-byte bound"
        )));
    }
    let resolved = std::fs::canonicalize(path)?;
    if !path_is_within(root, &resolved) {
        return Err(PowerError::InvalidRequest(format!(
            "release handoff file '{portable_path}' resolves outside the staged root"
        )));
    }

    let mut source = File::open(&resolved)?;
    let opened_metadata = source.metadata()?;
    if !opened_metadata.is_file() || opened_metadata.len() != scanned_metadata.len() {
        return Err(PowerError::InvalidRequest(format!(
            "release handoff file '{portable_path}' changed while its inventory was captured"
        )));
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes = 0_u64;
    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes = bytes.checked_add(read as u64).ok_or_else(|| {
            PowerError::InvalidRequest(format!(
                "release handoff file '{portable_path}' byte count overflowed"
            ))
        })?;
        if bytes > MAX_FILE_BYTES || bytes > opened_metadata.len() {
            return Err(PowerError::InvalidRequest(format!(
                "release handoff file '{portable_path}' changed outside its declared bound while being read"
            )));
        }
        hasher.update(&buffer[..read]);
    }
    let final_metadata = source.metadata()?;
    if bytes != opened_metadata.len() || final_metadata.len() != opened_metadata.len() {
        return Err(PowerError::InvalidRequest(format!(
            "release handoff file '{portable_path}' changed while being hashed"
        )));
    }
    Ok(ReleaseHandoffFile {
        path: portable_path.to_string(),
        bytes,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

fn total_bytes(files: &[ReleaseHandoffFile]) -> Result<u64> {
    files.iter().try_fold(0_u64, |total, file| {
        let total = total.checked_add(file.bytes).ok_or_else(|| {
            PowerError::InvalidRequest("release handoff total byte count overflowed".to_string())
        })?;
        if total > MAX_TOTAL_BYTES {
            return Err(PowerError::InvalidRequest(format!(
                "release handoff exceeds the {MAX_TOTAL_BYTES}-byte aggregate bound"
            )));
        }
        Ok(total)
    })
}

fn validate_portable_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.len() > MAX_PATH_BYTES
        || path.starts_with('/')
        || path.contains('\\')
    {
        return Err(PowerError::InvalidFormat(format!(
            "release handoff file '{path}' must use a bounded portable relative path"
        )));
    }
    let components = path.split('/').collect::<Vec<_>>();
    if components.is_empty()
        || components.len() > MAX_DEPTH + 1
        || components
            .iter()
            .any(|component| validate_component(component).is_err())
    {
        return Err(PowerError::InvalidFormat(format!(
            "release handoff file '{path}' must use a bounded portable relative path"
        )));
    }
    Ok(())
}

fn validate_component(component: &str) -> Result<()> {
    let portable_stem = component
        .split_once('.')
        .map_or(component, |(stem, _)| stem)
        .to_ascii_lowercase();
    let windows_reserved = matches!(portable_stem.as_str(), "con" | "prn" | "aux" | "nul")
        || portable_stem.strip_prefix("com").is_some_and(|ordinal| {
            matches!(ordinal, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
        })
        || portable_stem.strip_prefix("lpt").is_some_and(|ordinal| {
            matches!(ordinal, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
        });
    let portable = !component.is_empty()
        && component.len() <= MAX_COMPONENT_BYTES
        && !matches!(component, "." | "..")
        && !component.ends_with([' ', '.'])
        && !windows_reserved
        && !component
            .chars()
            .any(|character| character.is_control() || "<>:\"|?*/\\".contains(character));
    if portable {
        Ok(())
    } else {
        Err(PowerError::InvalidFormat(
            "release handoff paths must use portable UTF-8 components".to_string(),
        ))
    }
}

fn ensure_manifest_outside_root(root: &Path, manifest: &Path, existing: bool) -> Result<()> {
    let resolved = if existing {
        std::fs::canonicalize(manifest).map_err(|error| {
            PowerError::InvalidRequest(format!(
                "release handoff manifest '{}' is unavailable: {error}",
                manifest.display()
            ))
        })?
    } else {
        let name = manifest.file_name().ok_or_else(|| {
            PowerError::InvalidRequest("release handoff manifest must name a file".to_string())
        })?;
        let parent = manifest
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::canonicalize(parent)?.join(name)
    };
    if path_is_within(root, &resolved) {
        return Err(PowerError::InvalidRequest(
            "release handoff manifest must be stored outside the staged artifact root".to_string(),
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
fn path_is_within(root: &Path, candidate: &Path) -> bool {
    candidate.starts_with(root)
}

#[cfg(windows)]
fn path_is_within(root: &Path, candidate: &Path) -> bool {
    let root = root.to_string_lossy().replace('/', "\\");
    let candidate = candidate.to_string_lossy().replace('/', "\\");
    candidate.eq_ignore_ascii_case(&root)
        || candidate.get(root.len()..).is_some_and(|suffix| {
            candidate[..root.len()].eq_ignore_ascii_case(&root) && suffix.starts_with('\\')
        })
}

fn is_link_like(metadata: &std::fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_paths_reject_traversal_platform_separators_and_controls() {
        for path in ["artifact.json", "hardware/apple.txt", "nested/a-b_1.txt"] {
            validate_portable_path(path).unwrap();
        }
        for path in [
            "",
            "/absolute",
            "../outside",
            "nested/../outside",
            "windows\\path",
            "bad:name",
            "hardware/CON.txt",
            "hardware/com1",
            "trailing. ",
            "line\nbreak",
        ] {
            assert!(validate_portable_path(path).is_err(), "accepted {path:?}");
        }
    }

    #[test]
    fn source_binding_is_exact_and_bounded() {
        validate_source_binding("1.0.0-rc.1", &"a".repeat(40)).unwrap();
        assert!(validate_source_binding("1.0.0", &"A".repeat(40)).is_err());
        assert!(validate_source_binding("1.0.0", &"a".repeat(39)).is_err());
        assert!(validate_source_binding("version with spaces", &"a".repeat(40)).is_err());
    }
}
