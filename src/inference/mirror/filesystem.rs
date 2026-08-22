use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};
use crate::inference::filesystem::sync_directory;
use crate::weight_collection::portable_relative_path;

use super::{check_cancelled, WeightMirrorPlannedFile};

const COPY_BUFFER_BYTES: usize = 1024 * 1024;
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) fn resolve_destination(
    primary_roots: &[PathBuf],
    destination: &Path,
) -> Result<PathBuf> {
    if destination.as_os_str().is_empty() {
        return Err(PowerError::InvalidRequest(
            "partial weight mirror destination must not be empty".to_string(),
        ));
    }
    let resolved = match std::fs::symlink_metadata(destination) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(PowerError::InvalidRequest(
                    "partial weight mirror destination must be a real directory".to_string(),
                ));
            }
            std::fs::canonicalize(destination)?
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = destination
                .parent()
                .filter(|path| !path.as_os_str().is_empty());
            let parent = parent.unwrap_or_else(|| Path::new("."));
            let parent = std::fs::canonicalize(parent).map_err(|error| {
                PowerError::Io(std::io::Error::new(
                    error.kind(),
                    format!("failed to resolve partial weight mirror parent: {error}"),
                ))
            })?;
            if !std::fs::metadata(&parent)?.is_dir() {
                return Err(PowerError::InvalidRequest(
                    "partial weight mirror parent must be a directory".to_string(),
                ));
            }
            let name = destination.file_name().ok_or_else(|| {
                PowerError::InvalidRequest(
                    "partial weight mirror destination must name a directory".to_string(),
                )
            })?;
            parent.join(name)
        }
        Err(error) => return Err(error.into()),
    };
    if primary_roots.iter().any(|primary| {
        resolved == *primary || resolved.starts_with(primary) || primary.starts_with(&resolved)
    }) {
        return Err(PowerError::InvalidRequest(
            "partial weight mirror destination must be separate from every primary collection root"
                .to_string(),
        ));
    }
    Ok(resolved)
}

pub(super) fn inspect_destination(
    destination: &Path,
    files: &mut [WeightMirrorPlannedFile],
) -> Result<Vec<String>> {
    if !destination.exists() {
        return Ok(Vec::new());
    }
    let selected = files
        .iter()
        .map(|file| file.relative_path.as_str())
        .collect::<BTreeSet<_>>();
    let mut conflicts = Vec::new();
    let mut pending = vec![destination.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let mut entries = std::fs::read_dir(&directory)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                return Err(PowerError::InvalidRequest(
                    "partial weight mirror destination contains a symbolic link".to_string(),
                ));
            }
            if file_type.is_dir() {
                pending.push(entry.path());
                continue;
            }
            if !file_type.is_file()
                || entry.path().extension().and_then(|value| value.to_str()) != Some("safetensors")
            {
                continue;
            }
            let entry_path = entry.path();
            let relative = entry_path.strip_prefix(destination).map_err(|_| {
                PowerError::InvalidRequest(
                    "partial weight mirror file escaped its destination".to_string(),
                )
            })?;
            let relative = portable_relative_path(relative)?;
            if !selected.contains(relative.as_str()) {
                conflicts.push(relative);
            }
        }
    }
    for file in files {
        let target = destination.join(&file.relative_path);
        match std::fs::symlink_metadata(&target) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink()
                    || !metadata.is_file()
                    || metadata.len() != file.bytes
                    || hash_file(&target)? != file.sha256
                {
                    conflicts.push(file.relative_path.clone());
                } else {
                    file.reused = true;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    conflicts.sort();
    conflicts.dedup();
    Ok(conflicts)
}

fn hash_file(path: &Path) -> Result<String> {
    let mut input = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(super) fn ensure_target_parent(destination: &Path, target: &Path) -> Result<()> {
    let relative_parent = target
        .parent()
        .and_then(|parent| parent.strip_prefix(destination).ok())
        .ok_or_else(|| {
            PowerError::InvalidRequest(
                "partial weight mirror target escaped its destination".to_string(),
            )
        })?;
    let mut current = destination.to_path_buf();
    for component in relative_parent.components() {
        let Component::Normal(name) = component else {
            return Err(PowerError::InvalidRequest(
                "partial weight mirror target has a non-canonical parent".to_string(),
            ));
        };
        current.push(name);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(PowerError::InvalidRequest(
                        "partial weight mirror target parent is not a real directory".to_string(),
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                std::fs::create_dir(&current)?;
                if let Some(parent) = current.parent() {
                    sync_directory(parent)?;
                }
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

pub(super) fn copy_verified_no_replace(
    source: &Path,
    target: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    cancellation: &CancellationToken,
) -> Result<()> {
    if std::fs::symlink_metadata(target).is_ok() {
        return Err(PowerError::InvalidRequest(
            "partial weight mirror target appeared after planning".to_string(),
        ));
    }
    let parent = target.parent().ok_or_else(|| {
        PowerError::InvalidRequest("partial weight mirror target has no parent".to_string())
    })?;
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            PowerError::InvalidRequest(
                "partial weight mirror file names must be valid UTF-8".to_string(),
            )
        })?;
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{file_name}.a3s-power-{}-{sequence}.part",
        std::process::id()
    ));
    let result = (|| {
        let source_metadata = std::fs::symlink_metadata(source)?;
        if source_metadata.file_type().is_symlink()
            || !source_metadata.is_file()
            || source_metadata.len() != expected_bytes
        {
            return Err(PowerError::IntegrityCheckFailed {
                model: "partial weight mirror source".to_string(),
                expected: format!("{expected_bytes} bytes"),
                actual: format!("{} bytes", source_metadata.len()),
            });
        }
        let mut input = File::open(source)?;
        let mut output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        let mut digest = Sha256::new();
        let mut copied = 0_u64;
        let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
        loop {
            check_cancelled(cancellation)?;
            let read = input.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            output.write_all(&buffer[..read])?;
            digest.update(&buffer[..read]);
            copied = copied
                .checked_add(u64::try_from(read).map_err(|_| {
                    PowerError::InvalidFormat(
                        "partial weight mirror read length exceeds u64".to_string(),
                    )
                })?)
                .ok_or_else(|| {
                    PowerError::InvalidFormat(
                        "partial weight mirror copied byte count overflowed".to_string(),
                    )
                })?;
        }
        check_cancelled(cancellation)?;
        let actual_sha256 = format!("{:x}", digest.finalize());
        if copied != expected_bytes || actual_sha256 != expected_sha256 {
            return Err(PowerError::IntegrityCheckFailed {
                model: "partial weight mirror source".to_string(),
                expected: format!("{expected_bytes} bytes, sha256 {expected_sha256}"),
                actual: format!("{copied} bytes, sha256 {actual_sha256}"),
            });
        }
        output.sync_all()?;
        drop(output);
        std::fs::hard_link(&temporary, target).map_err(|error| {
            PowerError::Io(std::io::Error::new(
                error.kind(),
                format!("failed to publish partial weight mirror without replacement: {error}"),
            ))
        })?;
        std::fs::remove_file(&temporary)?;
        sync_directory(parent)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

#[cfg(unix)]
pub(super) fn available_space(path: &Path) -> Result<u64> {
    use std::ffi::CString;
    use std::mem::MaybeUninit;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        PowerError::InvalidRequest(
            "partial weight mirror destination contains an embedded NUL".to_string(),
        )
    })?;
    let mut statistics = MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path` is NUL terminated and `statistics` points to writable,
    // correctly aligned storage initialized by a successful `statvfs` call.
    if unsafe { libc::statvfs(path.as_ptr(), statistics.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: the preceding `statvfs` call succeeded and initialized the value.
    let statistics = unsafe { statistics.assume_init() };
    let bytes = u128::from(statistics.f_bavail).saturating_mul(u128::from(statistics.f_frsize));
    Ok(u64::try_from(bytes).unwrap_or(u64::MAX))
}

#[cfg(windows)]
pub(super) fn available_space(path: &Path) -> Result<u64> {
    use std::iter;
    use std::os::windows::ffi::OsStrExt;

    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(iter::once(0))
        .collect::<Vec<_>>();
    let mut available = 0_u64;
    // SAFETY: `wide` is NUL terminated and `available` is a valid writable
    // pointer. The unused output pointers are permitted to be null.
    if unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(available)
}
