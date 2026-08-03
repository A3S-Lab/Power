use std::cmp::Reverse;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use tokio_util::sync::CancellationToken;
use zeroize::Zeroizing;

use crate::error::{PowerError, Result};
use crate::inference::filesystem::sync_directory;
use crate::inference::InferenceLimits;

use super::envelope::{maximum_encoded_bytes, SEALED_STATE_HEADER_BYTES};
use super::types::{
    RecoveredSealedState, SealedStateBinding, SealedStateKey, SealedStateRecoverySource,
    SealedStateRollbackPolicy, SealedStateScope,
};
use super::{check_cancelled, SealedStateEnvelope};

/// Caller-invoked, single-writer store for one sealed model-state stream.
///
/// Publication uses a same-directory pending file, a synchronized previous
/// generation, and a synchronized primary rename. Recovery authenticates both
/// primary and backup and selects the highest generation allowed by the
/// caller's rollback floor. Pending files are never treated as committed.
pub struct SealedStateStore {
    target: PathBuf,
    backup: PathBuf,
    pending: PathBuf,
    writer: Mutex<()>,
}

impl SealedStateStore {
    pub fn new(target: impl AsRef<Path>) -> Result<Self> {
        let target = resolve_target(target.as_ref())?;
        validate_regular_or_absent(&target, "sealed state target")?;
        let file_name = target
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "sealed state target name must be valid UTF-8".to_string(),
                )
            })?;
        let parent = target.parent().ok_or_else(|| {
            PowerError::InvalidRequest("sealed state target must have a parent".to_string())
        })?;
        let backup = parent.join(format!(".{file_name}.a3s-power-sealed-state.backup"));
        let pending = parent.join(format!(".{file_name}.a3s-power-sealed-state.pending"));
        Ok(Self {
            target,
            backup,
            pending,
            writer: Mutex::new(()),
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Authenticates and recovers the highest committed generation permitted
    /// by the caller's rollback floor.
    ///
    /// This method performs blocking filesystem and cryptographic work. Async
    /// callers must invoke it from their existing bounded blocking path.
    pub fn load(
        &self,
        binding: &SealedStateBinding,
        key: &SealedStateKey,
        scope: SealedStateScope<'_>,
        rollback: SealedStateRollbackPolicy,
        limits: &InferenceLimits,
        cancellation: &CancellationToken,
    ) -> Result<Option<RecoveredSealedState>> {
        let _writer = self.lock();
        self.load_locked(binding, key, scope, rollback, limits, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    /// Authenticates and monotonically publishes one sealed generation.
    ///
    /// This method performs blocking filesystem and cryptographic work. Async
    /// callers must invoke it from their existing bounded blocking path.
    pub fn commit(
        &self,
        envelope: &SealedStateEnvelope,
        binding: &SealedStateBinding,
        key: &SealedStateKey,
        scope: SealedStateScope<'_>,
        rollback: SealedStateRollbackPolicy,
        limits: &InferenceLimits,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        let _writer = self.lock();
        check_cancelled(cancellation)?;
        let opened = envelope.open(binding, key, scope, rollback, limits, cancellation)?;
        let next_generation = opened.generation();
        drop(opened);

        let current = self.load_locked(
            binding,
            key,
            scope,
            SealedStateRollbackPolicy::new(0),
            limits,
            cancellation,
        )?;
        if let Some(current) = current {
            if next_generation <= current.generation() {
                return Err(PowerError::PolicyViolation(format!(
                    "sealed state generation {next_generation} must be greater than the committed generation {}",
                    current.generation()
                )));
            }
            let source = current.source();
            drop(current);
            if source == SealedStateRecoverySource::Backup {
                remove_regular_if_present(&self.target, "sealed state primary")?;
                std::fs::rename(&self.backup, &self.target).map_err(|error| {
                    contextual_io(error, "failed to restore committed sealed state backup")
                })?;
                sync_directory(self.parent())?;
            }
        }

        remove_regular_if_present(&self.pending, "sealed state pending file")?;
        let write_result = self.write_pending(envelope, cancellation);
        if let Err(error) = write_result {
            let _ = remove_regular_if_present(&self.pending, "sealed state pending file");
            return Err(error);
        }

        check_cancelled(cancellation).inspect_err(|_| {
            let _ = remove_regular_if_present(&self.pending, "sealed state pending file");
        })?;

        if let Err(error) = remove_regular_if_present(&self.backup, "sealed state backup") {
            let _ = remove_regular_if_present(&self.pending, "sealed state pending file");
            return Err(error);
        }
        sync_directory(self.parent())?;

        let had_primary = regular_exists(&self.target, "sealed state primary")?;
        if had_primary {
            if let Err(error) = std::fs::rename(&self.target, &self.backup) {
                let _ = remove_regular_if_present(&self.pending, "sealed state pending file");
                return Err(contextual_io(
                    error,
                    "failed to preserve the prior sealed state generation",
                ));
            }
            if let Err(error) = sync_directory(self.parent()) {
                let _ = restore_backup(&self.backup, &self.target, self.parent());
                let _ = remove_regular_if_present(&self.pending, "sealed state pending file");
                return Err(error);
            }
        }

        if let Err(error) = std::fs::rename(&self.pending, &self.target) {
            if had_primary {
                let _ = restore_backup(&self.backup, &self.target, self.parent());
            }
            let _ = remove_regular_if_present(&self.pending, "sealed state pending file");
            return Err(contextual_io(
                error,
                "failed to publish the sealed state generation",
            ));
        }
        sync_directory(self.parent())?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn load_locked(
        &self,
        binding: &SealedStateBinding,
        key: &SealedStateKey,
        scope: SealedStateScope<'_>,
        rollback: SealedStateRollbackPolicy,
        limits: &InferenceLimits,
        cancellation: &CancellationToken,
    ) -> Result<Option<RecoveredSealedState>> {
        check_cancelled(cancellation)?;
        let mut inspections = Vec::with_capacity(2);
        let mut last_error = None;
        for (path, source) in [
            (&self.target, SealedStateRecoverySource::Primary),
            (&self.backup, SealedStateRecoverySource::Backup),
        ] {
            match inspect_candidate(path, source, limits) {
                Ok(Some(inspection)) => inspections.push(inspection),
                Ok(None) => {}
                Err(error) => last_error = Some(error),
            }
            check_cancelled(cancellation)?;
        }
        inspections.sort_by_key(|inspection| {
            Reverse((
                inspection.generation,
                inspection.source == SealedStateRecoverySource::Primary,
            ))
        });

        for inspection in inspections {
            let path = match inspection.source {
                SealedStateRecoverySource::Primary => &self.target,
                SealedStateRecoverySource::Backup => &self.backup,
            };
            match read_candidate(
                path,
                inspection.source,
                binding,
                key,
                scope,
                rollback,
                limits,
                cancellation,
            ) {
                Ok(Some(candidate)) if candidate.generation() == inspection.generation => {
                    return Ok(Some(candidate));
                }
                Ok(Some(_)) => {
                    last_error = Some(PowerError::InvalidFormat(
                        "sealed state candidate changed generation during recovery".to_string(),
                    ));
                }
                Ok(None) => {}
                Err(error) => last_error = Some(error),
            }
            check_cancelled(cancellation)?;
        }
        match last_error {
            Some(error) => Err(error),
            None => Ok(None),
        }
    }

    fn write_pending(
        &self,
        envelope: &SealedStateEnvelope,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        check_cancelled(cancellation)?;
        let mut output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&self.pending)
            .map_err(|error| contextual_io(error, "failed to create sealed state pending file"))?;
        output
            .write_all(envelope.encoded())
            .map_err(|error| contextual_io(error, "failed to write sealed state pending file"))?;
        output.sync_all().map_err(|error| {
            contextual_io(error, "failed to synchronize sealed state pending file")
        })?;
        check_cancelled(cancellation)
    }

    fn parent(&self) -> &Path {
        // `resolve_target` always constructs a file below a canonical parent.
        self.target.parent().unwrap_or_else(|| Path::new("."))
    }

    fn lock(&self) -> MutexGuard<'_, ()> {
        self.writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl std::fmt::Debug for SealedStateStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SealedStateStore")
            .field("target", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

#[allow(clippy::too_many_arguments)]
fn read_candidate(
    path: &Path,
    source: SealedStateRecoverySource,
    binding: &SealedStateBinding,
    key: &SealedStateKey,
    scope: SealedStateScope<'_>,
    rollback: SealedStateRollbackPolicy,
    limits: &InferenceLimits,
    cancellation: &CancellationToken,
) -> Result<Option<RecoveredSealedState>> {
    check_cancelled(cancellation)?;
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PowerError::PolicyViolation(
            "sealed state candidate must be a regular file, not a link or directory".to_string(),
        ));
    }
    let maximum = maximum_encoded_bytes(limits)?;
    if metadata.len() > maximum {
        return Err(PowerError::InvalidRequest(format!(
            "sealed state candidate exceeds the {maximum} byte bound"
        )));
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| {
        PowerError::InvalidRequest("sealed state candidate cannot fit address space".to_string())
    })?;
    let mut encoded = Zeroizing::new(Vec::with_capacity(capacity));
    let input = File::open(path)?;
    input
        .take(maximum.saturating_add(1))
        .read_to_end(&mut encoded)?;
    if u64::try_from(encoded.len()).map_or(true, |bytes| bytes > maximum) {
        return Err(PowerError::InvalidRequest(
            "sealed state candidate grew beyond its configured bound while reading".to_string(),
        ));
    }
    check_cancelled(cancellation)?;
    let envelope = SealedStateEnvelope::import_owned(encoded, limits)?;
    let state = envelope.open(binding, key, scope, rollback, limits, cancellation)?;
    Ok(Some(RecoveredSealedState { source, state }))
}

#[derive(Debug, Clone, Copy)]
struct CandidateInspection {
    source: SealedStateRecoverySource,
    generation: u64,
}

fn inspect_candidate(
    path: &Path,
    source: SealedStateRecoverySource,
    limits: &InferenceLimits,
) -> Result<Option<CandidateInspection>> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PowerError::PolicyViolation(
            "sealed state candidate must be a regular file, not a link or directory".to_string(),
        ));
    }
    let maximum = maximum_encoded_bytes(limits)?;
    if metadata.len() > maximum {
        return Err(PowerError::InvalidRequest(format!(
            "sealed state candidate exceeds the {maximum} byte bound"
        )));
    }
    let mut header = [0_u8; SEALED_STATE_HEADER_BYTES];
    File::open(path)?.read_exact(&mut header)?;
    let generation = SealedStateEnvelope::inspect_generation(&header, metadata.len(), limits)?;
    Ok(Some(CandidateInspection { source, generation }))
}

fn resolve_target(target: &Path) -> Result<PathBuf> {
    if target.as_os_str().is_empty() {
        return Err(PowerError::InvalidRequest(
            "sealed state target must not be empty".to_string(),
        ));
    }
    let file_name = target.file_name().ok_or_else(|| {
        PowerError::InvalidRequest("sealed state target must name a file".to_string())
    })?;
    let file_name_text = file_name.to_str().ok_or_else(|| {
        PowerError::InvalidRequest("sealed state target name must be valid UTF-8".to_string())
    })?;
    if file_name_text.is_empty() || file_name_text.chars().any(char::is_control) {
        return Err(PowerError::InvalidRequest(
            "sealed state target name is empty or contains control characters".to_string(),
        ));
    }
    let parent = target
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let metadata = std::fs::symlink_metadata(parent)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(PowerError::PolicyViolation(
            "sealed state parent must be a real directory".to_string(),
        ));
    }
    Ok(std::fs::canonicalize(parent)?.join(file_name))
}

fn validate_regular_or_absent(path: &Path, label: &str) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => Err(
            PowerError::PolicyViolation(format!("{label} must be a regular file")),
        ),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn regular_exists(path: &Path, label: &str) -> Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => Err(
            PowerError::PolicyViolation(format!("{label} must be a regular file")),
        ),
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn remove_regular_if_present(path: &Path, label: &str) -> Result<()> {
    if regular_exists(path, label)? {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

fn restore_backup(backup: &Path, target: &Path, parent: &Path) -> Result<()> {
    if regular_exists(target, "sealed state primary")?
        || !regular_exists(backup, "sealed state backup")?
    {
        return Ok(());
    }
    std::fs::rename(backup, target)?;
    sync_directory(parent)
}

fn contextual_io(error: std::io::Error, context: &str) -> PowerError {
    PowerError::Io(std::io::Error::new(
        error.kind(),
        format!("{context}: {error}"),
    ))
}
