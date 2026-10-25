//! Measured model assembly for one explicit managed ACL binding. Source paths
//! and hashes are supplied by the admitted immutable model directory.
use crate::{
    config::{PowerConfig, XolotlModelPart},
    error::{PowerError, Result},
    model::manifest::{ModelFormat, ModelManifest, ModelParameters},
};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
const MAX_BYTES: u64 = 512 * 1024 * 1024 * 1024;
const BUFFER_BYTES: usize = 8 * 1024 * 1024;
fn refused(reason: impl Into<String>) -> PowerError {
    PowerError::Config(format!("managed Xolotl model: {}", reason.into()))
}
fn valid_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
struct Cancel(Arc<AtomicBool>);
impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
struct Partial(PathBuf);
impl Drop for Partial {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
struct Assembly {
    name: String,
    path: PathBuf,
    expected: String,
    size: u64,
    parts: Vec<XolotlModelPart>,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
}
impl Assembly {
    fn check(&self) -> Result<()> {
        if self.cancel.load(Ordering::Acquire) {
            return Err(refused("assembly was cancelled"));
        }
        if Instant::now() >= self.deadline {
            return Err(refused("assembly exceeded its deadline"));
        }
        Ok(())
    }
    fn verify(&self) -> Result<ModelManifest> {
        self.check()?;
        let mut file = regular_file(&self.path)?;
        if file.metadata()?.len() != self.size {
            return Err(refused(
                "joined GGUF size differs from its admitted members",
            ));
        }
        let mut buffer = vec![0; BUFFER_BYTES];
        let mut hash = Sha256::new();
        let mut done = 0_u64;
        let mut magic = Vec::with_capacity(4);
        loop {
            self.check()?;
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            done = done
                .checked_add(n as u64)
                .ok_or_else(|| refused("joined GGUF length overflowed"))?;
            if done > self.size {
                return Err(refused("joined GGUF grew beyond its admitted size"));
            }
            if magic.len() < 4 {
                magic.extend_from_slice(&buffer[..n.min(4 - magic.len())]);
            }
            hash.update(&buffer[..n]);
        }
        if done != self.size
            || magic != b"GGUF"
            || format!("{:x}", hash.finalize()) != self.expected
        {
            return Err(refused(
                "joined GGUF failed its exact size, magic, or SHA-256 binding",
            ));
        }
        self.manifest()
    }
    fn manifest(&self) -> Result<ModelManifest> {
        // The existing constructor supplies neutral metadata only; format, content,
        // size and path below are the measured local GGUF facts, never remote state.
        Ok(ModelManifest {
            format: ModelFormat::Gguf,
            size: self.size,
            sha256: self.expected.clone(),
            path: self.path.clone(),
            ..ModelManifest::remote(&self.name)
        })
    }
    fn prepare(&self) -> Result<ModelManifest> {
        self.check()?;
        match std::fs::symlink_metadata(&self.path) {
            Ok(_) => return self.verify(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        if self.parts.is_empty() {
            return Err(refused("registered joined GGUF is absent"));
        }
        let parent = self
            .path
            .parent()
            .ok_or_else(|| refused("joined GGUF has no parent directory"))?;
        std::fs::create_dir_all(parent)?;
        let partial =
            Partial(parent.join(format!(".power-model-{}.partial", uuid::Uuid::new_v4())));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        }
        let mut output = options.open(&partial.0)?;
        let mut buffer = vec![0; BUFFER_BYTES];
        let mut combined = Sha256::new();
        let mut magic = Vec::with_capacity(4);
        for part in &self.parts {
            self.check()?;
            let mut input = regular_file(&part.path)?;
            if input.metadata()?.len() != part.size_bytes {
                return Err(refused("source member size changed"));
            }
            let mut hash = Sha256::new();
            let mut left = part.size_bytes;
            while left > 0 {
                self.check()?;
                let size = buffer
                    .len()
                    .min(usize::try_from(left).unwrap_or(usize::MAX));
                let n = input.read(&mut buffer[..size])?;
                if n == 0 {
                    return Err(refused("source member ended before its admitted size"));
                }
                if magic.len() < 4 {
                    magic.extend_from_slice(&buffer[..n.min(4 - magic.len())]);
                }
                hash.update(&buffer[..n]);
                combined.update(&buffer[..n]);
                output.write_all(&buffer[..n])?;
                left -= n as u64;
            }
            if input.read(&mut [0; 1])? != 0 || format!("{:x}", hash.finalize()) != part.sha256 {
                return Err(refused(
                    "source member failed its exact size or SHA-256 binding",
                ));
            }
        }
        if magic != b"GGUF" || format!("{:x}", combined.finalize()) != self.expected {
            return Err(refused(
                "assembled GGUF failed its admitted magic or SHA-256 binding",
            ));
        }
        output.sync_all()?;
        drop(output);
        // Atomic no-clobber publication; a concurrent or pre-existing destination is
        // accepted only after a full independent content check.
        match std::fs::hard_link(&partial.0, &self.path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return self.verify()
            }
            Err(error) => return Err(error.into()),
        }
        File::open(parent)?.sync_all()?;
        self.manifest()
    }
}
fn regular_file(path: &Path) -> Result<File> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(refused(
            "model member or destination is not a regular non-symlink file",
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(refused("model member changed its file type"));
    }
    Ok(file)
}
fn verify_dspark_sidecar(
    dspark: crate::config::XolotlDSparkConfig,
    assembly: &Assembly,
) -> Result<()> {
    dspark.validate()?;
    let size = regular_file(&dspark.sidecar)?.metadata()?.len();
    if !(4..=16 * 1024 * 1024 * 1024).contains(&size) {
        return Err(refused("DSpark sidecar exceeds its 16 GiB bound"));
    }
    // Reuse the regular-file, cancellation, exact digest and GGUF-magic proof.
    // This value is never registered as a second served model.
    Assembly {
        name: "DSpark draft".into(),
        path: dspark.sidecar,
        expected: dspark.sidecar_sha256,
        size,
        parts: Vec::new(),
        deadline: assembly.deadline,
        cancel: assembly.cancel.clone(),
    }
    .verify()?;
    Ok(())
}
pub(crate) async fn prepare_managed_model(
    config: &PowerConfig,
    previous: Option<ModelManifest>,
) -> Result<ModelManifest> {
    if !cfg!(target_os = "linux") {
        return Err(refused("managed assembly requires Linux"));
    }
    let managed = config
        .xolotl
        .managed
        .as_ref()
        .ok_or_else(|| refused("managed ACL binding is absent"))?;
    if managed.model_name.is_empty()
        || managed.model_name.len() > 256
        || managed.model_name.contains(['\0', '\r', '\n'])
    {
        return Err(refused("model identity is invalid"));
    }
    if !valid_hash(&managed.model_sha256)
        || !(1..=3600).contains(&managed.startup_timeout_seconds)
        || managed.model_parts.len() > 64
    {
        return Err(refused(
            "digest, member count or deadline exceeds its bounds",
        ));
    }
    let path = PathBuf::from(
        config
            .xolotl
            .model
            .as_ref()
            .ok_or_else(|| refused("xolotl.model is unset"))?,
    );
    if !path.is_absolute() {
        return Err(refused("xolotl.model must be absolute"));
    }
    if let Some(previous) = &previous {
        if previous.path != path || previous.sha256 != managed.model_sha256 {
            return Err(refused(
                "registered model changed its path or content binding",
            ));
        }
    }
    let mut size = 0_u64;
    let mut paths = std::collections::BTreeSet::new();
    for part in &managed.model_parts {
        if !part.path.is_absolute()
            || part.size_bytes == 0
            || !valid_hash(&part.sha256)
            || !paths.insert(&part.path)
        {
            return Err(refused("source members are invalid or duplicated"));
        }
        size = size
            .checked_add(part.size_bytes)
            .filter(|n| *n <= MAX_BYTES)
            .ok_or_else(|| refused("source members exceed 512 GiB"))?;
    }
    if managed.model_parts.is_empty() {
        size = previous
            .as_ref()
            .ok_or_else(|| refused("no registered joined model or admitted source members"))?
            .size;
    }
    if size < 4 || size > MAX_BYTES {
        return Err(refused("joined GGUF length is outside its bounds"));
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let guard = Cancel(cancel.clone());
    let assembly = Assembly {
        name: managed.model_name.clone(),
        path,
        expected: managed.model_sha256.clone(),
        size,
        parts: managed.model_parts.clone(),
        deadline: Instant::now()
            .checked_add(Duration::from_secs(managed.startup_timeout_seconds))
            .ok_or_else(|| refused("assembly deadline overflowed"))?,
        cancel,
    };
    let dspark = config.xolotl.dspark.clone();
    let result = tokio::task::spawn_blocking(move || {
        if let Some(dspark) = dspark {
            verify_dspark_sidecar(dspark, &assembly)?;
        }
        assembly.prepare()
    })
    .await
    .map_err(|_| refused("assembly task failed"))?;
    drop(guard);
    let mut manifest = result?;
    if let Some(tuning) = &config.xolotl.tuning {
        tuning.validate()?;
        manifest.parameters = Some(ModelParameters {
            context_length: Some(tuning.context_tokens),
            embedding_length: None,
            parameter_count: None,
            quantization: None,
        });
    }
    Ok(manifest)
}
#[cfg(all(test, target_os = "linux"))]
#[path = "prepare_tests.rs"]
mod tests;
