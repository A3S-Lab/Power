use std::path::{Path, PathBuf};

use a3s_power::error::{PowerError, Result};
use a3s_power::inference::{InferenceLimits, WeightStore};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde::Serialize;

const FIXTURE_PREFIX: &str = "a3s-power-tensor-batch-fixture-";

pub(super) struct FixtureWeightDirectory {
    path: PathBuf,
    cleanup: bool,
    persistent: bool,
}

impl FixtureWeightDirectory {
    pub(super) fn temporary(width: usize) -> Result<Self> {
        let unique = format!(
            "{FIXTURE_PREFIX}{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|error| PowerError::InferenceFailed(error.to_string()))?
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        create_new(&path, width, false)
    }

    pub(super) fn open(path: &Path) -> Result<Self> {
        let path = std::fs::canonicalize(path).map_err(|error| {
            PowerError::InvalidRequest(format!(
                "failed to resolve fixture weight directory '{}': {error}",
                path.display()
            ))
        })?;
        if !std::fs::symlink_metadata(&path)?.is_dir() {
            return Err(PowerError::InvalidRequest(format!(
                "fixture weight path '{}' must be a directory",
                path.display()
            )));
        }
        Ok(Self {
            path,
            cleanup: false,
            persistent: true,
        })
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn is_persistent(&self) -> bool {
        self.persistent
    }

    pub(super) fn persist(mut self) {
        self.cleanup = false;
        self.persistent = true;
    }
}

impl Drop for FixtureWeightDirectory {
    fn drop(&mut self) {
        if !self.cleanup {
            return;
        }
        let Ok(resolved) = std::fs::canonicalize(&self.path) else {
            return;
        };
        if resolved == self.path {
            let _ = std::fs::remove_dir_all(&resolved);
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct FixtureWeightReceipt {
    schema: &'static str,
    width: usize,
    weights_sha256: String,
    file_sha256: String,
    file_bytes: u64,
}

pub(super) struct PendingFixtureWeights {
    directory: FixtureWeightDirectory,
    receipt: FixtureWeightReceipt,
}

impl PendingFixtureWeights {
    pub(super) fn receipt(&self) -> &FixtureWeightReceipt {
        &self.receipt
    }

    pub(super) fn persist(self) {
        self.directory.persist();
    }
}

pub(super) fn materialize(path: &Path, width: usize) -> Result<PendingFixtureWeights> {
    let directory = create_new(path, width, true)?;
    let store = WeightStore::open(directory.path(), &InferenceLimits::default())?;
    validate(&store, width)?;
    let file = store.files().first().ok_or_else(|| {
        PowerError::InvalidFormat("fixture weight collection contains no file".to_string())
    })?;
    Ok(PendingFixtureWeights {
        receipt: FixtureWeightReceipt {
            schema: "a3s.power.release-fixture-weights.v1",
            width,
            weights_sha256: store.sha256().to_string(),
            file_sha256: file.sha256.clone(),
            file_bytes: file.bytes,
        },
        directory,
    })
}

pub(super) fn validate(store: &WeightStore, width: usize) -> Result<()> {
    let inventory = store.inventory().collect::<Vec<_>>();
    let files = store.files();
    let expected_bytes = u64::try_from(width)
        .ok()
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| PowerError::InvalidRequest("fixture weight size overflowed".to_string()))?;
    if files.len() != 1
        || files[0].relative_path != "fixture.safetensors"
        || inventory.len() != 1
        || inventory[0].name != "bias"
        || inventory[0].dtype != "f32"
        || inventory[0].shape != [width]
        || inventory[0].bytes != expected_bytes
    {
        return Err(PowerError::InvalidFormat(
            "fixture weight collection must contain only fixture.safetensors with one F32 bias tensor of the requested width"
                .to_string(),
        ));
    }
    Ok(())
}

fn create_new(path: &Path, width: usize, persistent: bool) -> Result<FixtureWeightDirectory> {
    if width == 0 {
        return Err(PowerError::InvalidRequest(
            "fixture width must be positive".to_string(),
        ));
    }
    let file_name = path.file_name().ok_or_else(|| {
        PowerError::InvalidRequest(format!(
            "fixture weight output '{}' must name a new directory",
            path.display()
        ))
    })?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = std::fs::canonicalize(parent).map_err(|error| {
        PowerError::InvalidRequest(format!(
            "fixture weight output parent '{}' is unavailable: {error}",
            parent.display()
        ))
    })?;
    let path = parent.join(file_name);
    std::fs::create_dir(&path).map_err(|error| {
        PowerError::InvalidRequest(format!(
            "failed to create new fixture weight directory '{}': {error}",
            path.display()
        ))
    })?;
    let directory = FixtureWeightDirectory {
        path,
        cleanup: true,
        persistent,
    };
    write_weights(directory.path(), width)?;
    Ok(directory)
}

fn write_weights(path: &Path, width: usize) -> Result<()> {
    let bias = vec![0.25_f32; width]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::F32, vec![width], &bias).map_err(|error| {
        PowerError::InvalidFormat(format!("failed to build fixture tensor: {error}"))
    })?;
    serialize_to_file(
        vec![("bias", view)],
        None,
        &path.join("fixture.safetensors"),
    )
    .map_err(|error| {
        PowerError::InvalidFormat(format!("failed to serialize fixture weights: {error}"))
    })
}
