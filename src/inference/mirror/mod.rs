use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};
use crate::inference::filesystem::sync_directory;

use super::WeightStore;

mod filesystem;

use filesystem::{
    available_space, copy_verified_no_replace, ensure_target_parent, inspect_destination,
    resolve_destination,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightMirrorCandidate {
    pub relative_path: String,
    pub benefit: u64,
}

impl WeightMirrorCandidate {
    pub fn new(relative_path: impl Into<String>, benefit: u64) -> Self {
        Self {
            relative_path: relative_path.into(),
            benefit,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WeightMirrorConfidentiality {
    #[default]
    DenyPlaintext,
    CallerManagedPlaintext,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightMirrorPolicy {
    pub max_bytes: u64,
    pub reserve_bytes: u64,
    #[serde(default)]
    pub confidentiality: WeightMirrorConfidentiality,
}

impl WeightMirrorPolicy {
    pub fn new(max_bytes: u64, reserve_bytes: u64) -> Result<Self> {
        if max_bytes == 0 {
            return Err(PowerError::Config(
                "partial weight mirror budget must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            max_bytes,
            reserve_bytes,
            confidentiality: WeightMirrorConfidentiality::DenyPlaintext,
        })
    }

    pub fn with_confidentiality(mut self, confidentiality: WeightMirrorConfidentiality) -> Self {
        self.confidentiality = confidentiality;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WeightMirrorPlanRejection {
    PlaintextStagingDenied,
    NoCandidateFits,
    InsufficientSpace,
    DestinationConflict,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightMirrorPlannedFile {
    pub relative_path: String,
    pub bytes: u64,
    pub sha256: String,
    pub benefit: u64,
    pub reused: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightMirrorPlan {
    pub schema: String,
    pub model_sha256: String,
    pub max_bytes: u64,
    pub reserve_bytes: u64,
    pub available_bytes: u64,
    pub selected_bytes: u64,
    pub reused_bytes: u64,
    pub copy_bytes: u64,
    pub files: Vec<WeightMirrorPlannedFile>,
    pub conflicts: Vec<String>,
    pub admitted: bool,
    pub rejection: Option<WeightMirrorPlanRejection>,
}

impl WeightMirrorPlan {
    pub const SCHEMA: &'static str = "a3s.power.weight-mirror-plan.v1";
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightMirrorStageReport {
    pub schema: String,
    pub plan: WeightMirrorPlan,
    pub copied_files: usize,
    pub copied_bytes: u64,
    pub reused_files: usize,
    pub reused_bytes: u64,
}

impl WeightMirrorStageReport {
    pub const SCHEMA: &'static str = "a3s.power.weight-mirror-stage.v1";
}

impl WeightStore {
    /// Plans an explicitly requested, usage-ranked partial mirror.
    ///
    /// Candidate benefits are model-owned opaque positive scores. Power uses
    /// them only to select complete, already verified SafeTensors files by
    /// benefit density. This method reads existing destination files to verify
    /// resumable copies and performs native filesystem-capacity discovery, so
    /// callers in async code should execute it on a blocking worker.
    pub fn plan_partial_mirror(
        &self,
        destination: impl AsRef<Path>,
        candidates: &[WeightMirrorCandidate],
        policy: &WeightMirrorPolicy,
    ) -> Result<WeightMirrorPlan> {
        validate_policy(policy)?;
        let destination = resolve_destination(self.roots(), destination.as_ref())?;
        let inventory = self
            .files()
            .iter()
            .map(|file| (file.relative_path.as_str(), file))
            .collect::<BTreeMap<_, _>>();
        let mut seen = BTreeSet::new();
        let mut ranked = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            validate_relative_path(&candidate.relative_path)?;
            if candidate.benefit == 0 {
                return Err(PowerError::InvalidRequest(format!(
                    "partial weight mirror candidate '{}' must have a positive benefit",
                    candidate.relative_path
                )));
            }
            if !seen.insert(candidate.relative_path.as_str()) {
                return Err(PowerError::InvalidRequest(format!(
                    "partial weight mirror candidate '{}' is duplicated",
                    candidate.relative_path
                )));
            }
            let file = inventory
                .get(candidate.relative_path.as_str())
                .ok_or_else(|| {
                    PowerError::InvalidRequest(format!(
                        "partial weight mirror candidate '{}' is not in the verified collection",
                        candidate.relative_path
                    ))
                })?;
            ranked.push((candidate, *file));
        }
        ranked.sort_by(|(left, left_file), (right, right_file)| {
            compare_candidates(left, left_file.bytes, right, right_file.bytes)
        });

        let mut selected_bytes = 0_u64;
        let mut files = Vec::new();
        for (candidate, file) in ranked {
            let Some(next_bytes) = selected_bytes.checked_add(file.bytes) else {
                return Err(PowerError::InvalidRequest(
                    "partial weight mirror selection byte count overflowed".to_string(),
                ));
            };
            if next_bytes > policy.max_bytes {
                continue;
            }
            selected_bytes = next_bytes;
            files.push(WeightMirrorPlannedFile {
                relative_path: file.relative_path.clone(),
                bytes: file.bytes,
                sha256: file.sha256.clone(),
                benefit: candidate.benefit,
                reused: false,
            });
        }

        let conflicts = inspect_destination(&destination, &mut files)?;
        let reused_bytes = files
            .iter()
            .filter(|file| file.reused)
            .try_fold(0_u64, |total, file| total.checked_add(file.bytes))
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "partial weight mirror reused byte count overflowed".to_string(),
                )
            })?;
        let copy_bytes = selected_bytes.checked_sub(reused_bytes).ok_or_else(|| {
            PowerError::InvalidRequest(
                "partial weight mirror copy byte count underflowed".to_string(),
            )
        })?;
        let capacity_root = if destination.exists() {
            destination.as_path()
        } else {
            destination.parent().ok_or_else(|| {
                PowerError::InvalidRequest(
                    "partial weight mirror destination has no parent".to_string(),
                )
            })?
        };
        let available_bytes = available_space(capacity_root)?;
        let required_bytes = copy_bytes.checked_add(policy.reserve_bytes);
        let rejection = if !conflicts.is_empty() {
            Some(WeightMirrorPlanRejection::DestinationConflict)
        } else if files.is_empty() {
            Some(WeightMirrorPlanRejection::NoCandidateFits)
        } else if required_bytes.is_none_or(|required| required > available_bytes) {
            Some(WeightMirrorPlanRejection::InsufficientSpace)
        } else if policy.confidentiality != WeightMirrorConfidentiality::CallerManagedPlaintext {
            Some(WeightMirrorPlanRejection::PlaintextStagingDenied)
        } else {
            None
        };

        Ok(WeightMirrorPlan {
            schema: WeightMirrorPlan::SCHEMA.to_string(),
            model_sha256: self.sha256().to_string(),
            max_bytes: policy.max_bytes,
            reserve_bytes: policy.reserve_bytes,
            available_bytes,
            selected_bytes,
            reused_bytes,
            copy_bytes,
            files,
            conflicts,
            admitted: rejection.is_none(),
            rejection,
        })
    }

    /// Stages a partial mirror without overwriting any existing file.
    ///
    /// Each SafeTensors file is copied into a same-directory temporary file,
    /// checked against the digest already verified by this store, synced, and
    /// atomically published through a no-replace hard link. Cancellation can
    /// leave completed exact files behind for a later resumable call, but never
    /// publishes a partial file. No receipt, routing history, path, or telemetry
    /// is persisted automatically. This is blocking filesystem work.
    pub fn stage_partial_mirror_blocking(
        &self,
        destination: impl AsRef<Path>,
        candidates: &[WeightMirrorCandidate],
        policy: &WeightMirrorPolicy,
        cancellation: &CancellationToken,
    ) -> Result<WeightMirrorStageReport> {
        check_cancelled(cancellation)?;
        let destination = resolve_destination(self.roots(), destination.as_ref())?;
        let plan = self.plan_partial_mirror(&destination, candidates, policy)?;
        if let Some(rejection) = &plan.rejection {
            return Err(plan_rejection_error(rejection));
        }

        if !destination.exists() {
            std::fs::create_dir(&destination).map_err(|error| {
                PowerError::Io(std::io::Error::new(
                    error.kind(),
                    format!("failed to create partial weight mirror: {error}"),
                ))
            })?;
            if let Some(parent) = destination.parent() {
                sync_directory(parent)?;
            }
        }

        let mut copied_files = 0_usize;
        let mut copied_bytes = 0_u64;
        let mut reused_files = 0_usize;
        let mut reused_bytes = 0_u64;
        for file in &plan.files {
            check_cancelled(cancellation)?;
            if file.reused {
                reused_files = reused_files.saturating_add(1);
                reused_bytes = reused_bytes.checked_add(file.bytes).ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "partial weight mirror reused byte count overflowed".to_string(),
                    )
                })?;
                continue;
            }
            let source = self.verified_file_path(&file.relative_path)?;
            let target = destination.join(&file.relative_path);
            ensure_target_parent(&destination, &target)?;
            copy_verified_no_replace(source, &target, file.bytes, &file.sha256, cancellation)?;
            copied_files = copied_files.saturating_add(1);
            copied_bytes = copied_bytes.checked_add(file.bytes).ok_or_else(|| {
                PowerError::InvalidRequest(
                    "partial weight mirror copied byte count overflowed".to_string(),
                )
            })?;
        }

        let mut verified_files = plan.files.clone();
        let conflicts = inspect_destination(&destination, &mut verified_files)?;
        if !conflicts.is_empty() || verified_files.iter().any(|file| !file.reused) {
            return Err(PowerError::IntegrityCheckFailed {
                model: "partial weight mirror".to_string(),
                expected: self.sha256().to_string(),
                actual: "destination files failed exact verification".to_string(),
            });
        }

        Ok(WeightMirrorStageReport {
            schema: WeightMirrorStageReport::SCHEMA.to_string(),
            plan,
            copied_files,
            copied_bytes,
            reused_files,
            reused_bytes,
        })
    }
}

fn validate_policy(policy: &WeightMirrorPolicy) -> Result<()> {
    if policy.max_bytes == 0 {
        return Err(PowerError::Config(
            "partial weight mirror budget must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn validate_relative_path(relative_path: &str) -> Result<()> {
    if relative_path.is_empty()
        || Path::new(relative_path).is_absolute()
        || Path::new(relative_path)
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(PowerError::InvalidRequest(format!(
            "partial weight mirror candidate '{relative_path}' is not a canonical relative path"
        )));
    }
    Ok(())
}

fn compare_candidates(
    left: &WeightMirrorCandidate,
    left_bytes: u64,
    right: &WeightMirrorCandidate,
    right_bytes: u64,
) -> Ordering {
    let left_density = u128::from(left.benefit).saturating_mul(u128::from(right_bytes));
    let right_density = u128::from(right.benefit).saturating_mul(u128::from(left_bytes));
    right_density
        .cmp(&left_density)
        .then_with(|| right.benefit.cmp(&left.benefit))
        .then_with(|| left_bytes.cmp(&right_bytes))
        .then_with(|| left.relative_path.cmp(&right.relative_path))
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        return Err(PowerError::InferenceFailed(
            "partial weight mirror staging was cancelled".to_string(),
        ));
    }
    Ok(())
}

fn plan_rejection_error(rejection: &WeightMirrorPlanRejection) -> PowerError {
    match rejection {
        WeightMirrorPlanRejection::PlaintextStagingDenied => PowerError::PolicyViolation(
            "partial weight mirror staging requires explicit caller-managed plaintext authority"
                .to_string(),
        ),
        WeightMirrorPlanRejection::NoCandidateFits => PowerError::InvalidRequest(
            "no partial weight mirror candidate fits the configured budget".to_string(),
        ),
        WeightMirrorPlanRejection::InsufficientSpace => PowerError::Config(
            "partial weight mirror would violate the configured free-space reserve".to_string(),
        ),
        WeightMirrorPlanRejection::DestinationConflict => PowerError::IntegrityCheckFailed {
            model: "partial weight mirror".to_string(),
            expected: "only exact files from the selected verified collection".to_string(),
            actual: "destination contains conflicting files".to_string(),
        },
    }
}

#[cfg(test)]
mod tests;
