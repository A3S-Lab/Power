/// Persistent pull state — tracks in-progress and completed model downloads.
///
/// State files are stored as JSON in `~/.a3s/power/pulls/<name-hash>.json`.
/// On server restart, callers can query `GET /v1/models/pull/:name/status`
/// to check whether a previous download completed, is still in progress, or failed.
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::dirs;
use crate::error::{PowerError, Result};

/// Status of a model pull operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PullStatus {
    /// Download is in progress.
    Pulling,
    /// Download completed successfully.
    Done,
    /// Download failed with an error message.
    Failed,
}

/// Persisted state for a single pull operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullState {
    /// Model name (e.g. `bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M`).
    pub name: String,
    /// Current status.
    pub status: PullStatus,
    /// Bytes downloaded so far.
    pub completed: u64,
    /// Total bytes (0 if unknown).
    pub total: u64,
    /// Error message if status is `Failed`.
    pub error: Option<String>,
    /// RFC3339 timestamp when the pull started.
    pub started_at: String,
    /// RFC3339 timestamp when the pull finished (success or failure).
    pub finished_at: Option<String>,
}

impl PullState {
    /// Create a new in-progress pull state.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            status: PullStatus::Pulling,
            completed: 0,
            total: 0,
            error: None,
            started_at: chrono::Utc::now().to_rfc3339(),
            finished_at: None,
        }
    }

    /// Derive the state file path from the model name.
    pub fn path(name: &str) -> PathBuf {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(name.as_bytes());
        let digest = h.finalize();
        dirs::pulls_dir().join(format!("{:x}.json", digest))
    }

    /// Load state from disk. Returns `None` if no state file exists.
    pub fn load(name: &str) -> Option<Self> {
        let path = Self::path(name);
        let content = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Persist state to disk.
    pub fn save(&self) -> Result<()> {
        let dir = dirs::pulls_dir();
        std::fs::create_dir_all(&dir).map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "failed to create pulls dir: {e}"
            )))
        })?;
        let path = Self::path(&self.name);
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, content).map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "failed to write pull state: {e}"
            )))
        })
    }

    /// Mark as done and persist.
    pub fn mark_done(&mut self) -> Result<()> {
        self.status = PullStatus::Done;
        self.finished_at = Some(chrono::Utc::now().to_rfc3339());
        self.save()
    }

    /// Mark as failed and persist.
    pub fn mark_failed(&mut self, error: &str) -> Result<()> {
        self.status = PullStatus::Failed;
        self.error = Some(error.to_string());
        self.finished_at = Some(chrono::Utc::now().to_rfc3339());
        self.save()
    }

    /// Update progress and persist.
    pub fn update_progress(&mut self, completed: u64, total: u64) -> Result<()> {
        self.completed = completed;
        self.total = total;
        self.save()
    }

    /// Delete the state file (cleanup after successful registration).
    pub fn delete(name: &str) {
        let _ = std::fs::remove_file(Self::path(name));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = PullState::new("owner/repo:Q4_K_M");
        state.save().unwrap();

        let loaded = PullState::load("owner/repo:Q4_K_M").unwrap();
        assert_eq!(loaded.name, "owner/repo:Q4_K_M");
        assert_eq!(loaded.status, PullStatus::Pulling);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_mark_done() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let mut state = PullState::new("owner/repo:Q4_K_M");
        state.save().unwrap();
        state.mark_done().unwrap();

        let loaded = PullState::load("owner/repo:Q4_K_M").unwrap();
        assert_eq!(loaded.status, PullStatus::Done);
        assert!(loaded.finished_at.is_some());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_mark_failed() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let mut state = PullState::new("owner/repo:Q4_K_M");
        state.save().unwrap();
        state.mark_failed("connection reset").unwrap();

        let loaded = PullState::load("owner/repo:Q4_K_M").unwrap();
        assert_eq!(loaded.status, PullStatus::Failed);
        assert_eq!(loaded.error.as_deref(), Some("connection reset"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_update_progress() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let mut state = PullState::new("owner/repo:Q4_K_M");
        state.save().unwrap();
        state.update_progress(1024, 4096).unwrap();

        let loaded = PullState::load("owner/repo:Q4_K_M").unwrap();
        assert_eq!(loaded.completed, 1024);
        assert_eq!(loaded.total, 4096);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_load_nonexistent_returns_none() {
        assert!(PullState::load("nonexistent/model:Q4_K_M").is_none());
    }

    #[test]
    fn test_path_is_deterministic() {
        assert_eq!(
            PullState::path("owner/repo:Q4_K_M"),
            PullState::path("owner/repo:Q4_K_M")
        );
    }

    #[test]
    fn test_path_differs_for_different_names() {
        assert_ne!(
            PullState::path("owner/repo-a:Q4_K_M"),
            PullState::path("owner/repo-b:Q4_K_M")
        );
    }

    #[test]
    #[serial]
    fn test_delete() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = PullState::new("owner/repo:Q4_K_M");
        state.save().unwrap();
        assert!(PullState::load("owner/repo:Q4_K_M").is_some());

        PullState::delete("owner/repo:Q4_K_M");
        assert!(PullState::load("owner/repo:Q4_K_M").is_none());

        std::env::remove_var("A3S_POWER_HOME");
    }
}
