use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use serde::Serialize;

/// Status of an audited operation.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AuditStatus {
    Success,
    Failure { reason: String },
}

/// A structured audit event capturing who did what, when, and the outcome.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEvent {
    /// ISO 8601 timestamp.
    pub timestamp: DateTime<Utc>,
    /// Unique request ID from RequestContext.
    pub request_id: String,
    /// Authenticated identity (key-0, key-1, etc.) if auth is enabled.
    pub auth_id: Option<String>,
    /// Action performed: "chat", "completion", "embedding", "model_load", "auth_failure", "startup".
    pub action: String,
    /// Model name involved, if applicable.
    pub model: Option<String>,
    /// Outcome of the operation.
    #[serde(flatten)]
    pub status: AuditStatus,
    /// Duration in milliseconds, if applicable.
    pub duration_ms: Option<u64>,
    /// Token count (output tokens), if applicable.
    pub tokens: Option<u64>,
}

impl AuditEvent {
    /// Create a success event.
    pub fn success(
        request_id: impl Into<String>,
        auth_id: Option<String>,
        action: impl Into<String>,
        model: Option<String>,
        duration_ms: Option<u64>,
        tokens: Option<u64>,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            request_id: request_id.into(),
            auth_id,
            action: action.into(),
            model,
            status: AuditStatus::Success,
            duration_ms,
            tokens,
        }
    }

    /// Create a failure event.
    pub fn failure(
        request_id: impl Into<String>,
        auth_id: Option<String>,
        action: impl Into<String>,
        model: Option<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            request_id: request_id.into(),
            auth_id,
            action: action.into(),
            model,
            status: AuditStatus::Failure {
                reason: reason.into(),
            },
            duration_ms: None,
            tokens: None,
        }
    }
}

/// Trait for audit log destinations.
///
/// This is an extension point — implement this trait to send audit events
/// to different backends (file, syslog, remote SIEM, etc.).
pub trait AuditLogger: Send + Sync {
    /// Record an audit event.
    fn log(&self, event: &AuditEvent);

    /// Flush any buffered events to the underlying sink.
    ///
    /// Called during graceful shutdown to ensure no events are lost.
    /// Default implementation is a no-op for loggers that write synchronously.
    fn flush(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::error::Result<()>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }
}

/// Writes audit events as JSON Lines to a file.
///
/// Each line is a complete JSON object representing one audit event.
/// The file is append-only and flushed after every write.
pub struct JsonLinesAuditLogger {
    file: Mutex<std::fs::File>,
    path: PathBuf,
}

impl JsonLinesAuditLogger {
    /// Open or create the audit log file at the given path.
    pub fn open(path: PathBuf) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            file: Mutex::new(file),
            path,
        })
    }

    /// Return the path to the audit log file.
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl AuditLogger for JsonLinesAuditLogger {
    fn log(&self, event: &AuditEvent) {
        match serde_json::to_string(event) {
            Ok(line) => {
                if let Ok(mut file) = self.file.lock() {
                    let _ = writeln!(file, "{}", line);
                    let _ = file.flush();
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to serialize audit event");
            }
        }
    }
}

/// A no-op audit logger for use when audit logging is disabled.
pub struct NoopAuditLogger;

impl AuditLogger for NoopAuditLogger {
    fn log(&self, _event: &AuditEvent) {}
}

/// Writes audit events to a file asynchronously via a background task.
///
/// File I/O is offloaded to a dedicated Tokio task so `log()` never blocks
/// the calling async worker. Uses an unbounded channel so callers never block.
pub struct AsyncJsonLinesAuditLogger {
    sender: tokio::sync::mpsc::UnboundedSender<AsyncAuditMsg>,
    path: PathBuf,
}

/// Messages sent to the background writer task.
enum AsyncAuditMsg {
    Line(String),
    Flush(tokio::sync::oneshot::Sender<()>),
}

impl AsyncJsonLinesAuditLogger {
    /// Open or create the audit log file and spawn the background writer task.
    pub fn open(path: PathBuf) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AsyncAuditMsg>();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            let mut async_file = tokio::fs::File::from_std(file);
            while let Some(msg) = rx.recv().await {
                match msg {
                    AsyncAuditMsg::Line(line) => {
                        let _ = async_file.write_all(line.as_bytes()).await;
                        let _ = async_file.write_all(b"\n").await;
                        let _ = async_file.flush().await;
                    }
                    AsyncAuditMsg::Flush(reply) => {
                        let _ = async_file.flush().await;
                        let _ = reply.send(());
                    }
                }
            }
        });

        Ok(Self { sender: tx, path })
    }

    /// Return the path to the audit log file.
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl AuditLogger for AsyncJsonLinesAuditLogger {
    fn log(&self, event: &AuditEvent) {
        if let Ok(line) = serde_json::to_string(event) {
            let _ = self.sender.send(AsyncAuditMsg::Line(line));
        }
    }

    fn flush(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::error::Result<()>> + Send + '_>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.sender.send(AsyncAuditMsg::Flush(tx));
        Box::pin(async move {
            rx.await.map_err(|_| {
                crate::error::PowerError::Server(
                    "Audit logger background task exited before flush completed".to_string(),
                )
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;

    #[test]
    fn test_audit_event_success_serializes() {
        let event = AuditEvent::success(
            "req-123",
            Some("key-0".to_string()),
            "chat",
            Some("llama3".to_string()),
            Some(150),
            Some(42),
        );
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("req-123"));
        assert!(json.contains("key-0"));
        assert!(json.contains("chat"));
        assert!(json.contains("llama3"));
        assert!(json.contains("\"status\":\"success\""));
        assert!(json.contains("150"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_audit_event_failure_serializes() {
        let event = AuditEvent::failure("req-456", None, "auth_failure", None, "invalid token");
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("req-456"));
        assert!(json.contains("auth_failure"));
        assert!(json.contains("\"status\":\"failure\""));
        assert!(json.contains("invalid token"));
    }

    #[test]
    fn test_json_lines_logger_writes_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("audit.jsonl");

        let logger = JsonLinesAuditLogger::open(log_path.clone()).unwrap();

        let event1 = AuditEvent::success(
            "req-1",
            None,
            "chat",
            Some("llama3".to_string()),
            Some(100),
            Some(10),
        );
        let event2 = AuditEvent::failure("req-2", None, "auth_failure", None, "bad key");

        logger.log(&event1);
        logger.log(&event2);

        // Read back and verify
        let file = std::fs::File::open(&log_path).unwrap();
        let lines: Vec<String> = std::io::BufReader::new(file)
            .lines()
            .map(|l| l.unwrap())
            .collect();

        assert_eq!(lines.len(), 2);
        let parsed1: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        let parsed2: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(parsed1["request_id"], "req-1");
        assert_eq!(parsed2["request_id"], "req-2");
        assert_eq!(parsed2["status"], "failure");
    }

    #[test]
    fn test_json_lines_logger_appends() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("audit.jsonl");

        // Write first event
        {
            let logger = JsonLinesAuditLogger::open(log_path.clone()).unwrap();
            logger.log(&AuditEvent::success(
                "req-1", None, "chat", None, None, None,
            ));
        }

        // Write second event with a new logger instance (simulates restart)
        {
            let logger = JsonLinesAuditLogger::open(log_path.clone()).unwrap();
            logger.log(&AuditEvent::success(
                "req-2", None, "chat", None, None, None,
            ));
        }

        let file = std::fs::File::open(&log_path).unwrap();
        let line_count = std::io::BufReader::new(file).lines().count();
        assert_eq!(line_count, 2);
    }

    #[test]
    fn test_json_lines_logger_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("nested").join("dirs").join("audit.jsonl");

        let logger = JsonLinesAuditLogger::open(log_path.clone()).unwrap();
        logger.log(&AuditEvent::success(
            "req-1", None, "startup", None, None, None,
        ));

        assert!(log_path.exists());
    }

    #[test]
    fn test_noop_logger_does_nothing() {
        let logger = NoopAuditLogger;
        // Should not panic
        logger.log(&AuditEvent::success(
            "req-1", None, "chat", None, None, None,
        ));
    }

    #[tokio::test]
    async fn test_async_json_lines_logger_writes_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("async_audit.jsonl");

        let logger = AsyncJsonLinesAuditLogger::open(log_path.clone()).unwrap();

        let event = AuditEvent::success(
            "req-async",
            None,
            "chat",
            Some("llama3".to_string()),
            Some(100),
            Some(10),
        );
        logger.log(&event);

        // Give the background task time to flush
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let file = std::fs::File::open(&log_path).unwrap();
        let lines: Vec<String> = std::io::BufReader::new(file)
            .lines()
            .map(|l| l.unwrap())
            .collect();

        assert_eq!(lines.len(), 1);
        let parsed: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(parsed["request_id"], "req-async");
        assert_eq!(parsed["action"], "chat");
    }

    #[tokio::test]
    async fn test_async_json_lines_logger_path() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("path_test.jsonl");
        let logger = AsyncJsonLinesAuditLogger::open(log_path.clone()).unwrap();
        assert_eq!(logger.path(), &log_path);
    }

    #[test]
    fn test_audit_event_has_timestamp() {
        let before = Utc::now();
        let event = AuditEvent::success("req-1", None, "chat", None, None, None);
        let after = Utc::now();
        assert!(event.timestamp >= before);
        assert!(event.timestamp <= after);
    }

    #[tokio::test]
    async fn test_async_logger_flush_waits_for_pending_writes() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("flush_test.jsonl");
        let logger = AsyncJsonLinesAuditLogger::open(log_path.clone()).unwrap();

        for i in 0..10 {
            let event = AuditEvent::success(
                &format!("req-{i}"),
                None,
                "chat",
                Some("model".to_string()),
                None,
                None,
            );
            logger.log(&event);
        }

        // flush() must wait until all 10 events are written
        logger.flush().await.unwrap();

        let content = std::fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 10, "expected 10 lines after flush, got {}", lines.len());
    }

    #[tokio::test]
    async fn test_noop_logger_flush_is_ok() {
        let logger = NoopAuditLogger;
        assert!(logger.flush().await.is_ok());
    }
}
