use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use serde::Serialize;

/// Open or create an audit log file with restricted permissions.
///
/// On Unix, the file is created with mode 0600 (owner read/write only),
/// preventing other users on the system from reading audit metadata.
/// On non-Unix platforms, falls back to standard file creation.
fn open_audit_file(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(path)
    }
    #[cfg(not(unix))]
    {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
    }
}

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
    fn flush(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::error::Result<()>> + Send + '_>>
    {
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
    ///
    /// On Unix, the file is created with mode 0600 (owner read/write only).
    pub fn open(path: PathBuf) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = open_audit_file(&path)?;
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

// ============================================================================
// Encrypted audit logger
// ============================================================================

/// Writes AES-256-GCM encrypted audit events to a file.
///
/// Each line has the format: `<nonce_hex>.<base64_ciphertext>`
///
/// The nonce is unique per entry (random, 12 bytes). The ciphertext is the
/// AES-256-GCM encryption of the JSON-serialized `AuditEvent`.
///
/// Use `a3s-power-audit decrypt` to read the log given the key.
pub struct EncryptedAuditLogger {
    /// AES-256-GCM key (32 bytes).
    key: [u8; 32],
    inner: JsonLinesAuditLogger,
}

impl EncryptedAuditLogger {
    /// Open or create an encrypted audit log file.
    ///
    /// The file is created with mode 0600 on Unix (owner read/write only).
    pub fn open(path: PathBuf, key: [u8; 32]) -> std::io::Result<Self> {
        let logger = JsonLinesAuditLogger::open(path)?;
        Ok(Self { key, inner: logger })
    }

    /// Encrypt a plaintext string and return `<nonce_hex>.<base64_ciphertext>`.
    fn encrypt_line(&self, plaintext: &str) -> Option<String> {
        use aes_gcm::aead::{Aead, KeyInit, OsRng};
        use aes_gcm::{AeadCore, Aes256Gcm};

        let cipher = Aes256Gcm::new_from_slice(&self.key).ok()?;
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = cipher.encrypt(&nonce, plaintext.as_bytes()).ok()?;

        let nonce_hex: String = nonce.iter().map(|b| format!("{b:02x}")).collect();
        let ct_b64 =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &ciphertext);
        Some(format!("{nonce_hex}.{ct_b64}"))
    }

    /// Decrypt a single log line produced by `encrypt_line`.
    pub fn decrypt_line(line: &str, key: &[u8; 32]) -> Option<String> {
        use aes_gcm::aead::{Aead, KeyInit};
        use aes_gcm::{Aes256Gcm, Nonce};

        let (nonce_hex, ct_b64) = line.split_once('.')?;

        // Decode nonce from hex (24 hex chars = 12 bytes)
        if nonce_hex.len() != 24 {
            return None;
        }
        let nonce_bytes: Vec<u8> = (0..12)
            .map(|i| u8::from_str_radix(&nonce_hex[i * 2..i * 2 + 2], 16).ok())
            .collect::<Option<Vec<_>>>()?;
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext =
            base64::Engine::decode(&base64::engine::general_purpose::STANDARD, ct_b64).ok()?;

        let cipher = Aes256Gcm::new_from_slice(key).ok()?;
        let plaintext = cipher.decrypt(nonce, ciphertext.as_ref()).ok()?;
        String::from_utf8(plaintext).ok()
    }

    /// Return the path to the audit log file.
    pub fn path(&self) -> &PathBuf {
        self.inner.path()
    }
}

impl AuditLogger for EncryptedAuditLogger {
    fn log(&self, event: &AuditEvent) {
        if let Ok(json) = serde_json::to_string(event) {
            if let Some(encrypted_line) = self.encrypt_line(&json) {
                if let Ok(mut file) = self.inner.file.lock() {
                    let _ = writeln!(file, "{}", encrypted_line);
                    let _ = file.flush();
                }
            }
        }
    }
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
    ///
    /// On Unix, the file is created with mode 0600 (owner read/write only).
    pub fn open(path: PathBuf) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = open_audit_file(&path)?;

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

    fn flush(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::error::Result<()>> + Send + '_>>
    {
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
        assert_eq!(
            lines.len(),
            10,
            "expected 10 lines after flush, got {}",
            lines.len()
        );
    }

    #[tokio::test]
    async fn test_noop_logger_flush_is_ok() {
        let logger = NoopAuditLogger;
        assert!(logger.flush().await.is_ok());
    }

    // --- EncryptedAuditLogger tests ---

    fn test_key() -> [u8; 32] {
        [0x42u8; 32]
    }

    #[test]
    fn test_encrypted_logger_encrypt_decrypt_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("encrypted_audit.log");
        let key = test_key();

        let logger = EncryptedAuditLogger::open(log_path.clone(), key).unwrap();
        let event = AuditEvent::success(
            "req-enc-1",
            Some("key-0".to_string()),
            "chat",
            Some("llama3".to_string()),
            Some(200),
            Some(15),
        );
        logger.log(&event);

        // Read back the encrypted line
        let content = std::fs::read_to_string(&log_path).unwrap();
        let line = content.lines().next().unwrap();

        // Verify it's not plaintext
        assert!(
            !line.contains("req-enc-1"),
            "log line must not contain plaintext"
        );
        assert!(
            line.contains('.'),
            "log line must have nonce.ciphertext format"
        );

        // Decrypt and verify
        let decrypted = EncryptedAuditLogger::decrypt_line(line, &key).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&decrypted).unwrap();
        assert_eq!(parsed["request_id"], "req-enc-1");
        assert_eq!(parsed["action"], "chat");
        assert_eq!(parsed["status"], "success");
    }

    #[test]
    fn test_encrypted_logger_wrong_key_fails_decryption() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("encrypted_audit2.log");
        let key = test_key();
        let wrong_key = [0xFFu8; 32];

        let logger = EncryptedAuditLogger::open(log_path.clone(), key).unwrap();
        logger.log(&AuditEvent::success(
            "req-1", None, "chat", None, None, None,
        ));

        let content = std::fs::read_to_string(&log_path).unwrap();
        let line = content.lines().next().unwrap();

        // Decryption with wrong key must fail
        assert!(
            EncryptedAuditLogger::decrypt_line(line, &wrong_key).is_none(),
            "decryption with wrong key must return None"
        );
    }

    #[test]
    fn test_encrypted_logger_each_entry_has_unique_nonce() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("nonce_test.log");
        let key = test_key();

        let logger = EncryptedAuditLogger::open(log_path.clone(), key).unwrap();
        for i in 0..5 {
            logger.log(&AuditEvent::success(
                &format!("req-{i}"),
                None,
                "chat",
                None,
                None,
                None,
            ));
        }

        let content = std::fs::read_to_string(&log_path).unwrap();
        let nonces: Vec<&str> = content
            .lines()
            .filter_map(|l| l.split_once('.').map(|(n, _)| n))
            .collect();

        assert_eq!(nonces.len(), 5);
        // All nonces must be unique
        let unique: std::collections::HashSet<_> = nonces.iter().collect();
        assert_eq!(unique.len(), 5, "each entry must have a unique nonce");
    }

    #[test]
    fn test_encrypted_logger_decrypt_line_invalid_input() {
        let key = test_key();
        // Missing dot separator
        assert!(EncryptedAuditLogger::decrypt_line("nodot", &key).is_none());
        // Wrong nonce length
        assert!(EncryptedAuditLogger::decrypt_line("tooshort.abc", &key).is_none());
        // Invalid base64
        assert!(EncryptedAuditLogger::decrypt_line(
            "aabbccddeeff00112233445566778899.!!!invalid!!!",
            &key
        )
        .is_none());
    }

    #[cfg(unix)]
    #[test]
    fn test_audit_file_permissions_are_0600() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("perm_test.jsonl");

        let _logger = JsonLinesAuditLogger::open(log_path.clone()).unwrap();

        let metadata = std::fs::metadata(&log_path).unwrap();
        let mode = metadata.permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "audit log must have 0600 permissions, got {mode:o}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_encrypted_audit_file_permissions_are_0600() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("enc_perm_test.log");

        let _logger = EncryptedAuditLogger::open(log_path.clone(), test_key()).unwrap();

        let metadata = std::fs::metadata(&log_path).unwrap();
        let mode = metadata.permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "encrypted audit log must have 0600 permissions, got {mode:o}"
        );
    }
}
