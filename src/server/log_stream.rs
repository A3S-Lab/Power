//! In-process log capture layer for streaming server logs to HTTP clients.
//!
//! `LogBuffer` stores the last N log entries in a ring buffer and broadcasts
//! new entries via a tokio broadcast channel.  `LogBufferLayer` is a
//! `tracing_subscriber::Layer` that writes matching events to the buffer.
//!
//! Attach the layer when building the global tracing subscriber, then pass the
//! `LogBuffer` into `AppState` so the `GET /v1/logs` SSE handler can serve it.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tokio::sync::broadcast;
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

/// Maximum entries kept in the ring buffer.
const BUFFER_CAPACITY: usize = 500;

/// A single captured log entry.
#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    /// ISO-8601 timestamp with milliseconds.
    pub ts: String,
    /// Level string: "ERROR", "WARN", "INFO", "DEBUG", "TRACE".
    pub level: String,
    /// Tracing target (module path, e.g. `a3s_power::model::pull`).
    pub target: String,
    /// Formatted log message.
    pub message: String,
}

/// Shared ring buffer + broadcast channel for log entries.
///
/// Clone-cheap: all clones share the same underlying storage.
#[derive(Clone)]
pub struct LogBuffer {
    inner: Arc<Mutex<VecDeque<LogEntry>>>,
    tx: broadcast::Sender<LogEntry>,
}

impl LogBuffer {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(256);
        Self {
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(BUFFER_CAPACITY))),
            tx,
        }
    }

    /// Push a new entry into the ring buffer and broadcast it to live subscribers.
    pub fn push(&self, entry: LogEntry) {
        {
            let mut buf = self.inner.lock().unwrap();
            if buf.len() >= BUFFER_CAPACITY {
                buf.pop_front();
            }
            buf.push_back(entry.clone());
        }
        // Ignore send errors — no active receivers is fine.
        let _ = self.tx.send(entry);
    }

    /// Return a snapshot of all buffered entries (oldest first).
    pub fn recent(&self) -> Vec<LogEntry> {
        self.inner.lock().unwrap().iter().cloned().collect()
    }

    /// Subscribe to the live broadcast channel for new entries.
    pub fn subscribe(&self) -> broadcast::Receiver<LogEntry> {
        self.tx.subscribe()
    }
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ── tracing Layer ─────────────────────────────────────────────────────────────

/// A `tracing_subscriber::Layer` that captures log events into a `LogBuffer`.
///
/// Only events whose target starts with one of the configured prefixes are
/// captured to avoid noise from third-party crates.
pub struct LogBufferLayer {
    buffer: LogBuffer,
    /// Target prefixes to capture (e.g. `["a3s_power", "safeclaw"]`).
    filter: Vec<String>,
}

impl LogBufferLayer {
    /// Create a layer that captures `a3s_power` and `safeclaw` targets.
    pub fn new(buffer: LogBuffer) -> Self {
        Self {
            buffer,
            filter: vec!["a3s_power".to_string(), "safeclaw".to_string()],
        }
    }
}

impl<S: Subscriber> Layer<S> for LogBufferLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let target = event.metadata().target();
        if !self.filter.iter().any(|p| target.starts_with(p.as_str())) {
            return;
        }

        let level = event.metadata().level().as_str().to_uppercase();

        let mut visitor = MessageVisitor(String::new());
        event.record(&mut visitor);

        self.buffer.push(LogEntry {
            ts: chrono::Utc::now()
                .to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            level,
            target: target.to_string(),
            message: visitor.0,
        });
    }
}

// ── visitor that extracts the "message" field ─────────────────────────────────

struct MessageVisitor(String);

impl tracing::field::Visit for MessageVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.0 = value.to_string();
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{value:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_recent_roundtrip() {
        let buf = LogBuffer::new();
        buf.push(LogEntry {
            ts: "2024-01-01T00:00:00.000Z".to_string(),
            level: "INFO".to_string(),
            target: "a3s_power".to_string(),
            message: "hello".to_string(),
        });
        let entries = buf.recent();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].message, "hello");
    }

    #[test]
    fn ring_buffer_evicts_oldest_at_capacity() {
        let buf = LogBuffer::new();
        for i in 0..=BUFFER_CAPACITY {
            buf.push(LogEntry {
                ts: String::new(),
                level: "INFO".to_string(),
                target: "a3s_power".to_string(),
                message: format!("msg-{i}"),
            });
        }
        let entries = buf.recent();
        assert_eq!(entries.len(), BUFFER_CAPACITY);
        assert_eq!(entries[0].message, "msg-1");
    }

    #[tokio::test]
    async fn subscribe_receives_new_entries() {
        let buf = LogBuffer::new();
        let mut rx = buf.subscribe();

        buf.push(LogEntry {
            ts: String::new(),
            level: "INFO".to_string(),
            target: "a3s_power".to_string(),
            message: "broadcast".to_string(),
        });

        let entry = rx.try_recv().unwrap();
        assert_eq!(entry.message, "broadcast");
    }
}
