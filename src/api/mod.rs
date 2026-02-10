pub mod autoload;
pub mod health;
pub mod native;
pub mod openai;
pub mod sse;
pub mod types;

/// Format a UTC timestamp in Ollama's wire format: nanosecond precision with Z suffix.
///
/// Example: `2024-01-15T10:30:00.000000000Z`
pub fn ollama_timestamp() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S%.9fZ")
        .to_string()
}

/// Format an existing chrono DateTime in Ollama's wire format.
pub fn format_ollama_timestamp(dt: &chrono::DateTime<chrono::Utc>) -> String {
    dt.format("%Y-%m-%dT%H:%M:%S%.9fZ").to_string()
}
