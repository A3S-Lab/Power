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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_timestamp_format() {
        let ts = ollama_timestamp();
        // Should end with Z
        assert!(ts.ends_with('Z'), "timestamp should end with Z: {ts}");
        // Should contain T separator
        assert!(ts.contains('T'), "timestamp should contain T: {ts}");
        // Should have nanosecond precision (9 digits after decimal)
        let dot_pos = ts.rfind('.').expect("should have decimal point");
        let z_pos = ts.rfind('Z').unwrap();
        let nano_digits = z_pos - dot_pos - 1;
        assert_eq!(nano_digits, 9, "should have 9 nanosecond digits");
    }

    #[test]
    fn test_format_ollama_timestamp_specific() {
        use chrono::TimeZone;
        let dt = chrono::Utc
            .with_ymd_and_hms(2024, 1, 15, 10, 30, 0)
            .unwrap();
        let ts = format_ollama_timestamp(&dt);
        assert_eq!(ts, "2024-01-15T10:30:00.000000000Z");
    }

    #[test]
    fn test_format_ollama_timestamp_ends_with_z() {
        let dt = chrono::Utc::now();
        let ts = format_ollama_timestamp(&dt);
        assert!(ts.ends_with('Z'));
    }
}
