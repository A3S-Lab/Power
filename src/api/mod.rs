pub mod autoload;
pub mod health;
pub mod openai;
pub mod types;

/// Format a UTC timestamp in RFC 3339 format.
pub fn timestamp() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Format an existing chrono DateTime in RFC 3339 format.
pub fn format_timestamp(dt: &chrono::DateTime<chrono::Utc>) -> String {
    dt.to_rfc3339()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_format() {
        let ts = timestamp();
        assert!(ts.contains('T'), "timestamp should contain T: {ts}");
        assert!(
            ts.ends_with('Z') || ts.contains('+'),
            "timestamp should have timezone: {ts}"
        );
    }

    #[test]
    fn test_format_timestamp_specific() {
        use chrono::TimeZone;
        let dt = chrono::Utc
            .with_ymd_and_hms(2024, 1, 15, 10, 30, 0)
            .unwrap();
        let ts = format_timestamp(&dt);
        assert!(ts.starts_with("2024-01-15T10:30:00"));
    }

    #[test]
    fn test_format_timestamp_roundtrip() {
        let dt = chrono::Utc::now();
        let ts = format_timestamp(&dt);
        assert!(!ts.is_empty());
    }
}
