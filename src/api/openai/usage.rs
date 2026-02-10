use std::collections::HashMap;

use axum::extract::{Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::server::state::AppState;

/// Query parameters for the GET /v1/usage endpoint.
#[derive(Debug, Deserialize)]
pub struct UsageQuery {
    /// Start date filter (ISO 8601, e.g. "2024-01-01" or "2024-01-01T00:00:00Z").
    pub start: Option<String>,
    /// End date filter (ISO 8601).
    pub end: Option<String>,
    /// Filter by model name.
    pub model: Option<String>,
}

/// Aggregated usage data for a single (date, model) bucket.
#[derive(Debug, Clone, Serialize)]
pub struct UsageBucket {
    pub date: String,
    pub model: String,
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub total_duration_secs: f64,
    pub estimated_cost_dollars: f64,
}

/// Summary totals across all buckets.
#[derive(Debug, Clone, Serialize)]
pub struct UsageTotals {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub estimated_cost_dollars: f64,
}

/// Response body for GET /v1/usage.
#[derive(Debug, Clone, Serialize)]
pub struct UsageResponse {
    pub data: Vec<UsageBucket>,
    pub total: UsageTotals,
}

/// Parse a date string into a UTC DateTime.
/// Accepts "YYYY-MM-DD" or full ISO 8601 "YYYY-MM-DDTHH:MM:SSZ".
fn parse_date(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    // Try full ISO 8601 first
    if let Ok(dt) = s.parse::<chrono::DateTime<chrono::Utc>>() {
        return Some(dt);
    }
    // Try date-only format (treat as start of day UTC)
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(0, 0, 0)?;
        return Some(chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(
            dt,
            chrono::Utc,
        ));
    }
    None
}

/// Aggregate usage records into (date, model) buckets.
pub fn aggregate_usage(
    records: &[crate::server::metrics::UsageRecord],
) -> (Vec<UsageBucket>, UsageTotals) {
    // Key: (date_string, model)
    let mut buckets: HashMap<(String, String), UsageBucket> = HashMap::new();

    for record in records {
        let date = record.timestamp.format("%Y-%m-%d").to_string();
        let key = (date.clone(), record.model.clone());

        let bucket = buckets.entry(key).or_insert_with(|| UsageBucket {
            date,
            model: record.model.clone(),
            requests: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            total_duration_secs: 0.0,
            estimated_cost_dollars: 0.0,
        });

        bucket.requests += 1;
        bucket.prompt_tokens += record.prompt_tokens as u64;
        bucket.completion_tokens += record.completion_tokens as u64;
        bucket.total_tokens += record.total_tokens as u64;
        bucket.total_duration_secs += record.duration_secs;
        bucket.estimated_cost_dollars += record.cost_dollars;
    }

    let mut data: Vec<UsageBucket> = buckets.into_values().collect();
    data.sort_by(|a, b| a.date.cmp(&b.date).then(a.model.cmp(&b.model)));

    let total = UsageTotals {
        requests: data.iter().map(|b| b.requests).sum(),
        prompt_tokens: data.iter().map(|b| b.prompt_tokens).sum(),
        completion_tokens: data.iter().map(|b| b.completion_tokens).sum(),
        total_tokens: data.iter().map(|b| b.total_tokens).sum(),
        estimated_cost_dollars: data.iter().map(|b| b.estimated_cost_dollars).sum(),
    };

    (data, total)
}

/// GET /v1/usage - Usage and cost dashboard data.
pub async fn handler(
    State(state): State<AppState>,
    Query(query): Query<UsageQuery>,
) -> impl IntoResponse {
    let start = query.start.as_deref().and_then(parse_date);
    let end = query.end.as_deref().and_then(parse_date);
    let model = query.model.as_deref();

    let records = state.metrics.query_usage(start, end, model);
    let (data, total) = aggregate_usage(&records);

    Json(UsageResponse { data, total })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::metrics::UsageRecord;

    #[test]
    fn test_parse_date_iso8601() {
        let dt = parse_date("2024-06-15T10:30:00Z");
        assert!(dt.is_some());
        let dt = dt.unwrap();
        assert_eq!(dt.format("%Y-%m-%d").to_string(), "2024-06-15");
    }

    #[test]
    fn test_parse_date_date_only() {
        let dt = parse_date("2024-06-15");
        assert!(dt.is_some());
        let dt = dt.unwrap();
        assert_eq!(dt.format("%Y-%m-%d").to_string(), "2024-06-15");
    }

    #[test]
    fn test_parse_date_invalid() {
        assert!(parse_date("not-a-date").is_none());
        assert!(parse_date("").is_none());
    }

    #[test]
    fn test_aggregate_usage_empty() {
        let (data, total) = aggregate_usage(&[]);
        assert!(data.is_empty());
        assert_eq!(total.requests, 0);
        assert_eq!(total.total_tokens, 0);
    }

    #[test]
    fn test_aggregate_usage_single_record() {
        let now = chrono::Utc::now();
        let records = vec![UsageRecord {
            timestamp: now,
            model: "llama3".to_string(),
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            duration_secs: 1.5,
            cost_dollars: 0.0,
        }];

        let (data, total) = aggregate_usage(&records);
        assert_eq!(data.len(), 1);
        assert_eq!(data[0].model, "llama3");
        assert_eq!(data[0].requests, 1);
        assert_eq!(data[0].prompt_tokens, 100);
        assert_eq!(data[0].completion_tokens, 50);
        assert_eq!(data[0].total_tokens, 150);
        assert_eq!(total.requests, 1);
        assert_eq!(total.total_tokens, 150);
    }

    #[test]
    fn test_aggregate_usage_groups_by_date_and_model() {
        let today = chrono::Utc::now();
        let yesterday = today - chrono::Duration::days(1);

        let records = vec![
            UsageRecord {
                timestamp: today,
                model: "llama3".to_string(),
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                duration_secs: 1.0,
                cost_dollars: 0.0,
            },
            UsageRecord {
                timestamp: today,
                model: "llama3".to_string(),
                prompt_tokens: 200,
                completion_tokens: 100,
                total_tokens: 300,
                duration_secs: 2.0,
                cost_dollars: 0.0,
            },
            UsageRecord {
                timestamp: yesterday,
                model: "qwen".to_string(),
                prompt_tokens: 50,
                completion_tokens: 25,
                total_tokens: 75,
                duration_secs: 0.5,
                cost_dollars: 0.0,
            },
        ];

        let (data, total) = aggregate_usage(&records);
        // Should have 2 buckets: (yesterday, qwen) and (today, llama3)
        assert_eq!(data.len(), 2);
        assert_eq!(total.requests, 3);
        assert_eq!(total.total_tokens, 525);

        // The today/llama3 bucket should have 2 requests aggregated
        let llama_bucket = data.iter().find(|b| b.model == "llama3").unwrap();
        assert_eq!(llama_bucket.requests, 2);
        assert_eq!(llama_bucket.prompt_tokens, 300);
        assert_eq!(llama_bucket.completion_tokens, 150);
        assert_eq!(llama_bucket.total_tokens, 450);
    }

    #[test]
    fn test_aggregate_usage_sorted_by_date_then_model() {
        let today = chrono::Utc::now();
        let yesterday = today - chrono::Duration::days(1);

        let records = vec![
            UsageRecord {
                timestamp: today,
                model: "zephyr".to_string(),
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                duration_secs: 0.1,
                cost_dollars: 0.0,
            },
            UsageRecord {
                timestamp: yesterday,
                model: "alpha".to_string(),
                prompt_tokens: 20,
                completion_tokens: 10,
                total_tokens: 30,
                duration_secs: 0.2,
                cost_dollars: 0.0,
            },
            UsageRecord {
                timestamp: today,
                model: "alpha".to_string(),
                prompt_tokens: 30,
                completion_tokens: 15,
                total_tokens: 45,
                duration_secs: 0.3,
                cost_dollars: 0.0,
            },
        ];

        let (data, _) = aggregate_usage(&records);
        assert_eq!(data.len(), 3);
        // Sorted: yesterday/alpha, today/alpha, today/zephyr
        assert_eq!(data[0].model, "alpha");
        assert!(data[0].date < data[1].date || data[0].model <= data[1].model);
        assert_eq!(data[2].model, "zephyr");
    }

    #[tokio::test]
    async fn test_usage_handler_empty() {
        use crate::backend::BackendRegistry;
        use crate::config::PowerConfig;
        use crate::model::registry::ModelRegistry;
        use crate::server::router;
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use std::sync::Arc;
        use tower::util::ServiceExt;

        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        let app = router::build(state);
        let req = Request::builder()
            .uri("/v1/usage")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["data"].as_array().unwrap().is_empty());
        assert_eq!(json["total"]["requests"], 0);
        assert_eq!(json["total"]["total_tokens"], 0);
    }

    #[tokio::test]
    async fn test_usage_handler_with_model_filter() {
        use crate::backend::BackendRegistry;
        use crate::config::PowerConfig;
        use crate::model::registry::ModelRegistry;
        use crate::server::router;
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use std::sync::Arc;
        use tower::util::ServiceExt;

        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );

        // Add some usage records
        state.metrics.record_usage(UsageRecord {
            timestamp: chrono::Utc::now(),
            model: "llama3".to_string(),
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            duration_secs: 1.0,
            cost_dollars: 0.0,
        });
        state.metrics.record_usage(UsageRecord {
            timestamp: chrono::Utc::now(),
            model: "qwen".to_string(),
            prompt_tokens: 200,
            completion_tokens: 100,
            total_tokens: 300,
            duration_secs: 2.0,
            cost_dollars: 0.0,
        });

        let app = router::build(state);
        let req = Request::builder()
            .uri("/v1/usage?model=llama3")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let data = json["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["model"], "llama3");
        assert_eq!(json["total"]["requests"], 1);
    }

    #[test]
    fn test_parse_date_with_time() {
        let dt = parse_date("2024-01-15T14:30:45Z");
        assert!(dt.is_some());
        let dt = dt.unwrap();
        assert_eq!(dt.format("%Y-%m-%d").to_string(), "2024-01-15");
    }

    #[test]
    fn test_parse_date_malformed() {
        assert!(parse_date("2024-13-01").is_none()); // Invalid month
        assert!(parse_date("2024-01-32").is_none()); // Invalid day
        // Note: "24-01-01" might parse as a valid date in some formats, so skip this test
    }

    #[test]
    fn test_aggregate_usage_with_cost() {
        let now = chrono::Utc::now();
        let records = vec![
            UsageRecord {
                timestamp: now,
                model: "gpt4".to_string(),
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                duration_secs: 1.0,
                cost_dollars: 0.05,
            },
            UsageRecord {
                timestamp: now,
                model: "gpt4".to_string(),
                prompt_tokens: 200,
                completion_tokens: 100,
                total_tokens: 300,
                duration_secs: 2.0,
                cost_dollars: 0.10,
            },
        ];

        let (data, total) = aggregate_usage(&records);
        assert_eq!(data.len(), 1);
        assert!((data[0].estimated_cost_dollars - 0.15).abs() < 0.001);
        assert!((total.estimated_cost_dollars - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_aggregate_usage_duration_accumulation() {
        let now = chrono::Utc::now();
        let records = vec![
            UsageRecord {
                timestamp: now,
                model: "test".to_string(),
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                duration_secs: 1.5,
                cost_dollars: 0.0,
            },
            UsageRecord {
                timestamp: now,
                model: "test".to_string(),
                prompt_tokens: 20,
                completion_tokens: 10,
                total_tokens: 30,
                duration_secs: 2.5,
                cost_dollars: 0.0,
            },
        ];

        let (data, _) = aggregate_usage(&records);
        assert_eq!(data.len(), 1);
        assert_eq!(data[0].total_duration_secs, 4.0);
    }

    #[test]
    fn test_aggregate_usage_multiple_models_same_day() {
        let now = chrono::Utc::now();
        let records = vec![
            UsageRecord {
                timestamp: now,
                model: "model-a".to_string(),
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                duration_secs: 1.0,
                cost_dollars: 0.0,
            },
            UsageRecord {
                timestamp: now,
                model: "model-b".to_string(),
                prompt_tokens: 200,
                completion_tokens: 100,
                total_tokens: 300,
                duration_secs: 2.0,
                cost_dollars: 0.0,
            },
            UsageRecord {
                timestamp: now,
                model: "model-c".to_string(),
                prompt_tokens: 50,
                completion_tokens: 25,
                total_tokens: 75,
                duration_secs: 0.5,
                cost_dollars: 0.0,
            },
        ];

        let (data, total) = aggregate_usage(&records);
        assert_eq!(data.len(), 3);
        assert_eq!(total.requests, 3);
        assert_eq!(total.total_tokens, 525);
    }

    #[test]
    fn test_usage_bucket_clone() {
        let bucket = UsageBucket {
            date: "2024-01-01".to_string(),
            model: "test".to_string(),
            requests: 10,
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            total_duration_secs: 5.0,
            estimated_cost_dollars: 0.1,
        };
        let cloned = bucket.clone();
        assert_eq!(bucket.date, cloned.date);
        assert_eq!(bucket.requests, cloned.requests);
    }

    #[test]
    fn test_usage_totals_clone() {
        let totals = UsageTotals {
            requests: 100,
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
            estimated_cost_dollars: 1.5,
        };
        let cloned = totals.clone();
        assert_eq!(totals.requests, cloned.requests);
        assert_eq!(totals.estimated_cost_dollars, cloned.estimated_cost_dollars);
    }

    #[test]
    fn test_usage_response_clone() {
        let response = UsageResponse {
            data: vec![],
            total: UsageTotals {
                requests: 0,
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                estimated_cost_dollars: 0.0,
            },
        };
        let cloned = response.clone();
        assert_eq!(response.data.len(), cloned.data.len());
    }

    #[test]
    fn test_usage_query_debug() {
        let query = UsageQuery {
            start: Some("2024-01-01".to_string()),
            end: Some("2024-12-31".to_string()),
            model: Some("test".to_string()),
        };
        let debug = format!("{:?}", query);
        assert!(debug.contains("UsageQuery"));
    }
}
