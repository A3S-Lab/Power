//! Upstream-reported Prism `llama-server` timings (not Power-verified digests).

use serde::{Deserialize, Serialize};

use crate::backend::SpeculativeMetricsSnapshot;

/// Timings object returned by Prism / llama.cpp OpenAI-compatible servers.
///
/// Marked **upstream-reported**: Power did not hash draft artifacts or verify
/// acceptance locally unless a supervisor recorded digests separately.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct UpstreamTimings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_n: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_n: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_n: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_per_second: Option<f64>,
    /// Present when Prism DSpark (or other draft) is engaged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_n: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_n_accepted: Option<u64>,
}

impl UpstreamTimings {
    pub fn from_response(json: &serde_json::Value) -> Option<Self> {
        let timings = json.get("timings")?;
        serde_json::from_value(timings.clone()).ok()
    }

    pub fn prompt_eval_duration_ns(&self) -> Option<u64> {
        self.prompt_ms
            .filter(|ms| *ms >= 0.0)
            .map(|ms| (ms * 1_000_000.0) as u64)
    }

    pub fn speculation_engaged(&self) -> bool {
        self.draft_n.unwrap_or(0) > 0
    }

    pub fn into_observation_json(self) -> serde_json::Value {
        serde_json::json!({
            "source": "upstream-reported",
            "timings": self,
        })
    }
}

/// Cumulative upstream-reported speculative counters for health / observation.
#[derive(Debug, Default)]
pub struct UpstreamSpecMetrics {
    pub requests: u64,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub emitted_tokens: u64,
    pub decode_duration_ns: u64,
}

impl UpstreamSpecMetrics {
    pub fn observe(&mut self, timings: &UpstreamTimings) {
        self.requests = self.requests.saturating_add(1);
        if let Some(n) = timings.draft_n {
            self.drafted_tokens = self.drafted_tokens.saturating_add(n);
        }
        if let Some(n) = timings.draft_n_accepted {
            self.accepted_tokens = self.accepted_tokens.saturating_add(n);
        }
        if let Some(n) = timings.predicted_n {
            self.emitted_tokens = self.emitted_tokens.saturating_add(n);
        }
        if let Some(ms) = timings.predicted_ms.filter(|ms| *ms >= 0.0) {
            self.decode_duration_ns = self
                .decode_duration_ns
                .saturating_add((ms * 1_000_000.0) as u64);
        }
    }

    pub fn snapshot(&self, model: &str, strategy: &str) -> SpeculativeMetricsSnapshot {
        SpeculativeMetricsSnapshot {
            backend: "prism".into(),
            model: model.into(),
            strategy: strategy.into(),
            requests: self.requests,
            rounds: 0,
            target_passes: 0,
            drafted_tokens: self.drafted_tokens,
            accepted_tokens: self.accepted_tokens,
            emitted_tokens: self.emitted_tokens,
            decode_duration_ns: self.decode_duration_ns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_baseline_timings_without_draft() {
        let json = serde_json::json!({
            "timings": {
                "predicted_n": 8,
                "predicted_ms": 100.0,
                "predicted_per_second": 80.0,
                "prompt_ms": 50.0
            }
        });
        let t = UpstreamTimings::from_response(&json).unwrap();
        assert!(!t.speculation_engaged());
        assert_eq!(t.prompt_eval_duration_ns(), Some(50_000_000));
        let obs = t.into_observation_json();
        assert_eq!(obs["source"], "upstream-reported");
    }

    #[test]
    fn parses_dspark_draft_counters() {
        let json = serde_json::json!({
            "timings": {
                "draft_n": 40,
                "draft_n_accepted": 36,
                "predicted_n": 100,
                "predicted_ms": 500.0
            }
        });
        let t = UpstreamTimings::from_response(&json).unwrap();
        assert!(t.speculation_engaged());
        let mut metrics = UpstreamSpecMetrics::default();
        metrics.observe(&t);
        let snap = metrics.snapshot("bonsai2", "prism-dspark-upstream");
        assert_eq!(snap.drafted_tokens, 40);
        assert_eq!(snap.accepted_tokens, 36);
        assert_eq!(snap.emitted_tokens, 100);
    }
}
