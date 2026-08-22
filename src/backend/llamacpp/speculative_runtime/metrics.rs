use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use crate::backend::SpeculativeMetricsSnapshot;
use crate::server::lock::mutex_lock;
use crate::speculative::SpeculativeMetrics;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct TelemetryKey {
    model: String,
    strategy: String,
}

#[derive(Debug, Clone, Copy, Default)]
struct TelemetryTotals {
    requests: u64,
    rounds: u64,
    target_passes: u64,
    drafted_tokens: u64,
    accepted_tokens: u64,
    emitted_tokens: u64,
    decode_duration_ns: u64,
}

/// Process-lifetime counters for completed llama.cpp speculative requests.
#[derive(Debug, Default)]
pub(crate) struct SpeculativeTelemetry {
    totals: Mutex<HashMap<TelemetryKey, TelemetryTotals>>,
}

impl SpeculativeTelemetry {
    fn record(
        &self,
        model: &str,
        strategy: &str,
        metrics: SpeculativeMetrics,
        emitted_tokens: usize,
        decode_duration_ns: u64,
    ) {
        let key = TelemetryKey {
            model: model.to_string(),
            strategy: strategy.to_string(),
        };
        let mut totals = mutex_lock(&self.totals);
        let entry = totals.entry(key).or_default();
        entry.requests = entry.requests.saturating_add(1);
        entry.rounds = entry.rounds.saturating_add(metrics.rounds);
        entry.target_passes = entry.target_passes.saturating_add(metrics.target_passes);
        entry.drafted_tokens = entry.drafted_tokens.saturating_add(metrics.drafted_tokens);
        entry.accepted_tokens = entry
            .accepted_tokens
            .saturating_add(metrics.accepted_tokens);
        entry.emitted_tokens = entry
            .emitted_tokens
            .saturating_add(u64::try_from(emitted_tokens).unwrap_or(u64::MAX));
        entry.decode_duration_ns = entry.decode_duration_ns.saturating_add(decode_duration_ns);
    }

    pub(crate) fn snapshot(&self) -> Vec<SpeculativeMetricsSnapshot> {
        mutex_lock(&self.totals)
            .iter()
            .map(|(key, totals)| SpeculativeMetricsSnapshot {
                backend: "llama.cpp".to_string(),
                model: key.model.clone(),
                strategy: key.strategy.clone(),
                requests: totals.requests,
                rounds: totals.rounds,
                target_passes: totals.target_passes,
                drafted_tokens: totals.drafted_tokens,
                accepted_tokens: totals.accepted_tokens,
                emitted_tokens: totals.emitted_tokens,
                decode_duration_ns: totals.decode_duration_ns,
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct FrCoverageMetrics {
    pub(super) target_samples: u64,
    pub(super) target_samples_in_token_id_prefix: u64,
    pub(super) rejected_rounds: u64,
    pub(super) corrections_outside_token_id_prefix: u64,
}

impl FrCoverageMetrics {
    // This is intentionally a token-ID-prefix diagnostic. It is exact for the
    // legacy prefix shortlist, but it is not membership coverage for a ranked
    // d2t vocabulary whose compact rows can map to arbitrary target token IDs.
    pub(super) fn observe_target_sample(&mut self, token: i32, fr_vocab_size: Option<u32>) {
        let Some(fr_vocab_size) = fr_vocab_size else {
            return;
        };
        self.target_samples = self.target_samples.saturating_add(1);
        if u32::try_from(token).is_ok_and(|token| token < fr_vocab_size) {
            self.target_samples_in_token_id_prefix =
                self.target_samples_in_token_id_prefix.saturating_add(1);
        }
    }

    pub(super) fn observe_rejection(&mut self, correction: i32, fr_vocab_size: Option<u32>) {
        let Some(fr_vocab_size) = fr_vocab_size else {
            return;
        };
        self.rejected_rounds = self.rejected_rounds.saturating_add(1);
        if u32::try_from(correction).map_or(true, |token| token >= fr_vocab_size) {
            self.corrections_outside_token_id_prefix =
                self.corrections_outside_token_id_prefix.saturating_add(1);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MtpPhaseTimings {
    pub(super) draft_ns: u64,
    pub(super) target_decode_ns: u64,
    pub(super) target_only_decode_ns: u64,
    pub(super) accepted_prefix_sync_ns: u64,
    pub(super) sampling_ns: u64,
    pub(super) state_management_ns: u64,
    pub(super) streaming_ns: u64,
    pub(super) fallback_replays: u32,
    pub(super) rollback_guard_activations: u32,
    pub(super) rollback_guard_draft_limit: usize,
    pub(super) rollback_guard_after_round: Option<u64>,
    pub(super) max_rejected_suffix: usize,
    pub(super) target_only_tokens: u64,
    pub(super) target_only_after_round: Option<u64>,
    // Counts draft steps captured by llama.cpp's device-resident recurrent path;
    // a failed first probe is visible as a one-step partial chain before fallback.
    pub(super) recurrent_draft_chains: u64,
    pub(super) recurrent_draft_steps: u64,
    pub(super) accepted_prefix_histogram: [u32; 65],
    pub(super) draft_limit_histogram: [u32; 65],
    pub(super) fr: FrCoverageMetrics,
}

impl Default for MtpPhaseTimings {
    fn default() -> Self {
        Self {
            draft_ns: 0,
            target_decode_ns: 0,
            target_only_decode_ns: 0,
            accepted_prefix_sync_ns: 0,
            sampling_ns: 0,
            state_management_ns: 0,
            streaming_ns: 0,
            fallback_replays: 0,
            rollback_guard_activations: 0,
            rollback_guard_draft_limit: 0,
            rollback_guard_after_round: None,
            max_rejected_suffix: 0,
            target_only_tokens: 0,
            target_only_after_round: None,
            recurrent_draft_chains: 0,
            recurrent_draft_steps: 0,
            accepted_prefix_histogram: [0; 65],
            draft_limit_histogram: [0; 65],
            fr: FrCoverageMetrics::default(),
        }
    }
}

impl MtpPhaseTimings {
    fn instrumented_ns(self) -> u64 {
        self.draft_ns
            .saturating_add(self.target_decode_ns)
            .saturating_add(self.target_only_decode_ns)
            .saturating_add(self.accepted_prefix_sync_ns)
            .saturating_add(self.sampling_ns)
            .saturating_add(self.state_management_ns)
            .saturating_add(self.streaming_ns)
    }
}

pub(super) fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

pub(super) fn log_metrics(
    model: &str,
    strategy: &str,
    telemetry: &SpeculativeTelemetry,
    metrics: SpeculativeMetrics,
    timings: MtpPhaseTimings,
    emitted_tokens: usize,
    decode_started: Instant,
) {
    let elapsed_duration_ns = elapsed_ns(decode_started);
    telemetry.record(
        model,
        strategy,
        metrics,
        emitted_tokens,
        elapsed_duration_ns,
    );
    let elapsed = elapsed_duration_ns as f64 / 1_000_000_000.0;
    let tokens_per_second = if elapsed > 0.0 {
        emitted_tokens as f64 / elapsed
    } else {
        0.0
    };
    tracing::info!(
        model,
        strategy,
        rounds = metrics.rounds,
        drafted_tokens = metrics.drafted_tokens,
        accepted_tokens = metrics.accepted_tokens,
        emitted_tokens,
        verified_emitted_tokens = metrics.emitted_tokens,
        acceptance_rate = metrics.acceptance_rate(),
        tokens_per_target_pass = metrics.tokens_per_target_pass(),
        tokens_per_second,
        draft_duration_ns = timings.draft_ns,
        target_decode_duration_ns = timings.target_decode_ns,
        target_only_decode_duration_ns = timings.target_only_decode_ns,
        accepted_prefix_sync_duration_ns = timings.accepted_prefix_sync_ns,
        sampling_duration_ns = timings.sampling_ns,
        state_management_duration_ns = timings.state_management_ns,
        streaming_duration_ns = timings.streaming_ns,
        runtime_overhead_ns = elapsed_duration_ns.saturating_sub(timings.instrumented_ns()),
        fallback_replays = timings.fallback_replays,
        rollback_guard_activations = timings.rollback_guard_activations,
        rollback_guard_draft_limit = timings.rollback_guard_draft_limit,
        rollback_guard_after_round = ?timings.rollback_guard_after_round,
        max_rejected_suffix = timings.max_rejected_suffix,
        target_only_tokens = timings.target_only_tokens,
        target_only_after_round = ?timings.target_only_after_round,
        recurrent_draft_chains = timings.recurrent_draft_chains,
        recurrent_draft_steps = timings.recurrent_draft_steps,
        accepted_prefix_histogram = ?timings.accepted_prefix_histogram,
        draft_limit_histogram = ?timings.draft_limit_histogram,
        fr_coverage_scope = "token_id_prefix",
        fr_target_samples = timings.fr.target_samples,
        fr_target_samples_in_token_id_prefix = timings.fr.target_samples_in_token_id_prefix,
        fr_rejected_rounds = timings.fr.rejected_rounds,
        fr_corrections_outside_token_id_prefix = timings.fr.corrections_outside_token_id_prefix,
        "llama.cpp speculative completion finished"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_aggregates_by_model_and_strategy() {
        let telemetry = SpeculativeTelemetry::default();
        telemetry.record(
            "target",
            "dspark",
            SpeculativeMetrics {
                rounds: 2,
                target_passes: 2,
                drafted_tokens: 8,
                accepted_tokens: 7,
                emitted_tokens: 9,
            },
            10,
            1_000,
        );
        telemetry.record(
            "target",
            "dspark",
            SpeculativeMetrics {
                rounds: 1,
                target_passes: 1,
                drafted_tokens: 4,
                accepted_tokens: 4,
                emitted_tokens: 5,
            },
            5,
            2_000,
        );

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].requests, 2);
        assert_eq!(snapshot[0].rounds, 3);
        assert_eq!(snapshot[0].drafted_tokens, 12);
        assert_eq!(snapshot[0].accepted_tokens, 11);
        assert_eq!(snapshot[0].emitted_tokens, 15);
        assert_eq!(snapshot[0].decode_duration_ns, 3_000);
    }
}
