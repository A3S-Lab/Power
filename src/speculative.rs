//! Model-neutral speculative-decoding primitives.
//!
//! Power owns reusable proposal, adaptive-length, and exact-acceptance
//! mechanics. Model backends remain responsible for draft weights, target
//! block verification, and transactional KV, recurrent, convolution, sampler,
//! and decoder state.
//!
//! [`SpeculativeStrategy`] is the model-neutral control-plane choice. Backends
//! advertise capabilities and resolve `auto` to their own safe default. The
//! zero-weight strategies are implemented here through [`SpecMode`]:
//! - [`SpecMode::Off`] — plain autoregressive decoding, no draft.
//! - [`SpecMode::PromptLookup`] — match the generated suffix n-gram against the
//!   prompt. Zero draft cost; wins when output overlaps input (summarize,
//!   JSON with known keys, code completion).
//! - [`SpecMode::NgramContext`] — zero-weight self-speculation: an online
//!   n-gram model over the *full running sequence* (prompt + generated),
//!   prefix-chained, so it also accelerates free-form generation.
//!
//! # Relationship to DSpark
//!
//! DSpark combines a parallel draft backbone, a lightweight prefix-dependent
//! head, adaptive verification length, and lossless target verification. Power
//! owns its strategy selection, scheduling, verification accounting, and state
//! transaction contract. Model backends own the tensor graph and weights.
//! [`NgramContextDrafter`] is only a zero-weight integration baseline; it is
//! not a trained DSpark drafter. Qwen, Llama, DeepSeek, and future architectures
//! plug in behind the same backend capability boundary instead of becoming
//! variants in the shared runtime.
//!
//! # State Rollback
//!
//! The caller must transactionally retain the accepted prefix and discard the
//! rejected suffix. This includes every mutable architecture state, not only a
//! transformer KV cache.

/// Default number of draft tokens to propose per speculation round.
pub const DRAFT_K: usize = 4;

mod adaptive;

pub use adaptive::{AdaptiveK, AdaptiveSpeculationController};

#[cfg(feature = "server")]
pub mod benchmark;

/// Model-neutral speculative-decoding strategy selected by configuration.
///
/// A strategy being parseable does not mean that every backend or model can
/// execute it. The backend must resolve it through [`SpeculativeCapabilities`]
/// and fail closed when the required adapter or model tensors are absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SpeculativeStrategy {
    /// Let the backend select its documented safe default.
    #[default]
    Auto,
    /// Plain autoregressive decoding.
    Off,
    /// Zero-weight prompt lookup.
    PromptLookup,
    /// Zero-weight online n-gram drafting.
    NgramContext,
    /// A separately loaded small draft model.
    DraftModel,
    /// Native multi-token prediction tensors embedded in the target model.
    Mtp,
    /// Dynamic tree drafting with a compatible model adapter.
    Dflash,
    /// DSpark parallel drafting with a compatible model adapter.
    Dspark,
}

impl SpeculativeStrategy {
    /// Parse a configuration value without silently changing its meaning.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "off" | "none" | "false" => Some(Self::Off),
            "prompt-lookup" | "prompt_lookup" | "lookup" | "true" => Some(Self::PromptLookup),
            "ngram-context" | "ngram_context" | "context" => Some(Self::NgramContext),
            "draft-model" | "draft_model" | "draft" => Some(Self::DraftModel),
            "mtp" | "multi-token-prediction" | "multi_token_prediction" => Some(Self::Mtp),
            "dflash" => Some(Self::Dflash),
            "dspark" => Some(Self::Dspark),
            _ => None,
        }
    }

    /// Stable configuration spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::PromptLookup => "prompt-lookup",
            Self::NgramContext => "ngram-context",
            Self::DraftModel => "draft-model",
            Self::Mtp => "mtp",
            Self::Dflash => "dflash",
            Self::Dspark => "dspark",
        }
    }

    /// Return the in-process zero-weight implementation, when applicable.
    pub const fn zero_weight_mode(self) -> Option<SpecMode> {
        match self {
            Self::Off => Some(SpecMode::Off),
            Self::PromptLookup => Some(SpecMode::PromptLookup),
            Self::NgramContext => Some(SpecMode::NgramContext),
            Self::Auto | Self::DraftModel | Self::Mtp | Self::Dflash | Self::Dspark => None,
        }
    }
}

/// Set of speculative strategies implemented by a backend/model pair.
///
/// The compact bitset keeps model names and backend types out of the shared
/// layer while still making capability negotiation explicit and testable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeCapabilities(u16);

impl Default for SpeculativeCapabilities {
    fn default() -> Self {
        Self::none()
    }
}

impl SpeculativeCapabilities {
    /// No speculative strategies beyond plain autoregressive decoding.
    pub const fn none() -> Self {
        Self(1 << SpeculativeStrategy::Off as u8)
    }

    /// Add a supported strategy.
    pub const fn with(mut self, strategy: SpeculativeStrategy) -> Self {
        if !matches!(strategy, SpeculativeStrategy::Auto) {
            self.0 |= 1 << strategy as u8;
        }
        self
    }

    /// Whether this backend/model pair implements `strategy`.
    pub const fn supports(self, strategy: SpeculativeStrategy) -> bool {
        matches!(strategy, SpeculativeStrategy::Auto) || (self.0 & (1 << strategy as u8)) != 0
    }

    /// Resolve `auto` and reject unsupported explicit selections.
    pub fn resolve(
        self,
        requested: SpeculativeStrategy,
        backend_default: SpeculativeStrategy,
    ) -> Result<SpeculativeStrategy, UnsupportedSpeculativeStrategy> {
        let resolved = if matches!(requested, SpeculativeStrategy::Auto) {
            backend_default
        } else {
            requested
        };
        if self.supports(resolved) {
            Ok(resolved)
        } else {
            Err(UnsupportedSpeculativeStrategy {
                requested: resolved,
            })
        }
    }
}

/// An explicit strategy is unavailable for the selected backend/model pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedSpeculativeStrategy {
    pub requested: SpeculativeStrategy,
}

impl std::fmt::Display for UnsupportedSpeculativeStrategy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "speculative strategy '{}' is not supported by this backend/model",
            self.requested.as_str()
        )
    }
}

impl std::error::Error for UnsupportedSpeculativeStrategy {}

/// Per-request counters shared by all speculative implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpeculativeMetrics {
    pub rounds: u64,
    pub target_passes: u64,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub emitted_tokens: u64,
}

impl SpeculativeMetrics {
    /// Record one target verification pass.
    pub fn record_round(&mut self, drafted: usize, accepted: usize, emitted: usize) {
        self.rounds = self.rounds.saturating_add(1);
        self.target_passes = self.target_passes.saturating_add(1);
        self.drafted_tokens = self.drafted_tokens.saturating_add(drafted as u64);
        self.accepted_tokens = self.accepted_tokens.saturating_add(accepted as u64);
        self.emitted_tokens = self.emitted_tokens.saturating_add(emitted as u64);
    }

    /// Accepted draft-token fraction, or zero before the first draft.
    pub fn acceptance_rate(self) -> f64 {
        if self.drafted_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / self.drafted_tokens as f64
        }
    }

    /// Output tokens emitted per target verification pass.
    pub fn tokens_per_target_pass(self) -> f64 {
        if self.target_passes == 0 {
            0.0
        } else {
            self.emitted_tokens as f64 / self.target_passes as f64
        }
    }
}

/// Lossless result of sampling a target model over one proposed block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedBlock<T> {
    /// Number of draft tokens accepted before correction or bonus sampling.
    pub accepted: usize,
    /// Accepted drafts followed by a correction or bonus token, unless an
    /// accepted draft itself terminated verification.
    pub emitted: Vec<T>,
}

/// Verify proposed token IDs using target samples at successive batch rows.
///
/// `sample(i)` must sample target logits for row `i`. It is called only up to
/// the first mismatch; when all drafts match, row `drafts.len()` supplies the
/// bonus token. This preserves sampler state exactly because rejected suffixes
/// are never sampled.
pub fn verify_token_block<T>(drafts: &[T], mut sample: impl FnMut(usize) -> T) -> VerifiedBlock<T>
where
    T: Copy + Eq,
{
    verify_token_block_until(drafts, &mut sample, |_| false)
}

/// Verify a proposed block while stopping after an accepted terminal token.
///
/// The terminal predicate is only evaluated for matching draft tokens. A
/// mismatch correction and an all-accepted bonus already end the target
/// sampling pass, so no later row can be observed in either case.
pub fn verify_token_block_until<T>(
    drafts: &[T],
    mut sample: impl FnMut(usize) -> T,
    mut is_terminal: impl FnMut(T) -> bool,
) -> VerifiedBlock<T>
where
    T: Copy + Eq,
{
    let mut emitted = Vec::with_capacity(drafts.len() + 1);
    for (index, &draft) in drafts.iter().enumerate() {
        let target = sample(index);
        if target != draft {
            emitted.push(target);
            return VerifiedBlock {
                accepted: index,
                emitted,
            };
        }
        emitted.push(draft);
        if is_terminal(draft) {
            return VerifiedBlock {
                accepted: index + 1,
                emitted,
            };
        }
    }
    emitted.push(sample(drafts.len()));
    VerifiedBlock {
        accepted: drafts.len(),
        emitted,
    }
}

/// Bound a proposal so one output slot remains for the target correction or
/// bonus token and the verification batch stays within context capacity.
pub fn bounded_draft_len(
    requested: usize,
    remaining_tokens: usize,
    remaining_context: usize,
) -> usize {
    requested
        .min(remaining_tokens.saturating_sub(1))
        .min(remaining_context)
}

/// Minimum logical batch size needed to execute an MTP proposal safely.
///
/// The target evaluates the current anchor plus at most `draft_max` proposed
/// tokens in one batch. llama.cpp's recurrent allocator also requires the
/// physical batch to be strictly larger than its anchor-plus-snapshot tail.
/// Power caps resident snapshots at `draft_max`, so one additional staging row
/// covers both constraints without coupling request validation to that cap.
pub const fn minimum_mtp_batch(draft_max: u32) -> u32 {
    draft_max.saturating_add(2)
}

/// Minimum n-gram size for prompt lookup matching.
const MIN_NGRAM: usize = 2;

/// Maximum n-gram size to try for prompt lookup matching.
const MAX_NGRAM: usize = 5;

/// Speculation mode — selects the draft strategy for the decode loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpecMode {
    /// No speculation — plain autoregressive decoding.
    Off,
    /// Prompt-lookup: match the generated suffix n-gram against the prompt only.
    #[default]
    PromptLookup,
    /// Online n-gram over the full running sequence (prompt + generated).
    NgramContext,
}

impl SpecMode {
    /// Parse a config string (case-insensitive).
    ///
    /// Unknown values return `None` so configuration validation can fail closed
    /// instead of silently selecting a different decoding policy.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "false" => Some(Self::Off),
            "prompt-lookup" | "prompt_lookup" | "lookup" | "true" => Some(Self::PromptLookup),
            "ngram-context" | "ngram_context" | "context" => Some(Self::NgramContext),
            _ => None,
        }
    }

    /// Build the drafter for this mode. `Off` has no drafter.
    pub fn drafter(self) -> Option<Box<dyn Drafter>> {
        match self {
            Self::Off => None,
            Self::PromptLookup => Some(Box::new(PromptLookupDrafter)),
            Self::NgramContext => Some(Box::new(NgramContextDrafter)),
        }
    }
}

/// A draft strategy: propose continuation tokens to verify speculatively.
///
/// Implementations must be cheap relative to a full forward pass — the whole
/// point is to avoid per-token model passes. A trained draft head would
/// implement this trait too (running its own small forward), replacing the
/// statistical drafters without touching the decode loop.
pub trait Drafter: Send + Sync {
    /// Propose up to `max_draft` tokens continuing `generated_ids`, given the
    /// original `input_ids` (prompt). Returns an empty vec when no confident
    /// draft is available.
    fn draft(&self, input_ids: &[u32], generated_ids: &[u32], max_draft: usize) -> Vec<u32>;
}

/// Prompt-lookup drafter: suffix n-gram matched against the prompt only.
pub struct PromptLookupDrafter;

impl Drafter for PromptLookupDrafter {
    fn draft(&self, input_ids: &[u32], generated_ids: &[u32], max_draft: usize) -> Vec<u32> {
        prompt_lookup_draft(input_ids, generated_ids, max_draft)
    }
}

/// Zero-weight n-gram self-speculative drafter.
///
/// Searches the model's **own generated output** first (captures the
/// self-repetition common in code, lists and structured text), then falls back
/// to the prompt. The single n-gram lookup returns a prefix-consistent block of
/// continuation tokens — the statistical analogue of DSpark's prefix-dependent
/// sequential head, with zero trained weights.
pub struct NgramContextDrafter;

impl Drafter for NgramContextDrafter {
    fn draft(&self, input_ids: &[u32], generated_ids: &[u32], max_draft: usize) -> Vec<u32> {
        // Match the recent suffix inside earlier generated output. The trailing
        // suffix itself is never matched (the search requires start <= len-n-1).
        if generated_ids.len() > MIN_NGRAM {
            let max_n = MAX_NGRAM.min(generated_ids.len());
            for n in (MIN_NGRAM..=max_n).rev() {
                let suffix = &generated_ids[generated_ids.len() - n..];
                if let Some(c) = find_ngram_continuation(generated_ids, suffix, max_draft) {
                    if !c.is_empty() {
                        return c;
                    }
                }
            }
        }
        // Fall back to prompt-lookup.
        prompt_lookup_draft(input_ids, generated_ids, max_draft)
    }
}

/// Lossless rejection-sampling acceptance over a verified draft block.
///
/// For each draft position, sample the target token from that position's logits
/// with `sample` (which must apply the same temperature/top-p/rng the main loop
/// uses). Accept the draft while it equals the freshly sampled target; at the
/// first mismatch the sampled target token is the **lossless correction** and
/// the walk stops. If every draft is accepted, `bonus_logits` is sampled for
/// one free extra token.
///
/// This is distribution-exact: every emitted token (accepted draft, correction,
/// or bonus) equals a sample from the target distribution at its position, so
/// the output is identical to plain sampling. For greedy decoding `sample` is
/// argmax and acceptance reduces to "draft matches the model's argmax".
///
/// Returns `(n_accepted, correction_or_bonus_token)`.
pub fn accept_block(
    drafts: &[u32],
    target_logits: &[Vec<f32>],
    bonus_logits: &[f32],
    sample: &mut impl FnMut(&[f32]) -> u32,
) -> (usize, u32) {
    for (i, &d) in drafts.iter().enumerate() {
        debug_assert!(i < target_logits.len(), "missing target logits for draft");
        let target = sample(&target_logits[i]);
        if target != d {
            return (i, target);
        }
    }
    (drafts.len(), sample(bonus_logits))
}

/// Look up the most recent n-gram in the generated token sequence and find
/// a matching continuation in the input tokens.
///
/// Returns up to `max_draft` candidate token IDs from the input that follow
/// the matched n-gram, or an empty vec if no match is found.
pub fn prompt_lookup_draft(input_ids: &[u32], generated_ids: &[u32], max_draft: usize) -> Vec<u32> {
    if generated_ids.len() < MIN_NGRAM || input_ids.len() < MIN_NGRAM + 1 {
        return Vec::new();
    }

    // Try decreasing n-gram sizes for the best match
    let max_n = MAX_NGRAM.min(generated_ids.len());
    for n in (MIN_NGRAM..=max_n).rev() {
        let suffix = &generated_ids[generated_ids.len() - n..];

        // Search for this n-gram in the input tokens
        if let Some(candidates) = find_ngram_continuation(input_ids, suffix, max_draft) {
            if !candidates.is_empty() {
                return candidates;
            }
        }
    }

    Vec::new()
}

/// Find the last occurrence of `ngram` in `tokens` and return up to
/// `max_count` tokens that follow it.
fn find_ngram_continuation(tokens: &[u32], ngram: &[u32], max_count: usize) -> Option<Vec<u32>> {
    let n = ngram.len();
    if tokens.len() < n + 1 {
        return None;
    }

    // Search backwards for the most recent match (more likely to be relevant)
    for start in (0..=tokens.len() - n - 1).rev() {
        if tokens[start..start + n] == *ngram {
            let cont_start = start + n;
            let cont_end = (cont_start + max_count).min(tokens.len());
            return Some(tokens[cont_start..cont_end].to_vec());
        }
    }

    None
}

/// Get the argmax token from a logits vector (greedy selection).
pub fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_lookup_finds_continuation() {
        // Input: "The cat sat on the mat"
        let input = vec![1, 2, 3, 4, 1, 2, 5];
        // Generated so far: "the cat" → [1, 2]
        let generated = vec![1, 2];
        let draft = prompt_lookup_draft(&input, &generated, 4);
        // Should find [1, 2] at position 0 → continuation is [3, 4, 1, 2]
        // Or at position 4 → continuation is [5]
        // We search backwards, so position 4 is found first → [5]
        assert_eq!(draft, vec![5]);
    }

    #[test]
    fn test_prompt_lookup_longer_ngram() {
        let input = vec![1, 2, 3, 4, 5, 1, 2, 3, 6, 7];
        // Generated: [1, 2, 3] — 3-gram match
        let generated = vec![1, 2, 3];
        let draft = prompt_lookup_draft(&input, &generated, 4);
        // Backwards search: [1,2,3] at position 5 → continuation [6, 7]
        assert_eq!(draft, vec![6, 7]);
    }

    #[test]
    fn test_prompt_lookup_no_match() {
        let input = vec![1, 2, 3, 4, 5];
        let generated = vec![9, 8]; // not in input
        let draft = prompt_lookup_draft(&input, &generated, 4);
        assert!(draft.is_empty());
    }

    #[test]
    fn test_prompt_lookup_too_short() {
        let input = vec![1, 2, 3];
        let generated = vec![1]; // less than MIN_NGRAM
        let draft = prompt_lookup_draft(&input, &generated, 4);
        assert!(draft.is_empty());
    }

    #[test]
    fn test_prompt_lookup_max_draft_limit() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let generated = vec![1, 2];
        let draft = prompt_lookup_draft(&input, &generated, 2);
        // [1,2] at position 0 → continuation limited to 2 tokens: [3, 4]
        assert_eq!(draft.len(), 2);
    }

    #[test]
    fn test_argmax_token() {
        assert_eq!(argmax_token(&logits_with_max(3, 10)), 3);
        assert_eq!(argmax_token(&logits_with_max(0, 10)), 0);
        assert_eq!(argmax_token(&logits_with_max(9, 10)), 9);
    }

    #[test]
    fn test_find_ngram_continuation() {
        let tokens = vec![1, 2, 3, 4, 5];
        assert_eq!(
            find_ngram_continuation(&tokens, &[2, 3], 3),
            Some(vec![4, 5])
        );
        assert_eq!(find_ngram_continuation(&tokens, &[9, 9], 3), None);
    }

    #[test]
    fn test_find_ngram_at_end() {
        let tokens = vec![1, 2, 3, 4, 5];
        // [4, 5] is at the end — no continuation
        assert_eq!(find_ngram_continuation(&tokens, &[4, 5], 3), None);
    }

    // ── New API: modes, drafters, adaptive-K, lossless accept ─────────────────

    #[test]
    fn test_spec_mode_parse() {
        assert_eq!(SpecMode::parse("off"), Some(SpecMode::Off));
        assert_eq!(
            SpecMode::parse("Prompt-Lookup"),
            Some(SpecMode::PromptLookup)
        );
        assert_eq!(SpecMode::parse("dspark"), None);
        assert_eq!(SpecMode::parse("context"), Some(SpecMode::NgramContext));
        assert_eq!(SpecMode::parse("bogus"), None);
        assert_eq!(SpecMode::default(), SpecMode::PromptLookup);
    }

    #[test]
    fn test_spec_mode_drafter_presence() {
        assert!(SpecMode::Off.drafter().is_none());
        assert!(SpecMode::PromptLookup.drafter().is_some());
        assert!(SpecMode::NgramContext.drafter().is_some());
    }

    #[test]
    fn test_model_neutral_strategy_parse() {
        assert_eq!(
            SpeculativeStrategy::parse("multi-token-prediction"),
            Some(SpeculativeStrategy::Mtp)
        );
        assert_eq!(
            SpeculativeStrategy::parse("DSpark"),
            Some(SpeculativeStrategy::Dspark)
        );
        assert_eq!(
            SpeculativeStrategy::parse("draft_model"),
            Some(SpeculativeStrategy::DraftModel)
        );
        assert_eq!(SpeculativeStrategy::parse("bogus"), None);
    }

    #[test]
    fn test_capability_resolution_is_backend_and_model_specific() {
        let capabilities = SpeculativeCapabilities::none()
            .with(SpeculativeStrategy::Mtp)
            .with(SpeculativeStrategy::Dspark);

        assert_eq!(
            capabilities
                .resolve(SpeculativeStrategy::Auto, SpeculativeStrategy::Mtp)
                .unwrap(),
            SpeculativeStrategy::Mtp
        );
        assert_eq!(
            capabilities
                .resolve(SpeculativeStrategy::Dspark, SpeculativeStrategy::Off)
                .unwrap(),
            SpeculativeStrategy::Dspark
        );
        assert_eq!(
            capabilities
                .resolve(SpeculativeStrategy::Dflash, SpeculativeStrategy::Off)
                .unwrap_err()
                .requested,
            SpeculativeStrategy::Dflash
        );
    }

    #[test]
    fn test_verify_token_block_rejects_without_sampling_suffix() {
        let drafts = [10, 20, 30];
        let target = [10, 99, 88, 77];
        let mut sampled_rows = Vec::new();
        let verified = verify_token_block(&drafts, |row| {
            sampled_rows.push(row);
            target[row]
        });

        assert_eq!(verified.accepted, 1);
        assert_eq!(verified.emitted, vec![10, 99]);
        assert_eq!(sampled_rows, vec![0, 1]);
    }

    #[test]
    fn test_verify_token_block_emits_bonus_after_full_acceptance() {
        let drafts = [10, 20, 30];
        let target = [10, 20, 30, 40];
        let verified = verify_token_block(&drafts, |row| target[row]);

        assert_eq!(verified.accepted, drafts.len());
        assert_eq!(verified.emitted, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_verify_token_block_stops_at_accepted_terminal() {
        let drafts = [10, 20, 30];
        let target = [10, 20, 30, 40];
        let mut sampled_rows = Vec::new();
        let verified = verify_token_block_until(
            &drafts,
            |row| {
                sampled_rows.push(row);
                target[row]
            },
            |token| token == 20,
        );

        assert_eq!(verified.accepted, 2);
        assert_eq!(verified.emitted, vec![10, 20]);
        assert_eq!(sampled_rows, vec![0, 1]);
    }

    #[test]
    fn test_bounded_draft_len_reserves_correction_and_context() {
        assert_eq!(bounded_draft_len(4, 5, 8), 4);
        assert_eq!(bounded_draft_len(4, 3, 8), 2);
        assert_eq!(bounded_draft_len(4, 5, 2), 2);
        assert_eq!(bounded_draft_len(4, 1, 8), 0);
        assert_eq!(bounded_draft_len(4, 0, 8), 0);
    }

    #[test]
    fn test_minimum_mtp_batch_covers_verification_and_recurrent_tail() {
        assert_eq!(minimum_mtp_batch(0), 2);
        assert_eq!(minimum_mtp_batch(3), 5);
        assert_eq!(minimum_mtp_batch(u32::MAX), u32::MAX);
    }

    #[test]
    fn test_speculative_metrics_report_runtime_gain() {
        let mut metrics = SpeculativeMetrics::default();
        metrics.record_round(3, 2, 3);
        metrics.record_round(3, 3, 4);

        assert_eq!(metrics.rounds, 2);
        assert!((metrics.acceptance_rate() - (5.0 / 6.0)).abs() < f64::EPSILON);
        assert!((metrics.tokens_per_target_pass() - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ngram_context_drafts_from_own_output() {
        // No prompt overlap, but the generation repeats a pattern: "a b c ... a b ?"
        // Generated: [1,2,3, 9, 1,2] — suffix [1,2] earlier continues with 3.
        let input = vec![100, 101]; // unrelated prompt
        let generated = vec![1, 2, 3, 9, 1, 2];
        let draft = NgramContextDrafter.draft(&input, &generated, 4);
        assert_eq!(draft, vec![3, 9, 1, 2]); // continuation after the earlier [1,2]
    }

    #[test]
    fn test_ngram_context_falls_back_to_prompt() {
        // Generation has no internal repetition, but the suffix matches the prompt.
        let input = vec![1, 2, 3, 4, 5];
        let generated = vec![7, 1, 2]; // suffix [1,2] not repeated in generated → use prompt
        let draft = NgramContextDrafter.draft(&input, &generated, 3);
        assert_eq!(draft, vec![3, 4, 5]);
    }

    #[test]
    fn test_prompt_lookup_drafter_matches_function() {
        let input = vec![1, 2, 3, 4, 1, 2, 5];
        let generated = vec![1, 2];
        assert_eq!(
            PromptLookupDrafter.draft(&input, &generated, 4),
            prompt_lookup_draft(&input, &generated, 4)
        );
    }

    #[test]
    fn test_adaptive_k_grows_on_high_acceptance() {
        let mut a = AdaptiveK::new(4, 1, 8);
        for _ in 0..10 {
            a.update(4, 4); // perfect acceptance
        }
        assert!(a.current() > 4, "K should grow under high acceptance");
        assert!(a.current() <= 8, "K must respect max");
    }

    #[test]
    fn test_adaptive_k_shrinks_on_low_acceptance() {
        let mut a = AdaptiveK::new(4, 1, 8);
        for _ in 0..10 {
            a.update(0, 4); // nothing accepted
        }
        assert!(a.current() < 4, "K should shrink under low acceptance");
        assert!(a.current() >= 1, "K must respect min");
    }

    #[test]
    fn test_adaptive_k_clamps_and_ignores_empty() {
        let mut a = AdaptiveK::new(100, 2, 6);
        assert_eq!(a.current(), 6, "initial clamped to max");
        let before = a.current();
        a.update(0, 0); // no draft → no change
        assert_eq!(a.current(), before);
    }

    #[test]
    fn adaptive_speculation_warm_starts_inside_rollback_window() {
        let controller = AdaptiveSpeculationController::new(7, 1, 7, 6);

        assert_eq!(controller.draft_limit(), Some(6));
        assert_eq!(controller.effective_max(), 7);
        assert_eq!(controller.rollback_guard_after_round(), None);
    }

    #[test]
    fn adaptive_speculation_clamps_only_after_replay_worthy_rejection() {
        let mut controller = AdaptiveSpeculationController::new(7, 1, 7, 6);

        controller.observe(6, 6);
        assert_eq!(controller.draft_limit(), Some(7));
        assert_eq!(controller.effective_max(), 7);
        assert_eq!(controller.rollback_guard_after_round(), None);

        controller.observe(0, 7);
        assert_eq!(controller.effective_max(), 6);
        assert_eq!(controller.rollback_guard_after_round(), Some(2));
    }

    #[test]
    fn adaptive_speculation_keeps_wide_perfect_acceptance_path() {
        let mut controller = AdaptiveSpeculationController::new(7, 1, 7, 6);

        for _ in 0..16 {
            let drafted = controller.draft_limit().unwrap();
            controller.observe(drafted, drafted);
        }

        assert_eq!(controller.draft_limit(), Some(7));
        assert_eq!(controller.effective_max(), 7);
        assert_eq!(controller.rollback_guard_after_round(), None);
    }

    #[test]
    fn adaptive_speculation_shrinks_fast_after_zero_acceptance() {
        let mut controller = AdaptiveSpeculationController::new(7, 1, 7, 7);

        controller.observe(0, 7);
        assert_eq!(controller.draft_limit(), Some(4));
        controller.observe(0, 4);
        assert_eq!(controller.draft_limit(), Some(2));
    }

    #[test]
    fn adaptive_speculation_regrows_after_full_acceptance() {
        let mut controller = AdaptiveSpeculationController::new(3, 1, 7, 7);

        controller.observe(3, 3);
        controller.observe(4, 4);

        assert_eq!(controller.draft_limit(), Some(5));
    }

    #[test]
    fn adaptive_speculation_keeps_width_when_target_pass_is_well_amortized() {
        let mut controller = AdaptiveSpeculationController::new(10, 1, 10, 10);

        controller.observe(5, 10);
        assert_eq!(controller.draft_limit(), Some(10));
        controller.observe(7, 10);
        assert_eq!(controller.draft_limit(), Some(10));
    }

    #[test]
    fn adaptive_speculation_jumps_wide_after_first_safe_full_acceptance() {
        let mut controller = AdaptiveSpeculationController::new(10, 1, 10, 6);

        controller.observe(6, 6);
        assert_eq!(controller.draft_limit(), Some(10));
    }

    #[test]
    fn adaptive_speculation_closes_wide_probe_after_partial_acceptance() {
        let mut controller = AdaptiveSpeculationController::new(10, 1, 10, 6);

        controller.observe(5, 6);
        controller.observe(6, 6);
        controller.observe(6, 6);
        assert_eq!(controller.draft_limit(), Some(6));
    }

    #[test]
    fn adaptive_speculation_opens_target_only_circuit_for_sustained_low_yield() {
        let mut controller = AdaptiveSpeculationController::new(7, 1, 7, 7);

        for round in 0..12 {
            let accepted = usize::from(round % 3 == 0);
            let drafted = controller.draft_limit().unwrap();
            controller.observe(accepted, drafted);
        }

        assert_eq!(controller.draft_limit(), None);
        assert_eq!(controller.target_only_after_round(), Some(12));
    }

    #[test]
    fn adaptive_speculation_keeps_healthy_mixed_workload_enabled() {
        let mut controller = AdaptiveSpeculationController::new(7, 1, 7, 7);

        for round in 0..24 {
            let accepted = if round % 3 == 0 { 0 } else { 2 };
            let drafted = controller.draft_limit().unwrap();
            controller.observe(accepted.min(drafted), drafted);
        }

        assert!(controller.draft_limit().is_some());
        assert_eq!(controller.target_only_after_round(), None);
    }

    #[test]
    fn test_accept_block_all_accepted_returns_bonus() {
        let drafts = vec![1, 2, 3];
        let targets = vec![
            logits_with_max(1, 10),
            logits_with_max(2, 10),
            logits_with_max(3, 10),
        ];
        let bonus = logits_with_max(7, 10);
        let mut argmax = |l: &[f32]| argmax_token(l);
        let (n, tok) = accept_block(&drafts, &targets, &bonus, &mut argmax);
        assert_eq!(n, 3);
        assert_eq!(tok, 7, "bonus token sampled after full acceptance");
    }

    #[test]
    fn test_accept_block_rejects_midway_with_correction() {
        let drafts = vec![1, 2, 3, 4];
        // Position 2 mismatches: target argmax is 9, not the drafted 3.
        let targets = vec![
            logits_with_max(1, 10),
            logits_with_max(2, 10),
            logits_with_max(9, 10),
            logits_with_max(4, 10),
        ];
        let bonus = logits_with_max(0, 10);
        let mut argmax = |l: &[f32]| argmax_token(l);
        let (n, tok) = accept_block(&drafts, &targets, &bonus, &mut argmax);
        assert_eq!(n, 2, "accept the matching prefix");
        assert_eq!(tok, 9, "correction is the freshly sampled target token");
    }

    #[test]
    fn test_accept_block_rejects_first() {
        let drafts = vec![5, 2, 3];
        let targets = vec![
            logits_with_max(1, 10),
            logits_with_max(2, 10),
            logits_with_max(3, 10),
        ];
        let bonus = logits_with_max(0, 10);
        let mut argmax = |l: &[f32]| argmax_token(l);
        let (n, tok) = accept_block(&drafts, &targets, &bonus, &mut argmax);
        assert_eq!(n, 0);
        assert_eq!(tok, 1, "correction replaces the rejected first draft");
    }

    /// Helper: create a logits vector where `token_id` has the max value.
    fn logits_with_max(token_id: usize, vocab: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; vocab];
        if token_id < vocab {
            logits[token_id] = 10.0;
        }
        logits
    }
}
