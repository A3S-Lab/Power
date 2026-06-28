//! Self-speculative decoding for picolm — multiple inference modes.
//!
//! Speculative decoding drafts several continuation tokens cheaply, then
//! verifies them in a **single batched layer-streaming pass** (load each
//! layer's weights once, push the whole draft block through). On the
//! memory-bandwidth-bound streaming path this turns K tokens into ~one
//! weight-streaming pass — a near-Kx win when weight loading dominates.
//!
//! Modes ([`SpecMode`]) select the draft strategy:
//! - [`SpecMode::Off`] — plain autoregressive decoding, no draft.
//! - [`SpecMode::PromptLookup`] — match the generated suffix n-gram against the
//!   prompt. Zero draft cost; wins when output overlaps input (summarize,
//!   JSON with known keys, code completion).
//! - [`SpecMode::NgramContext`] — DSpark-like self-speculation: an online
//!   n-gram model over the *full running sequence* (prompt + generated),
//!   prefix-chained, so it also accelerates free-form generation.
//!
//! # Relationship to DSpark
//!
//! DSpark (DeepSeek, 2026) is "semi-parallel" speculative decoding: a parallel
//! backbone proposes a whole block, a lightweight sequential head refines each
//! position with a prefix-dependent bias, an adaptive scheduler sizes the
//! verify length, and rejection sampling keeps it lossless. This module mirrors
//! that *scaffold* with a statistical drafter ([`NgramContextDrafter`], a
//! zero-weight stand-in for DSpark's trained head), an [`AdaptiveK`] scheduler,
//! and lossless [`accept_block`] verification. The [`Drafter`] trait is the
//! seam where a real trained draft head drops in when weights are available.
//!
//! # KV Cache Rollback
//!
//! Rejected draft tokens are rolled back via `KvCache::truncate()`.
//! The verify pass re-populates the KV cache with correct values.

/// Default number of draft tokens to propose per speculation round.
pub const DRAFT_K: usize = 4;

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
    /// DSpark-like: online n-gram over the full running sequence (prompt +
    /// generated), so free-form output is accelerated too.
    NgramContext,
}

impl SpecMode {
    /// Parse a config string (case-insensitive). Unknown values fall back to
    /// the default and are reported by the caller.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "false" => Some(Self::Off),
            "prompt-lookup" | "prompt_lookup" | "lookup" | "true" => Some(Self::PromptLookup),
            "ngram-context" | "ngram_context" | "context" | "dspark" => Some(Self::NgramContext),
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

/// DSpark-like self-speculative drafter.
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

/// Adaptive draft-length controller — DSpark's load-aware scheduler analogue.
///
/// Tracks an exponential moving average of the per-round acceptance ratio and
/// grows the draft length when speculation is paying off, shrinks it when
/// drafts are mostly rejected (so a bad streak costs at most `min` wasted
/// verify slots rather than a fixed large block).
#[derive(Debug, Clone)]
pub struct AdaptiveK {
    k: usize,
    min: usize,
    max: usize,
    ema: f32,
}

impl AdaptiveK {
    /// EMA smoothing factor (higher = more reactive).
    const ALPHA: f32 = 0.3;
    /// Grow K above this acceptance ratio, shrink below `1 - GROW`.
    const GROW: f32 = 0.6;

    pub fn new(initial: usize, min: usize, max: usize) -> Self {
        let min = min.max(1);
        let max = max.max(min);
        Self {
            k: initial.clamp(min, max),
            min,
            max,
            ema: 0.5,
        }
    }

    /// Current draft length to propose.
    pub fn current(&self) -> usize {
        self.k
    }

    /// Update after a verify round: `accepted` of `drafted` tokens kept.
    pub fn update(&mut self, accepted: usize, drafted: usize) {
        if drafted == 0 {
            return;
        }
        let ratio = accepted as f32 / drafted as f32;
        self.ema = Self::ALPHA * ratio + (1.0 - Self::ALPHA) * self.ema;
        if self.ema > Self::GROW && self.k < self.max {
            self.k += 1;
        } else if self.ema < 1.0 - Self::GROW && self.k > self.min {
            self.k -= 1;
        }
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
        assert_eq!(SpecMode::parse("dspark"), Some(SpecMode::NgramContext));
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
