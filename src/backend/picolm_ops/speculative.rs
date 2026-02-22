//! Self-speculative decoding for picolm.
//!
//! Uses prompt-lookup decoding: matches n-grams from the generated text
//! against the input prompt to predict likely continuations. Draft tokens
//! are verified in batch through the full model.
//!
//! This approach has zero draft cost (no extra forward passes) and works
//! well for tasks where output overlaps with input (summarization, JSON
//! with known keys, code completion).
//!
//! # KV Cache Rollback
//!
//! Rejected draft tokens are rolled back via `KvCache::truncate()`.
//! The verify pass re-populates the KV cache with correct values.

/// Number of draft tokens to generate per speculation round.
pub const DRAFT_K: usize = 4;

/// Minimum n-gram size for prompt lookup matching.
const MIN_NGRAM: usize = 2;

/// Maximum n-gram size to try for prompt lookup matching.
const MAX_NGRAM: usize = 5;

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

/// Compare draft tokens against verified tokens and return the number accepted.
///
/// Uses greedy comparison: accepts the longest prefix where draft and verify
/// agree on the argmax token.
pub fn count_accepted(draft_tokens: &[u32], verify_logits: &[Vec<f32>]) -> usize {
    let mut accepted = 0;
    for (i, draft_tok) in draft_tokens.iter().enumerate() {
        if i >= verify_logits.len() {
            break;
        }
        let verify_argmax = argmax_token(&verify_logits[i]);
        if *draft_tok == verify_argmax {
            accepted += 1;
        } else {
            break;
        }
    }
    accepted
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
    fn test_count_accepted_all_match() {
        let draft = vec![1, 2, 3, 4];
        let verify = vec![
            logits_with_max(1, 10),
            logits_with_max(2, 10),
            logits_with_max(3, 10),
            logits_with_max(4, 10),
        ];
        assert_eq!(count_accepted(&draft, &verify), 4);
    }

    #[test]
    fn test_count_accepted_none_match() {
        let draft = vec![1, 2, 3, 4];
        let verify = vec![
            logits_with_max(5, 10),
            logits_with_max(2, 10),
            logits_with_max(3, 10),
            logits_with_max(4, 10),
        ];
        assert_eq!(count_accepted(&draft, &verify), 0);
    }

    #[test]
    fn test_count_accepted_partial() {
        let draft = vec![1, 2, 3, 4];
        let verify = vec![
            logits_with_max(1, 10),
            logits_with_max(2, 10),
            logits_with_max(9, 10),
            logits_with_max(4, 10),
        ];
        assert_eq!(count_accepted(&draft, &verify), 2);
    }

    #[test]
    fn test_count_accepted_empty() {
        assert_eq!(count_accepted(&[], &[]), 0);
    }

    #[test]
    fn test_count_accepted_fewer_verify() {
        let draft = vec![1, 2, 3, 4];
        let verify = vec![logits_with_max(1, 10), logits_with_max(2, 10)];
        assert_eq!(count_accepted(&draft, &verify), 2);
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

    /// Helper: create a logits vector where `token_id` has the max value.
    fn logits_with_max(token_id: usize, vocab: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; vocab];
        if token_id < vocab {
            logits[token_id] = 10.0;
        }
        logits
    }
}
