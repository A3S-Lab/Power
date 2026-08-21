/// Bounded suffix state for exact stop-sequence matching.
///
/// A stop match can only end at the current streamed token boundary, so the
/// full generated response never needs to be retained or cloned. Keeping one
/// maximum-length UTF-8 suffix preserves the existing `ends_with` semantics
/// while making preview cloning independent of response length.
#[derive(Debug, Clone)]
pub(super) struct StopSequenceTracker {
    suffix: String,
    max_sequence_bytes: usize,
}

impl StopSequenceTracker {
    pub(super) fn new(stop_sequences: &[String]) -> Self {
        Self {
            suffix: String::new(),
            max_sequence_bytes: stop_sequences.iter().map(String::len).max().unwrap_or(0),
        }
    }

    /// Append one decoded token piece and report whether a stop sequence now
    /// terminates the generated text.
    pub(super) fn push(&mut self, text: &str, stop_sequences: &[String]) -> bool {
        if stop_sequences.is_empty() {
            return false;
        }
        self.suffix.push_str(text);
        if stop_sequences
            .iter()
            .any(|stop| self.suffix.ends_with(stop))
        {
            return true;
        }
        self.trim_to_match_window();
        false
    }

    fn trim_to_match_window(&mut self) {
        if self.max_sequence_bytes == 0 {
            self.suffix.clear();
            return;
        }
        if self.suffix.len() <= self.max_sequence_bytes {
            return;
        }

        let mut start = self.suffix.len() - self.max_sequence_bytes;
        while start > 0 && !self.suffix.is_char_boundary(start) {
            start -= 1;
        }
        self.suffix.drain(..start);
    }

    #[cfg(test)]
    fn retained_bytes(&self) -> usize {
        self.suffix.len()
    }
}

#[cfg(test)]
mod tests {
    use super::StopSequenceTracker;

    #[test]
    fn detects_a_stop_split_across_token_pieces() {
        let stops = vec!["</answer>".to_string()];
        let mut tracker = StopSequenceTracker::new(&stops);

        assert!(!tracker.push("prefix </ans", &stops));
        assert!(tracker.push("wer>", &stops));
    }

    #[test]
    fn preview_clone_does_not_advance_stream_state() {
        let stops = vec!["END".to_string()];
        let mut tracker = StopSequenceTracker::new(&stops);
        assert!(!tracker.push("prefix E", &stops));

        let mut preview = tracker.clone();
        assert!(preview.push("ND", &stops));
        assert!(!tracker.push("x", &stops));
    }

    #[test]
    fn retains_only_a_bounded_ascii_suffix() {
        let stops = vec!["STOP".to_string()];
        let mut tracker = StopSequenceTracker::new(&stops);

        assert!(!tracker.push(&"x".repeat(16_384), &stops));
        assert!(tracker.retained_bytes() <= 4);
    }

    #[test]
    fn preserves_utf8_boundaries_needed_by_a_future_match() {
        let stops = vec!["\u{1f642}END".to_string()];
        let mut tracker = StopSequenceTracker::new(&stops);

        assert!(!tracker.push("long-prefix-\u{1f642}", &stops));
        assert!(tracker.push("END", &stops));
    }

    #[test]
    fn empty_stop_sequence_keeps_string_ends_with_semantics() {
        let stops = vec![String::new()];
        let mut tracker = StopSequenceTracker::new(&stops);

        assert!(tracker.push("token", &stops));
    }

    #[test]
    fn absent_stop_sequences_do_not_retain_token_text() {
        let mut tracker = StopSequenceTracker::new(&[]);

        assert!(!tracker.push(&"x".repeat(16_384), &[]));
        assert_eq!(tracker.retained_bytes(), 0);
    }
}
