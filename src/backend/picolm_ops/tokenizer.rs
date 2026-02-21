//! BPE tokenizer loaded from GGUF metadata.
//!
//! LLaMA uses SentencePiece BPE. The vocabulary, merge scores, and token
//! types are stored in GGUF metadata keys `tokenizer.ggml.*`.

use std::collections::HashMap;

/// BPE tokenizer loaded from GGUF metadata.
pub struct BpeTokenizer {
    /// Token ID → string piece
    vocab: Vec<String>,
    /// Token ID → merge priority score (lower = merge first)
    scores: Vec<f32>,
    /// String piece → token ID (for encoding)
    piece_to_id: HashMap<String, u32>,
    /// Byte fallback tokens: byte value → token ID
    byte_tokens: [Option<u32>; 256],
    pub bos_id: u32,
    pub eos_id: u32,
}

impl BpeTokenizer {
    /// Build a tokenizer from GGUF metadata arrays.
    pub fn from_gguf(
        tokens: &[String],
        scores: &[f32],
        token_types: &[i32],
        bos_id: u32,
        eos_id: u32,
    ) -> Self {
        let mut piece_to_id = HashMap::with_capacity(tokens.len());
        let mut byte_tokens = [None; 256];

        for (id, token) in tokens.iter().enumerate() {
            piece_to_id.insert(token.clone(), id as u32);

            // Detect byte fallback tokens: <0xHH>
            if token_types.get(id).copied() == Some(6) || is_byte_token(token) {
                if let Some(byte_val) = parse_byte_token(token) {
                    byte_tokens[byte_val as usize] = Some(id as u32);
                }
            }
        }

        // Pad scores if shorter than vocab
        let mut scores_vec = scores.to_vec();
        scores_vec.resize(tokens.len(), 0.0);

        Self {
            vocab: tokens.to_vec(),
            scores: scores_vec,
            piece_to_id,
            byte_tokens,
            bos_id,
            eos_id,
        }
    }

    /// Encode text into token IDs using SentencePiece-style BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![self.bos_id];

        if text.is_empty() {
            return ids;
        }

        // SentencePiece convention: prepend space
        let text = format!(" {text}");

        // Initialize: try to match each character as a single-char token,
        // fall back to byte tokens for unknown characters.
        let mut tokens: Vec<u32> = Vec::new();
        for ch in text.chars() {
            let s = ch.to_string();
            // SentencePiece uses ▁ (U+2581) for space
            let sp_s = s.replace(' ', "\u{2581}");
            if let Some(&id) = self.piece_to_id.get(&sp_s) {
                tokens.push(id);
            } else if let Some(&id) = self.piece_to_id.get(&s) {
                tokens.push(id);
            } else {
                // Byte fallback: encode each UTF-8 byte
                for byte in ch.to_string().as_bytes() {
                    if let Some(id) = self.byte_tokens[*byte as usize] {
                        tokens.push(id);
                    }
                    // Skip bytes with no fallback token (shouldn't happen with proper vocab)
                }
            }
        }

        // BPE merge loop: repeatedly merge the highest-priority adjacent pair
        loop {
            if tokens.len() < 2 {
                break;
            }

            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;
            let mut best_id = 0u32;

            for i in 0..tokens.len() - 1 {
                let left = &self.vocab[tokens[i] as usize];
                let right = &self.vocab[tokens[i + 1] as usize];
                let merged = format!("{left}{right}");
                if let Some(&merged_id) = self.piece_to_id.get(&merged) {
                    let score = self.scores[merged_id as usize];
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                        best_id = merged_id;
                    }
                }
            }

            if best_idx == usize::MAX {
                break; // No more merges possible
            }

            // Apply merge
            tokens[best_idx] = best_id;
            tokens.remove(best_idx + 1);
        }

        ids.extend_from_slice(&tokens);
        ids
    }

    /// Decode a single token ID to its string piece.
    /// Returns `None` for EOS token.
    pub fn decode(&self, token_id: u32) -> Option<String> {
        if token_id == self.eos_id {
            return None;
        }
        if token_id as usize >= self.vocab.len() {
            return None;
        }
        let piece = &self.vocab[token_id as usize];

        // Handle byte tokens: <0xHH> → single byte
        if let Some(byte_val) = parse_byte_token(piece) {
            return Some(String::from_utf8_lossy(&[byte_val]).into_owned());
        }

        // Handle SentencePiece space: ▁ (U+2581) → ' '
        Some(piece.replace('\u{2581}', " "))
    }

    /// Decode a sequence of token IDs into a string.
    pub fn decode_all(&self, token_ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in token_ids {
            if let Some(piece) = self.decode(id) {
                out.push_str(&piece);
            }
        }
        out
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Check if a token string looks like a byte fallback token: `<0xHH>`
fn is_byte_token(s: &str) -> bool {
    s.len() == 6 && s.starts_with("<0x") && s.ends_with('>')
}

/// Parse a byte fallback token `<0xHH>` into its byte value.
fn parse_byte_token(s: &str) -> Option<u8> {
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        u8::from_str_radix(&s[3..5], 16).ok()
    } else {
        None
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tokenizer() -> BpeTokenizer {
        // Minimal vocab with merge chain: h+e→he, he+l→hel, hel+lo→hello, ▁+hello→▁hello
        let tokens = vec![
            "<unk>".to_string(),         // 0
            "<s>".to_string(),           // 1 (BOS)
            "</s>".to_string(),          // 2 (EOS)
            "\u{2581}hello".to_string(), // 3
            "\u{2581}world".to_string(), // 4
            "\u{2581}hel".to_string(),   // 5
            "lo".to_string(),            // 6
            "\u{2581}".to_string(),      // 7 (space)
            "h".to_string(),             // 8
            "e".to_string(),             // 9
            "l".to_string(),             // 10
            "o".to_string(),             // 11
            "w".to_string(),             // 12
            "r".to_string(),             // 13
            "d".to_string(),             // 14
            "<0x41>".to_string(),        // 15 (byte 'A')
            "he".to_string(),            // 16
            "hel".to_string(),           // 17
            "hello".to_string(),         // 18
        ];
        // Higher score = merge first in SentencePiece BPE
        let scores = vec![
            0.0, 0.0, 0.0,
            -100.0, // 3: "▁hello" — final merge (lowest priority)
            -100.0, // 4: "▁world"
            -50.0,  // 5: "▁hel"
            -3.0,   // 6: "lo"
            -10.0,  // 7: "▁"
            -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
            0.0,    // 15: byte token
            -1.0,   // 16: "he" — merge first (highest score)
            -2.0,   // 17: "hel"
            -2.5,   // 18: "hello"
        ];
        let types = vec![0i32; tokens.len()];
        BpeTokenizer::from_gguf(&tokens, &scores, &types, 1, 2)
    }

    #[test]
    fn test_decode_eos_is_none() {
        let tok = make_test_tokenizer();
        assert!(tok.decode(2).is_none());
    }

    #[test]
    fn test_decode_normal_token() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.decode(3).unwrap(), " hello");
    }

    #[test]
    fn test_decode_byte_token() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.decode(15).unwrap(), "A");
    }

    #[test]
    fn test_encode_starts_with_bos() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("hello");
        assert_eq!(ids[0], 1); // BOS
    }

    #[test]
    fn test_encode_merges() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("hello");
        // Should merge to "▁hello" (id=3) since it has the best score
        assert!(ids.contains(&3), "expected token 3 (▁hello), got {ids:?}");
    }

    #[test]
    fn test_encode_empty() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("");
        assert_eq!(ids, vec![1]); // Just BOS
    }

    #[test]
    fn test_vocab_size() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.vocab_size(), 19);
    }

    #[test]
    fn test_parse_byte_token() {
        assert_eq!(parse_byte_token("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_token("<0x00>"), Some(0x00));
        assert_eq!(parse_byte_token("hello"), None);
        assert_eq!(parse_byte_token("<0x>"), None);
    }
}
