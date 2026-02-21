//! BPE tokenizer loaded from GGUF metadata.
//!
//! Supports two tokenizer families:
//! - **SentencePiece BPE** (LLaMA): uses `▁` (U+2581) for space, merge scores
//! - **GPT-style BPE** (Qwen, GPT-2): uses `Ġ` for space, `Ċ` for newline,
//!   greedy longest-match when scores are absent

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
    /// Whether this is a GPT-style tokenizer (Ġ prefix, no scores)
    gpt_style: bool,
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

        // Detect GPT-style tokenizer: scores are empty and vocab contains Ġ-prefixed tokens
        let has_scores = !scores.is_empty() && scores.iter().any(|&s| s != 0.0);
        let has_gpt_prefix = tokens.iter().any(|t| t.starts_with('Ġ'));
        let gpt_style = !has_scores && has_gpt_prefix;

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
            gpt_style,
            bos_id,
            eos_id,
        }
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.gpt_style {
            self.encode_gpt(text)
        } else {
            self.encode_sentencepiece(text)
        }
    }

    /// GPT-style encoding: convert text to Ġ/Ċ representation, then greedy longest-match.
    fn encode_gpt(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();

        // Don't add BOS for GPT-style — the chat template handles special tokens
        // Split text into segments: special tokens vs normal text
        let segments = split_special_tokens(text, &self.piece_to_id);

        for segment in segments {
            match segment {
                Segment::Special(id) => ids.push(id),
                Segment::Text(s) => {
                    // Convert to GPT byte representation
                    let encoded = to_gpt_bytes(&s);
                    self.greedy_longest_match(&encoded, &mut ids);
                }
            }
        }

        ids
    }

    /// Greedy longest-match tokenization.
    fn greedy_longest_match(&self, text: &str, ids: &mut Vec<u32>) {
        let chars: Vec<char> = text.chars().collect();
        let mut pos = 0;

        while pos < chars.len() {
            let mut best_len = 0;
            let mut best_id = 0u32;

            // Try progressively longer substrings
            let max_len = (chars.len() - pos).min(64); // cap search length
            for end in 1..=max_len {
                let candidate: String = chars[pos..pos + end].iter().collect();
                if let Some(&id) = self.piece_to_id.get(&candidate) {
                    best_len = end;
                    best_id = id;
                }
            }

            if best_len > 0 {
                ids.push(best_id);
                pos += best_len;
            } else {
                // Single character fallback — try byte tokens
                let ch = chars[pos];
                let mut found = false;
                for byte in ch.to_string().as_bytes() {
                    if let Some(id) = self.byte_tokens[*byte as usize] {
                        ids.push(id);
                        found = true;
                    }
                }
                if !found {
                    // Skip unknown character
                }
                pos += 1;
            }
        }
    }

    /// SentencePiece-style BPE encoding with merge scores.
    fn encode_sentencepiece(&self, text: &str) -> Vec<u32> {
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
                break;
            }

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

        // Skip special/control tokens
        if piece.starts_with("<|") && piece.ends_with("|>") {
            return Some(String::new());
        }

        // Handle byte tokens: <0xHH> → single byte
        if let Some(byte_val) = parse_byte_token(piece) {
            return Some(String::from_utf8_lossy(&[byte_val]).into_owned());
        }

        if self.gpt_style {
            // GPT-style: Ġ → space, Ċ → newline
            Some(from_gpt_bytes(piece))
        } else {
            // SentencePiece: ▁ (U+2581) → space
            Some(piece.replace('\u{2581}', " "))
        }
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

// ── GPT byte encoding ────────────────────────────────────────────────────────

/// Convert text to GPT-style byte representation.
/// Space → Ġ (U+0120), newline → Ċ (U+010A), tab → ĉ (U+0109), etc.
fn to_gpt_bytes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        out.push(gpt_byte_encode(byte));
    }
    out
}

/// Convert GPT-style byte representation back to text.
fn from_gpt_bytes(piece: &str) -> String {
    let mut out = Vec::new();
    for ch in piece.chars() {
        if let Some(byte) = gpt_byte_decode(ch) {
            out.push(byte);
        } else {
            // Multi-byte UTF-8 char, pass through
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            out.extend_from_slice(s.as_bytes());
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// GPT-2 byte-to-unicode mapping for printable range.
/// Maps bytes 0..255 to Unicode codepoints, avoiding control characters.
fn gpt_byte_encode(byte: u8) -> char {
    // The GPT-2 byte encoder maps:
    // 33..=126 → same codepoint (printable ASCII)
    // 161..=172 → same codepoint
    // 174..=255 → same codepoint
    // Everything else → 256 + byte
    match byte {
        b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF => byte as char,
        _ => char::from_u32(256 + byte as u32).unwrap_or('?'),
    }
}

/// Reverse of gpt_byte_encode.
fn gpt_byte_decode(ch: char) -> Option<u8> {
    let cp = ch as u32;
    match cp {
        0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => Some(cp as u8),
        256..=511 => Some((cp - 256) as u8),
        _ => None,
    }
}

// ── Special token splitting ──────────────────────────────────────────────────

enum Segment {
    Special(u32),
    Text(String),
}

/// Split text into segments of special tokens and normal text.
fn split_special_tokens(text: &str, piece_to_id: &HashMap<String, u32>) -> Vec<Segment> {
    // Collect special tokens (those with <|...|> pattern)
    let mut specials: Vec<(&str, u32)> = piece_to_id
        .iter()
        .filter(|(k, _)| k.starts_with("<|") && k.ends_with("|>"))
        .map(|(k, &v)| (k.as_str(), v))
        .collect();
    // Sort by length descending for greedy matching
    specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let mut segments = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        // Try to match a special token at the current position
        let mut found = false;
        for &(token_str, token_id) in &specials {
            if remaining.starts_with(token_str) {
                segments.push(Segment::Special(token_id));
                remaining = &remaining[token_str.len()..];
                found = true;
                break;
            }
        }
        if !found {
            // Find the next special token occurrence
            let mut next_special_pos = remaining.len();
            for &(token_str, _) in &specials {
                if let Some(pos) = remaining.find(token_str) {
                    if pos < next_special_pos {
                        next_special_pos = pos;
                    }
                }
            }
            // Everything before the next special token is normal text
            let normal = &remaining[..next_special_pos];
            if !normal.is_empty() {
                segments.push(Segment::Text(normal.to_string()));
            }
            remaining = &remaining[next_special_pos..];
        }
    }

    segments
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
        let scores = vec![
            0.0, 0.0, 0.0, -100.0, -100.0, -50.0, -3.0, -10.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
            -5.0, 0.0, -1.0, -2.0, -2.5,
        ];
        let types = vec![0i32; tokens.len()];
        BpeTokenizer::from_gguf(&tokens, &scores, &types, 1, 2)
    }

    fn make_gpt_tokenizer() -> BpeTokenizer {
        // GPT-style vocab (Ġ prefix for space)
        let tokens = vec![
            "!".to_string(),            // 0
            "Ġ".to_string(),            // 1 (space)
            "Ċ".to_string(),            // 2 (newline)
            "Hello".to_string(),        // 3
            "ĠWorld".to_string(),       // 4
            "Ġis".to_string(),          // 5
            "2".to_string(),            // 6
            "+".to_string(),            // 7
            "?".to_string(),            // 8
            "<|im_start|>".to_string(), // 9
            "<|im_end|>".to_string(),   // 10
            "user".to_string(),         // 11
            "What".to_string(),         // 12
            "ĠWhat".to_string(),        // 13
        ];
        // No scores → GPT-style
        let scores = vec![];
        let types = vec![1i32; tokens.len()];
        BpeTokenizer::from_gguf(&tokens, &scores, &types, 9, 10)
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

    // ── GPT-style tests ──

    #[test]
    fn test_gpt_style_detected() {
        let tok = make_gpt_tokenizer();
        assert!(tok.gpt_style);
    }

    #[test]
    fn test_gpt_encode_special_tokens() {
        let tok = make_gpt_tokenizer();
        let ids = tok.encode("<|im_start|>user");
        assert_eq!(ids[0], 9); // <|im_start|>
        assert_eq!(ids[1], 11); // user
    }

    #[test]
    fn test_gpt_decode_space() {
        let tok = make_gpt_tokenizer();
        assert_eq!(tok.decode(4).unwrap(), " World");
    }

    #[test]
    fn test_gpt_byte_roundtrip() {
        // Space → Ġ → space
        assert_eq!(gpt_byte_encode(b' '), 'Ġ');
        assert_eq!(gpt_byte_decode('Ġ'), Some(b' '));
        // Newline → Ċ → newline
        assert_eq!(gpt_byte_encode(b'\n'), 'Ċ');
        assert_eq!(gpt_byte_decode('Ċ'), Some(b'\n'));
        // Printable ASCII passes through
        assert_eq!(gpt_byte_encode(b'A'), 'A');
        assert_eq!(gpt_byte_decode('A'), Some(b'A'));
    }

    #[test]
    fn test_gpt_decode_skips_special() {
        let tok = make_gpt_tokenizer();
        // Special tokens decode to empty string
        assert_eq!(tok.decode(9).unwrap(), "");
    }
}
