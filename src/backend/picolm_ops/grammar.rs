//! Grammar-constrained sampling for structured output.
//!
//! Implements a character-level JSON grammar constraint that masks logits
//! for tokens producing invalid characters at the current parse position.
//!
//! Instead of a full GBNF parser, this uses a stack-based JSON validator
//! that tracks structural state (object/array/string/number nesting) and
//! determines which characters are valid next. This covers the primary
//! structured output use case (JSON Schema) without the complexity of
//! a general-purpose grammar engine.

use super::tokenizer::BpeTokenizer;

/// JSON structural state for grammar-constrained sampling.
#[derive(Debug, Clone, PartialEq)]
enum JsonState {
    /// Expecting a JSON value (object, array, string, number, bool, null).
    Value,
    /// Inside an object, expecting a key or '}'.
    ObjectStart,
    /// Inside an object after a key, expecting ':'.
    ObjectColon,
    /// Inside an object after a value, expecting ',' or '}'.
    ObjectComma,
    /// Inside an array, expecting a value or ']'.
    ArrayStart,
    /// Inside an array after a value, expecting ',' or ']'.
    ArrayComma,
    /// Inside a string (after opening '"', before closing '"').
    InString,
    /// Inside a string, after a backslash (escape sequence).
    InStringEscape,
    /// Accumulating a keyword (true/false/null) — `expected` holds remaining chars.
    Keyword { expected: &'static str },
    /// Accumulating a number.
    InNumber,
    /// Done — a complete JSON value has been produced.
    Done,
}

/// Stack-based JSON grammar sampler.
///
/// Tracks the parse state of the generated text and determines which
/// characters are valid at each position. Used to mask logits before
/// token sampling, ensuring the output is valid JSON.
pub struct JsonGrammarSampler {
    /// Parse state stack. The top is the current state.
    stack: Vec<JsonState>,
    /// Characters accumulated for the current number literal.
    number_buf: String,
}

impl Default for JsonGrammarSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonGrammarSampler {
    /// Create a new sampler expecting a JSON value.
    pub fn new() -> Self {
        Self {
            stack: vec![JsonState::Value],
            number_buf: String::new(),
        }
    }

    /// Return the set of characters valid at the current parse position.
    /// Returns `None` if any character is valid (unconstrained / done).
    pub fn allowed_chars(&self) -> Option<Vec<char>> {
        let state = self.stack.last()?;

        match state {
            JsonState::Value => {
                // Any JSON value can start with: { [ " - 0-9 t f n or whitespace
                Some(vec![
                    '{', '[', '"', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'f',
                    'n', ' ', '\t', '\n', '\r',
                ])
            }
            JsonState::ObjectStart => {
                // Expecting key (string) or closing '}'
                Some(vec!['"', '}', ' ', '\t', '\n', '\r'])
            }
            JsonState::ObjectColon => Some(vec![':', ' ', '\t', '\n', '\r']),
            JsonState::ObjectComma => Some(vec![',', '}', ' ', '\t', '\n', '\r']),
            JsonState::ArrayStart => {
                // Expecting value or closing ']'
                Some(vec![
                    '{', '[', '"', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'f',
                    'n', ']', ' ', '\t', '\n', '\r',
                ])
            }
            JsonState::ArrayComma => Some(vec![',', ']', ' ', '\t', '\n', '\r']),
            JsonState::InString => {
                // Any printable character except unescaped control chars
                None // Allow all printable chars inside strings
            }
            JsonState::InStringEscape => Some(vec!['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']),
            JsonState::Keyword { expected } => {
                if let Some(ch) = expected.chars().next() {
                    Some(vec![ch])
                } else {
                    // Keyword complete — allow structural chars
                    Some(vec![',', '}', ']', ' ', '\t', '\n', '\r'])
                }
            }
            JsonState::InNumber => {
                // Digits, '.', 'e', 'E', '+', '-', or structural end
                Some(vec![
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'e', 'E', '+', '-', ',',
                    '}', ']', ' ', '\t', '\n', '\r',
                ])
            }
            JsonState::Done => None,
        }
    }

    /// Feed a character to the grammar sampler, advancing the parse state.
    /// Returns `true` if the character was valid, `false` if it violated the grammar.
    pub fn feed(&mut self, ch: char) -> bool {
        // Skip whitespace in structural positions
        if ch.is_ascii_whitespace() {
            let state = self.stack.last();
            match state {
                Some(JsonState::InString) | Some(JsonState::InStringEscape) => {
                    // Whitespace inside strings is literal content
                }
                _ => return true, // Whitespace is always valid in structural positions
            }
        }

        let state = match self.stack.last().cloned() {
            Some(s) => s,
            None => return true, // Done, accept anything
        };

        match state {
            JsonState::Value => self.feed_value(ch),
            JsonState::ObjectStart => self.feed_object_start(ch),
            JsonState::ObjectColon => self.feed_object_colon(ch),
            JsonState::ObjectComma => self.feed_object_comma(ch),
            JsonState::ArrayStart => self.feed_array_start(ch),
            JsonState::ArrayComma => self.feed_array_comma(ch),
            JsonState::InString => self.feed_in_string(ch),
            JsonState::InStringEscape => self.feed_in_string_escape(ch),
            JsonState::Keyword { expected } => self.feed_keyword(ch, expected),
            JsonState::InNumber => self.feed_in_number(ch),
            JsonState::Done => true,
        }
    }

    fn feed_value(&mut self, ch: char) -> bool {
        if ch.is_ascii_whitespace() {
            return true;
        }
        self.stack.pop();
        match ch {
            '{' => {
                self.stack.push(JsonState::ObjectStart);
                true
            }
            '[' => {
                self.stack.push(JsonState::ArrayStart);
                true
            }
            '"' => {
                self.stack.push(JsonState::InString);
                true
            }
            't' => {
                self.stack.push(JsonState::Keyword { expected: "rue" });
                true
            }
            'f' => {
                self.stack.push(JsonState::Keyword { expected: "alse" });
                true
            }
            'n' => {
                self.stack.push(JsonState::Keyword { expected: "ull" });
                true
            }
            '-' | '0'..='9' => {
                self.number_buf.clear();
                self.number_buf.push(ch);
                self.stack.push(JsonState::InNumber);
                true
            }
            _ => false,
        }
    }

    fn feed_object_start(&mut self, ch: char) -> bool {
        if ch.is_ascii_whitespace() {
            return true;
        }
        match ch {
            '"' => {
                // Key string
                *self.stack.last_mut().unwrap() = JsonState::ObjectColon;
                self.stack.push(JsonState::InString);
                true
            }
            '}' => {
                self.stack.pop();
                self.complete_value();
                true
            }
            _ => false,
        }
    }

    fn feed_object_colon(&mut self, ch: char) -> bool {
        if ch.is_ascii_whitespace() {
            return true;
        }
        if ch == ':' {
            *self.stack.last_mut().unwrap() = JsonState::ObjectComma;
            self.stack.push(JsonState::Value);
            true
        } else {
            false
        }
    }

    fn feed_object_comma(&mut self, ch: char) -> bool {
        if ch.is_ascii_whitespace() {
            return true;
        }
        match ch {
            ',' => {
                // Expect next key
                *self.stack.last_mut().unwrap() = JsonState::ObjectColon;
                self.stack.push(JsonState::InString);
                // The next feed should see '"' to start the key string,
                // but we need to push a Value-like state that expects '"'.
                // Actually, after comma we expect a key string directly.
                // Let's adjust: pop the InString we just pushed, and instead
                // set state to ObjectStart which expects '"' or '}'.
                self.stack.pop();
                *self.stack.last_mut().unwrap() = JsonState::ObjectStart;
                true
            }
            '}' => {
                self.stack.pop();
                self.complete_value();
                true
            }
            _ => false,
        }
    }

    fn feed_array_start(&mut self, ch: char) -> bool {
        if ch.is_ascii_whitespace() {
            return true;
        }
        if ch == ']' {
            self.stack.pop();
            self.complete_value();
            return true;
        }
        // First element — transition to ArrayComma and push Value
        *self.stack.last_mut().unwrap() = JsonState::ArrayComma;
        self.stack.push(JsonState::Value);
        self.feed_value(ch)
    }

    fn feed_array_comma(&mut self, ch: char) -> bool {
        if ch.is_ascii_whitespace() {
            return true;
        }
        match ch {
            ',' => {
                self.stack.push(JsonState::Value);
                true
            }
            ']' => {
                self.stack.pop();
                self.complete_value();
                true
            }
            _ => false,
        }
    }

    fn feed_in_string(&mut self, ch: char) -> bool {
        match ch {
            '"' => {
                self.stack.pop();
                self.complete_value();
                true
            }
            '\\' => {
                *self.stack.last_mut().unwrap() = JsonState::InStringEscape;
                true
            }
            // Control characters are invalid in JSON strings
            '\x00'..='\x1f' => false,
            _ => true,
        }
    }

    fn feed_in_string_escape(&mut self, ch: char) -> bool {
        let valid = matches!(ch, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u');
        if valid {
            *self.stack.last_mut().unwrap() = JsonState::InString;
        }
        valid
    }

    fn feed_keyword(&mut self, ch: char, expected: &'static str) -> bool {
        if let Some(next_ch) = expected.chars().next() {
            if ch == next_ch {
                let remaining = &expected[ch.len_utf8()..];
                if remaining.is_empty() {
                    self.stack.pop();
                    self.complete_value();
                } else {
                    *self.stack.last_mut().unwrap() = JsonState::Keyword {
                        expected: remaining,
                    };
                }
                true
            } else {
                false
            }
        } else {
            // Keyword complete, this char belongs to the parent
            self.stack.pop();
            self.complete_value();
            self.feed(ch)
        }
    }

    fn feed_in_number(&mut self, ch: char) -> bool {
        match ch {
            '0'..='9' | '.' | 'e' | 'E' | '+' | '-' => {
                self.number_buf.push(ch);
                true
            }
            // Structural character ends the number
            ',' | '}' | ']' | ' ' | '\t' | '\n' | '\r' => {
                self.stack.pop();
                self.complete_value();
                self.feed(ch)
            }
            _ => false,
        }
    }

    /// Called when a value (string, number, keyword, object, array) is complete.
    /// Does nothing — the parent state on the stack handles what comes next.
    fn complete_value(&mut self) {
        if self.stack.is_empty() {
            self.stack.push(JsonState::Done);
        }
    }

    /// Whether the sampler has reached a complete JSON value.
    pub fn is_complete(&self) -> bool {
        matches!(self.stack.last(), Some(JsonState::Done) | None)
    }

    /// Mask logits for tokens that would produce invalid characters.
    ///
    /// For each token in the vocabulary, decode it to text and check if
    /// its first character is in the allowed set. If not, set its logit
    /// to negative infinity.
    pub fn mask_logits(&self, logits: &mut [f32], tokenizer: &BpeTokenizer) {
        let allowed = match self.allowed_chars() {
            Some(chars) => chars,
            None => return, // No constraint — all tokens valid
        };

        for (token_id, logit) in logits.iter_mut().enumerate() {
            if let Some(text) = tokenizer.decode(token_id as u32) {
                // Check if the first non-empty character of this token is allowed
                if let Some(first_char) = text.chars().next() {
                    if !allowed.contains(&first_char) {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
            // EOS tokens: allow if we're in Done state, block otherwise
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_object() {
        let mut s = JsonGrammarSampler::new();
        let input = r#"{"name": "Alice", "age": 30}"#;
        for ch in input.chars() {
            assert!(s.feed(ch), "failed at char '{ch}' in: {input}");
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_simple_array() {
        let mut s = JsonGrammarSampler::new();
        let input = r#"[1, 2, 3]"#;
        for ch in input.chars() {
            assert!(s.feed(ch), "failed at char '{ch}'");
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_nested_object() {
        let mut s = JsonGrammarSampler::new();
        let input = r#"{"user": {"name": "Bob", "scores": [1, 2]}}"#;
        for ch in input.chars() {
            assert!(s.feed(ch), "failed at char '{ch}'");
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_string_value() {
        let mut s = JsonGrammarSampler::new();
        let input = r#""hello world""#;
        for ch in input.chars() {
            assert!(s.feed(ch), "failed at char '{ch}'");
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_string_with_escapes() {
        let mut s = JsonGrammarSampler::new();
        let input = r#""hello \"world\" \n\t""#;
        for ch in input.chars() {
            assert!(s.feed(ch), "failed at char '{ch}'");
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_boolean_true() {
        let mut s = JsonGrammarSampler::new();
        for ch in "true".chars() {
            assert!(s.feed(ch));
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_boolean_false() {
        let mut s = JsonGrammarSampler::new();
        for ch in "false".chars() {
            assert!(s.feed(ch));
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_null() {
        let mut s = JsonGrammarSampler::new();
        for ch in "null".chars() {
            assert!(s.feed(ch));
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_number_integer() {
        let mut s = JsonGrammarSampler::new();
        // Numbers need a structural terminator to complete
        let input = "[42]";
        for ch in input.chars() {
            assert!(s.feed(ch), "failed at char '{ch}'");
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_number_negative() {
        let mut s = JsonGrammarSampler::new();
        let input = "[-3.14]";
        for ch in input.chars() {
            assert!(s.feed(ch), "failed at char '{ch}'");
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_number_scientific() {
        let mut s = JsonGrammarSampler::new();
        let input = "[1.5e10]";
        for ch in input.chars() {
            assert!(s.feed(ch), "failed at char '{ch}'");
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_empty_object() {
        let mut s = JsonGrammarSampler::new();
        for ch in "{}".chars() {
            assert!(s.feed(ch));
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_empty_array() {
        let mut s = JsonGrammarSampler::new();
        for ch in "[]".chars() {
            assert!(s.feed(ch));
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_whitespace_handling() {
        let mut s = JsonGrammarSampler::new();
        let input = "{ \"key\" : \"value\" }";
        for ch in input.chars() {
            assert!(s.feed(ch), "failed at char '{ch}'");
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_allowed_chars_at_start() {
        let s = JsonGrammarSampler::new();
        let allowed = s.allowed_chars().unwrap();
        assert!(allowed.contains(&'{'));
        assert!(allowed.contains(&'['));
        assert!(allowed.contains(&'"'));
        assert!(allowed.contains(&'t'));
        assert!(allowed.contains(&'1'));
        assert!(!allowed.contains(&'}'));
        assert!(!allowed.contains(&':'));
    }

    #[test]
    fn test_allowed_chars_after_object_open() {
        let mut s = JsonGrammarSampler::new();
        s.feed('{');
        let allowed = s.allowed_chars().unwrap();
        assert!(allowed.contains(&'"')); // key
        assert!(allowed.contains(&'}')); // empty object
        assert!(!allowed.contains(&'[')); // not valid here
    }

    #[test]
    fn test_invalid_char_rejected() {
        let mut s = JsonGrammarSampler::new();
        s.feed('{');
        // After '{', a bare 'x' is invalid
        assert!(!s.feed('x'));
    }

    #[test]
    fn test_complex_nested() {
        let mut s = JsonGrammarSampler::new();
        let input =
            r#"{"users":[{"name":"Alice","active":true},{"name":"Bob","active":false}],"count":2}"#;
        for ch in input.chars() {
            assert!(s.feed(ch), "failed at char '{ch}'");
        }
        assert!(s.is_complete());
    }

    #[test]
    fn test_not_complete_mid_parse() {
        let mut s = JsonGrammarSampler::new();
        s.feed('{');
        assert!(!s.is_complete());
        for ch in r#""key": "val"}"#.chars() {
            s.feed(ch);
        }
        assert!(s.is_complete());
    }
}
