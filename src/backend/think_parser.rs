//! Streaming parser for `<think>...</think>` blocks emitted by reasoning models
//! (DeepSeek-R1, QwQ, etc.).
//!
//! Separates thinking/reasoning content from regular response content in a
//! token-by-token streaming pipeline. Handles partial tag matches across
//! token boundaries.
//!
//! # Usage
//! ```ignore
//! let mut parser = ThinkBlockParser::new();
//! for token in tokens {
//!     let (content, thinking) = parser.feed(&token);
//!     // content goes to the user-visible response
//!     // thinking goes to the reasoning/thinking field
//! }
//! let (content, thinking) = parser.flush();
//! ```

/// State machine states for think block detection.
#[derive(Debug, Clone, PartialEq)]
enum ThinkState {
    /// Normal content — pass through to content output.
    Normal,
    /// Accumulating characters that might be `<think>`.
    MaybeOpenTag,
    /// Inside a `<think>` block — content goes to thinking output.
    InsideThink,
    /// Inside a think block, accumulating characters that might be `</think>`.
    MaybeCloseTag,
}

const OPEN_TAG: &str = "<think>";
const CLOSE_TAG: &str = "</think>";

/// Streaming parser that separates `<think>...</think>` blocks from regular content.
#[derive(Debug)]
pub struct ThinkBlockParser {
    state: ThinkState,
    /// Buffer for accumulating partial tag matches.
    tag_buffer: String,
    /// Whether we have already seen and closed a think block.
    /// After the first complete think block, subsequent `<think>` tags are treated as literal text.
    seen_complete_block: bool,
}

impl Default for ThinkBlockParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ThinkBlockParser {
    /// Create a new parser in the initial state.
    pub fn new() -> Self {
        Self {
            state: ThinkState::Normal,
            tag_buffer: String::new(),
            seen_complete_block: false,
        }
    }

    /// Feed a token string into the parser.
    ///
    /// Returns `(content, thinking)` — the text to emit for each destination.
    /// Either or both may be empty.
    pub fn feed(&mut self, token: &str) -> (String, String) {
        let mut content = String::new();
        let mut thinking = String::new();

        for ch in token.chars() {
            match self.state {
                ThinkState::Normal => {
                    if ch == '<' && !self.seen_complete_block {
                        // Might be the start of <think>
                        self.tag_buffer.clear();
                        self.tag_buffer.push(ch);
                        self.state = ThinkState::MaybeOpenTag;
                    } else {
                        content.push(ch);
                    }
                }
                ThinkState::MaybeOpenTag => {
                    self.tag_buffer.push(ch);
                    if OPEN_TAG.starts_with(&self.tag_buffer) {
                        if self.tag_buffer == OPEN_TAG {
                            // Complete <think> tag matched
                            self.state = ThinkState::InsideThink;
                            self.tag_buffer.clear();
                        }
                        // else: still accumulating, keep going
                    } else {
                        // Not a match — flush buffer as content
                        content.push_str(&self.tag_buffer);
                        self.tag_buffer.clear();
                        self.state = ThinkState::Normal;
                    }
                }
                ThinkState::InsideThink => {
                    if ch == '<' {
                        // Might be the start of </think>
                        self.tag_buffer.clear();
                        self.tag_buffer.push(ch);
                        self.state = ThinkState::MaybeCloseTag;
                    } else {
                        thinking.push(ch);
                    }
                }
                ThinkState::MaybeCloseTag => {
                    self.tag_buffer.push(ch);
                    if CLOSE_TAG.starts_with(&self.tag_buffer) {
                        if self.tag_buffer == CLOSE_TAG {
                            // Complete </think> tag matched
                            self.state = ThinkState::Normal;
                            self.tag_buffer.clear();
                            self.seen_complete_block = true;
                        }
                        // else: still accumulating
                    } else {
                        // Not a match — flush buffer as thinking content
                        thinking.push_str(&self.tag_buffer);
                        self.tag_buffer.clear();
                        self.state = ThinkState::InsideThink;
                    }
                }
            }
        }

        (content, thinking)
    }

    /// Flush any remaining buffered content at the end of the stream.
    ///
    /// Returns `(content, thinking)` for the final output.
    pub fn flush(&mut self) -> (String, String) {
        let mut content = String::new();
        let mut thinking = String::new();

        match self.state {
            ThinkState::Normal => {
                // Nothing buffered
            }
            ThinkState::MaybeOpenTag => {
                // Incomplete open tag — emit as content
                content.push_str(&self.tag_buffer);
            }
            ThinkState::InsideThink => {
                // Unclosed think block — content stays as thinking
            }
            ThinkState::MaybeCloseTag => {
                // Incomplete close tag inside think — emit as thinking
                thinking.push_str(&self.tag_buffer);
            }
        }

        self.tag_buffer.clear();
        self.state = ThinkState::Normal;
        (content, thinking)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_think_block() {
        let mut parser = ThinkBlockParser::new();
        let (content, thinking) = parser.feed("Hello, world!");
        assert_eq!(content, "Hello, world!");
        assert_eq!(thinking, "");
        let (c, t) = parser.flush();
        assert_eq!(c, "");
        assert_eq!(t, "");
    }

    #[test]
    fn test_simple_think_block() {
        let mut parser = ThinkBlockParser::new();
        let (c, t) = parser.feed("<think>reasoning here</think>the answer");
        assert_eq!(t, "reasoning here");
        assert_eq!(c, "the answer");
    }

    #[test]
    fn test_think_block_across_tokens() {
        let mut parser = ThinkBlockParser::new();

        let (c1, t1) = parser.feed("<thi");
        assert_eq!(c1, "");
        assert_eq!(t1, "");

        let (c2, t2) = parser.feed("nk>");
        assert_eq!(c2, "");
        assert_eq!(t2, "");

        let (c3, t3) = parser.feed("I need to think");
        assert_eq!(c3, "");
        assert_eq!(t3, "I need to think");

        let (c4, t4) = parser.feed("</think>");
        assert_eq!(c4, "");
        assert_eq!(t4, "");

        let (c5, t5) = parser.feed("The answer is 42.");
        assert_eq!(c5, "The answer is 42.");
        assert_eq!(t5, "");
    }

    #[test]
    fn test_close_tag_across_tokens() {
        let mut parser = ThinkBlockParser::new();
        parser.feed("<think>");

        let (c1, t1) = parser.feed("reasoning</thi");
        assert_eq!(c1, "");
        assert_eq!(t1, "reasoning");

        let (c2, t2) = parser.feed("nk>answer");
        assert_eq!(c2, "answer");
        assert_eq!(t2, "");
    }

    #[test]
    fn test_empty_think_block() {
        let mut parser = ThinkBlockParser::new();
        let (c, t) = parser.feed("<think></think>answer");
        assert_eq!(c, "answer");
        assert_eq!(t, "");
    }

    #[test]
    fn test_think_block_at_end() {
        let mut parser = ThinkBlockParser::new();
        let (c, t) = parser.feed("prefix<think>reasoning</think>");
        assert_eq!(c, "prefix");
        assert_eq!(t, "reasoning");
    }

    #[test]
    fn test_only_thinking_no_content() {
        let mut parser = ThinkBlockParser::new();
        let (c, t) = parser.feed("<think>all reasoning</think>");
        assert_eq!(c, "");
        assert_eq!(t, "all reasoning");
        let (c2, t2) = parser.flush();
        assert_eq!(c2, "");
        assert_eq!(t2, "");
    }

    #[test]
    fn test_unclosed_think_block() {
        let mut parser = ThinkBlockParser::new();
        let (c, t) = parser.feed("<think>reasoning without close");
        assert_eq!(c, "");
        assert_eq!(t, "reasoning without close");
        let (c2, t2) = parser.flush();
        assert_eq!(c2, "");
        assert_eq!(t2, "");
    }

    #[test]
    fn test_partial_open_tag_not_matching() {
        let mut parser = ThinkBlockParser::new();
        let (c, t) = parser.feed("<this is not a think tag>");
        assert_eq!(c, "<this is not a think tag>");
        assert_eq!(t, "");
    }

    #[test]
    fn test_partial_open_tag_at_stream_end() {
        let mut parser = ThinkBlockParser::new();
        let (c, t) = parser.feed("hello<thi");
        assert_eq!(c, "hello");
        assert_eq!(t, "");
        let (c2, t2) = parser.flush();
        assert_eq!(c2, "<thi");
        assert_eq!(t2, "");
    }

    #[test]
    fn test_partial_close_tag_not_matching() {
        let mut parser = ThinkBlockParser::new();
        parser.feed("<think>");
        let (c, t) = parser.feed("text</other>more");
        assert_eq!(c, "");
        assert_eq!(t, "text</other>more");
    }

    #[test]
    fn test_partial_close_tag_at_stream_end() {
        let mut parser = ThinkBlockParser::new();
        parser.feed("<think>");
        let (c, t) = parser.feed("reasoning</thi");
        assert_eq!(c, "");
        assert_eq!(t, "reasoning");
        let (c2, t2) = parser.flush();
        assert_eq!(c2, "");
        assert_eq!(t2, "</thi");
    }

    #[test]
    fn test_second_think_block_treated_as_literal() {
        let mut parser = ThinkBlockParser::new();
        let (c1, t1) = parser.feed("<think>first</think>");
        assert_eq!(t1, "first");
        assert_eq!(c1, "");

        let (c2, t2) = parser.feed("middle<think>second</think>end");
        assert_eq!(c2, "middle<think>second</think>end");
        assert_eq!(t2, "");
    }

    #[test]
    fn test_token_by_token() {
        let mut parser = ThinkBlockParser::new();
        let input = "<think>AB</think>CD";
        let mut all_content = String::new();
        let mut all_thinking = String::new();

        for ch in input.chars() {
            let (c, t) = parser.feed(&ch.to_string());
            all_content.push_str(&c);
            all_thinking.push_str(&t);
        }
        let (c, t) = parser.flush();
        all_content.push_str(&c);
        all_thinking.push_str(&t);

        assert_eq!(all_thinking, "AB");
        assert_eq!(all_content, "CD");
    }

    #[test]
    fn test_angle_bracket_inside_think() {
        let mut parser = ThinkBlockParser::new();
        let (c, t) = parser.feed("<think>a < b and c > d</think>result");
        assert_eq!(t, "a < b and c > d");
        assert_eq!(c, "result");
    }

    #[test]
    fn test_newlines_in_think_block() {
        let mut parser = ThinkBlockParser::new();
        let (c, t) = parser.feed("<think>\nstep 1\nstep 2\n</think>\nanswer");
        assert_eq!(t, "\nstep 1\nstep 2\n");
        assert_eq!(c, "\nanswer");
    }

    #[test]
    fn test_think_with_leading_newline() {
        // DeepSeek-R1 often emits: \n<think>\n...\n</think>\n\nanswer
        let mut parser = ThinkBlockParser::new();

        let (c1, t1) = parser.feed("\n");
        assert_eq!(c1, "\n");
        assert_eq!(t1, "");

        let (c2, t2) = parser.feed("<think>");
        assert_eq!(c2, "");
        assert_eq!(t2, "");

        let (c3, t3) = parser.feed("\nreasoning\n");
        assert_eq!(c3, "");
        assert_eq!(t3, "\nreasoning\n");

        let (c4, t4) = parser.feed("</think>");
        assert_eq!(c4, "");
        assert_eq!(t4, "");

        let (c5, t5) = parser.feed("\n\nanswer");
        assert_eq!(c5, "\n\nanswer");
        assert_eq!(t5, "");
    }

    #[test]
    fn test_empty_input() {
        let mut parser = ThinkBlockParser::new();
        let (c, t) = parser.feed("");
        assert_eq!(c, "");
        assert_eq!(t, "");
        let (c2, t2) = parser.flush();
        assert_eq!(c2, "");
        assert_eq!(t2, "");
    }

    #[test]
    fn test_less_than_in_normal_text() {
        let mut parser = ThinkBlockParser::new();
        let (c, t) = parser.feed("x < y and a > b");
        assert_eq!(c, "x < y and a > b");
        assert_eq!(t, "");
    }

    #[test]
    fn test_less_than_followed_by_non_think() {
        let mut parser = ThinkBlockParser::new();
        let (c, t) = parser.feed("<div>hello</div>");
        assert_eq!(c, "<div>hello</div>");
        assert_eq!(t, "");
    }

    #[test]
    fn test_realistic_deepseek_r1_output() {
        let mut parser = ThinkBlockParser::new();
        let tokens = vec![
            "<think>",
            "\nThe user is asking about",
            " the capital of France.\n",
            "I know it's Paris.",
            "\n</think>",
            "\n\nThe capital",
            " of France is",
            " Paris.",
        ];

        let mut all_content = String::new();
        let mut all_thinking = String::new();

        for token in tokens {
            let (c, t) = parser.feed(token);
            all_content.push_str(&c);
            all_thinking.push_str(&t);
        }
        let (c, t) = parser.flush();
        all_content.push_str(&c);
        all_thinking.push_str(&t);

        assert_eq!(
            all_thinking,
            "\nThe user is asking about the capital of France.\nI know it's Paris.\n"
        );
        assert_eq!(all_content, "\n\nThe capital of France is Paris.");
    }
}
