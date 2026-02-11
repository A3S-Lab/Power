# Plan: Phase 15 — Thinking & Reasoning

## Goal

Add thinking/reasoning support for DeepSeek-R1, QwQ, and similar models. When `think` is enabled, the model's `<think>...</think>` blocks are separated from the response content and returned in a dedicated `thinking` field.

## Design

### Core: Streaming ThinkingParser (`src/backend/thinking_parser.rs` — NEW ~120 lines)

A stateful parser that processes streaming token chunks and separates `<think>...</think>` blocks from content. Handles partial tags across chunk boundaries via buffering.

```rust
pub struct ThinkingParser { buffer: String, state: State, thinking: String }
enum State { Normal, InTag, InThinking, InCloseTag }

pub struct ProcessedChunk {
    pub thinking: Option<String>,  // thinking delta (if any)
    pub content: String,           // normal content delta
}

impl ThinkingParser {
    pub fn new() -> Self;
    pub fn process(&mut self, text: &str) -> ProcessedChunk;
}
```

### Type Changes

**1. `ThinkValue` type** — `src/api/types.rs`
```rust
#[serde(untagged)]
pub enum ThinkValue { Bool(bool), Level(String) }
```
- Add `think: Option<ThinkValue>` to `GenerateRequest`, `NativeChatRequest`, `ChatCompletionRequest`
- Add `thinking: Option<String>` to `GenerateResponse`, `NativeChatResponse`
- Add `thinking: Option<String>` to `ChatDelta`, `ChatCompletionMessage`
- Add `reasoning_effort: Option<String>` to `ChatCompletionRequest`

**2. Backend types** — `src/backend/types.rs`
- Add `think: Option<bool>` to `ChatRequest`, `CompletionRequest` (simplified bool — API layer maps levels)
- Add `thinking: Option<String>` to `ChatResponseChunk`, `CompletionResponseChunk`

### Handler Integration

**3. Native generate** — `src/api/native/generate.rs`
- When `think` is enabled: wrap backend stream with `ThinkingParser`, map `ProcessedChunk` → `GenerateResponse` with `thinking` field
- When `think` is disabled: pass through unchanged (zero overhead)

**4. Native chat** — `src/api/native/chat.rs`
- Same pattern as generate

**5. OpenAI chat** — `src/api/openai/chat.rs`
- Map `reasoning_effort` → `think` on backend request
- Stream: emit `thinking` in `ChatDelta`
- Non-stream: include `thinking` in final `ChatCompletionMessage`

### CLI

**6. CLI flags** — `src/cli/mod.rs`
- Add `--think` (bool flag) and `--hidethinking` (bool flag) to `Run` command

**7. CLI run** — `src/cli/run.rs`
- Add `think: bool` and `hide_thinking: bool` to `RunOptions`
- Pass `think` to `ChatRequest`
- When streaming: if `hide_thinking` is false, print thinking content in dimmed style before response

### Backend module

**8. `src/backend/mod.rs`**
- Add `pub mod thinking_parser;`

## Files Changed

| File | Change |
|------|--------|
| `src/backend/thinking_parser.rs` | **NEW** — Stateful streaming parser (~120 lines) |
| `src/backend/mod.rs` | Add `pub mod thinking_parser;` |
| `src/backend/types.rs` | Add `think` to requests, `thinking` to response chunks |
| `src/api/types.rs` | Add `ThinkValue`, `think`/`thinking` fields, `reasoning_effort` |
| `src/api/native/generate.rs` | Wrap stream with ThinkingParser when think enabled |
| `src/api/native/chat.rs` | Wrap stream with ThinkingParser when think enabled |
| `src/api/openai/chat.rs` | Map reasoning_effort → think, emit thinking in delta |
| `src/cli/mod.rs` | Add `--think`, `--hidethinking` flags |
| `src/cli/run.rs` | Add think/hide_thinking to RunOptions, pass to backend |

## Implementation Order

1. `thinking_parser.rs` — core parser with tests (TDD)
2. `backend/types.rs` — add fields
3. `api/types.rs` — add ThinkValue + fields
4. `api/native/generate.rs` — integrate parser
5. `api/native/chat.rs` — integrate parser
6. `api/openai/chat.rs` — reasoning_effort mapping
7. `cli/mod.rs` + `cli/run.rs` — CLI flags
8. Update tests across all modified files
9. Update README.md documentation
