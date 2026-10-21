# WorkBuddy chat compatibility

`POST /v1/chat/completions` accepts the WorkBuddy message annotations `agent`,
`conversationRequestId`, `startsNewUserRequest`, `messageId`, `model`,
`requestModelId`, `requestModelName`, `stepSeq`, `traceId`, `argumentsDisplayText`,
`rawUsage`, and `usage` inside individual `messages`
objects. They are opaque client annotations, consumed at the request boundary.
They do not select a server agent, start or reset a server session, alter routing,
change policy, or become model prompt content. Values are not echoed to model
backends, responses, or inference receipts. No client detection or User-Agent
heuristic is used.

`argumentsDisplayText` is a client display record; it never replaces the typed
`tool_calls[].function.arguments`. Replayed `rawUsage` and `usage` never replace
backend-reported usage, inference receipts, or accounting. The response reports
the current inference's usage rather than trusting a client's history record.

Standard message fields, content parts, tool calls and tool results keep their
existing behavior. Completed assistant function calls may replay null or omitted
content through the existing request-only normalization. Other unknown fields,
including annotation names at the top request level or inside content parts,
remain subject to strict validation. The root `model` remains the required typed
routing field; a replayed message's `model` is only a client history label and
cannot change that route. Duplicate annotation keys are rejected by the Power
parser. The managed Gateway canonicalizes JSON before forwarding, so its earlier
boundary may already have coalesced duplicate keys.

WorkBuddy can continue using the OpenAI-compatible chat endpoint and its existing
API credential. Use `/v1` as the API base when the client appends
`/chat/completions`; if a client requires an exact URL, use
`/v1/chat/completions`. This compatibility change does not alter authentication,
model IDs, request-size limits, or firewall rules.

Regression checks:

```sh
cargo fmt --all -- --check
cargo test --locked --no-default-features --features server --lib workbuddy_compat_tests
cargo test --locked --no-default-features --features server --lib
```
