# Serving admission and qualification

HTTP admission defaults to one active request, eight queued requests and a
1000 ms queue deadline. Configure `max_concurrent_requests`,
`max_queued_requests` and `queue_timeout_ms` in ACL. Active capacity accepts
1..100000; waiting capacity accepts 0..100000; queue timeout accepts
1..300000 ms. Zero waiting capacity means immediate refusal when active
capacity is exhausted. Zero active capacity is a startup error.

All four admission endpoints (chat, completions, embeddings and SystemOne)
use the same finite policy. Queue overflow and expiry return HTTP 429; no
hidden retry starts another inference. Cancellation drops the owned backend
stream before releasing the HTTP permit. Verify actual GPU slot recovery
against the pinned engine; closing an HTTP socket alone is not GPU-stop proof.

`power_http_admission_wait_seconds` measures local admission wait, with fixed
outcomes admitted, queue_full, timeout, cancelled and unavailable.
`power_http_admission_refusals_total` records the corresponding refusal
reasons. These metrics exclude engine queue and contain no prompt labels.
Time to first meaningful output starts at HTTP handler entry and ignores
role-only frames. Buffered completions expose completion latency, not TTFT.

External Prism chat uses incremental upstream SSE for both API response modes.
Tool call identity and index remain stable while argument/name fragments are
accumulated. A failed or truncated stream is not a successful completion and
does not emit a success terminal. Authoritative upstream usage is propagated
when present; missing counts remain unknown. SSE chunks and bytes are never
token counts. Privacy suppression also covers nested usage and timing metadata.
Legacy completions have separate buffering behavior; chat streaming changes do
not establish a completions streaming speedup.

`num_parallel` in Power configuration is not proof of external llama-server
slots. Inspect the pinned engine's actual slot count and effective context per
sequence, and size HTTP admission accordingly. External engine prompt reuse
does not imply authenticated distributed-cache or state-transfer support.
Worker observation must continue to advertise unsupported capabilities as such.

Qualify old and candidate artifacts through the same Gateway and pinned model,
with literal engine/runtime identities, equal corpus, tokenizer/template,
sampling, context, cache and warmup policy. Separate cold autoload from warm
inference and report overload refusals alongside successful throughput. Record
cancelled, truncated, invalid and failed requests explicitly. A changed slot,
cache, speculation or NUMA profile needs a declared experimental factor.
Do not promote an artifact from a single short run or a fabricated token rate.
