# Keyed Prompt-Prefix Cache

Power exposes prompt-prefix KV reuse as an explicit request contract. A caller
opts in with `prompt_cache_key`; requests without that field remain stateless.
The contract is model-neutral, while each backend must advertise a compatible
implementation before Power forwards the key.

## Current support

| Path | Text chat | Text completion | Multimodal | Behavior when requested |
| --- | --- | --- | --- | --- |
| `llamacpp` / `llamacpp-cuda` | Prefix match | Prefix match | Not yet | Reuses a verified token prefix when llama.cpp can roll back the model's memory exactly; otherwise records a miss. |
| `mistralrs` | Not implemented | Not implemented | Not implemented | Returns `prompt_cache_unsupported`. |
| `picolm` | Not implemented | Not implemented | Not implemented | Returns `prompt_cache_unsupported`. |
| `proxy` | Not guaranteed | Not guaranteed | Not guaranteed | Returns `prompt_cache_unsupported`. |

Power never treats an ignored backend field as a cache hit. An invalid key or
unsupported selected backend returns an OpenAI-shaped `400` response before
model loading.

## Request contract

`prompt_cache_key` is an A3S extension accepted by both
`POST /v1/chat/completions` and `POST /v1/completions`. It must contain 1 to 256
UTF-8 bytes, must not have surrounding whitespace, and must not contain control
characters.

```json
{
  "model": "your-model",
  "messages": [
    {"role": "system", "content": "A long shared agent or RAG prefix..."},
    {"role": "user", "content": "The request-specific suffix"}
  ],
  "prompt_cache_key": "agent-policy-v3",
  "max_tokens": 64,
  "stream": true
}
```

The caller identifier is never used as the backend map key. Power derives an
opaque SHA-256 identity with separate authenticated identity, endpoint, model,
and caller-key domains. Two API keys, models, or endpoint types cannot address
the same context. In deployments without authentication, callers share the
explicit anonymous namespace and must be treated as one trust domain.

Attestation receipt decoding parameters include
`prompt_cache_key_sha256` only when caching is requested. The raw caller key is
not serialized into the receipt.

## Lifecycle and memory bound

llama.cpp holds reusable contexts per loaded model in a capacity-bounded LRU
map with an idle TTL. The default permits one resident context for five
minutes:

```acl
prompt_cache_max_entries = 1
prompt_cache_ttl_seconds = 300
```

The accepted ranges are 1 to 1,024 entries and 1 to 86,400 seconds. Equivalent
environment overrides are:

```text
A3S_POWER_PROMPT_CACHE_MAX_ENTRIES=4
A3S_POWER_PROMPT_CACHE_TTL_SECONDS=900
```

A KV context can consume substantial CPU or GPU memory. Raise the entry bound
only after measuring the exact context size and leaving headroom for model
weights, target batches, graph captures, and concurrent requests. Unloading a
model drops its resident contexts and updates the entry gauge.

On a keyed request, llama.cpp removes the context from the reusable map while
it is executing, tokenizes the new prompt, retains only the longest equal token
prefix that its model memory can roll back exactly, and evaluates the suffix.
Each keyed context reserves one recurrent rollback plane. It also captures the
prompt-boundary recurrent/SWA state on device; before returning the context, it
restores that state and removes the generated suffix. If either restoration or
partial removal cannot be proved, Power clears the context and records the next
lookup as a miss instead of reusing stale state.

Transformer KV can normally remove an arbitrary suffix. Hybrid recurrent
models have a bounded rollback window; strict prompt extension is therefore the
portable fast path, while an earlier divergent suffix may become an exact miss.
If the complete prompt matches, llama.cpp re-evaluates its final token because
KV truncation alone does not restore that token's logits. Only tokens confirmed
at the normalized prompt boundary are advertised as resident.

## Observability

`GET /health` reports the public request field, configured bounds,
authenticated namespace policy, and the backends that advertise support.
`GET /metrics` exports:

- `power_prompt_cache_requests_total`;
- `power_prompt_cache_hits_total` and `power_prompt_cache_misses_total`;
- `power_prompt_cache_reused_tokens_total`;
- `power_prompt_cache_evaluated_tokens_total`;
- `power_prompt_cache_evictions_total`; and
- `power_prompt_cache_entries`.

Every metric has a `backend` label. A hit means at least one token was reused;
it does not by itself prove a latency improvement.

Streaming text completions requested with
`stream_options.include_usage = true` also report backend
`prompt_eval_duration_ns` inside the final `a3s_performance` event. This keeps
backend prefill time separate from request-wide TTFT and steady decode timing.

## Speculation boundary

The current llama.cpp adapter does not compose a cached session context with
native MTP. Explicit `spec_mode = "mtp"` plus `prompt_cache_key` fails closed.
With `spec_mode = "auto"`, a keyed request selects exact target-only decoding
instead of claiming MTP. Combining both requires one reviewed state transaction
for target KV, MTP recurrent state, sampler state, rollback, and CUDA Graph
ownership; that adapter is not implemented yet.

This separates two measurements:

- speculative decoding improves effective generated tokens per target pass;
- prefix caching reduces repeated prefill work and time to first token.

Neither changes the other metric automatically. Quantization remains an
artifact and quality decision after these lossless execution paths are
measured.

## Reproducible acceptance check

Use a fixed long prefix, then make the paired warm prompt strictly extend the
cold prompt. Keep model bytes, binary revision, ACL, sampling, and host state
fixed.

1. Start a `llamacpp` or `llamacpp-cuda` build with `spec_mode = "off"` and a
   known cache capacity and TTL.
2. Record `/health`, `/v1/models/<name>`, and the prompt file SHA-256.
3. Record the prompt-cache counters from `/metrics`.
4. Send the first request with a fresh key. It must add one request and one
   miss.
5. Send the second request with the same key and append a short suffix to the
   complete cold prompt. It must add one request, one hit, and a positive
   reused token count.
6. Alternate which suffix is used to prime each pair. Report wall latency and server
   prompt-evaluation timing separately from steady decode token/s.
7. Reject the result if the selected backend, model hash, configuration,
   receipt policy, output policy, or metric deltas do not match the capture.

The checked-in zero-dependency client automates that contract. It requires an
idle single-request server, verifies the model hash and server revision, warms
model loading outside the sample window, alternates the two suffix orders, and
requires an exact miss followed by a hit for every fresh key:

```powershell
$revision = git rev-parse HEAD
py -3 tools/prompt_cache_benchmark.py `
  --model your-model `
  --expected-model-sha256 <64-hex-model-digest> `
  --server-revision $revision `
  --prefix-file .\benchmark-prefix.txt `
  --run-id pcache-20260822-a `
  --pairs 5 `
  --output .\target-prompt-cache\capture.json
```

The server must expose `num_parallel = 1`, no timing padding, unsuppressed token
metrics, and `spec_mode = "off"` or `"auto"`; the client fails otherwise. The
capture contains hashes and timing/counter evidence, but not the raw prefix,
caller keys, or generated text. Run its deterministic unit tests with:

```text
python3 -m unittest tools/test_prompt_cache_benchmark.py
```

For honest service reporting, publish the cache hit ratio, reused/evaluated
token totals, TTFT distribution, request-wide throughput, entry memory, and
evictions together. A large prefill speedup on one shared prefix is not a
decode-speed or concurrent-throughput claim.
