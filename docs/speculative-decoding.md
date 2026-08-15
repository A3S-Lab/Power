# Model-neutral Speculative Decoding

Speculative decoding is a Power runtime capability. It is not owned by Qwen,
DeepSeek, Llama, or any other model family. [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)
is the first dense hybrid acceptance target for the CUDA path, not a special
case in the shared runtime.

The design follows the separation used by
[DSpark](https://arxiv.org/abs/2607.05147): proposal generation is advisory,
while the target model remains the source of truth for every emitted token.
Power exposes one control plane for zero-weight lookup, independent draft
models, native MTP heads, DFlash, DSpark, and future proposal algorithms.

## Ownership boundary

Power owns model-independent behavior:

- strategy parsing and fail-closed capability negotiation;
- bounded draft and verification scheduling;
- lossless target sampling and accepted-prefix accounting;
- commit and rejected-suffix rollback orchestration;
- cancellation, output limits, stop sequences, and streaming;
- acceptance, target-pass, latency, throughput, and memory evidence.

Each backend/model adapter owns architecture-dependent behavior:

- checkpoint discovery, compatibility fingerprints, and tensor layouts;
- draft graphs, native heads, and accelerator kernels;
- target block evaluation and logits access;
- transactions for every mutable model state;
- tokenizer and vocabulary compatibility between target and drafter.

A conventional transformer adapter normally transactions KV state. A hybrid
adapter may also need recurrent, convolution, or state-space snapshots. Power
does not infer those layouts from a model name.

## Capability negotiation

`SpeculativeStrategy` is the stable configuration vocabulary. A backend/model
pair advertises `SpeculativeCapabilities`, then resolves `auto` to a safe local
default. An explicit unsupported strategy returns an error; it never silently
falls back to a cheaper algorithm and never relabels n-gram lookup as DSpark.

Current executable adapters are:

| Backend/model capability | Strategies |
| --- | --- |
| mistral.rs or proxy | `off` (`auto` resolves to `off`) |
| picolm | `off`, `prompt-lookup`, `ngram-context` |
| llama.cpp without native prediction tensors | `off` |
| llama.cpp with `*.nextn_predict_layers > 0` | `off`, `mtp` |

`draft-model`, `dflash`, and `dspark` are part of the shared protocol but stay
unavailable until the loaded model supplies a compatible adapter artifact and
the backend implements its graph. This is intentional fail-closed behavior.

## Verification transaction

One model-backed round has the following order:

1. Retain a checkpoint for all target, draft, sampler, and decoder state.
2. Ask the adapter for at most the scheduled number of draft tokens.
3. Evaluate the anchor and draft block in one target pass.
4. Sample target rows only through the first mismatch. If every draft matches,
   sample one target bonus token.
5. Commit the accepted prefix and discard every rejected state row.
6. Stream accepted tokens through a terminal token; otherwise append exactly
   one correction or bonus token.
7. On cancellation or failure, restore the last committed transaction.

For greedy decoding, the speculative and non-speculative token IDs must match
exactly. For stochastic decoding, target sampler state advances once for each
emitted target sample and never for an unobserved rejected suffix.

## Adapter families

- Prompt lookup and online n-grams are zero-weight integration baselines.
- Native MTP consumes prediction tensors shipped with a target checkpoint.
- A separate draft model uses a tokenizer-compatible smaller checkpoint.
- DFlash provides a parallel block-diffusion draft backbone.
- DSpark adds prefix-dependent Markov and confidence heads to parallel drafting.

DSpark adapters can target Qwen, Llama, DeepSeek, Gemma, MoE, or future
architectures. Compatibility is determined by adapter metadata and state
contracts, not by branches in the Power scheduler.

## Delivery and acceptance

1. Shared strategy, capability, exact-verification, adaptive-length, and metric
   primitives are implemented and covered by deterministic tests.
2. picolm consumes the shared strategy vocabulary while preserving its former
   adaptive draft-length default.
3. llama.cpp provides native MTP execution with transactional target and draft
   rollback for compatible models.
4. Qwen3.5-27B Q6_K is the first CUDA performance gate. Baseline and MTP runs
   use the same model digest, prompts, sampling settings, context, and hardware.
5. A separate DSpark artifact is admitted only after its target/tokenizer
   compatibility, provenance, peak memory, exactness, and speedup are measured.
6. At least one non-Qwen adapter must pass the same transaction and exactness
   suite before DSpark support is considered cross-architecture complete.

The Qwen3.5 performance gate is at least 100 generated tokens per second on the
acceptance host through Power's streaming API. Native-tool measurements are
diagnostic evidence; they do not replace the Power end-to-end result.

## Reproducible Power API benchmark

`a3s-power-speculative-bench` captures the acceptance evidence through
`POST /v1/completions`; it does not call llama.cpp directly. Build the server
and client from one clean revision:

```console
cargo build --release --no-default-features --features llamacpp-cuda \
  --bin a3s-power --bin a3s-power-speculative-bench
```

Use one reviewed Q6_K GGUF that retains the Qwen3.5 native prediction metadata
and tensors (`qwen35.nextn_predict_layers > 0`). Record its lowercase SHA-256,
register that exact file once, and use the same registry entry for both runs.
Do not substitute an unreviewed conversion merely because its filename says
Q6_K. Keep every performance-affecting ACL setting fixed across server
restarts, including `gpu`, `num_thread`, `flash_attention`, `num_parallel`,
`use_mlock`, TEE mode, timing padding, and these draft settings:

```acl
spec_draft_max = 3
spec_draft_min = 0
spec_draft_p_min = 0.0
suppress_token_metrics = false

gpu {
  gpu_layers = -1
  main_gpu = 0
}
```

Start Power with explicit `spec_mode = "off"`, capture the baseline, stop the
server, then restart the same binary and ACL with only
`spec_mode = "mtp"` changed. `GET /health` exposes the effective speculative
and non-secret inference settings; the benchmark records them and comparison
rejects configuration drift. `GET /v1/models/:name` supplies the registered
format, byte length, and SHA-256 checked by the client.

Use a fixed UTF-8 prompt that reliably reaches the output limit. Every warmup
and measured request uses greedy sampling (`temperature = 0`, `top_p = 1`), a
fixed seed and context, `keep_alive = -1`, streaming, and exact opted-in usage.
An early EOS or stop result fails the run instead of silently producing a
shorter, faster sample.

```console
a3s-power-speculative-bench run \
  --url http://127.0.0.1:11434 \
  --model qwen3.5-27b-q6-k \
  --model-sha256 <64-lowercase-hex> \
  --mode off \
  --power-commit <40-or-64-lowercase-git-revision> \
  --hardware-label rtx-4090-cuda \
  --prompt-file benchmark-prompt.txt \
  --max-tokens 256 --num-ctx 4096 --seed 42 \
  --warmup-runs 1 --samples 5 \
  --min-tokens-per-second 0 > baseline.json

a3s-power-speculative-bench run \
  --url http://127.0.0.1:11434 \
  --model qwen3.5-27b-q6-k \
  --model-sha256 <same-64-lowercase-hex> \
  --mode mtp \
  --power-commit <same-git-revision> \
  --hardware-label rtx-4090-cuda \
  --prompt-file benchmark-prompt.txt \
  --max-tokens 256 --num-ctx 4096 --seed 42 \
  --warmup-runs 1 --samples 5 \
  --min-tokens-per-second 100 > mtp.json

a3s-power-speculative-bench compare baseline.json mtp.json > comparison.json
```

For authenticated servers, put the bearer token in an environment variable and
add `--api-key-env <VARIABLE>`; the key is never accepted as a command-line
value. Plain HTTP is restricted to loopback hosts. Prompt content, prompt path,
server URL, model path, and API key are omitted from reports. The client hashes
the streamed UTF-8 output, verifies every inference receipt digest, and requires
output parity across samples and modes.

The threshold uses the median server-side steady-state decode rate:
`(completion_tokens - 1) / (last_token_time - first_token_time)`. Reports also
retain time to first token and client-observed end-to-end throughput. Power
emits these exact timings only in the final opted-in SSE usage event when token
metric suppression is disabled. A comparison passes only when the candidate's
declared threshold passes and its output digest matches the autoregressive
baseline. Until a real digest-pinned Qwen3.5-27B Q6_K capture is attached, the
100 token/s performance gate remains open.
