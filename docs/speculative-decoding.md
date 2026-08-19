# Model-neutral Speculative Decoding

Speculative decoding is a Power runtime capability. It is not owned by Qwen,
DeepSeek, Llama, or any other model family. [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
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
4. Qwen3.8-27B Q6_K is the first CUDA performance gate. Baseline and MTP runs
   use the same model digest, prompts, sampling settings, context, and hardware.
5. A separate DSpark artifact is admitted only after its target/tokenizer
   compatibility, provenance, peak memory, exactness, and speedup are measured.
6. At least one non-Qwen adapter must pass the same transaction and exactness
   suite before DSpark support is considered cross-architecture complete.

The Qwen3.8 performance gate is at least 100 generated tokens per second on the
acceptance host through Power's streaming API. This is an acceptance floor,
not a tuning ceiling. Native-tool measurements are diagnostic evidence; they
do not replace the Power end-to-end result.

## Reproducible Power API benchmark

`a3s-power-speculative-bench` captures the acceptance evidence through
`POST /v1/completions`; it does not call llama.cpp directly. Build the server
and client from one source state. Release captures must use a clean revision;
development captures must disclose the dirty state and record the executable
digest in companion evidence. The pinned llama-cpp-rs revision does not expose
Power's context extensions, so fetch it and apply the reviewed patch before a
CUDA build:

```powershell
cargo fetch
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\apply-llamacpp-power-patches.ps1
```

On non-Windows hosts, apply
`patches/llama-cpp-rs-dfd12e4-mtp-dynamic-k.patch` to the fetched
`llama-cpp-rs` checkout, then apply
`patches/llama-cpp-rs-dfd12e4-mtp-fr-spec.patch` to its nested
`llama-cpp-sys-2/llama.cpp` checkout. Then build both executables:

```console
cargo build --release --no-default-features --features llamacpp-cuda,llamacpp-mtp-fr \
  --bin a3s-power --bin a3s-power-speculative-bench
```

Use one reviewed Q6_K GGUF that retains the Qwen3.8 native prediction metadata
and tensors (`qwen35.nextn_predict_layers > 0`; `qwen35` is the GGUF
architecture identifier retained by Qwen3.8). Record its lowercase SHA-256,
register that exact file once, and use the same registry entry for both runs.
Do not substitute an unreviewed conversion merely because its filename says
Q6_K. Keep every performance-affecting ACL setting fixed across server
restarts, including `gpu`, `num_thread`, `flash_attention`, `num_parallel`,
`use_mlock`, `use_mmap`, TEE mode, timing padding, and these draft settings:

```acl
spec_draft_max = 7
spec_mtp_recurrent_snapshots = 7
# Optional, experimental MTP draft-head row limit. Compact d2t heads use
# frequency-rank order; legacy full heads use target token-ID order.
# Omit this line to project every available draft-head row.
# spec_mtp_fr_vocab_size = 8192
spec_draft_min = 0
spec_draft_p_min = 0.0
suppress_token_metrics = false

gpu {
  gpu_layers = -1
  main_gpu = 0
  # Any exact tensor-level CPU placement must remain identical across A/B runs.
  # cpu_tensors = ["output.weight"]
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

The correctness minimum is `num_batch >= spec_draft_max + 2`: one target
anchor plus the proposal rows and one staging slot required because
llama.cpp's recurrent splitter requires the physical batch to be strictly
larger than its anchor-plus-snapshot tail. The conservative `draft_max=6`
capture used `--num-batch 8`; the tuned RTX 4090 capture used `draft_max=7`
and `--num-batch 24`. Batch size is also a CUDA graph shape, so benchmark it
rather than assuming the minimum is fastest. Always use the identical value
for the baseline and candidate.

`spec_mtp_recurrent_snapshots` bounds resident target rollback state separately
from draft width. The effective value is capped by `spec_draft_max`. When a
rejected suffix exceeds that window, Power restores exactness by replaying the
committed target prefix. Record the same explicit value in both A/B configs;
reducing it can avoid a GPU-memory allocation cliff, while replay frequency can
reduce throughput. A three-run mixed-workload calibration made the boundary
explicit: fixed K7/S6 incurred 46 fallback replays per run and reached only
28.226 token/s, while fixed K7/S7 incurred none and reached 68.211 token/s.
Adaptive K7/S6 also incurred no replay because its proposal width is capped by
the rollback window, but reached 60.031 token/s. Use a complete K-sized window
for fixed general workloads; a narrower window is a measured peak-only tuning.

`spec_mtp_fr_vocab_size` is an experimental FR-Spec-inspired optimization for
the llama.cpp MTP context only. A full draft head selects rows in target
token-ID order. A compact head can instead carry an I64 `d2t` tensor whose rows
are ordered by a reproducible corpus-frequency ranking; draft logits are
scattered back into the full target vocabulary and every absent row remains
negative infinity. The target still verifies every proposal, so a draft-only
token is never committed. Draft acceptance, block-vs-serial numerical effects,
and output parity remain workload gates. Keep the artifact hash and explicit
row limit identical in controlled A/B configs.

```console
a3s-power-speculative-bench run \
  --url http://127.0.0.1:11434 \
  --model qwen3.8-27b-q6-k \
  --model-sha256 <64-lowercase-hex> \
  --mode off \
  --power-commit <40-or-64-lowercase-git-revision> \
  --hardware-label rtx-4090-qwen38-q6k \
  --prompt-file benchmark-prompt.txt \
  --max-tokens 256 --num-ctx 4096 --num-batch 24 --seed 42 \
  --warmup-runs 1 --samples 5 \
  --min-tokens-per-second 0 > baseline.json

a3s-power-speculative-bench run \
  --url http://127.0.0.1:11434 \
  --model qwen3.8-27b-q6-k \
  --model-sha256 <same-64-lowercase-hex> \
  --mode mtp \
  --power-commit <same-git-revision> \
  --hardware-label rtx-4090-qwen38-q6k \
  --prompt-file benchmark-prompt.txt \
  --max-tokens 256 --num-ctx 4096 --num-batch 24 --seed 42 \
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
baseline.

The [RTX 4090 acceptance capture](benchmarks/qwen3.8-27b-q6k-rtx4090/README.md)
pins the 22,884,408,288-byte GGUF by SHA-256 and records five measured samples
per mode. The final MTP configuration reached a 140.1600 token/s median and a
139.4793 token/s minimum, versus a 35.5793 token/s same-shape baseline median
(3.9394x speedup). Every sample produced the same output digest, so the
100 token/s gate and greedy parity both pass. A post-safety rebuild repeated
the A/B after the recurrent-batch validation fix and retained a 129.7065
token/s MTP median, 125.8369 token/s minimum, 4.0318x speedup, and identical
greedy output under an active Windows desktop; those companion reports are
kept beside the best capture rather than replacing it. A later Q6_K-derived
mixed-precision development artifact combined selective TBQ4-style FFN
requantization, backend CUDA sampling, Flash Attention, and output-reorder
elimination. The current path keeps the full draft vocabulary and batches all
dense pure-greedy verification rows into one matrix argmax plus one contiguous
device-to-host token copy. Pure-greedy MTP drafting with a zero probability
threshold also reads the backend-selected draft token directly, avoiding its
former Top-K probability and CPU-sampler path; positive `spec_draft_p_min`
values and non-greedy requests retain the general implementation. Two
independent nine-sample 1,024-token captures of the probability-guarded build
reach 187.6094 and 188.2972 token/s medians, with an 183.7360 token/s combined
minimum. An earlier 256-token capture fell to 159.8593 under heavy WDDM
contention. The current captured binary adds a benchmark-only physical-core
affinity control: an order-balanced 12-sample A/B improved the combined median
from 173.0114 to 176.8276 token/s, and an independent nine-sample confirmation
reached a 177.3062 median and 175.5958 minimum with every sample above 175.
The runner records both requested and effective masks; the accepted `0x55555`
mask is specific to the 10-core, 20-thread Xeon W5-2445 and is not a portable
runtime default. The shared WDDM display GPU can still erase this margin, so
175 is an observed boundary rather than a guaranteed service floor. Because
the runtime artifact selectively requantizes Q6_K source tensors, it is
reported separately from the untouched-Q6_K same-artifact comparison. Earlier
8,192-row prefix-FR
captures remain archived as historical, workload-sensitive experiments. A
backend-resident hidden-row carry prototype was also removed after an
order-reversed eight-sample A/B produced a 0.09% median gain but a 0.38% mean
regression. It reduced transfer staging without reducing target or draft graph
work, so the result stayed within WDDM noise and did not justify an additional
cross-context state contract.
