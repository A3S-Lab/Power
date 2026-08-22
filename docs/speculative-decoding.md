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

The current clean untouched-Q6_K capture raises the observed boundary to a
176.6109 token/s median for prefix-FR8192 K7/S6, with seven of nine 1,024-token
samples at or above 175. Its full-vocabulary K7/S7 control reached 147.0207
token/s. This remains benchmark evidence for one artifact, prompt, backend,
and RTX 4090 host; it does not change the model-neutral runtime contract.

## Reproducible Power API benchmark

`a3s-power-speculative-bench` captures the acceptance evidence through
`POST /v1/completions`; it does not call llama.cpp directly. Build the server
and client from one source state. Release captures must use a clean revision;
development captures must disclose the dirty state and record the executable
digest in companion evidence. The pinned llama-cpp-rs revision does not expose
Power's context extensions, so fetch it and apply the reviewed patch before a
CUDA build:

```powershell
cargo fetch --locked
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\apply-llamacpp-power-patches.ps1
```

On non-Windows hosts, apply
`patches/llama-cpp-rs-dfd12e4-mtp-dynamic-k.patch` to the fetched
`llama-cpp-rs` checkout, then apply
`patches/llama-cpp-rs-dfd12e4-mtp-fr-spec.patch` and
`patches/llama-cpp-rs-dfd12e4-cuda-high-priority.patch` to its nested
`llama-cpp-sys-2/llama.cpp` checkout. Then build both executables:

```console
cargo build --locked --release --no-default-features --features llamacpp-cuda,llamacpp-mtp-fr \
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

`tools/run-gguf-speculative-benchmark.ps1` is the model-neutral host runner.
It requires the registered model name, GGUF SHA-256, prompt, ACL profile, Power
home, and explicit strategy instead of embedding a Qwen model name or path. For
each selected NVIDIA device it records a configurable pre-start utilization
window, the process set, binary and input hashes, clock request, power plan,
and effective CPU affinity. Use
`tools/run-qwen38-q6k-benchmark.ps1` only as a compatibility wrapper for the
published Qwen capture. `-CudaHighPriority` opts into the reviewed
`GGML_CUDA_HIGH_PRIORITY=1` path and records that choice in both preflight and
environment receipts. The runner removes the variable for ordinary-stream
controls and restores the caller's environment in `finally`.

For a claimed service floor, set both gates. The median gate alone proves a
typical run; the all-sample gate requires the slowest measured request to meet
the floor:

```powershell
.\tools\run-gguf-speculative-benchmark.ps1 `
  -Label candidate-1024-9x `
  -Config .\candidate.acl `
  -Model my-registered-gguf `
  -ModelHash <64-lowercase-hex> `
  -PromptFile .\benchmark-prompt.txt `
  -PowerHome D:\models\power-home `
  -Mode mtp -MaxTokens 1024 -NumCtx 4096 -NumBatch 11 `
  -WarmupRuns 2 -Samples 9 `
  -ProcessPriority High `
  -ProcessorAffinityMask 349525 `
  -CudaHighPriority `
  -MinimumTokensPerSecond 175 `
  -MinimumSampleTokensPerSecond 175 `
  -NvidiaGpuIndices 0 `
  -MaximumIdleGpuUtilizationPercent 2 `
  -IdleGpuSampleCount 21 `
  -IdleGpuSampleIntervalMilliseconds 500
```

Twenty-one samples at 500 ms cover a 10-second quiet window. The runner writes
`<label>.preflight.json` before model startup; a failed idle or clock gate keeps
that receipt and never loads the model. `-PreflightOnly` validates and records
the host controls without starting the server. Once inference runs, the JSON
report and environment receipt are also written before a throughput-threshold
failure. These files preserve both environment and performance negative
evidence. On Windows PowerShell 5.1 they use BOM-free UTF-8; the comparison CLI
also accepts older reports that contain PowerShell's UTF-8 BOM.

The generic Windows runner defaults to the ordinary `target` Cargo directory
and does not require NVIDIA tooling. Select one or more devices with
`-NvidiaGpuIndices 0,1` only when NVIDIA idle-utilization evidence or clock
locking is part of the gate. `-NumCtx` is configurable for models whose
evaluated context is not 4,096 tokens. For a multi-GPU capture, make
`-HardwareLabel` identify the complete topology; the environment receipt keeps
the per-device indices and snapshots while the compact benchmark identity
reports the backend's primary detected adapter. The Qwen compatibility wrapper
retains its SM89-specific target directory and GPU 0 defaults.

The correctness minimum is `num_batch >= spec_draft_max + 2`: one target
anchor plus the proposal rows and one staging slot required because
llama.cpp's recurrent splitter requires the physical batch to be strictly
larger than its anchor-plus-snapshot tail. The conservative `draft_max=6`
profile uses `--num-batch 8`; the current RTX 4090 peak profile uses
`draft_max=7` and `--num-batch 11`. Batch size is also a CUDA graph shape: a
fixed K6/B8 mixed-task path and K7/B11 peak path outperformed larger allocations
on this host, while request-local variable K lost graph reuse. Benchmark the
actual model and workload instead of assuming either the minimum or a large
batch is universally fastest. Always use the identical value for a paired
baseline and candidate.

`spec_mtp_recurrent_snapshots` bounds resident target rollback state separately
from draft width. The effective value is capped by `spec_draft_max`. When a
rejected suffix exceeds that window, Power restores exactness by replaying the
committed target prefix. Fixed K>S configurations now wrap this fallback in a
request-local guard: the first exact replay is allowed, then all later rounds
in that request are clamped to the rollback-complete snapshot width. This
bounds replay without changing high-acceptance requests that never activate
the guard. Adaptive K>S configurations use the same rule: they start at the
configured width and clamp only after an observed rejection actually exceeds
the resident rollback window. Metrics report guarded requests, activations,
activation round, and the clamped draft limit.

Record the same explicit snapshot value in both A/B configs; reducing it can
avoid a GPU-memory allocation cliff, while recovery frequency can reduce
throughput. Before the guard, fixed K7/S6 incurred 46 fallback replays per
12-task run and reached 28.226 token/s. The current guard reduced that to 11
replays and 54.060 token/s; complete K7/S7 needed none and reached 68.205
token/s. The current 100-task K7/S7 matrix likewise recorded zero replay and
zero guard activation. Use a complete K-sized window for fixed general
workloads; narrower K7/S6 is a guarded peak-only tuning.

`spec_mtp_fr_vocab_size` is an experimental FR-Spec-inspired optimization for
the llama.cpp MTP context only. The reviewed patch currently advertises this
capability only for the GGUF `qwen35` adapter; another MTP architecture fails
closed when FR is requested and remains available through full-vocabulary MTP.
A full draft head selects rows in target token-ID order. A compact head can
instead carry an I64 `d2t` tensor whose rows are ordered by a reproducible
corpus-frequency ranking; draft logits are scattered back into the full target
vocabulary and every absent row remains negative infinity. The target still
verifies every proposal, so a draft-only token is never committed. Draft
acceptance, block-vs-serial numerical effects, and output parity remain
workload gates. Keep the artifact hash and explicit row limit identical in
controlled A/B configs.

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
  --min-tokens-per-second 100 \
  --min-sample-tokens-per-second 100 > mtp.json

a3s-power-speculative-bench compare baseline.json mtp.json > comparison.json
```

For authenticated servers, put the bearer token in an environment variable and
add `--api-key-env <VARIABLE>`; the key is never accepted as a command-line
value. Plain HTTP is restricted to loopback hosts. Prompt content, prompt path,
server URL, model path, and API key are omitted from reports. The client hashes
the streamed UTF-8 output, verifies every inference receipt digest, and requires
output parity across samples and modes.

The compatibility threshold uses the median server-side steady-state decode rate:
`(completion_tokens - 1) / (last_token_time - first_token_time)`. Reports also
retain time to first token and client-observed end-to-end throughput. Power
emits these exact timings only in the final opted-in SSE usage event when token
metric suppression is disabled. When
`--min-sample-tokens-per-second` is present, the report additionally requires
its minimum sample rate to pass. A comparison requires every declared
throughput gate and an output digest matching the autoregressive baseline.

The [RTX 4090 acceptance capture](benchmarks/qwen3.8-27b-q6k-rtx4090/README.md)
pins the 22,884,408,288-byte GGUF by SHA-256. The current clean capture records
one warm-up and nine measured 1,024-token samples per mode. Prefix-FR8192
K7/S6 reached a 176.6109 token/s median and 173.2630 minimum, versus a
147.0207 token/s full-vocabulary K7/S7 control; both emitted the same output
digest. A 12-task calibration reversed the ranking: full vocabulary reached
47.032 token/s request-wide with 52.30% proposal acceptance, while prefix FR
reached 37.290 token/s with 24.82% acceptance. Eleven tasks hit the output cap,
so the result establishes workload sensitivity rather than a quality score.

A 2026-08-22 execution-only follow-up kept those exact Q6_K bytes and split
the policy by workload. The peak profile uses fixed K7/S6, B11, Flash Attention
off, physical-core affinity, and high-priority CUDA streams. It reached a
172.252 token/s median and 171.250 minimum under a 5--8% busy Windows desktop;
the earlier quiet-host 176.6109 median remains the measured high-water mark.
Disabling CUDA graphs fell to 133.876 token/s, enabling the experimental graph
optimizer reached only 160.613 token/s, and `CUDA_DEVICE_MAX_CONNECTIONS=32`
reached 168.900 token/s. Those alternatives were rejected.

The paired mixed-task profile uses fixed K6/S6 and the minimum legal B8 shape.
At a 256-token cap it reached 49.025 token/s versus 29.381 token/s with
speculation off, a 66.86% gain, with identical final answers and the same 9/12
lenient and strict score in both modes. Proposal acceptance was only 26.81%,
but the fixed graph emitted 2.591 verified tokens per target pass with no
replay. Adaptive K reported a higher 50.07% acceptance yet slowed to 35.178
token/s because variable verification shapes and a target-only circuit reduced
CUDA graph reuse. Optimize useful tokens per total phase time, not acceptance
percentage in isolation.

The CUDA backend already uses dynamic Q8_1 activation quantization for
quantized matrix-matrix kernels. Forcing the eight-row Q6 verification shape
from MMVQ into that MMQ path reduced its five-sample median from 143.6024 to
132.2581 token/s for the 5-8-row route and to 116.1835 token/s for exactly
eight rows. Those dependency-checkout experiments retained output identity but
were reverted; on Ada SM89, their setup cost exceeded the benefit at this row
shape.

The historical same-artifact configuration reached a 140.1600 token/s median
and a 139.4793 token/s minimum, versus a 35.5793 token/s same-shape baseline
median (3.9394x speedup). A post-safety rebuild repeated the A/B after the
recurrent-batch validation fix and retained a 129.7065 token/s MTP median,
125.8369 token/s minimum, 4.0318x speedup, and identical
greedy output under an active Windows desktop; those companion reports are
kept beside the best capture rather than replacing it. A later Q6_K-derived
mixed-precision development artifact combined selective TBQ4-style FFN
requantization, backend CUDA sampling, Flash Attention, and output-reorder
elimination. The current path keeps the full draft vocabulary and batches all
dense pure-greedy verification rows into one matrix argmax plus one contiguous
device-to-host token copy. Pure-greedy MTP drafting with a zero probability
threshold also reads the backend-selected draft token directly, avoiding its
former Top-K probability and CPU-sampler path; positive `spec_draft_p_min`
values and non-greedy requests retain the general implementation. Current
nine-sample 1,024-token captures reached 177.7165 token/s median and 176.7287
minimum for guarded K7/S6, and 175.2089 median and 174.2211 minimum for
rollback-complete K7/S7. The S6 guard did not activate on that high-acceptance
peak prompt. The runner records both requested and effective masks; the
accepted `0x55555` mask is specific to the 10-core, 20-thread Xeon W5-2445 and
is not a portable runtime default. Earlier quiet-WDDM captures reached
187.6094 and 188.2972 token/s, while another 256-token capture fell to 159.8593
under contention. The shared display GPU can erase the margin, so 175 is an
observed boundary rather than a guaranteed service floor.

On the current cyclically ordered 3x100 task matrix, K7/S7 reached 83.228
token/s request-wide throughput versus 38.724 token/s for TBQ4-off, with
76/100 versus 70/100 lenient answers and 66/100 versus 64/100 strict answers.
The paired differences were not statistically significant at `p < 0.05`, so
the capture shows no observed quality decline but does not establish a general
intelligence gain. All predictions were stable across repeats, acceptance was
51.33%, and replay and guard activation were zero.

The mixed runtime artifact selectively requantizes Q6_K source tensors and is
reported separately from the current untouched-Q6_K captures. Prefix-FR8192
is also kept peak-only because its current full head is limited in target-token
ID order, not by a corpus-frequency-ranked `d2t` map, and its acceptance can
collapse on multilingual workloads. A backend-resident hidden-row carry
prototype was also removed after an
order-reversed eight-sample A/B produced a 0.09% median gain but a 0.38% mean
regression. It reduced transfer staging without reducing target or draft graph
work, so the result stayed within WDDM noise and did not justify an additional
cross-context state contract.
