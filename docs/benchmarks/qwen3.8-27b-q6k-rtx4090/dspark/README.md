# Native DSpark Q4 on Qwen3.8-27B Q6_K

This capture measures Power's native external-draft path through the real
streaming `POST /v1/completions` API. The target remains the untouched Q6_K
GGUF. A separate DSpark Q4 artifact proposes tokens; the Q6_K target verifies
every emitted token.

## Accepted result

The paired runs use the same clean Power commit, CUDA binaries, target model,
prompt, request shape, greedy sampling, and output limit.

| Mode | Measured decode samples | Median decode | Minimum | Median end to end | Peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| Q6_K target only | 32.645, 32.249, 32.024 token/s | 32.249 token/s | 32.024 token/s | 25.171 token/s | 21,860 MiB |
| Q6_K + DSpark Q4, K10/S6 | 169.561, 167.102, 169.324 token/s | **169.324 token/s** | **167.102 token/s** | **65.825 token/s** | 23,847 MiB |

The median decode speedup is **5.250x** and the median request-wide speedup is
**2.615x**. All three DSpark samples passed the 160 token/s all-sample gate.
The request SHA-256, every 256-token output SHA-256, and every execution receipt
SHA-256 match the target-only control exactly.

The DSpark run accepted 229 of 252 proposed tokens per request (90.873%), used
26 target verification rounds, emitted 9.8077 verified tokens per target pass,
and performed zero fallback replays. Its longest rejected suffix was six
tokens, exactly the resident S6 rollback window.

Run the offline verifier without a model or GPU:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\verify-dspark-evidence.ps1 -Json
```

The checked-in [target-only report](target-only.json),
[DSpark report](dspark-q4-k10-s6.json), and
[compact environment receipt](evidence.json) bind the raw samples, artifact
identities, executable hashes, telemetry summary, and capture revision.

## Adaptive request-local result

Fixed K10/S6 is the narrow peak high-water mark, but its representative
100-task run replayed every request. The current controller starts at the
rollback-complete K6 shape. A fully accepted first probe jumps directly to
K10; a partial first probe closes the wide path for that request. Healthy
partial rounds retain their graph shape, while twelve sustained low-yield
rounds open a one-way target-only circuit. This policy is model-neutral and is
shared by native MTP, DFlash, and DSpark adapters.

The clean `cbdb3f673446b3532c9683dabc816a149ae27b1f` capture used the same
Q6_K target, DSpark artifact, context 512, batch 12, 256-token greedy request,
2745 MHz clock lock, high-priority CUDA streams, High process priority, and
`0x55555` CPU affinity.

| Adaptive K10/S6 capture | Result |
| --- | ---: |
| Decode samples | 166.988, 160.881, 164.756 token/s |
| Median / minimum decode | **164.756 / 160.881 token/s** |
| Median end to end | 63.535 token/s |
| Proposal acceptance | 229/247, **92.713%** |
| Verified tokens per target pass | **9.8077** |
| Fallback replay / guard activation | **0 / 0** |
| Output / receipt identity | identical across all 3 samples |

The median and every sample passed the 160 token/s gate. This is a stable
160-plus boundary on the recorded prompt, not a 175 token/s service floor. It
trades 2.7% of the historical fixed-K10 median for a request-local safe start
and two reusable high-yield CUDA Graph shapes.

The paired context-1024, batch-12 quality run tells the broader story:

| Mode | Lenient | Strict | Truncated | Request-wide throughput |
| --- | ---: | ---: | ---: | ---: |
| Q6_K target-only | 67/100 | 58/100 | 40/100 | 22.872 token/s |
| Q6_K + adaptive DSpark Q4 | 69/100 | 56/100 | 42/100 | **31.052 token/s** |

The request-wide speedup is **1.358x**. Adaptive DSpark recorded five lenient
gains and three losses, one strict gain and three losses, 89/100 extracted-
answer parity, and 55/100 complete-output parity. All 57 tasks untruncated in
both modes retained the same extracted answer. Runtime telemetry recorded
62.878% proposal acceptance, 3.373 verified tokens per target pass, 24
target-only requests and 2,934 target-only tokens, with zero fallback replay
and zero rollback-guard activation.

The lenient score rose by two while strict score fell by two. Neither delta is
statistically persuasive. A hash-locked follow-up selected every observed
lenient or strict loss plus one positive control, then raised the generation
budget without modifying the reviewed task cache:

| Loss-focused follow-up | Answer result | Truncation | Request-wide result |
| --- | --- | ---: | ---: |
| 512 tokens, 5 tasks, 3 alternating repetitions | 5/5 target/DSpark parity in every repetition; 0 gains; 0 losses | 1 task per mode | 24.967 vs **30.521 token/s**; **1.222x** |
| 1,024 tokens, 5 tasks, 1 pair | **5/5 untruncated target/DSpark parity**; both modes 4/5 | 0 tasks | quality-only capture |

The original three lenient losses were therefore truncation-sensitive, not a
reproduced DSpark answer regression. One task that appeared correct for the
target at 256 tokens continued to a different, wrong explicit answer at the
larger budget in both modes, demonstrating why a cutoff-position answer is not
a sound quality oracle. This is focused diagnostic evidence, not a new general
accuracy estimate.

Complete outputs still differed on all five selected tasks. Batched
target-verification and serial target-only CUDA shapes can follow different
deterministic floating-point trajectories while every committed speculative
token remains target-authoritative. The adaptive profile therefore remains
opt-in; target-only remains the strict production default until the desired
contract is defined and passed. See the
[follow-up evidence and exact commands](quality/README.md#adaptive-truncation-follow-up).

Verify the path-free package without the model or an NVIDIA GPU:

```powershell
py -3.13 .\tools\dspark_adaptive_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\adaptive\evidence.json `
  --json
```

Adding `--require-production-default` is expected to fail with exact-output
parity `55/100`. The checked-in [adaptive evidence](adaptive/evidence.json)
pins the clean commit, binaries, input and artifact digests, peak samples,
host controls, raw-capture hashes, runtime telemetry, and all 100 task pairs.
The separate [follow-up evidence](quality/followup-evidence.json) pins the
clean loss-focused 512- and 1,024-token captures and verifies their selected
answer non-regression while keeping exact-output admission closed.

## Representative 100-task diagnostic

The peak prompt is intentionally easy to predict. A separate capture therefore
ran the fixed 50-task MMLU, 20-task GSM8K, and 30-task C-Eval workload three
times per mode in alternating order. Both modes used context 1,024, batch 12,
greedy sampling, the same target bytes, a 256-token maximum, and one excluded
warm-up. All 600 API requests completed successfully.

| Mode | Lenient | Strict | Truncated | Request-wide throughput runs | Mean |
| --- | ---: | ---: | ---: | --- | ---: |
| Q6_K target-only | 67/100 | 58/100 | 40/100 | 22.450, 22.707, 22.696 token/s | 22.618 token/s |
| Q6_K + DSpark Q4 K10/S6 | **73/100** | **59/100** | 40/100 | 32.546, 32.356, 33.133 token/s | **32.678 token/s** |

The measured workload speedup is **1.445x**. Within each mode, every prediction
and complete output digest was stable across all three repetitions. In the
paired comparison DSpark had six lenient gains and no losses (`p=0.03125`), two
strict gains and one loss (`p=1.0`), 91/100 extracted-answer parity, and 54/100
complete output-hash parity. All 58 requests that were untruncated in both
modes retained the same extracted answer.

This does **not** show an intelligence loss, and it does not show that DSpark
improves intelligence. The score increase is a property of this fixed sample.
More importantly for the runtime contract, deterministic DSpark and target-only
execution followed different complete token trajectories on 46 tasks. The
K10/S6 profile therefore fails the exact-output production-default gate even
though every committed speculative token was target-verified.

Workload acceptance was 44.726% (14,719 accepted of 32,909 drafted), with
3.674 verified tokens per target pass. Each DSpark repetition recorded 100
fallback replays and 100 rollback-guard activations. By domain, acceptance was
46.218% for MMLU, 55.395% for GSM8K, and 37.197% for C-Eval. These values
explain why the 169.324 token/s peak cannot be generalized: proposal coverage,
replay cost, and output length dominate mixed workloads.

The path-free [quality evidence package](quality/evidence.json) pins the clean
source and server, both model digests, task and configuration identities, six
raw-report hashes, GPU admission samples, per-run aggregates, and all paired
task vectors. Verify it on any machine without the model or an NVIDIA GPU:

```powershell
py -3.13 .\tools\qwen38_quality_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality\evidence.json `
  --json
```

Use `--require-production-default` to apply the lossless release gate. It is
expected to fail this diagnostic because exact output parity is 54/100. The
default verifier succeeds because the evidence is internally consistent and
explicitly classified as `diagnostic-output-divergence`.

## What was optimized

The result is not a DFlash-plus-DSpark stack. DFlash and DSpark define different
draft-model tensor contracts and proposal graphs. Power admits one declared
external draft per target and selects the matching strategy. Labeling a DSpark
artifact as DFlash is rejected by the model inspector because the DSpark-only
Markov and confidence tensors are present.

The accepted DSpark path combines:

- a content-addressed external GGUF draft whose kind, provenance, tensor
  contract, and artifact identity are verified at registration and load, with
  full target/draft vocabulary compatibility checked when contexts bind;
- separate target and draft llama.cpp model instances with one shared exact
  verification transaction;
- ten-token fixed proposals so one expensive target pass can commit about ten
  output tokens;
- six resident recurrent rollback snapshots, with exact committed-prefix
  replay available when a rejected suffix exceeds that window;
- an optional request-local adaptive controller that probes inside S6, jumps
  directly to K10 after one fully accepted probe, and moves sustained low-yield
  requests to target-only without an automatic full-prefix replay;
- stable batch-12 CUDA Graph shapes, full target/draft GPU offload, Flash
  Attention, high-priority CUDA streams, and single-model/single-request
  scheduling;
- path-free loaded-artifact proof in health, process-lifetime speculative
  counters, and per-run GPU telemetry.

Development sweeps showed why K and S must be tuned together. K4/S4 reached
about 118 token/s, K6/S6 141 token/s, and K8/S6 149 token/s. Wider complete
rollback profiles crossed a VRAM allocation cliff: K7/S7 and K8/S8 fell to
roughly 48 and 60 token/s. K10/S5 triggered one exact replay and fell to
101.066 token/s; K10/S6 avoided replay and became the accepted profile.

## DFlash-family status

A genuine DFlash v1 artifact has **not** been benchmarked. The only DFlash-named
diagnostic deliberately forced the DSpark artifact through a DFlash command;
it accepted 10 of 970 proposals (1.031%, mean accepted length 1.04) and ran at
roughly 17.3 token/s. That is negative compatibility evidence, not a DFlash
performance result. A defensible DFlash comparison requires a GGUF whose
declared architecture, block size, target-layer metadata, tokenizer, and
target identity satisfy the DFlash contract.

Consequently:

- DSpark Q4: implemented and measured natively in Power;
- DFlash v1: contract implemented and fail-closed, but no compatible artifact
  has been accepted on this host;
- DFlash2: its distinct GGUF contract is validated, and an
  [exact upstream standalone capture](../dflash2/README.md) exists; native Power
  execution remains fail-closed until the pinned Rust binding is updated;
- DFlash-family + DSpark: not an additive execution mode; a future router may select
  one compatible drafter per model/request, but it must not execute both as one
  proposal graph.

## Fixed identities

| Item | Value |
| --- | --- |
| Power commit | `c272e35365fb25a057a8ee4c04c20d8a35cb4b05` |
| Target | 22,884,408,288 bytes; `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| DSpark Q4 | 1,104,594,816 bytes; `12003c7f2642e2e87e979729e16947a913e2213d82136cb5024a36ec4871fef2` |
| DSpark source | `DimInfer/Qwen3.8-27B-Dspark-v1`, revision `10b6bccfcc109bda0666d0aed4b7871aac357b99`, Apache-2.0 |
| Prompt | `../prompt.txt`; `d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89` |
| Request | `d49166116d50d478cd475287fd90a4b57b98848877119e71636b310c87ec8181` |
| Output | `584e2b93ba21d7c727456567762c6bbacc150d43156c73ed91c1c0cbb13be6eb` |
| Server binary | `c95031269b7626ebd8e843b5e0f25e9e2378b25c3b078ba8df3e26613cb05498` |
| Benchmark client | `c6813ecceaede07aebe82d74cd91c797ca097c430fd8986491d3db6790e26bfd` |
| Host | Windows 11 build 22631; RTX 4090 24,564 MiB; driver 610.74; CUDA UMD 13.3; Xeon w5-2445 |

## Reproduce the capture

### 1. Verify the model files

Place the target and DSpark files at paths of your choice, then verify both
before registration:

```powershell
$target = 'D:\models\a3s-power\qwen38\full\Qwen3.8-27B-Q6_K.gguf'
$draft = 'D:\models\a3s-power\qwen38\dspark\Qwen3.8-27B-DSpark-Q4_K_M.gguf'

(Get-Item -LiteralPath $target).Length
(Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash
(Get-Item -LiteralPath $draft).Length
(Get-FileHash -Algorithm SHA256 -LiteralPath $draft).Hash
```

Edit only the local paths in [the registration body](register-model.example.json)
and the `data_dir` in both ACL files. Do not add client-supplied `size`,
`sha256`, or `target_sha256` fields: Power computes and stores those identities
from the actual files.

### 2. Build the pinned CUDA backend

```powershell
cargo fetch --locked
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\apply-llamacpp-power-patches.ps1

$env:CMAKE_GENERATOR = 'Ninja'
$env:CMAKE_CUDA_ARCHITECTURES = '89'
cargo build --locked --release --no-default-features `
  --features llamacpp-cuda,llamacpp-mtp-fr `
  --target-dir target-native-sm89-ninja `
  --bin a3s-power --bin a3s-power-speculative-bench
```

The patch tool must apply or confirm the dynamic-K, FR, high-priority CUDA, and
external-draft patches. A build with `llamacpp` instead of `llamacpp-cuda` is a
CPU control and cannot reproduce these numbers.

### 3. Register the target and draft once

Start Power temporarily with either checked-in ACL:

```powershell
.\target-native-sm89-ninja\release\a3s-power.exe serve `
  --config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\target-only.acl
```

In a second terminal:

```powershell
$body = Get-Content -Raw -Encoding UTF8 `
  .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\register-model.example.json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:11439/v1/models `
  -ContentType 'application/json' -Body $body
```

Stop the temporary server. Registration parses both GGUF headers, validates the
declared DSpark tensor contract, hashes both files off the async executor, and
writes a manifest bound to the target digest. Runtime binding additionally
compares the target and draft vocabulary type, size, BOS/EOS policy, and every
token string before accepting the decoder pair.

### 4. Run the paired API benchmark

Use a clean worktree and the Windows High performance power plan. The accepted
run used a shared display GPU with 6--9% idle utilization, high process
priority, CUDA high-priority streams, one warm-up, and three measured samples:

```powershell
$common = @{
  Model = 'qwen3.8-27b-q6-k-dspark-q4'
  ModelHash = '562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727'
  PromptFile = '.\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt'
  PowerHome = 'D:\models\a3s-power\dspark-home'
  BenchmarkRoot = 'D:\models\a3s-power\dspark-benchmark'
  Samples = 3
  WarmupRuns = 1
  MaxTokens = 256
  NumCtx = 512
  NumBatch = 12
  ProcessPriority = 'High'
  NvidiaGpuIndices = 0
  MaximumIdleGpuUtilizationPercent = 20
  TargetDirectory = 'target-native-sm89-ninja'
  Port = 11439
  CudaHighPriority = $true
  RequireHighPerformancePowerPlan = $true
  RequireCleanTree = $true
}

.\tools\run-gguf-speculative-benchmark.ps1 @common `
  -Label native-target-only-c512-b12-256t-3x `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\target-only.acl `
  -Mode off

.\tools\run-gguf-speculative-benchmark.ps1 @common `
  -Label native-dspark-q4-k10-s6-c512-b12-256t-3x `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\dspark-q4-k10-s6.acl `
  -Mode dspark -MinimumTokensPerSecond 160 `
  -MinimumSampleTokensPerSecond 160

$adaptive = @{} + $common
$adaptive.MaximumIdleGpuUtilizationPercent = 15
$adaptive.LockGpuClockMHz = 2745
$adaptive.ProcessorAffinityMask = 349525

.\tools\run-gguf-speculative-benchmark.ps1 @adaptive `
  -Label adaptive-dspark-q4-k10-s6-c512-b12-256t-3x `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality-adaptive-k10-s6.acl `
  -Mode dspark -MinimumTokensPerSecond 160 `
  -MinimumSampleTokensPerSecond 160
```

The DSpark profile peaked at 23,847 MiB, leaving only 717 MiB on this GPU.
Model load can fail under extra WDDM allocations even when utilization is low.
Treat a quiet GPU and adequate free VRAM as admission requirements; if the host
cannot provide that margin, reduce context or proposal shape and record a new
result rather than presenting an unstable K10/S6 number.

### 5. Run the representative quality matrix

Use the reviewed offline task cache so dataset drift cannot alter the sample.
The runner starts a fresh server for every mode, rotates execution order, and
requires three consecutive idle-GPU samples before every server launch:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\run-qwen38-quality-matrix.ps1 `
  -Q6PowerHome D:\models\a3s-power\dspark-home `
  -Profile dspark-q4 `
  -PreparedTaskCache .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\quality\tasks-v1.json `
  -TargetDirectory target-native-sm89-ninja `
  -OutputRoot D:\models\a3s-power\quality-dspark-replay `
  -Model qwen3.8-27b-q6-k-dspark-q4 `
  -Repetitions 3 -NumCtx 1024 -NumBatch 12 -MaxTokensCap 256 `
  -ProcessPriority High -ProcessorAffinityMask 349525 `
  -NvidiaGpuIndex 0 -MaximumIdleGpuUtilizationPercent 20 `
  -MinimumIdleGpuMemoryFreeMiB 23000 `
  -IdleGpuSampleCount 3 -IdleGpuSampleIntervalMilliseconds 500 `
  -IdleGpuWaitSeconds 120 `
  -RequireHighPerformancePowerPlan -RequireCleanTree
```

Replay the current adaptive matrix with its attested host controls:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\run-qwen38-quality-matrix.ps1 `
  -Q6PowerHome D:\models\a3s-power\dspark-home `
  -Profile dspark-q4 `
  -RuntimeConfig .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality-adaptive-k10-s6.acl `
  -PreparedTaskCache .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\quality\tasks-v1.json `
  -TargetDirectory target-native-sm89-ninja `
  -OutputRoot D:\models\a3s-power\quality-dspark-adaptive `
  -Model qwen3.8-27b-q6-k-dspark-q4 `
  -Repetitions 1 -NumCtx 1024 -NumBatch 12 -MaxTokensCap 256 `
  -ProcessPriority High -ProcessorAffinityMask 349525 `
  -LockGpuClockMHz 2745 -CudaHighPriority `
  -NvidiaGpuIndex 0 -MaximumIdleGpuUtilizationPercent 15 `
  -MinimumIdleGpuMemoryFreeMiB 23000 `
  -IdleGpuSampleCount 3 -IdleGpuSampleIntervalMilliseconds 500 `
  -IdleGpuWaitSeconds 300 `
  -RequireHighPerformancePowerPlan -RequireCleanTree
```

Package a completed raw capture for review with:

```powershell
py -3.13 .\tools\qwen38_quality_evidence.py capture `
  --capture-root D:\models\a3s-power\quality-dspark-replay `
  --output D:\models\a3s-power\quality-dspark-replay\evidence.json
```

That newly generated package will bind its own raw files, but the checked-in
verifier intentionally recognizes only the published clean capture. Updating
the repository pin requires review of the new source, binary, artifact, GPU,
task, configuration, and report identities.

Package the adaptive peak and quality captures together after reviewing their
raw hashes:

```powershell
py -3.13 .\tools\dspark_adaptive_evidence.py capture `
  --quality-root D:\models\a3s-power\qwen38\benchmark\quality-dspark-adaptive-one-shot-k10s6-c1024-b12-256t-1x-cbdb3f6 `
  --peak-report D:\models\a3s-power\qwen38\benchmark\adaptive-dspark-20260823\dspark-k10s6-adaptive-one-shot-cbdb3f6-clock2745-affinity-peak-3x.json `
  --peak-preflight D:\models\a3s-power\qwen38\benchmark\adaptive-dspark-20260823\dspark-k10s6-adaptive-one-shot-cbdb3f6-clock2745-affinity-peak-3x.preflight.json `
  --peak-environment D:\models\a3s-power\qwen38\benchmark\adaptive-dspark-20260823\dspark-k10s6-adaptive-one-shot-cbdb3f6-clock2745-affinity-peak-3x.environment.json `
  --peak-server-log D:\models\a3s-power\qwen38\benchmark\adaptive-dspark-20260823\dspark-k10s6-adaptive-one-shot-cbdb3f6-clock2745-affinity-peak-3x.stdout.log `
  --output D:\models\a3s-power\adaptive-dspark-evidence.json
```

The published verifier is intentionally pinned to the reviewed source hashes;
a fresh capture is expected to require a source and evidence review before it
can replace the checked-in package.

## Claim boundary

The fixed 169.324 token/s and adaptive 164.756 token/s results are
single-request, short-context, deterministic boundaries on one prompt and one
RTX 4090. The adaptive minimum of 160.881 token/s supports a controlled
160-plus claim, not a 175 token/s floor. Its 100-task diagnostic gained 1.358x
request-wide throughput but included three paired lenient losses and only
55/100 complete-output matches. Both K10/S6 variants are consequently opt-in
and not eligible as lossless production defaults. Stochastic sampling,
concurrency, long contexts, other models, drivers, and GPU memory pressure
require independent quality and performance gates.
