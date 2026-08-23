# Q6_K-only DFlash2 experiment

This capture answers one narrow question: can DFlash2 accelerate the unchanged
Qwen3.8-27B Q6_K target on this RTX 4090 host?

The target is always the same 22,884,408,288-byte Q6_K GGUF. The 1.14 GB Q4
DFlash2 file is an auxiliary proposer only; it is never scored or reported as
the target model. No Q4 target result is included in this directory.

## Result

The exact upstream DFlash2 llama.cpp revision was tested in a standalone paired
runner. Power's pinned `llama-cpp-rs` binding does not contain that revision,
so native Power execution deliberately fails closed. These numbers are
experimental backend evidence, not a native Power service claim.

| Fixed target and execution mode | Quality proxy | Request-wide throughput | Repetitive-prompt steady decode |
| --- | --- | ---: | ---: |
| Q6_K target only | 9/12 lenient and strict in every repetition | 29.702 token/s mean; 30.134 median | 35.380 token/s median |
| Q6_K target + DFlash2 Q4 proposer, K7 | 9/12 lenient and strict in every repetition | **45.143 token/s mean; 47.155 median** | **108.429 token/s median** |

The mixed 12-task workload ran three times per mode in alternating order: 72
successful requests in total. DFlash2 increased mean request-wide throughput
by 1.520x. Its three candidate runs were 54.586, 33.689, and 47.155 token/s,
so this capture does not establish a stable production floor.

The steady-decode gate used one deterministic integer-sequence prompt, one
warm-up, and three 256-token samples. DFlash2 produced 103.742, 108.429, and
110.209 token/s versus 35.284, 35.380, and 35.387 token/s for target-only. It
accepted 666 of 678 proposals (98.230%), verified 7.727 emitted tokens per
target pass, and matched the target-only output digest in all six measured
samples. That 3.065x result is a high-coverage synthetic boundary, not general
chat, coding, RAG, or Agent throughput.

## Quality boundary

The quality calibration contains four MMLU, four GSM8K, and four C-Eval tasks.
Both modes retained 9/12 lenient and strict answers in all three repetitions.
Across the paired tasks there were zero answer gains, zero answer losses, and
12/12 extracted-answer parity. Nine tasks were untruncated in both modes and
all nine retained the same answer.

Complete response hashes matched on only 7/12 tasks. Batched target
verification can follow a different finite-precision CUDA trajectory from
serial target-only decoding even when every committed token is target
verified. The result therefore shows no observed score regression on this
small fixed calibration; it does not prove general intelligence equivalence
and it is not eligible as a lossless production default.

On the mixed workload DFlash2 accepted 1,754 of 3,246 proposals (54.036%),
verified 4.964 tokens per target pass, and recorded zero fallback replay.
Loaded GPU memory was 23,781--23,911 MiB for the candidate versus
21,866--22,078 MiB for target-only. The remaining physical headroom was too
small for another large resident graph or draft model.

## Why this is not 175 token/s

DFlash2 improves emitted tokens per target forward, but it does not remove the
target forward, proposal graph, synchronization, sampling, or memory traffic.
This implementation proposes at most seven tokens, so even perfect acceptance
emits at most the seven proposals plus one target token per verification pass.
The synthetic prompt nearly reaches that acceptance ceiling and still measures
108.429 token/s. Real mixed prompts reduce acceptance to 54.036% and expose
host/WDDM variance.

Flash Attention was enabled, ordinary CUDA graph reuse was retained, target and
draft layers were fully offloaded, the GPU clock request was 2,745 MHz, and the
server used high process priority with physical-core affinity. A separate
forced-cuBLAS build and extra concurrent-graph tuning regressed the result, so
neither is promoted. Tensor sharing could recover memory when target and draft
layouts permit immutable aliasing, but it does not erase the proposal and
verification compute needed to close the 108-to-175 gap.

## Pinned identities

| Component | Identity |
| --- | --- |
| Power source | `32bc4ea54bc2889e7ada584b4b7ad04616e703f6`, clean |
| llama.cpp DFlash2 source | PR 27342, commit `1deefcca395743049c3820ab8f9b15043f3e9446` |
| `llama-server.exe` | `d4fcedab36dc30795c77ea1990c2d1496d27c15d99797a24af54ee5c2e792910` |
| Q6_K target | 22,884,408,288 bytes; `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| DFlash2 proposer | 1,143,006,752 bytes; `18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594` |
| Peak report | `5f0c66d6a9669fbd85fae64e9d7c43c3217195966dc5a9cd8a7b97bb596da689` |
| Quality environment | `608a7f761e9b4575be7ce2c0c3c49cc9f609954b0474a6aec0f69fefc49ac615` |
| Quality aggregate | `becc158ec023e739d01946084ca8ba4b1863a3c5fdad89228eafbdbc5a802b42` |

The checked-in [path-free evidence](evidence.json) also pins all runtime DLLs,
six raw quality reports, runner/evaluator hashes, GPU admission windows,
process controls, per-run rates, and the 12 paired task vectors. It contains no
local model path or GPU UUID.

## Verify without a GPU

The offline verifier needs only Python and this repository:

```powershell
py -3.13 .\tools\dflash2_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\evidence.json `
  --json
```

Applying the production-default gate is expected to fail because complete
output parity is 7/12 and native Power DFlash2 execution is unavailable:

```powershell
py -3.13 .\tools\dflash2_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\evidence.json `
  --require-production-default
```

## Reproduce the raw measurements

Use a clean Power checkout at commit `32bc4ea54bc2889e7ada584b4b7ad04616e703f6`,
Windows 11, CUDA, Visual Studio 2022 Build Tools, CMake, and Ninja. Fetch and
build the exact upstream proposal revision in a Developer PowerShell:

```powershell
$llamaSource = Join-Path $PWD 'llama-dflash2-pr27342'
$llamaBuild = Join-Path $PWD 'llama-dflash2-sm89'

git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git $llamaSource
git -C $llamaSource fetch origin refs/pull/27342/head
git -C $llamaSource checkout --detach 1deefcca395743049c3820ab8f9b15043f3e9446

cmake -S $llamaSource -B $llamaBuild -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_ARCHITECTURES=89 `
  -DGGML_CUDA=ON `
  -DGGML_CUDA_FA=ON `
  -DGGML_CUDA_FA_ALL_QUANTS=OFF `
  -DGGML_CUDA_FORCE_CUBLAS=OFF `
  -DGGML_CUDA_FORCE_MMQ=OFF `
  -DLLAMA_CURL=OFF
cmake --build $llamaBuild --target llama-server -j 20
```

Set the three local artifact paths, verify their hashes, then run the paired
steady-decode gate:

```powershell
$llamaBin = Join-Path $llamaBuild 'bin'
$target = '<path-to-Qwen3.8-27B-Q6_K.gguf>'
$draft = '<path-to-Qwen3.8-27B-DFlash2-Q4_K_M.gguf>'
$captureRoot = '<empty-output-directory>'

Get-FileHash $target -Algorithm SHA256
Get-FileHash $draft -Algorithm SHA256

.\tools\run-llamacpp-external-draft-benchmark.ps1 `
  -LlamaBinDirectory $llamaBin `
  -TargetModel $target `
  -TargetSha256 562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727 `
  -DraftModel $draft `
  -DraftSha256 18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594 `
  -LlamaCppCommit 1deefcca395743049c3820ab8f9b15043f3e9446 `
  -PromptFile .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt `
  -Output (Join-Path $captureRoot 'performance\report.json') `
  -DraftMode dflash2 -Samples 3 -WarmupRuns 1 -MaxTokens 256 `
  -ContextSize 512 -BatchSize 12 -Threads 10 -DraftMax 7 `
  -TargetGpuLayers all -DraftGpuLayers all -ServerVerbosity 3 `
  -ProcessPriority High -ProcessorAffinityMask 0x55555 `
  -LockGpuClockMHz 2745 -MaximumIdleGpuUtilizationPercent 15 `
  -MinimumIdleGpuMemoryFreeMiB 23000 -IdleGpuSampleCount 3 `
  -IdleGpuSampleIntervalMilliseconds 500 -IdleGpuWaitSeconds 120 `
  -RequireHighPerformancePowerPlan -RequireCleanTree
```

Run the balanced calibration. The runner automatically alternates mode order
and uses the hash-locked `calibration-v1.selection.json` file:

```powershell
.\tools\run-llamacpp-external-draft-quality.ps1 `
  -LlamaBinDirectory $llamaBin `
  -TargetModel $target `
  -TargetSha256 562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727 `
  -DraftModel $draft `
  -DraftSha256 18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594 `
  -LlamaCppCommit 1deefcca395743049c3820ab8f9b15043f3e9446 `
  -OutputDirectory (Join-Path $captureRoot 'quality') `
  -DraftMode dflash2 -Repetitions 3 -ContextSize 1024 `
  -BatchSize 12 -Threads 10 -DraftMax 7 -MaxTokensCap 256 `
  -TargetGpuLayers all -DraftGpuLayers all -ServerVerbosity 3 `
  -ProcessPriority High -ProcessorAffinityMask 0x55555 `
  -LockGpuClockMHz 2745 -MaximumIdleGpuUtilizationPercent 15 `
  -MinimumIdleGpuMemoryFreeMiB 23000 -IdleGpuSampleCount 3 `
  -IdleGpuSampleIntervalMilliseconds 500 -IdleGpuWaitSeconds 120 `
  -RequireHighPerformancePowerPlan -RequireCleanTree
```

The historical evidence packager intentionally accepts only the pinned raw
hashes. Reconstruct it from those exact outputs with:

```powershell
py -3.13 .\tools\dflash2_evidence.py capture `
  --performance-report (Join-Path $captureRoot 'performance\report.json') `
  --quality-root (Join-Path $captureRoot 'quality') `
  --output (Join-Path $captureRoot 'evidence.json')
```

A new source revision or artifact produces a new evidence generation; do not
overwrite this historical contract with incomparable results.
