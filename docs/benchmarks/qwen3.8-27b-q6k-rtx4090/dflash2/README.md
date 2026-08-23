# Q6_K-only native DFlash2 evidence

This capture answers one narrow question: can native Power DFlash2 accelerate
the unchanged Qwen3.8-27B Q6_K target on this RTX 4090 host?

The target in every row is the same 22,884,408,288-byte Q6_K GGUF. The 1.14 GB
Q4 DFlash2 file is an auxiliary proposer only: it is never used as the target,
never scored as the target, and never allowed to commit a token without Q6_K
verification. This directory contains no Q4 target result.

## Native Power result

Power commit `72a1ecd352fee1a9416241ccb3ca298ebcde5e09` contains the typed
DFlash2 binding and the reviewed runtime port to the pinned llama.cpp source.
The paired capture changed only `spec_mode` between `off` and `dflash2`.

| Unchanged Q6_K target | Median decode | Minimum decode | Median end-to-end | Minimum end-to-end |
| --- | ---: | ---: | ---: | ---: |
| Target-only | 33.075 token/s | 32.938 token/s | 25.744 token/s | 25.593 token/s |
| **Native DFlash2, K7/S6** | **144.453 token/s** | **141.267 token/s** | **63.182 token/s** | **62.636 token/s** |

The native decode speedup is **4.367x** and the median end-to-end speedup is
**2.454x**. `Decode` measures the intervals between generated tokens after the
first token; `end-to-end` includes prompt processing, startup latency inside
the request, generation, and HTTP overhead. The two rates are not
interchangeable.

The five DFlash2 decode samples were 141.267, 144.052, 144.836, 145.149, and
144.453 token/s. The five target-only samples were 33.056, 33.075, 32.938,
33.213, and 33.213 token/s. Every measured request generated 256 tokens and all
ten outputs had SHA-256
`584e2b93ba21d7c727456567762c6bbacc150d43156c73ed91c1c0cbb13be6eb`.
The DFlash2 runtime logs recorded 222 accepted proposals out of 226
(98.230%), 7.727 emitted tokens per target pass, and zero fallback replay in
each measured request. The warm-up is retained in the raw capture but excluded
from the five-sample statistics.

This is a high-acceptance integer-sequence boundary, not general chat, coding,
RAG, or Agent throughput. It establishes native execution and exact output for
this deterministic request. It does not establish a stable 175 token/s floor
or lossless behavior across arbitrary prompts.

Checked-in evidence:

- [native-target-only.json](native-target-only.json): five target-only samples;
- [native-dflash2-k7-s6.json](native-dflash2-k7-s6.json): five native DFlash2 samples;
- [native-comparison.json](native-comparison.json): canonical report hashes,
  speedup, threshold status, and cross-mode output parity.

## Representative quality boundary

The broader quality result predates the native binding and used the exact
upstream llama.cpp DFlash2 implementation in a standalone paired runner. It is
still the relevant quality warning because the native port contains the same
proposal graph and target-verification algorithm.

| Fixed target and execution mode | Quality proxy | Request-wide throughput | Repetitive-prompt steady decode |
| --- | --- | ---: | ---: | ---: |
| Q6_K target only | 9/12 lenient and strict in every repetition | 29.702 token/s mean; 30.134 median | 35.380 token/s median |
| Q6_K target + DFlash2 Q4 proposer, K7 | 9/12 lenient and strict in every repetition | **45.143 token/s mean; 47.155 median** | **108.429 token/s median** |

The fixed MMLU/GSM8K/C-Eval workload ran three times per mode in alternating
order: 72 successful requests in total. DFlash2 increased mean request-wide
throughput by 1.520x. Its three candidate runs were 54.586, 33.689, and 47.155
token/s, so this capture does not establish a stable production floor.

Both modes retained 9/12 lenient and strict answers in all three repetitions.
Across paired tasks there were zero answer gains, zero answer losses, 12/12
extracted-answer parity, and 9/9 parity where neither response was truncated.
Complete response hashes matched on only 7/12 tasks. Batched target
verification can follow a different finite-precision CUDA trajectory from
serial target-only decoding even though every committed token is target
verified. The small fixed calibration therefore shows no observed score
regression, but it does not prove general intelligence equivalence and does not
qualify DFlash2 as a lossless production default.

On the mixed workload DFlash2 accepted 1,754 of 3,246 proposals (54.036%),
verified 4.964 tokens per target pass, and recorded zero fallback replay.
Loaded GPU memory was 23,781--23,911 MiB for the candidate versus
21,866--22,078 MiB for target-only. A quiet 24 GB device is required.

## Why this is not 175 token/s

DFlash2 raises the number of useful tokens produced by one target forward; it
does not remove the proposal graph, target verification, synchronization,
sampling, or memory traffic. K7 can emit at most seven accepted proposals plus
one target token per verification pass. The native boundary already accepts
98.230% of proposals and emits 7.727 tokens per target pass. Only 1.770
percentage points of proposal-acceptance headroom remain, while reaching 175
from 144.453 token/s requires roughly 21.1% more decode throughput.

The next useful work is therefore lower proposal/verification latency, stable
CUDA Graph shapes, sampling overhead, and exclusive device scheduling—not
claiming that acceptance alone can close the gap. Flash Attention, ordinary
CUDA Graph reuse, full target/draft offload, a 2,745 MHz clock request, a
high-priority CUDA stream, High process priority, and physical-core affinity
were already active. A separate forced-cuBLAS build and extra concurrent-graph
tuning regressed the historical result and are not promoted.

Tensor sharing can recover memory when target and proposer layouts permit safe
immutable aliasing. It cannot erase the proposal and target compute, and the
current artifacts do not claim such aliasing.

## Pinned identities

### Native Power capture

| Component | Identity |
| --- | --- |
| Power source | `72a1ecd352fee1a9416241ccb3ca298ebcde5e09`, clean |
| llama-cpp-rs source | `dfd12e4d334846367e4284a2a7763fe92c1bf676` |
| Nested llama.cpp base | `e79e4bf660e19f2ad851e06c6913f7a8c5852621` |
| DFlash2 runtime source | upstream commits `5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4` and `1deefcca395743049c3820ab8f9b15043f3e9446`, reviewed port |
| `a3s-power.exe` | `eb5122c7fbb02cd736f94d40e74f404fa44da535e4e54a87496ec685f2b511a5` |
| `a3s-power-speculative-bench.exe` | `f6bd00817b4810299e6ab4cb1b93caf5a7ca4448a81decd72aae1af49f5376d6` |
| Q6_K target | 22,884,408,288 bytes; `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| DFlash2 proposer | 1,143,006,752 bytes; `18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594` |
| Prompt | `d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89` |
| Canonical target-only report | `1a5628c50fb8d8e4fdb91c9b48f3e6a0dba143fe514dab96453b15036d893408` |
| Canonical DFlash2 report | `50fd88f282713c675976e2c0255f706eb5e9fa688093cb47d6950359a81891ae` |
| Host | Windows x86_64; RTX 4090; driver 610.74; CUDA SM89 build |

### Historical quality capture

| Component | Identity |
| --- | --- |
| Power source | `32bc4ea54bc2889e7ada584b4b7ad04616e703f6`, clean |
| Upstream llama.cpp DFlash2 | PR 27342, commit `1deefcca395743049c3820ab8f9b15043f3e9446` |
| `llama-server.exe` | `d4fcedab36dc30795c77ea1990c2d1496d27c15d99797a24af54ee5c2e792910` |
| Peak report | `5f0c66d6a9669fbd85fae64e9d7c43c3217195966dc5a9cd8a7b97bb596da689` |
| Quality environment | `608a7f761e9b4575be7ce2c0c3c49cc9f609954b0474a6aec0f69fefc49ac615` |
| Quality aggregate | `becc158ec023e739d01946084ca8ba4b1863a3c5fdad89228eafbdbc5a802b42` |

The historical [path-free evidence](evidence.json) pins the raw quality
reports, runtime DLLs, runner/evaluator hashes, GPU admission windows, process
controls, per-run rates, and all paired task vectors. It contains no local
model path or GPU UUID.

## Verify without a GPU

Recompute the native comparison from the two checked-in reports:

```powershell
a3s-power-speculative-bench compare `
  .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\native-target-only.json `
  .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\native-dflash2-k7-s6.json
```

Verify the historical quality package with Python:

```powershell
py -3.13 .\tools\dflash2_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\evidence.json `
  --json
```

Applying `--require-production-default` is expected to fail because complete
output parity in the representative quality calibration is 7/12. Native
execution availability is no longer a failure reason.

## Reproduce native Power measurements

Use a clean Power checkout at commit
`72a1ecd352fee1a9416241ccb3ca298ebcde5e09`, Windows 11, CUDA 12.6,
Visual Studio 2022 Build Tools, CMake, Ninja, and libclang. Apply the reviewed
patch stack before compiling; the installer invalidates stale Cargo dependency
artifacts whenever it changes the fetched source.

```powershell
cargo fetch --locked
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\apply-llamacpp-power-patches.ps1

$env:CMAKE_GENERATOR = 'Ninja'
$env:CMAKE_CUDA_ARCHITECTURES = '89'
cargo build --locked --release --bins `
  --target-dir target-native-sm89-ninja `
  --no-default-features `
  --features llamacpp-cuda,llamacpp-mtp-fr
```

Edit [register-model.example.json](register-model.example.json) so its two
paths point to the hash-verified local artifacts. The model registry is chosen
by `A3S_POWER_HOME`, not by the ACL `data_dir` field. Start Power once with the
same home, register through the public API, then stop that server before the
runner starts its own process:

```powershell
$powerHome = 'D:\models\a3s-power\qwen38\power-home-dflash2-native'
$env:A3S_POWER_HOME = $powerHome

.\target-native-sm89-ninja\release\a3s-power.exe serve `
  --config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\native-dflash2-k7-s6.acl

# Run in a second terminal while the server is active.
Invoke-RestMethod -Method Post -ContentType 'application/json' `
  -InFile .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\register-model.example.json `
  -Uri http://127.0.0.1:11441/v1/models
```

Use one external output directory so generated evidence does not dirty the
checkout. Run both modes under the same controls; only the ACL and `-Mode`
change:

```powershell
$captureRoot = 'D:\models\a3s-power\dflash2-native-replay'
$powerHome = 'D:\models\a3s-power\qwen38\power-home-dflash2-native'
$common = @{
  Model = 'qwen3.8-27b-q6-k-dflash2'
  ModelHash = '562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727'
  PromptFile = '.\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt'
  PowerHome = $powerHome
  BenchmarkRoot = $captureRoot
  Samples = 5
  WarmupRuns = 1
  MaxTokens = 256
  NumCtx = 512
  NumBatch = 12
  ProcessPriority = 'High'
  ProcessorAffinityMask = 349525
  LockGpuClockMHz = 2745
  MaximumIdleGpuUtilizationPercent = 15
  IdleGpuSampleCount = 5
  RuntimeGpuSampleIntervalMilliseconds = 100
  TargetDirectory = 'target-native-sm89-ninja'
  Port = 11441
  HardwareLabel = 'rtx-4090-qwen38-q6k-native-dflash2'
  CudaHighPriority = $true
  RequireHighPerformancePowerPlan = $true
  RequireCleanTree = $true
}

.\tools\run-gguf-speculative-benchmark.ps1 @common `
  -Label native-dflash2-k7-s6 `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\native-dflash2-k7-s6.acl `
  -Mode dflash2

.\tools\run-gguf-speculative-benchmark.ps1 @common `
  -Label native-target-only `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\native-target-only.acl `
  -Mode off

.\target-native-sm89-ninja\release\a3s-power-speculative-bench.exe compare `
  (Join-Path $captureRoot 'native-target-only.json') `
  (Join-Path $captureRoot 'native-dflash2-k7-s6.json')
```

The runner records source, binaries, model, ACL, prompt, GPU, power plan,
affinity, clock, idle admission, runtime telemetry, reports, and logs. It resets
the requested clock in `finally`. A replay on different source, model bytes,
driver, hardware, or host load is a new result and must retain its own evidence.

## Reproduce the historical quality measurements

The historical standalone runner remains useful for repeating the broader
quality matrix. Use a clean Power checkout at
`32bc4ea54bc2889e7ada584b4b7ad04616e703f6`, then build exact upstream
llama.cpp PR 27342 at `1deefcca395743049c3820ab8f9b15043f3e9446`:

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

Set the local paths, verify both artifacts, and run the paired steady-decode
gate:

```powershell
$llamaBin = Join-Path $llamaBuild 'bin'
$target = '<path-to-Qwen3.8-27B-Q6_K.gguf>'
$draft = '<path-to-Qwen3.8-27B-DFlash2-Q4_K_M.gguf>'
$captureRoot = 'D:\models\a3s-power\dflash2-upstream-replay'

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

Run the balanced calibration. The runner alternates mode order and uses the
hash-locked `calibration-v1.selection.json` task set:

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

Repackage only the exact pinned raw reports:

```powershell
py -3.13 .\tools\dflash2_evidence.py capture `
  --performance-report (Join-Path $captureRoot 'performance\report.json') `
  --quality-root (Join-Path $captureRoot 'quality') `
  --output (Join-Path $captureRoot 'evidence.json')
```

A new source revision or artifact is a new evidence generation; do not
overwrite the historical quality contract with incomparable results.
