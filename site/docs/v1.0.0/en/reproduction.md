---
title: Reproduction
description: Reproduce the A3S Power Qwen3.8-27B boundary, from offline evidence verification to a full RTX 4090 replay.
---

# Reproduction

This protocol separates reproduction into two acceptance levels. **Offline evidence verification** runs on any machine. A host matching the fixed hardware and software controls can continue with a **performance replay** that emits a new environment receipt. Both levels use the same model, prompt, ACL, and output identities.

:::warning Boundary of the claim
The current untouched-Q6_K capture records a clean source revision and exact binary identities. A replay from a newer revision is still a new experiment and must retain its own Git revision, binary hashes, and environment receipt. Older mixed-artifact captures that disclose a dirty worktree remain historical evidence.
:::

## Acceptance baseline

| Item | Fixed value |
| --- | --- |
| Model artifact | Untouched Q6_K GGUF, 22,884,408,288 bytes |
| Model SHA-256 | `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| Peak mode | Prefix-FR8192 MTP K7/S6; exact target verification |
| Work shape | 1 warm-up + 9 measured requests; 1,024 generated tokens each; batch 11; greedy; short-batch Flash Attention off |
| Latest exact-build result | **174.4133 token/s median**; 172.7230 minimum; 177.1497 maximum; 4 / 9 samples at least 175 |
| Full-vocabulary control | 147.0207 token/s median, 146.0917 minimum |
| 12-task request-wide calibration | Off 29.713; full vocabulary 47.032; prefix FR 37.290 token/s |
| Output SHA-256 | `a54538eaaf6cc0b8b43cbafd489c7779f0f5206c93d5034fd3a16f4366a90523` |

## 1. Acquire the source and freeze experiment identity

Run in Windows PowerShell:

```powershell
git clone https://github.com/A3S-Lab/Power.git
Set-Location Power

$powerCommit = (git rev-parse HEAD).Trim()
$dirtyFiles = @(git status --porcelain)
if ($dirtyFiles.Count -ne 0) {
  throw 'A clean worktree is required'
}
$powerCommit
```

The latest peak capture records clean source revision `da2c1dd5a2c6a573ef8be7789de4a67fdb2a0eb0`; the active quality matrix records `64aef15ddff7232c6261385700c8a912d1ed0963`. Replaying from a newer clean revision is valid, but it is a new experiment and must retain its own evidence.

## 2. Run the model-free offline verifiers first

The active Q6_K-only verifier does not load the 22.88 GB model and does not
require an NVIDIA GPU. It pins the full compact payload and recomputes the clean
600-request quality matrix:

```powershell
py -3.13 .\tools\test_qwen38_q6_quality_evidence.py
py -3.13 .\tools\qwen38_q6_quality_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\quality\pure-q6-rtx4090-3x.evidence.json `
  --json
```

It recomputes 23.642 versus 41.035 request-wide token/s. Exact output parity is
50/100 and strict scoring has two losses, so `--require-lossless` fails and the
MTP mode remains opt-in.

### Verify archived peak and mixed-artifact evidence

The historical verifier checks 14 file SHA-256 values and recomputes the older
peak and mixed-artifact records:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\verify-qwen38-q6k-evidence.ps1 -Json
```

A passing process exits with code `0` and includes:

```json
{
  "status": "passed",
  "verified_file_hashes": 14,
  "quality": {
    "completed_requests": 900,
    "request_wide_tokens_per_second": 83.22814601950864
  },
  "pure_q6": {
    "full_vocabulary_k7_s7_median": 147.020656574707,
    "prefix_fr8192_k7_s6_median": 176.6108685085471
  }
}
```

Any changed evidence byte or statistic produces a nonzero exit code and names the mismatched field.

## 3. Match the acceptance host

These token/s values are boundaries on this host, not cross-hardware promises:

| Layer | Acceptance environment |
| --- | --- |
| OS | Windows 11 build 22631 |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9 |
| CPU | Intel Xeon w5-2445, 10 cores / 20 logical processors |
| Driver and CUDA | NVIDIA 610.74; CUDA 12.6 |
| Toolchain | Rust 1.97.1, supported MSVC, CMake, Ninja, libclang |
| Host controls | High Performance power plan; High process priority; GPU at 2745 MHz |
| CPU affinity | `0x55555` (decimal `349525`, valid only for this CPU topology) |

The same protocol can run after changing the GPU, driver, clock, display load, or CPU topology, but label that result as a new platform. Do not copy an inapplicable affinity mask merely to pass the gate.

## 4. Build the pinned CUDA profile

```powershell
cargo fetch
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\apply-llamacpp-power-patches.ps1

$env:CMAKE_GENERATOR = 'Ninja'
$env:CMAKE_CUDA_ARCHITECTURES = '89'
cargo build --release --bins `
  --target-dir target-native-sm89-ninja `
  --no-default-features `
  --features llamacpp-cuda,llamacpp-mtp-fr
```

The patch tool must confirm that the binding and embedded llama.cpp patches are applied. The runner rejects any backend other than exclusive `llama.cpp`.

## 5. Verify the model and inputs

Register the untouched Q6_K artifact as `qwen3.8-27b-q6-k`, then select its Power data directory. These three repository inputs must match exactly:

| Input | SHA-256 |
| --- | --- |
| `prompt.txt` | `d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89` |
| Pure Q6_K full-vocabulary K7/S7 ACL | `eb445101c1e33a035c9b1d120fec12d9b21e6ce1b2fe5486ad46bee52878a588` |
| Pure Q6_K prefix-FR8192 K7/S6 ACL | `9b1213df972ea3731010a1fa72b0d553ba73da42f31e92eaa4fecd3156cbf2ef` |

```powershell
$powerHome = 'D:\models\a3s-power\qwen38\power-home'
$manifestPath = Join-Path $powerHome `
  'models\manifests\qwen3.8-27b-q6-k.json'
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$model = Get-Item -LiteralPath $manifest.path

if ($model.Length -ne 22884408288) { throw 'Unexpected model size' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $model.FullName).Hash -ne
    '562FBF760503008F118E5DF38DE5B3E97992D1F693F475815631198547486727') {
  throw 'Unexpected model hash'
}
```

## 6. Replay the prefix-FR8192 peak end to end

Close GPU-consuming applications and run in a terminal allowed to lock the GPU clock:

```powershell
$benchmarkRoot = 'D:\models\a3s-power\qwen38\benchmark'
$powerHome = 'D:\models\a3s-power\qwen38\power-home'

.\tools\run-qwen38-q6k-benchmark.ps1 `
  -Label pure-q6-fr8192-k7s6-replay `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\pure-q6-mtp7-snap6-fr8192-host-staged.acl `
  -PromptFile .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt `
  -BenchmarkRoot $benchmarkRoot `
  -PowerHome $powerHome `
  -ModelHash 562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727 `
  -MaxTokens 1024 -NumBatch 14 -WarmupRuns 1 -Samples 9 `
  -MinimumTokensPerSecond 175 -ProcessPriority High `
  -ProcessorAffinityMask 349525 -LockGpuClockMHz 2745 `
  -TargetDirectory target-native-sm89-ninja `
  -RequireHighPerformancePowerPlan -RequireCleanTree
```

The runner restores the GPU clock in `finally` and retains failed reports, so environment contention can be distinguished from a real performance regression.

## 7. Accept the new result

Retain these files under `$benchmarkRoot`:

- `pure-q6-fr8192-k7s6-replay.json`: nine raw samples, statistics, and output digest;
- `pure-q6-fr8192-k7s6-replay.environment.json`: Git, binary, model, ACL, prompt, GPU, process, and power identities;
- matching stdout and stderr logs: backend initialization and failure diagnosis.

The replay passes only when all nine requests generate 1,024 tokens, the steady median is at least 175 token/s, every output SHA-256 is identical, model identity matches exactly, the backend is exclusive, the worktree is clean, and the requested host controls actually took effect.

## 8. Verify the Q6_K-only native DFlash2 capture

Both modes use the same 22.88 GB Q6_K target. The 1.14 GB Q4 DFlash2 artifact
is an auxiliary proposer, never the target. Recompute the native comparison
without a model or GPU:

```powershell
a3s-power-speculative-bench compare `
  .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\native-target-only.json `
  .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\native-dflash2-k7-s6.json
```

It verifies 33.075 versus 144.453 token/s median decode, a 4.367x speedup,
and exact output parity. Median end-to-end throughput is 25.744 versus 63.182
token/s. The [complete DFlash2 guide](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dflash2)
pins clean source commit `72a1ecd`, artifact and binary hashes,
`A3S_POWER_HOME` registration, host controls, quality evidence, and exact
replay commands.

## 9. Reproduce the native DSpark gate

The external-DSpark package is a separate paired experiment. It keeps the same
22,884,408,288-byte Q6_K target and binds the 1,104,594,816-byte DSpark Q4
artifact with SHA-256
`12003c7f2642e2e87e979729e16947a913e2213d82136cb5024a36ec4871fef2`.
Run its model-free verifier first:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\verify-dspark-evidence.ps1 -Json

py -3.13 .\tools\qwen38_quality_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality\evidence.json `
  --json
```

The accepted context-512, batch-12 capture reports 32.249 token/s target-only
and 169.324 token/s with DSpark K10/S6, with a 167.102 token/s minimum. All
three 256-token outputs and receipts match exactly. Peak VRAM is 23,847 MiB,
so a quiet device and adequate free memory are required even when GPU
utilization is low.

The second verifier authenticates the context-1024, batch-12, 600-request
quality capture. It recomputes the 22.618 versus 32.678 token/s workload rates,
1.445x speedup, fixed-task scores, replay telemetry, and all 100 paired task
vectors. Adding `--require-production-default` is expected to fail because only
54/100 complete outputs match; the published K10/S6 matrix is diagnostic, not
a lossless default.

Use the [DSpark reproduction package](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark)
for the typed registration body, target-only and DSpark ACL files, raw reports,
artifact revisions, and exact paired runner commands. DFlash is not part of
that result: DFlash and DSpark are alternative artifact contracts, and no
compatible DFlash GGUF has completed this gate.

The complete [Windows/CUDA guide](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md) also covers the paired full-vocabulary control, the 12-task pure-Q6_K calibration, and the previous mixed-artifact gates. The [quality-matrix protocol](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/README.md#reproduce) explains how to rerun the existing 100-task × 3-run evaluation.
