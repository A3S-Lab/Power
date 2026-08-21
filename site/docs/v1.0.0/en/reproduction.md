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
| Work shape | 1 warm-up + 9 measured requests; 1,024 generated tokens each; batch 14; greedy |
| Steady gate | 175 token/s median; committed result **176.6109**, minimum 173.2630; 7 / 9 samples at least 175 |
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

The current pure-Q6_K gate records clean source revision `eb6aeda59561eff3e4e7592704cab6fc863b72c7`. Replaying from a newer clean revision is valid, but it is a new experiment and must not overwrite the checked-in JSON.

## 2. Run the model-free offline verifier first

This command does not load the 22.88 GB model and does not require an NVIDIA GPU. It verifies 14 file SHA-256 values and recomputes sample counts, medians, minima, quality scores, request-wide throughput, acceptance, replay counts, and deterministic output identity:

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

The 176.61 token/s value is a boundary on this host, not a cross-hardware promise:

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

The complete [Windows/CUDA guide](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md) also covers the paired full-vocabulary control, the 12-task pure-Q6_K calibration, and the previous mixed-artifact gates. The [quality-matrix protocol](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/README.md#reproduce) explains how to rerun the existing 100-task × 3-run evaluation.
