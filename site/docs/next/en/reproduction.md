---
title: Reproduction
description: Reproduce the current A3S Power Q6_K execution-path case study, from 23 offline evidence hashes to a full RTX 4090 replay.
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
| Current shared-desktop capture | **172.8353 token/s median**; 171.2981 minimum; 175.5329 maximum; 1 / 9 samples at least 175 |
| Earlier quiet-host high-water mark | **176.6109 token/s median**; 173.2630 minimum; 7 / 9 samples at least 175 |
| Full-vocabulary control | 147.0207 token/s median, 146.0917 minimum |
| Current 12-task paired calibration | Off 28.713; fixed K6/S6/B8 46.923 token/s; 63.42% gain |
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

The current deep-optimization capture records clean source revision `f6326bb05bb8101c2335ec7c3c2f1e261fd86071`. Replaying from a newer clean revision is valid, but it is a new experiment and must not overwrite the checked-in JSON.

## 2. Run the model-free offline verifier first

This command does not load the 22.88 GB model and does not require an NVIDIA GPU. It verifies 23 file SHA-256 values and recomputes sample counts, medians, minima, quality scores, request-wide throughput, acceptance, replay counts, paired answers, and deterministic output identity:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\verify-qwen38-q6k-evidence.ps1 -Json
```

A passing process exits with code `0` and includes:

```json
{
  "status": "passed",
  "verified_file_hashes": 23,
  "quality": {
    "completed_requests": 900,
    "request_wide_tokens_per_second": 83.22814601950864
  },
  "pure_q6": {
    "full_vocabulary_k7_s7_median": 147.020656574707,
    "prefix_fr8192_k7_s6_median": 176.6108685085471
  },
  "deep_optimization": {
    "peak": {
      "median_decode_tokens_per_second": 172.8353133057359,
      "minimum_decode_tokens_per_second": 171.29810355919784
    },
    "general": {
      "target_only_tokens_per_second": 28.71272184998198,
      "mtp_tokens_per_second": 46.92338764288924,
      "speedup_percent": 63.4236833695329,
      "paired_final_answers": 12,
      "fallback_replays": 0
    }
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
| Driver and CUDA | NVIDIA 610.74; CUDA UMD 13.3; build toolchain pinned by the environment receipt |
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

The patch tool must confirm the binding, MTP/FR, and high-priority CUDA-stream patches. The runner rejects any backend other than exclusive `llama.cpp`.

## 5. Verify the model and inputs

Register the untouched Q6_K artifact as `qwen3.8-27b-q6-k`, then select its Power data directory. These three repository inputs must match exactly:

| Input | SHA-256 |
| --- | --- |
| `prompt.txt` | `d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89` |
| Pure Q6_K full-vocabulary K7/S7 ACL | `eb445101c1e33a035c9b1d120fec12d9b21e6ce1b2fe5486ad46bee52878a588` |
| Current pure-Q6_K prefix-FR8192 K7/S6/B11 ACL | `674d3a36e0f0019c9e39e60994ea40eee0477615827464edee1fb9627a74cdec` |
| Current pure-Q6_K prefix-FR8192 K6/S6/B8 ACL | `b4f3db4229bfad05371bbed0ce1fec165aa2b05279405078aa8f7721721abb37` |

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

Close unnecessary GPU applications and run in a terminal allowed to lock the GPU clock. This command uses a zero performance threshold to capture the real measurement. Only a separate quiet-host service gate should require every sample to reach 175:

```powershell
$benchmarkRoot = 'D:\models\a3s-power\qwen38\benchmark'
$powerHome = 'D:\models\a3s-power\qwen38\power-home'

.\tools\run-qwen38-q6k-benchmark.ps1 `
  -Label pure-q6-fr8192-k7s6-b11-cudahigh `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\pure-q6-mtp7-snap6-fr8192-rtx4090-throughput.acl `
  -PromptFile .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt `
  -BenchmarkRoot $benchmarkRoot `
  -PowerHome $powerHome `
  -ModelHash 562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727 `
  -MaxTokens 1024 -NumBatch 11 -WarmupRuns 1 -Samples 9 `
  -MinimumTokensPerSecond 0 -ProcessPriority High `
  -ProcessorAffinityMask 349525 -LockGpuClockMHz 2745 `
  -CudaHighPriority `
  -MaximumIdleGpuUtilizationPercent 8 -IdleGpuSampleCount 3 `
  -TargetDirectory target-native-sm89-ninja `
  -RequireHighPerformancePowerPlan -RequireCleanTree
```

The runner restores the GPU clock in `finally` and retains failed reports, so environment contention can be distinguished from a real performance regression.

## 7. Accept the new result

Retain these files under `$benchmarkRoot`:

- `pure-q6-fr8192-k7s6-b11-cudahigh.json`: nine raw samples, statistics, and output digest;
- `pure-q6-fr8192-k7s6-b11-cudahigh.environment.json`: Git, binary, model, ACL, prompt, GPU, process, and power identities;
- `pure-q6-fr8192-k7s6-b11-cudahigh.preflight.json`: startup identity and host-control checks;
- matching stdout and stderr logs: backend initialization and failure diagnosis.

The capture is valid only when all nine requests generate 1,024 tokens, every output SHA-256 is identical, model identity matches exactly, the backend is exclusive, the worktree is clean, and stream priority, affinity, clock, and power controls actually took effect. The caller decides whether it meets a host-specific deployment SLO; the historical 176.61 result is not an automatic failure threshold for the current shared desktop.

## 8. Verify the Q6_K-only DFlash2 prototype

Both paired modes use the same 22.88 GB Q6_K target. The 1.14 GB Q4 DFlash2
artifact is an auxiliary proposer, never a target result. Verify the path-free
peak and three-order quality package without a model or GPU:

```powershell
py -3.13 .\tools\dflash2_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\evidence.json `
  --json
```

The verifier recomputes 35.380 versus 108.429 token/s on the repetitive prompt,
29.702 versus 45.143 token/s on the fixed 12-task workload, 12/12 answer
parity, and 7/12 complete-output parity. Adding
`--require-production-default` must fail because the output gate is not exact
and the pinned Power binding does not execute DFlash2. Use the
[full DFlash2 guide](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dflash2)
for upstream PR 27342, artifact/runtime hashes, CUDA build flags, host controls,
and exact paired commands.

## 9. Reproduce the native DSpark gate

The external-DSpark package is a separate paired experiment. It keeps the same
22,884,408,288-byte Q6_K target and binds the 1,104,594,816-byte DSpark Q4
artifact with SHA-256
`12003c7f2642e2e87e979729e16947a913e2213d82136cb5024a36ec4871fef2`.
Run the four model-free verifiers first:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\verify-dspark-evidence.ps1 -Json

py -3.13 .\tools\qwen38_quality_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality\evidence.json `
  --json

py -3.13 .\tools\dspark_adaptive_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\adaptive\evidence.json `
  --json

py -3.13 .\tools\dspark_quality_followup_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality\followup-evidence.json `
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

The third verifier binds the current clean request-local capture. It
recomputes a 164.756 token/s peak median and 160.881 minimum, identical peak
output and receipt hashes, zero replay, the 22.872 versus 31.052 token/s
quality-workload rates, 1.358x speedup, and all 100 paired task vectors. The
candidate recorded five lenient gains and three losses and only 55/100
complete-output parity, so `--require-production-default` must fail here too.

The fourth verifier binds the clean loss-focused follow-up at commit
`7bdeb960f5a38ea7515c67a12636a29198fd95f6`. It checks three alternating
512-token repetitions and one 1,024-token pair. Every repetition retained 5/5
paired answers with zero gains and zero losses; the 1,024-token pair completed
all five tasks with 5/5 answer parity. The 512-token workload reached 30.521
versus 24.967 token/s. Complete output parity remained 0/5, so adding
`--require-production-default` is expected to fail.

Reproduce those two matrices with the hash-locked
`dspark/quality/divergence-v1.selection.json` file and
`run-qwen38-quality-matrix.ps1`: use `-MaxTokensOverride 512 -NumCtx 1024
-Repetitions 3` for the performance follow-up, then `-MaxTokensOverride 1024
-NumCtx 2048 -Repetitions 1` for the no-truncation quality check. The exact
host controls and complete command are in the
[follow-up protocol](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark/quality/README.md#adaptive-truncation-follow-up).

The exact performance commands in the package attest the requested 2745 MHz
GPU clock lock, high-priority CUDA streams, High process priority, `0x55555`
affinity, clean worktree, idle GPU, and at least 23,000 MiB free VRAM. Those
controls are evidence from this host, not portable defaults for other CPU or
GPU topologies.

Use the [DSpark reproduction package](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark)
for the typed registration body, target-only and DSpark ACL files, raw reports,
artifact revisions, and exact paired runner commands. DFlash v1 is not part of
that result: DFlash v1 and DSpark are alternative artifact contracts, and no
compatible DFlash v1 GGUF has completed this gate. DFlash2 uses the separate
prototype package in step 8.

The complete [Windows/CUDA guide](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md) also covers the paired full-vocabulary control, the 12-task pure-Q6_K calibration, and the previous mixed-artifact gates. The [quality-matrix protocol](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/README.md#reproduce) explains how to rerun the existing 100-task × 3-run evaluation.
