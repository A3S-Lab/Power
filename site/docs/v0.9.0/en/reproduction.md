---
title: Reproduction
description: Reproduce the A3S Power Qwen3.8-27B boundary, from offline evidence verification to a full RTX 4090 replay.
---

# Reproduction

This protocol separates reproduction into two acceptance levels. **Offline evidence verification** runs on any machine. A host matching the fixed hardware and software controls can continue with a **performance replay** that emits a new environment receipt. Both levels use the same model, prompt, ACL, and output identities.

:::warning Boundary of the claim
The historical raw capture discloses a dirty worktree, so a clean source checkout cannot rebuild its byte-identical historical binary. Offline verification exactly recomputes the committed evidence. A performance replay validates the same behavior and gate, but must retain its new Git revision, binary hashes, and environment receipt.
:::

## Acceptance baseline

| Item | Fixed value |
| --- | --- |
| Model artifact | Q6_K-derived TBQ4 mixed artifact, 19,187,686,464 bytes |
| Model SHA-256 | `5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6` |
| Balanced mode | Full-vocabulary MTP K7/S7, rollback-complete, zero replay |
| Work shape | 1 warm-up + 9 measured requests; 1,024 generated tokens each; batch 14; greedy |
| Steady gate | 175 token/s median; committed result 175.2089, minimum 174.2211 |
| Request-wide throughput | 83.228 token/s across a 100-task × 3-run quality matrix |
| Fixed-task quality | 76/100 lenient; 66/100 strict; 900/900 requests completed |
| Proposal acceptance | 51.33% |
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

The committed steady-decode gate records source revision `f6a26a7eb51dc5a73b5aa5d0ff1f4f388f32606a`; the complete quality matrix records `4406c9c5aa67b8ad861898866e04d7dfbf4cbf2b`. Replaying from a newer clean revision is valid, but it is a new experiment and must not overwrite the historical JSON.

## 2. Run the model-free offline verifier first

This command does not load the 19.19 GB model and does not require an NVIDIA GPU. It verifies six file SHA-256 values and recomputes sample counts, medians, minima, quality scores, request-wide throughput, acceptance, replay counts, and deterministic output identity:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\verify-qwen38-q6k-evidence.ps1 -Json
```

A passing process exits with code `0` and includes:

```json
{
  "status": "passed",
  "verified_file_hashes": 6,
  "quality": {
    "completed_requests": 900,
    "request_wide_tokens_per_second": 83.22814601950864
  },
  "steady_decode": {
    "rollback_complete_k7_s7_median": 175.20889378841997
  }
}
```

Any changed evidence byte or statistic produces a nonzero exit code and names the mismatched field.

## 3. Match the acceptance host

The 175 token/s value is a boundary on this host, not a cross-hardware promise:

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

Register the mixed artifact as `qwen3.8-27b-q6-k`, then select its Power data directory. These three repository inputs must match exactly:

| Input | SHA-256 |
| --- | --- |
| `prompt.txt` | `d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89` |
| K7/S7 ACL | `759ef6e5e60a08939ed747558992fa3031d63d2ecd59dacfdae59790cc6ff79a` |
| K7/S6 ACL | `2f348cca96282a22650d9766cffa81251ea10a5e34a089bcc91b0822ab5c1d0e` |

```powershell
$powerHome = 'D:\models\a3s-power\qwen38\power-home-tbq4-0-ffn'
$manifestPath = Join-Path $powerHome `
  'models\manifests\qwen3.8-27b-q6-k.json'
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$model = Get-Item -LiteralPath $manifest.path

if ($model.Length -ne 19187686464) { throw 'Unexpected model size' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $model.FullName).Hash -ne
    '5F578B395F61DCAAC9698FE222D988F461FD902CE9494E8A06D8B9AAE4E7E2A6') {
  throw 'Unexpected model hash'
}
```

## 6. Replay balanced K7/S7 end to end

Close GPU-consuming applications and run in a terminal allowed to lock the GPU clock:

```powershell
$benchmarkRoot = 'D:\models\a3s-power\qwen38\benchmark'
$powerHome = 'D:\models\a3s-power\qwen38\power-home-tbq4-0-ffn'

.\tools\run-qwen38-q6k-benchmark.ps1 `
  -Label rollback-complete-s7-affinity-1024-9x-replay `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\mtp7-snap7-full-vocab-cpu-embedding.acl `
  -PromptFile .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt `
  -BenchmarkRoot $benchmarkRoot `
  -PowerHome $powerHome `
  -ModelHash 5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6 `
  -MaxTokens 1024 -NumBatch 14 -WarmupRuns 1 -Samples 9 `
  -MinimumTokensPerSecond 175 -ProcessPriority High `
  -ProcessorAffinityMask 349525 -LockGpuClockMHz 2745 `
  -TargetDirectory target-native-sm89-ninja `
  -RequireHighPerformancePowerPlan -RequireCleanTree
```

The runner restores the GPU clock in `finally` and retains failed reports, so environment contention can be distinguished from a real performance regression.

## 7. Accept the new result

Retain these files under `$benchmarkRoot`:

- `rollback-complete-s7-affinity-1024-9x-replay.json`: nine raw samples, statistics, and output digest;
- `rollback-complete-s7-affinity-1024-9x-replay.environment.json`: Git, binary, model, ACL, prompt, GPU, process, and power identities;
- matching stdout and stderr logs: backend initialization and failure diagnosis.

The replay passes only when all nine requests generate 1,024 tokens, the steady median is at least 175 token/s, every output SHA-256 is identical, model identity matches exactly, the backend is exclusive, the worktree is clean, and the requested host controls actually took effect.

The complete [Windows/CUDA guide](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md) also covers the guarded K7/S6 peak gate and source validation. The [quality-matrix protocol](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/README.md#reproduce) explains how to rerun the 100-task × 3-run evaluation.
