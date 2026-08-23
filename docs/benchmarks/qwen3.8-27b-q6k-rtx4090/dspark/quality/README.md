# DSpark 100-task quality evidence

[`evidence.json`](evidence.json) is the path-free, machine-readable package for
the context-1024 Q6_K target-only versus external-DSpark K10/S6 capture. It
contains 600 successful API requests: 100 fixed MMLU/GSM8K/C-Eval tasks, two
modes, and three alternating repetitions.

| Mode | Lenient | Strict | Mean request-wide throughput |
| --- | ---: | ---: | ---: |
| Q6_K target-only | 67/100 | 58/100 | 22.618 token/s |
| Q6_K + DSpark Q4 K10/S6 | 73/100 | 59/100 | 32.678 token/s |

The capture is deterministic within each mode and records a 1.445x workload
speedup. Complete target/DSpark outputs match on only 54/100 tasks, so the
package is classified as diagnostic output divergence and is not eligible as
a lossless production default.

Verify every pinned identity and derived metric without a model or GPU:

```powershell
py -3.13 .\tools\qwen38_quality_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality\evidence.json `
  --json
```

`--require-production-default` is expected to reject this capture. See the
[parent report](../README.md#representative-100-task-diagnostic) for analysis,
raw-capture reproduction, and the distinction from the 169.324 token/s peak
prompt.

## Adaptive truncation follow-up

The later adaptive one-run matrix reported three lenient losses and three
strict losses at a 256-token limit. All lenient losses reached or approached
that limit, while every task untruncated in both modes retained the same
extracted answer. The hash-locked
[`divergence-v1.selection.json`](divergence-v1.selection.json) therefore
selects every observed loss plus one positive control. It is a diagnostic
subset, not a replacement for the reviewed 100-task sample.

The clean `7bdeb960f5a38ea7515c67a12636a29198fd95f6` follow-up used the
unchanged Q6_K target and DSpark Q4 draft, greedy seed 42, batch 12, adaptive
K10/S6, a 2745 MHz clock lock, high-priority CUDA streams, High process
priority, `0x55555` CPU affinity, and three consecutive idle-GPU admission
samples before every server start.

| Follow-up | Q6_K target-only | Adaptive DSpark | Paired result |
| --- | ---: | ---: | --- |
| 512-token override, 5 tasks × 3 alternating repetitions | 4/5 lenient and strict; 1 truncated; 24.967 token/s | 4/5 lenient and strict; 1 truncated; **30.521 token/s** | 5/5 answer parity in every repetition; 0 gains; 0 losses; **1.222x** workload throughput |
| 1,024-token override, 5 tasks × 1 pair | 4/5 lenient and strict; 0 truncated | 4/5 lenient and strict; 0 truncated | **5/5 untruncated answer parity**; 0 gains; 0 losses |

Both modes were prediction- and content-deterministic across all three
512-token repetitions. Cross-mode complete-output parity was 0/5 at both
budgets: batched speculative verification and serial target-only execution
followed different deterministic Q6_K CUDA trajectories even though the
target verified every committed speculative token. The 1,024-token run is
classified as quality-only because shared-GPU contention made its throughput
comparison unfavorable; it is not used as a performance claim.

This closes the observed answer losses as truncation-sensitive diagnostics;
it does not establish general intelligence equivalence. Exact-output
production-default admission remains closed. Verify the path-free package,
including every raw environment, aggregate, report, tool, model, task, and
selection hash, with:

```powershell
py -3.13 .\tools\dspark_quality_followup_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality\followup-evidence.json `
  --json
```

Adding `--require-production-default` is expected to fail with exact output
parity `0/5`.

### Reproduce the focused matrix

The runner now treats a token override, selected-task digest, runner hash,
evaluator hash, and reporter hash as part of the reusable environment
identity. `MaxTokensOverride` can raise a reviewed task's normal limit without
changing the source task cache; it is mutually exclusive with
`MaxTokensCap`.

```powershell
$common = @{
  Q6PowerHome = 'D:\models\a3s-power\dspark-home'
  Profile = 'dspark-q4'
  Model = 'qwen3.8-27b-q6-k-dspark-q4'
  RuntimeConfig = '.\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality-adaptive-k10-s6.acl'
  PreparedTaskCache = '.\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\quality\tasks-v1.json'
  TaskSelection = '.\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality\divergence-v1.selection.json'
  TargetDirectory = 'target-native-sm89-ninja'
  NumBatch = 12
  ProcessPriority = 'High'
  ProcessorAffinityMask = 349525
  LockGpuClockMHz = 2745
  CudaHighPriority = $true
  MaximumIdleGpuUtilizationPercent = 15
  MinimumIdleGpuMemoryFreeMiB = 23000
  IdleGpuSampleCount = 3
  IdleGpuSampleIntervalMilliseconds = 500
  IdleGpuWaitSeconds = 300
  RequireHighPerformancePowerPlan = $true
  RequireCleanTree = $true
  IncludeContent = $true
}

.\tools\run-qwen38-quality-matrix.ps1 @common `
  -OutputRoot D:\benchmarks\dspark-followup-512 `
  -Repetitions 3 -NumCtx 1024 -MaxTokensOverride 512

.\tools\run-qwen38-quality-matrix.ps1 @common `
  -OutputRoot D:\benchmarks\dspark-followup-1024 `
  -Repetitions 1 -NumCtx 2048 -MaxTokensOverride 1024
```

On WDDM, a process can claim VRAM after the pre-launch admission samples.
Keep other GPU workloads stopped. If a model load fails, rerun the same
command: only complete reports with the exact environment identity are reused.
