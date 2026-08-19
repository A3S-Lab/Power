# Qwen3.8-27B representative quality and throughput matrix

This suite complements the repetitive-prompt peak benchmark in the parent
directory. It measures three inference modes on one fixed 100-task workload,
repeats every mode three times, rotates execution order, and retains the
machine-readable task set and per-request evidence needed to reproduce or
audit the result.

## 100-task prefix-FR result

Within this 100-task prefix-FR matrix, the winner is **TBQ4 with speculation
off**. It was 20.8% faster than the untouched Q6_K artifact while scoring
72/100 instead of 66/100 on this sample. That six-answer difference is not
statistically significant (`p=0.1796`, paired exact McNemar test), so it is
evidence of no detected regression here, not evidence that quantization
improves the model.

MTP + FR retained the same 72/100 lenient score as TBQ4 without speculation,
but was 33.0% slower overall. Its 25.55% workload-wide draft acceptance was
far below the 93.62% acceptance of the repetitive prompt used by the 175
token/s development gate.

`TBQ4` here means the documented Q6_K-derived mixed artifact: main-block FFN
tensors are Q4_0, the MTP block remains Q6_K, and the separate MTP head is
Q4_K. It is not an untouched six-bit model or a uniform four-bit model.

| Mode | Lenient score | Strict score | Truncated | Mean workload throughput | Three-run range | Median per-run task latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untouched Q6_K, speculation off | 66/100 | 59/100 | 40/100 | 34.551 token/s | 34.360--34.756 | 6.421 s |
| TBQ4 mixed artifact, speculation off | 72/100 | 64/100 | 34/100 | **41.745 token/s** | 41.517--42.140 | **5.057 s** |
| TBQ4 + MTP + FR | 72/100 | 60/100 | 39/100 | 27.951 token/s | 27.705--28.351 | 5.728 s |

Workload throughput is total completion tokens divided by summed request wall
time. It includes prompt processing, generation, and request overhead, but
excludes model loading and one warm-up request per server cycle. It is not the
server's steady-state decode rate and must not be compared directly with the
184.3665 token/s repetitive-prompt peak.

### Domain breakdown

| Fixed sample | Untouched Q6_K score / throughput | TBQ4 off score / throughput | TBQ4 + MTP + FR score / throughput | MTP draft acceptance |
| --- | ---: | ---: | ---: | ---: |
| MMLU, 50 tasks | 60.0% / 34.360 token/s | 64.0% / 41.628 token/s | 66.0% / 36.027 token/s | 30.75% |
| GSM8K, 20 tasks | 75.0% / 34.865 token/s | 90.0% / 42.108 token/s | 95.0% / 45.001 token/s | 36.03% |
| C-Eval, 30 tasks | 70.0% / 34.636 token/s | 73.3% / 41.639 token/s | 66.7% / 15.985 token/s | 14.21% |

MTP was 6.9% faster than TBQ4-off on GSM8K, but 13.5% slower on MMLU and
61.6% slower on C-Eval. The leading-8,192-row FR draft vocabulary is therefore
not a safe universal performance default; its language and domain coverage
must be improved or the runtime must disable it when observed acceptance is
too low.

### Paired quality evidence

All three repetitions produced the same prediction and output hash for every
task within each mode. Repetition reduces timing noise; deterministic repeats
do not turn this 100-task sample into 300 independent quality observations.

For Q6_K to TBQ4-off, the paired lenient comparison had 10 gains and 4 losses
(`p=0.1796`), 81/100 answer parity, and 56/57 answer parity where neither
response was truncated. For TBQ4-off to TBQ4 + MTP + FR, it had 5 gains and 5
losses (`p=1.0`), 88/100 answer parity, and 59/59 answer parity where neither
response was truncated. Strict scoring changed from 64/100 to 60/100 with 3
gains and 7 losses (`p=0.3438`).

Target verification prevents unverified draft proposals from being committed,
but it does not promise bitwise-identical text across serial and block execution
paths. The MTP comparison had only 34/100 exact output-hash matches. Truncation,
floating-point kernel ordering, and changed generation boundaries are all
visible in this workload, so release gates should use paired task scores in
addition to a single prompt digest.

## Full-vocabulary rollback calibration

After removing the token-ID-prefix FR limit, the current binary replayed a
fixed 12-task calibration three times. This smaller suite uses four MMLU, four
GSM8K, and four C-Eval tasks selected by the checked-in manifest. It is a
configuration calibration, not a replacement for the 100-task matrix above.

| Mode | Mean workload throughput | Three-run range | Acceptance | Fallback replays per run | Lenient / strict score |
| --- | ---: | ---: | ---: | ---: | ---: |
| TBQ4, speculation off | 35.048 token/s | 34.806--35.330 | -- | -- | 5/12 / 3/12 |
| Full-vocab fixed K7/S6 | 28.226 token/s | 27.904--28.451 | 48.54% | 46 | 4/12 / 3/12 |
| Full-vocab adaptive K7/S6 | 60.031 token/s | 58.676--60.749 | 65.50% | 0 | 5/12 / 3/12 |
| Full-vocab fixed K7/S7 | **68.211 token/s** | **67.559--68.609** | 49.67% | 0 | 5/12 / 3/12 |

Fixed K7/S7 was 94.6% faster than speculation-off and 13.6% faster than
adaptive K7/S6 on this calibration. Fixed K7/S6 was 19.5% slower than
speculation-off because every repetition incurred 46 exact prefix replays.
The seventh snapshot is therefore not optional for a fixed K7 mixed-workload
configuration. Adaptive K7/S6 remains rollback-safe because its controller
caps the effective proposal width to the six-snapshot window, but that
conservative width leaves throughput below fixed K7/S7.

All predictions were stable across the three repetitions of each mode. The
strict score was 3/12 in every mode; the fixed S6 path changed one lenient
answer while traversing its replay-heavy block schedule. This tiny, highly
truncated sample is evidence for rollback and throughput behavior only. The
machine-readable summary pins the binary, model, task selection, per-run
throughput, acceptance, replay counts, and scores in
[full-vocab-rollback-calibration-rtx4090-3x.json](full-vocab-rollback-calibration-rtx4090-3x.json).

## Locked inputs and environment

The task cache has SHA-256
`5798257e18b81188749196d34359278dfadf7986776eb2bd66d629cbfc33813c`
over its canonical task array. It contains fixed dataset row windows rather
than a random sample:

- 50 MMLU test tasks from five declared windows.
- 20 GSM8K `main` test tasks.
- 30 C-Eval validation tasks from ten declared subjects.

The run used temperature 0, top-p 1, seed 42, a 4,096-token context, one
parallel request, full CUDA layer offload, Flash Attention, ten host threads,
and one excluded warm-up request before each 100-task mode run. Each server was
restarted between modes. The three-mode cyclic order was Q6/TBQ4/MTP,
TBQ4/MTP/Q6, then MTP/Q6/TBQ4.

| Identity | Captured value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB reported, compute capability 8.9 |
| Driver | 610.74 |
| CPU | Intel Xeon w5-2445, 10 cores / 20 logical processors |
| OS | Windows 11 build 22631 |
| Host controls | High-performance power plan; High process priority |
| Power commit | `955b4552ca091af07818573e803f9369488a63f9` |
| Server executable SHA-256 | `a2bc32b6be65bc79cc757716e27ac64c03b7bc05d0c96a1f91aabe6dfb0acbde` |
| Untouched Q6_K | 22,884,408,288 bytes; `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| TBQ4 mixed artifact | 19,187,686,464 bytes; `5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6` |

The runner verifies both GGUF byte lengths, manifest hashes, and full file
hashes by default. It also records the executable, Git state, hardware, OS,
power plan, process priority, configuration hash, and task-manifest hash in
the output environment file. This capture records a dirty worktree only
because the new harness and evidence files were still untracked; its Git status
contains no tracked server-source modification, and the executable hash is the
authoritative binary identity.

[`matrix.acl`](matrix.acl) is the byte-exact historical K7/S6 runtime ACL
whose SHA-256 is pinned by the environment capture. It intentionally omits
configuration fields added after commit `955b4552ca091af07818573e803f9369488a63f9`.
Do not normalize it to current defaults: exact replay of the published
prefix-FR result requires the captured source revision and server executable,
while running the same task matrix with the current binary is a new experiment.

The nine archived per-request reports use historical schema `report.v2`.
The current evaluator writes `report.v3`, which adds task-source, task-selection,
request, batch-size, and token-cap identity. The shared result and summary fields
still reaggregate exactly, but a current replay is expected to have new report
timestamps, schema identifiers, and file hashes.

## Reproduce

Run from the `crates/power` directory on Windows. Python uses only its standard
library; the PowerShell runner also requires `nvidia-smi`, a release
`a3s-power.exe`, and two Power homes whose model manifests point to the exact
artifacts above. Build prerequisites, the pinned CUDA profile, input-hash
checks, and the complete source-validation command set are documented in the
[parent reproduction guide](../REPRODUCE.md).

First verify the evaluator:

```powershell
py -3.13 .\tools\test_qwen38_quality_eval.py
py -3.13 -m py_compile `
  .\tools\qwen38_quality_eval.py `
  .\tools\qwen38_quality_report.py
```

Then run the complete three-by-three matrix using the reviewed offline task
cache:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\run-qwen38-quality-matrix.ps1 `
  -Q6PowerHome D:\models\a3s-power\qwen38\power-home `
  -Tbq4PowerHome D:\models\a3s-power\qwen38\power-home-tbq4-0-ffn `
  -PreparedTaskCache .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\quality\tasks-v1.json `
  -TargetDirectory target-native-sm89-ninja `
  -OutputRoot target-qwen38-quality-replay `
  -Repetitions 3 `
  -ProcessPriority High `
  -RequireHighPerformancePowerPlan `
  -RequireCleanTree
```

Replay the smaller full-vocabulary rollback calibration with the current
mixed artifact and explicit snapshot windows:

```powershell
.\tools\run-qwen38-mtp-sweep.ps1 `
  -PowerHome D:\models\a3s-power\qwen38\power-home-tbq4-0-ffn `
  -FrVocabSizes @(0) -DraftMaxValues @(7) `
  -MtpRecurrentSnapshots 6 -MtpRecurrentChain $true `
  -NumBatchValues @(14) `
  -Policies @('fixed','adaptive') -IncludeOffBaseline `
  -Repetitions 3 -MaxTokensCap 128 `
  -TargetDirectory target-native-sm89-ninja `
  -OutputRoot target-qwen38-mtp-full-vocab-s6-3x `
  -RequireHighPerformancePowerPlan -ProcessPriority High

.\tools\run-qwen38-mtp-sweep.ps1 `
  -PowerHome D:\models\a3s-power\qwen38\power-home-tbq4-0-ffn `
  -FrVocabSizes @(0) -DraftMaxValues @(7) `
  -MtpRecurrentSnapshots 7 -MtpRecurrentChain $true `
  -NumBatchValues @(14) `
  -Policies @('fixed') -Repetitions 3 -MaxTokensCap 128 `
  -TargetDirectory target-native-sm89-ninja `
  -OutputRoot target-qwen38-mtp-full-vocab-s7-fixed-3x `
  -RequireHighPerformancePowerPlan -ProcessPriority High
```

Omit `-PreparedTaskCache` to fetch the declared rows from the Hugging Face
dataset server. The fetch is accepted only when its canonical task digest
matches the reviewed manifest, so upstream drift fails closed. A complete
per-mode report is resumable; an incomplete report is restarted. The runner
starts and stops only the server processes it owns and restores all modified
environment variables.

The evaluator sends manual Qwen ChatML through `POST /v1/completions`. Choice
tasks allow 256 completion tokens and GSM8K allows 384. Lenient scoring uses an
explicit final marker when present and otherwise extracts the last plausible
choice or number; strict scoring requires the explicit marker. Both are
reported because 34--40 responses per mode reached the token limit.

## Evidence map

- [Task selection manifest](tasks-v1.manifest.json) and [reviewed task cache](tasks-v1.json)
- [Byte-exact historical runtime ACL](matrix.acl)
- [Captured machine and artifact environment](environment-rtx4090-3x.json)
- [Three-run aggregate, paired comparisons, and report hashes](results-rtx4090-3x.json)
- [Nine compact per-request reports](runs/)
- [Full-vocabulary rollback calibration](full-vocab-rollback-calibration-rtx4090-3x.json)
- [Evaluator](../../../../tools/qwen38_quality_eval.py),
  [report aggregation](../../../../tools/qwen38_quality_report.py),
  [runner](../../../../tools/run-qwen38-quality-matrix.ps1), and
  [unit tests](../../../../tools/test_qwen38_quality_eval.py)

The compact reports intentionally omit generated text by default. They retain
task IDs, expected and extracted answers, completion usage, finish reasons,
latency, output hashes, receipt hashes, errors, and parsed MTP runtime metrics.
Use `-IncludeContent` only when storing model output is appropriate.

## Scope

This is a fixed smoke-quality sample, not an official full MMLU, GSM8K, or
C-Eval score. It has no sampling confidence over the broader datasets, and its
high truncation rate makes the strict/lenient distinction material. The
results apply to the pinned artifacts, executable, configuration, prompts,
and machine above. They establish that the 175 token/s peak is
workload-sensitive and that TBQ4-off was the fastest mode in the historical
100-task prefix-FR matrix. The smaller current-binary calibration separately
establishes that full-vocabulary fixed K7/S7 can outperform TBQ4-off when the
rollback window covers the complete proposal, but it is not broad enough to
replace the 100-task release result.
