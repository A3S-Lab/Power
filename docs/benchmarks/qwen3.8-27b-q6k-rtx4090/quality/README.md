# Qwen3.8-27B representative quality and throughput matrix

This suite complements the repetitive-prompt peak benchmark in the parent
directory. It measures three inference modes on one fixed 100-task workload,
repeats every mode three times, rotates execution order, and retains the
machine-readable task set and per-request evidence needed to reproduce or
audit the result.

## External-DSpark quality diagnostic

The native external-DSpark K10/S6 profile has now been replayed against the
same fixed 100-task MMLU/GSM8K/C-Eval set, three times per mode and in rotating
order. This is a separate context-1024, batch-12 capture using the untouched
Q6_K target and the verified 1.10 GB DSpark Q4 artifact.

| Mode | Lenient | Strict | Truncated | Mean request-wide throughput |
| --- | ---: | ---: | ---: | ---: |
| Q6_K target-only | 67/100 | 58/100 | 40/100 | 22.618 token/s |
| Q6_K + DSpark Q4 K10/S6 | **73/100** | **59/100** | 40/100 | **32.678 token/s** |

Both modes produced identical predictions and response hashes across their
three repetitions. The paired DSpark comparison had six lenient gains and no
losses (`p=0.03125`), two strict gains and one loss (`p=1.0`), 91/100 answer
parity, and 54/100 complete response-hash parity. All 58 tasks that were
untruncated in both modes retained the same extracted answer. The measured
score did not fall, but the complete-output divergence prevents a claim of
lossless equivalence or improved intelligence.

DSpark increased workload throughput by **1.445x**, accepted 44.726% of
proposals, and committed 3.674 verified tokens per target pass. Each 100-task
run recorded 100 exact fallback replays and 100 rollback-guarded requests.
That makes fixed K10/S6 a high-acceptance peak-prompt profile, not the balanced
quality default. See the [full DSpark analysis, verifier, and reproduction
protocol](../dspark/README.md#representative-100-task-diagnostic).

## Current 100-task full-vocabulary K7/S7 result

The current rollback-complete profile selects **TBQ4 + full-vocabulary MTP,
fixed K7/S7**. Across the same 100 tasks and three cyclically ordered runs, it
was 114.9% faster than TBQ4 with speculation off and produced no fallback
replays or rollback-guard activations.

| Mode | Lenient score | Strict score | Truncated | Mean request-wide throughput | Three-run range | Mean per-run median task latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untouched Q6_K, speculation off | 67/100 | 60/100 | 38/100 | 30.883 token/s | 30.065--31.954 | 7.095 s |
| TBQ4 mixed artifact, speculation off | 70/100 | 64/100 | 36/100 | 38.724 token/s | 37.254--39.539 | 5.488 s |
| TBQ4 + full-vocabulary MTP, fixed K7/S7 | **76/100** | **66/100** | **34/100** | **83.228 token/s** | **80.747--84.995** | **2.330 s** |

Request-wide throughput is total completion tokens divided by summed request
wall time. It includes prefill, generation, and request overhead, but excludes
model loading and one warm-up request per server cycle. The server's aggregate
generation metric for K7/S7 was 104.107--110.004 token/s; the separate
1,024-token steady-decode gate reached a 175.2089 token/s median. These three
rates answer different questions and must not be substituted for one another.

### Current domain breakdown

| Fixed sample | Untouched Q6_K score / throughput | TBQ4 off score / throughput | Full-vocabulary K7/S7 score / throughput | K7/S7 acceptance |
| --- | ---: | ---: | ---: | ---: |
| MMLU, 50 tasks | 60.0% / 30.456 token/s | 64.0% / 37.909 token/s | **70.0% / 78.570 token/s** | 49.26% |
| GSM8K, 20 tasks | 75.0% / 31.985 token/s | 85.0% / 40.167 token/s | **90.0% / 95.681 token/s** | 56.31% |
| C-Eval, 30 tasks | 73.3% / 30.832 token/s | 70.0% / 39.071 token/s | **76.7% / 82.782 token/s** | 51.17% |

### Current paired quality evidence

All three repetitions produced identical predictions and content hashes within
each mode. Repetition reduces timing noise; quality still has 100 independent
task observations, not 300. All 900 requests completed without error.

Relative to TBQ4-off, K7/S7 had seven lenient gains and one loss (`p=0.0703`,
paired exact McNemar), and three strict gains and one loss (`p=0.625`). It had
89/100 answer parity and 34/100 exact content-hash parity. Where neither answer
was truncated, all 62/62 extracted answers matched. The observed scores rose
from 70 to 76 lenient and from 64 to 66 strict; therefore this sample shows
**no observed intelligence decrease at a 175.2089 token/s median steady
decode**. The
differences are not significant at `p < 0.05`, so the result also does not prove
that block execution improves general intelligence.

The K7/S7 runtime accepted 51.33% of proposals and verified 4.543 tokens per
target pass. Each of the three 100-request runs recorded zero fallback replay,
zero guarded request, and zero guard activation. The compact checked-in summary
pins the scores, runtime metrics, peak gates, identities, and source-evidence
hashes in
[full-vocabulary-s7-current-rtx4090-3x.json](full-vocabulary-s7-current-rtx4090-3x.json).

## Untouched-Q6_K prefix-FR calibration

The current pure-Q6_K peak profile was also replayed once on the fixed 12-task
subset with a 128-token cap. This is a throughput and proposal-coverage probe,
not a replacement for the repeated 100-task matrix:

| Pure Q6_K mode, K7/S6, batch 14 | Request-wide throughput | Acceptance | Lenient / strict | Truncated |
| --- | ---: | ---: | ---: | ---: |
| Speculation off | 29.7127 token/s | -- | 4/12 / 3/12 | 11/12 |
| Full-vocabulary MTP | **47.0324 token/s** | **52.30%** | **5/12 / 3/12** | 11/12 |
| Prefix-FR8192 MTP | 37.2900 token/s | 24.82% | 4/12 / 3/12 | 11/12 |

Full-vocabulary MTP was 58.29% faster than autoregressive mode. Prefix-FR8192
was 25.50% faster than autoregressive mode but 20.71% slower than full
vocabulary because proposal coverage fell on this multilingual mix. This is why
the 176.6109 token/s repetitive-prompt peak is not reported as general
request-wide throughput.

The [raw sweep](pure-q6-fr8192-calibration-rtx4090-1x.json),
[environment receipt](pure-q6-fr8192-calibration-rtx4090-1x.environment.json),
[pure-Q6_K analysis](../PURE-Q6.md), and
[reproduction command](../REPRODUCE.md#5-reproduce-the-representative-quality-tests)
are checked in. Eleven truncated tasks make the answer counts unsuitable for a
general quality conclusion; the exact pure-Q6_K prefix-FR profile still needs a
full repeated matrix before any such claim.

## Historical 100-task prefix-FR result

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

## Historical pre-guard rollback calibration

After removing the token-ID-prefix FR limit, the pre-guard binary replayed a
fixed 12-task calibration three times. This smaller suite uses four MMLU, four
GSM8K, and four C-Eval tasks selected by the checked-in manifest. It is a
configuration calibration, not a replacement for either 100-task matrix.

| Mode | Mean workload throughput | Three-run range | Acceptance | Fallback replays per run | Lenient / strict score |
| --- | ---: | ---: | ---: | ---: | ---: |
| TBQ4, speculation off | 35.048 token/s | 34.806--35.330 | -- | -- | 5/12 / 3/12 |
| Full-vocab fixed K7/S6 | 28.226 token/s | 27.904--28.451 | 48.54% | 46 | 4/12 / 3/12 |
| Full-vocab adaptive K7/S6 | 60.031 token/s | 58.676--60.749 | 65.50% | 0 | 5/12 / 3/12 |
| Full-vocab fixed K7/S7 | **68.211 token/s** | **67.559--68.609** | 49.67% | 0 | 5/12 / 3/12 |

Fixed K7/S7 was 94.6% faster than speculation-off and 13.6% faster than
adaptive K7/S6 on this calibration. Fixed K7/S6 was 19.5% slower than
speculation-off because every repetition incurred 46 exact prefix replays.
At that revision, the seventh snapshot was required to avoid repeated recovery
in a fixed K7 mixed-workload configuration. Adaptive K7/S6 remained
rollback-safe because its controller capped proposal width to the six-snapshot
window, but that conservative width left throughput below fixed K7/S7. The
current guard changes the fixed-S6 recovery behavior as documented below.

All predictions were stable across the three repetitions of each mode. The
strict score was 3/12 in every mode; the fixed S6 path changed one lenient
answer while traversing its replay-heavy block schedule. This tiny, highly
truncated sample is evidence for rollback and throughput behavior only. The
machine-readable summary pins the binary, model, task selection, per-run
throughput, acceptance, replay counts, and scores in
[full-vocab-rollback-calibration-rtx4090-3x.json](full-vocab-rollback-calibration-rtx4090-3x.json).

## Current rollback-guard calibration

The request-local guard removes the unbounded replay behavior from fixed K7/S6.
It permits the first exact replay, then permanently clamps that request's
proposal width to its six resident snapshots. The high-acceptance peak path is
unchanged because the guard never activates there.

| Current mode | Mean workload throughput | Three-run range | Acceptance | Replays / guarded requests / activations per run | Lenient / strict score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Guarded fixed K7/S6 | 54.060 token/s | 53.686--54.331 | 53.07% | 11 / 11 / 11 | 5/12 / 3/12 |
| Rollback-complete fixed K7/S7 | **68.205 token/s** | **66.924--68.922** | 49.67% | **0 / 0 / 0** | 5/12 / 3/12 |

Guarded S6 is 91.5% faster than the pre-guard fixed-S6 result, while S7 remains
26.2% faster than guarded S6 on this calibration and needs no recovery path.
The corresponding 1,024-token peak gates reached 177.7165 token/s median with
a 176.7287 minimum for guarded S6 (9/9 samples at least 175) and 175.2089
token/s median with a 174.2211 minimum for S7 (5/9 samples at least 175; median
gate passed). This makes S7 the balanced default and guarded S6 an explicit
peak-throughput profile.

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
TBQ4/MTP/Q6, then MTP/Q6/TBQ4. The current capture also requested processor
affinity mask `0x55555`.

| Current full-vocabulary identity | Captured value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB reported, compute capability 8.9 |
| Driver | 610.74 |
| CPU | Intel Xeon w5-2445, 10 cores / 20 logical processors |
| OS | Windows 11 build 22631 |
| Host controls | High-performance power plan; High process priority; affinity `0x55555` |
| Power commit | `4406c9c5aa67b8ad861898866e04d7dfbf4cbf2b` (clean) |
| Server executable SHA-256 | `2beb4cd460eee49ea8ab350bf19b4941e2cd121faa62a44a26846eec6eb66082` |
| Runtime profile | [`full-vocabulary-current.acl`](full-vocabulary-current.acl), fixed K7/S7, batch 14 |
| Runtime ACL SHA-256 | `c2b3aca41323b6c3d0ae101b0b68764a99e0a9318df565d13612dd718eb11767` |
| Untouched Q6_K | 22,884,408,288 bytes; `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| TBQ4 mixed artifact | 19,187,686,464 bytes; `5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6` |

The runner verifies both GGUF byte lengths, manifest hashes, and full file
hashes by default. It also records the executable, Git state, hardware, OS,
power plan, process priority, configuration hash, and task-manifest hash in
the output environment file. The current S7 capture started from a clean tree,
did not reuse reports from another commit, and pins aggregate JSON SHA-256
`cadf445f2d3fb6c00924669b86bbb8f900b8ee945fe4bafa9222443081482123`
and environment SHA-256
`5f92af26a9b611b6825ca18694ee6fb5dfc2cfe28ac8b7b6dac16ee44fb58df3`.

[`matrix.acl`](matrix.acl) is the byte-exact historical K7/S6 runtime ACL
whose SHA-256 is pinned by the environment capture. It intentionally omits
configuration fields added after commit `955b4552ca091af07818573e803f9369488a63f9`.
Do not normalize it to current defaults: exact replay of the published
prefix-FR result requires that captured source revision and server executable.
Its historical server SHA-256 is
`a2bc32b6be65bc79cc757716e27ac64c03b7bc05d0c96a1f91aabe6dfb0acbde`;
it is not the identity of the current full-vocabulary result.

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

Then run the current complete three-by-three full-vocabulary S7 matrix using
the reviewed offline task cache:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\run-qwen38-quality-matrix.ps1 `
  -Q6PowerHome D:\models\a3s-power\qwen38\power-home `
  -Tbq4PowerHome D:\models\a3s-power\qwen38\power-home-tbq4-0-ffn `
  -Profile full-vocabulary-current `
  -PreparedTaskCache .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\quality\tasks-v1.json `
  -TargetDirectory target-native-sm89-ninja `
  -OutputRoot target-qwen38-quality-full-vocabulary-s7 `
  -Repetitions 3 -NumBatch 14 `
  -ProcessPriority High `
  -ProcessorAffinityMask 349525 `
  -RequireHighPerformancePowerPlan `
  -RequireCleanTree
```

`-Profile full-vocabulary-current` selects the checked-in K7/S7 ACL and fails
if a compatible report from a different commit would otherwise be reused.
Use a new output directory for every experiment. Omit the profile only when
deliberately replaying the historical prefix-FR matrix.

Replay the smaller current-binary rollback calibration with host-staged
recurrent state and explicit snapshot windows:

```powershell
.\tools\run-qwen38-mtp-sweep.ps1 `
  -PowerHome D:\models\a3s-power\qwen38\power-home-tbq4-0-ffn `
  -FrVocabSizes @(0) -DraftMaxValues @(7) `
  -MtpRecurrentSnapshots 6 -MtpRecurrentChain $false `
  -NumBatchValues @(14) `
  -Policies @('fixed') `
  -Repetitions 3 -MaxTokensCap 128 `
  -TargetDirectory target-native-sm89-ninja `
  -OutputRoot target-qwen38-mtp-guarded-s6-current-3x `
  -RequireHighPerformancePowerPlan -ProcessPriority High

.\tools\run-qwen38-mtp-sweep.ps1 `
  -PowerHome D:\models\a3s-power\qwen38\power-home-tbq4-0-ffn `
  -FrVocabSizes @(0) -DraftMaxValues @(7) `
  -MtpRecurrentSnapshots 7 -MtpRecurrentChain $false `
  -NumBatchValues @(14) `
  -Policies @('fixed') -Repetitions 3 -MaxTokensCap 128 `
  -TargetDirectory target-native-sm89-ninja `
  -OutputRoot target-qwen38-mtp-rollback-complete-s7-current-3x `
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
reported because 34--38 responses per current mode reached the token limit
(34--40 in the historical prefix-FR capture).

## Evidence map

- [Task selection manifest](tasks-v1.manifest.json) and [reviewed task cache](tasks-v1.json)
- [Current full-vocabulary S7 compact result and source hashes](full-vocabulary-s7-current-rtx4090-3x.json)
- [Current full-vocabulary S7 runtime ACL](full-vocabulary-current.acl)
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
and machine above. The current 100-task matrix establishes no observed quality
regression for full-vocabulary fixed K7/S7 and a 114.9% request-wide throughput
gain over TBQ4-off on this workload. Its paired score differences are not
statistically significant at `p < 0.05`, so neither general quality parity nor
an intelligence improvement may be inferred beyond this sample. The 175.2089
token/s result remains a warmed-up repetitive-prompt boundary on a shared WDDM
display GPU, not representative application throughput or a service-level
guarantee.
