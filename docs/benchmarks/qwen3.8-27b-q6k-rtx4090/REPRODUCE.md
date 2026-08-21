# Reproducing the Qwen3.8-27B RTX 4090 results

This guide reproduces the checked-in evidence at two levels:

- **Offline verification** checks the archived reports, hashes, thresholds, and
  deterministic outputs without loading the model.
- **Performance replay** rebuilds Power, verifies the exact model and prompt,
  and runs the same streaming API workload on the acceptance host.

Run every command from the `crates/power` directory. The published performance
capture was made on Windows 11 with an RTX 4090 and a 10-core / 20-thread Intel
Xeon w5-2445. The CPU affinity mask and GPU clock control are specific to that
host; do not copy them to a different topology without a new A/B measurement.

## Current untouched Q6-K target

| Input or result | Current value |
| --- | --- |
| Model | Untouched Q6_K GGUF, 22,884,408,288 bytes |
| Model SHA-256 | `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| Clean source revision | `eb6aeda59561eff3e4e7592704cab6fc863b72c7` |
| Workload | 1 warm-up, 9 measured requests, 1,024 generated tokens each, batch 14 |
| Full-vocabulary K7/S7 control | 147.0207 token/s median; 146.0917 minimum |
| Prefix-FR8192 K7/S6 peak | **176.6109 token/s median**; 173.2630 minimum; 7 / 9 at least 175 |
| Peak median end to end | 167.3519 token/s |
| Output SHA-256 | `a54538eaaf6cc0b8b43cbafd489c7779f0f5206c93d5034fd3a16f4366a90523` |
| Peak report | [`pure-q6-fr8192-1024-9x.json`](pure-q6-fr8192-1024-9x.json) |
| Full-vocabulary report | [`pure-q6-full-vocabulary-1024-9x.json`](pure-q6-full-vocabulary-1024-9x.json) |

Prefix-FR8192 is a peak profile for measured high-coverage workloads. The
current full head uses target-token-ID order rather than a corpus-frequency
`d2t` map. On the checked-in 12-task calibration, full-vocabulary K7/S6 was
faster request-wide than prefix FR, so full vocabulary remains the balanced
choice. The [pure-Q6_K report](PURE-Q6.md) separates that real-workload result
from the long repetitive steady-decode gate.

## Previous mixed-artifact acceptance targets

| Input or result | Current value |
| --- | --- |
| Model | Q6_K-derived mixed TBQ4 artifact, 19,187,686,464 bytes |
| Model SHA-256 | `5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6` |
| Server SHA-256 | `2beb4cd460eee49ea8ab350bf19b4941e2cd121faa62a44a26846eec6eb66082` |
| Prompt SHA-256 | `d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89` |
| Workload | 1 warm-up, 9 measured requests, 1,024 generated tokens each, batch 14 |
| Guarded K7/S6 | 177.7165 token/s median; 176.7287 minimum; 9 / 9 at least 175 |
| Rollback-complete K7/S7 | 175.2089 token/s median; 174.2211 minimum; median gate passed, 5 / 9 at least 175 |
| Output SHA-256 | `a54538eaaf6cc0b8b43cbafd489c7779f0f5206c93d5034fd3a16f4366a90523` |
| Current 3x100 K7/S7 quality | 76/100 lenient; 66/100 strict; 83.228 token/s request-wide; zero replay |
| Compact evidence | [`quality/full-vocabulary-s7-current-rtx4090-3x.json`](quality/full-vocabulary-s7-current-rtx4090-3x.json) |

K7/S7 is the balanced mixed-workload profile. Guarded K7/S6 is the peak
profile: it preserves the high-acceptance fast path, but a low-acceptance
request may perform one exact replay before being clamped to six proposals.
The performance threshold is median-based; only the guarded S6 capture kept
every current sample above 175 token/s.

## Historical checked-in raw acceptance capture

| Input or result | Published value |
| --- | --- |
| Model | Q6_K-derived mixed TBQ4 artifact, 19,187,686,464 bytes |
| Model SHA-256 | `5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6` |
| Prompt SHA-256 | `d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89` |
| Runtime ACL SHA-256 | `2f348cca96282a22650d9766cffa81251ea10a5e34a089bcc91b0822ab5c1d0e` |
| Workload | 1 warm-up, 9 measured requests, 1,024 generated tokens each |
| Decode result | 177.3062 token/s median; 175.5958 token/s minimum |
| Gate | 9 / 9 samples at or above 175 token/s |
| Output SHA-256 | `a54538eaaf6cc0b8b43cbafd489c7779f0f5206c93d5034fd3a16f4366a90523` |
| Report SHA-256 | `ad478dfdb3df7d3560fb39eeb2be854715bbb483a985c8e10793df053c0c2d72` |
| Environment SHA-256 | `c0cb352e99dcdd2c9bd487cb7501a1f231c97ef5a661c26446ef3c2e4f5eda8d` |

The historical environment discloses a dirty worktree based on commit
`955b4552ca091af07818573e803f9369488a63f9`; its server and client executable
digests are therefore the exact historical binary identities. A replay from
the merged clean commit is a new capture. Its Git revision, executable hashes,
timestamps, and report hash will differ even when the measured behavior is the
same.

The historical evidence JSON, ACL, and prompt files are marked `-text` in
`.gitattributes`. Git therefore preserves their captured bytes instead of
applying platform line-ending conversion, so the published SHA-256 values are
stable across clones.

## 1. Build the CUDA profile

Install Rust 1.97.1, CUDA 12.6, CMake, Ninja, a supported MSVC toolchain, and
libclang. Fetch the pinned binding, apply both reviewed patches idempotently,
then build the SM 89 release profile:

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

The patch command must report that the binding and nested llama.cpp patches are
applied. The benchmark runner later rejects a server that initializes anything
other than the exclusive `llama.cpp` backend.

## 2. Verify the model, prompt, and ACL

Register the untouched Q6_K file as `qwen3.8-27b-q6-k` in the selected Power
home, then verify the exact artifact and current pure-Q6_K inputs:

```powershell
$powerDataRoot = 'D:\models\a3s-power\qwen38\power-home'
$modelManifestPath = Join-Path $powerDataRoot `
  'models\manifests\qwen3.8-27b-q6-k.json'
$modelManifest = Get-Content -Raw -LiteralPath $modelManifestPath |
  ConvertFrom-Json
$modelFile = Get-Item -LiteralPath $modelManifest.path

if ($modelFile.Length -ne 22884408288) {
  throw "Unexpected model length: $($modelFile.Length)"
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $modelFile.FullName).Hash -ne
    '562FBF760503008F118E5DF38DE5B3E97992D1F693F475815631198547486727') {
  throw 'Unexpected model hash'
}

$evidenceRoot = '.\docs\benchmarks\qwen3.8-27b-q6k-rtx4090'
$purePeakConfig = Join-Path $evidenceRoot `
  'pure-q6-mtp7-snap6-fr8192-host-staged.acl'
$pureFullConfig = Join-Path $evidenceRoot `
  'pure-q6-mtp7-snap7-host-staged.acl'
$promptPath = Join-Path $evidenceRoot 'prompt.txt'

$expectedHashes = @{
  $purePeakConfig =
    '9B1213DF972EA3731010A1FA72B0D553BA73DA42F31E92EAA4FECD3156CBF2EF'
  $pureFullConfig =
    'EB445101C1E33A035C9B1D120FEC12D9B21E6CE1B2FE5486AD46BEE52878A588'
  $promptPath =
    'D95A5E4DAD822BA9C84138F7A120017318BCB3A6A90E77246A8EC4EDE0E65D89'
}
foreach ($entry in $expectedHashes.GetEnumerator()) {
  if ((Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Key).Hash -ne
      $entry.Value) {
    throw "Hash mismatch: $($entry.Key)"
  }
}
```

For the previous mixed-artifact capture, rebuild it from the untouched Q6_K
source with the
quantization steps in the [benchmark record](README.md#historical-prefix-fr-175-tokens-gate).
Register the resulting file as `qwen3.8-27b-q6-k` in the selected Power home,
then verify every local input before starting the server:

```powershell
$powerDataRoot = 'D:\models\a3s-power\qwen38\power-home-tbq4-0-ffn'
$modelManifestPath = Join-Path $powerDataRoot `
  'models\manifests\qwen3.8-27b-q6-k.json'
$modelManifest = Get-Content -Raw -LiteralPath $modelManifestPath |
  ConvertFrom-Json
$modelFile = Get-Item -LiteralPath $modelManifest.path

if ($modelFile.Length -ne 19187686464) {
  throw "Unexpected model length: $($modelFile.Length)"
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $modelFile.FullName).Hash -ne
    '5F578B395F61DCAAC9698FE222D988F461FD902CE9494E8A06D8B9AAE4E7E2A6') {
  throw 'Unexpected model hash'
}

$promptPath = '.\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt'
$peakConfigPath = '.\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\mtp7-snap6-full-vocab-cpu-embedding.acl'
$balancedConfigPath = '.\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\mtp7-snap7-full-vocab-cpu-embedding.acl'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $promptPath).Hash -ne
    'D95A5E4DAD822BA9C84138F7A120017318BCB3A6A90E77246A8EC4EDE0E65D89') {
  throw 'Unexpected prompt hash'
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $peakConfigPath).Hash -ne
    '2F348CCA96282A22650D9766CFFA81251EA10A5E34A089BCC91B0822AB5C1D0E') {
  throw 'Unexpected K7/S6 ACL hash'
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $balancedConfigPath).Hash -ne
    '759EF6E5E60A08939ED747558992FA3031D63D2ECD59DACFDAE59790CC6FF79A') {
  throw 'Unexpected K7/S7 ACL hash'
}
```

## 3. Run the current untouched Q6-K gates

Use an otherwise idle GPU. The clock-lock operation may require an elevated
terminal and is reset by the runner in its `finally` block. The runner records
the current Git revision, clean/dirty state, executable hashes, prompt and ACL
hashes, requested and effective CPU affinity, power plan, GPU state, and GPU
process snapshot in a companion environment file. Run the full-vocabulary
control first, then the prefix-FR8192 peak with the same model and work shape:

```powershell
.\tools\run-qwen38-q6k-benchmark.ps1 `
  -Label pure-q6-full-vocabulary-replay `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\pure-q6-mtp7-snap7-host-staged.acl `
  -PromptFile .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt `
  -BenchmarkRoot D:\models\a3s-power\qwen38\benchmark `
  -PowerHome D:\models\a3s-power\qwen38\power-home `
  -ModelHash 562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727 `
  -MaxTokens 1024 -NumBatch 14 -WarmupRuns 1 -Samples 9 `
  -MinimumTokensPerSecond 0 -ProcessPriority High `
  -ProcessorAffinityMask 349525 -LockGpuClockMHz 2745 `
  -TargetDirectory target-native-sm89-ninja `
  -RequireHighPerformancePowerPlan -RequireCleanTree

.\tools\run-qwen38-q6k-benchmark.ps1 `
  -Label pure-q6-fr8192-k7s6-replay `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\pure-q6-mtp7-snap6-fr8192-host-staged.acl `
  -PromptFile .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt `
  -BenchmarkRoot D:\models\a3s-power\qwen38\benchmark `
  -PowerHome D:\models\a3s-power\qwen38\power-home `
  -ModelHash 562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727 `
  -MaxTokens 1024 -NumBatch 14 -WarmupRuns 1 -Samples 9 `
  -MinimumTokensPerSecond 175 -ProcessPriority High `
  -ProcessorAffinityMask 349525 -LockGpuClockMHz 2745 `
  -TargetDirectory target-native-sm89-ninja `
  -RequireHighPerformancePowerPlan -RequireCleanTree
```

The peak command fails when the median misses 175 token/s, the model identity
changes, the output is non-deterministic, a request stops before 1,024 tokens,
the backend is not exclusive, or a required host control is absent. The
full-vocabulary control deliberately uses a zero threshold because its purpose
is the paired 147.0207 token/s baseline, not the 175 gate.

### Replay the previous mixed-artifact gates

```powershell
.\tools\run-qwen38-q6k-benchmark.ps1 `
  -Label rollback-guard-s6-affinity-1024-9x-replay `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\mtp7-snap6-full-vocab-cpu-embedding.acl `
  -PromptFile .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt `
  -BenchmarkRoot D:\models\a3s-power\qwen38\benchmark `
  -PowerHome D:\models\a3s-power\qwen38\power-home-tbq4-0-ffn `
  -ModelHash 5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6 `
  -MaxTokens 1024 -NumBatch 14 -WarmupRuns 1 -Samples 9 `
  -MinimumTokensPerSecond 175 -ProcessPriority High `
  -ProcessorAffinityMask 349525 `
  -LockGpuClockMHz 2745 `
  -TargetDirectory target-native-sm89-ninja `
  -RequireHighPerformancePowerPlan -RequireCleanTree

.\tools\run-qwen38-q6k-benchmark.ps1 `
  -Label rollback-complete-s7-affinity-1024-9x-replay `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\mtp7-snap7-full-vocab-cpu-embedding.acl `
  -PromptFile .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt `
  -BenchmarkRoot D:\models\a3s-power\qwen38\benchmark `
  -PowerHome D:\models\a3s-power\qwen38\power-home-tbq4-0-ffn `
  -ModelHash 5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6 `
  -MaxTokens 1024 -NumBatch 14 -WarmupRuns 1 -Samples 9 `
  -MinimumTokensPerSecond 175 -ProcessPriority High `
  -ProcessorAffinityMask 349525 `
  -LockGpuClockMHz 2745 `
  -TargetDirectory target-native-sm89-ninja `
  -RequireHighPerformancePowerPlan -RequireCleanTree
```

The mixed-artifact command returns nonzero when the median misses 175 token/s,
the model identity changes, the output is non-deterministic, a request stops
before 1,024 tokens, the backend is not exclusive, or a required host control
is absent.
Failed reports are retained for diagnosis. Use S7 for the balanced gate and S6
only when intentionally measuring the guarded peak profile.

For the historical short-window shape, use `-MaxTokens 256 -NumBatch 20` and a
new label. A clock lock reduces one source of variance but cannot eliminate
contention from other WDDM clients on a shared display GPU.

## 4. Verify the archived performance evidence offline

Run the checked-in verifier first. It requires neither the model nor an NVIDIA
GPU, verifies 14 pinned file hashes, and recomputes the pure-Q6_K and mixed
quality, workload, steady-decode, and deterministic-output values:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\verify-qwen38-q6k-evidence.ps1 -Json
```

A passing run exits with code zero and reports `"status": "passed"`, including:

```json
{
  "verified_file_hashes": 14,
  "pure_q6": {
    "full_vocabulary_k7_s7_median": 147.020656574707,
    "prefix_fr8192_k7_s6_median": 176.6108685085471,
    "prefix_fr8192_samples_at_or_above_175": 7,
    "calibration": {
      "autoregressive_tokens_per_second": 29.712723837098697,
      "full_vocabulary_tokens_per_second": 47.03236986836804,
      "prefix_fr8192_tokens_per_second": 37.29003139316878
    }
  }
}
```

The expanded commands below document the individual assertions implemented by
the verifier and remain useful when diagnosing a mismatch.

First verify the current compact evidence and its distinction between
request-wide workload throughput and steady decode:

```powershell
$currentPath = '.\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\quality\full-vocabulary-s7-current-rtx4090-3x.json'
$current = Get-Content -Raw -LiteralPath $currentPath | ConvertFrom-Json
$s7 = $current.modes | Where-Object name -eq 'tbq4-mtp-full-vocab-k7-s7'

if ($s7.lenient_score -ne 76 -or
    $s7.strict_score -ne 66 -or
    [math]::Abs($s7.mean_workload_tokens_per_second - 83.22814601950864) -gt 1e-9 -or
    ($s7.fallback_replays_per_run -join ',') -ne '0,0,0' -or
    [math]::Abs($current.peak_gates.k7_s7_rollback_complete.median_decode_tokens_per_second -
      175.20889378841997) -gt 1e-9 -or
    $current.peak_gates.k7_s6_guarded.samples_at_or_above_175 -ne 9) {
  throw 'Current compact evidence did not validate'
}
```

The following historical check recomputes the median and minimum from all nine
checked-in raw samples, verifies the threshold and output digest, and verifies
the archived report, environment, ACL, and prompt hashes:

```powershell
$evidenceRoot = '.\docs\benchmarks\qwen3.8-27b-q6k-rtx4090'
$reportPath = Join-Path $evidenceRoot 'final-affinity-1024-9x.json'
$environmentPath = Join-Path $evidenceRoot `
  'final-affinity-1024-9x.environment.json'
$report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
$rates = @($report.samples.decode_tokens_per_second | Sort-Object)
$median = $rates[[math]::Floor($rates.Count / 2)]
$minimum = $rates[0]

if ($rates.Count -ne 9 -or
    [math]::Abs($median - 177.30624292681546) -gt 1e-9 -or
    [math]::Abs($minimum - 175.59583804564133) -gt 1e-9 -or
    -not $report.threshold_passed -or
    @($rates | Where-Object { $_ -lt 175 }).Count -ne 0 -or
    @($report.samples.output_sha256 | Sort-Object -Unique).Count -ne 1 -or
    $report.output_sha256 -ne
      'a54538eaaf6cc0b8b43cbafd489c7779f0f5206c93d5034fd3a16f4366a90523') {
  throw 'Archived performance evidence did not validate'
}

$expectedHashes = @{
  $reportPath = 'AD478DFDB3DF7D3560FB39EEB2BE854715BBB483A985C8E10793DF053C0C2D72'
  $environmentPath = 'C0CB352E99DCDD2C9BD487CB7501A1F231C97EF5A661C26446EF3C2E4F5EDA8D'
  (Join-Path $evidenceRoot 'mtp7-snap6-full-vocab-cpu-embedding.acl') =
    '2F348CCA96282A22650D9766CFFA81251EA10A5E34A089BCC91B0822AB5C1D0E'
  (Join-Path $evidenceRoot 'prompt.txt') =
    'D95A5E4DAD822BA9C84138F7A120017318BCB3A6A90E77246A8EC4EDE0E65D89'
}
foreach ($entry in $expectedHashes.GetEnumerator()) {
  if ((Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Key).Hash -ne
      $entry.Value) {
    throw "Hash mismatch: $($entry.Key)"
  }
}
```

## 5. Reproduce the representative quality tests

Replay the current pure-Q6_K 12-task throughput and acceptance calibration with
a splatted argument table. Supplying the Boolean recurrent-chain option this
way avoids PowerShell's cross-process Boolean argument ambiguity:

```powershell
$sweepArgs = @{
  PowerHome = 'D:\models\a3s-power\qwen38\power-home'
  FrVocabSizes = @(0, 8192)
  DraftMaxValues = @(7)
  MtpRecurrentSnapshots = 6
  MtpRecurrentChain = $false
  NumBatchValues = @(14)
  Policies = @('fixed')
  IncludeOffBaseline = $true
  Repetitions = 1
  MaxTokensCap = 128
  ProcessPriority = 'High'
  TargetDirectory = 'target-native-sm89-ninja'
  OutputRoot = 'target-qwen38-pure-q6-calibration'
  ModelHash =
    '562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727'
  VerifyModelFile = $true
  RequireHighPerformancePowerPlan = $true
}
.\tools\run-qwen38-mtp-sweep.ps1 @sweepArgs
```

The expected mode labels are `off-b14`, `frfull-k7-s6-b14-fixed`, and
`fr8192-k7-s6-b14-fixed`. Compare the new `sweep.json` with the checked-in
[raw calibration](quality/pure-q6-fr8192-calibration-rtx4090-1x.json). Eleven
of twelve tasks per mode were truncated at 128 tokens in the accepted capture;
do not present its answer counts as a general quality score.

The separate [quality matrix guide](quality/README.md#reproduce) contains the
complete commands for:

- the current full-vocabulary K7/S7 three-mode, 100-task, three-repetition
  MMLU/GSM8K/C-Eval matrix;
- the current guarded K7/S6 versus rollback-complete K7/S7 calibration; and
- deliberate replay of the historical prefix-FR matrix.

Use its reviewed offline task cache to avoid dataset drift. Do not compare its
request-wide throughput directly with the steady-state decode rate above.

## 6. Re-run implementation validation

The pre-submit source and evidence state used for this documentation passed the
following checks. Documentation and general library checks were rerun on
2026-08-21; the unchanged CUDA implementation rows retain the 2026-08-20
release validation:

| Check | Result |
| --- | --- |
| Rust library tests | 1,538 passed, 0 failed |
| CUDA speculative-runtime tests | 14 passed, 0 failed |
| CUDA release build | Passed |
| CUDA release Clippy with warnings denied | Passed |
| Python harness tests | 28 passed, 0 failed |
| Rust formatting | Passed |
| PowerShell syntax | 3 benchmark scripts and 21 documentation blocks parsed |
| Benchmark evidence | One-command verifier passed; pure-Q6_K and mixed assertions plus 14 pinned file hashes verified |
| Quality archive | 900 current requests completed; aggregate, environment, manifest, ACL, and source hashes pinned |
| Documentation links | All local links in changed documents resolved, 0 missing |
| Documentation site | TypeScript passed; 33 bilingual/versioned pages built and verified |

Re-run the same checks with:

```powershell
cargo fmt --all -- --check
cargo test --lib --quiet

cargo test --release --lib `
  --target-dir target-native-sm89-ninja `
  --no-default-features `
  --features llamacpp-cuda,llamacpp-mtp-fr `
  backend::llamacpp::speculative_runtime

cargo build --release --bins `
  --target-dir target-native-sm89-ninja `
  --no-default-features `
  --features llamacpp-cuda,llamacpp-mtp-fr

cargo clippy --release --bins `
  --target-dir target-native-sm89-ninja `
  --no-default-features `
  --features llamacpp-cuda,llamacpp-mtp-fr `
  -- -D warnings

$llamaCppCandidates = @(
  Get-ChildItem `
    (Join-Path $env:USERPROFILE '.cargo\git\checkouts') `
    -Directory -Filter 'llama-cpp-rs-*' |
  ForEach-Object {
    Join-Path $_.FullName `
      'dfd12e4\llama-cpp-sys-2\llama.cpp\gguf-py'
  } |
  Where-Object { Test-Path -LiteralPath $_ -PathType Container }
)
if ($llamaCppCandidates.Count -ne 1) {
  throw 'Expected one pinned llama.cpp gguf-py checkout'
}
$env:PYTHONPATH = $llamaCppCandidates[0]

py -3.13 .\tools\test_add_gguf_mtp_head.py
py -3.13 .\tools\test_build_fr_vocabulary.py
py -3.13 .\tools\test_qwen38_quality_eval.py
py -3.13 -m py_compile `
  .\tools\add-gguf-mtp-head.py `
  .\tools\build-fr-vocabulary.py `
  .\tools\qwen38_quality_eval.py `
  .\tools\qwen38_quality_report.py
```

The performance replay is deliberately not a CI test: the current gate requires
the pinned 22.88 GB pure-Q6_K model, a specific RTX 4090 host, administrative
clock control, and an exclusive benchmark window. The previous mixed gate uses
its separate 19.19 GB artifact identity. The checked-in JSON remains the
auditable result; new measurements should be added as new captures instead of
overwriting it.
