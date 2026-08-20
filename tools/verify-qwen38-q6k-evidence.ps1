[CmdletBinding()]
param(
  [string]$EvidenceRoot = '',
  [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) {
  $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
  $EvidenceRoot = Join-Path $scriptRoot `
    '..\docs\benchmarks\qwen3.8-27b-q6k-rtx4090'
}

function Assert-Equal {
  param(
    [Parameter(Mandatory = $true)]
    [AllowNull()]
    [object]$Actual,
    [Parameter(Mandatory = $true)]
    [AllowNull()]
    [object]$Expected,
    [Parameter(Mandatory = $true)]
    [string]$Label
  )

  if ($Actual -cne $Expected) {
    throw "$Label mismatch: expected '$Expected', got '$Actual'"
  }
}

function Assert-Near {
  param(
    [Parameter(Mandatory = $true)]
    [double]$Actual,
    [Parameter(Mandatory = $true)]
    [double]$Expected,
    [Parameter(Mandatory = $true)]
    [string]$Label,
    [double]$Tolerance = 1e-9
  )

  if ([math]::Abs($Actual - $Expected) -gt $Tolerance) {
    throw "$Label mismatch: expected '$Expected', got '$Actual'"
  }
}

function Assert-FileHash {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$Expected,
    [Parameter(Mandatory = $true)]
    [string]$Label
  )

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "$Label is missing: $Path"
  }

  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
  if ($actual -cne $Expected.ToUpperInvariant()) {
    throw "$Label SHA-256 mismatch: expected '$Expected', got '$actual'"
  }
}

$resolvedEvidenceRoot = (Resolve-Path -LiteralPath $EvidenceRoot).Path
$currentPath = Join-Path $resolvedEvidenceRoot `
  'quality\full-vocabulary-s7-current-rtx4090-3x.json'
$historicalReportPath = Join-Path $resolvedEvidenceRoot `
  'final-affinity-1024-9x.json'
$historicalEnvironmentPath = Join-Path $resolvedEvidenceRoot `
  'final-affinity-1024-9x.environment.json'
$s6ConfigPath = Join-Path $resolvedEvidenceRoot `
  'mtp7-snap6-full-vocab-cpu-embedding.acl'
$s7ConfigPath = Join-Path $resolvedEvidenceRoot `
  'mtp7-snap7-full-vocab-cpu-embedding.acl'
$promptPath = Join-Path $resolvedEvidenceRoot 'prompt.txt'

$expectedHashes = @(
  @{
    Path = $currentPath
    Hash = '83D418651959E49074511BCD03CA33633BD0E26BC079D9758F26519DB6DBEB31'
    Label = 'Current compact evidence'
  },
  @{
    Path = $historicalReportPath
    Hash = 'AD478DFDB3DF7D3560FB39EEB2BE854715BBB483A985C8E10793DF053C0C2D72'
    Label = 'Historical performance report'
  },
  @{
    Path = $historicalEnvironmentPath
    Hash = 'C0CB352E99DCDD2C9BD487CB7501A1F231C97EF5A661C26446EF3C2E4F5EDA8D'
    Label = 'Historical environment receipt'
  },
  @{
    Path = $s6ConfigPath
    Hash = '2F348CCA96282A22650D9766CFFA81251EA10A5E34A089BCC91B0822AB5C1D0E'
    Label = 'K7/S6 runtime configuration'
  },
  @{
    Path = $s7ConfigPath
    Hash = '759EF6E5E60A08939ED747558992FA3031D63D2ECD59DACFDAE59790CC6FF79A'
    Label = 'K7/S7 runtime configuration'
  },
  @{
    Path = $promptPath
    Hash = 'D95A5E4DAD822BA9C84138F7A120017318BCB3A6A90E77246A8EC4EDE0E65D89'
    Label = 'Benchmark prompt'
  }
)

foreach ($expectedHash in $expectedHashes) {
  Assert-FileHash `
    -Path $expectedHash.Path `
    -Expected $expectedHash.Hash `
    -Label $expectedHash.Label
}

$current = Get-Content -Raw -LiteralPath $currentPath | ConvertFrom-Json
Assert-Equal $current.schema `
  'a3s.power.qwen38-full-vocabulary-summary.v1' 'Current evidence schema'
Assert-Equal $current.identity.tbq4_model_sha256 `
  '5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6' `
  'TBQ4 model identity'
Assert-Equal ([long]$current.identity.tbq4_model_bytes) ([long]19187686464) `
  'TBQ4 model byte length'
Assert-Equal $current.identity.gpu 'NVIDIA GeForce RTX 4090' 'Acceptance GPU'
Assert-Equal ([int]$current.workload.tasks) 100 'Quality task count'
Assert-Equal ([int]$current.workload.repetitions) 3 'Quality repetition count'
Assert-Equal ([int]$current.workload.num_batch) 14 'Quality batch size'
Assert-Equal ([int]$current.workload.warmup_requests_per_run) 1 `
  'Quality warm-up count'

$currentModes = @($current.modes)
Assert-Equal $currentModes.Count 3 'Current mode count'
$s7Modes = @($currentModes | Where-Object {
    $_.name -eq 'tbq4-mtp-full-vocab-k7-s7'
  })
Assert-Equal $s7Modes.Count 1 'K7/S7 mode count'
$s7 = $s7Modes[0]
Assert-Equal ([int]$s7.lenient_score) 76 'K7/S7 lenient score'
Assert-Equal ([int]$s7.strict_score) 66 'K7/S7 strict score'
Assert-Equal ([int]$s7.prediction_stable_tasks) 100 `
  'K7/S7 stable prediction count'
Assert-Equal ([int]$s7.content_stable_tasks) 100 `
  'K7/S7 stable content count'
Assert-Near ([double]$s7.mean_workload_tokens_per_second) `
  83.22814601950864 'K7/S7 request-wide throughput'
Assert-Near ([double]$s7.weighted_acceptance_rate) `
  0.5132958564931783 'K7/S7 proposal acceptance'
Assert-Equal ($s7.fallback_replays_per_run -join ',') '0,0,0' `
  'K7/S7 fallback replay counts'
Assert-Equal ($s7.rollback_guard_activations_per_run -join ',') '0,0,0' `
  'K7/S7 rollback guard activations'

$s6Gate = $current.peak_gates.k7_s6_guarded
Assert-Equal ([int]$s6Gate.samples) 9 'K7/S6 sample count'
Assert-Equal ([int]$s6Gate.samples_at_or_above_175) 9 `
  'K7/S6 samples at or above 175 token/s'
Assert-Near ([double]$s6Gate.median_decode_tokens_per_second) `
  177.71645508101074 'K7/S6 median steady decode'
Assert-Near ([double]$s6Gate.minimum_decode_tokens_per_second) `
  176.72867856598307 'K7/S6 minimum steady decode'

$s7Gate = $current.peak_gates.k7_s7_rollback_complete
Assert-Equal ([int]$s7Gate.samples) 9 'K7/S7 sample count'
Assert-Equal ([int]$s7Gate.samples_at_or_above_175) 5 `
  'K7/S7 samples at or above 175 token/s'
Assert-Near ([double]$s7Gate.median_decode_tokens_per_second) `
  175.20889378841997 'K7/S7 median steady decode'
Assert-Near ([double]$s7Gate.minimum_decode_tokens_per_second) `
  174.2211132623549 'K7/S7 minimum steady decode'
Assert-Equal $s7Gate.output_sha256 `
  'a54538eaaf6cc0b8b43cbafd489c7779f0f5206c93d5034fd3a16f4366a90523' `
  'K7/S7 deterministic output identity'

$historical = Get-Content -Raw -LiteralPath $historicalReportPath |
  ConvertFrom-Json
$rates = @($historical.samples.decode_tokens_per_second | Sort-Object)
$outputHashes = @($historical.samples.output_sha256 | Sort-Object -Unique)
$median = [double]$rates[[math]::Floor($rates.Count / 2)]
$minimum = [double]$rates[0]

Assert-Equal $historical.schema 'a3s.power.speculative-benchmark.v1' `
  'Historical report schema'
Assert-Equal $rates.Count 9 'Historical sample count'
Assert-Near $median 177.30624292681546 'Historical median steady decode'
Assert-Near $minimum 175.59583804564133 'Historical minimum steady decode'
Assert-Equal ([bool]$historical.threshold_passed) $true `
  'Historical threshold result'
Assert-Equal @($rates | Where-Object { $_ -lt 175 }).Count 0 `
  'Historical samples below 175 token/s'
Assert-Equal $outputHashes.Count 1 'Historical unique output identity count'
Assert-Equal $historical.output_sha256 `
  'a54538eaaf6cc0b8b43cbafd489c7779f0f5206c93d5034fd3a16f4366a90523' `
  'Historical deterministic output identity'
Assert-Equal @($historical.samples | Where-Object {
    $_.completion_tokens -ne 1024
  }).Count 0 'Historical short output count'

$completedRequests = [int]$current.workload.tasks *
  [int]$current.workload.repetitions * $currentModes.Count
$result = [ordered]@{
  schema = 'a3s.power.evidence-verification.v1'
  status = 'passed'
  evidence_root = $resolvedEvidenceRoot
  verified_file_hashes = $expectedHashes.Count
  quality = [ordered]@{
    completed_requests = $completedRequests
    lenient_score = [int]$s7.lenient_score
    strict_score = [int]$s7.strict_score
    request_wide_tokens_per_second = [double]$s7.mean_workload_tokens_per_second
    weighted_acceptance_rate = [double]$s7.weighted_acceptance_rate
    fallback_replays = 0
  }
  steady_decode = [ordered]@{
    rollback_complete_k7_s7_median = `
      [double]$s7Gate.median_decode_tokens_per_second
    rollback_complete_k7_s7_minimum = `
      [double]$s7Gate.minimum_decode_tokens_per_second
    guarded_k7_s6_median = [double]$s6Gate.median_decode_tokens_per_second
    guarded_k7_s6_minimum = [double]$s6Gate.minimum_decode_tokens_per_second
  }
  deterministic_output_sha256 = $s7Gate.output_sha256
}

if ($Json) {
  $result | ConvertTo-Json -Depth 5
} else {
  [pscustomobject]$result
}
