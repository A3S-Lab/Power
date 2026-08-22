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
    '..\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark'
}

function Assert-Equal {
  param(
    [Parameter(Mandatory = $true)][AllowNull()][object]$Actual,
    [Parameter(Mandatory = $true)][AllowNull()][object]$Expected,
    [Parameter(Mandatory = $true)][string]$Label
  )
  if ($Actual -cne $Expected) {
    throw "$Label mismatch: expected '$Expected', got '$Actual'"
  }
}

function Assert-Near {
  param(
    [Parameter(Mandatory = $true)][double]$Actual,
    [Parameter(Mandatory = $true)][double]$Expected,
    [Parameter(Mandatory = $true)][string]$Label,
    [double]$Tolerance = 1e-9
  )
  if ([Math]::Abs($Actual - $Expected) -gt $Tolerance) {
    throw "$Label mismatch: expected '$Expected', got '$Actual'"
  }
}

function Assert-FileHash {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Expected,
    [Parameter(Mandatory = $true)][string]$Label
  )
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "$Label is missing: $Path"
  }
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
  if ($actual -cne $Expected.ToUpperInvariant()) {
    throw "$Label SHA-256 mismatch: expected '$Expected', got '$actual'"
  }
}

function Get-Median {
  param([Parameter(Mandatory = $true)][double[]]$Values)
  if ($Values.Count -eq 0) {
    throw 'Cannot compute a median from an empty set'
  }
  $ordered = @($Values | Sort-Object)
  $middle = [Math]::Floor($ordered.Count / 2)
  if (($ordered.Count % 2) -eq 1) {
    return [double]$ordered[$middle]
  }
  return ([double]$ordered[$middle - 1] + [double]$ordered[$middle]) / 2.0
}

$root = (Resolve-Path -LiteralPath $EvidenceRoot).Path
$evidencePath = Join-Path $root 'evidence.json'
$targetPath = Join-Path $root 'target-only.json'
$dsparkPath = Join-Path $root 'dspark-q4-k10-s6.json'
$targetConfigPath = Join-Path $root 'target-only.acl'
$dsparkConfigPath = Join-Path $root 'dspark-q4-k10-s6.acl'
$promptPath = Join-Path (Split-Path -Parent $root) 'prompt.txt'

Assert-FileHash $targetPath `
  '549988CD4BCDCEA0A6F055168B3DABD8CB037D4303009EAE251926A7BDE9838E' `
  'Target-only report'
Assert-FileHash $dsparkPath `
  '6D36F3121B96B51B563AB9CDE160BEF34552D22A74BC91C784D2962065C0E0F9' `
  'DSpark report'
Assert-FileHash $targetConfigPath `
  '60D529652B4019C5E590E3D06981F60C1BAB442A351D62A0D0DA72D97A00848C' `
  'Target-only ACL'
Assert-FileHash $dsparkConfigPath `
  '3B7A302B9DF77C5CC7F581E0B531791FCB59610F080D18899F82E5D69FB75C74' `
  'DSpark ACL'
Assert-FileHash $promptPath `
  'D95A5E4DAD822BA9C84138F7A120017318BCB3A6A90E77246A8EC4EDE0E65D89' `
  'Benchmark prompt'

$evidence = Get-Content -Raw -Encoding UTF8 -LiteralPath $evidencePath |
  ConvertFrom-Json
$target = Get-Content -Raw -Encoding UTF8 -LiteralPath $targetPath |
  ConvertFrom-Json
$dspark = Get-Content -Raw -Encoding UTF8 -LiteralPath $dsparkPath |
  ConvertFrom-Json

Assert-Equal $target.schema 'a3s.power.speculative-benchmark.v1' `
  'Target schema'
Assert-Equal $dspark.schema 'a3s.power.speculative-benchmark.v1' `
  'DSpark schema'
Assert-Equal $target.identity.power_commit `
  'c272e35365fb25a057a8ee4c04c20d8a35cb4b05' 'Target commit'
Assert-Equal $dspark.identity.power_commit $target.identity.power_commit `
  'Paired commit'
Assert-Equal $target.identity.speculative.mode 'off' 'Target mode'
Assert-Equal $dspark.identity.speculative.mode 'dspark' 'DSpark mode'
Assert-Equal $target.identity.model_sha256 $dspark.identity.model_sha256 `
  'Paired target identity'
Assert-Equal $target.workload.request_sha256 $dspark.workload.request_sha256 `
  'Paired request identity'
Assert-Equal $target.output_sha256 $dspark.output_sha256 `
  'Paired output identity'
Assert-Equal @($target.samples).Count 3 'Target sample count'
Assert-Equal @($dspark.samples).Count 3 'DSpark sample count'

$targetRates = @($target.samples | ForEach-Object {
  Assert-Equal $_.completion_tokens 256 'Target completion length'
  Assert-Equal $_.output_sha256 $target.output_sha256 'Target sample output'
  [double]$_.decode_tokens_per_second
})
$dsparkRates = @($dspark.samples | ForEach-Object {
  Assert-Equal $_.completion_tokens 256 'DSpark completion length'
  Assert-Equal $_.output_sha256 $dspark.output_sha256 'DSpark sample output'
  if ([double]$_.decode_tokens_per_second -lt 160.0) {
    throw "DSpark sample fell below 160 token/s: $($_.decode_tokens_per_second)"
  }
  [double]$_.decode_tokens_per_second
})

$targetMedian = Get-Median $targetRates
$dsparkMedian = Get-Median $dsparkRates
$targetMinimum = [double]($targetRates | Measure-Object -Minimum).Minimum
$dsparkMinimum = [double]($dsparkRates | Measure-Object -Minimum).Minimum
$decodeSpeedup = $dsparkMedian / $targetMedian

Assert-Near $targetMedian ([double]$target.median_decode_tokens_per_second) `
  'Target median'
Assert-Near $dsparkMedian ([double]$dspark.median_decode_tokens_per_second) `
  'DSpark median'
Assert-Near $targetMinimum ([double]$target.minimum_decode_tokens_per_second) `
  'Target minimum'
Assert-Near $dsparkMinimum ([double]$dspark.minimum_decode_tokens_per_second) `
  'DSpark minimum'
Assert-Near $decodeSpeedup ([double]$evidence.comparison.decode_speedup) `
  'Decode speedup'
Assert-Equal $evidence.comparison.output_sha256 $target.output_sha256 `
  'Evidence output identity'
Assert-Equal $evidence.comparison.exact_greedy_output_match $true `
  'Exact greedy output match'

$result = [ordered]@{
  status = 'passed'
  power_commit = $target.identity.power_commit
  request_sha256 = $target.workload.request_sha256
  output_sha256 = $target.output_sha256
  target_only_median_tokens_per_second = $targetMedian
  target_only_minimum_tokens_per_second = $targetMinimum
  dspark_median_tokens_per_second = $dsparkMedian
  dspark_minimum_tokens_per_second = $dsparkMinimum
  decode_speedup = $decodeSpeedup
  exact_greedy_output_match = $true
}

if ($Json) {
  $result | ConvertTo-Json -Depth 4
} else {
  Write-Host 'DSpark evidence: PASS'
  Write-Host ("  target-only median: {0:N3} token/s" -f $targetMedian)
  Write-Host ("  DSpark median:      {0:N3} token/s" -f $dsparkMedian)
  Write-Host ("  speedup:             {0:N3}x" -f $decodeSpeedup)
  Write-Host '  exact output:        yes'
}
