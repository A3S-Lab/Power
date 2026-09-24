# Same-host PTQ1_0 vs PQ2_0 packing compare through a3s-power.
#
# First-principles: packing advisory tables are hypotheses until this script
# publishes request-wide timings under docs/benchmarks/bonsai2-…/.
#
# Requires two Prism upstreams (or sequential restarts) already serving the
# registered aliases -Ptq1Model / -Pq2Model on -PowerUrl.
#
# Example (after both servers are up and models registered):
#   .\tools\run-prism-packing-compare.ps1 `
#     -Ptq1Model bonsai2-ptq1 -Pq2Model bonsai2-pq2 `
#     -OutDir docs\benchmarks\bonsai2-27b-packing-rtx4090

[CmdletBinding()]
param(
    [string]$PowerUrl = "http://127.0.0.1:11435",
    [string]$Ptq1PrismUrl = "http://127.0.0.1:8080",
    [string]$Pq2PrismUrl = "http://127.0.0.1:8081",
    [string]$Ptq1Model = "bonsai2-ptq1",
    [string]$Pq2Model = "bonsai2-pq2",
    [int]$Samples = 3,
    [int]$MaxTokens = 128,
    [string]$Prompt = "Write a Python function is_prime(n) that returns True if n is prime. Code only.",
    [string]$OutDir = "docs\benchmarks\bonsai2-27b-packing-rtx4090"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Bench = Join-Path $ScriptDir "run-prism-baseline-bench.ps1"

function Invoke-PackingBench([string]$Label, [string]$PrismUrl, [string]$Model) {
    Write-Host "=== packing=$Label model=$Model prism=$PrismUrl ==="
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("a3s-power-packing-" + [guid]::NewGuid().ToString("n"))
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    try {
            & $Bench -PowerUrl $PowerUrl -PrismUrl $PrismUrl -Model $Model `
            -Samples $Samples -MaxTokens $MaxTokens -Prompt $Prompt -OutDir $tmp
        $file = Get-ChildItem $tmp -Filter "power-baseline-bench-*.json" -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $file) {
            throw "baseline bench failed for $Label (no JSON; lastExit=$LASTEXITCODE)"
        }
        $json = Get-Content $file.FullName -Raw | ConvertFrom-Json
        return [pscustomobject]@{
            packing = $Label
            model = $Model
            prism_url = $PrismUrl
            predicted_per_second_mean = $json.predicted_per_second_mean
            predicted_per_second_min = $json.predicted_per_second_min
            predicted_per_second_max = $json.predicted_per_second_max
            prompt_per_second_mean = (
                @($json.rows | Where-Object { $_.prompt_per_second -ne $null } |
                    ForEach-Object { [double]$_.prompt_per_second } |
                    Measure-Object -Average).Average
            )
            artifact = $file.FullName
            rows = $json.rows
        }
    }
    finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
}

$ptq1 = Invoke-PackingBench "PTQ1_0" $Ptq1PrismUrl $Ptq1Model
$pq2 = Invoke-PackingBench "PQ2_0" $Pq2PrismUrl $Pq2Model

$decodeWinner = if ($null -eq $ptq1.predicted_per_second_mean -or $null -eq $pq2.predicted_per_second_mean) {
    "incomplete"
}
elseif ($ptq1.predicted_per_second_mean -ge $pq2.predicted_per_second_mean) { "PTQ1_0" }
else { "PQ2_0" }

$prefillWinner = if ($null -eq $ptq1.prompt_per_second_mean -or $null -eq $pq2.prompt_per_second_mean) {
    "incomplete"
}
elseif ($ptq1.prompt_per_second_mean -ge $pq2.prompt_per_second_mean) { "PTQ1_0" }
else { "PQ2_0" }

$summary = [ordered]@{
    schema = "a3s.power.prism-packing-compare.v1"
    host_note = "same Power URL; distinct Prism upstreams per packing"
    power_url = $PowerUrl
    samples = $Samples
    max_tokens = $MaxTokens
    prompt = $Prompt
    decode_winner = $decodeWinner
    prefill_winner = $prefillWinner
    advisory_hypothesis = "Ada/L4 often prefer PTQ1_0 decode; PQ2_0 often wins prefill — this capture is the evidence"
    packs = @($ptq1, $pq2)
}

$json = $summary | ConvertTo-Json -Depth 10
Write-Host $json

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$path = Join-Path $OutDir "power-packing-compare-$stamp.json"
Set-Content -Path $path -Value $json -Encoding utf8
Write-Host "WROTE $path"
Write-Host "decode_winner=$decodeWinner prefill_winner=$prefillWinner"
