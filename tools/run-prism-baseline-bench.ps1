# Reproducible Prism baseline acceleration timings through a3s-power.
#
# Requires:
#   - Prism llama-server on -PrismUrl (health OK)
#   - a3s-power on -PowerUrl with prism_upstream + prism_profile=baseline
#   - Model already registered (POST /v1/models) as -Model
#
# Example:
#   .\tools\run-prism-baseline-bench.ps1 -Samples 3 -OutDir docs\benchmarks\bonsai2-27b-ptq1-rtx4090

[CmdletBinding()]
param(
    [string]$PowerUrl = "http://127.0.0.1:11435",
    [string]$PrismUrl = "http://127.0.0.1:8080",
    [string]$Model = "bonsai2-ptq1",
    [int]$Samples = 3,
    [int]$MaxTokens = 128,
    [string]$Prompt = "Write a Python function is_prime(n) that returns True if n is prime. Code only.",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

function Require-Health([string]$Base, [string]$Label) {
    $r = Invoke-WebRequest -Uri "$Base/health" -TimeoutSec 5
    if ($r.StatusCode -ne 200) { throw "$Label health returned $($r.StatusCode)" }
}

Require-Health $PrismUrl "Prism"
# Power may not expose /health the same way — probe models list.
$null = Invoke-WebRequest -Uri "$PowerUrl/v1/models" -TimeoutSec 5

$results = @()
for ($i = 1; $i -le $Samples; $i++) {
    $body = @{
        model = $Model
        messages = @(@{ role = "user"; content = $Prompt })
        temperature = 0.0
        max_tokens = $MaxTokens
        stream = $false
    } | ConvertTo-Json -Depth 6

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri "$PowerUrl/v1/chat/completions" -Method POST `
        -ContentType "application/json" -Body $body -TimeoutSec 600
    $sw.Stop()

    $timings = $null
    if ($resp.prism_upstream -and $resp.prism_upstream.timings) {
        $timings = $resp.prism_upstream.timings
    }
    $row = [ordered]@{
        sample = $i
        wall_ms = $sw.ElapsedMilliseconds
        source = if ($resp.prism_upstream) { $resp.prism_upstream.source } else { $null }
        predicted_per_second = if ($timings) { $timings.predicted_per_second } else { $null }
        predicted_n = if ($timings) { $timings.predicted_n } else { $null }
        predicted_ms = if ($timings) { $timings.predicted_ms } else { $null }
        prompt_per_second = if ($timings) { $timings.prompt_per_second } else { $null }
        draft_n = if ($timings) { $timings.draft_n } else { $null }
        draft_n_accepted = if ($timings) { $timings.draft_n_accepted } else { $null }
        finish_reason = $resp.choices[0].finish_reason
    }
    $results += [pscustomobject]$row
    Write-Host ("sample={0} predicted_per_second={1} wall_ms={2}" -f $i, $row.predicted_per_second, $row.wall_ms)
}

$speeds = @($results | Where-Object { $_.predicted_per_second -ne $null } | ForEach-Object { [double]$_.predicted_per_second })
$summary = [ordered]@{
    profile = "baseline"
    model = $Model
    power_url = $PowerUrl
    prism_url = $PrismUrl
    samples = $Samples
    max_tokens = $MaxTokens
    prompt = $Prompt
    predicted_per_second_mean = if ($speeds.Count) { ($speeds | Measure-Object -Average).Average } else { $null }
    predicted_per_second_min = if ($speeds.Count) { ($speeds | Measure-Object -Minimum).Minimum } else { $null }
    predicted_per_second_max = if ($speeds.Count) { ($speeds | Measure-Object -Maximum).Maximum } else { $null }
    dspark_compare = "skipped — Bonsai-2 official dspark-dflash drafter not pinned"
    rows = $results
}

$json = $summary | ConvertTo-Json -Depth 8
Write-Host $json

if ($OutDir) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $path = Join-Path $OutDir "power-baseline-bench-$stamp.json"
    Set-Content -Path $path -Value $json -Encoding utf8
    Write-Host "WROTE $path"
}
