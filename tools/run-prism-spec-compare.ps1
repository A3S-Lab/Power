# Compare Prism speculation profiles through a3s-power (baseline vs mtp/dspark).
#
# Requires a healthy Prism upstream already started with the profile under test,
# and Power registered with matching prism_profile in config.acl.
#
# Example (after MTP upstream is up):
#   .\tools\run-prism-spec-compare.ps1 -Model bonsai2-ptq1-mtp -ProfileLabel mtp `
#     -OutDir docs\benchmarks\bonsai2-27b-ptq1-rtx4090

[CmdletBinding()]
param(
    [string]$PowerUrl = "http://127.0.0.1:11435",
    [string]$PrismUrl = "http://127.0.0.1:8080",
    [string]$Model = "bonsai2-ptq1",
    [string]$ProfileLabel = "baseline",
    [int]$Samples = 3,
    [int]$MaxTokens = 128,
    [string]$Prompt = "Write a Python function is_prime(n) that returns True if n is prime. Code only.",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

$null = Invoke-WebRequest -Uri "$PrismUrl/health" -TimeoutSec 5
$null = Invoke-WebRequest -Uri "$PowerUrl/v1/models" -TimeoutSec 5

$rows = @()
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
    $draftN = if ($timings) { $timings.draft_n } else { $null }
    $draftAcc = if ($timings) { $timings.draft_n_accepted } else { $null }
    $accept = $null
    if ($null -ne $draftN -and [double]$draftN -gt 0 -and $null -ne $draftAcc) {
        $accept = [double]$draftAcc / [double]$draftN
    }
    $row = [ordered]@{
        sample = $i
        profile = $ProfileLabel
        wall_ms = $sw.ElapsedMilliseconds
        source = if ($resp.prism_upstream) { $resp.prism_upstream.source } else { $null }
        predicted_per_second = if ($timings) { $timings.predicted_per_second } else { $null }
        predicted_n = if ($timings) { $timings.predicted_n } else { $null }
        prompt_per_second = if ($timings) { $timings.prompt_per_second } else { $null }
        draft_n = $draftN
        draft_n_accepted = $draftAcc
        draft_accept_rate = $accept
        finish_reason = $resp.choices[0].finish_reason
    }
    $rows += [pscustomobject]$row
    Write-Host ("sample={0} tok/s={1} draft_n={2} accept={3:N3} wall_ms={4}" -f `
        $i, $row.predicted_per_second, $draftN, $accept, $row.wall_ms)
}

$speeds = @($rows | Where-Object { $_.predicted_per_second -ne $null } | ForEach-Object { [double]$_.predicted_per_second })
$accepts = @($rows | Where-Object { $_.draft_accept_rate -ne $null } | ForEach-Object { [double]$_.draft_accept_rate })
$summary = [ordered]@{
    schema = "a3s.power.prism-spec-compare.v1"
    profile = $ProfileLabel
    model = $Model
    power_url = $PowerUrl
    prism_url = $PrismUrl
    samples = $Samples
    max_tokens = $MaxTokens
    prompt = $Prompt
    predicted_per_second_mean = if ($speeds.Count) { ($speeds | Measure-Object -Average).Average } else { $null }
    predicted_per_second_min = if ($speeds.Count) { ($speeds | Measure-Object -Minimum).Minimum } else { $null }
    predicted_per_second_max = if ($speeds.Count) { ($speeds | Measure-Object -Maximum).Maximum } else { $null }
    draft_accept_rate_mean = if ($accepts.Count) { ($accepts | Measure-Object -Average).Average } else { $null }
    speculation_engaged = ($rows | Where-Object { $_.draft_n -ne $null -and [double]$_.draft_n -gt 0 }).Count -gt 0
    rows = $rows
}

$json = $summary | ConvertTo-Json -Depth 8
Write-Host $json
if ($OutDir) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $path = Join-Path $OutDir "power-spec-$ProfileLabel-$stamp.json"
    Set-Content -Path $path -Value $json -Encoding utf8
    Write-Host "WROTE $path"
}
