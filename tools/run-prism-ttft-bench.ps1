# E2E TTFT through a3s-power Prism path (SSE vs buffered completion).
#
# Measures client-observed time-to-first-content-token on stream:true, and
# full-response wall on stream:false for the same prompt/tokens. Upstream
# must already be healthy; Power must have the model registered.
#
# Example:
#   .\tools\run-prism-ttft-bench.ps1 -Samples 5 -OutDir docs\benchmarks\bonsai2-27b-ptq1-rtx4090

[CmdletBinding()]
param(
    [string]$PowerUrl = "http://127.0.0.1:11435",
    [string]$PrismUrl = "http://127.0.0.1:8080",
    [string]$Model = "bonsai2-ptq1",
    [int]$Samples = 5,
    [int]$MaxTokens = 64,
    [string]$Prompt = "Write a Python function is_prime(n) that returns True if n is prime. Code only.",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

function Require-Reachable([string]$Uri, [string]$Label) {
    $r = Invoke-WebRequest -Uri $Uri -TimeoutSec 5
    if ($r.StatusCode -ne 200) { throw "$Label returned $($r.StatusCode)" }
}

Require-Reachable "$PrismUrl/health" "Prism health"
Require-Reachable "$PowerUrl/v1/models" "Power models"

function Measure-StreamTtft {
    param([hashtable]$Body)
    $json = $Body | ConvertTo-Json -Depth 6
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $ttftMs = $null
    $contentChars = 0
    $done = $false

    $req = [System.Net.HttpWebRequest]::Create("$PowerUrl/v1/chat/completions")
    $req.Method = "POST"
    $req.ContentType = "application/json"
    $req.Accept = "text/event-stream"
    $req.Timeout = 600000
    $req.ReadWriteTimeout = 600000
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $req.ContentLength = $bytes.Length
    $reqStream = $req.GetRequestStream()
    $reqStream.Write($bytes, 0, $bytes.Length)
    $reqStream.Close()

    $resp = $req.GetResponse()
    $reader = New-Object System.IO.StreamReader($resp.GetResponseStream())
    $lineBuf = New-Object System.Text.StringBuilder
    while (-not $reader.EndOfStream) {
        $line = $reader.ReadLine()
        if ($null -eq $line) { break }
        if ($line.StartsWith("data:")) {
            $payload = $line.Substring(5).Trim()
            if ($payload -eq "[DONE]") {
                $done = $true
                break
            }
            try {
                $obj = $payload | ConvertFrom-Json
                $deltaText = $null
                if ($obj.choices -and $obj.choices[0].delta) {
                    $d = $obj.choices[0].delta
                    # Bonsai/Prism may emit reasoning_content before content.
                    if ($d.content -and [string]$d.content.Length -gt 0) {
                        $deltaText = [string]$d.content
                    } elseif ($d.reasoning_content -and [string]$d.reasoning_content.Length -gt 0) {
                        $deltaText = [string]$d.reasoning_content
                    } elseif ($d.thinking -and [string]$d.thinking.Length -gt 0) {
                        $deltaText = [string]$d.thinking
                    }
                }
                if ($deltaText) {
                    if ($null -eq $ttftMs) {
                        $ttftMs = $sw.ElapsedMilliseconds
                    }
                    $contentChars += $deltaText.Length
                }
                if ($obj.choices -and $obj.choices[0].finish_reason) {
                    $done = $true
                }
            } catch {
                # ignore malformed SSE frames
            }
        }
    }
    $reader.Close()
    $resp.Close()
    $sw.Stop()
    return [pscustomobject]@{
        ttft_ms = $ttftMs
        wall_ms = $sw.ElapsedMilliseconds
        content_chars = $contentChars
        done = $done
    }
}

function Measure-BufferedWall {
    param([hashtable]$Body)
    $json = $Body | ConvertTo-Json -Depth 6
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri "$PowerUrl/v1/chat/completions" -Method POST `
        -ContentType "application/json" -Body $json -TimeoutSec 600
    $sw.Stop()
    $text = ""
    if ($resp.choices -and $resp.choices[0].message) {
        $text = [string]$resp.choices[0].message.content
    }
    return [pscustomobject]@{
        wall_ms = $sw.ElapsedMilliseconds
        content_chars = $text.Length
        finish_reason = $resp.choices[0].finish_reason
    }
}

$streamRows = @()
$bufferedRows = @()

# Warm health cache / model once
$null = Measure-BufferedWall @{
    model = $Model
    messages = @(@{ role = "user"; content = "ping" })
    temperature = 0.0
    max_tokens = 8
    stream = $false
}

for ($i = 1; $i -le $Samples; $i++) {
    $streamBody = @{
        model = $Model
        messages = @(@{ role = "user"; content = $Prompt })
        temperature = 0.0
        max_tokens = $MaxTokens
        stream = $true
    }
    $s = Measure-StreamTtft -Body $streamBody
    $streamRows += [pscustomobject]@{
        sample = $i
        mode = "stream"
        ttft_ms = $s.ttft_ms
        wall_ms = $s.wall_ms
        content_chars = $s.content_chars
    }
    Write-Host ("stream sample={0} ttft_ms={1} wall_ms={2}" -f $i, $s.ttft_ms, $s.wall_ms)

    $bufBody = @{
        model = $Model
        messages = @(@{ role = "user"; content = $Prompt })
        temperature = 0.0
        max_tokens = $MaxTokens
        stream = $false
    }
    $b = Measure-BufferedWall -Body $bufBody
    $bufferedRows += [pscustomobject]@{
        sample = $i
        mode = "buffered"
        ttft_ms = $null
        wall_ms = $b.wall_ms
        content_chars = $b.content_chars
        finish_reason = $b.finish_reason
    }
    Write-Host ("buffered sample={0} wall_ms={1}" -f $i, $b.wall_ms)
}

$ttfts = @($streamRows | Where-Object { $_.ttft_ms -ne $null } | ForEach-Object { [double]$_.ttft_ms })
$streamWalls = @($streamRows | ForEach-Object { [double]$_.wall_ms })
$bufWalls = @($bufferedRows | ForEach-Object { [double]$_.wall_ms })

function Median([double[]]$xs) {
    if (-not $xs -or $xs.Count -eq 0) { return $null }
    $sorted = $xs | Sort-Object
    $n = $sorted.Count
    if ($n % 2 -eq 1) { return $sorted[[int]($n / 2)] }
    return ($sorted[$n / 2 - 1] + $sorted[$n / 2]) / 2.0
}

$summary = [ordered]@{
    schema = "a3s.power.prism-ttft-bench.v1"
    profile = "baseline"
    model = $Model
    power_url = $PowerUrl
    prism_url = $PrismUrl
    samples = $Samples
    max_tokens = $MaxTokens
    prompt = $Prompt
    stream_ttft_ms_median = Median $ttfts
    stream_ttft_ms_mean = if ($ttfts.Count) { ($ttfts | Measure-Object -Average).Average } else { $null }
    stream_ttft_ms_min = if ($ttfts.Count) { ($ttfts | Measure-Object -Minimum).Minimum } else { $null }
    stream_ttft_ms_max = if ($ttfts.Count) { ($ttfts | Measure-Object -Maximum).Maximum } else { $null }
    stream_wall_ms_median = Median $streamWalls
    buffered_wall_ms_median = Median $bufWalls
    note = "TTFT is client-observed first non-empty delta content|reasoning_content|thinking on Power SSE; buffered has no first-token until full JSON."
    stream_rows = $streamRows
    buffered_rows = $bufferedRows
}

$json = $summary | ConvertTo-Json -Depth 8
Write-Host $json

if ($OutDir) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $path = Join-Path $OutDir "power-ttft-bench-$stamp.json"
    Set-Content -Path $path -Value $json -Encoding utf8
    Write-Host "WROTE $path"
}
