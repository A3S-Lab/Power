# Sequential Prism packing compare (single GPU)
#
# Same-host PTQ1_0 then PQ2_0 through one Power + one Prism port, restarting
# the upstream between packs. Prefer run-prism-packing-compare.ps1 when two
# Prism processes / enough VRAM are available.
#
# Example:
#   .\tools\run-prism-packing-compare-sequential.ps1 `
#     -Ptq1ModelPath "D:\Bonsai-demo\models\...\PTQ1_0.gguf" `
#     -Pq2ModelPath "D:\Bonsai-demo\models\...\PQ2_0.gguf" `
#     -BinDir "D:\Bonsai-demo\bin\cuda" `
#     -OutDir docs\benchmarks\bonsai2-27b-packing-rtx4090

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Ptq1ModelPath,
    [Parameter(Mandatory = $true)][string]$Pq2ModelPath,
    [Parameter(Mandatory = $true)][string]$BinDir,
    [string]$PowerUrl = "http://127.0.0.1:11435",
    [string]$PrismUrl = "http://127.0.0.1:8080",
    [string]$Ptq1Alias = "bonsai2-ptq1",
    [string]$Pq2Alias = "bonsai2-pq2",
    [int]$Port = 8080,
    [int]$Samples = 3,
    [int]$MaxTokens = 128,
    [string]$OutDir = "docs\benchmarks\bonsai2-27b-packing-rtx4090"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartPrism = Join-Path $ScriptDir "start-prism-upstream.ps1"
$Bench = Join-Path $ScriptDir "run-prism-baseline-bench.ps1"

function Stop-PortListener([int]$ListenPort) {
    $conns = Get-NetTCPConnection -LocalPort $ListenPort -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

function Wait-HttpOk([string]$Url, [int]$TimeoutSec = 600) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $Url -TimeoutSec 5
            if ($r.StatusCode -eq 200) { return }
        } catch {
            Start-Sleep -Seconds 3
        }
    }
    throw "timeout waiting for $Url"
}

function Invoke-Pack([string]$Label, [string]$ModelPath, [string]$Alias) {
    Write-Host "=== packing=$Label alias=$Alias ==="
    Stop-PortListener $Port
    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", $StartPrism,
        "-Profile", "baseline",
        "-Alias", $Alias,
        "-Model", $ModelPath,
        "-BinDir", $BinDir,
        "-Port", "$Port",
        "-Ctx", "8192",
        "-Np", "1"
    ) -PassThru -WindowStyle Minimized
    try {
        Wait-HttpOk "$PrismUrl/health" 900
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("a3s-power-pack-" + [guid]::NewGuid().ToString("n"))
        New-Item -ItemType Directory -Force -Path $tmp | Out-Null
        try {
            & $Bench -PowerUrl $PowerUrl -PrismUrl $PrismUrl -Model $Alias `
                -Samples $Samples -MaxTokens $MaxTokens -OutDir $tmp
            $file = Get-ChildItem $tmp -Filter "power-baseline-bench-*.json" -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if (-not $file) {
                throw "bench failed for $Label (no power-baseline-bench-*.json; lastExit=$LASTEXITCODE)"
            }
            $json = Get-Content $file.FullName -Raw | ConvertFrom-Json
            return [pscustomobject]@{
                packing = $Label
                model = $Alias
                model_path = $ModelPath
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
    finally {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
        Stop-PortListener $Port
    }
}

$ptq1 = Invoke-Pack "PTQ1_0" $Ptq1ModelPath $Ptq1Alias
$pq2 = Invoke-Pack "PQ2_0" $Pq2ModelPath $Pq2Alias

$decodeWinner = if ($null -eq $ptq1.predicted_per_second_mean -or $null -eq $pq2.predicted_per_second_mean) {
    "incomplete"
} elseif ($ptq1.predicted_per_second_mean -ge $pq2.predicted_per_second_mean) { "PTQ1_0" }
else { "PQ2_0" }

$prefillWinner = if ($null -eq $ptq1.prompt_per_second_mean -or $null -eq $pq2.prompt_per_second_mean) {
    "incomplete"
} elseif ($ptq1.prompt_per_second_mean -ge $pq2.prompt_per_second_mean) { "PTQ1_0" }
else { "PQ2_0" }

$summary = [ordered]@{
    schema = "a3s.power.prism-packing-compare.v1"
    host_note = "sequential single-GPU: restart Prism between packs; same Power URL"
    mode = "sequential"
    decode_winner = $decodeWinner
    prefill_winner = $prefillWinner
    ptq1 = $ptq1
    pq2 = $pq2
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$path = Join-Path $OutDir "power-packing-compare-$stamp.json"
($summary | ConvertTo-Json -Depth 10) | Set-Content -Path $path -Encoding utf8
Write-Host "WROTE $path"
Write-Host "decode_winner=$decodeWinner prefill_winner=$prefillWinner"
