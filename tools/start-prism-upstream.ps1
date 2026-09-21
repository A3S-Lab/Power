# Start a Prism llama-server with a Power-aligned acceleration profile.
#
# Profiles (see docs/prism-acceleration-plan.md):
#   baseline — FA + offload; prompt-cache friendly; multi-slot OK
#   dspark   — requires -Drafter; single slot; no cross-request cache
#   kv4      — baseline + Q4 KV cache
#
# Example:
#   .\tools\start-prism-upstream.ps1 -Profile baseline -Alias bonsai2-ptq1 `
#     -Model "D:\Bonsai-demo\models\bonsai2-gguf\27B\Ternary-Bonsai-2-27B-PTQ1_0.gguf" `
#     -BinDir "D:\Bonsai-demo\bin\cuda" -Port 8080

[CmdletBinding()]
param(
    [ValidateSet("baseline", "dspark", "kv4")]
    [string]$Profile = "baseline",
    [Parameter(Mandatory = $true)]
    [string]$Model,
    [Parameter(Mandatory = $true)]
    [string]$BinDir,
    [string]$Drafter = "",
    [string]$Mmproj = "",
    [string]$Alias = "bonsai2-ptq1",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8080,
    [int]$Ctx = 8192,
    [int]$Ngl = 99,
    [int]$Np = 4
)

$ErrorActionPreference = "Stop"
$server = Join-Path $BinDir "llama-server.exe"
if (-not (Test-Path $server)) {
    throw "llama-server.exe not found in $BinDir"
}
if (-not (Test-Path $Model)) {
    throw "Model not found: $Model"
}

$env:Path = "$BinDir;$env:Path"

$argsList = @(
    "-m", $Model,
    "--host", $HostAddress,
    "--port", "$Port",
    "-a", $Alias,
    "-ngl", "$Ngl",
    "-fa", "on",
    "-c", "$Ctx",
    "--jinja",
    "--temp", "1.0",
    "--top-p", "0.95",
    "--top-k", "20"
)

if ($Mmproj -and (Test-Path $Mmproj)) {
    $argsList += @("--mmproj", $Mmproj)
}

switch ($Profile) {
    "baseline" {
        $argsList += @("-np", "$Np")
        Write-Host "Prism profile=baseline (prompt-cache friendly, -np $Np)"
    }
    "kv4" {
        $argsList += @("-np", "$Np", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0")
        Write-Host "Prism profile=kv4 (Q4 KV + baseline serving)"
    }
    "dspark" {
        if (-not $Drafter -or -not (Test-Path $Drafter)) {
            throw "profile=dspark requires -Drafter pointing at a *dspark-dflash*.gguf (Bonsai 2 official drafter may not exist yet)."
        }
        if ($Ctx -lt 16384) { $Ctx = 16384 }
        $argsList = @(
            "-m", $Model,
            "--host", $HostAddress,
            "--port", "$Port",
            "-a", $Alias,
            "-ngl", "$Ngl",
            "-fa", "on",
            "-c", "$Ctx",
            "--jinja",
            "-md", $Drafter,
            "--spec-type", "draft-dspark",
            "--spec-draft-n-max", "4",
            "-ngld", "999",
            "-np", "1"
        )
        if ($Mmproj -and (Test-Path $Mmproj)) {
            $argsList += @("--mmproj", $Mmproj)
        }
        Write-Host "Prism profile=dspark (single slot; cross-request prompt cache disabled)"
        Write-Host "  Drafter: $Drafter"
    }
}

Write-Host "  Model:   $Model"
Write-Host "  Binary:  $server"
Write-Host "  Listen:  http://${HostAddress}:$Port"
Write-Host "  Alias:   $Alias"
Write-Host "  Power:   prism_upstream=`"http://${HostAddress}:$Port`" prism_profile=`"$Profile`""
Write-Host ""

& $server @argsList
exit $LASTEXITCODE
