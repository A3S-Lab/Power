# Start a Prism llama-server with a Power-aligned acceleration profile.
#
# Profiles (see docs/prism-acceleration-plan.md):
#   baseline — FA + offload; prompt-cache friendly; multi-slot OK
#   dspark   — requires -Drafter (*dspark-dflash*); single slot; no cross-request cache
#              (Bonsai-2 has no official DSpark pin yet — use mtp for Bonsai-2 speculation)
#   mtp      — in-file / grafted MTP head; --spec-type draft-mtp; single slot
#   kv4      — baseline + Q4 KV cache
#
#   dflash   — community DFlash2; requires patched BinDir (Prism+DFlash2 SM89 build)
#              + -Drafter Bonsai-2-27B-DFlash2-Q8_0.gguf; stock Prism fails tensor load
#
# Example:
#   .\tools\start-prism-upstream.ps1 -Profile baseline -Alias bonsai2-ptq1 `
#     -Model "D:\Bonsai-demo\models\bonsai2-gguf\27B\Ternary-Bonsai-2-27B-PTQ1_0.gguf" `
#     -BinDir "D:\Bonsai-demo\bin\cuda" -Port 8080
#
#   .\tools\start-prism-upstream.ps1 -Profile mtp -Alias bonsai2-ptq1-mtp `
#     -Model "...\Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf" `
#     -BinDir "D:\Bonsai-demo\bin\cuda" -Port 8080 -DraftNMax 1
#
#   .\tools\start-prism-upstream.ps1 -Profile dflash -Alias bonsai2-pq2-dflash `
#     -Model "...\Ternary-Bonsai-2-27B-PQ2_0.gguf" `
#     -Drafter "...\Bonsai-2-27B-DFlash2-Q8_0.gguf" `
#     -BinDir "...\llama\build-sm89\bin" -Port 8080 -DraftNMax 3 -Reasoning off

[CmdletBinding()]
param(
    [ValidateSet("baseline", "dspark", "mtp", "kv4", "dflash")]
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
    [int]$Np = 4,
    [int]$DraftNMax = 0,
    # Decode-focused: disable thinking so Power e2e tok/s tracks llama-bench tg.
    [ValidateSet("on", "off", "auto")]
    [string]$Reasoning = "auto",
    [int]$ReasoningBudget = -1
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
    "--top-k", "20",
    "--reasoning", $Reasoning
)
if ($ReasoningBudget -ge 0) {
    $argsList += @("--reasoning-budget", "$ReasoningBudget")
}

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
            throw "profile=dspark requires -Drafter pointing at a *dspark-dflash*.gguf (Bonsai 2 official drafter may not exist yet; use -Profile mtp)."
        }
        if ($Ctx -lt 16384) { $Ctx = 16384 }
        $nMax = if ($DraftNMax -gt 0) { $DraftNMax } else { 4 }
        $argsList = @(
            "-m", $Model,
            "--host", $HostAddress,
            "--port", "$Port",
            "-a", $Alias,
            "-ngl", "$Ngl",
            "-fa", "on",
            "-c", "$Ctx",
            "--jinja",
            "--reasoning", $Reasoning,
            "-md", $Drafter,
            "--spec-type", "draft-dspark",
            "--spec-draft-n-max", "$nMax",
            "-ngld", "999",
            "-np", "1"
        )
        if ($ReasoningBudget -ge 0) { $argsList += @("--reasoning-budget", "$ReasoningBudget") }
        if ($Mmproj -and (Test-Path $Mmproj)) {
            $argsList += @("--mmproj", $Mmproj)
        }
        Write-Host "Prism profile=dspark (single slot; cross-request prompt cache disabled)"
        Write-Host "  Drafter: $Drafter"
    }
    "mtp" {
        if ($Ctx -lt 16384) { $Ctx = 16384 }
        # sudoingx fat PTQ1 MTP graft: n-max 1 is the safe default on stock Prism;
        # ProCreations PQ2 MTP often uses 2 with a patched embedding runtime.
        $nMax = if ($DraftNMax -gt 0) { $DraftNMax } else { 1 }
        $argsList = @(
            "-m", $Model,
            "--host", $HostAddress,
            "--port", "$Port",
            "-a", $Alias,
            "-ngl", "$Ngl",
            "-fa", "on",
            "-c", "$Ctx",
            "--jinja",
            "--reasoning", $Reasoning,
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "$nMax",
            "-np", "1"
        )
        if ($ReasoningBudget -ge 0) { $argsList += @("--reasoning-budget", "$ReasoningBudget") }
        if ($Mmproj -and (Test-Path $Mmproj)) {
            $argsList += @("--mmproj", $Mmproj)
        }
        Write-Host "Prism profile=mtp (in-file MTP; single slot; draft-n-max=$nMax)"
    }
    "dflash" {
        if (-not $Drafter -or -not (Test-Path $Drafter)) {
            throw "profile=dflash requires -Drafter pointing at a Bonsai-2-matched DFlash2 GGUF."
        }
        if ($Ctx -lt 16384) { $Ctx = 16384 }
        $nMax = if ($DraftNMax -gt 0) { $DraftNMax } else { 8 }
        $argsList = @(
            "-m", $Model,
            "--host", $HostAddress,
            "--port", "$Port",
            "-a", $Alias,
            "-ngl", "$Ngl",
            "-fa", "on",
            "-c", "$Ctx",
            "--jinja",
            "--reasoning", $Reasoning,
            "-md", $Drafter,
            "--spec-type", "draft-dflash",
            "--spec-draft-n-max", "$nMax",
            "-ngld", "999",
            "-np", "1"
        )
        if ($ReasoningBudget -ge 0) { $argsList += @("--reasoning-budget", "$ReasoningBudget") }
        if ($Mmproj -and (Test-Path $Mmproj)) {
            $argsList += @("--mmproj", $Mmproj)
        }
        Write-Host "Prism profile=dflash (community DFlash2; single slot; draft-n-max=$nMax)"
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
