[CmdletBinding()]
param(
    [string]$LlamaCppRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$powerRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$patchPath = Join-Path $powerRoot "patches\llama-cpp-rs-dfd12e4-mtp-fr-spec.patch"
$expectedBindingRevision = "dfd12e4d334846367e4284a2a7763fe92c1bf676"
$expectedLlamaRevision = "e79e4bf660e19f2ad851e06c6913f7a8c5852621"

if (-not (Test-Path -LiteralPath $patchPath -PathType Leaf)) {
    throw "Power llama.cpp patch is missing: $patchPath"
}

if (-not $LlamaCppRoot) {
    if (-not $env:USERPROFILE) {
        throw "USERPROFILE is unavailable; pass -LlamaCppRoot explicitly."
    }

    $checkoutRoot = Join-Path $env:USERPROFILE ".cargo\git\checkouts"
    $candidates = @(
        Get-ChildItem -LiteralPath $checkoutRoot -Directory -Filter "llama-cpp-rs-*" -ErrorAction SilentlyContinue |
            ForEach-Object {
                Join-Path $_.FullName "dfd12e4\llama-cpp-sys-2\llama.cpp"
            } |
            Where-Object { Test-Path -LiteralPath $_ -PathType Container }
    )

    if ($candidates.Count -ne 1) {
        throw "Expected one fetched llama-cpp-rs dfd12e4 checkout, found $($candidates.Count); pass -LlamaCppRoot explicitly."
    }
    $LlamaCppRoot = $candidates[0]
}

$LlamaCppRoot = (Resolve-Path -LiteralPath $LlamaCppRoot).Path
$bindingRoot = (Resolve-Path -LiteralPath (Join-Path $LlamaCppRoot "..\..")).Path

$bindingRevision = (& git -C $bindingRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $bindingRevision -ne $expectedBindingRevision) {
    throw "Expected llama-cpp-rs revision $expectedBindingRevision, found '$bindingRevision'."
}

$llamaRevision = (& git -C $LlamaCppRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $llamaRevision -ne $expectedLlamaRevision) {
    throw "Expected llama.cpp revision $expectedLlamaRevision, found '$llamaRevision'."
}

& git -C $LlamaCppRoot apply --check --reverse $patchPath 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Output "Power llama.cpp patches are already applied."
    exit 0
}

& git -C $LlamaCppRoot apply --check $patchPath
if ($LASTEXITCODE -ne 0) {
    throw "Power llama.cpp patch does not apply cleanly to $LlamaCppRoot."
}

& git -C $LlamaCppRoot apply $patchPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to apply the Power llama.cpp patch."
}

Write-Output "Applied Power llama.cpp patches to $LlamaCppRoot."
