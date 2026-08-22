[CmdletBinding()]
param(
    [string]$LlamaCppRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$powerRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$llamaPatchPath = Join-Path $powerRoot "patches\llama-cpp-rs-dfd12e4-mtp-fr-spec.patch"
$bindingPatchPath = Join-Path $powerRoot "patches\llama-cpp-rs-dfd12e4-mtp-dynamic-k.patch"
$externalDraftBindingPatchPath = Join-Path $powerRoot "patches\llama-cpp-rs-dfd12e4-external-draft.patch"
$cudaPriorityPatchPath = Join-Path $powerRoot "patches\llama-cpp-rs-dfd12e4-cuda-high-priority.patch"
$expectedBindingRevision = "dfd12e4d334846367e4284a2a7763fe92c1bf676"
$expectedLlamaRevision = "e79e4bf660e19f2ad851e06c6913f7a8c5852621"

foreach ($patchPath in @(
    $llamaPatchPath,
    $bindingPatchPath,
    $externalDraftBindingPatchPath,
    $cudaPriorityPatchPath
)) {
    if (-not (Test-Path -LiteralPath $patchPath -PathType Leaf)) {
        throw "Power llama.cpp patch is missing: $patchPath"
    }
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

function Apply-ReviewedPatch {
    param(
        [string]$Root,
        [string]$Patch,
        [string]$Label
    )

    & git -C $Root apply --check --reverse $Patch 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Output "Power $Label patch is already applied."
        return
    }

    & git -C $Root apply --check $Patch
    if ($LASTEXITCODE -ne 0) {
        throw "Power $Label patch does not apply cleanly to $Root."
    }

    & git -C $Root apply $Patch
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to apply the Power $Label patch."
    }

    Write-Output "Applied Power $Label patch to $Root."
}

Apply-ReviewedPatch -Root $bindingRoot -Patch $bindingPatchPath -Label 'binding'
Apply-ReviewedPatch -Root $bindingRoot -Patch $externalDraftBindingPatchPath -Label 'external-draft binding'
Apply-ReviewedPatch -Root $LlamaCppRoot -Patch $llamaPatchPath -Label 'llama.cpp'
Apply-ReviewedPatch -Root $LlamaCppRoot -Patch $cudaPriorityPatchPath -Label 'CUDA high-priority stream'
