[CmdletBinding()]
param(
    [string]$LlamaCppRoot,
    [switch]$SkipCargoClean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$powerRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$llamaPatchPath = Join-Path $powerRoot "patches\llama-cpp-rs-dfd12e4-mtp-fr-spec.patch"
$bindingPatchPath = Join-Path $powerRoot "patches\llama-cpp-rs-dfd12e4-mtp-dynamic-k.patch"
$externalDraftBindingPatchPath = Join-Path $powerRoot "patches\llama-cpp-rs-dfd12e4-external-draft.patch"
$dflash2BindingPatchPath = Join-Path $powerRoot "patches\llama-cpp-rs-dfd12e4-dflash2-binding.patch"
$dflash2PatchPath = Join-Path $powerRoot "patches\llama-cpp-rs-dfd12e4-dflash2.patch"
$cudaPriorityPatchPath = Join-Path $powerRoot "patches\llama-cpp-rs-dfd12e4-cuda-high-priority.patch"
$expectedBindingRevision = "dfd12e4d334846367e4284a2a7763fe92c1bf676"
$expectedLlamaRevision = "e79e4bf660e19f2ad851e06c6913f7a8c5852621"
$script:appliedAnyPatch = $false

foreach ($patchPath in @(
    $llamaPatchPath,
    $bindingPatchPath,
    $externalDraftBindingPatchPath,
    $dflash2BindingPatchPath,
    $dflash2PatchPath,
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
        [string]$Label,
        [string]$SentinelPath,
        [string]$SentinelText
    )

    if ($SentinelPath -and $SentinelText) {
        $sentinelFile = Join-Path $Root $SentinelPath
        if ((Test-Path -LiteralPath $sentinelFile -PathType Leaf) -and
            [System.IO.File]::ReadAllText($sentinelFile).Contains($SentinelText)) {
            Write-Output "Power $Label patch is already applied."
            return
        }
    }

    # A failed reverse-check is the normal signal that a patch still needs to
    # be applied. Capture native exit codes explicitly so ErrorActionPreference
    # does not turn that probe into a terminating PowerShell error.
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & git -C $Root apply --check --reverse $Patch 2>$null
        $reverseCheckExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedErrorActionPreference
    }
    if ($reverseCheckExitCode -eq 0) {
        Write-Output "Power $Label patch is already applied."
        return
    }

    $ErrorActionPreference = 'Continue'
    try {
        & git -C $Root apply --check $Patch
        $forwardCheckExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedErrorActionPreference
    }
    if ($forwardCheckExitCode -ne 0) {
        throw "Power $Label patch does not apply cleanly to $Root."
    }

    & git -C $Root apply $Patch
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to apply the Power $Label patch."
    }

    $script:appliedAnyPatch = $true
    Write-Output "Applied Power $Label patch to $Root."
}

Apply-ReviewedPatch -Root $bindingRoot -Patch $bindingPatchPath -Label 'binding' `
    -SentinelPath 'llama-cpp-sys-2\wrapper_common.h' `
    -SentinelText 'llama_rs_mtp_speculative_last_recurrent_steps'
Apply-ReviewedPatch -Root $bindingRoot -Patch $externalDraftBindingPatchPath -Label 'external-draft binding' `
    -SentinelPath 'llama-cpp-sys-2\wrapper_common.h' `
    -SentinelText 'llama_rs_external_draft_speculative_init'
Apply-ReviewedPatch -Root $bindingRoot -Patch $dflash2BindingPatchPath -Label 'DFlash2 binding' `
    -SentinelPath 'llama-cpp-2\src\speculative.rs' `
    -SentinelText 'Dflash2 = 3,'
Apply-ReviewedPatch -Root $LlamaCppRoot -Patch $dflash2PatchPath -Label 'DFlash2 runtime' `
    -SentinelPath 'common\speculative.cpp' `
    -SentinelText 'DFlash2 selector produced no lattice'
Apply-ReviewedPatch -Root $LlamaCppRoot -Patch $llamaPatchPath -Label 'llama.cpp' `
    -SentinelPath 'src\llama-ext.h' `
    -SentinelText 'llama_mtp_recurrent_begin'
Apply-ReviewedPatch -Root $LlamaCppRoot -Patch $cudaPriorityPatchPath -Label 'CUDA high-priority stream' `
    -SentinelPath 'ggml\src\ggml-cuda\common.cuh' `
    -SentinelText 'GGML_CUDA_HIGH_PRIORITY'

if ($script:appliedAnyPatch -and -not $SkipCargoClean) {
    Write-Output "Invalidating cached llama-cpp build artifacts after patching."
    & cargo clean --manifest-path (Join-Path $powerRoot 'Cargo.toml') `
        -p llama-cpp-2 -p llama-cpp-sys-2
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to invalidate cached llama-cpp build artifacts."
    }
} elseif ($script:appliedAnyPatch) {
    Write-Warning "Patched sources may be hidden by stale Cargo artifacts; clean llama-cpp-2 and llama-cpp-sys-2 before building."
}
