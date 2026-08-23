$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$qualityFiles = @(
    (Join-Path $PSScriptRoot 'run-llamacpp-external-draft-quality.ps1'),
    (Join-Path $PSScriptRoot 'lib/llamacpp-external-draft-quality.ps1')
)
$parseErrors = $null
foreach ($qualityFile in $qualityFiles) {
    [System.Management.Automation.Language.Parser]::ParseFile(
        $qualityFile,
        [ref]$null,
        [ref]$parseErrors
    ) | Out-Null
    if ($parseErrors.Count -gt 0) {
        throw "External-draft quality tooling has PowerShell parse errors: $parseErrors"
    }
}

$MaxTokens = 1
$ContextSize = 64
$BatchSize = 8
$DraftMax = 1
. (Join-Path $PSScriptRoot 'lib/llamacpp-external-draft-benchmark.ps1')

function Assert-Equal([string]$Expected, [string]$Actual, [string]$Label) {
    if ($Expected -ne $Actual) {
        throw "$Label differs: expected '$Expected', got '$Actual'"
    }
}

Assert-Equal 'dflash' (Get-ExternalDraftServerMode 'dflash') 'DFlash backend mode'
Assert-Equal 'dflash' (Get-ExternalDraftServerMode 'dflash2') 'DFlash2 backend mode'
Assert-Equal 'dspark' (Get-ExternalDraftServerMode 'dspark') 'DSpark backend mode'

$invalidFailed = $false
try {
    $null = Get-ExternalDraftServerMode 'unknown'
} catch {
    $invalidFailed = $true
}
if (-not $invalidFailed) {
    throw 'Unknown external-draft mode did not fail closed'
}

$skipped = New-SkippedModeResult 'draft-dflash2' 'baseline failed'
Assert-Equal 'skipped' $skipped.status 'Skipped result status'
Assert-Equal 'baseline failed' $skipped.error 'Skipped result reason'
if (@($skipped.samples).Count -ne 0) {
    throw 'Skipped result unexpectedly contains benchmark samples'
}

Write-Output 'llama.cpp external-draft benchmark contract: PASS'
