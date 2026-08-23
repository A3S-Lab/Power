$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert-Equal {
    param(
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [object]$Actual,

        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [object]$Expected,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if ($Actual -cne $Expected) {
        throw "$Label differs: expected '$Expected', got '$Actual'"
    }
}

function Assert-True {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$Condition,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not $Condition) {
        throw $Label
    }
}

$helper = Join-Path $PSScriptRoot 'lib/qwen38-quality-profile.ps1'
. $helper

$pure = Resolve-Qwen38QualityProfile -Profile 'pure-q6'
Assert-Equal $pure.config_relative_path `
    'quality\full-vocabulary-current.acl' 'Pure Q6 configuration'
Assert-Equal ([bool]$pure.requires_tbq4) $false 'Pure Q6 TBQ4 requirement'
Assert-Equal $pure.q6_external_draft_kind $null `
    'Pure Q6 external draft kind'
Assert-Equal @($pure.modes).Count 2 'Pure Q6 mode count'
Assert-Equal (@($pure.modes.label) -join ',') `
    'q6-off,q6-mtp-full-vocab' 'Pure Q6 mode order'
Assert-True `
    (@($pure.modes | Where-Object { $_.model_role -cne 'q6' }).Count -eq 0) `
    'Every pure-Q6 mode must use the Q6 target artifact'
Assert-True `
    (@($pure.modes | Where-Object { $null -ne $_.external_draft_kind }).Count -eq 0) `
    'Pure-Q6 acceptance must not load an external draft artifact'
Assert-Equal $pure.modes[0].spec_mode 'off' 'Pure Q6 control strategy'
Assert-Equal $pure.modes[1].spec_mode 'mtp' 'Pure Q6 optimized strategy'
Assert-Equal $pure.modes[1].fr_vocab_size $null `
    'Pure Q6 balanced MTP vocabulary'
Assert-Equal ($pure.comparisons[0] -join ' -> ') `
    'q6-off -> q6-mtp-full-vocab' 'Pure Q6 paired comparison'

$historical = Resolve-Qwen38QualityProfile -Profile 'full-vocabulary-current'
Assert-Equal ([bool]$historical.requires_tbq4) $true `
    'Historical mixed-artifact requirement'
Assert-True `
    (@($historical.modes | Where-Object { $_.model_role -ceq 'tbq4' }).Count -eq 2) `
    'Historical mixed-artifact profile must remain explicit'

$dspark = Resolve-Qwen38QualityProfile -Profile 'dspark-q4'
Assert-Equal ([bool]$dspark.requires_tbq4) $false `
    'DSpark target artifact requirement'
Assert-Equal $dspark.q6_external_draft_kind 'dspark' `
    'DSpark external draft kind'
Assert-True `
    (@($dspark.modes | Where-Object { $_.model_role -cne 'q6' }).Count -eq 0) `
    'DSpark must retain Q6_K as the target artifact'

$runner = Join-Path $PSScriptRoot 'run-qwen38-quality-matrix.ps1'
$described = @(& $runner -Q6PowerHome 'unused' -DescribeProfile) -join `
    [Environment]::NewLine | ConvertFrom-Json
Assert-Equal (@($described.modes.label) -join ',') `
    'q6-off,q6-mtp-full-vocab' 'Runner default mode order'
Assert-Equal ([bool]$described.requires_tbq4) $false `
    'Runner default TBQ4 requirement'

Write-Output 'Qwen3.8 quality profile contract: PASS'
