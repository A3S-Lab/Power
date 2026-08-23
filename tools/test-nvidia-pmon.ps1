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

$helper = Join-Path $PSScriptRoot 'lib/nvidia-pmon.ps1'
. $helper

$baseline = ConvertFrom-NvidiaPmonOutput -Lines @(
    '# gpu pid type sm mem enc dec jpg ofa command',
    '0 2140 C+G 5 0 - - - - dwm.exe',
    '0 7676 C+G - - 1 - - - GameViewerServer.exe'
)
Assert-Equal @($baseline).Count 2 'Baseline process count'
Assert-Equal ([int]$baseline[0].pid) 2140 'Baseline PID'
Assert-Equal ([int]$baseline[0].sm_utilization_percent) 5 `
    'Baseline SM utilization'
Assert-Equal $baseline[1].sm_utilization_percent $null `
    'Unavailable SM utilization'

$observed = ConvertFrom-NvidiaPmonOutput -Lines @(
    '# gpu pid type sm mem enc dec jpg ofa command',
    '0 2140 C+G 4 0 - - - - dwm.exe',
    '0 63924 C+G 74 96 - - - - a3s-power.exe',
    '0 99680 C 20 25 - - - - a3s_use_ocr.exe',
    '0 99680 C 15 25 - - - - a3s_use_ocr.exe',
    '0 120000 C 1 1 - - - - harmless.exe',
    'malformed row'
)
$summary = Get-NvidiaPmonInterferenceSummary `
    -Samples $observed `
    -AllowedProcessIds @(2140, 7676, 63924) `
    -MaximumForeignSmUtilizationPercent 2

Assert-Equal ([int]$summary.parsed_samples) 5 'Parsed sample count'
Assert-Equal ([bool]$summary.interference_detected) $true `
    'Foreign GPU interference result'
Assert-Equal @($summary.foreign_processes).Count 1 `
    'Foreign process count'
Assert-Equal ([int]$summary.foreign_processes[0].pid) 99680 `
    'Foreign process PID'
Assert-Equal ([string]$summary.foreign_processes[0].command) `
    'a3s_use_ocr.exe' 'Foreign process command'
Assert-Equal ([int]$summary.foreign_processes[0].maximum_sm_utilization_percent) 20 `
    'Foreign process peak SM utilization'
Assert-Equal ([int]$summary.foreign_processes[0].samples_over_limit) 2 `
    'Foreign process violating sample count'

$allowed = Get-NvidiaPmonInterferenceSummary `
    -Samples $observed `
    -AllowedProcessIds @(2140, 7676, 63924, 99680) `
    -MaximumForeignSmUtilizationPercent 2
Assert-Equal ([bool]$allowed.interference_detected) $false `
    'Allowed process result'

Write-Output 'NVIDIA pmon interference contract: PASS'
