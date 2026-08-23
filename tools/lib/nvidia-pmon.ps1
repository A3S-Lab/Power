function ConvertFrom-NvidiaPmonOutput {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$Lines
    )

    $rows = [System.Collections.Generic.List[object]]::new()
    foreach ($line in $Lines) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or
            $trimmed.StartsWith('#')) {
            continue
        }

        $fields = @($trimmed -split '\s+', 10)
        if ($fields.Count -lt 10) {
            continue
        }
        $gpuIndex = 0
        $processId = 0
        if (-not [int]::TryParse($fields[0], [ref]$gpuIndex) -or
            -not [int]::TryParse($fields[1], [ref]$processId)) {
            continue
        }

        $smUtilization = $null
        if ($fields[3] -ne '-') {
            $parsedUtilization = 0
            if (-not [int]::TryParse(
                    $fields[3],
                    [ref]$parsedUtilization
                )) {
                continue
            }
            $smUtilization = $parsedUtilization
        }

        $rows.Add([pscustomobject][ordered]@{
            gpu_index = $gpuIndex
            pid = $processId
            process_type = $fields[2]
            sm_utilization_percent = $smUtilization
            command = $fields[9]
        })
    }
    return $rows.ToArray()
}

function Get-NvidiaPmonInterferenceSummary {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [object[]]$Samples,

        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [int[]]$AllowedProcessIds,

        [Parameter(Mandatory = $true)]
        [ValidateRange(0, 100)]
        [int]$MaximumForeignSmUtilizationPercent
    )

    $allowed = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($processId in $AllowedProcessIds) {
        $null = $allowed.Add($processId)
    }

    $violations = @(
        $Samples | Where-Object {
            $null -ne $_.sm_utilization_percent -and
            -not $allowed.Contains([int]$_.pid) -and
            [int]$_.sm_utilization_percent -gt
                $MaximumForeignSmUtilizationPercent
        }
    )
    $foreignProcesses = @(
        $violations |
            Group-Object pid |
            Sort-Object { [int]$_.Name } |
            ForEach-Object {
                $group = @($_.Group)
                [pscustomobject][ordered]@{
                    pid = [int]$_.Name
                    command = [string]$group[0].command
                    maximum_sm_utilization_percent = [int](
                        $group.sm_utilization_percent |
                            Measure-Object -Maximum
                    ).Maximum
                    samples_over_limit = $group.Count
                }
            }
    )

    [pscustomobject][ordered]@{
        parsed_samples = $Samples.Count
        maximum_foreign_sm_utilization_percent =
            $MaximumForeignSmUtilizationPercent
        interference_detected = $foreignProcesses.Count -gt 0
        foreign_processes = $foreignProcesses
    }
}

function Get-NvidiaPmonSnapshot {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateScript({ $_ -ge 0 })]
        [int]$GpuIndex
    )

    $lines = @(& nvidia-smi.exe pmon `
        "--id=$GpuIndex" '-s' 'u' '-c' '1')
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to capture NVIDIA GPU $GpuIndex process baseline"
    }
    return @(ConvertFrom-NvidiaPmonOutput -Lines $lines)
}

function Start-NvidiaPmonMonitor {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateScript({ $_ -ge 0 })]
        [int]$GpuIndex,

        [Parameter(Mandatory = $true)]
        [string]$OutputPath,

        [Parameter(Mandatory = $true)]
        [string]$ErrorPath
    )

    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        throw 'nvidia-smi.exe is required for continuous GPU exclusivity'
    }
    $process = Start-Process -FilePath $nvidiaSmi.Source `
        -ArgumentList @('pmon', "--id=$GpuIndex", '-s', 'u', '-d', '1') `
        -RedirectStandardOutput $OutputPath `
        -RedirectStandardError $ErrorPath `
        -WindowStyle Hidden `
        -PassThru
    Start-Sleep -Milliseconds 250
    if ($process.HasExited) {
        $message = Get-Content -LiteralPath $ErrorPath -Raw `
            -ErrorAction SilentlyContinue
        throw "NVIDIA pmon exited before quality evaluation: $message"
    }
    return $process
}

function Complete-NvidiaPmonMonitor {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,

        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [object[]]$Baseline,

        [Parameter(Mandatory = $true)]
        [int]$ServerProcessId,

        [Parameter(Mandatory = $true)]
        [string]$OutputPath,

        [Parameter(Mandatory = $true)]
        [ValidateRange(0, 100)]
        [int]$MaximumForeignSmUtilizationPercent
    )

    $exitedUnexpectedly = $Process.HasExited
    if (-not $Process.HasExited) {
        $Process.Kill()
    }
    $Process.WaitForExit()

    $failure = $null
    $summary = $null
    if ($exitedUnexpectedly) {
        $failure = 'NVIDIA pmon exited before the mode completed'
    } elseif (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
        $failure = 'NVIDIA pmon did not create its evidence log'
    } else {
        $lines = @(Get-Content -LiteralPath $OutputPath)
        $samples = @(ConvertFrom-NvidiaPmonOutput -Lines $lines)
        if ($samples.Count -eq 0) {
            $failure = 'NVIDIA pmon recorded no process samples'
        } else {
            $allowedProcessIds = @(
                @($Baseline.pid) + @($ServerProcessId) |
                    Sort-Object -Unique
            )
            $summary = Get-NvidiaPmonInterferenceSummary `
                -Samples $samples `
                -AllowedProcessIds $allowedProcessIds `
                -MaximumForeignSmUtilizationPercent `
                    $MaximumForeignSmUtilizationPercent
        }
    }

    [pscustomobject][ordered]@{
        summary = $summary
        failure = $failure
    }
}
