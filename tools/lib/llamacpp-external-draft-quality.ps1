function Invoke-QualityPython([string[]]$Arguments) {
    & $PythonLauncher "-$PythonVersion" @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python quality evaluator exited with code $LASTEXITCODE"
    }
}

function Get-QualityReportMetadata([string]$Path) {
    $json = @(
        & $PythonLauncher "-$PythonVersion" $evaluator `
            'inspect-report' '--report' $Path
    ) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0) {
        throw "Python report inspection exited with code $LASTEXITCODE"
    }
    return $json | ConvertFrom-Json
}

function Write-EnvironmentReport {
    $json = $script:environment | ConvertTo-Json -Depth 12
    [System.IO.File]::WriteAllText(
        $environmentPath,
        $json + [Environment]::NewLine,
        $utf8NoBom
    )
}

function Invoke-QualityMode(
    [hashtable]$Mode,
    [int]$Repetition,
    [int]$OrderIndex
) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port is already in use"
    }

    $stem = 'r{0:d2}-o{1}-{2}' -f $Repetition, $OrderIndex, $Mode.label
    $stdoutPath = Join-Path $outputPath "$stem.stdout.log"
    $stderrPath = Join-Path $outputPath "$stem.stderr.log"
    $reportPath = Join-Path $outputPath "$stem.json"
    $process = $null
    $client = $null
    $admission = Wait-NvidiaGpuIdleAdmission
    $script:environment.gpu_admissions += [ordered]@{
        run = $stem
        admission = $admission
    }
    Write-EnvironmentReport

    try {
        $arguments = New-ServerArguments ([bool]$Mode.use_draft)
        $argumentLine = Join-WindowsCommandLineArguments $arguments
        $process = Start-Process `
            -FilePath $serverPath `
            -ArgumentList $argumentLine `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -WindowStyle Hidden `
            -PassThru
        $process.PriorityClass = $ProcessPriority
        $effectivePriority = [string]$process.PriorityClass
        if ($ProcessorAffinityMask -gt 0) {
            $process.ProcessorAffinity = [IntPtr]::new([int64]$ProcessorAffinityMask)
            $effectiveAffinityValue = [uint64]$process.ProcessorAffinity.ToInt64()
            if ($effectiveAffinityValue -ne $ProcessorAffinityMask) {
                throw ('Requested processor affinity 0x{0:x} became 0x{1:x}' -f `
                    $ProcessorAffinityMask, $effectiveAffinityValue)
            }
        } else {
            $effectiveAffinityValue = [uint64]$process.ProcessorAffinity.ToInt64()
        }

        $client = [System.Net.Http.HttpClient]::new()
        $client.Timeout = [TimeSpan]::FromSeconds(300)
        $peakUsedMiB = Get-GpuUsedMiB
        Wait-ServerReady $process $client ([ref]$peakUsedMiB) 300
        $script:environment.processes += [ordered]@{
            run = $stem
            requested_priority = $ProcessPriority
            effective_priority = $effectivePriority
            requested_affinity = if ($ProcessorAffinityMask -gt 0) {
                '0x{0:x}' -f $ProcessorAffinityMask
            } else {
                $null
            }
            effective_affinity = '0x{0:x}' -f $effectiveAffinityValue
            loaded_gpu_memory_used_mib = Get-GpuUsedMiB
            startup_peak_gpu_memory_used_mib = $peakUsedMiB
        }
        Write-EnvironmentReport

        $pythonArguments = @(
            $evaluator,
            'run',
            '--url', "http://127.0.0.1:$Port",
            '--model', $Model,
            '--mode-label', $Mode.label,
            '--repetition', [string]$Repetition,
            '--order-index', [string]$OrderIndex,
            '--model-sha256', $actualTargetHash,
            '--server-sha256', $serverHash,
            '--power-commit', $powerCommit,
            '--tasks', $tasksPath,
            '--manifest', $manifestPath,
            '--output', $reportPath,
            '--server-log', $stderrPath,
            '--warmup-requests', '1',
            '--num-ctx', [string]$ContextSize,
            '--num-batch', [string]$BatchSize,
            '--seed', '42',
            '--timeout-seconds', '900'
        )
        if ($selectionPath) {
            $pythonArguments += @('--task-selection', $selectionPath)
        }
        if ($Mode.use_draft) {
            $pythonArguments += @('--speculative-strategy', $DraftMode)
        }
        if ($MaxTokensCap -gt 0) {
            $pythonArguments += @('--max-tokens-cap', [string]$MaxTokensCap)
        }
        if ($IncludeContent) {
            $pythonArguments += '--include-content'
        }
        Invoke-QualityPython $pythonArguments

        $metadata = Get-QualityReportMetadata $reportPath
        if ([int]$metadata.result_count -ne $expectedTaskCount -or
            [int]$metadata.completed -ne $expectedTaskCount -or
            [int]$metadata.errors -ne 0) {
            throw "Mode $($Mode.label) did not complete $expectedTaskCount error-free tasks"
        }
        if ($Mode.use_draft -and
            [string]$metadata.speculative_strategy -ne $DraftMode) {
            throw "Mode $($Mode.label) did not report the expected $DraftMode runtime"
        }
        if (-not $Mode.use_draft -and [bool]$metadata.has_speculative_runtime) {
            throw "Mode $($Mode.label) unexpectedly reported a speculative runtime"
        }
    } finally {
        if ($null -ne $client) {
            $client.Dispose()
        }
        Stop-Server $process
    }
}
