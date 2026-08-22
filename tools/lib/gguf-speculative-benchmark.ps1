function ConvertTo-WindowsCommandLineArgument([string]$Argument) {
    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    $builder = [System.Text.StringBuilder]::new()
    $null = $builder.Append('"')
    $backslashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }
        if ($character -eq '"') {
            $null = $builder.Append((('\' * (($backslashes * 2) + 1)) -join ''))
            $null = $builder.Append('"')
        } else {
            if ($backslashes -gt 0) {
                $null = $builder.Append((('\' * $backslashes) -join ''))
            }
            $null = $builder.Append($character)
        }
        $backslashes = 0
    }
    if ($backslashes -gt 0) {
        $null = $builder.Append((('\' * ($backslashes * 2)) -join ''))
    }
    $null = $builder.Append('"')
    return $builder.ToString()
}

function Join-WindowsCommandLineArguments([string[]]$Arguments) {
    return (@(
        $Arguments | ForEach-Object { ConvertTo-WindowsCommandLineArgument $_ }
    ) -join ' ')
}

function Get-NvidiaRuntimeSamples([int[]]$GpuIndices) {
    $observedAt = [DateTimeOffset]::UtcNow.ToString('o')
    $samples = @()
    foreach ($gpuIndex in $GpuIndices) {
        $lines = @(& nvidia-smi.exe `
            "--id=$gpuIndex" `
            '--query-gpu=index,utilization.gpu,memory.used,power.draw' `
            '--format=csv,noheader,nounits')
        if ($LASTEXITCODE -ne 0 -or $lines.Count -ne 1) {
            throw "Failed to capture runtime telemetry for NVIDIA GPU $gpuIndex"
        }
        $fields = @($lines[0].Split(',') | ForEach-Object { $_.Trim() })
        if ($fields.Count -ne 4) {
            throw "NVIDIA GPU $gpuIndex returned malformed runtime telemetry"
        }

        $reportedIndex = 0
        $utilization = 0
        $memoryUsed = 0
        if (-not [int]::TryParse($fields[0], [ref]$reportedIndex) -or
            -not [int]::TryParse($fields[1], [ref]$utilization) -or
            -not [int]::TryParse($fields[2], [ref]$memoryUsed)) {
            throw "NVIDIA GPU $gpuIndex returned non-numeric runtime telemetry"
        }
        $powerDraw = 0.0
        $hasPowerDraw = [double]::TryParse(
            $fields[3],
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$powerDraw
        )
        $samples += [pscustomobject][ordered]@{
            observed_at = $observedAt
            gpu_index = $reportedIndex
            utilization_percent = $utilization
            memory_used_mib = $memoryUsed
            power_draw_watts = if ($hasPowerDraw) { $powerDraw } else { $null }
        }
    }
    return $samples
}

function Invoke-MonitoredNativeProcess(
    [string]$FilePath,
    [string[]]$Arguments,
    [string]$StandardOutputPath,
    [string]$StandardErrorPath,
    [int[]]$GpuIndices,
    [int]$SampleIntervalMilliseconds
) {
    $argumentLine = Join-WindowsCommandLineArguments $Arguments
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = $argumentLine
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $samples = @()
    $stdoutTask = $null
    $stderrTask = $null
    $started = $false
    try {
        if (-not $process.Start()) {
            throw "Failed to start native process: $FilePath"
        }
        $started = $true
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()

        if ($GpuIndices.Count -eq 0) {
            $process.WaitForExit()
        } else {
            while (-not $process.HasExited) {
                $samples += @(Get-NvidiaRuntimeSamples $GpuIndices)
                if (-not $process.HasExited) {
                    Start-Sleep -Milliseconds $SampleIntervalMilliseconds
                }
            }
            $process.WaitForExit()
            $samples += @(Get-NvidiaRuntimeSamples $GpuIndices)
        }
    } catch {
        if ($started -and -not $process.HasExited) {
            $process.Kill()
            $process.WaitForExit()
        }
        $process.Dispose()
        throw
    }

    $exitCode = [int]$process.ExitCode
    $stdoutText = $stdoutTask.GetAwaiter().GetResult()
    $stderrText = $stderrTask.GetAwaiter().GetResult()
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($StandardOutputPath, $stdoutText, $utf8NoBom)
    [System.IO.File]::WriteAllText($StandardErrorPath, $stderrText, $utf8NoBom)
    $process.Dispose()

    $perGpu = @()
    foreach ($gpuIndex in $GpuIndices) {
        $gpuSamples = @($samples | Where-Object { $_.gpu_index -eq $gpuIndex })
        $perGpu += [pscustomobject][ordered]@{
            gpu_index = $gpuIndex
            sample_count = $gpuSamples.Count
            peak_memory_used_mib = if ($gpuSamples.Count -gt 0) {
                [int]($gpuSamples | Measure-Object -Property memory_used_mib -Maximum).Maximum
            } else {
                $null
            }
            peak_utilization_percent = if ($gpuSamples.Count -gt 0) {
                [int]($gpuSamples | Measure-Object -Property utilization_percent -Maximum).Maximum
            } else {
                $null
            }
            peak_power_draw_watts = if (@($gpuSamples | Where-Object { $null -ne $_.power_draw_watts }).Count -gt 0) {
                [double]($gpuSamples | Where-Object { $null -ne $_.power_draw_watts } |
                    Measure-Object -Property power_draw_watts -Maximum).Maximum
            } else {
                $null
            }
        }
    }

    return [pscustomobject][ordered]@{
        exit_code = $exitCode
        stdout = $stdoutText
        stderr = $stderrText
        telemetry = [ordered]@{
            sample_interval_milliseconds = $SampleIntervalMilliseconds
            samples = $samples
            per_gpu = $perGpu
        }
    }
}
