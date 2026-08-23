function Get-ExternalDraftServerMode([string]$Mode) {
    switch ($Mode) {
        'dflash' { return 'dflash' }
        'dflash2' { return 'dflash' }
        'dspark' { return 'dspark' }
        default { throw "Unsupported external-draft benchmark mode: $Mode" }
    }
}

function Assert-Leaf([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label does not exist: $Path"
    }
}

function Get-NormalizedSha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-LlamaRuntimeFileIdentity([string]$BinDirectory) {
    return @(
        Get-ChildItem -LiteralPath $BinDirectory -File |
            Where-Object {
                $_.Name -eq 'llama-server.exe' -or $_.Extension -eq '.dll'
            } |
            Sort-Object Name |
            ForEach-Object {
                [ordered]@{
                    file = $_.Name
                    size = [int64]$_.Length
                    sha256 = Get-NormalizedSha256 $_.FullName
                }
            }
    )
}

function Wait-NvidiaGpuIdleAdmission {
    $startedAt = [DateTimeOffset]::UtcNow
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $accepted = [System.Collections.Generic.List[object]]::new()
    $observedSamples = 0
    $lastSample = $null

    while ($stopwatch.Elapsed.TotalSeconds -lt $IdleGpuWaitSeconds) {
        $lines = @(& nvidia-smi.exe `
            "--id=$NvidiaGpuIndex" `
            '--query-gpu=index,utilization.gpu,memory.free' `
            '--format=csv,noheader,nounits')
        if ($LASTEXITCODE -ne 0 -or $lines.Count -ne 1) {
            throw "Failed to sample NVIDIA GPU $NvidiaGpuIndex for idle admission"
        }
        $fields = @($lines[0].Split(',') | ForEach-Object { $_.Trim() })
        if ($fields.Count -ne 3) {
            throw "NVIDIA GPU $NvidiaGpuIndex returned malformed idle telemetry"
        }

        $reportedIndex = 0
        $utilization = 0
        $memoryFree = 0
        if (-not [int]::TryParse($fields[0], [ref]$reportedIndex) -or
            -not [int]::TryParse($fields[1], [ref]$utilization) -or
            -not [int]::TryParse($fields[2], [ref]$memoryFree)) {
            throw "NVIDIA GPU $NvidiaGpuIndex returned non-numeric idle telemetry"
        }
        if ($reportedIndex -ne $NvidiaGpuIndex) {
            throw "NVIDIA GPU index mismatch: requested $NvidiaGpuIndex, got $reportedIndex"
        }

        $observedSamples++
        $lastSample = [pscustomobject][ordered]@{
            observed_at = [DateTimeOffset]::UtcNow.ToString('o')
            gpu_index = $reportedIndex
            utilization_percent = $utilization
            memory_free_mib = $memoryFree
        }
        if ($utilization -le $MaximumIdleGpuUtilizationPercent -and
            $memoryFree -ge $MinimumIdleGpuMemoryFreeMiB) {
            $accepted.Add($lastSample)
        } else {
            $accepted.Clear()
        }
        if ($accepted.Count -eq $IdleGpuSampleCount) {
            $stopwatch.Stop()
            return [ordered]@{
                started_at = $startedAt.ToString('o')
                completed_at = [DateTimeOffset]::UtcNow.ToString('o')
                elapsed_milliseconds = [int64]$stopwatch.ElapsedMilliseconds
                observed_sample_count = $observedSamples
                accepted_samples = $accepted.ToArray()
            }
        }
        Start-Sleep -Milliseconds $IdleGpuSampleIntervalMilliseconds
    }

    $last = if ($lastSample) {
        "$($lastSample.utilization_percent)% utilization, $($lastSample.memory_free_mib) MiB free"
    } else {
        'no valid sample'
    }
    throw "NVIDIA GPU $NvidiaGpuIndex did not satisfy the idle gate within $IdleGpuWaitSeconds seconds; last sample: $last"
}

function Get-TextSha256([string]$Value) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
        return ([System.BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    } finally {
        $algorithm.Dispose()
    }
}

function Get-Median([double[]]$Values) {
    if ($Values.Count -eq 0) {
        return $null
    }
    $ordered = @($Values | Sort-Object)
    $middle = [math]::Floor($ordered.Count / 2)
    if (($ordered.Count % 2) -eq 1) {
        return [double]$ordered[$middle]
    }
    return ([double]$ordered[$middle - 1] + [double]$ordered[$middle]) / 2.0
}

function Get-GpuUsedMiB {
    $value = @(& nvidia-smi.exe `
        "--id=$NvidiaGpuIndex" `
        '--query-gpu=memory.used' `
        '--format=csv,noheader,nounits')
    if ($LASTEXITCODE -ne 0 -or $value.Count -ne 1) {
        throw "Failed to sample NVIDIA GPU $NvidiaGpuIndex memory"
    }
    return [int]$value[0].Trim()
}

function Update-PeakGpuMemory([ref]$PeakUsedMiB) {
    $used = Get-GpuUsedMiB
    if ($used -gt $PeakUsedMiB.Value) {
        $PeakUsedMiB.Value = $used
    }
}

function Get-MetricValue([string]$Metrics, [string]$Name) {
    $pattern = '(?m)^' + [regex]::Escape($Name) + '\s+(?<value>[-+0-9.eE]+)\s*$'
    $match = [regex]::Match($Metrics, $pattern)
    if (-not $match.Success) {
        return 0.0
    }
    return [double]::Parse(
        $match.Groups['value'].Value,
        [System.Globalization.CultureInfo]::InvariantCulture
    )
}

function Invoke-HttpJson(
    [System.Net.Http.HttpClient]$Client,
    [string]$Uri,
    [hashtable]$Body,
    [ref]$PeakUsedMiB
) {
    $json = $Body | ConvertTo-Json -Depth 8 -Compress
    $content = [System.Net.Http.StringContent]::new(
        $json,
        [System.Text.Encoding]::UTF8,
        'application/json'
    )
    try {
        $task = $Client.PostAsync($Uri, $content)
        while (-not $task.IsCompleted) {
            Update-PeakGpuMemory $PeakUsedMiB
            Start-Sleep -Milliseconds $GpuSampleIntervalMilliseconds
        }
        $response = $task.GetAwaiter().GetResult()
        try {
            $rawTask = $response.Content.ReadAsStringAsync()
            $raw = $rawTask.GetAwaiter().GetResult()
            if (-not $response.IsSuccessStatusCode) {
                throw "HTTP $([int]$response.StatusCode): $raw"
            }
            return $raw | ConvertFrom-Json
        } finally {
            $response.Dispose()
        }
    } finally {
        $content.Dispose()
    }
}

function Wait-ServerReady(
    [System.Diagnostics.Process]$Process,
    [System.Net.Http.HttpClient]$Client,
    [ref]$PeakUsedMiB,
    [int]$TimeoutSeconds
) {
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($stopwatch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if ($Process.HasExited) {
            throw "llama-server exited during startup with code $($Process.ExitCode)"
        }
        Update-PeakGpuMemory $PeakUsedMiB
        try {
            $task = $Client.GetAsync("http://127.0.0.1:$Port/health")
            $response = $task.GetAwaiter().GetResult()
            try {
                if ([int]$response.StatusCode -eq 200) {
                    return
                }
            } finally {
                $response.Dispose()
            }
        } catch {
        }
        Start-Sleep -Milliseconds 250
    }
    throw "llama-server was not ready within $TimeoutSeconds seconds"
}

function Get-ServerMetrics([System.Net.Http.HttpClient]$Client) {
    $task = $Client.GetStringAsync("http://127.0.0.1:$Port/metrics")
    return $task.GetAwaiter().GetResult()
}

function New-ServerArguments([bool]$UseDraft) {
    $arguments = @(
        '-m', $targetPath,
        '-ngl', $TargetGpuLayers,
        '--fit', 'off',
        '-c', [string]$ContextSize,
        '-b', [string]$BatchSize,
        '-ub', [string]$BatchSize,
        '-fa', 'on',
        '-t', [string]$Threads,
        '--prio', '2',
        '-np', '1',
        '--host', '127.0.0.1',
        '--port', [string]$Port,
        '--metrics',
        '--no-warmup',
        '--offline',
        '--log-colors', 'off',
        '-lv', [string]$ServerVerbosity
    )
    if ($UseDraft) {
        $serverDraftMode = Get-ExternalDraftServerMode $DraftMode
        $arguments += @(
            '-md', $draftPath,
            '--spec-type', "draft-$serverDraftMode",
            '--spec-draft-n-max', [string]$DraftMax,
            '-ngld', $DraftGpuLayers,
            '-td', [string]$Threads,
            '--prio-draft', '2'
        )
    } else {
        $arguments += @('--spec-type', 'none')
    }
    return $arguments
}

function Stop-Server([AllowNull()][System.Diagnostics.Process]$Process) {
    if ($null -ne $Process -and -not $Process.HasExited) {
        $Process.Kill()
        $Process.WaitForExit()
    }
    for ($attempt = 0; $attempt -lt 50; $attempt++) {
        if (-not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
            return
        }
        Start-Sleep -Milliseconds 100
    }
    throw "Port $Port remained in use after llama-server shutdown"
}

function New-SkippedModeResult([string]$Name, [string]$Reason) {
    return [ordered]@{
        name = $Name
        status = 'skipped'
        error = $Reason
        started_at = $null
        completed_at = $null
        admission = $null
        process = $null
        memory = $null
        samples = @()
        summary = [ordered]@{
            median_tokens_per_second = $null
            minimum_tokens_per_second = $null
            maximum_tokens_per_second = $null
            draft_tokens = 0
            accepted_draft_tokens = 0
            acceptance_rate = $null
            verification_steps = 0
            mean_accepted_length = $null
        }
        logs = $null
    }
}

function Invoke-Mode([string]$Name, [bool]$UseDraft, [hashtable]$RequestBody) {
    $stdoutPath = Join-Path $outputDirectory "$Name.stdout.log"
    $stderrPath = Join-Path $outputDirectory "$Name.stderr.log"
    $process = $null
    $client = $null
    $idleUsedMiB = Get-GpuUsedMiB
    $peakUsedMiB = $idleUsedMiB
    $startedAt = [DateTimeOffset]::UtcNow
    $samplesResult = @()
    $metricsBefore = ''
    $metricsAfter = ''
    $failure = $null
    $admission = $null
    $effectivePriority = $null
    $effectiveAffinity = $null

    try {
        $admission = Wait-NvidiaGpuIdleAdmission
        $arguments = New-ServerArguments $UseDraft
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
            $effectiveAffinity = '0x{0:x}' -f $effectiveAffinityValue
        } else {
            $effectiveAffinity = '0x{0:x}' -f [uint64]$process.ProcessorAffinity.ToInt64()
        }

        $client = [System.Net.Http.HttpClient]::new()
        $client.Timeout = [TimeSpan]::FromSeconds(900)
        Wait-ServerReady $process $client ([ref]$peakUsedMiB) 300

        for ($warmup = 0; $warmup -lt $WarmupRuns; $warmup++) {
            $null = Invoke-HttpJson `
                $client `
                "http://127.0.0.1:$Port/completion" `
                $RequestBody `
                ([ref]$peakUsedMiB)
        }

        $metricsBefore = Get-ServerMetrics $client
        for ($sample = 0; $sample -lt $Samples; $sample++) {
            $response = Invoke-HttpJson `
                $client `
                "http://127.0.0.1:$Port/completion" `
                $RequestBody `
                ([ref]$peakUsedMiB)
            if ($null -eq $response.timings -or $null -eq $response.content) {
                throw 'llama-server completion response omitted timings or content'
            }
            $samplesResult += [ordered]@{
                index = $sample + 1
                output_sha256 = Get-TextSha256 ([string]$response.content)
                predicted_tokens = [int]$response.timings.predicted_n
                predicted_milliseconds = [double]$response.timings.predicted_ms
                predicted_tokens_per_second = [double]$response.timings.predicted_per_second
                prompt_tokens = [int]$response.timings.prompt_n
                prompt_milliseconds = [double]$response.timings.prompt_ms
                draft_tokens = if ($null -ne $response.timings.PSObject.Properties['draft_n']) {
                    [int]$response.timings.draft_n
                } else {
                    0
                }
                accepted_draft_tokens = if (
                    $null -ne $response.timings.PSObject.Properties['draft_n_accepted']
                ) {
                    [int]$response.timings.draft_n_accepted
                } else {
                    0
                }
            }
        }
        $metricsAfter = Get-ServerMetrics $client
        Update-PeakGpuMemory ([ref]$peakUsedMiB)
    } catch {
        $failure = $_.Exception.Message
    } finally {
        if ($null -ne $client) {
            $client.Dispose()
        }
        Stop-Server $process
    }

    $rates = @(
        $samplesResult |
            ForEach-Object { [double]$_['predicted_tokens_per_second'] }
    )
    $draftTokens = [int64](
        ($samplesResult | ForEach-Object { [int64]$_['draft_tokens'] } | Measure-Object -Sum).Sum
    )
    $acceptedTokens = [int64](
        ($samplesResult | ForEach-Object { [int64]$_['accepted_draft_tokens'] } | Measure-Object -Sum).Sum
    )
    $verificationSteps = [int64](
        (Get-MetricValue $metricsAfter 'llamacpp:spec_decode_num_drafts_total') -
        (Get-MetricValue $metricsBefore 'llamacpp:spec_decode_num_drafts_total')
    )
    $stdoutHash = if (Test-Path -LiteralPath $stdoutPath -PathType Leaf) {
        Get-NormalizedSha256 $stdoutPath
    } else {
        $null
    }
    $stderrHash = if (Test-Path -LiteralPath $stderrPath -PathType Leaf) {
        Get-NormalizedSha256 $stderrPath
    } else {
        $null
    }

    return [ordered]@{
        name = $Name
        status = if ($null -eq $failure) { 'passed' } else { 'failed' }
        error = $failure
        started_at = $startedAt.ToString('o')
        completed_at = [DateTimeOffset]::UtcNow.ToString('o')
        admission = $admission
        process = [ordered]@{
            requested_priority = $ProcessPriority
            effective_priority = $effectivePriority
            requested_affinity = if ($ProcessorAffinityMask -gt 0) {
                '0x{0:x}' -f $ProcessorAffinityMask
            } else {
                $null
            }
            effective_affinity = $effectiveAffinity
        }
        memory = [ordered]@{
            idle_used_mib = $idleUsedMiB
            peak_used_mib = $peakUsedMiB
            peak_increment_mib = $peakUsedMiB - $idleUsedMiB
        }
        samples = $samplesResult
        summary = [ordered]@{
            median_tokens_per_second = Get-Median $rates
            minimum_tokens_per_second = if ($rates.Count -gt 0) {
                [double]($rates | Measure-Object -Minimum).Minimum
            } else {
                $null
            }
            maximum_tokens_per_second = if ($rates.Count -gt 0) {
                [double]($rates | Measure-Object -Maximum).Maximum
            } else {
                $null
            }
            draft_tokens = $draftTokens
            accepted_draft_tokens = $acceptedTokens
            acceptance_rate = if ($draftTokens -gt 0) { $acceptedTokens / $draftTokens } else { $null }
            verification_steps = $verificationSteps
            mean_accepted_length = if ($verificationSteps -gt 0) {
                1.0 + ($acceptedTokens / $verificationSteps)
            } else {
                $null
            }
        }
        logs = [ordered]@{
            stdout = [ordered]@{ file = [System.IO.Path]::GetFileName($stdoutPath); sha256 = $stdoutHash }
            stderr = [ordered]@{ file = [System.IO.Path]::GetFileName($stderrPath); sha256 = $stderrHash }
        }
    }
}
