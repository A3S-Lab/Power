param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9][a-z0-9-]*$')]
    [string]$Label,

    [Parameter(Mandatory = $true)]
    [string]$Config,

    [Parameter(Mandatory = $true)]
    [string]$Model,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ModelHash,

    [Parameter(Mandatory = $true)]
    [string]$PromptFile,

    [Parameter(Mandatory = $true)]
    [string]$PowerHome,

    [ValidateSet('off', 'prompt-lookup', 'ngram-context', 'draft-model', 'mtp', 'dflash', 'dspark')]
    [string]$Mode = 'mtp',

    [ValidateRange(1, 100)]
    [int]$Samples = 3,

    [ValidateRange(0, 20)]
    [int]$WarmupRuns = 1,

    [ValidateRange(2, 4096)]
    [int]$MaxTokens = 256,

    [ValidateRange(2, 1048576)]
    [int]$NumCtx = 4096,

    [ValidateRange(1, 4096)]
    [int]$NumBatch = 24,

    [ValidateRange(0.0, 1000000.0)]
    [double]$MinimumTokensPerSecond = 0.0,

    [ValidateRange(0.0, 1000000.0)]
    [Nullable[double]]$MinimumSampleTokensPerSecond,

    [ValidateSet('Normal', 'AboveNormal', 'High')]
    [string]$ProcessPriority = 'High',

    [UInt64]$ProcessorAffinityMask = 0,

    [ValidateRange(0, 10000)]
    [int]$LockGpuClockMHz = 0,

    [ValidateRange(0, 100)]
    [int]$MaximumIdleGpuUtilizationPercent = 100,

    [ValidateRange(1, 120)]
    [int]$IdleGpuSampleCount = 3,

    [ValidateRange(100, 60000)]
    [int]$IdleGpuSampleIntervalMilliseconds = 500,

    [ValidateScript({ $_ -ge 0 })]
    [int[]]$NvidiaGpuIndices = @(),

    [switch]$RequireHighPerformancePowerPlan,

    [switch]$RequireCleanTree,

    [string]$TargetDirectory = 'target',

    [string]$BenchmarkRoot = 'target-gguf-speculative-benchmark',

    [ValidateRange(1, 65535)]
    [int]$Port = 11434,

    [string]$HardwareLabel,

    [switch]$PreflightOnly,

    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._+-]*$')]
    [string]$ExpectedBackend = 'llama.cpp',

    [string]$RustLog = 'a3s_power::backend::llamacpp::speculative_runtime=info,a3s_power=info'
)

$ErrorActionPreference = 'Stop'

$powerRoot = Split-Path -Parent $PSScriptRoot
$server = Join-Path $powerRoot "$TargetDirectory\release\a3s-power.exe"
$benchmark = Join-Path $powerRoot "$TargetDirectory\release\a3s-power-speculative-bench.exe"
$prompt = [System.IO.Path]::GetFullPath($PromptFile)
$configPath = [System.IO.Path]::GetFullPath($Config)
$benchmarkRootPath = if ([System.IO.Path]::IsPathRooted($BenchmarkRoot)) {
    [System.IO.Path]::GetFullPath($BenchmarkRoot)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $powerRoot $BenchmarkRoot))
}
$stdout = Join-Path $benchmarkRootPath "$Label.stdout.log"
$stderr = Join-Path $benchmarkRootPath "$Label.stderr.log"
$report = Join-Path $benchmarkRootPath "$Label.json"
$environmentReport = Join-Path $benchmarkRootPath "$Label.environment.json"
$preflightReport = Join-Path $benchmarkRootPath "$Label.preflight.json"
$encodedModel = [Uri]::EscapeDataString($Model)
$normalizedModelHash = $ModelHash.ToLowerInvariant()
$effectiveHardwareLabel = if ([string]::IsNullOrWhiteSpace($HardwareLabel)) {
    "$Model-$Label"
} else {
    $HardwareLabel
}
$process = $null
$effectiveProcessorAffinityMask = $null

if ([string]::IsNullOrWhiteSpace($Model) -or $Model.Length -gt 256 -or $Model.Trim() -ne $Model) {
    throw 'Model must be a non-empty name of at most 256 characters without surrounding whitespace'
}
if ([string]::IsNullOrWhiteSpace($effectiveHardwareLabel) -or
    $effectiveHardwareLabel.Length -gt 256 -or
    $effectiveHardwareLabel.Trim() -ne $effectiveHardwareLabel) {
    throw 'HardwareLabel must contain at most 256 characters without surrounding whitespace'
}
if ($NvidiaGpuIndices.Count -ne @($NvidiaGpuIndices | Sort-Object -Unique).Count) {
    throw 'NvidiaGpuIndices must not contain duplicate device indices'
}
if ($NvidiaGpuIndices.Count -eq 0 -and
    ($LockGpuClockMHz -gt 0 -or
     $MaximumIdleGpuUtilizationPercent -lt 100 -or
     $PSBoundParameters.ContainsKey('IdleGpuSampleCount') -or
     $PSBoundParameters.ContainsKey('IdleGpuSampleIntervalMilliseconds'))) {
    throw 'NvidiaGpuIndices is required when an NVIDIA clock lock or idle-utilization gate is requested'
}
if ($MaxTokens -gt $NumCtx -or $NumBatch -gt $NumCtx) {
    throw 'NumCtx must be at least MaxTokens and NumBatch'
}

foreach ($requiredPath in @($server, $benchmark, $prompt, $configPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required benchmark input does not exist: $requiredPath"
    }
}

New-Item -ItemType Directory -Force -Path $benchmarkRootPath | Out-Null

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use"
}

$powerCommit = (& git -C $powerRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to resolve the Power commit'
}
$gitStatus = @(& git -C $powerRoot status --porcelain=v1)
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to inspect the Power git worktree'
}
if ($RequireCleanTree -and $gitStatus.Count -ne 0) {
    throw 'The Power worktree must be clean for this capture'
}

$activePowerScheme = (& powercfg.exe /getactivescheme) -join [Environment]::NewLine
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to inspect the active Windows power scheme'
}
if ($RequireHighPerformancePowerPlan -and
    $activePowerScheme -notmatch '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c') {
    throw 'The Windows High performance power plan is required for this capture'
}

$serverHash = (Get-FileHash -LiteralPath $server -Algorithm SHA256).Hash.ToLowerInvariant()
$benchmarkHash = (Get-FileHash -LiteralPath $benchmark -Algorithm SHA256).Hash.ToLowerInvariant()
$configHash = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
$promptHash = (Get-FileHash -LiteralPath $prompt -Algorithm SHA256).Hash.ToLowerInvariant()
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

$env:A3S_POWER_HOME = $PowerHome
$env:RUST_LOG = $RustLog
$lockedGpuIndices = @()
$gpuSnapshot = @()
$gpuProcessSnapshot = @()
$idleGpuUtilization = @()
$maximumObservedGpuUtilization = $null
$idleWindowStartedAt = $null
$idleWindowCompletedAt = $null
$idleWindowElapsedMilliseconds = $null
$preflightFailure = $null
$preflightHash = $null

try {
    if ($NvidiaGpuIndices.Count -gt 0) {
        if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
            $preflightFailure = [ordered]@{
                code = 'nvidia-smi-unavailable'
                message = 'nvidia-smi.exe is required when NvidiaGpuIndices is configured'
            }
        }

        if (-not $preflightFailure) {
            $idleWindowStartedAt = [DateTimeOffset]::UtcNow.ToString('o')
            $idleWindowStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            for ($sampleIndex = 0; $sampleIndex -lt $IdleGpuSampleCount; $sampleIndex++) {
                foreach ($gpuIndex in $NvidiaGpuIndices) {
                    $utilization = @(& nvidia-smi.exe `
                        --id=$gpuIndex `
                        --query-gpu=utilization.gpu `
                        --format=csv,noheader,nounits)
                    if ($LASTEXITCODE -ne 0 -or $utilization.Count -ne 1) {
                        $preflightFailure = [ordered]@{
                            code = 'nvidia-idle-sample-failed'
                            message = "Failed to capture idle utilization for NVIDIA GPU $gpuIndex"
                        }
                        break
                    }
                    $parsedUtilization = 0
                    if (-not [int]::TryParse($utilization[0].Trim(), [ref]$parsedUtilization) -or
                        $parsedUtilization -lt 0 -or
                        $parsedUtilization -gt 100) {
                        $preflightFailure = [ordered]@{
                            code = 'nvidia-idle-sample-invalid'
                            message = "NVIDIA GPU $gpuIndex reported an invalid utilization value: $($utilization[0])"
                        }
                        break
                    }
                    $idleGpuUtilization += [ordered]@{
                        sample = $sampleIndex
                        gpu_index = $gpuIndex
                        utilization_percent = $parsedUtilization
                        observed_at = [DateTimeOffset]::UtcNow.ToString('o')
                    }
                }
                if ($preflightFailure) {
                    break
                }
                if ($sampleIndex -lt ($IdleGpuSampleCount - 1)) {
                    Start-Sleep -Milliseconds $IdleGpuSampleIntervalMilliseconds
                }
            }
            $idleWindowStopwatch.Stop()
            $idleWindowElapsedMilliseconds = [int64]$idleWindowStopwatch.ElapsedMilliseconds
            $idleWindowCompletedAt = [DateTimeOffset]::UtcNow.ToString('o')
        }

        if (-not $preflightFailure) {
            $maximumObservedGpuUtilization = [int](@(
                $idleGpuUtilization | ForEach-Object { $_.utilization_percent }
            ) | Measure-Object -Maximum).Maximum
            if ($maximumObservedGpuUtilization -gt
                $MaximumIdleGpuUtilizationPercent) {
                $observations = @(
                    $idleGpuUtilization | ForEach-Object {
                        "sample $($_.sample), GPU $($_.gpu_index)=$($_.utilization_percent)%"
                    }
                ) -join '; '
                $preflightFailure = [ordered]@{
                    code = 'nvidia-idle-utilization-exceeded'
                    message = "GPU idle utilization exceeded $MaximumIdleGpuUtilizationPercent percent: $observations"
                }
            }
        }

        if (-not $preflightFailure -and $LockGpuClockMHz -gt 0) {
            foreach ($gpuIndex in $NvidiaGpuIndices) {
                & nvidia-smi.exe `
                    --id=$gpuIndex `
                    --lock-gpu-clocks="$LockGpuClockMHz,$LockGpuClockMHz" | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    $preflightFailure = [ordered]@{
                        code = 'nvidia-clock-lock-failed'
                        message = "Failed to lock NVIDIA GPU $gpuIndex at $LockGpuClockMHz MHz"
                    }
                    break
                }
                $lockedGpuIndices += $gpuIndex
            }
        }

        if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
            foreach ($gpuIndex in $NvidiaGpuIndices) {
                $snapshot = @(& nvidia-smi.exe `
                    --id=$gpuIndex `
                    --query-gpu=index,name,driver_version,pstate,clocks.current.graphics,clocks.max.graphics,power.limit,temperature.gpu,memory.total `
                    --format=csv,noheader,nounits)
                if ($LASTEXITCODE -ne 0 -or $snapshot.Count -ne 1) {
                    if (-not $preflightFailure) {
                        $preflightFailure = [ordered]@{
                            code = 'nvidia-state-snapshot-failed'
                            message = "Failed to capture NVIDIA GPU $gpuIndex state"
                        }
                    }
                    break
                }
                $gpuSnapshot += $snapshot[0]
            }
            $gpuProcessSnapshot = @(& nvidia-smi.exe)
            if ($LASTEXITCODE -ne 0) {
                if (-not $preflightFailure) {
                    $preflightFailure = [ordered]@{
                        code = 'nvidia-process-snapshot-failed'
                        message = 'Failed to capture the NVIDIA GPU process state'
                    }
                }
            }
        }
    }

    $preflight = [ordered]@{
        schema = 'a3s.power.speculative-benchmark.preflight.v1'
        created_at = [DateTimeOffset]::UtcNow.ToString('o')
        passed = $null -eq $preflightFailure
        failure = $preflightFailure
        power_commit = $powerCommit
        dirty_worktree = $gitStatus.Count -ne 0
        git_status = $gitStatus
        requirements = [ordered]@{
            clean_tree = [bool]$RequireCleanTree
            high_performance_power_plan = [bool]$RequireHighPerformancePowerPlan
            expected_exclusive_backend = $ExpectedBackend
        }
        active_power_scheme = $activePowerScheme.Trim()
        server = [ordered]@{
            path = [System.IO.Path]::GetFullPath($server)
            sha256 = $serverHash
        }
        benchmark_client = [ordered]@{
            path = [System.IO.Path]::GetFullPath($benchmark)
            sha256 = $benchmarkHash
        }
        config = [ordered]@{
            path = $configPath
            sha256 = $configHash
        }
        prompt = [ordered]@{
            path = $prompt
            sha256 = $promptHash
        }
        model = [ordered]@{
            name = $Model
            sha256 = $normalizedModelHash
        }
        gpu = [ordered]@{
            provider = if ($NvidiaGpuIndices.Count -gt 0) { 'nvidia' } else { 'none' }
            indices = $NvidiaGpuIndices
            requested_clock_lock_mhz = if ($LockGpuClockMHz -gt 0) { $LockGpuClockMHz } else { $null }
            clock_lock_applied_indices = $lockedGpuIndices
            maximum_idle_utilization_percent = if ($NvidiaGpuIndices.Count -gt 0) {
                $MaximumIdleGpuUtilizationPercent
            } else {
                $null
            }
            idle_sample_count = if ($NvidiaGpuIndices.Count -gt 0) { $IdleGpuSampleCount } else { $null }
            idle_sample_interval_milliseconds = if ($NvidiaGpuIndices.Count -gt 0) {
                $IdleGpuSampleIntervalMilliseconds
            } else {
                $null
            }
            idle_window_duration_milliseconds = if ($NvidiaGpuIndices.Count -gt 0) {
                [int64]($IdleGpuSampleCount - 1) * $IdleGpuSampleIntervalMilliseconds
            } else {
                $null
            }
            observed_idle_window_started_at = $idleWindowStartedAt
            observed_idle_window_completed_at = $idleWindowCompletedAt
            observed_idle_window_duration_milliseconds = $idleWindowElapsedMilliseconds
            maximum_observed_idle_utilization_percent = $maximumObservedGpuUtilization
            idle_utilization_samples = $idleGpuUtilization
            nvidia_smi = $gpuSnapshot
            process_snapshot = $gpuProcessSnapshot
        }
    }
    $preflightJson = $preflight | ConvertTo-Json -Depth 6
    [System.IO.File]::WriteAllText(
        $preflightReport,
        $preflightJson + [Environment]::NewLine,
        $utf8NoBom
    )
    $preflightHash = (Get-FileHash -LiteralPath $preflightReport -Algorithm SHA256).Hash.ToLowerInvariant()

    if ($preflightFailure) {
        throw $preflightFailure.message
    }
    if ($PreflightOnly) {
        $preflightJson
        return
    }

    # Start-Process joins ArgumentList entries into one Windows command line.
    # Use option=value and quote the canonical path so configs under a path
    # containing whitespace remain one native argument on PowerShell 5.1.
    $serverArguments = @(
        'serve',
        "--config=`"$configPath`"",
        '--host=127.0.0.1',
        "--port=$Port"
    )
    $process = Start-Process -FilePath $server `
        -ArgumentList $serverArguments `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    $process.PriorityClass = $ProcessPriority
    if ($ProcessorAffinityMask -gt 0) {
        $process.ProcessorAffinity = [IntPtr]::new([int64]$ProcessorAffinityMask)
    }
    $effectiveProcessorAffinityMask = '0x{0:x}' -f [uint64]$process.ProcessorAffinity.ToInt64()

    $ready = $false
    for ($attempt = 0; $attempt -lt 180; $attempt++) {
        if ($process.HasExited) {
            $message = Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue
            throw "Server exited before becoming ready: $message"
        }
        try {
            $response = Invoke-WebRequest -UseBasicParsing `
                -Uri "http://127.0.0.1:$Port/v1/models/$encodedModel" `
                -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        throw "Server did not expose the registered model '$Model'"
    }

    $startupLog = Get-Content -LiteralPath $stdout -Raw -ErrorAction Stop
    $backendMatch = [regex]::Match(
        $startupLog,
        'Initialized backends\s+backends=\[(?<names>[^\]]*)\]'
    )
    if (-not $backendMatch.Success) {
        throw 'Server startup log did not expose the initialized backend set'
    }
    $backendNames = @(
        [regex]::Matches($backendMatch.Groups['names'].Value, '"(?<name>[^"]+)"') |
            ForEach-Object { $_.Groups['name'].Value }
    )
    if ($backendNames.Count -ne 1 -or $backendNames[0] -ne $ExpectedBackend) {
        throw "Benchmark requires the exclusive '$ExpectedBackend' backend; server initialized: $($backendNames -join ', ')"
    }

    $benchmarkArguments = @(
        'run',
        '--url', "http://127.0.0.1:$Port",
        '--model', $Model,
        '--model-sha256', $normalizedModelHash,
        '--mode', $Mode,
        '--power-commit', $powerCommit,
        '--hardware-label', $effectiveHardwareLabel,
        '--prompt-file', $prompt,
        '--max-tokens', [string]$MaxTokens,
        '--num-ctx', [string]$NumCtx,
        '--num-batch', [string]$NumBatch,
        '--seed', '42',
        '--warmup-runs', [string]$WarmupRuns,
        '--samples', [string]$Samples,
        '--min-tokens-per-second', [string]$MinimumTokensPerSecond,
        '--timeout-secs', '900'
    )
    if ($PSBoundParameters.ContainsKey('MinimumSampleTokensPerSecond')) {
        $benchmarkArguments += @(
            '--min-sample-tokens-per-second',
            [string]$MinimumSampleTokensPerSecond
        )
    }
    $rawLines = @(& $benchmark @benchmarkArguments)
    $benchmarkExitCode = $LASTEXITCODE
    $raw = $rawLines -join [Environment]::NewLine
    if ([string]::IsNullOrWhiteSpace($raw)) {
        throw "Benchmark exited with code $benchmarkExitCode without a JSON report"
    }
    [System.IO.File]::WriteAllText(
        $report,
        $raw + [Environment]::NewLine,
        $utf8NoBom
    )

    $environment = [ordered]@{
        schema = 'a3s.power.speculative-benchmark.environment.v1'
        created_at = [DateTimeOffset]::UtcNow.ToString('o')
        power_commit = $powerCommit
        dirty_worktree = $gitStatus.Count -ne 0
        git_status = $gitStatus
        preflight = [ordered]@{
            path = [System.IO.Path]::GetFullPath($preflightReport)
            sha256 = $preflightHash
        }
        server = [ordered]@{
            path = [System.IO.Path]::GetFullPath($server)
            sha256 = $serverHash
            expected_exclusive_backend = $ExpectedBackend
        }
        benchmark_client = [ordered]@{
            path = [System.IO.Path]::GetFullPath($benchmark)
            sha256 = $benchmarkHash
        }
        config = [ordered]@{
            path = $configPath
            sha256 = $configHash
            expected_mode = $Mode
        }
        prompt = [ordered]@{
            path = [System.IO.Path]::GetFullPath($prompt)
            sha256 = $promptHash
        }
        model = [ordered]@{
            name = $Model
            sha256 = $normalizedModelHash
        }
        thresholds = [ordered]@{
            median_tokens_per_second = $MinimumTokensPerSecond
            every_sample_tokens_per_second = if (
                $PSBoundParameters.ContainsKey('MinimumSampleTokensPerSecond')
            ) {
                [double]$MinimumSampleTokensPerSecond
            } else {
                $null
            }
        }
        active_power_scheme = $activePowerScheme.Trim()
        process_priority = $ProcessPriority
        process_affinity = [ordered]@{
            requested_mask = if ($ProcessorAffinityMask -gt 0) {
                '0x{0:x}' -f $ProcessorAffinityMask
            } else {
                $null
            }
            effective_mask = $effectiveProcessorAffinityMask
            logical_processor_count = [Environment]::ProcessorCount
        }
        gpu = [ordered]@{
            provider = if ($NvidiaGpuIndices.Count -gt 0) { 'nvidia' } else { 'none' }
            indices = $NvidiaGpuIndices
            clock_lock_mhz = if ($lockedGpuIndices.Count -gt 0) { $LockGpuClockMHz } else { $null }
            maximum_idle_utilization_percent = if ($NvidiaGpuIndices.Count -gt 0) {
                $MaximumIdleGpuUtilizationPercent
            } else {
                $null
            }
            idle_sample_count = if ($NvidiaGpuIndices.Count -gt 0) { $IdleGpuSampleCount } else { $null }
            idle_sample_interval_milliseconds = if ($NvidiaGpuIndices.Count -gt 0) {
                $IdleGpuSampleIntervalMilliseconds
            } else {
                $null
            }
            idle_window_duration_milliseconds = if ($NvidiaGpuIndices.Count -gt 0) {
                [int64]($IdleGpuSampleCount - 1) * $IdleGpuSampleIntervalMilliseconds
            } else {
                $null
            }
            observed_idle_window_started_at = $idleWindowStartedAt
            observed_idle_window_completed_at = $idleWindowCompletedAt
            observed_idle_window_duration_milliseconds = $idleWindowElapsedMilliseconds
            maximum_observed_idle_utilization_percent = $maximumObservedGpuUtilization
            idle_utilization_samples = $idleGpuUtilization
            idle_utilization_samples_percent = if ($NvidiaGpuIndices.Count -eq 1) {
                @($idleGpuUtilization | ForEach-Object { $_.utilization_percent })
            } else {
                $null
            }
            nvidia_smi = $gpuSnapshot
            process_snapshot = $gpuProcessSnapshot
        }
        report = [ordered]@{
            path = [System.IO.Path]::GetFullPath($report)
            sha256 = (Get-FileHash -LiteralPath $report -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $environmentJson = $environment | ConvertTo-Json -Depth 6
    [System.IO.File]::WriteAllText(
        $environmentReport,
        $environmentJson + [Environment]::NewLine,
        $utf8NoBom
    )
    if ($benchmarkExitCode -ne 0) {
        throw "Benchmark exited with code $benchmarkExitCode; report and environment were retained"
    }
    $raw
} finally {
    if ($process -and -not $process.HasExited) {
        $process.Kill()
        $process.WaitForExit()
    }
    foreach ($gpuIndex in $lockedGpuIndices) {
        & nvidia-smi.exe --id=$gpuIndex --reset-gpu-clocks | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Failed to reset NVIDIA GPU $gpuIndex graphics clock"
        }
    }
}
