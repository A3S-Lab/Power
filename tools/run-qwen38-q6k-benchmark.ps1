param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9][a-z0-9-]*$')]
    [string]$Label,

    [Parameter(Mandatory = $true)]
    [string]$Config,

    [ValidateRange(1, 100)]
    [int]$Samples = 3,

    [ValidateRange(0, 20)]
    [int]$WarmupRuns = 1,

    [ValidateRange(1, 4096)]
    [int]$MaxTokens = 256,

    [ValidateRange(1, 4096)]
    [int]$NumBatch = 24,

    [ValidateRange(0.0, 1000000.0)]
    [double]$MinimumTokensPerSecond = 0.0,

    [ValidateSet('Normal', 'AboveNormal', 'High')]
    [string]$ProcessPriority = 'High',

    [UInt64]$ProcessorAffinityMask = 0,

    [ValidateRange(0, 10000)]
    [int]$LockGpuClockMHz = 0,

    [switch]$RequireHighPerformancePowerPlan,

    [switch]$RequireCleanTree,

    [string]$TargetDirectory = 'target-native-sm89-ninja',

    [string]$BenchmarkRoot = 'D:\models\a3s-power\qwen38\benchmark',

    [string]$PromptFile,

    [string]$PowerHome = 'D:\models\a3s-power\qwen38\power-home',

    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._+-]*$')]
    [string]$ExpectedBackend = 'llama.cpp',

    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ModelHash = '562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727',

    [string]$RustLog = 'a3s_power::backend::llamacpp::speculative_runtime=info,a3s_power=info'
)

$ErrorActionPreference = 'Stop'

$powerRoot = Split-Path -Parent $PSScriptRoot
$server = Join-Path $powerRoot "$TargetDirectory\release\a3s-power.exe"
$benchmark = Join-Path $powerRoot "$TargetDirectory\release\a3s-power-speculative-bench.exe"
$prompt = if ([string]::IsNullOrWhiteSpace($PromptFile)) {
    Join-Path $powerRoot 'docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt'
} else {
    [System.IO.Path]::GetFullPath($PromptFile)
}
$stdout = Join-Path $BenchmarkRoot "$Label.stdout.log"
$stderr = Join-Path $BenchmarkRoot "$Label.stderr.log"
$report = Join-Path $BenchmarkRoot "$Label.json"
$environmentReport = Join-Path $BenchmarkRoot "$Label.environment.json"
$model = 'qwen3.8-27b-q6-k'
$process = $null
$effectiveProcessorAffinityMask = $null

foreach ($requiredPath in @($server, $benchmark, $prompt, $Config)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required benchmark input does not exist: $requiredPath"
    }
}

if (Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 11434 is already in use'
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
$configHash = (Get-FileHash -LiteralPath $Config -Algorithm SHA256).Hash.ToLowerInvariant()
$promptHash = (Get-FileHash -LiteralPath $prompt -Algorithm SHA256).Hash.ToLowerInvariant()

$env:A3S_POWER_HOME = $PowerHome
$env:RUST_LOG = $RustLog
$gpuClockLocked = $false
$gpuSnapshot = @()
$gpuProcessSnapshot = @()

try {
    if ($LockGpuClockMHz -gt 0) {
        & nvidia-smi.exe --lock-gpu-clocks="$LockGpuClockMHz,$LockGpuClockMHz" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to lock the GPU graphics clock at $LockGpuClockMHz MHz"
        }
        $gpuClockLocked = $true
    }
    $gpuSnapshot = @(& nvidia-smi.exe `
        --query-gpu=name,driver_version,pstate,clocks.current.graphics,clocks.max.graphics,power.limit,temperature.gpu,memory.total `
        --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to capture the NVIDIA GPU state'
    }
    $gpuProcessSnapshot = @(& nvidia-smi.exe)
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to capture the NVIDIA GPU process state'
    }

    $process = Start-Process -FilePath $server `
        -ArgumentList @('serve', '--config', $Config) `
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
                -Uri "http://127.0.0.1:11434/v1/models/$model" `
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
        throw 'Server did not expose the registered Q6_K model'
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

    $rawLines = @(& $benchmark run `
        --url 'http://127.0.0.1:11434' `
        --model $model `
        --model-sha256 $ModelHash `
        --mode mtp `
        --power-commit $powerCommit `
        --hardware-label "rtx4090-qwen38-q6k-$Label" `
        --prompt-file $prompt `
        --max-tokens $MaxTokens `
        --num-ctx 4096 `
        --num-batch $NumBatch `
        --seed 42 `
        --warmup-runs $WarmupRuns `
        --samples $Samples `
        --min-tokens-per-second $MinimumTokensPerSecond `
        --timeout-secs 900)
    $benchmarkExitCode = $LASTEXITCODE
    $raw = $rawLines -join [Environment]::NewLine
    if ([string]::IsNullOrWhiteSpace($raw)) {
        throw "Benchmark exited with code $benchmarkExitCode without a JSON report"
    }
    Set-Content -LiteralPath $report -Value $raw -Encoding utf8

    $environment = [ordered]@{
        schema = 'a3s.power.speculative-benchmark.environment.v1'
        created_at = [DateTimeOffset]::UtcNow.ToString('o')
        power_commit = $powerCommit
        dirty_worktree = $gitStatus.Count -ne 0
        git_status = $gitStatus
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
            path = [System.IO.Path]::GetFullPath($Config)
            sha256 = $configHash
        }
        prompt = [ordered]@{
            path = [System.IO.Path]::GetFullPath($prompt)
            sha256 = $promptHash
        }
        model_sha256 = $ModelHash.ToLowerInvariant()
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
            clock_lock_mhz = if ($LockGpuClockMHz -gt 0) { $LockGpuClockMHz } else { $null }
            nvidia_smi = $gpuSnapshot
            process_snapshot = $gpuProcessSnapshot
        }
        report = [ordered]@{
            path = [System.IO.Path]::GetFullPath($report)
            sha256 = (Get-FileHash -LiteralPath $report -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $environment | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $environmentReport -Encoding utf8
    if ($benchmarkExitCode -ne 0) {
        throw "Benchmark exited with code $benchmarkExitCode; report and environment were retained"
    }
    $raw
} finally {
    if ($process -and -not $process.HasExited) {
        $process.Kill()
        $process.WaitForExit()
    }
    if ($gpuClockLocked) {
        & nvidia-smi.exe --reset-gpu-clocks | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning 'Failed to reset the GPU graphics clock'
        }
    }
}
