param(
    [Parameter(Mandatory = $true)]
    [string]$LlamaBinDirectory,

    [Parameter(Mandatory = $true)]
    [string]$TargetModel,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$TargetSha256,

    [Parameter(Mandatory = $true)]
    [string]$DraftModel,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$DraftSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$LlamaCppCommit,

    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$LlamaCppRsCommit = '',

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [ValidateSet('dflash', 'dflash2', 'dspark')]
    [string]$DraftMode = 'dflash2',

    [ValidateRange(1, 20)]
    [int]$Repetitions = 3,

    [ValidateRange(64, 1048576)]
    [int]$ContextSize = 1024,

    [ValidateRange(8, 4096)]
    [int]$BatchSize = 12,

    [ValidateRange(1, 64)]
    [int]$Threads = 10,

    [ValidateRange(1, 64)]
    [int]$DraftMax = 7,

    [ValidateRange(0, 4096)]
    [int]$MaxTokensCap = 256,

    [ValidatePattern('^(all|auto|[0-9]+)$')]
    [string]$TargetGpuLayers = 'all',

    [ValidatePattern('^(all|auto|[0-9]+)$')]
    [string]$DraftGpuLayers = 'all',

    [ValidateRange(1, 65535)]
    [int]$Port = 11539,

    [ValidateRange(0, 31)]
    [int]$NvidiaGpuIndex = 0,

    [ValidateRange(0, 7)]
    [int]$ServerVerbosity = 3,

    [ValidateSet('Normal', 'AboveNormal', 'High')]
    [string]$ProcessPriority = 'High',

    [UInt64]$ProcessorAffinityMask = 0,

    [ValidateRange(0, 10000)]
    [int]$LockGpuClockMHz = 0,

    [switch]$CudaGraphOptimization,

    [ValidateSet(0, 1, 2, 4, 8, 16, 32)]
    [int]$CudaDeviceMaxConnections = 0,

    [ValidateRange(0, 100)]
    [int]$MaximumIdleGpuUtilizationPercent = 15,

    [ValidateRange(0, 262144)]
    [int]$MinimumIdleGpuMemoryFreeMiB = 23000,

    [ValidateRange(1, 120)]
    [int]$IdleGpuSampleCount = 3,

    [ValidateRange(100, 60000)]
    [int]$IdleGpuSampleIntervalMilliseconds = 500,

    [ValidateRange(1, 300)]
    [int]$IdleGpuWaitSeconds = 120,

    [string]$Tasks,

    [string]$TaskManifest,

    [string]$TaskSelection,

    [switch]$AllTasks,

    [string]$Model = 'qwen3.8-27b-q6-k',

    [string]$PythonLauncher = 'py',

    [string]$PythonVersion = '3.13',

    [switch]$IncludeContent,

    [switch]$RequireHighPerformancePowerPlan,

    [switch]$RequireCleanTree
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Add-Type -AssemblyName System.Net.Http

if ($BatchSize -le ($DraftMax + 1)) {
    throw 'BatchSize must exceed the target anchor plus every draft token'
}
if ($AllTasks -and -not [string]::IsNullOrWhiteSpace($TaskSelection)) {
    throw 'AllTasks and TaskSelection are mutually exclusive'
}

$powerRoot = Split-Path -Parent $PSScriptRoot
$qualityRoot = Join-Path $powerRoot 'docs\benchmarks\qwen3.8-27b-q6k-rtx4090\quality'
$evaluator = Join-Path $PSScriptRoot 'qwen38_quality_eval.py'
$reporter = Join-Path $PSScriptRoot 'qwen38_quality_report.py'
$binDirectory = [System.IO.Path]::GetFullPath($LlamaBinDirectory)
$serverPath = Join-Path $binDirectory 'llama-server.exe'
$targetPath = [System.IO.Path]::GetFullPath($TargetModel)
$draftPath = [System.IO.Path]::GetFullPath($DraftModel)
$outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
$tasksPath = if ([string]::IsNullOrWhiteSpace($Tasks)) {
    Join-Path $qualityRoot 'tasks-v1.json'
} else {
    [System.IO.Path]::GetFullPath($Tasks)
}
$manifestPath = if ([string]::IsNullOrWhiteSpace($TaskManifest)) {
    Join-Path $qualityRoot 'tasks-v1.manifest.json'
} else {
    [System.IO.Path]::GetFullPath($TaskManifest)
}
$selectionPath = if ($AllTasks) {
    $null
} elseif ([string]::IsNullOrWhiteSpace($TaskSelection)) {
    Join-Path $qualityRoot 'calibration-v1.selection.json'
} else {
    [System.IO.Path]::GetFullPath($TaskSelection)
}
$environmentPath = Join-Path $outputPath 'environment.json'
$aggregateJsonPath = Join-Path $outputPath 'quality-matrix.json'
$aggregateMarkdownPath = Join-Path $outputPath 'quality-matrix.md'
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

. (Join-Path $PSScriptRoot 'lib/gguf-speculative-benchmark.ps1')
. (Join-Path $PSScriptRoot 'lib/llamacpp-external-draft-benchmark.ps1')
. (Join-Path $PSScriptRoot 'lib/llamacpp-external-draft-quality.ps1')

$requiredPaths = [ordered]@{
    'llama-server' = $serverPath
    'TargetModel' = $targetPath
    'DraftModel' = $draftPath
    'Quality evaluator' = $evaluator
    'Quality reporter' = $reporter
    'Task cache' = $tasksPath
    'Task manifest' = $manifestPath
}
if ($selectionPath) {
    $requiredPaths['Task selection'] = $selectionPath
}
foreach ($required in $requiredPaths.GetEnumerator()) {
    Assert-Leaf $required.Value $required.Key
}
if (-not (Get-Command $PythonLauncher -ErrorAction SilentlyContinue)) {
    throw "Python launcher is not available: $PythonLauncher"
}
if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    throw 'nvidia-smi.exe is required for the external-draft quality benchmark'
}

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$actualTargetHash = Get-NormalizedSha256 $targetPath
$actualDraftHash = Get-NormalizedSha256 $draftPath
if ($actualTargetHash -ne $TargetSha256.ToLowerInvariant()) {
    throw "Target model SHA-256 mismatch: expected $TargetSha256, got $actualTargetHash"
}
if ($actualDraftHash -ne $DraftSha256.ToLowerInvariant()) {
    throw "Draft model SHA-256 mismatch: expected $DraftSha256, got $actualDraftHash"
}

$powerCommit = (& git -C $powerRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to resolve the Power commit'
}
$gitStatus = @(& git -C $powerRoot status --porcelain=v1)
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to inspect the Power worktree'
}
if ($RequireCleanTree -and $gitStatus.Count -gt 0) {
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

$taskInspectionArguments = @(
    $evaluator,
    'inspect-selection',
    '--tasks', $tasksPath,
    '--manifest', $manifestPath
)
if ($selectionPath) {
    $taskInspectionArguments += @('--task-selection', $selectionPath)
    $taskInspectionJson = @(
        & $PythonLauncher "-$PythonVersion" @taskInspectionArguments
    ) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0) {
        throw "Python task selection inspection exited with code $LASTEXITCODE"
    }
    $taskInspection = $taskInspectionJson | ConvertFrom-Json
    $expectedTaskCount = [int]$taskInspection.task_count
    $tasksDigest = [string]$taskInspection.tasks_sha256
} else {
    $taskPayload = Get-Content -LiteralPath $tasksPath -Raw | ConvertFrom-Json
    $expectedTaskCount = @($taskPayload.tasks).Count
    $tasksDigest = [string]$taskPayload.tasks_sha256
}
if ($expectedTaskCount -le 0) {
    throw 'The selected quality workload contains no tasks'
}

$serverHash = Get-NormalizedSha256 $serverPath
$runtimeFiles = Get-LlamaRuntimeFileIdentity $binDirectory
if (@($runtimeFiles).Count -lt 2) {
    throw 'llama.cpp runtime identity must include llama-server.exe and its DLLs'
}
$gpuIdentity = @(& nvidia-smi.exe `
    "--id=$NvidiaGpuIndex" `
    '--query-gpu=name,uuid,driver_version,memory.total,compute_cap,pstate,power.limit,clocks.max.graphics' `
    '--format=csv,noheader,nounits')
if ($LASTEXITCODE -ne 0 -or $gpuIdentity.Count -ne 1) {
    throw "Failed to capture NVIDIA GPU $NvidiaGpuIndex identity"
}

$modes = @(
    @{ label = 'q6-target-only'; use_draft = $false },
    @{ label = "q6-$DraftMode"; use_draft = $true }
)
$environment = [ordered]@{
    schema = 'a3s.power.llamacpp-external-draft-quality.environment.v1'
    created_at = [DateTimeOffset]::UtcNow.ToString('o')
    identity = [ordered]@{
        power_commit = $powerCommit
        dirty_worktree = $gitStatus.Count -gt 0
        git_status = $gitStatus
        llama_cpp_commit = $LlamaCppCommit.ToLowerInvariant()
        llama_cpp_rs_commit = if ($LlamaCppRsCommit) {
            $LlamaCppRsCommit.ToLowerInvariant()
        } else {
            $null
        }
        backend_source = if ($LlamaCppRsCommit) { 'llama-cpp-rs' } else { 'llama.cpp' }
        llama_runtime_files = $runtimeFiles
        target = [ordered]@{
            file = [System.IO.Path]::GetFileName($targetPath)
            size = [int64](Get-Item -LiteralPath $targetPath).Length
            sha256 = $actualTargetHash
            quantization = 'Q6_K'
        }
        draft = [ordered]@{
            file = [System.IO.Path]::GetFileName($draftPath)
            size = [int64](Get-Item -LiteralPath $draftPath).Length
            sha256 = $actualDraftHash
            mode = $DraftMode
            backend_mode = Get-ExternalDraftServerMode $DraftMode
        }
    }
    workload = [ordered]@{
        tasks_file = [System.IO.Path]::GetFileName($tasksPath)
        tasks_file_sha256 = Get-NormalizedSha256 $tasksPath
        tasks_sha256 = $tasksDigest
        manifest_sha256 = Get-NormalizedSha256 $manifestPath
        selection_file = if ($selectionPath) {
            [System.IO.Path]::GetFileName($selectionPath)
        } else {
            $null
        }
        selection_sha256 = if ($selectionPath) {
            Get-NormalizedSha256 $selectionPath
        } else {
            $null
        }
        task_count = $expectedTaskCount
        repetitions = $Repetitions
        total_requests = $expectedTaskCount * $Repetitions * $modes.Count
        max_tokens_cap = if ($MaxTokensCap -gt 0) { $MaxTokensCap } else { $null }
        seed = 42
        temperature = 0.0
    }
    configuration = [ordered]@{
        context_size = $ContextSize
        batch_size = $BatchSize
        threads = $Threads
        draft_max = $DraftMax
        target_gpu_layers = $TargetGpuLayers
        draft_gpu_layers = $DraftGpuLayers
        flash_attention = $true
        fit = $false
        parallel_slots = 1
        server_verbosity = $ServerVerbosity
        cyclic_mode_order = $true
    }
    tools = [ordered]@{
        runner_sha256 = Get-NormalizedSha256 $PSCommandPath
        evaluator_sha256 = Get-NormalizedSha256 $evaluator
        reporter_sha256 = Get-NormalizedSha256 $reporter
    }
    environment = [ordered]@{
        gpu = $gpuIdentity[0]
        gpu_index = $NvidiaGpuIndex
        cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()
        os = [System.Environment]::OSVersion.VersionString
        active_power_scheme = $activePowerScheme.Trim()
        process_priority = $ProcessPriority
        process_affinity = [ordered]@{
            requested_mask = if ($ProcessorAffinityMask -gt 0) {
                '0x{0:x}' -f $ProcessorAffinityMask
            } else {
                $null
            }
            logical_processor_count = [Environment]::ProcessorCount
        }
        gpu_controls = [ordered]@{
            requested_clock_lock_mhz = if ($LockGpuClockMHz -gt 0) {
                $LockGpuClockMHz
            } else {
                $null
            }
            clock_lock_applied = $false
            maximum_idle_utilization_percent = $MaximumIdleGpuUtilizationPercent
            minimum_idle_memory_free_mib = $MinimumIdleGpuMemoryFreeMiB
            consecutive_idle_sample_count = $IdleGpuSampleCount
            idle_sample_interval_milliseconds = $IdleGpuSampleIntervalMilliseconds
            idle_wait_seconds = $IdleGpuWaitSeconds
        }
        cuda_runtime = [ordered]@{
            graph_optimization = [bool]$CudaGraphOptimization
            device_max_connections = if ($CudaDeviceMaxConnections -gt 0) {
                $CudaDeviceMaxConnections
            } else {
                $null
            }
        }
    }
    gpu_admissions = @()
    processes = @()
    outputs = @()
}
Write-EnvironmentReport

$gpuClockLocked = $false
$savedCudaGraphOptimization = Get-Item Env:GGML_CUDA_GRAPH_OPT -ErrorAction SilentlyContinue
$savedCudaDeviceMaxConnections = Get-Item Env:CUDA_DEVICE_MAX_CONNECTIONS -ErrorAction SilentlyContinue
try {
    if ($CudaGraphOptimization) {
        $env:GGML_CUDA_GRAPH_OPT = '1'
    } else {
        Remove-Item Env:GGML_CUDA_GRAPH_OPT -ErrorAction SilentlyContinue
    }
    if ($CudaDeviceMaxConnections -gt 0) {
        $env:CUDA_DEVICE_MAX_CONNECTIONS = [string]$CudaDeviceMaxConnections
    } else {
        Remove-Item Env:CUDA_DEVICE_MAX_CONNECTIONS -ErrorAction SilentlyContinue
    }
    if ($LockGpuClockMHz -gt 0) {
        & nvidia-smi.exe `
            "--id=$NvidiaGpuIndex" `
            --lock-gpu-clocks="$LockGpuClockMHz,$LockGpuClockMHz" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to lock NVIDIA GPU $NvidiaGpuIndex at $LockGpuClockMHz MHz"
        }
        $gpuClockLocked = $true
        $environment.environment.gpu_controls.clock_lock_applied = $true
        Write-EnvironmentReport
    }

    for ($repetition = 1; $repetition -le $Repetitions; $repetition++) {
        for ($position = 0; $position -lt $modes.Count; $position++) {
            $mode = $modes[($position + $repetition - 1) % $modes.Count]
            Write-Output (
                "Running repetition $repetition/$Repetitions, " +
                "order $($position + 1)/$($modes.Count): $($mode.label)"
            )
            Invoke-QualityMode $mode $repetition ($position + 1)
        }
    }
} finally {
    if ($gpuClockLocked) {
        & nvidia-smi.exe "--id=$NvidiaGpuIndex" --reset-gpu-clocks | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Failed to reset NVIDIA GPU $NvidiaGpuIndex graphics clock"
        }
    }
    if ($savedCudaGraphOptimization) {
        $env:GGML_CUDA_GRAPH_OPT = $savedCudaGraphOptimization.Value
    } else {
        Remove-Item Env:GGML_CUDA_GRAPH_OPT -ErrorAction SilentlyContinue
    }
    if ($savedCudaDeviceMaxConnections) {
        $env:CUDA_DEVICE_MAX_CONNECTIONS = $savedCudaDeviceMaxConnections.Value
    } else {
        Remove-Item Env:CUDA_DEVICE_MAX_CONNECTIONS -ErrorAction SilentlyContinue
    }
}

$reports = @(
    Get-ChildItem -LiteralPath $outputPath -Filter 'r??-o?-*.json' -File |
        Sort-Object Name |
        ForEach-Object { $_.FullName }
)
$expectedReportCount = $Repetitions * $modes.Count
if ($reports.Count -ne $expectedReportCount) {
    throw "Expected $expectedReportCount quality reports, found $($reports.Count)"
}
$aggregateArguments = @(
    $evaluator,
    'aggregate',
    '--reports'
) + $reports + @(
    '--pair', 'q6-target-only', "q6-$DraftMode",
    '--output-json', $aggregateJsonPath,
    '--output-markdown', $aggregateMarkdownPath
)
Invoke-QualityPython $aggregateArguments
$environment.outputs = @(
    Get-ChildItem -LiteralPath $outputPath -File |
        Where-Object { $_.FullName -ne $environmentPath } |
        Sort-Object Name |
        ForEach-Object {
            [ordered]@{
                file = $_.Name
                size = [int64]$_.Length
                sha256 = Get-NormalizedSha256 $_.FullName
            }
        }
)
Write-EnvironmentReport
Write-Output "External-draft Q6_K quality matrix complete: $aggregateJsonPath"
