param(
    [Parameter(Mandatory = $true)]
    [string]$Q6PowerHome,

    [string]$Tbq4PowerHome,

    [ValidateSet(
        'pure-q6',
        'prefix-fr-release',
        'full-vocabulary-current',
        'dspark-q4'
    )]
    [string]$Profile = 'pure-q6',

    [string]$RuntimeConfig,

    [ValidateRange(1, 20)]
    [int]$Repetitions = 3,

    [ValidateRange(1, 4096)]
    [int]$NumBatch = 14,

    [ValidateRange(128, 131072)]
    [int]$NumCtx = 4096,

    [ValidateRange(0, 4096)]
    [int]$MaxTokensCap = 0,

    [ValidateRange(0, 4096)]
    [int]$MaxTokensOverride = 0,

    [ValidateRange(1, 65535)]
    [int]$Port = 11436,

    [ValidateSet('Normal', 'AboveNormal', 'High')]
    [string]$ProcessPriority = 'High',

    [UInt64]$ProcessorAffinityMask = 0,

    [ValidateRange(0, 10000)]
    [int]$LockGpuClockMHz = 0,

    [switch]$CudaHighPriority,

    [ValidateScript({ $_ -ge 0 })]
    [int]$NvidiaGpuIndex = 0,

    [ValidateRange(0, 100)]
    [int]$MaximumIdleGpuUtilizationPercent = 100,

    [ValidateRange(0, 262144)]
    [int]$MinimumIdleGpuMemoryFreeMiB = 0,

    [ValidateRange(1, 120)]
    [int]$IdleGpuSampleCount = 3,

    [ValidateRange(100, 60000)]
    [int]$IdleGpuSampleIntervalMilliseconds = 500,

    [ValidateRange(1, 300)]
    [int]$IdleGpuWaitSeconds = 60,

    [string]$TargetDirectory = 'target-native-sm89-ninja',

    [string]$OutputRoot = 'target-qwen38-quality',

    [string]$Model = 'qwen3.8-27b-q6-k',

    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$Q6ModelHash = '562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727',

    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$Tbq4ModelHash = '5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6',

    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$DsparkDraftHash = '12003c7f2642e2e87e979729e16947a913e2213d82136cb5024a36ec4871fef2',

    [string]$PythonLauncher = 'py',

    [string]$PythonVersion = '3.13',

    [string]$PreparedTaskCache,

    [string]$TaskSelection,

    [bool]$VerifyModelFiles = $true,

    [switch]$RequireCleanTree,

    [switch]$RequireHighPerformancePowerPlan,

    [switch]$IncludeContent,

    [switch]$ReuseCompatibleReportsAcrossCommits,

    [switch]$DescribeProfile
)

$ErrorActionPreference = 'Stop'

if ($MaxTokensCap -gt 0 -and $MaxTokensOverride -gt 0) {
    throw 'MaxTokensCap and MaxTokensOverride are mutually exclusive'
}

$powerRoot = Split-Path -Parent $PSScriptRoot
$profileHelper = Join-Path $PSScriptRoot 'lib/qwen38-quality-profile.ps1'
. $profileHelper
$profileDefinition = Resolve-Qwen38QualityProfile -Profile $Profile
if ($DescribeProfile) {
    $profileDefinition | ConvertTo-Json -Depth 6
    return
}
$server = Join-Path $powerRoot "$TargetDirectory\release\a3s-power.exe"
$evaluator = Join-Path $PSScriptRoot 'qwen38_quality_eval.py'
$reporter = Join-Path $PSScriptRoot 'qwen38_quality_report.py'
$qwenBenchmarkRoot = Join-Path $powerRoot 'docs\benchmarks\qwen3.8-27b-q6k-rtx4090'
$benchmarkRoot = Join-Path $qwenBenchmarkRoot 'quality'
$manifest = Join-Path $benchmarkRoot 'tasks-v1.manifest.json'
$config = if (-not [string]::IsNullOrWhiteSpace($RuntimeConfig)) {
    [System.IO.Path]::GetFullPath($RuntimeConfig)
} else {
    Join-Path $qwenBenchmarkRoot $profileDefinition.config_relative_path
}
$output = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    [System.IO.Path]::GetFullPath($OutputRoot)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $powerRoot $OutputRoot))
}
$tasks = Join-Path $output 'tasks-v1.json'
$taskSelectionPath = if ([string]::IsNullOrWhiteSpace($TaskSelection)) {
    $null
} else {
    [System.IO.Path]::GetFullPath($TaskSelection)
}
$environmentPath = Join-Path $output 'environment.json'
$aggregateJson = Join-Path $output 'quality-matrix.json'
$aggregateMarkdown = Join-Path $output 'quality-matrix.md'
$pythonPrefix = @("-$PythonVersion")

function Assert-PortAvailable {
    param([int]$CandidatePort)

    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $CandidatePort
    )
    try {
        $listener.Start()
    } catch {
        throw "Port $CandidatePort is already in use"
    } finally {
        $listener.Stop()
    }
}

function Get-ModelIdentity {
    param(
        [string]$PowerHome,
        [string]$ExpectedHash,
        [string]$ExpectedExternalDraftKind,
        [string]$ExpectedExternalDraftHash
    )

    $manifestPath = Join-Path $PowerHome "models\manifests\$Model.json"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Model manifest does not exist: $manifestPath"
    }
    $modelManifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($modelManifest.sha256 -ne $ExpectedHash) {
        throw "Model manifest hash mismatch in $manifestPath"
    }
    if (-not (Test-Path -LiteralPath $modelManifest.path -PathType Leaf)) {
        throw "GGUF does not exist: $($modelManifest.path)"
    }
    $modelFile = Get-Item -LiteralPath $modelManifest.path
    if ($modelFile.Length -ne $modelManifest.size) {
        throw "GGUF byte length differs from the manifest: $($modelManifest.path)"
    }
    if ($VerifyModelFiles) {
        $actualHash = (Get-FileHash -LiteralPath $modelManifest.path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $ExpectedHash) {
            throw "GGUF SHA-256 mismatch: $($modelManifest.path)"
        }
    }
    $externalDraftIdentity = $null
    if (-not [string]::IsNullOrWhiteSpace($ExpectedExternalDraftKind)) {
        if ($null -eq $modelManifest.external_draft) {
            throw "Model manifest does not bind an external draft: $manifestPath"
        }
        $draft = $modelManifest.external_draft
        if ($draft.kind -ne $ExpectedExternalDraftKind) {
            throw "External draft kind mismatch in $manifestPath"
        }
        if ($draft.sha256 -ne $ExpectedExternalDraftHash) {
            throw "External draft hash mismatch in $manifestPath"
        }
        if ($draft.target_sha256 -ne $ExpectedHash) {
            throw "External draft target hash mismatch in $manifestPath"
        }
        if (-not (Test-Path -LiteralPath $draft.path -PathType Leaf)) {
            throw "External draft GGUF does not exist: $($draft.path)"
        }
        $draftFile = Get-Item -LiteralPath $draft.path
        if ($draftFile.Length -ne $draft.size) {
            throw "External draft byte length differs from the manifest: $($draft.path)"
        }
        if ($VerifyModelFiles) {
            $actualDraftHash = (Get-FileHash -LiteralPath $draft.path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actualDraftHash -ne $ExpectedExternalDraftHash) {
                throw "External draft GGUF SHA-256 mismatch: $($draft.path)"
            }
        }
        $externalDraftIdentity = [ordered]@{
            kind = $draft.kind
            path = [System.IO.Path]::GetFullPath($draft.path)
            size = [int64]$draft.size
            sha256 = $draft.sha256
            target_sha256 = $draft.target_sha256
            source = $draft.source
            revision = $draft.revision
            license = $draft.license
            file_hash_verified = $VerifyModelFiles
        }
    }
    [ordered]@{
        power_home = [System.IO.Path]::GetFullPath($PowerHome)
        manifest = [System.IO.Path]::GetFullPath($manifestPath)
        path = [System.IO.Path]::GetFullPath($modelManifest.path)
        size = [int64]$modelManifest.size
        sha256 = $modelManifest.sha256
        file_hash_verified = $VerifyModelFiles
        external_draft = $externalDraftIdentity
    }
}

function Wait-NvidiaGpuIdleAdmission {
    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        throw 'nvidia-smi.exe is required for the quality benchmark GPU admission gate'
    }

    $startedAt = [DateTimeOffset]::UtcNow
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $accepted = [System.Collections.Generic.List[object]]::new()
    $observedSamples = 0
    $lastSample = $null
    while ($stopwatch.Elapsed.TotalSeconds -lt $IdleGpuWaitSeconds) {
        $lines = @(& $nvidiaSmi.Source `
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

function Invoke-Python {
    param([string[]]$Arguments)

    & $PythonLauncher @pythonPrefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python evaluator exited with code $LASTEXITCODE"
    }
}

function Get-ReportMetadata {
    param([string]$Path)

    $json = @(& $PythonLauncher @pythonPrefix $evaluator `
        'inspect-report' '--report' $Path) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0) {
        throw "Python report inspection exited with code $LASTEXITCODE"
    }
    return $json | ConvertFrom-Json
}

function Get-GpuCompatibilityIdentity {
    param([object[]]$GpuRows)

    return (@(
        $GpuRows | ForEach-Object {
            [string]$_ -replace ', P[0-9]+,', ','
        }
    ) -join "`n")
}

function Invoke-ModeRun {
    param(
        [hashtable]$Mode,
        [int]$Repetition,
        [int]$OrderIndex,
        [string]$ServerHash,
        [string]$PowerCommit,
        [bool]$ReuseCompatible
    )

    Assert-PortAvailable -CandidatePort $Port
    $runStem = 'r{0:d2}-o{1}-{2}' -f $Repetition, $OrderIndex, $Mode.label
    $stdout = Join-Path $output "$runStem.stdout.log"
    $stderr = Join-Path $output "$runStem.stderr.log"
    $report = Join-Path $output "$runStem.json"
    $process = $null

    $admission = Wait-NvidiaGpuIdleAdmission
    $script:gpuAdmissions += [ordered]@{
        run = $runStem
        admission = $admission
    }
    $environment.gpu_admissions = @($script:gpuAdmissions)
    $environment | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $environmentPath -Encoding utf8

    if ($ReuseCompatible -and (Test-Path -LiteralPath $report -PathType Leaf)) {
        try {
            $existing = Get-ReportMetadata -Path $report
            $identityMatches =
                $existing.schema -eq 'a3s.power.quality-eval.report.v3' -and
                $existing.mode_label -eq $Mode.label -and
                [int]$existing.repetition -eq $Repetition -and
                [int]$existing.order_index -eq $OrderIndex -and
                $existing.model -eq $Model -and
                $existing.model_sha256 -eq $Mode.model_hash -and
                $existing.tasks_sha256 -eq $taskDigest -and
                $existing.server_sha256 -eq $ServerHash -and
                ($existing.power_commit -eq $PowerCommit -or
                    $ReuseCompatibleReportsAcrossCommits) -and
                [int]$existing.seed -eq 42 -and
                [int]$existing.num_ctx -eq $NumCtx -and
                [int]$existing.num_batch -eq $NumBatch -and
                (($MaxTokensCap -eq 0 -and $null -eq $existing.max_tokens_cap) -or
                    [int]$existing.max_tokens_cap -eq $MaxTokensCap) -and
                (($MaxTokensOverride -eq 0 -and
                        $null -eq $existing.max_tokens_override) -or
                    [int]$existing.max_tokens_override -eq $MaxTokensOverride) -and
                [int]$existing.warmup_requests -eq 1
            $isComplete =
                [int]$existing.result_count -eq $expectedTaskCount -and
                $existing.completed_at -and
                [int]$existing.completed -eq $expectedTaskCount -and
                [int]$existing.errors -eq 0
            $runtimeMatches = if ($Mode.spec_mode -eq 'off') {
                -not [bool]$existing.has_speculative_runtime
            } else {
                [bool]$existing.has_speculative_runtime -and
                    $existing.speculative_strategy -eq $Mode.spec_mode
            }
            if ($identityMatches -and $isComplete -and $runtimeMatches) {
                Write-Output "Reusing complete report: $report"
                return
            }
        } catch {
        }
    }
    foreach ($staleRunFile in @($report, $stdout, $stderr)) {
        Remove-Item -LiteralPath $staleRunFile -Force -ErrorAction SilentlyContinue
    }

    $env:A3S_POWER_HOME = $Mode.power_home
    $env:A3S_POWER_DATA_DIR = $Mode.power_home
    $env:A3S_POWER_SPEC_MODE = $Mode.spec_mode
    if ($null -eq $Mode.fr_vocab_size) {
        Remove-Item Env:A3S_POWER_SPEC_MTP_FR_VOCAB_SIZE -ErrorAction SilentlyContinue
    } else {
        $env:A3S_POWER_SPEC_MTP_FR_VOCAB_SIZE = [string]$Mode.fr_vocab_size
    }
    $env:RUST_LOG = 'a3s_power::backend::llamacpp::speculative_runtime=info,a3s_power=info'

    try {
        $process = Start-Process -FilePath $server `
            -ArgumentList @(
                'serve',
                '--config', $config,
                '--host', '127.0.0.1',
                '--port', [string]$Port
            ) `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -WindowStyle Hidden `
            -PassThru
        $process.PriorityClass = $ProcessPriority
        if ($ProcessorAffinityMask -gt 0) {
            $process.ProcessorAffinity = [IntPtr]::new([int64]$ProcessorAffinityMask)
            $effectiveAffinity = [uint64]$process.ProcessorAffinity.ToInt64()
            if ($effectiveAffinity -ne $ProcessorAffinityMask) {
                throw ('Requested processor affinity 0x{0:x} became 0x{1:x}' -f `
                    $ProcessorAffinityMask, $effectiveAffinity)
            }
        }

        $ready = $false
        for ($attempt = 0; $attempt -lt 240; $attempt++) {
            if ($process.HasExited) {
                $message = Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue
                throw "Server exited before becoming ready: $message"
            }
            try {
                $probe = Invoke-WebRequest -UseBasicParsing `
                    -Uri "http://127.0.0.1:$Port/v1/models/$Model" `
                    -TimeoutSec 2
                if ($probe.StatusCode -eq 200) {
                    $ready = $true
                    break
                }
            } catch {
            }
            Start-Sleep -Milliseconds 500
        }
        if (-not $ready) {
            throw "Server did not expose model $Model"
        }

        $arguments = @(
            $evaluator,
            'run',
            '--url', "http://127.0.0.1:$Port",
            '--model', $Model,
            '--mode-label', $Mode.label,
            '--repetition', [string]$Repetition,
            '--order-index', [string]$OrderIndex,
            '--model-sha256', $Mode.model_hash,
            '--server-sha256', $ServerHash,
            '--power-commit', $PowerCommit,
            '--tasks', $tasks,
            '--manifest', $manifest,
            '--output', $report,
            '--server-log', $stdout,
            '--warmup-requests', '1',
            '--num-ctx', [string]$NumCtx,
            '--num-batch', [string]$NumBatch,
            '--seed', '42',
            '--timeout-seconds', '900'
        )
        if ($MaxTokensCap -gt 0) {
            $arguments += @('--max-tokens-cap', [string]$MaxTokensCap)
        }
        if ($MaxTokensOverride -gt 0) {
            $arguments += @('--max-tokens-override', [string]$MaxTokensOverride)
        }
        if ($taskSelectionPath) {
            $arguments += @('--task-selection', $taskSelectionPath)
        }
        if ($IncludeContent) {
            $arguments += '--include-content'
        }
        Invoke-Python -Arguments $arguments
        $completed = Get-ReportMetadata -Path $report
        if ([int]$completed.result_count -ne $expectedTaskCount -or
            [int]$completed.completed -ne $expectedTaskCount -or
            [int]$completed.errors -ne 0) {
            throw "Mode $($Mode.label) repetition $Repetition did not complete $expectedTaskCount error-free requests"
        }
    } finally {
        if ($process -and -not $process.HasExited) {
            $process.Kill()
            $process.WaitForExit()
        }
    }
}

$requiredPaths = @($server, $evaluator, $reporter, $manifest, $config)
if ($taskSelectionPath) {
    $requiredPaths += $taskSelectionPath
}
foreach ($requiredPath in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required quality benchmark input does not exist: $requiredPath"
    }
}
if (-not (Get-Command $PythonLauncher -ErrorAction SilentlyContinue)) {
    throw "Python launcher is not available: $PythonLauncher"
}
$taskManifest = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json

$gitStatus = @(& git -C $powerRoot status --porcelain=v1)
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to inspect the Power git worktree'
}
if ($RequireCleanTree -and $gitStatus.Count -ne 0) {
    throw 'The Power worktree must be clean for this capture'
}
$powerCommit = (& git -C $powerRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to resolve the Power commit'
}

$activePowerScheme = (& powercfg.exe /getactivescheme) -join [Environment]::NewLine
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to inspect the active Windows power scheme'
}
if ($RequireHighPerformancePowerPlan -and
    $activePowerScheme -notmatch '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c') {
    throw 'The Windows High performance power plan is required for this capture'
}

New-Item -ItemType Directory -Force -Path $output | Out-Null
$q6Identity = if ($profileDefinition.q6_external_draft_kind -eq 'dspark') {
    Get-ModelIdentity -PowerHome $Q6PowerHome -ExpectedHash $Q6ModelHash `
        -ExpectedExternalDraftKind 'dspark' `
        -ExpectedExternalDraftHash $DsparkDraftHash
} else {
    Get-ModelIdentity -PowerHome $Q6PowerHome -ExpectedHash $Q6ModelHash
}
$tbq4Identity = $null
if ($profileDefinition.requires_tbq4) {
    if ([string]::IsNullOrWhiteSpace($Tbq4PowerHome)) {
        throw "Tbq4PowerHome is required for profile '$Profile'"
    }
    $tbq4Identity = Get-ModelIdentity `
        -PowerHome $Tbq4PowerHome -ExpectedHash $Tbq4ModelHash
}
$modes = @($profileDefinition.modes | ForEach-Object {
    $modelIdentity = if ($_.model_role -eq 'q6') {
        $q6Identity
    } else {
        $tbq4Identity
    }
    $externalDraftHash = if ($_.external_draft_kind -eq 'dspark') {
        $DsparkDraftHash
    } else {
        $null
    }
    @{
        label = $_.label
        power_home = $modelIdentity.power_home
        model_hash = $modelIdentity.sha256
        spec_mode = $_.spec_mode
        fr_vocab_size = $_.fr_vocab_size
        external_draft_sha256 = $externalDraftHash
    }
})
$comparisons = @($profileDefinition.comparisons)
$serverHash = (Get-FileHash -LiteralPath $server -Algorithm SHA256).Hash.ToLowerInvariant()
$gpu = @(& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,pstate,power.limit --format=csv,noheader,nounits)
$cpu = Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
$os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,TotalVisibleMemorySize

$taskManifestHash = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
$configHash = (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash.ToLowerInvariant()
$environment = [ordered]@{
    schema = 'a3s.power.quality-eval.environment.v1'
    created_at = [DateTimeOffset]::UtcNow.ToString('o')
    power_commit = $powerCommit
    dirty_worktree = $gitStatus.Count -ne 0
    git_status = $gitStatus
    server = [ordered]@{
        path = [System.IO.Path]::GetFullPath($server)
        sha256 = $serverHash
    }
    benchmark_tools = [ordered]@{
        runner_sha256 = (Get-FileHash -LiteralPath $PSCommandPath `
            -Algorithm SHA256).Hash.ToLowerInvariant()
        evaluator_sha256 = (Get-FileHash -LiteralPath $evaluator `
            -Algorithm SHA256).Hash.ToLowerInvariant()
        reporter_sha256 = (Get-FileHash -LiteralPath $reporter `
            -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    q6_model = $q6Identity
    tbq4_model = $tbq4Identity
    gpu = $gpu
    cpu = $cpu
    os = $os
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
    host_controls = [ordered]@{
        cuda_high_priority = [bool]$CudaHighPriority
        gpu_clock = [ordered]@{
            gpu_index = $NvidiaGpuIndex
            requested_mhz = if ($LockGpuClockMHz -gt 0) {
                $LockGpuClockMHz
            } else {
                $null
            }
            lock_applied = $false
        }
    }
    gpu_admission = [ordered]@{
        gpu_index = $NvidiaGpuIndex
        maximum_idle_utilization_percent = $MaximumIdleGpuUtilizationPercent
        minimum_idle_memory_free_mib = $MinimumIdleGpuMemoryFreeMiB
        consecutive_sample_count = $IdleGpuSampleCount
        sample_interval_milliseconds = $IdleGpuSampleIntervalMilliseconds
        wait_seconds = $IdleGpuWaitSeconds
    }
    gpu_admissions = @()
    num_ctx = $NumCtx
    num_batch = $NumBatch
    max_tokens_cap = if ($MaxTokensCap -gt 0) { $MaxTokensCap } else { $null }
    max_tokens_override = if ($MaxTokensOverride -gt 0) {
        $MaxTokensOverride
    } else {
        $null
    }
    task_selection = if ($taskSelectionPath) {
        [ordered]@{
            path = $taskSelectionPath
            sha256 = (Get-FileHash -LiteralPath $taskSelectionPath `
                -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    } else {
        $null
    }
    repetitions = $Repetitions
    profile = $Profile
    compatible_report_commit_reuse = [bool]$ReuseCompatibleReportsAcrossCommits
    modes = @($modes | ForEach-Object {
        [ordered]@{
            label = $_.label
            model_sha256 = $_.model_hash.ToLowerInvariant()
            spec_mode = $_.spec_mode
            fr_vocab_size = $_.fr_vocab_size
            external_draft_sha256 = $_.external_draft_sha256
        }
    })
    order = "$($modes.Count)-mode cyclic Latin rotation"
    task_manifest_sha256 = $taskManifestHash
    config_sha256 = $configHash
}
$reuseCompatible = $false
if (Test-Path -LiteralPath $environmentPath -PathType Leaf) {
    try {
        $previous = Get-Content -LiteralPath $environmentPath -Raw | ConvertFrom-Json
        $reuseCompatible =
            $previous.schema -eq $environment.schema -and
            ($previous.power_commit -eq $environment.power_commit -or
                ($ReuseCompatibleReportsAcrossCommits -and
                    -not [bool]$previous.dirty_worktree -and
                    -not [bool]$environment.dirty_worktree)) -and
            $previous.server.sha256 -eq $environment.server.sha256 -and
            $previous.benchmark_tools.runner_sha256 -eq
                $environment.benchmark_tools.runner_sha256 -and
            $previous.benchmark_tools.evaluator_sha256 -eq
                $environment.benchmark_tools.evaluator_sha256 -and
            $previous.benchmark_tools.reporter_sha256 -eq
                $environment.benchmark_tools.reporter_sha256 -and
            $previous.q6_model.sha256 -eq $environment.q6_model.sha256 -and
            $previous.q6_model.external_draft.sha256 -eq
                $environment.q6_model.external_draft.sha256 -and
            $previous.tbq4_model.sha256 -eq $environment.tbq4_model.sha256 -and
            $previous.active_power_scheme -eq $environment.active_power_scheme -and
            $previous.process_priority -eq $environment.process_priority -and
            $previous.process_affinity.requested_mask -eq
                $environment.process_affinity.requested_mask -and
            [bool]$previous.host_controls.cuda_high_priority -eq
                $environment.host_controls.cuda_high_priority -and
            $previous.host_controls.gpu_clock.gpu_index -eq
                $environment.host_controls.gpu_clock.gpu_index -and
            $previous.host_controls.gpu_clock.requested_mhz -eq
                $environment.host_controls.gpu_clock.requested_mhz -and
            [int]$previous.gpu_admission.gpu_index -eq
                $environment.gpu_admission.gpu_index -and
            [int]$previous.gpu_admission.maximum_idle_utilization_percent -eq
                $environment.gpu_admission.maximum_idle_utilization_percent -and
            [int]$previous.gpu_admission.minimum_idle_memory_free_mib -eq
                $environment.gpu_admission.minimum_idle_memory_free_mib -and
            [int]$previous.gpu_admission.consecutive_sample_count -eq
                $environment.gpu_admission.consecutive_sample_count -and
            [int]$previous.gpu_admission.sample_interval_milliseconds -eq
                $environment.gpu_admission.sample_interval_milliseconds -and
            [int]$previous.gpu_admission.wait_seconds -eq
                $environment.gpu_admission.wait_seconds -and
            [int]$previous.num_ctx -eq $environment.num_ctx -and
            [int]$previous.num_batch -eq $environment.num_batch -and
            (($null -eq $previous.max_tokens_cap -and
                    $null -eq $environment.max_tokens_cap) -or
                [int]$previous.max_tokens_cap -eq
                    [int]$environment.max_tokens_cap) -and
            (($null -eq $previous.max_tokens_override -and
                    $null -eq $environment.max_tokens_override) -or
                [int]$previous.max_tokens_override -eq
                    [int]$environment.max_tokens_override) -and
            $previous.task_selection.sha256 -eq
                $environment.task_selection.sha256 -and
            [int]$previous.repetitions -eq $environment.repetitions -and
            $previous.profile -eq $environment.profile -and
            (@($previous.modes.label) -join ',') -eq
                (@($environment.modes.label) -join ',') -and
            $previous.task_manifest_sha256 -eq $environment.task_manifest_sha256 -and
            $previous.config_sha256 -eq $environment.config_sha256 -and
            (Get-GpuCompatibilityIdentity $previous.gpu) -eq
                (Get-GpuCompatibilityIdentity $environment.gpu)
    } catch {
        $reuseCompatible = $false
    }
}
Write-Output "Existing report environment compatible: $reuseCompatible"
$gpuAdmissions = @()
$environment | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $environmentPath -Encoding utf8

if ($PreparedTaskCache) {
    if (-not (Test-Path -LiteralPath $PreparedTaskCache -PathType Leaf)) {
        throw "Prepared task cache does not exist: $PreparedTaskCache"
    }
    $preparedPath = [System.IO.Path]::GetFullPath($PreparedTaskCache)
    if ($preparedPath -ne [System.IO.Path]::GetFullPath($tasks)) {
        Copy-Item -LiteralPath $preparedPath -Destination $tasks -Force
    }
}

Invoke-Python -Arguments @(
    $evaluator,
    'prepare',
    '--manifest', $manifest,
    '--output', $tasks
)
$taskPayload = Get-Content -LiteralPath $tasks -Raw | ConvertFrom-Json
$taskDigest = $taskPayload.tasks_sha256
if ($taskDigest -ne $taskManifest.expected_tasks_sha256) {
    throw 'Prepared task digest differs from the reviewed manifest'
}
$expectedTaskCount = @($taskPayload.tasks).Count
if ($expectedTaskCount -le 0) {
    throw 'Prepared task cache contains no tasks'
}
if ($taskSelectionPath) {
    $selectionJson = @(& $PythonLauncher @pythonPrefix $evaluator `
        'inspect-selection' `
        '--tasks' $tasks `
        '--manifest' $manifest `
        '--task-selection' $taskSelectionPath) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0) {
        throw "Python task-selection inspection exited with code $LASTEXITCODE"
    }
    $selectionMetadata = $selectionJson | ConvertFrom-Json
    if ($selectionMetadata.schema -ne
        'a3s.power.quality-eval.selection-inspection.v1') {
        throw 'Python task-selection inspection returned an unsupported schema'
    }
    $taskDigest = [string]$selectionMetadata.tasks_sha256
    $expectedTaskCount = [int]$selectionMetadata.task_count
    if ($expectedTaskCount -le 0) {
        throw 'Task selection contains no tasks'
    }
}

$environmentNames = @(
    'A3S_POWER_HOME',
    'A3S_POWER_DATA_DIR',
    'A3S_POWER_SPEC_MODE',
    'A3S_POWER_SPEC_MTP_FR_VOCAB_SIZE',
    'GGML_CUDA_HIGH_PRIORITY',
    'RUST_LOG'
)
$savedEnvironment = @{}
foreach ($name in $environmentNames) {
    $item = Get-Item "Env:$name" -ErrorAction SilentlyContinue
    $savedEnvironment[$name] = if ($item) { $item.Value } else { $null }
}

$gpuClockLocked = $false
try {
    if ($CudaHighPriority) {
        $env:GGML_CUDA_HIGH_PRIORITY = '1'
    } else {
        Remove-Item Env:GGML_CUDA_HIGH_PRIORITY -ErrorAction SilentlyContinue
    }
    if ($LockGpuClockMHz -gt 0) {
        $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
        if (-not $nvidiaSmi) {
            throw 'nvidia-smi.exe is required when LockGpuClockMHz is configured'
        }
        & $nvidiaSmi.Source `
            "--id=$NvidiaGpuIndex" `
            --lock-gpu-clocks="$LockGpuClockMHz,$LockGpuClockMHz" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to lock NVIDIA GPU $NvidiaGpuIndex at $LockGpuClockMHz MHz"
        }
        $gpuClockLocked = $true
        $environment.host_controls.gpu_clock.lock_applied = $true
    }
    $environment | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $environmentPath -Encoding utf8

    for ($repetition = 1; $repetition -le $Repetitions; $repetition++) {
        for ($position = 0; $position -lt $modes.Count; $position++) {
            $mode = $modes[($position + $repetition - 1) % $modes.Count]
            Write-Output "Running repetition $repetition/$Repetitions, order $($position + 1)/$($modes.Count): $($mode.label)"
            Invoke-ModeRun `
                -Mode $mode `
                -Repetition $repetition `
                -OrderIndex ($position + 1) `
                -ServerHash $serverHash `
                -PowerCommit $powerCommit `
                -ReuseCompatible $reuseCompatible
        }
    }
} finally {
    if ($gpuClockLocked) {
        & nvidia-smi.exe `
            "--id=$NvidiaGpuIndex" --reset-gpu-clocks | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Failed to reset NVIDIA GPU $NvidiaGpuIndex graphics clock"
        }
    }
    foreach ($name in $environmentNames) {
        if ($null -eq $savedEnvironment[$name]) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$name" $savedEnvironment[$name]
        }
    }
}

$reports = @(
    Get-ChildItem -LiteralPath $output -Filter 'r??-o?-*.json' -File |
        Sort-Object Name |
        ForEach-Object { $_.FullName }
)
$expectedReports = $Repetitions * $modes.Count
if ($reports.Count -ne $expectedReports) {
    throw "Expected $expectedReports reports, found $($reports.Count)"
}
$aggregateArguments = @($evaluator, 'aggregate', '--reports') + $reports + @(
    '--output-json', $aggregateJson,
    '--output-markdown', $aggregateMarkdown
)
foreach ($comparison in $comparisons) {
    $aggregateArguments += @('--pair', $comparison[0], $comparison[1])
}
Invoke-Python -Arguments $aggregateArguments
Write-Output "Quality matrix complete: $aggregateJson"
