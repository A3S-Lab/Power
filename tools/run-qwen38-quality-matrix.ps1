param(
    [Parameter(Mandatory = $true)]
    [string]$Q6PowerHome,

    [Parameter(Mandatory = $true)]
    [string]$Tbq4PowerHome,

    [ValidateSet('prefix-fr-release', 'full-vocabulary-current')]
    [string]$Profile = 'prefix-fr-release',

    [string]$RuntimeConfig,

    [ValidateRange(1, 20)]
    [int]$Repetitions = 3,

    [ValidateRange(1, 4096)]
    [int]$NumBatch = 14,

    [ValidateRange(1, 65535)]
    [int]$Port = 11436,

    [ValidateSet('Normal', 'AboveNormal', 'High')]
    [string]$ProcessPriority = 'High',

    [UInt64]$ProcessorAffinityMask = 0,

    [string]$TargetDirectory = 'target-native-sm89-ninja',

    [string]$OutputRoot = 'target-qwen38-quality',

    [string]$Model = 'qwen3.8-27b-q6-k',

    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$Q6ModelHash = '562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727',

    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$Tbq4ModelHash = '5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6',

    [string]$PythonLauncher = 'py',

    [string]$PythonVersion = '3.13',

    [string]$PreparedTaskCache,

    [bool]$VerifyModelFiles = $true,

    [switch]$RequireCleanTree,

    [switch]$RequireHighPerformancePowerPlan,

    [switch]$IncludeContent
)

$ErrorActionPreference = 'Stop'

$powerRoot = Split-Path -Parent $PSScriptRoot
$server = Join-Path $powerRoot "$TargetDirectory\release\a3s-power.exe"
$evaluator = Join-Path $PSScriptRoot 'qwen38_quality_eval.py'
$qwenBenchmarkRoot = Join-Path $powerRoot 'docs\benchmarks\qwen3.8-27b-q6k-rtx4090'
$benchmarkRoot = Join-Path $qwenBenchmarkRoot 'quality'
$manifest = Join-Path $benchmarkRoot 'tasks-v1.manifest.json'
$config = if (-not [string]::IsNullOrWhiteSpace($RuntimeConfig)) {
    [System.IO.Path]::GetFullPath($RuntimeConfig)
} elseif ($Profile -eq 'full-vocabulary-current') {
    Join-Path $benchmarkRoot 'full-vocabulary-current.acl'
} else {
    Join-Path $benchmarkRoot 'matrix.acl'
}
$output = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    [System.IO.Path]::GetFullPath($OutputRoot)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $powerRoot $OutputRoot))
}
$tasks = Join-Path $output 'tasks-v1.json'
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
        [string]$ExpectedHash
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
    [ordered]@{
        power_home = [System.IO.Path]::GetFullPath($PowerHome)
        manifest = [System.IO.Path]::GetFullPath($manifestPath)
        path = [System.IO.Path]::GetFullPath($modelManifest.path)
        size = [int64]$modelManifest.size
        sha256 = $modelManifest.sha256
        file_hash_verified = $VerifyModelFiles
    }
}

function Invoke-Python {
    param([string[]]$Arguments)

    & $PythonLauncher @pythonPrefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python evaluator exited with code $LASTEXITCODE"
    }
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

    if ($ReuseCompatible -and (Test-Path -LiteralPath $report -PathType Leaf)) {
        try {
            $existing = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
            $identityMatches =
                $existing.schema -eq 'a3s.power.quality-eval.report.v3' -and
                $existing.mode_label -eq $Mode.label -and
                [int]$existing.repetition -eq $Repetition -and
                [int]$existing.order_index -eq $OrderIndex -and
                $existing.model -eq $Model -and
                $existing.model_sha256 -eq $Mode.model_hash -and
                $existing.tasks_sha256 -eq $taskDigest -and
                $existing.server_sha256 -eq $ServerHash -and
                $existing.power_commit -eq $PowerCommit -and
                [int]$existing.seed -eq 42 -and
                [int]$existing.request.num_ctx -eq 4096 -and
                [int]$existing.request.num_batch -eq $NumBatch -and
                [int]$existing.request.warmup_requests -eq 1
            $isComplete =
                $existing.results.Count -eq $expectedTaskCount -and
                $existing.completed_at -and
                [int]$existing.summary.overall.completed -eq $expectedTaskCount -and
                [int]$existing.summary.overall.errors -eq 0
            $runtimeMatches =
                ($Mode.spec_mode -ne 'mtp') -or
                ($null -ne $existing.speculative_runtime)
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
            '--num-batch', [string]$NumBatch,
            '--seed', '42',
            '--timeout-seconds', '900'
        )
        if ($IncludeContent) {
            $arguments += '--include-content'
        }
        Invoke-Python -Arguments $arguments
        $completed = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
        if ($completed.results.Count -ne $expectedTaskCount -or
            [int]$completed.summary.overall.completed -ne $expectedTaskCount -or
            [int]$completed.summary.overall.errors -ne 0) {
            throw "Mode $($Mode.label) repetition $Repetition did not complete $expectedTaskCount error-free requests"
        }
    } finally {
        if ($process -and -not $process.HasExited) {
            $process.Kill()
            $process.WaitForExit()
        }
    }
}

foreach ($requiredPath in @($server, $evaluator, $manifest, $config)) {
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
$q6Identity = Get-ModelIdentity -PowerHome $Q6PowerHome -ExpectedHash $Q6ModelHash
$tbq4Identity = Get-ModelIdentity -PowerHome $Tbq4PowerHome -ExpectedHash $Tbq4ModelHash
$mtpLabel = if ($Profile -eq 'full-vocabulary-current') {
    'tbq4-mtp-full-vocab'
} else {
    'tbq4-mtp-fr'
}
$mtpFrVocabSize = if ($Profile -eq 'full-vocabulary-current') { $null } else { 8192 }
$modes = @(
    @{
        label = 'q6-off'
        power_home = $q6Identity.power_home
        model_hash = $Q6ModelHash
        spec_mode = 'off'
        fr_vocab_size = $null
    },
    @{
        label = 'tbq4-off'
        power_home = $tbq4Identity.power_home
        model_hash = $Tbq4ModelHash
        spec_mode = 'off'
        fr_vocab_size = $null
    },
    @{
        label = $mtpLabel
        power_home = $tbq4Identity.power_home
        model_hash = $Tbq4ModelHash
        spec_mode = 'mtp'
        fr_vocab_size = $mtpFrVocabSize
    }
)
$comparisons = @(
    @('q6-off', 'tbq4-off'),
    @('tbq4-off', $modes[2].label)
)
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
    num_batch = $NumBatch
    repetitions = $Repetitions
    profile = $Profile
    modes = @($modes | ForEach-Object {
        [ordered]@{
            label = $_.label
            model_sha256 = $_.model_hash.ToLowerInvariant()
            spec_mode = $_.spec_mode
            fr_vocab_size = $_.fr_vocab_size
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
            $previous.power_commit -eq $environment.power_commit -and
            $previous.server.sha256 -eq $environment.server.sha256 -and
            $previous.q6_model.sha256 -eq $environment.q6_model.sha256 -and
            $previous.tbq4_model.sha256 -eq $environment.tbq4_model.sha256 -and
            $previous.active_power_scheme -eq $environment.active_power_scheme -and
            $previous.process_priority -eq $environment.process_priority -and
            $previous.process_affinity.requested_mask -eq
                $environment.process_affinity.requested_mask -and
            [int]$previous.num_batch -eq $environment.num_batch -and
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

$environmentNames = @(
    'A3S_POWER_HOME',
    'A3S_POWER_DATA_DIR',
    'A3S_POWER_SPEC_MODE',
    'A3S_POWER_SPEC_MTP_FR_VOCAB_SIZE',
    'RUST_LOG'
)
$savedEnvironment = @{}
foreach ($name in $environmentNames) {
    $item = Get-Item "Env:$name" -ErrorAction SilentlyContinue
    $savedEnvironment[$name] = if ($item) { $item.Value } else { $null }
}

try {
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
