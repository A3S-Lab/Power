[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PowerHome,

    [int[]]$FrVocabSizes = @(0, 8192, 16384, 32768, 65536),

    [int[]]$DraftMaxValues = @(7),

    [ValidateRange(1, 64)]
    [int]$MtpRecurrentSnapshots = 6,

    [bool]$MtpRecurrentChain = $true,

    [int[]]$NumBatchValues = @(14),

    [ValidateSet('fixed', 'adaptive')]
    [string[]]$Policies = @('fixed'),

    [switch]$IncludeOffBaseline,

    [ValidateRange(1, 10)]
    [int]$Repetitions = 1,

    [ValidateRange(1, 4096)]
    [int]$MaxTokensCap = 128,

    [ValidateRange(1, 65535)]
    [int]$Port = 11437,

    [ValidateSet('Normal', 'AboveNormal', 'High')]
    [string]$ProcessPriority = 'High',

    [string]$TargetDirectory = 'target-native-sm89-ninja',

    [string]$OutputRoot = 'target-qwen38-mtp-sweep',

    [string]$Model = 'qwen3.8-27b-q6-k',

    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ModelHash = '5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6',

    [string]$PythonLauncher = 'py',

    [string]$PythonVersion = '3.13',

    [bool]$VerifyModelFile = $true,

    [switch]$RequireHighPerformancePowerPlan,

    [switch]$IncludeContent
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$powerRoot = Split-Path -Parent $PSScriptRoot
$server = Join-Path $powerRoot "$TargetDirectory\release\a3s-power.exe"
$evaluator = Join-Path $PSScriptRoot 'qwen38_quality_eval.py'
$benchmarkRoot = Join-Path $powerRoot 'docs\benchmarks\qwen3.8-27b-q6k-rtx4090\quality'
$config = Join-Path $benchmarkRoot 'matrix.acl'
$tasks = Join-Path $benchmarkRoot 'tasks-v1.json'
$manifest = Join-Path $benchmarkRoot 'tasks-v1.manifest.json'
$selection = Join-Path $benchmarkRoot 'calibration-v1.selection.json'
$output = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    [System.IO.Path]::GetFullPath($OutputRoot)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $powerRoot $OutputRoot))
}
$environmentPath = Join-Path $output 'environment.json'
$aggregateJson = Join-Path $output 'sweep.json'
$aggregateMarkdown = Join-Path $output 'sweep.md'
$pythonPrefix = @("-$PythonVersion")

function Invoke-Python {
    param([string[]]$Arguments)

    & $PythonLauncher @pythonPrefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python evaluator exited with code $LASTEXITCODE"
    }
}

function Assert-PortAvailable {
    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $Port
    )
    try {
        $listener.Start()
    } catch {
        throw "Port $Port is already in use"
    } finally {
        $listener.Stop()
    }
}

foreach ($requiredPath in @($server, $evaluator, $config, $tasks, $manifest, $selection)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required sweep input does not exist: $requiredPath"
    }
}
if (-not (Get-Command $PythonLauncher -ErrorAction SilentlyContinue)) {
    throw "Python launcher is unavailable: $PythonLauncher"
}
foreach ($frVocabSize in $FrVocabSizes) {
    if ($frVocabSize -ne 0 -and ($frVocabSize -lt 1024 -or $frVocabSize -gt 1048576)) {
        throw "FR vocabulary sizes must be zero or between 1024 and 1048576"
    }
}
foreach ($draftMax in $DraftMaxValues) {
    if ($draftMax -lt 1 -or $draftMax -gt 64) {
        throw "Draft widths must be between 1 and 64"
    }
    if ($MtpRecurrentSnapshots -gt $draftMax) {
        throw "MTP recurrent snapshots $MtpRecurrentSnapshots cannot exceed draft width $draftMax"
    }
    foreach ($numBatch in $NumBatchValues) {
        if ($numBatch -lt $draftMax + 2) {
            throw "num_batch $numBatch cannot execute draft width $draftMax; require at least $($draftMax + 2)"
        }
    }
}

$modelManifestPath = Join-Path $PowerHome "models\manifests\$Model.json"
if (-not (Test-Path -LiteralPath $modelManifestPath -PathType Leaf)) {
    throw "Model manifest does not exist: $modelManifestPath"
}
$modelManifest = Get-Content -LiteralPath $modelManifestPath -Raw | ConvertFrom-Json
if ($modelManifest.sha256 -ne $ModelHash) {
    throw "Model manifest hash differs from -ModelHash"
}
if (-not (Test-Path -LiteralPath $modelManifest.path -PathType Leaf)) {
    throw "GGUF does not exist: $($modelManifest.path)"
}
$modelFile = Get-Item -LiteralPath $modelManifest.path
if ($modelFile.Length -ne $modelManifest.size) {
    throw "GGUF byte length differs from its manifest"
}
if ($VerifyModelFile) {
    $actualModelHash = (Get-FileHash -LiteralPath $modelManifest.path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualModelHash -ne $ModelHash) {
        throw "GGUF SHA-256 differs from -ModelHash"
    }
}

if ($RequireHighPerformancePowerPlan) {
    $activePowerScheme = (& powercfg.exe /getactivescheme) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0 -or
        $activePowerScheme -notmatch '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c') {
        throw 'The Windows High performance power plan is required'
    }
} else {
    $activePowerScheme = (& powercfg.exe /getactivescheme) -join [Environment]::NewLine
}

$modes = @()
if ($IncludeOffBaseline) {
    foreach ($numBatch in $NumBatchValues) {
        $modes += [pscustomobject]@{
            label = "off-b$numBatch"
            spec_mode = 'off'
            fr_vocab_size = 0
            draft_max = $DraftMaxValues[0]
            recurrent_snapshots = $MtpRecurrentSnapshots
            recurrent_chain = $MtpRecurrentChain
            num_batch = $numBatch
            adaptive = $false
        }
    }
}
foreach ($frVocabSize in $FrVocabSizes) {
    foreach ($draftMax in $DraftMaxValues) {
        foreach ($numBatch in $NumBatchValues) {
            foreach ($policy in $Policies) {
                $frLabel = if ($frVocabSize -eq 0) { 'full' } else { [string]$frVocabSize }
                $modes += [pscustomobject]@{
                    label = "fr$frLabel-k$draftMax-s$MtpRecurrentSnapshots-b$numBatch-$policy"
                    spec_mode = 'mtp'
                    fr_vocab_size = $frVocabSize
                    draft_max = $draftMax
                    recurrent_snapshots = $MtpRecurrentSnapshots
                    recurrent_chain = $MtpRecurrentChain
                    num_batch = $numBatch
                    adaptive = $policy -eq 'adaptive'
                }
            }
        }
    }
}
if ($modes.Count -eq 0) {
    throw 'The sweep grid is empty'
}

New-Item -ItemType Directory -Force -Path $output | Out-Null
$serverHash = (Get-FileHash -LiteralPath $server -Algorithm SHA256).Hash.ToLowerInvariant()
$powerCommit = (& git -C $powerRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to resolve the Power commit'
}
$gitStatus = @(& git -C $powerRoot status --porcelain=v1)
$gpu = @(& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits)
$selectionPayload = Get-Content -LiteralPath $selection -Raw | ConvertFrom-Json
$taskCount = $selectionPayload.task_ids.Count
$environmentIdentity = [ordered]@{
    power_commit = $powerCommit
    server_sha256 = $serverHash
    model_sha256 = $ModelHash
    model_bytes = [int64]$modelFile.Length
    gpu = $gpu
    active_power_scheme = $activePowerScheme.Trim()
    process_priority = $ProcessPriority
    repetitions = $Repetitions
    max_tokens_cap = $MaxTokensCap
    config_sha256 = (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash.ToLowerInvariant()
    tasks_sha256 = (Get-FileHash -LiteralPath $tasks -Algorithm SHA256).Hash.ToLowerInvariant()
    selection_sha256 = (Get-FileHash -LiteralPath $selection -Algorithm SHA256).Hash.ToLowerInvariant()
    modes = $modes
}
$environment = [ordered]@{
    schema = 'a3s.power.mtp-sweep.environment.v1'
    created_at = [DateTimeOffset]::UtcNow.ToString('o')
    identity = $environmentIdentity
    dirty_worktree = $gitStatus.Count -ne 0
    git_status = $gitStatus
    model_path = [System.IO.Path]::GetFullPath($modelManifest.path)
    model_file_hash_verified = $VerifyModelFile
}
$reuseCompatible = $false
if (Test-Path -LiteralPath $environmentPath -PathType Leaf) {
    try {
        $previous = Get-Content -LiteralPath $environmentPath -Raw | ConvertFrom-Json
        $previousIdentity = $previous.identity | ConvertTo-Json -Depth 12 -Compress
        $currentIdentity = $environmentIdentity | ConvertTo-Json -Depth 12 -Compress
        $reuseCompatible = $previous.schema -eq $environment.schema -and
            $previousIdentity -eq $currentIdentity
    } catch {
        $reuseCompatible = $false
    }
}
$environment | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $environmentPath -Encoding utf8
Write-Output "Existing sweep environment compatible: $reuseCompatible"

$environmentNames = @(
    'A3S_POWER_HOME',
    'A3S_POWER_DATA_DIR',
    'A3S_POWER_SPEC_MODE',
    'A3S_POWER_SPEC_DRAFT_MAX',
    'A3S_POWER_SPEC_MTP_RECURRENT_SNAPSHOTS',
    'A3S_POWER_SPEC_MTP_RECURRENT_CHAIN',
    'A3S_POWER_SPEC_MTP_ADAPTIVE',
    'A3S_POWER_SPEC_MTP_FR_VOCAB_SIZE',
    'RUST_LOG'
)
$savedEnvironment = @{}
foreach ($name in $environmentNames) {
    $item = Get-Item "Env:$name" -ErrorAction SilentlyContinue
    $savedEnvironment[$name] = if ($item) { $item.Value } else { $null }
}
$reportPaths = @()

try {
    for ($repetition = 1; $repetition -le $Repetitions; $repetition++) {
        for ($position = 0; $position -lt $modes.Count; $position++) {
            $mode = $modes[($position + $repetition - 1) % $modes.Count]
            $orderIndex = $position + 1
            $runStem = 'r{0:d2}-o{1:d2}-{2}' -f $repetition, $orderIndex, $mode.label
            $stdout = Join-Path $output "$runStem.stdout.log"
            $stderr = Join-Path $output "$runStem.stderr.log"
            $report = Join-Path $output "$runStem.json"
            $reportPaths += $report

            if ($reuseCompatible -and (Test-Path -LiteralPath $report -PathType Leaf)) {
                try {
                    $existing = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
                    $existingFr = $existing.health.speculative.mtp_fr_vocab_size
                    $expectedFr = if ($mode.fr_vocab_size -eq 0) { $null } else { $mode.fr_vocab_size }
                    $canReuse =
                        $existing.schema -eq 'a3s.power.quality-eval.report.v3' -and
                        $existing.mode_label -eq $mode.label -and
                        [int]$existing.repetition -eq $repetition -and
                        [int]$existing.order_index -eq $orderIndex -and
                        $existing.server_sha256 -eq $serverHash -and
                        $existing.health.speculative.mode -eq $mode.spec_mode -and
                        [int]$existing.request.num_batch -eq $mode.num_batch -and
                        [int]$existing.request.max_tokens_cap -eq $MaxTokensCap -and
                        ($mode.spec_mode -eq 'off' -or (
                            [int]$existing.health.speculative.draft_max -eq $mode.draft_max -and
                            [int]$existing.health.speculative.mtp_recurrent_snapshots -eq $mode.recurrent_snapshots -and
                            [bool]$existing.health.speculative.mtp_recurrent_chain -eq $mode.recurrent_chain -and
                            [bool]$existing.health.speculative.mtp_adaptive -eq $mode.adaptive
                        )) -and
                        $existingFr -eq $expectedFr -and
                        $existing.results.Count -eq $taskCount -and
                        [int]$existing.summary.overall.errors -eq 0 -and
                        ($mode.spec_mode -eq 'off' -or $null -ne $existing.speculative_runtime)
                    if ($canReuse) {
                        Write-Output "Reusing complete report: $report"
                        continue
                    }
                } catch {
                }
            }

            # A report can contain all task rows but still be incomplete when
            # the previous process stopped before runtime-log aggregation. The
            # evaluator intentionally resumes task rows, so starting it with a
            # fresh server log would otherwise leave only the warm-up MTP
            # record. Once reuse is rejected, restart this deterministic run
            # from a clean report and clean log pair.
            foreach ($staleRunFile in @($report, $stdout, $stderr)) {
                Remove-Item -LiteralPath $staleRunFile -Force -ErrorAction SilentlyContinue
            }

            Assert-PortAvailable
            $env:A3S_POWER_HOME = [System.IO.Path]::GetFullPath($PowerHome)
            $env:A3S_POWER_DATA_DIR = $env:A3S_POWER_HOME
            $env:A3S_POWER_SPEC_MODE = $mode.spec_mode
            $env:A3S_POWER_SPEC_DRAFT_MAX = [string]$mode.draft_max
            $env:A3S_POWER_SPEC_MTP_RECURRENT_SNAPSHOTS = [string]$mode.recurrent_snapshots
            $env:A3S_POWER_SPEC_MTP_RECURRENT_CHAIN = $mode.recurrent_chain.ToString().ToLowerInvariant()
            $env:A3S_POWER_SPEC_MTP_ADAPTIVE = $mode.adaptive.ToString().ToLowerInvariant()
            if ($mode.fr_vocab_size -eq 0) {
                Remove-Item Env:A3S_POWER_SPEC_MTP_FR_VOCAB_SIZE -ErrorAction SilentlyContinue
            } else {
                $env:A3S_POWER_SPEC_MTP_FR_VOCAB_SIZE = [string]$mode.fr_vocab_size
            }
            $env:RUST_LOG = 'a3s_power::backend::llamacpp::speculative_runtime=info,a3s_power=info'

            Write-Output "Running $($mode.label), repetition $repetition/$Repetitions"
            $process = $null
            try {
                $process = Start-Process -FilePath $server `
                    -ArgumentList @(
                        'serve', '--config', $config,
                        '--host', '127.0.0.1', '--port', [string]$Port
                    ) `
                    -RedirectStandardOutput $stdout `
                    -RedirectStandardError $stderr `
                    -WindowStyle Hidden `
                    -PassThru
                $process.PriorityClass = $ProcessPriority
                $ready = $false
                for ($attempt = 0; $attempt -lt 240; $attempt++) {
                    if ($process.HasExited) {
                        throw "Server exited before becoming ready: $(Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue)"
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
                    $evaluator, 'run',
                    '--url', "http://127.0.0.1:$Port",
                    '--model', $Model,
                    '--mode-label', $mode.label,
                    '--repetition', [string]$repetition,
                    '--order-index', [string]$orderIndex,
                    '--model-sha256', $ModelHash,
                    '--server-sha256', $serverHash,
                    '--power-commit', $powerCommit,
                    '--tasks', $tasks,
                    '--manifest', $manifest,
                    '--task-selection', $selection,
                    '--output', $report,
                    '--server-log', $stdout,
                    '--num-batch', [string]$mode.num_batch,
                    '--max-tokens-cap', [string]$MaxTokensCap,
                    '--warmup-requests', '1',
                    '--seed', '42',
                    '--timeout-seconds', '900'
                )
                if ($IncludeContent) {
                    $arguments += '--include-content'
                }
                Invoke-Python -Arguments $arguments
                $completed = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
                if ($completed.results.Count -ne $taskCount -or
                    [int]$completed.summary.overall.errors -ne 0) {
                    throw "Sweep mode $($mode.label) did not complete $taskCount error-free requests"
                }
            } finally {
                if ($process -and -not $process.HasExited) {
                    $process.Kill()
                    $process.WaitForExit()
                }
            }
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

$aggregateArguments = @(
    $evaluator,
    'aggregate-sweep',
    '--reports'
) + $reportPaths + @(
    '--output-json', $aggregateJson,
    '--output-markdown', $aggregateMarkdown
)
Invoke-Python -Arguments $aggregateArguments
Write-Output "MTP sweep complete: $aggregateJson"
