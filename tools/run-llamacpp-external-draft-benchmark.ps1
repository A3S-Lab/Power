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

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$LlamaCppRsCommit,

    [Parameter(Mandatory = $true)]
    [string]$PromptFile,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [ValidateSet('dflash', 'dspark')]
    [string]$DraftMode = 'dspark',

    [ValidateRange(1, 20)]
    [int]$Samples = 3,

    [ValidateRange(0, 10)]
    [int]$WarmupRuns = 1,

    [ValidateRange(2, 4096)]
    [int]$MaxTokens = 256,

    [ValidateRange(64, 1048576)]
    [int]$ContextSize = 512,

    [ValidateRange(8, 4096)]
    [int]$BatchSize = 512,

    [ValidateRange(1, 64)]
    [int]$Threads = 10,

    [ValidateRange(1, 64)]
    [int]$DraftMax = 4,

    [ValidatePattern('^(all|auto|[0-9]+)$')]
    [string]$TargetGpuLayers = 'all',

    [ValidatePattern('^(all|auto|[0-9]+)$')]
    [string]$DraftGpuLayers = 'all',

    [ValidateRange(1, 65535)]
    [int]$Port = 11538,

    [ValidateRange(0, 31)]
    [int]$NvidiaGpuIndex = 0,

    [ValidateRange(50, 2000)]
    [int]$GpuSampleIntervalMilliseconds = 100,

    [switch]$RequireCleanTree
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Add-Type -AssemblyName System.Net.Http

$powerRoot = Split-Path -Parent $PSScriptRoot
$binDirectory = [System.IO.Path]::GetFullPath($LlamaBinDirectory)
$serverPath = Join-Path $binDirectory 'llama-server.exe'
$targetPath = [System.IO.Path]::GetFullPath($TargetModel)
$draftPath = [System.IO.Path]::GetFullPath($DraftModel)
$promptPath = [System.IO.Path]::GetFullPath($PromptFile)
$outputPath = [System.IO.Path]::GetFullPath($Output)
$outputDirectory = Split-Path -Parent $outputPath
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

. (Join-Path $PSScriptRoot 'lib/gguf-speculative-benchmark.ps1')
. (Join-Path $PSScriptRoot 'lib/llamacpp-external-draft-benchmark.ps1')

$requiredPaths = [ordered]@{
    'llama-server' = $serverPath
    'TargetModel' = $targetPath
    'DraftModel' = $draftPath
    'PromptFile' = $promptPath
}
foreach ($required in $requiredPaths.GetEnumerator()) {
    Assert-Leaf $required.Value $required.Key
}
if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    throw 'nvidia-smi.exe is required for the external-draft benchmark'
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use"
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$normalizedTargetHash = $TargetSha256.ToLowerInvariant()
$normalizedDraftHash = $DraftSha256.ToLowerInvariant()
$actualTargetHash = Get-NormalizedSha256 $targetPath
$actualDraftHash = Get-NormalizedSha256 $draftPath
if ($actualTargetHash -ne $normalizedTargetHash) {
    throw "Target model SHA-256 mismatch: expected $normalizedTargetHash, got $actualTargetHash"
}
if ($actualDraftHash -ne $normalizedDraftHash) {
    throw "Draft model SHA-256 mismatch: expected $normalizedDraftHash, got $actualDraftHash"
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

$promptText = [System.IO.File]::ReadAllText($promptPath)
$requestBody = @{
    prompt = $promptText
    n_predict = $MaxTokens
    temperature = 0.0
    top_k = 1
    top_p = 1.0
    min_p = 0.0
    seed = 42
    ignore_eos = $true
    cache_prompt = $false
    stream = $false
    repeat_penalty = 1.0
    frequency_penalty = 0.0
    presence_penalty = 0.0
}

$gpuIdentity = @(& nvidia-smi.exe `
    "--id=$NvidiaGpuIndex" `
    '--query-gpu=name,uuid,driver_version,memory.total' `
    '--format=csv,noheader,nounits')
$activePowerScheme = (& powercfg.exe /getactivescheme) -join [Environment]::NewLine
$baseline = Invoke-Mode 'target-only' $false $requestBody
Start-Sleep -Seconds 2
$candidate = Invoke-Mode "draft-$DraftMode" $true $requestBody

$baselineHashes = @($baseline.samples | ForEach-Object { $_.output_sha256 } | Sort-Object -Unique)
$candidateHashes = @($candidate.samples | ForEach-Object { $_.output_sha256 } | Sort-Object -Unique)
$parity = $baselineHashes.Count -eq 1 -and
    $candidateHashes.Count -eq 1 -and
    $baselineHashes[0] -eq $candidateHashes[0]
$baselineRate = $baseline.summary.median_tokens_per_second
$candidateRate = $candidate.summary.median_tokens_per_second

$report = [ordered]@{
    schema = 'a3s.power.llamacpp-external-draft-benchmark.v1'
    created_at = [DateTimeOffset]::UtcNow.ToString('o')
    identity = [ordered]@{
        power_commit = $powerCommit
        dirty_worktree = $gitStatus.Count -gt 0
        git_status = $gitStatus
        llama_cpp_commit = $LlamaCppCommit.ToLowerInvariant()
        llama_cpp_rs_commit = $LlamaCppRsCommit.ToLowerInvariant()
        llama_server_sha256 = Get-NormalizedSha256 $serverPath
        target = [ordered]@{
            file = [System.IO.Path]::GetFileName($targetPath)
            size = (Get-Item -LiteralPath $targetPath).Length
            sha256 = $actualTargetHash
        }
        draft = [ordered]@{
            file = [System.IO.Path]::GetFileName($draftPath)
            size = (Get-Item -LiteralPath $draftPath).Length
            sha256 = $actualDraftHash
            mode = $DraftMode
        }
        prompt = [ordered]@{
            file = [System.IO.Path]::GetFileName($promptPath)
            size = (Get-Item -LiteralPath $promptPath).Length
            sha256 = Get-NormalizedSha256 $promptPath
        }
    }
    configuration = [ordered]@{
        samples = $Samples
        warmup_runs = $WarmupRuns
        max_tokens = $MaxTokens
        context_size = $ContextSize
        batch_size = $BatchSize
        threads = $Threads
        draft_max = $DraftMax
        target_gpu_layers = $TargetGpuLayers
        draft_gpu_layers = $DraftGpuLayers
        flash_attention = $true
        fit = $false
        parallel_slots = 1
        seed = 42
        greedy = $true
    }
    environment = [ordered]@{
        gpu = $gpuIdentity
        gpu_index = $NvidiaGpuIndex
        cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()
        os = [System.Environment]::OSVersion.VersionString
        active_power_scheme = $activePowerScheme.Trim()
    }
    baseline = $baseline
    candidate = $candidate
    comparison = [ordered]@{
        deterministic_output_parity = $parity
        output_sha256 = if ($parity) { $baselineHashes[0] } else { $null }
        throughput_speedup = if ($null -ne $baselineRate -and $baselineRate -gt 0 -and $null -ne $candidateRate) {
            $candidateRate / $baselineRate
        } else {
            $null
        }
        throughput_change_percent = if ($null -ne $baselineRate -and $baselineRate -gt 0 -and $null -ne $candidateRate) {
            100.0 * (($candidateRate / $baselineRate) - 1.0)
        } else {
            $null
        }
    }
}

$json = $report | ConvertTo-Json -Depth 12
[System.IO.File]::WriteAllText($outputPath, $json + [Environment]::NewLine, $utf8NoBom)
$json

if ($baseline.status -ne 'passed' -or $candidate.status -ne 'passed' -or -not $parity) {
    exit 1
}
