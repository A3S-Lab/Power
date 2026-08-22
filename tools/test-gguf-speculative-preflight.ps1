$ErrorActionPreference = 'Stop'

function Assert-Equal {
    param(
        [Parameter(Mandatory = $true)]
        $Actual,

        [Parameter(Mandatory = $true)]
        $Expected,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if ($Actual -ne $Expected) {
        throw "$Message (expected '$Expected', got '$Actual')"
    }
}

function Assert-True {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$Condition,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

$powerRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot 'run-gguf-speculative-benchmark.ps1'
$runtimeHelper = Join-Path $PSScriptRoot 'lib/gguf-speculative-benchmark.ps1'
. $runtimeHelper
$testId = [Guid]::NewGuid().ToString('N')
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "a3s-power-preflight-test-$testId"
$targetName = "target-preflight-contract-test-$testId"
$targetRoot = Join-Path $powerRoot $targetName
$releaseRoot = Join-Path $targetRoot 'release'
$outputRoot = Join-Path $tempRoot 'output'
$fakeNvidiaLog = Join-Path $tempRoot 'nvidia-smi.log'
$config = Join-Path $tempRoot 'test.acl'
$prompt = Join-Path $tempRoot 'prompt.txt'
$originalFakeUtilization = $env:A3S_POWER_FAKE_NVIDIA_UTILIZATION
$originalFakeLog = $env:A3S_POWER_FAKE_NVIDIA_LOG
$originalCudaHighPriority = $env:GGML_CUDA_HIGH_PRIORITY
$originalNvidiaSmiFunction = Get-Item `
    -LiteralPath 'Function:\global:nvidia-smi.exe' `
    -ErrorAction SilentlyContinue

try {
    New-Item -ItemType Directory -Force -Path $releaseRoot, $outputRoot | Out-Null
    [System.IO.File]::WriteAllBytes(
        (Join-Path $releaseRoot 'a3s-power.exe'),
        [byte[]](1, 2, 3)
    )
    [System.IO.File]::WriteAllBytes(
        (Join-Path $releaseRoot 'a3s-power-speculative-bench.exe'),
        [byte[]](4, 5, 6)
    )
    [System.IO.File]::WriteAllText($config, "model {}`r`n")
    [System.IO.File]::WriteAllText($prompt, "preflight contract`r`n")

    Assert-Equal `
        (ConvertTo-WindowsCommandLineArgument 'path with spaces\') `
        '"path with spaces\\"' `
        'Windows native argument quoting'
    $helperStdout = Join-Path $outputRoot 'native-helper.stdout.log'
    $helperStderr = Join-Path $outputRoot 'native-helper.stderr.log'
    $helperRun = Invoke-MonitoredNativeProcess `
        'where.exe' `
        @('powershell.exe') `
        $helperStdout `
        $helperStderr `
        @() `
        100
    Assert-Equal ([int]$helperRun.exit_code) 0 'Monitored native process exit code'
    Assert-True `
        (-not [string]::IsNullOrWhiteSpace($helperRun.stdout)) `
        'Monitored native process must capture standard output'
    Assert-Equal $helperRun.stderr '' 'Monitored native process standard error'

    Set-Item -LiteralPath 'Function:\global:nvidia-smi.exe' -Value {
        $flatArguments = @($args | ForEach-Object { $_ })
        $invocation = $flatArguments -join ' '
        $log = $env:A3S_POWER_FAKE_NVIDIA_LOG
        if (-not [string]::IsNullOrEmpty($log)) {
            [System.IO.File]::AppendAllText(
                $log,
                $invocation + [Environment]::NewLine
            )
        }

        if ($invocation.Contains('--query-gpu=utilization.gpu')) {
            if ([string]::IsNullOrEmpty(
                $env:A3S_POWER_FAKE_NVIDIA_UTILIZATION
            )) {
                Write-Output '0'
            } else {
                Write-Output $env:A3S_POWER_FAKE_NVIDIA_UTILIZATION
            }
        } elseif ($invocation.Contains('--query-gpu=index,name,driver_version')) {
            Write-Output '0, Fake NVIDIA GPU, 999.0, P0, 2745, 2745, 450, 40, 24564'
        } elseif ($args.Count -eq 0) {
            Write-Output 'fake NVIDIA process snapshot'
        }

        $global:LASTEXITCODE = 0
    }
    $env:A3S_POWER_FAKE_NVIDIA_LOG = $fakeNvidiaLog

    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        0
    )
    $listener.Start()
    $testPort = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
    $listener.Stop()

    $common = @{
        Config = $config
        Model = 'preflight-contract-model'
        ModelHash = ('0' * 64)
        PromptFile = $prompt
        PowerHome = $tempRoot
        MaxTokens = 2
        NumCtx = 8
        NumBatch = 2
        NvidiaGpuIndices = @(0)
        IdleGpuSampleIntervalMilliseconds = 100
        TargetDirectory = $targetName
        BenchmarkRoot = $outputRoot
        Port = $testPort
        PreflightOnly = $true
    }

    $env:A3S_POWER_FAKE_NVIDIA_UTILIZATION = '9'
    $failureParameters = $common.Clone()
    $failureParameters.Label = 'idle-gate-failure'
    $failureParameters.MaximumIdleGpuUtilizationPercent = 2
    $failureParameters.IdleGpuSampleCount = 4
    $failureMessage = $null
    try {
        & $runner @failureParameters | Out-Null
    } catch {
        $failureMessage = $_.Exception.Message
    }
    Assert-True ($null -ne $failureMessage) 'The busy-GPU preflight must fail'
    Assert-True `
        $failureMessage.StartsWith('GPU idle utilization exceeded 2 percent:') `
        'The busy-GPU failure must explain the rejected utilization gate'

    $failureReceiptPath = Join-Path $outputRoot 'idle-gate-failure.preflight.json'
    Assert-True `
        (Test-Path -LiteralPath $failureReceiptPath -PathType Leaf) `
        'The failed preflight receipt must be retained'
    $failureReceipt = Get-Content -LiteralPath $failureReceiptPath -Raw | ConvertFrom-Json
    Assert-Equal $failureReceipt.schema `
        'a3s.power.speculative-benchmark.preflight.v1' `
        'Failed preflight schema'
    Assert-Equal ([bool]$failureReceipt.passed) $false 'Failed preflight state'
    Assert-Equal $failureReceipt.failure.code `
        'nvidia-idle-utilization-exceeded' `
        'Failed preflight reason'
    Assert-Equal ([int]$failureReceipt.gpu.idle_sample_count) 4 `
        'Failed preflight configured sample count'
    Assert-Equal ([int]$failureReceipt.gpu.idle_utilization_samples.Count) 4 `
        'Failed preflight captured sample count'
    Assert-Equal ([int]$failureReceipt.gpu.idle_window_duration_milliseconds) 300 `
        'Failed preflight quiet-window duration'
    Assert-True `
        ([int]$failureReceipt.gpu.observed_idle_window_duration_milliseconds -ge 300) `
        'Failed preflight must record the elapsed quiet window'
    Assert-True `
        (-not [string]::IsNullOrWhiteSpace(
            $failureReceipt.gpu.idle_utilization_samples[0].observed_at
        )) `
        'Failed preflight samples must carry observation timestamps'
    Assert-Equal ([int]$failureReceipt.gpu.maximum_observed_idle_utilization_percent) 9 `
        'Failed preflight maximum utilization'

    $firstBytes = [System.IO.File]::ReadAllBytes($failureReceiptPath)
    $hasUtf8Bom = $firstBytes.Length -ge 3 -and
        $firstBytes[0] -eq 0xef -and
        $firstBytes[1] -eq 0xbb -and
        $firstBytes[2] -eq 0xbf
    Assert-Equal $hasUtf8Bom $false 'Preflight receipts must use BOM-free UTF-8'

    $env:A3S_POWER_FAKE_NVIDIA_UTILIZATION = '1'
    $successParameters = $common.Clone()
    $successParameters.Label = 'preflight-only-success'
    $successParameters.MaximumIdleGpuUtilizationPercent = 2
    $successParameters.IdleGpuSampleCount = 3
    $successParameters.LockGpuClockMHz = 2745
    $successParameters.CudaHighPriority = $true
    $env:GGML_CUDA_HIGH_PRIORITY = 'restore-after-preflight'
    & $runner @successParameters | Out-Null

    $successReceiptPath = Join-Path $outputRoot 'preflight-only-success.preflight.json'
    $successReceipt = Get-Content -LiteralPath $successReceiptPath -Raw | ConvertFrom-Json
    Assert-Equal ([bool]$successReceipt.passed) $true 'Successful preflight state'
    Assert-Equal ([int]$successReceipt.gpu.idle_utilization_samples.Count) 3 `
        'Successful preflight captured sample count'
    Assert-Equal ([int]$successReceipt.gpu.maximum_observed_idle_utilization_percent) 1 `
        'Successful preflight maximum utilization'
    Assert-Equal ([int]$successReceipt.gpu.clock_lock_applied_indices[0]) 0 `
        'Successful preflight clock-lock device'
    Assert-Equal ([bool]$successReceipt.gpu.cuda_high_priority) $true `
        'Successful preflight CUDA stream priority'
    Assert-Equal $env:GGML_CUDA_HIGH_PRIORITY 'restore-after-preflight' `
        'Preflight must restore the caller CUDA stream-priority environment'
    Assert-Equal `
        (Test-Path -LiteralPath (Join-Path $outputRoot 'preflight-only-success.json')) `
        $false `
        'Preflight-only mode must not create a benchmark report'
    Assert-Equal `
        (Test-Path -LiteralPath (Join-Path $outputRoot 'preflight-only-success.environment.json')) `
        $false `
        'Preflight-only mode must not create a benchmark environment receipt'

    $fakeInvocations = Get-Content -LiteralPath $fakeNvidiaLog -Raw
    Assert-True `
        $fakeInvocations.Contains('--lock-gpu-clocks=2745,2745') `
        'The requested clock lock must be applied during preflight'
    Assert-True `
        $fakeInvocations.Contains('--reset-gpu-clocks') `
        'Preflight-only mode must reset an applied clock lock'

    Write-Output 'GGUF speculative benchmark preflight contract: PASS'
} finally {
    if ($null -eq $originalNvidiaSmiFunction) {
        Remove-Item `
            -LiteralPath 'Function:\global:nvidia-smi.exe' `
            -ErrorAction SilentlyContinue
    } else {
        Set-Item `
            -LiteralPath 'Function:\global:nvidia-smi.exe' `
            -Value $originalNvidiaSmiFunction.ScriptBlock
    }
    if ($null -eq $originalFakeUtilization) {
        Remove-Item Env:A3S_POWER_FAKE_NVIDIA_UTILIZATION -ErrorAction SilentlyContinue
    } else {
        $env:A3S_POWER_FAKE_NVIDIA_UTILIZATION = $originalFakeUtilization
    }
    if ($null -eq $originalFakeLog) {
        Remove-Item Env:A3S_POWER_FAKE_NVIDIA_LOG -ErrorAction SilentlyContinue
    } else {
        $env:A3S_POWER_FAKE_NVIDIA_LOG = $originalFakeLog
    }
    if ($null -eq $originalCudaHighPriority) {
        Remove-Item Env:GGML_CUDA_HIGH_PRIORITY -ErrorAction SilentlyContinue
    } else {
        $env:GGML_CUDA_HIGH_PRIORITY = $originalCudaHighPriority
    }

    $resolvedPowerRoot = [System.IO.Path]::GetFullPath($powerRoot).TrimEnd('\')
    $resolvedTargetRoot = [System.IO.Path]::GetFullPath($targetRoot)
    $expectedTargetPrefix = "$resolvedPowerRoot\target-preflight-contract-test-"
    if ($resolvedTargetRoot.StartsWith(
        $expectedTargetPrefix,
        [StringComparison]::OrdinalIgnoreCase
    ) -and (Test-Path -LiteralPath $resolvedTargetRoot)) {
        [System.IO.Directory]::Delete($resolvedTargetRoot, $true)
    }

    $resolvedTempRoot = [System.IO.Path]::GetFullPath($tempRoot)
    $systemTempRoot = [System.IO.Path]::GetFullPath(
        [System.IO.Path]::GetTempPath()
    ).TrimEnd('\')
    $expectedTempPrefix = "$systemTempRoot\a3s-power-preflight-test-"
    if ($resolvedTempRoot.StartsWith(
        $expectedTempPrefix,
        [StringComparison]::OrdinalIgnoreCase
    ) -and (Test-Path -LiteralPath $resolvedTempRoot)) {
        [System.IO.Directory]::Delete($resolvedTempRoot, $true)
    }
}
