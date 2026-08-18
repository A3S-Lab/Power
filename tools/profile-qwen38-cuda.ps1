param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9][a-z0-9-]*$')]
    [string]$Label,

    [Parameter(Mandatory = $true)]
    [string]$Config,

    [ValidateRange(1, 4096)]
    [int]$MaxTokens = 128,

    [ValidateRange(1, 4096)]
    [int]$NumBatch = 24,

    [ValidateRange(0, 20)]
    [int]$WarmupRuns = 1,

    [string]$TargetDirectory = 'target-native-sm89-ninja',

    [string]$BenchmarkRoot = 'D:\models\a3s-power\qwen38\benchmark',

    [string]$PowerHome = 'D:\models\a3s-power\qwen38\power-home',

    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ModelHash = '562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727',

    [string]$Nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.2\target-windows-x64\nsys.exe'
)

$ErrorActionPreference = 'Stop'

$powerRoot = Split-Path -Parent $PSScriptRoot
$server = Join-Path $powerRoot "$TargetDirectory\release\a3s-power.exe"
$benchmark = Join-Path $powerRoot "$TargetDirectory\release\a3s-power-speculative-bench.exe"
$prompt = Join-Path $BenchmarkRoot 'prompt.txt'
$prefix = Join-Path $BenchmarkRoot $Label
$model = 'qwen3.8-27b-q6-k'
$powerCommit = '491184ada54699ddfc4b40246cd6aee92d7550dd'
$profile = $null
$gpuMonitor = $null
$serverProcess = $null

foreach ($requiredPath in @($Nsys, $server, $benchmark, $prompt, $Config)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required profiling input does not exist: $requiredPath"
    }
}

if (Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 11434 is already in use'
}

function Get-OwnedServerProcess {
    $connection = Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $connection) {
        return $null
    }

    $candidate = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
    if (-not $candidate -or $candidate.Path -ne $server) {
        throw "Port 11434 is owned by an unexpected process: $($connection.OwningProcess)"
    }
    $candidate
}

$env:A3S_POWER_HOME = $PowerHome
$env:RUST_LOG = 'a3s_power::backend::llamacpp::speculative_runtime=info,a3s_power=info'

try {
    $gpuMonitor = Start-Process -FilePath 'nvidia-smi.exe' `
        -ArgumentList @('pmon', '-s', 'um', '-d', '1') `
        -RedirectStandardOutput "$prefix.pmon.log" `
        -WindowStyle Hidden `
        -PassThru

    $profile = Start-Process -FilePath $Nsys `
        -ArgumentList @(
            'profile',
            '--trace=cuda,nvtx',
            '--sample=none',
            '--cpuctxsw=none',
            '--force-overwrite=true',
            '--output', $prefix,
            $server, 'serve', '--config', $Config
        ) `
        -RedirectStandardOutput "$prefix.stdout.log" `
        -RedirectStandardError "$prefix.stderr.log" `
        -WindowStyle Hidden `
        -PassThru

    $ready = $false
    for ($attempt = 0; $attempt -lt 180; $attempt++) {
        if ($profile.HasExited) {
            $message = Get-Content -LiteralPath "$prefix.stderr.log" -Raw -ErrorAction SilentlyContinue
            throw "Profiled server exited before becoming ready: $message"
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
        throw 'Profiled server did not expose the registered model'
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
        --samples 1 `
        --min-tokens-per-second 0 `
        --timeout-secs 900)
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark exited with code $LASTEXITCODE"
    }
    $rawLines -join [Environment]::NewLine |
        Set-Content -LiteralPath "$prefix.benchmark.json" -Encoding utf8

    $serverProcess = Get-OwnedServerProcess
    if (-not $serverProcess) {
        throw 'Profiled server stopped before the capture could be finalized'
    }
    $serverProcess.Kill()
    $serverProcess.WaitForExit()
    $profile.WaitForExit(180000) | Out-Null
    if (-not $profile.HasExited) {
        throw 'Nsight Systems did not finalize the report within 180 seconds'
    }
} finally {
    if (-not $serverProcess) {
        $serverProcess = Get-OwnedServerProcess
    }
    if ($serverProcess -and -not $serverProcess.HasExited) {
        $serverProcess.Kill()
        $serverProcess.WaitForExit()
    }
    if ($profile -and -not $profile.HasExited) {
        $profile.Kill()
        $profile.WaitForExit()
    }
    if ($gpuMonitor -and -not $gpuMonitor.HasExited) {
        $gpuMonitor.Kill()
        $gpuMonitor.WaitForExit()
    }
}

Get-Item -LiteralPath "$prefix.nsys-rep"
