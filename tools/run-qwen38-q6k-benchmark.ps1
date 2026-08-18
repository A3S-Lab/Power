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

    [switch]$RequireHighPerformancePowerPlan,

    [string]$TargetDirectory = 'target-native-sm89-ninja',

    [string]$BenchmarkRoot = 'D:\models\a3s-power\qwen38\benchmark',

    [string]$PowerHome = 'D:\models\a3s-power\qwen38\power-home',

    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ModelHash = '562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727',

    [string]$RustLog = 'a3s_power::backend::llamacpp::speculative_runtime=info,a3s_power=info'
)

$ErrorActionPreference = 'Stop'

$powerRoot = Split-Path -Parent $PSScriptRoot
$server = Join-Path $powerRoot "$TargetDirectory\release\a3s-power.exe"
$benchmark = Join-Path $powerRoot "$TargetDirectory\release\a3s-power-speculative-bench.exe"
$prompt = Join-Path $BenchmarkRoot 'prompt.txt'
$stdout = Join-Path $BenchmarkRoot "$Label.stdout.log"
$stderr = Join-Path $BenchmarkRoot "$Label.stderr.log"
$report = Join-Path $BenchmarkRoot "$Label.json"
$model = 'qwen3.8-27b-q6-k'
$powerCommit = '491184ada54699ddfc4b40246cd6aee92d7550dd'
$process = $null

foreach ($requiredPath in @($server, $benchmark, $prompt, $Config)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required benchmark input does not exist: $requiredPath"
    }
}

if (Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 11434 is already in use'
}

if ($RequireHighPerformancePowerPlan) {
    $activePowerScheme = (& powercfg.exe /getactivescheme) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0 -or
        $activePowerScheme -notmatch '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c') {
        throw 'The Windows High performance power plan is required for this capture'
    }
}

$env:A3S_POWER_HOME = $PowerHome
$env:RUST_LOG = $RustLog

try {
    $process = Start-Process -FilePath $server `
        -ArgumentList @('serve', '--config', $Config) `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    $process.PriorityClass = $ProcessPriority

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
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark exited with code $LASTEXITCODE"
    }

    $raw = $rawLines -join [Environment]::NewLine
    Set-Content -LiteralPath $report -Value $raw -Encoding utf8
    $raw
} finally {
    if ($process -and -not $process.HasExited) {
        $process.Kill()
        $process.WaitForExit()
    }
}
