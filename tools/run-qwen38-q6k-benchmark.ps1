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

    [switch]$CudaHighPriority,

    [ValidateRange(0, 100)]
    [int]$MaximumIdleGpuUtilizationPercent = 100,

    [ValidateRange(1, 120)]
    [int]$IdleGpuSampleCount = 3,

    [ValidateRange(100, 60000)]
    [int]$IdleGpuSampleIntervalMilliseconds = 500,

    [ValidateScript({ $_ -ge 0 })]
    [int[]]$NvidiaGpuIndices = @(0),

    [switch]$RequireHighPerformancePowerPlan,

    [switch]$RequireCleanTree,

    [string]$TargetDirectory = 'target-native-sm89-ninja',

    [string]$BenchmarkRoot = 'D:\models\a3s-power\qwen38\benchmark',

    [string]$PromptFile,

    [string]$PowerHome = 'D:\models\a3s-power\qwen38\power-home',

    [switch]$PreflightOnly,

    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._+-]*$')]
    [string]$ExpectedBackend = 'llama.cpp',

    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ModelHash = '562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727',

    [string]$RustLog = 'a3s_power::backend::llamacpp::speculative_runtime=info,a3s_power=info'
)

$powerRoot = Split-Path -Parent $PSScriptRoot
$genericRunner = Join-Path $PSScriptRoot 'run-gguf-speculative-benchmark.ps1'
$effectivePrompt = if ([string]::IsNullOrWhiteSpace($PromptFile)) {
    Join-Path $powerRoot 'docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt'
} else {
    [System.IO.Path]::GetFullPath($PromptFile)
}

$runnerParameters = @{
    Label = $Label
    Config = $Config
    Model = 'qwen3.8-27b-q6-k'
    ModelHash = $ModelHash
    PromptFile = $effectivePrompt
    PowerHome = $PowerHome
    Samples = $Samples
    WarmupRuns = $WarmupRuns
    MaxTokens = $MaxTokens
    NumCtx = $NumCtx
    NumBatch = $NumBatch
    MinimumTokensPerSecond = $MinimumTokensPerSecond
    ProcessPriority = $ProcessPriority
    ProcessorAffinityMask = $ProcessorAffinityMask
    LockGpuClockMHz = $LockGpuClockMHz
    CudaHighPriority = $CudaHighPriority
    MaximumIdleGpuUtilizationPercent = $MaximumIdleGpuUtilizationPercent
    IdleGpuSampleCount = $IdleGpuSampleCount
    IdleGpuSampleIntervalMilliseconds = $IdleGpuSampleIntervalMilliseconds
    NvidiaGpuIndices = $NvidiaGpuIndices
    RequireHighPerformancePowerPlan = $RequireHighPerformancePowerPlan
    RequireCleanTree = $RequireCleanTree
    TargetDirectory = $TargetDirectory
    BenchmarkRoot = $BenchmarkRoot
    Port = 11434
    HardwareLabel = "rtx4090-qwen38-q6k-$Label"
    PreflightOnly = $PreflightOnly
    ExpectedBackend = $ExpectedBackend
    RustLog = $RustLog
}
if ($PSBoundParameters.ContainsKey('MinimumSampleTokensPerSecond')) {
    $runnerParameters.MinimumSampleTokensPerSecond = $MinimumSampleTokensPerSecond
}

& $genericRunner @runnerParameters
