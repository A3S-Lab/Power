param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9][a-z0-9-]*$')]
    [string]$Label,

    [ValidateRange(1, 100)]
    [int]$Samples = 3,

    [ValidateRange(0, 20)]
    [int]$WarmupRuns = 1,

    [ValidateRange(1, 4096)]
    [int]$MaxTokens = 256,

    [string]$BenchmarkRoot = 'D:\models\a3s-power\qwen38\benchmark',

    [string]$Server = 'D:\models\a3s-power\qwen38\tools\llama-b10405-native-sm89-vs3\bin\llama-server.exe',

    [string]$Model = 'D:\models\a3s-power\qwen38\full\Qwen3.8-27B-Q6_K.gguf'
)

$ErrorActionPreference = 'Stop'

$promptPath = Join-Path $BenchmarkRoot 'prompt.txt'
$stdout = Join-Path $BenchmarkRoot "$Label.native.stdout.log"
$stderr = Join-Path $BenchmarkRoot "$Label.native.stderr.log"
$report = Join-Path $BenchmarkRoot "$Label.native.json"
$port = 11435
$process = $null

foreach ($requiredPath in @($Server, $Model, $promptPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required benchmark input does not exist: $requiredPath"
    }
}

if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $port is already in use"
}

$arguments = @(
    '--model', $Model,
    '--host', '127.0.0.1',
    '--port', "$port",
    '--ctx-size', '4096',
    '--batch-size', '24',
    '--ubatch-size', '24',
    '--threads', '10',
    '--threads-batch', '10',
    '--n-gpu-layers', 'all',
    '--flash-attn', 'on',
    '--parallel', '1',
    '--backend-sampling',
    '--spec-type', 'draft-mtp',
    '--spec-draft-n-max', '7',
    '--spec-draft-n-min', '0',
    '--spec-draft-p-min', '0',
    '--no-webui'
)

try {
    $process = Start-Process -FilePath $Server `
        -ArgumentList $arguments `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru

    $ready = $false
    for ($attempt = 0; $attempt -lt 180; $attempt++) {
        if ($process.HasExited) {
            $message = Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue
            throw "Server exited before becoming ready: $message"
        }
        try {
            $response = Invoke-WebRequest -UseBasicParsing `
                -Uri "http://127.0.0.1:$port/health" `
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
        throw 'Native llama.cpp server did not become ready'
    }

    $prompt = [IO.File]::ReadAllText($promptPath)
    $body = @{
        prompt = $prompt
        n_predict = $MaxTokens
        temperature = 0
        seed = 42
        cache_prompt = $false
        ignore_eos = $true
        stream = $false
    } | ConvertTo-Json -Compress

    $results = [System.Collections.Generic.List[object]]::new()
    $totalRuns = $WarmupRuns + $Samples
    for ($run = 0; $run -lt $totalRuns; $run++) {
        $response = Invoke-RestMethod `
            -Method Post `
            -Uri "http://127.0.0.1:$port/completion" `
            -ContentType 'application/json' `
            -Body $body `
            -TimeoutSec 900

        if ($run -ge $WarmupRuns) {
            $results.Add([pscustomobject]@{
                sample = $run - $WarmupRuns + 1
                predicted_tokens = $response.timings.predicted_n
                predicted_ms = $response.timings.predicted_ms
                tokens_per_second = $response.timings.predicted_per_second
                drafted_tokens = $response.timings.draft_n
                accepted_tokens = $response.timings.draft_n_accepted
                content_sha256 = [Convert]::ToHexString(
                    [Security.Cryptography.SHA256]::HashData(
                        [Text.Encoding]::UTF8.GetBytes([string]$response.content)
                    )
                ).ToLowerInvariant()
            })
        }
    }

    $rates = @($results | ForEach-Object tokens_per_second | Sort-Object)
    $summary = [pscustomobject]@{
        label = $Label
        samples = $results
        minimum_tokens_per_second = $rates[0]
        median_tokens_per_second = $rates[[Math]::Floor($rates.Count / 2)]
        maximum_tokens_per_second = $rates[-1]
    }
    $json = $summary | ConvertTo-Json -Depth 5
    Set-Content -LiteralPath $report -Value $json -Encoding utf8
    $json
} finally {
    if ($process -and -not $process.HasExited) {
        $process.Kill()
        $process.WaitForExit()
    }
}
