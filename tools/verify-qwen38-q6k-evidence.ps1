[CmdletBinding()]
param(
  [string]$EvidenceRoot = '',
  [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) {
  $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
  $EvidenceRoot = Join-Path $scriptRoot `
    '..\docs\benchmarks\qwen3.8-27b-q6k-rtx4090'
}

function Assert-Equal {
  param(
    [Parameter(Mandatory = $true)]
    [AllowNull()]
    [object]$Actual,
    [Parameter(Mandatory = $true)]
    [AllowNull()]
    [object]$Expected,
    [Parameter(Mandatory = $true)]
    [string]$Label
  )

  if ($Actual -cne $Expected) {
    throw "$Label mismatch: expected '$Expected', got '$Actual'"
  }
}

function Assert-Near {
  param(
    [Parameter(Mandatory = $true)]
    [double]$Actual,
    [Parameter(Mandatory = $true)]
    [double]$Expected,
    [Parameter(Mandatory = $true)]
    [string]$Label,
    [double]$Tolerance = 1e-9
  )

  if ([math]::Abs($Actual - $Expected) -gt $Tolerance) {
    throw "$Label mismatch: expected '$Expected', got '$Actual'"
  }
}

function Assert-FileHash {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$Expected,
    [Parameter(Mandatory = $true)]
    [string]$Label
  )

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "$Label is missing: $Path"
  }

  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
  if ($actual -cne $Expected.ToUpperInvariant()) {
    throw "$Label SHA-256 mismatch: expected '$Expected', got '$actual'"
  }
}

$resolvedEvidenceRoot = (Resolve-Path -LiteralPath $EvidenceRoot).Path
$currentPath = Join-Path $resolvedEvidenceRoot `
  'quality\full-vocabulary-s7-current-rtx4090-3x.json'
$historicalReportPath = Join-Path $resolvedEvidenceRoot `
  'final-affinity-1024-9x.json'
$historicalEnvironmentPath = Join-Path $resolvedEvidenceRoot `
  'final-affinity-1024-9x.environment.json'
$s6ConfigPath = Join-Path $resolvedEvidenceRoot `
  'mtp7-snap6-full-vocab-cpu-embedding.acl'
$s7ConfigPath = Join-Path $resolvedEvidenceRoot `
  'mtp7-snap7-full-vocab-cpu-embedding.acl'
$promptPath = Join-Path $resolvedEvidenceRoot 'prompt.txt'
$purePeakReportPath = Join-Path $resolvedEvidenceRoot `
  'pure-q6-fr8192-1024-9x.json'
$purePeakEnvironmentPath = Join-Path $resolvedEvidenceRoot `
  'pure-q6-fr8192-1024-9x.environment.json'
$pureFullReportPath = Join-Path $resolvedEvidenceRoot `
  'pure-q6-full-vocabulary-1024-9x.json'
$pureFullEnvironmentPath = Join-Path $resolvedEvidenceRoot `
  'pure-q6-full-vocabulary-1024-9x.environment.json'
$pureCalibrationPath = Join-Path $resolvedEvidenceRoot `
  'quality\pure-q6-fr8192-calibration-rtx4090-1x.json'
$pureCalibrationEnvironmentPath = Join-Path $resolvedEvidenceRoot `
  'quality\pure-q6-fr8192-calibration-rtx4090-1x.environment.json'
$purePeakConfigPath = Join-Path $resolvedEvidenceRoot `
  'pure-q6-mtp7-snap6-fr8192-host-staged.acl'
$pureFullConfigPath = Join-Path $resolvedEvidenceRoot `
  'pure-q6-mtp7-snap7-host-staged.acl'
$deepPeakReportPath = Join-Path $resolvedEvidenceRoot `
  'deepopt-20260822\peak\deepopt-final-f6326bb-k7s6-b11-faoff-cudahigh-1024-9x.json'
$deepPeakEnvironmentPath = Join-Path $resolvedEvidenceRoot `
  'deepopt-20260822\peak\deepopt-final-f6326bb-k7s6-b11-faoff-cudahigh-1024-9x.environment.json'
$deepPeakPreflightPath = Join-Path $resolvedEvidenceRoot `
  'deepopt-20260822\peak\deepopt-final-f6326bb-k7s6-b11-faoff-cudahigh-1024-9x.preflight.json'
$deepQualityRoot = Join-Path $resolvedEvidenceRoot 'deepopt-20260822\quality'
$deepQualityEnvironmentPath = Join-Path $deepQualityRoot 'environment.json'
$deepOffReportPath = Join-Path $deepQualityRoot 'r01-o01-off-b8.json'
$deepMtpReportPath = Join-Path $deepQualityRoot `
  'r01-o02-fr8192-k6-s6-b8-fixed.json'
$deepSweepPath = Join-Path $deepQualityRoot 'sweep.json'
$deepPeakConfigPath = Join-Path $resolvedEvidenceRoot `
  'pure-q6-mtp7-snap6-fr8192-rtx4090-throughput.acl'
$deepGeneralConfigPath = Join-Path $resolvedEvidenceRoot `
  'pure-q6-mtp6-snap6-fr8192-rtx4090-general.acl'

$expectedHashes = @(
  @{
    Path = $currentPath
    Hash = '83D418651959E49074511BCD03CA33633BD0E26BC079D9758F26519DB6DBEB31'
    Label = 'Current compact evidence'
  },
  @{
    Path = $historicalReportPath
    Hash = 'AD478DFDB3DF7D3560FB39EEB2BE854715BBB483A985C8E10793DF053C0C2D72'
    Label = 'Historical performance report'
  },
  @{
    Path = $historicalEnvironmentPath
    Hash = 'C0CB352E99DCDD2C9BD487CB7501A1F231C97EF5A661C26446EF3C2E4F5EDA8D'
    Label = 'Historical environment receipt'
  },
  @{
    Path = $s6ConfigPath
    Hash = '2F348CCA96282A22650D9766CFFA81251EA10A5E34A089BCC91B0822AB5C1D0E'
    Label = 'K7/S6 runtime configuration'
  },
  @{
    Path = $s7ConfigPath
    Hash = '759EF6E5E60A08939ED747558992FA3031D63D2ECD59DACFDAE59790CC6FF79A'
    Label = 'K7/S7 runtime configuration'
  },
  @{
    Path = $promptPath
    Hash = 'D95A5E4DAD822BA9C84138F7A120017318BCB3A6A90E77246A8EC4EDE0E65D89'
    Label = 'Benchmark prompt'
  },
  @{
    Path = $purePeakReportPath
    Hash = '26AE70F1606A57E562ACDEFA2CFD5679246E47C7ECBCD103B06A6E9EDFBE790C'
    Label = 'Pure Q6 prefix-FR performance report'
  },
  @{
    Path = $purePeakEnvironmentPath
    Hash = 'AFD4115FEED91B1C40C712DC5C0EF242A7532C394D0B93C03FEEF9DAF61D5AB8'
    Label = 'Pure Q6 prefix-FR environment receipt'
  },
  @{
    Path = $pureFullReportPath
    Hash = '714BB2DCEBD79ED09E23F4A9735EF0F435090E140B81B92157E2262023719E24'
    Label = 'Pure Q6 full-vocabulary performance report'
  },
  @{
    Path = $pureFullEnvironmentPath
    Hash = 'D6845400CFFE267EB71BDCA1FAE4805AD32FE3170D6E0EAE3D8B1C287E525E60'
    Label = 'Pure Q6 full-vocabulary environment receipt'
  },
  @{
    Path = $pureCalibrationPath
    Hash = '8A867228EE360441C43BB6037F8DAA437ED91C4B544CF51142DCCFEA913FD171'
    Label = 'Pure Q6 workload calibration'
  },
  @{
    Path = $pureCalibrationEnvironmentPath
    Hash = '456E6F3DCED8C3C940113FBD28A1083975A93B45A58FC1DF8440652DFBC29796'
    Label = 'Pure Q6 workload environment receipt'
  },
  @{
    Path = $purePeakConfigPath
    Hash = '9B1213DF972EA3731010A1FA72B0D553BA73DA42F31E92EAA4FECD3156CBF2EF'
    Label = 'Pure Q6 K7/S6 prefix-FR configuration'
  },
  @{
    Path = $pureFullConfigPath
    Hash = 'EB445101C1E33A035C9B1D120FEC12D9B21E6CE1B2FE5486AD46BEE52878A588'
    Label = 'Pure Q6 K7/S7 full-vocabulary configuration'
  },
  @{
    Path = $deepPeakReportPath
    Hash = '9D8D767EACCDBEA5C3AD09783556ED940A6D5E66ECFEA482E80B58DB631492CA'
    Label = 'Deep-optimization peak report'
  },
  @{
    Path = $deepPeakEnvironmentPath
    Hash = '2C43D2AD8703AEE64051B363FD58735965950BDD6FFFB6607B09660A90934C63'
    Label = 'Deep-optimization peak environment receipt'
  },
  @{
    Path = $deepPeakPreflightPath
    Hash = '6CA1259C687A4DD08C3759CC59F3A74FD4A5F7F2A0DEA96A6DE3DA6DF0995C2A'
    Label = 'Deep-optimization peak preflight receipt'
  },
  @{
    Path = $deepQualityEnvironmentPath
    Hash = '162C9994B0000FCCAA9269770C1EFD7AA15561761D242E33E8015761D1DCA22B'
    Label = 'Deep-optimization quality environment receipt'
  },
  @{
    Path = $deepOffReportPath
    Hash = '86CE4BD3B912B20B90B6C42754500AAD341F0B0131B90A88361073354E91B958'
    Label = 'Deep-optimization target-only quality report'
  },
  @{
    Path = $deepMtpReportPath
    Hash = 'EB6CDD7B02F196FC3411CA4DA8D88BAB489BCC98274DA18EE88D7EE826C0526D'
    Label = 'Deep-optimization MTP quality report'
  },
  @{
    Path = $deepSweepPath
    Hash = '05F29C83397E664D02563EF45396BFBCFABACA91AE2FBDE853063CF91A9B4E7F'
    Label = 'Deep-optimization paired quality aggregate'
  },
  @{
    Path = $deepPeakConfigPath
    Hash = '674D3A36E0F0019C9E39E60994EA40EEE0477615827464EDEE1FB9627A74CDEC'
    Label = 'Deep-optimization peak configuration'
  },
  @{
    Path = $deepGeneralConfigPath
    Hash = 'B4F3DB4229BFAD05371BBED0CE1FEC165AA2B05279405078AA8F7721721ABB37'
    Label = 'Deep-optimization general configuration'
  }
)

foreach ($expectedHash in $expectedHashes) {
  Assert-FileHash `
    -Path $expectedHash.Path `
    -Expected $expectedHash.Hash `
    -Label $expectedHash.Label
}

$current = Get-Content -Raw -LiteralPath $currentPath | ConvertFrom-Json
Assert-Equal $current.schema `
  'a3s.power.qwen38-full-vocabulary-summary.v1' 'Current evidence schema'
Assert-Equal $current.identity.tbq4_model_sha256 `
  '5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6' `
  'TBQ4 model identity'
Assert-Equal ([long]$current.identity.tbq4_model_bytes) ([long]19187686464) `
  'TBQ4 model byte length'
Assert-Equal $current.identity.gpu 'NVIDIA GeForce RTX 4090' 'Acceptance GPU'
Assert-Equal ([int]$current.workload.tasks) 100 'Quality task count'
Assert-Equal ([int]$current.workload.repetitions) 3 'Quality repetition count'
Assert-Equal ([int]$current.workload.num_batch) 14 'Quality batch size'
Assert-Equal ([int]$current.workload.warmup_requests_per_run) 1 `
  'Quality warm-up count'

$currentModes = @($current.modes)
Assert-Equal $currentModes.Count 3 'Current mode count'
$s7Modes = @($currentModes | Where-Object {
    $_.name -eq 'tbq4-mtp-full-vocab-k7-s7'
  })
Assert-Equal $s7Modes.Count 1 'K7/S7 mode count'
$s7 = $s7Modes[0]
Assert-Equal ([int]$s7.lenient_score) 76 'K7/S7 lenient score'
Assert-Equal ([int]$s7.strict_score) 66 'K7/S7 strict score'
Assert-Equal ([int]$s7.prediction_stable_tasks) 100 `
  'K7/S7 stable prediction count'
Assert-Equal ([int]$s7.content_stable_tasks) 100 `
  'K7/S7 stable content count'
Assert-Near ([double]$s7.mean_workload_tokens_per_second) `
  83.22814601950864 'K7/S7 request-wide throughput'
Assert-Near ([double]$s7.weighted_acceptance_rate) `
  0.5132958564931783 'K7/S7 proposal acceptance'
Assert-Equal ($s7.fallback_replays_per_run -join ',') '0,0,0' `
  'K7/S7 fallback replay counts'
Assert-Equal ($s7.rollback_guard_activations_per_run -join ',') '0,0,0' `
  'K7/S7 rollback guard activations'

$s6Gate = $current.peak_gates.k7_s6_guarded
Assert-Equal ([int]$s6Gate.samples) 9 'K7/S6 sample count'
Assert-Equal ([int]$s6Gate.samples_at_or_above_175) 9 `
  'K7/S6 samples at or above 175 token/s'
Assert-Near ([double]$s6Gate.median_decode_tokens_per_second) `
  177.71645508101074 'K7/S6 median steady decode'
Assert-Near ([double]$s6Gate.minimum_decode_tokens_per_second) `
  176.72867856598307 'K7/S6 minimum steady decode'

$s7Gate = $current.peak_gates.k7_s7_rollback_complete
Assert-Equal ([int]$s7Gate.samples) 9 'K7/S7 sample count'
Assert-Equal ([int]$s7Gate.samples_at_or_above_175) 5 `
  'K7/S7 samples at or above 175 token/s'
Assert-Near ([double]$s7Gate.median_decode_tokens_per_second) `
  175.20889378841997 'K7/S7 median steady decode'
Assert-Near ([double]$s7Gate.minimum_decode_tokens_per_second) `
  174.2211132623549 'K7/S7 minimum steady decode'
Assert-Equal $s7Gate.output_sha256 `
  'a54538eaaf6cc0b8b43cbafd489c7779f0f5206c93d5034fd3a16f4366a90523' `
  'K7/S7 deterministic output identity'

$historical = Get-Content -Raw -LiteralPath $historicalReportPath |
  ConvertFrom-Json
$rates = @($historical.samples.decode_tokens_per_second | Sort-Object)
$outputHashes = @($historical.samples.output_sha256 | Sort-Object -Unique)
$median = [double]$rates[[math]::Floor($rates.Count / 2)]
$minimum = [double]$rates[0]

Assert-Equal $historical.schema 'a3s.power.speculative-benchmark.v1' `
  'Historical report schema'
Assert-Equal $rates.Count 9 'Historical sample count'
Assert-Near $median 177.30624292681546 'Historical median steady decode'
Assert-Near $minimum 175.59583804564133 'Historical minimum steady decode'
Assert-Equal ([bool]$historical.threshold_passed) $true `
  'Historical threshold result'
Assert-Equal @($rates | Where-Object { $_ -lt 175 }).Count 0 `
  'Historical samples below 175 token/s'
Assert-Equal $outputHashes.Count 1 'Historical unique output identity count'
Assert-Equal $historical.output_sha256 `
  'a54538eaaf6cc0b8b43cbafd489c7779f0f5206c93d5034fd3a16f4366a90523' `
  'Historical deterministic output identity'
Assert-Equal @($historical.samples | Where-Object {
    $_.completion_tokens -ne 1024
  }).Count 0 'Historical short output count'

$pureModelSha256 =
  '562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727'
$pureOutputSha256 =
  'a54538eaaf6cc0b8b43cbafd489c7779f0f5206c93d5034fd3a16f4366a90523'
$pureCommit = 'eb6aeda59561eff3e4e7592704cab6fc863b72c7'

$purePeak = Get-Content -Raw -LiteralPath $purePeakReportPath |
  ConvertFrom-Json
$purePeakRates = @($purePeak.samples.decode_tokens_per_second | Sort-Object)
$purePeakEndToEnd = @(
  $purePeak.samples.end_to_end_tokens_per_second | Sort-Object
)
$purePeakHashes = @($purePeak.samples.output_sha256 | Sort-Object -Unique)

Assert-Equal $purePeak.schema 'a3s.power.speculative-benchmark.v1' `
  'Pure Q6 prefix-FR report schema'
Assert-Equal $purePeak.identity.power_commit $pureCommit `
  'Pure Q6 prefix-FR source revision'
Assert-Equal $purePeak.identity.model_sha256 $pureModelSha256 `
  'Pure Q6 prefix-FR model identity'
Assert-Equal ([long]$purePeak.identity.model_bytes) ([long]22884408288) `
  'Pure Q6 prefix-FR model byte length'
Assert-Equal $purePeak.identity.speculative.mode 'mtp' `
  'Pure Q6 prefix-FR speculative mode'
Assert-Equal ([int]$purePeak.identity.speculative.draft_max) 7 `
  'Pure Q6 prefix-FR draft width'
Assert-Equal ([int]$purePeak.identity.speculative.mtp_recurrent_snapshots) 6 `
  'Pure Q6 prefix-FR recurrent snapshots'
Assert-Equal ([bool]$purePeak.identity.speculative.mtp_recurrent_chain) $false `
  'Pure Q6 prefix-FR recurrent chain'
Assert-Equal ([int]$purePeak.identity.speculative.mtp_fr_vocab_size) 8192 `
  'Pure Q6 prefix-FR row limit'
Assert-Equal ([int]$purePeak.workload.max_tokens) 1024 `
  'Pure Q6 prefix-FR generated-token count'
Assert-Equal ([int]$purePeak.workload.num_batch) 14 `
  'Pure Q6 prefix-FR batch size'
Assert-Equal ([int]$purePeak.warmup_runs) 1 `
  'Pure Q6 prefix-FR warm-up count'
Assert-Equal $purePeakRates.Count 9 'Pure Q6 prefix-FR sample count'
Assert-Near ([double]$purePeakRates[4]) 176.6108685085471 `
  'Pure Q6 prefix-FR median steady decode'
Assert-Near ([double]$purePeakRates[0]) 173.26295503882207 `
  'Pure Q6 prefix-FR minimum steady decode'
Assert-Near ([double]$purePeakEndToEnd[4]) 167.3518723471296 `
  'Pure Q6 prefix-FR median end-to-end throughput'
Assert-Equal @($purePeakRates | Where-Object { $_ -ge 175 }).Count 7 `
  'Pure Q6 prefix-FR samples at or above 175 token/s'
Assert-Equal ([bool]$purePeak.threshold_passed) $true `
  'Pure Q6 prefix-FR threshold result'
Assert-Equal $purePeakHashes.Count 1 `
  'Pure Q6 prefix-FR unique output identity count'
Assert-Equal $purePeak.output_sha256 $pureOutputSha256 `
  'Pure Q6 prefix-FR deterministic output identity'
Assert-Equal @($purePeak.samples | Where-Object {
    $_.completion_tokens -ne 1024
  }).Count 0 'Pure Q6 prefix-FR short output count'

$pureFull = Get-Content -Raw -LiteralPath $pureFullReportPath |
  ConvertFrom-Json
$pureFullRates = @($pureFull.samples.decode_tokens_per_second | Sort-Object)
$pureFullEndToEnd = @(
  $pureFull.samples.end_to_end_tokens_per_second | Sort-Object
)
$pureFullHashes = @($pureFull.samples.output_sha256 | Sort-Object -Unique)

Assert-Equal $pureFull.schema 'a3s.power.speculative-benchmark.v1' `
  'Pure Q6 full-vocabulary report schema'
Assert-Equal $pureFull.identity.power_commit $pureCommit `
  'Pure Q6 full-vocabulary source revision'
Assert-Equal $pureFull.identity.model_sha256 $pureModelSha256 `
  'Pure Q6 full-vocabulary model identity'
Assert-Equal ([long]$pureFull.identity.model_bytes) ([long]22884408288) `
  'Pure Q6 full-vocabulary model byte length'
Assert-Equal ([int]$pureFull.identity.speculative.draft_max) 7 `
  'Pure Q6 full-vocabulary draft width'
Assert-Equal ([int]$pureFull.identity.speculative.mtp_recurrent_snapshots) 7 `
  'Pure Q6 full-vocabulary recurrent snapshots'
Assert-Equal ([bool]$pureFull.identity.speculative.mtp_recurrent_chain) $false `
  'Pure Q6 full-vocabulary recurrent chain'
Assert-Equal ([int]$pureFull.workload.max_tokens) 1024 `
  'Pure Q6 full-vocabulary generated-token count'
Assert-Equal ([int]$pureFull.workload.num_batch) 14 `
  'Pure Q6 full-vocabulary batch size'
Assert-Equal $pureFullRates.Count 9 'Pure Q6 full-vocabulary sample count'
Assert-Near ([double]$pureFullRates[4]) 147.020656574707 `
  'Pure Q6 full-vocabulary median steady decode'
Assert-Near ([double]$pureFullRates[0]) 146.09169791727078 `
  'Pure Q6 full-vocabulary minimum steady decode'
Assert-Near ([double]$pureFullEndToEnd[4]) 140.25733769822205 `
  'Pure Q6 full-vocabulary median end-to-end throughput'
Assert-Equal $pureFullHashes.Count 1 `
  'Pure Q6 full-vocabulary unique output identity count'
Assert-Equal $pureFull.output_sha256 $pureOutputSha256 `
  'Pure Q6 full-vocabulary deterministic output identity'
Assert-Equal @($pureFull.samples | Where-Object {
    $_.completion_tokens -ne 1024
  }).Count 0 'Pure Q6 full-vocabulary short output count'

$purePeakEnvironment = Get-Content -Raw `
  -LiteralPath $purePeakEnvironmentPath | ConvertFrom-Json
$pureFullEnvironment = Get-Content -Raw `
  -LiteralPath $pureFullEnvironmentPath | ConvertFrom-Json
foreach ($environment in @($purePeakEnvironment, $pureFullEnvironment)) {
  Assert-Equal $environment.schema `
    'a3s.power.speculative-benchmark.environment.v1' `
    'Pure Q6 performance environment schema'
  Assert-Equal $environment.power_commit $pureCommit `
    'Pure Q6 performance environment revision'
  Assert-Equal ([bool]$environment.dirty_worktree) $false `
    'Pure Q6 performance clean-worktree state'
  Assert-Equal $environment.model_sha256 $pureModelSha256 `
    'Pure Q6 performance environment model identity'
  Assert-Equal $environment.server.sha256 `
    'c6ba312db786b45d81e8feaa286df7793ad2f072a81e4d0ab37ad39756ec95fa' `
    'Pure Q6 performance server identity'
  Assert-Equal $environment.benchmark_client.sha256 `
    'd1b5760849f31d5d9b76e64548dd85110db3b961cd032eb818917d88e4d452da' `
    'Pure Q6 performance client identity'
  Assert-Equal $environment.process_affinity.requested_mask '0x55555' `
    'Pure Q6 requested processor affinity'
  Assert-Equal $environment.process_affinity.effective_mask '0x55555' `
    'Pure Q6 effective processor affinity'
  Assert-Equal ([int]$environment.gpu.clock_lock_mhz) 2745 `
    'Pure Q6 requested GPU clock lock'
}
Assert-Equal $purePeakEnvironment.config.sha256 `
  '9b1213df972ea3731010a1fa72b0d553ba73da42f31e92eaa4fecd3156cbf2ef' `
  'Pure Q6 prefix-FR environment configuration identity'
Assert-Equal $purePeakEnvironment.report.sha256 `
  '26ae70f1606a57e562acdefa2cfd5679246e47c7ecbcd103b06a6e9edfbe790c' `
  'Pure Q6 prefix-FR environment report identity'
Assert-Equal $pureFullEnvironment.config.sha256 `
  'eb445101c1e33a035c9b1d120fec12d9b21e6ce1b2fe5486ad46bee52878a588' `
  'Pure Q6 full-vocabulary environment configuration identity'
Assert-Equal $pureFullEnvironment.report.sha256 `
  '714bb2dcebd79ed09e23f4a9735ef0f435090e140b81b92157e2262023719e24' `
  'Pure Q6 full-vocabulary environment report identity'

$pureCalibration = Get-Content -Raw -LiteralPath $pureCalibrationPath |
  ConvertFrom-Json
$pureCalibrationEnvironment = Get-Content -Raw `
  -LiteralPath $pureCalibrationEnvironmentPath | ConvertFrom-Json
$pureCalibrationModes = @($pureCalibration.modes.psobject.Properties)

Assert-Equal $pureCalibration.schema 'a3s.power.quality-eval.sweep.v1' `
  'Pure Q6 calibration schema'
Assert-Equal ([int]$pureCalibration.repetitions) 1 `
  'Pure Q6 calibration repetition count'
Assert-Equal $pureCalibrationModes.Count 3 `
  'Pure Q6 calibration mode count'
Assert-Equal $pureCalibrationEnvironment.schema `
  'a3s.power.mtp-sweep.environment.v1' `
  'Pure Q6 calibration environment schema'
Assert-Equal $pureCalibrationEnvironment.identity.power_commit $pureCommit `
  'Pure Q6 calibration source revision'
Assert-Equal $pureCalibrationEnvironment.identity.model_sha256 $pureModelSha256 `
  'Pure Q6 calibration model identity'
Assert-Equal ([long]$pureCalibrationEnvironment.identity.model_bytes) `
  ([long]22884408288) 'Pure Q6 calibration model byte length'
Assert-Equal ([bool]$pureCalibrationEnvironment.dirty_worktree) $false `
  'Pure Q6 calibration clean-worktree state'
Assert-Equal ([bool]$pureCalibrationEnvironment.model_file_hash_verified) $true `
  'Pure Q6 calibration model hash verification'

$pureOff = $pureCalibration.modes.'off-b14'
$pureFullCalibration = $pureCalibration.modes.'frfull-k7-s6-b14-fixed'
$pureFrCalibration = $pureCalibration.modes.'fr8192-k7-s6-b14-fixed'
foreach ($mode in @($pureOff, $pureFullCalibration, $pureFrCalibration)) {
  Assert-Equal $mode.model_sha256 $pureModelSha256 `
    'Pure Q6 calibration mode model identity'
  Assert-Equal ([int]$mode.request.num_batch) 14 `
    'Pure Q6 calibration batch size'
  Assert-Equal ([int]$mode.request.max_tokens_cap) 128 `
    'Pure Q6 calibration output cap'
  Assert-Equal ([int]$mode.task_count) 12 `
    'Pure Q6 calibration task count'
  Assert-Equal ([int]$mode.prediction_stable_tasks) 12 `
    'Pure Q6 calibration stable prediction count'
  Assert-Equal ([int]$mode.runs[0].summary.overall.completed) 12 `
    'Pure Q6 calibration completed task count'
  Assert-Equal ([int]$mode.runs[0].summary.overall.errors) 0 `
    'Pure Q6 calibration error count'
  Assert-Equal ([int]$mode.runs[0].summary.overall.truncated) 11 `
    'Pure Q6 calibration truncated task count'
}
Assert-Near ([double]$pureOff.aggregate_completion_tokens_per_second.mean) `
  29.712723837098697 'Pure Q6 autoregressive request-wide throughput'
Assert-Equal ([int]$pureOff.runs[0].summary.overall.correct) 4 `
  'Pure Q6 autoregressive lenient score'
Assert-Equal ([int]$pureOff.runs[0].summary.overall.strict_correct) 3 `
  'Pure Q6 autoregressive strict score'
Assert-Near `
  ([double]$pureFullCalibration.aggregate_completion_tokens_per_second.mean) `
  47.03236986836804 'Pure Q6 full-vocabulary request-wide throughput'
Assert-Near `
  ([double]$pureFullCalibration.speculative_runtime.weighted_acceptance_rate.mean) `
  0.5230414746543779 'Pure Q6 full-vocabulary proposal acceptance'
Assert-Equal ([int]$pureFullCalibration.runs[0].summary.overall.correct) 5 `
  'Pure Q6 full-vocabulary lenient score'
Assert-Equal ([int]$pureFullCalibration.runs[0].summary.overall.strict_correct) 3 `
  'Pure Q6 full-vocabulary strict score'
Assert-Near `
  ([double]$pureFrCalibration.aggregate_completion_tokens_per_second.mean) `
  37.29003139316878 'Pure Q6 prefix-FR request-wide throughput'
Assert-Near `
  ([double]$pureFrCalibration.speculative_runtime.weighted_acceptance_rate.mean) `
  0.24824880919024936 'Pure Q6 prefix-FR proposal acceptance'
Assert-Equal ([int]$pureFrCalibration.runs[0].summary.overall.correct) 4 `
  'Pure Q6 prefix-FR lenient score'
Assert-Equal ([int]$pureFrCalibration.runs[0].summary.overall.strict_correct) 3 `
  'Pure Q6 prefix-FR strict score'

$deepCommit = 'f6326bb05bb8101c2335ec7c3c2f1e261fd86071'
$deepServerSha256 =
  'a2b1ef3eab435dca02ca6dc41415f21c91c0f84d424ebfd0c7c589a992c555cc'
$deepPeak = Get-Content -Raw -LiteralPath $deepPeakReportPath |
  ConvertFrom-Json
$deepPeakEnvironment = Get-Content -Raw `
  -LiteralPath $deepPeakEnvironmentPath | ConvertFrom-Json
$deepPeakPreflight = Get-Content -Raw `
  -LiteralPath $deepPeakPreflightPath | ConvertFrom-Json
$deepPeakRates = @($deepPeak.samples.decode_tokens_per_second | Sort-Object)
$deepPeakEndToEnd = @(
  $deepPeak.samples.end_to_end_tokens_per_second | Sort-Object
)
$deepPeakHashes = @($deepPeak.samples.output_sha256 | Sort-Object -Unique)

Assert-Equal $deepPeak.schema 'a3s.power.speculative-benchmark.v1' `
  'Deep-optimization peak report schema'
Assert-Equal $deepPeak.identity.power_commit $deepCommit `
  'Deep-optimization peak source revision'
Assert-Equal $deepPeak.identity.model_sha256 $pureModelSha256 `
  'Deep-optimization peak model identity'
Assert-Equal ([long]$deepPeak.identity.model_bytes) ([long]22884408288) `
  'Deep-optimization peak model byte length'
Assert-Equal ([int]$deepPeak.identity.speculative.draft_max) 7 `
  'Deep-optimization peak draft width'
Assert-Equal `
  ([int]$deepPeak.identity.speculative.mtp_recurrent_snapshots) 6 `
  'Deep-optimization peak recurrent snapshots'
Assert-Equal ([int]$deepPeak.identity.speculative.mtp_fr_vocab_size) 8192 `
  'Deep-optimization peak FR row limit'
Assert-Equal ([bool]$deepPeak.identity.inference.flash_attention) $false `
  'Deep-optimization peak Flash Attention mode'
Assert-Equal ([int]$deepPeak.workload.num_batch) 11 `
  'Deep-optimization peak batch size'
Assert-Equal $deepPeakRates.Count 9 'Deep-optimization peak sample count'
Assert-Near ([double]$deepPeakRates[4]) 172.8353133057359 `
  'Deep-optimization peak median steady decode'
Assert-Near ([double]$deepPeakRates[0]) 171.29810355919784 `
  'Deep-optimization peak minimum steady decode'
Assert-Near ([double]$deepPeakEndToEnd[4]) 162.7538614545675 `
  'Deep-optimization peak median end-to-end throughput'
Assert-Equal @($deepPeakRates | Where-Object { $_ -ge 175 }).Count 1 `
  'Deep-optimization peak samples at or above 175 token/s'
Assert-Equal $deepPeakHashes.Count 1 `
  'Deep-optimization peak unique output identity count'
Assert-Equal $deepPeak.output_sha256 $pureOutputSha256 `
  'Deep-optimization peak deterministic output identity'

foreach ($receipt in @($deepPeakEnvironment, $deepPeakPreflight)) {
  Assert-Equal $receipt.power_commit $deepCommit `
    'Deep-optimization peak receipt revision'
  Assert-Equal ([bool]$receipt.dirty_worktree) $false `
    'Deep-optimization peak clean-worktree state'
  Assert-Equal $receipt.server.sha256 $deepServerSha256 `
    'Deep-optimization peak server identity'
  Assert-Equal $receipt.config.sha256 `
    '674d3a36e0f0019c9e39e60994ea40eee0477615827464edee1fb9627a74cdec' `
    'Deep-optimization peak configuration identity'
  Assert-Equal ([bool]$receipt.gpu.cuda_high_priority) $true `
    'Deep-optimization peak CUDA stream priority'
  Assert-Equal ([int]$receipt.gpu.maximum_observed_idle_utilization_percent) 7 `
    'Deep-optimization peak observed idle utilization'
}
Assert-Equal $deepPeakEnvironment.process_affinity.requested_mask '0x55555' `
  'Deep-optimization peak requested processor affinity'
Assert-Equal $deepPeakEnvironment.process_affinity.effective_mask '0x55555' `
  'Deep-optimization peak effective processor affinity'
Assert-Equal ([int]$deepPeakEnvironment.gpu.clock_lock_mhz) 2745 `
  'Deep-optimization peak GPU clock lock'

$deepQualityEnvironment = Get-Content -Raw `
  -LiteralPath $deepQualityEnvironmentPath | ConvertFrom-Json
$deepSweep = Get-Content -Raw -LiteralPath $deepSweepPath | ConvertFrom-Json
$deepOffReport = Get-Content -Raw -LiteralPath $deepOffReportPath |
  ConvertFrom-Json
$deepMtpReport = Get-Content -Raw -LiteralPath $deepMtpReportPath |
  ConvertFrom-Json
$deepOffMode = $deepSweep.modes.'off-b8'
$deepMtpMode = $deepSweep.modes.'fr8192-k6-s6-b8-fixed'

Assert-Equal $deepQualityEnvironment.identity.power_commit $deepCommit `
  'Deep-optimization quality source revision'
Assert-Equal $deepQualityEnvironment.identity.server_sha256 $deepServerSha256 `
  'Deep-optimization quality server identity'
Assert-Equal $deepQualityEnvironment.identity.model_sha256 $pureModelSha256 `
  'Deep-optimization quality model identity'
Assert-Equal ([bool]$deepQualityEnvironment.dirty_worktree) $false `
  'Deep-optimization quality clean-worktree state'
Assert-Equal ([bool]$deepQualityEnvironment.model_file_hash_verified) $true `
  'Deep-optimization quality model hash verification'
Assert-Equal $deepQualityEnvironment.identity.process_affinity_mask '0x55555' `
  'Deep-optimization quality processor affinity'
Assert-Equal ([bool]$deepQualityEnvironment.identity.cuda_high_priority) $true `
  'Deep-optimization quality CUDA stream priority'
Assert-Equal $deepQualityEnvironment.identity.config_sha256 `
  'b4f3db4229bfad05371bbed0ce1fec165aa2b05279405078aa8f7721721abb37' `
  'Deep-optimization quality configuration identity'
Assert-Equal ([int]$deepQualityEnvironment.identity.max_tokens_cap) 256 `
  'Deep-optimization quality output cap'
Assert-Equal @($deepQualityEnvironment.identity.modes).Count 2 `
  'Deep-optimization quality mode count'

foreach ($mode in @($deepOffMode, $deepMtpMode)) {
  Assert-Equal $mode.model_sha256 $pureModelSha256 `
    'Deep-optimization quality mode model identity'
  Assert-Equal ([int]$mode.request.num_batch) 8 `
    'Deep-optimization quality batch size'
  Assert-Equal ([int]$mode.request.max_tokens_cap) 256 `
    'Deep-optimization quality request output cap'
  Assert-Equal ([int]$mode.task_count) 12 `
    'Deep-optimization quality task count'
  Assert-Equal ([int]$mode.runs[0].summary.overall.completed) 12 `
    'Deep-optimization quality completed task count'
  Assert-Equal ([int]$mode.runs[0].summary.overall.errors) 0 `
    'Deep-optimization quality error count'
  Assert-Equal ([int]$mode.runs[0].summary.overall.correct) 9 `
    'Deep-optimization quality lenient score'
  Assert-Equal ([int]$mode.runs[0].summary.overall.strict_correct) 9 `
    'Deep-optimization quality strict score'
  Assert-Equal ([int]$mode.runs[0].summary.overall.truncated) 3 `
    'Deep-optimization quality truncated task count'
}
Assert-Near ([double]$deepOffMode.aggregate_completion_tokens_per_second.mean) `
  28.71272184998198 'Deep-optimization target-only throughput'
Assert-Near ([double]$deepMtpMode.aggregate_completion_tokens_per_second.mean) `
  46.92338764288924 'Deep-optimization MTP throughput'
Assert-Near `
  ([double]$deepMtpMode.speculative_runtime.weighted_acceptance_rate.mean) `
  0.26814756049186833 'Deep-optimization MTP proposal acceptance'
Assert-Near `
  ([double]$deepMtpMode.speculative_runtime.verified_tokens_per_target_pass.mean) `
  2.5912322274881516 'Deep-optimization verified tokens per target pass'
Assert-Near ([double]$deepMtpMode.speculative_runtime.fallback_replays.mean) `
  0.0 'Deep-optimization MTP fallback replays'

$samePredictions = 0
$sameContent = 0
foreach ($offResult in @($deepOffReport.results)) {
  $mtpResult = @($deepMtpReport.results | Where-Object {
      $_.id -ceq $offResult.id
    })
  Assert-Equal $mtpResult.Count 1 `
    "Deep-optimization paired result $($offResult.id)"
  if ($offResult.prediction -ceq $mtpResult[0].prediction) {
    $samePredictions++
  }
  if ($offResult.content_sha256 -ceq $mtpResult[0].content_sha256) {
    $sameContent++
  }
}
Assert-Equal $samePredictions 12 `
  'Deep-optimization paired final-answer identity count'
Assert-Equal $sameContent 8 `
  'Deep-optimization paired full-content identity count'

$completedRequests = [int]$current.workload.tasks *
  [int]$current.workload.repetitions * $currentModes.Count
$result = [ordered]@{
  schema = 'a3s.power.evidence-verification.v1'
  status = 'passed'
  evidence_root = $resolvedEvidenceRoot
  verified_file_hashes = $expectedHashes.Count
  quality = [ordered]@{
    completed_requests = $completedRequests
    lenient_score = [int]$s7.lenient_score
    strict_score = [int]$s7.strict_score
    request_wide_tokens_per_second = [double]$s7.mean_workload_tokens_per_second
    weighted_acceptance_rate = [double]$s7.weighted_acceptance_rate
    fallback_replays = 0
  }
  steady_decode = [ordered]@{
    rollback_complete_k7_s7_median = `
      [double]$s7Gate.median_decode_tokens_per_second
    rollback_complete_k7_s7_minimum = `
      [double]$s7Gate.minimum_decode_tokens_per_second
    guarded_k7_s6_median = [double]$s6Gate.median_decode_tokens_per_second
    guarded_k7_s6_minimum = [double]$s6Gate.minimum_decode_tokens_per_second
  }
  pure_q6 = [ordered]@{
    model_sha256 = $pureModelSha256
    model_bytes = 22884408288
    full_vocabulary_k7_s7_median = [double]$pureFullRates[4]
    full_vocabulary_k7_s7_minimum = [double]$pureFullRates[0]
    prefix_fr8192_k7_s6_median = [double]$purePeakRates[4]
    prefix_fr8192_k7_s6_minimum = [double]$purePeakRates[0]
    prefix_fr8192_samples_at_or_above_175 = @(
      $purePeakRates | Where-Object { $_ -ge 175 }
    ).Count
    prefix_fr8192_speedup_percent =
      (($purePeakRates[4] / $pureFullRates[4]) - 1.0) * 100.0
    calibration = [ordered]@{
      autoregressive_tokens_per_second =
        [double]$pureOff.aggregate_completion_tokens_per_second.mean
      full_vocabulary_tokens_per_second =
        [double]$pureFullCalibration.aggregate_completion_tokens_per_second.mean
      prefix_fr8192_tokens_per_second =
        [double]$pureFrCalibration.aggregate_completion_tokens_per_second.mean
      truncated_tasks_per_mode = 11
    }
  }
  deep_optimization = [ordered]@{
    power_commit = $deepCommit
    peak = [ordered]@{
      median_decode_tokens_per_second = [double]$deepPeakRates[4]
      minimum_decode_tokens_per_second = [double]$deepPeakRates[0]
      samples_at_or_above_175 = @(
        $deepPeakRates | Where-Object { $_ -ge 175 }
      ).Count
      cuda_high_priority = $true
      observed_idle_gpu_utilization_percent = 7
    }
    general = [ordered]@{
      target_only_tokens_per_second =
        [double]$deepOffMode.aggregate_completion_tokens_per_second.mean
      mtp_tokens_per_second =
        [double]$deepMtpMode.aggregate_completion_tokens_per_second.mean
      speedup_percent = (
        ([double]$deepMtpMode.aggregate_completion_tokens_per_second.mean /
          [double]$deepOffMode.aggregate_completion_tokens_per_second.mean) - 1.0
      ) * 100.0
      weighted_acceptance_rate =
        [double]$deepMtpMode.speculative_runtime.weighted_acceptance_rate.mean
      verified_tokens_per_target_pass =
        [double]$deepMtpMode.speculative_runtime.verified_tokens_per_target_pass.mean
      paired_final_answers = $samePredictions
      paired_full_content = $sameContent
      fallback_replays = 0
    }
  }
  deterministic_output_sha256 = $s7Gate.output_sha256
}

if ($Json) {
  $result | ConvertTo-Json -Depth 5
} else {
  [pscustomobject]$result
}
