# Capture a local CUDA confidential-GPU *source* (not promoted).
#
# Output remains ordinary local CUDA evidence plus an accelerator declaration.
# Promotion to confidential-gpu requires real SEV-SNP + NVIDIA NRAS via
# a3s-power-verify --promote-capture on confidential hardware.
#
# Usage (x64 VS developer shell, clean checkout of source parent):
#   powershell -File tools/release-capture/capture-confidential-source.ps1 `
#     -OutputRoot D:\captures\a3s-power-<shortsha>\confidential-source

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [string]$PowerCommit = "",
    [string]$PolicyPath = "docs/benchmarks/release-contract-windows-20260910/local-execution-policy.json",
    [string]$DeviceClassCuda = "NVIDIA GeForce RTX 4090 24 GiB; driver 610.74",
    [string]$CpuModel = "Intel(R) Xeon(R) w5-2445",
    [long]$RamBytes = 137071693824
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (git status --porcelain) {
    throw "capture requires a clean git worktree"
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "requires cl.exe on PATH (x64 VS developer shell)"
}

if (-not $PowerCommit) {
    $PowerCommit = (git rev-parse HEAD).Trim()
}
$head = (git rev-parse HEAD).Trim()
if ($head -ne $PowerCommit) {
    throw "HEAD does not match -PowerCommit; detach to the source parent first"
}

$policyHash = (Get-FileHash $PolicyPath -Algorithm SHA256).Hash.ToLowerInvariant()
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$weights = Join-Path $OutputRoot "fixture-weights"

cargo run --locked --release --no-default-features `
    --features embedded-inference `
    --bin a3s-power-tensor-batch-bench -- materialize-release-fixture-weights `
    --directory $weights `
    --width 4096 `
    --output (Join-Path $OutputRoot "fixture-weights.receipt.json")

cargo run --locked --release --no-default-features `
    --features embedded-cuda `
    --bin a3s-power-tensor-batch-bench -- release-confidential-fixture `
    --device cuda:0 `
    --fixture-weights $weights `
    --output (Join-Path $OutputRoot "confidential-source-cuda.json") `
    --accelerator-declaration-output (Join-Path $OutputRoot "accelerator-declaration.json") `
    --power-commit $PowerCommit `
    --filesystem-class ntfs `
    --device-class $DeviceClassCuda `
    --cpu-model $CpuModel `
    --ram-bytes $RamBytes `
    --tee-policy-sha256 $policyHash `
    --host-fixed-bytes 67108864 `
    --host-scratch-bytes 67108864 `
    --device-fixed-bytes 67108864 `
    --device-scratch-bytes 67108864 `
    --items 8 --width 4096 `
    --warmup-rounds 2 --measured-rounds 9

@"
This directory holds a LOCAL CUDA confidential-GPU source capture only.
It is NOT production confidential-gpu evidence until promoted with a real
SEV-SNP report and NVIDIA NRAS verdict on confidential hardware.
Power commit: $PowerCommit
"@ | Set-Content -Path (Join-Path $OutputRoot "NOT-PROMOTED.txt") -Encoding utf8

if (git status --porcelain) {
    throw "capture worktree became dirty"
}

Write-Host "Wrote confidential SOURCE under $OutputRoot (not promoted)."
