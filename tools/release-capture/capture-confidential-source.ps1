# Capture a local CUDA confidential-GPU *source* (not promoted).
#
# Output remains ordinary local CUDA evidence plus an accelerator declaration.
# Promotion to confidential-gpu requires real SEV-SNP + NVIDIA NRAS via
# a3s-power-verify --promote-capture on confidential hardware.
#
# Usage (x64 VS developer shell, clean checkout of source parent):
#   powershell -File tools/release-capture/capture-confidential-source.ps1 `
#     -OutputRoot D:\captures\a3s-power-<shortsha>\confidential-source
#
# By default HEAD / -PowerCommit must equal the v1.0.0 freeze parent. Pass
# -AllowAnySourceParent only when intentionally recutting a new evidence parent.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [string]$PowerCommit = "",
    [string]$ExpectedSourceParent = "514031dc74edd72da7c3bfee40144a38d2d91434",
    [switch]$AllowAnySourceParent,
    [string]$PolicyPath = "docs/benchmarks/release-contract-windows-20260910/local-execution-policy.json",
    [string]$DeviceClassCuda = "NVIDIA GeForce RTX 4090 24 GiB; driver 610.74",
    [string]$CpuModel = "Intel(R) Xeon(R) w5-2445",
    [long]$RamBytes = 137071693824
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "ensure-nvcc-ccbin.ps1")
Ensure-A3SPowerNvccCcbIn

if (git status --porcelain) {
    throw "capture requires a clean git worktree"
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "requires cl.exe on PATH (add Hostx64\\x64 or let ensure-nvcc-ccbin.ps1 set NVCC_CCBIN)"
}

if (-not $PowerCommit) {
    $PowerCommit = (git rev-parse HEAD).Trim()
}
$head = (git rev-parse HEAD).Trim()
if ($head -ne $PowerCommit) {
    throw "HEAD does not match -PowerCommit; detach to the source parent first"
}

if (-not $AllowAnySourceParent) {
    if ($ExpectedSourceParent.Length -ne 40) {
        throw "-ExpectedSourceParent must be a 40-character lowercase SHA"
    }
    if ($ExpectedSourceParent -cne $ExpectedSourceParent.ToLowerInvariant()) {
        throw "-ExpectedSourceParent must be lowercase"
    }
    if ($PowerCommit -ne $ExpectedSourceParent) {
        throw "PowerCommit $PowerCommit is not release source parent $ExpectedSourceParent; detach first or pass -AllowAnySourceParent only when recutting"
    }
}

if (-not (Test-Path -LiteralPath $PolicyPath)) {
    throw @"
missing policy file: $PolicyPath
The freeze parent does not contain this file. Export it outside the worktree:
  git show main:docs/benchmarks/release-contract-windows-20260910/local-execution-policy.json > D:\captures\local-execution-policy.json
Then re-run with -PolicyPath D:\captures\local-execution-policy.json from a clean detached freeze-parent worktree.
The script itself may live on a main checkout; do not copy it into the freeze-parent tree.
"@
}
$policyHash = (Get-FileHash -LiteralPath $PolicyPath -Algorithm SHA256).Hash.ToLowerInvariant()
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
