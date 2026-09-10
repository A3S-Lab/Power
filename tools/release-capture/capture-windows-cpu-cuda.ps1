# Capture exact-parent Windows CPU + CUDA release fixtures for a frozen Power tip.
#
# Usage (from a clean Power checkout of the intended source parent):
#   powershell -File tools/release-capture/capture-windows-cpu-cuda.ps1 `
#     -OutputRoot D:\captures\a3s-power-<shortsha>
#
# Requires: clean git tree, rustc, CUDA toolkit, and an x64 VS developer
# environment for the CUDA step (cl.exe on PATH). Does not claim Metal or
# confidential-GPU evidence.

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

if (-not $PowerCommit) {
    $PowerCommit = (git rev-parse HEAD).Trim()
}
if ($PowerCommit.Length -ne 40) {
    throw "power commit must be a 40-character lowercase SHA"
}
if ($PowerCommit -cne $PowerCommit.ToLowerInvariant()) {
    throw "power commit must be lowercase"
}

$head = (git rev-parse HEAD).Trim()
if ($head -ne $PowerCommit) {
    throw "HEAD ($head) does not match -PowerCommit ($PowerCommit); use a detached checkout of the source parent"
}

if (-not (Test-Path $PolicyPath)) {
    throw "missing policy file: $PolicyPath"
}
$policyHash = (Get-FileHash $PolicyPath -Algorithm SHA256).Hash.ToLowerInvariant()

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

Write-Host "Capturing CPU against $PowerCommit ..."
cargo run --locked --release --no-default-features `
    --features embedded-inference `
    --bin a3s-power-tensor-batch-bench -- release-fixture `
    --output (Join-Path $OutputRoot "cpu.json") `
    --device cpu `
    --power-commit $PowerCommit `
    --filesystem-class ntfs `
    --device-class "Intel Xeon w5-2445 CPU" `
    --cpu-model $CpuModel `
    --ram-bytes $RamBytes `
    --tee-policy-sha256 $policyHash `
    --host-fixed-bytes 67108864 `
    --host-scratch-bytes 67108864 `
    --device-fixed-bytes 0 `
    --device-scratch-bytes 0 `
    --items 8 --width 4096 `
    --warmup-rounds 2 --measured-rounds 9

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "CUDA capture requires cl.exe on PATH (open an x64 VS developer shell)"
}

Write-Host "Capturing CUDA against $PowerCommit ..."
cargo run --locked --release --no-default-features `
    --features embedded-cuda `
    --bin a3s-power-tensor-batch-bench -- release-fixture `
    --output (Join-Path $OutputRoot "cuda.json") `
    --device cuda:0 `
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

if (git status --porcelain) {
    throw "capture worktree became dirty"
}

Write-Host "Wrote CPU and CUDA captures under $OutputRoot"
Write-Host "Still required for production: Metal + SEV-SNP confidential-GPU on the same parent."
