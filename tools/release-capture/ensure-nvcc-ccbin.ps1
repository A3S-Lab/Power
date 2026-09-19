# Ensure NVCC can host-compile on Windows VS Build Tools without a nested
# VsDevCmd environment (nested vcvars fails with "command too long").
#
# Prefer an existing NVCC_CCBIN. Otherwise create/use a space-free junction at
# C:\vsbt -> VS 2022 BuildTools and point NVCC_CCBIN at Hostx64\x64.
#
# Dot-source from CUDA capture scripts before `cargo ... --features embedded-cuda`.

function Add-A3SPowerCcbinToPath {
    param([Parameter(Mandatory = $true)][string]$Ccbin)
    $needle = $Ccbin.TrimEnd('\')
    $present = $false
    foreach ($part in ($env:Path -split ';')) {
        if ($part.TrimEnd('\') -eq $needle) {
            $present = $true
            break
        }
    }
    if (-not $present) {
        $env:Path = "$Ccbin;$env:Path"
    }
}

function Ensure-A3SPowerNvccCcbIn {
    if ($env:NVCC_CCBIN -and (Test-Path -LiteralPath (Join-Path $env:NVCC_CCBIN "cl.exe"))) {
        Add-A3SPowerCcbinToPath -Ccbin $env:NVCC_CCBIN
        Write-Host "Using existing NVCC_CCBIN=$env:NVCC_CCBIN"
        return
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw "vswhere.exe not found; install VS 2022 Build Tools with MSVC x64"
    }

    $vsRoot = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsRoot) {
        throw "VS 2022 MSVC x64 tools not found"
    }

    $msvcRoot = Join-Path $vsRoot "VC\Tools\MSVC"
    $hostDir = Get-ChildItem -LiteralPath $msvcRoot -Directory |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "bin\Hostx64\x64" } |
        Where-Object { Test-Path -LiteralPath (Join-Path $_ "cl.exe") } |
        Select-Object -First 1
    if (-not $hostDir) {
        throw "cl.exe (Hostx64\\x64) not found under $msvcRoot"
    }

    # Space-free junction so nvcc's relative vcvars64.bat lookup succeeds.
    $junction = "C:\vsbt"
    if (-not (Test-Path -LiteralPath $junction)) {
        cmd /c "mklink /J `"$junction`" `"$vsRoot`"" | Out-Host
    }

    $rel = $hostDir.Substring($vsRoot.Length).TrimStart('\')
    $ccbin = Join-Path $junction $rel
    if (-not (Test-Path -LiteralPath (Join-Path $ccbin "cl.exe"))) {
        throw "junction ccbin missing cl.exe: $ccbin"
    }

    $env:NVCC_CCBIN = $ccbin
    Add-A3SPowerCcbinToPath -Ccbin $ccbin
    Write-Host "Set NVCC_CCBIN=$env:NVCC_CCBIN"
    Write-Host "Do not nest an outer VsDevCmd before cargo embedded-cuda; let nvcc call vcvars once."
}
