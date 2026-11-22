# Install meetnote: creates a private venv and a meetnote.cmd shim in ~\.local\bin,
# so `meetnote` works from any terminal. (ASCII only: Windows PowerShell 5.1 misreads UTF-8 without BOM.)
# Not "Stop": PS 5.1 would treat pip's stderr warnings as fatal; we check $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$venv = Join-Path $root ".venv"

if (-not (Test-Path $venv)) {
    Write-Host "Creating venv at $venv"
    py -3 -m venv $venv
}
$py = Join-Path $venv "Scripts\python.exe"
& $py -m pip install --upgrade pip -q
& $py -m pip install -e $root
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $root ".env.example") $envFile
    Write-Host "Created $envFile - fill in DEEPSEEK_API_KEY there."
}

$binDir = Join-Path $env:USERPROFILE ".local\bin"
New-Item -ItemType Directory -Force $binDir | Out-Null
$shim = Join-Path $binDir "meetnote.cmd"
"@echo off`r`n`"$venv\Scripts\meetnote.exe`" %*" | Set-Content -Encoding ascii $shim
Write-Host "Created $shim"

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ";") -notcontains $binDir) {
    Write-Host "Note: $binDir is not on your user PATH. Add it, or call $shim directly."
}
Write-Host "Done. Try: meetnote --help"
