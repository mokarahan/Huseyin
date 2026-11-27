# Fetch the official NASA source, then prepare the .mat files.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw 'Run win\installer.ps1 first.' }
& $py main.py download
if ($LASTEXITCODE -ne 0) { throw 'NASA download failed. See README / offline mode.' }
& $py main.py prepare
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed: check data\processed\audit.json' }
& $py main.py audit
