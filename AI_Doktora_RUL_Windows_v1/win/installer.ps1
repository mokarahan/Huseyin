# Prerequisites: Python 3.9 x64 installed from python.org, py.exe launcher.
# Files are resolved relative to their own location, not from $HOME.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host 'AI-Doktora / Windows / Python 3.9'
& py -3.9 -c "import sys; assert sys.version_info[:2] == (3,9), sys.version"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.9 x64 not found. Install Python 3.9.' }
& py -3.9 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed.' }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& .\.venv\Scripts\python.exe -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw 'Automated tests failed.' }
Write-Host 'Installation complete. Read README.md, then run win\nasa.ps1'
