# Prerequis : Python 3.9 x64 installe depuis python.org, lanceur py.exe.
# Les fichiers sont resolus depuis leur emplacement, pas depuis $HOME.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host 'AI-Doktora / Windows / Python 3.9'
& py -3.9 -c "import sys; assert sys.version_info[:2] == (3,9), sys.version"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.9 x64 introuvable. Installer Python 3.9.' }
& py -3.9 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Creation venv echouee.' }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Mise a jour pip echouee.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Installation dependances echouee.' }
& .\.venv\Scripts\python.exe -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw 'Tests automatiques echoues.' }
Write-Host 'Installation terminee. Lire README.md, puis executer win\nasa.ps1'
