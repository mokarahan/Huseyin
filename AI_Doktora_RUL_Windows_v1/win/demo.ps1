$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw 'Installer les dependances.' }
& $py main.py demo
if ($LASTEXITCODE -ne 0) { throw 'Dossier data\processed deja utilise ? Ne pas ecraser NASA.' }
& $py main.py train --model hybrid --epochs 2 --test B0005 --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Demo echouee.' }
