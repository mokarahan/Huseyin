$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw 'Install the dependencies first.' }
& $py main.py demo
if ($LASTEXITCODE -ne 0) { throw 'Is the data\processed folder already in use? Do not overwrite NASA data.' }
& $py main.py train --model hybrid --epochs 2 --test B0005 --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Demo failed.' }
