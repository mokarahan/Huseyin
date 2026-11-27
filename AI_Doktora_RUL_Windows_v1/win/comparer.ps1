param([int]$Epochs=100, [int]$Window=10, [int]$Seed=42)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw 'Install the dependencies first (win\installer.ps1).' }
& $py main.py compare --epochs $Epochs --window $Window --seed $Seed
if ($LASTEXITCODE -ne 0) { throw 'Comparison failed.' }
Write-Host 'Reports: results\experiments\comparison.csv and comparison_summary.json'
