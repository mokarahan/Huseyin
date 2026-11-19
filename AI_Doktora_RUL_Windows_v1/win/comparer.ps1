param([int]$Epochs=100, [int]$Window=10, [int]$Seed=42)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw 'Installer les dependances (win\installer.ps1).' }
& $py main.py compare --epochs $Epochs --window $Window --seed $Seed
if ($LASTEXITCODE -ne 0) { throw 'Comparaison echouee.' }
Write-Host 'Rapports : results\experiments\comparison.csv et comparison_summary.json'
