# Récupère la source officielle NASA, puis prépare les .mat.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw 'Executer win\installer.ps1 avant.' }
& $py main.py download
if ($LASTEXITCODE -ne 0) { throw 'Echec du telechargement NASA. Voir README / mode hors-ligne.' }
& $py main.py prepare
if ($LASTEXITCODE -ne 0) { throw 'Echec preparation : examiner data\processed\audit.json' }
& $py main.py audit
