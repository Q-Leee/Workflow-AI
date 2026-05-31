# Start WorkFlow AI backend using the project virtualenv (not system Python).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$venvUvicorn = Join-Path $PSScriptRoot ".venv\Scripts\uvicorn.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtualenv not found. Run from backend folder:" -ForegroundColor Red
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $venvUvicorn)) {
    Write-Host "Installing dependencies into .venv ..." -ForegroundColor Yellow
    & $venvPython -m pip install -r requirements.txt
}

Write-Host "Starting backend at http://127.0.0.1:8003 (venv Python)" -ForegroundColor Green
Write-Host "Use port 8003 if 8001/8002 show stale /health (ghost processes)." -ForegroundColor DarkGray
& $venvUvicorn app.main:app --reload --host 127.0.0.1 --port 8003
