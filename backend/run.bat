@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\uvicorn.exe" (
  echo Virtualenv missing. Run:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)
echo Starting backend at http://127.0.0.1:8000
.venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8000
