@echo off
REM Modified: 2026-02-07T07:10:00Z | Author: COPILOT | Change: Create batch fix script for runner AI access
REM SLATE Runner AI Access Fix (Windows Batch)
REM Ensures self-hosted runner can access local AI services

echo ============================================================
echo   SLATE Runner AI Access Fix (Batch)
echo   %DATE% %TIME%
echo ============================================================
echo.

set WORKSPACE=%~dp0
set WORKSPACE=%WORKSPACE:~0,-1%
set RUNNER_DIR=%WORKSPACE%\actions-runner
set PYTHON=%WORKSPACE%\.venv\Scripts\python.exe

REM ---- Step 1: Check Ollama ----
echo [1/4] Checking Ollama...
"%PYTHON%" -c "import urllib.request, json; r=urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=5); d=json.loads(r.read()); print(f'  OK: {len(d.get(\"models\",[]))} models')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo   FAIL: Ollama not running
    echo   Starting Ollama...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
    "%PYTHON%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=5); print('  OK: Ollama started')" 2>nul
    if %ERRORLEVEL% neq 0 (
        echo   ERROR: Could not start Ollama
    )
)

REM ---- Step 2: Create .env ----
echo.
echo [2/4] Setting up runner .env...
(
echo OLLAMA_HOST=127.0.0.1:11434
echo SLATE_OLLAMA_URL=http://127.0.0.1:11434
echo CUDA_VISIBLE_DEVICES=0,1
echo SLATE_GPU_COUNT=2
echo SLATE_WORKSPACE=%WORKSPACE%
echo PYTHONPATH=%WORKSPACE%
echo PYTHONIOENCODING=utf-8
echo SLATE_RUNNER=true
) > "%RUNNER_DIR%\.env"
echo   OK: .env created/updated

REM ---- Step 3: Run diagnostic ----
echo.
echo [3/4] Running full diagnostic...
"%PYTHON%" "%WORKSPACE%\fix_workflow_ai_access.py" --fix
if %ERRORLEVEL% neq 0 (
    echo   WARN: Some checks failed
)

REM ---- Step 4: GPU check ----
echo.
echo [4/4] GPU status...
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>nul
if %ERRORLEVEL% neq 0 (
    echo   FAIL: nvidia-smi not available
)

echo.
echo ============================================================
echo   Fix Complete
echo   Restart runner: cd %RUNNER_DIR% ^& .\run.cmd
echo ============================================================
pause
