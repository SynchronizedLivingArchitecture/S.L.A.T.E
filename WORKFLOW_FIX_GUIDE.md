# SLATE Workflow Fix Guide
# Modified: 2026-02-07T07:20:00Z | Author: COPILOT | Change: Create comprehensive workflow fix guide

## Problem Summary

Workflows fail when the self-hosted runner cannot access local AI services (Ollama)
running on `127.0.0.1:11434`. Common symptoms:

- `Ollama: NOT RUNNING` errors in agentic workflow
- Connection refused to `127.0.0.1:11434`
- Inference timeouts in CI/CD jobs
- 164+ accumulated failed workflow runs

## Root Causes

| Cause | Symptom | Fix |
|-------|---------|-----|
| Ollama not running | `Connection refused` | Start Ollama service |
| Missing env vars | Runner can't find Ollama | Update `.env` and pre-job hook |
| Runner not using hooks | Pre-job vars not set | Verify `hooks.json` config |
| GPU not accessible | CUDA errors | Check `CUDA_VISIBLE_DEVICES` |
| Runner service crashed | Jobs stay queued | Restart runner process |

## Quick Fix (Automated)

### Option 1: Python Diagnostic + Fix
```powershell
# From workspace root
.\.venv\Scripts\python.exe fix_workflow_ai_access.py --fix
```

### Option 2: PowerShell Fix Script
```powershell
.\fix-runner-ai-access.ps1
```

### Option 3: Batch File (double-click)
```
fix-runner-ai-access.bat
```

## Manual Fix Steps

### Step 1: Verify Ollama
```powershell
# Check if Ollama is responding
Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5

# If not running, start it
ollama serve

# Verify models are loaded
ollama list
```

Expected models:
- `llama3.2:3b` - Fast inference
- `mistral-nemo` / `mistral` - General purpose
- `slate-coder` - 12B code generation (custom)
- `slate-fast` - 3B classification (custom)
- `slate-planner` - 7B planning (custom)

### Step 2: Verify Runner Environment

Check that `actions-runner/.env` contains:
```
OLLAMA_HOST=127.0.0.1:11434
SLATE_OLLAMA_URL=http://127.0.0.1:11434
CUDA_VISIBLE_DEVICES=0,1
SLATE_GPU_COUNT=2
SLATE_WORKSPACE=<your-workspace-path>
PYTHONPATH=<your-workspace-path>
PYTHONIOENCODING=utf-8
SLATE_RUNNER=true
```

### Step 3: Verify Pre-Job Hook

Check `actions-runner/hooks/pre-job.ps1` contains:
```powershell
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:SLATE_OLLAMA_URL = "http://127.0.0.1:11434"
$env:CUDA_VISIBLE_DEVICES = "0,1"
```

Check `actions-runner/hooks/hooks.json` points to the hook:
```json
{
  "pre_job": {
    "path": "<workspace>\\actions-runner\\hooks\\pre-job.ps1",
    "args": []
  }
}
```

### Step 4: Restart Runner
```powershell
cd <workspace>\actions-runner

# If running as service
.\svc.cmd stop
.\svc.cmd start

# If running interactively
# Stop the running process (Ctrl+C)
.\run.cmd
```

### Step 5: Test AI Access
```powershell
# Quick inference test
.\.venv\Scripts\python.exe -c @"
import urllib.request, json, time
payload = json.dumps({
    'model': 'llama3.2:3b',
    'prompt': 'Say OK',
    'stream': False,
    'options': {'num_predict': 5}
}).encode()
req = urllib.request.Request(
    'http://127.0.0.1:11434/api/generate',
    data=payload,
    headers={'Content-Type': 'application/json'}
)
start = time.time()
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read())
print(f'Response: {data.get("response", "")[:50]}')
print(f'Tokens: {data.get("eval_count", 0)}')
print(f'Time: {time.time()-start:.1f}s')
"@
```

### Step 6: Dispatch Test Workflow
```powershell
# Run SLATE system status via CI
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --dispatch "agentic.yml"

# Or dispatch the new git intelligence workflow
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --dispatch "git-intelligence.yml"
```

## Workflow Architecture

### AI-Enabled Workflows

| Workflow | File | AI Usage |
|----------|------|----------|
| Agentic AI | `agentic.yml` | Autonomous task execution, inference benchmarks |
| Git Intelligence | `git-intelligence.yml` | Git history analysis, failure patterns, security scan |
| Fork Intelligence | `fork-intelligence.yml` | AI-powered fork analysis |

### Workflow → AI Service Flow
```
GitHub Actions Runner
  ├── pre-job.ps1 (sets OLLAMA_HOST, CUDA_VISIBLE_DEVICES)
  ├── Job Step: python slate/... 
  │   └── urllib.request → http://127.0.0.1:11434/api/generate
  │       └── Ollama → GPU 0/1 → Response
  └── Post-job cleanup
```

### GPU Model Assignment
```
GPU 0 (RTX 5070 Ti, 16GB):
  - slate-coder (12B) — code generation
  - slate-planner (7B) — planning/analysis
  - mistral-nemo (12B) — general

GPU 1 (RTX 5070 Ti, 16GB):
  - slate-fast (3B) — fast classification
  - llama3.2:3b — fast inference
  - phi — small model
```

## Monitoring

### Check Workflow Status
```powershell
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --status
```

### Check Recent Failures
```powershell
.\.venv\Scripts\python.exe _temp\check_failures.py
```

### Full Diagnostic
```powershell
.\.venv\Scripts\python.exe fix_workflow_ai_access.py
```

## Prevention

The pre-job hook (`actions-runner/hooks/pre-job.ps1`) now includes:
1. Ollama connectivity verification
2. GPU availability check 
3. Python environment validation
4. All required environment variables

This runs automatically before every GitHub Actions job on this runner.
