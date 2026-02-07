---
mode: agent
description: "Run a full SLATE system health check: GPU, services, runner, workflows"
---

You are the SLATE system operator. Run a comprehensive health check.

## Steps
1. Run `E:\11132025\.venv\Scripts\python.exe slate/slate_status.py --quick` to get system status
2. Run `E:\11132025\.venv\Scripts\python.exe slate/slate_runtime.py --check-all` to check all integrations
3. Check the runner status: `E:\11132025\.venv\Scripts\python.exe slate/slate_runner_manager.py --status`
4. Check workflow queue: `E:\11132025\.venv\Scripts\python.exe slate/slate_workflow_manager.py --status`

## Output Format
Summarize findings as a status dashboard:
- **System**: Python version, GPU count, memory
- **Runner**: Online/offline, labels, active jobs
- **Workflows**: Active/queued runs, any failures
- **Issues**: Any warnings or errors found

## Rules
- All commands use the venv Python at `E:\11132025\.venv\Scripts\python.exe`
- Never use `curl.exe` — use Python `urllib.request` for HTTP
- All operations are LOCAL ONLY (127.0.0.1)
