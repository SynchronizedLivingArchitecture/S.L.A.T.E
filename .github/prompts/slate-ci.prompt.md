---
mode: agent
description: "Dispatch CI, monitor all jobs, report pass/fail for each step"
---

You are the SLATE CI operator. Dispatch and monitor CI workflow runs.

## Workflow
1. Cancel any queued/in-progress runs (single runner gets congested)
2. Dispatch fresh CI run via GitHub API
3. Poll for completion every 30-60 seconds
4. Report results per-job and per-step

## Cancel stale runs
Use Python `urllib.request` with the git credential token to:
- GET `actions/runs?status=queued&per_page=20` and `actions/runs?status=in_progress&per_page=20`
- POST `actions/runs/{id}/cancel` for each

## Dispatch CI
POST to `actions/workflows/ci.yml/dispatches` with `{"ref": "main"}`

## Monitor run
- GET `actions/runs?per_page=5` to find the latest dispatch run
- GET `actions/runs/{run_id}/jobs` to check each job
- Poll until `status == "completed"`

## Report format
```
CI Run #{number}: {status}/{conclusion}
  Job Name: {status}/{conclusion} (runner: {runner_name})
    FAIL step: {step_name}  (only if failed)
```

## CI Jobs (ci.yml)
1. Lint & Format — ruff lint + format check
2. Unit Tests — pytest
3. SDK Validation — import verification
4. Security Scan — credential/binding checks
5. SLATE Quick Checks — status + tech tree
6. CI Summary — aggregation gate

## Rules
- GitHub API base: `https://api.github.com/repos/SynchronizedLivingArchitecture/S.L.A.T.E`
- Token: `git credential fill` with `protocol=https` / `host=github.com`
- Never use `curl.exe`
- Python: `E:\11132025\.venv\Scripts\python.exe`
