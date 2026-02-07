# S.L.A.T.E. Copilot Instructions
# Modified: 2026-02-07T00:30:00Z | Author: COPILOT | Change: Full rewrite with SLATE protocol tooling

## System Overview
SLATE (Synchronized Living Architecture for Transformation and Evolution) is a local-first
AI agent orchestration framework. All operations are LOCAL ONLY (127.0.0.1). Version 2.4.0.

Repository: `SynchronizedLivingArchitecture/S.L.A.T.E`
Python: 3.11+ via `.venv` at `E:\11132025\.venv\Scripts\python.exe`
Runner: Self-hosted GitHub Actions runner `slate-runner` at `E:\11132025\actions-runner`

## Format Rules
All code edits MUST include a timestamp + author comment:
```python
# Modified: YYYY-MM-DDTHH:MM:SSZ | Author: COPILOT | Change: description
```

## SLATE Protocol Commands
```bash
# System status
python slate/slate_status.py --quick          # Quick health check
python slate/slate_status.py --json           # Machine-readable status

# Runtime integration check
python slate/slate_runtime.py --check-all     # All integrations
python slate/slate_runtime.py --json          # JSON output

# Hardware & GPU optimization
python slate/slate_hardware_optimizer.py       # Detect GPUs
python slate/slate_hardware_optimizer.py --optimize       # Apply optimizations
python slate/slate_hardware_optimizer.py --install-pytorch # Install correct PyTorch

# Runner management
python slate/slate_runner_manager.py --detect  # Detect runner
python slate/slate_runner_manager.py --status  # Runner status
python slate/slate_runner_manager.py --dispatch "ci.yml"  # Dispatch workflow

# Orchestrator (all services)
python slate/slate_orchestrator.py start       # Start all services
python slate/slate_orchestrator.py stop        # Stop all services
python slate/slate_orchestrator.py status      # Service status

# Workflow manager
python slate/slate_workflow_manager.py --status   # Task status
python slate/slate_workflow_manager.py --cleanup   # Clean stale tasks
python slate/slate_workflow_manager.py --enforce   # Enforce completion

# Benchmarks
python slate/slate_benchmark.py                # Run benchmarks
```

## Project Structure
```
slate/              # Core SDK modules
  slate_status.py           # System health checker
  slate_runtime.py          # Integration & dependency checker
  slate_hardware_optimizer.py  # GPU detection & PyTorch optimization
  slate_runner_manager.py   # GitHub Actions runner management
  slate_orchestrator.py     # Unified service orchestrator
  slate_workflow_manager.py # Task lifecycle & PR workflows
  slate_benchmark.py        # Performance benchmarks
  slate_fork_manager.py     # Fork contribution workflow
  mcp_server.py             # MCP server for Claude Code
  slate_terminal_monitor.py # Terminal activity tracking
  install_tracker.py        # Installation tracking

agents/             # API servers & agent modules
  runner_api.py             # RunnerAPI class for CI integration
  slate_dashboard_server.py # FastAPI dashboard (port 8080)
  install_api.py            # Installation API

skills/             # Copilot Chat skill definitions
  slate-status/             # Status checking skill
  slate-runner/             # Runner management skill
  slate-orchestrator/       # Service orchestration skill
  slate-workflow/           # Workflow management skill
  slate-help/               # Help & documentation skill

actions-runner/     # GitHub Actions self-hosted runner
  .runner                   # Runner config (agent: slate-runner)
  .env                      # Environment: CUDA, Python, GPU config
  hooks/pre-job.ps1         # Pre-job hook: GPU env + PATH setup

.github/
  workflows/                # CI/CD workflow definitions
    ci.yml                  # Main CI: lint, tests, SDK, security
    slate.yml               # Integration tests
    pr.yml                  # PR validation
    nightly.yml             # Nightly health checks
    cd.yml                  # Build & deploy
    docs.yml                # Documentation validation
    fork-validation.yml     # Fork security gate
    contributor-pr.yml      # External contributor PRs
  slate.config.yaml         # Master SLATE configuration
```

## Self-Hosted Runner Details
- **Name**: slate-runner
- **Labels**: `[self-hosted, Windows, X64, slate, gpu, cuda, gpu-2, blackwell]`
- **Work folder**: `slate_work`
- **GPUs**: 2x NVIDIA GeForce RTX 5070 Ti (Blackwell, compute 12.0)
- **Pre-job hook**: Sets `CUDA_VISIBLE_DEVICES=0,1`, SLATE env vars, Python PATH
- **Python PATH**: Each workflow job has a `Setup Python` step that adds venv to `GITHUB_PATH`
- **No `actions/setup-python`**: Removed — use the local `.venv` instead

## Workflow Conventions
- All jobs use `runs-on: [self-hosted, slate]`
- Default shell: `powershell`
- Python path step: `'E:\11132025\.venv\Scripts' | Out-File -Append $env:GITHUB_PATH`
- YAML paths use single quotes to avoid backslash escape issues

## Agent Routing (from slate.config.yaml)
| Pattern | Agent | Role |
|---------|-------|------|
| implement, code, build, fix | ALPHA | Coding |
| test, validate, verify | BETA | Testing |
| analyze, plan, research | GAMMA | Planning |
| claude, mcp, sdk, integration | DELTA | External Bridge |
| complex, multi-step | COPILOT | Full orchestration |

## Security Rules
- ALL network bindings: `127.0.0.1` ONLY — never `0.0.0.0`
- No external telemetry
- No `curl.exe` (freezes on this system)
- Protected files in forks: `.github/workflows/*`, `CODEOWNERS`, action guards
- Blocked patterns: `eval(`, `exec(os`, `rm -rf /`, `base64.b64decode`

## Terminal Rules
- Use `isBackground=true` for long-running commands (servers, watchers, runner)
- Never use `curl.exe` — use Python `urllib.request` instead
- Python executable: `E:\11132025\.venv\Scripts\python.exe`
- Git credential: `git credential fill` with `protocol=https` / `host=github.com`

## GitHub API Access
```python
# Get token from git credential manager
import subprocess
result = subprocess.run(['git', 'credential', 'fill'],
    input='protocol=https\nhost=github.com\n',
    capture_output=True, text=True)
token = [l.split('=',1)[1] for l in result.stdout.splitlines()
         if l.startswith('password=')][0]
```
Repository API base: `https://api.github.com/repos/SynchronizedLivingArchitecture/S.L.A.T.E`
