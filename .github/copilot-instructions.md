# S.L.A.T.E. Copilot Agent Instructions
# Modified: 2026-02-06T23:30:00Z | Author: COPILOT | Change: Add VS Code integration section and extension dependencies
# ═══════════════════════════════════════════════════════════════════════════════
#
# This file configures GitHub Copilot as a SLATE-aware coding agent.
# Copilot reads this to understand the project, its conventions, and tools.
#
# Reference: https://docs.github.com/en/copilot/customizing-copilot/adding-repository-instructions
# ═══════════════════════════════════════════════════════════════════════════════

## Project Identity

- **Name**: S.L.A.T.E. (Synchronized Living Architecture for Transformation and Evolution)
- **Type**: Local-first AI agent orchestration framework
- **Language**: Python 3.11+
- **Repository**: `SynchronizedLivingArchitecture/S.L.A.T.E.`
- **Branch Strategy**: `main` (stable), feature branches off `main`
- **License**: MIT

## Architecture

SLATE is a multi-agent system with 4 agents (ALPHA, BETA, GAMMA, DELTA), a
FastAPI dashboard on port 8080, GPU-accelerated ML pipelines, and a self-hosted
GitHub Actions runner with 2x NVIDIA RTX 5070 Ti GPUs.

### Key Directories

| Path | Purpose |
|------|---------|
| `slate/` | Core SDK, status, runtime, runner manager, ML orchestrator |
| `agents/` | Dashboard server, install API, agent implementations |
| `slate_core/` | Shared infrastructure (file_lock, message_broker) |
| `tests/` | pytest test suite |
| `specs/` | Feature specifications |
| `.github/workflows/` | 17 GitHub Actions workflows |
| `docs/wiki/` | 14 wiki pages |

### Important Files

| File | Purpose |
|------|---------|
| `slate/slate_sdk.py` | Unified SDK (setup, status, verify) |
| `slate/slate_status.py` | System status checker |
| `slate/slate_runtime.py` | Integration runtime checks |
| `slate/slate_runner_manager.py` | Self-hosted runner manager |
| `slate/slate_mcp_server.py` | MCP server for AI assistants |
| `slate/action_guard.py` | Security validation |
| `agents/slate_dashboard_server.py` | FastAPI dashboard |
| `install_slate.py` | 12-step installation script |
| `clean_tasks.json` | Active task queue |

## Coding Conventions

### Required: Timestamp + Author on All Code Edits

```python
# Modified: YYYY-MM-DDTHH:MM:SSZ | Author: COPILOT | Change: description
```

### Python Style

- Type hints required on all function signatures
- Google-style docstrings
- Imports: add `WORKSPACE_ROOT` to `sys.path` for cross-module imports
- UI: Glassmorphism theme (75% opacity, muted pastels)
- Task files: use `slate_core/file_lock.py` for `current_tasks.json`

### Commit Format

```
feat(module): Short description

- Change detail 1
- Change detail 2
```

Prefixes: `feat`, `fix`, `docs`, `refactor`, `test`, `ci`, `chore`

## Commands (Run These with `.venv` Python)

```powershell
# Status
.\.venv\Scripts\python.exe slate/slate_status.py --quick

# Runtime checks
.\.venv\Scripts\python.exe slate/slate_runtime.py --check-all

# Hardware optimizer
.\.venv\Scripts\python.exe slate/slate_hardware_optimizer.py

# Runner status
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --status

# Tests
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Dashboard
.\.venv\Scripts\python.exe agents/slate_dashboard_server.py
```

## Security Rules

- **All servers bind to `127.0.0.1` ONLY** — never `0.0.0.0`
- No external network calls unless explicitly requested
- Never use `curl.exe` (freezes on Windows)
- Never use `eval()` or `exec()` with untrusted input
- ActionGuard validates all agent actions
- SDK Source Guard blocks untrusted packages

## Terminal Rules

- Use `isBackground=true` for long-running commands (servers, watchers)
- Use `.\.venv\Scripts\python.exe` not bare `python`
- Never use `&&` in PowerShell — use `;` instead

## Self-Hosted Runner

- **GPUs**: 2x NVIDIA GeForce RTX 5070 Ti (Blackwell)
- **Labels**: `self-hosted, slate, gpu, windows, cuda, gpu-2, multi-gpu, blackwell`
- **Directory**: `E:\11132025\actions-runner`
- **CUDA_VISIBLE_DEVICES**: `0,1`

## MCP Server

SLATE exposes tools to Copilot via an MCP server at `slate/slate_mcp_server.py`.
The server provides: `slate_get_status`, `slate_run_check`, `slate_list_tasks`,
`slate_gpu_info`, `slate_agent_status`, `slate_runner_status`, `slate_search_code`,
`slate_dashboard_url`.

## VS Code Integration

SLATE is designed as a VS Code-first development environment. The workspace
ships with full editor configuration so contributors have the correct tools
from the first open.

### Required Extensions (`.vscode/extensions.json`)

| Extension | ID | Purpose |
|-----------|----|--------|
| GitHub Copilot | `github.copilot` | AI code completion |
| Copilot Chat | `github.copilot-chat` | Agent chat + MCP tools |
| GitHub Actions | `github.vscode-github-actions` | Workflow authoring, run monitoring |
| GitHub Pull Requests | `github.vscode-pull-request-github` | PR review, issue management |
| Python | `ms-python.python` | Language support, debugging |
| Pylance | `ms-python.vscode-pylance` | Type checking, IntelliSense |
| Debugpy | `ms-python.debugpy` | Python debugging |

### Workspace Config Files

| File | Purpose |
|------|---------|
| `.vscode/extensions.json` | Extension recommendations (prompted on open) |
| `.vscode/mcp.json` | MCP server registration for Copilot Chat |
| `.vscode/tasks.json` | Build, run, and test tasks |

## GitHub Integrations

| Integration | Status | Config |
|-------------|--------|--------|
| Actions (17 workflows) | Active | `.github/workflows/` |
| Projects (3 templates) | Active | `.github/projects.json` |
| Issues (3 templates) | Active | `.github/ISSUE_TEMPLATE/` |
| PR Template | Active | `.github/PULL_REQUEST_TEMPLATE.md` |
| Dependabot | Active | `.github/dependabot.yml` |
| CodeQL | Active | `.github/workflows/codeql.yml` |
| CODEOWNERS | Active | `.github/CODEOWNERS` |
| Self-Hosted Runner | Active | `slate/slate_runner_manager.py` |
| Releases & Packages | Active | `.github/workflows/release.yml` |
| Labels (auto-sync) | Active | `.github/labels.yml` |
| Security Advisories | Active | `.github/SECURITY.md` |
| Funding | Active | `.github/FUNDING.yml` |
| Wiki | Active | `docs/wiki/` (14 pages) |
| VS Code Extensions | Active | `.vscode/extensions.json` |
| VS Code MCP Tools | Active | `.vscode/mcp.json` |
| VS Code Tasks | Active | `.vscode/tasks.json` |

## Test-Driven Development

All code changes must be accompanied by tests. Target 50%+ coverage.

```
1. WRITE TEST → failing test defining expected behavior
2. RUN TEST → verify it fails
3. IMPLEMENT → minimum code to pass
4. RUN TEST → verify it passes
5. REFACTOR → clean up while keeping tests green
```
