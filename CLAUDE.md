# S.L.A.T.E. Development Guidelines

**S.L.A.T.E.** = Synchronized Living Architecture for Transformation and Evolution

**Constitution**: `.specify/memory/constitution.md` — Supersedes all other practices
Last updated: 2026-02-06

---

## Quick Start (5 Minutes)

```powershell
# Clone and install
git clone https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E..git
cd S.L.A.T.E.
python install_slate.py

# Verify installation
.\.venv\Scripts\python.exe slate/slate_sdk.py --verify

# Start dashboard
.\.venv\Scripts\python.exe agents/slate_dashboard_server.py
```

**With self-hosted runner:**
```powershell
python install_slate.py --runner --runner-token YOUR_TOKEN
```

---

## SLATE SDK

The SDK provides unified setup, configuration, and integration. Always use the SDK for system operations.

### CLI Commands

```powershell
# Full setup (creates venv, installs deps, configures system)
python slate/slate_sdk.py --setup

# Setup with runner
python slate/slate_sdk.py --setup --runner --runner-token YOUR_TOKEN

# Check status (human-readable)
python slate/slate_sdk.py --status

# Check status (JSON for scripts)
python slate/slate_sdk.py --status --json

# Verify installation
python slate/slate_sdk.py --verify

# Integrate git remotes
python slate/slate_sdk.py --integrate-git

# Integrate runner
python slate/slate_sdk.py --integrate-runner
```

### Python API

```python
from slate.slate_sdk import SlateSDK

sdk = SlateSDK()

# Full setup
result = sdk.setup(include_runner=True)

# Get status
status = sdk.get_status()

# Verify installation
verification = sdk.verify()
```

---

## Architecture

SLATE uses a **GitHub-centric architecture** with self-hosted runners for GPU-accelerated CI/CD:

```
GitHub Repository (S.L.A.T.E.)
         │
    ┌────┴────┐
    ▼         ▼
  Cloud     Self-Hosted
  Runner     Runner
             (2x RTX 5070 Ti)
         │
         ▼
   GPU Workloads
   (tests, AI, builds)
```

### Single-Repository Model

| Repository | Purpose | Branch |
|------------|---------|--------|
| **S.L.A.T.E.** | Main product — all development | `main` (stable), feature branches |

> **Note**: S.L.A.T.E.-BETA is deprecated. All work is in the single `S.L.A.T.E.` repository.

---

## Technologies

- Python 3.11+ (backend)
- Vanilla JavaScript + D3.js v7 (frontend)
- FastAPI (dashboard on port 8080)
- GitHub Actions (CI/CD)
- Self-hosted runner (GPU compute)

### Local AI Providers (FREE - No Cloud Costs)

| Provider | Port | Models | Status |
|----------|------|--------|--------|
| Ollama | 11434 | mistral-nemo, llama3.2, phi | Active |
| Foundry Local | 5272 | Phi-3, Mistral-7B (ONNX) | Active |

**Security**: ActionGuard blocks ALL paid cloud APIs. Only localhost AI is allowed.

---

## Project Structure

```text
slate/           # Core SLATE modules
  slate_sdk.py         # Unified SDK (setup, status, verify)
  slate_status.py      # System status checker
  slate_runner_manager.py    # GitHub runner management
  slate_github_integration.py  # GitHub API integration
  slate_fork_manager.py      # Fork contribution workflow
  slate_project_manager.py   # GitHub Projects integration
  unified_ai_backend.py      # AI routing (Ollama, Foundry)
  action_guard.py            # Security validation
  sdk_source_guard.py        # Package trust verification
agents/                # Agent implementations
  slate_dashboard_server.py  # Dashboard server
slate_core/            # Shared infrastructure
specs/                 # Feature specifications
tests/                 # Test suite
.github/
  workflows/           # GitHub Actions workflows
  projects.json        # GitHub Projects configuration
```

---

## Commands

```powershell
# SLATE Status
.\.venv\Scripts\python.exe slate/slate_status.py --quick

# SDK Status
.\.venv\Scripts\python.exe slate/slate_sdk.py --status

# Run tests
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Lint
ruff check .

# Start dashboard
.\.venv\Scripts\python.exe agents/slate_dashboard_server.py

# Check AI backends
.\.venv\Scripts\python.exe slate/unified_ai_backend.py --status
```

---

## Self-Hosted Runner (Standard Protocol)

SLATE operations use local GPU runners for CI/CD. **This is standard protocol.**

### Current Runner

- **Name**: `slate-DESKTOP-R3UD82D`
- **Status**: Online
- **Labels**: `self-hosted`, `slate`, `gpu`, `windows`, `cuda`, `gpu-2`, `blackwell`
- **Hardware**: 2x RTX 5070 Ti (CUDA compute)

### Runner Management

```powershell
# Check status
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --status

# Start runner
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --start

# Stop runner
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --stop

# Install as Windows service
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --install-service
```

### Get Runner Token

1. Go to https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./settings/actions/runners
2. Click "New self-hosted runner"
3. Copy the token
4. Run: `python slate/slate_sdk.py --setup --runner --runner-token YOUR_TOKEN`

### Runner Labels (Auto-Detected)

| Label | Meaning |
|-------|---------|
| `self-hosted` | Not GitHub-hosted |
| `slate` | SLATE system runner |
| `gpu` | GPU available |
| `windows` | Windows OS |
| `cuda` | CUDA compute capable |
| `gpu-2` | 2 GPUs detected |
| `blackwell` | RTX 50 series |

---

## GitHub Projects Integration

SLATE syncs tasks to GitHub Projects for planning and tracking.

### Commands

```powershell
# List projects
.\.venv\Scripts\python.exe slate/slate_project_manager.py --list

# Create development project
.\.venv\Scripts\python.exe slate/slate_project_manager.py --create --template development

# Sync tasks to project
.\.venv\Scripts\python.exe slate/slate_project_manager.py --sync --project 1
```

### Project Templates

| Template | Description |
|----------|-------------|
| `development` | Sprint board with status, priority, agent fields |
| `roadmap` | Timeline view for milestones |
| `sprint` | 2-week sprint with story points |

---

## SLATE Protocol Automation

Automated workflows for sync, validate, and deploy operations.

### Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `self-hosted-runner.yml` | push, PR | GPU-enabled tests and builds |
| `slate-protocol.yml` | manual, nightly | Sync, validate, deploy |

### Protocol Commands

```powershell
# Trigger validation
gh workflow run slate-protocol.yml -f operation=validate

# Trigger sync
gh workflow run slate-protocol.yml -f operation=sync

# Trigger deploy
gh workflow run slate-protocol.yml -f operation=deploy
```

---

## Git Integration

### Configure Remotes

```powershell
# Automatic (recommended)
python slate/slate_sdk.py --integrate-git

# Manual
git remote set-url origin https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E..git
git remote add beta https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.-BETA.git
```

### Branch Naming Convention (Enforced)

All branches **must** follow this naming convention:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New features | `feature/user-auth` |
| `bugfix/` | Bug fixes | `bugfix/login-crash` |
| `refactor/` | Code refactoring | `refactor/api-cleanup` |
| `docs/` | Documentation | `docs/api-reference` |

**Prohibited patterns:** `001-*`, `002-*`, numbered prefixes, generic names.

### Development Workflow

```powershell
# Create branch (use correct prefix!)
git checkout -b feature/my-feature

# Make changes, commit
git add .
git commit -m "feat: description"

# Push to origin (triggers CI on self-hosted runner)
git push origin HEAD

# Create PR on GitHub
gh pr create --title "feat: description" --body "..."
```

### Pushing Workflow Files (IMPORTANT)

GitHub requires **workflow scope** to push changes to `.github/workflows/`. This is a security
feature. SLATE provides automatic handling:

```powershell
# Automatic workflow push (handles auth)
python slate/slate_workflow_push.py

# Check if workflow scope is available
python slate/slate_workflow_push.py --check

# Re-authenticate with workflow scope
python slate/slate_workflow_push.py --auth
```

**When you see this error:**
```
refusing to allow an OAuth App to create or update workflow without `workflow` scope
```

Run: `python slate/slate_workflow_push.py --auth` — this opens a browser to add the
workflow scope. Then retry the push.

**GitHub CLI alternative:**
```powershell
gh auth login -s workflow -w    # Add workflow scope
git push origin HEAD            # Push changes
```

---

## Agent System

| Agent | Role | Preference |
|-------|------|------------|
| ALPHA | Coding & implementation | GPU-preferred |
| BETA | Testing & validation | GPU-preferred |
| GAMMA | Planning & triage | CPU-preferred |
| DELTA | Claude Code bridge | CLI-based |

Tasks use `assigned_to` field for routing. `assigned_to: "auto"` uses ML-based smart routing.

---

## Security

- **All servers bind to `127.0.0.1` only** — never `0.0.0.0`
- No external network calls unless explicitly requested
- ActionGuard validates all agent actions
- GitHub runner executes trusted workflows only
- SDK Source Guard blocks untrusted packages

---

## Code Style

- Python: Type hints required. Google-style docstrings.
- Imports: Add `WORKSPACE_ROOT` to `sys.path` when importing cross-module.
- UI: Glassmorphism theme (75% opacity, muted pastels).
- Task files: Use `slate_core/file_lock.py` for `current_tasks.json`.

---

## Test-Driven Development (Constitution Mandate)

All code changes must be accompanied by tests. Target 50%+ coverage.

```text
1. WRITE TEST → failing test defining expected behavior
2. RUN TEST → verify it fails (red)
3. IMPLEMENT → minimum code to pass
4. RUN TEST → verify it passes (green)
5. REFACTOR → clean up while keeping tests green
```

---

## Resources

- **Repository**: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.
- **Wiki**: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./wiki
- **Issues**: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./issues
- **Projects**: https://github.com/orgs/SynchronizedLivingArchitecture/projects

---

## MCP Server Integration (Claude + Copilot)

SLATE provides a Model Context Protocol (MCP) server that exposes system tools to
AI assistants. The server is at `slate/slate_mcp_server.py`.

### MCP Tools Available

| Tool | Description |
|------|-------------|
| `slate_get_status` | System status (GPU, SDK, runner, agents) |
| `slate_run_check` | Run status/runtime/hardware/SDK checks |
| `slate_list_tasks` | List tasks from the task queue |
| `slate_gpu_info` | Detailed GPU information (VRAM, utilization, CUDA) |
| `slate_agent_status` | Agent system health (all 4 agents) |
| `slate_runner_status` | Self-hosted GitHub Actions runner status |
| `slate_search_code` | Search SLATE codebase |
| `slate_dashboard_url` | Get dashboard URL and running status |

### MCP Resources

| URI | Description |
|-----|-------------|
| `slate://status` | Current system status text |
| `slate://tasks` | Task queue as JSON |
| `slate://gpu` | nvidia-smi output |

### Running the MCP Server

```powershell
# stdio mode (for VS Code Copilot / Claude Desktop)
..\.venv\Scripts\python.exe slate/slate_mcp_server.py

# SSE mode (for web clients)
..\.venv\Scripts\python.exe slate/slate_mcp_server.py --sse --port 6274

# Verify tools are registered
..\.venv\Scripts\python.exe slate/slate_mcp_server.py --verify
```

### Claude Desktop Configuration

Add to `%APPDATA%/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "slate-system": {
      "command": "C:/path/to/S.L.A.T.E./.venv/Scripts/python.exe",
      "args": ["C:/path/to/S.L.A.T.E./slate/slate_mcp_server.py"]
    }
  }
}
```

### VS Code Copilot MCP Configuration

Registered in `.vscode/mcp.json` — Copilot Chat automatically loads the SLATE MCP
server when the workspace is open.

### Claude Skills (`.claude/skills/`)

| Skill | Purpose |
|-------|---------|
| `slate-status` | System health checks and diagnostics |
| `slate-tasks` | Task queue management |
| `slate-agents` | Agent system (ALPHA, BETA, GAMMA, DELTA) |
| `slate-benchmark` | GPU benchmarks and hardware optimization |
| `slate-traces` | Observability, metrics, and tracing |

---

## GitHub Integration Checklist

SLATE maintains full GitHub integration for project management:

| Integration | Status | Setup |
|-------------|--------|-------|
| **GitHub Actions** (17 workflows) | ✅ | `.github/workflows/` |
| **Self-Hosted Runner** (2x GPU) | ✅ | `install_slate.py --runner` |
| **GitHub Projects** (3 templates) | ✅ | `.github/projects.json` |
| **Issue Templates** (3 types) | ✅ | `.github/ISSUE_TEMPLATE/` |
| **PR Template** | ✅ | `.github/PULL_REQUEST_TEMPLATE.md` |
| **Dependabot** | ✅ | `.github/dependabot.yml` |
| **CodeQL Analysis** | ✅ | `.github/workflows/codeql.yml` |
| **CODEOWNERS** | ✅ | `.github/CODEOWNERS` |
| **Releases & Packages** | ✅ | `.github/workflows/release.yml` |
| **Label Sync** | ✅ | `.github/labels.yml` |
| **Security Advisories** | ✅ | `.github/SECURITY.md` |
| **Funding** | ✅ | `.github/FUNDING.yml` |
| **Wiki** (14 pages) | ✅ | `docs/wiki/` |
| **Copilot Agent** | ✅ | `.github/copilot-instructions.md` |
| **Copilot MCP Server** | ✅ | `.vscode/mcp.json` |
| **Claude Skills** | ✅ | `.claude/skills/` |
| **Claude MCP Config** | ✅ | `slate/slate_mcp_server.py` |

---

*S.L.A.T.E. - Local AI, Unlimited Potential*
