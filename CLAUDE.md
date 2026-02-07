# S.L.A.T.E. Development Guidelines
<!-- Modified: 2026-02-07T04:57:00Z | Author: COPILOT | Change: Add AAA standards and fix MCP JSON indentation -->

**S.L.A.T.E.** = Synchronized Living Architecture for Transformation and Evolution

**Constitution**: `.specify/memory/constitution.md` — Supersedes all other practices
Last updated: 2026-02-06

## Active Technologies
- Python 3.11+ (backend), Vanilla JavaScript + D3.js v7 (frontend)
- FastAPI (dashboard server on port 8080)
- D3.js v7 (bundled locally, tech tree visualization)
- **Ollama** (local LLM inference, mistral-nemo on dual RTX 5070 Ti) - localhost:11434
- **Foundry Local** (ONNX-optimized local inference) - localhost:5272
- ChromaDB (local vector store for RAG memory)

## Claude Code Integration

SLATE provides slash commands and MCP tools for Claude Code integration.

### Slash Commands (Project-Level)

Commands are defined in `.claude/commands/` and available when working in this project:

| Command | Description |
|---------|-------------|
| `/slate [start\|stop\|status]` | Manage SLATE orchestrator |
| `/slate-status` | Check system and service status |
| `/slate-workflow` | Manage task workflow queue |
| `/slate-runner` | Manage GitHub Actions runner |
| `/slate-discussions` | Manage GitHub Discussions |
| `/slate-multirunner` | Manage multi-runner system |
| `/slate-help` | Show all available commands |

### MCP Server Setup

The SLATE MCP server provides AI tools. Add to `~/.claude/config.json`:

```json
{
  "mcpServers": {
      "slate": {
         "command": "<workspace>\\.venv\\Scripts\\python.exe",
         "args": ["<workspace>\\slate\\mcp_server.py"],
         "env": {
            "SLATE_WORKSPACE": "<workspace>",
            "PYTHONPATH": "<workspace>"
         }
      }
  }
}
```

## AAA Standards

SLATE follows AAA standards across testing, accessibility, security, and performance.

### Testing (Arrange-Act-Assert)
- Use explicit Arrange, Act, Assert sections in tests
- Use pytest and pytest-asyncio

### Accessibility (WCAG AAA)
- Keyboard accessible UI, visible focus, high contrast
- Avoid motion without a disable option

### Security/Compliance
- Local-only bindings (`127.0.0.1`)
- No dynamic execution (`eval`, `exec`)
- No secrets or tokens in code or logs

### Performance/Reliability
- Validate with `slate/slate_benchmark.py`
- Add timeouts and retries for IO

### MCP Tools

| Tool | Description |
|------|-------------|
| `slate_status` | Check all services and GPU status |
| `slate_workflow` | Manage task queue |
| `slate_orchestrator` | Start/stop services |
| `slate_runner` | Manage GitHub runner |
| `slate_ai` | Execute AI tasks via local LLMs |

### Structure

```text
.claude/commands/     # Claude Code slash commands
  slate.md            # /slate command
  slate-status.md     # /slate-status command
  slate-workflow.md   # /slate-workflow command
  slate-runner.md     # /slate-runner command
  slate-discussions.md  # /slate-discussions command
  slate-multirunner.md  # /slate-multirunner command
  slate-help.md       # /slate-help command
slate/mcp_server.py   # MCP server implementation
```

## Local AI Providers (FREE - No Cloud Costs)

| Provider | Port | Models | Status |
|----------|------|--------|--------|
| Ollama | 11434 | mistral-nemo, llama3.2, phi, llama2, mistral | Active |
| Foundry Local | 5272 | Phi-3, Mistral-7B (ONNX) | Active |

```powershell
# Check provider status
.\.venv\Scripts\python.exe slate/foundry_local.py --check

# List all local models
.\.venv\Scripts\python.exe slate/foundry_local.py --models

# Generate with auto-provider selection
.\.venv\Scripts\python.exe slate/foundry_local.py --generate "your prompt"

# Download Foundry Local models (run in PowerShell)
foundry model download microsoft/Phi-3.5-mini-instruct-onnx
```

**Security**: ActionGuard blocks ALL paid cloud APIs (OpenAI, Anthropic, etc.). Only localhost AI services are allowed.

## Unified AI Backend

SLATE routes all AI tasks through `unified_ai_backend.py` which prioritizes FREE local backends:

```
Task Type          -> Best Backend     (Cost)
─────────────────────────────────────────────
code_generation    -> ollama_local     (FREE)
code_review        -> ollama_local     (FREE)
test_generation    -> ollama_local     (FREE)
bug_fix            -> ollama_local     (FREE)
refactoring        -> ollama_local     (FREE)
documentation      -> ollama_local     (FREE)
analysis           -> ollama_local     (FREE)
research           -> ollama_local     (FREE)
planning           -> speckit          (FREE)
```

```powershell
# Check all backend status
.\.venv\Scripts\python.exe slate/unified_ai_backend.py --status

# Execute task with auto-routing
.\.venv\Scripts\python.exe slate/unified_ai_backend.py --task "your task"
```

**Key Files**:
- `slate/unified_ai_backend.py` - Central routing (Ollama, Foundry, Copilot, Claude)
- `slate/foundry_local.py` - Foundry Local + Ollama unified client
- `slate/inference_instructions.py` - ML-based code generation guidance
- `slate/action_guard.py` - Security (blocks paid APIs)

## AI Orchestrator (Automated Maintenance)

SLATE includes an AI orchestrator that automatically maintains the codebase using local Ollama models.

### Capabilities

| Feature | Schedule | Description |
|---------|----------|-------------|
| Quick Analysis | Every 4 hours | Analyze recently changed files |
| Full Analysis | Daily 2am | Complete codebase analysis |
| Documentation | Daily | Auto-generate/update docs |
| GitHub Monitor | Daily | Analyze workflows and integrations |
| Model Training | Weekly Sunday | Train custom SLATE model |

### Commands

```powershell
# Check AI orchestrator status
.\.venv\Scripts\python.exe slate/slate_ai_orchestrator.py --status

# Warmup AI models
.\.venv\Scripts\python.exe slate/slate_ai_orchestrator.py --warmup

# Analyze recently changed files
.\.venv\Scripts\python.exe slate/slate_ai_orchestrator.py --analyze-recent

# Full codebase analysis
.\.venv\Scripts\python.exe slate/slate_ai_orchestrator.py --analyze-codebase

# Update documentation
.\.venv\Scripts\python.exe slate/slate_ai_orchestrator.py --update-docs

# Monitor GitHub integrations
.\.venv\Scripts\python.exe slate/slate_ai_orchestrator.py --monitor-github

# Collect training data
.\.venv\Scripts\python.exe slate/slate_ai_orchestrator.py --collect-training

# Train custom model
.\.venv\Scripts\python.exe slate/slate_ai_orchestrator.py --train
```

### Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ai-maintenance.yml` | Every 4h, 2am, push | Codebase analysis and doc updates |
| `ai-training.yml` | Weekly Sunday 5am | Model training and customization |
| `fork-intelligence.yml` | Every 6h | AI-powered fork analysis |

### Custom Model

SLATE can train a custom model (`slate-custom`) tuned for the codebase:
- Based on `mistral-nemo`
- Trained on SLATE code patterns
- Understands project architecture
- Updated weekly with new code

## Project Structure

```text
slate/             # Core SLATE engine modules
agents/            # Dashboard server and legacy agent code (deprecated)
slate_core/        # Shared infrastructure (locks, memory, GPU scheduler)
specs/             # Active specifications
src/               # Source code (backend/frontend)
tests/             # Test suite
skills/            # Claude Code skill definitions
commands/          # Claude Code command help
hooks/             # Claude Code automation hooks
.claude-plugin/    # Claude Code plugin manifest
.specify/          # Constitution, memory, feedback
.slate_tech_tree/  # Tech tree state (tech_tree.json)
.slate_changes/    # Detected code changes and snapshots
.slate_nemo/       # Nemo knowledge base and training sessions
.slate_errors/     # Error logs with context
.slate_index/      # ChromaDB vector index
.slate_prompts/    # Ingested prompts and intent learning data
```

## Commands

```powershell
# Start SLATE (dashboard + runner + workflow monitor)
.\.venv\Scripts\python.exe slate/slate_orchestrator.py start

# Check all services status
.\.venv\Scripts\python.exe slate/slate_orchestrator.py status

# Stop all SLATE services
.\.venv\Scripts\python.exe slate/slate_orchestrator.py stop

# Quick system status (auto-detects GPU, services)
.\.venv\Scripts\python.exe slate/slate_status.py --quick

# Workflow health (stale tasks, abandoned, duplicates)
.\.venv\Scripts\python.exe slate/slate_workflow_manager.py --status

# Run tests
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Lint
ruff check .
```

## Code Style

- Python: Type hints required. Google-style docstrings. Use `Annotated` for tool parameters.
- Imports: Add `WORKSPACE_ROOT` to `sys.path` when importing cross-module.
- UI: Glassmorphism theme (75% opacity, muted pastels). System fonts only (Consolas, Segoe UI).
- Task files: Always use `slate_core/file_lock.py` for `current_tasks.json` (prevents race conditions).

## Security Architecture — LOCAL ONLY

- **All servers bind to `127.0.0.1` only** — never `0.0.0.0`
- No external network calls unless explicitly requested by user
- ActionGuard (`slate/action_guard.py`) validates all actions
- Content Security Policy enforced — no external CDN/fonts
- Rate limiting active on dashboard API endpoints

### SDK Source Guard (Trusted Publishers Only)

SDKSourceGuard (`slate/sdk_source_guard.py`) enforces that ALL packages come from trusted primary sources:

| Trusted Source | Examples |
|----------------|----------|
| Microsoft | azure-*, onnxruntime |
| NVIDIA | nvidia-cuda-*, triton |
| Anthropic | anthropic SDK |
| Meta/Facebook | torch, torchvision |
| Google | tensorflow, jax |
| Hugging Face | transformers, datasets |
| Python Foundation | pip, setuptools |

```powershell
# Check SDK security status
.\.venv\Scripts\python.exe slate/sdk_source_guard.py --report

# Validate a specific package
.\.venv\Scripts\python.exe slate/sdk_source_guard.py --validate "some-package"

# Check all requirements.txt packages
.\.venv\Scripts\python.exe slate/sdk_source_guard.py --check-requirements
```

**Blocked Sources:**
- Unknown PyPI publishers
- Untrusted GitHub organizations
- Known typosquatting packages
- Suspicious naming patterns

## Task Execution System

SLATE uses GitHub Actions with a self-hosted runner for all task execution. The deprecated agent system (ALPHA, BETA, GAMMA, DELTA) has been replaced by workflow-based execution.

Tasks in `current_tasks.json` are processed by GitHub Actions workflows.
Use `assigned_to: "workflow"` for workflow-based execution.

## Task Management

- Task queue: `current_tasks.json` (use FileLock for atomic access)
- Priorities: `DO_THIS_NOW.txt` for immediate priorities
- Tech tree: `.slate_tech_tree/tech_tree.json` directs development focus
- Spec lifecycle: `draft → specified → planned → tasked → implementing → complete`

## GitHub Project Boards

SLATE uses GitHub Projects V2 for task tracking and workflow management. The KANBAN board is the primary source for workflow execution.

### Project Board Mapping

| # | Board | Purpose |
|---|-------|---------|
| 5 | **KANBAN** | Primary workflow source - active tasks |
| 7 | **BUG TRACKING** | Bug-related issues and fixes |
| 8 | **ITERATIVE DEV** | Pull requests and iterations |
| 10 | **ROADMAP** | Completed features and enhancements |
| 4 | **PLANNING** | Planning and design tasks |
| 6 | **FUTURE RELEASE** | Future version items |

### Project Board Commands

```powershell
# Check all project boards status
.\.venv\Scripts\python.exe slate/slate_project_board.py --status

# Update all boards from current_tasks.json
.\.venv\Scripts\python.exe slate/slate_project_board.py --update-all

# Sync KANBAN to local tasks
.\.venv\Scripts\python.exe slate/slate_project_board.py --sync

# Push pending tasks to KANBAN
.\.venv\Scripts\python.exe slate/slate_project_board.py --push
```

### Auto-Categorization

Tasks are automatically routed to boards by keywords:
- **BUG TRACKING**: bug, fix, crash, error, broken
- **ROADMAP**: feat, add, new, implement, create
- **PLANNING**: plan, design, architect, research
- **KANBAN**: default for active work

### Workflow Automation

The `project-automation.yml` workflow:
- Runs every 30 minutes (scheduled)
- Auto-adds issues/PRs to boards based on labels
- PII scanning before public board exposure
- Bidirectional sync with `current_tasks.json`

## GitHub Discussions

SLATE integrates GitHub Discussions for community engagement and feature ideation.

### Discussion Categories

| Category | Routing | Action |
|----------|---------|--------|
| Announcements | None | Informational only |
| Ideas | ROADMAP board | Creates tracking issue |
| Q&A | Metrics tracking | Monitors response time |
| Show and Tell | Engagement log | Community showcase |
| General | Engagement log | Community discussion |

### Discussion Commands

```powershell
# Check discussion system status
.\.venv\Scripts\python.exe slate/slate_discussion_manager.py --status

# List unanswered Q&A discussions
.\.venv\Scripts\python.exe slate/slate_discussion_manager.py --unanswered

# Sync actionable discussions to task queue
.\.venv\Scripts\python.exe slate/slate_discussion_manager.py --sync-tasks

# Generate engagement metrics
.\.venv\Scripts\python.exe slate/slate_discussion_manager.py --metrics
```

### Discussion Automation

The `discussion-automation.yml` workflow:
- Triggers on discussion events (create, edit, label, answer)
- Hourly scheduled processing for metrics
- PII scanning before processing
- Routes Ideas/Bugs to issue tracker
- Tracks Q&A response times

## Test-Driven Development (Constitution Mandate)

All code changes must be accompanied by tests. Target 50%+ coverage for `slate/` and `slate_core/`.

```text
1. WRITE TEST → failing test defining expected behavior
2. RUN TEST → verify it fails (red)
3. IMPLEMENT → minimum code to pass
4. RUN TEST → verify it passes (green)
5. REFACTOR → clean up while keeping tests green
```

## Dual-Repository System

SLATE uses a dual-repo model for development and distribution:

```
SLATE (origin)         = Main repository (the product)
       ↑
       │ contribute-to-main.yml
       │
SLATE-BETA (beta)      = Developer fork (where development happens)
```

### Git Remote Configuration

```powershell
# Check remotes
git remote -v
# Should show:
# origin  https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.git
# beta    https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E-BETA.git
```

### Development Workflow (BETA → SLATE)

1. **Develop on BETA branch**
   ```powershell
   git checkout -b feature/my-feature
   # Make changes, commit
   ```

2. **Sync BETA with SLATE main** (get latest)
   ```powershell
   git fetch origin
   git merge origin/main
   ```

3. **Push to BETA**
   ```powershell
   git push beta HEAD:main
   ```

4. **Contribute to SLATE main**
   - Run the `contribute-to-main.yml` workflow on BETA
   - OR push directly if you have access:
   ```powershell
   git push origin HEAD:main
   ```

### Required Setup

1. **MAIN_REPO_TOKEN** secret on BETA repo
   - Settings → Secrets → Actions → Add `MAIN_REPO_TOKEN`
   - Use a PAT with `repo` and `workflow` scope

2. **GitHub CLI with workflow scope**
   ```powershell
   gh auth login --scopes workflow
   ```

## Fork Contributions

SLATE uses a secure fork validation system for external contributions:

1. Fork the repository from https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.
2. Create a local SLATE installation with your own git
3. Run `python slate/slate_fork_manager.py --init` to set up
4. Make changes following SLATE prerequisites
5. Submit PR - it will be validated by fork-validation workflow

**Required checks before PR merge:**
- Security Gate (no workflow modifications)
- SDK Source Guard (trusted publishers only)
- SLATE Prerequisites (core modules valid)
- ActionGuard Audit (no security bypasses)
- Malicious Code Scan (no obfuscated code)

## GitHub Integration

Repository: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.

- Branch protection on `main` requires reviews and passing checks
- CODEOWNERS enforces review requirements for critical paths
- All PRs must pass SLATE compatibility checklist

## GitHub Workflow System

SLATE uses GitHub as a **task execution platform**. The GitHub ecosystem manages the entire project lifecycle: issues → tasks → workflow execution → PR completion.

### Workflow Architecture

```
GitHub Issues/Tasks → current_tasks.json → Workflow Dispatch → Self-hosted Runner
        ↓                     ↓                   ↓                    ↓
    Tracking              Task Queue          CI/CD Jobs          AI Execution
                                                  ↓                    ↓
                                              Validation     →    PR/Commit
```

### Workflow Files

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR | Smoke tests, lint, unit tests, security |
| `slate.yml` | Core path changes | Tech tree, task queue, validation |
| `nightly.yml` | Daily 4am UTC | Full test suite, dependency audit |
| `cd.yml` | Tags/main | Build EXE, create releases |
| `fork-validation.yml` | Fork PRs | Security gate |

### Auto-Configured Runner

SLATE **auto-detects** and configures the GitHub Actions runner. No manual setup required:

```powershell
# Auto-detect runner, GPUs, and GitHub state
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --status

# Auto-configure hooks, environment, and labels
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --setup

# Dispatch a workflow for execution
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --dispatch "ci.yml"
```

The runner manager automatically:
- **Detects** GPU configuration (count, architecture, CUDA capability)
- **Creates** pre-job hooks for environment setup
- **Generates** labels (self-hosted, slate, gpu, cuda, blackwell, multi-gpu)
- **Configures** SLATE workspace paths and venv

### Workflow Management

```powershell
# Analyze task workflow health
.\.venv\Scripts\python.exe slate/slate_workflow_manager.py --status

# Auto-cleanup stale/abandoned/duplicate tasks
.\.venv\Scripts\python.exe slate/slate_workflow_manager.py --cleanup

# Check if workflow can accept new tasks
.\.venv\Scripts\python.exe slate/slate_workflow_manager.py --enforce
```

Automatic rules:
- **Stale** (in-progress > 4h) → auto-reset to pending
- **Abandoned** (pending > 24h) → flagged for review
- **Duplicates** → auto-archived
- **Max concurrent** → 5 tasks before blocking

## System Auto-Detection

SLATE auto-detects all components on startup:

```powershell
# Full system auto-detection
.\.venv\Scripts\python.exe slate/slate_status.py --quick

# Runtime integration check
.\.venv\Scripts\python.exe slate/slate_runtime.py --check-all

# JSON for automation pipelines
.\.venv\Scripts\python.exe slate/slate_status.py --json
```

### Auto-Detected Components

| Component | Detection Method | Auto-Config |
|-----------|------------------|-------------|
| Python | Version check | Validates 3.11+ |
| GPU | nvidia-smi | Compute cap, memory, architecture |
| PyTorch | Import + CUDA | Version, device count |
| Ollama | Service query | Model list |
| GitHub Runner | .runner file | Labels, hooks, service |
| venv | Path check | Activation |

## System Services

| Service | Port | Auto-Checked By |
|---------|------|-----------------|
| Dashboard | 8080 | `slate/slate_status.py` |
| Ollama | 11434 | `slate/slate_status.py` |
| Foundry Local | 5272 | `slate/slate_status.py` |
| GitHub Runner | N/A | `slate/slate_runner_manager.py` |

## Quick Reference

```powershell
# Full system status (auto-detects everything)
.\.venv\Scripts\python.exe slate/slate_status.py --quick

# Workflow health
.\.venv\Scripts\python.exe slate/slate_workflow_manager.py --status

# Runner status and auto-config
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --status

# Run tests
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Start dashboard
.\.venv\Scripts\python.exe agents/slate_dashboard_server.py
```
