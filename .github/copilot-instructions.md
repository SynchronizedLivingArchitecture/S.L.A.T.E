# S.L.A.T.E. Copilot Instructions
# Modified: 2026-02-07T22:00:00Z | Author: COPILOT | Change: Add document evolution rules, chat participant enforcement, aggressive execution patterns

## Workspace

**This workspace is a S.L.A.T.E. installation.**
All paths are relative to the workspace root — no hardcoded directories.

- **Repo**: `SynchronizedLivingArchitecture/S.L.A.T.E`
- **Purpose**: Full SLATE SDK, inference pipeline, GPU benchmarking, CI/CD
- **Git**: `origin` → user's fork or `https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.git`

When install issues are found:
1. Fix the issue in the relevant installer/SDK file
2. Include `# Modified:` timestamp comment per SLATE rules
3. Commit with `fix(installer):` prefix
4. Push to the appropriate branch

## MANDATORY PROTOCOL  Read Before Any Operation

**Every Copilot session MUST begin by running SLATE protocols before performing work.**
This is NOT optional. These protocols exist to ensure system consistency, prevent
regressions, and maintain the integrity of the agentic AI pipeline.

### Session Start Protocol (REQUIRED)
Before ANY code changes, file creation, or task work, execute these in order:
```bash
python slate/slate_status.py --quick          # 1. System health  confirm GPUs, Python, Ollama
python slate/slate_runtime.py --check-all     # 2. Verify all 7 integrations are active
python slate/slate_workflow_manager.py --status # 3. Check task queue  respect completion rules
```

### Pre-Commit Protocol (REQUIRED)
Before committing or suggesting a commit:
```bash
python slate/slate_workflow_manager.py --enforce  # Block if tasks are stale/overloaded
python slate/slate_workflow_manager.py --cleanup  # Archive deprecated/test tasks
```

### Code Edit Rules (ENFORCED)
Every code edit MUST include a timestamp + author comment:
```python
# Modified: YYYY-MM-DDTHH:MM:SSZ | Author: COPILOT | Change: description
```

## Built-In Safeguards (ENFORCED)

SLATE enforces these protections automatically:

1) **ActionGuard** - Blocks dangerous patterns
   - Destructive commands (`rm -rf`, `format`, `del /s`)
   - Network exposure (`0.0.0.0` bindings)
   - Dynamic execution (`eval`, `exec`)
   - External paid API calls

2) **SDK Source Guard** - Trusted publishers only
   - Microsoft, NVIDIA, Meta, Google, Hugging Face
   - Unknown PyPI packages blocked

3) **PII Scanner** - Before GitHub sync
   - API keys, tokens, credentials detected
   - Personal info blocked from public boards

4) **Resource Limits**
   - Max concurrent tasks enforced
   - Stale tasks (>4h) auto-flagged
   - GPU memory monitored per-runner

## System Overview
SLATE (Synchronized Living Architecture for Transformation and Evolution) is a local-first
AI agent orchestration framework. All operations are LOCAL ONLY (127.0.0.1). Version 2.4.0.

Repository: `SynchronizedLivingArchitecture/S.L.A.T.E`
Python: 3.11+ via `.venv` at `<workspace>\.venv\Scripts\python.exe`
Runner: Self-hosted GitHub Actions runner `slate-runner` at `<workspace>\actions-runner`

## SLATE Protocol Commands  Use These, Not Ad-Hoc Commands

### System Health (run FIRST in every session)
```bash
python slate/slate_status.py --quick          # Quick health check
python slate/slate_status.py --json           # Machine-readable status
```

### Runtime Integration Check
```bash
python slate/slate_runtime.py --check-all     # All 7 integrations (Python, GPU, PyTorch, Transformers, Ollama, ChromaDB, venv)
python slate/slate_runtime.py --json          # JSON output
```

### Hardware & GPU Optimization
```bash
python slate/slate_hardware_optimizer.py       # Detect GPUs
python slate/slate_hardware_optimizer.py --optimize       # Apply optimizations
python slate/slate_gpu_manager.py --status     # Dual-GPU load balancing status
python slate/slate_gpu_manager.py --preload    # Preload models to assigned GPUs
```

### Task & Workflow Management
```bash
python slate/slate_workflow_manager.py --status   # Task queue status
python slate/slate_workflow_manager.py --cleanup   # Clean stale/deprecated tasks
python slate/slate_workflow_manager.py --enforce   # Enforce completion before new tasks
```

### Runner & CI/CD
```bash
python slate/slate_runner_manager.py --status  # Runner status
python slate/slate_runner_manager.py --dispatch "ci.yml"  # Dispatch workflow
```

### Service Orchestration
```bash
python slate/slate_orchestrator.py status      # Service status
python slate/slate_orchestrator.py start       # Start all services (dashboard, runner, monitor)
python slate/slate_orchestrator.py stop        # Stop all services
```

### ML / Agentic AI (GPU Inference)
```bash
python slate/ml_orchestrator.py --status       # ML pipeline status
python slate/ml_orchestrator.py --index-now    # Build codebase embedding index (uses ChromaDB)
python slate/ml_orchestrator.py --benchmarks   # Inference benchmarks
python slate/slate_model_trainer.py --status   # SLATE custom model status
python slate/slate_chromadb.py --status        # ChromaDB vector store status
python slate/slate_chromadb.py --index         # Index codebase into ChromaDB
python slate/slate_chromadb.py --search "query" # Semantic search
```

### Autonomous Loops
```bash
python slate/slate_unified_autonomous.py --status   # Autonomous loop status
python slate/slate_unified_autonomous.py --discover  # Discover available tasks
python slate/copilot_slate_runner.py --status        # Copilot runner bridge status
python slate/integrated_autonomous_loop.py --status  # Integrated loop status
```

### Benchmarks
```bash
python slate/slate_benchmark.py                # System benchmarks (CPU, memory, disk, GPU)
```

### Project Boards
```bash
python slate/slate_project_board.py --status   # GitHub Projects V2 board status
```

## Project Structure
```
slate/              # Core SDK modules (30 Python files)
  slate_status.py           # System health checker
  slate_runtime.py          # Integration & dependency checker (7 integrations)
  slate_hardware_optimizer.py  # GPU detection & PyTorch optimization
  slate_gpu_manager.py      # Dual-GPU load balancing for Ollama
  slate_runner_manager.py   # GitHub Actions runner management
  slate_orchestrator.py     # Unified service orchestrator
  slate_workflow_manager.py # Task lifecycle & PR workflows
  slate_workflow_analyzer.py # Meta-workflow analysis & deprecation detection
  slate_benchmark.py        # Performance benchmarks
  slate_fork_manager.py     # Fork contribution workflow
  slate_chromadb.py         # ChromaDB vector store integration
  ml_orchestrator.py        # ML inference orchestrator (Ollama + PyTorch)
  slate_model_trainer.py    # Custom SLATE model builder
  slate_unified_autonomous.py   # Unified autonomous task loop
  integrated_autonomous_loop.py # Self-healing autonomous brain
  copilot_slate_runner.py   # Copilot  autonomous bridge
  slate_project_board.py    # GitHub Projects V2 integration
  mcp_server.py             # MCP server for Claude Code
  action_guard.py           # Security enforcement (ActionGuard)
  sdk_source_guard.py       # SDK source validation
  pii_scanner.py            # PII detection
  slate_multi_runner.py     # Multi-runner parallelism
  slate_real_multi_runner.py # Real multi-runner implementation
  slate_runner_benchmark.py # Runner capacity benchmarks
  feature_flags.py          # Feature flag system
  install_tracker.py        # Installation tracking
  runner_cost_tracker.py    # Runner cost tracking
  runner_fallback.py        # Runner fallback logic
  slate_terminal_monitor.py # Terminal activity tracking
  slate_discussion_manager.py # Discussion automation

agents/             # API servers & agent modules
  runner_api.py             # RunnerAPI class (GitHub API integration)
  slate_dashboard_server.py # FastAPI dashboard (127.0.0.1:8080)
  install_api.py            # Installation API

models/             # Ollama Modelfiles for SLATE custom models
  Modelfile.slate-coder     # 12B code generation (mistral-nemo base)
  Modelfile.slate-fast      # 3B classification/summary (llama3.2 base)
  Modelfile.slate-planner   # 7B planning/analysis (mistral base)

plugins/            # VS Code & Claude extensions
  slate-copilot/            # @slate chat participant (TypeScript)
  slate-sdk/                # Claude Code plugin

skills/             # Copilot Chat skill definitions
  slate-status/             # Status checking skill
  slate-runner/             # Runner management skill
  slate-orchestrator/       # Service orchestration skill
  slate-workflow/           # Workflow management skill
  slate-help/               # Help & documentation skill

.github/
  workflows/                # 19 CI/CD workflow definitions
    ci.yml                  # Main CI: lint, tests, SDK, security, GPU validation
    cd.yml                  # Build & deploy
    slate.yml               # Integration tests
    agentic.yml             # GPU inference & autonomous agent loop
    codeql.yml              # Security analysis
    pr.yml                  # PR validation
    nightly.yml             # Nightly health checks
    docs.yml                # Documentation validation
    fork-validation.yml     # Fork security gate
    contributor-pr.yml      # External contributor PRs
    multi-runner.yml        # Multi-runner parallelism
    docker.yml              # Container builds
    release.yml             # Release management
  slate.config.yaml         # Master SLATE configuration
```

## Self-Hosted Runner Details
- **Name**: slate-runner
- **Labels**: `[self-hosted, Windows, X64, slate, gpu, cuda, gpu-2, blackwell]`
- **Work folder**: `slate_work`
- **GPUs**: 2x NVIDIA GeForce RTX 5070 Ti (Blackwell, compute 12.0, 16GB each)
- **Pre-job hook**: Sets `CUDA_VISIBLE_DEVICES=0,1`, SLATE env vars, Python PATH
- **SLATE Custom Models**: slate-coder (12B), slate-fast (3B), slate-planner (7B)

## Workflow Conventions
- All jobs use `runs-on: [self-hosted, slate]`
- Default shell: `powershell`
- Python path step: `"$env:GITHUB_WORKSPACE\.venv\Scripts" | Out-File -Append $env:GITHUB_PATH`
- YAML paths use single quotes to avoid backslash escape issues

## Agent Routing (from slate.config.yaml)
| Pattern | Agent | Role | GPU |
|---------|-------|------|-----|
| implement, code, build, fix | ALPHA | Coding | Yes |
| test, validate, verify, coverage | BETA | Testing | Yes |
| analyze, plan, research, document | GAMMA | Planning | No |
| claude, mcp, sdk, integration | DELTA | External Bridge | No |
| diagnose, investigate, troubleshoot, interactive, explain | COPILOT_CHAT | Chat Participant | No |
| complex, multi-step | COPILOT | Full orchestration | Yes |

## @slate Participant as Subagent (COPILOT_CHAT)
The @slate VS Code chat participant is registered as the **COPILOT_CHAT** agent in the
SLATE agent registry. This enables bidirectional task flow between the autonomous loop
and the interactive chat interface.

### Bridge Architecture
```
Autonomous Loop ──▶ copilot_agent_bridge.py ──▶ .slate_copilot_bridge.json
                                                       │
@slate Participant ◀── slate_agentBridge tool ◀────────┘
       │
       ▼
.slate_copilot_bridge_results.json ──▶ Autonomous Loop picks up results
```

### Bridge Commands
```bash
python slate/copilot_agent_bridge.py --status     # Bridge health
python slate/copilot_agent_bridge.py --pending     # Pending tasks for @slate
python slate/copilot_agent_bridge.py --results     # Completed results
python slate/copilot_agent_bridge.py --cleanup     # Clean stale entries
```

### How It Works
1. Autonomous loop classifies a task matching `diagnose|investigate|troubleshoot|interactive|explain`
2. Task is routed to COPILOT_CHAT agent → enqueued to `.slate_copilot_bridge.json`
3. @slate participant polls via `slate_agentBridge` tool (action: 'poll')
4. Participant processes the task using its full tool suite (20+ tools)
5. Results written back via `slate_agentBridge` tool (action: 'complete')
6. Copilot Runner picks up results and updates task status

## Security Rules (ENFORCED by ActionGuard)
- ALL network bindings: `127.0.0.1` ONLY  never `0.0.0.0`
- No external telemetry (ChromaDB telemetry disabled)
- No `curl.exe` (freezes on this system  use `urllib.request`)
- Protected files in forks: `.github/workflows/*`, `CODEOWNERS`, action guards
- Blocked patterns: `eval(`, `exec(os`, `rm -rf /`, `base64.b64decode`

## Terminal Rules
- Use `isBackground=true` for long-running commands (servers, watchers, runner)
- Never use `curl.exe`  use Python `urllib.request` or PowerShell `Invoke-RestMethod`
- Python executable: `./.venv/Scripts/python.exe` (Windows) or `./.venv/bin/python` (Linux/macOS)
- Always use `encoding='utf-8'` when opening files in Python on Windows
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

## When Copilot Does NOT Run Protocols
If you skip the mandatory protocols above, you risk:
1. **Stale task overload**  creating new tasks when existing ones need completion first
2. **Integration drift**  editing code that depends on an offline service (Ollama, ChromaDB)
3. **Security violations**  introducing `0.0.0.0` bindings or blocked patterns unchecked
4. **Version mismatch**  SDK version diverging from pyproject.toml
5. **GPU misconfiguration**  wrong model placement on dual-GPU system

**Always run the protocols. They take <5 seconds and prevent hours of debugging.**

## Document Evolution Rules (ENFORCED)

These instruction files are **append-only living documents**. All protocol files
(`copilot-instructions.md`, `AGENTS.md`, `slate.config.yaml`) follow strict evolution rules:

1. **ADDITIVE ONLY** — New behaviors, sections, and rules are APPENDED. Never overwrite,
   remove, or replace existing behaviors unless explicitly deprecated via `# DEPRECATED:`.
2. **Preserve All Existing Rules** — Every existing section, behavior, safeguard, and protocol
   remains intact. New additions extend the system, they do not contradict or supersede.
3. **Timestamp Every Change** — Each edit includes `# Modified:` with ISO timestamp, author, and
   a change description that says what was ADDED, not what was "updated" or "replaced".
4. **No Silent Overwrites** — If a behavior needs to change, add a NEW section that augments
   the original. Reference the original section. Never delete the original.
5. **Deprecation Protocol** — To retire a behavior: add `# DEPRECATED: YYYY-MM-DD | Reason:`
   above the section. The section stays in-document for history. A new section replaces it.
6. **Version Accumulation** — The document grows over time. This is intentional. It forms the
   complete behavioral history of the system.

## Chat Participant Enforcement (MANDATORY)

SLATE provides three VS Code chat participants. All agents and subagents MUST use them
aggressively for execution rather than falling back to raw terminal commands or manual work.

### Available Chat Participants

| Participant | Extension | Purpose | When to Use |
|-------------|-----------|---------|-------------|
| `@slate` | `slate.slate-copilot` | SLATE system ops — runner, CI, GPU, orchestrator, specs, agents | System management, health checks, deployments, agent routing |
| `@vscode` | Built-in | VS Code editor operations — files, settings, extensions, debugging | File operations, editor configuration, extension management |

### @slate Tools (20+ LM Tools)

| Tool | Purpose |
|------|--------|
| `slate_systemStatus` | System health check |
| `slate_runtimeCheck` | Integration verification |
| `slate_hardwareInfo` | GPU detection & optimization |
| `slate_runnerStatus` | Runner management |
| `slate_orchestrator` | Service lifecycle |
| `slate_workflow` | Task management |
| `slate_benchmark` | Performance benchmarks |
| `slate_securityAudit` | Security scanning |
| `slate_specKit` | Spec processing, wiki generation, roadmap |
| `slate_planContext` | Compressed context for token efficiency |
| `slate_devCycle` | Development stage management |
| `slate_codeGuidance` | Stage-aligned code guidance |
| `slate_agentBridge` | Autonomous loop ↔ chat bridge |
| `slate_autonomous` | Autonomous task execution |
| `slate_executeWork` | Direct work execution |
| `slate_handoff` | Task delegation between agents |
| `slate_forkCheck` | Fork sync and security |
| `slate_gpuManager` | Dual-GPU load balancing |
| `slate_learningProgress` | Learning/XP tracking |
| `slate_runCommand` | Escape hatch — run any SLATE Python script |

### Enforcement Rules

1. **Subagents MUST use `@slate`** — When delegating to `runSubagent` with agent name `slate`,
   the subagent has full access to all 20+ SLATE tools. Always prefer `@slate` over raw
   terminal commands for SLATE operations.
2. **Spec work MUST use `slate_specKit`** — Never process specs manually. Use
   `@slate /specs` or invoke `slate_specKit` tool directly.
3. **System operations MUST use `@slate`** — Health checks, runner management, CI dispatch,
   GPU management — all go through `@slate` participant, not ad-hoc Python commands.
4. **Editor operations SHOULD use `@vscode`** — File management, settings changes, extension
   installs — prefer `@vscode` participant when available.
5. **Multi-participant chaining** — Complex tasks should chain participants:
   - `@slate /status` → verify system health
   - `@slate /specs` → process specifications
   - `@vscode` → apply editor-level changes
6. **Never bypass participants for covered operations** — If a participant tool exists for the
   operation, USE IT. Don't fall back to manual terminal unless the tool fails.

### Aggressive Execution Patterns

Participants are NOT optional helpers — they are the PRIMARY execution interface.
Every task should be routed through the appropriate participant FIRST.

```
Task Arrives → Classify Intent → Route to Participant → Execute via Tools → Report
                                        ↓
                            @slate (system/infra/specs/agents)
                            @vscode (editor/files/settings)
                            Terminal (ONLY as fallback)
```

**Escalation Order:**
1. Try participant tool first (e.g., `slate_workflow` for task management)
2. Try participant command second (e.g., `@slate /status`)
3. Try subagent delegation third (e.g., `runSubagent` with `slate` agent)
4. Fall back to terminal ONLY if all participant paths fail

### Subagent Delegation with Participants

When using `runSubagent`, always specify the appropriate agent:

| Task Type | Subagent | Participant Used |
|-----------|----------|------------------|
| System health, runner, CI | `slate` | `@slate` |
| Spec processing, roadmap | `slate` | `@slate /specs` |
| Agent management, autonomous loops | `slate` | `@slate /agents` |
| Multi-step complex operations | `slate` | `@slate` (full orchestration) |
| Code research, file analysis | Default | `@vscode` |
| Planning, architecture | `Plan` | N/A |

