# S.L.A.T.E. Development Guidelines

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

## Local AI Providers (FREE - No Cloud Costs)

| Provider | Port | Models | Status |
|----------|------|--------|--------|
| Ollama | 11434 | mistral-nemo, llama3.2, phi, llama2, mistral | Active |
| Foundry Local | 5272 | Phi-3, Mistral-7B (ONNX) | Active |

```powershell
# Check provider status
.\.venv\Scripts\python.exe aurora_core/foundry_local.py --check

# List all local models
.\.venv\Scripts\python.exe aurora_core/foundry_local.py --models

# Generate with auto-provider selection
.\.venv\Scripts\python.exe aurora_core/foundry_local.py --generate "your prompt"

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
.\.venv\Scripts\python.exe aurora_core/unified_ai_backend.py --status

# Execute task with auto-routing
.\.venv\Scripts\python.exe aurora_core/unified_ai_backend.py --task "your task"
```

**Key Files**:
- `aurora_core/unified_ai_backend.py` - Central routing (Ollama, Foundry, Copilot, Claude)
- `aurora_core/foundry_local.py` - Foundry Local + Ollama unified client
- `aurora_core/inference_instructions.py` - ML-based code generation guidance
- `aurora_core/action_guard.py` - Security (blocks paid APIs)

## Project Structure

```text
aurora_core/       # Core SLATE engine modules
agents/            # Agent implementations (ALPHA, BETA, GAMMA, DELTA)
slate_core/        # Shared infrastructure (agents, locks, memory, GPU scheduler)
specs/             # Active specifications
src/               # Source code (backend/frontend)
tests/             # Test suite
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
# Run tests
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Run with coverage
.\.venv\Scripts\python.exe -m pytest tests/ --cov=aurora_core -q

# Lint
ruff check .

# Start dashboard
.\.venv\Scripts\python.exe agents/aurora_dashboard_server.py

# Start orchestrator (prevents duplicate processes)
.\.venv\Scripts\python.exe slate_core/slate_orchestrator.py start

# Check SLATE status
.\.venv\Scripts\python.exe aurora_core/slate_status.py --quick

# Check task summary
.\.venv\Scripts\python.exe aurora_core/slate_status.py --tasks
```

## Code Style

- Python: Type hints required. Google-style docstrings. Use `Annotated` for tool parameters.
- Imports: Add `WORKSPACE_ROOT` to `sys.path` when importing cross-module.
- UI: Glassmorphism theme (75% opacity, muted pastels). System fonts only (Consolas, Segoe UI).
- Task files: Always use `slate_core/file_lock.py` for `current_tasks.json` (prevents race conditions).

## Security Architecture — LOCAL ONLY

- **All servers bind to `127.0.0.1` only** — never `0.0.0.0`
- No external network calls unless explicitly requested by user
- ActionGuard (`aurora_core/action_guard.py`) validates all agent actions
- Content Security Policy enforced — no external CDN/fonts
- Rate limiting active on dashboard API endpoints

### SDK Source Guard (Trusted Publishers Only)

SDKSourceGuard (`aurora_core/sdk_source_guard.py`) enforces that ALL packages come from trusted primary sources:

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
.\.venv\Scripts\python.exe aurora_core/sdk_source_guard.py --report

# Validate a specific package
.\.venv\Scripts\python.exe aurora_core/sdk_source_guard.py --validate "some-package"

# Check all requirements.txt packages
.\.venv\Scripts\python.exe aurora_core/sdk_source_guard.py --check-requirements
```

**Blocked Sources:**
- Unknown PyPI publishers
- Untrusted GitHub organizations
- Known typosquatting packages
- Suspicious naming patterns

## Agent System

| Agent | Role | Preference |
| --- | --- | --- |
| ALPHA | Coding & implementation | GPU-preferred |
| BETA | Testing & validation | GPU-preferred |
| GAMMA | Planning & triage | CPU-preferred |
| DELTA | Claude Code bridge | CLI-based |

Tasks in `current_tasks.json` use `assigned_to` field for routing.
Tasks with `assigned_to: "auto"` use ML-based smart routing.

## Task Management

- Task queue: `current_tasks.json` (use FileLock for atomic access)
- Priorities: `DO_THIS_NOW.txt` for immediate priorities
- Tech tree: `.slate_tech_tree/tech_tree.json` directs development focus
- Spec lifecycle: `draft → specified → planned → tasked → implementing → complete`

## Test-Driven Development (Constitution Mandate)

All code changes must be accompanied by tests. Target 50%+ coverage for `aurora_core/` and `slate_core/`.

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
# origin  https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E..git
# beta    https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.-BETA.git
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
3. Run `python aurora_core/slate_fork_manager.py --init` to set up
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

## Self-Hosted GitHub Actions Runners

SLATE supports self-hosted runners for GPU-accelerated CI/CD:

```powershell
# Check runner status
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --status

# Download and setup
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --download

# Configure (get token from GitHub repo settings)
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --configure --token YOUR_TOKEN

# Start runner (interactive)
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --start

# Start as Windows service (persistent)
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --start --service
```

**Runner Labels** (auto-detected):
- `self-hosted`, `slate`, `gpu`, `windows`
- `cuda`, `gpu-2` (based on GPU count)
- `blackwell` / `ada-lovelace` / `ampere` (GPU architecture)

**Key Files**:
- `aurora_core/slate_runner_manager.py` - Runner management CLI
- `.github/workflows/gpu-tests.yml` - GPU test workflow
- `docs/wiki/Self-Hosted-Runners.md` - Full documentation

**Get Registration Token**:
1. Go to https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./settings/actions/runners
2. Click "New self-hosted runner"
3. Copy the token
