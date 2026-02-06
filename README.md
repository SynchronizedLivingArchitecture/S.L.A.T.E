<p align="center">
  <img src="docs/assets/slate-logo.png" alt="SLATE Logo" width="200" height="200">
</p>

<h1 align="center">S.L.A.T.E.</h1>

<p align="center">
  <strong>Synchronized Living Architecture for Transformation and Evolution</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg" alt="PyTorch 2.7+"></a>
  <a href="https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build Status"></a>
  <a href="https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/wiki"><img src="https://img.shields.io/badge/docs-wiki-blue.svg" alt="Documentation"></a>
</p>

<p align="center">
  A <strong>local-first</strong> AI agent orchestration framework that coordinates multiple AI models<br>
  while keeping your code and data on your machine.
</p>

---

> **Warning**: This project is experimental and entirely "vibe-coded" - its accuracy relies on AI assistance. Not suitable for production use.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Core Modules](#core-modules)
- [Agent System](#agent-system)
- [Local AI Providers](#local-ai-providers)
- [Dashboard](#dashboard)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Security](#security)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

## Overview

SLATE is a local-first AI orchestration system that:

- **Runs entirely on your machine** - No cloud dependencies, no data leaves localhost
- **Coordinates multiple AI models** - Ollama, Foundry Local, and API-based models
- **Optimizes for your hardware** - Auto-detects GPUs and configures optimal settings
- **Manages complex workflows** - Task queuing, agent routing, and parallel execution

```
┌─────────────────────────────────────────────────────────────────┐
│                        S.L.A.T.E.                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │  ALPHA  │  │  BETA   │  │  GAMMA  │  │  DELTA  │   Agents   │
│  │ Coding  │  │ Testing │  │Planning │  │ Bridge  │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                  │
│  ┌────┴────────────┴────────────┴────────────┴────┐            │
│  │              Task Router / Scheduler            │            │
│  └────────────────────┬────────────────────────────┘            │
│                       │                                         │
│  ┌────────────────────┴────────────────────────────┐            │
│  │              AI Backend Selector                │            │
│  └──────┬──────────────┬──────────────┬───────────┘            │
│         │              │              │                         │
│  ┌──────┴──────┐ ┌─────┴─────┐ ┌──────┴──────┐                 │
│  │   Ollama    │ │  Foundry  │ │  External   │   Backends      │
│  │ mistral-nemo│ │   Local   │ │    APIs     │                 │
│  └─────────────┘ └───────────┘ └─────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### Local-First Design
- All services bind to `127.0.0.1` only
- No telemetry or external data collection
- Your code and prompts stay on your machine

### Multi-Model Orchestration
- **Ollama**: mistral-nemo, llama3.2, phi, codellama
- **Foundry Local**: ONNX-optimized models (Phi-3, Mistral-7B)
- **External APIs**: Optional Claude, GPT integration

### Hardware Optimization
| Architecture | GPUs | Optimizations |
|-------------|------|---------------|
| **Blackwell** | RTX 50xx | TF32, BF16, Flash Attention 2, CUDA Graphs |
| **Ada Lovelace** | RTX 40xx | TF32, BF16, Flash Attention, CUDA Graphs |
| **Ampere** | RTX 30xx, A100 | TF32, BF16, Flash Attention |
| **Turing** | RTX 20xx | FP16, Flash Attention |
| **CPU-Only** | Any | AVX2/AVX-512 optimizations |

### Development Tools
- Real-time dashboard (port 8080)
- Task queue with priority management
- Metrics aggregation and profiling
- Feature flags for safe rollouts
- Audit trail logging

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.git
cd S.L.A.T.E

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install SLATE

```bash
python install_slate.py
```

This will:
- Detect your hardware configuration
- Install appropriate PyTorch version
- Configure Ollama integration
- Set up the dashboard

### 3. Verify Installation

```bash
# Quick status check
python slate/slate_status.py --quick

# Full system check
python slate/slate_runtime.py --check-all
```

### 4. Start the Dashboard

```bash
python agents/slate_dashboard_server.py
```

Open http://127.0.0.1:8080 in your browser.

## System Requirements

### Minimum
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.11+
- **RAM**: 8GB
- **Disk**: 5GB free space

### Recommended
- **GPU**: NVIDIA RTX 3060 or better
- **RAM**: 16GB+
- **VRAM**: 8GB+
- **Disk**: 20GB+ (for models)

### Optional
- **Ollama**: For local LLM inference
- **Foundry Local**: For ONNX-optimized models
- **Redis**: For distributed task queuing

## Core Modules

```
slate/
├── Core Infrastructure
│   ├── message_broker.py      # Redis/memory pub-sub messaging
│   ├── rag_memory.py          # ChromaDB vector memory
│   ├── gpu_scheduler.py       # GPU workload distribution
│   ├── file_lock.py           # Thread-safe task queue
│   └── llm_cache.py           # Response caching
│
├── AI Backends
│   ├── unified_ai_backend.py  # Central routing
│   ├── ollama_client.py       # Ollama integration
│   ├── foundry_local.py       # Foundry Local client
│   └── action_guard.py        # Security layer
│
├── System Tooling
│   ├── metrics_aggregator.py  # Prometheus-format metrics
│   ├── integration_health_check.py  # Health diagnostics
│   ├── performance_profiler.py  # CPU/memory profiling
│   ├── load_balancer.py       # Agent work distribution
│   ├── feature_flags.py       # Feature toggles
│   └── audit_trail.py         # Action logging
│
└── CLI Tools
    ├── slatepi_status.py      # System status
    ├── slatepi_runtime.py     # Integration checker
    └── slatepi_benchmark.py   # Performance testing
```

## Agent System

SLATE coordinates multiple specialized agents:

| Agent | Role | Hardware | Capabilities |
|-------|------|----------|--------------|
| **ALPHA** | Implementation | GPU | Code generation, refactoring, bug fixes |
| **BETA** | Validation | GPU | Testing, code review, security analysis |
| **GAMMA** | Planning | CPU | Architecture, research, documentation |
| **DELTA** | Integration | API | External service coordination |

### Task Routing

Tasks are automatically routed based on:
- Complexity score (0-100)
- Hardware requirements
- Agent availability
- Priority level

```python
# Example: Create a task
from slate import create_task

task = create_task(
    title="Implement user authentication",
    description="Add JWT-based auth to the API",
    priority=2,  # 1=highest, 5=lowest
    assigned_to="ALPHA"
)
```

## Local AI Providers

### Ollama (Recommended)

```bash
# Install Ollama (Windows)
winget install Ollama.Ollama

# Pull recommended models
ollama pull mistral-nemo
ollama pull phi
ollama pull codellama

# Verify
curl http://127.0.0.1:11434/api/tags
```

### Foundry Local

```bash
# Install via foundry CLI
foundry model download microsoft/Phi-3.5-mini-instruct-onnx

# Check status
python slate/foundry_local.py --check
```

### Provider Priority

SLATE automatically selects the best available backend:

```
1. Ollama (localhost:11434) - Fast, GPU-optimized
2. Foundry Local (localhost:5272) - ONNX efficiency
3. External APIs - Fallback only
```

## Dashboard

The web dashboard provides:

- **System Overview**: CPU, GPU, memory usage
- **Task Queue**: Pending, in-progress, completed tasks
- **Agent Status**: Health and activity of each agent
- **Metrics**: Response times, token usage, error rates
- **Logs**: Real-time log streaming

### Glassmorphism Theme

The UI uses a modern glassmorphism design:
- 75% opacity panels
- Muted pastel accents
- System fonts (Consolas, Segoe UI)

## CLI Reference

### Status Commands

```bash
# Quick status
python slate/slate_status.py --quick

# Task summary
python slate/slate_status.py --tasks

# Full integration check
python slate/slate_runtime.py --check-all
```

### Hardware Commands

```bash
# Detect hardware
python slate/slate_hardware_optimizer.py

# Install optimal PyTorch
python slate/slate_hardware_optimizer.py --install-pytorch

# Apply optimizations
python slate/slate_hardware_optimizer.py --optimize
```

### Benchmark Commands

```bash
# Full benchmark
python slate/slate_benchmark.py

# CPU only
python slate/slate_benchmark.py --cpu-only

# Quick benchmark
python slate/slate_benchmark.py --quick
```

### AI Backend Commands

```bash
# Check all backends
python slate/unified_ai_backend.py --status

# Generate with specific backend
python slate/foundry_local.py --generate "Explain async/await"

# List local models
python slate/foundry_local.py --models
```

## Configuration

### Main Configuration

Configuration is stored in multiple locations:

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package configuration |
| `.github/copilot-instructions.md` | AI agent instructions |
| `current_tasks.json` | Active task queue |
| `.specify/memory/constitution.md` | Project principles |

### Environment Variables

```bash
# Optional: Override defaults
export SLATE_OLLAMA_HOST=127.0.0.1
export SLATE_OLLAMA_PORT=11434
export SLATE_DASHBOARD_PORT=8080
export SLATE_LOG_LEVEL=INFO
```

### Feature Flags

```python
from slate import is_enabled

if is_enabled("slate.new_router"):
    # Use new ML-based routing
    pass
```

## Security

### Local-Only Architecture

- All servers bind to `127.0.0.1`
- No external network calls without explicit request
- ActionGuard validates all agent actions

### Content Security

- CSP headers enforced
- No external CDN or fonts
- Rate limiting on API endpoints

### API Key Management

- BYOK (Bring Your Own Key) model
- Keys stored in local environment
- Never transmitted or logged

## Contributing

We welcome contributions! Please follow these guidelines:

### Code Style

- Python: Type hints required, Google-style docstrings
- Use `Annotated` for tool parameters
- Include modification timestamps and author attribution

### Commit Format

```
feat(module): Short description

- Detailed change 1
- Detailed change 2

Co-Authored-By: Your Name <email>
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

## Roadmap

### v2.5 (Current Focus)
- [ ] Enhanced documentation and wiki
- [ ] Improved error handling
- [ ] Performance optimizations

### v3.0 (Planned)
- [ ] Distributed agent coordination
- [ ] Model fine-tuning pipeline
- [ ] Visual workflow editor

### Future
- [ ] Multi-machine clustering
- [ ] Custom model training
- [ ] Plugin ecosystem

## GitHub Agentic Workflow System

SLATE uses GitHub as an **agentic task execution platform**. The entire project lifecycle is managed autonomously: issues become tasks, agents claim work, workflows execute, and PRs track completion.

### Architecture

```
GitHub Issues → current_tasks.json → Agent Routing → Workflow Dispatch → PR/Commit
     ↓                 ↓                  ↓                  ↓              ↓
  Tracking         Task Queue         ALPHA/BETA        Self-hosted     Completion
                                     GAMMA/DELTA          Runner
```

### Workflow Overview

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR | Smoke tests, linting, unit tests, security scans |
| `slate.yml` | Push to core paths | SLATE-specific validation (tech tree, task queue, agents) |
| `nightly.yml` | Daily at 4am UTC | Full test suite, dependency checks, SDK import audit |
| `cd.yml` | Push to main/tags | Build EXE, create releases |
| `fork-validation.yml` | Fork PRs | Security gate, prerequisite validation |

### Auto-Configured Runner

SLATE **automatically detects and configures** the GitHub Actions runner. No manual setup required:

```bash
# Check runner status, GPU config, and GitHub state
python slate/slate_runner_manager.py --status

# Auto-configure hooks, environment, and labels
python slate/slate_runner_manager.py --setup

# Dispatch a workflow for agentic execution
python slate/slate_runner_manager.py --dispatch "ci.yml"

# JSON output for automation
python slate/slate_runner_manager.py --detect --json
```

The runner manager automatically:
- **Detects** GPU configuration (count, architecture, CUDA capability)
- **Creates** pre-job hooks for SLATE environment
- **Generates** labels (self-hosted, slate, gpu, cuda, blackwell, multi-gpu)
- **Configures** workspace paths and Python venv

### Workflow Management

```bash
# Check workflow health and task status
python slate/slate_workflow_manager.py --status

# Auto-cleanup stale, abandoned, and duplicate tasks
python slate/slate_workflow_manager.py --cleanup

# Check if new tasks can be accepted
python slate/slate_workflow_manager.py --enforce
```

Automatic task lifecycle:
- **Stale tasks** (in-progress > 4h) → auto-reset to pending
- **Abandoned tasks** (pending > 24h) → flagged for review
- **Duplicates** → auto-archived
- **Max concurrent** → 5 in-progress before blocking

### Required Status Checks

All PRs to `main` must pass:
- CodeQL Advanced
- SLATE Integration / SLATE Status Check
- SLATE Integration / Tech Tree Validation
- Fork Validation / Security Gate
- Fork Validation / SLATE Prerequisites

## Dual-Repository Development

SLATE uses a dual-repo model for structured development:

```
SLATE (origin)         = Main repository (the product)
       ↑
       │ contribute-to-main.yml
       │
SLATE-BETA (beta)      = Developer fork
```

### Setup

```bash
# Verify remotes
git remote -v
# Should show:
# origin  https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.git
# beta    https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E-BETA.git

# Add beta remote if missing
git remote add beta https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E-BETA.git
```

### Development Workflow

1. **Create feature branch**: `git checkout -b feature/my-feature`
2. **Sync with main**: `git fetch origin && git merge origin/main`
3. **Push to beta**: `git push beta HEAD:main`
4. **Contribute to main**: Run `contribute-to-main.yml` workflow

### Required Secrets

Set `MAIN_REPO_TOKEN` in BETA repo (Settings → Secrets → Actions):
- PAT with `repo` and `workflow` scope

## System Architecture

### Core Services

| Service | Port | Purpose |
|---------|------|---------|
| Dashboard | 8080 | Web UI for monitoring |
| Ollama | 11434 | Local LLM inference |
| Foundry Local | 5272 | ONNX model inference |

### Directory Structure

```
slate/                 # Core SDK modules
agents/                # Agent implementations
slate_core/            # Shared infrastructure
slate/                 # SLATE workflow tools
.github/               # CI/CD workflows
actions-runner/        # Self-hosted runner (optional)
  hooks/               # Pre/post job scripts
  _work/               # Workflow workspace
.slate_tech_tree/      # Tech tree state
.slate_nemo/           # Nemo knowledge base
.slate_errors/         # Error logs
.slate_index/          # ChromaDB index
```

### Key Configuration Files

| File | Purpose |
|------|---------|
| `.github/slate.config.yaml` | SLATE system configuration |
| `.github/workflows/*.yml` | CI/CD pipelines |
| `current_tasks.json` | Active task queue |
| `.slate_tech_tree/tech_tree.json` | Development roadmap |
| `pyproject.toml` | Package configuration |

## Fork Contribution System

External contributors use a secure fork validation system:

### Getting Started

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/S.L.A.T.E.git

# 3. Initialize SLATE workspace
python slate/slate_fork_manager.py --init --name "Your Name" --email "you@example.com"

# 4. Validate before PR
python slate/slate_fork_manager.py --validate
```

### Security Checks

Fork PRs must pass:
- **Security Gate**: No workflow modifications
- **SDK Source Guard**: Trusted publishers only
- **SLATE Prerequisites**: Core modules valid
- **ActionGuard Audit**: No security bypasses
- **Malicious Code Scan**: No obfuscated code

### Protected Files

These cannot be modified by forks:
- `.github/workflows/*`
- `.github/CODEOWNERS`
- `slate/action_guard.py`
- `slate/sdk_source_guard.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- [GitHub Repository](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E)
- [Wiki Documentation](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/wiki)
- [Issue Tracker](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/issues)
- [Discussions](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/discussions)

---

<p align="center">
  <strong>S.L.A.T.E.</strong> - Synchronized Living Architecture for Transformation and Evolution
</p>

<p align="center">
  Made with AI assistance
</p>
