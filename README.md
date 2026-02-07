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

> **Status**: v2.4 - Production-ready local AI orchestration with GitHub Actions integration

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [GitHub Project Boards](#github-project-boards)
- [Multi-Runner System](#multi-runner-system)
- [Docker Deployment](#docker-deployment)
- [Local AI Providers](#local-ai-providers)
- [CLI Reference](#cli-reference)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

## Overview

SLATE is a local-first AI orchestration system that:

- **Runs entirely on your machine** - No cloud dependencies, no data leaves localhost
- **Coordinates multiple AI models** - Ollama, Foundry Local, and API-based models
- **Optimizes for your hardware** - Auto-detects GPUs and configures optimal settings
- **Manages complex workflows** - GitHub Projects, multi-runner execution, parallel processing

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              S.L.A.T.E. v2.4                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    GitHub Projects V2 (Task Source)                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │    │
│  │  │  KANBAN  │  │   BUGS   │  │ ITERATIVE│  │ ROADMAP  │  │ PLANNING │   │    │
│  │  │  (5)     │  │   (7)    │  │   (8)    │  │  (10)    │  │   (4)    │   │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │    │
│  └───────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────┘    │
│          │             │             │             │             │               │
│          └─────────────┴──────┬──────┴─────────────┴─────────────┘               │
│                               ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     slate_project_board.py                               │    │
│  │                   (Bidirectional Sync Engine)                            │    │
│  └──────────────────────────────┬──────────────────────────────────────────┘    │
│                                 ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        current_tasks.json                                │    │
│  │                      (Local Task Queue)                                  │    │
│  └──────────────────────────────┬──────────────────────────────────────────┘    │
│                                 ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     Multi-Runner System (19 Runners)                     │    │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │    │
│  │  │  GPU Runners (7)           │  CPU Runners (12)                    │  │    │
│  │  │  ├─ 6x Light (2GB VRAM)    │  ├─ 12x Parallel Workers             │  │    │
│  │  │  └─ 1x Heavy (12GB VRAM)   │  └─ 24 CPU Threads                   │  │    │
│  │  └───────────────────────────────────────────────────────────────────┘  │    │
│  └──────────────────────────────┬──────────────────────────────────────────┘    │
│                                 ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        AI Backend Router                                 │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │    │
│  │  │    Ollama    │    │   Foundry    │    │  External    │               │    │
│  │  │ mistral-nemo │    │    Local     │    │    APIs      │               │    │
│  │  │ :11434       │    │   :5272      │    │  (optional)  │               │    │
│  │  └──────────────┘    └──────────────┘    └──────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Map

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         SLATE Module Architecture                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  slate/                                                                   │
│  ├── Orchestration                                                        │
│  │   ├── slate_orchestrator.py      # Service lifecycle management        │
│  │   ├── slate_project_board.py     # GitHub Projects V2 sync             │
│  │   ├── slate_workflow_manager.py  # Task lifecycle & cleanup            │
│  │   └── slate_workflow_analyzer.py # Meta-workflow self-management       │
│  │                                                                        │
│  ├── Execution                                                            │
│  │   ├── slate_multi_runner.py      # 19 parallel runner coordination     │
│  │   ├── slate_runner_manager.py    # GitHub Actions runner setup         │
│  │   ├── slate_runner_benchmark.py  # Resource capacity benchmarking      │
│  │   └── runner_fallback.py         # Cost-aware runner selection         │
│  │                                                                        │
│  ├── AI Backends                                                          │
│  │   ├── unified_ai_backend.py      # Central routing (FREE local first)  │
│  │   ├── foundry_local.py           # Ollama + Foundry client             │
│  │   └── ollama_client.py           # Direct Ollama integration           │
│  │                                                                        │
│  ├── System                                                               │
│  │   ├── slate_status.py            # Quick system health check           │
│  │   ├── slate_runtime.py           # Integration verification            │
│  │   ├── slate_hardware_optimizer.py# GPU detection & PyTorch setup       │
│  │   └── slate_benchmark.py         # Performance testing                 │
│  │                                                                        │
│  └── Security                                                             │
│      ├── action_guard.py            # API blocking & action validation    │
│      ├── sdk_source_guard.py        # Package publisher verification      │
│      └── pii_scanner.py             # PII detection for public boards     │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Key Features

### Local-First Design
- All services bind to `127.0.0.1` only
- No telemetry or external data collection
- Your code and prompts stay on your machine
- FREE local AI inference (no cloud costs)

### GitHub Projects Integration
- **KANBAN** (Project 5): Primary workflow source
- **BUG TRACKING** (Project 7): Auto-routed bug fixes
- **ITERATIVE DEV** (Project 8): PR tracking
- **ROADMAP** (Project 10): Completed features
- Scheduled sync every 30 minutes

### Multi-Runner Execution
- **19 parallel runners** across dual GPUs
- 6 GPU light runners (2GB VRAM each)
- 1 GPU heavy runner (12GB VRAM)
- 12 CPU parallel workers
- Resource-aware task distribution

### Hardware Optimization
| Architecture | GPUs | Optimizations |
|-------------|------|---------------|
| **Blackwell** | RTX 50xx | TF32, BF16, Flash Attention 2, CUDA Graphs |
| **Ada Lovelace** | RTX 40xx | TF32, BF16, Flash Attention, CUDA Graphs |
| **Ampere** | RTX 30xx, A100 | TF32, BF16, Flash Attention |
| **CPU-Only** | Any | AVX2/AVX-512 optimizations |

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.git
cd S.L.A.T.E

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Quick status check
python slate/slate_status.py --quick

# Full system check
python slate/slate_runtime.py --check-all

# Check project boards
python slate/slate_project_board.py --status
```

### 3. Start SLATE

```bash
# Start all services (dashboard, runner, workflow monitor)
python slate/slate_orchestrator.py start

# Or start dashboard only
python agents/slate_dashboard_server.py
```

Open http://127.0.0.1:8080 in your browser.

## GitHub Project Boards

SLATE uses GitHub Projects V2 as the task management layer with bidirectional sync.

### Board Structure

| # | Board | Purpose | Auto-Route Keywords |
|---|-------|---------|---------------------|
| 5 | **KANBAN** | Active work queue | Default for pending tasks |
| 7 | **BUG TRACKING** | Bug fixes | bug, fix, crash, error |
| 8 | **ITERATIVE DEV** | Pull requests | PRs auto-added |
| 10 | **ROADMAP** | Completed features | feat, add, implement |
| 4 | **PLANNING** | Design work | plan, design, architect |

### Project Board Commands

```bash
# Check all boards status
python slate/slate_project_board.py --status

# Update all boards from current_tasks.json
python slate/slate_project_board.py --update-all

# Sync KANBAN items to local tasks
python slate/slate_project_board.py --sync

# Push pending tasks to KANBAN
python slate/slate_project_board.py --push

# Process KANBAN items
python slate/slate_project_board.py --process
```

### Automation

The `project-automation.yml` workflow:
- **Scheduled**: Runs every 30 minutes
- **PII Scanning**: Blocks sensitive data from public boards
- **Auto-routing**: Issues/PRs added to boards by labels
- **Bidirectional sync**: current_tasks.json ↔ GitHub Projects

## Multi-Runner System

SLATE maximizes hardware utilization with 19 parallel runners.

### Runner Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│                  Multi-Runner Distribution                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPU 0 (RTX 5070 Ti - 16GB)          GPU 1 (RTX 5070 Ti - 16GB) │
│  ┌─────────────────────────┐         ┌─────────────────────────┐│
│  │ Light Runners (3)       │         │ Light Runners (3)       ││
│  │ ├─ 2GB VRAM each        │         │ ├─ 2GB VRAM each        ││
│  │ └─ Inference tasks      │         │ └─ Inference tasks      ││
│  │                         │         │                         ││
│  │ Heavy Runner (shared)   │◄────────┤ Heavy Runner (shared)   ││
│  │ └─ 12GB VRAM            │         │ └─ Fine-tuning, batch   ││
│  └─────────────────────────┘         └─────────────────────────┘│
│                                                                  │
│  CPU Pool (24 threads)                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ CPU Runners (12) - 2 threads each                           ││
│  │ └─ Linting, testing, file operations, git commands          ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Runner Commands

```bash
# Check runner capacity and status
python slate/slate_multi_runner.py --status

# Run benchmark to calibrate runners
python slate/slate_runner_benchmark.py

# Dispatch task with resource awareness
python slate/slate_multi_runner.py --dispatch "task_type=inference"
```

### Task Routing

| Task Type | Runner Type | Resource |
|-----------|-------------|----------|
| Inference | GPU Light | 2GB VRAM |
| Fine-tuning | GPU Heavy | 12GB VRAM |
| Code generation | GPU Light | 2GB VRAM |
| Linting | CPU | 2 threads |
| Testing | CPU | 2 threads |
| Git operations | CPU | 2 threads |

## Docker Deployment

SLATE provides GPU and CPU Docker variants.

### Docker Images

| Image | Base | Size | Use Case |
|-------|------|------|----------|
| `slate:latest` | CUDA 12.4 | ~8GB | GPU inference |
| `slate:cpu` | Python 3.11-slim | ~500MB | CPU-only deployment |

### Quick Start with Docker

```bash
# GPU version with Ollama sidecar
docker-compose up -d

# CPU-only version
docker build -f Dockerfile.cpu -t slate:cpu .
docker run -p 8080:8080 slate:cpu
```

### Docker Compose Services

```yaml
services:
  slate:           # Main SLATE container (GPU)
  slate-ollama:    # Ollama sidecar for LLM inference
```

## Local AI Providers

SLATE prioritizes FREE local AI backends.

### Provider Priority

```
1. Ollama (localhost:11434)      - FREE, GPU-optimized
2. Foundry Local (localhost:5272) - FREE, ONNX efficiency
3. External APIs                  - Blocked by default (ActionGuard)
```

### Ollama Setup

```bash
# Install Ollama
winget install Ollama.Ollama  # Windows
curl -fsSL https://ollama.com/install.sh | sh  # Linux

# Pull recommended models
ollama pull mistral-nemo
ollama pull phi
ollama pull codellama

# Verify
curl http://127.0.0.1:11434/api/tags
```

### AI Backend Commands

```bash
# Check all backends
python slate/unified_ai_backend.py --status

# List local models
python slate/foundry_local.py --models

# Generate with local model
python slate/foundry_local.py --generate "Explain async/await"
```

## CLI Reference

### System Status

```bash
python slate/slate_status.py --quick          # Quick health check
python slate/slate_status.py --json           # JSON output
python slate/slate_runtime.py --check-all     # Full integration check
```

### Orchestrator

```bash
python slate/slate_orchestrator.py start      # Start all services
python slate/slate_orchestrator.py stop       # Stop all services
python slate/slate_orchestrator.py status     # Check service status
```

### Project Boards

```bash
python slate/slate_project_board.py --status      # Board status
python slate/slate_project_board.py --update-all  # Sync all boards
python slate/slate_project_board.py --push        # Push to KANBAN
python slate/slate_project_board.py --sync        # Pull from KANBAN
```

### Workflow Management

```bash
python slate/slate_workflow_manager.py --status   # Task status
python slate/slate_workflow_manager.py --cleanup  # Clean stale tasks
python slate/slate_workflow_manager.py --enforce  # Enforce completion
```

### Hardware

```bash
python slate/slate_hardware_optimizer.py              # Detect GPUs
python slate/slate_hardware_optimizer.py --optimize   # Apply optimizations
python slate/slate_benchmark.py                       # Run benchmarks
```

## Security

### Local-Only Architecture

- All servers bind to `127.0.0.1`
- ActionGuard blocks ALL paid cloud APIs
- SDK Source Guard validates package publishers
- PII Scanner protects public project boards

### Protected Components

| Guard | Purpose |
|-------|---------|
| `action_guard.py` | Block unauthorized API calls |
| `sdk_source_guard.py` | Verify package publishers |
| `pii_scanner.py` | Detect PII before public exposure |

### Trusted Publishers Only

Only packages from verified sources are allowed:
- Microsoft (azure-*, onnxruntime)
- NVIDIA (nvidia-cuda-*, triton)
- Meta (torch, torchvision)
- Google (tensorflow, jax)
- Hugging Face (transformers, datasets)

## Contributing

### Fork Validation

External contributors must pass security checks:

```bash
# Initialize fork
python slate/slate_fork_manager.py --init

# Validate before PR
python slate/slate_fork_manager.py --validate
```

### Required Checks

- **Security Gate**: No workflow modifications
- **SDK Source Guard**: Trusted publishers only
- **SLATE Prerequisites**: Core modules valid
- **Malicious Code Scan**: No obfuscated code

### Code Style

- Python: Type hints required, Google-style docstrings
- Modification timestamp format: `# Modified: YYYY-MM-DDTHH:MM:SSZ | Author: NAME | Change: description`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- [GitHub Repository](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E)
- [Wiki Documentation](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/wiki)
- [Issue Tracker](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/issues)
- [Project Boards](https://github.com/orgs/SynchronizedLivingArchitecture/projects)

---

<p align="center">
  <strong>S.L.A.T.E.</strong> - Synchronized Living Architecture for Transformation and Evolution
</p>

<p align="center">
  Made with AI assistance | Local-first by design
</p>
