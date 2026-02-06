# Packages & Dependencies

This page documents S.L.A.T.E.'s package structure, dependency management, and module architecture.

## Package Overview

S.L.A.T.E. is organized into three core Python packages:

```
slate/
├── aurora_core/          # SDK — runtime, hardware, benchmarks, agents
│   ├── __init__.py       # Version: 2.4.0
│   ├── slate_status.py   # System status & health checks
│   ├── slate_runtime.py  # Runtime verification
│   ├── slate_benchmark.py        # System benchmarks
│   ├── slate_hardware_optimizer.py # GPU/CUDA optimization
│   ├── slate_fork_manager.py     # Git fork management
│   ├── install_tracker.py        # Install progress tracking + SSE
│   ├── ml_orchestrator.py        # ML pipeline orchestration
│   └── ...
├── agents/               # Dashboard, API servers, agent runners
│   ├── aurora_dashboard_server.py # FastAPI dashboard
│   ├── install_api.py    # Install progress API + SSE endpoints
│   └── ...
└── aurora_slate/         # Templates, static assets, HTML
    └── install.html      # Install progress dashboard
```

## Dependency Tiers

### Tier 1 — Core (Always Required)

| Package | Version | Purpose |
|---------|---------|---------|
| `psutil` | ≥ 5.9.0 | System monitoring, CPU/RAM/disk stats |
| `rich` | ≥ 13.0.0 | Terminal formatting, progress bars |
| `aiohttp` | ≥ 3.9.0 | Async HTTP client for API integrations |
| `python-dotenv` | ≥ 1.0.0 | Environment variable management |
| `typing-extensions` | ≥ 4.0.0 | Python typing backports |

### Tier 2 — Web / Dashboard

| Package | Version | Purpose |
|---------|---------|---------|
| `hypercorn` | ≥ 0.16.0 | ASGI server for dashboard |
| `quart` | ≥ 0.19.0 | Async Flask-like web framework |
| `fastapi` | ≥ 0.100.0 | API framework (install dashboard) |
| `uvicorn` | ≥ 0.25.0 | ASGI server (FastAPI) |

### Tier 3 — AI Toolkit (Optional)

| Package | Version | Purpose |
|---------|---------|---------|
| `transformers` | ≥ 4.40.0 | Hugging Face model loading |
| `accelerate` | ≥ 0.30.0 | Multi-GPU/mixed-precision training |
| `peft` | ≥ 0.10.0 | Parameter-efficient fine-tuning (LoRA) |
| `torch` | ≥ 2.2.0 | PyTorch (CPU or CUDA) |
| `bitsandbytes` | ≥ 0.43.0 | 4-bit/8-bit quantization |

### Tier 4 — Observability

| Package | Version | Purpose |
|---------|---------|---------|
| `opentelemetry-api` | ≥ 1.20.0 | Tracing API |
| `opentelemetry-sdk` | ≥ 1.20.0 | Tracing SDK |
| `opentelemetry-exporter-otlp` | ≥ 1.20.0 | OTLP trace export |

### Tier 5 — Development

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥ 8.0.0 | Test framework |
| `pytest-asyncio` | ≥ 0.23.0 | Async test support |
| `pytest-cov` | ≥ 4.1.0 | Coverage reporting |
| `black` | ≥ 24.0.0 | Code formatter |
| `isort` | ≥ 5.13.0 | Import sorter |
| `mypy` | ≥ 1.8.0 | Static type checker |

## Installation Modes

### Standard Install
```bash
python install_slate.py
```
Installs Tier 1 + 2 packages from `requirements.txt`.

### Developer Install
```bash
python install_slate.py --dev
pip install -e ".[dev]"
```
Installs all tiers including testing and linting tools.

### AI-Enhanced Install
```bash
pip install -e ".[ai]"
```
Adds Tier 3 AI packages for model training and inference.

### GPU-Accelerated
```bash
python aurora_core/slate_hardware_optimizer.py --install-pytorch
```
Installs the correct PyTorch build for your NVIDIA GPU's compute capability:
- **Blackwell** (CC 12.x): CUDA 12.8+
- **Ada Lovelace** (CC 8.9): CUDA 12.1+
- **Ampere** (CC 8.x): CUDA 11.8+
- **Turing** (CC 7.5): CUDA 11.8

## pyproject.toml

SLATE uses modern Python packaging via `pyproject.toml`:

```toml
[project]
name = "slate"
version = "2.4.0"
requires-python = ">=3.11"

[project.scripts]
slate-status = "aurora_core.slate_status:main"
slate-runtime = "aurora_core.slate_runtime:main"
slate-benchmark = "aurora_core.slate_benchmark:main"
slate-hardware = "aurora_core.slate_hardware_optimizer:main"
```

After `pip install -e .`, these CLI commands become available:

```bash
slate-status         # Quick health check
slate-runtime        # Full runtime verification
slate-benchmark      # System benchmarks
slate-hardware       # Hardware detection + optimization
```

## Automated Dependency Updates

SLATE uses [Dependabot](https://docs.github.com/en/code-security/dependabot) for automated dependency management:

- **pip** dependencies scanned weekly (Mondays)
- **GitHub Actions** versions scanned weekly
- Dependencies grouped by category (ai-toolkit, observability, testing, web)
- PRs auto-created with `deps:` or `ci:` commit prefixes

Configuration: [`.github/dependabot.yml`](../../.github/dependabot.yml)

## Module Import Map

```
aurora_core.slate_status          → slate-status CLI
aurora_core.slate_runtime         → slate-runtime CLI
aurora_core.slate_benchmark       → slate-benchmark CLI
aurora_core.slate_hardware_optimizer → slate-hardware CLI
aurora_core.slate_fork_manager    → Git fork management
aurora_core.install_tracker       → Install progress tracking
aurora_core.ml_orchestrator       → ML pipeline orchestration
agents.aurora_dashboard_server    → Dashboard web server
agents.install_api                → Install API + SSE
```

## Version Management

SLATE version is defined in:
- `pyproject.toml` → `project.version = "2.4.0"`
- `aurora_core/__init__.py` → `__version__ = "2.4.0"`

The CI pipeline validates version consistency across these files.
