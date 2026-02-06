# S.L.A.T.E.

**S**ystem **L**earning **A**gent for **T**ask **E**xecution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)

> A local-only AI agent orchestration system with automatic hardware detection and optimization.

This is an experimental layer built between AI / and software. 
- THIS IS NOT SUITABLE FOR PRODUCTION - 
- USE AT OWN RISK 

## Features

- **🔒 Local-Only**: All operations run on 127.0.0.1 - your data never leaves your machine
- **🎮 GPU Auto-Detection**: Automatically detects NVIDIA GPUs (Blackwell, Ada Lovelace, Ampere, Turing)
- **⚡ PyTorch Optimization**: Configures torch.compile, TF32, BF16, Flash Attention based on your hardware
- **🤖 Multi-Agent Orchestration**: Coordinate multiple AI agents with task routing
- **📊 Dashboard**: Real-time monitoring via web interface
- **🔧 AI Toolkit Integration**: Microsoft AI Toolkit for VS Code with LoRA/QLoRA fine-tuning
- **📋 Task Management**: JSON-based task queue with status tracking

## Quick Start

```bash
# Clone the repository
git clone https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E..git
cd S.L.A.T.E.

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run installation
python install_slate.py
```

## Requirements

- **Python**: 3.11+
- **OS**: Windows 10/11, Linux, macOS
- **GPU** (Optional): NVIDIA GPU with CUDA support
- **RAM**: 16GB+ recommended
- **Disk**: 10GB+ free space

## Hardware Support

| Architecture | GPUs | Features |
|-------------|------|----------|
| **Blackwell** | RTX 50xx | TF32, BF16, Flash Attention 2, CUDA Graphs |
| **Ada Lovelace** | RTX 40xx | TF32, BF16, Flash Attention, CUDA Graphs |
| **Ampere** | RTX 30xx, A100 | TF32, BF16, Flash Attention |
| **Turing** | RTX 20xx | FP16, Flash Attention |
| **CPU-Only** | Any | AVX2 optimizations |

## Core Commands

```bash
# Check system status
python aurora_core/slate_status.py --quick

# Full integration check (10 integrations)
python aurora_core/slate_runtime.py --check-all

# Hardware detection and optimization
python aurora_core/slate_hardware_optimizer.py
python aurora_core/slate_hardware_optimizer.py --install-pytorch  # Install CUDA PyTorch
python aurora_core/slate_hardware_optimizer.py --optimize  # Apply optimizations

# System benchmark
python aurora_core/slate_benchmark.py

# AI Toolkit
python aurora_core/slate_ai_toolkit_integration.py --status
python aurora_core/slate_ai_toolkit_integration.py --install-deps

# Terminal monitor (prevent freezes)
python aurora_core/slate_terminal_monitor.py --status
```

## Project Structure

```
S.L.A.T.E./
├── aurora_core/           # Core Python modules
│   ├── slate_status.py           # Quick status check
│   ├── slate_runtime.py          # Integration checker
│   ├── slate_benchmark.py        # System benchmarks
│   ├── slate_hardware_optimizer.py  # GPU detection
│   ├── slate_ai_toolkit_integration.py  # AI Toolkit
│   └── slate_terminal_monitor.py  # Safe terminal execution
├── agents/                # Dashboard and agent servers
├── tests/                 # pytest test suite
├── .github/              # CI/CD workflows
├── install_slate.py      # Installation script
├── requirements.txt      # Python dependencies
└── README.md
```

## Agent System

SLATE uses a multi-agent architecture:

| Agent | Role | Target |
|-------|------|--------|
| **ALPHA** | Coding, Implementation | GPU |
| **BETA** | Testing, Validation | GPU |
| **GAMMA** | Planning, Research | CPU |
| **DELTA** | External Integration | API |

## Dashboard

Start the dashboard server:

```bash
python agents/aurora_dashboard_server.py
```

Open in browser: http://127.0.0.1:8080

## Configuration

SLATE uses YAML-based configuration files:

- `.github/copilot-instructions.md` - Agent instructions
- `.github/slate.slate` - Main configuration
- `current_tasks.json` - Task queue

## Security

- **Local-Only**: All services bind to 127.0.0.1
- **No Telemetry**: No data sent externally
- **BYOK**: Bring Your Own Key for API access
- **IP Protection**: Your code stays on your machine

## Integration with VS Code

SLATE integrates with:

- **GitHub Copilot**: Via Copilot SDK
- **AI Toolkit**: Microsoft AI Toolkit extension
- **Ollama**: Local LLM inference

## Contributing

Contributions are welcome! Please read our contributing guidelines.

1. Fork the repository
2. Create a feature branch
3. Follow SLATE format rules (timestamps + author attribution)
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.)
- [Issue Tracker](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./issues)

---

**S.L.A.T.E.** - Synchronized Living Architecture for Transformation and Evolution
