# Getting Started

This guide will walk you through installing and setting up SLATE on your system.

## Prerequisites

### Required
- **Python 3.11+** - [Download](https://www.python.org/downloads/)
- **Git** - [Download](https://git-scm.com/downloads)
- **8GB RAM** minimum

### Recommended
- **NVIDIA GPU** with CUDA support (RTX 20xx or newer)
- **16GB+ RAM**
- **Ollama** for local LLM inference

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.git
cd S.L.A.T.E
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Installation Script

```bash
python install_slate.py
```

This script will:
1. Detect your hardware (GPU, CPU, RAM)
2. Install appropriate PyTorch version
3. Configure Ollama integration (if available)
4. Set up configuration files
5. Verify the installation

## Quick Start

### Verify Installation

```bash
# Check system status
python slate/slate_status.py --quick
```

Expected output:
```
SLATE Status
============
Version: 2.4.0
Python: 3.11.x
GPU: NVIDIA RTX xxxx (detected)
Ollama: Connected
Status: Ready
```

### Start the Dashboard

```bash
python agents/slate_dashboard_server.py
```

Open your browser to: http://127.0.0.1:8080

### Check All Integrations

```bash
python slate/slate_runtime.py --check-all
```

This verifies:
- Ollama connection
- Foundry Local (if installed)
- GPU detection
- ChromaDB vector store
- Task queue
- File locks
- Configuration files
- Agent system

## Your First Task

### Using the Task Queue

Tasks are managed in `current_tasks.json`:

```json
{
  "tasks": [
    {
      "id": "task_001",
      "title": "Fix login bug",
      "description": "Users can't login with special characters in password",
      "status": "pending",
      "assigned_to": "ALPHA",
      "priority": 2
    }
  ]
}
```

### Using Python API

```python
from slate import create_task, get_tasks

# Create a new task
task = create_task(
    title="Implement dark mode",
    description="Add dark mode toggle to settings page",
    priority=3
)
print(f"Created task: {task.id}")

# List all tasks
tasks = get_tasks()
for t in tasks:
    print(f"- [{t.status}] {t.title}")
```

## Installing Ollama

For the best local AI experience, install Ollama:

### Windows
```bash
winget install Ollama.Ollama
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### macOS
```bash
brew install ollama
```

### Pull Recommended Models

```bash
# Primary model (7B, fast)
ollama pull mistral-nemo

# Lightweight model (2.7B, very fast)
ollama pull phi

# Code-specialized model
ollama pull codellama
```

### Verify Ollama

```bash
curl http://127.0.0.1:11434/api/tags
```

## Hardware Optimization

SLATE automatically detects your hardware. For manual optimization:

```bash
# Detect hardware capabilities
python slate/slate_hardware_optimizer.py

# Install optimal PyTorch for your GPU
python slate/slate_hardware_optimizer.py --install-pytorch

# Apply runtime optimizations
python slate/slate_hardware_optimizer.py --optimize
```

## Next Steps

- [Learn about the Architecture](Architecture)
- [Understand the Agent System](Agents)
- [Configure AI Backends](AI-Backends)
- [Explore CLI Commands](CLI-Reference)
