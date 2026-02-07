# Getting Started
<!-- Modified: 2026-02-07T14:30:00Z | Author: CLAUDE | Change: Add themed styling and visual elements -->

<div align="center">

**Turn your local hardware into an AI operations center for GitHub**

*One command. Full ecosystem.*

</div>

---

## Quick Install

```bash
git clone https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.git && cd S.L.A.T.E && python install_slate.py
```

The installer handles everything automatically with a live dashboard at **http://127.0.0.1:8080**

## Prerequisites

<table>
<tr>
<th>Required</th>
<th>Recommended</th>
</tr>
<tr>
<td>
<ul>
<li><strong>Python 3.11+</strong> - <a href="https://www.python.org/downloads/">Download</a></li>
<li><strong>Git</strong> - <a href="https://git-scm.com/downloads">Download</a></li>
<li><strong>8GB RAM</strong> minimum</li>
</ul>
</td>
<td>
<ul>
<li><strong>NVIDIA GPU</strong> with CUDA (RTX 20xx+)</li>
<li><strong>16GB+ RAM</strong></li>
<li><strong>VS Code</strong> with Claude Code extension</li>
</ul>
</td>
</tr>
</table>

## What Gets Installed

<table>
<tr>
<th>Component</th>
<th>Purpose</th>
<th>Cost</th>
</tr>
<tr>
<td><strong>Ollama</strong></td>
<td>Local LLM inference (mistral-nemo, llama3.2)</td>
<td align="center"><code>FREE</code></td>
</tr>
<tr>
<td><strong>ChromaDB</strong></td>
<td>Vector store for codebase memory</td>
<td align="center"><code>FREE</code></td>
</tr>
<tr>
<td><strong>PyTorch</strong></td>
<td>GPU-optimized for your hardware</td>
<td align="center"><code>FREE</code></td>
</tr>
<tr>
<td><strong>GitHub Runner</strong></td>
<td>Self-hosted Actions runner with AI access</td>
<td align="center"><code>FREE</code></td>
</tr>
<tr>
<td><strong>Dashboard</strong></td>
<td>Real-time monitoring at localhost:8080</td>
<td align="center"><code>FREE</code></td>
</tr>
</table>

## Manual Installation

If you prefer step-by-step:

```bash
# Clone
git clone https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.git
cd S.L.A.T.E

# Virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Dependencies
pip install -r requirements.txt

# Full install with dashboard
python install_slate.py
```

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
Version: 2.5.0
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
- Workflow system

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
      "assigned_to": "workflow",
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
