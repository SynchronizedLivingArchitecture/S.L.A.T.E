# AI Backends
<!-- Modified: 2026-02-07T14:30:00Z | Author: CLAUDE | Change: Create comprehensive AI backends documentation -->

SLATE routes all AI tasks through local backends first, ensuring **zero cloud costs** for standard operations.

## Backend Priority

<table>
<tr>
<th colspan="4" align="center">AI Backend Selection Order</th>
</tr>
<tr>
<td align="center"><strong>1. Ollama</strong><br><sub>Primary local LLM</sub><br><code>:11434</code></td>
<td align="center"><strong>2. Foundry Local</strong><br><sub>ONNX optimized</sub><br><code>:5272</code></td>
<td align="center"><strong>3. ChromaDB</strong><br><sub>Vector memory</sub><br><code>local</code></td>
<td align="center"><strong>4. External APIs</strong><br><sub>Blocked by default</sub><br><code>ActionGuard</code></td>
</tr>
</table>

## Ollama

Primary local LLM backend for SLATE. Runs entirely on your hardware.

### Installation

| Platform | Command |
|:---------|:--------|
| **Windows** | `winget install Ollama.Ollama` |
| **Linux** | `curl -fsSL https://ollama.com/install.sh \| sh` |
| **macOS** | `brew install ollama` |

### Recommended Models

| Model | Size | Speed | Use Case |
|:------|:-----|:------|:---------|
| `mistral-nemo` | 7B | Fast | General purpose, code review |
| `phi` | 2.7B | Very Fast | Quick tasks, validation |
| `codellama` | 7B | Fast | Code-specific tasks |
| `llama3.2` | 3B | Fast | Lightweight inference |

```bash
# Pull recommended models
ollama pull mistral-nemo
ollama pull phi
ollama pull codellama
```

### Verification

```bash
# Check Ollama status
curl http://127.0.0.1:11434/api/tags

# Test inference
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "mistral-nemo",
  "prompt": "Hello, world!",
  "stream": false
}'
```

### SLATE Integration

```python
from slate.ollama_client import OllamaClient

client = OllamaClient()

# Check connection
status = client.check()
print(f"Ollama: {'Connected' if status else 'Offline'}")

# List models
models = client.list_models()
for model in models:
    print(f"  - {model['name']}")

# Generate
response = client.generate("Write a Python function to sort a list")
print(response)
```

### Configuration

| Environment Variable | Default | Description |
|:---------------------|:--------|:------------|
| `SLATE_OLLAMA_HOST` | `127.0.0.1` | Ollama server host |
| `SLATE_OLLAMA_PORT` | `11434` | Ollama server port |
| `SLATE_OLLAMA_MODEL` | `mistral-nemo` | Default model |
| `SLATE_OLLAMA_TIMEOUT` | `120` | Request timeout (seconds) |

---

## Foundry Local

Microsoft's ONNX-optimized local inference engine. Excellent for lightweight models.

### Installation

```powershell
# Install Foundry Local CLI
winget install Microsoft.FoundryLocal

# Download models
foundry model download microsoft/Phi-3.5-mini-instruct-onnx
foundry model download microsoft/Mistral-7B-Instruct-v0.3-onnx
```

### Available Models

| Model | Parameters | Quantization | VRAM |
|:------|:-----------|:-------------|:-----|
| Phi-3.5-mini | 3.8B | INT4 | 2GB |
| Phi-3.5-MoE | 16x3.8B | INT4 | 8GB |
| Mistral-7B | 7B | INT4 | 4GB |
| Llama-3.2 | 3B | INT4 | 2GB |

### SLATE Integration

```python
from slate.foundry_local import FoundryClient

client = FoundryClient()

# Check availability
if client.is_available():
    print("Foundry Local: Ready")

# List models
models = client.list_models()
for model in models:
    print(f"  - {model}")

# Generate
response = client.generate(
    prompt="Explain async/await in Python",
    model="phi-3.5-mini"
)
print(response)
```

### Configuration

| Environment Variable | Default | Description |
|:---------------------|:--------|:------------|
| `SLATE_FOUNDRY_HOST` | `127.0.0.1` | Foundry server host |
| `SLATE_FOUNDRY_PORT` | `5272` | Foundry server port |

---

## ChromaDB

Vector store for persistent codebase memory and RAG operations.

### Features

| Feature | Description |
|:--------|:------------|
| **Semantic Search** | Find similar code patterns |
| **Codebase Memory** | Store project context across sessions |
| **RAG Support** | Retrieval-augmented generation |
| **Local Storage** | Data stays on your machine |

### SLATE Integration

```python
from slate.rag_memory import get_memory_manager

memory = get_memory_manager()

# Store context
memory.store_short_term("user_preference", {"theme": "dark"})

# Semantic search
results = memory.search("authentication flow", top_k=5)
for result in results:
    print(f"  - {result['content'][:50]}...")

# Long-term storage
memory.store_long_term("project_patterns", {
    "naming": "snake_case",
    "testing": "pytest"
})
```

### Memory Types

| Type | Duration | Use Case |
|:-----|:---------|:---------|
| **Short-term** | Session | Current context, user preferences |
| **Long-term** | Persistent | Project knowledge, learned patterns |
| **Episodic** | Persistent | Task history, outcomes |

---

## Unified AI Backend

SLATE routes all AI tasks through `unified_ai_backend.py` which automatically selects the best backend.

### Task Routing

| Task Type | Best Backend | Cost |
|:----------|:-------------|:-----|
| `code_generation` | Ollama | FREE |
| `code_review` | Ollama | FREE |
| `test_generation` | Ollama | FREE |
| `bug_fix` | Ollama | FREE |
| `refactoring` | Ollama | FREE |
| `documentation` | Ollama | FREE |
| `analysis` | Ollama | FREE |
| `research` | Ollama | FREE |
| `planning` | Spec-Kit | FREE |

### Commands

```bash
# Check all backend status
python slate/unified_ai_backend.py --status

# Test specific backend
python slate/unified_ai_backend.py --test ollama
python slate/unified_ai_backend.py --test foundry

# Generate with auto-routing
python slate/unified_ai_backend.py --generate "Write hello world"

# Force specific backend
python slate/unified_ai_backend.py --generate "..." --backend ollama
```

### Python API

```python
from slate.unified_ai_backend import UnifiedBackend

backend = UnifiedBackend()

# Get status
status = backend.get_status()
print(f"Ollama: {status['ollama']}")
print(f"Foundry: {status['foundry']}")

# Auto-route generation
response = backend.generate(
    prompt="Write a REST API endpoint",
    task_type="code_generation"
)
print(response)
```

---

## Security: ActionGuard

All external API calls are **blocked by default** through ActionGuard.

### Blocked Patterns

| Pattern | Reason |
|:--------|:-------|
| `api.openai.com` | Paid cloud API |
| `api.anthropic.com` | Paid cloud API |
| `api.cohere.com` | Paid cloud API |
| `*.amazonaws.com` | Cloud infrastructure |
| `*.azure.com` | Cloud infrastructure |

### Allowing External APIs

If you need external API access:

```python
from slate.action_guard import ActionGuard

guard = ActionGuard()

# Check if blocked
is_blocked = guard.is_blocked("https://api.openai.com/v1/chat")
print(f"Blocked: {is_blocked}")  # True

# Whitelist (requires explicit configuration)
# Edit .slate_identity/action_guard_config.json
```

---

## GPU Optimization

SLATE automatically optimizes for your GPU architecture.

### Supported Architectures

| Architecture | GPUs | Optimizations |
|:-------------|:-----|:--------------|
| **Blackwell** | RTX 50xx | TF32, BF16, Flash Attention 2, CUDA Graphs |
| **Ada Lovelace** | RTX 40xx | TF32, BF16, Flash Attention, CUDA Graphs |
| **Ampere** | RTX 30xx, A100 | TF32, BF16, Flash Attention |
| **Turing** | RTX 20xx | FP16, Tensor Cores |
| **CPU-Only** | Any | AVX2/AVX-512 optimizations |

### Commands

```bash
# Detect hardware
python slate/slate_hardware_optimizer.py

# Apply optimizations
python slate/slate_hardware_optimizer.py --optimize

# Install optimal PyTorch
python slate/slate_hardware_optimizer.py --install-pytorch
```

---

## Troubleshooting

### Ollama Not Connecting

```bash
# Check if Ollama is running
curl http://127.0.0.1:11434/api/tags

# Start Ollama service
ollama serve
```

### No GPU Detected

```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of VRAM

```bash
# Check current usage
nvidia-smi

# Use smaller model
ollama pull phi  # 2.7B instead of 7B
```

---

## Next Steps

- [CLI Reference](CLI-Reference) - Command-line tools
- [Configuration](Configuration) - Settings and customization
- [Architecture](Architecture) - System design
