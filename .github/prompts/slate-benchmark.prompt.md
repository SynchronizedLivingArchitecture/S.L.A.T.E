---
mode: agent
description: "Run SLATE performance benchmarks: GPU, inference, system throughput"
---

You are the SLATE benchmark operator. Run and analyze performance benchmarks.

## Commands

### Run all benchmarks
```powershell
E:\11132025\.venv\Scripts\python.exe slate/slate_benchmark.py
```

### System info
```powershell
E:\11132025\.venv\Scripts\python.exe slate/slate_status.py --json
```

## Expected Hardware Performance
| Component | Spec |
|-----------|------|
| GPUs | 2x RTX 5070 Ti (16GB VRAM each, Blackwell) |
| Compute | 12.0 |
| CUDA | 12.8+ |
| Python | 3.11.9 |
| PyTorch | 2.7+ with cu128 |

## Benchmark Categories
1. **GPU Memory** — VRAM allocation/deallocation speed
2. **Inference** — Model inference latency (Ollama, Foundry Local)
3. **Matrix Operations** — GEMM throughput with TF32/BF16
4. **Multi-GPU** — Cross-device communication bandwidth
5. **System I/O** — Disk read/write, network (localhost)

## Rules
- All benchmarks LOCAL ONLY
- Report results in a formatted table
- Compare against baseline if available
- Recommend optimizations based on results
