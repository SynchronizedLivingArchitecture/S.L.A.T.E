#!/usr/bin/env python3
# Modified: 2026-02-06T12:00:00Z | Author: COPILOT | Change: Created centralized utilities for GPU and environment checks
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_utils [python]
# Purpose: Shared utility functions for SLATE system
# ═══════════════════════════════════════════════════════════════════════════════

import subprocess
import sys
import os

def get_gpu_info():
    """Detect NVIDIA GPUs and return a standardized info dict."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "compute_capability": parts[2],
                        "memory_total": parts[3],
                        "memory_free": parts[4]
                    })
            return {"available": True, "count": len(gpus), "gpus": gpus}
        return {"available": False, "count": 0, "gpus": [], "reason": "No GPUs reported by nvidia-smi"}
    except FileNotFoundError:
        return {"available": False, "count": 0, "gpus": [], "reason": "nvidia-smi not found"}
    except Exception as e:
        return {"available": False, "count": 0, "gpus": [], "error": str(e)}

def get_pytorch_info():
    """Check PyTorch installation and CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda if cuda_available else None,
            "device_count": torch.cuda.device_count() if cuda_available else 0
        }
    except ImportError:
        return {"installed": False}
    except Exception as e:
        return {"installed": True, "error": str(e)}

def check_ollama():
    """Check if Ollama is available and list models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            models = [l.split()[0] for l in lines[1:] if l.strip()]
            return {"available": True, "model_count": len(models), "models": models[:10]}
        return {"available": False, "model_count": 0, "reason": f"Ollama returned exit code {result.returncode}"}
    except FileNotFoundError:
        return {"available": False, "model_count": 0, "reason": "ollama command not found"}
    except Exception as e:
        return {"available": False, "model_count": 0, "error": str(e)}

def is_venv():
    """Check if currently running in a virtual environment."""
    return hasattr(sys, 'real_prefix') or (sys.base_prefix != sys.prefix)
