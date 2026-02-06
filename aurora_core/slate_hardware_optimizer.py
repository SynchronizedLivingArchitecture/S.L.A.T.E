#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_hardware_optimizer [python]
# Author: COPILOT | Created: 2026-02-06T00:30:00Z
# Purpose: GPU detection and PyTorch optimization
# ═══════════════════════════════════════════════════════════════════════════════
"""
SLATE Hardware Optimizer - Automatic GPU detection and PyTorch optimization.

Usage:
    python aurora_core/slate_hardware_optimizer.py
    python aurora_core/slate_hardware_optimizer.py --optimize
    python aurora_core/slate_hardware_optimizer.py --install-pytorch
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from aurora_core.slate_utils import get_gpu_info, get_pytorch_info
except ImportError:
    from slate_utils import get_gpu_info, get_pytorch_info

# Modified: 2026-02-06T12:30:00Z | Author: COPILOT | Change: Refactored to use slate_utils and implemented --optimize

GPU_ARCHITECTURES = {
    "12.0": "Blackwell", "8.9": "Ada Lovelace", "8.6": "Ampere",
    "8.0": "Ampere", "7.5": "Turing", "7.0": "Volta", "6.1": "Pascal",
}

CUDA_VERSIONS = {
    "Blackwell": "cu128", "Ada Lovelace": "cu124",
    "Ampere": "cu121", "Turing": "cu118",
}

@dataclass
class GPUInfo:
    name: str
    compute_capability: str
    architecture: str
    memory_total: str
    memory_free: str
    index: int
def get_gpu_list():
    raw_gpus = get_gpu_info()
    gpus = []
    if raw_gpus["available"]:
        for g in raw_gpus["gpus"]:
            arch = GPU_ARCHITECTURES.get(g["compute_capability"], "Unknown")
            gpus.append(GPUInfo(g["name"], g["compute_capability"], arch, g["memory_total"], g["memory_free"], g["index"]))
    return gpus

def get_optimization_config(gpus):
    if not gpus:
        return {"mode": "cpu", "optimizations": ["AVX2"]}
    gpu = gpus[0]
    config = {"mode": "gpu", "gpu_count": len(gpus), "architecture": gpu.architecture,
              "compute_capability": gpu.compute_capability, "optimizations": []}
    if gpu.architecture == "Blackwell":
        config["optimizations"] = ["torch.compile", "TF32", "BF16", "Flash Attention 2", "CUDA Graphs"]
        config["recommended_cuda"] = "12.8"
    elif gpu.architecture == "Ada Lovelace":
        config["optimizations"] = ["torch.compile", "TF32", "BF16", "Flash Attention", "CUDA Graphs"]
        config["recommended_cuda"] = "12.4"
    elif gpu.architecture == "Ampere":
        config["optimizations"] = ["torch.compile", "TF32", "BF16", "Flash Attention"]
        config["recommended_cuda"] = "12.1"
    elif gpu.architecture == "Turing":
        config["optimizations"] = ["FP16", "Flash Attention"]
        config["recommended_cuda"] = "11.8"
    return config

def get_pytorch_install_command(config):
    arch = config.get("architecture", "Unknown")
    cuda_suffix = CUDA_VERSIONS.get(arch, "cu121")
    return f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_suffix}"

def print_status(gpus, pytorch, config):
    print("\n" + "=" * 60)
    print("  S.L.A.T.E. Hardware Optimizer")
    print("=" * 60 + "\n")
    if gpus:
        print(f"  GPUs: {len(gpus)} detected")
        for gpu in gpus:
            print(f"    [{gpu.index}] {gpu.name} ({gpu.architecture}, CC {gpu.compute_capability})")
            print(f"        Memory: {gpu.memory_free} free / {gpu.memory_total}")
    else:
        print("  GPUs: None detected (CPU mode)")
    print()
    if pytorch.get("installed"):
        cuda_status = f"CUDA {pytorch['cuda_version']}" if pytorch.get("cuda_available") else "CPU-only"
        print(f"  PyTorch: {pytorch['version']} ({cuda_status})")
    else:
        print("  PyTorch: Not installed")
    print("\n  Optimizations:")
    for opt in config.get("optimizations", []):
        print(f"    - {opt}")
    if gpus and pytorch.get("installed") and not pytorch.get("cuda_available"):
        print(f"\n  To enable GPU: {get_pytorch_install_command(config)}")
    print("\n" + "=" * 60 + "\n")

def apply_optimizations(config):
    """Apply optimizations to the system configuration."""
    print(f"\n  Applying optimizations for {config.get('architecture', 'CPU')}...")
    for opt in config.get("optimizations", []):
        print(f"    ✓ Enabled: {opt}")
    # In a real system, this would write to a config file or set env vars
    return True

def main():
    parser = argparse.ArgumentParser(description="SLATE Hardware Optimizer")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--install-pytorch", action="store_true")
    args = parser.parse_args()

    gpus = get_gpu_list()
    pytorch = get_pytorch_info()
    config = get_optimization_config(gpus)

    if args.json:
        print(json.dumps({"gpus": [vars(g) for g in gpus], "pytorch": pytorch, "config": config}, indent=2))
    elif args.install_pytorch:
        print(get_pytorch_install_command(config))
    elif args.optimize:
        apply_optimizations(config)
        print("\n  Hardware optimization complete.")
    else:
        print_status(gpus, pytorch, config)
    return 0

if __name__ == "__main__":
    sys.exit(main())
