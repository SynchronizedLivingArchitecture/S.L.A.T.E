#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_status [python]
# Author: COPILOT | Created: 2026-02-06T00:30:00Z | Modified: 2026-02-06T00:30:00Z
# Purpose: Quick status check for SLATE system
# ═══════════════════════════════════════════════════════════════════════════════
"""
SLATE Status Checker
====================
Quick system status check.

Usage:
    python aurora_core/slate_status.py --quick
    python aurora_core/slate_status.py --json
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from aurora_core.slate_utils import get_gpu_info, get_pytorch_info, check_ollama
except ImportError:
    from slate_utils import get_gpu_info, get_pytorch_info, check_ollama

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Modified: 2026-02-06T12:10:00Z | Author: COPILOT | Change: Refactored to use slate_utils and added header

def get_python_info():
    """Get Python version info."""
    version = sys.version_info
    return {
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "executable": sys.executable,
        "ok": version.major >= 3 and version.minor >= 11
    }


def get_system_info():
    """Get system resource info."""
    if not HAS_PSUTIL:
        return {"available": False}
    
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(Path.cwd())
    
    return {
        "available": True,
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_total_gb": round(mem.total / (1024**3), 1),
        "memory_available_gb": round(mem.available / (1024**3), 1),
        "memory_percent": mem.percent,
        "disk_total_gb": round(disk.total / (1024**3), 1),
        "disk_free_gb": round(disk.free / (1024**3), 1)
    }


def get_status():
    """Get full system status."""
    return {
        "timestamp": datetime.now().isoformat(),
        "python": get_python_info(),
        "gpu": get_gpu_info(),
        "system": get_system_info(),
        "pytorch": get_pytorch_info(),
        "ollama": check_ollama()
    }


def print_quick_status(status: dict):
    """Print quick status summary."""
    print()
    print("=" * 50)
    print("  S.L.A.T.E. Status")
    print("=" * 50)
    print()
    
    # Python
    py = status["python"]
    icon = "✓" if py["ok"] else "✗"
    print(f"  Python:   {icon} {py['version']}")
    
    # GPU
    gpu = status["gpu"]
    if gpu["available"]:
        print(f"  GPU:      ✓ {gpu['count']} NVIDIA GPU(s)")
        for g in gpu["gpus"]:
            print(f"            - {g['name']} ({g['memory_total']})")
    else:
        print("  GPU:      ○ CPU-only mode")
    
    # System
    sys_info = status["system"]
    if sys_info.get("available"):
        print(f"  CPU:      {sys_info['cpu_count']} cores ({sys_info['cpu_percent']}% used)")
        print(f"  Memory:   {sys_info['memory_available_gb']}/{sys_info['memory_total_gb']} GB free")
        print(f"  Disk:     {sys_info['disk_free_gb']}/{sys_info['disk_total_gb']} GB free")
    
    # PyTorch
    pt = status["pytorch"]
    if pt.get("installed"):
        cuda_status = f"CUDA {pt['cuda_version']}" if pt.get("cuda_available") else "CPU"
        print(f"  PyTorch:  ✓ {pt['version']} ({cuda_status})")
    else:
        print("  PyTorch:  ○ Not installed")
    
    # Ollama
    ollama = status["ollama"]
    if ollama.get("available"):
        print(f"  Ollama:   ✓ {ollama['model_count']} models")
    else:
        print("  Ollama:   ○ Not available")
    
    print()
    print("=" * 50)
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SLATE Status Checker")
    parser.add_argument("--quick", action="store_true", help="Quick status")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()
    
    status = get_status()
    
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print_quick_status(status)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
