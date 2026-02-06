#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_runtime [python]
# Author: COPILOT | Created: 2026-02-06T00:30:00Z
# Purpose: Runtime integration checker
# ═══════════════════════════════════════════════════════════════════════════════
"""
SLATE Runtime Checker - Check all integrations and dependencies.

Usage:
    python aurora_core/slate_runtime.py --check-all
    python aurora_core/slate_runtime.py --json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from aurora_core.slate_utils import get_gpu_info, get_pytorch_info, check_ollama, is_venv
except ImportError:
    from slate_utils import get_gpu_info, get_pytorch_info, check_ollama, is_venv

# Modified: 2026-02-06T12:20:00Z | Author: COPILOT | Change: Refactored to use slate_utils

def check_integration(name, check_fn, details_fn=None):
    try:
        status = check_fn()
        details = details_fn() if details_fn and status else None
        return {"name": name, "status": "active" if status else "inactive", "details": details}
    except Exception as e:
        return {"name": name, "status": "error", "error": str(e)}

def check_python(): return sys.version_info >= (3, 11)
def python_details(): return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def pytorch_details():
    info = get_pytorch_info()
    if not info["installed"]: return "Not installed"
    cuda = f", CUDA {info['cuda_version']}" if info['cuda_available'] else ", CPU"
    return f"{info['version']}{cuda}"

def check_transformers():
    try: import transformers; return True
    except: return False

INTEGRATIONS = [
    ("Python 3.11+", check_python, python_details),
    ("Virtual Env", is_venv, None),
    ("NVIDIA GPU", lambda: get_gpu_info()["available"], None),
    ("PyTorch", lambda: get_pytorch_info()["installed"], pytorch_details),
    ("Transformers", check_transformers, None),
    ("Ollama", lambda: check_ollama()["available"], None),
]

def check_all():
    results = {"timestamp": datetime.now().isoformat(), "integrations": []}
    for name, check_fn, details_fn in INTEGRATIONS:
        results["integrations"].append(check_integration(name, check_fn, details_fn))
    active = sum(1 for i in results["integrations"] if i["status"] == "active")
    results["summary"] = {"active": active, "total": len(results["integrations"])}
    return results

def print_results(results):
    print("\n" + "=" * 60)
    print("  S.L.A.T.E. Runtime Check")
    print("=" * 60 + "\n")
    for item in results["integrations"]:
        icon = "\u2713" if item["status"] == "active" else "\u25cb" if item["status"] == "inactive" else "\u2717"
        details = f" ({item['details']})" if item.get("details") else ""
        print(f"  {icon} {item['name']}{details}")
    s = results["summary"]
    print(f"\n  Summary: {s['active']}/{s['total']} integrations active")
    print("\n" + "=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="SLATE Runtime Checker")
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = check_all()
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)
    return 0

if __name__ == "__main__":
    sys.exit(main())
