#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# SLATE Status Checker
# Author: Claude | Modified: 2026-02-06
# Purpose: Quick status check for SLATE system (GitHub Runner focused)
# ═══════════════════════════════════════════════════════════════════════════════
"""
SLATE Status Checker
====================
Quick system status check focused on GitHub Runner integration.

Usage:
    python slate/slate_status.py --quick
    python slate/slate_status.py --json
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Modified: 2026-02-07T04:00:00Z | Author: COPILOT | Change: Add WORKSPACE_ROOT for runner dir
WORKSPACE_ROOT = Path(__file__).parent.parent

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_python_info():
    """Get Python version info."""
    version = sys.version_info
    return {
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "executable": sys.executable,
        "ok": version.major >= 3 and version.minor >= 11
    }


def get_gpu_info():
    """Detect NVIDIA GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "name": parts[0],
                        "memory_total": parts[1],
                        "memory_free": parts[2]
                    })
            return {"available": True, "count": len(gpus), "gpus": gpus}
        return {"available": False, "count": 0, "gpus": []}
    except Exception:
        return {"available": False, "count": 0, "gpus": []}


def get_system_info():
    """Get system resource info."""
    if not HAS_PSUTIL:
        return {"available": False}

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.cwd()))

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


def get_runner_info():
    """Check GitHub Actions self-hosted runner status."""
    runner_dir = WORKSPACE_ROOT / "actions-runner"
    runner_config = runner_dir / ".runner"
    slate_config = runner_dir / ".slate_runner_config.json"

    info = {
        "installed": runner_dir.exists() and (runner_dir / "config.cmd").exists(),
        "configured": runner_config.exists(),
        "running": False,
        "name": None,
        "repo": None,
        "labels": []
    }

    # Check if runner process is active
    if HAS_PSUTIL:
        for proc in psutil.process_iter(['name']):
            try:
                if 'Runner.Listener' in proc.info.get('name', ''):
                    info["running"] = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    # Load configuration
    if runner_config.exists():
        try:
            config = json.loads(runner_config.read_text())
            info["name"] = config.get("agentName")
            info["repo"] = config.get("gitHubUrl")
        except Exception:
            pass

    if slate_config.exists():
        try:
            config = json.loads(slate_config.read_text())
            info["labels"] = config.get("labels", [])
        except Exception:
            pass

    return info


def get_github_info():
    """Check GitHub CLI and authentication status."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10
        )
        authenticated = result.returncode == 0

        username = None
        if authenticated:
            user_result = subprocess.run(
                ["gh", "api", "user", "-q", ".login"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if user_result.returncode == 0:
                username = user_result.stdout.strip()

        return {
            "cli_installed": True,
            "authenticated": authenticated,
            "username": username
        }
    except FileNotFoundError:
        return {"cli_installed": False, "authenticated": False}
    except Exception:
        return {"cli_installed": True, "authenticated": False}


def get_status():
    """Get full system status."""
    return {
        "timestamp": datetime.now().isoformat(),
        "python": get_python_info(),
        "gpu": get_gpu_info(),
        "system": get_system_info(),
        "runner": get_runner_info(),
        "github": get_github_info()
    }


def print_quick_status(status: dict):
    """Print quick status summary."""
    print()
    print("=" * 50)
    print("  S.L.A.T.E. Status (GitHub Runner)")
    print("=" * 50)
    print()

    # Python
    py = status["python"]
    icon = "[OK]" if py["ok"] else "[!!]"
    print(f"  Python:   {icon} {py['version']}")

    # GPU
    gpu = status["gpu"]
    if gpu["available"]:
        print(f"  GPU:      [OK] {gpu['count']} NVIDIA GPU(s)")
        for g in gpu["gpus"]:
            print(f"            - {g['name']} ({g['memory_total']})")
    else:
        print("  GPU:      [--] None detected")

    # System
    sys_info = status["system"]
    if sys_info.get("available"):
        print(f"  CPU:      {sys_info['cpu_count']} cores ({sys_info['cpu_percent']}% used)")
        print(f"  Memory:   {sys_info['memory_available_gb']}/{sys_info['memory_total_gb']} GB free")
        print(f"  Disk:     {sys_info['disk_free_gb']}/{sys_info['disk_total_gb']} GB free")

    # GitHub Runner (primary)
    runner = status.get("runner", {})
    runner_name = runner.get('name') or 'self-hosted'
    if runner.get("running"):
        print(f"  Runner:   [OK] {runner_name} (listening)")
        if runner.get("labels"):
            print(f"            Labels: {', '.join(runner['labels'][:5])}")
    elif runner.get("configured"):
        print(f"  Runner:   [--] {runner_name} (stopped)")
    elif runner.get("installed"):
        print("  Runner:   [--] Installed but not configured")
    else:
        print("  Runner:   [!!] Not installed")

    # GitHub CLI
    github = status.get("github", {})
    if github.get("authenticated"):
        print(f"  GitHub:   [OK] {github.get('username', 'authenticated')}")
    elif github.get("cli_installed"):
        print("  GitHub:   [--] CLI installed, not authenticated")
    else:
        print("  GitHub:   [--] CLI not installed")

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
