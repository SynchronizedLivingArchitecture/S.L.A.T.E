#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: install_slate [python]
# Author: COPILOT | Created: 2026-02-06T00:30:00Z | Modified: 2026-02-06T00:30:00Z
# Purpose: SLATE public installation script
# ═══════════════════════════════════════════════════════════════════════════════
"""
S.L.A.T.E. Installation Script
===============================
Installs and configures SLATE for your system.

Usage:
    python install_slate.py
    python install_slate.py --skip-gpu
    python install_slate.py --dev
"""

import os
import subprocess
import sys
from pathlib import Path

# Try to import from aurora_core if available
try:
    sys.path.append(str(Path(__file__).parent / "aurora_core"))
    from slate_utils import get_gpu_info
except ImportError:
    get_gpu_info = None

# Modified: 2026-02-06T12:50:00Z | Author: COPILOT | Change: Refactored to use slate_utils for hardware detection

WORKSPACE_ROOT = Path(__file__).parent


def print_banner():
    """Print installation banner."""
    print()
    print("═" * 70)
    print("  S.L.A.T.E. Installation")
    print("  System Learning Agent for Task Execution")
    print("═" * 70)
    print()


def check_python_version():
    """Check Python version."""
    print("[1/6] Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"  ✗ Python 3.11+ required, found {version.major}.{version.minor}")
        return False
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def create_venv():
    """Create virtual environment if not exists."""
    print("[2/6] Setting up virtual environment...")
    venv_path = WORKSPACE_ROOT / ".venv"
    
    if venv_path.exists():
        print("  ✓ Virtual environment exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print("  ✓ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to create venv: {e}")
        return False


def get_pip():
    """Get pip executable path."""
    if os.name == "nt":
        return WORKSPACE_ROOT / ".venv" / "Scripts" / "pip.exe"
    return WORKSPACE_ROOT / ".venv" / "bin" / "pip"


def install_requirements():
    """Install Python requirements."""
    print("[3/6] Installing dependencies...")
    pip = get_pip()
    
    if not pip.exists():
        print("  ✗ pip not found in venv")
        return False
    
    req_file = WORKSPACE_ROOT / "requirements.txt"
    if not req_file.exists():
        print("  ✗ requirements.txt not found")
        return False
    
    try:
        subprocess.run(
            [str(pip), "install", "-r", str(req_file), "--quiet"],
            check=True,
            timeout=300
        )
        print("  ✓ Dependencies installed")
        return True
    except subprocess.TimeoutExpired:
        print("  ✗ Installation timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Installation failed: {e}")
        return False


def detect_hardware():
    """Detect system hardware."""
    print("[4/6] Detecting hardware...")
    
    if get_gpu_info is None:
        print("  ○ GPU detection utilities not available (CPU mode)")
        return True

    gpu_status = get_gpu_info()
    if gpu_status["available"]:
        print(f"  ✓ Found {gpu_status['count']} NVIDIA GPU(s):")
        for gpu in gpu_status["gpus"]:
            cc = gpu["compute_capability"]
            if cc.startswith("12."):
                arch = "Blackwell"
            elif cc == "8.9":
                arch = "Ada Lovelace"
            elif cc.startswith("8."):
                arch = "Ampere"
            elif cc == "7.5":
                arch = "Turing"
            else:
                arch = "Unknown"
            print(f"      {gpu['name']} ({arch}, CC {cc}, {gpu['memory_total']})")
        return True
    else:
        reason = gpu_status.get("reason", gpu_status.get("error", "Unknown reason"))
        print(f"  ○ GPU detection: {reason} (CPU mode)")
        return True


def create_directories():
    """Create required directories."""
    print("[5/6] Creating directories...")
    
    dirs = [
        WORKSPACE_ROOT / "aurora_core",
        WORKSPACE_ROOT / "agents",
        WORKSPACE_ROOT / "tests",
        WORKSPACE_ROOT / ".github",
    ]
    
    for d in dirs:
        d.mkdir(exist_ok=True)
    
    (WORKSPACE_ROOT / "aurora_core" / "__init__.py").touch()
    (WORKSPACE_ROOT / "agents" / "__init__.py").touch()
    
    print("  ✓ Directories created")
    return True


def run_benchmark():
    """Run system benchmark."""
    print("[6/6] Running benchmark...")
    
    benchmark_script = WORKSPACE_ROOT / "aurora_core" / "slate_benchmark.py"
    if not benchmark_script.exists():
        print("  ○ Benchmark script not found (skipping)")
        return True
    
    python_exe = WORKSPACE_ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
    if os.name == "nt":
        python_exe = python_exe.with_suffix(".exe")
    
    try:
        result = subprocess.run(
            [str(python_exe), str(benchmark_script), "--json"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("  ✓ Benchmark complete")
        else:
            print("  ○ Benchmark skipped")
        return True
    except Exception:
        print("  ○ Benchmark skipped")
        return True


def print_completion():
    """Print completion message."""
    print()
    print("═" * 70)
    print("  S.L.A.T.E. Installation Complete!")
    print("═" * 70)
    print()
    print("  Next steps:")
    print()
    if os.name == "nt":
        print("    1. Activate: .\\.venv\\Scripts\\activate")
    else:
        print("    1. Activate: source .venv/bin/activate")
    print("    2. Check status: python aurora_core/slate_status.py --quick")
    print("    3. Run benchmark: python aurora_core/slate_benchmark.py")
    print("    4. Detect hardware: python aurora_core/slate_hardware_optimizer.py")
    print()
    print("  For GPU support (optional):")
    print("    python aurora_core/slate_hardware_optimizer.py --install-pytorch")
    print()


def main():
    """Main installation entry point."""
    print_banner()
    
    steps = [
        check_python_version,
        create_venv,
        install_requirements,
        detect_hardware,
        create_directories,
        run_benchmark,
    ]
    
    for step in steps:
        if not step():
            print("\n✗ Installation failed")
            return 1
    
    print_completion()
    return 0


if __name__ == "__main__":
    sys.exit(main())
