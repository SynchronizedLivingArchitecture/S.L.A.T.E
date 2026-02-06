#!/usr/bin/env python3
# Modified: 2026-02-06T11:15:00Z | Author: COPILOT | Change: Added terminal monitor and AI toolkit to protocols
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_protocols [python]
# Purpose: Central runner for SLATE system protocols
# ═══════════════════════════════════════════════════════════════════════════════
"""
SLATE Protocols Runner
======================
Executes all core system checks and benchmarks.

Usage:
    python aurora_core/slate_protocols.py
    slate-protocols
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_command(cmd):
    """Run a system command and print its output."""
    print(f"\n>> Running: {' '.join(cmd)}")
    print("-" * 60)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error executing {' '.join(cmd)}: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
    print("-" * 60)

def main():
    """Main entry point to run all protocols."""
    print("\n" + "=" * 60)
    print("  S.L.A.T.E. Protocols Execution")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 60)

    base_dir = Path(__file__).parent.parent

    protocols = [
        [sys.executable, str(base_dir / "aurora_core" / "slate_status.py"), "--quick"],
        [sys.executable, str(base_dir / "aurora_core" / "slate_runtime.py"), "--check-all"],
        [sys.executable, str(base_dir / "aurora_core" / "slate_hardware_optimizer.py")],
        [sys.executable, str(base_dir / "aurora_core" / "slate_benchmark.py")],
        [sys.executable, str(base_dir / "aurora_core" / "slate_terminal_monitor.py"), "--status"],
        [sys.executable, str(base_dir / "aurora_core" / "slate_ai_toolkit_integration.py"), "--status"],
    ]

    for protocol in protocols:
        run_command(protocol)

    print("\n" + "=" * 60)
    print("  Protocols Execution Complete")
    print("=" * 60 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
