#!/usr/bin/env python3
# Modified: 2026-02-06T11:00:00Z | Author: COPILOT | Change: Created AI Toolkit integration script
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_ai_toolkit_integration [python]
# Purpose: Integration with Microsoft AI Toolkit for VS Code
# ═══════════════════════════════════════════════════════════════════════════════
"""
SLATE AI Toolkit Integration
============================
Handles dependencies and status for the AI Toolkit.

Usage:
    python aurora_core/slate_ai_toolkit_integration.py --status
    python aurora_core/slate_ai_toolkit_integration.py --install-deps
"""

import argparse
import subprocess
import sys

DEPENDENCIES = ["transformers", "accelerate", "peft"]

def check_status():
    """Check if AI Toolkit dependencies are installed."""
    print("\n  AI Toolkit Dependency Status:")
    print("  " + "-" * 30)
    all_installed = True
    for dep in DEPENDENCIES:
        try:
            __import__(dep)
            print(f"    ✓ {dep:15} : Installed")
        except ImportError:
            print(f"    ✗ {dep:15} : Not Found")
            all_installed = False
    print("  " + "-" * 30)
    return all_installed

def install_deps():
    """Install required dependencies."""
    print("\n  Installing AI Toolkit dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install"] + DEPENDENCIES, check=True)
        print("\n  ✓ Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ Installation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="SLATE AI Toolkit Integration")
    parser.add_argument("--status", action="store_true", help="Check dependency status")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    args = parser.parse_args()

    if args.install_deps:
        install_deps()
    elif args.status:
        if check_status():
            print("\n  ✓ AI Toolkit integration is ready.")
        else:
            print("\n  ○ Some dependencies are missing. Run with --install-deps to fix.")
    else:
        parser.print_help()

if __name__ == "__main__":
    sys.exit(main())
