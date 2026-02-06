#!/usr/bin/env python3
"""
SLATE Terminal Monitor - Safe command execution with timeouts.

Usage:
    python aurora_core/slate_terminal_monitor.py --status
    python aurora_core/slate_terminal_monitor.py --advice "pip install torch"
    python aurora_core/slate_terminal_monitor.py --run "python --version"
"""

import argparse
import json
import subprocess
import sys
import shlex
from datetime import datetime

# Modified: 2026-02-06T12:40:00Z | Author: COPILOT | Change: Fixed shell injection and expanded blocked commands

BLOCKED_COMMANDS = ["curl.exe", "Start-Sleep", "wget", "curl", "nc", "netcat", "rm", "sh", "bash"]
LONG_RUNNING = {
    "pip install": {"timeout": 300, "background": True},
    "npm install": {"timeout": 180, "background": True},
    "docker build": {"timeout": 600, "background": True},
    "ollama pull": {"timeout": 600, "background": True},
    "git clone": {"timeout": 120, "background": False},
}
DEFAULT_TIMEOUT = 60

def is_blocked(command):
    cmd_lower = command.lower()
    for blocked in BLOCKED_COMMANDS:
        if blocked.lower() in cmd_lower:
            return True, blocked
    return False, ""

def get_command_config(command):
    cmd_lower = command.lower()
    for pattern, config in LONG_RUNNING.items():
        if pattern.lower() in cmd_lower:
            return config
    return {"timeout": DEFAULT_TIMEOUT, "background": False}

def get_advice(command):
    blocked, reason = is_blocked(command)
    if blocked:
        return {"safe": False, "reason": f"Blocked: {reason}"}
    config = get_command_config(command)
    return {"safe": True, "timeout": config["timeout"], "background": config["background"]}

def run_command(command, timeout=None, background=False):
    blocked, reason = is_blocked(command)
    if blocked:
        return {"success": False, "error": f"Blocked: {reason}"}
    config = get_command_config(command)
    timeout = timeout or config["timeout"]

    try:
        # Use shlex to safely split the command and avoid shell=True (shell injection)
        cmd_args = shlex.split(command)

        if background:
            subprocess.Popen(cmd_args, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"success": True, "background": True}

        result = subprocess.run(cmd_args, shell=False, capture_output=True, text=True, timeout=timeout)
        return {"success": result.returncode == 0, "output": result.stdout[:5000]}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_status():
    return {"timestamp": datetime.now().isoformat(), "blocked": BLOCKED_COMMANDS,
            "long_running": list(LONG_RUNNING.keys()), "default_timeout": DEFAULT_TIMEOUT}

def main():
    parser = argparse.ArgumentParser(description="SLATE Terminal Monitor")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--advice", type=str)
    parser.add_argument("--run", type=str)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.advice:
        advice = get_advice(args.advice)
        print(json.dumps(advice, indent=2) if args.json else
              ("Safe" if advice["safe"] else f"Blocked: {advice['reason']}"))
    elif args.run:
        result = run_command(args.run, args.timeout, args.background)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(result.get("output", result.get("error", "Done")))
        return 0 if result["success"] else 1
    else:
        status = get_status()
        print(json.dumps(status, indent=2) if args.json else
              f"Blocked: {status['blocked']}\nLong-running: {status['long_running']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
