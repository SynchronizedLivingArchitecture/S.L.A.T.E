#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_service_watchdog [python]
# Author: Claude | Created: 2026-02-07T06:00:00Z
# Purpose: Service watchdog with automatic restart for SLATE services
# ═══════════════════════════════════════════════════════════════════════════════
"""
SLATE Service Watchdog
======================
Monitors SLATE services and automatically restarts them when they crash.

Features:
- Health check polling for dashboard server
- Process monitoring for runner
- Automatic restart with exponential backoff
- Integration with VSCode plugin via status file
- GitHub runner compatibility

Usage:
    python slate/slate_service_watchdog.py start     # Start watchdog
    python slate/slate_service_watchdog.py status    # Check status
    python slate/slate_service_watchdog.py restart   # Restart all services
"""

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# Configuration
DASHBOARD_PORT = 8080
DASHBOARD_URL = f"http://127.0.0.1:{DASHBOARD_PORT}/health"
HEALTH_CHECK_INTERVAL = 10  # seconds
MAX_RESTART_ATTEMPTS = 5
RESTART_BACKOFF_BASE = 2  # seconds, doubles each attempt
PID_FILE = WORKSPACE_ROOT / ".slate_watchdog.pid"
STATE_FILE = WORKSPACE_ROOT / ".slate_watchdog_state.json"


class ServiceWatchdog:
    """Monitors and auto-restarts SLATE services."""

    def __init__(self):
        self.workspace = WORKSPACE_ROOT
        self.running = False
        self._shutdown_event = threading.Event()
        self.restart_counts: Dict[str, int] = {}
        self.last_restart: Dict[str, float] = {}
        self.processes: Dict[str, subprocess.Popen] = {}

    def _get_python(self) -> str:
        """Get venv Python path."""
        if os.name == "nt":
            return str(self.workspace / ".venv" / "Scripts" / "python.exe")
        return str(self.workspace / ".venv" / "bin" / "python")

    def _save_state(self, state: Dict[str, Any]):
        """Save watchdog state to file for VSCode plugin to read."""
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_state(self) -> Dict[str, Any]:
        """Load watchdog state from file."""
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                pass
        return {"services": {}, "started_at": None}

    def _write_pid(self):
        """Write current PID to file."""
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

    def _clear_pid(self):
        """Remove PID file."""
        if PID_FILE.exists():
            PID_FILE.unlink()

    def _check_existing(self) -> Optional[int]:
        """Check if watchdog is already running."""
        if not PID_FILE.exists():
            return None
        try:
            pid = int(PID_FILE.read_text().strip())
            if os.name == "nt":
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    capture_output=True, text=True
                )
                if str(pid) in result.stdout:
                    return pid
            else:
                os.kill(pid, 0)
                return pid
        except (ValueError, OSError, ProcessLookupError):
            self._clear_pid()
        return None

    def check_dashboard(self) -> bool:
        """Check if dashboard server is responding."""
        try:
            req = urllib.request.urlopen(DASHBOARD_URL, timeout=5)
            return req.status == 200
        except Exception:
            return False

    def check_runner(self) -> bool:
        """Check if GitHub runner process is running."""
        try:
            if os.name == "nt":
                result = subprocess.run(
                    ["powershell", "-Command",
                     "Get-Process -Name 'Runner.Listener' -ErrorAction SilentlyContinue"],
                    capture_output=True, text=True, timeout=5
                )
                return "Runner.Listener" in result.stdout or result.returncode == 0
            else:
                result = subprocess.run(
                    ["pgrep", "-f", "Runner.Listener"],
                    capture_output=True, timeout=5
                )
                return result.returncode == 0
        except Exception:
            return False

    def start_dashboard(self) -> bool:
        """Start the dashboard server."""
        dashboard_script = self.workspace / "agents" / "slate_dashboard_server.py"
        if not dashboard_script.exists():
            return False

        try:
            python = self._get_python()
            process = subprocess.Popen(
                [python, str(dashboard_script)],
                cwd=str(self.workspace),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            self.processes["dashboard"] = process
            print(f"  [WATCHDOG] Dashboard started (PID {process.pid})")
            return True
        except Exception as e:
            print(f"  [WATCHDOG] Dashboard start failed: {e}")
            return False

    def start_runner(self) -> bool:
        """Start the GitHub runner."""
        runner_dir = self.workspace / "actions-runner"
        run_cmd = runner_dir / "run.cmd"
        if not run_cmd.exists():
            return False

        try:
            if os.name == "nt":
                process = subprocess.Popen(
                    ["cmd", "/c", str(run_cmd)],
                    cwd=str(runner_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
                )
            else:
                process = subprocess.Popen(
                    [str(run_cmd)],
                    cwd=str(runner_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
            self.processes["runner"] = process
            print(f"  [WATCHDOG] Runner started (PID {process.pid})")
            return True
        except Exception as e:
            print(f"  [WATCHDOG] Runner start failed: {e}")
            return False

    def restart_service(self, service_name: str) -> bool:
        """Restart a service with exponential backoff."""
        now = time.time()
        count = self.restart_counts.get(service_name, 0)
        last = self.last_restart.get(service_name, 0)

        # Reset counter if it's been more than 5 minutes since last restart
        if now - last > 300:
            count = 0

        if count >= MAX_RESTART_ATTEMPTS:
            print(f"  [WATCHDOG] {service_name}: Max restart attempts reached, waiting...")
            # Reset after 10 minutes of waiting
            if now - last > 600:
                self.restart_counts[service_name] = 0
            return False

        # Exponential backoff
        backoff = RESTART_BACKOFF_BASE * (2 ** count)
        if now - last < backoff:
            return False

        print(f"  [WATCHDOG] Restarting {service_name} (attempt {count + 1}/{MAX_RESTART_ATTEMPTS})")

        success = False
        if service_name == "dashboard":
            success = self.start_dashboard()
        elif service_name == "runner":
            success = self.start_runner()

        self.restart_counts[service_name] = count + 1
        self.last_restart[service_name] = now

        return success

    def health_check_loop(self):
        """Main health check loop."""
        while self.running and not self._shutdown_event.is_set():
            state = {"services": {}, "watchdog_running": True}

            # Check dashboard
            dashboard_ok = self.check_dashboard()
            state["services"]["dashboard"] = {
                "running": dashboard_ok,
                "port": DASHBOARD_PORT,
                "restart_count": self.restart_counts.get("dashboard", 0)
            }
            if not dashboard_ok:
                self.restart_service("dashboard")
                # Give it time to start
                time.sleep(3)
                state["services"]["dashboard"]["running"] = self.check_dashboard()

            # Check runner
            runner_ok = self.check_runner()
            state["services"]["runner"] = {
                "running": runner_ok,
                "restart_count": self.restart_counts.get("runner", 0)
            }
            if not runner_ok:
                self.restart_service("runner")
                time.sleep(3)
                state["services"]["runner"]["running"] = self.check_runner()

            # Save state for VSCode plugin
            self._save_state(state)

            # Wait for next check
            self._shutdown_event.wait(HEALTH_CHECK_INTERVAL)

    def start(self) -> bool:
        """Start the watchdog."""
        existing_pid = self._check_existing()
        if existing_pid:
            print(f"[!] Watchdog already running (PID {existing_pid})")
            return False

        print()
        print("=" * 60)
        print("  SLATE Service Watchdog Starting")
        print("=" * 60)
        print()

        self._write_pid()
        atexit.register(self._clear_pid)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True

        # Initial service check and start
        if not self.check_dashboard():
            print("  [WATCHDOG] Dashboard not running, starting...")
            self.start_dashboard()
            time.sleep(2)

        if not self.check_runner():
            print("  [WATCHDOG] Runner not running, starting...")
            self.start_runner()
            time.sleep(2)

        print()
        print(f"  Dashboard: {'Running' if self.check_dashboard() else 'Starting...'}")
        print(f"  Runner:    {'Running' if self.check_runner() else 'Starting...'}")
        print()
        print(f"  Health check interval: {HEALTH_CHECK_INTERVAL}s")
        print("  Press Ctrl+C to stop")
        print()

        # Start health check loop
        self.health_check_loop()

        self.stop()
        return True

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n  [WATCHDOG] Shutdown signal received...")
        self.running = False
        self._shutdown_event.set()

    def stop(self):
        """Stop the watchdog."""
        print()
        print("  [WATCHDOG] Stopping...")
        self._shutdown_event.set()
        self._save_state({"services": {}, "watchdog_running": False})
        self._clear_pid()
        print("  [WATCHDOG] Stopped")

    def status(self) -> Dict[str, Any]:
        """Get status of all services."""
        return {
            "watchdog": {
                "running": self._check_existing() is not None,
                "pid": self._check_existing()
            },
            "dashboard": {
                "running": self.check_dashboard(),
                "port": DASHBOARD_PORT,
                "url": f"http://127.0.0.1:{DASHBOARD_PORT}"
            },
            "runner": {
                "running": self.check_runner()
            }
        }

    def print_status(self):
        """Print formatted status."""
        status = self.status()
        print()
        print("=" * 60)
        print("  SLATE Service Watchdog Status")
        print("=" * 60)
        print()

        watchdog = status["watchdog"]
        if watchdog["running"]:
            print(f"  Watchdog:   Running (PID {watchdog['pid']})")
        else:
            print("  Watchdog:   Stopped")

        dashboard = status["dashboard"]
        if dashboard["running"]:
            print(f"  Dashboard:  Running > {dashboard['url']}")
        else:
            print("  Dashboard:  Stopped")

        runner = status["runner"]
        if runner["running"]:
            print("  Runner:     Running")
        else:
            print("  Runner:     Stopped")

        print()
        print("=" * 60)
        print()

    def restart_all(self):
        """Force restart all services."""
        print()
        print("  [WATCHDOG] Restarting all services...")

        # Stop existing
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq slate_dashboard*"],
                          capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "Runner.Listener.exe"],
                          capture_output=True)

        time.sleep(2)

        # Restart
        self.restart_counts = {}
        self.start_dashboard()
        time.sleep(2)
        self.start_runner()

        print("  [WATCHDOG] Services restarted")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SLATE Service Watchdog - Monitors and auto-restarts services"
    )
    parser.add_argument("command", nargs="?", default="status",
                       choices=["start", "stop", "status", "restart"],
                       help="Command to run")
    parser.add_argument("--json", action="store_true", help="JSON output for status")

    args = parser.parse_args()
    watchdog = ServiceWatchdog()

    if args.command == "start":
        watchdog.start()
    elif args.command == "stop":
        existing = watchdog._check_existing()
        if existing:
            try:
                os.kill(existing, signal.SIGTERM)
                print(f"[OK] Sent stop signal to watchdog PID {existing}")
            except Exception as e:
                print(f"[!] Failed to stop: {e}")
        else:
            print("[!] Watchdog not running")
    elif args.command == "restart":
        watchdog.restart_all()
    elif args.command == "status":
        if args.json:
            print(json.dumps(watchdog.status(), indent=2))
        else:
            watchdog.print_status()


if __name__ == "__main__":
    main()
