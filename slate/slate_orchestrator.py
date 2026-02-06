#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_orchestrator [python]
# Author: Claude | Created: 2026-02-06T23:30:00Z
# Purpose: Unified SLATE system orchestrator - manages all services including GitHub runner
# ═══════════════════════════════════════════════════════════════════════════════
"""
SLATE Orchestrator
==================
Manages the complete SLATE system lifecycle:
- Dashboard server (FastAPI on port 8080)
- GitHub Actions runner (self-hosted)
- Workflow manager (task lifecycle)
- System health monitoring

Usage:
    python slate/slate_orchestrator.py start      # Start all SLATE services
    python slate/slate_orchestrator.py stop       # Stop all services
    python slate/slate_orchestrator.py status     # Show service status
    python slate/slate_orchestrator.py restart    # Restart all services
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# PID file for tracking running orchestrator
PID_FILE = WORKSPACE_ROOT / ".slate_orchestrator.pid"
STATE_FILE = WORKSPACE_ROOT / ".slate_orchestrator_state.json"


class SlateOrchestrator:
    """Manages all SLATE system services."""

    def __init__(self):
        self.workspace = WORKSPACE_ROOT
        self.runner_dir = self.workspace / "actions-runner"
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = False
        self._shutdown_event = threading.Event()

    def _get_python(self) -> str:
        """Get venv Python path."""
        if os.name == "nt":
            return str(self.workspace / ".venv" / "Scripts" / "python.exe")
        return str(self.workspace / ".venv" / "bin" / "python")

    def _save_state(self, state: Dict[str, Any]):
        """Save orchestrator state to file."""
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_state(self) -> Dict[str, Any]:
        """Load orchestrator state from file."""
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
        """Check if orchestrator is already running."""
        if not PID_FILE.exists():
            return None
        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if process exists
            if os.name == "nt":
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    capture_output=True, text=True
                )
                if str(pid) in result.stdout:
                    return pid
            else:
                os.kill(pid, 0)  # Signal 0 = check if exists
                return pid
        except (ValueError, OSError, ProcessLookupError):
            self._clear_pid()
        return None

    def start_runner(self) -> bool:
        """Start the GitHub Actions runner."""
        run_cmd = self.runner_dir / "run.cmd"
        if not run_cmd.exists():
            print("  [!] Runner not installed at", self.runner_dir)
            return False

        # FIRST: Check if runner process is already running locally
        try:
            proc_check = subprocess.run(
                ["powershell", "-Command",
                 "Get-Process -Name 'Runner.Listener' -ErrorAction SilentlyContinue | Select-Object Id"],
                capture_output=True, text=True, timeout=5
            )
            if proc_check.returncode == 0 and proc_check.stdout.strip() and "Id" in proc_check.stdout:
                print("  [OK] Runner process already running")
                return True
        except Exception:
            pass

        # SECOND: Check GitHub API for runner status
        try:
            from slate.slate_runner_manager import SlateRunnerManager
            manager = SlateRunnerManager()
            detection = manager.detect()

            gh = detection.get("github", {})
            if gh.get("authenticated"):
                gh_cli = manager.gh_cli
                result = subprocess.run(
                    [gh_cli, "api", "repos/SynchronizedLivingArchitecture/S.L.A.T.E/actions/runners",
                     "--jq", ".runners[] | select(.name==\"slate-DESKTOP-R3UD82D\") | .status"],
                    capture_output=True, text=True, timeout=10, cwd=str(self.workspace)
                )
                if result.returncode == 0 and "online" in result.stdout:
                    print("  [OK] Runner already online (GitHub)")
                    return True
        except Exception:
            pass

        # Start runner process (only if not already running)
        try:
            if os.name == "nt":
                # Windows: start in background
                process = subprocess.Popen(
                    ["cmd", "/c", str(run_cmd)],
                    cwd=str(self.runner_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
                )
            else:
                process = subprocess.Popen(
                    [str(run_cmd)],
                    cwd=str(self.runner_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )

            self.processes["runner"] = process
            print(f"  [OK] Runner started (PID {process.pid})")
            return True
        except Exception as e:
            print(f"  [X] Runner failed: {e}")
            return False

    def start_dashboard(self) -> bool:
        """Start the SLATE dashboard server."""
        dashboard_script = self.workspace / "agents" / "slate_dashboard_server.py"
        if not dashboard_script.exists():
            # Try alternate path
            dashboard_script = self.workspace / "agents" / "slate_dashboard_server.py"

        if not dashboard_script.exists():
            print("  [!] Dashboard server not found")
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
            print(f"  [OK] Dashboard started (PID {process.pid}) > http://127.0.0.1:8080")
            return True
        except Exception as e:
            print(f"  [X] Dashboard failed: {e}")
            return False

    def start_workflow_monitor(self) -> bool:
        """Start the workflow monitor for task lifecycle management."""
        # This runs as a background thread, not a separate process
        try:
            from slate.slate_workflow_manager import SlateWorkflowManager

            def monitor_loop():
                manager = SlateWorkflowManager()
                while not self._shutdown_event.is_set():
                    try:
                        # Check and cleanup every 5 minutes
                        analysis = manager.analyze_tasks()
                        if analysis.get("needs_attention"):
                            manager.cleanup(dry_run=False)
                    except Exception:
                        pass
                    self._shutdown_event.wait(300)  # 5 minute interval

            thread = threading.Thread(target=monitor_loop, daemon=True, name="workflow-monitor")
            thread.start()
            print("  [OK] Workflow monitor started")
            return True
        except Exception as e:
            print(f"  [!] Workflow monitor skipped: {e}")
            return True  # Non-fatal

    def start(self) -> bool:
        """Start all SLATE services."""
        existing_pid = self._check_existing()
        if existing_pid:
            print(f"[!] SLATE orchestrator already running (PID {existing_pid})")
            return False

        print()
        print("=" * 60)
        print("  Starting SLATE Orchestrator")
        print("=" * 60)
        print()

        self._write_pid()
        atexit.register(self._clear_pid)

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        services_started = 0
        services_total = 3

        # 1. Start GitHub Runner
        print("  [1/3] GitHub Runner...")
        if self.start_runner():
            services_started += 1
        time.sleep(1)

        # 2. Start Dashboard
        print("  [2/3] Dashboard Server...")
        if self.start_dashboard():
            services_started += 1
        time.sleep(0.5)

        # 3. Start Workflow Monitor
        print("  [3/3] Workflow Monitor...")
        if self.start_workflow_monitor():
            services_started += 1

        print()
        print("=" * 60)
        print(f"  SLATE Started: {services_started}/{services_total} services")
        print("=" * 60)
        print()
        print("  Dashboard:  http://127.0.0.1:8080")
        print("  Runner:     slate-DESKTOP-R3UD82D")
        print()
        print("  Press Ctrl+C to stop")
        print()

        # Save state
        self._save_state({
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "services": {
                "runner": "runner" in self.processes,
                "dashboard": "dashboard" in self.processes,
                "workflow_monitor": True
            },
            "pid": os.getpid()
        })

        self.running = True

        # Keep running until shutdown
        try:
            while self.running and not self._shutdown_event.is_set():
                # Check if child processes are still running
                for name, proc in list(self.processes.items()):
                    if proc.poll() is not None:
                        print(f"  [!] {name} exited with code {proc.returncode}")
                        del self.processes[name]

                time.sleep(5)
        except KeyboardInterrupt:
            pass

        self.stop()
        return True

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n  [!] Shutdown signal received...")
        self.running = False
        self._shutdown_event.set()

    def stop(self):
        """Stop all SLATE services."""
        print()
        print("  Stopping SLATE services...")

        self._shutdown_event.set()

        # Stop managed processes
        for name, proc in self.processes.items():
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"  [OK] {name} stopped")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  [!] {name} killed")
            except Exception as e:
                print(f"  [!] {name} stop error: {e}")

        # Stop any orphaned runner processes
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/IM", "Runner.Listener.exe"],
                    capture_output=True
                )
        except Exception:
            pass

        self._save_state({
            "status": "stopped",
            "stopped_at": datetime.now(timezone.utc).isoformat(),
            "services": {}
        })

        self._clear_pid()
        print("  [OK] SLATE stopped")
        print()

    def status(self) -> Dict[str, Any]:
        """Get status of all SLATE services."""
        result = {
            "orchestrator": {"running": False, "pid": None},
            "runner": {"running": False, "status": "unknown"},
            "dashboard": {"running": False, "port": 8080},
            "workflow": {"task_count": 0, "healthy": False}
        }

        # Check orchestrator
        existing_pid = self._check_existing()
        if existing_pid:
            result["orchestrator"]["running"] = True
            result["orchestrator"]["pid"] = existing_pid

        # Check runner via GitHub API
        try:
            from slate.slate_runner_manager import SlateRunnerManager
            manager = SlateRunnerManager()
            gh_cli = manager.gh_cli

            check = subprocess.run(
                [gh_cli, "api", "repos/SynchronizedLivingArchitecture/S.L.A.T.E/actions/runners",
                 "--jq", ".runners[0] | {status, busy}"],
                capture_output=True, text=True, timeout=10, cwd=str(self.workspace)
            )
            if check.returncode == 0:
                data = json.loads(check.stdout)
                result["runner"]["running"] = data.get("status") == "online"
                result["runner"]["status"] = data.get("status", "unknown")
                result["runner"]["busy"] = data.get("busy", False)
        except Exception:
            pass

        # Check dashboard
        try:
            import urllib.request
            req = urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
            result["dashboard"]["running"] = req.status == 200
        except Exception:
            pass

        # Check workflow
        try:
            from slate.slate_workflow_manager import SlateWorkflowManager
            manager = SlateWorkflowManager()
            analysis = manager.analyze_tasks()
            result["workflow"]["task_count"] = analysis.get("total", 0)
            result["workflow"]["healthy"] = not analysis.get("needs_attention", True)
            result["workflow"]["in_progress"] = analysis.get("by_status", {}).get("in-progress", 0)
        except Exception:
            pass

        return result

    def print_status(self):
        """Print formatted status."""
        status = self.status()

        print()
        print("=" * 60)
        print("  SLATE System Status")
        print("=" * 60)
        print()

        # Orchestrator
        orch = status["orchestrator"]
        if orch["running"]:
            print(f"  Orchestrator:  Running (PID {orch['pid']})")
        else:
            print("  Orchestrator:  Stopped")

        # Runner
        runner = status["runner"]
        if runner["running"]:
            busy = " (busy)" if runner.get("busy") else ""
            print(f"  Runner:        Online{busy}")
        else:
            print(f"  Runner:        {runner['status']}")

        # Dashboard
        dash = status["dashboard"]
        if dash["running"]:
            print(f"  Dashboard:     Running > http://127.0.0.1:{dash['port']}")
        else:
            print("  Dashboard:     Stopped")

        # Workflow
        wf = status["workflow"]
        health = "Healthy" if wf["healthy"] else "Needs attention"
        print(f"  Workflow:      {wf['task_count']} tasks ({health})")
        if wf.get("in_progress", 0) > 0:
            print(f"                 {wf['in_progress']} in progress")

        print()
        print("=" * 60)
        print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SLATE Orchestrator - Manages all SLATE services"
    )
    parser.add_argument("command", nargs="?", default="status",
                       choices=["start", "stop", "status", "restart"],
                       help="Command to run")
    parser.add_argument("--json", action="store_true", help="JSON output for status")

    args = parser.parse_args()

    orchestrator = SlateOrchestrator()

    if args.command == "start":
        orchestrator.start()

    elif args.command == "stop":
        existing = orchestrator._check_existing()
        if existing:
            try:
                os.kill(existing, signal.SIGTERM)
                print(f"[OK] Sent stop signal to PID {existing}")
                time.sleep(2)
            except Exception as e:
                print(f"[!] Failed to stop: {e}")
        else:
            print("[!] Orchestrator not running")
        orchestrator.stop()

    elif args.command == "restart":
        existing = orchestrator._check_existing()
        if existing:
            try:
                os.kill(existing, signal.SIGTERM)
                time.sleep(3)
            except Exception:
                pass
        orchestrator.start()

    elif args.command == "status":
        if args.json:
            print(json.dumps(orchestrator.status(), indent=2))
        else:
            orchestrator.print_status()


if __name__ == "__main__":
    main()
