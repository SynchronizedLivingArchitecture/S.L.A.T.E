#!/usr/bin/env python3
"""
SLATE Copilot Runner
=====================
# Modified: 2026-02-07T04:30:00Z | Author: COPILOT | Change: Initial implementation

Bridges VS Code Copilot Chat participant to the SLATE autonomous system.
Processes tasks dispatched from the @slate chat participant and feeds
results back. Acts as the Copilot-driven orchestration layer.

Architecture:
    @slate Chat Participant --> Copilot Runner --> Unified Autonomous
         |                         |                     |
         v                         v                     v
    User Intent              Task Queue            ML Inference
         |                         |                     |
         v                         v                     v
    Response <---------- Completion Tracking <---- GPU Workers

Usage:
    python slate/copilot_slate_runner.py --start --max-tasks 50
    python slate/copilot_slate_runner.py --status
    python slate/copilot_slate_runner.py --stop
    python slate/copilot_slate_runner.py --queue "fix the dashboard"
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Modified: 2026-02-07T04:30:00Z | Author: COPILOT | Change: workspace setup
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

STATE_FILE = WORKSPACE_ROOT / ".slate_copilot_runner.json"
QUEUE_FILE = WORKSPACE_ROOT / ".slate_copilot_queue.json"
TASK_FILE = WORKSPACE_ROOT / "current_tasks.json"
LOG_DIR = WORKSPACE_ROOT / "slate_logs" / "copilot_runner"
PID_FILE = WORKSPACE_ROOT / ".slate_copilot_runner.pid"


class CopilotSlateRunner:
    """Copilot-driven task runner for SLATE."""

    # Modified: 2026-02-07T04:30:00Z | Author: COPILOT | Change: runner core
    def __init__(self):
        self.workspace = WORKSPACE_ROOT
        self.state = self._load_state()
        self.autonomous = None  # Lazy-loaded
        self._running = True
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _get_autonomous(self):
        """Lazy-load the unified autonomous loop."""
        if self.autonomous is None:
            from slate.slate_unified_autonomous import UnifiedAutonomousLoop
            self.autonomous = UnifiedAutonomousLoop()
        return self.autonomous

    def _load_state(self) -> dict:
        """Load runner state."""
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "status": "stopped",
            "started_at": None,
            "pid": None,
            "tasks_processed": 0,
            "tasks_queued": 0,
            "last_task": None,
            "copilot_requests": [],
        }

    def _save_state(self):
        """Save runner state."""
        STATE_FILE.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")

    def _log(self, msg: str, level: str = "INFO"):
        """Log a message."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{ts}] [{level}] {msg}"
        print(line)
        log_file = LOG_DIR / f"copilot_runner_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------
    # Queue Management
    # ------------------------------------------------------------------

    def _load_queue(self) -> list[dict]:
        """Load the Copilot request queue."""
        if QUEUE_FILE.exists():
            try:
                return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []

    def _save_queue(self, queue: list[dict]):
        """Save the Copilot request queue."""
        QUEUE_FILE.write_text(json.dumps(queue, indent=2, default=str), encoding="utf-8")

    def queue_task(self, description: str, priority: str = "medium",
                   source: str = "copilot_chat") -> dict:
        """Add a task to the queue from Copilot Chat."""
        task = {
            "id": f"copilot_{int(time.time())}_{os.getpid()}",
            "title": description[:100],
            "description": description,
            "priority": priority,
            "source": source,
            "status": "pending",
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add to queue file
        queue = self._load_queue()
        queue.append(task)
        self._save_queue(queue)

        # Also inject into current_tasks.json
        self._inject_task(task)

        self.state["tasks_queued"] = self.state.get("tasks_queued", 0) + 1
        self.state["copilot_requests"].append({
            "id": task["id"],
            "description": description[:80],
            "time": task["queued_at"],
        })
        # Keep recent requests trimmed
        if len(self.state["copilot_requests"]) > 50:
            self.state["copilot_requests"] = self.state["copilot_requests"][-50:]
        self._save_state()

        self._log(f"Queued: {description[:60]}")
        return task

    def _inject_task(self, task: dict):
        """Inject a Copilot task into current_tasks.json."""
        try:
            data = {"tasks": [], "last_updated": ""}
            if TASK_FILE.exists():
                data = json.loads(TASK_FILE.read_text(encoding="utf-8"))

            data["tasks"].append({
                "id": task["id"],
                "title": task["title"],
                "description": task["description"],
                "priority": task["priority"],
                "source": task["source"],
                "status": "pending",
                "created_at": task.get("queued_at", datetime.now(timezone.utc).isoformat()),
            })
            data["last_updated"] = datetime.now(timezone.utc).isoformat()
            TASK_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            self._log(f"Failed to inject task: {e}", "ERROR")

    # ------------------------------------------------------------------
    # Runner Start/Stop
    # ------------------------------------------------------------------

    def start(self, max_tasks: int = 50, stop_on_empty: bool = False):
        """Start the Copilot runner."""
        self.state["status"] = "running"
        self.state["started_at"] = datetime.now(timezone.utc).isoformat()
        self.state["pid"] = os.getpid()
        self._save_state()

        # Write PID file
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

        # Handle graceful shutdown
        def _shutdown(sig, frame):
            self._log("Shutdown signal received")
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        self._log(f"Copilot Runner started (PID={os.getpid()}, max={max_tasks})")

        tasks_run = 0
        auto = self._get_autonomous()

        while self._running and tasks_run < max_tasks:
            # Check for Copilot-queued tasks first
            queue = self._load_queue()
            pending_queue = [q for q in queue if q.get("status") == "pending"]

            if pending_queue:
                task = pending_queue[0]
                self._log(f"Processing Copilot request: {task.get('title', '')[:50]}")
                task["status"] = "processing"
                self._save_queue(queue)

                result = auto.execute_task(task)
                tasks_run += 1
                self.state["tasks_processed"] += 1
                self.state["last_task"] = {
                    "id": task["id"],
                    "title": task.get("title", ""),
                    "result": "success" if result.get("success") else "failed",
                    "time": datetime.now(timezone.utc).isoformat(),
                }
                self._save_state()

                # Remove from queue
                queue = [q for q in self._load_queue() if q["id"] != task["id"]]
                self._save_queue(queue)
                continue

            # Fall back to autonomous discovery
            tasks = auto.discover_tasks()
            pending = [t for t in tasks if t.get("status") == "pending"]

            if not pending:
                if stop_on_empty:
                    self._log("No tasks, stopping (--stop-on-empty)")
                    break
                time.sleep(15)
                continue

            # Priority sort
            priority_map = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            pending.sort(key=lambda t: priority_map.get(t.get("priority", "medium"), 2))

            task = pending[0]
            result = auto.execute_task(task)
            tasks_run += 1
            self.state["tasks_processed"] += 1
            self.state["last_task"] = {
                "id": task.get("id", ""),
                "title": task.get("title", ""),
                "result": "success" if result.get("success") else "failed",
                "time": datetime.now(timezone.utc).isoformat(),
            }
            self._save_state()

            # Adapt every 5 tasks
            if tasks_run % 5 == 0:
                auto.adapt()

            time.sleep(2)

        self.state["status"] = "stopped"
        self._save_state()
        if PID_FILE.exists():
            PID_FILE.unlink()
        self._log(f"Copilot Runner stopped ({tasks_run} tasks processed)")

    def stop(self):
        """Stop the Copilot runner."""
        if PID_FILE.exists():
            try:
                pid = int(PID_FILE.read_text(encoding="utf-8").strip())
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                                   capture_output=True, timeout=10)
                else:
                    os.kill(pid, signal.SIGTERM)
                self._log(f"Sent stop signal to PID {pid}")
            except (ValueError, ProcessLookupError, OSError) as e:
                self._log(f"Could not stop PID: {e}", "WARN")
            finally:
                PID_FILE.unlink(missing_ok=True)

        self.state["status"] = "stopped"
        self._save_state()
        print("Copilot Runner stopped")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get runner status."""
        is_alive = False
        if PID_FILE.exists():
            try:
                pid = int(PID_FILE.read_text(encoding="utf-8").strip())
                if sys.platform == "win32":
                    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                       capture_output=True, text=True, timeout=5)
                    is_alive = str(pid) in r.stdout
                else:
                    os.kill(pid, 0)
                    is_alive = True
            except Exception:
                pass

        queue = self._load_queue()
        pending = [q for q in queue if q.get("status") == "pending"]

        return {
            "status": "running" if is_alive else self.state.get("status", "stopped"),
            "pid": self.state.get("pid"),
            "process_alive": is_alive,
            "started_at": self.state.get("started_at"),
            "tasks_processed": self.state.get("tasks_processed", 0),
            "tasks_queued": len(pending),
            "last_task": self.state.get("last_task"),
            "copilot_requests_total": len(self.state.get("copilot_requests", [])),
        }

    def print_status(self):
        """Print human-readable status."""
        s = self.get_status()
        running = s["status"] == "running" and s.get("process_alive", False)
        print("=" * 60)
        print("  SLATE Copilot Runner")
        print("=" * 60)
        print(f"\n  Status:     {'RUNNING' if running else 'STOPPED'}")
        print(f"  PID:        {s.get('pid', 'N/A')}")
        print(f"  Started:    {s.get('started_at', 'Never')}")
        print(f"  Processed:  {s['tasks_processed']}")
        print(f"  Queued:     {s['tasks_queued']}")
        print(f"  Copilot Requests: {s['copilot_requests_total']}")

        last = s.get("last_task")
        if last:
            print(f"\n  Last Task: [{last.get('result', '?')}] {last.get('title', '?')[:50]}")
            print(f"             {last.get('time', '?')}")

        print("\n" + "=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SLATE Copilot Runner")
    parser.add_argument("--start", action="store_true", help="Start runner")
    parser.add_argument("--stop", action="store_true", help="Stop runner")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--queue", type=str, help="Queue a task from CLI")
    parser.add_argument("--max-tasks", type=int, default=50, help="Max tasks")
    parser.add_argument("--stop-on-empty", action="store_true", help="Stop when empty")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    runner = CopilotSlateRunner()

    if args.start:
        runner.start(max_tasks=args.max_tasks, stop_on_empty=args.stop_on_empty)
    elif args.stop:
        runner.stop()
    elif args.queue:
        task = runner.queue_task(args.queue)
        print(f"Queued: {task['id']}")
    elif args.json:
        print(json.dumps(runner.get_status(), indent=2, default=str))
    else:
        runner.print_status()


if __name__ == "__main__":
    main()
