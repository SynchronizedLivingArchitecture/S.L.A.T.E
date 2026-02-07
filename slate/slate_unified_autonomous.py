#!/usr/bin/env python3
"""
SLATE Unified Autonomous Loop
==============================
# Modified: 2026-02-07T04:30:00Z | Author: COPILOT | Change: Initial implementation

Adaptive autonomous agent that discovers, routes, and executes tasks
using local GPU inference. Pulls from KANBAN, tech tree, and codebase
analysis to find work, then uses Ollama for intelligent task processing.

Architecture:
    KANBAN Board --> Task Discovery --> ML Classification --> Agent Routing
         ^              |                    |                    |
         |              v                    v                    v
    Project Board   Codebase Scan     Ollama Inference    Runner Dispatch
         |              |                    |                    |
         v              v                    v                    v
    Completion <--- Result Tracking <--- Execution <--------- GPU Workers

Usage:
    python slate/slate_unified_autonomous.py --run --max 50    # Run loop
    python slate/slate_unified_autonomous.py --status          # Show status
    python slate/slate_unified_autonomous.py --discover        # Discover tasks
    python slate/slate_unified_autonomous.py --single          # Run one task
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Modified: 2026-02-07T04:30:00Z | Author: COPILOT | Change: workspace setup
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# Force UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TASK_FILE = WORKSPACE_ROOT / "current_tasks.json"
STATE_FILE = WORKSPACE_ROOT / ".slate_autonomous_state.json"
LOG_DIR = WORKSPACE_ROOT / "slate_logs" / "autonomous"

# Agent routing patterns
AGENT_PATTERNS = {
    "ALPHA": ["implement", "code", "build", "fix", "create", "add", "refactor", "write"],
    "BETA": ["test", "validate", "verify", "coverage", "check", "lint", "format"],
    "GAMMA": ["analyze", "plan", "research", "document", "review", "design"],
    "DELTA": ["claude", "mcp", "sdk", "integration", "api", "plugin"],
    "COPILOT": ["complex", "multi-step", "orchestrate", "deploy", "release"],
}


class UnifiedAutonomousLoop:
    """Adaptive autonomous agent loop for SLATE."""

    # Modified: 2026-02-07T04:30:00Z | Author: COPILOT | Change: autonomous loop core
    def __init__(self):
        self.workspace = WORKSPACE_ROOT
        self.state = self._load_state()
        self.ml = None  # Lazy-loaded ML orchestrator
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _get_ml(self):
        """Lazy-load ML orchestrator."""
        if self.ml is None:
            from slate.ml_orchestrator import MLOrchestrator
            self.ml = MLOrchestrator()
        return self.ml

    def _load_state(self) -> dict:
        """Load autonomous loop state."""
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return {
            "started_at": None,
            "tasks_discovered": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "cycles": 0,
            "last_cycle": None,
            "adaptations": [],
            "history": [],
        }

    def _save_state(self):
        """Save autonomous loop state."""
        STATE_FILE.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")

    def _log(self, msg: str, level: str = "INFO"):
        """Log a message."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{ts}] [{level}] {msg}"
        print(line)
        log_file = LOG_DIR / f"autonomous_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------
    # Task Discovery
    # ------------------------------------------------------------------

    def discover_tasks(self) -> list[dict]:
        """Discover pending tasks from all sources."""
        tasks = []

        # Source 1: current_tasks.json (pending tasks)
        tasks.extend(self._discover_from_task_file())

        # Source 2: KANBAN board
        tasks.extend(self._discover_from_kanban())

        # Source 3: Codebase analysis (TODOs, FIXMEs, errors)
        tasks.extend(self._discover_from_codebase())

        # Source 4: Test coverage gaps
        tasks.extend(self._discover_from_coverage())

        # Deduplicate by title similarity
        seen_titles = set()
        unique = []
        for task in tasks:
            key = task.get("title", "")[:50].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                unique.append(task)

        self.state["tasks_discovered"] += len(unique)
        self._save_state()

        return unique

    def _discover_from_task_file(self) -> list[dict]:
        """Load pending tasks from current_tasks.json."""
        if not TASK_FILE.exists():
            return []
        try:
            data = json.loads(TASK_FILE.read_text(encoding="utf-8"))
            pending = [
                t for t in data.get("tasks", [])
                if t.get("status") in ("pending", "in_progress")
                and not self._is_stale(t)
            ]
            return pending
        except Exception as e:
            self._log(f"Error reading task file: {e}", "WARN")
            return []

    def _discover_from_kanban(self) -> list[dict]:
        """Sync pending items from KANBAN board."""
        try:
            result = subprocess.run(
                [sys.executable, str(self.workspace / "slate" / "slate_project_board.py"), "--sync"],
                capture_output=True, text=True, timeout=30, cwd=str(self.workspace),
                encoding="utf-8", errors="replace"
            )
            if result.returncode == 0:
                # Re-read task file after sync
                return self._discover_from_task_file()
        except Exception as e:
            self._log(f"KANBAN sync failed: {e}", "WARN")
        return []

    def _discover_from_codebase(self) -> list[dict]:
        """Find TODOs and FIXMEs in codebase."""
        tasks = []
        patterns = ["TODO:", "FIXME:", "HACK:", "BUG:"]
        scan_dirs = ["slate", "agents"]

        for dir_name in scan_dirs:
            dir_path = self.workspace / dir_name
            if not dir_path.exists():
                continue
            for py_file in dir_path.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(content.split("\n"), 1):
                        for pattern in patterns:
                            if pattern in line:
                                rel = str(py_file.relative_to(self.workspace))
                                msg = line.strip().lstrip("#").strip()
                                tasks.append({
                                    "id": f"codebase_{rel}_{i}",
                                    "title": f"{pattern} in {rel}:{i}",
                                    "description": msg,
                                    "priority": "low" if pattern == "TODO:" else "medium",
                                    "source": "codebase_scan",
                                    "file_paths": rel,
                                    "status": "pending",
                                })
                except Exception:
                    pass
        return tasks[:10]  # Cap at 10 to avoid noise

    def _discover_from_coverage(self) -> list[dict]:
        """Find files without tests."""
        tasks = []
        test_dir = self.workspace / "tests"
        slate_dir = self.workspace / "slate"

        if not slate_dir.exists():
            return []

        tested_modules = set()
        if test_dir.exists():
            for test_file in test_dir.rglob("test_*.py"):
                # Extract module name from test filename
                name = test_file.stem.replace("test_", "")
                tested_modules.add(name)

        for py_file in slate_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module = py_file.stem
            if module not in tested_modules:
                tasks.append({
                    "id": f"coverage_{module}",
                    "title": f"Add tests for slate/{module}.py",
                    "description": f"No test file found for slate/{module}.py",
                    "priority": "low",
                    "source": "coverage_gap",
                    "file_paths": f"slate/{module}.py",
                    "status": "pending",
                })
        return tasks[:5]

    def _is_stale(self, task: dict) -> bool:
        """Check if a task is stale (in-progress too long)."""
        started = task.get("started_at")
        if not started or task.get("status") != "in_progress":
            return False
        try:
            start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - start_dt).total_seconds()
            return elapsed > 4 * 3600  # 4 hours
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Task Classification & Routing
    # ------------------------------------------------------------------

    def classify_task(self, task: dict) -> dict:
        """Classify task and determine routing using ML."""
        title = task.get("title", "")
        desc = task.get("description", "")
        combined = f"{title}. {desc}"

        # Fast pattern matching first
        for agent, patterns in AGENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in combined.lower():
                    return {"agent": agent, "method": "pattern", "confidence": 0.9}

        # Fall back to ML inference if Ollama available
        try:
            ml = self._get_ml()
            if ml.ollama.is_running():
                result = ml.classify_task(combined)
                return {
                    "agent": result.get("routed_agent", "ALPHA"),
                    "method": "ml_inference",
                    "confidence": 0.8,
                    "classification": result.get("classification", ""),
                }
        except Exception:
            pass

        # Default to ALPHA (coding agent)
        return {"agent": "ALPHA", "method": "default", "confidence": 0.5}

    # ------------------------------------------------------------------
    # Task Execution
    # ------------------------------------------------------------------

    def execute_task(self, task: dict) -> dict:
        """Execute a single task through the appropriate agent."""
        task_id = task.get("id", "unknown")
        title = task.get("title", "untitled")
        self._log(f"Executing: {title} [{task_id}]")

        # Classify
        routing = self.classify_task(task)
        agent = routing["agent"]
        self._log(f"  Routed to {agent} (method={routing['method']}, conf={routing['confidence']:.1f})")

        # Mark in-progress
        self._update_task_status(task_id, "in_progress")

        try:
            # Execute based on agent type
            if agent in ("ALPHA", "BETA"):
                result = self._execute_code_task(task, agent)
            elif agent == "GAMMA":
                result = self._execute_analysis_task(task)
            elif agent == "DELTA":
                result = self._execute_integration_task(task)
            else:
                result = self._execute_complex_task(task)

            if result.get("success"):
                self._update_task_status(task_id, "completed")
                self.state["tasks_completed"] += 1
                self._log(f"  Completed: {title}")
            else:
                self._update_task_status(task_id, "failed",
                                         error=result.get("error", "Unknown error"))
                self.state["tasks_failed"] += 1
                self._log(f"  Failed: {title} - {result.get('error', '')}", "ERROR")

            # Record in history
            self.state["history"].append({
                "task_id": task_id,
                "title": title,
                "agent": agent,
                "success": result.get("success", False),
                "time": datetime.now(timezone.utc).isoformat(),
                "duration_s": result.get("duration_s", 0),
            })
            # Keep history trimmed
            if len(self.state["history"]) > 100:
                self.state["history"] = self.state["history"][-100:]

            self._save_state()
            return result

        except Exception as e:
            self._update_task_status(task_id, "failed", error=str(e))
            self.state["tasks_failed"] += 1
            self._log(f"  Exception: {e}", "ERROR")
            self._save_state()
            return {"success": False, "error": str(e)}

    def _execute_code_task(self, task: dict, agent: str) -> dict:
        """Execute a coding/testing task using ML inference."""
        start = time.time()
        title = task.get("title", "")
        desc = task.get("description", "")
        file_paths = task.get("file_paths", "")

        try:
            ml = self._get_ml()
            if not ml.ollama.is_running():
                return {"success": False, "error": "Ollama not running"}

            # Read relevant files for context
            context = ""
            if file_paths:
                for fp in str(file_paths).split(","):
                    fp = fp.strip()
                    full_path = self.workspace / fp
                    if full_path.exists():
                        try:
                            content = full_path.read_text(encoding="utf-8", errors="replace")
                            context += f"\n--- {fp} ---\n{content[:3000]}\n"
                        except Exception:
                            pass

            # Generate solution
            task_type = "code_generation" if agent == "ALPHA" else "code_review"
            system_prompt = (
                "You are SLATE, an autonomous coding agent. "
                "Analyze the task and provide a concrete solution. "
                "If code changes are needed, output the full modified code. "
                "Be precise and actionable."
            )
            prompt = f"Task: {title}\nDescription: {desc}\n"
            if context:
                prompt += f"\nRelevant code:\n{context[:5000]}"

            result = ml.infer(prompt, task_type=task_type, system=system_prompt,
                              max_tokens=2048, temperature=0.3)

            elapsed = time.time() - start
            return {
                "success": True,
                "agent": agent,
                "response": result.get("response", "")[:500],
                "model": result.get("model", ""),
                "tokens": result.get("tokens", 0),
                "tok_per_sec": result.get("tok_per_sec", 0),
                "duration_s": round(elapsed, 1),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "duration_s": round(time.time() - start, 1)}

    def _execute_analysis_task(self, task: dict) -> dict:
        """Execute an analysis/planning task."""
        start = time.time()
        try:
            ml = self._get_ml()
            result = ml.infer(
                f"Analyze: {task.get('title', '')}. {task.get('description', '')}",
                task_type="planning",
                system="You are a senior architect. Provide a structured analysis with recommendations.",
                max_tokens=1024,
            )
            return {
                "success": True,
                "agent": "GAMMA",
                "response": result.get("response", "")[:500],
                "duration_s": round(time.time() - start, 1),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "duration_s": round(time.time() - start, 1)}

    def _execute_integration_task(self, task: dict) -> dict:
        """Execute an integration task."""
        start = time.time()
        try:
            ml = self._get_ml()
            result = ml.infer(
                f"Integration task: {task.get('title', '')}. {task.get('description', '')}",
                task_type="general",
                system="You are an integration specialist. Describe the integration steps needed.",
                max_tokens=1024,
            )
            return {
                "success": True,
                "agent": "DELTA",
                "response": result.get("response", "")[:500],
                "duration_s": round(time.time() - start, 1),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "duration_s": round(time.time() - start, 1)}

    def _execute_complex_task(self, task: dict) -> dict:
        """Execute a complex multi-step task."""
        start = time.time()
        try:
            ml = self._get_ml()
            # Break down into steps
            plan_result = ml.infer(
                f"Break this task into 3-5 concrete steps:\n{task.get('title', '')}\n{task.get('description', '')}",
                task_type="planning",
                system="Output a numbered list of steps. Be specific and actionable.",
                max_tokens=512, temperature=0.2,
            )
            return {
                "success": True,
                "agent": "COPILOT",
                "plan": plan_result.get("response", ""),
                "duration_s": round(time.time() - start, 1),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "duration_s": round(time.time() - start, 1)}

    def _update_task_status(self, task_id: str, status: str, error: str = ""):
        """Update task status in current_tasks.json."""
        if not TASK_FILE.exists():
            return
        try:
            data = json.loads(TASK_FILE.read_text(encoding="utf-8"))
            for task in data.get("tasks", []):
                if task.get("id") == task_id:
                    task["status"] = status
                    now = datetime.now(timezone.utc).isoformat()
                    if status == "in_progress":
                        task["started_at"] = now
                        task["started_by"] = "autonomous_loop"
                    elif status == "completed":
                        task["completed_at"] = now
                        task["completed_by"] = "autonomous_loop"
                    elif status == "failed":
                        task["failed_at"] = now
                        task["error"] = error
                    break
            data["last_updated"] = datetime.now(timezone.utc).isoformat()
            TASK_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            self._log(f"Failed to update task status: {e}", "ERROR")

    # ------------------------------------------------------------------
    # Adaptive Learning
    # ------------------------------------------------------------------

    def adapt(self):
        """Adapt strategy based on recent results."""
        history = self.state.get("history", [])
        if len(history) < 5:
            return

        recent = history[-10:]
        success_rate = sum(1 for h in recent if h.get("success")) / len(recent)

        adaptations = []

        if success_rate < 0.5:
            adaptations.append({
                "type": "reduce_complexity",
                "reason": f"Low success rate: {success_rate:.0%}",
                "action": "Prioritize simpler tasks",
                "time": datetime.now(timezone.utc).isoformat(),
            })
            self._log(f"Adapting: low success rate ({success_rate:.0%}), prioritizing simpler tasks", "ADAPT")

        # Check for agent-specific failures
        agent_stats = {}
        for h in recent:
            agent = h.get("agent", "?")
            if agent not in agent_stats:
                agent_stats[agent] = {"success": 0, "fail": 0}
            if h.get("success"):
                agent_stats[agent]["success"] += 1
            else:
                agent_stats[agent]["fail"] += 1

        for agent, stats in agent_stats.items():
            total = stats["success"] + stats["fail"]
            if total >= 3 and stats["fail"] / total > 0.7:
                adaptations.append({
                    "type": "agent_issue",
                    "agent": agent,
                    "reason": f"{agent} failing {stats['fail']}/{total}",
                    "action": f"Re-route {agent} tasks to fallback",
                    "time": datetime.now(timezone.utc).isoformat(),
                })
                self._log(f"Adapting: {agent} has high failure rate, re-routing", "ADAPT")

        if adaptations:
            self.state["adaptations"].extend(adaptations)
            if len(self.state["adaptations"]) > 50:
                self.state["adaptations"] = self.state["adaptations"][-50:]
            self._save_state()

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    def run(self, max_tasks: int = 50, stop_on_empty: bool = False,
            cycle_delay: float = 30.0):
        """Run the autonomous loop."""
        self.state["started_at"] = datetime.now(timezone.utc).isoformat()
        self._save_state()
        self._log(f"Autonomous loop started (max={max_tasks})")

        tasks_executed = 0

        while tasks_executed < max_tasks:
            self.state["cycles"] += 1
            self.state["last_cycle"] = datetime.now(timezone.utc).isoformat()
            self._save_state()

            # Discover tasks
            tasks = self.discover_tasks()
            pending = [t for t in tasks if t.get("status") == "pending"]

            if not pending:
                if stop_on_empty:
                    self._log("No pending tasks, stopping (--stop-on-empty)")
                    break
                self._log(f"No pending tasks, waiting {cycle_delay}s...")
                time.sleep(cycle_delay)
                continue

            # Sort by priority
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            pending.sort(key=lambda t: priority_order.get(t.get("priority", "medium"), 2))

            # Execute top task
            task = pending[0]
            self.execute_task(task)
            tasks_executed += 1

            # Adapt after every 5 tasks
            if tasks_executed % 5 == 0:
                self.adapt()

            # Brief pause between tasks
            time.sleep(2)

        self._log(f"Autonomous loop finished: {tasks_executed} tasks executed")
        self.print_status()

    def run_single(self) -> dict:
        """Discover and execute a single task."""
        tasks = self.discover_tasks()
        pending = [t for t in tasks if t.get("status") == "pending"]
        if not pending:
            self._log("No pending tasks found")
            return {"success": False, "error": "No pending tasks"}

        # Pick highest priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        pending.sort(key=lambda t: priority_order.get(t.get("priority", "medium"), 2))
        return self.execute_task(pending[0])

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get autonomous loop status."""
        # Get ML status
        ml_status = {}
        try:
            ml = self._get_ml()
            ml_status = {
                "ollama_running": ml.ollama.is_running(),
                "models": len(ml.ollama.list_models()),
                "loaded": len(ml.ollama.running_models()),
            }
        except Exception:
            ml_status = {"ollama_running": False}

        return {
            "started_at": self.state.get("started_at"),
            "cycles": self.state.get("cycles", 0),
            "last_cycle": self.state.get("last_cycle"),
            "tasks_discovered": self.state.get("tasks_discovered", 0),
            "tasks_completed": self.state.get("tasks_completed", 0),
            "tasks_failed": self.state.get("tasks_failed", 0),
            "success_rate": (
                self.state.get("tasks_completed", 0) /
                max(self.state.get("tasks_completed", 0) + self.state.get("tasks_failed", 0), 1)
            ),
            "adaptations": len(self.state.get("adaptations", [])),
            "recent_history": self.state.get("history", [])[-5:],
            "ml": ml_status,
        }

    def print_status(self):
        """Print human-readable status."""
        status = self.get_status()
        print("=" * 60)
        print("  SLATE Unified Autonomous Loop")
        print("=" * 60)
        print(f"\n  Started:    {status['started_at'] or 'Never'}")
        print(f"  Cycles:     {status['cycles']}")
        print(f"  Last Cycle: {status['last_cycle'] or 'N/A'}")
        print(f"\n  Discovered: {status['tasks_discovered']}")
        print(f"  Completed:  {status['tasks_completed']}")
        print(f"  Failed:     {status['tasks_failed']}")
        print(f"  Success:    {status['success_rate']:.0%}")
        print(f"  Adaptations: {status['adaptations']}")

        ml = status.get("ml", {})
        print(f"\n  Ollama:     {'Running' if ml.get('ollama_running') else 'Stopped'}")
        print(f"  Models:     {ml.get('models', 0)} available, {ml.get('loaded', 0)} loaded")

        if status["recent_history"]:
            print("\n  Recent Tasks:")
            for h in status["recent_history"]:
                icon = "+" if h.get("success") else "x"
                print(f"    [{icon}] {h.get('title', '?')[:50]} ({h.get('agent', '?')}, {h.get('duration_s', 0)}s)")

        print("\n" + "=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SLATE Unified Autonomous Loop")
    parser.add_argument("--run", action="store_true", help="Run autonomous loop")
    parser.add_argument("--max", type=int, default=50, help="Max tasks to execute")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--discover", action="store_true", help="Discover tasks only")
    parser.add_argument("--single", action="store_true", help="Execute one task")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--stop-on-empty", action="store_true", help="Stop when no tasks")
    args = parser.parse_args()

    loop = UnifiedAutonomousLoop()

    if args.run:
        loop.run(max_tasks=args.max, stop_on_empty=args.stop_on_empty)
    elif args.discover:
        tasks = loop.discover_tasks()
        pending = [t for t in tasks if t.get("status") == "pending"]
        print(f"Discovered {len(tasks)} tasks ({len(pending)} pending):")
        for t in pending[:15]:
            print(f"  [{t.get('priority', '?'):>6}] {t.get('title', '?')[:60]} ({t.get('source', '?')})")
    elif args.single:
        result = loop.run_single()
        print(json.dumps(result, indent=2, default=str))
    elif args.json:
        print(json.dumps(loop.get_status(), indent=2, default=str))
    else:
        loop.print_status()


if __name__ == "__main__":
    main()
