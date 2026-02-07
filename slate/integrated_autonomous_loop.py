#!/usr/bin/env python3
"""
SLATE Integrated Autonomous Loop
==================================
# Modified: 2026-02-07T04:30:00Z | Author: COPILOT | Change: Initial implementation

Top-level integration layer that ties all SLATE autonomous components into
a single adaptive system:
    - ML Orchestrator (GPU inference via Ollama + PyTorch)
    - Unified Autonomous Loop (task discovery + execution)
    - Copilot Runner (chat participant bridge)
    - Orchestrator (service lifecycle)
    - Multi-Runner Coordinator (parallel runner dispatch)
    - Project Board (KANBAN sync)
    - Workflow Manager (task lifecycle)

This is the "brain" — it coordinates everything and provides the top-level
autonomous control loop with self-healing and adaptation.

Usage:
    python slate/integrated_autonomous_loop.py --max 100        # Run loop
    python slate/integrated_autonomous_loop.py --status         # System status
    python slate/integrated_autonomous_loop.py --generate       # Generate tech tree tasks
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Modified: 2026-02-07T04:30:00Z | Author: COPILOT | Change: workspace setup
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

STATE_FILE = WORKSPACE_ROOT / ".slate_integrated_state.json"
LOG_DIR = WORKSPACE_ROOT / "slate_logs" / "integrated"
PYTHON = sys.executable


class IntegratedAutonomousLoop:
    """Top-level SLATE autonomous integration."""

    # Modified: 2026-02-07T04:30:00Z | Author: COPILOT | Change: integrated loop core
    def __init__(self):
        self.workspace = WORKSPACE_ROOT
        self.state = self._load_state()
        self._components = {}  # Lazy-loaded components
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "started_at": None,
            "cycles": 0,
            "last_cycle": None,
            "components_healthy": 0,
            "components_total": 7,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "self_heals": 0,
            "adaptations": [],
            "health_history": [],
        }

    def _save_state(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")

    def _log(self, msg: str, level: str = "INFO"):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{ts}] [INTEGRATED] [{level}] {msg}"
        print(line)
        log_file = LOG_DIR / f"integrated_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------
    # Component Health Checks
    # ------------------------------------------------------------------

    def check_health(self) -> dict:
        """Check all SLATE components."""
        checks = {}

        # 1. Ollama
        checks["ollama"] = self._check_ollama()

        # 2. PyTorch / GPU
        checks["gpu"] = self._check_gpu()

        # 3. Orchestrator services
        checks["orchestrator"] = self._check_component("slate_orchestrator", ["status"])

        # 4. ML Orchestrator
        checks["ml_orchestrator"] = self._check_component("ml_orchestrator", ["--status"])

        # 5. Multi-runner
        checks["multi_runner"] = self._check_file_exists("slate/slate_multi_runner.py")

        # 6. Workflow manager
        checks["workflow_manager"] = self._check_component("slate_workflow_manager", ["--status"])

        # 7. Project board
        checks["project_board"] = self._check_component("slate_project_board", ["--status"])

        healthy = sum(1 for v in checks.values() if v.get("healthy", False))
        self.state["components_healthy"] = healthy
        self.state["components_total"] = len(checks)
        self.state["health_history"].append({
            "time": datetime.now(timezone.utc).isoformat(),
            "healthy": healthy,
            "total": len(checks),
        })
        if len(self.state["health_history"]) > 100:
            self.state["health_history"] = self.state["health_history"][-100:]
        self._save_state()

        return checks

    def _check_ollama(self) -> dict:
        """Check if Ollama is running and has models."""
        try:
            import urllib.request
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = data.get("models", [])
                return {"healthy": True, "models": len(models)}
        except Exception as e:
            return {"healthy": False, "error": str(e)[:80]}

    def _check_gpu(self) -> dict:
        """Check GPU availability."""
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                gpus = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
                return {"healthy": True, "gpus": len(gpus), "details": gpus}
        except Exception as e:
            return {"healthy": False, "error": str(e)[:80]}
        return {"healthy": False}

    def _check_component(self, module: str, args: list) -> dict:
        """Check a SLATE component by running it."""
        script = self.workspace / "slate" / f"{module}.py"
        if not script.exists():
            return {"healthy": False, "error": f"{module}.py not found"}
        try:
            r = subprocess.run(
                [PYTHON, str(script)] + args,
                capture_output=True, text=True, timeout=30,
                cwd=str(self.workspace), encoding="utf-8", errors="replace",
            )
            return {"healthy": r.returncode == 0, "output_lines": len(r.stdout.split("\n"))}
        except Exception as e:
            return {"healthy": False, "error": str(e)[:80]}

    def _check_file_exists(self, path: str) -> dict:
        """Check if a file exists."""
        full = self.workspace / path
        return {"healthy": full.exists(), "path": path}

    # ------------------------------------------------------------------
    # Self-Healing
    # ------------------------------------------------------------------

    def self_heal(self, health: dict) -> list[str]:
        """Attempt to fix unhealthy components."""
        fixes = []

        # Fix Ollama not running
        if not health.get("ollama", {}).get("healthy"):
            self._log("Self-heal: attempting to start Ollama", "HEAL")
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                time.sleep(3)
                fixes.append("Started Ollama serve")
            except Exception as e:
                fixes.append(f"Failed to start Ollama: {e}")

        # Fix orchestrator
        if not health.get("orchestrator", {}).get("healthy"):
            self._log("Self-heal: restarting orchestrator", "HEAL")
            try:
                subprocess.run(
                    [PYTHON, str(self.workspace / "slate" / "slate_orchestrator.py"), "start"],
                    capture_output=True, text=True, timeout=30,
                    cwd=str(self.workspace),
                )
                fixes.append("Restarted orchestrator")
            except Exception as e:
                fixes.append(f"Orchestrator restart failed: {e}")

        if fixes:
            self.state["self_heals"] += len(fixes)
            self._save_state()

        return fixes

    # ------------------------------------------------------------------
    # Tech Tree Task Generation
    # ------------------------------------------------------------------

    def generate_tech_tree_tasks(self) -> list[dict]:
        """Generate tasks from tech tree / project analysis."""
        tasks = []

        # Analyze codebase for improvement areas
        slate_dir = self.workspace / "slate"
        if not slate_dir.exists():
            return tasks

        for py_file in sorted(slate_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                lines = content.split("\n")
                line_count = len(lines)

                # Check for missing docstrings
                has_module_doc = '"""' in "\n".join(lines[:5])
                if not has_module_doc:
                    tasks.append({
                        "id": f"techtree_doc_{py_file.stem}",
                        "title": f"Add module docstring to {py_file.name}",
                        "priority": "low",
                        "source": "tech_tree",
                        "file_paths": str(py_file.relative_to(self.workspace)),
                        "status": "pending",
                    })

                # Check for missing type hints (basic heuristic)
                func_defs = [l for l in lines if l.strip().startswith("def ") and "->" not in l]
                if len(func_defs) > 3:
                    tasks.append({
                        "id": f"techtree_types_{py_file.stem}",
                        "title": f"Add type hints to {py_file.name} ({len(func_defs)} functions)",
                        "priority": "low",
                        "source": "tech_tree",
                        "file_paths": str(py_file.relative_to(self.workspace)),
                        "status": "pending",
                    })

                # Check for error handling
                try_blocks = sum(1 for l in lines if l.strip().startswith("try:"))
                bare_except = sum(1 for l in lines if "except:" in l or "except Exception:" in l)
                if bare_except > try_blocks * 0.5 and bare_except > 2:
                    tasks.append({
                        "id": f"techtree_errors_{py_file.stem}",
                        "title": f"Improve error handling in {py_file.name}",
                        "priority": "medium",
                        "source": "tech_tree",
                        "file_paths": str(py_file.relative_to(self.workspace)),
                        "status": "pending",
                    })

            except Exception:
                pass

        self._log(f"Generated {len(tasks)} tech tree tasks")
        return tasks

    # ------------------------------------------------------------------
    # Main Integrated Loop
    # ------------------------------------------------------------------

    def run(self, max_tasks: int = 100, cycle_delay: float = 30.0):
        """Run the integrated autonomous loop."""
        self.state["started_at"] = datetime.now(timezone.utc).isoformat()
        self._save_state()
        self._log(f"Integrated Autonomous Loop started (max={max_tasks})")

        # Import the unified loop
        from slate.slate_unified_autonomous import UnifiedAutonomousLoop
        auto = UnifiedAutonomousLoop()

        tasks_run = 0

        while tasks_run < max_tasks:
            self.state["cycles"] += 1
            self.state["last_cycle"] = datetime.now(timezone.utc).isoformat()
            self._save_state()
            self._log(f"--- Cycle {self.state['cycles']} ---")

            # Phase 1: Health check
            health = self.check_health()
            healthy_count = sum(1 for v in health.values() if v.get("healthy"))
            self._log(f"Health: {healthy_count}/{len(health)} components healthy")

            # Phase 2: Self-heal if needed
            if healthy_count < len(health):
                fixes = self.self_heal(health)
                for fix in fixes:
                    self._log(f"  Heal: {fix}")

            # Phase 3: Check Ollama (required for inference)
            if not health.get("ollama", {}).get("healthy"):
                self._log("Ollama not available, waiting...", "WARN")
                time.sleep(cycle_delay)
                continue

            # Phase 4: Generate new tasks if running low
            existing = auto.discover_tasks()
            pending = [t for t in existing if t.get("status") == "pending"]
            if len(pending) < 3:
                new_tasks = self.generate_tech_tree_tasks()
                if new_tasks:
                    self._inject_tasks(new_tasks[:5])
                    self._log(f"Injected {min(len(new_tasks), 5)} tech tree tasks")

            # Phase 5: Execute tasks via unified loop
            batch_size = min(5, max_tasks - tasks_run)
            for _ in range(batch_size):
                tasks = auto.discover_tasks()
                pending = [t for t in tasks if t.get("status") == "pending"]
                if not pending:
                    break

                # Priority sort
                prio = {"critical": 0, "high": 1, "medium": 2, "low": 3}
                pending.sort(key=lambda t: prio.get(t.get("priority", "medium"), 2))

                result = auto.execute_task(pending[0])
                tasks_run += 1

                if result.get("success"):
                    self.state["tasks_completed"] += 1
                else:
                    self.state["tasks_failed"] += 1
                self._save_state()

            # Phase 6: Adapt
            auto.adapt()

            # Phase 7: Sync results to KANBAN
            self._sync_to_kanban()

            self._log(f"Cycle complete: {tasks_run} total tasks run")
            time.sleep(cycle_delay)

        self._log(f"Integrated loop finished: {tasks_run} tasks executed")
        self.print_status()

    def _inject_tasks(self, tasks: list[dict]):
        """Inject generated tasks into current_tasks.json."""
        try:
            data = {"tasks": [], "last_updated": ""}
            if (self.workspace / "current_tasks.json").exists():
                data = json.loads((self.workspace / "current_tasks.json").read_text(encoding="utf-8"))

            existing_ids = {t.get("id") for t in data.get("tasks", [])}
            for task in tasks:
                if task.get("id") not in existing_ids:
                    data["tasks"].append(task)

            data["last_updated"] = datetime.now(timezone.utc).isoformat()
            (self.workspace / "current_tasks.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as e:
            self._log(f"Failed to inject tasks: {e}", "ERROR")

    def _sync_to_kanban(self):
        """Push completed tasks back to KANBAN board."""
        try:
            subprocess.run(
                [PYTHON, str(self.workspace / "slate" / "slate_project_board.py"), "--push"],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.workspace), encoding="utf-8", errors="replace",
            )
        except Exception:
            pass  # Non-critical

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get integrated loop status."""
        health = self.check_health()
        healthy = sum(1 for v in health.values() if v.get("healthy"))

        return {
            "started_at": self.state.get("started_at"),
            "cycles": self.state.get("cycles", 0),
            "last_cycle": self.state.get("last_cycle"),
            "components": {
                "healthy": healthy,
                "total": len(health),
                "details": {k: v.get("healthy", False) for k, v in health.items()},
            },
            "tasks_completed": self.state.get("tasks_completed", 0),
            "tasks_failed": self.state.get("tasks_failed", 0),
            "self_heals": self.state.get("self_heals", 0),
            "adaptations": len(self.state.get("adaptations", [])),
        }

    def print_status(self):
        """Print human-readable status."""
        s = self.get_status()
        health = s.get("components", {}).get("details", {})

        print("=" * 65)
        print("  SLATE Integrated Autonomous Loop")
        print("=" * 65)
        print(f"\n  Started:     {s.get('started_at', 'Never')}")
        print(f"  Cycles:      {s['cycles']}")
        print(f"  Last Cycle:  {s.get('last_cycle', 'N/A')}")
        print(f"\n  Components:  {s['components']['healthy']}/{s['components']['total']} healthy")

        for name, ok in health.items():
            icon = "+" if ok else "x"
            print(f"    [{icon}] {name}")

        total = s["tasks_completed"] + s["tasks_failed"]
        rate = s["tasks_completed"] / max(total, 1)
        print(f"\n  Completed:   {s['tasks_completed']}")
        print(f"  Failed:      {s['tasks_failed']}")
        print(f"  Success:     {rate:.0%}")
        print(f"  Self-Heals:  {s['self_heals']}")
        print(f"  Adaptations: {s['adaptations']}")
        print("\n" + "=" * 65)


def main():
    parser = argparse.ArgumentParser(description="SLATE Integrated Autonomous Loop")
    parser.add_argument("--max", type=int, default=100, help="Max tasks")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--generate", action="store_true", help="Generate tech tree tasks")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    loop = IntegratedAutonomousLoop()

    if args.generate:
        tasks = loop.generate_tech_tree_tasks()
        print(f"Generated {len(tasks)} tasks:")
        for t in tasks:
            print(f"  [{t.get('priority', '?'):>6}] {t.get('title', '')[:60]}")
    elif args.json:
        print(json.dumps(loop.get_status(), indent=2, default=str))
    elif args.status or (not args.max or args.max == 100):
        loop.print_status()
    else:
        loop.run(max_tasks=args.max)


if __name__ == "__main__":
    main()
