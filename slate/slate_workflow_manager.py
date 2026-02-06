#!/usr/bin/env python3
"""
SLATE Workflow Manager
======================
# Modified: 2026-02-06T22:00:00Z | Author: Claude | Change: Initial implementation

Manages task lifecycle, PR workflows, and ensures work completion before new tasks.

Key Features:
- Automatic stale task detection and cleanup
- Task completion enforcement (no new tasks until old complete)
- PR/branch workflow integration
- Deprecated task removal
- Re-engagement of abandoned tasks

Usage:
    python slate/slate_workflow_manager.py --status
    python slate/slate_workflow_manager.py --cleanup
    python slate/slate_workflow_manager.py --enforce
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    import filelock
except ImportError:
    filelock = None  # Graceful degradation


class SlateWorkflowManager:
    """Manages SLATE task and PR workflows with automatic cleanup."""

    # Configuration
    TASK_FILE = WORKSPACE_ROOT / "current_tasks.json"
    ARCHIVE_FILE = WORKSPACE_ROOT / ".slate_archive" / "archived_tasks.json"
    STALE_HOURS = 4  # Tasks in-progress for more than this are stale
    ABANDONED_HOURS = 24  # Tasks pending for more than this are abandoned
    MAX_CONCURRENT_TASKS = 5  # Max tasks allowed before requiring cleanup
    TEST_TASK_PATTERNS = ["Test Task", "test task", "Integration test task"]

    def __init__(self):
        self.gh_cli = self._find_gh_cli()
        self._ensure_archive_dir()

    def _find_gh_cli(self) -> str:
        """Find gh CLI path."""
        gh_path = WORKSPACE_ROOT / ".tools" / "gh.exe"
        if gh_path.exists():
            return str(gh_path)
        return "gh"  # Fall back to PATH

    def _ensure_archive_dir(self):
        """Ensure archive directory exists."""
        self.ARCHIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not self.ARCHIVE_FILE.exists():
            self.ARCHIVE_FILE.write_text('{"archived": [], "last_archive": null}')

    def _load_tasks(self) -> Dict[str, Any]:
        """Load current tasks with file locking."""
        if not self.TASK_FILE.exists():
            return {"tasks": [], "created_at": datetime.now(timezone.utc).isoformat()}

        if filelock:
            lock = filelock.FileLock(str(self.TASK_FILE) + ".lock", timeout=10)
            with lock:
                return json.loads(self.TASK_FILE.read_text(encoding="utf-8"))
        else:
            return json.loads(self.TASK_FILE.read_text(encoding="utf-8"))

    def _save_tasks(self, data: Dict[str, Any]):
        """Save tasks with file locking."""
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        if filelock:
            lock = filelock.FileLock(str(self.TASK_FILE) + ".lock", timeout=10)
            with lock:
                self.TASK_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            self.TASK_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _archive_tasks(self, tasks: List[Dict[str, Any]], reason: str):
        """Archive removed tasks for audit trail."""
        archive = json.loads(self.ARCHIVE_FILE.read_text(encoding="utf-8"))
        for task in tasks:
            task["archived_at"] = datetime.now(timezone.utc).isoformat()
            task["archive_reason"] = reason
            archive["archived"].append(task)
        archive["last_archive"] = datetime.now(timezone.utc).isoformat()
        self.ARCHIVE_FILE.write_text(json.dumps(archive, indent=2), encoding="utf-8")

    def _parse_timestamp(self, ts: Optional[str]) -> Optional[datetime]:
        """Parse ISO timestamp string to datetime."""
        if not ts:
            return None
        try:
            ts = ts.replace("Z", "+00:00")
            if "+" not in ts and ts.count("-") == 2:
                ts += "+00:00"
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None

    def _is_stale(self, task: Dict[str, Any]) -> bool:
        """Check if a task is stale (in-progress too long)."""
        if task.get("status") != "in-progress":
            return False

        started = self._parse_timestamp(
            task.get("picked_up_at") or task.get("started_at") or task.get("claimed_at")
        )
        if not started:
            return False

        age_hours = (datetime.now(timezone.utc) - started).total_seconds() / 3600
        return age_hours > self.STALE_HOURS

    def _is_abandoned(self, task: Dict[str, Any]) -> bool:
        """Check if a task is abandoned (pending too long with no activity)."""
        if task.get("status") != "pending":
            return False

        created = self._parse_timestamp(task.get("created_at"))
        if not created:
            return False

        age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
        return age_hours > self.ABANDONED_HOURS

    def _is_test_task(self, task: Dict[str, Any]) -> bool:
        """Check if task is a test/temporary task."""
        name = task.get("name", "") or task.get("title", "")
        desc = task.get("description", "") or ""
        payload_desc = task.get("payload", {}).get("description", "") if isinstance(task.get("payload"), dict) else ""

        full_text = f"{name} {desc} {payload_desc}".lower()
        return any(pattern.lower() in full_text for pattern in self.TEST_TASK_PATTERNS)

    def _is_deprecated(self, task: Dict[str, Any]) -> bool:
        """Check if task references deprecated code/features."""
        deprecated_refs = [
            "aurora_core", "aurora-core", "slatepi_", "001-", "002-",
            "old_", "deprecated_", "legacy_"
        ]

        file_paths = task.get("file_paths", "")
        description = task.get("description", "") or ""
        title = task.get("title", "") or task.get("name", "")

        full_text = f"{file_paths} {description} {title}".lower()
        return any(ref in full_text for ref in deprecated_refs)

    def analyze_tasks(self) -> Dict[str, Any]:
        """Analyze current task state and identify issues."""
        data = self._load_tasks()
        tasks = data.get("tasks", [])

        analysis = {
            "total": len(tasks),
            "by_status": {"pending": 0, "in-progress": 0, "completed": 0, "other": 0},
            "stale": [],
            "abandoned": [],
            "test_tasks": [],
            "deprecated": [],
            "duplicates": [],
            "healthy": [],
            "needs_attention": False,
            "can_accept_new": True,
            "recommendations": []
        }

        seen_titles = {}

        for task in tasks:
            task_id = task.get("id", "unknown")
            status = task.get("status", "other")
            title = task.get("title") or task.get("name") or task.get("payload", {}).get("description", "")

            # Count by status
            if status in analysis["by_status"]:
                analysis["by_status"][status] += 1
            else:
                analysis["by_status"]["other"] += 1

            # Check for issues
            if self._is_stale(task):
                analysis["stale"].append({"id": task_id, "title": title})
            elif self._is_abandoned(task):
                analysis["abandoned"].append({"id": task_id, "title": title})
            elif self._is_test_task(task):
                analysis["test_tasks"].append({"id": task_id, "title": title})
            elif self._is_deprecated(task):
                analysis["deprecated"].append({"id": task_id, "title": title})
            else:
                analysis["healthy"].append({"id": task_id, "title": title})

            # Check for duplicates
            if title in seen_titles:
                analysis["duplicates"].append({"id": task_id, "title": title, "duplicate_of": seen_titles[title]})
            else:
                seen_titles[title] = task_id

        # Determine if workflow needs attention
        problem_count = (
            len(analysis["stale"]) +
            len(analysis["abandoned"]) +
            len(analysis["test_tasks"]) +
            len(analysis["deprecated"]) +
            len(analysis["duplicates"])
        )

        analysis["needs_attention"] = problem_count > 0
        analysis["can_accept_new"] = (
            analysis["by_status"]["in-progress"] < self.MAX_CONCURRENT_TASKS and
            len(analysis["stale"]) == 0
        )

        # Generate recommendations
        if analysis["stale"]:
            analysis["recommendations"].append(
                f"Reset {len(analysis['stale'])} stale tasks (in-progress > {self.STALE_HOURS}h)"
            )
        if analysis["abandoned"]:
            analysis["recommendations"].append(
                f"Review {len(analysis['abandoned'])} abandoned tasks (pending > {self.ABANDONED_HOURS}h)"
            )
        if analysis["test_tasks"]:
            analysis["recommendations"].append(
                f"Remove {len(analysis['test_tasks'])} test/temporary tasks"
            )
        if analysis["deprecated"]:
            analysis["recommendations"].append(
                f"Archive {len(analysis['deprecated'])} deprecated tasks"
            )
        if analysis["duplicates"]:
            analysis["recommendations"].append(
                f"Deduplicate {len(analysis['duplicates'])} duplicate tasks"
            )
        if not analysis["can_accept_new"]:
            analysis["recommendations"].append(
                "Complete or reset existing tasks before adding new ones"
            )

        return analysis

    def cleanup(self, dry_run: bool = False) -> Dict[str, Any]:
        """Clean up stale, test, deprecated, and duplicate tasks."""
        data = self._load_tasks()
        tasks = data.get("tasks", [])

        result = {
            "dry_run": dry_run,
            "removed": [],
            "reset": [],
            "kept": [],
            "archived_count": 0
        }

        to_remove = []
        to_reset = []
        to_keep = []
        seen_titles = set()

        for task in tasks:
            task_id = task.get("id", "unknown")
            title = task.get("title") or task.get("name") or str(task.get("payload", {}).get("description", ""))

            if self._is_test_task(task):
                to_remove.append((task, "test_task"))
                result["removed"].append({"id": task_id, "reason": "test_task"})
                continue

            if self._is_deprecated(task):
                to_remove.append((task, "deprecated"))
                result["removed"].append({"id": task_id, "reason": "deprecated"})
                continue

            if title in seen_titles:
                to_remove.append((task, "duplicate"))
                result["removed"].append({"id": task_id, "reason": "duplicate"})
                continue
            seen_titles.add(title)

            if self._is_stale(task):
                to_reset.append(task)
                result["reset"].append({"id": task_id, "reason": "stale"})
                continue

            to_keep.append(task)
            result["kept"].append({"id": task_id})

        if not dry_run:
            if to_remove:
                self._archive_tasks([t[0] for t in to_remove], "workflow_cleanup")
                result["archived_count"] = len(to_remove)

            for task in to_reset:
                task["status"] = "pending"
                task["reset_at"] = datetime.now(timezone.utc).isoformat()
                task["reset_reason"] = "stale_auto_reset"
                task.pop("picked_up_at", None)
                task.pop("claimed_at", None)
                task.pop("copilot_session_id", None)
                to_keep.append(task)

            data["tasks"] = to_keep
            data["last_cleaned"] = datetime.now(timezone.utc).isoformat()
            data["clean_reason"] = f"Workflow cleanup: {len(to_remove)} removed, {len(to_reset)} reset"
            self._save_tasks(data)

        return result

    def enforce_completion(self) -> Dict[str, Any]:
        """Enforce task completion - blocks new tasks if too many are in-progress."""
        analysis = self.analyze_tasks()

        result = {
            "can_accept_new": analysis["can_accept_new"],
            "in_progress_count": analysis["by_status"]["in-progress"],
            "max_allowed": self.MAX_CONCURRENT_TASKS,
            "blocking_tasks": [],
            "action_required": None
        }

        if not analysis["can_accept_new"]:
            data = self._load_tasks()
            for task in data.get("tasks", []):
                if task.get("status") == "in-progress":
                    result["blocking_tasks"].append({
                        "id": task.get("id"),
                        "title": task.get("title") or task.get("name"),
                        "started": task.get("picked_up_at") or task.get("started_at")
                    })

            if analysis["stale"]:
                result["action_required"] = "reset_stale"
            else:
                result["action_required"] = "complete_existing"

        return result

    def print_status(self):
        """Print formatted workflow status."""
        analysis = self.analyze_tasks()

        print("\n" + "=" * 60)
        print("  SLATE Workflow Status")
        print("=" * 60)

        print(f"\nTotal Tasks: {analysis['total']}")
        print(f"  - Pending:     {analysis['by_status']['pending']}")
        print(f"  - In Progress: {analysis['by_status']['in-progress']}")
        print(f"  - Completed:   {analysis['by_status']['completed']}")

        print(f"\nCan Accept New Tasks: {'Yes' if analysis['can_accept_new'] else 'NO'}")

        if analysis["needs_attention"]:
            print("\n[!] Issues Found:")
            if analysis["stale"]:
                print(f"  - {len(analysis['stale'])} stale tasks (in-progress > {self.STALE_HOURS}h)")
            if analysis["abandoned"]:
                print(f"  - {len(analysis['abandoned'])} abandoned tasks")
            if analysis["test_tasks"]:
                print(f"  - {len(analysis['test_tasks'])} test tasks")
            if analysis["deprecated"]:
                print(f"  - {len(analysis['deprecated'])} deprecated tasks")
            if analysis["duplicates"]:
                print(f"  - {len(analysis['duplicates'])} duplicate tasks")

        if analysis["recommendations"]:
            print("\n[>] Recommendations:")
            for rec in analysis["recommendations"]:
                print(f"  - {rec}")

        print("\n" + "=" * 60)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SLATE Workflow Manager - Task lifecycle and PR workflow management"
    )
    parser.add_argument("--status", action="store_true", help="Show workflow status")
    parser.add_argument("--analyze", action="store_true", help="Analyze tasks (JSON output)")
    parser.add_argument("--cleanup", action="store_true", help="Clean up stale/test/deprecated tasks")
    parser.add_argument("--dry-run", action="store_true", help="Show what cleanup would do")
    parser.add_argument("--enforce", action="store_true", help="Check if new tasks can be accepted")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    manager = SlateWorkflowManager()

    if args.status:
        if args.json:
            print(json.dumps(manager.analyze_tasks(), indent=2))
        else:
            manager.print_status()

    elif args.analyze:
        print(json.dumps(manager.analyze_tasks(), indent=2))

    elif args.cleanup:
        result = manager.cleanup(dry_run=args.dry_run)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            action = "Would remove" if args.dry_run else "Removed"
            print(f"\n{action} {len(result['removed'])} tasks")
            print(f"{'Would reset' if args.dry_run else 'Reset'} {len(result['reset'])} stale tasks")
            print(f"Kept {len(result['kept'])} healthy tasks")
            if not args.dry_run:
                print(f"Archived {result['archived_count']} tasks")

    elif args.enforce:
        result = manager.enforce_completion()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["can_accept_new"]:
                print("[OK] Workflow ready for new tasks")
            else:
                print("[X] Cannot accept new tasks")
                print(f"   In progress: {result['in_progress_count']}/{result['max_allowed']}")
                if result["action_required"]:
                    print(f"   Action: {result['action_required']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
