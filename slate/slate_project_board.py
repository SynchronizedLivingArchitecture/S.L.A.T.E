#!/usr/bin/env python3
"""SLATE Project Board Processor - GitHub Projects V2 Integration.

Syncs between GitHub Projects and current_tasks.json, prioritizing the KANBAN
board as the primary source of truth for workflow execution.

Project Board Mapping:
    5 - KANBAN (primary workflow source)
    7 - BUG TRACKING (bugs)
    8 - ITERATIVE DEVELOPMENT (active PRs)
    10 - ROADMAP (features/enhancements)

Usage:
    python slate/slate_project_board.py --status       # Show all projects
    python slate/slate_project_board.py --sync         # Sync KANBAN to tasks
    python slate/slate_project_board.py --push         # Push tasks to KANBAN
    python slate/slate_project_board.py --process      # Process KANBAN items
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# Simple file locking for Windows
import msvcrt
import contextlib


@contextlib.contextmanager
def file_lock(filepath: Path, timeout: float = 10.0):
    """Simple file lock context manager for Windows."""
    lock_path = filepath.with_suffix(filepath.suffix + ".lock")
    lock_file = None
    try:
        lock_file = open(lock_path, "w", encoding="utf-8")
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        yield
    except OSError:
        # Lock failed, proceed anyway
        yield
    finally:
        if lock_file:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            lock_file.close()
            try:
                lock_path.unlink()
            except OSError:
                pass

# Constants
GH_CLI = r"C:\Program Files\GitHub CLI\gh.exe"
PROJECT_OWNER = "SynchronizedLivingArchitecture"
TASKS_FILE = WORKSPACE_ROOT / "current_tasks.json"

# Project IDs
PROJECTS = {
    "kanban": 5,
    "bugs": 7,
    "iterative": 8,
    "roadmap": 10,
    "planning": 4,
    "future": 6,
    "launch": 9,
    "introspection": 11,
}


def run_gh(*args: str, capture: bool = True) -> tuple[int, str]:
    """Run gh CLI command and return exit code and output."""
    cmd = [GH_CLI] + list(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=30,
            encoding="utf-8",
        )
        return result.returncode, result.stdout.strip() if capture else ""
    except subprocess.TimeoutExpired:
        return 1, "Command timed out"
    except FileNotFoundError:
        return 1, f"gh CLI not found at {GH_CLI}"


def list_projects() -> list[dict[str, Any]]:
    """List all projects for the organization."""
    code, output = run_gh(
        "project", "list", "--owner", PROJECT_OWNER, "--format", "json"
    )
    if code != 0:
        print(f"Error listing projects: {output}")
        return []
    try:
        data = json.loads(output)
        return data.get("projects", [])
    except json.JSONDecodeError:
        # Parse table format
        projects = []
        for line in output.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 3:
                projects.append({
                    "number": int(parts[0]),
                    "title": parts[1],
                    "state": parts[2],
                })
        return projects


def get_project_items(project_number: int, limit: int = 50) -> list[dict[str, Any]]:
    """Get items from a project."""
    code, output = run_gh(
        "project", "item-list", str(project_number),
        "--owner", PROJECT_OWNER,
        "--limit", str(limit),
        "--format", "json"
    )
    if code != 0:
        # Try without JSON format
        code, output = run_gh(
            "project", "item-list", str(project_number),
            "--owner", PROJECT_OWNER,
            "--limit", str(limit),
        )
        if code != 0:
            return []
        # Parse table format
        items = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                items.append({
                    "type": parts[0],
                    "title": parts[1],
                    "number": int(parts[2]) if parts[2].isdigit() else 0,
                    "repository": parts[3] if len(parts) > 3 else "",
                    "id": parts[4] if len(parts) > 4 else "",
                })
        return items
    try:
        data = json.loads(output)
        return data.get("items", [])
    except json.JSONDecodeError:
        return []


def get_project_fields(project_number: int) -> list[dict[str, str]]:
    """Get field definitions for a project."""
    code, output = run_gh(
        "project", "field-list", str(project_number),
        "--owner", PROJECT_OWNER,
    )
    if code != 0:
        return []
    fields = []
    for line in output.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            fields.append({
                "name": parts[0],
                "type": parts[1],
                "id": parts[2],
            })
    return fields


def add_item_to_project(project_number: int, title: str) -> tuple[bool, str]:
    """Add a draft item to a project."""
    code, output = run_gh(
        "project", "item-create", str(project_number),
        "--owner", PROJECT_OWNER,
        "--title", title,
    )
    return code == 0, output


def load_tasks() -> dict[str, Any]:
    """Load current_tasks.json with file locking."""
    if not TASKS_FILE.exists():
        return {"tasks": [], "created_at": datetime.now(timezone.utc).isoformat()}

    with file_lock(TASKS_FILE):
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)


def save_tasks(data: dict[str, Any]) -> None:
    """Save current_tasks.json with file locking."""
    with file_lock(TASKS_FILE):
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def sync_kanban_to_tasks() -> dict[str, int]:
    """Sync KANBAN board items to current_tasks.json."""
    stats = {"added": 0, "skipped": 0, "errors": 0}

    items = get_project_items(PROJECTS["kanban"])
    if not items:
        print("No items in KANBAN or error fetching")
        return stats

    task_data = load_tasks()
    existing_titles = {t.get("title", "") for t in task_data.get("tasks", [])}

    for item in items:
        title = item.get("title", "")
        if not title or title in existing_titles:
            stats["skipped"] += 1
            continue

        # Create task from KANBAN item
        new_task = {
            "id": f"kanban_{item.get('id', '')[:8]}",
            "title": title,
            "description": f"Synced from KANBAN project board (item type: {item.get('type', 'unknown')})",
            "priority": "medium",
            "assigned_to": "workflow",
            "source": "project_board",
            "project_item_id": item.get("id", ""),
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        task_data.setdefault("tasks", []).append(new_task)
        stats["added"] += 1
        print(f"  + Added: {title}")

    if stats["added"] > 0:
        save_tasks(task_data)
        print(f"\nSynced {stats['added']} items from KANBAN to tasks")

    return stats


def push_tasks_to_kanban() -> dict[str, int]:
    """Push pending tasks from current_tasks.json to KANBAN board."""
    stats = {"added": 0, "skipped": 0, "errors": 0}

    task_data = load_tasks()
    pending_tasks = [
        t for t in task_data.get("tasks", [])
        if t.get("status") == "pending" and t.get("source") != "project_board"
    ]

    if not pending_tasks:
        print("No pending tasks to push")
        return stats

    # Get existing KANBAN items to avoid duplicates
    kanban_items = get_project_items(PROJECTS["kanban"])
    existing_titles = {item.get("title", "") for item in kanban_items}

    for task in pending_tasks:
        title = task.get("title", "")
        if not title or title in existing_titles:
            stats["skipped"] += 1
            continue

        success, output = add_item_to_project(PROJECTS["kanban"], title)
        if success:
            stats["added"] += 1
            print(f"  + Pushed: {title}")
        else:
            stats["errors"] += 1
            print(f"  ! Error: {title} - {output}")

    print(f"\nPushed {stats['added']} tasks to KANBAN")
    return stats


def process_kanban() -> dict[str, int]:
    """Process KANBAN items by triggering workflows."""
    stats = {"processed": 0, "skipped": 0}

    items = get_project_items(PROJECTS["kanban"])
    if not items:
        print("No items in KANBAN")
        return stats

    print(f"Found {len(items)} items in KANBAN:")
    for item in items:
        item_type = item.get("type", "unknown")
        title = item.get("title", "untitled")
        number = item.get("number", "?")
        print(f"  [{item_type}] #{number}: {title}")
        stats["processed"] += 1

    return stats


def categorize_task(task: dict[str, Any]) -> str:
    """Categorize a task to determine which board it belongs to."""
    title = task.get("title", "").lower()
    desc = task.get("description", "").lower()
    source = task.get("source", "")

    # Bug-related keywords
    if any(kw in title or kw in desc for kw in ["bug", "fix", "crash", "error", "broken"]):
        return "bugs"

    # Feature/enhancement keywords
    if any(kw in title or kw in desc for kw in ["feat", "add", "new", "implement", "create", "enhancement"]):
        return "roadmap"

    # Planning keywords
    if any(kw in title or kw in desc for kw in ["plan", "design", "architect", "research"]):
        return "planning"

    # Future release keywords
    if any(kw in title or kw in desc for kw in ["future", "next", "phase", "v2", "v3"]):
        return "future"

    # Default to kanban for active work
    return "kanban"


def push_completed_to_roadmap() -> dict[str, int]:
    """Push completed feature tasks to ROADMAP board."""
    stats = {"added": 0, "skipped": 0, "errors": 0}

    task_data = load_tasks()
    completed_tasks = [
        t for t in task_data.get("tasks", [])
        if t.get("status") == "completed"
    ]

    if not completed_tasks:
        print("No completed tasks to push")
        return stats

    # Get existing ROADMAP items to avoid duplicates
    roadmap_items = get_project_items(PROJECTS["roadmap"])
    existing_titles = {item.get("title", "") for item in roadmap_items}

    for task in completed_tasks:
        title = task.get("title", "")
        if not title or title in existing_titles:
            stats["skipped"] += 1
            continue

        category = categorize_task(task)
        if category in ["roadmap", "kanban"]:  # Features go to roadmap
            success, output = add_item_to_project(PROJECTS["roadmap"], title)
            if success:
                stats["added"] += 1
                print(f"  + ROADMAP: {title[:60]}")
            else:
                stats["errors"] += 1

    print(f"\nPushed {stats['added']} completed tasks to ROADMAP")
    return stats


def push_bugs_to_tracking() -> dict[str, int]:
    """Push bug-related tasks to BUG TRACKING board."""
    stats = {"added": 0, "skipped": 0, "errors": 0}

    task_data = load_tasks()
    all_tasks = task_data.get("tasks", [])

    # Get existing BUG TRACKING items
    bug_items = get_project_items(PROJECTS["bugs"])
    existing_titles = {item.get("title", "") for item in bug_items}

    for task in all_tasks:
        title = task.get("title", "")
        if not title or title in existing_titles:
            stats["skipped"] += 1
            continue

        category = categorize_task(task)
        if category == "bugs":
            success, output = add_item_to_project(PROJECTS["bugs"], title)
            if success:
                stats["added"] += 1
                print(f"  + BUG TRACKING: {title[:60]}")
            else:
                stats["errors"] += 1

    if stats["added"] > 0:
        print(f"\nPushed {stats['added']} bug tasks to BUG TRACKING")
    return stats


def update_all_boards() -> dict[str, Any]:
    """Update all project boards based on current_tasks.json."""
    print("=" * 60)
    print("  Updating All SLATE Project Boards")
    print("=" * 60)
    print()

    results = {
        "kanban": {"added": 0, "skipped": 0},
        "roadmap": {"added": 0, "skipped": 0},
        "bugs": {"added": 0, "skipped": 0},
        "iterative": {"added": 0, "skipped": 0},
    }

    task_data = load_tasks()
    all_tasks = task_data.get("tasks", [])

    # Get existing items from all boards
    kanban_items = get_project_items(PROJECTS["kanban"])
    roadmap_items = get_project_items(PROJECTS["roadmap"])
    bug_items = get_project_items(PROJECTS["bugs"])
    iterative_items = get_project_items(PROJECTS["iterative"])

    kanban_titles = {item.get("title", "") for item in kanban_items}
    roadmap_titles = {item.get("title", "") for item in roadmap_items}
    bug_titles = {item.get("title", "") for item in bug_items}
    iterative_titles = {item.get("title", "") for item in iterative_items}

    print("Processing tasks...")
    for task in all_tasks:
        title = task.get("title", "")
        if not title:
            continue

        status = task.get("status", "pending")
        category = categorize_task(task)

        # Pending tasks go to KANBAN
        if status == "pending" and title not in kanban_titles:
            success, _ = add_item_to_project(PROJECTS["kanban"], title)
            if success:
                results["kanban"]["added"] += 1
                print(f"  + KANBAN: {title[:55]}")
            kanban_titles.add(title)

        # Completed feature tasks go to ROADMAP
        if status == "completed" and category in ["roadmap", "kanban"]:
            if title not in roadmap_titles:
                success, _ = add_item_to_project(PROJECTS["roadmap"], title)
                if success:
                    results["roadmap"]["added"] += 1
                    print(f"  + ROADMAP: {title[:55]}")
                roadmap_titles.add(title)

        # Bug tasks go to BUG TRACKING
        if category == "bugs" and title not in bug_titles:
            success, _ = add_item_to_project(PROJECTS["bugs"], title)
            if success:
                results["bugs"]["added"] += 1
                print(f"  + BUG TRACKING: {title[:55]}")
            bug_titles.add(title)

    print()
    print("-" * 60)
    print("  Summary")
    print("-" * 60)
    total = sum(r["added"] for r in results.values())
    print(f"  KANBAN:       +{results['kanban']['added']} items")
    print(f"  ROADMAP:      +{results['roadmap']['added']} items")
    print(f"  BUG TRACKING: +{results['bugs']['added']} items")
    print(f"  Total:        +{total} items added")

    return results


def print_status() -> None:
    """Print status of all project boards."""
    print("=" * 60)
    print("  SLATE Project Boards Status")
    print("=" * 60)
    print()

    # List all projects
    projects = list_projects()
    if not projects:
        # Fallback to direct query
        for name, number in sorted(PROJECTS.items(), key=lambda x: x[1]):
            items = get_project_items(number)
            print(f"  {number:2d}. {name.upper():20s} [{len(items)} items]")
    else:
        for proj in projects:
            num = proj.get("number", 0)
            title = proj.get("title", "untitled")
            state = proj.get("state", "unknown")
            items = get_project_items(num)
            print(f"  {num:2d}. {title:35s} [{len(items):2d} items] ({state})")

    print()
    print("-" * 60)
    print("  KANBAN Board (Primary Workflow Source)")
    print("-" * 60)

    items = get_project_items(PROJECTS["kanban"])
    if items:
        for item in items:
            item_type = item.get("type", "unknown")
            title = item.get("title", "untitled")[:50]
            print(f"  [{item_type:11s}] {title}")
    else:
        print("  (no items)")

    print()
    print("-" * 60)
    print("  Local Task Queue (current_tasks.json)")
    print("-" * 60)

    task_data = load_tasks()
    tasks = task_data.get("tasks", [])
    pending = [t for t in tasks if t.get("status") == "pending"]
    completed = [t for t in tasks if t.get("status") == "completed"]

    print(f"  Total: {len(tasks)} | Pending: {len(pending)} | Completed: {len(completed)}")

    if pending:
        print()
        print("  Pending tasks:")
        for task in pending[:5]:
            title = task.get("title", "untitled")[:50]
            priority = task.get("priority", "medium")
            print(f"    [{priority:6s}] {title}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SLATE Project Board Processor"
    )
    parser.add_argument("--status", action="store_true", help="Show project status")
    parser.add_argument("--sync", action="store_true", help="Sync KANBAN to tasks")
    parser.add_argument("--push", action="store_true", help="Push tasks to KANBAN")
    parser.add_argument("--process", action="store_true", help="Process KANBAN items")
    parser.add_argument("--update-all", action="store_true", help="Update all boards")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.update_all:
        results = update_all_boards()
        if args.json:
            print(json.dumps(results))
        return 0

    if args.sync:
        stats = sync_kanban_to_tasks()
        if args.json:
            print(json.dumps(stats))
        return 0

    if args.push:
        stats = push_tasks_to_kanban()
        if args.json:
            print(json.dumps(stats))
        return 0

    if args.process:
        stats = process_kanban()
        if args.json:
            print(json.dumps(stats))
        return 0

    # Default: show status
    print_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
