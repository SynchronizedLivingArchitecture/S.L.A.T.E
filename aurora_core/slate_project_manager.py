#!/usr/bin/env python3
"""
SLATE Project Manager - GitHub Projects Integration
Author: COPILOT | Created: 2026-02-06T00:00:00Z

Integrates SLATE task system with GitHub Projects for planning and tracking.
Syncs tasks between current_tasks.json and GitHub Projects.

Usage:
    python aurora_core/slate_project_manager.py --status
    python aurora_core/slate_project_manager.py --sync
    python aurora_core/slate_project_manager.py --create-project "SLATE Development"
    python aurora_core/slate_project_manager.py --list-projects
"""

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

logger = logging.getLogger("aurora.project_manager")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL: constants
# Author: COPILOT | Created: 2026-02-06T00:00:00Z
# ═══════════════════════════════════════════════════════════════════════════════

GH_PATH = WORKSPACE_ROOT / ".tools" / "gh.exe"
OWNER = "SynchronizedLivingArchitecture"
REPO = "S.L.A.T.E."
TASKS_FILE = WORKSPACE_ROOT / "current_tasks.json"
PROJECT_CONFIG_FILE = WORKSPACE_ROOT / ".github" / "projects.json"

# Project templates
PROJECT_TEMPLATES = {
    "development": {
        "title": "SLATE Development",
        "description": "Active development tasks and features for S.L.A.T.E.",
        "fields": [
            {"name": "Status", "type": "single_select", "options": [
                "Backlog", "Ready", "In Progress", "In Review", "Done"
            ]},
            {"name": "Priority", "type": "single_select", "options": [
                "Critical", "High", "Medium", "Low"
            ]},
            {"name": "Agent", "type": "single_select", "options": [
                "ALPHA", "BETA", "GAMMA", "DELTA", "COPILOT", "Auto"
            ]},
            {"name": "Sprint", "type": "iteration"},
            {"name": "Effort", "type": "number"},
            {"name": "Due Date", "type": "date"},
        ],
        "views": [
            {"name": "Board", "layout": "board", "group_by": "Status"},
            {"name": "By Agent", "layout": "board", "group_by": "Agent"},
            {"name": "Backlog", "layout": "table", "filter": "status:Backlog"},
            {"name": "Roadmap", "layout": "roadmap", "date_field": "Due Date"},
        ]
    },
    "roadmap": {
        "title": "SLATE Roadmap",
        "description": "Long-term planning and milestones for S.L.A.T.E.",
        "fields": [
            {"name": "Status", "type": "single_select", "options": [
                "Planning", "In Progress", "Complete", "Deferred"
            ]},
            {"name": "Quarter", "type": "single_select", "options": [
                "Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"
            ]},
            {"name": "Category", "type": "single_select", "options": [
                "Feature", "Infrastructure", "Performance", "Security", "Documentation"
            ]},
            {"name": "Target Version", "type": "text"},
            {"name": "Start Date", "type": "date"},
            {"name": "End Date", "type": "date"},
        ],
        "views": [
            {"name": "Timeline", "layout": "roadmap", "date_field": "Start Date"},
            {"name": "By Quarter", "layout": "board", "group_by": "Quarter"},
            {"name": "By Category", "layout": "table", "group_by": "Category"},
        ]
    },
    "sprint": {
        "title": "SLATE Sprint",
        "description": "Current sprint planning and tracking",
        "fields": [
            {"name": "Status", "type": "single_select", "options": [
                "To Do", "In Progress", "Review", "Done"
            ]},
            {"name": "Story Points", "type": "number"},
            {"name": "Assignee", "type": "single_select", "options": [
                "ALPHA", "BETA", "GAMMA", "DELTA"
            ]},
            {"name": "Sprint Goal", "type": "text"},
        ],
        "views": [
            {"name": "Sprint Board", "layout": "board", "group_by": "Status"},
            {"name": "Burndown", "layout": "table"},
        ]
    }
}

# Task status mapping from SLATE to GitHub Projects
STATUS_MAPPING = {
    "pending": "Backlog",
    "ready": "Ready",
    "in_progress": "In Progress",
    "review": "In Review",
    "completed": "Done",
    "blocked": "Backlog",
    "failed": "Backlog",
}

# Priority mapping
PRIORITY_MAPPING = {
    "urgent": "Critical",
    "critical": "Critical",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
}

# Agent mapping
AGENT_MAPPING = {
    "ALPHA": "ALPHA",
    "BETA": "BETA",
    "GAMMA": "GAMMA",
    "DELTA": "DELTA",
    "COPILOT": "COPILOT",
    "auto": "Auto",
}


# ═══════════════════════════════════════════════════════════════════════════════
# CELL: data_classes
# Author: COPILOT | Created: 2026-02-06T00:00:00Z
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProjectConfig:
    """Configuration for a GitHub Project."""
    id: Optional[str] = None
    number: Optional[int] = None
    title: str = ""
    description: str = ""
    template: str = "development"
    url: Optional[str] = None
    created_at: Optional[str] = None
    synced_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "number": self.number,
            "title": self.title,
            "description": self.description,
            "template": self.template,
            "url": self.url,
            "created_at": self.created_at,
            "synced_at": self.synced_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProjectState:
    """State tracking for project operations."""
    projects: List[ProjectConfig] = field(default_factory=list)
    last_sync: Optional[str] = None
    sync_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "projects": [p.to_dict() for p in self.projects],
            "last_sync": self.last_sync,
            "sync_errors": self.sync_errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectState":
        state = cls()
        state.projects = [ProjectConfig.from_dict(p) for p in data.get("projects", [])]
        state.last_sync = data.get("last_sync")
        state.sync_errors = data.get("sync_errors", [])
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# CELL: project_manager
# Author: COPILOT | Created: 2026-02-06T00:00:00Z
# ═══════════════════════════════════════════════════════════════════════════════

class SlateProjectManager:
    """
    Manages GitHub Projects integration for SLATE.

    Features:
    - Create and configure GitHub Projects
    - Sync tasks between current_tasks.json and Projects
    - Manage project fields, views, and automation
    - Track project state and sync history
    """

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or WORKSPACE_ROOT
        self.gh_path = self.workspace / ".tools" / "gh.exe"
        self.config_file = self.workspace / ".github" / "projects.json"
        self.tasks_file = self.workspace / "current_tasks.json"
        self.state = self._load_state()

    def _load_state(self) -> ProjectState:
        """Load project state from config file."""
        if self.config_file.exists():
            try:
                data = json.loads(self.config_file.read_text())
                return ProjectState.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load project state: {e}")
        return ProjectState()

    def _save_state(self) -> None:
        """Save project state to config file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(json.dumps(self.state.to_dict(), indent=2))

    def _run_gh(self, args: List[str], capture: bool = True) -> subprocess.CompletedProcess:
        """Run GitHub CLI command."""
        cmd = [str(self.gh_path)] + args
        return subprocess.run(cmd, capture_output=capture, text=True)

    def _run_gh_api(self, endpoint: str, method: str = "GET",
                    data: Optional[Dict] = None) -> Dict[str, Any]:
        """Run GitHub API command."""
        args = ["api", endpoint]
        if method != "GET":
            args.extend(["--method", method])
        if data:
            for key, value in data.items():
                args.extend(["-f", f"{key}={value}"])

        result = self._run_gh(args)
        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"raw": result.stdout}
        return {"error": result.stderr}

    def check_auth_scopes(self) -> Dict[str, Any]:
        """Check if required auth scopes are available."""
        result = self._run_gh(["auth", "status"])
        output = result.stdout + result.stderr

        required_scopes = ["read:project", "project"]
        has_scopes = {scope: scope in output for scope in required_scopes}

        return {
            "authenticated": result.returncode == 0,
            "scopes": has_scopes,
            "all_scopes_present": all(has_scopes.values()),
            "missing_scopes": [s for s, v in has_scopes.items() if not v],
            "command_to_fix": "gh auth refresh -s read:project -s project" if not all(has_scopes.values()) else None,
        }

    def list_projects(self) -> Dict[str, Any]:
        """List all GitHub Projects for the organization."""
        result = self._run_gh([
            "project", "list",
            "--owner", OWNER,
            "--format", "json"
        ])

        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr,
                "projects": [],
            }

        try:
            projects = json.loads(result.stdout)
            return {
                "success": True,
                "projects": projects.get("projects", []),
                "total": len(projects.get("projects", [])),
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Failed to parse projects JSON",
                "projects": [],
            }

    def create_project(self, template: str = "development",
                       title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new GitHub Project from template.

        Args:
            template: Template name (development, roadmap, sprint)
            title: Optional custom title

        Returns:
            Result dict with project details
        """
        if template not in PROJECT_TEMPLATES:
            return {
                "success": False,
                "error": f"Unknown template: {template}. Available: {list(PROJECT_TEMPLATES.keys())}",
            }

        tmpl = PROJECT_TEMPLATES[template]
        project_title = title or tmpl["title"]

        # Create project
        result = self._run_gh([
            "project", "create",
            "--owner", OWNER,
            "--title", project_title,
            "--format", "json"
        ])

        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr,
            }

        try:
            project_data = json.loads(result.stdout)
            project_number = project_data.get("number")
            project_url = project_data.get("url")

            # Save to state
            config = ProjectConfig(
                id=project_data.get("id"),
                number=project_number,
                title=project_title,
                description=tmpl["description"],
                template=template,
                url=project_url,
                created_at=datetime.utcnow().isoformat(),
            )
            self.state.projects.append(config)
            self._save_state()

            return {
                "success": True,
                "project": config.to_dict(),
                "next_steps": [
                    f"Add fields: gh project field-create {project_number} --owner {OWNER} --name 'Status' --data-type SINGLE_SELECT",
                    f"View project: {project_url}",
                ],
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Failed to parse project creation response",
            }

    def get_project_fields(self, project_number: int) -> Dict[str, Any]:
        """Get fields for a project."""
        result = self._run_gh([
            "project", "field-list",
            str(project_number),
            "--owner", OWNER,
            "--format", "json"
        ])

        if result.returncode != 0:
            return {"success": False, "error": result.stderr, "fields": []}

        try:
            fields = json.loads(result.stdout)
            return {"success": True, "fields": fields.get("fields", [])}
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse fields", "fields": []}

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from current_tasks.json."""
        if not self.tasks_file.exists():
            return []

        try:
            data = json.loads(self.tasks_file.read_text())
            return data.get("tasks", [])
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")
            return []

    def sync_tasks_to_project(self, project_number: int) -> Dict[str, Any]:
        """
        Sync tasks from current_tasks.json to a GitHub Project.

        Args:
            project_number: The project number to sync to

        Returns:
            Sync results
        """
        tasks = self.load_tasks()
        if not tasks:
            return {
                "success": True,
                "message": "No tasks to sync",
                "synced": 0,
                "errors": [],
            }

        results = {
            "success": True,
            "synced": 0,
            "skipped": 0,
            "errors": [],
        }

        for task in tasks:
            task_id = task.get("id", "unknown")
            title = task.get("title", "Untitled")

            # Create issue for the task
            issue_result = self._run_gh([
                "issue", "create",
                "--repo", f"{OWNER}/{REPO}",
                "--title", f"[SLATE] {title}",
                "--body", self._format_task_body(task),
                "--label", "slate-task",
            ])

            if issue_result.returncode != 0:
                results["errors"].append(f"Task {task_id}: {issue_result.stderr}")
                continue

            # Extract issue number and add to project
            try:
                issue_url = issue_result.stdout.strip()
                issue_number = issue_url.split("/")[-1]

                # Add to project
                add_result = self._run_gh([
                    "project", "item-add",
                    str(project_number),
                    "--owner", OWNER,
                    "--url", issue_url,
                ])

                if add_result.returncode == 0:
                    results["synced"] += 1
                else:
                    results["errors"].append(f"Task {task_id}: Failed to add to project")
            except Exception as e:
                results["errors"].append(f"Task {task_id}: {str(e)}")

        # Update sync timestamp
        self.state.last_sync = datetime.utcnow().isoformat()
        self.state.sync_errors = results["errors"]
        self._save_state()

        return results

    def _format_task_body(self, task: Dict[str, Any]) -> str:
        """Format task as issue body."""
        body = f"""## Task Details

| Property | Value |
|----------|-------|
| ID | `{task.get('id', 'N/A')}` |
| Priority | {task.get('priority', 'medium')} |
| Assigned To | {task.get('assigned_to', 'auto')} |
| Status | {task.get('status', 'pending')} |
| Created | {task.get('created_at', 'N/A')} |

## Description

{task.get('description', 'No description provided.')}

---
*Synced from SLATE task queue*
"""
        return body

    def get_status(self) -> Dict[str, Any]:
        """Get current project manager status."""
        auth = self.check_auth_scopes()
        tasks = self.load_tasks()

        return {
            "auth": auth,
            "projects": [p.to_dict() for p in self.state.projects],
            "tasks": {
                "total": len(tasks),
                "by_status": self._count_by_field(tasks, "status"),
                "by_agent": self._count_by_field(tasks, "assigned_to"),
            },
            "last_sync": self.state.last_sync,
            "sync_errors": self.state.sync_errors,
        }

    def _count_by_field(self, items: List[Dict], field: str) -> Dict[str, int]:
        """Count items by field value."""
        counts: Dict[str, int] = {}
        for item in items:
            value = str(item.get(field, "unknown"))
            counts[value] = counts.get(value, 0) + 1
        return counts


# ═══════════════════════════════════════════════════════════════════════════════
# CELL: cli
# Author: COPILOT | Created: 2026-02-06T00:00:00Z
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SLATE Project Manager - GitHub Projects Integration",
        epilog="""
Examples:
  # Check status and auth
  python slate_project_manager.py --status

  # List existing projects
  python slate_project_manager.py --list-projects

  # Create a new development project
  python slate_project_manager.py --create-project development

  # Sync tasks to a project
  python slate_project_manager.py --sync --project 1

  # Check auth scopes
  python slate_project_manager.py --check-auth
"""
    )

    parser.add_argument("--status", action="store_true", help="Show project manager status")
    parser.add_argument("--list-projects", action="store_true", help="List GitHub Projects")
    parser.add_argument("--create-project", type=str, metavar="TEMPLATE",
                        help="Create project from template (development, roadmap, sprint)")
    parser.add_argument("--title", type=str, help="Custom project title")
    parser.add_argument("--sync", action="store_true", help="Sync tasks to project")
    parser.add_argument("--project", type=int, help="Project number for sync")
    parser.add_argument("--check-auth", action="store_true", help="Check auth scopes")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    args = parser.parse_args()
    manager = SlateProjectManager()

    if args.check_auth:
        result = manager.check_auth_scopes()
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("\n[SLATE Project Manager] Auth Check")
            print("=" * 50)
            print(f"  Authenticated: {'Yes' if result['authenticated'] else 'No'}")
            print(f"  All Scopes:    {'Yes' if result['all_scopes_present'] else 'No'}")
            if result['missing_scopes']:
                print(f"  Missing:       {', '.join(result['missing_scopes'])}")
                print(f"\n  Fix: {result['command_to_fix']}")

    elif args.status:
        result = manager.get_status()
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("\n[SLATE Project Manager] Status")
            print("=" * 50)
            print(f"  Auth OK:       {'Yes' if result['auth']['all_scopes_present'] else 'No'}")
            print(f"  Projects:      {len(result['projects'])}")
            print(f"  Tasks:         {result['tasks']['total']}")
            print(f"  Last Sync:     {result['last_sync'] or 'Never'}")

            if result['projects']:
                print("\n  Configured Projects:")
                for p in result['projects']:
                    print(f"    - {p['title']} (#{p['number']})")

            if not result['auth']['all_scopes_present']:
                print(f"\n  Run: {result['auth']['command_to_fix']}")

    elif args.list_projects:
        result = manager.list_projects()
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            if result['success']:
                print(f"\n[SLATE] GitHub Projects ({result['total']})")
                print("=" * 50)
                for p in result['projects']:
                    print(f"  #{p.get('number', '?')}: {p.get('title', 'Untitled')}")
                    if p.get('url'):
                        print(f"        {p['url']}")
            else:
                print(f"Error: {result['error']}")

    elif args.create_project:
        result = manager.create_project(args.create_project, args.title)
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            if result['success']:
                print("\n[SLATE] Project Created")
                print("=" * 50)
                print(f"  Title:  {result['project']['title']}")
                print(f"  Number: #{result['project']['number']}")
                print(f"  URL:    {result['project']['url']}")
                print("\n  Next Steps:")
                for step in result.get('next_steps', []):
                    print(f"    - {step}")
            else:
                print(f"Error: {result['error']}")

    elif args.sync:
        if not args.project:
            print("Error: --project NUMBER required for sync")
            sys.exit(1)
        result = manager.sync_tasks_to_project(args.project)
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("\n[SLATE] Task Sync Results")
            print("=" * 50)
            print(f"  Synced:  {result['synced']}")
            print(f"  Errors:  {len(result['errors'])}")
            if result['errors']:
                for err in result['errors'][:5]:
                    print(f"    - {err}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
