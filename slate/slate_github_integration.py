#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_github_integration [python]
# Author: Claude | Created: 2026-02-06T23:00:00Z
# Purpose: Full SLATE GitHub integration with Actions, Runners, and Workflows
# ═══════════════════════════════════════════════════════════════════════════════
"""
SLATE GitHub Integration
=========================

Unified integration between SLATE and GitHub:
- Self-hosted runner management
- Workflow triggering and monitoring
- GitHub API interactions
- CI/CD status tracking

Usage:
    python slate/slate_github_integration.py --status
    python slate/slate_github_integration.py --trigger-workflow gpu-tests
    python slate/slate_github_integration.py --runner-start
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup path
WORKSPACE_ROOT = Path(__file__).parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

logger = logging.getLogger("slate.github_integration")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL: constants [python]
# ═══════════════════════════════════════════════════════════════════════════════

SLATE_REPO = "SynchronizedLivingArchitecture/S.L.A.T.E."

# Modified: 2026-02-07T04:00:00Z | Author: COPILOT | Change: Use WORKSPACE_ROOT for runner dir
DEFAULT_RUNNER_DIR = WORKSPACE_ROOT / "actions-runner"


@dataclass
class GitHubStatus:
    """GitHub integration status."""
    cli_installed: bool = False
    authenticated: bool = False
    username: Optional[str] = None
    token_scopes: List[str] = field(default_factory=list)


@dataclass
class RunnerStatus:
    """Self-hosted runner status."""
    installed: bool = False
    configured: bool = False
    running: bool = False
    name: Optional[str] = None
    repo_url: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    agent_id: Optional[int] = None


@dataclass
class WorkflowRun:
    """GitHub Actions workflow run."""
    id: int
    name: str
    status: str
    conclusion: Optional[str]
    created_at: str
    head_branch: str
    html_url: str


class SlateGitHubIntegration:
    """
    SLATE GitHub Integration Manager.

    Provides unified interface for:
    - GitHub CLI operations
    - Self-hosted runner management
    - Workflow triggering and monitoring
    - Repository management
    """

    def __init__(self):
        """Initialize integration."""
        self.repo = SLATE_REPO
        self.runner_dir = DEFAULT_RUNNER_DIR

    # ═══════════════════════════════════════════════════════════════════════════
    # GitHub CLI Operations
    # ═══════════════════════════════════════════════════════════════════════════

    def get_github_status(self) -> GitHubStatus:
        """Get GitHub CLI and authentication status."""
        status = GitHubStatus()

        try:
            # Check if gh is installed
            result = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            status.cli_installed = result.returncode == 0

            if not status.cli_installed:
                return status

            # Check authentication
            auth_result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10
            )
            status.authenticated = auth_result.returncode == 0

            if status.authenticated:
                # Get username
                user_result = subprocess.run(
                    ["gh", "api", "user", "-q", ".login"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if user_result.returncode == 0:
                    status.username = user_result.stdout.strip()

                # Check for workflow scope
                if "workflow" in auth_result.stdout.lower():
                    status.token_scopes.append("workflow")
                if "repo" in auth_result.stdout.lower():
                    status.token_scopes.append("repo")

        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Error checking GitHub status: {e}")

        return status

    def run_gh_command(self, args: List[str], timeout: int = 30) -> Dict[str, Any]:
        """
        Run a GitHub CLI command.

        Args:
            args: Command arguments (without 'gh' prefix)
            timeout: Command timeout in seconds

        Returns:
            Dict with success, stdout, stderr
        """
        try:
            result = subprocess.run(
                ["gh"] + args,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except FileNotFoundError:
            return {"success": False, "error": "GitHub CLI not installed"}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # Runner Management
    # ═══════════════════════════════════════════════════════════════════════════

    def get_runner_status(self) -> RunnerStatus:
        """Get self-hosted runner status."""
        status = RunnerStatus()

        # Check if installed
        config_cmd = self.runner_dir / "config.cmd" if os.name == "nt" else self.runner_dir / "config.sh"
        status.installed = config_cmd.exists()

        if not status.installed:
            return status

        # Check configuration
        runner_file = self.runner_dir / ".runner"
        status.configured = runner_file.exists()

        if status.configured:
            try:
                config = json.loads(runner_file.read_text())
                status.name = config.get("agentName")
                status.repo_url = config.get("gitHubUrl")
                status.agent_id = config.get("agentId")
            except Exception:
                pass

        # Load SLATE config for labels
        slate_config = self.runner_dir / ".slate_runner_config.json"
        if slate_config.exists():
            try:
                config = json.loads(slate_config.read_text())
                status.labels = config.get("labels", [])
            except Exception:
                pass

        # Check if running
        try:
            import psutil
            for proc in psutil.process_iter(['name']):
                if 'Runner.Listener' in proc.info.get('name', ''):
                    status.running = True
                    break
        except Exception:
            pass

        return status

    def start_runner(self, background: bool = True) -> Dict[str, Any]:
        """
        Start the self-hosted runner.

        Args:
            background: Run in background

        Returns:
            Result dict
        """
        status = self.get_runner_status()

        if not status.installed:
            return {"success": False, "error": "Runner not installed"}
        if not status.configured:
            return {"success": False, "error": "Runner not configured"}
        if status.running:
            return {"success": True, "message": "Runner already running"}

        run_cmd = self.runner_dir / ("run.cmd" if os.name == "nt" else "run.sh")

        try:
            if background:
                if os.name == "nt":
                    subprocess.Popen(
                        ["cmd", "/c", str(run_cmd)],
                        cwd=str(self.runner_dir),
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                else:
                    subprocess.Popen(
                        [str(run_cmd)],
                        cwd=str(self.runner_dir),
                        start_new_session=True
                    )
                return {"success": True, "message": "Runner started in background"}
            else:
                subprocess.run(
                    [str(run_cmd)],
                    cwd=str(self.runner_dir)
                )
                return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_runner(self) -> Dict[str, Any]:
        """Stop the self-hosted runner."""
        try:
            import psutil
            for proc in psutil.process_iter(['name', 'pid']):
                if 'Runner.Listener' in proc.info.get('name', ''):
                    proc.terminate()
                    return {"success": True, "message": f"Stopped runner (PID {proc.pid})"}
            return {"success": False, "error": "Runner not running"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # Workflow Operations
    # ═══════════════════════════════════════════════════════════════════════════

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List available workflows."""
        result = self.run_gh_command([
            "workflow", "list",
            "--repo", self.repo,
            "--json", "name,id,state"
        ])

        if result["success"]:
            try:
                return json.loads(result["stdout"])
            except json.JSONDecodeError:
                return []
        return []

    def trigger_workflow(
        self,
        workflow: str,
        ref: str = "main",
        inputs: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Trigger a workflow dispatch.

        Args:
            workflow: Workflow name or filename
            ref: Git ref (branch/tag)
            inputs: Workflow inputs

        Returns:
            Result dict
        """
        args = [
            "workflow", "run", workflow,
            "--repo", self.repo,
            "--ref", ref
        ]

        if inputs:
            for key, value in inputs.items():
                args.extend(["-f", f"{key}={value}"])

        result = self.run_gh_command(args)

        if result["success"]:
            return {"success": True, "message": f"Triggered workflow: {workflow}"}
        return {"success": False, "error": result.get("stderr", "Unknown error")}

    def get_workflow_runs(
        self,
        workflow: Optional[str] = None,
        limit: int = 10
    ) -> List[WorkflowRun]:
        """
        Get recent workflow runs.

        Args:
            workflow: Filter by workflow name
            limit: Max runs to return

        Returns:
            List of WorkflowRun objects
        """
        args = [
            "run", "list",
            "--repo", self.repo,
            "--limit", str(limit),
            "--json", "databaseId,displayTitle,status,conclusion,createdAt,headBranch,url"
        ]

        if workflow:
            args.extend(["--workflow", workflow])

        result = self.run_gh_command(args)

        if result["success"]:
            try:
                runs_data = json.loads(result["stdout"])
                return [
                    WorkflowRun(
                        id=r["databaseId"],
                        name=r["displayTitle"],
                        status=r["status"],
                        conclusion=r.get("conclusion"),
                        created_at=r["createdAt"],
                        head_branch=r["headBranch"],
                        html_url=r["url"]
                    )
                    for r in runs_data
                ]
            except (json.JSONDecodeError, KeyError):
                pass
        return []

    def get_run_status(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow run."""
        result = self.run_gh_command([
            "run", "view", str(run_id),
            "--repo", self.repo,
            "--json", "status,conclusion,jobs"
        ])

        if result["success"]:
            try:
                return json.loads(result["stdout"])
            except json.JSONDecodeError:
                pass
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # Repository Operations
    # ═══════════════════════════════════════════════════════════════════════════

    def get_repo_info(self) -> Dict[str, Any]:
        """Get repository information."""
        result = self.run_gh_command([
            "repo", "view", self.repo,
            "--json", "name,owner,defaultBranchRef,isPrivate,url"
        ])

        if result["success"]:
            try:
                return json.loads(result["stdout"])
            except json.JSONDecodeError:
                pass
        return {}

    def get_open_prs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get open pull requests."""
        result = self.run_gh_command([
            "pr", "list",
            "--repo", self.repo,
            "--limit", str(limit),
            "--json", "number,title,state,author,headRefName,url"
        ])

        if result["success"]:
            try:
                return json.loads(result["stdout"])
            except json.JSONDecodeError:
                pass
        return []

    # ═══════════════════════════════════════════════════════════════════════════
    # Full Status
    # ═══════════════════════════════════════════════════════════════════════════

    def get_full_status(self) -> Dict[str, Any]:
        """Get complete integration status."""
        github = self.get_github_status()
        runner = self.get_runner_status()

        status = {
            "timestamp": datetime.now().isoformat(),
            "repository": self.repo,
            "github": {
                "cli_installed": github.cli_installed,
                "authenticated": github.authenticated,
                "username": github.username,
                "scopes": github.token_scopes
            },
            "runner": {
                "installed": runner.installed,
                "configured": runner.configured,
                "running": runner.running,
                "name": runner.name,
                "repo_url": runner.repo_url,
                "labels": runner.labels,
                "agent_id": runner.agent_id
            }
        }

        # Add workflow info if authenticated
        if github.authenticated:
            runs = self.get_workflow_runs(limit=5)
            status["recent_runs"] = [
                {
                    "id": r.id,
                    "name": r.name,
                    "status": r.status,
                    "conclusion": r.conclusion,
                    "branch": r.head_branch
                }
                for r in runs
            ]

        return status


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════════════════

def print_status(status: Dict[str, Any]):
    """Print status in human-readable format."""
    print()
    print("=" * 60)
    print("  SLATE GitHub Integration Status")
    print("=" * 60)
    print()

    # GitHub
    gh = status["github"]
    if gh["authenticated"]:
        print(f"  GitHub:     [OK] {gh['username']}")
        if gh["scopes"]:
            print(f"              Scopes: {', '.join(gh['scopes'])}")
    elif gh["cli_installed"]:
        print("  GitHub:     [--] CLI installed, not authenticated")
    else:
        print("  GitHub:     [!!] CLI not installed")

    # Runner
    runner = status["runner"]
    if runner["running"]:
        print(f"  Runner:     [OK] {runner['name']} (listening)")
        print(f"              Labels: {', '.join(runner['labels'][:5])}")
    elif runner["configured"]:
        print(f"  Runner:     [--] {runner['name']} (stopped)")
    elif runner["installed"]:
        print("  Runner:     [--] Installed but not configured")
    else:
        print("  Runner:     [!!] Not installed")

    # Recent runs
    if "recent_runs" in status:
        print()
        print("  Recent Workflow Runs:")
        for run in status["recent_runs"][:5]:
            icon = "[OK]" if run["conclusion"] == "success" else "[!!]" if run["conclusion"] else "[..]"
            print(f"    {icon} {run['name'][:40]} ({run['status']})")

    print()
    print("=" * 60)
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SLATE GitHub Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check integration status
  python slate_github_integration.py --status

  # Start the runner
  python slate_github_integration.py --runner-start

  # Trigger a workflow
  python slate_github_integration.py --trigger-workflow gpu-tests

  # List recent workflow runs
  python slate_github_integration.py --list-runs
"""
    )

    parser.add_argument("--status", action="store_true", help="Show integration status")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    # Repository is always SLATE main (BETA deprecated)
    parser.add_argument("--runner-start", action="store_true", help="Start the runner")
    parser.add_argument("--runner-stop", action="store_true", help="Stop the runner")
    parser.add_argument("--trigger-workflow", type=str, metavar="NAME", help="Trigger workflow dispatch")
    parser.add_argument("--list-runs", action="store_true", help="List recent workflow runs")
    parser.add_argument("--list-workflows", action="store_true", help="List available workflows")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    integration = SlateGitHubIntegration()

    if args.status:
        status = integration.get_full_status()
        if args.json_output:
            print(json.dumps(status, indent=2))
        else:
            print_status(status)

    elif args.runner_start:
        result = integration.start_runner(background=True)
        if result["success"]:
            print(f"[OK] {result.get('message', 'Runner started')}")
        else:
            print(f"[ERROR] {result.get('error', 'Failed to start runner')}")
            sys.exit(1)

    elif args.runner_stop:
        result = integration.stop_runner()
        if result["success"]:
            print(f"[OK] {result.get('message', 'Runner stopped')}")
        else:
            print(f"[ERROR] {result.get('error', 'Failed to stop runner')}")
            sys.exit(1)

    elif args.trigger_workflow:
        result = integration.trigger_workflow(args.trigger_workflow)
        if result["success"]:
            print(f"[OK] {result.get('message')}")
        else:
            print(f"[ERROR] {result.get('error')}")
            sys.exit(1)

    elif args.list_runs:
        runs = integration.get_workflow_runs(limit=10)
        if args.json_output:
            print(json.dumps([
                {"id": r.id, "name": r.name, "status": r.status, "conclusion": r.conclusion}
                for r in runs
            ], indent=2))
        else:
            print()
            print("Recent Workflow Runs:")
            print("-" * 60)
            for run in runs:
                icon = "[OK]" if run.conclusion == "success" else "[!!]" if run.conclusion else "[..]"
                print(f"  {icon} #{run.id} {run.name[:45]}")
                print(f"      Status: {run.status} | Branch: {run.head_branch}")
            print()

    elif args.list_workflows:
        workflows = integration.list_workflows()
        if args.json_output:
            print(json.dumps(workflows, indent=2))
        else:
            print()
            print("Available Workflows:")
            print("-" * 40)
            for wf in workflows:
                state_icon = "[OK]" if wf.get("state") == "active" else "[--]"
                print(f"  {state_icon} {wf.get('name')}")
            print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
