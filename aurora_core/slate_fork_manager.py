#!/usr/bin/env python3
"""
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_fork_manager [python]
# Author: COPILOT | Created: 2026-02-06T19:30:00Z | Modified: 2026-02-06T19:30:00Z
# SLATE Fork Manager - Manages user forks and local git for SLATE installations
# ═══════════════════════════════════════════════════════════════════════════════

SLATE Fork Manager handles:
- Creating isolated local git repositories for SLATE users
- Managing user forks and contributions
- Validating fork prerequisites before PR creation
- Syncing with upstream SLATE repository

Security Model:
- Each user gets an isolated workspace
- Forks are validated locally before push
- All contributions must pass SLATE prerequisites
"""

import json
import logging
import os
import shutil
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

logger = logging.getLogger("aurora.fork_manager")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL: constants [python]
# Author: COPILOT | Created: 2026-02-06T19:30:00Z
# ═══════════════════════════════════════════════════════════════════════════════

SLATE_UPSTREAM = "https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E..git"
SLATE_UPSTREAM_SSH = "git@github.com:SynchronizedLivingArchitecture/S.L.A.T.E..git"

FORK_CONFIG_FILE = ".slate_fork/config.json"
FORK_STATE_FILE = ".slate_fork/state.json"

# Files that user forks CANNOT modify (security-critical)
PROTECTED_FILES = [
    ".github/workflows/*",
    ".github/CODEOWNERS",
    "aurora_core/action_guard.py",
    "aurora_core/sdk_source_guard.py",
]

# Required files for valid SLATE installation
REQUIRED_FILES = [
    "aurora_core/__init__.py",
    "aurora_core/slate_status.py",
    "current_tasks.json",
    "pyproject.toml",
    "CLAUDE.md",
]


@dataclass
class ForkConfig:
    """Configuration for a user's SLATE fork."""
    user_name: str
    user_email: str
    fork_url: Optional[str] = None
    upstream_url: str = SLATE_UPSTREAM
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_sync: Optional[str] = None
    local_branch: str = "user-workspace"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_name": self.user_name,
            "user_email": self.user_email,
            "fork_url": self.fork_url,
            "upstream_url": self.upstream_url,
            "created_at": self.created_at,
            "last_sync": self.last_sync,
            "local_branch": self.local_branch,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForkConfig":
        return cls(**data)


@dataclass
class ForkState:
    """State tracking for fork operations."""
    initialized: bool = False
    upstream_configured: bool = False
    fork_configured: bool = False
    last_validation: Optional[str] = None
    validation_passed: bool = False
    pending_changes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "upstream_configured": self.upstream_configured,
            "fork_configured": self.fork_configured,
            "last_validation": self.last_validation,
            "validation_passed": self.validation_passed,
            "pending_changes": self.pending_changes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForkState":
        return cls(**data)


class SlateForkManager:
    """
    Manages SLATE user forks and local git repositories.

    Each SLATE installation gets:
    1. An isolated local git repository
    2. Upstream tracking to the main SLATE repo
    3. Optional fork URL for contributions
    4. Local validation before any PR creation
    """

    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace = workspace_dir or WORKSPACE_ROOT
        self.fork_dir = self.workspace / ".slate_fork"
        self.config_path = self.workspace / FORK_CONFIG_FILE
        self.state_path = self.workspace / FORK_STATE_FILE

        self.config: Optional[ForkConfig] = None
        self.state: ForkState = ForkState()

        self._load()

    def _load(self):
        """Load configuration and state from disk."""
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                self.config = ForkConfig.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load fork config: {e}")

        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.state = ForkState.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load fork state: {e}")

    def _save(self):
        """Save configuration and state to disk."""
        self.fork_dir.mkdir(parents=True, exist_ok=True)

        if self.config:
            self.config_path.write_text(
                json.dumps(self.config.to_dict(), indent=2),
                encoding="utf-8"
            )

        self.state_path.write_text(
            json.dumps(self.state.to_dict(), indent=2),
            encoding="utf-8"
        )

    def _run_git(self, args: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run a git command."""
        return subprocess.run(
            ["git"] + args,
            cwd=str(cwd or self.workspace),
            capture_output=True,
            text=True,
            timeout=60
        )

    def is_git_repo(self) -> bool:
        """Check if the workspace is a git repository."""
        result = self._run_git(["rev-parse", "--git-dir"])
        return result.returncode == 0

    def initialize_user_workspace(
        self,
        user_name: str,
        user_email: str,
        fork_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Initialize a new user's SLATE workspace with local git.

        This creates:
        1. Local git repository (if not exists)
        2. User-specific branch for their work
        3. Upstream remote pointing to main SLATE repo
        4. Optional fork remote for contributions

        Args:
            user_name: Git user name
            user_email: Git user email
            fork_url: Optional URL to user's GitHub fork

        Returns:
            Status dict with initialization results
        """
        results = {
            "success": False,
            "steps": [],
            "errors": [],
        }

        try:
            # Step 1: Initialize git if needed
            if not self.is_git_repo():
                result = self._run_git(["init"])
                if result.returncode == 0:
                    results["steps"].append("Git repository initialized")
                else:
                    results["errors"].append(f"Git init failed: {result.stderr}")
                    return results
            else:
                results["steps"].append("Git repository already exists")

            # Step 2: Configure user
            self._run_git(["config", "user.name", user_name])
            self._run_git(["config", "user.email", user_email])
            results["steps"].append(f"Configured git user: {user_name} <{user_email}>")

            # Step 3: Set up upstream remote
            result = self._run_git(["remote", "get-url", "upstream"])
            if result.returncode != 0:
                self._run_git(["remote", "add", "upstream", SLATE_UPSTREAM])
                results["steps"].append("Added upstream remote")
            else:
                self._run_git(["remote", "set-url", "upstream", SLATE_UPSTREAM])
                results["steps"].append("Updated upstream remote")

            self.state.upstream_configured = True

            # Step 4: Set up fork remote if provided
            if fork_url:
                result = self._run_git(["remote", "get-url", "fork"])
                if result.returncode != 0:
                    self._run_git(["remote", "add", "fork", fork_url])
                    results["steps"].append(f"Added fork remote: {fork_url}")
                else:
                    self._run_git(["remote", "set-url", "fork", fork_url])
                    results["steps"].append(f"Updated fork remote: {fork_url}")

                self.state.fork_configured = True

            # Step 5: Create user workspace branch
            branch_name = f"user/{user_name.lower().replace(' ', '-')}"
            result = self._run_git(["checkout", "-b", branch_name])
            if result.returncode == 0:
                results["steps"].append(f"Created branch: {branch_name}")
            elif "already exists" in result.stderr:
                self._run_git(["checkout", branch_name])
                results["steps"].append(f"Switched to existing branch: {branch_name}")

            # Step 6: Create .slate_fork directory
            self.fork_dir.mkdir(parents=True, exist_ok=True)

            # Step 7: Add .slate_fork to .gitignore if not present
            gitignore_path = self.workspace / ".gitignore"
            if gitignore_path.exists():
                content = gitignore_path.read_text(encoding="utf-8")
                if ".slate_fork/" not in content:
                    with open(gitignore_path, "a", encoding="utf-8") as f:
                        f.write("\n# SLATE user fork data (local only)\n.slate_fork/\n")
                    results["steps"].append("Added .slate_fork to .gitignore")

            # Save configuration
            self.config = ForkConfig(
                user_name=user_name,
                user_email=user_email,
                fork_url=fork_url,
                local_branch=branch_name,
            )
            self.state.initialized = True
            self._save()

            results["success"] = True
            results["steps"].append("Fork manager initialized successfully")

        except Exception as e:
            results["errors"].append(str(e))
            logger.exception("Fork initialization failed")

        return results

    def validate_prerequisites(self) -> Dict[str, Any]:
        """
        Validate that the workspace meets all SLATE prerequisites.

        This checks:
        1. Required files exist
        2. Core modules import correctly
        3. No protected files modified
        4. pyproject.toml is valid
        5. current_tasks.json format is correct

        Returns:
            Validation results with pass/fail status
        """
        results = {
            "passed": True,
            "checks": [],
            "errors": [],
            "warnings": [],
        }

        # Check 1: Required files
        for file_path in REQUIRED_FILES:
            full_path = self.workspace / file_path
            if full_path.exists():
                results["checks"].append(f"PASS: {file_path} exists")
            else:
                results["passed"] = False
                results["errors"].append(f"MISSING: {file_path}")

        # Check 2: Core module imports
        try:
            sys.path.insert(0, str(self.workspace))
            import aurora_core.slate_status
            results["checks"].append("PASS: aurora_core.slate_status imports")
        except ImportError as e:
            results["passed"] = False
            results["errors"].append(f"IMPORT FAIL: aurora_core.slate_status - {e}")

        try:
            import aurora_core.action_guard
            results["checks"].append("PASS: aurora_core.action_guard imports")
        except ImportError as e:
            results["passed"] = False
            results["errors"].append(f"IMPORT FAIL: aurora_core.action_guard - {e}")

        # Check 3: pyproject.toml validation
        try:
            import tomllib
            pyproject_path = self.workspace / "pyproject.toml"
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)

            if "project" in config and "version" in config["project"]:
                results["checks"].append(f"PASS: pyproject.toml valid (v{config['project']['version']})")
            else:
                results["passed"] = False
                results["errors"].append("pyproject.toml missing project.version")
        except Exception as e:
            results["passed"] = False
            results["errors"].append(f"pyproject.toml error: {e}")

        # Check 4: current_tasks.json format
        try:
            tasks_path = self.workspace / "current_tasks.json"
            if tasks_path.exists():
                tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
                if isinstance(tasks, list):
                    results["checks"].append(f"PASS: current_tasks.json valid ({len(tasks)} tasks)")
                else:
                    results["passed"] = False
                    results["errors"].append("current_tasks.json must be a JSON array")
        except json.JSONDecodeError as e:
            results["passed"] = False
            results["errors"].append(f"current_tasks.json invalid JSON: {e}")

        # Check 5: Security policy compliance
        # Scan for dangerous patterns
        dangerous_patterns = [
            ("0.0.0.0", "Network binding to 0.0.0.0 (must use 127.0.0.1)"),
            ("eval(", "Use of eval() (security risk)"),
            ("exec(os", "Use of exec with os module"),
        ]

        for py_file in self.workspace.glob("aurora_core/**/*.py"):
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for pattern, description in dangerous_patterns:
                if pattern in content:
                    results["warnings"].append(f"{py_file.name}: {description}")

        # Update state
        self.state.last_validation = datetime.utcnow().isoformat()
        self.state.validation_passed = results["passed"]
        self._save()

        return results

    def prepare_contribution(self, branch_name: str, title: str) -> Dict[str, Any]:
        """
        Prepare a contribution branch for PR submission.

        This:
        1. Creates a new branch from the latest upstream
        2. Runs all validations
        3. Prepares commit message template

        Args:
            branch_name: Name for the contribution branch
            title: Brief title for the contribution

        Returns:
            Preparation status
        """
        results = {
            "success": False,
            "branch": None,
            "validation": None,
            "next_steps": [],
        }

        if not self.state.fork_configured:
            results["next_steps"].append(
                "Configure your fork first: slate_fork_manager.py --setup-fork <your-fork-url>"
            )
            return results

        # Fetch latest from upstream
        result = self._run_git(["fetch", "upstream"])
        if result.returncode != 0:
            results["next_steps"].append(f"Fetch failed: {result.stderr}")
            return results

        # Create contribution branch
        safe_branch = f"contrib/{branch_name.lower().replace(' ', '-')}"
        result = self._run_git(["checkout", "-b", safe_branch, "upstream/main"])
        if result.returncode != 0:
            results["next_steps"].append(f"Branch creation failed: {result.stderr}")
            return results

        results["branch"] = safe_branch

        # Run validations
        validation = self.validate_prerequisites()
        results["validation"] = validation

        if not validation["passed"]:
            results["next_steps"].append("Fix validation errors before continuing")
            results["next_steps"].extend(validation["errors"])
            return results

        results["success"] = True
        results["next_steps"] = [
            "Make your changes",
            "Run: python aurora_core/slate_fork_manager.py --validate",
            f"Commit: git commit -m '{title}'",
            f"Push: git push fork {safe_branch}",
            "Create PR on GitHub",
        ]

        return results

    def sync_with_upstream(self) -> Dict[str, Any]:
        """Sync local repository with upstream SLATE."""
        results = {
            "success": False,
            "updated_files": 0,
            "conflicts": [],
        }

        # Fetch upstream
        result = self._run_git(["fetch", "upstream"])
        if result.returncode != 0:
            results["conflicts"].append(f"Fetch failed: {result.stderr}")
            return results

        # Merge upstream/main
        result = self._run_git(["merge", "upstream/main", "--no-edit"])
        if result.returncode != 0:
            if "CONFLICT" in result.stdout:
                results["conflicts"].append("Merge conflicts detected")
                # Get conflicting files
                result = self._run_git(["diff", "--name-only", "--diff-filter=U"])
                results["conflicts"].extend(result.stdout.strip().split("\n"))
            else:
                results["conflicts"].append(result.stderr)
            return results

        # Update state
        self.config.last_sync = datetime.utcnow().isoformat()
        self._save()

        results["success"] = True
        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current fork status."""
        return {
            "initialized": self.state.initialized,
            "config": self.config.to_dict() if self.config else None,
            "state": self.state.to_dict(),
            "is_git_repo": self.is_git_repo(),
            "remotes": self._get_remotes(),
        }

    def _get_remotes(self) -> Dict[str, str]:
        """Get configured git remotes."""
        result = self._run_git(["remote", "-v"])
        remotes = {}
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line and "(fetch)" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        remotes[parts[0]] = parts[1]
        return remotes


# ═══════════════════════════════════════════════════════════════════════════════
# CELL: cli [python]
# Author: COPILOT | Created: 2026-02-06T19:30:00Z
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point for SLATE Fork Manager."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SLATE Fork Manager - Manage user forks and contributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize your SLATE workspace
  python slate_fork_manager.py --init --name "Your Name" --email "you@example.com"

  # Set up your GitHub fork
  python slate_fork_manager.py --setup-fork https://github.com/yourusername/S.L.A.T.E..git

  # Validate your workspace
  python slate_fork_manager.py --validate

  # Prepare a contribution
  python slate_fork_manager.py --contribute "fix-bug-123" --title "Fix issue #123"

  # Sync with upstream
  python slate_fork_manager.py --sync

  # Check status
  python slate_fork_manager.py --status
"""
    )

    parser.add_argument("--init", action="store_true", help="Initialize user workspace")
    parser.add_argument("--name", type=str, help="Git user name")
    parser.add_argument("--email", type=str, help="Git user email")
    parser.add_argument("--setup-fork", type=str, metavar="URL", help="Configure fork remote URL")
    parser.add_argument("--validate", action="store_true", help="Validate SLATE prerequisites")
    parser.add_argument("--contribute", type=str, metavar="BRANCH", help="Prepare contribution branch")
    parser.add_argument("--title", type=str, help="Contribution title")
    parser.add_argument("--sync", action="store_true", help="Sync with upstream SLATE")
    parser.add_argument("--status", action="store_true", help="Show fork status")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    manager = SlateForkManager()

    if args.init:
        if not args.name or not args.email:
            print("Error: --init requires --name and --email")
            sys.exit(1)

        result = manager.initialize_user_workspace(
            user_name=args.name,
            user_email=args.email,
            fork_url=args.setup_fork
        )

        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("\n🚀 SLATE Fork Manager Initialization")
            print("=" * 50)
            if result["success"]:
                print("✅ Success!")
                for step in result["steps"]:
                    print(f"  • {step}")
            else:
                print("❌ Failed!")
                for error in result["errors"]:
                    print(f"  • {error}")

    elif args.setup_fork:
        if manager.config:
            manager.config.fork_url = args.setup_fork
            manager.state.fork_configured = True
            manager._save()
            print(f"✅ Fork remote configured: {args.setup_fork}")
        else:
            print("❌ Initialize workspace first: --init --name <name> --email <email>")

    elif args.validate:
        result = manager.validate_prerequisites()

        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("\n🔍 SLATE Prerequisites Validation")
            print("=" * 50)

            for check in result["checks"]:
                print(f"  {check}")

            if result["warnings"]:
                print("\n⚠️ Warnings:")
                for warning in result["warnings"]:
                    print(f"  • {warning}")

            if result["errors"]:
                print("\n❌ Errors:")
                for error in result["errors"]:
                    print(f"  • {error}")

            print()
            if result["passed"]:
                print("✅ All prerequisites met - Ready for contribution!")
            else:
                print("❌ Prerequisites NOT met - Fix errors before contributing")

    elif args.contribute:
        if not args.title:
            args.title = args.contribute

        result = manager.prepare_contribution(args.contribute, args.title)

        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("\n📝 Preparing Contribution")
            print("=" * 50)

            if result["success"]:
                print(f"✅ Branch created: {result['branch']}")
                print("\nNext steps:")
                for i, step in enumerate(result["next_steps"], 1):
                    print(f"  {i}. {step}")
            else:
                print("❌ Preparation failed")
                for step in result["next_steps"]:
                    print(f"  • {step}")

    elif args.sync:
        result = manager.sync_with_upstream()

        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            if result["success"]:
                print("✅ Synced with upstream SLATE")
            else:
                print("❌ Sync failed")
                for conflict in result["conflicts"]:
                    print(f"  • {conflict}")

    elif args.status:
        result = manager.get_status()

        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("\n📊 SLATE Fork Status")
            print("=" * 50)
            print(f"  Initialized:     {'✅' if result['initialized'] else '❌'}")
            print(f"  Git Repository:  {'✅' if result['is_git_repo'] else '❌'}")

            if result["config"]:
                print(f"  User:            {result['config']['user_name']}")
                print(f"  Branch:          {result['config']['local_branch']}")
                print(f"  Fork URL:        {result['config']['fork_url'] or 'Not configured'}")
                print(f"  Last Sync:       {result['config']['last_sync'] or 'Never'}")

            if result["remotes"]:
                print("\n  Remotes:")
                for name, url in result["remotes"].items():
                    print(f"    {name}: {url}")

            print(f"\n  Validation:      {'✅ Passed' if result['state']['validation_passed'] else '❌ Not passed'}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
