#!/usr/bin/env python3
"""
SLATE SDK - Unified Setup and Integration Module
Author: COPILOT | Created: 2026-02-06T00:00:00Z

Provides a unified interface for SLATE setup, configuration, and integration.
Consolidates all setup operations into a single SDK entry point.

Usage:
    python aurora_core/slate_sdk.py --setup           # Full SLATE setup
    python aurora_core/slate_sdk.py --status          # System status
    python aurora_core/slate_sdk.py --configure       # Interactive configuration
    python aurora_core/slate_sdk.py --verify          # Verify installation
    python aurora_core/slate_sdk.py --integrate-git   # Setup git integration
    python aurora_core/slate_sdk.py --integrate-runner # Setup self-hosted runner
"""

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup path
WORKSPACE_ROOT = Path(__file__).parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

logger = logging.getLogger("aurora.sdk")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL: constants
# Author: COPILOT | Created: 2026-02-06T00:00:00Z
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "2.4.0"
SDK_CONFIG_FILE = WORKSPACE_ROOT / ".slate_sdk" / "config.json"
SDK_STATE_FILE = WORKSPACE_ROOT / ".slate_sdk" / "state.json"

GITHUB_OWNER = "SynchronizedLivingArchitecture"
GITHUB_REPO = "S.L.A.T.E."
GITHUB_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}"

REQUIRED_PYTHON_VERSION = (3, 11)

CORE_MODULES = [
    "aurora_core.slate_status",
    "aurora_core.action_guard",
    "aurora_core.sdk_source_guard",
    "aurora_core.unified_ai_backend",
    "aurora_core.slate_fork_manager",
    "aurora_core.slate_project_manager",
]

OPTIONAL_MODULES = [
    "aurora_core.slate_runner_manager",
    "aurora_core.foundry_local",
    "aurora_core.rag_memory",
]

AI_BACKENDS = {
    "ollama": {"port": 11434, "check_url": "http://127.0.0.1:11434/api/tags"},
    "foundry": {"port": 5272, "check_url": "http://127.0.0.1:5272/health"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# CELL: data_classes
# Author: COPILOT | Created: 2026-02-06T00:00:00Z
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SDKConfig:
    """SLATE SDK configuration."""
    version: str = VERSION
    workspace: str = str(WORKSPACE_ROOT)
    github_owner: str = GITHUB_OWNER
    github_repo: str = GITHUB_REPO
    runner_enabled: bool = False
    runner_name: Optional[str] = None
    git_configured: bool = False
    git_remotes: Dict[str, str] = field(default_factory=dict)
    ai_backends: List[str] = field(default_factory=list)
    gpu_count: int = 0
    installed_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "workspace": self.workspace,
            "github_owner": self.github_owner,
            "github_repo": self.github_repo,
            "runner_enabled": self.runner_enabled,
            "runner_name": self.runner_name,
            "git_configured": self.git_configured,
            "git_remotes": self.git_remotes,
            "ai_backends": self.ai_backends,
            "gpu_count": self.gpu_count,
            "installed_at": self.installed_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDKConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SDKState:
    """SLATE SDK runtime state."""
    initialized: bool = False
    modules_loaded: List[str] = field(default_factory=list)
    modules_failed: List[str] = field(default_factory=list)
    backends_online: List[str] = field(default_factory=list)
    runner_status: str = "unknown"
    last_check: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "modules_loaded": self.modules_loaded,
            "modules_failed": self.modules_failed,
            "backends_online": self.backends_online,
            "runner_status": self.runner_status,
            "last_check": self.last_check,
            "errors": self.errors,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CELL: sdk_class
# Author: COPILOT | Created: 2026-02-06T00:00:00Z
# ═══════════════════════════════════════════════════════════════════════════════

class SlateSDK:
    """
    SLATE Software Development Kit.

    Provides unified interface for:
    - Installation and setup
    - Configuration management
    - Git integration
    - Runner integration
    - Module loading
    - Status checking
    """

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or WORKSPACE_ROOT
        self.config_dir = self.workspace / ".slate_sdk"
        self.config_file = self.config_dir / "config.json"
        self.state_file = self.config_dir / "state.json"
        self.config = self._load_config()
        self.state = SDKState()

    def _load_config(self) -> SDKConfig:
        """Load SDK configuration."""
        if self.config_file.exists():
            try:
                data = json.loads(self.config_file.read_text())
                return SDKConfig.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load SDK config: {e}")
        return SDKConfig()

    def _save_config(self) -> None:
        """Save SDK configuration."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config.updated_at = datetime.utcnow().isoformat()
        self.config_file.write_text(json.dumps(self.config.to_dict(), indent=2))

    def _save_state(self) -> None:
        """Save SDK state."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.state.last_check = datetime.utcnow().isoformat()
        self.state_file.write_text(json.dumps(self.state.to_dict(), indent=2))

    def _run_cmd(self, cmd: List[str], cwd: Optional[Path] = None,
                 timeout: int = 60) -> Tuple[bool, str, str]:
        """Run a command and return (success, stdout, stderr)."""
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd or self.workspace),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)

    # ─── Setup Methods ────────────────────────────────────────────────────────

    def setup(self, include_runner: bool = False,
              runner_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Full SLATE SDK setup.

        Args:
            include_runner: Whether to setup self-hosted runner
            runner_token: GitHub token for runner registration

        Returns:
            Setup results
        """
        results = {
            "success": True,
            "steps": [],
            "errors": [],
        }

        # Step 1: Verify Python version
        step = self._verify_python()
        results["steps"].append(step)
        if not step["success"]:
            results["success"] = False
            results["errors"].append(step["error"])
            return results

        # Step 2: Check workspace structure
        step = self._verify_workspace()
        results["steps"].append(step)

        # Step 3: Load core modules
        step = self._load_modules()
        results["steps"].append(step)

        # Step 4: Detect hardware
        step = self._detect_hardware()
        results["steps"].append(step)

        # Step 5: Check AI backends
        step = self._check_backends()
        results["steps"].append(step)

        # Step 6: Configure git
        step = self._configure_git()
        results["steps"].append(step)

        # Step 7: Setup runner (optional)
        if include_runner:
            step = self._setup_runner(runner_token)
            results["steps"].append(step)

        # Save configuration
        self.config.installed_at = self.config.installed_at or datetime.utcnow().isoformat()
        self._save_config()
        self._save_state()

        # Check for any failures
        for step in results["steps"]:
            if not step.get("success", True) and step.get("required", False):
                results["success"] = False

        return results

    def _verify_python(self) -> Dict[str, Any]:
        """Verify Python version meets requirements."""
        version = sys.version_info[:2]
        required = REQUIRED_PYTHON_VERSION
        success = version >= required

        return {
            "name": "verify_python",
            "success": success,
            "required": True,
            "version": f"{version[0]}.{version[1]}",
            "required_version": f"{required[0]}.{required[1]}",
            "error": None if success else f"Python {required[0]}.{required[1]}+ required",
        }

    def _verify_workspace(self) -> Dict[str, Any]:
        """Verify workspace structure."""
        required_paths = [
            "aurora_core/__init__.py",
            "pyproject.toml",
            "requirements.txt",
        ]

        missing = []
        for path in required_paths:
            if not (self.workspace / path).exists():
                missing.append(path)

        return {
            "name": "verify_workspace",
            "success": len(missing) == 0,
            "required": True,
            "missing_files": missing,
        }

    def _load_modules(self) -> Dict[str, Any]:
        """Load and verify core modules."""
        loaded = []
        failed = []

        for module_name in CORE_MODULES:
            try:
                __import__(module_name)
                loaded.append(module_name)
            except ImportError as e:
                failed.append({"module": module_name, "error": str(e)})

        self.state.modules_loaded = loaded
        self.state.modules_failed = [f["module"] for f in failed]

        return {
            "name": "load_modules",
            "success": len(failed) == 0,
            "required": False,
            "loaded": len(loaded),
            "failed": failed,
        }

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect GPU and hardware configuration."""
        gpu_count = 0
        gpus = []

        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpus.append({
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory": torch.cuda.get_device_properties(i).total_memory // (1024**3),
                    })
        except ImportError:
            pass

        self.config.gpu_count = gpu_count

        return {
            "name": "detect_hardware",
            "success": True,
            "gpu_count": gpu_count,
            "gpus": gpus,
            "platform": platform.system(),
            "cpu_count": os.cpu_count(),
        }

    def _check_backends(self) -> Dict[str, Any]:
        """Check AI backend availability."""
        import urllib.request
        import urllib.error

        online = []
        offline = []

        for name, config in AI_BACKENDS.items():
            try:
                req = urllib.request.Request(config["check_url"], method="GET")
                urllib.request.urlopen(req, timeout=2)
                online.append(name)
            except (urllib.error.URLError, Exception):
                offline.append(name)

        self.config.ai_backends = online
        self.state.backends_online = online

        return {
            "name": "check_backends",
            "success": len(online) > 0,
            "online": online,
            "offline": offline,
        }

    def _configure_git(self) -> Dict[str, Any]:
        """Configure git remotes for SLATE."""
        remotes = {}

        # Check current remotes
        success, stdout, _ = self._run_cmd(["git", "remote", "-v"])
        if success:
            for line in stdout.strip().split("\n"):
                if line and "(fetch)" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        remotes[parts[0]] = parts[1]

        self.config.git_configured = "origin" in remotes
        self.config.git_remotes = remotes

        return {
            "name": "configure_git",
            "success": True,
            "remotes": remotes,
            "origin_configured": "origin" in remotes,
        }

    def _setup_runner(self, token: Optional[str] = None) -> Dict[str, Any]:
        """Setup self-hosted runner."""
        try:
            from aurora_core.slate_runner_manager import SlateRunnerManager
            manager = SlateRunnerManager()
            status = manager.get_runner_status()

            self.config.runner_enabled = status.get("installed", False)
            self.config.runner_name = status.get("name")
            self.state.runner_status = status.get("status", "unknown")

            return {
                "name": "setup_runner",
                "success": True,
                "status": status,
            }
        except Exception as e:
            return {
                "name": "setup_runner",
                "success": False,
                "error": str(e),
            }

    # ─── Status Methods ───────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive SDK status."""
        # Refresh state
        self._load_modules()
        self._check_backends()

        # Check runner
        runner_status = "unknown"
        try:
            from aurora_core.slate_runner_manager import SlateRunnerManager
            manager = SlateRunnerManager()
            status = manager.get_runner_status()
            runner_status = status.get("status", "unknown")
        except Exception:
            pass

        self.state.runner_status = runner_status
        self._save_state()

        return {
            "version": VERSION,
            "workspace": str(self.workspace),
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.system(),
        }

    def verify(self) -> Dict[str, Any]:
        """Verify SLATE installation."""
        checks = []

        # Python version
        checks.append({
            "check": "python_version",
            "passed": sys.version_info[:2] >= REQUIRED_PYTHON_VERSION,
            "value": f"{sys.version_info.major}.{sys.version_info.minor}",
        })

        # Virtual environment
        venv_path = self.workspace / ".venv"
        checks.append({
            "check": "virtual_environment",
            "passed": venv_path.exists(),
            "value": str(venv_path) if venv_path.exists() else "Not found",
        })

        # Core modules
        modules_ok = len(self.state.modules_failed) == 0
        checks.append({
            "check": "core_modules",
            "passed": modules_ok,
            "value": f"{len(self.state.modules_loaded)} loaded",
            "failed": self.state.modules_failed,
        })

        # AI backends
        backends_ok = len(self.state.backends_online) > 0
        checks.append({
            "check": "ai_backends",
            "passed": backends_ok,
            "value": ", ".join(self.state.backends_online) or "None online",
        })

        # Git configured
        checks.append({
            "check": "git_configured",
            "passed": self.config.git_configured,
            "value": "origin" if self.config.git_configured else "Not configured",
        })

        # Runner
        runner_ok = self.state.runner_status in ["online", "listening"]
        checks.append({
            "check": "self_hosted_runner",
            "passed": runner_ok,
            "value": self.state.runner_status,
        })

        all_passed = all(c["passed"] for c in checks if c["check"] != "self_hosted_runner")

        return {
            "verified": all_passed,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ─── Integration Methods ──────────────────────────────────────────────────

    def integrate_git(self) -> Dict[str, Any]:
        """Setup git integration with proper remotes."""
        results = {"success": True, "actions": []}

        # Check if git repo
        success, _, _ = self._run_cmd(["git", "status"])
        if not success:
            results["success"] = False
            results["error"] = "Not a git repository"
            return results

        # Setup origin remote
        origin_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}.git"
        success, stdout, _ = self._run_cmd(["git", "remote", "get-url", "origin"])

        if not success:
            # Add origin
            self._run_cmd(["git", "remote", "add", "origin", origin_url])
            results["actions"].append(f"Added origin: {origin_url}")
        elif stdout.strip() != origin_url:
            # Update origin
            self._run_cmd(["git", "remote", "set-url", "origin", origin_url])
            results["actions"].append(f"Updated origin: {origin_url}")

        # Setup beta remote
        beta_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}-BETA.git"
        success, _, _ = self._run_cmd(["git", "remote", "get-url", "beta"])
        if not success:
            self._run_cmd(["git", "remote", "add", "beta", beta_url])
            results["actions"].append(f"Added beta: {beta_url}")

        # Refresh config
        self._configure_git()
        self._save_config()

        return results

    def integrate_runner(self, token: Optional[str] = None) -> Dict[str, Any]:
        """Setup self-hosted runner integration."""
        try:
            from aurora_core.slate_runner_manager import SlateRunnerManager
            manager = SlateRunnerManager()

            status = manager.get_runner_status()

            if status.get("installed"):
                return {
                    "success": True,
                    "message": "Runner already installed",
                    "status": status,
                }

            # Would need token for full setup
            return {
                "success": False,
                "message": "Runner not installed. Run: python install_slate.py --runner",
                "instructions": [
                    "1. Get a runner token from GitHub Settings → Actions → Runners",
                    "2. Run: python install_slate.py --runner --runner-token YOUR_TOKEN",
                    "3. Or manually: python aurora_core/slate_runner_manager.py --install",
                ],
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


# ═══════════════════════════════════════════════════════════════════════════════
# CELL: cli
# Author: COPILOT | Created: 2026-02-06T00:00:00Z
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SLATE SDK - Unified Setup and Integration",
        epilog="""
Examples:
  # Full setup
  python slate_sdk.py --setup

  # Setup with runner
  python slate_sdk.py --setup --runner

  # Check status
  python slate_sdk.py --status

  # Verify installation
  python slate_sdk.py --verify

  # Configure git
  python slate_sdk.py --integrate-git

  # Setup runner
  python slate_sdk.py --integrate-runner
"""
    )

    parser.add_argument("--setup", action="store_true", help="Run full SDK setup")
    parser.add_argument("--status", action="store_true", help="Show SDK status")
    parser.add_argument("--verify", action="store_true", help="Verify installation")
    parser.add_argument("--configure", action="store_true", help="Interactive configuration")
    parser.add_argument("--integrate-git", action="store_true", help="Setup git integration")
    parser.add_argument("--integrate-runner", action="store_true", help="Setup runner integration")
    parser.add_argument("--runner", action="store_true", help="Include runner in setup")
    parser.add_argument("--runner-token", type=str, help="GitHub token for runner")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")

    args = parser.parse_args()
    sdk = SlateSDK()

    if args.setup:
        result = sdk.setup(include_runner=args.runner, runner_token=args.runner_token)
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("\n[SLATE SDK] Setup Complete")
            print("=" * 50)
            for step in result["steps"]:
                status = "[OK]" if step.get("success", True) else "[FAIL]"
                print(f"  {status} {step['name']}")
            print(f"\n  Overall: {'Success' if result['success'] else 'Failed'}")

    elif args.status:
        result = sdk.get_status()
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("\n[SLATE SDK] Status")
            print("=" * 50)
            print(f"  Version:     {result['version']}")
            print(f"  Platform:    {result['platform']}")
            print(f"  Python:      {result['python_version']}")
            print(f"  GPUs:        {result['config']['gpu_count']}")
            print(f"  AI Backends: {', '.join(result['config']['ai_backends']) or 'None'}")
            print(f"  Runner:      {result['state']['runner_status']}")
            print(f"  Git:         {'Configured' if result['config']['git_configured'] else 'Not configured'}")

    elif args.verify:
        # First refresh modules
        sdk._load_modules()
        sdk._check_backends()
        result = sdk.verify()
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("\n[SLATE SDK] Verification")
            print("=" * 50)
            for check in result["checks"]:
                status = "[OK]" if check["passed"] else "[--]"
                print(f"  {status} {check['check']}: {check['value']}")
            print(f"\n  Verified: {'Yes' if result['verified'] else 'No'}")

    elif args.integrate_git:
        result = sdk.integrate_git()
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("\n[SLATE SDK] Git Integration")
            print("=" * 50)
            if result["success"]:
                for action in result["actions"]:
                    print(f"  [OK] {action}")
                if not result["actions"]:
                    print("  Git already configured")
            else:
                print(f"  [FAIL] {result.get('error', 'Unknown error')}")

    elif args.integrate_runner:
        result = sdk.integrate_runner(args.runner_token)
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("\n[SLATE SDK] Runner Integration")
            print("=" * 50)
            if result["success"]:
                print(f"  {result['message']}")
            else:
                print(f"  {result.get('message', result.get('error', 'Failed'))}")
                if "instructions" in result:
                    print("\n  Instructions:")
                    for inst in result["instructions"]:
                        print(f"    {inst}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
