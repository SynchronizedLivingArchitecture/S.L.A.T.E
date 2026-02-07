#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_runner_manager [python]
# Author: Claude | Created: 2026-02-06T23:00:00Z
# Purpose: GitHub Actions runner auto-detection and management for agentic workflows
# ═══════════════════════════════════════════════════════════════════════════════
"""
SLATE Runner Manager
====================
Automatically detects and configures GitHub Actions self-hosted runners.
Integrates with GitHub ecosystem for agentic task management.

Features:
- Auto-detect runner installation and status
- Configure runner environment for GPU/CUDA
- Integrate with GitHub API for workflow dispatch
- Manage hooks for SLATE environment setup

Usage:
    python slate/slate_runner_manager.py --detect
    python slate/slate_runner_manager.py --setup
    python slate/slate_runner_manager.py --status
    python slate/slate_runner_manager.py --dispatch "workflow_name"
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).parent.parent
RUNNER_PATHS = [
    WORKSPACE_ROOT / "actions-runner",
    WORKSPACE_ROOT.parent / "actions-runner",
    Path("C:/actions-runner"),
    Path(os.path.expanduser("~/actions-runner")),
]


class SlateRunnerManager:
    """Manages GitHub Actions runner for SLATE agentic workflows."""

    def __init__(self):
        self.runner_dir = self._find_runner()
        self.workspace = WORKSPACE_ROOT
        self.gh_cli = self._find_gh_cli()

    def _find_runner(self) -> Optional[Path]:
        """Auto-detect GitHub Actions runner installation."""
        for path in RUNNER_PATHS:
            runner_config = path / ".runner"
            if runner_config.exists():
                return path
        return None

    def _find_gh_cli(self) -> str:
        """Find gh CLI path."""
        gh_path = WORKSPACE_ROOT / ".tools" / "gh.exe"
        if gh_path.exists():
            return str(gh_path)
        return "gh"

    def detect(self) -> Dict[str, Any]:
        """Detect runner configuration and system state."""
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runner_installed": self.runner_dir is not None,
            "runner_dir": str(self.runner_dir) if self.runner_dir else None,
            "runner_config": None,
            "slate_config": None,
            "hooks": [],
            "gpu_info": self._detect_gpu(),
            "environment": self._detect_environment(),
            "github": self._detect_github(),
            "ready": False
        }

        if self.runner_dir:
            result["runner_config"] = self._read_runner_config()
            result["slate_config"] = self._read_slate_config()
            result["hooks"] = self._detect_hooks()
            result["ready"] = self._check_ready(result)

        return result

    def _read_runner_config(self) -> Optional[Dict[str, Any]]:
        """Read .runner configuration file."""
        if not self.runner_dir:
            return None
        runner_file = self.runner_dir / ".runner"
        if runner_file.exists():
            try:
                return json.loads(runner_file.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _read_slate_config(self) -> Optional[Dict[str, Any]]:
        """Read SLATE-specific runner configuration."""
        if not self.runner_dir:
            return None
        slate_config = self.runner_dir / ".slate_runner_config.json"
        if slate_config.exists():
            try:
                return json.loads(slate_config.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _detect_hooks(self) -> List[str]:
        """Detect installed runner hooks."""
        if not self.runner_dir:
            return []
        hooks_dir = self.runner_dir / "hooks"
        if not hooks_dir.exists():
            return []
        return [f.name for f in hooks_dir.iterdir() if f.is_file()]

    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect NVIDIA GPU configuration."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                gpus = []
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpus.append({
                            "index": parts[0],
                            "name": parts[1],
                            "compute_capability": parts[2],
                            "memory": parts[3],
                            "architecture": self._get_arch(parts[2])
                        })
                return {"available": True, "count": len(gpus), "gpus": gpus}
        except Exception:
            pass
        return {"available": False, "count": 0, "gpus": []}

    def _get_arch(self, compute_cap: str) -> str:
        """Map compute capability to architecture name."""
        if compute_cap.startswith("12."):
            return "Blackwell"
        elif compute_cap == "8.9":
            return "Ada Lovelace"
        elif compute_cap.startswith("8."):
            return "Ampere"
        elif compute_cap == "7.5":
            return "Turing"
        return "Unknown"

    def _detect_environment(self) -> Dict[str, Any]:
        """Detect SLATE environment variables."""
        return {
            "SLATE_RUNNER": os.environ.get("SLATE_RUNNER", "false"),
            "SLATE_WORKSPACE": os.environ.get("SLATE_WORKSPACE", str(WORKSPACE_ROOT)),
            "SLATE_GPU_COUNT": os.environ.get("SLATE_GPU_COUNT", "0"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "in_github_actions": os.environ.get("GITHUB_ACTIONS", "false") == "true"
        }

    def _detect_github(self) -> Dict[str, Any]:
        """Detect GitHub repository and authentication state."""
        result = {"authenticated": False, "repo": None, "workflows": []}

        try:
            auth_check = subprocess.run(
                [self.gh_cli, "auth", "status"],
                capture_output=True, text=True, timeout=10
            )
            result["authenticated"] = auth_check.returncode == 0

            if result["authenticated"]:
                repo_check = subprocess.run(
                    [self.gh_cli, "repo", "view", "--json", "nameWithOwner,url"],
                    capture_output=True, text=True, timeout=10, cwd=str(WORKSPACE_ROOT)
                )
                if repo_check.returncode == 0:
                    result["repo"] = json.loads(repo_check.stdout)

                workflow_check = subprocess.run(
                    [self.gh_cli, "workflow", "list", "--json", "name,state"],
                    capture_output=True, text=True, timeout=10, cwd=str(WORKSPACE_ROOT)
                )
                if workflow_check.returncode == 0:
                    result["workflows"] = json.loads(workflow_check.stdout)
        except Exception:
            pass

        return result

    def _check_ready(self, detection: Dict[str, Any]) -> bool:
        """Check if runner is ready for agentic workflows."""
        if not detection.get("runner_installed"):
            return False
        if not detection.get("runner_config"):
            return False
        if not detection.get("github", {}).get("authenticated"):
            return False
        return True

    def setup(self, force: bool = False) -> Dict[str, Any]:
        """Auto-configure runner for SLATE agentic workflows."""
        result = {"success": False, "actions": [], "errors": []}

        detection = self.detect()

        if not detection["runner_installed"]:
            result["errors"].append("Runner not installed. Run: gh runner download")
            return result

        # Create SLATE config if missing
        if not detection["slate_config"] or force:
            try:
                config = self._generate_slate_config(detection)
                config_path = self.runner_dir / ".slate_runner_config.json"
                config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                result["actions"].append(f"Created {config_path}")
            except Exception as e:
                result["errors"].append(f"Failed to create config: {e}")

        # Create hooks directory and pre-job hook
        hooks_dir = self.runner_dir / "hooks"
        hooks_dir.mkdir(exist_ok=True)

        pre_job_hook = hooks_dir / "pre-job.ps1"
        if not pre_job_hook.exists() or force:
            try:
                hook_content = self._generate_pre_job_hook(detection)
                pre_job_hook.write_text(hook_content, encoding="utf-8")
                result["actions"].append(f"Created {pre_job_hook}")
            except Exception as e:
                result["errors"].append(f"Failed to create hook: {e}")

        # Create environment script
        env_script = self.runner_dir / "slate_env.ps1"
        if not env_script.exists() or force:
            try:
                env_content = self._generate_env_script(detection)
                env_script.write_text(env_content, encoding="utf-8")
                result["actions"].append(f"Created {env_script}")
            except Exception as e:
                result["errors"].append(f"Failed to create env script: {e}")

        result["success"] = len(result["errors"]) == 0
        return result

    def _generate_slate_config(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SLATE runner configuration."""
        gpu_info = detection.get("gpu_info", {})
        runner_config = detection.get("runner_config", {})

        labels = ["self-hosted", "slate", "windows"]
        if gpu_info.get("available"):
            labels.extend(["gpu", "cuda"])
            if gpu_info.get("count", 0) > 1:
                labels.append("multi-gpu")
                labels.append(f"gpu-{gpu_info['count']}")
            for gpu in gpu_info.get("gpus", []):
                arch = gpu.get("architecture", "").lower()
                if arch and arch not in labels:
                    labels.append(arch)

        return {
            "name": runner_config.get("agentName", f"slate-{os.environ.get('COMPUTERNAME', 'local')}"),
            "repo_url": runner_config.get("gitHubUrl", ""),
            "runner_dir": str(self.runner_dir),
            "workspace": str(self.workspace),
            "labels": labels,
            "gpu_count": gpu_info.get("count", 0),
            "gpus": gpu_info.get("gpus", []),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "auto_configured": True
        }

    def _generate_pre_job_hook(self, detection: Dict[str, Any]) -> str:
        """Generate pre-job PowerShell hook for GPU setup."""
        gpu_info = detection.get("gpu_info", {})
        gpu_count = gpu_info.get("count", 0)
        cuda_devices = ",".join(str(i) for i in range(gpu_count)) if gpu_count > 0 else ""

        gpu_lines = ""
        for gpu in gpu_info.get("gpus", []):
            gpu_lines += f'$env:SLATE_GPU_{gpu["index"]} = "{gpu["name"]}"\n'

        return f'''# SLATE Pre-Job Hook - Auto-generated
# Modified: {datetime.now(timezone.utc).isoformat()} | Author: SLATE Runner Manager
# Sets GPU and SLATE environment for every GitHub Actions job

$env:CUDA_VISIBLE_DEVICES = "{cuda_devices}"
$env:SLATE_WORKSPACE = "{self.workspace}"
$env:SLATE_VENV = "{self.workspace}\\.venv"
$env:SLATE_RUNNER = "true"
$env:SLATE_GPU_COUNT = "{gpu_count}"
{gpu_lines}$env:PYTHONPATH = "{self.workspace}"
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "{self.workspace}\\.venv\\Scripts;$env:PATH"

# Verify GPUs are accessible
try {{
    $gpuInfo = nvidia-smi --query-gpu=index,name,utilization.gpu --format=csv,noheader 2>$null
    if ($gpuInfo) {{
        $gpuLines = ($gpuInfo -split "`n" | Where-Object {{ $_.Trim() }})
        Write-Host "[SLATE] Pre-job: $($gpuLines.Count) GPU(s) active, CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES"
    }}
}} catch {{
    Write-Host "[SLATE] Pre-job: GPU check skipped"
}}
'''

    def _generate_env_script(self, detection: Dict[str, Any]) -> str:
        """Generate environment setup PowerShell script."""
        gpu_info = detection.get("gpu_info", {})
        gpu_count = gpu_info.get("count", 0)
        cuda_devices = ",".join(str(i) for i in range(gpu_count)) if gpu_count > 0 else ""

        gpu_lines = ""
        for gpu in gpu_info.get("gpus", []):
            gpu_lines += f'$env:SLATE_GPU_{gpu["index"]} = "{gpu["name"]}"\n'

        return f'''# SLATE Runner Environment Setup - Auto-generated
# Modified: {datetime.now(timezone.utc).isoformat()} | Author: SLATE Runner Manager
# Source this script to configure the SLATE runner environment

$env:SLATE_WORKSPACE = "{self.workspace}"
$env:SLATE_VENV = "{self.workspace}\\.venv"
$env:SLATE_RUNNER = "true"
$env:SLATE_GPU_COUNT = "{gpu_count}"
{gpu_lines}$env:CUDA_VISIBLE_DEVICES = "{cuda_devices}"
$env:PYTHONPATH = "{self.workspace};$env:PYTHONPATH"
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "{self.workspace}\\.venv\\Scripts;$env:PATH"

# Activate venv
& "{self.workspace}\\.venv\\Scripts\\Activate.ps1"

Write-Host "[SLATE] Runner environment loaded"
Write-Host "  Workspace:  $env:SLATE_WORKSPACE"
Write-Host "  Python:     $((& python --version 2>&1))"
Write-Host "  GPU count:  $env:SLATE_GPU_COUNT"
Write-Host "  CUDA devs:  $env:CUDA_VISIBLE_DEVICES"

# Quick status
try {{
    & python slate/slate_status.py --quick 2>$null | Out-Null
    Write-Host "  SLATE:      systems initialized"
}} catch {{
    Write-Host "  SLATE:      status check skipped"
}}
'''

    def dispatch_workflow(self, workflow: str, inputs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Dispatch a GitHub Actions workflow for agentic task execution."""
        result = {"success": False, "workflow": workflow, "run_id": None, "error": None}

        cmd = [self.gh_cli, "workflow", "run", workflow]
        if inputs:
            for key, value in inputs.items():
                cmd.extend(["-f", f"{key}={value}"])

        try:
            run_result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, cwd=str(WORKSPACE_ROOT)
            )
            if run_result.returncode == 0:
                result["success"] = True
                # Get the run ID from the latest run
                list_result = subprocess.run(
                    [self.gh_cli, "run", "list", "--workflow", workflow, "--limit", "1", "--json", "databaseId"],
                    capture_output=True, text=True, timeout=10, cwd=str(WORKSPACE_ROOT)
                )
                if list_result.returncode == 0:
                    runs = json.loads(list_result.stdout)
                    if runs:
                        result["run_id"] = runs[0].get("databaseId")
            else:
                result["error"] = run_result.stderr.strip()
        except Exception as e:
            result["error"] = str(e)

        return result

    def get_runner_status(self) -> Dict[str, Any]:
        """Get current runner service status."""
        if not self.runner_dir:
            return {"installed": False, "running": False}

        result = {"installed": True, "running": False, "service_name": None, "process_running": False, "pid": None}

        # Check for runner process (foreground mode)
        try:
            proc_check = subprocess.run(
                ["powershell", "-Command",
                 "Get-Process -Name 'Runner.Listener' -ErrorAction SilentlyContinue | Select-Object Id | ConvertTo-Json"],
                capture_output=True, text=True, timeout=10
            )
            if proc_check.returncode == 0 and proc_check.stdout.strip():
                proc_data = json.loads(proc_check.stdout)
                if isinstance(proc_data, dict):
                    result["process_running"] = True
                    result["running"] = True
                    result["pid"] = proc_data.get("Id")
                elif isinstance(proc_data, list) and proc_data:
                    result["process_running"] = True
                    result["running"] = True
                    result["pid"] = proc_data[0].get("Id")
        except Exception:
            pass

        # Check for Windows service
        try:
            svc_status = subprocess.run(
                ["powershell", "-Command",
                 "Get-Service -Name '*actions*' | Select-Object Name,Status | ConvertTo-Json"],
                capture_output=True, text=True, timeout=10
            )
            if svc_status.returncode == 0 and svc_status.stdout.strip():
                services = json.loads(svc_status.stdout)
                if isinstance(services, dict):
                    services = [services]
                for svc in services:
                    if "actions" in svc.get("Name", "").lower():
                        result["service_name"] = svc["Name"]
                        result["running"] = svc.get("Status") == 4  # 4 = Running
                        break
        except Exception:
            pass

        return result

    def print_status(self):
        """Print formatted runner status."""
        detection = self.detect()
        service = self.get_runner_status()

        print("\n" + "=" * 60)
        print("  SLATE GitHub Runner Status")
        print("=" * 60)

        # Runner installation
        if detection["runner_installed"]:
            print(f"\n  Runner:     Installed at {detection['runner_dir']}")
            config = detection.get("runner_config", {})
            if config:
                print(f"  Agent:      {config.get('agentName', 'unknown')}")
                print(f"  GitHub:     {config.get('gitHubUrl', 'not configured')}")
        else:
            print("\n  Runner:     Not installed")
            print("              Run: python slate/slate_runner_manager.py --setup")

        # Service status
        if service.get("running"):
            if service.get("service_name"):
                print(f"  Service:    Running ({service.get('service_name')})")
            elif service.get("process_running"):
                print(f"  Process:    Running (PID {service.get('pid')})")
            else:
                print(f"  Status:     Running")
        elif service.get("installed"):
            print(f"  Service:    Stopped ({service.get('service_name')})")
        else:
            print("  Service:    Not configured")

        # GPU
        gpu = detection.get("gpu_info", {})
        if gpu.get("available"):
            print(f"\n  GPUs:       {gpu['count']} detected")
            for g in gpu.get("gpus", []):
                print(f"              - {g['name']} ({g['architecture']}, {g['memory']})")
        else:
            print("\n  GPUs:       None (CPU-only mode)")

        # GitHub
        gh = detection.get("github", {})
        if gh.get("authenticated"):
            repo = gh.get("repo", {})
            print(f"\n  GitHub:     Authenticated")
            if repo:
                print(f"  Repository: {repo.get('nameWithOwner', 'unknown')}")
            workflows = gh.get("workflows", [])
            active = sum(1 for w in workflows if w.get("state") == "active")
            print(f"  Workflows:  {active}/{len(workflows)} active")
        else:
            print("\n  GitHub:     Not authenticated")
            print("              Run: gh auth login")

        # Ready state
        print(f"\n  Ready:      {'Yes' if detection['ready'] else 'No'}")

        # Hooks
        hooks = detection.get("hooks", [])
        if hooks:
            print(f"  Hooks:      {', '.join(hooks)}")

        print("\n" + "=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SLATE GitHub Runner Manager - Auto-detect and configure runners"
    )
    parser.add_argument("--detect", action="store_true", help="Detect runner configuration")
    parser.add_argument("--setup", action="store_true", help="Auto-setup runner for SLATE")
    parser.add_argument("--force", action="store_true", help="Force reconfiguration")
    parser.add_argument("--status", action="store_true", help="Show runner status")
    parser.add_argument("--dispatch", metavar="WORKFLOW", help="Dispatch a workflow")
    # Modified: 2026-02-07T04:45:00Z | Author: COPILOT | Change: Add agentic dispatch shortcuts
    parser.add_argument("--agentic", metavar="MODE", nargs="?", const="autonomous",
                        choices=["autonomous", "single-task", "inference-bench", "discover", "health-check"],
                        help="Dispatch agentic AI workflow (default: autonomous)")
    parser.add_argument("--max-tasks", type=str, default="10", help="Max tasks for agentic mode")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    manager = SlateRunnerManager()

    if args.detect:
        result = manager.detect()
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            manager.print_status()

    elif args.setup:
        result = manager.setup(force=args.force)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["success"]:
                print("\n[OK] Runner configured for SLATE")
                for action in result["actions"]:
                    print(f"  - {action}")
            else:
                print("\n[X] Setup failed:")
                for error in result["errors"]:
                    print(f"  - {error}")

    elif args.status:
        if args.json:
            detection = manager.detect()
            service = manager.get_runner_status()
            print(json.dumps({"detection": detection, "service": service}, indent=2, default=str))
        else:
            manager.print_status()

    elif args.dispatch:
        result = manager.dispatch_workflow(args.dispatch)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["success"]:
                print(f"[OK] Dispatched {args.dispatch}")
                if result["run_id"]:
                    print(f"     Run ID: {result['run_id']}")
            else:
                print(f"[X] Failed: {result['error']}")

    # Modified: 2026-02-07T04:45:00Z | Author: COPILOT | Change: Agentic dispatch shortcut
    elif args.agentic:
        inputs = {"mode": args.agentic, "max_tasks": args.max_tasks}
        result = manager.dispatch_workflow("agentic.yml", inputs=inputs)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["success"]:
                print(f"[OK] Dispatched agentic AI ({args.agentic}, max={args.max_tasks})")
                if result["run_id"]:
                    print(f"     Run ID: {result['run_id']}")
            else:
                print(f"[X] Failed: {result['error']}")

    else:
        manager.print_status()


if __name__ == "__main__":
    main()
